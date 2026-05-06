"""Tests for the orphan-task reconciler beat.

Covers three cases:
  1. A PROCESSING row with `updated_at` older than the cutoff is
     reconciled: status → FAILED, rejection_reason set, audit row added.
  2. A PROCESSING row that's still recent is NOT reconciled.
  3. Idempotency: a second run on the same already-reconciled row is a
     no-op (the row is no longer PROCESSING, so it's not re-selected;
     no duplicate AuditLog appears).

The async core function is exercised directly via the per-test
rolled-back `db_session` fixture, mirroring how `run_pipeline` is tested
in `test_pipeline_runner.py`. A single end-to-end test also exercises
`reconcile_stale_submissions.run()` (the Celery task entrypoint) by
swapping in a session factory bound to the test connection so we can
observe its commit without leaving rows behind across tests.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from grader.db.models import AuditLog, Submission, SubmissionStatus, User
from grader.workers import reconciler
from grader.workers.reconciler import (
    RECONCILE_AUDIT_ACTION,
    RECONCILE_REASON,
    STALE_AFTER,
    reconcile_stale_submissions,
    reconcile_stale_submissions_async,
)


async def _make_user(db: AsyncSession) -> User:
    user = User(clerk_id=f"u_{uuid.uuid4().hex[:8]}", email=f"{uuid.uuid4().hex[:8]}@x")
    db.add(user)
    await db.flush()
    return user


async def _make_submission(
    db: AsyncSession,
    user: User,
    *,
    status: SubmissionStatus,
    age: timedelta,
) -> Submission:
    """Insert a submission and force `updated_at` to `now - age`.

    `updated_at` has `onupdate=func.now()` so we can't just set the
    attribute; we need a follow-up UPDATE that bypasses the ORM hook
    after the row is flushed."""
    sub = Submission(user_id=user.id, status=status)
    db.add(sub)
    await db.flush()
    target = datetime.now(timezone.utc) - age
    await db.execute(
        update(Submission).where(Submission.id == sub.id).values(updated_at=target)
    )
    await db.flush()
    await db.refresh(sub)
    return sub


async def _audit_count(db: AsyncSession, submission_id: uuid.UUID) -> int:
    n = await db.execute(
        select(func.count(AuditLog.id))
        .where(AuditLog.submission_id == submission_id)
        .where(AuditLog.action == RECONCILE_AUDIT_ACTION)
    )
    return int(n.scalar_one())


# --------------------------------------------------------------------------
# Positive: stale row gets reconciled
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reconciles_stale_processing_row(db_session: AsyncSession) -> None:
    user = await _make_user(db_session)
    sub = await _make_submission(
        db_session,
        user,
        status=SubmissionStatus.PROCESSING,
        age=STALE_AFTER + timedelta(seconds=30),
    )

    n = await reconcile_stale_submissions_async(db_session)

    assert n == 1
    refreshed = await db_session.get(Submission, sub.id)
    assert refreshed is not None
    assert refreshed.status == SubmissionStatus.FAILED
    assert refreshed.rejection_reason == RECONCILE_REASON
    assert await _audit_count(db_session, sub.id) == 1


# --------------------------------------------------------------------------
# Negative: recent PROCESSING row is left alone
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_does_not_reconcile_recent_processing_row(db_session: AsyncSession) -> None:
    user = await _make_user(db_session)
    sub = await _make_submission(
        db_session,
        user,
        status=SubmissionStatus.PROCESSING,
        age=timedelta(seconds=30),  # well within the 5-min window
    )

    n = await reconcile_stale_submissions_async(db_session)

    assert n == 0
    refreshed = await db_session.get(Submission, sub.id)
    assert refreshed is not None
    assert refreshed.status == SubmissionStatus.PROCESSING
    assert refreshed.rejection_reason is None
    assert await _audit_count(db_session, sub.id) == 0


@pytest.mark.asyncio
async def test_does_not_reconcile_completed_or_failed(db_session: AsyncSession) -> None:
    user = await _make_user(db_session)
    completed = await _make_submission(
        db_session,
        user,
        status=SubmissionStatus.COMPLETED,
        age=STALE_AFTER + timedelta(minutes=10),
    )
    already_failed = await _make_submission(
        db_session,
        user,
        status=SubmissionStatus.FAILED,
        age=STALE_AFTER + timedelta(minutes=10),
    )

    n = await reconcile_stale_submissions_async(db_session)

    assert n == 0
    assert (await db_session.get(Submission, completed.id)).status == SubmissionStatus.COMPLETED
    assert (await db_session.get(Submission, already_failed.id)).status == SubmissionStatus.FAILED


# --------------------------------------------------------------------------
# Idempotency: a second run after the first is a no-op
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_idempotent_second_run(db_session: AsyncSession) -> None:
    user = await _make_user(db_session)
    sub = await _make_submission(
        db_session,
        user,
        status=SubmissionStatus.PROCESSING,
        age=STALE_AFTER + timedelta(minutes=1),
    )

    first = await reconcile_stale_submissions_async(db_session)
    second = await reconcile_stale_submissions_async(db_session)

    assert first == 1
    assert second == 0  # row is now FAILED, not selected again
    assert await _audit_count(db_session, sub.id) == 1
    refreshed = await db_session.get(Submission, sub.id)
    assert refreshed is not None
    assert refreshed.rejection_reason == RECONCILE_REASON


# --------------------------------------------------------------------------
# Celery task entrypoint via .run()
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_celery_task_run_invokes_reconciler(
    _engine, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise the Celery task entrypoint via `.run()` (not `.delay()`).

    Unlike the other tests, this one cannot use the per-test
    rolled-back `db_session` because the task opens its own session and
    commits — invisible across that boundary. We seed via a committing
    session bound to the test engine, then have the task open its own
    engine inside the thread (asyncpg connection pools are bound to the
    event loop they were created in, so we cannot reuse the test's
    engine across the `asyncio.run` boundary that `.run()` triggers).
    """
    import asyncio
    import os

    from sqlalchemy import delete
    from sqlalchemy.ext.asyncio import async_sessionmaker

    setup_factory = async_sessionmaker(_engine, expire_on_commit=False)
    db_url = os.environ.get(
        "TEST_DATABASE_URL",
        "postgresql+asyncpg://grader:grader@localhost:5432/grader_test",
    )

    # Seed via a committing session so the task's fresh session sees the row.
    async with setup_factory() as setup:
        user = await _make_user(setup)
        sub = await _make_submission(
            setup,
            user,
            status=SubmissionStatus.PROCESSING,
            age=STALE_AFTER + timedelta(seconds=45),
        )
        await setup.commit()
        sub_id = sub.id
        user_id = user.id

    # Patch the reconciler's session factory with one that builds its own
    # engine on first use *inside* the thread's event loop. We use a
    # callable masquerading as the factory: when invoked, it builds an
    # engine + sessionmaker against the same test DB URL and returns a
    # session — exactly the shape `_run_async` expects from
    # `async_session_maker()`.
    from sqlalchemy.ext.asyncio import create_async_engine

    # Run the entire `.run()` invocation in a worker thread that builds
    # and disposes its own engine. asyncpg connections are bound to the
    # event loop they were created in, so the engine MUST live and die
    # inside the same `asyncio.run` call — we cannot reuse it across
    # loops. We avoid reuse entirely: one-shot factory, one-shot engine.
    def _run_in_thread() -> int:
        thread_engine = create_async_engine(db_url, pool_pre_ping=True)
        thread_maker = async_sessionmaker(thread_engine, expire_on_commit=False)
        monkeypatch.setattr(reconciler, "async_session_maker", thread_maker)
        try:
            return reconcile_stale_submissions.run()
        finally:
            # Dispose inside the thread's own loop. `.run()` already
            # consumed its asyncio.run; we need a fresh one for dispose.
            import asyncio as _a

            _a.run(thread_engine.dispose())

    try:
        # `.run()` calls asyncio.run internally; that cannot nest inside
        # the test's running loop. Hand off to a thread that builds +
        # disposes its own engine.
        n = await asyncio.to_thread(_run_in_thread)
        assert n == 1

        async with setup_factory() as verify:
            refreshed = await verify.get(Submission, sub_id)
            assert refreshed is not None
            assert refreshed.status == SubmissionStatus.FAILED
            assert refreshed.rejection_reason == RECONCILE_REASON
            audit_n = await verify.execute(
                select(func.count(AuditLog.id))
                .where(AuditLog.submission_id == sub_id)
                .where(AuditLog.action == RECONCILE_AUDIT_ACTION)
            )
            assert int(audit_n.scalar_one()) == 1
    finally:
        # Cleanup so this test doesn't pollute the shared schema for
        # other tests in the session.
        async with setup_factory() as cleanup:
            await cleanup.execute(delete(AuditLog).where(AuditLog.submission_id == sub_id))
            await cleanup.execute(delete(Submission).where(Submission.id == sub_id))
            await cleanup.execute(delete(User).where(User.id == user_id))
            await cleanup.commit()
