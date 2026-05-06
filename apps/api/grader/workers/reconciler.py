"""Orphan-task reconciler: rescue submissions stuck in PROCESSING.

The submit endpoint's 503 fix handles synchronous broker failures (the
broker rejects the enqueue, the row never leaves PENDING). But a Celery
worker that crashes mid-pipeline — OOM, SIGKILL, segfault from torch,
host reboot — can leave a row in PROCESSING with no task to drive it
forward. Without this beat, those rows are wedged forever; the user
never sees a result and never sees a failure.

This task scans for PROCESSING rows whose `updated_at` is older than
`STALE_AFTER` and flips them to FAILED with an audit-log entry. It is
intentionally NOT a re-enqueue: a 5-min-old orphan probably ran far
enough into the heavy ML pipeline (DINOv2 embed, dewarp, grading) that
re-running risks duplicating expensive work. Failing the row out and
letting the user re-submit is the conservative call.

Concurrency safety: the SELECT uses `FOR UPDATE SKIP LOCKED` so two
reconciler runs in flight (e.g. beat misfire, two API replicas with
embedded beat) don't double-process the same row. SKIP LOCKED is
specifically what makes the second runner take a different slice
instead of waiting for the first to commit. Combined with the 50-row
LIMIT, each run is bounded; backlog drains across consecutive ticks.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from grader.db.models import AuditLog, Submission, SubmissionStatus
from grader.db.session import async_session_maker
from grader.workers.celery_app import celery_app

log = logging.getLogger(__name__)

# A pipeline run that hasn't touched its submission row for 5 minutes is
# considered orphaned. The pipeline runner flushes audit-log rows at
# every stage transition, which bumps `submission.updated_at` via the
# session's onupdate hook on related writes — so a healthy long-running
# task won't be flagged. Tune upward if a single stage ever exceeds 5min.
STALE_AFTER = timedelta(minutes=5)

# Per-tick row cap. Bounded so a sudden backlog (e.g. broker outage
# recovery) doesn't turn the reconciler into a long-running transaction
# that holds locks across the whole submissions table. The next tick
# picks up where this one left off.
RECONCILE_BATCH_LIMIT = 50

RECONCILE_REASON = "worker_timeout_reconciled"
RECONCILE_AUDIT_ACTION = "reconcile_stale_processing"


async def reconcile_stale_submissions_async(db: AsyncSession) -> int:
    """Scan for stale PROCESSING rows, flip to FAILED, and audit each.

    Returns the number of rows reconciled. The caller owns the session
    and the transaction boundary — this matches the `run_pipeline` /
    `run_pipeline_sync` split so tests can call this directly with the
    per-test rolled-back session and the sync wrapper can open + commit
    its own session.
    """
    now = datetime.now(timezone.utc)
    cutoff = now - STALE_AFTER

    stale = (
        await db.execute(
            select(Submission)
            .where(Submission.status == SubmissionStatus.PROCESSING)
            .where(Submission.updated_at < cutoff)
            .with_for_update(skip_locked=True)
            .limit(RECONCILE_BATCH_LIMIT)
        )
    ).scalars().all()

    if not stale:
        return 0

    for sub in stale:
        stale_for_seconds = (now - sub.updated_at).total_seconds()
        sub.status = SubmissionStatus.FAILED
        # Don't clobber a pre-existing rejection_reason — it would be
        # bizarre to find one on a PROCESSING row, but if the runner
        # somehow set one before crashing, we keep it as the more
        # specific signal and only fill in our generic reason when the
        # field is empty.
        if not sub.rejection_reason:
            sub.rejection_reason = RECONCILE_REASON
        db.add(
            AuditLog(
                submission_id=sub.id,
                actor="reconciler",
                action=RECONCILE_AUDIT_ACTION,
                payload={"stale_for_seconds": stale_for_seconds},
            )
        )

    await db.flush()
    log.info("reconciler: flipped %d stale PROCESSING rows to FAILED", len(stale))
    return len(stale)


def _run_async(
    session_factory: async_sessionmaker[AsyncSession] | None = None,
) -> int:
    # Resolve the factory at call time (not at function-def time) so
    # tests can monkeypatch `reconciler.async_session_maker` and have
    # the swap take effect for `.run()` invocations.
    factory = session_factory if session_factory is not None else async_session_maker

    async def _runner() -> int:
        async with factory() as session:
            try:
                count = await reconcile_stale_submissions_async(session)
                await session.commit()
                return count
            except Exception:
                await session.rollback()
                raise

    return asyncio.run(_runner())


@celery_app.task(name="grader.workers.reconciler.reconcile_stale_submissions")
def reconcile_stale_submissions() -> int:
    """Celery entrypoint. Opens its own session, commits on success."""
    # Read the module-level binding fresh on each call so tests can
    # monkeypatch it. Pass it through explicitly to `_run_async`.
    from grader.workers import reconciler as _self

    return _run_async(_self.async_session_maker)
