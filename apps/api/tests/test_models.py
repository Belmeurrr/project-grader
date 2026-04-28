"""DB model tests — instantiation, relationships, enum values, cascade behavior.

These hit a real Postgres because the schema uses Postgres-only types
(pgvector, JSONB, native ENUMs)."""

from __future__ import annotations

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from grader.db.models import (
    AuditLog,
    AuthenticityResult,
    AuthenticityVerdict,
    CardSet,
    CardVariant,
    Game,
    Grade,
    GradingScheme,
    ShotKind,
    Submission,
    SubmissionShot,
    SubmissionStatus,
    User,
)


@pytest.mark.asyncio
async def test_user_create_roundtrip(db_session: AsyncSession) -> None:
    user = User(clerk_id="user_test_1", email="t1@example.com", display_name="Test")
    db_session.add(user)
    await db_session.commit()
    await db_session.refresh(user)

    fetched = await db_session.scalar(select(User).where(User.clerk_id == "user_test_1"))
    assert fetched is not None
    assert fetched.id == user.id
    assert fetched.training_consent is False  # server default


@pytest.mark.asyncio
async def test_card_set_and_variant_relationship(db_session: AsyncSession) -> None:
    cs = CardSet(game=Game.MTG, code="LEA", name="Limited Edition Alpha")
    db_session.add(cs)
    await db_session.flush()

    cv = CardVariant(
        game=Game.MTG,
        set_id=cs.id,
        card_number="195",
        name="Black Lotus",
        is_foil=False,
    )
    db_session.add(cv)
    await db_session.commit()
    await db_session.refresh(cs, ["variants"])

    assert len(cs.variants) == 1
    assert cs.variants[0].name == "Black Lotus"


@pytest.mark.asyncio
async def test_submission_shot_cascade_delete(db_session: AsyncSession) -> None:
    user = User(clerk_id="user_cascade", email="cascade@example.com")
    db_session.add(user)
    await db_session.flush()

    sub = Submission(user_id=user.id, status=SubmissionStatus.PENDING)
    db_session.add(sub)
    await db_session.flush()

    shot = SubmissionShot(
        submission_id=sub.id,
        kind=ShotKind.FRONT_FULL,
        s3_key="submissions/x/front.jpg",
        width_px=2000,
        height_px=2800,
    )
    db_session.add(shot)
    await db_session.commit()

    await db_session.delete(sub)
    await db_session.commit()

    remaining = await db_session.scalar(select(SubmissionShot).where(SubmissionShot.id == shot.id))
    assert remaining is None


@pytest.mark.asyncio
async def test_grade_unique_per_scheme(db_session: AsyncSession) -> None:
    user = User(clerk_id="user_grade", email="grade@example.com")
    db_session.add(user)
    await db_session.flush()
    sub = Submission(user_id=user.id, status=SubmissionStatus.COMPLETED)
    db_session.add(sub)
    await db_session.flush()

    g1 = Grade(
        submission_id=sub.id,
        scheme=GradingScheme.PSA,
        centering=10.0,
        corners=10.0,
        edges=10.0,
        surface=10.0,
        final=10.0,
        confidence=0.95,
    )
    db_session.add(g1)
    await db_session.commit()

    g2 = Grade(
        submission_id=sub.id,
        scheme=GradingScheme.PSA,
        centering=9.0,
        corners=9.0,
        edges=9.0,
        surface=9.0,
        final=9.0,
        confidence=0.9,
    )
    db_session.add(g2)
    with pytest.raises(Exception):  # IntegrityError from unique composite index
        await db_session.commit()


@pytest.mark.asyncio
async def test_authenticity_one_to_one(db_session: AsyncSession) -> None:
    user = User(clerk_id="user_auth", email="auth@example.com")
    db_session.add(user)
    await db_session.flush()
    sub = Submission(user_id=user.id, status=SubmissionStatus.COMPLETED)
    db_session.add(sub)
    await db_session.flush()

    db_session.add(
        AuthenticityResult(
            submission_id=sub.id,
            verdict=AuthenticityVerdict.AUTHENTIC,
            confidence=0.92,
            detector_scores={"rosette": 0.95, "color": 0.88},
            reasons=["rosette match", "color in distribution"],
        )
    )
    await db_session.commit()

    db_session.add(
        AuthenticityResult(
            submission_id=sub.id,
            verdict=AuthenticityVerdict.SUSPICIOUS,
            confidence=0.4,
            detector_scores={},
            reasons=[],
        )
    )
    with pytest.raises(Exception):  # unique constraint on submission_id
        await db_session.commit()


@pytest.mark.asyncio
async def test_audit_log_with_jsonb_payload(db_session: AsyncSession) -> None:
    log = AuditLog(actor="system", action="submission.created", payload={"foo": "bar", "n": 1})
    db_session.add(log)
    await db_session.commit()

    fetched = await db_session.scalar(select(AuditLog).where(AuditLog.id == log.id))
    assert fetched is not None
    assert fetched.payload["foo"] == "bar"
    assert fetched.payload["n"] == 1


def test_enum_values_match_migration_constants() -> None:
    """Guards against a model enum drifting from the migration's ENUM type values."""
    import importlib.util
    from pathlib import Path

    migration_path = (
        Path(__file__).parent.parent
        / "alembic"
        / "versions"
        / "20260428_0001_initial_schema.py"
    )
    spec = importlib.util.spec_from_file_location("initial_migration", migration_path)
    assert spec is not None and spec.loader is not None
    migration = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(migration)

    assert tuple(g.value for g in Game) == migration.GAME_VALUES
    assert tuple(s.value for s in SubmissionStatus) == migration.SUBMISSION_STATUS_VALUES
    assert tuple(s.value for s in ShotKind) == migration.SHOT_KIND_VALUES
    assert tuple(s.value for s in GradingScheme) == migration.GRADING_SCHEME_VALUES
    assert tuple(v.value for v in AuthenticityVerdict) == migration.AUTHENTICITY_VERDICT_VALUES
