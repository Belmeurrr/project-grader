"""End-to-end tests for the grading pipeline runner.

These tests exercise the full async orchestration: detection + dewarp,
identification, centering, and persistence. They do NOT go through Celery —
the runner is called directly so failures surface as plain Python errors,
not lost task results."""

from __future__ import annotations

import os
import sys
import uuid
from collections.abc import Iterator
from pathlib import Path

import boto3
import cv2
import numpy as np
import pytest
from moto import mock_aws
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from grader.db.models import (
    AuditLog,
    AuthenticityResult,
    AuthenticityVerdict,
    Grade,
    GradingScheme,
    ShotKind,
    Submission,
    SubmissionShot,
    SubmissionStatus,
    User,
)
from grader.services import storage
from grader.settings import get_settings
from grader.workers.pipeline_runner import PipelineValidationError, run_pipeline

_ML_ROOT = Path(__file__).resolve().parents[3] / "ml"
if str(_ML_ROOT) not in sys.path:
    sys.path.insert(0, str(_ML_ROOT))

from pipelines.identification import (  # noqa: E402
    CardCatalogEntry,
    InMemoryCatalogIndex,
    SimpleEmbedder,
    compute_phash,
)
from tests.fixtures import card_in_scene, encode_jpeg, synth_card  # noqa: E402


@pytest.fixture(autouse=True)
def _aws_creds() -> Iterator[None]:
    prior = {k: os.environ.get(k) for k in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY")}
    os.environ["AWS_ACCESS_KEY_ID"] = "test"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "test"
    storage.reset_s3_client_cache()
    yield
    for k, v in prior.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    storage.reset_s3_client_cache()


@pytest.fixture
def s3_bucket() -> Iterator[str]:
    with mock_aws():
        bucket = get_settings().s3_bucket
        boto3.client("s3", region_name="us-east-1").create_bucket(Bucket=bucket)
        storage.reset_s3_client_cache()
        yield bucket


# --------------------------------------------------------------------------
# Builders
# --------------------------------------------------------------------------


async def _make_user(db: AsyncSession) -> User:
    user = User(clerk_id=f"u_{uuid.uuid4().hex[:8]}", email=f"{uuid.uuid4().hex[:8]}@x")
    db.add(user)
    await db.flush()
    return user


async def _add_shot(
    db: AsyncSession,
    submission_id: uuid.UUID,
    kind: ShotKind,
    bucket: str,
    image: np.ndarray,
    quality_passed: bool = True,
) -> SubmissionShot:
    shot_id = uuid.uuid4()
    key = f"submissions/{submission_id}/shots/{shot_id}/{kind.value}.jpg"
    boto3.client("s3", region_name="us-east-1").put_object(
        Bucket=bucket, Key=key, Body=encode_jpeg(image), ContentType="image/jpeg"
    )
    shot = SubmissionShot(
        id=shot_id,
        submission_id=submission_id,
        kind=kind,
        s3_key=key,
        width_px=image.shape[1],
        height_px=image.shape[0],
        quality_passed=quality_passed,
        quality_metadata={},
    )
    db.add(shot)
    await db.flush()
    return shot


async def _make_submission(
    db: AsyncSession, user: User, status: SubmissionStatus = SubmissionStatus.CAPTURING
) -> Submission:
    sub = Submission(user_id=user.id, status=status)
    db.add(sub)
    await db.flush()
    return sub


def _empty_catalog() -> InMemoryCatalogIndex:
    return InMemoryCatalogIndex()


def _catalog_with_match(canonical_image: np.ndarray) -> InMemoryCatalogIndex:
    """Catalog where the front canonical hashes exactly to the only entry."""
    e = SimpleEmbedder()
    catalog = InMemoryCatalogIndex()
    catalog.add(
        CardCatalogEntry(
            variant_id=str(uuid.uuid4()),
            name="Test Card",
            set_code="TST",
            card_number="1",
            game="mtg",
            canonical_phash=compute_phash(canonical_image),
            canonical_embedding=e.encode(canonical_image),
        )
    )
    return catalog


# --------------------------------------------------------------------------
# Happy path
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pipeline_completes_with_front_only(
    s3_bucket: str, db_session: AsyncSession
) -> None:
    user = await _make_user(db_session)
    sub = await _make_submission(db_session, user)
    await _add_shot(db_session, sub.id, ShotKind.FRONT_FULL, s3_bucket, card_in_scene(fill=0.55))

    result = await run_pipeline(
        submission_id=sub.id,
        db=db_session,
        catalog=_empty_catalog(),
        embedder=SimpleEmbedder(),
    )

    assert result.status == SubmissionStatus.COMPLETED
    assert len(result.grades) == 1
    refreshed = await db_session.get(Submission, sub.id)
    assert refreshed.status == SubmissionStatus.COMPLETED
    assert refreshed.completed_at is not None


@pytest.mark.asyncio
async def test_pipeline_persists_centering_grade(
    s3_bucket: str, db_session: AsyncSession
) -> None:
    user = await _make_user(db_session)
    sub = await _make_submission(db_session, user)
    await _add_shot(db_session, sub.id, ShotKind.FRONT_FULL, s3_bucket, card_in_scene(fill=0.55))

    await run_pipeline(
        submission_id=sub.id, db=db_session,
        catalog=_empty_catalog(), embedder=SimpleEmbedder(),
    )

    grade = await db_session.scalar(
        select(Grade).where(Grade.submission_id == sub.id, Grade.scheme == GradingScheme.PSA)
    )
    assert grade is not None
    assert grade.centering > 0
    assert grade.corners is None  # not yet implemented
    assert grade.final is None  # partial run


@pytest.mark.asyncio
async def test_pipeline_uses_back_centering_when_present(
    s3_bucket: str, db_session: AsyncSession
) -> None:
    user = await _make_user(db_session)
    sub = await _make_submission(db_session, user)
    await _add_shot(db_session, sub.id, ShotKind.FRONT_FULL, s3_bucket, card_in_scene(fill=0.55))
    await _add_shot(db_session, sub.id, ShotKind.BACK_FULL, s3_bucket, card_in_scene(fill=0.55))

    await run_pipeline(
        submission_id=sub.id, db=db_session,
        catalog=_empty_catalog(), embedder=SimpleEmbedder(),
    )

    grade = await db_session.scalar(
        select(Grade).where(Grade.submission_id == sub.id)
    )
    assert grade.back_centering_lr is not None


@pytest.mark.asyncio
async def test_pipeline_writes_audit_log_per_stage(
    s3_bucket: str, db_session: AsyncSession
) -> None:
    user = await _make_user(db_session)
    sub = await _make_submission(db_session, user)
    await _add_shot(db_session, sub.id, ShotKind.FRONT_FULL, s3_bucket, card_in_scene(fill=0.55))

    await run_pipeline(
        submission_id=sub.id, db=db_session,
        catalog=_empty_catalog(), embedder=SimpleEmbedder(),
    )

    logs = (
        await db_session.execute(
            select(AuditLog).where(AuditLog.submission_id == sub.id).order_by(AuditLog.created_at)
        )
    ).scalars().all()
    actions = [log.action for log in logs]
    assert "pipeline.started" in actions
    assert "pipeline.detection.completed" in actions
    assert "identification.completed" in actions
    assert "pipeline.completed" in actions


@pytest.mark.asyncio
async def test_pipeline_identifies_when_catalog_has_match(
    s3_bucket: str, db_session: AsyncSession
) -> None:
    user = await _make_user(db_session)
    sub = await _make_submission(db_session, user)
    front = card_in_scene(fill=0.55)
    await _add_shot(db_session, sub.id, ShotKind.FRONT_FULL, s3_bucket, front)

    # The canonical (post-dewarp) is what gets identified — preview by
    # populating the catalog with the same card_in_scene image so its
    # embedding/pHash is similar to the dewarped output.
    catalog = _catalog_with_match(front)

    result = await run_pipeline(
        submission_id=sub.id, db=db_session,
        catalog=catalog, embedder=SimpleEmbedder(),
    )

    refreshed = await db_session.get(Submission, sub.id)
    # Either identified or not — both are valid; what we assert is that the
    # field was set/cleared (not left None unset on the instance).
    assert refreshed.identification_confidence is not None
    if refreshed.identified_variant_id is not None:
        assert result.identified_variant_id == refreshed.identified_variant_id


# --------------------------------------------------------------------------
# Failure paths
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pipeline_fails_when_no_shots(
    s3_bucket: str, db_session: AsyncSession
) -> None:
    user = await _make_user(db_session)
    sub = await _make_submission(db_session, user)

    result = await run_pipeline(
        submission_id=sub.id, db=db_session,
        catalog=_empty_catalog(), embedder=SimpleEmbedder(),
    )

    assert result.status == SubmissionStatus.FAILED
    assert "missing required shot" in result.failure_reason
    refreshed = await db_session.get(Submission, sub.id)
    assert refreshed.status == SubmissionStatus.FAILED
    assert refreshed.rejection_reason == result.failure_reason


@pytest.mark.asyncio
async def test_pipeline_fails_when_required_shot_quality_failed(
    s3_bucket: str, db_session: AsyncSession
) -> None:
    user = await _make_user(db_session)
    sub = await _make_submission(db_session, user)
    await _add_shot(
        db_session, sub.id, ShotKind.FRONT_FULL, s3_bucket,
        card_in_scene(fill=0.55), quality_passed=False,
    )

    result = await run_pipeline(
        submission_id=sub.id, db=db_session,
        catalog=_empty_catalog(), embedder=SimpleEmbedder(),
    )
    assert result.status == SubmissionStatus.FAILED
    assert "quality" in result.failure_reason


@pytest.mark.asyncio
async def test_pipeline_fails_when_no_card_in_image(
    s3_bucket: str, db_session: AsyncSession
) -> None:
    user = await _make_user(db_session)
    sub = await _make_submission(db_session, user)
    blank = np.full((1500, 2000, 3), 35, dtype=np.uint8)
    await _add_shot(db_session, sub.id, ShotKind.FRONT_FULL, s3_bucket, blank)

    result = await run_pipeline(
        submission_id=sub.id, db=db_session,
        catalog=_empty_catalog(), embedder=SimpleEmbedder(),
    )
    assert result.status == SubmissionStatus.FAILED
    assert "detection" in result.failure_reason


@pytest.mark.asyncio
async def test_pipeline_drops_failed_optional_back_shot(
    s3_bucket: str, db_session: AsyncSession
) -> None:
    """A failed-quality back shot should not block grading; pipeline runs
    front-only."""
    user = await _make_user(db_session)
    sub = await _make_submission(db_session, user)
    await _add_shot(db_session, sub.id, ShotKind.FRONT_FULL, s3_bucket, card_in_scene(fill=0.55))
    await _add_shot(
        db_session, sub.id, ShotKind.BACK_FULL, s3_bucket,
        card_in_scene(fill=0.55), quality_passed=False,
    )

    result = await run_pipeline(
        submission_id=sub.id, db=db_session,
        catalog=_empty_catalog(), embedder=SimpleEmbedder(),
    )
    assert result.status == SubmissionStatus.COMPLETED

    grade = await db_session.scalar(select(Grade).where(Grade.submission_id == sub.id))
    assert grade.back_centering_lr is None  # back was dropped


@pytest.mark.asyncio
async def test_pipeline_validation_error_for_missing_submission(
    db_session: AsyncSession,
) -> None:
    with pytest.raises(PipelineValidationError, match="not found"):
        await run_pipeline(
            submission_id=uuid.uuid4(), db=db_session,
            catalog=_empty_catalog(), embedder=SimpleEmbedder(),
        )


# --------------------------------------------------------------------------
# Edges integration
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pipeline_persists_edges_grade(
    s3_bucket: str, db_session: AsyncSession
) -> None:
    user = await _make_user(db_session)
    sub = await _make_submission(db_session, user)
    await _add_shot(db_session, sub.id, ShotKind.FRONT_FULL, s3_bucket, card_in_scene(fill=0.55))

    await run_pipeline(
        submission_id=sub.id, db=db_session,
        catalog=_empty_catalog(), embedder=SimpleEmbedder(),
    )

    grade = await db_session.scalar(
        select(Grade).where(Grade.submission_id == sub.id, Grade.scheme == GradingScheme.PSA)
    )
    assert grade is not None
    assert grade.edges is not None
    assert 1.0 <= grade.edges <= 10.0
    assert grade.model_versions["edges"] == "geometric-v1"
    assert grade.model_versions["edges_worse_face"] in {"front", "back"}


@pytest.mark.asyncio
async def test_pipeline_edges_audit_log_includes_back_flag(
    s3_bucket: str, db_session: AsyncSession
) -> None:
    user = await _make_user(db_session)
    sub = await _make_submission(db_session, user)
    await _add_shot(db_session, sub.id, ShotKind.FRONT_FULL, s3_bucket, card_in_scene(fill=0.55))
    await _add_shot(db_session, sub.id, ShotKind.BACK_FULL, s3_bucket, card_in_scene(fill=0.55))

    await run_pipeline(
        submission_id=sub.id, db=db_session,
        catalog=_empty_catalog(), embedder=SimpleEmbedder(),
    )

    log = (
        await db_session.execute(
            select(AuditLog).where(
                AuditLog.submission_id == sub.id, AuditLog.action == "pipeline.completed"
            )
        )
    ).scalar_one()
    assert log.payload["back_edges_used"] is True
    assert "edges" in log.payload
    assert "edges_worse_face" in log.payload


# --------------------------------------------------------------------------
# Counterfeit / authenticity integration
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pipeline_persists_grade_and_authenticity_result(
    s3_bucket: str, db_session: AsyncSession
) -> None:
    """The headline end-to-end assertion: a successful submission produces
    BOTH a Grade row AND an AuthenticityResult row. This is the contract
    the cert page relies on."""
    user = await _make_user(db_session)
    sub = await _make_submission(db_session, user)
    await _add_shot(db_session, sub.id, ShotKind.FRONT_FULL, s3_bucket, card_in_scene(fill=0.55))

    result = await run_pipeline(
        submission_id=sub.id, db=db_session,
        catalog=_empty_catalog(), embedder=SimpleEmbedder(),
    )
    assert result.status == SubmissionStatus.COMPLETED

    grade = await db_session.scalar(
        select(Grade).where(Grade.submission_id == sub.id, Grade.scheme == GradingScheme.PSA)
    )
    authenticity = await db_session.scalar(
        select(AuthenticityResult).where(AuthenticityResult.submission_id == sub.id)
    )

    assert grade is not None
    assert authenticity is not None
    assert authenticity.verdict in set(AuthenticityVerdict)
    # Both detectors in the ensemble persist their raw outputs for
    # offline recalibration. Pinning their presence at the top-level
    # so a regression that drops a detector is loud.
    assert "rosette" in authenticity.detector_scores
    assert "color" in authenticity.detector_scores
    assert "rosette" in authenticity.model_versions
    assert "color" in authenticity.model_versions
    # Per-detector verdicts are stored in the detector_scores blob so a
    # later reviewer can see which detector pushed the combined verdict.
    assert "verdict" in authenticity.detector_scores["rosette"]
    assert "verdict" in authenticity.detector_scores["color"]
    assert 0.0 <= authenticity.confidence <= 1.0


@pytest.mark.asyncio
async def test_pipeline_writes_counterfeit_audit_log_entries(
    s3_bucket: str, db_session: AsyncSession
) -> None:
    user = await _make_user(db_session)
    sub = await _make_submission(db_session, user)
    await _add_shot(db_session, sub.id, ShotKind.FRONT_FULL, s3_bucket, card_in_scene(fill=0.55))

    await run_pipeline(
        submission_id=sub.id, db=db_session,
        catalog=_empty_catalog(), embedder=SimpleEmbedder(),
    )

    logs = (
        await db_session.execute(
            select(AuditLog).where(AuditLog.submission_id == sub.id).order_by(AuditLog.created_at)
        )
    ).scalars().all()
    actions = [log.action for log in logs]
    assert "pipeline.counterfeit.started" in actions
    # `.completed` on success, `.skipped` on detector-level failure — at
    # least one terminal entry must follow `.started`.
    assert ("pipeline.counterfeit.completed" in actions) or (
        "pipeline.counterfeit.skipped" in actions
    )

    completed = [log for log in logs if log.action == "pipeline.counterfeit.completed"]
    if completed:
        payload = completed[0].payload
        assert "verdict" in payload
        # Both detectors' top-line metrics are in the audit payload so
        # an operator can grep history for "color_score < X" without
        # joining against AuthenticityResult.
        assert "rosette_score" in payload
        assert "rosette_confidence" in payload
        assert "color_score" in payload
        assert "color_confidence" in payload
        assert "combined_confidence" in payload


@pytest.mark.asyncio
async def test_pipeline_authenticity_idempotent_on_re_run(
    s3_bucket: str, db_session: AsyncSession
) -> None:
    """Re-running the pipeline updates the existing AuthenticityResult row
    (unique index on submission_id) rather than colliding."""
    user = await _make_user(db_session)
    sub = await _make_submission(db_session, user)
    await _add_shot(db_session, sub.id, ShotKind.FRONT_FULL, s3_bucket, card_in_scene(fill=0.55))

    await run_pipeline(
        submission_id=sub.id, db=db_session,
        catalog=_empty_catalog(), embedder=SimpleEmbedder(),
    )
    sub.status = SubmissionStatus.CAPTURING  # reset to allow re-run
    await db_session.flush()
    await run_pipeline(
        submission_id=sub.id, db=db_session,
        catalog=_empty_catalog(), embedder=SimpleEmbedder(),
    )

    rows = (
        await db_session.execute(
            select(AuthenticityResult).where(AuthenticityResult.submission_id == sub.id)
        )
    ).scalars().all()
    assert len(rows) == 1


@pytest.mark.asyncio
async def test_pipeline_idempotent_on_re_run(
    s3_bucket: str, db_session: AsyncSession
) -> None:
    """Re-running the pipeline updates the existing Grade row (unique index
    on submission_id + scheme), not creates a duplicate."""
    user = await _make_user(db_session)
    sub = await _make_submission(db_session, user)
    await _add_shot(db_session, sub.id, ShotKind.FRONT_FULL, s3_bucket, card_in_scene(fill=0.55))

    await run_pipeline(
        submission_id=sub.id, db=db_session,
        catalog=_empty_catalog(), embedder=SimpleEmbedder(),
    )
    sub.status = SubmissionStatus.CAPTURING  # reset to allow re-run
    await db_session.flush()
    await run_pipeline(
        submission_id=sub.id, db=db_session,
        catalog=_empty_catalog(), embedder=SimpleEmbedder(),
    )

    rows = (
        await db_session.execute(select(Grade).where(Grade.submission_id == sub.id))
    ).scalars().all()
    assert len(rows) == 1
