"""Pipeline runner — orchestrates a full grading job for one submission.

Sequence (v1, centering-only):

  1. Validate the submission has all required quality-passed shots.
  2. For each shot needing detection (front, back), run detect+dewarp and
     persist canonical images to S3.
  3. Run identification on the front canonical.
  4. Run centering measurement and persist a Grade row.
  5. Mark the submission completed.

Failures at any stage transition the submission to FAILED with a reason
written to `submission.rejection_reason`. Each transition is also written
to the audit log so disputes can be reproduced.

This module is sync-callable through `run_pipeline_sync` (used by the
Celery task) and async-callable through `run_pipeline` (used by tests
that already have an event loop and a session).
"""

from __future__ import annotations

import asyncio
import sys
import uuid
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.orm import selectinload

from grader.db.models import (
    AuditLog,
    Grade,
    GradingScheme,
    ShotKind,
    Submission,
    SubmissionShot,
    SubmissionStatus,
)
from grader.services import detection, grading, identification

_ML_ROOT = Path(__file__).resolve().parents[3] / "ml"
if str(_ML_ROOT) not in sys.path:
    sys.path.insert(0, str(_ML_ROOT))

from pipelines.identification import (  # noqa: E402
    CatalogIndex,
    ImageEmbedder,
)


# Shots that v1 (centering-only) requires before grading can run.
# Front is required; back is optional (PSA centering can grade front-only).
REQUIRED_SHOTS: tuple[ShotKind, ...] = (ShotKind.FRONT_FULL,)
OPTIONAL_SHOTS: tuple[ShotKind, ...] = (ShotKind.BACK_FULL,)


@dataclass
class PipelineRunResult:
    submission_id: uuid.UUID
    status: SubmissionStatus
    grades: list[Grade] = field(default_factory=list)
    failure_reason: str | None = None
    identified_variant_id: uuid.UUID | None = None


class PipelineValidationError(Exception):
    """Pre-flight check failed — submission isn't ready to grade."""


# --------------------------------------------------------------------------
# Pre-flight validation
# --------------------------------------------------------------------------


def _shots_by_kind(shots: Iterable[SubmissionShot]) -> dict[ShotKind, SubmissionShot]:
    """Latest shot per kind. The shot register endpoint rejects duplicate
    shot_ids, but a user can re-take a kind by requesting a new shot_id —
    the most recent passing shot wins."""
    out: dict[ShotKind, SubmissionShot] = {}
    for s in shots:
        prior = out.get(s.kind)
        if prior is None or s.created_at > prior.created_at:
            out[s.kind] = s
    return out


def _validate_ready(submission: Submission) -> dict[ShotKind, SubmissionShot]:
    by_kind = _shots_by_kind(submission.shots)

    for required in REQUIRED_SHOTS:
        shot = by_kind.get(required)
        if shot is None:
            raise PipelineValidationError(f"missing required shot: {required.value}")
        if not shot.quality_passed:
            raise PipelineValidationError(
                f"required shot {required.value} did not pass quality gate"
            )

    # Optional shots are accepted only if they passed quality. A failed
    # back is silently dropped — better to grade front-only than to fail
    # the whole submission on a bad back capture.
    for optional in OPTIONAL_SHOTS:
        shot = by_kind.get(optional)
        if shot is not None and not shot.quality_passed:
            by_kind.pop(optional)

    return by_kind


# --------------------------------------------------------------------------
# Stage runners
# --------------------------------------------------------------------------


def _run_detect_and_dewarp(
    shots_by_kind: dict[ShotKind, SubmissionShot],
) -> dict[ShotKind, str]:
    """Run Stage 1 + 2 on each shot that needs a canonical image. Returns a
    map kind → canonical_s3_key. Raises DetectionFailedError on hard fails."""
    canonicals: dict[ShotKind, str] = {}
    for kind in (*REQUIRED_SHOTS, *OPTIONAL_SHOTS):
        shot = shots_by_kind.get(kind)
        if shot is None:
            continue
        result = detection.detect_and_dewarp_shot(shot.s3_key, kind)
        canonicals[kind] = result.canonical_s3_key
    return canonicals


# --------------------------------------------------------------------------
# Top-level runners
# --------------------------------------------------------------------------


async def run_pipeline(
    submission_id: uuid.UUID,
    db: AsyncSession,
    catalog: CatalogIndex,
    embedder: ImageEmbedder,
) -> PipelineRunResult:
    """Run the full grading pipeline for a submission.

    The session is the caller's responsibility — we do NOT commit here.
    The caller (Celery task or test) wraps the call in a transaction and
    decides when to flush/commit/rollback. This keeps test isolation
    clean (per-test rollback) and lets the worker batch DB writes."""
    submission = await db.get(
        Submission,
        submission_id,
        options=[selectinload(Submission.shots), selectinload(Submission.grades)],
    )
    if submission is None:
        raise PipelineValidationError(f"submission {submission_id} not found")

    # Pre-flight: required shots present + quality passed.
    try:
        shots_by_kind = _validate_ready(submission)
    except PipelineValidationError as e:
        return await _mark_failed(submission, db, str(e))

    submission.status = SubmissionStatus.PROCESSING
    db.add(_audit(submission.id, "pipeline.started", {}))
    await db.flush()

    # Stage 1 + 2: detect + dewarp.
    try:
        canonicals = _run_detect_and_dewarp(shots_by_kind)
    except detection.DetectionFailedError as e:
        return await _mark_failed(submission, db, f"detection_failed: {e}")

    if ShotKind.FRONT_FULL not in canonicals:
        return await _mark_failed(
            submission, db, "detection_failed: front canonical not produced"
        )

    db.add(
        _audit(
            submission.id,
            "pipeline.detection.completed",
            {kind.value: key for kind, key in canonicals.items()},
        )
    )
    await db.flush()

    # Stage 3: identification (soft-fail; we still grade unidentified cards).
    front_canonical_key = canonicals[ShotKind.FRONT_FULL]
    try:
        id_outcome = await identification.identify_canonical_for_submission(
            submission_id=submission.id,
            canonical_s3_key=front_canonical_key,
            catalog=catalog,
            embedder=embedder,
            db=db,
        )
        identified_variant = (
            uuid.UUID(id_outcome.result.chosen.entry.variant_id)
            if id_outcome.result.chosen
            else None
        )
    except identification.IdentificationFailedError as e:
        # Hard fail only if the canonical itself is unreadable. Catalog miss
        # is handled inside identify_canonical_for_submission as a soft fail.
        return await _mark_failed(submission, db, f"identification_failed: {e}")

    # Stage 4 (partial): centering.
    try:
        centering_result = grading.grade_centering(
            front_canonical_s3_key=front_canonical_key,
            back_canonical_s3_key=canonicals.get(ShotKind.BACK_FULL),
        )
    except grading.GradingFailedError as e:
        return await _mark_failed(submission, db, f"grading_failed: {e}")

    grade = await grading.persist_centering_grade(
        submission_id=submission.id,
        result=centering_result,
        db=db,
        scheme=GradingScheme.PSA,
    )

    # Stage 4 (partial cont.): edges. Geometric, deterministic, no model. Runs
    # after centering so the Grade row already exists.
    try:
        edges_result = grading.grade_edges(
            front_canonical_s3_key=front_canonical_key,
            back_canonical_s3_key=canonicals.get(ShotKind.BACK_FULL),
        )
    except grading.GradingFailedError as e:
        return await _mark_failed(submission, db, f"grading_edges_failed: {e}")

    grade = await grading.persist_edges_grade(
        submission_id=submission.id,
        result=edges_result,
        db=db,
        scheme=GradingScheme.PSA,
    )

    submission.status = SubmissionStatus.COMPLETED
    submission.completed_at = datetime.now(timezone.utc)
    db.add(
        _audit(
            submission.id,
            "pipeline.completed",
            {
                "centering": centering_result.psa_subgrade,
                "edges": edges_result.psa_subgrade,
                "edges_worse_face": edges_result.worse_face,
                "confidence": grade.confidence,
                "back_centering_used": centering_result.back_measurement is not None,
                "back_edges_used": edges_result.back_measurement is not None,
            },
        )
    )
    await db.flush()

    return PipelineRunResult(
        submission_id=submission.id,
        status=SubmissionStatus.COMPLETED,
        grades=[grade],
        identified_variant_id=identified_variant,
    )


def run_pipeline_sync(
    submission_id: uuid.UUID,
    session_factory: async_sessionmaker[AsyncSession],
    catalog: CatalogIndex,
    embedder: ImageEmbedder,
) -> PipelineRunResult:
    """Sync entrypoint for Celery. Opens its own session, commits on success,
    rolls back on uncaught error (failures handled inside run_pipeline are
    persisted as part of normal flow)."""

    async def _runner() -> PipelineRunResult:
        async with session_factory() as session:
            try:
                result = await run_pipeline(submission_id, session, catalog, embedder)
                await session.commit()
                return result
            except Exception:
                await session.rollback()
                raise

    return asyncio.run(_runner())


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


async def _mark_failed(
    submission: Submission, db: AsyncSession, reason: str
) -> PipelineRunResult:
    submission.status = SubmissionStatus.FAILED
    submission.rejection_reason = reason
    db.add(_audit(submission.id, "pipeline.failed", {"reason": reason}))
    await db.flush()
    return PipelineRunResult(
        submission_id=submission.id,
        status=SubmissionStatus.FAILED,
        failure_reason=reason,
    )


def _audit(submission_id: uuid.UUID, action: str, payload: dict) -> AuditLog:
    return AuditLog(
        submission_id=submission_id,
        actor="pipeline_runner",
        action=action,
        payload=payload,
    )
