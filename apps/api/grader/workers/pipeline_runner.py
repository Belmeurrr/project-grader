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
import uuid
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone

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
from grader.services import counterfeit, detection, grading, identification
from grader.settings import get_settings
from pipelines.identification import (
    CatalogIndex,
    ImageEmbedder,
)


# Shots that v1 (centering-only) requires before grading can run.
# Front is required; back is optional (PSA centering can grade front-only).
# Tilt_30 is also optional (the wizard step is opt-in) — used by the
# holographic-parallax counterfeit detector when present.
REQUIRED_SHOTS: tuple[ShotKind, ...] = (ShotKind.FRONT_FULL,)
OPTIONAL_SHOTS: tuple[ShotKind, ...] = (ShotKind.BACK_FULL, ShotKind.TILT_30)


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
    map kind → canonical_s3_key. Raises DetectionFailedError on hard fails
    for REQUIRED + back; tilt_30 is best-effort (a 30° tilt that fails
    detection just means the holographic-parallax detector abstains)."""
    canonicals: dict[ShotKind, str] = {}
    for kind in (*REQUIRED_SHOTS, ShotKind.BACK_FULL):
        shot = shots_by_kind.get(kind)
        if shot is None:
            continue
        result = detection.detect_and_dewarp_shot(shot.s3_key, kind)
        canonicals[kind] = result.canonical_s3_key
    # tilt_30 is by construction a steeper angle than front/back shots —
    # relax the irregularity cap so a 30° tilt isn't rejected by the
    # same gate that protects against bent cards on the straight-on
    # shots. And soft-fail: a wonky tilt capture shouldn't sink a
    # submission; the holographic detector will abstain UNVERIFIED.
    tilt_shot = shots_by_kind.get(ShotKind.TILT_30)
    if tilt_shot is not None:
        try:
            tilt_result = detection.detect_and_dewarp_shot(
                tilt_shot.s3_key, ShotKind.TILT_30, max_irregularity=0.45
            )
            canonicals[ShotKind.TILT_30] = tilt_result.canonical_s3_key
        except detection.DetectionFailedError:
            pass
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
        # Force re-issue of the load options. Without this, if the submission
        # is already in the session's identity map (common in tests, possible
        # in workers that share sessions), `db.get` returns the cached instance
        # and silently drops the selectinload — leaving `submission.shots` to
        # lazy-load from a sync attribute access, which the asyncpg backend
        # can't service.
        populate_existing=True,
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

    # Stage 3.5: counterfeit-authenticity check. Five detectors:
    #   - rosette FFT (image-only)
    #   - color profile (image-only)
    #   - embedding anomaly (depends on identification result + reference
    #     embeddings store; gracefully abstains when either is missing)
    #   - typography (depends on identification's matched card name +
    #     RapidOCR; gracefully abstains when either is missing)
    #   - holographic parallax (depends on the optional tilt_30 shot;
    #     gracefully abstains when tilt isn't captured or no foil region
    #     is detected on the front shot)
    # The combined verdict is the conservative ensemble in
    # `_combine_verdicts`. Soft-fail: a detector blow-up should not block
    # grading; the verdict (including UNVERIFIED on failure) is recorded
    # so the cert page always has *some* authenticity row to display.
    db.add(_audit(submission.id, "pipeline.counterfeit.started", {}))
    await db.flush()
    try:
        rosette_measurement = counterfeit.analyze_rosette(front_canonical_key)
        color_measurement = counterfeit.analyze_color_profile(front_canonical_key)
        chosen = id_outcome.result.chosen
        embedding_measurement = counterfeit.analyze_embedding_anomaly(
            submitted_embedding=id_outcome.result.submitted_embedding,
            manufacturer=chosen.entry.game if chosen is not None else None,
            variant_id=chosen.entry.variant_id if chosen is not None else None,
            references_store_path=get_settings().references_embeddings_path,
        )
        typography_measurement = counterfeit.analyze_typography_service(
            front_canonical_key,
            chosen.entry.name if chosen is not None else None,
        )
        # tilt_30 is optional — pass None to the service when no canonical
        # was produced (either the shot wasn't captured, or detection
        # failed on it). The detector turns None into an UNVERIFIED
        # abstain with abstain_reason='tilt_not_captured'.
        tilt_canonical_key = canonicals.get(ShotKind.TILT_30)
        holographic_measurement = counterfeit.analyze_holographic_service(
            front_canonical_key,
            tilt_canonical_key,
        )
        authenticity = await counterfeit.persist_authenticity_result(
            submission_id=submission.id,
            rosette=rosette_measurement,
            color=color_measurement,
            embedding=embedding_measurement,
            typography=typography_measurement,
            holographic=holographic_measurement,
            db=db,
        )
        db.add(
            _audit(
                submission.id,
                "pipeline.counterfeit.completed",
                {
                    "verdict": authenticity.verdict.value,
                    "rosette_score": float(rosette_measurement.rosette_score),
                    "rosette_confidence": float(rosette_measurement.confidence),
                    "rosette_analyzed_patches": int(
                        rosette_measurement.analyzed_patches
                    ),
                    "color_score": float(color_measurement.color_score),
                    "color_confidence": float(color_measurement.confidence),
                    "color_p95_chroma": float(color_measurement.p95_chroma),
                    "embedding_score": float(embedding_measurement.embedding_score),
                    "embedding_confidence": float(embedding_measurement.confidence),
                    "embedding_n_references": int(
                        embedding_measurement.n_references
                    ),
                    "typography_score": float(typography_measurement.score),
                    "typography_confidence": float(typography_measurement.confidence),
                    "typography_levenshtein": (
                        int(typography_measurement.levenshtein_distance)
                        if typography_measurement.levenshtein_distance is not None
                        else None
                    ),
                    "holographic_score": float(holographic_measurement.score),
                    "holographic_confidence": float(holographic_measurement.confidence),
                    "holographic_flow_ratio": (
                        float(holographic_measurement.flow_ratio)
                        if holographic_measurement.flow_ratio is not None
                        else None
                    ),
                    "holographic_mask_fraction": (
                        float(holographic_measurement.holo_mask_fraction)
                        if holographic_measurement.holo_mask_fraction is not None
                        else None
                    ),
                    "tilt_canonical_present": tilt_canonical_key is not None,
                    "combined_confidence": float(authenticity.confidence),
                },
            )
        )
        await db.flush()
    except counterfeit.CounterfeitFailedError as e:
        db.add(
            _audit(
                submission.id,
                "pipeline.counterfeit.skipped",
                {"reason": str(e)},
            )
        )
        await db.flush()

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
