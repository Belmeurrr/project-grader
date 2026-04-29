"""Grading service.

Computes per-criterion subgrades and a final scheme-specific grade for a
submission whose canonical images have already been produced by Stage 2.
Currently implements centering only — corners, edges, surface are
placeholders that will be filled in as their respective ML models land.

The PSA-style final grade is `min(centering, corners, edges, surface)`
with the published rounding rules. When some criteria are missing
(partial grading run), the final grade is left as None: the API surfaces
that as "partial grade" rather than fabricating a number.
"""

from __future__ import annotations

import sys
import uuid
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from grader.db.models import Grade, GradingScheme, ShotKind, SubmissionShot
from grader.services import storage

_ML_ROOT = Path(__file__).resolve().parents[4] / "ml"
if str(_ML_ROOT) not in sys.path:
    sys.path.insert(0, str(_ML_ROOT))

from pipelines.grading.centering import (  # noqa: E402
    CenteringMeasurement,
    measure_centering,
    psa_subgrade_from_ratios,
)
from pipelines.grading.edges import (  # noqa: E402
    EdgeMeasurement,
    measure_edges,
)


@dataclass(frozen=True)
class CenteringGradeResult:
    """Output of grade_centering. Mirrors what gets persisted on the Grade row."""

    front_measurement: CenteringMeasurement
    back_measurement: CenteringMeasurement | None
    psa_subgrade: float
    confidence: float


@dataclass(frozen=True)
class EdgesGradeResult:
    """Output of grade_edges. Per-face measurements + the worse subgrade.

    PSA evaluates edges across both faces of a card; the worse face dominates
    the grade. We persist the worse subgrade and remember which face it came
    from in the model_versions metadata for forensic review."""

    front_measurement: EdgeMeasurement
    back_measurement: EdgeMeasurement | None
    psa_subgrade: float
    worse_face: str  # "front" or "back"
    confidence: float


class GradingFailedError(Exception):
    pass


def _load_canonical_bgr(s3_key: str) -> np.ndarray:
    raw = storage.get_shot_bytes(s3_key)
    arr = np.frombuffer(raw, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise GradingFailedError(f"could not decode canonical at {s3_key}")
    return image


def grade_centering(
    front_canonical_s3_key: str,
    back_canonical_s3_key: str | None = None,
) -> CenteringGradeResult:
    """Run the centering measurement. Front canonical is required; back is optional.

    The PSA subgrade uses front-only when no back is available (per PSA's
    looser back tolerances and the front-dominant scoring rule)."""
    front_img = _load_canonical_bgr(front_canonical_s3_key)
    front = measure_centering(front_img)

    back: CenteringMeasurement | None = None
    if back_canonical_s3_key:
        back_img = _load_canonical_bgr(back_canonical_s3_key)
        back = measure_centering(back_img)

    psa = psa_subgrade_from_ratios(
        front=front.ratios,
        back=back.ratios if back else None,
    )

    # Confidence: average of front + back inner-border-detection confidences.
    confidences = [front.confidence] + ([back.confidence] if back else [])
    confidence = float(sum(confidences) / len(confidences))

    return CenteringGradeResult(
        front_measurement=front,
        back_measurement=back,
        psa_subgrade=psa,
        confidence=confidence,
    )


def grade_edges(
    front_canonical_s3_key: str,
    back_canonical_s3_key: str | None = None,
) -> EdgesGradeResult:
    """Run the edges geometric measurement on front and (optionally) back.

    Both faces are graded independently; the *worse* of the two is the
    final edges subgrade — that's how PSA scores edges in practice."""
    front_img = _load_canonical_bgr(front_canonical_s3_key)
    front = measure_edges(front_img)

    back: EdgeMeasurement | None = None
    if back_canonical_s3_key:
        back_img = _load_canonical_bgr(back_canonical_s3_key)
        back = measure_edges(back_img)

    if back is None or front.psa_subgrade <= back.psa_subgrade:
        worse_face, worse_subgrade = "front", front.psa_subgrade
    else:
        worse_face, worse_subgrade = "back", back.psa_subgrade

    confidences = [front.confidence] + ([back.confidence] if back else [])
    return EdgesGradeResult(
        front_measurement=front,
        back_measurement=back,
        psa_subgrade=worse_subgrade,
        worse_face=worse_face,
        confidence=float(sum(confidences) / len(confidences)),
    )


# --------------------------------------------------------------------------
# Final grade composition
# --------------------------------------------------------------------------


def compute_psa_final(
    centering: float,
    corners: float | None,
    edges: float | None,
    surface: float | None,
) -> float | None:
    """PSA's final-grade rule is essentially `min` across criteria, with
    half-step rounding. Returns None if any required criterion is missing
    (we never fabricate a number).
    """
    components = [centering, corners, edges, surface]
    if any(c is None for c in components):
        return None
    final = min(c for c in components if c is not None)
    # PSA grades are whole or half steps; clamp to that grid.
    return float(round(final * 2) / 2)


# --------------------------------------------------------------------------
# Grade persistence
# --------------------------------------------------------------------------


async def persist_centering_grade(
    submission_id: uuid.UUID,
    result: CenteringGradeResult,
    db: AsyncSession,
    *,
    scheme: GradingScheme = GradingScheme.PSA,
    model_versions: dict[str, str] | None = None,
) -> Grade:
    """Insert or update the centering portion of a Grade row for the submission.

    A Grade row is unique per (submission_id, scheme) — see the unique index
    `ix_grades_submission_scheme`. If a row already exists for this scheme
    (e.g. a previous partial run), update its centering fields rather than
    inserting a duplicate that would violate the constraint."""
    existing = await db.scalar(
        select(Grade).where(
            Grade.submission_id == submission_id,
            Grade.scheme == scheme,
        )
    )

    f = result.front_measurement.ratios
    b = result.back_measurement.ratios if result.back_measurement else None

    fields = {
        "centering": result.psa_subgrade,
        "front_centering_lr": float(f.left),
        "front_centering_tb": float(f.top),
        "back_centering_lr": float(b.left) if b else None,
        "back_centering_tb": float(b.top) if b else None,
        "confidence": result.confidence,
        "model_versions": {"centering": "geometric-v1", **(model_versions or {})},
    }
    fields["final"] = compute_psa_final(
        centering=result.psa_subgrade,
        corners=existing.corners if existing else None,
        edges=existing.edges if existing else None,
        surface=existing.surface if existing else None,
    )

    if existing is None:
        grade = Grade(
            submission_id=submission_id,
            scheme=scheme,
            **fields,
        )
        db.add(grade)
        await db.flush()
        return grade

    for key, value in fields.items():
        setattr(existing, key, value)
    await db.flush()
    return existing


async def persist_edges_grade(
    submission_id: uuid.UUID,
    result: EdgesGradeResult,
    db: AsyncSession,
    *,
    scheme: GradingScheme = GradingScheme.PSA,
    model_versions: dict[str, str] | None = None,
) -> Grade:
    """Update the edges portion of a Grade row for the submission.

    Centering must already be present on the row — it's NOT NULL, and the
    pipeline always runs centering before edges. Calling this without a
    prior centering grade is a programming error and raises."""
    existing = await db.scalar(
        select(Grade).where(
            Grade.submission_id == submission_id,
            Grade.scheme == scheme,
        )
    )
    if existing is None:
        raise GradingFailedError(
            "cannot persist edges grade before centering grade exists"
        )

    existing_versions = dict(existing.model_versions or {})
    existing_versions.update({"edges": "geometric-v1"})
    if model_versions:
        existing_versions.update(model_versions)
    existing_versions["edges_worse_face"] = result.worse_face

    existing.edges = result.psa_subgrade
    # Confidence is the conservative min over criteria contributed so far.
    existing.confidence = float(min(existing.confidence, result.confidence))
    existing.model_versions = existing_versions
    existing.final = compute_psa_final(
        centering=existing.centering,
        corners=existing.corners,
        edges=result.psa_subgrade,
        surface=existing.surface,
    )
    await db.flush()
    return existing
