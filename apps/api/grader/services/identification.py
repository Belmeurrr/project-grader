"""Identification service — bridges Stage 2 canonical S3 image to the
ml/pipelines/identification module, persists the result to the
submissions row, and writes an audit-log entry.

The grading worker calls `identify_canonical_for_submission` after
detection + dewarp persists `canonical/<kind>.png`. The function:

  1. Pulls the canonical PNG from S3.
  2. Runs the identifier against the configured catalog + embedder.
  3. Updates `submissions.identified_variant_id` and
     `submissions.identification_confidence`.
  4. Writes one audit_log row capturing the top candidates.
"""

from __future__ import annotations

import sys
import uuid
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession

from grader.db.models import AuditLog, Submission
from grader.services import storage

_ML_ROOT = Path(__file__).resolve().parents[4] / "ml"
if str(_ML_ROOT) not in sys.path:
    sys.path.insert(0, str(_ML_ROOT))

from pipelines.identification import (  # noqa: E402
    CatalogIndex,
    IdentificationResult,
    ImageEmbedder,
    identify,
)


@dataclass(frozen=True)
class IdentificationOutcome:
    submission_id: uuid.UUID
    canonical_s3_key: str
    result: IdentificationResult


class IdentificationFailedError(Exception):
    pass


def load_canonical_bgr(s3_key: str) -> np.ndarray:
    raw = storage.get_shot_bytes(s3_key)
    arr = np.frombuffer(raw, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise IdentificationFailedError(f"could not decode {s3_key}")
    return image


async def identify_canonical_for_submission(
    submission_id: uuid.UUID,
    canonical_s3_key: str,
    catalog: CatalogIndex,
    embedder: ImageEmbedder,
    db: AsyncSession,
) -> IdentificationOutcome:
    """Run identification against the canonical image and persist the result.

    Note: we pass `db` in rather than opening one ourselves because the
    grading pipeline batches several DB writes in a single transaction
    around this call (audit log, submission update, downstream stage
    inputs)."""
    image = load_canonical_bgr(canonical_s3_key)
    result = identify(image, catalog=catalog, embedder=embedder)

    submission = await db.get(Submission, submission_id)
    if submission is None:
        raise IdentificationFailedError(f"submission {submission_id} not found")

    if result.chosen is not None:
        submission.identified_variant_id = uuid.UUID(result.chosen.entry.variant_id)
        submission.identification_confidence = float(result.confidence)
    else:
        submission.identified_variant_id = None
        submission.identification_confidence = float(result.confidence)

    db.add(
        AuditLog(
            submission_id=submission_id,
            actor="identification_service",
            action="identification.completed",
            payload={
                "canonical_s3_key": canonical_s3_key,
                "identified": result.identified,
                "confidence": float(result.confidence),
                "candidates": [
                    {
                        "variant_id": c.entry.variant_id,
                        "name": c.entry.name,
                        "score": c.score,
                        "method": c.method,
                    }
                    for c in result.candidates
                ],
            },
        )
    )
    return IdentificationOutcome(
        submission_id=submission_id,
        canonical_s3_key=canonical_s3_key,
        result=result,
    )
