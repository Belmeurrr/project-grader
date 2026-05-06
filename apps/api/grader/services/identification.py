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

import uuid
from dataclasses import dataclass

import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession

from grader.db.models import AuditLog, Submission
from grader.services import storage
from pipelines.identification import (
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


def _load_canonical_bgr(s3_key: str) -> np.ndarray:
    """Service-typed wrapper around storage.load_canonical_bgr.

    Catches the storage-level CanonicalLoadError and re-raises as an
    IdentificationFailedError so the grading worker can route load
    failures the same as identification failures (matches the pattern
    used by services/grading.py and services/counterfeit.py)."""
    try:
        return storage.load_canonical_bgr(s3_key)
    except storage.CanonicalLoadError as e:
        raise IdentificationFailedError(str(e)) from e


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
    image = _load_canonical_bgr(canonical_s3_key)
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
