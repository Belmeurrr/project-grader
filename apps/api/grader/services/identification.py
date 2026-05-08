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

import asyncio
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
)
from pipelines.identification.hashing import compute_phash
from pipelines.identification.identify import (
    EMBEDDING_DEFAULT_TOP_K,
    MIN_ACCEPT_CONFIDENCE,
    PHASH_EXACT_MAX_DIST,
    PHASH_NEAR_MAX_DIST,
    _calibrate_confidence,
    _candidate_from_phash,
    _merge_and_rank,
)


@dataclass(frozen=True)
class IdentificationOutcome:
    submission_id: uuid.UUID
    canonical_s3_key: str
    result: IdentificationResult


class IdentificationFailedError(Exception):
    pass


async def _identify_async(
    image: np.ndarray,
    catalog: CatalogIndex,
    embedder: ImageEmbedder,
) -> IdentificationResult:
    """Async port of `pipelines.identification.identify.identify`.
    Calls the catalog's native async methods so we don't cross-loop
    its asyncpg connections."""
    # The catalog Protocol type hint is sync; we cast to the concrete
    # PgVectorCatalogIndex via runtime access to its `_…_async` methods.
    cat = catalog
    if image.dtype != np.uint8:
        raise ValueError(f"expected uint8 image, got {image.dtype}")

    phash = compute_phash(image)
    phash_hits = await cat._find_by_phash_async(  # type: ignore[attr-defined]
        phash, max_distance=PHASH_NEAR_MAX_DIST, limit=EMBEDDING_DEFAULT_TOP_K * 2
    )

    if phash_hits and phash_hits[0].distance <= PHASH_EXACT_MAX_DIST:
        cand = _candidate_from_phash(phash_hits[0])
        return IdentificationResult(
            candidates=[cand], chosen=cand, confidence=cand.score
        )

    # Embedder.encode is a synchronous DinoV2 forward pass — bounce
    # to a thread so it doesn't block the event loop.
    embedding = await asyncio.to_thread(embedder.encode, image)
    embedding_hits = await cat._nearest_by_embedding_async(  # type: ignore[attr-defined]
        embedding, top_k=EMBEDDING_DEFAULT_TOP_K
    )

    candidates = _merge_and_rank(phash_hits, embedding_hits)[
        :EMBEDDING_DEFAULT_TOP_K
    ]
    if not candidates:
        return IdentificationResult(
            candidates=[],
            chosen=None,
            confidence=0.0,
            submitted_embedding=embedding,
        )

    chosen = candidates[0]
    confidence = _calibrate_confidence(chosen, candidates)
    if confidence < MIN_ACCEPT_CONFIDENCE:
        return IdentificationResult(
            candidates=candidates,
            chosen=None,
            confidence=confidence,
            submitted_embedding=embedding,
        )
    return IdentificationResult(
        candidates=candidates,
        chosen=chosen,
        confidence=confidence,
        submitted_embedding=embedding,
    )


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
    # We can't call the sync `identify(...)` here because its catalog
    # wrappers spawn `asyncio.run(...)` for the DB calls, and those new
    # loops can't share asyncpg connections with the worker's main
    # loop. Bouncing through `asyncio.to_thread` doesn't fix it either
    # (the new thread's loop still isn't where the asyncpg engine
    # lives). So drive the identification flow async, calling the
    # catalog's private async methods directly. The compute-heavy bits
    # (embedder.encode) get bumped to a thread pool.
    result = await _identify_async(image, catalog=catalog, embedder=embedder)

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
