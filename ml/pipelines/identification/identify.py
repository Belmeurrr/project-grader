"""Identification orchestrator.

Combines pHash pre-filter, embedding nearest-neighbor lookup, and a
ranking step into a single `identify` call. Returns up to top_k
candidates with a calibrated confidence and the chosen best match
(None if no candidate meets the confidence floor).

Confidence calibration:
  - pHash exact match (distance == 0):                conf = 1.00
  - pHash close match (distance ≤ 2):                  conf = 0.95
  - pHash near match  (distance ≤ 6):                  conf = 0.85
  - Embedding top-1 with margin > 0.10 over top-2:    conf = top1_sim
  - Otherwise:                                         conf scaled by gap

The actual numbers above are tunable; what matters is the calibration is
*monotonic* with how confident a real grader would be. The API surfaces
this number directly; the UI shows "Identified" vs "Possible match" vs
"Unable to identify" based on bands."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from pipelines.identification.catalog import CardCatalogEntry, CatalogHit, CatalogIndex
from pipelines.identification.embedding import ImageEmbedder
from pipelines.identification.hashing import compute_dhash, compute_phash, hamming_distance


@dataclass(frozen=True)
class IdentificationCandidate:
    entry: CardCatalogEntry
    score: float                 # higher = better, calibrated to [0, 1]
    method: str                  # "phash" | "embedding"
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class IdentificationResult:
    candidates: list[IdentificationCandidate]
    chosen: IdentificationCandidate | None
    confidence: float            # confidence in `chosen`, or 0 if None

    @property
    def identified(self) -> bool:
        return self.chosen is not None


# Tunable thresholds. Live here, not in settings, because changing them
# changes model behavior — they need to be versioned with the algorithm.
PHASH_EXACT_MAX_DIST = 0
PHASH_CLOSE_MAX_DIST = 2
PHASH_NEAR_MAX_DIST = 6
EMBEDDING_DEFAULT_TOP_K = 5
MIN_ACCEPT_CONFIDENCE = 0.55


def identify(
    canonical_image: NDArray[np.uint8],
    catalog: CatalogIndex,
    embedder: ImageEmbedder,
    top_k: int = EMBEDDING_DEFAULT_TOP_K,
    min_accept_confidence: float = MIN_ACCEPT_CONFIDENCE,
) -> IdentificationResult:
    """Run identification on a dewarped canonical card image.

    The image is the output of detection + dewarp (Stages 1 and 2), so
    it's already 750x1050 BGR uint8 in canonical orientation."""
    if canonical_image.dtype != np.uint8:
        raise ValueError(f"expected uint8 image, got {canonical_image.dtype}")

    phash = compute_phash(canonical_image)
    phash_hits = catalog.find_by_phash(phash, max_distance=PHASH_NEAR_MAX_DIST, limit=top_k * 2)

    # Short-circuit on a confident pHash exact match.
    if phash_hits and phash_hits[0].distance <= PHASH_EXACT_MAX_DIST:
        cand = _candidate_from_phash(phash_hits[0])
        return IdentificationResult(
            candidates=[cand],
            chosen=cand,
            confidence=cand.score,
        )

    # Always also fetch embedding neighbors — pHash near-misses are the
    # primary failure mode when foils, parallels, or lighting shift bits.
    embedding = embedder.encode(canonical_image)
    embedding_hits = catalog.nearest_by_embedding(embedding, top_k=top_k)

    candidates = _merge_and_rank(phash_hits, embedding_hits)[:top_k]
    if not candidates:
        return IdentificationResult(candidates=[], chosen=None, confidence=0.0)

    chosen = candidates[0]
    confidence = _calibrate_confidence(chosen, candidates)
    if confidence < min_accept_confidence:
        return IdentificationResult(
            candidates=candidates,
            chosen=None,
            confidence=confidence,
        )
    return IdentificationResult(
        candidates=candidates,
        chosen=chosen,
        confidence=confidence,
    )


def _candidate_from_phash(hit: CatalogHit) -> IdentificationCandidate:
    d = int(hit.distance)
    if d <= PHASH_EXACT_MAX_DIST:
        score = 1.0
    elif d <= PHASH_CLOSE_MAX_DIST:
        score = 0.95
    elif d <= PHASH_NEAR_MAX_DIST:
        score = 0.85 - 0.02 * (d - PHASH_CLOSE_MAX_DIST)
    else:
        score = 0.50
    return IdentificationCandidate(
        entry=hit.entry,
        score=float(score),
        method="phash",
        metadata={"phash_distance": d},
    )


def _candidate_from_embedding(hit: CatalogHit) -> IdentificationCandidate:
    # cosine distance is 1 - cos_sim; convert to similarity (higher better).
    sim = max(0.0, 1.0 - float(hit.distance))
    return IdentificationCandidate(
        entry=hit.entry,
        score=sim,
        method="embedding",
        metadata={"cosine_distance": float(hit.distance)},
    )


def _merge_and_rank(
    phash_hits: list[CatalogHit],
    embedding_hits: list[CatalogHit],
) -> list[IdentificationCandidate]:
    """Combine pHash and embedding candidates, dedupe by variant_id, and
    return them sorted by score descending. When the same variant appears
    in both lists, take the higher-scoring source."""
    by_id: dict[str, IdentificationCandidate] = {}
    for h in phash_hits:
        cand = _candidate_from_phash(h)
        existing = by_id.get(cand.entry.variant_id)
        if existing is None or cand.score > existing.score:
            by_id[cand.entry.variant_id] = cand
    for h in embedding_hits:
        cand = _candidate_from_embedding(h)
        existing = by_id.get(cand.entry.variant_id)
        if existing is None or cand.score > existing.score:
            by_id[cand.entry.variant_id] = cand
    return sorted(by_id.values(), key=lambda c: c.score, reverse=True)


def _calibrate_confidence(
    chosen: IdentificationCandidate,
    ranked: list[IdentificationCandidate],
) -> float:
    """Confidence factors in the chosen candidate's raw score AND its margin
    over the runner-up. A high score with a tiny margin is suspicious — that's
    a parallel or alternate-art collision that needs disambiguation."""
    if chosen.method == "phash" and chosen.score >= 0.95:
        return chosen.score
    if len(ranked) < 2:
        return chosen.score
    margin = chosen.score - ranked[1].score
    # Penalize tight margins; reward wide ones, but bound to chosen.score.
    margin_factor = min(1.0, max(0.5, margin * 5.0 + 0.5))
    return float(chosen.score * margin_factor)
