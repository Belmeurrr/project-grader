"""k-NN reference counterfeit detector (ensemble #1 of 7).

Sibling — and intentionally distinct — counterfeit detector to
embedding-anomaly (#7). Both consume the same `(n, d)` reference
embedding array out of `reference_embeddings.npz`, but they reduce that
array to a score along **different geometric axes**:

  - embedding-anomaly (#7) collapses the reference set to a single
    centroid and scores cosine-distance-to-centroid. Treats the
    authentic exemplars as a single point in embedding space.
  - k-NN reference (#1, here) computes cosine distance from the
    submitted embedding to **every** authentic exemplar, takes the
    `k` smallest, and scores the **mean** of those.

Why both: real authentic exemplars don't sit on top of each other in
embedding space — they cluster on a manifold (different angles, lighting,
wear states). The centroid is a single average of that manifold, so
authentic submissions that land elsewhere on the manifold can be far
from the centroid even though they're near plenty of authentic
exemplars individually. The k-NN signal is robust to that — if the
manifold is locally dense around the submission, the top-k mean
distance is small even when the centroid distance is moderate.
Conversely, a fake that lands inside the centroid's radius (rare but
possible — a counterfeiter who's averaged a lot of training data could
hit the mean) lands far from any specific authentic exemplar, so the
top-k mean catches it where the centroid doesn't.

So this detector is most useful precisely on the cases the centroid
gets wrong, and they coexist as separate signals in the ensemble.
Don't fold them — duplication of "load references; compute distance"
plumbing is cheap; conflating two different reductions of the same
input would lose the cross-check.

Algorithm:
  1. Inputs: a 1-D submitted embedding (d,) and an (n_refs, d) array
     of authentic reference embeddings for the SAME variant.
  2. If submitted is None → abstain UNVERIFIED `no_submitted_embedding`.
  3. If references is None or n_refs < k → abstain UNVERIFIED
     `insufficient_references`. The whole point of "top-k" is that it
     measures local manifold density; with n < k there isn't a top-k
     to take, the signal collapses to "distance to whichever reference
     happened to land closest" (exactly what we DON'T want — that's
     just nearest-neighbor with k=1, which is fragile to a single
     outlier reference).
  4. L2-normalize submitted + every reference row.
  5. Cosine distances `d_i = 1 - dot(submitted_unit, ref_i_unit)`.
  6. Take the k smallest distances; compute their mean = `mean_topk_distance`.
  7. Logistic squash on `(midpoint - mean_topk_distance)` →
     score in [0, 1], higher = more authentic-like.
  8. Confidence: full at n_refs >= 10, scales down to 0.4 at n=k=3,
     abstain (caller treats as UNVERIFIED) below.

Detects: visually-passable counterfeits that DO land near the
centroid but DON'T land near any specific authentic exemplar — i.e.,
fakes built from "average" training data. Also recovers authentic
recall on legitimate angle/lighting variation that pulls a real card
away from the centroid but onto the manifold near other real exemplars.

Does NOT detect: counterfeits that share enough exemplars in the
reference set to land near k of them (extremely rare with the variant-
sized reference budgets we run), or variants we have no references for.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray


# Default `k` for the top-k aggregation. Three is the smallest k where
# "mean of k" is meaningfully more robust than "min of one" — it averages
# out a single noisy outlier reference without diluting the local-density
# signal across the whole reference set. Larger k is also reasonable
# (k=5 is the obvious next step) but the floor on minimum-references is
# fixed at k, so raising the default would push the abstain bar higher.
DEFAULT_K: int = 3

# Logistic squash midpoint + slope for the mean top-k cosine distance.
#
# Empirical anchors against synthetic embeddings (see test_knn_reference_detector):
#   - Submitted ≈ one of the references (perfect manifold hit):
#         mean_topk ≈ 0.001 - 0.01    → score ≈ 0.99
#   - Same variant, near multiple authentic exemplars:
#         mean_topk ≈ 0.05 - 0.15      → score ≈ 0.95 - 0.80
#   - Submitted on the manifold but slightly off the dense cluster:
#         mean_topk ≈ 0.15 - 0.25     → score ≈ 0.50
#   - Different variant or random submission:
#         mean_topk ≈ 0.4 - 0.7       → score ≈ 0.05
#
# Same shape as embedding-anomaly's logistic for consistency: midpoint
# 0.25 places the boundary between "plausible same-variant manifold
# distance" and "you're not actually on this variant's manifold," and
# slope 15 makes the transition firm. Because k-NN distances are smaller
# than centroid distances on the same data (top-k ≤ all-distances by
# construction), the score curve sits slightly more authentic-leaning
# at any given distance — which is fine, because the recalibration
# tool ratchets these per-detector against the real corpus.
#
# These are placeholders; the recalibration tool will ratchet them later.
DEFAULT_DISTANCE_MIDPOINT: float = 0.25
DEFAULT_DISTANCE_SLOPE: float = 15.0


# Confidence ramp by number of references. The detector REQUIRES at
# least k references to compute a top-k at all (below k → abstain). At
# the floor (n_refs == k) the variance in top-k mean across draws of
# the reference set is high, so confidence is bounded down to 0.4 to
# match the ensemble's MIN_CONFIDENCE gate (so n=k still produces a
# usable signal but doesn't carry more weight than the data supports).
# Saturates at 1.0 by n=10, matching the embedding-anomaly ramp.
_CONFIDENCE_BY_NREFS: dict[int, float] = {
    3: 0.4,
    5: 0.7,
    10: 1.0,
}


@dataclass(frozen=True)
class KnnReferenceResult:
    """Output of `analyze_knn_reference`.

    Fields:
      score: [0, 1], higher = more likely authentic. 0.5 in abstain
        paths so the score itself is never confidently wrong; the
        verdict mapper consults `confidence` to decide UNVERIFIED.
      confidence: [0, 1]. Below the ensemble's MIN_CONFIDENCE
        threshold (~0.4) means abstain — see `abstain_reason`.
      mean_topk_distance: mean cosine distance to the k nearest
        authentic exemplars. Provided for debug / forensic review and
        so a future calibration pass can recompute scores from history
        without re-running the embedder.
      n_references_used: how many authentic reference embeddings were
        in the lookup at the moment of scoring (the full set, not just
        the top-k subset). Determines confidence.
      k: the k actually used for the top-k aggregation. Recorded
        separately from `DEFAULT_K` because future per-variant tuning
        may pick a different k for sparse variants.
      abstain_reason: short tag explaining why the detector abstained
        (`no_submitted_embedding`, `insufficient_references`). None on
        the happy path.
      manufacturer_profile: which calibration profile was used. v1
        always "generic"; future versions branch per-manufacturer once
        enough authentic exemplars accumulate per variant.
      metadata: backend extras (per-reference distances, embedding dim,
        etc.) for forensic review and offline calibration analyses.
    """

    score: float
    confidence: float
    mean_topk_distance: float
    n_references_used: int
    k: int
    abstain_reason: Optional[str]
    manufacturer_profile: str = "generic"
    metadata: dict[str, Any] = field(default_factory=dict)


def _logistic(x: float, midpoint: float, slope: float) -> float:
    """Numerically-stable logistic squash to [0, 1]."""
    z = slope * (x - midpoint)
    if z >= 0:
        ez = float(np.exp(-z))
        return 1.0 / (1.0 + ez)
    ez = float(np.exp(z))
    return ez / (1.0 + ez)


def _l2_normalize(v: NDArray[np.float32], eps: float = 1e-12) -> NDArray[np.float32]:
    """Normalize a vector or row-batch of vectors to unit L2 length.

    Same shape as the embedding-anomaly helper — we don't import that
    one across module boundaries because the two detectors are
    intentionally independent (sibling implementations sharing a
    geometric primitive, not one importing the other)."""
    if v.ndim == 1:
        n = float(np.linalg.norm(v))
        return (v / max(n, eps)).astype(np.float32)
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    return (v / np.maximum(norms, eps)).astype(np.float32)


def _confidence_from_n_refs(n: int) -> float:
    """Linearly interpolate confidence between the anchor points in
    `_CONFIDENCE_BY_NREFS`. Returns 0.0 below the lowest anchor (n < k);
    saturates at the highest anchor (n >= 10).

    The lowest anchor is `k` (3 by default) because below `k` we abstain
    outright — interpolation never enters that regime."""
    keys = sorted(_CONFIDENCE_BY_NREFS.keys())
    if n < keys[0]:
        return 0.0
    if n >= keys[-1]:
        return _CONFIDENCE_BY_NREFS[keys[-1]]
    for i in range(len(keys) - 1):
        lo, hi = keys[i], keys[i + 1]
        if lo <= n <= hi:
            if hi == lo:
                return _CONFIDENCE_BY_NREFS[lo]
            t = (n - lo) / (hi - lo)
            return float(
                _CONFIDENCE_BY_NREFS[lo]
                + t * (_CONFIDENCE_BY_NREFS[hi] - _CONFIDENCE_BY_NREFS[lo])
            )
    return _CONFIDENCE_BY_NREFS[keys[-1]]


def _abstain(
    *,
    reason: str,
    n_references_used: int,
    k: int,
    metadata: Optional[dict[str, Any]] = None,
) -> KnnReferenceResult:
    """Build an abstain-shaped KnnReferenceResult. Mirrors the typography
    + holographic detectors' no-signal helpers so the ensemble verdict
    mapper treats the abstain identically across detectors."""
    return KnnReferenceResult(
        score=0.5,
        confidence=0.0,
        mean_topk_distance=0.0,
        n_references_used=n_references_used,
        k=k,
        abstain_reason=reason,
        manufacturer_profile="generic",
        metadata=metadata or {"reason": reason},
    )


def analyze_knn_reference(
    submitted_embedding: NDArray[np.floating] | None,
    reference_embeddings: NDArray[np.floating] | None,
    *,
    k: int = DEFAULT_K,
    distance_midpoint: float = DEFAULT_DISTANCE_MIDPOINT,
    distance_slope: float = DEFAULT_DISTANCE_SLOPE,
) -> KnnReferenceResult:
    """Score a submitted embedding against an authentic reference set
    using the mean cosine distance to the top-k nearest references.

    Args:
        submitted_embedding: 1-D array of shape (d,). Floating-point
            embedding from the identification stage. None → abstain.
        reference_embeddings: 2-D array of shape (n_refs, d) of
            authentic reference embeddings for the SAME VARIANT as the
            submitted card. None or n_refs < k → abstain. Caller is
            responsible for variant-matching (the detector itself
            doesn't know about variants, just embeddings).
        k: number of nearest references to average over. Default 3.
            The minimum-reference count to avoid abstain is also `k`.
        distance_midpoint: cosine-distance value at which the score is
            0.5. Default 0.25 (well above plausible same-variant
            manifold distances, well below cross-variant distances).
        distance_slope: logistic slope. Higher = sharper transition.

    Returns:
        KnnReferenceResult. Always returns a result (never raises) —
        the abstain path is a result with confidence=0, not an
        exception. Mirrors the other detectors' contract.

    Raises:
        ValueError: on dtype/shape/dim mismatch between submitted and
            references when both are non-None and non-empty.
    """
    # --- abstain on missing submitted embedding ---
    if submitted_embedding is None:
        return _abstain(
            reason="no_submitted_embedding",
            n_references_used=0,
            k=k,
        )

    # --- input validation on submitted ---
    submitted = np.asarray(submitted_embedding)
    if not np.issubdtype(submitted.dtype, np.floating):
        raise ValueError(
            f"submitted must be floating-point, got {submitted.dtype}"
        )
    if submitted.ndim != 1:
        raise ValueError(
            f"submitted must be 1-D (d,), got shape {submitted.shape}"
        )
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if not (0 < distance_slope):
        raise ValueError(f"distance_slope must be positive, got {distance_slope}")

    embedding_dim = int(submitted.shape[0])

    # --- abstain on missing or insufficient references ---
    if reference_embeddings is None:
        return _abstain(
            reason="insufficient_references",
            n_references_used=0,
            k=k,
            metadata={
                "reason": "insufficient_references",
                "n_references": 0,
                "k": k,
                "embedding_dim": embedding_dim,
            },
        )

    references = np.asarray(reference_embeddings)
    if references.ndim != 2:
        raise ValueError(
            f"references must be 2-D (n_refs, d), got shape {references.shape}"
        )
    if not np.issubdtype(references.dtype, np.floating):
        raise ValueError(
            f"references must be floating-point, got {references.dtype}"
        )
    if references.size and references.shape[1] != embedding_dim:
        raise ValueError(
            f"dim mismatch: submitted has {embedding_dim} dims, "
            f"references have {references.shape[1]}"
        )

    n_refs = int(references.shape[0])
    if n_refs < k:
        return _abstain(
            reason="insufficient_references",
            n_references_used=n_refs,
            k=k,
            metadata={
                "reason": "insufficient_references",
                "n_references": n_refs,
                "k": k,
                "embedding_dim": embedding_dim,
            },
        )

    # --- compute pairwise cosine distances + top-k mean ---
    s_unit = _l2_normalize(submitted.astype(np.float32))
    r_unit = _l2_normalize(references.astype(np.float32))

    cos_sims = (r_unit @ s_unit).astype(np.float32)
    # Numerical safety: cos sim can drift fractionally outside [-1, 1].
    cos_sims = np.clip(cos_sims, -1.0, 1.0)
    distances = 1.0 - cos_sims  # shape (n_refs,), each in [0, 2]

    # `np.partition` puts the k smallest at the front in O(n). We don't
    # need them in sorted order — only the mean — so partition + slice
    # is the right primitive (sort would do extra work).
    if k >= n_refs:
        topk = distances
    else:
        topk = np.partition(distances, k - 1)[:k]
    mean_topk = float(topk.mean())

    score = _logistic(distance_midpoint - mean_topk, 0.0, distance_slope)
    confidence = _confidence_from_n_refs(n_refs)

    return KnnReferenceResult(
        score=float(score),
        confidence=float(confidence),
        mean_topk_distance=mean_topk,
        n_references_used=n_refs,
        k=k,
        abstain_reason=None,
        manufacturer_profile="generic",
        metadata={
            "embedding_dim": embedding_dim,
            "all_distances": [float(d) for d in distances.tolist()],
            "topk_distances": [float(d) for d in np.sort(topk).tolist()],
            "distance_midpoint": float(distance_midpoint),
            "distance_slope": float(distance_slope),
        },
    )


__all__ = [
    "DEFAULT_DISTANCE_MIDPOINT",
    "DEFAULT_DISTANCE_SLOPE",
    "DEFAULT_K",
    "KnnReferenceResult",
    "analyze_knn_reference",
]
