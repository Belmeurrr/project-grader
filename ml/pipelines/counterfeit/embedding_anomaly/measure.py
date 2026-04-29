"""Embedding-anomaly counterfeit detector (ensemble #7 of 7).

Compares the submitted card's identification embedding against the
distribution of embeddings from known authentic exemplars of the same
variant. Counterfeit cards — even visually-passable ones — sit further
in embedding space from the authentic centroid than genuine cards do,
because the model picks up subtle texture / ink / substrate cues the
counterfeiter can't perfectly replicate.

This is the most reference-dependent detector in the ensemble. The
others (rosette, color profile, typography, substrate) work on a
single canonical card with no external data; this one needs a
populated set of authentic exemplars per variant. It's powerful when
that data exists (high specificity for known variants) and abstains
when it doesn't.

Algorithm:
  1. L2-normalize the submitted embedding and all reference embeddings.
     Defensive — most embedders return unit vectors already, but
     normalizing here means the detector works regardless.
  2. If n_refs == 0: return uncertain (score=0.5, conf=0). Cannot speak.
  3. Compute the centroid: L2-normalized mean of reference embeddings.
     For n_refs == 1 the centroid IS the single reference.
  4. Cosine distance from submitted to centroid:
        d = 1 - dot(submitted_unit, centroid_unit)
     Equivalent to 1 - cosine_similarity. Range [0, 2], typical
     same-variant range [0.0, 0.20].
  5. Logistic squash: score = sigmoid((midpoint - d) * slope). Lower
     distance → higher score (more authentic-like).
  6. Confidence ramps with n_refs: 0 → 0.0, 1 → 0.4, 2 → 0.6, 3 → 0.7,
     5+ → 0.85, 10+ → 1.0. The detector CAN speak with one reference,
     but the variance estimate is unreliable; more refs means the
     centroid is closer to the true variant center.

Design choice — fixed distance threshold vs. per-variant z-score:
    With ≥3 reference embeddings we could compute a per-variant
    cosine-distance distribution and score against z-score. That's
    the principled approach but it requires several authentic
    exemplars per variant. Until the corpus has that, a fixed
    threshold (calibrated against synthetic embeddings) is the only
    workable option. The current implementation uses fixed-threshold
    everywhere; swapping to z-score per variant when refs are dense
    enough is a one-function-body change that doesn't touch the
    public surface.

Detects: visually-passable counterfeits whose ink/substrate/print
fingerprint differs from the authentic distribution. Particularly
useful for SECOND-GENERATION counterfeits (good photocopies of real
slabs) where rosette and color profile may both pass.
Does NOT detect: counterfeits that share the embedding manifold with
authentics (rare — would require physical-world reproduction of the
print process), or cards from variants we have no references for.
That's why this is one detector in a 7-detector ensemble, not a
standalone authenticator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

# Logistic squash midpoint for the cosine distance.
#
# Empirical anchors against synthetic embeddings (see test_embedding_anomaly):
#   - Same image embedded twice with tiny noise: d ≈ 0.001 - 0.005
#   - Same variant, different lighting/angle (visual transformations
#     applied to the same source): d ≈ 0.05 - 0.15
#   - Different variant, same set: d ≈ 0.4 - 0.6
#   - Random embedding: d ≈ 0.7 - 1.0
#
# Midpoint at 0.25 places the decision boundary in the gap between
# "same variant" and "different variant" — well above plausible same-
# variant variation, well below cross-variant distances.
DEFAULT_DISTANCE_MIDPOINT: float = 0.25
DEFAULT_DISTANCE_SLOPE: float = 15.0

# Confidence ramp by number of references. The detector CAN speak with
# one reference but the variance estimate (= distance threshold tuning)
# is unreliable. Linearly interpolated between these anchor points;
# values below 1 ref → 0, above 10 → 1.
_CONFIDENCE_BY_NREFS: dict[int, float] = {
    0: 0.0,
    1: 0.4,
    2: 0.6,
    3: 0.7,
    5: 0.85,
    10: 1.0,
}


@dataclass(frozen=True)
class EmbeddingAnomalyMeasurement:
    """Output of `measure_embedding_anomaly`.

    Fields:
      embedding_score: [0, 1], higher = more likely authentic.
        0.5 means "uncertain" (returned when no references are
        available).
      distance_from_centroid: cosine distance (1 - cos similarity)
        from the submitted embedding to the L2-normalized centroid
        of the reference embeddings. Provided for debug / forensic
        review and so a future calibration pass can recompute scores
        from history without re-running the embedder.
      n_references: how many authentic reference embeddings the score
        was computed against. Determines confidence.
      confidence: [0, 1]. Ramps with n_references; reaches 0 with no
        references and saturates at 1.0 around 10+ refs.
      manufacturer_profile: which calibration profile was used. v1
        always "generic"; future versions will branch per-manufacturer
        once enough authentic exemplars have accumulated per variant.
      metadata: backend extras (per-reference distances, embedding
        dim, etc.) for UI overlay and offline calibration analyses.
    """

    embedding_score: float
    distance_from_centroid: float
    n_references: int
    confidence: float
    manufacturer_profile: str
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
    """Normalize a vector or batch of vectors to unit L2 length.

    For a 2-D input (n, d) normalizes each row independently. Uses a
    tiny epsilon to avoid divide-by-zero on a zero vector — that
    vector's normalized form remains zero, and downstream cosine
    distance against a zero vector falls out of the formula naturally.
    """
    if v.ndim == 1:
        n = float(np.linalg.norm(v))
        return (v / max(n, eps)).astype(np.float32)
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    return (v / np.maximum(norms, eps)).astype(np.float32)


def _confidence_from_n_refs(n: int) -> float:
    """Linearly interpolate confidence between the anchor points in
    `_CONFIDENCE_BY_NREFS`. Clamps below 0 and above 10."""
    if n <= 0:
        return 0.0
    if n >= 10:
        return 1.0
    keys = sorted(_CONFIDENCE_BY_NREFS.keys())
    # Find the two anchor points bracketing n.
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


def measure_embedding_anomaly(
    submitted: NDArray[np.floating],
    references: NDArray[np.floating],
    *,
    distance_midpoint: float = DEFAULT_DISTANCE_MIDPOINT,
    distance_slope: float = DEFAULT_DISTANCE_SLOPE,
) -> EmbeddingAnomalyMeasurement:
    """Score a submitted embedding against a set of authentic references.

    Args:
        submitted: 1-D array of shape (d,). Floating-point embedding
            from the identification stage.
        references: 2-D array of shape (n_refs, d). Authentic
            reference embeddings for the SAME VARIANT as the
            submitted card. Caller is responsible for variant-matching
            (the detector itself doesn't know about variants — just
            embeddings).
        distance_midpoint: cosine-distance value at which the score is
            0.5. Default 0.25 (well above plausible same-variant
            variation, well below cross-variant distances).
        distance_slope: logistic slope. Higher = sharper transition.

    Returns:
        EmbeddingAnomalyMeasurement with embedding_score in [0, 1]
        (higher = more likely authentic) and confidence in [0, 1].

    Raises:
        ValueError: on dtype/shape/dim mismatch between submitted and
            references.
    """
    # --- input validation ---
    if not np.issubdtype(submitted.dtype, np.floating):
        raise ValueError(f"submitted must be floating-point, got {submitted.dtype}")
    if not np.issubdtype(references.dtype, np.floating):
        raise ValueError(f"references must be floating-point, got {references.dtype}")
    if submitted.ndim != 1:
        raise ValueError(
            f"submitted must be 1-D (d,), got shape {submitted.shape}"
        )
    if references.ndim != 2:
        raise ValueError(
            f"references must be 2-D (n_refs, d), got shape {references.shape}"
        )
    if references.size and references.shape[1] != submitted.shape[0]:
        raise ValueError(
            f"dim mismatch: submitted has {submitted.shape[0]} dims, "
            f"references have {references.shape[1]}"
        )
    if not (0 < distance_slope):
        raise ValueError(f"distance_slope must be positive, got {distance_slope}")

    n_refs = int(references.shape[0])
    embedding_dim = int(submitted.shape[0])

    # --- abstain on no references ---
    if n_refs == 0:
        return EmbeddingAnomalyMeasurement(
            embedding_score=0.5,
            distance_from_centroid=0.0,
            n_references=0,
            confidence=0.0,
            manufacturer_profile="generic",
            metadata={
                "reason": "no_references",
                "embedding_dim": embedding_dim,
            },
        )

    # --- compute centroid + distance ---
    s_unit = _l2_normalize(submitted.astype(np.float32))
    r_unit = _l2_normalize(references.astype(np.float32))
    # Mean of unit vectors then re-normalize. Unlike a raw mean, this
    # gives the geometric center on the unit hypersphere — robust to
    # near-zero-magnitude refs.
    centroid_raw = r_unit.mean(axis=0)
    centroid = _l2_normalize(centroid_raw)

    cos_sim = float(np.dot(s_unit, centroid))
    # Numerical safety: cos sim can drift fractionally outside [-1, 1].
    cos_sim = max(-1.0, min(1.0, cos_sim))
    distance = 1.0 - cos_sim

    # --- per-reference distances for forensics ---
    # Distance from each reference to the centroid; gives a sense of
    # how tight the variant cluster actually is. Useful for offline
    # tuning of the per-variant z-score variant of the detector.
    ref_to_centroid = (1.0 - r_unit @ centroid).astype(np.float32)
    ref_distances = [float(d) for d in ref_to_centroid.tolist()]

    score = _logistic(distance_midpoint - distance, 0.0, distance_slope)
    confidence = _confidence_from_n_refs(n_refs)

    return EmbeddingAnomalyMeasurement(
        embedding_score=float(score),
        distance_from_centroid=float(distance),
        n_references=n_refs,
        confidence=confidence,
        manufacturer_profile="generic",
        metadata={
            "embedding_dim": embedding_dim,
            "ref_distances_from_centroid": ref_distances,
            "centroid_norm_before_renormalize": float(np.linalg.norm(centroid_raw)),
            "distance_midpoint": float(distance_midpoint),
            "distance_slope": float(distance_slope),
        },
    )


def is_likely_authentic(
    measurement: EmbeddingAnomalyMeasurement,
    threshold: float = 0.5,
) -> bool:
    """Convenience predicate matching `measurement.embedding_score >= threshold`.

    Mirrors the rosette and color detectors' predicates so call sites
    can be uniform across the counterfeit ensemble. May add confidence-
    gated abstain or per-manufacturer thresholds in future without
    changing call sites."""
    return measurement.embedding_score >= threshold
