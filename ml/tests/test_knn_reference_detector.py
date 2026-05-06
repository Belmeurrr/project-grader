"""k-NN reference counterfeit detector tests.

Pins the public contract of `analyze_knn_reference` against synthetic
embedding distributions. Sister test to `test_embedding_anomaly` —
both detectors consume the same shape of input but reduce it
differently. The "manifold-distance vs centroid-distance" test below
captures the entire reason this detector exists alongside #7: same
inputs, materially different scores when the reference set is
spread along a manifold instead of clustered around a single point.

Synthetic embedding strategy mirrors test_embedding_anomaly: random
unit vectors with controlled perturbations. Cosine distance only
depends on direction, so this is a faithful proxy for real embedder
behavior — what the detector measures is the geometry of "near vs
far," not anything embedder-specific.
"""

from __future__ import annotations

import numpy as np
import pytest

from pipelines.counterfeit.embedding_anomaly import (
    measure_embedding_anomaly,
)
from pipelines.counterfeit.knn_reference import (
    DEFAULT_K,
    KnnReferenceResult,
    analyze_knn_reference,
)

EMBEDDING_DIM = 64
SEED = 0xC4FF


def _rand_unit(rng: np.random.Generator, dim: int = EMBEDDING_DIM) -> np.ndarray:
    v = rng.standard_normal(dim).astype(np.float32)
    n = np.linalg.norm(v)
    return (v / max(n, 1e-12)).astype(np.float32)


def _perturb(
    base: np.ndarray, rng: np.random.Generator, noise_scale: float = 0.05
) -> np.ndarray:
    v = base + rng.standard_normal(base.shape).astype(np.float32) * noise_scale
    n = np.linalg.norm(v)
    return (v / max(n, 1e-12)).astype(np.float32)


# -----------------------------
# Authentic: submitted near multiple references → high score
# -----------------------------


def test_submitted_near_multiple_references_scores_high() -> None:
    """Submitted lands among 5 same-variant references → mean top-3
    distance is small → score high."""
    rng = np.random.default_rng(SEED)
    base = _rand_unit(rng)
    refs = np.stack([_perturb(base, rng) for _ in range(5)])
    submitted = _perturb(base, rng)

    result = analyze_knn_reference(submitted, refs)
    assert isinstance(result, KnnReferenceResult)
    assert result.abstain_reason is None
    assert result.score >= 0.85, (
        f"near-cluster scored {result.score} "
        f"(mean_topk={result.mean_topk_distance:.4f}, n={result.n_references_used})"
    )
    assert result.k == DEFAULT_K
    # The detector reports the actual top-k list in metadata for
    # forensic review; assert it has the right cardinality.
    assert len(result.metadata["topk_distances"]) == DEFAULT_K


# -----------------------------
# Counterfeit: submitted far from all references → low score
# -----------------------------


def test_submitted_far_from_all_references_scores_low() -> None:
    rng = np.random.default_rng(SEED)
    base = _rand_unit(rng)
    refs = np.stack([_perturb(base, rng) for _ in range(5)])
    # Random submitted vector — orthogonal in expectation to base.
    submitted = _rand_unit(rng)

    result = analyze_knn_reference(submitted, refs)
    assert result.abstain_reason is None
    assert result.score <= 0.10, (
        f"far-from-cluster scored {result.score} "
        f"(mean_topk={result.mean_topk_distance:.4f})"
    )


def test_near_strictly_beats_far() -> None:
    """Sanity: a near submission must always score higher than a far one
    against the same reference set."""
    rng = np.random.default_rng(SEED)
    base = _rand_unit(rng)
    refs = np.stack([_perturb(base, rng) for _ in range(5)])
    near = analyze_knn_reference(_perturb(base, rng), refs)
    far = analyze_knn_reference(_rand_unit(rng), refs)
    assert near.score > far.score


# -----------------------------
# Insufficient references → abstain
# -----------------------------


def test_abstain_when_fewer_than_k_references() -> None:
    """n_refs < k means we can't take a top-k; the detector must abstain
    UNVERIFIED rather than fall back to top-(n_refs) — else the signal
    silently degrades into nearest-neighbor on a 1- or 2-element set,
    which is exactly what we don't want."""
    rng = np.random.default_rng(SEED)
    base = _rand_unit(rng)
    submitted = _perturb(base, rng)
    refs = np.stack([_perturb(base, rng) for _ in range(2)])

    result = analyze_knn_reference(submitted, refs, k=3)
    assert result.abstain_reason == "insufficient_references"
    assert result.score == 0.5
    assert result.confidence == 0.0
    assert result.n_references_used == 2
    assert result.k == 3


def test_abstain_when_zero_references() -> None:
    rng = np.random.default_rng(SEED)
    submitted = _rand_unit(rng)
    refs = np.empty((0, EMBEDDING_DIM), dtype=np.float32)

    result = analyze_knn_reference(submitted, refs, k=3)
    assert result.abstain_reason == "insufficient_references"
    assert result.confidence == 0.0
    assert result.n_references_used == 0


def test_abstain_when_references_is_none() -> None:
    """`None` for references is explicitly accepted — the service
    wrapper passes through a missing-variant lookup as None."""
    rng = np.random.default_rng(SEED)
    submitted = _rand_unit(rng)
    result = analyze_knn_reference(submitted, None, k=3)
    assert result.abstain_reason == "insufficient_references"
    assert result.confidence == 0.0
    assert result.n_references_used == 0


# -----------------------------
# Missing submitted embedding → abstain
# -----------------------------


def test_abstain_when_no_submitted_embedding() -> None:
    rng = np.random.default_rng(SEED)
    base = _rand_unit(rng)
    refs = np.stack([_perturb(base, rng) for _ in range(5)])

    result = analyze_knn_reference(None, refs)
    assert result.abstain_reason == "no_submitted_embedding"
    assert result.score == 0.5
    assert result.confidence == 0.0
    assert result.n_references_used == 0


# -----------------------------
# Confidence ramp
# -----------------------------


def test_confidence_ramp_anchors() -> None:
    """k=3: floor confidence 0.4 at n=k, 0.7 at n=5, 1.0 at n=10+."""
    rng = np.random.default_rng(SEED)
    base = _rand_unit(rng)
    submitted = _perturb(base, rng)
    pool = np.stack([_perturb(base, rng) for _ in range(15)])

    expected = {3: 0.4, 5: 0.7, 10: 1.0, 15: 1.0}
    for n, conf in expected.items():
        r = analyze_knn_reference(submitted, pool[:n])
        assert r.confidence == pytest.approx(conf, abs=0.01), (
            f"n={n} → conf={r.confidence}, expected {conf}"
        )


def test_confidence_interpolates_between_anchors() -> None:
    """n=4 sits between 3 (0.4) and 5 (0.7) — expect ~0.55."""
    rng = np.random.default_rng(SEED)
    base = _rand_unit(rng)
    submitted = _perturb(base, rng)
    refs = np.stack([_perturb(base, rng) for _ in range(4)])
    r = analyze_knn_reference(submitted, refs)
    assert r.confidence == pytest.approx(0.55, abs=0.02)


# -----------------------------
# k-NN ≠ centroid: the entire reason this detector exists
# -----------------------------


def test_manifold_distribution_produces_different_score_from_centroid() -> None:
    """Reference set spread along a single axis (a synthetic "manifold").
    A submission that lands AT ONE END of the spread is close to the
    references at that end but far from the centroid (which sits in
    the middle of the spread). The k-NN detector should give it a
    notably HIGHER score than the centroid detector — that's the whole
    reason both detectors exist as separate signals. Lock in a
    meaningful gap so a future calibration change that accidentally
    collapses the two signals fails loudly here.
    """
    rng = np.random.default_rng(SEED)
    base = _rand_unit(rng)

    # Build a "manifold": pick a perpendicular direction and stretch
    # references along it. After re-normalization the spread becomes
    # an arc on the unit hypersphere. 1.4 is large enough that the
    # centroid (the renormalized mean of the arc endpoints) sits
    # meaningfully off the manifold — that's the geometric pathology
    # the centroid detector struggles with.
    perp = rng.standard_normal(EMBEDDING_DIM).astype(np.float32)
    perp -= float(perp @ base) * base  # orthogonalize against base
    perp /= max(float(np.linalg.norm(perp)), 1e-12)

    # 7 references stretched ±1.4 along `perp`. The endpoints end up
    # roughly orthogonal to the midpoints after renormalization, so
    # the arc has real curvature.
    spread_factors = np.linspace(-1.4, 1.4, 7).astype(np.float32)
    refs_raw = np.stack([base + s * perp for s in spread_factors])
    refs = (refs_raw / np.linalg.norm(refs_raw, axis=1, keepdims=True)).astype(
        np.float32
    )

    # Submitted lands AT ONE END of the spread (matches the +1.4
    # endpoint exactly). That gives it cosine distance ≈ 0 to row 6
    # and small-to-moderate distance to rows 4,5 — top-3 mean is
    # small. But the centroid is the L2-renormalized mean of the arc;
    # because the arc is symmetric around `base`, the centroid is
    # close to `base` (perpendicular components cancel), so the
    # centroid distance to the submission is large.
    submitted_raw = base + 1.4 * perp
    submitted = (submitted_raw / np.linalg.norm(submitted_raw)).astype(np.float32)

    knn = analyze_knn_reference(submitted, refs, k=3)
    centroid = measure_embedding_anomaly(submitted, refs)

    assert knn.abstain_reason is None
    # The k-NN distance is the mean of "0 + two moderately-close
    # arc neighbors"; the centroid distance is the curvature-induced
    # gap from the submission to the arc midpoint. They should differ
    # by a meaningful margin.
    assert knn.mean_topk_distance < centroid.distance_from_centroid, (
        f"expected k-NN top-{knn.k} mean distance < centroid distance; "
        f"got knn_topk={knn.mean_topk_distance:.3f}, "
        f"centroid={centroid.distance_from_centroid:.3f}"
    )
    # And the resulting scores should be materially different —
    # the gap is the entire justification for running both detectors.
    assert (knn.score - centroid.embedding_score) > 0.10, (
        f"k-NN should score noticeably higher than centroid for a "
        f"manifold-end submission; got "
        f"knn={knn.score:.3f}, centroid={centroid.embedding_score:.3f}, "
        f"knn_topk={knn.mean_topk_distance:.3f}, "
        f"centroid_dist={centroid.distance_from_centroid:.3f}"
    )


# -----------------------------
# Edge cases
# -----------------------------


def test_submitted_identical_to_a_reference_scores_max() -> None:
    """Submitted == one of the references → that reference contributes
    distance ≈ 0 to the top-k → mean is dominated by the near-zero
    member; score saturates near 1."""
    rng = np.random.default_rng(SEED)
    base = _rand_unit(rng)
    refs_list = [_perturb(base, rng) for _ in range(5)]
    refs = np.stack(refs_list)
    submitted = refs_list[2].copy()

    r = analyze_knn_reference(submitted, refs)
    assert r.abstain_reason is None
    # The submitted-equal-reference produces one zero-distance entry;
    # the other top-k members are at the cluster's pairwise spread
    # (~sqrt(2)*noise_scale ≈ 0.07 for noise_scale=0.05). Mean top-3
    # ≈ 0.06; score with midpoint=0.25, slope=15 lands ~0.94. Anchor
    # firmly above 0.9 — the point is that submitting one of the
    # exemplars VERBATIM scores authentic-side with margin, not that
    # it saturates exactly to 1.
    assert r.score >= 0.9
    # And in particular: lower top-k distance than a random submission.
    rng2 = np.random.default_rng(SEED + 1)
    far = analyze_knn_reference(_rand_unit(rng2), refs)
    assert r.score > far.score


def test_handles_unnormalized_inputs() -> None:
    """The detector L2-normalizes defensively — magnitude-scaling the
    inputs should not change the score."""
    rng = np.random.default_rng(SEED)
    base = _rand_unit(rng)
    refs = np.stack([_perturb(base, rng) for _ in range(5)])
    submitted = _perturb(base, rng)

    refs_scaled = (
        refs * np.array([0.5, 2.0, 7.3, 0.1, 100.0])[:, None].astype(np.float32)
    )
    submitted_scaled = submitted * 17.5

    r_norm = analyze_knn_reference(submitted, refs)
    r_scaled = analyze_knn_reference(
        submitted_scaled.astype(np.float32), refs_scaled.astype(np.float32)
    )
    assert r_scaled.score == pytest.approx(r_norm.score, abs=1e-4)
    assert r_scaled.mean_topk_distance == pytest.approx(
        r_norm.mean_topk_distance, abs=1e-4
    )


def test_k_equal_to_n_references_uses_all() -> None:
    """When k == n_refs, the top-k is the full reference set; mean of all
    distances is the score's input."""
    rng = np.random.default_rng(SEED)
    base = _rand_unit(rng)
    refs = np.stack([_perturb(base, rng) for _ in range(3)])
    submitted = _perturb(base, rng)

    r = analyze_knn_reference(submitted, refs, k=3)
    assert r.abstain_reason is None
    # Mean top-3 == mean of all 3 distances.
    expected_mean = float(np.mean(r.metadata["all_distances"]))
    assert r.mean_topk_distance == pytest.approx(expected_mean, abs=1e-6)


def test_metadata_includes_per_reference_distances() -> None:
    rng = np.random.default_rng(SEED)
    base = _rand_unit(rng)
    refs = np.stack([_perturb(base, rng) for _ in range(5)])
    submitted = _perturb(base, rng)
    r = analyze_knn_reference(submitted, refs)
    all_d = r.metadata["all_distances"]
    topk_d = r.metadata["topk_distances"]
    assert len(all_d) == 5
    assert len(topk_d) == DEFAULT_K
    # The top-k distances are a subset of the full distance list.
    for d in topk_d:
        assert any(abs(d - a) < 1e-6 for a in all_d), (
            f"top-k member {d} not in full distance list {all_d}"
        )
    # Top-k are the smallest k distances.
    assert max(topk_d) <= sorted(all_d)[DEFAULT_K]


# -----------------------------
# Input validation
# -----------------------------


def test_rejects_integer_submitted() -> None:
    submitted = np.zeros(EMBEDDING_DIM, dtype=np.int32)
    refs = np.zeros((3, EMBEDDING_DIM), dtype=np.float32)
    with pytest.raises(ValueError, match="floating-point"):
        analyze_knn_reference(submitted, refs)


def test_rejects_integer_references() -> None:
    submitted = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    refs = np.zeros((3, EMBEDDING_DIM), dtype=np.int32)
    with pytest.raises(ValueError, match="floating-point"):
        analyze_knn_reference(submitted, refs)


def test_rejects_2d_submitted() -> None:
    submitted = np.zeros((3, EMBEDDING_DIM), dtype=np.float32)
    refs = np.zeros((3, EMBEDDING_DIM), dtype=np.float32)
    with pytest.raises(ValueError, match="1-D"):
        analyze_knn_reference(submitted, refs)


def test_rejects_1d_references() -> None:
    submitted = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    refs = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    with pytest.raises(ValueError, match="2-D"):
        analyze_knn_reference(submitted, refs)


def test_rejects_dim_mismatch() -> None:
    submitted = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    refs = np.zeros((3, EMBEDDING_DIM * 2), dtype=np.float32)
    with pytest.raises(ValueError, match="dim mismatch"):
        analyze_knn_reference(submitted, refs)


def test_rejects_invalid_k() -> None:
    rng = np.random.default_rng(SEED)
    submitted = _rand_unit(rng)
    refs = np.stack([_rand_unit(rng) for _ in range(3)])
    with pytest.raises(ValueError, match="k must be"):
        analyze_knn_reference(submitted, refs, k=0)


def test_rejects_invalid_distance_slope() -> None:
    rng = np.random.default_rng(SEED)
    submitted = _rand_unit(rng)
    refs = np.stack([_rand_unit(rng) for _ in range(3)])
    with pytest.raises(ValueError, match="slope"):
        analyze_knn_reference(submitted, refs, distance_slope=0.0)


# -----------------------------
# Output contract
# -----------------------------


def test_output_is_frozen_dataclass_with_all_fields() -> None:
    rng = np.random.default_rng(SEED)
    base = _rand_unit(rng)
    refs = np.stack([_perturb(base, rng) for _ in range(5)])
    result = analyze_knn_reference(_perturb(base, rng), refs)
    assert isinstance(result.score, float)
    assert isinstance(result.confidence, float)
    assert isinstance(result.mean_topk_distance, float)
    assert isinstance(result.n_references_used, int)
    assert isinstance(result.k, int)
    assert isinstance(result.manufacturer_profile, str)
    assert isinstance(result.metadata, dict)
    assert 0.0 <= result.score <= 1.0
    assert 0.0 <= result.confidence <= 1.0


def test_output_is_immutable() -> None:
    rng = np.random.default_rng(SEED)
    base = _rand_unit(rng)
    refs = np.stack([_perturb(base, rng) for _ in range(5)])
    result = analyze_knn_reference(_perturb(base, rng), refs)
    with pytest.raises((AttributeError, Exception)):
        result.score = 0.0  # type: ignore[misc]
