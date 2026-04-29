"""Embedding-anomaly counterfeit detector tests.

Pins the public contract of `measure_embedding_anomaly` against
synthetic embedding distributions. Mirrors test_rosette.py and
test_color_profile.py — the detector is deterministic and threshold-
based, so these tests act as both correctness checks and a regression
guard for future calibration tweaks (changing
DEFAULT_DISTANCE_MIDPOINT or the confidence ramp would have to update
the calibrated assertions here).

Synthetic embedding strategy:
    Embeddings are sampled from numpy random with a fixed seed, then
    L2-normalized. "Same variant" embeddings are perturbations of a
    base vector with small additive noise; "different variant"
    embeddings are independent random unit vectors. Because cosine
    distance only depends on direction, this is a faithful proxy for
    real embedder behavior — the geometry of "near vs far" is what
    the detector measures, not any property of the embedder itself.
"""

from __future__ import annotations

import numpy as np
import pytest

from pipelines.counterfeit.embedding_anomaly import (
    EmbeddingAnomalyMeasurement,
    is_likely_authentic,
    measure_embedding_anomaly,
)

EMBEDDING_DIM = 64
SEED = 0xE111


def _rand_unit(rng: np.random.Generator, dim: int = EMBEDDING_DIM) -> np.ndarray:
    """Random unit-length vector for use as a synthetic embedding."""
    v = rng.standard_normal(dim).astype(np.float32)
    n = np.linalg.norm(v)
    return (v / max(n, 1e-12)).astype(np.float32)


def _perturb(
    base: np.ndarray, rng: np.random.Generator, noise_scale: float = 0.05
) -> np.ndarray:
    """Same-variant-style perturbation: base + small noise, re-normalized.

    With noise_scale = 0.05 the cosine distance to the base lands in
    the 0.001-0.005 range — well below the detector's 0.25 midpoint."""
    v = base + rng.standard_normal(base.shape).astype(np.float32) * noise_scale
    n = np.linalg.norm(v)
    return (v / max(n, 1e-12)).astype(np.float32)


# -----------------------------
# Same-variant vs different-variant discrimination (the primary signal)
# -----------------------------


def test_submitted_near_centroid_scores_high() -> None:
    """An embedding near the centroid of the reference cluster should
    score near 1.0 — same-variant authentic-style."""
    rng = np.random.default_rng(SEED)
    base = _rand_unit(rng)
    refs = np.stack([_perturb(base, rng) for _ in range(5)])
    submitted = _perturb(base, rng)

    result = measure_embedding_anomaly(submitted, refs)
    assert isinstance(result, EmbeddingAnomalyMeasurement)
    assert result.embedding_score >= 0.85, (
        f"near-centroid scored {result.embedding_score} "
        f"(distance={result.distance_from_centroid:.4f}, n_refs={result.n_references})"
    )


def test_submitted_far_from_centroid_scores_low() -> None:
    """An embedding pulled from a totally different direction should
    score near 0.0 — counterfeit-style anomaly."""
    rng = np.random.default_rng(SEED)
    base = _rand_unit(rng)
    refs = np.stack([_perturb(base, rng) for _ in range(5)])
    # Submitted is independent of base — random unit vector. Expected
    # cosine distance to centroid ≈ 1 (random vectors in high dim
    # average ~orthogonal).
    submitted = _rand_unit(rng)

    result = measure_embedding_anomaly(submitted, refs)
    assert result.embedding_score <= 0.10, (
        f"far-from-centroid scored {result.embedding_score} "
        f"(distance={result.distance_from_centroid:.4f}, n_refs={result.n_references})"
    )


def test_near_strictly_beats_far() -> None:
    """Sanity: even after recalibration, a near-centroid submission must
    always score higher than a far-from-centroid one against the same
    reference set."""
    rng = np.random.default_rng(SEED)
    base = _rand_unit(rng)
    refs = np.stack([_perturb(base, rng) for _ in range(5)])
    near = measure_embedding_anomaly(_perturb(base, rng), refs)
    far = measure_embedding_anomaly(_rand_unit(rng), refs)
    assert near.embedding_score > far.embedding_score


def test_distance_decreases_as_submitted_approaches_centroid() -> None:
    """Walk a submitted embedding from far → near the centroid; score
    should be monotonically non-decreasing."""
    rng = np.random.default_rng(SEED)
    base = _rand_unit(rng)
    refs = np.stack([_perturb(base, rng) for _ in range(5)])
    far = _rand_unit(rng)
    # Linear interpolations from far → base, re-normalized at each step.
    walks = []
    for t in [0.0, 0.25, 0.50, 0.75, 1.0]:
        v = (1.0 - t) * far + t * base
        v = v / max(np.linalg.norm(v), 1e-12)
        walks.append(v.astype(np.float32))
    scores = [measure_embedding_anomaly(v, refs).embedding_score for v in walks]
    for i in range(len(scores) - 1):
        assert scores[i] <= scores[i + 1] + 1e-3, (
            f"non-monotone at step {i}: scores={scores}"
        )


# -----------------------------
# Reference-count / confidence semantics
# -----------------------------


def test_zero_references_returns_uncertain() -> None:
    """No references → can't speak. Score=0.5, confidence=0."""
    rng = np.random.default_rng(SEED)
    submitted = _rand_unit(rng)
    refs = np.empty((0, EMBEDDING_DIM), dtype=np.float32)

    result = measure_embedding_anomaly(submitted, refs)
    assert result.embedding_score == 0.5
    assert result.confidence == 0.0
    assert result.n_references == 0
    assert result.distance_from_centroid == 0.0
    assert result.metadata.get("reason") == "no_references"


def test_single_reference_can_still_score_with_low_confidence() -> None:
    """One reference → detector CAN speak (the centroid IS that ref)
    but confidence is reduced (0.4 per the calibration table)."""
    rng = np.random.default_rng(SEED)
    base = _rand_unit(rng)
    refs = np.stack([_perturb(base, rng)])
    near = measure_embedding_anomaly(_perturb(base, rng), refs)
    far = measure_embedding_anomaly(_rand_unit(rng), refs)
    # Both can produce a score, but the confidence should be bounded.
    assert near.confidence == pytest.approx(0.4, abs=0.001)
    assert far.confidence == pytest.approx(0.4, abs=0.001)
    # The score discrimination still works.
    assert near.embedding_score > far.embedding_score


def test_confidence_ramps_up_with_n_refs() -> None:
    """As n_refs increases the confidence rises along the calibration
    table: 1→0.4, 2→0.6, 3→0.7, 5→0.85, 10+→1.0."""
    rng = np.random.default_rng(SEED)
    base = _rand_unit(rng)
    submitted = _perturb(base, rng)
    refs_pool = np.stack([_perturb(base, rng) for _ in range(15)])

    expected = {1: 0.4, 2: 0.6, 3: 0.7, 5: 0.85, 10: 1.0, 15: 1.0}
    for n, conf in expected.items():
        r = measure_embedding_anomaly(submitted, refs_pool[:n])
        assert r.confidence == pytest.approx(conf, abs=0.01), (
            f"n_refs={n} → confidence={r.confidence}, expected {conf}"
        )


def test_confidence_interpolates_between_anchor_points() -> None:
    """n_refs=4 sits between 3 (0.7) and 5 (0.85); should be ~0.775."""
    rng = np.random.default_rng(SEED)
    base = _rand_unit(rng)
    submitted = _perturb(base, rng)
    refs = np.stack([_perturb(base, rng) for _ in range(4)])
    r = measure_embedding_anomaly(submitted, refs)
    assert r.confidence == pytest.approx(0.775, abs=0.01)


# -----------------------------
# Edge cases
# -----------------------------


def test_submitted_identical_to_a_reference_scores_high() -> None:
    """If the submitted embedding equals one of the references, distance
    to centroid is the in-cluster spread (NOT zero — the centroid is
    the mean of all refs, not any single one). Score should be high but
    not necessarily saturated; depends on cluster spread vs. midpoint."""
    rng = np.random.default_rng(SEED)
    base = _rand_unit(rng)
    refs_list = [_perturb(base, rng) for _ in range(5)]
    refs = np.stack(refs_list)
    # Submitted is exactly one of the references.
    submitted = refs_list[2]
    r = measure_embedding_anomaly(submitted, refs)
    # In-cluster spread with noise_scale=0.05 puts distance ~0.05; that
    # gives a score around 0.95 with the default midpoint=0.25 and
    # slope=15. Anchor at >=0.85 to stay above the same-variant
    # threshold without being brittle to seed-level cluster shape.
    assert r.embedding_score >= 0.85


def test_all_identical_references_give_zero_distance_when_match() -> None:
    """Edge case: all references are identical and submitted matches.
    Distance should be 0 (or numerically near it)."""
    rng = np.random.default_rng(SEED)
    base = _rand_unit(rng)
    refs = np.stack([base for _ in range(3)])
    r = measure_embedding_anomaly(base, refs)
    assert r.distance_from_centroid < 1e-4
    assert r.embedding_score >= 0.95


def test_metadata_includes_per_reference_distances() -> None:
    rng = np.random.default_rng(SEED)
    base = _rand_unit(rng)
    refs = np.stack([_perturb(base, rng) for _ in range(5)])
    submitted = _perturb(base, rng)
    r = measure_embedding_anomaly(submitted, refs)
    ref_distances = r.metadata.get("ref_distances_from_centroid")
    assert isinstance(ref_distances, list)
    assert len(ref_distances) == 5
    # Each ref-to-centroid distance is in [0, 2] for cosine.
    for d in ref_distances:
        assert 0.0 <= d <= 2.0


def test_handles_unnormalized_inputs() -> None:
    """The detector L2-normalizes defensively. Magnitude-scaling the
    inputs should not change the score."""
    rng = np.random.default_rng(SEED)
    base = _rand_unit(rng)
    refs = np.stack([_perturb(base, rng) for _ in range(5)])
    submitted = _perturb(base, rng)

    # Scale all inputs by arbitrary positive constants.
    refs_scaled = refs * np.array([0.5, 2.0, 7.3, 0.1, 100.0])[:, None].astype(np.float32)
    submitted_scaled = submitted * 17.5

    r_norm = measure_embedding_anomaly(submitted, refs)
    r_scaled = measure_embedding_anomaly(submitted_scaled, refs_scaled.astype(np.float32))
    assert r_scaled.embedding_score == pytest.approx(r_norm.embedding_score, abs=1e-4)


# -----------------------------
# Input validation
# -----------------------------


def test_rejects_integer_submitted() -> None:
    submitted = np.zeros(EMBEDDING_DIM, dtype=np.int32)
    refs = np.zeros((3, EMBEDDING_DIM), dtype=np.float32)
    with pytest.raises(ValueError, match="floating-point"):
        measure_embedding_anomaly(submitted, refs)


def test_rejects_integer_references() -> None:
    submitted = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    refs = np.zeros((3, EMBEDDING_DIM), dtype=np.int32)
    with pytest.raises(ValueError, match="floating-point"):
        measure_embedding_anomaly(submitted, refs)


def test_rejects_2d_submitted() -> None:
    submitted = np.zeros((3, EMBEDDING_DIM), dtype=np.float32)
    refs = np.zeros((3, EMBEDDING_DIM), dtype=np.float32)
    with pytest.raises(ValueError, match="1-D"):
        measure_embedding_anomaly(submitted, refs)


def test_rejects_1d_references() -> None:
    submitted = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    refs = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    with pytest.raises(ValueError, match="2-D"):
        measure_embedding_anomaly(submitted, refs)


def test_rejects_dim_mismatch() -> None:
    submitted = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    refs = np.zeros((3, EMBEDDING_DIM * 2), dtype=np.float32)
    with pytest.raises(ValueError, match="dim mismatch"):
        measure_embedding_anomaly(submitted, refs)


def test_rejects_invalid_distance_slope() -> None:
    rng = np.random.default_rng(SEED)
    submitted = _rand_unit(rng)
    refs = np.stack([_rand_unit(rng) for _ in range(3)])
    with pytest.raises(ValueError, match="slope"):
        measure_embedding_anomaly(submitted, refs, distance_slope=0.0)


# -----------------------------
# Output contract
# -----------------------------


def test_output_is_frozen_dataclass_with_all_fields() -> None:
    rng = np.random.default_rng(SEED)
    base = _rand_unit(rng)
    refs = np.stack([_perturb(base, rng) for _ in range(5)])
    result = measure_embedding_anomaly(_perturb(base, rng), refs)
    assert isinstance(result.embedding_score, float)
    assert isinstance(result.distance_from_centroid, float)
    assert isinstance(result.n_references, int)
    assert isinstance(result.confidence, float)
    assert isinstance(result.manufacturer_profile, str)
    assert isinstance(result.metadata, dict)
    assert 0.0 <= result.embedding_score <= 1.0
    assert 0.0 <= result.confidence <= 1.0


def test_output_is_immutable() -> None:
    rng = np.random.default_rng(SEED)
    base = _rand_unit(rng)
    refs = np.stack([_perturb(base, rng) for _ in range(5)])
    result = measure_embedding_anomaly(_perturb(base, rng), refs)
    with pytest.raises((AttributeError, Exception)):
        result.embedding_score = 0.0  # type: ignore[misc]


# -----------------------------
# is_likely_authentic predicate
# -----------------------------


def test_is_likely_authentic_matches_threshold_compare() -> None:
    rng = np.random.default_rng(SEED)
    base = _rand_unit(rng)
    refs = np.stack([_perturb(base, rng) for _ in range(5)])
    near = measure_embedding_anomaly(_perturb(base, rng), refs)
    far = measure_embedding_anomaly(_rand_unit(rng), refs)
    assert is_likely_authentic(near) is True
    assert is_likely_authentic(far) is False


def test_is_likely_authentic_respects_custom_threshold() -> None:
    rng = np.random.default_rng(SEED)
    base = _rand_unit(rng)
    refs = np.stack([_perturb(base, rng) for _ in range(5)])
    result = measure_embedding_anomaly(_perturb(base, rng), refs)
    # At a stricter threshold, marginal scores become "not authentic".
    assert is_likely_authentic(result, threshold=0.999) is (
        result.embedding_score >= 0.999
    )
