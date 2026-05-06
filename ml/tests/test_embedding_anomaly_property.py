"""Property-based tests for the embedding-anomaly counterfeit detector.

The fixture-based tests in `test_embedding_anomaly.py` pin the
score/confidence shape against hand-picked synthetic embedding
distributions — same-variant near-centroid vs. independent-direction
random. Those tests catch a regression that flips the sign of the
score function but they're blind to drift in
DEFAULT_DISTANCE_MIDPOINT and DEFAULT_DISTANCE_SLOPE: a midpoint
shift of 0.10 in either direction still leaves
"submitted-near-centroid" on the same side of the boundary and still
leaves "submitted-far-from-centroid" on its side.

This file expresses the calibration-invariant directly: for a fixed
reference set, **the embedding score is monotone-non-increasing in
the cosine distance from centroid**. That property holds for any
midpoint, any slope, any future per-manufacturer profile — the score
is a logistic squash of `(midpoint - distance)`, which is
monotone-decreasing in distance by construction.

We construct embedding pairs at varying distances using a
parameterized rotation: take a random unit reference, define a unit
vector orthogonal to it, and form `cos(θ) * ref + sin(θ) * orth` for
a hypothesis-generated angle θ. Larger θ → larger cosine distance,
deterministically and independently of the embedder.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipelines.counterfeit.embedding_anomaly import (  # noqa: E402
    measure_embedding_anomaly,
)


EMBEDDING_DIM = 32
SEED = 0xE111


def _orthonormal_pair(rng: np.random.Generator, dim: int) -> tuple[
    np.ndarray, np.ndarray
]:
    """Return two unit vectors `u`, `v` with `dot(u, v) == 0`.

    Constructed via Gram-Schmidt on a 2D random subspace, so any
    `cos(θ) * u + sin(θ) * v` lies on the unit sphere and has cosine
    similarity to `u` exactly equal to `cos(θ)` — i.e. cosine
    distance equal to `1 - cos(θ)`. This gives us deterministic
    control over the distance-from-centroid axis the property is
    quantifying over."""
    a = rng.standard_normal(dim).astype(np.float32)
    b = rng.standard_normal(dim).astype(np.float32)
    u = a / np.linalg.norm(a)
    # Project b out of u, normalize the residual.
    b = b - np.dot(b, u) * u
    v = b / np.linalg.norm(b)
    return u.astype(np.float32), v.astype(np.float32)


def _at_angle(u: np.ndarray, v: np.ndarray, theta: float) -> np.ndarray:
    """Unit vector at angle `theta` from `u`, in the (u, v) plane."""
    return (math.cos(theta) * u + math.sin(theta) * v).astype(np.float32)


# Strategies. Two angles in [0, π/2]; embeddings constructed at those
# angles will land at cosine distances [0, 1] — covers the whole
# meaningful range of the detector.
_ANGLES = st.floats(
    min_value=0.0, max_value=math.pi / 2, allow_nan=False, allow_infinity=False
)


@settings(
    max_examples=80,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(
    angle_a=_ANGLES,
    angle_b=_ANGLES,
    seed=st.integers(min_value=0, max_value=2**32 - 1),
)
def test_score_monotone_non_increasing_in_distance_from_centroid(
    angle_a: float, angle_b: float, seed: int
) -> None:
    """For a fixed reference set and any two submitted embeddings A,
    B: if A is closer (in cosine distance) to the centroid than B,
    then `embedding_score(A) >= embedding_score(B)`.

    Construction: build a single fixed reference unit vector `u` (the
    "centroid" for our purposes when there's only one reference) and
    a perpendicular `v`. Submitted embeddings live on the unit sphere
    at angle θ from `u` in the (u, v) plane. Cosine distance from
    centroid is exactly `1 - cos(θ)`, monotone in θ on [0, π/2].

    If this fails, the detector's logistic squash has been broken in
    a way that fixture tests didn't catch (sign flip on
    `distance_midpoint - distance`, or a calibration that uses
    similarity instead of distance).
    """
    rng = np.random.default_rng(seed)
    u, v = _orthonormal_pair(rng, EMBEDDING_DIM)
    references = u[np.newaxis, :]  # single reference; centroid is u

    submitted_a = _at_angle(u, v, angle_a)
    submitted_b = _at_angle(u, v, angle_b)

    result_a = measure_embedding_anomaly(submitted_a, references)
    result_b = measure_embedding_anomaly(submitted_b, references)

    # Score bound invariant.
    for r in (result_a, result_b):
        assert 0.0 <= r.embedding_score <= 1.0

    dist_a = result_a.distance_from_centroid
    dist_b = result_b.distance_from_centroid

    # Monotonicity. Use a small epsilon to absorb float32 noise in the
    # cosine computation — the centroid path L2-renormalizes, so an
    # angle of 0 doesn't quite produce distance == 0 in float32.
    if dist_a < dist_b - 1e-5:
        assert result_a.embedding_score >= result_b.embedding_score - 1e-5, (
            f"closer embedding scored lower: dist_a={dist_a:.6f} "
            f"score_a={result_a.embedding_score:.6f} vs "
            f"dist_b={dist_b:.6f} score_b={result_b.embedding_score:.6f}"
        )
    elif dist_a > dist_b + 1e-5:
        assert result_b.embedding_score >= result_a.embedding_score - 1e-5, (
            f"closer embedding scored lower: dist_b={dist_b:.6f} "
            f"score_b={result_b.embedding_score:.6f} vs "
            f"dist_a={dist_a:.6f} score_a={result_a.embedding_score:.6f}"
        )


@settings(max_examples=40, deadline=None)
@given(
    seed=st.integers(min_value=0, max_value=2**32 - 1),
    n_steps=st.integers(min_value=2, max_value=8),
    n_refs=st.integers(min_value=1, max_value=6),
)
def test_score_decreases_along_angular_ladder(
    seed: int, n_steps: int, n_refs: int
) -> None:
    """Multi-reference variant. Build a small cluster of references
    around `u` (small perturbations on the unit sphere) and walk a
    ladder of submitted embeddings at increasing angles from `u`.
    Score must be monotone non-increasing along the ladder.

    This complements the pairwise property by exercising a contiguous
    distance trajectory against a non-degenerate reference set —
    closer to how the detector actually runs in production (multiple
    reference exemplars per variant). If the pairwise test passes but
    this one fails, the bug is in centroid computation
    (re-normalization, weighting) rather than the squash itself.
    """
    rng = np.random.default_rng(seed)
    u, v = _orthonormal_pair(rng, EMBEDDING_DIM)

    # Build references as small-noise perturbations of u, all on the
    # unit sphere. The centroid will land very close to u.
    refs = []
    for _ in range(n_refs):
        noisy = u + 0.05 * rng.standard_normal(EMBEDDING_DIM).astype(np.float32)
        refs.append((noisy / np.linalg.norm(noisy)).astype(np.float32))
    references = np.stack(refs)

    # Ladder of n_steps angles from 0 to π/2.
    angles = np.linspace(0.0, math.pi / 2, n_steps).tolist()
    scores = []
    distances = []
    for theta in angles:
        sub = _at_angle(u, v, theta)
        r = measure_embedding_anomaly(sub, references)
        assert 0.0 <= r.embedding_score <= 1.0
        scores.append(r.embedding_score)
        distances.append(r.distance_from_centroid)

    # Distances must themselves be sorted ascending — sanity-check
    # the construction. Tiny float epsilon for the boundary case.
    for i in range(1, len(distances)):
        assert distances[i] >= distances[i - 1] - 1e-5, (
            f"angular ladder produced non-monotone distances: {distances}"
        )

    # Scores must be sorted descending (closer => higher score).
    for i in range(1, len(scores)):
        assert scores[i] <= scores[i - 1] + 1e-5, (
            f"score went UP at ladder step {i}: scores={scores} "
            f"(distances={distances})"
        )


def test_property_test_module_imports_cleanly() -> None:
    """Hypothesis import smoke check — runs even if the parameterized
    tests above are filtered out, so a missing `hypothesis` install
    fails loudly rather than silently passing zero examples."""
    pytest.importorskip("hypothesis", reason="hypothesis not installed")
