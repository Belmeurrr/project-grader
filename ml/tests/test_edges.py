"""Edges measurement tests.

Verifies the geometric edge-defect detection on synthetic canonical cards.
The PSA subgrade table is calibrated against perimeter-anomaly fraction;
these tests pin its monotonicity and the side-localization behavior."""

from __future__ import annotations

import numpy as np
import pytest

from pipelines.grading.edges import (
    EdgeMeasurement,
    SideMeasurement,
    measure_edges,
    psa_subgrade_from_anomaly_fraction,
)
from pipelines.grading.edges.measure import SIDES
from tests.fixtures import canonical_clean, canonical_with_edge_defect


# -----------------------------
# Subgrade lookup table
# -----------------------------


def test_psa_subgrade_clean_is_10() -> None:
    assert psa_subgrade_from_anomaly_fraction(0.0) == 10.0
    assert psa_subgrade_from_anomaly_fraction(0.001) == 10.0


def test_psa_subgrade_monotonic_decreasing() -> None:
    fractions = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.8]
    grades = [psa_subgrade_from_anomaly_fraction(f) for f in fractions]
    for a, b in zip(grades, grades[1:]):
        assert a >= b, f"grades not monotonic: {grades}"


def test_psa_subgrade_extreme_is_1() -> None:
    assert psa_subgrade_from_anomaly_fraction(1.0) == 1.0
    assert psa_subgrade_from_anomaly_fraction(0.99) == 1.0


def test_psa_subgrade_rejects_negative() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        psa_subgrade_from_anomaly_fraction(-0.01)


# -----------------------------
# measure_edges output shape
# -----------------------------


def test_measure_edges_returns_4_sides() -> None:
    result = measure_edges(canonical_clean())
    assert isinstance(result, EdgeMeasurement)
    assert len(result.sides) == 4
    assert {s.side for s in result.sides} == set(SIDES)


def test_measure_edges_sides_are_in_canonical_order() -> None:
    result = measure_edges(canonical_clean())
    assert [s.side for s in result.sides] == list(SIDES)


def test_measure_edges_subgrade_in_valid_range() -> None:
    result = measure_edges(canonical_clean())
    assert 1.0 <= result.psa_subgrade <= 10.0


def test_measure_edges_confidence_in_unit_range() -> None:
    result = measure_edges(canonical_clean())
    assert 0.0 <= result.confidence <= 1.0


def test_measure_edges_heatmap_matches_input_dimensions() -> None:
    img = canonical_clean()
    result = measure_edges(img, return_heatmap=True)
    assert result.heatmap is not None
    assert result.heatmap.shape == img.shape[:2]
    assert result.heatmap.dtype == np.uint8
    assert set(np.unique(result.heatmap)).issubset({0, 255})


def test_measure_edges_heatmap_omitted_when_disabled() -> None:
    result = measure_edges(canonical_clean(), return_heatmap=False)
    assert result.heatmap is None


# -----------------------------
# Clean baseline behavior
# -----------------------------


def test_clean_card_grades_high() -> None:
    result = measure_edges(canonical_clean())
    assert result.psa_subgrade >= 9.0, (
        f"clean card scored {result.psa_subgrade}; "
        f"sides: {[(s.side, s.anomaly_fraction) for s in result.sides]}"
    )


def test_clean_card_has_low_overall_anomaly_fraction() -> None:
    result = measure_edges(canonical_clean())
    assert result.overall_anomaly_fraction < 0.02


# -----------------------------
# Side-localization
# -----------------------------


@pytest.mark.parametrize("side", ["top", "right", "bottom", "left"])
def test_chip_localizes_to_correct_side(side: str) -> None:
    result = measure_edges(canonical_with_edge_defect(side=side, length_px=80, severity="chip"))
    side_scores = {s.side: s.anomaly_fraction for s in result.sides}
    other_max = max(score for s, score in side_scores.items() if s != side)
    assert side_scores[side] > other_max, (
        f"defect on {side}, got {side_scores}"
    )


def test_defect_drops_overall_subgrade() -> None:
    clean = measure_edges(canonical_clean()).psa_subgrade
    defect = measure_edges(
        canonical_with_edge_defect(side="top", length_px=200, severity="chip")
    ).psa_subgrade
    assert defect < clean


# -----------------------------
# Monotonic severity
# -----------------------------


def test_subgrade_monotonic_with_chip_size() -> None:
    sizes = [20, 80, 160, 320]
    grades = [
        measure_edges(canonical_with_edge_defect("top", length_px=s, severity="chip")).psa_subgrade
        for s in sizes
    ]
    for a, b in zip(grades, grades[1:]):
        assert a >= b, f"larger chip didn't lower grade: {list(zip(sizes, grades))}"


def test_whitening_also_drops_subgrade() -> None:
    clean = measure_edges(canonical_clean()).psa_subgrade
    whitening = measure_edges(
        canonical_with_edge_defect(side="left", length_px=200, severity="whitening")
    ).psa_subgrade
    assert whitening < clean


# -----------------------------
# Validation
# -----------------------------


def test_measure_edges_rejects_non_uint8() -> None:
    with pytest.raises(ValueError, match="uint8"):
        measure_edges(canonical_clean().astype(np.float32))


def test_measure_edges_rejects_too_small_image() -> None:
    tiny = np.zeros((10, 10, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="too small"):
        measure_edges(tiny, strip_px=8)


def test_measure_edges_accepts_grayscale_input() -> None:
    """Production canonical is BGR, but the algorithm works on grayscale too —
    callers that already have luminance shouldn't pay a conversion cost."""
    gray = np.full((1050, 750), 200, dtype=np.uint8)
    result = measure_edges(gray)
    assert result.psa_subgrade == 10.0  # totally uniform → no anomalies
