"""Property-based tests for the typography counterfeit detector.

The fixture-based tests in `test_typography_detector.py` pin the
score/confidence shape at a handful of hand-chosen edit distances —
enough to catch a regression that breaks the obvious cases, but blind
to drift in the calibration constants. If a future commit lowers
`DEFAULT_DISTANCE_MIDPOINT` from 0.35 to 0.20, several of those pinned
fixtures will keep passing because the picked distances still land on
the right side of the threshold.

This file expresses the calibration-invariant property directly: for a
fixed expected name and a fixed pair of generated extracted strings,
**the one with smaller Levenshtein distance must score >= the one
with larger Levenshtein distance**. That property holds for any
midpoint, any slope, any future per-game profile — the score function
is logistic-monotone in distance by construction. If it ever stops
holding, the calibration code is broken regardless of the constant
values.

We also assert score-bound invariants (output stays in [0, 1] for any
input the detector accepts without abstaining), which guards against
overflow / underflow in the logistic squash.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from hypothesis import HealthCheck, assume, given, settings, strategies as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipelines.counterfeit.typography import analyze_typography  # noqa: E402
from pipelines.counterfeit.typography.detector import (  # noqa: E402
    _levenshtein,
    _normalize_for_compare,
)


# A canonical-shaped image. Every test uses the same one so the OCR
# stub is the only varying input — keeps the property attributable to
# the detector's text-comparison code, not to ROI cropping.
_CANONICAL_IMAGE = np.full((1050, 750, 3), 200, dtype=np.uint8)


def _stub_ocr(text: str):
    """OCR-fn stub returning a fixed string regardless of input image.
    Mirrors test_typography_detector.py's helper."""

    def _fn(_: np.ndarray) -> list[str]:
        return [text]

    return _fn


# Strategies. Ascii-letters-only keeps generated names away from the
# unicode normalization edge cases that aren't part of the calibration
# property under test (`_normalize_for_compare` strips punctuation, so
# we sidestep that whole branch by sticking to alnum). Length capped
# at 30 — real card names are usually 5-30 chars; the property holds
# at any length but tiny inputs stress the denominator clamp.
_LOWER_LETTERS = "abcdefghijklmnopqrstuvwxyz"
_NAMES = st.text(
    alphabet=_LOWER_LETTERS + " ",
    min_size=3,
    max_size=30,
).map(lambda s: s.strip()).filter(lambda s: len(s) >= 3 and any(c.isalpha() for c in s))


@settings(
    max_examples=80,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(expected=_NAMES, extracted_a=_NAMES, extracted_b=_NAMES)
def test_score_monotone_non_increasing_in_levenshtein_distance(
    expected: str, extracted_a: str, extracted_b: str
) -> None:
    """For a fixed expected name and any two extracted strings A and B,
    if the normalized Levenshtein distance of A is <= that of B, then
    the typography score for A must be >= the score for B.

    This is the invariant the calibration constants must always
    preserve. A future change to DEFAULT_DISTANCE_MIDPOINT or
    DEFAULT_DISTANCE_SLOPE shifts the absolute scores but cannot break
    this ordering — the logistic is monotone-decreasing in
    `distance_norm` by construction. If this property fails, the
    detector's distance-to-score mapping has been broken (e.g. someone
    accidentally swapped the sign on `distance_midpoint - distance`).
    """
    norm_expected = _normalize_for_compare(expected)
    norm_a = _normalize_for_compare(extracted_a)
    norm_b = _normalize_for_compare(extracted_b)
    # Skip degenerate normalizations (the helper can collapse to empty).
    assume(norm_expected and norm_a and norm_b)

    # Compute distances exactly the same way the detector does.
    dist_a = _levenshtein(norm_a, norm_expected) / max(
        len(norm_expected), len(norm_a), 1
    )
    dist_b = _levenshtein(norm_b, norm_expected) / max(
        len(norm_expected), len(norm_b), 1
    )

    result_a = analyze_typography(
        _CANONICAL_IMAGE, expected, ocr_fn=_stub_ocr(extracted_a)
    )
    result_b = analyze_typography(
        _CANONICAL_IMAGE, expected, ocr_fn=_stub_ocr(extracted_b)
    )
    # Both must be non-abstain (we generated valid expected + image).
    assert result_a.abstain_reason is None
    assert result_b.abstain_reason is None

    # Score bound invariant (cheap; catches logistic overflow).
    for r in (result_a, result_b):
        assert 0.0 <= r.score <= 1.0

    # The monotonicity direction itself.
    if dist_a < dist_b:
        assert result_a.score >= result_b.score - 1e-6, (
            f"closer match scored lower: dist_a={dist_a:.4f} score_a="
            f"{result_a.score:.4f} vs dist_b={dist_b:.4f} score_b="
            f"{result_b.score:.4f}"
        )
    elif dist_a > dist_b:
        assert result_b.score >= result_a.score - 1e-6, (
            f"closer match scored lower: dist_b={dist_b:.4f} score_b="
            f"{result_b.score:.4f} vs dist_a={dist_a:.4f} score_a="
            f"{result_a.score:.4f}"
        )
    else:
        # Equal distances must yield equal scores (deterministic
        # function of distance once the OCR stub fixes the input).
        assert abs(result_a.score - result_b.score) < 1e-6


@settings(max_examples=50, deadline=None)
@given(expected=_NAMES, n_edits=st.integers(min_value=0, max_value=10))
def test_more_edits_never_increases_score(expected: str, n_edits: int) -> None:
    """A targeted ladder: starting from a perfect match and prepending
    junk characters, each step adds one edit. Score must never go UP
    along the ladder.

    This complements the unordered-pair test above by exercising a
    path through edit-distance space directly: the ladder constructs
    increasing distances by construction, so we can assert score is
    sorted descending without re-deriving the distance ordering.
    """
    norm_expected = _normalize_for_compare(expected)
    assume(norm_expected)
    # Generate ladder of strings with monotonically growing prefix.
    ladder = [expected]
    for i in range(n_edits):
        # Prepend a single distinct character — adds one substitution
        # at minimum (and possibly an insertion). Use the tail of the
        # alphabet to avoid accidentally matching characters in
        # `expected` and shrinking the distance.
        ladder.append("z" * (i + 1) + " " + expected)

    scores = []
    for s in ladder:
        result = analyze_typography(
            _CANONICAL_IMAGE, expected, ocr_fn=_stub_ocr(s)
        )
        assume(result.abstain_reason is None)
        scores.append(result.score)

    for i in range(1, len(scores)):
        assert scores[i] <= scores[i - 1] + 1e-6, (
            f"score went UP at ladder step {i}: scores={scores}"
        )
