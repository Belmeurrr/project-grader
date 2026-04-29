"""Color-profile counterfeit detector tests.

Pins the public contract of `measure_color_profile` against synthetic
high-chroma vs. desaturated fixtures. Mirrors test_rosette.py — the
detector is deterministic and threshold-based, so these tests act as
both correctness checks and a regression guard for future calibration
tweaks (changing CHROMA_MIDPOINT or the gain cap would have to update
the calibrated assertions here)."""

from __future__ import annotations

import numpy as np
import pytest

from pipelines.counterfeit.color import (
    ColorProfileMeasurement,
    is_likely_authentic,
    measure_color_profile,
)
from tests.fixtures import synth_card


# -----------------------------
# High-chroma vs desaturated discrimination (the primary signal)
# -----------------------------


def test_high_chroma_card_scores_high() -> None:
    """An authentic-style high-saturation print produces high chroma in
    the inner area → score should be near 1.0."""
    # Pure saturated blue in BGR (B=255, G=0, R=0). CIELAB chroma is well
    # above the authentic baseline.
    card = synth_card(image_color=(255, 0, 0))
    result = measure_color_profile(card)
    assert isinstance(result, ColorProfileMeasurement)
    assert result.color_score >= 0.85, (
        f"high-chroma scored {result.color_score} "
        f"(p95_chroma={result.p95_chroma}, conf={result.confidence})"
    )


def test_low_chroma_card_scores_low() -> None:
    """A desaturated inkjet-style print clips at lower chroma → low score."""
    # Muted gray-blue (BGR=(140, 130, 120)): close to neutral, p95 chroma
    # should be in the inkjet-cap range (<30).
    card = synth_card(image_color=(140, 130, 120))
    result = measure_color_profile(card)
    assert result.color_score <= 0.20, (
        f"low-chroma scored {result.color_score} "
        f"(p95_chroma={result.p95_chroma}, conf={result.confidence})"
    )


def test_high_chroma_strictly_beats_low_chroma() -> None:
    """Sanity: even after recalibration, saturated must always score
    higher than desaturated on the same fixtures."""
    high = measure_color_profile(synth_card(image_color=(255, 0, 0)))
    low = measure_color_profile(synth_card(image_color=(140, 130, 120)))
    assert high.color_score > low.color_score


# -----------------------------
# Monotonicity with chroma
# -----------------------------


def test_chroma_monotone_with_saturation() -> None:
    """As we walk from neutral to saturated colors, the score should be
    monotonically non-decreasing (chroma signal increases)."""
    # Walk along the blue axis: closer to saturated blue → higher chroma.
    walks = [
        synth_card(image_color=(140, 140, 140)),  # near-gray
        synth_card(image_color=(180, 100, 100)),  # mild blue
        synth_card(image_color=(220, 60, 60)),    # strong blue
        synth_card(image_color=(255, 0, 0)),      # saturated blue
    ]
    scores = [measure_color_profile(c).color_score for c in walks]
    # Allow ties (logistic saturation), but no drops.
    for i in range(len(scores) - 1):
        assert scores[i] <= scores[i + 1] + 1e-3, (
            f"non-monotone at step {i}: scores={scores}"
        )


# -----------------------------
# White-balance calibration
# -----------------------------


def test_calibration_corrects_warm_lighting_cast() -> None:
    """A card photographed under warm tungsten light has its border tinted
    yellowish. After white-balance calibration, the inner-area chroma
    should reflect the inks, not the lighting."""
    # Simulate warm cast: border still ~white-ish but with reduced blue
    # channel; inner image is saturated red (high chroma either way).
    warm_border = (210, 222, 230)   # less blue than 230,230,230 baseline
    saturated = synth_card(image_color=(0, 0, 255), border_color=warm_border)
    neutral = synth_card(image_color=(0, 0, 255))
    warm_result = measure_color_profile(saturated)
    neutral_result = measure_color_profile(neutral)
    # Both should score high (the inner color is genuinely saturated).
    # Critically, the calibration should NOT crash, and the warm result
    # should be within ~0.1 of the neutral result — calibration removed
    # most of the lighting confounder.
    assert warm_result.color_score >= 0.7
    assert abs(warm_result.color_score - neutral_result.color_score) <= 0.15


def test_unreliable_border_returns_uncertain() -> None:
    """A heavily-tinted 'border' that's nowhere near white should mark the
    measurement as low-confidence. The detector should NOT confidently
    assert authentic or counterfeit when its calibration reference is bad."""
    # Border that's very dark green: failed dewarp landed us on the card's
    # printed area, not the white border.
    bad_border = synth_card(border_color=(20, 80, 20), image_color=(255, 0, 0))
    result = measure_color_profile(bad_border)
    # Either: low confidence (correctly abstaining) OR uncertain score
    # (~0.5). Both are acceptable outcomes for unreliable input.
    assert result.confidence < 0.5 or 0.3 < result.color_score < 0.7, (
        f"bad-border result was unexpectedly confident: "
        f"score={result.color_score} conf={result.confidence}"
    )


def test_underexposed_card_returns_uncertain() -> None:
    """If the entire card is severely underexposed, the 'white' border has
    luminance well below the reliability floor — should abstain."""
    dark = synth_card(border_color=(60, 60, 60), image_color=(40, 20, 20))
    result = measure_color_profile(dark)
    assert result.confidence == 0.0
    assert 0.4 <= result.color_score <= 0.6


# -----------------------------
# Input validation
# -----------------------------


def test_rejects_non_uint8() -> None:
    img = np.zeros((1050, 750, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="uint8"):
        measure_color_profile(img)


def test_rejects_2d_grayscale() -> None:
    img = np.zeros((1050, 750), dtype=np.uint8)
    with pytest.raises(ValueError, match="3-channel"):
        measure_color_profile(img)


def test_rejects_non_3_channel() -> None:
    img = np.zeros((1050, 750, 4), dtype=np.uint8)
    with pytest.raises(ValueError, match="3-channel"):
        measure_color_profile(img)


def test_rejects_too_small_image() -> None:
    img = np.zeros((20, 20, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="too small"):
        measure_color_profile(img)


def test_rejects_invalid_inner_inset() -> None:
    img = synth_card()
    with pytest.raises(ValueError, match="inner_inset_px"):
        measure_color_profile(img, inner_inset_px=0)


def test_rejects_invalid_border_depth() -> None:
    img = synth_card()
    with pytest.raises(ValueError, match="depth"):
        measure_color_profile(img, border_sample_depth_px=0)


# -----------------------------
# Output contract
# -----------------------------


def test_output_is_frozen_dataclass_with_all_fields() -> None:
    result = measure_color_profile(synth_card(image_color=(255, 0, 0)))
    # Required fields on the contract
    assert isinstance(result.color_score, float)
    assert isinstance(result.p95_chroma, float)
    assert isinstance(result.border_white_bgr, tuple)
    assert len(result.border_white_bgr) == 3
    assert isinstance(result.border_stddev, float)
    assert isinstance(result.gain_applied, tuple)
    assert len(result.gain_applied) == 3
    assert isinstance(result.confidence, float)
    assert isinstance(result.manufacturer_profile, str)
    assert isinstance(result.metadata, dict)
    # Score and confidence in [0, 1]
    assert 0.0 <= result.color_score <= 1.0
    assert 0.0 <= result.confidence <= 1.0


def test_output_is_immutable() -> None:
    result = measure_color_profile(synth_card(image_color=(255, 0, 0)))
    with pytest.raises((AttributeError, Exception)):
        result.color_score = 0.0  # type: ignore[misc]


def test_metadata_includes_pixel_counts() -> None:
    result = measure_color_profile(synth_card(image_color=(255, 0, 0)))
    # On a successful measurement these are populated.
    if result.confidence > 0:
        assert result.metadata.get("border_pixels_sampled", 0) > 0
        assert result.metadata.get("inner_pixels_analyzed", 0) > 0


# -----------------------------
# is_likely_authentic predicate
# -----------------------------


def test_is_likely_authentic_matches_threshold_compare() -> None:
    high = measure_color_profile(synth_card(image_color=(255, 0, 0)))
    low = measure_color_profile(synth_card(image_color=(140, 130, 120)))
    assert is_likely_authentic(high) is True
    assert is_likely_authentic(low) is False


def test_is_likely_authentic_respects_custom_threshold() -> None:
    result = measure_color_profile(synth_card(image_color=(180, 100, 100)))
    # At a stricter threshold, marginal cards become "not authentic".
    assert is_likely_authentic(result, threshold=0.99) is False
