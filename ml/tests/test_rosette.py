"""Print-rosette FFT counterfeit detector tests.

Pins the public contract of `measure_rosette` against synthetic halftone
and continuous-tone fixtures. The detector is deterministic and threshold-
based, so these tests act as both correctness checks and a regression
guard for future algorithm tweaks (changing the prominence midpoint or
patch-size default would have to update the calibrated assertions here)."""

from __future__ import annotations

import numpy as np
import pytest

from pipelines.counterfeit.rosette import (
    RosetteMeasurement,
    is_likely_authentic,
    measure_rosette,
)
from tests.fixtures import (
    synth_card_with_pattern,
    synth_continuous_tone_card,
    synth_halftone_card,
)


# -----------------------------
# Halftone vs. continuous-tone discrimination (the primary signal)
# -----------------------------


def test_halftone_card_scores_high() -> None:
    """An authentic-style halftone print produces a strong rosette peak in
    the configured frequency band → score should be near 1.0."""
    result = measure_rosette(synth_halftone_card(cell_size=6))
    assert isinstance(result, RosetteMeasurement)
    assert result.rosette_score >= 0.7, (
        f"halftone scored {result.rosette_score} (peak={result.peak_strength}, "
        f"patches={result.analyzed_patches})"
    )


def test_continuous_tone_card_scores_low() -> None:
    """An inkjet-style continuous-tone print has no halftone peak → low score."""
    result = measure_rosette(synth_continuous_tone_card())
    assert result.rosette_score <= 0.3, (
        f"continuous-tone scored {result.rosette_score} "
        f"(peak={result.peak_strength}, patches={result.analyzed_patches})"
    )


def test_halftone_strictly_beats_continuous_tone() -> None:
    """Sanity: even after recalibration, halftone must always score higher
    than continuous-tone on the same fixtures."""
    ht = measure_rosette(synth_halftone_card(cell_size=6))
    ct = measure_rosette(synth_continuous_tone_card())
    assert ht.rosette_score > ct.rosette_score


# -----------------------------
# Monotonicity with halftone clarity
# -----------------------------


def test_jitter_lowers_rosette_score() -> None:
    """More dot-position jitter = noisier halftone pattern = weaker FFT peak.
    Specifically: a perfectly-periodic grid must score higher than a
    significantly-jittered one."""
    clean = measure_rosette(synth_halftone_card(cell_size=6, jitter=0.0, seed=7))
    jittered = measure_rosette(synth_halftone_card(cell_size=6, jitter=2.0, seed=7))
    assert clean.rosette_score >= jittered.rosette_score
    # Peak strength is the more sensitive signal; it should drop noticeably.
    assert clean.peak_strength > jittered.peak_strength


def test_extreme_jitter_lowers_peak_strength() -> None:
    sweep = [
        measure_rosette(synth_halftone_card(cell_size=6, jitter=j, seed=11))
        for j in (0.0, 5.0)
    ]
    assert sweep[0].peak_strength > sweep[1].peak_strength


# -----------------------------
# Edge / degenerate cases
# -----------------------------


def test_random_noise_image_returns_low_confidence_or_low_score() -> None:
    """A pure-noise image has no flat patches and no rosette structure;
    confidence should be low and the score should not falsely claim
    authenticity."""
    rng = np.random.default_rng(0)
    noise = rng.integers(0, 256, size=(1050, 750, 3), dtype=np.uint8)
    result = measure_rosette(noise)
    # Either no flat patches were found (confidence 0, score 0.5 = uncertain)
    # OR patches were found but the score should be low (no rosette).
    if result.analyzed_patches == 0:
        assert result.confidence == 0.0
        assert result.rosette_score == 0.5
    else:
        assert result.rosette_score < 0.5


def test_textured_card_still_finds_some_flat_patches_or_is_uncertain() -> None:
    """A card with random rectangles should be either fully analyzed (with
    a low-ish score) or report few flat patches; either way it must not
    look authentic."""
    result = measure_rosette(synth_card_with_pattern(seed=3))
    if result.analyzed_patches == 0:
        assert result.rosette_score == 0.5
        assert result.confidence == 0.0
    else:
        # No halftone structure → must not pass the authenticity threshold.
        assert result.rosette_score < 0.7


# -----------------------------
# Validation
# -----------------------------


def test_rejects_non_uint8() -> None:
    img = synth_halftone_card().astype(np.float32)
    with pytest.raises(ValueError, match="uint8"):
        measure_rosette(img)


def test_rejects_2d_grayscale() -> None:
    """Production input is BGR canonical; the rosette algorithm currently
    only accepts 3-channel input (it converts internally)."""
    img = synth_halftone_card()[:, :, 0]
    with pytest.raises(ValueError, match="3-channel"):
        measure_rosette(img)


def test_rejects_non_3_channel() -> None:
    img = np.zeros((1050, 750, 4), dtype=np.uint8)  # RGBA
    with pytest.raises(ValueError, match="3-channel"):
        measure_rosette(img)


def test_rejects_too_small_image() -> None:
    tiny = np.zeros((50, 50, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="too small"):
        measure_rosette(tiny)


def test_rejects_invalid_patch_size() -> None:
    img = synth_halftone_card()
    with pytest.raises(ValueError, match="patch_size"):
        measure_rosette(img, patch_size=0)


def test_rejects_invalid_n_patches() -> None:
    img = synth_halftone_card()
    with pytest.raises(ValueError, match="n_patches"):
        measure_rosette(img, n_patches=0)


def test_rejects_invalid_freq_band() -> None:
    img = synth_halftone_card()
    with pytest.raises(ValueError, match="expected_freq_band"):
        measure_rosette(img, expected_freq_band=(10.0, 5.0))  # high < low
    with pytest.raises(ValueError, match="expected_freq_band"):
        measure_rosette(img, expected_freq_band=(0.0, 10.0))  # low not > 0


# -----------------------------
# Output dataclass shape
# -----------------------------


def test_output_is_frozen_dataclass_with_all_fields() -> None:
    result = measure_rosette(synth_halftone_card(cell_size=6))
    assert isinstance(result, RosetteMeasurement)
    assert isinstance(result.rosette_score, float)
    assert 0.0 <= result.rosette_score <= 1.0
    assert isinstance(result.peak_strength, float)
    assert isinstance(result.analyzed_patches, int)
    assert isinstance(result.confidence, float)
    assert 0.0 <= result.confidence <= 1.0
    assert isinstance(result.manufacturer_profile, str)
    assert result.manufacturer_profile == "generic"
    assert isinstance(result.metadata, dict)


def test_output_is_immutable() -> None:
    result = measure_rosette(synth_halftone_card(cell_size=6))
    with pytest.raises((AttributeError, Exception)):
        result.rosette_score = 0.0  # type: ignore[misc]


def test_analyzed_patches_within_requested_bound() -> None:
    """analyzed_patches must be 0..n_patches inclusive."""
    n = 4
    result = measure_rosette(synth_halftone_card(cell_size=6), n_patches=n)
    assert 0 <= result.analyzed_patches <= n


def test_confidence_drops_when_few_patches() -> None:
    """Confidence is linear in patches_found / n_patches; if we ask for
    more patches than the inner area can yield, confidence should reflect
    the shortfall (may be < 1)."""
    # Both calls should run; the n=20 ask may or may not all be filled.
    r5 = measure_rosette(synth_halftone_card(cell_size=6), n_patches=5)
    r20 = measure_rosette(synth_halftone_card(cell_size=6), n_patches=20)
    assert r5.confidence == 1.0  # 5/5 patches found
    # n=20 will be capped at however many flat regions actually exist.
    assert 0.0 <= r20.confidence <= 1.0
    assert r20.analyzed_patches >= r5.analyzed_patches


# -----------------------------
# is_likely_authentic
# -----------------------------


def test_is_likely_authentic_matches_threshold_compare() -> None:
    high = measure_rosette(synth_halftone_card(cell_size=6))
    low = measure_rosette(synth_continuous_tone_card())
    assert is_likely_authentic(high) is (high.rosette_score >= 0.5)
    assert is_likely_authentic(low) is (low.rosette_score >= 0.5)


def test_is_likely_authentic_respects_custom_threshold() -> None:
    m = measure_rosette(synth_continuous_tone_card())
    # With threshold 0.0, anything counts as authentic.
    assert is_likely_authentic(m, threshold=0.0) is True
    # With threshold 1.0, nothing counts.
    assert is_likely_authentic(m, threshold=1.0) is False
