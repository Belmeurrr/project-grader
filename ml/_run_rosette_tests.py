"""Inline test runner for the print-rosette FFT counterfeit detector.

Mirrors the pytest assertions in tests/test_rosette.py. Removed once
`uv sync` provides pytest in the local environment."""

from __future__ import annotations

import sys
import traceback

sys.path.insert(0, ".")

import numpy as np

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

ran, passed, failed = 0, 0, 0


def case(name, fn):
    global ran, passed, failed
    ran += 1
    try:
        fn()
        print(f"PASS {name}")
        passed += 1
    except AssertionError as e:
        print(f"FAIL {name}: AssertionError: {e}")
        failed += 1
    except Exception as e:
        print(f"FAIL {name}: {type(e).__name__}: {e}")
        traceback.print_exc()
        failed += 1


# Halftone vs continuous discrimination

def t1():
    r = measure_rosette(synth_halftone_card(cell_size=6))
    assert isinstance(r, RosetteMeasurement)
    assert r.rosette_score >= 0.7, (
        f"halftone scored {r.rosette_score} (peak={r.peak_strength}, "
        f"patches={r.analyzed_patches})"
    )
case("halftone_card_scores_high", t1)


def t2():
    r = measure_rosette(synth_continuous_tone_card())
    assert r.rosette_score <= 0.3, (
        f"continuous scored {r.rosette_score} (peak={r.peak_strength}, "
        f"patches={r.analyzed_patches})"
    )
case("continuous_tone_card_scores_low", t2)


def t3():
    ht = measure_rosette(synth_halftone_card(cell_size=6))
    ct = measure_rosette(synth_continuous_tone_card())
    assert ht.rosette_score > ct.rosette_score
case("halftone_strictly_beats_continuous_tone", t3)


# Monotonicity with halftone clarity

def t4():
    clean = measure_rosette(synth_halftone_card(cell_size=6, jitter=0.0, seed=7))
    jittered = measure_rosette(synth_halftone_card(cell_size=6, jitter=2.0, seed=7))
    assert clean.rosette_score >= jittered.rosette_score
    assert clean.peak_strength > jittered.peak_strength
case("jitter_lowers_rosette_score", t4)


def t5():
    s0 = measure_rosette(synth_halftone_card(cell_size=6, jitter=0.0, seed=11))
    s5 = measure_rosette(synth_halftone_card(cell_size=6, jitter=5.0, seed=11))
    assert s0.peak_strength > s5.peak_strength
case("extreme_jitter_lowers_peak_strength", t5)


# Edge cases

def t6():
    rng = np.random.default_rng(0)
    noise = rng.integers(0, 256, size=(1050, 750, 3), dtype=np.uint8)
    r = measure_rosette(noise)
    if r.analyzed_patches == 0:
        assert r.confidence == 0.0
        assert r.rosette_score == 0.5
    else:
        assert r.rosette_score < 0.5, f"got {r.rosette_score}"
case("random_noise_returns_low_confidence_or_low_score", t6)


def t7():
    r = measure_rosette(synth_card_with_pattern(seed=3))
    if r.analyzed_patches == 0:
        assert r.rosette_score == 0.5
        assert r.confidence == 0.0
    else:
        assert r.rosette_score < 0.7
case("textured_card_does_not_falsely_pass", t7)


# Validation

def t8():
    img = synth_halftone_card().astype(np.float32)
    try:
        measure_rosette(img)
    except ValueError as e:
        assert "uint8" in str(e)
        return
    raise AssertionError("expected ValueError for non-uint8")
case("rejects_non_uint8", t8)


def t9():
    img = synth_halftone_card()[:, :, 0]
    try:
        measure_rosette(img)
    except ValueError as e:
        assert "3-channel" in str(e)
        return
    raise AssertionError("expected ValueError for grayscale")
case("rejects_2d_grayscale", t9)


def t10():
    img = np.zeros((1050, 750, 4), dtype=np.uint8)
    try:
        measure_rosette(img)
    except ValueError as e:
        assert "3-channel" in str(e)
        return
    raise AssertionError("expected ValueError for 4-channel")
case("rejects_non_3_channel", t10)


def t11():
    tiny = np.zeros((50, 50, 3), dtype=np.uint8)
    try:
        measure_rosette(tiny)
    except ValueError as e:
        assert "too small" in str(e)
        return
    raise AssertionError("expected ValueError for too-small")
case("rejects_too_small_image", t11)


def t12():
    img = synth_halftone_card()
    try:
        measure_rosette(img, patch_size=0)
    except ValueError as e:
        assert "patch_size" in str(e)
        return
    raise AssertionError("expected ValueError for patch_size=0")
case("rejects_invalid_patch_size", t12)


def t13():
    img = synth_halftone_card()
    try:
        measure_rosette(img, n_patches=0)
    except ValueError as e:
        assert "n_patches" in str(e)
        return
    raise AssertionError("expected ValueError for n_patches=0")
case("rejects_invalid_n_patches", t13)


def t14():
    img = synth_halftone_card()
    try:
        measure_rosette(img, expected_freq_band=(10.0, 5.0))
    except ValueError as e:
        assert "expected_freq_band" in str(e)
        return
    raise AssertionError("expected ValueError for inverted band")
case("rejects_inverted_freq_band", t14)


def t15():
    img = synth_halftone_card()
    try:
        measure_rosette(img, expected_freq_band=(0.0, 10.0))
    except ValueError as e:
        assert "expected_freq_band" in str(e)
        return
    raise AssertionError("expected ValueError for low<=0")
case("rejects_freq_band_low_zero", t15)


# Output shape

def t16():
    r = measure_rosette(synth_halftone_card(cell_size=6))
    assert isinstance(r, RosetteMeasurement)
    assert isinstance(r.rosette_score, float)
    assert 0.0 <= r.rosette_score <= 1.0
    assert isinstance(r.peak_strength, float)
    assert isinstance(r.analyzed_patches, int)
    assert isinstance(r.confidence, float)
    assert 0.0 <= r.confidence <= 1.0
    assert r.manufacturer_profile == "generic"
    assert isinstance(r.metadata, dict)
case("output_dataclass_shape", t16)


def t17():
    r = measure_rosette(synth_halftone_card(cell_size=6))
    try:
        r.rosette_score = 0.0  # type: ignore[misc]
    except (AttributeError, Exception):
        return
    raise AssertionError("expected mutation to fail on frozen dataclass")
case("output_is_immutable", t17)


def t18():
    n = 4
    r = measure_rosette(synth_halftone_card(cell_size=6), n_patches=n)
    assert 0 <= r.analyzed_patches <= n
case("analyzed_patches_within_bound", t18)


def t19():
    r5 = measure_rosette(synth_halftone_card(cell_size=6), n_patches=5)
    r20 = measure_rosette(synth_halftone_card(cell_size=6), n_patches=20)
    assert r5.confidence == 1.0
    assert 0.0 <= r20.confidence <= 1.0
    assert r20.analyzed_patches >= r5.analyzed_patches
case("confidence_drops_when_few_patches", t19)


# is_likely_authentic

def t20():
    high = measure_rosette(synth_halftone_card(cell_size=6))
    low = measure_rosette(synth_continuous_tone_card())
    assert is_likely_authentic(high) is (high.rosette_score >= 0.5)
    assert is_likely_authentic(low) is (low.rosette_score >= 0.5)
case("is_likely_authentic_matches_threshold_compare", t20)


def t21():
    m = measure_rosette(synth_continuous_tone_card())
    assert is_likely_authentic(m, threshold=0.0) is True
    assert is_likely_authentic(m, threshold=1.0) is False
case("is_likely_authentic_respects_custom_threshold", t21)


print()
print(f"{passed}/{ran} passed, {failed} failed")
sys.exit(0 if failed == 0 else 1)
