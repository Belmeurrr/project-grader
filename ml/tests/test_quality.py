"""Quality pipeline tests — blur, glare, card detection, perspective, full report."""

from __future__ import annotations

import numpy as np
import pytest

from pipelines.quality import (
    QualityThresholds,
    blur_score,
    detect_card_bbox,
    evaluate_shot,
    fill_ratio,
    glare_score,
    has_glare,
    is_sharp,
    perspective_deg,
)
from tests.fixtures import (
    blurry,
    card_in_scene,
    synth_card,
    with_glare,
)


# -----------------------------
# Blur
# -----------------------------


def test_sharp_synthetic_card_is_sharp() -> None:
    img = card_in_scene()
    assert is_sharp(img, threshold=100.0)


def test_blurred_card_fails_sharpness_check() -> None:
    img = blurry(card_in_scene(), k=35)
    assert not is_sharp(img, threshold=100.0)
    assert blur_score(img) < 100.0


def test_blur_score_is_higher_on_sharper_image() -> None:
    sharp = card_in_scene()
    soft = blurry(sharp, k=15)
    assert blur_score(sharp) > blur_score(soft)


# -----------------------------
# Glare
# -----------------------------


def test_clean_card_has_negligible_glare() -> None:
    img = card_in_scene()
    assert glare_score(img) < 0.005


def test_glare_blob_triggers_detector() -> None:
    img = with_glare(card_in_scene(), fraction=0.05)
    assert glare_score(img) >= 0.04
    assert has_glare(img, max_fraction=0.005)


def test_glare_score_rejects_non_bgr() -> None:
    gray = np.zeros((100, 100), dtype=np.uint8)
    with pytest.raises(ValueError, match="BGR"):
        glare_score(gray)


# -----------------------------
# Card detection / fill / perspective
# -----------------------------


def test_card_detected_in_clean_scene() -> None:
    img = card_in_scene(fill=0.55)
    card = detect_card_bbox(img)
    assert card is not None
    assert 0.10 <= fill_ratio(card, img.shape[:2]) <= 0.95


def test_card_not_detected_in_empty_scene() -> None:
    blank = np.full((1000, 1000, 3), 35, dtype=np.uint8)
    card = detect_card_bbox(blank)
    assert card is None


def test_perspective_deg_small_for_overhead_shot() -> None:
    img = card_in_scene(fill=0.55, perspective_skew_px=0)
    card = detect_card_bbox(img)
    assert card is not None
    assert perspective_deg(card) < 3.0


def test_perspective_deg_large_for_tilted_shot() -> None:
    img = card_in_scene(fill=0.55, perspective_skew_px=120)
    card = detect_card_bbox(img)
    assert card is not None
    assert perspective_deg(card) > 8.0


# -----------------------------
# Full evaluate_shot report
# -----------------------------


def test_clean_front_full_passes() -> None:
    img = card_in_scene(fill=0.55)
    report = evaluate_shot(img, "front_full")
    assert report.passed, report.reasons
    assert report.card_detected
    assert report.blur is not None and report.blur > 100
    assert report.glare is not None and report.glare < 0.005


def test_blurry_shot_fails_with_blur_reason() -> None:
    img = blurry(card_in_scene(fill=0.55), k=35)
    report = evaluate_shot(img, "front_full")
    assert not report.passed
    assert any("blurry" in r for r in report.reasons)


def test_glary_shot_fails_with_glare_reason() -> None:
    img = with_glare(card_in_scene(fill=0.55), fraction=0.05)
    report = evaluate_shot(img, "front_full")
    assert not report.passed
    assert any("glare" in r for r in report.reasons)


def test_flash_shot_tolerates_glare_that_would_fail_no_flash() -> None:
    img = with_glare(card_in_scene(fill=0.55), fraction=0.04)
    no_flash = evaluate_shot(img, "front_full")
    flash = evaluate_shot(img, "front_full_flash")
    assert not no_flash.passed
    # The flash shot may still fail other checks but must not fail on glare alone.
    assert not any("glare" in r for r in flash.reasons), flash.reasons


def test_card_too_far_fails_with_size_reason() -> None:
    img = card_in_scene(fill=0.20)
    report = evaluate_shot(img, "front_full")
    assert not report.passed
    assert any(("too small" in r) or ("Couldn't find the card" in r) for r in report.reasons)


def test_corner_zoom_uses_relaxed_thresholds() -> None:
    img = card_in_scene(fill=0.30)
    report = evaluate_shot(img, "corner_tl")
    # corner zooms have lower min_fill and don't require detection
    assert not any("too small" in r for r in report.reasons)


def test_evaluate_shot_rejects_non_uint8() -> None:
    img = card_in_scene().astype(np.float32)
    with pytest.raises(ValueError, match="uint8"):
        evaluate_shot(img, "front_full")


def test_quality_report_serializes_to_dict() -> None:
    img = card_in_scene(fill=0.55)
    report = evaluate_shot(img, "front_full")
    d = report.to_dict()
    assert d["passed"] is True
    assert "blur" in d and "glare" in d


def test_custom_thresholds_override_defaults() -> None:
    img = card_in_scene(fill=0.55)
    strict = QualityThresholds(min_blur=10_000.0)  # impossibly strict
    report = evaluate_shot(img, "front_full", thresholds=strict)
    assert not report.passed
    assert any("blurry" in r for r in report.reasons)
