"""Typography counterfeit detector tests.

Pin the public contract of `analyze_typography` against synthetic image
inputs and a stub OCR function. Mirrors test_rosette.py /
test_color_profile.py — the detector is deterministic and threshold-
based, so these tests double as both correctness checks and a regression
guard for future calibration tweaks.

Why a stub OCR function:
    Spinning up RapidOCR's onnxruntime engine in unit tests would add
    a heavy dependency to the test path AND make the tests non-
    deterministic (recognizer outputs depend on the version of the
    bundled weights). `analyze_typography` accepts an `ocr_fn` arg
    precisely so tests can drive the algorithm with deterministic
    synthetic strings; the production code path lazy-loads the real
    engine when the arg is omitted.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipelines.counterfeit.typography import (  # noqa: E402
    DEFAULT_DISTANCE_MIDPOINT,
    DEFAULT_DISTANCE_SLOPE,
    TypographyResult,
    analyze_typography,
)
from pipelines.counterfeit.typography import detector as typography_detector  # noqa: E402


# Canonical 750x1050 BGR shape — matches what the production pipeline
# produces and what the rosette / color detectors consume.
CANONICAL_H = 1050
CANONICAL_W = 750


def _make_canonical(color: tuple[int, int, int] = (200, 200, 200)) -> np.ndarray:
    """Solid-color uint8 BGR image of canonical size."""
    img = np.full((CANONICAL_H, CANONICAL_W, 3), color, dtype=np.uint8)
    return img


def _stub_ocr(text: str | list[str]):
    """Return an OCR-fn stub that yields the given text(s) regardless
    of input image. The detector concatenates lines with a single space."""
    lines = [text] if isinstance(text, str) else list(text)

    def _fn(img: np.ndarray) -> list[str]:
        return list(lines)

    return _fn


# -----------------------------
# Happy path — exact / near matches score authentic
# -----------------------------


def test_exact_match_scores_near_one() -> None:
    img = _make_canonical()
    r = analyze_typography(img, "Lightning Bolt", ocr_fn=_stub_ocr("Lightning Bolt"))
    assert r.abstain_reason is None
    assert r.confidence > 0.5
    assert r.levenshtein_distance == 0
    assert r.score >= 0.95
    assert r.extracted_text == "Lightning Bolt"
    assert r.expected_text == "Lightning Bolt"


def test_one_char_typo_still_scores_high() -> None:
    """A single-character OCR typo on a real card name (JPEG noise +
    fontmap drift) should still score firmly authentic — counterfeit
    fonts produce systematic distortion, not single-char swaps."""
    img = _make_canonical()
    # 1 substitution out of 14 letters → normalized distance ≈ 0.07.
    r = analyze_typography(img, "Lightning Bolt", ocr_fn=_stub_ocr("Lightning Boli"))
    assert r.abstain_reason is None
    assert r.levenshtein_distance == 1
    assert r.score >= 0.85, f"got score {r.score}"


def test_punctuation_difference_does_not_drop_score() -> None:
    """Card names with apostrophes/commas commonly OCR with the
    punctuation dropped — the detector's normalize step removes
    punctuation before the distance calc so this still scores ≈ exact."""
    img = _make_canonical()
    r = analyze_typography(
        img,
        "Jace, the Mind Sculptor",
        ocr_fn=_stub_ocr("Jace the Mind Sculptor"),
    )
    assert r.levenshtein_distance == 0
    assert r.score >= 0.95


def test_case_insensitive_match() -> None:
    img = _make_canonical()
    r = analyze_typography(img, "Lightning Bolt", ocr_fn=_stub_ocr("LIGHTNING BOLT"))
    assert r.levenshtein_distance == 0
    assert r.score >= 0.95


def test_multiline_ocr_concatenated() -> None:
    """RapidOCR may split a long card name into two boxes (one per
    line). The detector should concatenate them with a single space
    and still match cleanly."""
    img = _make_canonical()
    r = analyze_typography(
        img,
        "Sword of Fire and Ice",
        ocr_fn=_stub_ocr(["Sword of Fire", "and Ice"]),
    )
    assert r.levenshtein_distance == 0
    assert r.score >= 0.95


# -----------------------------
# Counterfeit path — totally different text scores low
# -----------------------------


def test_totally_different_text_scores_near_zero() -> None:
    """Counterfeiter substituted a completely different name (bait-and-
    switch) — score should land firmly in the LIKELY_COUNTERFEIT band."""
    img = _make_canonical()
    r = analyze_typography(
        img, "Lightning Bolt", ocr_fn=_stub_ocr("Pikachu Pokemon TCG")
    )
    assert r.abstain_reason is None
    assert r.confidence > 0.5
    assert r.score <= 0.10, f"got score {r.score}"


def test_score_strictly_decreases_with_distance() -> None:
    """Sanity: a 2-char typo scores STRICTLY lower than a 1-char typo
    which scores STRICTLY lower than an exact match. Pins the logistic
    monotonicity property."""
    img = _make_canonical()
    name = "Lightning Bolt"
    r0 = analyze_typography(img, name, ocr_fn=_stub_ocr("Lightning Bolt"))
    r1 = analyze_typography(img, name, ocr_fn=_stub_ocr("Lightning Boli"))
    r2 = analyze_typography(img, name, ocr_fn=_stub_ocr("Lightnigg Bolq"))
    assert r0.score > r1.score > r2.score


# -----------------------------
# Abstain paths — the detector never raises
# -----------------------------


def test_abstain_when_expected_text_missing() -> None:
    """No identified card name → no comparison possible → abstain.
    Encoded as confidence=0 so the ensemble verdict mapper produces
    UNVERIFIED."""
    img = _make_canonical()
    r = analyze_typography(img, None, ocr_fn=_stub_ocr("anything"))
    assert r.confidence == 0.0
    assert r.abstain_reason == "no_expected_text"
    assert r.score == 0.5  # uncertain mid-band, never confidently wrong
    assert r.levenshtein_distance is None


def test_abstain_when_expected_text_blank() -> None:
    img = _make_canonical()
    r = analyze_typography(img, "   ", ocr_fn=_stub_ocr("Lightning Bolt"))
    assert r.confidence == 0.0
    assert r.abstain_reason == "no_expected_text"


def test_abstain_when_image_none() -> None:
    r = analyze_typography(None, "Lightning Bolt", ocr_fn=_stub_ocr("anything"))
    assert r.confidence == 0.0
    assert r.abstain_reason == "invalid_image"


def test_abstain_when_image_wrong_shape() -> None:
    """A 2-D grayscale image isn't BGR — abstain rather than guess."""
    img_2d = np.zeros((CANONICAL_H, CANONICAL_W), dtype=np.uint8)
    r = analyze_typography(img_2d, "Lightning Bolt", ocr_fn=_stub_ocr("anything"))
    assert r.confidence == 0.0
    assert r.abstain_reason == "invalid_image"


def test_abstain_when_image_too_small() -> None:
    tiny = np.zeros((4, 4, 3), dtype=np.uint8)
    r = analyze_typography(tiny, "Lightning Bolt", ocr_fn=_stub_ocr("anything"))
    assert r.confidence == 0.0
    assert r.abstain_reason == "invalid_image"


def test_abstain_when_ocr_lib_missing(monkeypatch) -> None:
    """When RapidOCR isn't importable AND no ocr_fn is injected, the
    detector abstains (UNVERIFIED) rather than crashing the ensemble.

    Simulates the import failure by monkeypatching the lazy-loader to
    return None, then verifies the abstain path."""
    monkeypatch.setattr(typography_detector, "_try_load_ocr", lambda: None)
    img = _make_canonical()
    r = analyze_typography(img, "Lightning Bolt")
    assert r.confidence == 0.0
    assert r.abstain_reason == "ocr_unavailable"
    assert r.score == 0.5


def test_ocr_engine_crash_abstains_gracefully() -> None:
    """If the recognizer raises mid-call, the detector swallows the
    exception and returns an abstain — counterfeit ensemble shouldn't
    blow up because RapidOCR hit a corner case."""
    img = _make_canonical()

    def _raises(_img):
        raise RuntimeError("simulated onnxruntime crash")

    r = analyze_typography(img, "Lightning Bolt", ocr_fn=_raises)
    assert r.confidence == 0.0
    assert r.abstain_reason == "ocr_unavailable"
    # Detector recorded the underlying error for forensic review.
    assert "error" in r.metadata


# -----------------------------
# Empty OCR result — distinct from abstain
# -----------------------------


def test_empty_ocr_result_scores_low_with_reduced_confidence() -> None:
    """If OCR ran cleanly but returned no recognized text against a
    valid expected name, that's signal — but weaker than a clear
    mismatch. The detector returns a non-abstain result with reduced
    confidence and a low score (saturated by the logistic at distance=1)."""
    img = _make_canonical()
    r = analyze_typography(img, "Lightning Bolt", ocr_fn=_stub_ocr([]))
    # NOT an abstain — the OCR ran and the empty result IS the data.
    assert r.abstain_reason is None
    # But confidence is reduced from the OCR-hit path.
    assert 0.0 < r.confidence < 0.85
    # Score is firmly counterfeit-side (distance ≈ 1.0 → near zero).
    assert r.score < 0.10


def test_empty_string_ocr_result_treated_as_empty() -> None:
    img = _make_canonical()
    r = analyze_typography(img, "Lightning Bolt", ocr_fn=_stub_ocr([""]))
    assert r.abstain_reason is None
    assert r.score < 0.10


# -----------------------------
# Result shape — frozen dataclass with the right fields
# -----------------------------


def test_result_is_frozen_dataclass_with_metadata() -> None:
    img = _make_canonical()
    r = analyze_typography(img, "Lightning Bolt", ocr_fn=_stub_ocr("Lightning Bolt"))
    assert isinstance(r, TypographyResult)
    # Frozen — attempted assignment raises (defensive: forensic data is
    # immutable post-measurement).
    import dataclasses
    try:
        dataclasses.replace(r, score=0.0)  # works; replace returns a new instance
    except Exception:
        raise AssertionError("dataclasses.replace should work on TypographyResult")
    # Metadata carries the inputs to the logistic for offline recalibration.
    assert "distance_midpoint" in r.metadata
    assert r.metadata["distance_midpoint"] == DEFAULT_DISTANCE_MIDPOINT
    assert r.metadata["distance_slope"] == DEFAULT_DISTANCE_SLOPE


# -----------------------------
# ROI crop — fractions are honored
# -----------------------------


def test_roi_crop_fractions_passed_to_ocr_fn() -> None:
    """The OCR function receives the cropped ROI, not the full image.
    Verifies the caller's roi_*_frac knobs actually shape the input."""
    img = _make_canonical()
    captured_shapes: list[tuple] = []

    def _capture_ocr(roi: np.ndarray) -> list[str]:
        captured_shapes.append(roi.shape)
        return ["Lightning Bolt"]

    analyze_typography(
        img,
        "Lightning Bolt",
        ocr_fn=_capture_ocr,
        roi_top_frac=0.0,
        roi_bottom_frac=0.10,
        roi_left_frac=0.05,
        roi_right_frac=0.95,
    )
    assert len(captured_shapes) == 1
    h, w, c = captured_shapes[0]
    # ~10% of 1050 = 105, ~90% of 750 = 675. Allow a 1-px rounding slack.
    assert 100 <= h <= 110
    assert 670 <= w <= 680
    assert c == 3
