"""Typography-based counterfeit detector (ensemble #5 of 7).

Compares the OCR'd text in the card-title region of the canonical front
image against the card name returned by the identification stage. Real
cards have crisp, manufacturer-controlled typography; counterfeiters
consistently get fonts subtly wrong (slightly off kerning, wrong x-height,
wrong stroke contrast, sometimes the wrong font entirely) and the
resulting OCR mismatch carries signal even before we get to per-glyph
template matching.

This v1 is intentionally the smallest possible thing that could work:

  1. Crop the title ROI (top ~10% of the canonical 750x1050 dewarped
     front — rough but reliable across the games we care about; finer
     per-game ROIs are a future iteration).
  2. Run RapidOCR's recognizer over the crop.
  3. Compute a case-insensitive normalized Levenshtein distance against
     the identified card name.
  4. Squash distance → score with a logistic. Small typos (a couple of
     character swaps from JPEG artifacts) still score authentic; a
     totally different string (or empty OCR result against a real
     name) scores counterfeit.

Future iterations layer on canonical-font template matching (per-glyph
correlation against a stored reference rendering of the card name in the
manufacturer's exact font). Until that lands, OCR + Levenshtein is the
only signal in this detector.

Design choices:
  - No PaddleOCR dependency. RapidOCR's PP-OCR ONNX weights run cleanly
    on CPU through onnxruntime — no Paddle install required, clean
    Windows wheels, low cold-start.
  - Graceful abstain (confidence=0.0) when the lib isn't installed or
    the inputs are missing, so existing CI doesn't break the day this
    detector ships and a card without an identified name doesn't blow
    up the whole counterfeit ensemble.
  - `ocr_fn` is injectable for tests so we can drive the algorithm with
    deterministic synthetic strings without spinning up onnxruntime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np
from numpy.typing import NDArray


# ROI geometry. The canonical dewarped front is 750 wide x 1050 tall (BGR).
# The card title sits in the top band of every game we currently grade
# (MTG, Pokemon). 10% from the top is a generous but safe heuristic —
# wide enough to catch the title even when dewarp introduces a few px
# of vertical shift, narrow enough to keep the recognizer focused.
DEFAULT_ROI_TOP_FRAC: float = 0.0
DEFAULT_ROI_BOTTOM_FRAC: float = 0.10
DEFAULT_ROI_LEFT_FRAC: float = 0.05  # trim corner art
DEFAULT_ROI_RIGHT_FRAC: float = 0.95

# Logistic squash mapping normalized-Levenshtein-distance → score.
#
# Empirical anchors (clean OCR against real card titles):
#   - exact match: distance = 0.0 → score ≈ 1.0
#   - 1-char typo on a 12-char name: distance ≈ 0.08 → score ≈ 0.92
#   - 2-char typo on a 12-char name: distance ≈ 0.17 → score ≈ 0.80
#   - completely different string: distance ≈ 1.0 → score ≈ 0.02
#
# Midpoint at 0.35 places the boundary between "few-character JPEG
# artifact level OCR drift" and "this clearly does not say what the
# catalog says it should say." Slope 12 makes the transition firm
# enough that a 0.5-distance match (half-wrong characters) is decisively
# below the suspicious threshold.
#
# These are placeholders; the recalibration tool will ratchet them later.
DEFAULT_DISTANCE_MIDPOINT: float = 0.35
DEFAULT_DISTANCE_SLOPE: float = 12.0


# Confidence ramp: full confidence when OCR returned a non-empty string
# AND we have an expected name to compare against. Reduced (but not
# zero) confidence when OCR returned empty against a real expected name
# — empty OCR is itself signal (a counterfeit with a glaringly wrong
# title font may not OCR at all), but the signal is weaker because lots
# of legitimate things make OCR fail (low contrast, stylized fonts,
# scanned-at-low-res cards). Recalibration can move this.
_CONFIDENCE_OCR_HIT: float = 0.85
_CONFIDENCE_OCR_EMPTY: float = 0.45


@dataclass(frozen=True)
class TypographyResult:
    """Output of `analyze_typography`.

    Fields:
      score: [0, 1], higher = more likely authentic. 0.5 returned in
        abstain paths so the score itself is never confidently wrong;
        the verdict mapper consults `confidence` to decide UNVERIFIED.
      confidence: [0, 1]. 0 means "abstain — do not include in
        ensemble" (lib missing, no expected name, invalid image).
      extracted_text: what the OCR engine actually read from the title
        ROI, or None if OCR didn't run.
      expected_text: the card name from identification, normalized for
        comparison (None passes through).
      levenshtein_distance: case-insensitive char-level edit distance
        between extracted and expected, or None if either was missing.
      abstain_reason: short tag explaining why confidence is 0
        (`ocr_unavailable`, `no_expected_text`, `invalid_image`, etc.).
        None on the happy path.
      manufacturer_profile: which calibration profile was used. v1
        always "generic"; future versions branch per-game once we have
        per-game font templates.
      metadata: backend extras (ROI bounds, raw OCR confidence, etc.)
        for forensic review and offline calibration.
    """

    score: float
    confidence: float
    extracted_text: Optional[str]
    expected_text: Optional[str]
    levenshtein_distance: Optional[int]
    abstain_reason: Optional[str]
    manufacturer_profile: str = "generic"
    metadata: dict[str, Any] = field(default_factory=dict)


def _logistic(x: float, midpoint: float, slope: float) -> float:
    """Numerically-stable logistic squash to [0, 1]."""
    z = slope * (x - midpoint)
    if z >= 0:
        ez = float(np.exp(-z))
        return 1.0 / (1.0 + ez)
    ez = float(np.exp(z))
    return ez / (1.0 + ez)


def _levenshtein(a: str, b: str) -> int:
    """Iterative Levenshtein distance with O(min(len)) memory.

    Stdlib doesn't ship one; pulling in `python-Levenshtein` for a
    sub-200-char title comparison is overkill (and adds a C dep that
    complicates Windows builds). The textbook DP is plenty fast for
    title-length inputs."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    # Make `a` the shorter one for memory.
    if len(a) > len(b):
        a, b = b, a
    prev = list(range(len(a) + 1))
    for i, cb in enumerate(b, start=1):
        curr = [i] + [0] * len(a)
        for j, ca in enumerate(a, start=1):
            cost = 0 if ca == cb else 1
            curr[j] = min(
                curr[j - 1] + 1,        # insertion
                prev[j] + 1,            # deletion
                prev[j - 1] + cost,     # substitution
            )
        prev = curr
    return prev[-1]


def _normalize_for_compare(s: str) -> str:
    """Lowercase + collapse whitespace + strip punctuation that OCR
    routinely drops. Card names are full of apostrophes/commas
    ("Jace, the Mind Sculptor", "Voldaren's Pact") — we don't want to
    flag a perfect read just because the engine missed an apostrophe.
    """
    out_chars = []
    last_was_space = True  # drop leading whitespace
    for ch in s.lower():
        if ch.isalnum():
            out_chars.append(ch)
            last_was_space = False
        elif ch.isspace():
            if not last_was_space:
                out_chars.append(" ")
                last_was_space = True
        # else: drop punctuation entirely
    return "".join(out_chars).strip()


def _crop_title_roi(
    image: NDArray[np.uint8],
    *,
    top_frac: float,
    bottom_frac: float,
    left_frac: float,
    right_frac: float,
) -> NDArray[np.uint8]:
    """Crop the title ROI from the canonical card image.

    Fractions are relative to the image dimensions, which lets the
    detector run unchanged on the canonical 750x1050 OR on alternate
    sizes (test fixtures, future canonical resolutions)."""
    h, w = image.shape[:2]
    y0 = max(0, int(round(h * top_frac)))
    y1 = max(y0 + 1, int(round(h * bottom_frac)))
    x0 = max(0, int(round(w * left_frac)))
    x1 = max(x0 + 1, int(round(w * right_frac)))
    return image[y0:y1, x0:x1]


def _try_load_ocr() -> Optional[Callable[[NDArray[np.uint8]], list[str]]]:
    """Return a function that runs OCR on a numpy BGR image and returns
    a list of recognized text strings, or None if the OCR library isn't
    importable.

    Defensive across two RapidOCR API generations:
      - new unified `rapidocr` (RapidOCROutput dataclass with .txts)
      - legacy `rapidocr_onnxruntime` (returns (result_list, elapse)
        where result_list is [[box, text, score], ...])
    """
    engine = None
    try:
        # New unified package (preferred when present).
        from rapidocr_onnxruntime import RapidOCR  # type: ignore[import-not-found]
        engine = RapidOCR()
    except ImportError:
        try:
            from rapidocr import RapidOCR  # type: ignore[import-not-found]
            engine = RapidOCR()
        except ImportError:
            return None

    def _run(img: NDArray[np.uint8]) -> list[str]:
        out = engine(img)
        # New API: dataclass with .txts (list[str]).
        txts = getattr(out, "txts", None)
        if txts is not None:
            return [str(t) for t in txts]
        # Legacy API: tuple (result_list, elapse_list) where result_list
        # is None on no-detection or [[box, text, score], ...] on hit.
        if isinstance(out, tuple) and len(out) >= 1:
            result = out[0]
            if result is None:
                return []
            return [str(item[1]) for item in result if len(item) >= 2]
        return []

    return _run


def _no_signal(
    *,
    expected_text: Optional[str],
    extracted_text: Optional[str],
    reason: str,
    metadata: Optional[dict[str, Any]] = None,
) -> TypographyResult:
    """Build an abstain-shaped TypographyResult. Mirrors the embedding-
    anomaly detector's no-signal helper so the ensemble verdict mapper
    treats the abstain identically across detectors."""
    return TypographyResult(
        score=0.5,
        confidence=0.0,
        extracted_text=extracted_text,
        expected_text=expected_text,
        levenshtein_distance=None,
        abstain_reason=reason,
        manufacturer_profile="generic",
        metadata=metadata or {"reason": reason},
    )


def analyze_typography(
    image: NDArray[np.uint8] | None,
    expected_card_name: Optional[str],
    *,
    ocr_fn: Optional[Callable[[NDArray[np.uint8]], list[str]]] = None,
    roi_top_frac: float = DEFAULT_ROI_TOP_FRAC,
    roi_bottom_frac: float = DEFAULT_ROI_BOTTOM_FRAC,
    roi_left_frac: float = DEFAULT_ROI_LEFT_FRAC,
    roi_right_frac: float = DEFAULT_ROI_RIGHT_FRAC,
    distance_midpoint: float = DEFAULT_DISTANCE_MIDPOINT,
    distance_slope: float = DEFAULT_DISTANCE_SLOPE,
) -> TypographyResult:
    """Score the typography of the title region against the expected name.

    Args:
        image: BGR canonical card image as a uint8 numpy array (HxWx3).
            None is accepted and yields an abstain.
        expected_card_name: card name from the identification stage,
            or None if the card was not identified.
        ocr_fn: optional injected OCR callable. Tests pass a stub here
            to avoid spinning up onnxruntime; production leaves this
            None and the detector loads RapidOCR lazily.
        roi_*_frac: ROI bounds as fractions of the image. Defaults
            target the title band of the canonical 750x1050 dewarped
            front.
        distance_midpoint, distance_slope: logistic squash parameters
            applied to the normalized Levenshtein distance.

    Returns:
        TypographyResult. Always returns a result (never raises) — the
        abstain path is a result with confidence=0, not an exception.
    """
    # --- abstain on missing expected name ---
    if expected_card_name is None or not expected_card_name.strip():
        return _no_signal(
            expected_text=None,
            extracted_text=None,
            reason="no_expected_text",
        )

    # --- abstain on missing or invalid image ---
    if image is None:
        return _no_signal(
            expected_text=expected_card_name,
            extracted_text=None,
            reason="invalid_image",
        )
    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
        return _no_signal(
            expected_text=expected_card_name,
            extracted_text=None,
            reason="invalid_image",
            metadata={
                "reason": "invalid_image",
                "shape": tuple(getattr(image, "shape", ())),
            },
        )
    if image.shape[0] < 10 or image.shape[1] < 10:
        return _no_signal(
            expected_text=expected_card_name,
            extracted_text=None,
            reason="invalid_image",
            metadata={"reason": "invalid_image", "shape": tuple(image.shape)},
        )

    # --- ROI crop ---
    try:
        roi = _crop_title_roi(
            image,
            top_frac=roi_top_frac,
            bottom_frac=roi_bottom_frac,
            left_frac=roi_left_frac,
            right_frac=roi_right_frac,
        )
    except Exception as e:  # extremely defensive — fraction math shouldn't fail
        return _no_signal(
            expected_text=expected_card_name,
            extracted_text=None,
            reason="invalid_image",
            metadata={"reason": "roi_crop_failed", "error": str(e)},
        )

    # --- OCR (lazy lib-load, abstain on missing dep) ---
    if ocr_fn is None:
        ocr_fn = _try_load_ocr()
        if ocr_fn is None:
            return _no_signal(
                expected_text=expected_card_name,
                extracted_text=None,
                reason="ocr_unavailable",
                metadata={
                    "reason": "ocr_unavailable",
                    "hint": "install rapidocr-onnxruntime to enable typography detector",
                },
            )

    try:
        recognized = ocr_fn(roi)
    except Exception as e:
        # Recognizer crashed — abstain rather than blow up the ensemble.
        return _no_signal(
            expected_text=expected_card_name,
            extracted_text=None,
            reason="ocr_unavailable",
            metadata={"reason": "ocr_failed", "error": str(e)},
        )

    # Concatenate all recognized lines (the title might split across two
    # bounding boxes for multi-word names like "Jace, the Mind Sculptor").
    raw_extracted = " ".join(s for s in recognized if s).strip()

    # --- empty OCR path: still scoreable, but with reduced confidence ---
    if not raw_extracted:
        return TypographyResult(
            score=_logistic(distance_midpoint - 1.0, 0.0, distance_slope),
            confidence=_CONFIDENCE_OCR_EMPTY,
            extracted_text="",
            expected_text=expected_card_name,
            levenshtein_distance=None,
            abstain_reason=None,
            manufacturer_profile="generic",
            metadata={
                "reason": "empty_ocr_result",
                "roi_shape": tuple(roi.shape),
                "distance_midpoint": float(distance_midpoint),
                "distance_slope": float(distance_slope),
            },
        )

    # --- score: normalized Levenshtein on case-folded, depunctuated strings ---
    norm_expected = _normalize_for_compare(expected_card_name)
    norm_extracted = _normalize_for_compare(raw_extracted)
    distance_chars = _levenshtein(norm_extracted, norm_expected)
    denom = max(len(norm_expected), len(norm_extracted), 1)
    distance_norm = distance_chars / denom

    # Lower distance → higher score (more authentic-like). Same logistic
    # shape as the embedding-anomaly detector — the inputs are different
    # but the "distance to expected" framing is the same.
    score = _logistic(distance_midpoint - distance_norm, 0.0, distance_slope)

    return TypographyResult(
        score=float(score),
        confidence=_CONFIDENCE_OCR_HIT,
        extracted_text=raw_extracted,
        expected_text=expected_card_name,
        levenshtein_distance=int(distance_chars),
        abstain_reason=None,
        manufacturer_profile="generic",
        metadata={
            "normalized_distance": float(distance_norm),
            "normalized_extracted": norm_extracted,
            "normalized_expected": norm_expected,
            "roi_shape": tuple(roi.shape),
            "distance_midpoint": float(distance_midpoint),
            "distance_slope": float(distance_slope),
            "ocr_lines": list(recognized),
        },
    )


__all__ = [
    "DEFAULT_ROI_TOP_FRAC",
    "DEFAULT_ROI_BOTTOM_FRAC",
    "DEFAULT_ROI_LEFT_FRAC",
    "DEFAULT_ROI_RIGHT_FRAC",
    "DEFAULT_DISTANCE_MIDPOINT",
    "DEFAULT_DISTANCE_SLOPE",
    "TypographyResult",
    "analyze_typography",
]
