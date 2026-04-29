"""Color-profile counterfeit detector (ensemble #4 of 7).

Detects the gamut clipping that consumer-printer counterfeits exhibit
relative to authentic offset-printed cards. Authentic cards are printed
with process CMYK + spot inks on offset presses that reliably produce
strong chroma (C*ab routinely above 50 in saturated art areas). Bad
inkjet copies — the dominant counterfeit shape — clip at lower chroma
because consumer printer inks have a smaller gamut than process ink
sets and because halftone-on-paper interaction loses some saturation in
the conversion. So a card whose maximum chroma falls well below the
authentic-baseline is suspicious in a way that's hard to fake without a
real offset rig.

Algorithm:
  1. Locate the white border. Default canonical cards have ~40 px of
     white border (see tests.fixtures.synth_card); we sample the four
     border strips and compute a robust mean color. That mean serves
     as the "white reference" for chromatic adaptation.
  2. Compute the chromatic-adaptation gain that would map the border
     mean to neutral (R=G=B=255). Apply that gain to the inner area —
     this neutralizes ambient color cast (warm tungsten light, cool
     daylight, bluish phone screen reflections, etc.) so the chroma
     signal we measure comes from the inks, not the lighting.
  3. Convert the calibrated inner area BGR → CIELAB.
  4. Compute the 95th-percentile chroma C*ab = sqrt(a*² + b*²). 95th
     percentile rather than max because a single saturated noise
     pixel could otherwise dominate; we want the "characteristic
     peak chroma the inks reliably produce".
  5. Map p95 chroma → score via a logistic centered at a calibrated
     authentic-baseline.
  6. Confidence drops if the white border has unusually high std-dev
     (some other color is masquerading as a border, or the canonical
     was severely under/overexposed) — in that case our white-balance
     calibration is unreliable and the resulting chroma signal can't
     be trusted.

Why not a per-color histogram comparison? That would require a
per-manufacturer authentic distribution to compare against, and we
don't have one in the corpus yet. Max-chroma works as a useful
counterfeit signal in the meantime because gamut limits are physical
properties of the printer, not of any specific card design — even
without a reference distribution, it discriminates inkjet from
offset reliably on synthetic fixtures and (we'll see) real data once
the corpus crosses ~1k authentic samples.

Detects: low-gamut consumer-printer counterfeits.
Does NOT detect: counterfeits that print on real offset rigs, or
consumer-printer counterfeits whose authentic side already has muted
colors (e.g. some vintage cards). That's why this is one detector in
a 7-detector ensemble, not standalone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

# Inset (px) used when sampling the white border for white-balance
# calibration. Default canonical cards have ~40 px of white border;
# we sample inside the outermost few pixels (which can have edge
# softness or sub-pixel mis-dewarp) and outside the inner art area.
DEFAULT_BORDER_SAMPLE_INSET_PX: int = 8
DEFAULT_BORDER_SAMPLE_DEPTH_PX: int = 24

# Inset (px) used when sampling the inner art area for chroma. Steps
# in past the white border so we don't measure the white as part of
# our chroma signal (white has near-zero chroma by construction).
DEFAULT_INNER_INSET_PX: int = 56

# Logistic squash midpoint and slope for p95 chroma.
#
# CIELAB chroma C*ab = sqrt(a*² + b*²) ranges from 0 (neutral) to
# ~128 (theoretical max). Empirical anchors:
#   - Authentic offset-printed cards: p95 C*ab ≈ 50-80 in
#     saturated art areas. Synthetic high-saturation fixtures hit
#     C*ab > 60.
#   - Inkjet printer output of the same source image: p95 C*ab
#     ≈ 25-40. CMYK consumer printers can't physically reach the
#     saturation of process printing for many hues.
#   - Heavily faded / vintage cards: p95 C*ab can dip below 30 even
#     on authentic samples — these will score lower until we have a
#     per-manufacturer baseline distribution to compare against.
#
# Midpoint at 40 places the decision boundary in the middle of the
# inkjet/offset gap. Slope of 0.12 gives a smooth but decisive
# transition (logistic crosses ~0.5 at C*ab=40, ~0.95 at C*ab=64).
#
# These v1 thresholds are calibrated against synthetic fixtures.
# Will be recalibrated against real PSA pop-report images once the
# data flywheel pushes enough samples through.
CHROMA_MIDPOINT: float = 40.0
CHROMA_SLOPE: float = 0.12

# Std-dev of the border BGR (across border pixels, summed across
# channels) above which our white-balance calibration is considered
# unreliable. A clean printed border has std ≤ ~6 across channels
# under typical capture conditions; a noisy / partially-occluded /
# wrong-color border pushes std much higher.
MAX_RELIABLE_BORDER_STDDEV: float = 18.0

# Lower bound on the border mean luminance, in the 0-255 BGR sense.
# A "white border" that's actually dark gray suggests we're not
# looking at the white border at all — likely a severely
# under-exposed image or a wrong inset. Bail rather than calibrate
# against a non-white reference.
MIN_BORDER_MEAN_LUMINANCE: float = 140.0

# Cap on the chromatic-adaptation gain. Without this, a heavily
# tinted "white" border could push the gain to 5x+ on one channel,
# causing all inner pixels to clip at 255 in that channel and
# corrupting the chroma measurement. 1.6x covers normal ambient
# casts (~10% color shift); anything beyond that is a sign the
# border isn't really white and we should abstain.
MAX_GAIN_PER_CHANNEL: float = 1.6


@dataclass(frozen=True)
class ColorProfileMeasurement:
    """Output of `measure_color_profile`.

    Fields:
      color_score: [0, 1], higher = more likely authentic. 0.5 means
        "uncertain" (e.g. when the border couldn't be calibrated).
      p95_chroma: 95th-percentile CIELAB C*ab in the calibrated inner
        area. Provided for debug / forensic review.
      border_white_bgr: the (B, G, R) mean color we sampled as the
        white reference. Useful for offline calibration and for
        showing "this is what we treated as white" in the cert page.
      border_stddev: total channel-summed std-dev of the border
        sample. High values lower confidence.
      gain_applied: per-channel chromatic-adaptation gain that was
        applied to the inner area before chroma measurement. (1, 1, 1)
        means no calibration was needed; otherwise the values reflect
        the lighting cast we removed.
      confidence: [0, 1]. Drops as the border calibration becomes
        unreliable or as gains hit the safety cap.
      manufacturer_profile: which calibration profile was used.
        v1 always "generic" — branches per-manufacturer once we
        have authentic baselines for Topps / Pokemon / Konami / etc.
      metadata: backend extras (sample sizes, raw chroma stats) for
        UI overlay and offline calibration analyses.
    """

    color_score: float
    p95_chroma: float
    border_white_bgr: tuple[float, float, float]
    border_stddev: float
    gain_applied: tuple[float, float, float]
    confidence: float
    manufacturer_profile: str
    metadata: dict[str, Any] = field(default_factory=dict)


def _logistic(x: float, midpoint: float, slope: float) -> float:
    """Numerically-stable logistic squash to [0, 1]."""
    z = slope * (x - midpoint)
    if z >= 0:
        ez = float(np.exp(-z))
        return 1.0 / (1.0 + ez)
    ez = float(np.exp(z))
    return ez / (1.0 + ez)


def _sample_border(
    image: NDArray[np.uint8],
    inset_px: int,
    depth_px: int,
) -> NDArray[np.uint8]:
    """Concatenate four border strips (top, bottom, left, right) into a
    single (N, 3) BGR array.

    The strips overlap in the corners — that's intentional. Corners
    pick up a 2x weighting for the white-reference calculation, which
    is fine because corners are the most reliable parts of the white
    border (least edge-softness from dewarp interpolation)."""
    h, w = image.shape[:2]
    if h <= 2 * (inset_px + depth_px) or w <= 2 * (inset_px + depth_px):
        return np.empty((0, 3), dtype=np.uint8)

    # Top + bottom: full width, depth_px rows.
    top = image[inset_px : inset_px + depth_px, :, :]
    bot = image[h - inset_px - depth_px : h - inset_px, :, :]
    # Left + right: full height, depth_px cols.
    left = image[:, inset_px : inset_px + depth_px, :]
    right = image[:, w - inset_px - depth_px : w - inset_px, :]

    return np.concatenate(
        [
            top.reshape(-1, 3),
            bot.reshape(-1, 3),
            left.reshape(-1, 3),
            right.reshape(-1, 3),
        ],
        axis=0,
    )


def _gain_to_neutral(border_mean_bgr: NDArray[np.float32]) -> NDArray[np.float32]:
    """Per-channel multiplicative gain that maps the border mean to
    (255, 255, 255).

    Capped at MAX_GAIN_PER_CHANNEL — see module-level rationale."""
    eps = 1e-3
    raw = 255.0 / np.clip(border_mean_bgr, eps, None)
    return np.clip(raw, 0.5, MAX_GAIN_PER_CHANNEL).astype(np.float32)


def _apply_gain(
    image: NDArray[np.uint8],
    gain_bgr: NDArray[np.float32],
) -> NDArray[np.uint8]:
    """Multiply each channel by its gain, clipping to [0, 255]."""
    out = image.astype(np.float32) * gain_bgr.reshape(1, 1, 3)
    return np.clip(out, 0.0, 255.0).astype(np.uint8)


def _p95_chroma(image_bgr: NDArray[np.uint8]) -> float:
    """95th-percentile CIELAB chroma over the full input.

    Returns the C*ab value in CIELAB units; the conversion via
    cv2.COLOR_BGR2LAB uses uint8 packing where a* and b* are stored
    in [0, 255] with neutral at 128 (per OpenCV's convention)."""
    if image_bgr.size == 0:
        return 0.0
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    # OpenCV LAB packing: L in [0, 255] (= L*/100 * 255), a in [0, 255]
    # with neutral 128, b in [0, 255] with neutral 128. So a*-128 and
    # b*-128 give the signed CIELAB-style components, scaled the same.
    a = lab[:, :, 1].astype(np.int32) - 128
    b = lab[:, :, 2].astype(np.int32) - 128
    chroma = np.sqrt(a * a + b * b).astype(np.float32)
    return float(np.percentile(chroma, 95))


def measure_color_profile(
    canonical: NDArray[np.uint8],
    border_sample_inset_px: int = DEFAULT_BORDER_SAMPLE_INSET_PX,
    border_sample_depth_px: int = DEFAULT_BORDER_SAMPLE_DEPTH_PX,
    inner_inset_px: int = DEFAULT_INNER_INSET_PX,
) -> ColorProfileMeasurement:
    """Measure color-profile counterfeit signal on a canonical 750x1050 BGR card.

    Args:
        canonical: dewarped BGR uint8 card image. Production input is the
            canonical produced by Stage 2 (perspective correction); for
            unit tests the synthetic fixtures in tests.fixtures are used.
        border_sample_inset_px: how far in from each edge to start sampling
            the white border (skips edge-softness from dewarp interpolation).
        border_sample_depth_px: how many pixels deep into the border to
            sample. Stays inside a typical 40 px white border.
        inner_inset_px: how far in from each edge to start the inner-area
            chroma measurement (steps past the white border so we don't
            measure white as part of the chroma signal).

    Returns:
        ColorProfileMeasurement with color_score in [0, 1] (higher = more
        likely authentic) and confidence in [0, 1].

    Raises:
        ValueError: if the input fails the dtype/shape/size validation gate.
    """
    if canonical.dtype != np.uint8:
        raise ValueError(f"expected uint8 image, got {canonical.dtype}")
    if canonical.ndim != 3 or canonical.shape[2] != 3:
        raise ValueError(
            f"expected 3-channel BGR image, got shape {canonical.shape}"
        )
    if border_sample_inset_px < 0 or border_sample_depth_px <= 0:
        raise ValueError(
            f"border_sample_inset_px must be >=0 and depth must be >0, "
            f"got inset={border_sample_inset_px} depth={border_sample_depth_px}"
        )
    if inner_inset_px <= 0:
        raise ValueError(f"inner_inset_px must be positive, got {inner_inset_px}")

    h, w = canonical.shape[:2]
    if h < 2 * inner_inset_px + 8 or w < 2 * inner_inset_px + 8:
        raise ValueError(
            f"image too small for inner_inset_px={inner_inset_px}: got {(h, w)}"
        )

    # 1. Sample the border for the white reference.
    border = _sample_border(
        canonical,
        inset_px=border_sample_inset_px,
        depth_px=border_sample_depth_px,
    )
    if border.size == 0:
        return ColorProfileMeasurement(
            color_score=0.5,
            p95_chroma=0.0,
            border_white_bgr=(0.0, 0.0, 0.0),
            border_stddev=0.0,
            gain_applied=(1.0, 1.0, 1.0),
            confidence=0.0,
            manufacturer_profile="generic",
            metadata={"reason": "border_sample_empty"},
        )

    border_f = border.astype(np.float32)
    border_mean = border_f.mean(axis=0)
    border_std_per_channel = border_f.std(axis=0)
    border_stddev_total = float(border_std_per_channel.sum())
    border_luminance = float(border_mean.mean())

    # If the "white" reference is too dark (overexposure protection) or
    # too noisy (border-isn't-really-white protection), we can't trust
    # the calibration. Return uncertain rather than fabricate a verdict.
    if border_luminance < MIN_BORDER_MEAN_LUMINANCE or border_stddev_total > MAX_RELIABLE_BORDER_STDDEV * 3:
        return ColorProfileMeasurement(
            color_score=0.5,
            p95_chroma=0.0,
            border_white_bgr=tuple(float(v) for v in border_mean),  # type: ignore[arg-type]
            border_stddev=border_stddev_total,
            gain_applied=(1.0, 1.0, 1.0),
            confidence=0.0,
            manufacturer_profile="generic",
            metadata={
                "reason": "border_unreliable",
                "border_luminance": border_luminance,
                "border_stddev_total": border_stddev_total,
            },
        )

    # 2. Compute and apply chromatic-adaptation gain.
    gain = _gain_to_neutral(border_mean)
    inner = canonical[
        inner_inset_px : h - inner_inset_px,
        inner_inset_px : w - inner_inset_px,
        :,
    ]
    if inner.size == 0:
        return ColorProfileMeasurement(
            color_score=0.5,
            p95_chroma=0.0,
            border_white_bgr=tuple(float(v) for v in border_mean),  # type: ignore[arg-type]
            border_stddev=border_stddev_total,
            gain_applied=tuple(float(v) for v in gain),  # type: ignore[arg-type]
            confidence=0.0,
            manufacturer_profile="generic",
            metadata={"reason": "inner_area_empty"},
        )
    inner_calibrated = _apply_gain(inner, gain)

    # 3-4. CIELAB conversion + 95th-percentile chroma.
    p95 = _p95_chroma(inner_calibrated)

    # 5. Map to score via logistic.
    score = _logistic(p95, CHROMA_MIDPOINT, CHROMA_SLOPE)

    # 6. Confidence = 1 - (border_stddev / max_reliable), bounded to [0, 1].
    #    Hits 1.0 on a clean white border; falls toward 0 as the border
    #    becomes noisier. Also penalizes runs that hit the gain cap (a
    #    sign the border has a real color cast we couldn't fully correct).
    border_quality = max(0.0, 1.0 - border_stddev_total / MAX_RELIABLE_BORDER_STDDEV)
    gain_clipped = float(np.any(np.isclose(gain, MAX_GAIN_PER_CHANNEL)))
    gain_quality = 1.0 - 0.4 * gain_clipped
    confidence = float(np.clip(border_quality * gain_quality, 0.0, 1.0))

    return ColorProfileMeasurement(
        color_score=float(score),
        p95_chroma=float(p95),
        border_white_bgr=tuple(float(v) for v in border_mean),  # type: ignore[arg-type]
        border_stddev=border_stddev_total,
        gain_applied=tuple(float(v) for v in gain),  # type: ignore[arg-type]
        confidence=confidence,
        manufacturer_profile="generic",
        metadata={
            "border_pixels_sampled": int(border.shape[0]),
            "border_luminance": border_luminance,
            "inner_pixels_analyzed": int(inner.shape[0] * inner.shape[1]),
        },
    )


def is_likely_authentic(
    measurement: ColorProfileMeasurement,
    threshold: float = 0.5,
) -> bool:
    """Convenience predicate matching `measurement.color_score >= threshold`.

    Mirrors the rosette detector's predicate so call sites can be uniform
    across the counterfeit ensemble. May add confidence-gated abstain or
    per-manufacturer thresholds in future without changing call sites."""
    return measurement.color_score >= threshold
