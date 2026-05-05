"""Holographic-parallax counterfeit detector (ensemble #3 of 7).

The signal: real holographic foil prints exhibit angle-dependent
specular shifts — the diffraction grating physically redirects light
in different directions as the viewing angle changes. A counterfeit
"flat foil" — a matte sticker on top of flat ink, or glossy laminate
without a real grating — does not. So a foil region in two photos of
the same card taken at different angles will show a much larger
spatial shift on the authentic card than on the fake.

Algorithm (v1, intentionally first-pass):

  1. Take the canonical-dewarped front-of-card image AND a tilt_30
     image (also dewarped to the canonical 750x1050). Both BGR uint8.
  2. Build a coarse "holo region" mask from the front shot — pixels
     whose CIELAB chroma AND HSV saturation both clear thresholds
     (foil reads as bright + saturated). Erode/open to drop salt
     noise. If the mask covers <2% of pixels, the front shot doesn't
     have an obvious foil region — abstain UNVERIFIED.
  3. Compute dense optical flow between front and tilt with
     Farnebäck (`cv2.calcOpticalFlowFarneback`).
  4. Score = ratio of mean flow magnitude inside the mask vs.
     outside. Real holos shift SIGNIFICANTLY MORE than the rest of
     the card geometry; flat fakes shift roughly the same as
     non-foil regions. Logistic-squash the ratio to [0, 1].
  5. Confidence: high when both shots are present and the holo mask
     is large enough (≥5%); reduced when the mask is in the small
     band (2-5%); abstain when tilt is missing.

Limits of the v1 heuristic — DOCUMENT FOR FUTURE TUNING:
  - The chroma+saturation mask is a *crude* proxy for "is this
    pixel foil?". It will produce **false negatives** on cards
    whose holo region is desaturated foil (silver/gold mirror
    grains read as low-chroma in CIELAB). It will also produce
    **false positives** on the vibrant non-foil art of certain
    rare regular cards (full-art Pokémon, Secret Lair full-frame
    arts). The recalibration tool will ratchet the mask floor
    later; learned holo segmentation is on the roadmap.
  - We assume the canonical dewarp aligns the cards well between
    front and tilt. That's not strictly true — tilt_30 is by
    construction acquired at a different angle, so the dewarp
    output puts the card edges at the same canonical 750x1050
    rectangle, but interior content has subpixel shifts even on
    matte regions. We MASK-AWAY the global background flow by
    looking at the **ratio** of flow inside vs outside the holo
    mask, which cancels uniform shifts; the absolute flow
    magnitudes are not load-bearing.
  - Farnebäck is dense and slow but deterministic; we accept the
    cost given this detector runs once per submission.

Future iterations:
  - Multi-angle tilt (capture three angles, not one) gives a
    stronger separation signal — counterfeits are uniform across
    angles, so the variance in differential flow grows linearly
    with angle count.
  - Learned holo segmentation (a tiny U-Net trained on annotated
    foil/non-foil masks) replaces the chroma-saturation heuristic
    for the cards on the FN/FP lists above.
  - Per-manufacturer profiles — Pokémon vs MTG foil patterns
    differ in spatial frequency; per-game thresholds should be
    tighter than the current generic ones.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import cv2
import numpy as np
from numpy.typing import NDArray


# Canonical card dimensions for the dewarped inputs the detector
# expects. We don't enforce these exactly (any matched HxWxC pair is
# accepted) but the thresholds below assume images at roughly this
# resolution; very different sizes will need recalibration.
DEFAULT_CANONICAL_HEIGHT: int = 1050
DEFAULT_CANONICAL_WIDTH: int = 750


# Holo-region heuristic thresholds.
#
# Chroma in CIELAB: a*^2 + b*^2 (signed channels in cv2.COLOR_BGR2LAB
# returned as uint8 are in [0, 255] with 128 as neutral, so we shift to
# [-128, 127] before computing chroma). A neutral (gray) pixel sits at
# chroma ≈ 0; saturated print ink reaches 50+. Foil grains tend to
# spike well above 30 in any direction.
#
# Saturation (HSV): foil regions also stand out in H/S/V because the
# specular highlights kick the V channel high while keeping S non-zero
# (a true mirror would be desaturated; printed foils are not perfect
# mirrors — there's always some color cast).
#
# The combined gate (chroma > floor AND saturation > floor) keeps us
# from grabbing entire saturated card-art regions: art is high-chroma
# but the saturation-floor cuts most of it (printed inks have lower V
# than specular foil). It still false-positives some foil-like art —
# see the docstring's limit notes.
DEFAULT_CHROMA_THRESHOLD: float = 30.0
DEFAULT_SATURATION_THRESHOLD: int = 80  # 0-255, HSV S channel
# Mask cleanup — drop salt-noise pixels. 3x3 open/erode is plenty for
# the sub-percent dust we want to throw out.
DEFAULT_MORPH_KERNEL_SIZE: int = 3
# If the holo mask covers less than this fraction of pixels, the front
# image doesn't have an obvious foil region — abstain UNVERIFIED.
DEFAULT_MIN_MASK_FRACTION: float = 0.02
# Above this fraction, full confidence; between MIN and FULL, reduced.
DEFAULT_FULL_CONFIDENCE_MASK_FRACTION: float = 0.05


# Logistic squash mapping flow_ratio (mean(|flow|_holo) /
# mean(|flow|_non_holo)) → score.
#
# Empirical anchors (synthetic + first-pass calibration on internal
# fixtures — recalibration tool will ratchet later):
#   - flat fake (no parallax): ratio ≈ 1.0     → score ≈ 0.05
#   - weak holo (poor capture): ratio ≈ 1.5   → score ≈ 0.3
#   - decent holo: ratio ≈ 2.5                → score ≈ 0.85
#   - strong holo (well-lit shift): ratio ≥ 3 → score ≈ 0.95+
#
# Midpoint at 2.0 places the boundary squarely in the "this isn't
# clearly authentic and isn't clearly fake" band; slope 3.0 gives a
# firm-but-not-cliff-edged transition. These are placeholders;
# recalibration will move them.
DEFAULT_RATIO_MIDPOINT: float = 2.0
DEFAULT_RATIO_SLOPE: float = 3.0


# Confidence anchors. See the design-choices block in the module
# docstring for the rationale on the ramp.
_CONFIDENCE_FULL: float = 0.85
_CONFIDENCE_REDUCED: float = 0.55


# Farnebäck parameters. The defaults from the OpenCV docs are tuned
# for general-purpose video flow; we use slightly more aggressive
# pyramid + iterations because card holo flow is small (a few px) and
# sparse pyramid levels can miss it.
_FARNEBACK_PYR_SCALE: float = 0.5
_FARNEBACK_LEVELS: int = 3
_FARNEBACK_WINSIZE: int = 21
_FARNEBACK_ITERATIONS: int = 3
_FARNEBACK_POLY_N: int = 5
_FARNEBACK_POLY_SIGMA: float = 1.2
_FARNEBACK_FLAGS: int = 0


@dataclass(frozen=True)
class HolographicResult:
    """Output of `analyze_holographic_parallax`.

    Fields:
      score: [0, 1], higher = more likely authentic. 0.5 returned in
        abstain paths so the score itself is never confidently wrong;
        the verdict mapper consults `confidence` to decide UNVERIFIED.
      confidence: [0, 1]. 0 means "abstain — do not include in
        ensemble" (tilt missing, mask too small, flow failed).
      flow_ratio: mean(|flow|_holo) / mean(|flow|_non_holo), or None
        in abstain paths.
      holo_mask_fraction: fraction of pixels in the holo mask, or
        None in abstain paths that didn't get to mask construction.
      abstain_reason: short tag explaining why confidence is 0
        (`tilt_not_captured`, `shape_mismatch`, `no_holo_region`,
        `flow_computation_failed`, `invalid_image`). None on the
        happy path.
      manufacturer_profile: which calibration profile was used. v1
        always "generic"; future versions branch per-game.
      metadata: backend extras (mask fraction, raw flow stats, etc.)
        for forensic review and offline calibration.
    """

    score: float
    confidence: float
    flow_ratio: Optional[float]
    holo_mask_fraction: Optional[float]
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


def _no_signal(
    *,
    reason: str,
    flow_ratio: Optional[float] = None,
    holo_mask_fraction: Optional[float] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> HolographicResult:
    """Build an abstain-shaped HolographicResult. Mirrors the typography
    detector's no-signal helper so the ensemble verdict mapper treats
    the abstain identically across detectors."""
    return HolographicResult(
        score=0.5,
        confidence=0.0,
        flow_ratio=flow_ratio,
        holo_mask_fraction=holo_mask_fraction,
        abstain_reason=reason,
        manufacturer_profile="generic",
        metadata=metadata or {"reason": reason},
    )


def _is_valid_canonical(image: Optional[NDArray[np.uint8]]) -> bool:
    """Cheap shape + dtype sanity. Doesn't enforce 750x1050 exactly —
    any HxWx3 uint8 with both axes ≥ 32 is accepted (so synthetic
    test fixtures and alternate canonical sizes still work)."""
    if image is None:
        return False
    if not isinstance(image, np.ndarray):
        return False
    if image.ndim != 3 or image.shape[2] != 3:
        return False
    if image.shape[0] < 32 or image.shape[1] < 32:
        return False
    if image.dtype != np.uint8:
        return False
    return True


def _build_holo_mask(
    bgr: NDArray[np.uint8],
    *,
    chroma_threshold: float,
    saturation_threshold: int,
    morph_kernel_size: int,
) -> NDArray[np.uint8]:
    """Coarse foil-region mask. See the module docstring for why this
    is intentionally a first-pass heuristic and what its known
    failure modes are."""
    # CIELAB chroma. cv2 returns uint8 [0,255] for L*a*b* with 128 as
    # neutral on a/b. Shift to signed and compute Euclidean chroma.
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    a = lab[:, :, 1].astype(np.int16) - 128
    b = lab[:, :, 2].astype(np.int16) - 128
    chroma = np.sqrt((a.astype(np.float32) ** 2) + (b.astype(np.float32) ** 2))

    # HSV saturation.
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]

    raw = (chroma > chroma_threshold) & (s > saturation_threshold)
    mask = raw.astype(np.uint8) * 255

    # Open (erode then dilate) drops sub-kernel salt without shrinking
    # the foil region appreciably.
    if morph_kernel_size >= 2:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size)
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask


def _flow_magnitude(
    front_gray: NDArray[np.uint8],
    tilt_gray: NDArray[np.uint8],
) -> NDArray[np.float32]:
    """Dense Farnebäck flow magnitude. Returns an HxW float32 array of
    sqrt(dx^2+dy^2) per pixel."""
    flow = cv2.calcOpticalFlowFarneback(
        front_gray,
        tilt_gray,
        None,
        _FARNEBACK_PYR_SCALE,
        _FARNEBACK_LEVELS,
        _FARNEBACK_WINSIZE,
        _FARNEBACK_ITERATIONS,
        _FARNEBACK_POLY_N,
        _FARNEBACK_POLY_SIGMA,
        _FARNEBACK_FLAGS,
    )
    dx = flow[..., 0]
    dy = flow[..., 1]
    return np.sqrt(dx * dx + dy * dy).astype(np.float32)


def analyze_holographic_parallax(
    front_image: Optional[NDArray[np.uint8]],
    tilt_image: Optional[NDArray[np.uint8]],
    *,
    chroma_threshold: float = DEFAULT_CHROMA_THRESHOLD,
    saturation_threshold: int = DEFAULT_SATURATION_THRESHOLD,
    morph_kernel_size: int = DEFAULT_MORPH_KERNEL_SIZE,
    min_mask_fraction: float = DEFAULT_MIN_MASK_FRACTION,
    full_confidence_mask_fraction: float = DEFAULT_FULL_CONFIDENCE_MASK_FRACTION,
    ratio_midpoint: float = DEFAULT_RATIO_MIDPOINT,
    ratio_slope: float = DEFAULT_RATIO_SLOPE,
) -> HolographicResult:
    """Score holographic-parallax authenticity from a front + tilt pair.

    Args:
        front_image: BGR canonical front shot (uint8, HxWx3). None
            yields an abstain (`invalid_image`).
        tilt_image: BGR canonical tilt_30 shot (uint8, HxWx3). None
            yields an abstain (`tilt_not_captured`) — the wizard step
            is optional, so an unverified result is the right
            outcome rather than skipping silently.
        chroma_threshold, saturation_threshold, morph_kernel_size,
            min_mask_fraction, full_confidence_mask_fraction: holo-mask
            heuristic knobs. Module-level defaults are calibrated for
            the canonical 750x1050 dewarp.
        ratio_midpoint, ratio_slope: logistic squash applied to the
            inside-vs-outside flow ratio.

    Returns:
        HolographicResult. Always returns a result (never raises) —
        the abstain path is a result with confidence=0, not an
        exception. The verdict mapper in ensemble.py turns
        confidence=0 into UNVERIFIED.
    """
    # --- abstain when tilt isn't captured ---
    if tilt_image is None:
        return _no_signal(
            reason="tilt_not_captured",
            metadata={"reason": "tilt_not_captured"},
        )

    # --- abstain on missing/invalid front ---
    if not _is_valid_canonical(front_image):
        return _no_signal(
            reason="invalid_image",
            metadata={
                "reason": "invalid_image",
                "which": "front",
                "shape": tuple(getattr(front_image, "shape", ())),
            },
        )
    if not _is_valid_canonical(tilt_image):
        return _no_signal(
            reason="invalid_image",
            metadata={
                "reason": "invalid_image",
                "which": "tilt",
                "shape": tuple(getattr(tilt_image, "shape", ())),
            },
        )

    # --- abstain on shape mismatch ---
    # The flow computation requires identical shapes. If front and tilt
    # came back from the dewarp at different sizes (shouldn't happen,
    # but defend against it), abstain rather than rescaling — the score
    # would be meaningless.
    if front_image.shape != tilt_image.shape:
        return _no_signal(
            reason="shape_mismatch",
            metadata={
                "reason": "shape_mismatch",
                "front_shape": tuple(front_image.shape),
                "tilt_shape": tuple(tilt_image.shape),
            },
        )

    # --- holo-region mask on the front shot ---
    try:
        mask = _build_holo_mask(
            front_image,
            chroma_threshold=chroma_threshold,
            saturation_threshold=saturation_threshold,
            morph_kernel_size=morph_kernel_size,
        )
    except cv2.error as e:
        return _no_signal(
            reason="invalid_image",
            metadata={"reason": "mask_build_failed", "error": str(e)},
        )

    total_pixels = float(mask.shape[0] * mask.shape[1])
    if total_pixels <= 0:
        return _no_signal(
            reason="invalid_image",
            metadata={"reason": "empty_mask_shape"},
        )
    holo_pixels = float((mask > 0).sum())
    mask_fraction = holo_pixels / total_pixels

    if mask_fraction < min_mask_fraction:
        return _no_signal(
            reason="no_holo_region",
            holo_mask_fraction=mask_fraction,
            metadata={
                "reason": "no_holo_region",
                "mask_fraction": float(mask_fraction),
                "min_mask_fraction": float(min_mask_fraction),
            },
        )

    # --- dense Farnebäck flow ---
    try:
        front_gray = cv2.cvtColor(front_image, cv2.COLOR_BGR2GRAY)
        tilt_gray = cv2.cvtColor(tilt_image, cv2.COLOR_BGR2GRAY)
        flow_mag = _flow_magnitude(front_gray, tilt_gray)
    except (cv2.error, Exception) as e:  # noqa: BLE001 — defensive
        return _no_signal(
            reason="flow_computation_failed",
            holo_mask_fraction=mask_fraction,
            metadata={"reason": "flow_computation_failed", "error": str(e)},
        )

    if flow_mag.shape != (mask.shape[0], mask.shape[1]):
        # Defensive — shouldn't happen given the shape gate above, but
        # cv2 has surprised us before.
        return _no_signal(
            reason="flow_computation_failed",
            holo_mask_fraction=mask_fraction,
            metadata={
                "reason": "flow_shape_mismatch",
                "flow_shape": tuple(flow_mag.shape),
                "mask_shape": tuple(mask.shape),
            },
        )

    bool_mask = mask > 0
    inside = flow_mag[bool_mask]
    outside = flow_mag[~bool_mask]

    if inside.size == 0 or outside.size == 0:
        # Degenerate: mask covered the whole image (no outside) or the
        # cleanup ate the mask (no inside). Either way the ratio is
        # undefined.
        return _no_signal(
            reason="no_holo_region",
            holo_mask_fraction=mask_fraction,
            metadata={
                "reason": "degenerate_mask",
                "inside_pixels": int(inside.size),
                "outside_pixels": int(outside.size),
            },
        )

    inside_mean = float(inside.mean())
    outside_mean = float(outside.mean())

    # Floor the denominator to avoid divide-by-near-zero on perfectly
    # aligned non-foil regions (e.g. synthetic fixtures with zero
    # background motion). 0.05 px is a sub-pixel sentinel below which
    # we consider the non-foil region "stationary" and the ratio
    # numerically unstable; we cap inside-mean against the same floor
    # so the ratio doesn't blow up to infinity on synthetic shifts.
    EPS = 0.05
    denom = max(outside_mean, EPS)
    flow_ratio = inside_mean / denom

    # --- score + confidence ---
    score = _logistic(flow_ratio, ratio_midpoint, ratio_slope)
    if mask_fraction >= full_confidence_mask_fraction:
        confidence = _CONFIDENCE_FULL
    else:
        # Linear ramp from REDUCED at min_mask_fraction to FULL at
        # full_confidence_mask_fraction.
        span = max(
            full_confidence_mask_fraction - min_mask_fraction, 1e-6
        )
        t = (mask_fraction - min_mask_fraction) / span
        t = max(0.0, min(1.0, t))
        confidence = _CONFIDENCE_REDUCED + t * (_CONFIDENCE_FULL - _CONFIDENCE_REDUCED)

    return HolographicResult(
        score=float(score),
        confidence=float(confidence),
        flow_ratio=float(flow_ratio),
        holo_mask_fraction=float(mask_fraction),
        abstain_reason=None,
        manufacturer_profile="generic",
        metadata={
            "inside_mean_flow": inside_mean,
            "outside_mean_flow": outside_mean,
            "inside_pixels": int(inside.size),
            "outside_pixels": int(outside.size),
            "ratio_midpoint": float(ratio_midpoint),
            "ratio_slope": float(ratio_slope),
            "chroma_threshold": float(chroma_threshold),
            "saturation_threshold": int(saturation_threshold),
            "min_mask_fraction": float(min_mask_fraction),
        },
    )


__all__ = [
    "DEFAULT_CANONICAL_HEIGHT",
    "DEFAULT_CANONICAL_WIDTH",
    "DEFAULT_CHROMA_THRESHOLD",
    "DEFAULT_SATURATION_THRESHOLD",
    "DEFAULT_MORPH_KERNEL_SIZE",
    "DEFAULT_MIN_MASK_FRACTION",
    "DEFAULT_FULL_CONFIDENCE_MASK_FRACTION",
    "DEFAULT_RATIO_MIDPOINT",
    "DEFAULT_RATIO_SLOPE",
    "HolographicResult",
    "analyze_holographic_parallax",
]
