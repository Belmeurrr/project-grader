"""Substrate / paper-fluorescence counterfeit detector (ensemble #6 of 7).

Paired-flash differential CIELAB b* signal on the white border of the
canonical front. Counterfeit cards printed on inkjet/photo-paper with
optical brighteners (OBAs) absorb UV in the camera flash and re-emit at
~440nm, producing a measurable shift toward NEGATIVE b* (more blue) on
the unprinted border. Authentic offset-printed TCG cardstock contains
minimal brighteners — the b* shift between flash and no-flash is small.

Inputs (both BGR uint8 ndarrays at the SAME canonical resolution, e.g.
750x1050 from Stage 2 dewarp):
  - `front_image`: ambient-lit canonical front
  - `flash_image`: same card, same dewarp, captured with phone flash on

ROI is the OUTER WHITE BORDER, not the card body. Reasoning:
  - Border is unprinted cardstock — substrate fluorescence is unmasked
    by ink. Body mid-grays are ink-on-stock; ink dominates b* and
    swamps the brightener signal we want to measure.
  - Border is ~6-8% of canvas (BORDER_WIDTH_FRAC=0.04 each side, all
    four sides) but well over 10k pixels at 750x1050, plenty for stable
    median+MAD statistics.
  - Corner squares (sized 2*BORDER_WIDTH_FRAC*dim) are excluded so a
    single-corner ding / tape residue / ink bleed doesn't dominate.

Pipeline:
  1. Validate inputs (None → flash_not_captured; shape mismatch →
     shape_mismatch; corrupt or wrong-shape arrays → invalid_image).
  2. Build the boolean border mask once (top + bottom + left + right
     bands of width BORDER_WIDTH_FRAC, minus the corner squares).
  3. Exposure-normalize the flash image so a brighter flash exposure
     doesn't get scored as a real fluorescence shift. We match the
     median border L* by scaling the BGR image: scale =
     median(L_no_flash) / median(L_flash), then re-LAB. Equivalent to
     scaling the L channel directly but the BGR-scale path keeps
     gamma/saturation intact across both shots.
  4. Compute the differential b* on the border:
        delta_b = median(b*_flash[border]) - median(b*_no_flash[border])
     Robust median is the right primitive here — single edge artifacts
     (a tiny shadow on one side) shouldn't dominate.
  5. Score via logistic squash:
        score = 1 / (1 + exp(-(delta_b - DELTA_B_MIDPOINT) * DELTA_B_SLOPE))
     Anchors:
       - delta_b ≈ 0 (authentic, no fluorescence) → ~0.95 (authentic-side)
       - delta_b ≈ -3 (midpoint, borderline) → 0.5
       - delta_b ≈ -10 (cheap fake on photo paper) → ~0.05
  6. Confidence ramp:
       - border_mad < 1.0 (clean border) → full confidence
       - border_mad >= 3.0 (noisy/dirty border) → 0.4
       - linearly interpolated between
       - hard-capped at MAX_CONFIDENCE=0.7 — algorithmic ceiling vs.
         a UV-lamp ground truth (the phone flash is UV-poor; LED phosphor-
         converted whites have negligible <400nm power, so even a perfect
         brightener-positive read is much weaker than a real UV lamp would
         produce)

Limits (documented as a SOFT contributing vote, never gates verdict):
  - Phone flash UV is weak. The UV pump that actually excites the OBAs
    is a few percent of the total flash photons (LED whites are
    blue-pumped phosphor, not violet/UV-pumped). Score ceiling on cheap
    fakes is realistically 60-75%, not 95%.
  - Useless on super-fakes printed on brightener-free offset stock —
    they don't fluoresce by construction.
  - This detector does NOT gate the verdict alone. The ensemble's
    `combine_verdicts` requires either an explicit LIKELY_COUNTERFEIT
    or unanimous AUTHENTIC; substrate contributes to that conservative
    decision, never overrides it.
  - Foil-exclusion gate is OUT OF SCOPE here. A foil-stamped border
    can fluoresce or not depending on the foil chemistry. We don't have
    a foil mask for the BORDER specifically (the holographic detector's
    foil mask is for the card body), so we accept that foil-bordered
    cards add some noise to the border statistic. Future: AND the
    border mask with `~foil_mask` once a border foil detector exists.

Design choices mirror the typography + holographic detectors:
  - Pure numpy + opencv. No new deps.
  - Frozen dataclass result with a fixed-shape "abstain" path so the
    verdict mapper handles the no-signal case identically.
  - All thresholds module-level; recalibration tool can ratchet them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import cv2
import numpy as np
from numpy.typing import NDArray


# ROI geometry — fractional border width measured from each outer edge.
# 0.04 * 750 = 30px wide on the short side; 0.04 * 1050 = 42px on the
# long side. Total border-mask area on a 750x1050 canvas with corner
# exclusion (corner side = 2*0.04*max_dim = 84px square at each corner)
# works out to ~70k pixels — well above MIN_BORDER_PIXELS.
BORDER_WIDTH_FRAC: float = 0.04

# Floor on usable border pixels. Below this (e.g. someone passed a
# tiny test fixture, or a future re-canonicalization made the canvas
# smaller) we abstain — the median+MAD wouldn't be stable enough.
MIN_BORDER_PIXELS: int = 2000

# Whether to run the L*-median exposure-normalization step. Off-by-default
# is a footgun; turning it off only makes sense for synthetic tests
# where the two images are guaranteed identically exposed.
EXPOSURE_NORMALIZE: bool = True

# Logistic squash for delta_b → score. Negative delta_b means flash
# pushed b* MORE blue than ambient (fluorescence signature). Positive
# delta_b is rare and means flash pushed b* yellower (could be color-cast
# from the LED's correlated color temperature, or just noise) — those
# score ABOVE midpoint by construction, which is correct (no
# fluorescence detected).
DELTA_B_MIDPOINT: float = -3.0
DELTA_B_SLOPE: float = 0.4

# Hard cap on confidence — algorithmic ceiling vs. a true UV-lamp
# ground truth. See module docstring; phone-flash UV power is a few
# percent of a real 365nm UV lamp, so even a perfect brightener-positive
# read deserves <1.0 confidence in the ensemble.
MAX_CONFIDENCE: float = 0.7

# Confidence ramp endpoints by border-MAD on the b* channel. Tuned
# against synthetic fixtures: a clean uniform-stock border has b*-MAD
# around 0.5-1.0; a dirty/scratched/sticker'd border can hit 3+. The
# ramp is linear between these anchors, then clamped to [0.4, MAX].
_CONF_MAD_FULL: float = 1.0  # MAD ≤ this → full confidence (then capped)
_CONF_MAD_MIN: float = 3.0   # MAD ≥ this → minimum (0.4)
_CONF_MIN: float = 0.4
_CONF_FULL: float = 1.0


@dataclass(frozen=True)
class SubstrateResult:
    """Output of `analyze_substrate`.

    Fields:
      score: [0, 1], higher = more likely authentic. 0.5 in abstain
        paths so the score itself is never confidently wrong; the
        verdict mapper consults `confidence` to decide UNVERIFIED.
      confidence: [0, 1]. Below the ensemble's MIN_CONFIDENCE
        threshold (~0.4) means abstain — see `abstain_reason`. Hard-
        capped at MAX_CONFIDENCE=0.7 on the happy path.
      delta_b: median(b*_flash[border]) - median(b*_no_flash[border]).
        Negative values indicate fluorescence (counterfeit signature).
        None on abstain paths.
      border_mad: MAD of b* across the border on the no-flash image.
        Lower = cleaner border = more confident reading. None on abstain.
      n_border_pixels: number of pixels in the border mask actually
        used. Below MIN_BORDER_PIXELS triggers abstain.
      abstain_reason: short tag explaining why confidence is 0
        (`flash_not_captured`, `shape_mismatch`, `invalid_image`,
        `border_too_small`). None on the happy path.
      manufacturer_profile: which calibration profile was used. v1
        always "generic"; future versions branch per-game once
        manufacturer-specific brightener loadings are characterized.
      metadata: backend extras (border-px count, post-normalization
        L* medians, etc.) for forensic review and offline calibration.
    """

    score: float
    confidence: float
    delta_b: Optional[float]
    border_mad: Optional[float]
    n_border_pixels: int
    abstain_reason: Optional[str]
    manufacturer_profile: str = "generic"
    metadata: dict[str, Any] = field(default_factory=dict)


def _logistic(x: float, midpoint: float, slope: float) -> float:
    """Numerically-stable logistic squash to [0, 1]. Same shape as the
    typography + knn_reference helpers — kept local rather than
    cross-importing because the detectors are intentionally independent."""
    z = slope * (x - midpoint)
    if z >= 0:
        ez = float(np.exp(-z))
        return 1.0 / (1.0 + ez)
    ez = float(np.exp(z))
    return ez / (1.0 + ez)


def _build_border_mask(h: int, w: int, border_width_frac: float) -> NDArray[np.bool_]:
    """Boolean mask of the outer border ring with corner squares excluded.

    The border is `border_width_frac` of `min(h, w)` on each of the four
    sides. The corner squares (side = 2 * border_width_frac * min(h, w))
    are excluded — corners are where dings, ink bleed, and tape residue
    cluster, and we don't want a single-corner artifact dominating the
    median across the whole border.
    """
    mask = np.zeros((h, w), dtype=bool)
    # Use min(h, w) so the band thickness doesn't go asymmetric on
    # non-square canvases.
    band = max(1, int(round(border_width_frac * min(h, w))))
    corner = max(1, int(round(2 * border_width_frac * min(h, w))))

    # Top + bottom bands (full width)
    mask[:band, :] = True
    mask[h - band:, :] = True
    # Left + right bands (height between top/bottom bands)
    mask[:, :band] = True
    mask[:, w - band:] = True

    # Exclude corner squares. We exclude the corner-square box from
    # each corner; this carves out the corners on all four sides.
    mask[:corner, :corner] = False
    mask[:corner, w - corner:] = False
    mask[h - corner:, :corner] = False
    mask[h - corner:, w - corner:] = False

    return mask


def _abstain(
    *,
    reason: str,
    n_border_pixels: int = 0,
    metadata: Optional[dict[str, Any]] = None,
) -> SubstrateResult:
    """Build an abstain-shaped SubstrateResult. Mirrors the typography +
    holographic + knn_reference detectors' no-signal helpers so the
    ensemble verdict mapper treats the abstain identically across
    detectors."""
    return SubstrateResult(
        score=0.5,
        confidence=0.0,
        delta_b=None,
        border_mad=None,
        n_border_pixels=int(n_border_pixels),
        abstain_reason=reason,
        manufacturer_profile="generic",
        metadata=metadata or {"reason": reason},
    )


def _is_valid_bgr(image: Any) -> bool:
    """Sanity check on a candidate BGR ndarray: 3-D, 3-channel, HxWx3,
    non-trivial size."""
    if not isinstance(image, np.ndarray):
        return False
    if image.ndim != 3 or image.shape[2] != 3:
        return False
    if image.shape[0] < 10 or image.shape[1] < 10:
        return False
    return True


def _confidence_from_mad(border_mad: float) -> float:
    """Linearly interpolate confidence between the MAD anchors, capped
    at MAX_CONFIDENCE on the high end and floored at _CONF_MIN on the
    low end."""
    if border_mad <= _CONF_MAD_FULL:
        unclipped = _CONF_FULL
    elif border_mad >= _CONF_MAD_MIN:
        unclipped = _CONF_MIN
    else:
        t = (border_mad - _CONF_MAD_FULL) / (_CONF_MAD_MIN - _CONF_MAD_FULL)
        unclipped = _CONF_FULL + t * (_CONF_MIN - _CONF_FULL)
    return float(min(MAX_CONFIDENCE, max(_CONF_MIN, unclipped)))


def analyze_substrate(
    front_image: NDArray[np.uint8] | None,
    flash_image: NDArray[np.uint8] | None,
    *,
    border_width_frac: float = BORDER_WIDTH_FRAC,
    min_border_pixels: int = MIN_BORDER_PIXELS,
    exposure_normalize: bool = EXPOSURE_NORMALIZE,
    delta_b_midpoint: float = DELTA_B_MIDPOINT,
    delta_b_slope: float = DELTA_B_SLOPE,
) -> SubstrateResult:
    """Score the paired-flash CIELAB b* differential on the white border.

    Args:
        front_image: ambient-lit BGR canonical (HxWx3 uint8). None →
            abstain `invalid_image` (the pipeline never calls us with
            front=None — the front canonical is always required —
            but defensive).
        flash_image: flash-lit BGR canonical (HxWx3 uint8) at the same
            shape as front_image. None → abstain `flash_not_captured`
            (the flash shot is OPTIONAL in the wizard).
        border_width_frac: fractional thickness of the border ring.
        min_border_pixels: floor on usable border pixels — below this
            we abstain `border_too_small`.
        exposure_normalize: when True, scale the flash image's BGR
            values so its border L* median matches the no-flash one
            before computing delta_b. Off only for tests that pre-
            normalize their fixtures.
        delta_b_midpoint, delta_b_slope: logistic squash parameters.

    Returns:
        SubstrateResult. Always returns a result (never raises) — every
        failure mode is encoded as confidence=0 with an abstain_reason.
    """
    # --- abstain on missing flash (most common: optional capture skipped) ---
    if flash_image is None:
        return _abstain(reason="flash_not_captured")

    # --- input validation ---
    if not _is_valid_bgr(front_image):
        return _abstain(
            reason="invalid_image",
            metadata={
                "reason": "invalid_image",
                "which": "front",
                "shape": tuple(getattr(front_image, "shape", ())),
            },
        )
    if not _is_valid_bgr(flash_image):
        return _abstain(
            reason="invalid_image",
            metadata={
                "reason": "invalid_image",
                "which": "flash",
                "shape": tuple(getattr(flash_image, "shape", ())),
            },
        )
    # Both must be the SAME canonical shape so the border mask aligns.
    if front_image.shape != flash_image.shape:
        return _abstain(
            reason="shape_mismatch",
            metadata={
                "reason": "shape_mismatch",
                "front_shape": tuple(front_image.shape),
                "flash_shape": tuple(flash_image.shape),
            },
        )

    h, w = front_image.shape[:2]

    # --- border mask ---
    mask = _build_border_mask(h, w, border_width_frac)
    n_border = int(mask.sum())
    if n_border < min_border_pixels:
        return _abstain(
            reason="border_too_small",
            n_border_pixels=n_border,
            metadata={
                "reason": "border_too_small",
                "n_border_pixels": n_border,
                "min_border_pixels": int(min_border_pixels),
            },
        )

    # --- exposure-normalize the flash image to match the no-flash L* median ---
    # The flash image is brighter overall by construction; without this
    # step the flash b* would be dominated by exposure-driven LAB
    # nonlinearity rather than the fluorescence we want to measure.
    front_lab = cv2.cvtColor(front_image, cv2.COLOR_BGR2LAB).astype(np.float32)
    flash_lab_pre = cv2.cvtColor(flash_image, cv2.COLOR_BGR2LAB).astype(np.float32)

    L_front_border = front_lab[..., 0][mask]
    L_flash_border_pre = flash_lab_pre[..., 0][mask]
    median_L_front = float(np.median(L_front_border))
    median_L_flash_pre = float(np.median(L_flash_border_pre))

    if exposure_normalize and median_L_flash_pre > 1e-3:
        scale = median_L_front / median_L_flash_pre
        # Clip into a sane range so we don't amplify a degenerate flash
        # exposure into nonsense. Real flash-vs-ambient exposure ratios
        # land in [1.0, 4.0]ish; we accept a wider band defensively.
        scale = float(np.clip(scale, 0.25, 4.0))
        flash_bgr_scaled = np.clip(
            flash_image.astype(np.float32) * scale, 0.0, 255.0
        ).astype(np.uint8)
        flash_lab = cv2.cvtColor(flash_bgr_scaled, cv2.COLOR_BGR2LAB).astype(np.float32)
    else:
        scale = 1.0
        flash_lab = flash_lab_pre

    # --- compute differential b* on the border (robust median + MAD) ---
    # OpenCV's LAB scales b* to [0, 255] for uint8 inputs. After the
    # cvtColor on float32 inputs (which we do here, post-uint8 cast),
    # b* is centered at 128 with a range of [0, 255]. Subtract 128 to
    # recover the standard CIELAB b* in approximate units.
    b_front = front_lab[..., 2][mask] - 128.0
    b_flash = flash_lab[..., 2][mask] - 128.0

    median_b_front = float(np.median(b_front))
    median_b_flash = float(np.median(b_flash))
    delta_b = median_b_flash - median_b_front

    # MAD on the no-flash border tells us how clean the substrate
    # patch was BEFORE we touched the flash channel — that's the
    # quality signal for confidence weighting.
    border_mad = float(np.median(np.abs(b_front - median_b_front)))

    # --- score + confidence ---
    score = _logistic(delta_b - delta_b_midpoint, 0.0, delta_b_slope)
    confidence = _confidence_from_mad(border_mad)

    return SubstrateResult(
        score=float(score),
        confidence=float(confidence),
        delta_b=float(delta_b),
        border_mad=float(border_mad),
        n_border_pixels=n_border,
        abstain_reason=None,
        manufacturer_profile="generic",
        metadata={
            "median_b_front": median_b_front,
            "median_b_flash": median_b_flash,
            "median_L_front": median_L_front,
            "median_L_flash_pre_normalize": median_L_flash_pre,
            "exposure_scale_applied": scale,
            "delta_b_midpoint": float(delta_b_midpoint),
            "delta_b_slope": float(delta_b_slope),
            "border_width_frac": float(border_width_frac),
            "image_shape": tuple(front_image.shape),
        },
    )


__all__ = [
    "BORDER_WIDTH_FRAC",
    "MIN_BORDER_PIXELS",
    "EXPOSURE_NORMALIZE",
    "DELTA_B_MIDPOINT",
    "DELTA_B_SLOPE",
    "MAX_CONFIDENCE",
    "SubstrateResult",
    "analyze_substrate",
]
