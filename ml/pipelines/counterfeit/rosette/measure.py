"""Print-rosette FFT counterfeit detector.

The most novel detector in the 7-detector counterfeit-authenticity ensemble.
Detects the halftone "rosette" pattern that authentic trading cards exhibit
because they're printed at ~150 LPI on offset presses. Consumer-printer
counterfeits print continuous-tone via inkjet and produce no rosette
structure — so the presence/absence of this pattern is a manufacturer-level
fingerprint that works on cards-never-seen-before (no per-card reference
required, only knowledge of how the manufacturer prints).

Algorithm:
  1. Locate ~n_patches roughly-flat-color patches in the inner image area
     (avoiding the white border, which is ~40 px). A patch is patch_size x
     patch_size with low local std-dev — flat regions are where the
     halftone-dot pattern is most visible against a uniform background.
  2. For each patch: convert to grayscale, apply a Hann window (avoid FFT
     edge-ringing artifacts that create spurious low-frequency leakage),
     compute the 2-D FFT magnitude spectrum, take the radial profile (mean
     magnitude vs frequency, binned by integer radius).
  3. Look for a peak in the configured `expected_freq_band`. Per-patch
     prominence = peak_in_band / mean_off_band (a signal-to-baseline
     ratio that's robust to overall image energy).
  4. Aggregate across patches (mean prominence) and squash to [0, 1] via a
     logistic centered at a calibrated prominence threshold.
  5. Confidence drops if too few flat patches were found (the algorithm is
     unreliable on highly textured images where no clean halftone region is
     observable).

Why a radial profile? The 2-D rosette pattern produces ring-shaped peaks in
the magnitude spectrum at the dot frequency (rotation-invariant in the
spectrum because the print rotation is unknown — different ink colors
print at 15°/45°/75° to minimize moire, so the visible pattern is multiple
overlaid rotations). Averaging over rings collapses this into a 1-D
signature with a single peak at the dot frequency.

Why a Hann window? Without windowing, the rectangular-patch FFT exhibits
strong axis-aligned leakage (the "+ pattern" in any FFT of a patch with
non-periodic boundaries). That leakage often looks like a rosette peak and
gives false-positive scores on continuous-tone images. The Hann window
suppresses the boundary discontinuity and isolates true halftone signal.

Frequency-band defaults are calibrated for a 64 px patch from a 750x1050
canonical card. At ~150 LPI and a typical 300dpi canonical scan, halftone
cells fall around 6-12 px/cycle, mapping to spectral radius ~ patch_size /
cell_size ≈ 5-11 cycles per patch. Tunable via expected_freq_band.

Detects: continuous-tone counterfeits (the dominant consumer-printer fake).
Does not detect: counterfeits that themselves use offset/halftone printing
(rare — high BOM cost — but they do exist; that's why this detector is
one signal in a 7-detector ensemble, not a standalone authenticator).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

# Inner-area inset (px) used when scanning for flat patches. Default canonical
# cards have ~40 px of white border (see tests.fixtures.synth_card); we step
# in a bit further to avoid landing a patch on the inner-border transition.
DEFAULT_INNER_INSET_PX: int = 56

# Default expected halftone band, in spectral-radius units (cycles per patch).
# For patch_size=64 and ~150 LPI offset print at 300dpi canonical resolution,
# the dot spacing is roughly 6-12 px → 5-11 cycles per 64-px patch. Slightly
# widened for tolerance to scan-resolution and rotation variation.
DEFAULT_FREQ_BAND: tuple[float, float] = (5.0, 14.0)

# Logistic squash midpoint for the per-patch prominence ratio. A peak
# prominence of ~3.5x the off-band mean is the empirical decision boundary
# from synthetic halftone vs. continuous-tone fixtures: smooth gradients
# leak ~2x into low frequency bins (so even a perfectly continuous-tone
# image typically has prominence near 2), while a clean halftone pattern
# produces prominence well above 10. Putting the midpoint above the
# noise-floor of gradient leakage keeps continuous-tone scores low.
# Will be recalibrated against real PSA pop-report images once the data
# ingestion pipeline runs.
PROMINENCE_MIDPOINT: float = 3.5
PROMINENCE_SLOPE: float = 1.4

# Maximum std-dev of the *heavily-blurred* patch for it to count as "flat".
# Halftone dots themselves create high local contrast (std ~70 on a typical
# 64-px patch), so we measure flatness on a blurred version of the image:
# a patch is "flat" if its blurred form has low variance, meaning the
# underlying ink coverage is uniform regardless of the high-frequency dot
# structure. This is what we need for FFT analysis — the dots must be the
# dominant signal, not edges or text.
MAX_FLAT_PATCH_STDDEV: float = 12.0

# Gaussian-blur kernel used to reject patches dominated by macroscale
# structure (text, edges, gradients). Must be much larger than the
# halftone cell so even jittered/imperfect dot patterns smooth out, but
# smaller than typical text/edge widths so text patches still get rejected.
FLATNESS_BLUR_KERNEL: int = 31


@dataclass(frozen=True)
class RosetteMeasurement:
    """Output of `measure_rosette`.

    Fields:
      rosette_score: [0, 1], higher = more likely authentic. 0.5 indicates
        "uncertain" (e.g. when no flat patches were found).
      peak_strength: raw mean per-patch prominence (peak / off-band mean)
        across the analyzed patches. Provided for debug/forensic review.
      analyzed_patches: number of flat-color patches actually FFT'd. May be
        less than the requested n_patches if the inner area is highly
        textured.
      confidence: [0, 1]. Drops linearly as analyzed_patches falls below
        the requested count; reaches 0 if no patches were analyzed.
      manufacturer_profile: which calibration profile was used. v1 always
        "generic"; future versions will branch per-manufacturer (Topps vs.
        Pokemon Company vs. Konami vs. ...).
      metadata: backend extras (per-patch prominences, patch coords) for
        UI overlay and offline calibration analyses.
    """

    rosette_score: float
    peak_strength: float
    analyzed_patches: int
    confidence: float
    manufacturer_profile: str
    metadata: dict[str, Any] = field(default_factory=dict)


def _to_grayscale(image: NDArray[np.uint8]) -> NDArray[np.uint8]:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _logistic(x: float, midpoint: float, slope: float) -> float:
    """Numerically-stable logistic squash to [0, 1]."""
    z = slope * (x - midpoint)
    # Avoid overflow on extreme inputs.
    if z >= 0:
        ez = float(np.exp(-z))
        return 1.0 / (1.0 + ez)
    ez = float(np.exp(z))
    return ez / (1.0 + ez)


def _find_flat_patches(
    gray: NDArray[np.uint8],
    patch_size: int,
    n_patches: int,
    inner_inset_px: int,
    max_stddev: float,
) -> list[tuple[int, int]]:
    """Grid-scan for flat patches in the inner area.

    "Flat" here means the *blurred* patch has low std-dev — that is, the
    underlying mean ink coverage is uniform, even though the patch itself
    may contain high-contrast halftone dots. This is the right criterion
    for FFT analysis: we want patches whose low-frequency content is flat
    so that any spectral peaks come from the halftone pattern, not from
    text/edge structure.

    Returns a list of (y, x) top-left coordinates. Up to n_patches are
    returned, picked by lowest blurred-std (flattest first) for highest
    FFT signal-to-noise."""
    h, w = gray.shape
    y0 = inner_inset_px
    x0 = inner_inset_px
    y1 = h - inner_inset_px - patch_size
    x1 = w - inner_inset_px - patch_size
    if y1 <= y0 or x1 <= x0:
        return []

    blurred = cv2.GaussianBlur(gray, (FLATNESS_BLUR_KERNEL, FLATNESS_BLUR_KERNEL), 0)

    # Stride chosen so that the candidate grid contains many more positions
    # than n_patches, even on small inner areas, while keeping the scan
    # cheap (O(grid) of fast numpy slicing).
    stride = max(patch_size // 2, 16)

    candidates: list[tuple[float, int, int]] = []
    for y in range(y0, y1 + 1, stride):
        for x in range(x0, x1 + 1, stride):
            blurred_patch = blurred[y : y + patch_size, x : x + patch_size]
            std = float(blurred_patch.std())
            if std <= max_stddev:
                candidates.append((std, y, x))

    if not candidates:
        return []

    # Sort by std ascending (flattest first), then de-duplicate spatially so
    # we don't pick n_patches overlapping copies of the same flat region.
    candidates.sort(key=lambda t: t[0])
    selected: list[tuple[int, int]] = []
    min_separation = patch_size  # non-overlapping
    for _, y, x in candidates:
        if all(
            max(abs(y - py), abs(x - px)) >= min_separation for py, px in selected
        ):
            selected.append((y, x))
            if len(selected) >= n_patches:
                break
    return selected


def _radial_profile(magnitude: NDArray[np.float32]) -> NDArray[np.float32]:
    """Mean magnitude binned by integer radius from the spectrum center.

    Returns a 1-D array of length `min(h, w) // 2` (we only use frequencies
    inside the inscribed disk to avoid corner-of-spectrum artifacts)."""
    h, w = magnitude.shape
    cy, cx = h // 2, w // 2
    y, x = np.indices((h, w))
    r = np.hypot(y - cy, x - cx).astype(np.int32)

    max_r = min(cy, cx)
    # bincount-based mean: sum per bin / count per bin.
    flat_r = r.ravel()
    flat_m = magnitude.ravel()
    sums = np.bincount(flat_r, weights=flat_m, minlength=max_r + 1)
    counts = np.bincount(flat_r, minlength=max_r + 1)
    # Avoid divide-by-zero; bins with no pixels return 0.
    profile = np.zeros_like(sums, dtype=np.float32)
    nonzero = counts > 0
    profile[nonzero] = (sums[nonzero] / counts[nonzero]).astype(np.float32)
    return profile[:max_r]


def _patch_prominence(
    patch: NDArray[np.uint8],
    expected_freq_band: tuple[float, float],
) -> float:
    """Per-patch peak prominence in the expected halftone band.

    1. Hann-window the patch (avoid spectral leakage from boundaries).
    2. 2-D FFT magnitude, fft-shifted so DC is centered.
    3. Radial profile.
    4. peak_in_band / mean_off_band, with the DC bin and immediate
       low-frequency neighborhood excluded from "off-band" (DC dwarfs
       everything and would distort the baseline)."""
    h, w = patch.shape
    # Hann window applied separably (outer product of 1-D Hann windows).
    win_y = np.hanning(h).astype(np.float32)
    win_x = np.hanning(w).astype(np.float32)
    window = np.outer(win_y, win_x)
    windowed = patch.astype(np.float32) * window

    spectrum = np.fft.fftshift(np.fft.fft2(windowed))
    magnitude = np.abs(spectrum).astype(np.float32)

    profile = _radial_profile(magnitude)
    if profile.size == 0:
        return 0.0

    lo, hi = expected_freq_band
    lo_i = int(max(1, np.floor(lo)))
    hi_i = int(min(profile.size - 1, np.ceil(hi)))
    if hi_i <= lo_i:
        return 0.0

    in_band = profile[lo_i : hi_i + 1]
    if in_band.size == 0:
        return 0.0
    peak = float(in_band.max())

    # Off-band baseline: skip DC and the first few bins (low-freq energy
    # dominated by overall patch luminance + windowing tail), and skip the
    # in-band itself.
    dc_skip = 3
    off_lo = profile[dc_skip:lo_i] if lo_i > dc_skip else np.empty(0, dtype=np.float32)
    off_hi = profile[hi_i + 1 :]
    off = np.concatenate([off_lo, off_hi]) if (off_lo.size + off_hi.size) else np.empty(0, dtype=np.float32)
    if off.size == 0:
        return 0.0
    baseline = float(off.mean())
    if baseline <= 1e-6:
        return 0.0
    return peak / baseline


def measure_rosette(
    canonical: NDArray[np.uint8],
    patch_size: int = 64,
    n_patches: int = 5,
    expected_freq_band: tuple[float, float] = DEFAULT_FREQ_BAND,
    inner_inset_px: int = DEFAULT_INNER_INSET_PX,
) -> RosetteMeasurement:
    """Measure print-rosette signal on a canonical 750x1050 BGR card image.

    Args:
        canonical: dewarped BGR uint8 card image. Production input is the
            canonical produced by Stage 2 (perspective correction); for unit
            tests the synthetic fixtures in tests.fixtures are used.
        patch_size: side length of each square patch FFT'd. 64 is the
            default — large enough to resolve the halftone band cleanly,
            small enough to find multiple flat regions on a busy card.
        n_patches: how many flat patches to analyze. The score is the mean
            across them; more patches means more robust score but linear cost.
        expected_freq_band: (low, high) spectral-radius window where the
            halftone peak is expected. Default tuned for a 64 px patch on
            ~150 LPI offset print at 300dpi canonical resolution.
        inner_inset_px: how many pixels in from each image edge to start
            scanning for flat patches (avoids the white border).

    Returns:
        RosetteMeasurement with rosette_score in [0, 1] (higher = more likely
        authentic) and confidence in [0, 1].
    """
    if canonical.dtype != np.uint8:
        raise ValueError(f"expected uint8 image, got {canonical.dtype}")
    if canonical.ndim != 3 or canonical.shape[2] != 3:
        raise ValueError(
            f"expected 3-channel BGR image, got shape {canonical.shape}"
        )
    if patch_size <= 0:
        raise ValueError(f"patch_size must be positive, got {patch_size}")
    if n_patches <= 0:
        raise ValueError(f"n_patches must be positive, got {n_patches}")

    h, w = canonical.shape[:2]
    if h < patch_size + 2 * inner_inset_px or w < patch_size + 2 * inner_inset_px:
        raise ValueError(
            f"image too small for patch_size={patch_size} with "
            f"inner_inset_px={inner_inset_px}: got {(h, w)}"
        )

    lo, hi = expected_freq_band
    if not (0 < lo < hi):
        raise ValueError(
            f"expected_freq_band must be (low, high) with 0 < low < high, got {expected_freq_band}"
        )

    gray = _to_grayscale(canonical)
    patch_coords = _find_flat_patches(
        gray,
        patch_size=patch_size,
        n_patches=n_patches,
        inner_inset_px=inner_inset_px,
        max_stddev=MAX_FLAT_PATCH_STDDEV,
    )

    if not patch_coords:
        # No flat regions — algorithm can't speak. Return uncertain.
        return RosetteMeasurement(
            rosette_score=0.5,
            peak_strength=0.0,
            analyzed_patches=0,
            confidence=0.0,
            manufacturer_profile="generic",
            metadata={"per_patch_prominence": [], "patch_coords": []},
        )

    prominences: list[float] = []
    for y, x in patch_coords:
        patch = gray[y : y + patch_size, x : x + patch_size]
        prominences.append(_patch_prominence(patch, expected_freq_band))

    mean_prominence = float(np.mean(prominences))
    score = _logistic(mean_prominence, PROMINENCE_MIDPOINT, PROMINENCE_SLOPE)

    # Confidence: full when we got the requested number of patches, drops
    # linearly as fewer flat regions are found.
    confidence = float(min(1.0, len(patch_coords) / n_patches))

    return RosetteMeasurement(
        rosette_score=float(score),
        peak_strength=mean_prominence,
        analyzed_patches=len(patch_coords),
        confidence=confidence,
        manufacturer_profile="generic",
        metadata={
            "per_patch_prominence": [float(p) for p in prominences],
            "patch_coords": [(int(y), int(x)) for y, x in patch_coords],
            "expected_freq_band": tuple(float(v) for v in expected_freq_band),
            "patch_size": int(patch_size),
        },
    )


def is_likely_authentic(
    measurement: RosetteMeasurement,
    threshold: float = 0.5,
) -> bool:
    """Convenience predicate matching `measurement.rosette_score >= threshold`.

    Provided so callers don't have to care about the exact dataclass field
    name; the operational decision is "authentic vs. not", and we may add
    auxiliary signals (confidence-gated abstain, per-manufacturer thresholds)
    here in future without changing the call sites."""
    return measurement.rosette_score >= threshold
