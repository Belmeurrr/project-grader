"""Centering measurement.

Pure geometric measurement, no learning. The dewarped image is assumed to be a
canonical 750x1050 orthographic view of the card (Stage 2 of the pipeline).
We detect the inner printed border and compute the four border thicknesses
relative to the cut edge.

The output is a (left/right, top/bottom) ratio expressed PSA-style as e.g. "55/45".
A standalone subgrade is then derived via the PSA published rule table. This
component is the single highest-accuracy stage in the pipeline by construction.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class CenteringRatios:
    """Border thickness ratios on a 0..100 scale; each pair sums to 100."""

    left: float
    right: float
    top: float
    bottom: float

    @property
    def lr(self) -> tuple[float, float]:
        return (self.left, self.right)

    @property
    def tb(self) -> tuple[float, float]:
        return (self.top, self.bottom)

    @property
    def worst_axis(self) -> str:
        """Returns 'lr' or 'tb' — whichever axis is more off-center."""
        lr_offset = abs(self.left - 50.0)
        tb_offset = abs(self.top - 50.0)
        return "lr" if lr_offset >= tb_offset else "tb"

    @property
    def worst_pair(self) -> tuple[float, float]:
        return self.lr if self.worst_axis == "lr" else self.tb


@dataclass(frozen=True)
class CenteringMeasurement:
    """Full output of a centering measurement on one face of a card."""

    ratios: CenteringRatios
    inner_box_xyxy: tuple[int, int, int, int]
    outer_box_xyxy: tuple[int, int, int, int]
    confidence: float


def _to_grayscale(img: NDArray[np.uint8]) -> NDArray[np.uint8]:
    if img.ndim == 2:
        return img
    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _peak_centroid(profile: NDArray[np.float32], hi_quantile: float = 0.80) -> float:
    """Return the centroid (mean index) of pixels at or above hi_quantile of the
    profile peak. Robust to flat-topped peaks where np.argmax would arbitrarily
    pick the leftmost tied index, biasing measurements by ±1 pixel."""
    peak = float(profile.max())
    if peak <= 0.0:
        return float(profile.argmax())
    threshold = peak * hi_quantile
    indices = np.flatnonzero(profile >= threshold)
    if indices.size == 0:
        return float(profile.argmax())
    weights = profile[indices]
    return float(np.sum(indices * weights) / np.sum(weights))


def _detect_inner_border(
    gray: NDArray[np.uint8],
    outer_xyxy: tuple[int, int, int, int],
    search_fraction: float = 0.20,
) -> tuple[tuple[int, int, int, int], float]:
    """Find the inner printed border by scanning for the strongest gradient
    transitioning away from each outer edge.

    The card border is typically a sharp light/dark transition: the outer cut
    edge → white margin → printed image. We look for the printed-image edge by
    scanning each row/column inwards from the cut edge and locating the
    high-gradient transition past the white margin.

    Returns the inner box and a confidence score in [0, 1].
    """
    x0, y0, x1, y1 = outer_xyxy
    h = y1 - y0
    w = x1 - x0
    search_w = max(int(w * search_fraction), 5)
    search_h = max(int(h * search_fraction), 5)

    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_x_abs = np.abs(grad_x)
    grad_y_abs = np.abs(grad_y)

    left_band = grad_x_abs[y0:y1, x0 : x0 + search_w]
    left_profile = left_band.mean(axis=0)
    left_inner = x0 + int(round(_peak_centroid(left_profile)))

    right_band = grad_x_abs[y0:y1, x1 - search_w : x1]
    right_profile = right_band.mean(axis=0)
    right_inner = x1 - search_w + int(round(_peak_centroid(right_profile)))

    top_band = grad_y_abs[y0 : y0 + search_h, x0:x1]
    top_profile = top_band.mean(axis=1)
    top_inner = y0 + int(round(_peak_centroid(top_profile)))

    bottom_band = grad_y_abs[y1 - search_h : y1, x0:x1]
    bottom_profile = bottom_band.mean(axis=1)
    bottom_inner = y1 - search_h + int(round(_peak_centroid(bottom_profile)))

    def _prominence(profile: NDArray[np.float32]) -> float:
        peak = float(profile.max())
        median = float(np.median(profile)) + 1e-6
        return min(peak / (peak + median * 5.0), 1.0)

    confidence = float(
        np.mean(
            [
                _prominence(left_profile),
                _prominence(right_profile),
                _prominence(top_profile),
                _prominence(bottom_profile),
            ]
        )
    )

    return (left_inner, top_inner, right_inner, bottom_inner), confidence


def measure_centering(
    dewarped: NDArray[np.uint8],
    outer_xyxy: tuple[int, int, int, int] | None = None,
) -> CenteringMeasurement:
    """Measure centering on a dewarped (orthographic) card image.

    Args:
        dewarped: HxWx3 uint8 image, assumed to be the canonical orthographic
            view of one face of the card produced by Stage 2 of the pipeline.
        outer_xyxy: Optional explicit outer cut-edge box. If None, the outer
            box is taken to be the full image extent (typical for dewarped
            inputs where the card already fills the frame).

    Returns:
        CenteringMeasurement with ratios on a 0..100 scale and a confidence.
    """
    if dewarped.ndim not in (2, 3):
        raise ValueError(f"expected 2D or 3D image, got shape {dewarped.shape}")
    if dewarped.dtype != np.uint8:
        raise ValueError(f"expected uint8 image, got dtype {dewarped.dtype}")

    h, w = dewarped.shape[:2]
    if outer_xyxy is None:
        outer_xyxy = (0, 0, w, h)

    ox0, oy0, ox1, oy1 = outer_xyxy
    if not (0 <= ox0 < ox1 <= w and 0 <= oy0 < oy1 <= h):
        raise ValueError(f"outer_xyxy {outer_xyxy} is outside image bounds {(w, h)}")

    gray = _to_grayscale(dewarped)
    inner_xyxy, confidence = _detect_inner_border(gray, outer_xyxy)
    ix0, iy0, ix1, iy1 = inner_xyxy

    left = ix0 - ox0
    right = ox1 - ix1
    top = iy0 - oy0
    bottom = oy1 - iy1

    lr_total = left + right
    tb_total = top + bottom
    if lr_total <= 0 or tb_total <= 0:
        raise ValueError(
            f"degenerate border: lr_total={lr_total}, tb_total={tb_total} — inner box leaks past outer"
        )

    ratios = CenteringRatios(
        left=100.0 * left / lr_total,
        right=100.0 * right / lr_total,
        top=100.0 * top / tb_total,
        bottom=100.0 * bottom / tb_total,
    )

    return CenteringMeasurement(
        ratios=ratios,
        inner_box_xyxy=inner_xyxy,
        outer_box_xyxy=outer_xyxy,
        confidence=confidence,
    )


# PSA published centering tolerances (front, post-2025 update).
# Each entry: (max_offset_from_50, subgrade). max_offset is from the closer side,
# i.e. 55/45 → max_offset = 5. Sorted strictest first.
_PSA_FRONT_THRESHOLDS: tuple[tuple[float, float], ...] = (
    (5.0, 10.0),   # GEM-MT 10: ≤55/45
    (10.0, 9.0),   # MT 9:      ≤60/40
    (15.0, 8.0),   # NM-MT 8:   ≤65/35
    (20.0, 7.0),   # NM 7:      ≤70/30
    (25.0, 6.0),   # EX-MT 6:   ≤75/25
    (30.0, 5.0),   # EX 5:      ≤80/20
    (35.0, 4.0),   # VG-EX 4:   ≤85/15
    (40.0, 3.0),   # VG 3:      ≤90/10
    (45.0, 2.0),   # GOOD 2:    ≤95/5
    (50.0, 1.0),   # POOR 1
)

# PSA back centering tolerances are looser than front.
_PSA_BACK_THRESHOLDS: tuple[tuple[float, float], ...] = (
    (25.0, 10.0),  # ≤75/25
    (30.0, 9.0),   # ≤80/20
    (35.0, 8.0),
    (40.0, 7.0),
    (45.0, 6.0),
    (50.0, 5.0),
)


def psa_subgrade_from_ratios(
    front: CenteringRatios | None = None,
    back: CenteringRatios | None = None,
) -> float:
    """PSA-style centering subgrade. Held back by whichever face is worse.

    Front is required (back-only doesn't make physical sense). Back is optional;
    if not provided we assume the back is no worse than the front, which is the
    standard pre-grader assumption when only a front shot is captured.
    """
    if front is None:
        raise ValueError("front centering ratios are required")

    front_offset = abs(front.worst_pair[0] - 50.0)
    front_subgrade = _lookup_threshold(front_offset, _PSA_FRONT_THRESHOLDS)

    if back is None:
        return front_subgrade

    back_offset = abs(back.worst_pair[0] - 50.0)
    back_subgrade = _lookup_threshold(back_offset, _PSA_BACK_THRESHOLDS)
    return min(front_subgrade, back_subgrade)


def _lookup_threshold(offset: float, table: tuple[tuple[float, float], ...]) -> float:
    for max_offset, subgrade in table:
        if offset <= max_offset:
            return subgrade
    return 1.0
