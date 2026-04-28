"""Edges measurement.

Geometric-only edge defect detection on a canonical 750x1050 dewarped card.
No learned model. Ships before training data; gets replaced by a
segmentation model (SegFormer-B2) once labeled data is available.

Algorithm:
  1. Take a thin strip (default 8 px) along each of the 4 perimeter sides.
  2. For each side, compute a 1-D luminance signal projected along the
     side's long axis (mean of strip's perpendicular direction).
  3. Robustly estimate "expected" luminance via local median over a
     sliding window (default 30 px).
  4. Anomalies = pixels whose deviation from local median exceeds
     k * MAD (median absolute deviation).
  5. Per-side anomaly fraction = anomaly_count / side_length.
  6. Overall = perimeter-weighted aggregate.
  7. Map to PSA edge subgrade via published-tolerance table.

Detects: whitening, chips, dings, dirt marks. Does not detect: subtle
fraying that's lost in dewarp interpolation (that's what the SegFormer
v2 will catch).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np
from numpy.typing import NDArray

Side = Literal["top", "right", "bottom", "left"]
SIDES: tuple[Side, ...] = ("top", "right", "bottom", "left")


@dataclass(frozen=True)
class SideMeasurement:
    side: Side
    length_px: int
    anomaly_count: int
    anomaly_fraction: float
    luminance_std: float

    @property
    def luminance_smoothness(self) -> float:
        """1.0 = perfectly uniform, 0.0 = highly variable."""
        return float(max(0.0, 1.0 - self.luminance_std / 64.0))


@dataclass(frozen=True)
class EdgeMeasurement:
    sides: list[SideMeasurement]
    overall_anomaly_fraction: float
    psa_subgrade: float
    confidence: float
    heatmap: NDArray[np.uint8] | None = None  # 0/255 mask of anomaly pixels in the canonical


def _to_grayscale(image: NDArray[np.uint8]) -> NDArray[np.uint8]:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _anomaly_mask_along_signal(
    signal: NDArray[np.float32],
    mad_threshold: float = 4.0,
) -> NDArray[np.bool_]:
    """Mark positions whose deviation from the *global* side median exceeds
    k * MAD. The strip's typical luminance is the card's border; defects
    show up as significant deviations from that baseline.

    A sliding-window median was tried first but absorbs wide defects (the
    window then contains mostly defect pixels, hiding the anomaly). The
    global-baseline approach handles defects up to ~40% of the side
    length correctly. Beyond that the defect IS the baseline — but a card
    with 40%+ of one side defective is a PSA 1 either way, so the
    saturation point is acceptable."""
    n = signal.size
    if n == 0:
        return np.zeros(0, dtype=bool)

    median = float(np.median(signal))
    mad = float(np.median(np.abs(signal - median)))
    # MAD floor prevents flagging tiny float noise on a perfectly clean
    # constant signal (where MAD is mathematically 0).
    threshold = mad_threshold * max(mad, 1.5)
    return np.abs(signal - median) > threshold


def _strip(gray: NDArray[np.uint8], side: Side, strip_px: int) -> NDArray[np.uint8]:
    if side == "top":
        return gray[:strip_px, :]
    if side == "bottom":
        return gray[-strip_px:, :]
    if side == "left":
        return gray[:, :strip_px]
    return gray[:, -strip_px:]


def _project(strip: NDArray[np.uint8], side: Side) -> NDArray[np.float32]:
    """Reduce a 2-D strip to a 1-D signal along the side's long axis.

    Use a robust per-column statistic (median across the strip width) so
    that a single-row anomaly doesn't distort the whole column when
    averaged. We're looking for *spatial* anomalies along the edge, not
    cross-strip averages."""
    if side in ("top", "bottom"):
        return np.median(strip, axis=0).astype(np.float32)
    return np.median(strip, axis=1).astype(np.float32)


def _scatter_back_to_canvas(
    side: Side,
    mask_1d: NDArray[np.bool_],
    canvas_shape: tuple[int, int],
    strip_px: int,
) -> NDArray[np.uint8]:
    canvas = np.zeros(canvas_shape, dtype=np.uint8)
    if side == "top":
        canvas[:strip_px, mask_1d] = 255
    elif side == "bottom":
        canvas[-strip_px:, mask_1d] = 255
    elif side == "left":
        canvas[mask_1d, :strip_px] = 255
    else:
        canvas[mask_1d, -strip_px:] = 255
    return canvas


def measure_edges(
    canonical: NDArray[np.uint8],
    strip_px: int = 8,
    mad_threshold: float = 4.0,
    return_heatmap: bool = True,
) -> EdgeMeasurement:
    """Measure edge defects on a canonical 750x1050 card image.

    Args:
        canonical: dewarped BGR uint8 card image, full-card extent.
        strip_px: width of the perimeter strip examined per side.
        mad_threshold: anomaly = deviation > threshold * MAD (per side).
        return_heatmap: when True, return a binary mask of detected
            anomalies (image-sized, 0 or 255). Set False to skip the
            allocation if you don't need it.
    """
    if canonical.dtype != np.uint8:
        raise ValueError(f"expected uint8 image, got {canonical.dtype}")
    if canonical.ndim not in (2, 3):
        raise ValueError(f"expected 2- or 3-D image, got shape {canonical.shape}")

    gray = _to_grayscale(canonical)
    h, w = gray.shape
    if h < strip_px * 3 or w < strip_px * 3:
        raise ValueError(f"image too small for strip_px={strip_px}")

    sides_out: list[SideMeasurement] = []
    heatmap: NDArray[np.uint8] | None = (
        np.zeros((h, w), dtype=np.uint8) if return_heatmap else None
    )

    total_anomalies = 0
    total_perimeter = 0

    for side in SIDES:
        strip = _strip(gray, side, strip_px)
        signal = _project(strip, side)
        anomaly_mask = _anomaly_mask_along_signal(signal, mad_threshold=mad_threshold)
        length = int(signal.size)
        anomaly_count = int(anomaly_mask.sum())
        sides_out.append(
            SideMeasurement(
                side=side,
                length_px=length,
                anomaly_count=anomaly_count,
                anomaly_fraction=anomaly_count / length if length else 0.0,
                luminance_std=float(np.std(signal)),
            )
        )
        total_anomalies += anomaly_count
        total_perimeter += length
        if heatmap is not None:
            np.maximum(heatmap, _scatter_back_to_canvas(side, anomaly_mask, (h, w), strip_px), out=heatmap)

    overall = (total_anomalies / total_perimeter) if total_perimeter else 0.0
    subgrade = psa_subgrade_from_anomaly_fraction(overall)

    # Confidence: drops as the per-side detection becomes more variable.
    # A perfectly clean card (0 anomalies on every side) has confidence 1.0;
    # heavy disagreement between sides reduces certainty.
    fractions = np.asarray([s.anomaly_fraction for s in sides_out], dtype=np.float32)
    spread = float(fractions.max() - fractions.min()) if fractions.size else 0.0
    confidence = float(max(0.5, 1.0 - spread * 2.0))

    return EdgeMeasurement(
        sides=sides_out,
        overall_anomaly_fraction=float(overall),
        psa_subgrade=subgrade,
        confidence=confidence,
        heatmap=heatmap,
    )


# PSA edges thresholds. Anomaly fraction = fraction of perimeter pixels
# flagged as defects. Calibrated against PSA's published descriptions; will
# be retuned with real data once we have labeled examples.
_PSA_EDGE_THRESHOLDS: tuple[tuple[float, float], ...] = (
    (0.005, 10.0),
    (0.015, 9.0),
    (0.030, 8.0),
    (0.050, 7.0),
    (0.080, 6.0),
    (0.120, 5.0),
    (0.180, 4.0),
    (0.250, 3.0),
    (0.350, 2.0),
)


def psa_subgrade_from_anomaly_fraction(fraction: float) -> float:
    """Map perimeter anomaly fraction to PSA edge subgrade (1-10).

    Tunable; production will recalibrate against PSA-graded examples once
    the data ingestion pipeline is producing labeled images."""
    if fraction < 0:
        raise ValueError("anomaly fraction must be non-negative")
    for max_fraction, subgrade in _PSA_EDGE_THRESHOLDS:
        if fraction <= max_fraction:
            return subgrade
    return 1.0
