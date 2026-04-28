"""Glare detection.

Glare on card photos is a near-white, low-saturation, high-value pixel cluster
caused by direct light reflection on a glossy card or holo foil. We detect it
in HSV: (V > v_threshold) AND (S < s_threshold). Returns the fraction of
pixels in the region that are glare. The pipeline rejects shots over a small
fraction (default 0.5%) — even small glare patches destroy surface-defect
signal in the affected area.

Holo cards and foils have intrinsically high values in some regions; flash
shots will (intentionally) have more glare. The pipeline treats flash and
no-flash captures separately — see report.evaluate_shot.
"""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray


def glare_score(
    image: NDArray[np.uint8],
    v_threshold: int = 240,
    s_threshold: int = 30,
) -> float:
    """Fraction of pixels classified as glare. Range [0, 1]."""
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected BGR image, got shape {image.shape}")

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    s = hsv[..., 1]
    v = hsv[..., 2]
    glare_mask = (v > v_threshold) & (s < s_threshold)
    return float(glare_mask.mean())


def has_glare(
    image: NDArray[np.uint8],
    max_fraction: float = 0.005,
    v_threshold: int = 240,
    s_threshold: int = 30,
) -> bool:
    return glare_score(image, v_threshold=v_threshold, s_threshold=s_threshold) > max_fraction
