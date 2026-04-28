"""Blur detection via variance of the Laplacian.

Higher values are sharper. Card photos typically need >100 to be usable for
grading; below 50 is unmistakably blurry. The threshold is a setting, not a
constant, because phone cameras and webcams have different noise floors."""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray


def blur_score(image: NDArray[np.uint8]) -> float:
    """Variance of the Laplacian. Returns a non-negative float; higher = sharper."""
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def is_sharp(image: NDArray[np.uint8], threshold: float = 100.0) -> bool:
    return blur_score(image) >= threshold
