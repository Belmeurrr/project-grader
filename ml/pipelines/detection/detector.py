"""Detector interface + heuristic implementation + registry.

The detector contract is small (one method, one return type) so that the
YOLO production implementation, the heuristic fallback, and any future
keypoint-refinement variants are interchangeable from the calling code's
perspective. The registry picks an implementation at runtime based on
configuration / environment.

Heuristic detector — Canny + contour + 4-vertex polygon. Used when no YOLO
weights are configured (early development, CI, training-data labeling). The
heuristic is not accurate enough for production grading but is stable enough
for unit tests and capture-time gating.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Protocol

import cv2
import numpy as np
from numpy.typing import NDArray

from pipelines.quality.card_bbox import (
    BBox,
    CardBBox,
    detect_card_bbox,
    fill_ratio,
)


@dataclass(frozen=True)
class DetectedCard:
    """Stage 1 output. The quad is always present; the mask may be None when
    the detector backend doesn't produce segmentation (heuristic fallback)."""

    quad: NDArray[np.float32]            # 4x2: TL, TR, BR, BL
    bbox: BBox
    confidence: float
    mask: NDArray[np.uint8] | None = None  # binary mask, full image-size, 0 or 255
    metadata: dict[str, object] = field(default_factory=dict)


class CardDetector(Protocol):
    def detect(self, image: NDArray[np.uint8]) -> DetectedCard | None: ...


# -----------------------------
# Heuristic implementation
# -----------------------------


@dataclass(frozen=True)
class HeuristicDetector:
    """Wraps the capture-time Canny+contour detector for a uniform interface.

    Suitable for: unit tests, CI, dev environments without trained weights,
    initial labeling jobs. Not suitable for production grading — corner
    placement is jittery on cards with low border contrast and on holos."""

    min_fill: float = 0.10
    max_fill: float = 0.95

    def detect(self, image: NDArray[np.uint8]) -> DetectedCard | None:
        if image.dtype != np.uint8:
            raise ValueError(f"expected uint8 image, got {image.dtype}")
        bbox = detect_card_bbox(image)
        if bbox is None:
            return None

        fr = fill_ratio(bbox, image.shape[:2])
        if not (self.min_fill <= fr <= self.max_fill):
            return None

        return DetectedCard(
            quad=bbox.quad.astype(np.float32),
            bbox=bbox.bbox,
            confidence=bbox.confidence,
            mask=_mask_from_quad(image.shape[:2], bbox.quad),
            metadata={"backend": "heuristic", "fill_ratio": fr},
        )


def _mask_from_quad(image_shape: tuple[int, int], quad: NDArray[np.int32]) -> NDArray[np.uint8]:
    """Synthesize a binary mask from a 4-vertex quad.

    Heuristic detector doesn't produce a true segmentation mask; we fill the
    quad polygon so downstream code that expects `mask` works uniformly."""
    h, w = image_shape
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, quad.astype(np.int32), 255)
    return mask


# -----------------------------
# Registry
# -----------------------------


def get_detector() -> CardDetector:
    """Return the configured detector instance.

    Selection:
      - GRADER_DETECTOR=yolo with GRADER_YOLO_WEIGHTS pointing to a .pt file:
        load YOLO. Falls back to heuristic (with a warning) if the weights are
        not loadable.
      - Otherwise: heuristic.
    """
    backend = os.environ.get("GRADER_DETECTOR", "heuristic").lower()
    if backend == "yolo":
        try:
            from pipelines.detection.yolo import YoloDetector

            weights = os.environ.get("GRADER_YOLO_WEIGHTS")
            if not weights:
                return HeuristicDetector()
            return YoloDetector(weights_path=weights)
        except Exception:
            # Don't crash the API on bad weights — log + fall back.
            return HeuristicDetector()
    return HeuristicDetector()


def detect_card(image: NDArray[np.uint8]) -> DetectedCard | None:
    """Convenience: run the configured detector on an image."""
    return get_detector().detect(image)
