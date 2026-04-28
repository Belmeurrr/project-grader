"""YOLOv11-seg detector — production card detection backend.

Loads a fine-tuned YOLOv11-segmentation model via the ``ultralytics`` package.
The model is expected to output one segmentation mask per detected card; the
quad is recovered by fitting a min-area rotated rectangle to the largest
mask. This is robust to mask boundary noise and is what the original card
boundary looks like under perspective.

The class is import-safe even when ultralytics is not installed and when the
weights file is missing — both of those produce a `RuntimeError` only on
first inference, not at construction time. This keeps test environments and
the heuristic-fallback registry path clean.

Training of these weights lives in `ml/training/trainers/detection.py`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from pipelines.detection.detector import DetectedCard
from pipelines.quality.card_bbox import BBox


@dataclass
class YoloDetector:
    """Lazy-loading YOLOv11-seg wrapper.

    Construction is cheap; the model is loaded on first `.detect()`. This lets
    workers boot without paying GPU init time before they have a job."""

    weights_path: str
    confidence_threshold: float = 0.35
    iou_threshold: float = 0.50
    device: str = "cuda:0"
    _model: Any = field(default=None, init=False, repr=False)

    def _ensure_loaded(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise RuntimeError(
                "ultralytics is not installed; either install training deps "
                "(`uv sync --extra training` in ml/) or set GRADER_DETECTOR=heuristic"
            ) from e
        try:
            self._model = YOLO(self.weights_path)
        except Exception as e:
            raise RuntimeError(f"failed to load YOLO weights at {self.weights_path}: {e}") from e
        return self._model

    def detect(self, image: NDArray[np.uint8]) -> DetectedCard | None:
        if image.dtype != np.uint8:
            raise ValueError(f"expected uint8 image, got {image.dtype}")

        model = self._ensure_loaded()
        results = model.predict(
            source=image,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
        )
        if not results:
            return None

        result = results[0]
        masks = getattr(result, "masks", None)
        boxes = getattr(result, "boxes", None)
        if masks is None or boxes is None or len(masks.data) == 0:
            return None

        # Pick highest-confidence detection.
        confs = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else np.asarray(boxes.conf)
        best = int(np.argmax(confs))
        confidence = float(confs[best])

        mask_tensor = masks.data[best]
        mask_np = (
            mask_tensor.cpu().numpy() if hasattr(mask_tensor, "cpu") else np.asarray(mask_tensor)
        )
        # YOLOv11-seg masks come at the model's input resolution; resize to the
        # original image and binarize.
        h, w = image.shape[:2]
        mask = (cv2.resize(mask_np.astype(np.float32), (w, h)) > 0.5).astype(np.uint8) * 255

        return _detection_from_mask(mask, confidence)


def _detection_from_mask(mask: NDArray[np.uint8], confidence: float) -> DetectedCard | None:
    """Recover quad + bbox from a binary segmentation mask via rotated-rect fit."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    biggest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(biggest) < 100:
        return None

    rotated = cv2.minAreaRect(biggest)
    box = cv2.boxPoints(rotated)  # 4x2 float32, arbitrary order
    quad = _order_quad(box.astype(np.float32))

    x, y, w, h = cv2.boundingRect(biggest)
    return DetectedCard(
        quad=quad,
        bbox=BBox(int(x), int(y), int(w), int(h)),
        confidence=confidence,
        mask=mask,
        metadata={"backend": "yolo"},
    )


def _order_quad(pts: NDArray[np.float32]) -> NDArray[np.float32]:
    """Order 4 points as TL, TR, BR, BL by sum/diff of coords."""
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).flatten()
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    return np.stack([tl, tr, br, bl]).astype(np.float32)
