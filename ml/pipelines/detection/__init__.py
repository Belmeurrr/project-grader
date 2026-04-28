"""Stage 1 + 2 of the grading pipeline.

Stage 1 — Card detection & segmentation
    Input: raw client-uploaded image (any size).
    Output: DetectedCard (quad, bbox, optional mask, confidence).
    Implementations: YOLOv11-seg (production) and a heuristic Canny+contour
    detector (used as a fallback before weights are trained, and at capture
    time for fast quality gating).

Stage 2 — Perspective correction
    Input: original image + quad from Stage 1.
    Output: dewarped 750x1050 canonical orthographic view, plus a quad
    irregularity score that downstream stages use to reject bent / steeply
    shot cards.

The two stages are deliberately decoupled — Stage 2 has no learning, so
training improvements to Stage 1 don't require any change downstream."""

from pipelines.detection.detector import (
    CardDetector,
    DetectedCard,
    HeuristicDetector,
    detect_card,
    get_detector,
)
from pipelines.detection.dewarp import (
    CANONICAL_HEIGHT,
    CANONICAL_WIDTH,
    DewarpResult,
    dewarp_to_canonical,
    quad_irregularity,
)
from pipelines.detection.yolo import YoloDetector

__all__ = [
    "CANONICAL_HEIGHT",
    "CANONICAL_WIDTH",
    "CardDetector",
    "DetectedCard",
    "DewarpResult",
    "HeuristicDetector",
    "YoloDetector",
    "detect_card",
    "dewarp_to_canonical",
    "get_detector",
    "quad_irregularity",
]
