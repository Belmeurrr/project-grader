"""Dataset catalogs / manifests consumed by trainers and evaluation.

Each subpath here is the canonical location a trainer reads from.
For the YOLO detection trainer (see `ml/training/trainers/detection.py`),
this package exposes a builder that synthesizes a tiny labeled dataset
from `tests/fixtures.card_in_scene` so the training pipeline can be
exercised end-to-end before real PSA-labeled data is available.

Real labels (PSA-bordered cards with hand-drawn polygons) will replace
the synthetic builder once they exist; the manifest schema (ultralytics
YAML) does not change.
"""

from data.catalogs.build_detection_manifest import (
    DetectionManifestStats,
    build_detection_manifest,
    card_bbox_xyxy,
    card_bbox_yolo,
)

__all__ = [
    "DetectionManifestStats",
    "build_detection_manifest",
    "card_bbox_xyxy",
    "card_bbox_yolo",
]
