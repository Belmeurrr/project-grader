"""Detector interface, heuristic implementation, and registry-fallback tests.

The YOLO detector cannot be exercised here without a trained .pt file; we
test that it (a) is import-safe, (b) raises a useful error on first inference
without weights, and (c) the registry falls back to the heuristic when YOLO
weights are unavailable. End-to-end YOLO accuracy lives in evaluation/, not
unit tests."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager

import numpy as np
import pytest

from pipelines.detection import (
    DetectedCard,
    HeuristicDetector,
    YoloDetector,
    detect_card,
    get_detector,
)
from tests.fixtures import card_in_scene


@contextmanager
def _env(**kwargs: str) -> Iterator[None]:
    prior = {k: os.environ.get(k) for k in kwargs}
    try:
        for k, v in kwargs.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        yield
    finally:
        for k, v in prior.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


# -----------------------------
# Heuristic
# -----------------------------


def test_heuristic_detects_card_in_scene() -> None:
    img = card_in_scene(fill=0.55)
    det = HeuristicDetector().detect(img)
    assert det is not None
    assert isinstance(det, DetectedCard)
    assert det.quad.shape == (4, 2)
    assert det.confidence > 0
    assert det.mask is not None
    assert det.mask.shape == img.shape[:2]
    assert det.mask.dtype == np.uint8
    assert det.metadata["backend"] == "heuristic"


def test_heuristic_returns_none_on_blank() -> None:
    blank = np.full((1000, 1000, 3), 35, dtype=np.uint8)
    assert HeuristicDetector().detect(blank) is None


def test_heuristic_min_fill_filters_tiny_cards() -> None:
    img = card_in_scene(fill=0.05)  # very small
    det = HeuristicDetector(min_fill=0.20).detect(img)
    assert det is None


def test_heuristic_max_fill_filters_close_cards() -> None:
    img = card_in_scene(fill=0.90)
    det = HeuristicDetector(max_fill=0.50).detect(img)
    assert det is None


def test_heuristic_rejects_non_uint8() -> None:
    img = card_in_scene().astype(np.float32)
    with pytest.raises(ValueError, match="uint8"):
        HeuristicDetector().detect(img)


def test_heuristic_quad_is_ordered_TL_TR_BR_BL() -> None:
    img = card_in_scene(fill=0.55)
    det = HeuristicDetector().detect(img)
    assert det is not None
    tl, tr, br, bl = det.quad
    assert tl[0] < tr[0] and tl[1] < bl[1]   # TL is leftmost-topmost
    assert br[0] > bl[0] and br[1] > tr[1]   # BR is rightmost-bottommost


# -----------------------------
# YOLO (no weights — error path only)
# -----------------------------


def test_yolo_construction_does_not_load_weights() -> None:
    detector = YoloDetector(weights_path="/nonexistent/weights.pt")
    assert detector._model is None  # noqa: SLF001


def test_yolo_raises_runtime_error_on_missing_weights() -> None:
    detector = YoloDetector(weights_path="/nonexistent/weights.pt")
    img = card_in_scene(fill=0.55)
    with pytest.raises(RuntimeError):
        detector.detect(img)


# -----------------------------
# Registry
# -----------------------------


def test_registry_default_is_heuristic() -> None:
    with _env(GRADER_DETECTOR=None, GRADER_YOLO_WEIGHTS=None):
        det = get_detector()
        assert isinstance(det, HeuristicDetector)


def test_registry_falls_back_when_yolo_weights_missing_env() -> None:
    with _env(GRADER_DETECTOR="yolo", GRADER_YOLO_WEIGHTS=None):
        det = get_detector()
        # No weights env var → heuristic fallback (no loud failure).
        assert isinstance(det, HeuristicDetector)


def test_registry_falls_back_when_yolo_weights_path_unloadable() -> None:
    with _env(GRADER_DETECTOR="yolo", GRADER_YOLO_WEIGHTS="/nonexistent/weights.pt"):
        det = get_detector()
        # YoloDetector construction is cheap and won't throw — only inference
        # would. The registry contract is to return *some* detector.
        assert det is not None


def test_detect_card_helper_uses_registry() -> None:
    with _env(GRADER_DETECTOR=None, GRADER_YOLO_WEIGHTS=None):
        result = detect_card(card_in_scene(fill=0.55))
        assert result is not None
        assert result.metadata["backend"] == "heuristic"
