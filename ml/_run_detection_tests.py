"""Inline test runner for detection + training-config (pytest not installed locally).

Removed once `uv sync` provides pytest. Mirrors the assertions in the formal
test files."""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

sys.path.insert(0, ".")

import numpy as np
import yaml

from pipelines.detection import (
    CANONICAL_HEIGHT,
    CANONICAL_WIDTH,
    DetectedCard,
    HeuristicDetector,
    YoloDetector,
    detect_card,
    dewarp_to_canonical,
    get_detector,
    quad_irregularity,
)
from tests.fixtures import card_in_scene


ran, passed, failed = 0, 0, 0


def case(name, fn):
    global ran, passed, failed
    ran += 1
    try:
        fn()
        print(f"PASS {name}")
        passed += 1
    except AssertionError as e:
        print(f"FAIL {name}: AssertionError: {e}")
        failed += 1
    except Exception as e:
        print(f"FAIL {name}: {type(e).__name__}: {e}")
        traceback.print_exc()
        failed += 1


# quad_irregularity

def i1():
    quad = np.float32([[0, 0], [100, 0], [100, 140], [0, 140]])
    assert abs(quad_irregularity(quad)) < 1e-6
case("irregularity_zero_for_perfect_rect", i1)

def i2():
    flat = np.float32([[0, 0], [100, 0], [100, 140], [0, 140]])
    skewed = np.float32([[10, 0], [100, 0], [100, 140], [0, 140]])
    assert quad_irregularity(skewed) > quad_irregularity(flat)
case("irregularity_grows_with_skew", i2)

def i3():
    nearly_collapsed = np.float32([[0, 0], [100, 0], [50, 1], [50, 1]])
    v = quad_irregularity(nearly_collapsed)
    assert v >= 0.5, f"got {v}"
case("irregularity_high_for_collapsed_quad", i3)

def i4():
    degenerate = np.float32([[0, 0], [100, 0], [0, 0], [0, 100]])
    v = quad_irregularity(degenerate)
    assert 0.0 <= v <= 1.0
case("irregularity_clipped_to_one", i4)

def i5():
    try:
        quad_irregularity(np.zeros((3, 2), dtype=np.float32))
    except ValueError:
        return
    raise AssertionError("expected ValueError")
case("irregularity_rejects_wrong_shape", i5)


# dewarp

def _scene_with_known_quad():
    img = card_in_scene(fill=0.55)
    h, w = img.shape[:2]
    aspect = 1050 / 750
    target_w = int((0.55 * w * h / aspect) ** 0.5)
    target_h = int(target_w * aspect)
    cx = (w - target_w) // 2
    cy = (h - target_h) // 2
    quad = np.float32([
        [cx, cy],
        [cx + target_w, cy],
        [cx + target_w, cy + target_h],
        [cx, cy + target_h],
    ])
    return img, quad


def d1():
    img, quad = _scene_with_known_quad()
    r = dewarp_to_canonical(img, quad)
    assert r.canonical.shape == (CANONICAL_HEIGHT, CANONICAL_WIDTH, 3)
    assert r.canonical.dtype == np.uint8
case("dewarp_canonical_dimensions", d1)

def d2():
    img, quad = _scene_with_known_quad()
    r = dewarp_to_canonical(img, quad)
    assert r.homography.shape == (3, 3)
case("dewarp_homography_shape", d2)

def d3():
    img, quad = _scene_with_known_quad()
    r = dewarp_to_canonical(img, quad)
    cy, cx = CANONICAL_HEIGHT // 2, CANONICAL_WIDTH // 2
    px = r.canonical[cy, cx]
    # synth_card image_color (200,50,50) is BGR
    assert px[0] >= 150 and px[1] < 100 and px[2] < 100, f"center pixel {tuple(px)}"
case("dewarp_recovers_inner_image", d3)

def d4():
    img, quad = _scene_with_known_quad()
    r = dewarp_to_canonical(img, quad)
    assert r.irregularity < 0.05
case("dewarp_irregularity_low_for_overhead", d4)

def d5():
    img, quad = _scene_with_known_quad()
    try:
        dewarp_to_canonical(img.astype(np.float32), quad)
    except ValueError as e:
        assert "uint8" in str(e)
        return
    raise AssertionError("expected ValueError")
case("dewarp_rejects_non_uint8", d5)

def d6():
    img, quad = _scene_with_known_quad()
    gray = img[:, :, 0]
    try:
        dewarp_to_canonical(gray, quad)
    except ValueError as e:
        assert "3-channel" in str(e)
        return
    raise AssertionError("expected ValueError")
case("dewarp_rejects_grayscale", d6)

def d7():
    img, quad = _scene_with_known_quad()
    r = dewarp_to_canonical(img, quad, out_w=400, out_h=560)
    assert r.canonical.shape == (560, 400, 3)
case("dewarp_custom_dimensions", d7)


# heuristic detector

def h1():
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
case("heuristic_detects_card", h1)

def h2():
    blank = np.full((1000, 1000, 3), 35, dtype=np.uint8)
    assert HeuristicDetector().detect(blank) is None
case("heuristic_none_on_blank", h2)

def h3():
    img = card_in_scene(fill=0.05)
    assert HeuristicDetector(min_fill=0.20).detect(img) is None
case("heuristic_min_fill_filters_tiny", h3)

def h4():
    img = card_in_scene(fill=0.90)
    assert HeuristicDetector(max_fill=0.50).detect(img) is None
case("heuristic_max_fill_filters_close", h4)

def h5():
    img = card_in_scene().astype(np.float32)
    try:
        HeuristicDetector().detect(img)
    except ValueError as e:
        assert "uint8" in str(e)
        return
    raise AssertionError("expected ValueError")
case("heuristic_rejects_non_uint8", h5)

def h6():
    img = card_in_scene(fill=0.55)
    det = HeuristicDetector().detect(img)
    assert det is not None
    tl, tr, br, bl = det.quad
    assert tl[0] < tr[0] and tl[1] < bl[1]
    assert br[0] > bl[0] and br[1] > tr[1]
case("heuristic_quad_ordered_TL_TR_BR_BL", h6)


# yolo (no weights)

def y1():
    d = YoloDetector(weights_path="/nonexistent/weights.pt")
    assert d._model is None
case("yolo_construction_does_not_load", y1)

def y2():
    d = YoloDetector(weights_path="/nonexistent/weights.pt")
    img = card_in_scene(fill=0.55)
    try:
        d.detect(img)
    except RuntimeError:
        return
    raise AssertionError("expected RuntimeError")
case("yolo_raises_on_missing_weights", y2)


# registry

def _set_env(**kwargs):
    saved = {}
    for k, v in kwargs.items():
        saved[k] = os.environ.get(k)
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    return saved


def _restore_env(saved):
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def r1():
    saved = _set_env(GRADER_DETECTOR=None, GRADER_YOLO_WEIGHTS=None)
    try:
        det = get_detector()
        assert isinstance(det, HeuristicDetector)
    finally:
        _restore_env(saved)
case("registry_default_is_heuristic", r1)

def r2():
    saved = _set_env(GRADER_DETECTOR="yolo", GRADER_YOLO_WEIGHTS=None)
    try:
        det = get_detector()
        assert isinstance(det, HeuristicDetector)
    finally:
        _restore_env(saved)
case("registry_falls_back_when_yolo_missing_env", r2)

def r3():
    saved = _set_env(GRADER_DETECTOR="yolo", GRADER_YOLO_WEIGHTS="/nonexistent/weights.pt")
    try:
        det = get_detector()
        assert det is not None
    finally:
        _restore_env(saved)
case("registry_returns_some_detector_with_bad_path", r3)

def r4():
    saved = _set_env(GRADER_DETECTOR=None, GRADER_YOLO_WEIGHTS=None)
    try:
        result = detect_card(card_in_scene(fill=0.55))
        assert result is not None
        assert result.metadata["backend"] == "heuristic"
    finally:
        _restore_env(saved)
case("detect_card_helper_uses_registry", r4)


# training config

def _cfg():
    return yaml.safe_load(Path("training/configs/detection.yaml").read_text())

def tc1():
    assert isinstance(_cfg(), dict)
case("train_config_parses", tc1)

def tc2():
    cfg = _cfg()
    for k in ("dataset", "model", "train", "augment", "mlflow", "output"):
        assert k in cfg
case("train_config_top_level_keys", tc2)

def tc3():
    cfg = _cfg()
    assert cfg["dataset"]["num_classes"] == 1
    assert cfg["dataset"]["class_names"] == ["card"]
case("train_config_single_class", tc3)

def tc4():
    cfg = _cfg()
    assert cfg["augment"]["flipud"] == 0.0
    assert cfg["augment"]["fliplr"] == 0.0
case("train_config_no_flips", tc4)

def tc5():
    cfg = _cfg()
    assert 512 <= cfg["model"]["image_size"] <= 2048
case("train_config_image_size_sane", tc5)

def tc6():
    cfg = _cfg()
    assert cfg["model"]["base"].endswith("-seg.pt")
    assert "yolo11" in cfg["model"]["base"]
case("train_config_base_is_yolov11_seg", tc6)

def tc7():
    cfg = _cfg()
    t = cfg["train"]
    assert 1 <= t["epochs"] <= 1000
    assert 1 <= t["batch_size"] <= 256
    assert 1e-6 <= t["lr0"] <= 1e-1
    assert 0.0 <= t["lrf"] <= 1.0
    assert 0.0 <= t["weight_decay"] <= 1e-2
case("train_config_hyperparams_sane", tc7)


print()
print(f"{passed}/{ran} passed, {failed} failed")
sys.exit(0 if failed == 0 else 1)
