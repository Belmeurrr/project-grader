"""Tests for the synthetic detection manifest builder.

We can't run YOLO training in unit tests — that needs a GPU and minutes
of setup. What we can test is everything around it:

  * bbox math for `card_in_scene` is correct (hand-checked + pixel count
    cross-checks for both skewed and non-skewed scenes),
  * a tiny manifest writes the expected directory tree,
  * label files are valid YOLO format (1 class id, 4 floats in [0, 1]),
  * the dataset.yaml has the keys the trainer's config validator
    expects (`path`, `train`, `val`, `nc`, `names`) and `nc == 1`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

# Ensure ml/ root is on sys.path when running this test directly.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.catalogs.build_detection_manifest import (
    CLASS_ID_CARD,
    CLASS_NAME_CARD,
    NUM_CLASSES,
    build_detection_manifest,
    card_bbox_xyxy,
    card_bbox_yolo,
)
from tests.fixtures import card_in_scene


# ---------------------------------------------------------------------
# Bbox math
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "scene_w,scene_h,fill,expected_target_w,expected_target_h",
    [
        # target_w = sqrt(fill * W * H / aspect), target_h = target_w * aspect
        # aspect = 1050/750 = 1.4
        # 1500*2000=3_000_000; 0.55*3M/1.4 = 1_178_571.4 → sqrt ≈ 1085.62 → 1085
        # 1085 * 1.4 = 1519
        (1500, 2000, 0.55, 1085, 1519),
        # 0.20: 0.20*3M/1.4 = 428_571.4 → sqrt ≈ 654.65 → 654 ; 654*1.4=915.6→915
        (1500, 2000, 0.20, 654, 915),
    ],
)
def test_bbox_xyxy_known_geometry(
    scene_w, scene_h, fill, expected_target_w, expected_target_h
):
    x_min, y_min, x_max, y_max = card_bbox_xyxy(scene_w, scene_h, fill)
    assert x_max - x_min == expected_target_w
    assert y_max - y_min == expected_target_h
    # centered placement
    assert x_min == (scene_w - expected_target_w) // 2
    assert y_min == (scene_h - expected_target_h) // 2


def test_bbox_yolo_normalization():
    bcx, bcy, bw, bh = card_bbox_yolo(1500, 2000, 0.55)
    # centered → cx, cy ≈ 0.5
    assert abs(bcx - 0.5) < 1e-3
    assert abs(bcy - 0.5) < 1e-3
    assert 0.0 < bw <= 1.0
    assert 0.0 < bh <= 1.0


def test_bbox_yolo_all_in_unit_interval():
    """Across a wide span of (fill, skew) the normalized bbox stays in [0,1]."""
    for fill in (0.10, 0.25, 0.45, 0.65):
        for skew in (0, 25, 80):
            bcx, bcy, bw, bh = card_bbox_yolo(1500, 2000, fill, skew)
            for v in (bcx, bcy, bw, bh):
                assert 0.0 <= v <= 1.0


def _card_pixel_extent(scene: np.ndarray, bg_color=(35, 35, 35)) -> tuple[int, int, int, int]:
    """Find the tight axis-aligned bbox of non-background pixels.

    Returns (x_min, y_min, x_max, y_max) in pixel coords. `card_in_scene`
    uses bg_color (35, 35, 35) — anything that differs from that is a
    card pixel."""
    bg = np.asarray(bg_color, dtype=np.int16)
    diff = np.abs(scene.astype(np.int16) - bg).sum(axis=2)
    mask = diff > 5  # tiny tolerance for jpeg + interp slop on edges
    ys, xs = np.where(mask)
    assert xs.size > 0, "scene has no non-background pixels"
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def test_bbox_tightly_contains_unskewed_card():
    """For non-skewed scenes the bbox should match the rendered pixels exactly
    (modulo rounding from int target_w/target_h)."""
    scene = card_in_scene(fill=0.55, perspective_skew_px=0)
    h, w = scene.shape[:2]
    bx0, by0, bx1, by1 = card_bbox_xyxy(w, h, 0.55, 0)
    px0, py0, px1, py1 = _card_pixel_extent(scene)

    # bbox should fully contain the card pixels...
    assert bx0 <= px0
    assert by0 <= py0
    assert bx1 >= px1
    assert by1 >= py1
    # ...and be tight (within a couple of px of the actual extent).
    assert px0 - bx0 <= 2 and bx1 - px1 <= 2
    assert py0 - by0 <= 2 and by1 - py1 <= 2


def test_bbox_tightly_contains_skewed_card():
    """For skewed scenes the bbox should still tightly contain the warped card."""
    scene = card_in_scene(fill=0.55, perspective_skew_px=60)
    h, w = scene.shape[:2]
    bx0, by0, bx1, by1 = card_bbox_xyxy(w, h, 0.55, 60)
    px0, py0, px1, py1 = _card_pixel_extent(scene)

    assert bx0 <= px0
    assert by0 <= py0
    assert bx1 >= px1
    assert by1 >= py1
    # Tight: warpPerspective interpolation may bleed a few px outside
    # the geometric quad, so allow a small slack on the high side
    # and exactness on the low side.
    assert px0 - bx0 <= 3 and bx1 - px1 <= 3
    assert py0 - by0 <= 3 and by1 - py1 <= 3


# ---------------------------------------------------------------------
# Manifest build — files and YAML schema
# ---------------------------------------------------------------------


def test_build_writes_expected_tree(tmp_path: Path):
    stats = build_detection_manifest(
        out_dir=tmp_path,
        n_train=3,
        n_val=2,
        seed=42,
    )
    assert stats.out_dir == tmp_path
    assert stats.n_train == 3
    assert stats.n_val == 2
    assert stats.manifest_path == tmp_path / "dataset.yaml"
    assert stats.manifest_path.exists()

    train_imgs = sorted((tmp_path / "images" / "train").glob("*.jpg"))
    val_imgs = sorted((tmp_path / "images" / "val").glob("*.jpg"))
    train_lbls = sorted((tmp_path / "labels" / "train").glob("*.txt"))
    val_lbls = sorted((tmp_path / "labels" / "val").glob("*.txt"))
    assert len(train_imgs) == 3
    assert len(val_imgs) == 2
    assert len(train_lbls) == 3
    assert len(val_lbls) == 2

    # Stems pair up image ↔ label.
    assert [p.stem for p in train_imgs] == [p.stem for p in train_lbls]
    assert [p.stem for p in val_imgs] == [p.stem for p in val_lbls]


def test_label_files_valid_yolo_format(tmp_path: Path):
    build_detection_manifest(out_dir=tmp_path, n_train=4, n_val=2, seed=7)
    for split in ("train", "val"):
        for label_path in (tmp_path / "labels" / split).glob("*.txt"):
            text = label_path.read_text(encoding="utf-8").strip()
            # Single object per scene → one line.
            lines = text.splitlines()
            assert len(lines) == 1, f"{label_path} has {len(lines)} lines"
            parts = lines[0].split()
            assert len(parts) == 5, f"{label_path}: bad column count: {parts!r}"
            class_id = int(parts[0])
            cx, cy, bw, bh = (float(x) for x in parts[1:])
            assert class_id == CLASS_ID_CARD
            for v in (cx, cy, bw, bh):
                assert 0.0 <= v <= 1.0, f"{label_path}: value out of [0,1]: {v}"
            # bbox must have positive area; > 1% of the frame is a sane
            # floor for our `fill` range.
            assert bw > 0.01 and bh > 0.01


def test_manifest_yaml_schema(tmp_path: Path):
    build_detection_manifest(out_dir=tmp_path, n_train=2, n_val=2, seed=1)
    manifest = yaml.safe_load((tmp_path / "dataset.yaml").read_text(encoding="utf-8"))
    # Required ultralytics keys.
    for key in ("path", "train", "val", "nc", "names"):
        assert key in manifest, f"missing manifest key: {key}"
    assert manifest["nc"] == NUM_CLASSES
    # Names accepts either dict {0: name} or list[str] in ultralytics; we
    # always emit dict form for schema clarity.
    assert isinstance(manifest["names"], dict)
    assert manifest["names"][CLASS_ID_CARD] == CLASS_NAME_CARD
    # Path must be absolute so the trainer can resolve it from any cwd.
    assert Path(manifest["path"]).is_absolute()
    # train/val are dirs *relative to* `path`, per ultralytics convention.
    assert (tmp_path / manifest["train"]).is_dir()
    assert (tmp_path / manifest["val"]).is_dir()


def test_manifest_consistent_with_trainer_config(tmp_path: Path):
    """The manifest's `nc` must match `dataset.num_classes` in
    `training/configs/detection.yaml`. If they ever diverge (someone
    bumps to multi-class without regenerating the manifest), training
    will fail in confusing ways — surface it as a unit test."""
    build_detection_manifest(out_dir=tmp_path, n_train=1, n_val=1, seed=0)
    manifest = yaml.safe_load((tmp_path / "dataset.yaml").read_text(encoding="utf-8"))
    config_path = (
        Path(__file__).parent.parent / "training" / "configs" / "detection.yaml"
    )
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert manifest["nc"] == cfg["dataset"]["num_classes"]
    assert list(manifest["names"].values()) == cfg["dataset"]["class_names"]


def test_canonical_manifest_yaml_well_formed():
    """The committed-in placeholder at ml/data/catalogs/detection_dataset.yaml
    must parse and have the right schema even before the build script runs.
    `path` is intentionally a placeholder string the script overwrites."""
    canonical = (
        Path(__file__).parent.parent
        / "data"
        / "catalogs"
        / "detection_dataset.yaml"
    )
    manifest = yaml.safe_load(canonical.read_text(encoding="utf-8"))
    for key in ("path", "train", "val", "nc", "names"):
        assert key in manifest, f"canonical manifest missing key: {key}"
    assert manifest["nc"] == NUM_CLASSES
    assert manifest["names"][CLASS_ID_CARD] == CLASS_NAME_CARD
