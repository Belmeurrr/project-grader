"""Build a tiny labeled detection dataset from synthesized fixtures.

The YOLO-seg trainer at `ml/training/trainers/detection.py` wants an
ultralytics-format manifest:

    path: <abs root of dataset>
    train: images/train
    val:   images/val
    nc: 1
    names:
      0: card

Labels live under `<root>/labels/{train,val}/<stem>.txt` with one line
per object in normalized YOLO bbox format:

    <class_id> <cx> <cy> <w> <h>     (all floats in [0, 1])

We don't have real PSA-labeled data yet. To exercise the manifest
pipeline (and one day the trainer itself) we render `card_in_scene`
fixtures whose placement parameters we already know — that gives us
free, perfect ground truth.

Bbox math (axis-aligned, tight around the warped quad):

`card_in_scene` places a `target_w × target_h` resized card at scene
center `(cx, cy)`. With non-zero `perspective_skew_px`, it warps the
card to the destination quad

    TL = (cx,             cy + skew)
    TR = (cx + target_w,  cy)
    BR = (cx + target_w,  cy + target_h)
    BL = (cx,             cy + target_h - skew)

The tight axis-aligned rectangle around those four points is

    x_min = cx,                x_max = cx + target_w
    y_min = cy,                y_max = cy + target_h

(The skew moves *only* the left side's y-coords inward, so the
extrema match the non-skewed case in y as well.)

We therefore reuse one bbox formula for skewed and non-skewed scenes;
a unit test in `tests/test_detection_manifest.py` confirms the bbox
tightly contains the rendered card pixels for both cases.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import yaml

# Local import — fixtures live alongside the tests but are pure-numpy +
# pure-cv2 helpers so importing them at runtime is fine.
from tests.fixtures import CARD_H, CARD_W, card_in_scene, synth_card_with_pattern


# Manifest constants -------------------------------------------------

CLASS_ID_CARD = 0
CLASS_NAME_CARD = "card"
NUM_CLASSES = 1


@dataclass(frozen=True)
class DetectionManifestStats:
    """Outcome of one manifest build, useful for logging + smoke tests."""

    out_dir: Path
    n_train: int
    n_val: int
    manifest_path: Path


# Bbox math ----------------------------------------------------------


def card_bbox_xyxy(
    scene_w: int,
    scene_h: int,
    fill: float,
    perspective_skew_px: int = 0,
) -> tuple[int, int, int, int]:
    """Compute the axis-aligned tight bbox of a `card_in_scene` rendering.

    Returns `(x_min, y_min, x_max, y_max)` in pixel coords. Mirrors the
    exact placement math inside `tests.fixtures.card_in_scene` — keep
    them in lockstep if that function changes.
    """
    aspect_h_over_w = CARD_H / CARD_W
    target_w = int((fill * scene_w * scene_h / aspect_h_over_w) ** 0.5)
    target_h = int(target_w * aspect_h_over_w)
    cx = (scene_w - target_w) // 2
    cy = (scene_h - target_h) // 2

    # See the module docstring: with the asymmetric left-side-only skew
    # in card_in_scene, the axis-aligned bbox is the same as the
    # non-skewed case.
    x_min = cx
    y_min = cy
    x_max = cx + target_w
    y_max = cy + target_h
    # Defensive — keep the perspective_skew_px in the signature even
    # though it doesn't currently change the bbox; tests assert this
    # property explicitly so a future fixture change won't go silently
    # wrong.
    _ = perspective_skew_px
    return x_min, y_min, x_max, y_max


def card_bbox_yolo(
    scene_w: int,
    scene_h: int,
    fill: float,
    perspective_skew_px: int = 0,
) -> tuple[float, float, float, float]:
    """YOLO-normalized bbox: (cx, cy, w, h) all in [0, 1]."""
    x_min, y_min, x_max, y_max = card_bbox_xyxy(
        scene_w, scene_h, fill, perspective_skew_px
    )
    bw = (x_max - x_min) / scene_w
    bh = (y_max - y_min) / scene_h
    bcx = (x_min + x_max) / 2.0 / scene_w
    bcy = (y_min + y_max) / 2.0 / scene_h
    return bcx, bcy, bw, bh


# Sample generation --------------------------------------------------


def _sample_params(rng: np.random.Generator) -> dict:
    """Pick fill / skew / card-pattern seed for one synthetic scene.

    Ranges chosen to span what the heuristic detector + downstream
    quality pipeline already test against (fill 0.20–0.75, skew up to
    ~80 px). Stays well clear of pathological extremes."""
    return {
        "fill": float(rng.uniform(0.20, 0.65)),
        "perspective_skew_px": int(rng.integers(0, 80)),
        "card_seed": int(rng.integers(0, 1_000_000)),
    }


def _render_scene(
    scene_w: int,
    scene_h: int,
    fill: float,
    perspective_skew_px: int,
    card_seed: int,
) -> np.ndarray:
    card = synth_card_with_pattern(seed=card_seed)
    return card_in_scene(
        card=card,
        scene_w=scene_w,
        scene_h=scene_h,
        fill=fill,
        perspective_skew_px=perspective_skew_px,
    )


def _write_yolo_label(
    label_path: Path,
    scene_w: int,
    scene_h: int,
    fill: float,
    perspective_skew_px: int,
) -> None:
    bcx, bcy, bw, bh = card_bbox_yolo(scene_w, scene_h, fill, perspective_skew_px)
    # Validate before writing — clipping silently would mask bugs.
    for v in (bcx, bcy, bw, bh):
        if not (0.0 <= v <= 1.0):
            raise ValueError(
                f"bbox value out of [0,1]: cx={bcx} cy={bcy} w={bw} h={bh} "
                f"(scene={scene_w}x{scene_h}, fill={fill}, skew={perspective_skew_px})"
            )
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text(
        f"{CLASS_ID_CARD} {bcx:.6f} {bcy:.6f} {bw:.6f} {bh:.6f}\n",
        encoding="utf-8",
    )


def _write_scene(
    image_path: Path,
    scene: np.ndarray,
    jpeg_quality: int = 92,
) -> None:
    image_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(
        str(image_path),
        scene,
        [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)],
    )
    if not ok:
        raise RuntimeError(f"failed to write image: {image_path}")


def _write_manifest_yaml(
    out_dir: Path,
    train_rel: str,
    val_rel: str,
) -> Path:
    """Write the ultralytics-format dataset YAML and return its path.

    `path` is absolute so the trainer can resolve it from any cwd.
    `train` / `val` are relative-to-`path` per ultralytics convention."""
    manifest = {
        "path": str(out_dir.resolve()),
        "train": train_rel,
        "val": val_rel,
        "nc": NUM_CLASSES,
        "names": {CLASS_ID_CARD: CLASS_NAME_CARD},
    }
    manifest_path = out_dir / "dataset.yaml"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        yaml.safe_dump(manifest, sort_keys=False),
        encoding="utf-8",
    )
    return manifest_path


# Public API ---------------------------------------------------------


def build_detection_manifest(
    out_dir: Path | str,
    n_train: int = 80,
    n_val: int = 20,
    seed: int = 0,
    scene_w: int = 1500,
    scene_h: int = 2000,
    splits: Sequence[str] = ("train", "val"),
) -> DetectionManifestStats:
    """Synthesize a labeled detection dataset rooted at `out_dir`.

    Layout written:
        <out_dir>/dataset.yaml
        <out_dir>/images/train/scene_<i>.jpg
        <out_dir>/images/val/scene_<i>.jpg
        <out_dir>/labels/train/scene_<i>.txt
        <out_dir>/labels/val/scene_<i>.txt
    """
    if n_train < 1 or n_val < 1:
        raise ValueError(f"n_train and n_val must be >= 1, got {n_train}/{n_val}")
    if set(splits) != {"train", "val"}:
        raise ValueError(f"only train/val splits supported, got {splits!r}")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)

    counts = {"train": n_train, "val": n_val}
    for split in ("train", "val"):
        for i in range(counts[split]):
            params = _sample_params(rng)
            scene = _render_scene(
                scene_w=scene_w,
                scene_h=scene_h,
                fill=params["fill"],
                perspective_skew_px=params["perspective_skew_px"],
                card_seed=params["card_seed"],
            )
            stem = f"scene_{i:05d}"
            image_path = out_dir / "images" / split / f"{stem}.jpg"
            label_path = out_dir / "labels" / split / f"{stem}.txt"
            _write_scene(image_path, scene)
            _write_yolo_label(
                label_path,
                scene_w=scene_w,
                scene_h=scene_h,
                fill=params["fill"],
                perspective_skew_px=params["perspective_skew_px"],
            )

    manifest_path = _write_manifest_yaml(
        out_dir,
        train_rel="images/train",
        val_rel="images/val",
    )
    return DetectionManifestStats(
        out_dir=out_dir,
        n_train=n_train,
        n_val=n_val,
        manifest_path=manifest_path,
    )
