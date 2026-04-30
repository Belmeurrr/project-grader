"""Synthesize a labeled detection dataset for the YOLO-seg trainer.

One-shot driver — re-run when you want a fresh synthetic dataset, or
swap in real PSA-labeled images by writing them straight into
<out_dir>/{images,labels}/{train,val}/ and pointing the trainer at
<out_dir>/dataset.yaml.

Until real labeled data lands, this script renders `card_in_scene`
fixtures whose placement params we already know — that gives us free,
perfect ground truth bboxes to exercise the manifest format and
trainer plumbing.

Usage:
    uv run python -m scripts.build_detection_manifest \
        --out-dir ./datasets/detection_synth \
        --n-train 80 --n-val 20 --seed 0

Outputs (under --out-dir):
    dataset.yaml                    ultralytics manifest
    images/train/scene_*.jpg        rendered scenes
    images/val/scene_*.jpg
    labels/train/scene_*.txt        YOLO-format normalized bboxes
    labels/val/scene_*.txt

Optional:
    --also-write-canonical          mirror dataset.yaml to
                                    ml/data/catalogs/detection_dataset.yaml
                                    (the path the trainer config
                                    reads by default).

Exit codes:
    0  success
    1  unexpected exception (I/O, bbox math, etc.)
    2  bad CLI invocation
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

# Path tweak: this script is expected to run with the project's `ml/`
# dir on sys.path (via `python -m scripts.build_detection_manifest`
# from inside `ml/`). Defensive fallback for direct invocation.
_ML_ROOT = Path(__file__).resolve().parents[1]
if str(_ML_ROOT) not in sys.path:
    sys.path.insert(0, str(_ML_ROOT))

from data.catalogs import build_detection_manifest  # noqa: E402

_logger = logging.getLogger("build_detection_manifest")

_CANONICAL_MANIFEST = _ML_ROOT / "data" / "catalogs" / "detection_dataset.yaml"


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="build_detection_manifest",
        description="Synthesize a labeled detection dataset for the YOLO-seg trainer.",
    )
    p.add_argument(
        "--out-dir",
        required=True,
        help="Destination directory. Will be created if missing.",
    )
    p.add_argument(
        "--n-train",
        type=int,
        default=80,
        help="Number of training scenes to render. Default: 80.",
    )
    p.add_argument(
        "--n-val",
        type=int,
        default=20,
        help="Number of validation scenes to render. Default: 20.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed for placement parameters. Default: 0.",
    )
    p.add_argument(
        "--scene-w",
        type=int,
        default=1500,
        help="Rendered scene width in pixels. Default: 1500.",
    )
    p.add_argument(
        "--scene-h",
        type=int,
        default=2000,
        help="Rendered scene height in pixels. Default: 2000.",
    )
    p.add_argument(
        "--also-write-canonical",
        action="store_true",
        help=(
            "After building, copy the generated dataset.yaml to "
            "ml/data/catalogs/detection_dataset.yaml so the trainer's "
            "default config picks it up without env overrides."
        ),
    )
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    _logger.info(
        "building detection manifest out_dir=%s n_train=%d n_val=%d seed=%d",
        out_dir,
        args.n_train,
        args.n_val,
        args.seed,
    )

    try:
        stats = build_detection_manifest(
            out_dir=out_dir,
            n_train=args.n_train,
            n_val=args.n_val,
            seed=args.seed,
            scene_w=args.scene_w,
            scene_h=args.scene_h,
        )
    except Exception as e:  # noqa: BLE001 — log and exit 1 on anything
        _logger.exception("manifest build failed: %s", e)
        return 1

    _logger.info(
        "wrote manifest %s (train=%d, val=%d)",
        stats.manifest_path,
        stats.n_train,
        stats.n_val,
    )

    if args.also_write_canonical:
        try:
            shutil.copyfile(stats.manifest_path, _CANONICAL_MANIFEST)
            _logger.info(
                "mirrored manifest to canonical location %s",
                _CANONICAL_MANIFEST,
            )
        except OSError as e:
            _logger.exception("failed to mirror to canonical location: %s", e)
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
