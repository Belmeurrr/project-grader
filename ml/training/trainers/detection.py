"""YOLOv11-seg fine-tuning entrypoint for the card detection model.

Reads a Hydra config (default: ml/training/configs/detection.yaml), runs
ultralytics training, and logs to MLflow. Designed so a fresh team member
can train a new detector with one command once a dataset is available:

    uv run python -m training.trainers.detection

Dataset format: ultralytics standard YAML pointing to image and label dirs
with YOLO-format polygon labels. See `ml/data/catalogs/detection_dataset.yaml`.
That manifest is built by `ml/data/ingestion/`.

This module is intentionally a thin orchestrator — all training math is
inside ultralytics. We add MLflow tracking and config-driven defaults on top.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import hydra
import mlflow
from omegaconf import DictConfig, OmegaConf


def _validate_config(cfg: DictConfig) -> None:
    """Cheap config validation. Failing fast here beats failing 10 minutes
    into a training run because of a missing field."""
    required = [
        "dataset.manifest",
        "dataset.num_classes",
        "model.base",
        "model.image_size",
        "train.epochs",
        "train.batch_size",
    ]
    for path in required:
        node: Any = cfg
        for part in path.split("."):
            if not hasattr(node, part) and part not in node:
                raise ValueError(f"missing required config: {path}")
            node = getattr(node, part) if hasattr(node, part) else node[part]


def _train(cfg: DictConfig) -> dict[str, float]:
    """Run a YOLO training job and return key validation metrics."""
    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise RuntimeError(
            "ultralytics not installed; run `uv sync --extra training` from ml/"
        ) from e

    manifest = Path(cfg.dataset.manifest)
    if not manifest.exists():
        raise FileNotFoundError(
            f"dataset manifest not found: {manifest} — run ml/data/ingestion first"
        )

    model = YOLO(cfg.model.base)
    train_kwargs = {
        "data": str(manifest),
        "epochs": int(cfg.train.epochs),
        "batch": int(cfg.train.batch_size),
        "imgsz": int(cfg.model.image_size),
        "workers": int(cfg.train.workers),
        "patience": int(cfg.train.patience),
        "optimizer": str(cfg.train.optimizer),
        "lr0": float(cfg.train.lr0),
        "lrf": float(cfg.train.lrf),
        "weight_decay": float(cfg.train.weight_decay),
        "cos_lr": bool(cfg.train.cos_lr),
        "close_mosaic": int(cfg.train.close_mosaic),
        "amp": bool(cfg.train.amp),
        "device": cfg.train.device,
        "project": str(cfg.output.project),
        "name": str(cfg.output.name),
        # Augmentation
        "hsv_h": float(cfg.augment.hsv_h),
        "hsv_s": float(cfg.augment.hsv_s),
        "hsv_v": float(cfg.augment.hsv_v),
        "degrees": float(cfg.augment.degrees),
        "translate": float(cfg.augment.translate),
        "scale": float(cfg.augment.scale),
        "shear": float(cfg.augment.shear),
        "perspective": float(cfg.augment.perspective),
        "flipud": float(cfg.augment.flipud),
        "fliplr": float(cfg.augment.fliplr),
        "mosaic": float(cfg.augment.mosaic),
        "mixup": float(cfg.augment.mixup),
    }
    result = model.train(**train_kwargs)

    # Ultralytics returns a Result-like object; pull out the standard metrics.
    metrics_dict = getattr(result, "results_dict", None) or {}
    return {str(k): float(v) for k, v in metrics_dict.items() if isinstance(v, (int, float))}


@hydra.main(version_base=None, config_path="../configs", config_name="detection")
def main(cfg: DictConfig) -> None:
    _validate_config(cfg)

    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment)

    with mlflow.start_run(run_name=cfg.mlflow.run_name):
        # Log full resolved config for reproducibility.
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        mlflow.log_dict(cfg_dict, "config.yaml")
        mlflow.log_params(_flatten(cfg_dict))

        metrics = _train(cfg)
        for k, v in metrics.items():
            mlflow.log_metric(k.replace("/", "_"), v)

        out_dir = Path(cfg.output.project) / cfg.output.name / "weights"
        if (out_dir / "best.pt").exists():
            mlflow.log_artifact(str(out_dir / "best.pt"), artifact_path="weights")


def _flatten(d: object, prefix: str = "") -> dict[str, object]:
    out: dict[str, object] = {}
    if isinstance(d, dict):
        for k, v in d.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            out.update(_flatten(v, key))
    else:
        out[prefix] = d
    return out


if __name__ == "__main__":
    # Hydra parses sys.argv directly via the @hydra.main decorator.
    os.environ.setdefault("HYDRA_FULL_ERROR", "1")
    main()  # type: ignore[call-arg]
