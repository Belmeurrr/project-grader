"""EfficientNet-V2-S corner-grading regression trainer.

Reads `scraped.jsonl` produced by `ml/data/ingestion/psa_public_api`,
trains a backbone + Linear regression head against the per-corner
grade label, logs to MLflow, writes weights to `cfg.train.output_dir`.

Run (from ml/, with `uv sync --extra training`):

    uv run python -m training.trainers.corners

Override anything via Hydra:

    uv run python -m training.trainers.corners train.epochs=1 train.smoke_only=true

Architecture summary:
    backbone (timm EfficientNet-V2-S, ImageNet-21k pretrain) →
    global pool → Dropout → Linear(features → 256) → ReLU →
    Dropout → Linear(256 → 1)

Loss: MSELoss against the (clipped) per-corner grade.

Why a single-output regression and not ordinal regression / classification:
    For the skeleton, MSE-on-scalar gets us through the data-pipeline
    proof-of-life with the least moving parts. Once the corpus has
    enough records to actually train (≥1k), the swap-in candidates are
    (a) ordinal regression with K-1 sigmoid heads and (b) per-class
    classification with K classes. Both reuse the same backbone +
    Dataset; only the head + loss change. See the architecture plan.
"""

from __future__ import annotations

import logging
import math
import os
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

# Defensive sys.path: when this module is invoked as a script (rather
# than via `python -m`), ensure ml/ is importable. Mirrors detection.py.
_ML_ROOT = Path(__file__).resolve().parents[2]
if str(_ML_ROOT) not in sys.path:
    sys.path.insert(0, str(_ML_ROOT))


_logger = logging.getLogger("corners_trainer")


def _validate_config(cfg: DictConfig) -> None:
    """Cheap pre-flight. Failing here beats failing 10 minutes into
    a training run on a missing field."""
    required = [
        "dataset.jsonl",
        "dataset.min_samples",
        "dataset.val_fraction",
        "model.backbone",
        "model.image_size",
        "train.epochs",
        "train.batch_size",
        "train.lr",
        "train.output_dir",
    ]
    for path in required:
        node = cfg
        for part in path.split("."):
            if not hasattr(node, part) and part not in node:
                raise ValueError(f"missing required config: {path}")
            node = getattr(node, part) if hasattr(node, part) else node[part]


def _build_model(cfg: DictConfig):
    """timm backbone + 2-layer regression head. Imported lazily so the
    module is importable in stdlib-only environments for syntax checks."""
    import timm
    from torch import nn

    backbone = timm.create_model(
        cfg.model.backbone,
        pretrained=bool(cfg.model.pretrained),
        num_classes=0,  # disable timm's classifier — we add our own
        global_pool="avg",
    )
    in_features = backbone.num_features
    head = nn.Sequential(
        nn.Dropout(float(cfg.model.head_dropout)),
        nn.Linear(in_features, int(cfg.model.head_hidden)),
        nn.ReLU(inplace=True),
        nn.Dropout(float(cfg.model.head_dropout)),
        nn.Linear(int(cfg.model.head_hidden), 1),
    )
    return nn.Sequential(backbone, head)


def _device_from_cfg(name: str):
    import torch

    if name == "cpu":
        return torch.device("cpu")
    if name == "mps":
        if not torch.backends.mps.is_available():
            _logger.warning("requested mps but not available; falling back to cpu")
            return torch.device("cpu")
        return torch.device("mps")
    # numeric CUDA index or "cuda"
    try:
        return torch.device(f"cuda:{int(name)}" if str(name).isdigit() else name)
    except Exception:
        return torch.device("cpu")


def _train_one_epoch(model, loader, optimizer, loss_fn, device, log_every: int = 20):
    import torch

    model.train()
    total_loss = 0.0
    total_examples = 0
    for step, (x, y) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).unsqueeze(1)  # (B,) -> (B,1)
        optimizer.zero_grad()
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        bs = x.size(0)
        total_loss += float(loss.item()) * bs
        total_examples += bs
        if step % log_every == 0:
            _logger.info(
                "train step=%d loss=%.4f bs=%d", step, float(loss.item()), bs
            )
        if not math.isfinite(loss.item()):
            raise RuntimeError(f"non-finite loss at step {step}; aborting")
    avg = total_loss / max(total_examples, 1)
    return {"train_loss": avg, "train_examples": total_examples}


def _eval(model, loader, loss_fn, device):
    import torch

    model.eval()
    total_loss = 0.0
    total_examples = 0
    abs_errors: list[float] = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y_true = y.to(device, non_blocking=True).unsqueeze(1)
            pred = model(x)
            loss = loss_fn(pred, y_true)
            total_loss += float(loss.item()) * x.size(0)
            total_examples += x.size(0)
            abs_errors.extend((pred - y_true).abs().detach().cpu().numpy().tolist())
    if total_examples == 0:
        return {"val_loss": float("nan"), "val_mae": float("nan"), "val_examples": 0}
    return {
        "val_loss": total_loss / total_examples,
        "val_mae": sum(e[0] if isinstance(e, list) else float(e) for e in abs_errors) / total_examples,
        "val_examples": total_examples,
    }


def _run_training(cfg: DictConfig) -> dict[str, float]:
    """Build dataset + model + optimizer, run epochs, return final metrics."""
    import torch
    from torch import nn
    from torch.utils.data import DataLoader

    from training.datasets.psa_corners import (
        PSACornerDataset,
        build_corner_samples,
        split_by_cert,
    )

    jsonl_path = Path(str(cfg.dataset.jsonl)).expanduser()
    if not jsonl_path.exists():
        raise FileNotFoundError(
            f"dataset JSONL not found: {jsonl_path} — has the launchd ingest run?"
        )

    samples = build_corner_samples(
        jsonl_path,
        require_grade_in=(float(cfg.dataset.grade_min), float(cfg.dataset.grade_max)),
    )
    if len(samples) < int(cfg.dataset.min_samples):
        raise RuntimeError(
            f"only {len(samples)} corner samples found at {jsonl_path}; "
            f"need at least {int(cfg.dataset.min_samples)} to train. "
            "Wait for more daily ingest cycles or lower dataset.min_samples."
        )

    train_samples, val_samples = split_by_cert(
        samples,
        val_fraction=float(cfg.dataset.val_fraction),
        seed=int(cfg.dataset.split_seed),
    )
    _logger.info(
        "dataset stats total=%d train=%d val=%d",
        len(samples),
        len(train_samples),
        len(val_samples),
    )

    train_ds = PSACornerDataset(train_samples, crop_size=int(cfg.model.image_size), train=True)
    val_ds = PSACornerDataset(val_samples, crop_size=int(cfg.model.image_size), train=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg.train.batch_size),
        shuffle=True,
        num_workers=int(cfg.train.num_workers),
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg.train.batch_size),
        shuffle=False,
        num_workers=int(cfg.train.num_workers),
        pin_memory=False,
    )

    device = _device_from_cfg(str(cfg.train.device))
    _logger.info("device=%s", device)

    model = _build_model(cfg).to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.train.lr),
        weight_decay=float(cfg.train.weight_decay),
    )
    epochs = int(cfg.train.epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=float(cfg.train.lr_min)
    )

    out_dir = Path(str(cfg.train.output_dir)).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    if bool(cfg.train.smoke_only):
        _logger.info("smoke_only=True — running 1 mini-batch and exiting")
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            _logger.info("smoke step ok loss=%.4f", float(loss.item()))
            return {"smoke_loss": float(loss.item())}
        return {"smoke_loss": float("nan")}

    best_val_loss = float("inf")
    final_metrics: dict[str, float] = {}
    for epoch in range(epochs):
        _logger.info("epoch %d/%d lr=%.2e", epoch + 1, epochs, optimizer.param_groups[0]["lr"])
        train_metrics = _train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_metrics = _eval(model, val_loader, loss_fn, device)
        scheduler.step()

        _logger.info(
            "epoch %d done train_loss=%.4f val_loss=%.4f val_mae=%.4f",
            epoch + 1,
            train_metrics["train_loss"],
            val_metrics["val_loss"],
            val_metrics["val_mae"],
        )
        final_metrics = {**train_metrics, **val_metrics, "epoch": float(epoch + 1)}

        if val_metrics["val_loss"] < best_val_loss:
            best_val_loss = float(val_metrics["val_loss"])
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": OmegaConf.to_container(cfg, resolve=True),
                    "epoch": epoch + 1,
                    "val_loss": best_val_loss,
                },
                out_dir / "best.pt",
            )
            _logger.info("saved new best checkpoint val_loss=%.4f", best_val_loss)

    return final_metrics


def _flatten(d: object, prefix: str = "") -> dict[str, object]:
    out: dict[str, object] = {}
    if isinstance(d, dict):
        for k, v in d.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            out.update(_flatten(v, key))
    else:
        out[prefix] = d
    return out


@hydra.main(version_base=None, config_path="../configs", config_name="corners")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    _validate_config(cfg)

    # MLflow tracking is optional — if the server isn't reachable we
    # warn rather than fail (the skeleton's first runs will often be
    # local-only).
    mlflow = None
    try:
        import mlflow as _mlflow

        _mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
        _mlflow.set_experiment(cfg.mlflow.experiment)
        mlflow = _mlflow
    except Exception as e:
        if bool(cfg.mlflow.fail_on_tracking_error):
            raise
        _logger.warning("mlflow tracking unavailable, continuing without: %s", e)

    if mlflow is None:
        _run_training(cfg)
        return

    try:
        with mlflow.start_run(run_name=cfg.mlflow.run_name):
            cfg_dict = OmegaConf.to_container(cfg, resolve=True)
            mlflow.log_dict(cfg_dict, "config.yaml")
            mlflow.log_params(_flatten(cfg_dict))

            metrics = _run_training(cfg)
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    mlflow.log_metric(str(k).replace("/", "_"), float(v))

            best_path = Path(str(cfg.train.output_dir)).expanduser() / "best.pt"
            if best_path.exists():
                mlflow.log_artifact(str(best_path), artifact_path="weights")
    except Exception as e:
        if bool(cfg.mlflow.fail_on_tracking_error):
            raise
        _logger.warning("mlflow run failed mid-training, continuing: %s", e)


if __name__ == "__main__":
    os.environ.setdefault("HYDRA_FULL_ERROR", "1")
    main()  # type: ignore[call-arg]
