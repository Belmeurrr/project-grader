"""Surface-defect semantic-segmentation skeleton trainer.

Reads `scraped.jsonl` produced by `ml/data/ingestion/psa_public_api`,
trains a multi-scale encoder + SegFormer-style MLP decoder against
per-pixel defect class labels, logs to MLflow, writes weights to
`cfg.train.output_dir`.

Run (from ml/, with `uv sync --extra training`):

    uv run python -m training.trainers.surface train.smoke_only=true train.epochs=1

Override anything via Hydra, same as corners:

    uv run python -m training.trainers.surface train.epochs=1 model.image_size=256

Architecture summary:
    backbone (timm EfficientNet-V2-S, ImageNet-21k pretrain, multi-
        scale features via features_only=True) →
    project each feature map to a common channel count (1×1 conv) →
    upsample all to the largest feature map's spatial resolution →
    concatenate along channels →
    1×1 conv to N_CLASSES at that resolution →
    bilinear upsample to the input size

Loss: CrossEntropyLoss against the per-pixel class mask. Index 0 is
the background class.

Why a SegFormer-style MLP decoder:
    SegFormer's decoder takes multi-scale features, projects each to a
    common channel count, upsamples to the highest-resolution feature
    map, concatenates, and produces logits via a single 1×1 conv. The
    architecture is simple, fast, and competitive with heavier U-Net
    style decoders. Implemented here without a transformers-style
    encoder (timm doesn't ship Mix Transformer at the time of writing)
    so the decoder is the SegFormer half; the encoder is the same
    EfficientNet-V2-S the corners trainer uses.

Why the same backbone as corners:
    Two heads sharing one backbone (eventually) is a useful pattern:
    one feature extractor, multiple grading-criterion heads. Picking
    the same backbone in both trainers keeps that future option open
    without committing to it now. If/when we move to a Mix Transformer
    encoder, both trainers swap together.
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
# than via `python -m`), ensure ml/ is importable. Mirrors corners.py.
_ML_ROOT = Path(__file__).resolve().parents[2]
if str(_ML_ROOT) not in sys.path:
    sys.path.insert(0, str(_ML_ROOT))


_logger = logging.getLogger("surface_trainer")


def _validate_config(cfg: DictConfig) -> None:
    """Cheap pre-flight. Failing here beats failing 10 minutes into
    a training run on a missing field."""
    required = [
        "dataset.jsonl",
        "dataset.min_samples",
        "dataset.val_fraction",
        "model.backbone",
        "model.image_size",
        "model.decoder_channels",
        "model.defect_classes",
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
    """Multi-scale encoder + SegFormer-style MLP decoder.

    The encoder is timm in `features_only=True` mode, which exposes a
    list of feature maps at progressively-coarser resolutions
    (typically ~ /4, /8, /16, /32 of the input). The decoder projects
    each to a common channel count, upsamples to the finest feature
    map's resolution, concatenates, and emits per-class logits at that
    resolution. A final bilinear upsample takes logits to input
    resolution so the loss can compute against the full-resolution
    target mask.

    All torch / timm imports live inside this function so the module
    stays importable in a stdlib-only env for syntax checks (mirrors
    corners.py)."""
    import timm
    import torch
    import torch.nn.functional as F
    from torch import nn

    encoder = timm.create_model(
        cfg.model.backbone,
        pretrained=bool(cfg.model.pretrained),
        features_only=True,
        out_indices=(1, 2, 3, 4),  # /4, /8, /16, /32 — drop the /2 stem
    )

    feature_channels: list[int] = list(encoder.feature_info.channels())
    decoder_channels = int(cfg.model.decoder_channels)
    num_classes = len(cfg.model.defect_classes)

    class SegFormerHead(nn.Module):
        """SegFormer's All-MLP decoder, generalized for any multi-scale
        encoder. Projects each feature map to `decoder_channels`,
        bilinear-upsamples them all to the finest scale, concatenates,
        and 1×1-projects to `num_classes`. A second bilinear upsample
        in `forward` of the parent module brings logits to input size."""

        def __init__(self, in_channels: list[int], out_channels: int) -> None:
            super().__init__()
            self.projects = nn.ModuleList(
                nn.Conv2d(c, out_channels, kernel_size=1) for c in in_channels
            )
            self.fuse = nn.Sequential(
                nn.Conv2d(out_channels * len(in_channels), out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

        def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
            # Project + upsample everything to the finest feature map's
            # spatial resolution.
            target_size = features[0].shape[-2:]
            projected = []
            for f, proj in zip(features, self.projects):
                p = proj(f)
                if p.shape[-2:] != target_size:
                    p = F.interpolate(p, size=target_size, mode="bilinear", align_corners=False)
                projected.append(p)
            return self.fuse(torch.cat(projected, dim=1))

    class SurfaceModel(nn.Module):
        def __init__(self, encoder_, head_in_channels: list[int]) -> None:
            super().__init__()
            self.encoder = encoder_
            self.decoder = SegFormerHead(head_in_channels, decoder_channels)
            self.classifier = nn.Conv2d(decoder_channels, num_classes, kernel_size=1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            input_size = x.shape[-2:]
            features = self.encoder(x)
            decoded = self.decoder(features)
            logits = self.classifier(decoded)
            # Bilinear-upsample logits to input resolution so the loss
            # can compute against the full-res mask.
            return F.interpolate(
                logits, size=input_size, mode="bilinear", align_corners=False
            )

    return SurfaceModel(encoder, feature_channels)


def _device_from_cfg(name: str):
    """Mirror of corners._device_from_cfg. Kept duplicated rather than
    extracted to a shared helper because the corners trainer is the
    de-facto reference and we don't want to invent shared infrastructure
    for a single-line decision yet."""
    import torch

    if name == "cpu":
        return torch.device("cpu")
    if name == "mps":
        if not torch.backends.mps.is_available():
            _logger.warning("requested mps but not available; falling back to cpu")
            return torch.device("cpu")
        return torch.device("mps")
    try:
        return torch.device(f"cuda:{int(name)}" if str(name).isdigit() else name)
    except Exception:
        return torch.device("cpu")


def _train_one_epoch(model, loader, optimizer, loss_fn, device, log_every: int = 20):
    import torch  # noqa: F401  (loop body uses torch implicitly)

    model.train()
    total_loss = 0.0
    total_examples = 0
    for step, (x, y) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad()
        logits = model(x)  # (B, C, H, W)
        loss = loss_fn(logits, y)
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
    pixel_correct = 0
    pixel_total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            total_loss += float(loss.item()) * x.size(0)
            total_examples += x.size(0)
            pred = logits.argmax(dim=1)  # (B, H, W)
            pixel_correct += int((pred == y).sum().item())
            pixel_total += int(y.numel())
    if total_examples == 0:
        return {"val_loss": float("nan"), "val_pixel_acc": float("nan"), "val_examples": 0}
    return {
        "val_loss": total_loss / total_examples,
        "val_pixel_acc": pixel_correct / max(pixel_total, 1),
        "val_examples": total_examples,
    }


def _run_training(cfg: DictConfig) -> dict[str, float]:
    """Build dataset + model + optimizer, run epochs, return final
    metrics. Mirrors corners._run_training shape."""
    import torch
    from torch import nn
    from torch.utils.data import DataLoader

    from training.datasets.psa_surface import (
        PSASurfaceDataset,
        build_surface_samples,
        split_by_cert,
    )

    jsonl_path = Path(str(cfg.dataset.jsonl)).expanduser()
    if not jsonl_path.exists():
        raise FileNotFoundError(
            f"dataset JSONL not found: {jsonl_path} — has the launchd ingest run?"
        )

    samples = build_surface_samples(
        jsonl_path,
        require_grade_in=(float(cfg.dataset.grade_min), float(cfg.dataset.grade_max)),
    )
    if len(samples) < int(cfg.dataset.min_samples):
        raise RuntimeError(
            f"only {len(samples)} surface samples found at {jsonl_path}; "
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

    train_ds = PSASurfaceDataset(
        train_samples, image_size=int(cfg.model.image_size), train=True
    )
    val_ds = PSASurfaceDataset(
        val_samples, image_size=int(cfg.model.image_size), train=False
    )

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
    loss_fn = nn.CrossEntropyLoss()
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
            y = y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
            _logger.info(
                "smoke step ok loss=%.4f logits_shape=%s",
                float(loss.item()),
                tuple(logits.shape),
            )
            return {"smoke_loss": float(loss.item())}
        return {"smoke_loss": float("nan")}

    best_val_loss = float("inf")
    final_metrics: dict[str, float] = {}
    for epoch in range(epochs):
        _logger.info(
            "epoch %d/%d lr=%.2e", epoch + 1, epochs, optimizer.param_groups[0]["lr"]
        )
        train_metrics = _train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_metrics = _eval(model, val_loader, loss_fn, device)
        scheduler.step()

        _logger.info(
            "epoch %d done train_loss=%.4f val_loss=%.4f val_pixel_acc=%.4f",
            epoch + 1,
            train_metrics["train_loss"],
            val_metrics["val_loss"],
            val_metrics["val_pixel_acc"],
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
    """Same flattener as corners.py — kept duplicated to avoid extracting
    a shared helper before the third trainer asks for it."""
    out: dict[str, object] = {}
    if isinstance(d, dict):
        for k, v in d.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            out.update(_flatten(v, key))
    else:
        out[prefix] = d
    return out


@hydra.main(version_base=None, config_path="../configs", config_name="surface")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    _validate_config(cfg)

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
