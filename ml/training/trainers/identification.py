"""DinoV2 ViT-B/14 identification embedding trainer skeleton.

Reads `scraped.jsonl` produced by `ml/data/ingestion/psa_public_api`,
trains a DinoV2 backbone with triplet-margin loss against (anchor,
positive, negative) image triples, logs to MLflow, writes weights to
`cfg.train.output_dir`.

Run (from ml/, with `uv sync --extra training`):

    uv run python -m training.trainers.identification train.smoke_only=true train.epochs=1

Override anything via Hydra, same as corners + surface:

    uv run python -m training.trainers.identification train.epochs=1 model.image_size=224

Architecture summary:
    backbone (timm DinoV2 ViT-B/14, LVD-142M pretrain) →
    global pool ('token' = CLS embedding) →
    L2-normalized embedding (D=768)

Loss: TripletMarginLoss(margin=cfg.train.triplet_margin,
    p=cfg.train.triplet_p) on (anchor, positive, negative) embeddings.

Why DinoV2 ViT-B and not the same EfficientNet-V2-S as corners + surface:
    Identification is a retrieval problem — the model must produce a
    similarity space where same-variant pairs cluster and different-
    variant pairs separate. DinoV2's self-supervised pretraining
    objective is precisely this kind of representation learning, and
    its features outperform supervised ImageNet baselines on most
    retrieval benchmarks. Picking it for identification is the
    architecture plan's recommendation; the corners + surface heads
    stay on EfficientNet because they're regression / segmentation
    rather than retrieval.

Why triplet loss and not ArcFace / supervised contrastive:
    Triplet is the simplest metric-learning loss and the one with the
    fewest moving parts. The skeleton's job is plumbing, not SOTA
    accuracy. ArcFace + supervised contrastive both need a stable
    variant_id label that the PSA records don't carry today (see the
    psa_identification dataset docstring).

Why pretrained=false in the smoke run:
    Pulling the DinoV2 LVD-142M weights is a ~340MB download. The
    smoke run's whole point is plumbing-proof in seconds, so we skip
    the download and let the backbone init randomly. Real training
    sets pretrained=true.
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
# than via `python -m`), ensure ml/ is importable. Mirrors corners.py /
# surface.py.
_ML_ROOT = Path(__file__).resolve().parents[2]
if str(_ML_ROOT) not in sys.path:
    sys.path.insert(0, str(_ML_ROOT))


_logger = logging.getLogger("identification_trainer")


# DinoV2 backbone fallback chain. timm's exact id for DinoV2 ViT-B/14
# is `vit_base_patch14_dinov2.lvd142m` on most recent timm versions; on
# some versions the only available DinoV2 ViT-B is the register-token
# variant `vit_base_patch14_reg4_dinov2.lvd142m`. Same backbone weights,
# slightly different forward path (4 register tokens prepended). The
# trainer tries the cleaner id first and falls back if it's not in
# `timm.list_models()`.
_DINOV2_FALLBACKS: tuple[str, ...] = (
    "vit_base_patch14_dinov2.lvd142m",
    "vit_base_patch14_reg4_dinov2.lvd142m",
)


def _validate_config(cfg: DictConfig) -> None:
    """Cheap pre-flight. Failing here beats failing 10 minutes into
    a training run on a missing field."""
    required = [
        "dataset.jsonl",
        "dataset.min_samples",
        "dataset.val_fraction",
        "dataset.triplet_seed",
        "model.backbone",
        "model.image_size",
        "model.embedding_dim",
        "train.epochs",
        "train.batch_size",
        "train.lr",
        "train.triplet_margin",
        "train.triplet_p",
        "train.output_dir",
    ]
    for path in required:
        node = cfg
        for part in path.split("."):
            if not hasattr(node, part) and part not in node:
                raise ValueError(f"missing required config: {path}")
            node = getattr(node, part) if hasattr(node, part) else node[part]


def _resolve_backbone_name(requested: str) -> str:
    """Pick the configured DinoV2 backbone if timm has it, else the next
    fallback. Logs the choice so a fallback is visible in run logs.

    timm exposes models in two registries: `list_models()` returns
    architecture ids (e.g. `vit_base_patch14_dinov2`), and
    `list_pretrained()` returns architecture-with-pretrain-tag ids
    (e.g. `vit_base_patch14_dinov2.lvd142m`). The config uses the
    pretrain-tagged form so a real run uses the right weights, but a
    bare architecture id is also valid input. Check both registries +
    accept the bare-architecture prefix of a tagged id."""
    import timm

    arch_set = set(timm.list_models())
    pretrained_set = set(timm.list_pretrained())

    def _is_known(name: str) -> bool:
        if name in arch_set or name in pretrained_set:
            return True
        # Accept `arch.tag` if `arch` is a known timm architecture —
        # timm.create_model resolves the tag to a pretrained config.
        bare = name.split(".", 1)[0]
        return bare in arch_set

    if _is_known(requested):
        return requested
    for fallback in _DINOV2_FALLBACKS:
        if fallback != requested and _is_known(fallback):
            _logger.warning(
                "backbone %s not in timm; falling back to %s", requested, fallback
            )
            return fallback
    raise RuntimeError(
        f"no DinoV2 ViT-B backbone found in timm; tried {requested!r} "
        f"and fallbacks {_DINOV2_FALLBACKS!r}. Upgrade timm."
    )


def _build_model(cfg: DictConfig):
    """timm DinoV2 ViT-B/14 backbone, no projection head.

    Output: (B, D) embedding where D = model.embedding_dim (768 for
    ViT-B). The skeleton does NOT add a projection head — the
    backbone's pooled CLS-token embedding goes directly into the
    triplet loss. A 2-layer MLP projection head is a reasonable swap-in
    once we tune for retrieval recall, but for the skeleton's plumbing-
    proof job, fewer moving parts wins.

    Imported lazily so the module is importable in stdlib-only
    environments for syntax checks (mirrors corners.py / surface.py)."""
    import timm

    backbone_name = _resolve_backbone_name(str(cfg.model.backbone))
    image_size = int(cfg.model.image_size)
    # DinoV2 ViT-B/14 has a fixed pretrained patch grid (518 / 14 = 37
    # patches per side). Passing `img_size=` to timm.create_model
    # rebuilds the patch embedder for our chosen input size and
    # interpolates positional embeddings to match. Required when
    # `image_size != 518`. Must be a multiple of 14 (the patch size).
    if image_size % 14 != 0:
        raise ValueError(
            f"DinoV2 ViT-B/14 requires image_size divisible by 14; got "
            f"{image_size}. Pick a multiple of 14 (e.g. 126, 140, 224, 518)."
        )
    backbone = timm.create_model(
        backbone_name,
        pretrained=bool(cfg.model.pretrained),
        num_classes=0,  # disable timm's classifier
        global_pool="token",  # DinoV2 → return the CLS-token embedding
        img_size=image_size,
    )
    expected_dim = int(cfg.model.embedding_dim)
    actual_dim = int(backbone.num_features)
    if actual_dim != expected_dim:
        _logger.warning(
            "backbone num_features=%d but cfg.model.embedding_dim=%d — "
            "the config's embedding_dim is informational only; loss + "
            "logging will use the actual backbone output size.",
            actual_dim,
            expected_dim,
        )
    return backbone


def _device_from_cfg(name: str):
    """Mirror of corners._device_from_cfg / surface._device_from_cfg.
    Kept duplicated rather than extracted to a shared helper because
    each trainer is the de-facto reference for its head and we don't
    want to invent shared infrastructure for a single-line decision
    yet — the comment in surface.py applies here too."""
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
    for step, (anchor, positive, negative) in enumerate(loader):
        anchor = anchor.to(device, non_blocking=True)
        positive = positive.to(device, non_blocking=True)
        negative = negative.to(device, non_blocking=True)
        optimizer.zero_grad()
        a_emb = model(anchor)
        p_emb = model(positive)
        n_emb = model(negative)
        loss = loss_fn(a_emb, p_emb, n_emb)
        loss.backward()
        optimizer.step()
        bs = anchor.size(0)
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
    pos_distances: list[float] = []
    neg_distances: list[float] = []
    with torch.no_grad():
        for anchor, positive, negative in loader:
            anchor = anchor.to(device, non_blocking=True)
            positive = positive.to(device, non_blocking=True)
            negative = negative.to(device, non_blocking=True)
            a_emb = model(anchor)
            p_emb = model(positive)
            n_emb = model(negative)
            loss = loss_fn(a_emb, p_emb, n_emb)
            total_loss += float(loss.item()) * anchor.size(0)
            total_examples += anchor.size(0)
            # Per-pair distances for the val-time positive/negative
            # distance distributions. p=2 to match the most common
            # configuration; if the loss uses p=1 the relative
            # comparison still holds.
            d_pos = (a_emb - p_emb).pow(2).sum(dim=1).sqrt()
            d_neg = (a_emb - n_emb).pow(2).sum(dim=1).sqrt()
            pos_distances.extend(d_pos.detach().cpu().numpy().tolist())
            neg_distances.extend(d_neg.detach().cpu().numpy().tolist())
    if total_examples == 0:
        return {
            "val_loss": float("nan"),
            "val_pos_dist": float("nan"),
            "val_neg_dist": float("nan"),
            "val_examples": 0,
        }
    return {
        "val_loss": total_loss / total_examples,
        "val_pos_dist": sum(pos_distances) / max(len(pos_distances), 1),
        "val_neg_dist": sum(neg_distances) / max(len(neg_distances), 1),
        "val_examples": total_examples,
    }


def _run_training(cfg: DictConfig) -> dict[str, float]:
    """Build dataset + model + optimizer, run epochs, return final
    metrics. Mirrors corners._run_training / surface._run_training
    shape."""
    import torch
    from torch import nn
    from torch.utils.data import DataLoader

    from training.datasets.psa_identification import (
        IdentificationTripletDataset,
        build_identification_samples,
        split_by_cert,
    )

    jsonl_path = Path(str(cfg.dataset.jsonl)).expanduser()
    if not jsonl_path.exists():
        raise FileNotFoundError(
            f"dataset JSONL not found: {jsonl_path} — has the launchd ingest run?"
        )

    samples = build_identification_samples(
        jsonl_path,
        require_grade_in=(float(cfg.dataset.grade_min), float(cfg.dataset.grade_max)),
    )
    if len(samples) < int(cfg.dataset.min_samples):
        raise RuntimeError(
            f"only {len(samples)} identification samples found at {jsonl_path}; "
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

    train_ds = IdentificationTripletDataset(
        train_samples,
        image_size=int(cfg.model.image_size),
        train=True,
        seed=int(cfg.dataset.triplet_seed),
    )
    val_ds = IdentificationTripletDataset(
        val_samples,
        image_size=int(cfg.model.image_size),
        train=False,
        # Different seed for the val sampler so val negatives are
        # uncorrelated with train negatives.
        seed=int(cfg.dataset.triplet_seed) + 1,
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
    loss_fn = nn.TripletMarginLoss(
        margin=float(cfg.train.triplet_margin),
        p=float(cfg.train.triplet_p),
    )
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
        for anchor, positive, negative in train_loader:
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)
            a_emb = model(anchor)
            p_emb = model(positive)
            n_emb = model(negative)
            loss = loss_fn(a_emb, p_emb, n_emb)
            loss.backward()
            optimizer.step()
            _logger.info(
                "smoke step ok loss=%.4f anchor_shape=%s embedding_shape=%s",
                float(loss.item()),
                tuple(anchor.shape),
                tuple(a_emb.shape),
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
            "epoch %d done train_loss=%.4f val_loss=%.4f val_pos_dist=%.4f val_neg_dist=%.4f",
            epoch + 1,
            train_metrics["train_loss"],
            val_metrics["val_loss"],
            val_metrics["val_pos_dist"],
            val_metrics["val_neg_dist"],
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
    """Same flattener as corners.py / surface.py — kept duplicated to
    avoid extracting a shared helper. This is the third trainer; if a
    fourth lands, that's the trigger to extract a shared
    `training/_mlflow_utils.py` rather than the third copy paying the
    cost of premature abstraction."""
    out: dict[str, object] = {}
    if isinstance(d, dict):
        for k, v in d.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            out.update(_flatten(v, key))
    else:
        out[prefix] = d
    return out


@hydra.main(version_base=None, config_path="../configs", config_name="identification")
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
