"""PSA surface-defect segmentation dataset.

Reads the same JSONL records that `psa_corners.py` consumes
(`ml/data/ingestion/psa_public_api`) and yields (canonical_image,
per_pixel_class_mask) tensor pairs for training the surface-defect
semantic-segmentation model.

Why a dataset skeleton with no real labels:
    Per-defect-class pixel masks (scratch / print_line / indentation /
    stain / paper_loss / foil_scratch vs. background) do NOT exist in
    the corpus today. PSA Public API records have a numeric overall
    grade and a JPEG of the slabbed card — that's it. Building a
    labeled segmentation corpus is a separate, expensive step (manual
    annotation or a careful active-learning loop).

    Until then, this dataset emits an all-background placeholder mask
    for every record so the trainer can validate plumbing end-to-end:
    forward pass, loss computation, backward, optimizer step. The
    skeleton trainer's role is to PROVE THE PIPE WORKS, not to learn
    anything useful — same posture as the corners-trainer skeleton.

    When real labels arrive (in a separate label store), swap the
    placeholder by passing `mask_loader=...` to the dataset
    constructor. The mask_loader signature is `(record_dict) ->
    np.ndarray (H, W) of int class indices`. No changes to the
    trainer or the loss function are required.

Why we share the JSONL contract with the corners trainer:
    Same data flywheel feeds both heads. A second JSONL or a forked
    schema would invite the two trainers to drift; sharing means
    `front_image_path`, `grade`, `cert_id` mean the same thing in both
    contexts and improvements to the upstream ingest benefit every
    consumer.

Image source-of-truth:
    Reads `front_image_path` from the JSONL row directly. Records with
    `front_image_path = None` (cert had no images) or paths that no
    longer exist on disk are silently skipped at index time and
    surfaced via the manifest builder's caller (count of returned
    samples vs. records in JSONL).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

# torch + torchvision are optional at import time so the manifest
# builder + split helper can run in a stdlib-only environment (e.g.
# `make stats` style commands and the test suite). Mirrors psa_corners.
try:
    import numpy as np
    import torch
    from torch.utils.data import Dataset
    from PIL import Image
    from torchvision import transforms

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover — exercised in stdlib-only envs
    _TORCH_AVAILABLE = False
    Dataset = object  # type: ignore[misc, assignment]


_logger = logging.getLogger("psa_surface_dataset")


# Class indices and labels. Order MUST match `defect_classes` in
# `training/configs/surface.yaml` — the trainer reads N_CLASSES from
# `len(cfg.model.defect_classes)` so any reorder there should be
# reflected here too.
DEFECT_CLASSES: tuple[str, ...] = (
    "background",
    "scratch",
    "print_line",
    "indentation",
    "stain",
    "paper_loss",
    "foil_scratch",
)
NUM_CLASSES: int = len(DEFECT_CLASSES)
BACKGROUND_CLASS: int = 0


@dataclass(frozen=True)
class SurfaceSample:
    """One surface-segmentation training example.

    Pre-tensor representation. The Dataset.__getitem__ converts this
    into a (image_tensor, mask_tensor) pair via the transform pipeline."""

    cert_id: int
    image_path: Path
    grade: float
    # Whole record kept around so a real-mask `mask_loader` can use any
    # field it needs (set / year / variant) without re-reading the JSONL.
    record: dict


# Type alias for clarity. A mask_loader takes the full record dict and
# returns a (H, W) integer-class numpy array at the model's input
# resolution. Returning None is allowed and means "fall back to the
# all-background placeholder for this record" — useful when only a
# subset of records have real masks.
MaskLoader = Callable[[dict], "np.ndarray | None"]


def _read_manifest_records(jsonl_path: Path) -> Iterable[dict]:
    """Yield records from the scraper's JSONL output.

    Pure stdlib so it runs in environments without torch — useful for
    `make stats`-style commands and for the test harness. Mirrors the
    corners-dataset helper so a future refactor (e.g. moving to a
    shared `_jsonl.py`) can dedupe both with one change."""
    with jsonl_path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                _logger.warning("skipping malformed JSONL line %d: %s", line_no, e)


def build_surface_samples(
    jsonl_path: Path,
    *,
    require_grade_in: tuple[float, float] = (1.0, 10.0),
) -> list[SurfaceSample]:
    """Read the JSONL and emit one SurfaceSample per record.

    Pure-stdlib pre-tensor manifest. Does NOT load image bytes — only
    references paths. The Dataset wraps the result and adds the
    image-loading + transform layer.

    Records without an existing front_image_path or with grades outside
    `require_grade_in` are skipped. The returned list length is the
    authoritative sample count (the trainer's `min_samples` gate is
    measured against this).
    """
    samples: list[SurfaceSample] = []
    for rec in _read_manifest_records(jsonl_path):
        front = rec.get("front_image_path")
        grade = rec.get("grade")
        cert_id = rec.get("cert_id")
        if not front or grade is None or cert_id is None:
            continue
        if not (require_grade_in[0] <= float(grade) <= require_grade_in[1]):
            continue
        front_path = Path(front)
        if not front_path.exists():
            _logger.debug("front image missing for cert_id=%s path=%s", cert_id, front_path)
            continue
        samples.append(
            SurfaceSample(
                cert_id=int(cert_id),
                image_path=front_path,
                grade=float(grade),
                record=rec,
            )
        )
    return samples


def split_by_cert(
    samples: Sequence[SurfaceSample],
    *,
    val_fraction: float = 0.2,
    seed: int = 0xC0DE,
) -> tuple[list[SurfaceSample], list[SurfaceSample]]:
    """80/20 split by cert_id.

    Surface segmentation only emits ONE sample per cert (vs. corners'
    four), so a row-index split would already keep cards together.
    Splitting by cert_id is still safer because we may later emit
    multiple crops per card (sub-region tiles) without changing the
    splitter."""
    import random

    cert_ids = sorted({s.cert_id for s in samples})
    rng = random.Random(seed)
    rng.shuffle(cert_ids)
    n_val = max(1, int(len(cert_ids) * val_fraction)) if cert_ids else 0
    val_certs = set(cert_ids[:n_val])

    train, val = [], []
    for s in samples:
        (val if s.cert_id in val_certs else train).append(s)
    return train, val


def all_background_mask_loader(record: dict) -> "np.ndarray | None":
    """Default placeholder mask loader: returns None so the dataset's
    fallback (all-background mask) is used for every record.

    Exists as a named function so callers can reason about the default
    behavior explicitly and so log lines mentioning the loader's name
    are meaningful."""
    return None


class PSASurfaceDataset(Dataset):  # type: ignore[misc]
    """torch Dataset of (canonical_image, per_pixel_mask) examples.

    Constructor reads a list of SurfaceSample (e.g. from
    `build_surface_samples`) and applies the transform pipeline on
    `__getitem__`. Image loading is lazy per-item.

    Args:
      samples: pre-tensor manifest from `build_surface_samples`.
      image_size: side length of the square model input (e.g. 384).
        The image is letterbox-resized into a square of this size and
        the mask is resized to match.
      train: training-mode augmentations on/off.
      mask_loader: callable that returns a (H, W) integer-class numpy
        array for a record, or None to fall back to all-background.
        Default `all_background_mask_loader` returns None for every
        record — that's the skeleton's "no labels yet" mode.
      num_classes: total class count including background. Used as a
        sanity check on returned masks.

    Augmentation:
        Light, segmentation-aware. Flips are NOT applied (left/right
        defect placement is meaningful information and the model
        should learn it). Color jitter is mild because surface defects
        often present as subtle hue/luminance shifts and aggressive
        jitter would teach the model to ignore them.
    """

    def __init__(
        self,
        samples: Sequence[SurfaceSample],
        *,
        image_size: int = 384,
        train: bool = True,
        mask_loader: MaskLoader = all_background_mask_loader,
        num_classes: int = NUM_CLASSES,
        normalize_mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        normalize_std: tuple[float, float, float] = (0.229, 0.224, 0.225),
    ) -> None:
        if not _TORCH_AVAILABLE:
            raise RuntimeError(
                "torch + torchvision + Pillow + numpy required for "
                "PSASurfaceDataset; run `uv sync --extra training` from ml/"
            )
        self._samples = list(samples)
        self._image_size = int(image_size)
        self._mask_loader = mask_loader
        self._num_classes = int(num_classes)

        # The image transform mirrors the corners trainer's shape so
        # the two heads' input distributions are commensurable. Resize
        # to (S, S) is a stretch rather than a letterbox — cards have
        # a fixed 5:7 aspect ratio so the stretch is bounded and the
        # SegFormer-style decoder is robust to mild aspect changes.
        train_aug: list = [
            transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.03),
        ]
        eval_aug: list = []

        self._image_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                *(train_aug if train else eval_aug),
                transforms.ToTensor(),
                transforms.Normalize(mean=list(normalize_mean), std=list(normalize_std)),
            ]
        )

    def __len__(self) -> int:
        return len(self._samples)

    def _load_or_default_mask(self, sample: SurfaceSample) -> "np.ndarray":
        """Return the mask for `sample`. Tries the configured mask_loader;
        falls back to all-background when the loader returns None or the
        returned shape is wrong."""
        loaded = self._mask_loader(sample.record)
        if loaded is None:
            return np.zeros(
                (self._image_size, self._image_size), dtype=np.int64
            )
        if loaded.shape != (self._image_size, self._image_size):
            _logger.warning(
                "mask_loader returned shape %s, expected (%d, %d) — "
                "falling back to all-background",
                loaded.shape,
                self._image_size,
                self._image_size,
            )
            return np.zeros(
                (self._image_size, self._image_size), dtype=np.int64
            )
        if loaded.dtype != np.int64:
            loaded = loaded.astype(np.int64)
        # Defensive class-index clipping in case the loader returns
        # out-of-range labels (e.g. a future ontology change). Out-of-
        # range values become background so the loss stays well-defined.
        out_of_range = (loaded < 0) | (loaded >= self._num_classes)
        if out_of_range.any():
            loaded = loaded.copy()
            loaded[out_of_range] = BACKGROUND_CLASS
        return loaded

    def __getitem__(self, idx: int) -> tuple["torch.Tensor", "torch.Tensor"]:
        s = self._samples[idx]
        with Image.open(s.image_path) as full:
            full = full.convert("RGB")
            x = self._image_transform(full)

        mask = self._load_or_default_mask(s)
        # mask: int64 numpy of shape (S, S); CrossEntropyLoss expects
        # long target of shape (S, S). Convert here to keep the
        # collate_fn default-stack-able.
        y = torch.from_numpy(mask)
        return x, y


__all__ = [
    "BACKGROUND_CLASS",
    "DEFECT_CLASSES",
    "MaskLoader",
    "NUM_CLASSES",
    "PSASurfaceDataset",
    "SurfaceSample",
    "all_background_mask_loader",
    "build_surface_samples",
    "split_by_cert",
]
