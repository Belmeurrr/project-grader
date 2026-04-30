"""PSA card-identification triplet dataset.

Reads the same JSONL records that `psa_corners.py` and `psa_surface.py`
consume (`ml/data/ingestion/psa_public_api`) and yields
(anchor, positive, negative) image-tensor triples for training the
identification embedding model with a triplet-margin loss.

Why triplet loss for identification:
    Identification is variant-matching: given a card image, retrieve the
    closest entry in a manufacturer-canonical catalog by embedding
    distance. Metric-learning losses (triplet, contrastive, ArcFace)
    optimize the embedding space directly. Triplet loss is the simplest
    of the three and gives us a baseline; ArcFace / supervised
    contrastive are reasonable swap-ins once the corpus has stable
    variant ids.

Why "augmentation-based" positives, not "string-matched" positives:
    The PSA Public API records carry `card_name` + `set_name` as free-
    text strings, not a stable variant_id. True "same-variant"
    positives would require a string-matching heuristic (normalized
    name + set + maybe card_number) that's brittle and out of scope
    for a skeleton.

    The skeleton therefore picks the simpler workaround: anchor and
    positive are the SAME image with two independent random
    augmentations (RandomResizedCrop + ColorJitter + light affine).
    Negative is a random different image. This proves the triplet
    pipeline plumbs through forward/backward/optimizer; it does NOT
    teach the model to be invariant to printing differences (foil,
    border, alt-art, language). That's an explicit known limitation.

    Swap-in when the catalog has stable variant ids: see
    `IdentificationTripletDataset.__init__`. Replace the augmentation-
    based positive sampler with a same-variant-different-record
    positive sampler. Negatives stay random-different. The trainer +
    loss are unchanged.

Why we share the JSONL contract with the other trainers:
    Same data flywheel feeds all three heads. A second JSONL or a forked
    schema would invite the trainers to drift; sharing means
    `front_image_path`, `grade`, `cert_id` mean the same thing in every
    context and improvements to the upstream ingest benefit every
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
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

# torch + torchvision are optional at import time so the manifest
# builder + split helper + sampler can run in a stdlib-only environment
# (e.g. `make stats`-style commands and the test suite). Mirrors
# psa_corners and psa_surface.
try:
    import torch
    from torch.utils.data import Dataset
    from PIL import Image
    from torchvision import transforms

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover — exercised in stdlib-only envs
    _TORCH_AVAILABLE = False
    Dataset = object  # type: ignore[misc, assignment]


_logger = logging.getLogger("psa_identification_dataset")


@dataclass(frozen=True)
class IdentificationSample:
    """One identification training example.

    Pre-tensor representation. The Dataset.__getitem__ converts this
    into a (anchor, positive, negative) tensor triple via the transform
    pipeline + a random-different-sample negative sampler.

    `card_name` + `set_name` are kept on the sample so a future
    string-matched positive sampler can use them without re-reading the
    JSONL. The skeleton's augmentation-based positive sampler ignores
    them."""

    cert_id: int
    image_path: Path
    grade: float
    card_name: str
    set_name: str
    # Whole record kept around so a future "real positive" sampler can
    # use any field it needs (year / variant / number) without re-
    # reading the JSONL.
    record: dict


def _read_manifest_records(jsonl_path: Path) -> Iterable[dict]:
    """Yield records from the scraper's JSONL output.

    Pure stdlib so it runs in environments without torch — useful for
    `make stats`-style commands and for the test harness. Mirrors the
    corners + surface dataset helpers so a future refactor (e.g. moving
    to a shared `_jsonl.py`) can dedupe all three with one change."""
    with jsonl_path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                _logger.warning("skipping malformed JSONL line %d: %s", line_no, e)


def build_identification_samples(
    jsonl_path: Path,
    *,
    require_grade_in: tuple[float, float] = (1.0, 10.0),
) -> list[IdentificationSample]:
    """Read the JSONL and emit one IdentificationSample per record.

    Pure-stdlib pre-tensor manifest. Does NOT load image bytes — only
    references paths. The Dataset wraps the result and adds the
    image-loading + transform layer.

    Records without an existing front_image_path or with grades outside
    `require_grade_in` are skipped. The returned list length is the
    authoritative sample count (the trainer's `min_samples` gate is
    measured against this).
    """
    samples: list[IdentificationSample] = []
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
            IdentificationSample(
                cert_id=int(cert_id),
                image_path=front_path,
                grade=float(grade),
                card_name=str(rec.get("card_name") or ""),
                set_name=str(rec.get("set_name") or ""),
                record=rec,
            )
        )
    return samples


def split_by_cert(
    samples: Sequence[IdentificationSample],
    *,
    val_fraction: float = 0.2,
    seed: int = 0xC0DE,
) -> tuple[list[IdentificationSample], list[IdentificationSample]]:
    """80/20 split by cert_id.

    Identification triplets are sampled from within a split — anchors
    pair with positives + negatives drawn from the same side. A row-
    index split would already keep cards together (only one record per
    cert in the manifest builder), but we split by cert_id to keep the
    splitter shape identical to corners + surface and to leave room for
    a future "multiple crops per cert" mode without changing the
    splitter."""
    cert_ids = sorted({s.cert_id for s in samples})
    rng = random.Random(seed)
    rng.shuffle(cert_ids)
    n_val = max(1, int(len(cert_ids) * val_fraction)) if cert_ids else 0
    val_certs = set(cert_ids[:n_val])

    train, val = [], []
    for s in samples:
        (val if s.cert_id in val_certs else train).append(s)
    return train, val


def sample_triplet_indices(
    n_samples: int,
    anchor_index: int,
    *,
    rng: random.Random,
) -> tuple[int, int, int]:
    """Pure-stdlib triplet-index sampler.

    Returns `(anchor_index, positive_index, negative_index)` where:
      - `positive_index == anchor_index` because the skeleton's
        positive is the SAME image with a different augmentation (see
        module docstring). The Dataset applies two independent random
        transforms to that one image to produce the anchor + positive
        tensors.
      - `negative_index` is a uniform-random different sample.

    Raises `ValueError` if there's no possible negative (n_samples < 2).
    Exposed as a top-level function so the test harness can exercise
    the sampler logic without instantiating the torch Dataset."""
    if n_samples < 2:
        raise ValueError(
            f"need at least 2 samples to draw a triplet; got n={n_samples}"
        )
    if not (0 <= anchor_index < n_samples):
        raise ValueError(
            f"anchor_index={anchor_index} out of range for n={n_samples}"
        )
    # Sample a different index for negative. Reject-sample rather than
    # build a candidate list so we stay O(1) regardless of n_samples.
    while True:
        neg = rng.randrange(n_samples)
        if neg != anchor_index:
            return anchor_index, anchor_index, neg


class IdentificationTripletDataset(Dataset):  # type: ignore[misc]
    """torch Dataset of (anchor, positive, negative) triplets.

    Constructor reads a list of IdentificationSample (e.g. from
    `build_identification_samples`) and applies the augmentation
    pipeline on `__getitem__`. Image loading is lazy per-item.

    Args:
      samples: pre-tensor manifest from `build_identification_samples`.
      image_size: side length of the square model input. DinoV2 ViT-B/14
        wants multiples of 14; defaults vary, the trainer config sets
        the canonical value.
      train: training-mode augmentations on/off. Eval mode reuses the
        same triplet structure but with a deterministic transform on
        anchor + positive (so they are identical) — useful for
        validation distance metrics.
      seed: RNG seed for the negative sampler. Fixed seed makes the
        train loop reproducible. Workers re-seed via worker_init_fn
        in the trainer if num_workers > 0.

    Augmentation:
      Strong augmentation gap between anchor and positive is the
      whole point of the skeleton's positive-sampling strategy. We use
      RandomResizedCrop (different crop windows) + ColorJitter
      (different color profiles) + a small RandomAffine. The model
      should learn to embed the same card consistently across these
      perturbations even if it doesn't (yet) learn invariance to
      printing differences.

      Flips are NOT applied (card content is left/right meaningful).

    Future "string-matched positives" mode:
      Replace the body of `_sample_positive_index` with a lookup keyed
      on `(card_name, set_name)` (or a normalized variant). Negatives
      remain random-different. The trainer + loss are unchanged.
    """

    def __init__(
        self,
        samples: Sequence[IdentificationSample],
        *,
        image_size: int = 224,
        train: bool = True,
        seed: int = 0xD00D,
        normalize_mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        normalize_std: tuple[float, float, float] = (0.229, 0.224, 0.225),
    ) -> None:
        if not _TORCH_AVAILABLE:
            raise RuntimeError(
                "torch + torchvision + Pillow required for "
                "IdentificationTripletDataset; run `uv sync --extra training` "
                "from ml/"
            )
        if len(samples) < 2:
            raise ValueError(
                f"need at least 2 samples to form a triplet; got {len(samples)}"
            )
        self._samples = list(samples)
        self._image_size = int(image_size)
        self._train = bool(train)
        self._rng = random.Random(seed)

        if train:
            # Two views of the same image differ in crop window + color
            # so the network has to learn to embed them together. The
            # Compose object is applied independently to anchor +
            # positive in __getitem__ so the random parameters differ
            # between the two calls.
            self._anchor_transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop(
                        image_size,
                        scale=(0.7, 1.0),
                        ratio=(0.75, 1.3333333),
                    ),
                    transforms.ColorJitter(
                        brightness=0.20, contrast=0.20, saturation=0.10, hue=0.02
                    ),
                    transforms.RandomAffine(
                        degrees=3.0,
                        translate=(0.03, 0.03),
                        scale=(0.97, 1.03),
                        fill=0,
                    ),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=list(normalize_mean), std=list(normalize_std)
                    ),
                ]
            )
            # Positive view shares the same transform — the randomness
            # in RandomResizedCrop/ColorJitter/RandomAffine differs
            # because each `transform(img)` call samples fresh params.
            self._positive_transform = self._anchor_transform
        else:
            # Eval mode: deterministic resize + normalize. Anchor and
            # positive are identical tensors, which makes the val-time
            # triplet distance trivially zero on the positive arm — by
            # design, eval here is for the negative-distance distribution
            # only.
            self._anchor_transform = transforms.Compose(
                [
                    transforms.Resize((image_size, image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=list(normalize_mean), std=list(normalize_std)
                    ),
                ]
            )
            self._positive_transform = self._anchor_transform

        self._negative_transform = self._anchor_transform

    def __len__(self) -> int:
        return len(self._samples)

    def _sample_positive_index(self, anchor_index: int) -> int:
        """Augmentation-based positive: same-image, different-augmentation.

        Skeleton strategy — see module docstring. Returns the anchor
        index unchanged; the Dataset applies a different random
        transform call to produce the positive tensor."""
        return anchor_index

    def _sample_negative_index(self, anchor_index: int) -> int:
        _, _, neg = sample_triplet_indices(
            len(self._samples), anchor_index, rng=self._rng
        )
        return neg

    def _load_rgb(self, path: Path) -> "Image.Image":
        with Image.open(path) as full:
            return full.convert("RGB")

    def __getitem__(
        self, idx: int
    ) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        anchor_sample = self._samples[idx]
        positive_idx = self._sample_positive_index(idx)
        negative_idx = self._sample_negative_index(idx)

        anchor_img = self._load_rgb(anchor_sample.image_path)
        # Augmentation-based positive: re-load from disk (cheap and
        # keeps the lazy-load contract) and apply the transform a
        # second time. RandomResizedCrop + ColorJitter sample fresh
        # params per call so the two views differ.
        positive_img = self._load_rgb(self._samples[positive_idx].image_path)
        negative_img = self._load_rgb(self._samples[negative_idx].image_path)

        anchor = self._anchor_transform(anchor_img)
        positive = self._positive_transform(positive_img)
        negative = self._negative_transform(negative_img)
        return anchor, positive, negative


__all__ = [
    "IdentificationSample",
    "IdentificationTripletDataset",
    "build_identification_samples",
    "sample_triplet_indices",
    "split_by_cert",
]
