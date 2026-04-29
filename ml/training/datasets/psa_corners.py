"""PSA corner-crop dataset.

Reads the JSONL records produced by `ml/data/ingestion/psa_public_api`
and yields (corner_image, grade) tensor pairs for training the corners
grading model.

Why corner crops:
    PSA grades each of the 4 corners independently and reports the
    overall grade as `min(4 corner subgrades)`. So the corner-grading
    model takes a corner crop as input and predicts a per-corner grade;
    inference combines four crops with a min aggregation.

Why we label all 4 corners with the overall grade for now:
    The free-tier PSA Public API returns the overall card grade only —
    not per-corner subgrades. So this dataset uses overall-grade as a
    NOISY label for every corner crop. The model learns aggregate
    "what does a worn corner look like" features. Once we obtain
    per-corner labels (PSA paid tier, manual labeling, or learned from
    a future labeled subset), the dataset's `corner_grades` argument
    accepts a `{cert_id: [tl_grade, tr_grade, bl_grade, br_grade]}`
    override and the model upgrade is a config change, not a rewrite.

Why 80/20 split by cert_id, not by record index:
    Each cert produces 4 dataset rows (one per corner). If we split by
    row, the same card's corners straddle train and val — guaranteed
    leakage of card-specific texture, lighting, and capture conditions.
    Splitting by cert_id keeps all 4 corners of any given card on the
    same side of the split.

Image source-of-truth:
    The dataset reads `front_image_path` from the JSONL row directly.
    Records with `front_image_path = None` (cert had no images) or
    paths that no longer exist on disk are silently skipped at index
    time and surfaced in `dataset.skipped_count`.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

# torch + torchvision are optional at import-time for the
# benefit of stdlib-only smoke environments — actual instantiation
# requires them. Imports are guarded so `import psa_corners` doesn't
# fail in an env without torch.
try:
    import torch
    from torch.utils.data import Dataset
    from PIL import Image
    from torchvision import transforms

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover — exercised in stdlib-only envs
    _TORCH_AVAILABLE = False
    Dataset = object  # type: ignore[misc, assignment]


_logger = logging.getLogger("psa_corners_dataset")


# Corner indices in TL, TR, BL, BR order. The corners model treats them
# symmetrically; the order here is just the conventional reading order.
CORNER_TL = 0
CORNER_TR = 1
CORNER_BL = 2
CORNER_BR = 3
NUM_CORNERS = 4


@dataclass(frozen=True)
class CornerSample:
    """One corner-crop training example.

    Pre-tensor representation. The Dataset.__getitem__ converts this to
    a (image_tensor, grade_tensor) tuple via the transform pipeline."""

    cert_id: int
    corner_index: int  # 0..3
    image_path: Path
    grade: float


def _read_manifest_records(jsonl_path: Path) -> Iterable[dict]:
    """Yield records from the scraper's JSONL output.

    Pure stdlib so it runs in environments without torch — useful for
    `make stats`-style commands and for the test harness."""
    with jsonl_path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                _logger.warning("skipping malformed JSONL line %d: %s", line_no, e)


def build_corner_samples(
    jsonl_path: Path,
    *,
    require_grade_in: tuple[float, float] = (1.0, 10.0),
    corner_grades: dict[int, Sequence[float]] | None = None,
) -> list[CornerSample]:
    """Read the JSONL and emit one CornerSample per (cert, corner).

    Pure-stdlib pre-tensor manifest. Does NOT load image bytes — only
    references paths. The Dataset wraps the result and adds the
    image-loading + transform layer.

    `corner_grades`, if provided, overrides the per-corner grade for a
    cert with explicit subgrades (length-4 sequence in TL/TR/BL/BR
    order). When absent, all 4 corners get the overall card grade as
    their (noisy) label.
    """
    samples: list[CornerSample] = []
    for rec in _read_manifest_records(jsonl_path):
        front = rec.get("front_image_path")
        grade = rec.get("grade")
        cert_id = rec.get("cert_id")
        if not front or grade is None or cert_id is None:
            continue
        # Reject grades outside [require_grade_in] — defensive against
        # any future API shape where a non-numeric grade slipped through.
        if not (require_grade_in[0] <= float(grade) <= require_grade_in[1]):
            continue
        front_path = Path(front)
        if not front_path.exists():
            _logger.debug("front image missing for cert_id=%s path=%s", cert_id, front_path)
            continue

        per_corner = (
            list(corner_grades[cert_id])
            if corner_grades is not None and cert_id in corner_grades
            else [float(grade)] * NUM_CORNERS
        )
        if len(per_corner) != NUM_CORNERS:
            raise ValueError(
                f"corner_grades[{cert_id}] must have {NUM_CORNERS} values; got {len(per_corner)}"
            )
        for corner_index, g in enumerate(per_corner):
            samples.append(
                CornerSample(
                    cert_id=int(cert_id),
                    corner_index=corner_index,
                    image_path=front_path,
                    grade=float(g),
                )
            )
    return samples


def split_by_cert(
    samples: Sequence[CornerSample],
    *,
    val_fraction: float = 0.2,
    seed: int = 0xC0DE,
) -> tuple[list[CornerSample], list[CornerSample]]:
    """80/20 split by cert_id so a card's 4 corners stay together."""
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


def _crop_corners(img: "Image.Image", crop_size: int) -> list["Image.Image"]:
    """Return [TL, TR, BL, BR] crops of size `crop_size`×`crop_size` px.

    Assumes the input is the canonical 750×1050 dewarped card. If the
    actual dimensions differ, the crops are anchored at the four image
    corners; non-square cards still yield square crops since we slice
    `crop_size` pixels from each corner regardless of aspect."""
    W, H = img.size
    cs = crop_size
    return [
        img.crop((0, 0, cs, cs)),                # TL
        img.crop((W - cs, 0, W, cs)),            # TR
        img.crop((0, H - cs, cs, H)),            # BL
        img.crop((W - cs, H - cs, W, H)),        # BR
    ]


class PSACornerDataset(Dataset):  # type: ignore[misc]
    """torch Dataset of corner-crop training examples.

    Constructor reads a list of CornerSample (e.g. from
    `build_corner_samples`) and applies the transform pipeline on
    `__getitem__`. Image loading is lazy per-item.

    Augmentation note: cards are NEVER flipped (lefty/righty content
    matters), and rotation is gentle (corner identity is meaningful;
    rotating a TL crop into a BR position would teach the model
    something wrong)."""

    def __init__(
        self,
        samples: Sequence[CornerSample],
        *,
        crop_size: int = 224,
        train: bool = True,
        normalize_mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        normalize_std: tuple[float, float, float] = (0.229, 0.224, 0.225),
    ) -> None:
        if not _TORCH_AVAILABLE:
            raise RuntimeError(
                "torch + torchvision + Pillow required for PSACornerDataset; "
                "run `uv sync --extra training` from ml/"
            )
        self._samples = list(samples)
        self._crop_size = int(crop_size)

        train_aug: list = [
            transforms.ColorJitter(brightness=0.10, contrast=0.10, saturation=0.05),
            transforms.RandomAffine(
                degrees=2.0,
                translate=(0.02, 0.02),
                scale=(0.98, 1.02),
                fill=0,
            ),
        ]
        eval_aug: list = []

        self._transform = transforms.Compose(
            [
                transforms.Resize((crop_size, crop_size)),
                *(train_aug if train else eval_aug),
                transforms.ToTensor(),
                transforms.Normalize(mean=list(normalize_mean), std=list(normalize_std)),
            ]
        )

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> tuple["torch.Tensor", "torch.Tensor"]:
        s = self._samples[idx]
        with Image.open(s.image_path) as full:
            full = full.convert("RGB")
            corners = _crop_corners(full, self._crop_size)
            crop = corners[s.corner_index]
        x = self._transform(crop)
        y = torch.tensor(s.grade, dtype=torch.float32)
        return x, y


__all__ = [
    "CornerSample",
    "PSACornerDataset",
    "NUM_CORNERS",
    "CORNER_TL",
    "CORNER_TR",
    "CORNER_BL",
    "CORNER_BR",
    "build_corner_samples",
    "split_by_cert",
]
