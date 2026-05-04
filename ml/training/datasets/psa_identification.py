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

Positive sampling — supervised metric learning via name+set keys:
    The PSA Public API records carry `card_name` + `set_name` as free-
    text strings, not a stable variant_id. We synthesize a stable key
    by normalizing those two strings (lowercase, strip, collapse
    internal whitespace) and group samples by `(card_name, set_name)`.

    The positive sampler picks a uniformly random sample with the SAME
    key but a DIFFERENT `cert_id`. That trains the embedding to be
    invariant to the things that vary across two PSA-graded copies of
    the same printing — lighting, holder reflection, slab orientation,
    sensor noise — while staying separated from any other printing.

    Singleton keys (only one cert in the group) cannot supply a real
    positive, so they're dropped at manifest-build time. The trainer
    sees a strictly smaller corpus, but every remaining sample has at
    least one valid same-key, different-cert positive partner.

    Negatives stay random-different (uniform across the whole corpus).
    The trainer + loss are unchanged from the augmentation-only era.

    Why not ArcFace / supervised contrastive yet:
    Both want a stable variant_id. The (card_name, set_name) key is a
    reasonable proxy but it's noisy at the edges (typos, alt-art string
    variants, language). Triplet tolerates the noise; ArcFace's softmax
    over class ids would amplify it. Promote when the key gets cleaned.

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


def _normalize_key_part(s: str) -> str:
    """Normalize a card_name or set_name component for keying.

    Lowercase, strip surrounding whitespace, and collapse internal
    runs of whitespace to a single space. Two records with cosmetic
    differences ("  Pikachu  ", "pikachu") collide on the same key
    and become positive partners for each other.

    Conservative on purpose: we don't strip punctuation, normalize
    unicode, or split on `/` — that would risk wrong merges (e.g.
    "Mewtwo" vs "Mewtwo & Mew" need to stay distinct). The cost is
    that very mild typo variants stay separate; we'd rather drop a
    few singletons than poison a class with a wrong member."""
    return " ".join(str(s).lower().split())


def _build_key(card_name: str, set_name: str) -> tuple[str, str]:
    """Compute the supervised metric-learning key for a sample.

    Returns the `(normalized_card_name, normalized_set_name)` tuple
    used to group same-printing samples together for positive
    sampling."""
    return (_normalize_key_part(card_name), _normalize_key_part(set_name))


@dataclass(frozen=True)
class IdentificationSample:
    """One identification training example.

    Pre-tensor representation. The Dataset.__getitem__ converts this
    into a (anchor, positive, negative) tensor triple via the transform
    pipeline + a same-key-different-cert positive sampler + a random-
    different-sample negative sampler.

    `key` is the normalized `(card_name, set_name)` tuple — see
    `_build_key`. It's the supervision signal that lets two distinct
    PSA-graded copies of the same printing pair up as anchor +
    positive."""

    cert_id: int
    image_path: Path
    grade: float
    card_name: str
    set_name: str
    # Whole record kept around so a future positive sampler can use any
    # field it needs (year / variant / number) without re-reading the
    # JSONL.
    record: dict
    # Normalized (card_name, set_name) — populated by
    # `build_identification_samples`. Tests that construct
    # IdentificationSample directly may pass an explicit key or rely
    # on the post-init default below.
    key: tuple[str, str] = ("", "")

    def __post_init__(self) -> None:
        # If the caller didn't pass an explicit key, derive it from
        # the (card_name, set_name) on the sample. Frozen dataclass
        # so we have to bypass the field guard via object.__setattr__.
        if self.key == ("", ""):
            object.__setattr__(
                self, "key", _build_key(self.card_name, self.set_name)
            )


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
    `require_grade_in` are skipped.

    Samples are then grouped by their normalized `(card_name, set_name)`
    key. Any sample whose key group has fewer than two distinct
    `cert_id`s is DROPPED — a singleton key has no real positive
    partner under the same-key, different-cert positive sampling
    strategy, so feeding it to the trainer would either bias the loss
    (anchor==positive) or poison the negative pool. The drop count is
    logged at INFO so operators can see how much the corpus shrinks.

    The returned list length is the authoritative sample count
    (the trainer's `min_samples` gate is measured against this).
    """
    raw: list[IdentificationSample] = []
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
        card_name = str(rec.get("card_name") or "")
        set_name = str(rec.get("set_name") or "")
        raw.append(
            IdentificationSample(
                cert_id=int(cert_id),
                image_path=front_path,
                grade=float(grade),
                card_name=card_name,
                set_name=set_name,
                record=rec,
                key=_build_key(card_name, set_name),
            )
        )

    # Group by key, then drop samples whose group has fewer than two
    # distinct cert_ids. A "different cert" positive partner is required
    # for the metric-learning sampler.
    groups: dict[tuple[str, str], list[IdentificationSample]] = {}
    for s in raw:
        groups.setdefault(s.key, []).append(s)

    kept: list[IdentificationSample] = []
    dropped = 0
    for key, group in groups.items():
        distinct_certs = {s.cert_id for s in group}
        if len(distinct_certs) < 2:
            dropped += len(group)
            continue
        kept.extend(group)

    if dropped:
        _logger.info(
            "dropped %d singleton-key sample(s) without a same-key, "
            "different-cert positive partner (kept %d / %d total)",
            dropped, len(kept), len(raw),
        )
    return kept


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


def build_key_index(
    samples: Sequence[IdentificationSample],
) -> dict[tuple[str, str], list[int]]:
    """Group sample positions by their normalized `(card_name, set_name)` key.

    Returned mapping is `key -> [index, index, ...]`. Used by the
    positive sampler to look up same-key candidates without scanning
    the full sample list each call.

    Pure stdlib; exposed for the test harness."""
    out: dict[tuple[str, str], list[int]] = {}
    for i, s in enumerate(samples):
        out.setdefault(s.key, []).append(i)
    return out


def sample_triplet_indices(
    samples: Sequence[IdentificationSample],
    anchor_index: int,
    *,
    rng: random.Random,
    key_index: dict[tuple[str, str], list[int]] | None = None,
) -> tuple[int, int, int]:
    """Pure-stdlib triplet-index sampler.

    Returns `(anchor_index, positive_index, negative_index)` where:
      - `positive_index` is a uniformly random sample whose key matches
        the anchor's but whose `cert_id` differs (supervised metric
        learning — see module docstring).
      - `negative_index` is a uniform-random different sample.

    `key_index` is the precomputed grouping; if omitted we build it
    from `samples` (slower for hot loops, fine for tests).

    Raises `ValueError` if there's no possible negative (fewer than 2
    samples) or if the anchor's key has no other-cert partner. The
    latter shouldn't happen for samples produced by
    `build_identification_samples` (it drops singletons), but the
    error is loud rather than silently fall back.

    Exposed as a top-level function so the test harness can exercise
    the sampler logic without instantiating the torch Dataset."""
    n_samples = len(samples)
    if n_samples < 2:
        raise ValueError(
            f"need at least 2 samples to draw a triplet; got n={n_samples}"
        )
    if not (0 <= anchor_index < n_samples):
        raise ValueError(
            f"anchor_index={anchor_index} out of range for n={n_samples}"
        )

    if key_index is None:
        key_index = build_key_index(samples)

    anchor = samples[anchor_index]
    candidates = [
        i for i in key_index.get(anchor.key, ())
        if samples[i].cert_id != anchor.cert_id
    ]
    if not candidates:
        raise ValueError(
            f"no same-key, different-cert positive partner for anchor "
            f"index={anchor_index} key={anchor.key!r} cert_id={anchor.cert_id}"
            " — corpus likely wasn't filtered through "
            "build_identification_samples (which drops singletons)"
        )
    pos = rng.choice(candidates)

    # Sample a different index for negative. Reject-sample rather than
    # build a candidate list so we stay O(1) regardless of n_samples.
    while True:
        neg = rng.randrange(n_samples)
        if neg != anchor_index:
            return anchor_index, pos, neg


class IdentificationTripletDataset(Dataset):  # type: ignore[misc]
    """torch Dataset of (anchor, positive, negative) triplets.

    Constructor reads a list of IdentificationSample (e.g. from
    `build_identification_samples`) and applies the augmentation
    pipeline on `__getitem__`. Image loading is lazy per-item.

    Args:
      samples: pre-tensor manifest from `build_identification_samples`.
        Singleton-key samples MUST already be filtered out (the
        manifest builder does this); otherwise the positive sampler
        will fall back to anchor==positive and emit a warning.
      image_size: side length of the square model input. DinoV2 ViT-B/14
        wants multiples of 14; defaults vary, the trainer config sets
        the canonical value.
      train: training-mode augmentations on/off. Eval mode applies a
        deterministic transform but the anchor + positive still come
        from DIFFERENT records (same key, different cert) so they're
        not byte-identical the way the augmentation-only era was — by
        design, since we now want to measure how well the embedding
        generalizes across two distinct copies of the same printing.
      seed: RNG seed for both the positive AND negative samplers.
        Fixed seed makes the train loop reproducible. Workers re-seed
        via worker_init_fn in the trainer if num_workers > 0.

    Augmentation:
      Anchor and positive are separately-sourced images, so the
      augmentation gap that mattered in the augmentation-only era is
      no longer load-bearing. We keep the same RandomResizedCrop +
      ColorJitter + light RandomAffine because they still help the
      model generalize across crop / lighting / minor pose drift.

      Flips are NOT applied (card content is left/right meaningful).
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
        # Precomputed (card_name, set_name) → [sample_index, ...]. The
        # positive sampler hits this for an O(1) candidate-list lookup.
        self._key_index = build_key_index(self._samples)

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
        """Same-key, different-cert positive sampler.

        Looks up the precomputed `_key_index` for samples sharing the
        anchor's normalized `(card_name, set_name)` key, filters out
        any whose `cert_id` matches the anchor's, and picks one
        uniformly at random.

        If somehow no candidate is eligible (shouldn't happen because
        `build_identification_samples` drops singleton keys), falls
        back to the anchor index itself and logs a warning so the
        operator can investigate the upstream filter."""
        anchor = self._samples[anchor_index]
        candidates = [
            i for i in self._key_index.get(anchor.key, ())
            if self._samples[i].cert_id != anchor.cert_id
        ]
        if not candidates:
            _logger.warning(
                "no same-key, different-cert positive partner for "
                "anchor index=%d key=%r cert_id=%s; falling back to "
                "anchor itself (corpus probably wasn't filtered through "
                "build_identification_samples)",
                anchor_index, anchor.key, anchor.cert_id,
            )
            return anchor_index
        return self._rng.choice(candidates)

    def _sample_negative_index(self, anchor_index: int) -> int:
        n = len(self._samples)
        # Reject-sample uniformly. We don't filter by key here —
        # negatives are random-different across the entire corpus, so
        # an occasional same-key-different-cert pair will land in the
        # negative slot. That's accepted noise: the rate is O(group_size
        # / corpus_size) which is small once the corpus is real-sized,
        # and the loss tolerates it.
        while True:
            neg = self._rng.randrange(n)
            if neg != anchor_index:
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
        # Same-key, different-cert positive: a different on-disk image
        # of the same printing. The supervision signal is "two distinct
        # PSA-graded copies of the same card should embed close
        # together," not "an image and its augmentation."
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
    "build_key_index",
    "sample_triplet_indices",
    "split_by_cert",
]
