"""PSAIdentification dataset tests.

Stdlib-only path (manifest builder + splitter + triplet-index sampler)
is exercised without torch. The torch-dependent path
(IdentificationTripletDataset.__getitem__ + DataLoader collate) is
exercised when torch + Pillow are available; otherwise gracefully
skipped so the tests still run in stdlib-only environments.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import pytest

# Ensure ml/ root is on sys.path when running this test directly.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.datasets.psa_identification import (
    IdentificationSample,
    build_identification_samples,
    sample_triplet_indices,
    split_by_cert,
)


# Detect torch availability for the optional Dataset tests.
try:
    import torch
    from PIL import Image

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TORCH_AVAILABLE = False


# --------------------------------------------------------------------------
# Manifest builder
# --------------------------------------------------------------------------


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")


def _record(
    cert_id: int = 1,
    grade: float = 9.0,
    image_exists: bool = True,
    tmp_path: Path | None = None,
    card_name: str = "Card",
    set_name: str = "Test Set",
) -> dict:
    """Build a minimal scraped.jsonl row. If `image_exists` is True an
    empty file is created at the front_image_path so the dataset's
    `Path.exists()` check passes."""
    front = (tmp_path or Path()) / f"img_{cert_id}.jpg"
    if image_exists and tmp_path is not None:
        front.write_bytes(b"\xff\xd8\xff\xd9")  # smallest possible "JPEG"
    return {
        "cert_id": cert_id,
        "grade": grade,
        "card_name": f"{card_name} {cert_id}",
        "set_name": set_name,
        "year": 2020,
        "card_number": str(cert_id),
        "front_image_path": str(front) if image_exists else None,
        "back_image_path": None,
        "source_url": f"https://x/{cert_id}",
        "scraped_at": "2026-04-29T00:00:00Z",
    }


def test_build_identification_samples_emits_one_per_record(tmp_path: Path) -> None:
    jsonl = tmp_path / "scraped.jsonl"
    _write_jsonl(jsonl, [
        _record(cert_id=1, tmp_path=tmp_path),
        _record(cert_id=2, tmp_path=tmp_path),
    ])
    samples = build_identification_samples(jsonl)
    assert len(samples) == 2
    assert {s.cert_id for s in samples} == {1, 2}
    assert all(isinstance(s, IdentificationSample) for s in samples)


def test_build_identification_samples_skips_missing_front_image_path(tmp_path: Path) -> None:
    jsonl = tmp_path / "scraped.jsonl"
    _write_jsonl(jsonl, [
        _record(cert_id=1, tmp_path=tmp_path),
        _record(cert_id=2, image_exists=False, tmp_path=tmp_path),
    ])
    samples = build_identification_samples(jsonl)
    assert [s.cert_id for s in samples] == [1]


def test_build_identification_samples_skips_records_with_missing_image_file(tmp_path: Path) -> None:
    jsonl = tmp_path / "scraped.jsonl"
    rec = _record(cert_id=1, tmp_path=tmp_path)
    Path(rec["front_image_path"]).unlink()
    _write_jsonl(jsonl, [rec])
    samples = build_identification_samples(jsonl)
    assert samples == []


def test_build_identification_samples_filters_grades_outside_range(tmp_path: Path) -> None:
    jsonl = tmp_path / "scraped.jsonl"
    _write_jsonl(jsonl, [
        _record(cert_id=1, grade=0.5, tmp_path=tmp_path),
        _record(cert_id=2, grade=5.0, tmp_path=tmp_path),
        _record(cert_id=3, grade=11.0, tmp_path=tmp_path),
    ])
    samples = build_identification_samples(jsonl)
    assert [s.cert_id for s in samples] == [2]


def test_build_identification_samples_captures_card_and_set_name(tmp_path: Path) -> None:
    """card_name + set_name must be on the IdentificationSample so a
    future string-matched-positive sampler can use them without re-
    reading the JSONL."""
    jsonl = tmp_path / "scraped.jsonl"
    _write_jsonl(jsonl, [
        _record(cert_id=1, tmp_path=tmp_path, card_name="Pikachu", set_name="Base"),
    ])
    samples = build_identification_samples(jsonl)
    assert samples[0].card_name == "Pikachu 1"
    assert samples[0].set_name == "Base"


def test_build_identification_samples_preserves_full_record(tmp_path: Path) -> None:
    """The whole JSONL row is kept on the IdentificationSample so a
    future positive sampler can use any field (year / variant / number)
    without re-reading the manifest."""
    jsonl = tmp_path / "scraped.jsonl"
    rec = _record(cert_id=1, tmp_path=tmp_path)
    rec["custom_field"] = "future-positive-sampler-input"
    _write_jsonl(jsonl, [rec])
    samples = build_identification_samples(jsonl)
    assert samples[0].record["custom_field"] == "future-positive-sampler-input"


def test_build_identification_samples_tolerates_malformed_lines(tmp_path: Path) -> None:
    """A truncated trailing JSONL line (e.g. process killed mid-write)
    must not abort the build — the trainer should still see all the
    records that DID land cleanly. Mirrors the surface-dataset test."""
    jsonl = tmp_path / "scraped.jsonl"
    rec = _record(cert_id=1, tmp_path=tmp_path)
    jsonl.write_text(
        json.dumps(rec) + "\n"
        + '{"cert_id": 99, "grade": 9.0, "front_image_path": "x',  # truncated
        encoding="utf-8",
    )
    samples = build_identification_samples(jsonl)
    assert [s.cert_id for s in samples] == [1]


# --------------------------------------------------------------------------
# Split helper
# --------------------------------------------------------------------------


def _samples_for_certs(cert_ids: list[int], tmp_path: Path) -> list[IdentificationSample]:
    """Convenience: build IdentificationSamples from a list of cert_ids
    without going through JSONL."""
    out: list[IdentificationSample] = []
    for cid in cert_ids:
        path = tmp_path / f"img_{cid}.jpg"
        path.write_bytes(b"")
        out.append(IdentificationSample(
            cert_id=cid,
            image_path=path,
            grade=9.0,
            card_name=f"Card {cid}",
            set_name="Test",
            record={"cert_id": cid, "grade": 9.0},
        ))
    return out


def test_split_by_cert_keeps_card_intact(tmp_path: Path) -> None:
    samples = _samples_for_certs(list(range(20)), tmp_path)
    train, val = split_by_cert(samples, val_fraction=0.25, seed=42)
    train_certs = {s.cert_id for s in train}
    val_certs = {s.cert_id for s in val}
    assert not (train_certs & val_certs)


def test_split_by_cert_is_deterministic_on_seed(tmp_path: Path) -> None:
    samples = _samples_for_certs(list(range(30)), tmp_path)
    a_train, a_val = split_by_cert(samples, val_fraction=0.2, seed=123)
    b_train, b_val = split_by_cert(samples, val_fraction=0.2, seed=123)
    assert [s.cert_id for s in a_train] == [s.cert_id for s in b_train]
    assert [s.cert_id for s in a_val] == [s.cert_id for s in b_val]


def test_split_by_cert_handles_empty_input() -> None:
    train, val = split_by_cert([], val_fraction=0.2)
    assert train == []
    assert val == []


# --------------------------------------------------------------------------
# Triplet-index sampler (pure stdlib)
# --------------------------------------------------------------------------


def test_sample_triplet_indices_returns_anchor_as_positive() -> None:
    """Skeleton's positive strategy: anchor and positive are the SAME
    sample (the Dataset applies two random transforms to it). The
    sampler enforces this so a future swap-in (string-matched
    positives) is a single-function change."""
    rng = random.Random(0)
    a, p, n = sample_triplet_indices(10, anchor_index=3, rng=rng)
    assert a == 3
    assert p == 3


def test_sample_triplet_indices_negative_differs_from_anchor() -> None:
    """Negative must NEVER equal the anchor index — otherwise the
    triplet collapses (anchor==positive==negative implies zero
    gradient with margin>0 and an all-zero embedding fixed point)."""
    rng = random.Random(0)
    for _ in range(50):
        a, _, n = sample_triplet_indices(5, anchor_index=2, rng=rng)
        assert a == 2
        assert n != 2


def test_sample_triplet_indices_explores_all_negatives() -> None:
    """Over many draws, the negative sampler must hit every non-anchor
    index. This guards against an off-by-one in a future refactor that
    could otherwise silently exclude the last index."""
    rng = random.Random(0)
    n_samples = 6
    anchor = 0
    seen_negatives: set[int] = set()
    for _ in range(500):
        _, _, neg = sample_triplet_indices(n_samples, anchor_index=anchor, rng=rng)
        seen_negatives.add(neg)
    assert seen_negatives == {1, 2, 3, 4, 5}


def test_sample_triplet_indices_rejects_singleton_corpus() -> None:
    rng = random.Random(0)
    with pytest.raises(ValueError, match="at least 2 samples"):
        sample_triplet_indices(1, anchor_index=0, rng=rng)


def test_sample_triplet_indices_rejects_out_of_range_anchor() -> None:
    rng = random.Random(0)
    with pytest.raises(ValueError, match="out of range"):
        sample_triplet_indices(5, anchor_index=10, rng=rng)


# --------------------------------------------------------------------------
# Dataset (torch path — gracefully skipped when torch isn't installed)
# --------------------------------------------------------------------------


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="torch / Pillow not installed")
def test_dataset_returns_three_tensors_of_correct_shape(tmp_path: Path) -> None:
    from training.datasets.psa_identification import IdentificationTripletDataset

    samples: list[IdentificationSample] = []
    for i in range(4):
        p = tmp_path / f"img_{i}.jpg"
        Image.new("RGB", (256, 256), color=(i * 60, 40, 80)).save(p, "JPEG")
        samples.append(IdentificationSample(
            cert_id=i,
            image_path=p,
            grade=9.0,
            card_name=f"Card {i}",
            set_name="Test",
            record={"cert_id": i},
        ))

    ds = IdentificationTripletDataset(samples, image_size=64, train=True, seed=0)
    a, p_t, n = ds[0]
    assert a.shape == (3, 64, 64)
    assert p_t.shape == (3, 64, 64)
    assert n.shape == (3, 64, 64)


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="torch / Pillow not installed")
def test_dataset_anchor_and_positive_differ_under_train_aug(tmp_path: Path) -> None:
    """Augmentation-based positive: anchor and positive come from the
    SAME image but two independent random transforms must give two
    distinguishable tensors. If they're identical, RandomResizedCrop /
    ColorJitter aren't actually random and the skeleton's
    positive-sampling strategy is broken."""
    from training.datasets.psa_identification import IdentificationTripletDataset

    samples: list[IdentificationSample] = []
    for i in range(3):
        p = tmp_path / f"img_{i}.jpg"
        # A non-uniform image so RandomResizedCrop + ColorJitter actually
        # make a difference. A uniform color image would be invariant to
        # crop / brightness within the same image.
        img = Image.new("RGB", (256, 256), color=(0, 0, 0))
        # paint a diagonal so different crops capture different content
        for k in range(256):
            img.putpixel((k, k), (255, 255, 255))
            if k + 1 < 256:
                img.putpixel((k, k + 1), (255, 200, 0))
        img.save(p, "JPEG")
        samples.append(IdentificationSample(
            cert_id=i,
            image_path=p,
            grade=9.0,
            card_name=f"Card {i}",
            set_name="Test",
            record={"cert_id": i},
        ))

    ds = IdentificationTripletDataset(samples, image_size=64, train=True, seed=0)
    a, p_t, _ = ds[0]
    # The two augmented views of the same image should NOT be
    # bit-identical. Equality tolerance is exact-byte because the
    # transforms are stochastic.
    assert not torch.equal(a, p_t)


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="torch / Pillow not installed")
def test_dataset_eval_mode_makes_anchor_and_positive_identical(tmp_path: Path) -> None:
    """Eval mode uses a deterministic transform — anchor and positive
    must be byte-identical because the same image goes through the
    same transform twice. This guards the val-time
    positive-distance-distribution metric (which expects ~0 for the
    skeleton's same-image positive)."""
    from training.datasets.psa_identification import IdentificationTripletDataset

    samples: list[IdentificationSample] = []
    for i in range(3):
        p = tmp_path / f"img_{i}.jpg"
        Image.new("RGB", (200, 280), color=(i * 80, 40, 40)).save(p, "JPEG")
        samples.append(IdentificationSample(
            cert_id=i,
            image_path=p,
            grade=9.0,
            card_name=f"Card {i}",
            set_name="Test",
            record={"cert_id": i},
        ))

    ds = IdentificationTripletDataset(samples, image_size=64, train=False, seed=0)
    a, p_t, _ = ds[0]
    assert torch.equal(a, p_t)


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="torch / Pillow not installed")
def test_dataset_negative_image_differs_from_anchor(tmp_path: Path) -> None:
    """Negative sample comes from a DIFFERENT record than the anchor —
    which means a different on-disk image with a different mean color.
    Sanity-check: the negative tensor should not be identical to the
    anchor tensor in eval mode."""
    from training.datasets.psa_identification import IdentificationTripletDataset

    samples: list[IdentificationSample] = []
    for i in range(3):
        p = tmp_path / f"img_{i}.jpg"
        # Distinct colors per image so the negative tensor is
        # distinguishable from the anchor by mean intensity.
        Image.new("RGB", (200, 200), color=(i * 100, 0, 0)).save(p, "JPEG")
        samples.append(IdentificationSample(
            cert_id=i,
            image_path=p,
            grade=9.0,
            card_name=f"Card {i}",
            set_name="Test",
            record={"cert_id": i},
        ))

    ds = IdentificationTripletDataset(samples, image_size=64, train=False, seed=0)
    a, _, n = ds[0]
    assert not torch.equal(a, n)


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="torch / Pillow not installed")
def test_dataset_collates_into_batch(tmp_path: Path) -> None:
    """A DataLoader's default collate must work on triplets — i.e. all
    three tensors per item must have stack-compatible shapes. This
    catches accidental shape drift between samples (e.g. if augmentation
    introduced randomness in output size)."""
    from torch.utils.data import DataLoader

    from training.datasets.psa_identification import IdentificationTripletDataset

    samples: list[IdentificationSample] = []
    for i in range(4):
        p = tmp_path / f"img_{i}.jpg"
        Image.new("RGB", (200, 280), color=(i * 60, 40, 40)).save(p, "JPEG")
        samples.append(IdentificationSample(
            cert_id=i,
            image_path=p,
            grade=9.0,
            card_name=f"Card {i}",
            set_name="Test",
            record={"cert_id": i},
        ))

    ds = IdentificationTripletDataset(samples, image_size=64, train=False, seed=0)
    loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)
    a, p_t, n = next(iter(loader))
    assert a.shape == (4, 3, 64, 64)
    assert p_t.shape == (4, 3, 64, 64)
    assert n.shape == (4, 3, 64, 64)


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="torch / Pillow not installed")
def test_dataset_rejects_singleton_corpus(tmp_path: Path) -> None:
    """A 1-sample corpus has no possible negative — the Dataset
    constructor must reject it loudly rather than infinite-loop in the
    negative sampler."""
    from training.datasets.psa_identification import IdentificationTripletDataset

    p = tmp_path / "img.jpg"
    Image.new("RGB", (200, 200), color=(0, 0, 0)).save(p, "JPEG")
    sample = IdentificationSample(
        cert_id=1, image_path=p, grade=9.0,
        card_name="x", set_name="y", record={},
    )
    with pytest.raises(ValueError, match="at least 2 samples"):
        IdentificationTripletDataset([sample], image_size=64, train=True)
