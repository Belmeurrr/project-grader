"""PSASurface dataset tests.

Stdlib-only path (manifest builder + splitter + helpers) is exercised
without torch. The torch-dependent path (PSASurfaceDataset.__getitem__)
is exercised when torch + Pillow are available; otherwise gracefully
skipped so the tests still run in stdlib-only environments.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Ensure ml/ root is on sys.path when running this test directly.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.datasets.psa_surface import (
    BACKGROUND_CLASS,
    DEFECT_CLASSES,
    NUM_CLASSES,
    SurfaceSample,
    all_background_mask_loader,
    build_surface_samples,
    split_by_cert,
)


# Detect torch availability for the optional PSASurfaceDataset tests.
try:
    import numpy as np
    import torch
    from PIL import Image

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TORCH_AVAILABLE = False


# --------------------------------------------------------------------------
# Class taxonomy invariants
# --------------------------------------------------------------------------


def test_class_taxonomy_starts_with_background() -> None:
    assert DEFECT_CLASSES[BACKGROUND_CLASS] == "background"
    assert NUM_CLASSES == len(DEFECT_CLASSES)


def test_class_taxonomy_includes_all_planned_classes() -> None:
    expected = {
        "background",
        "scratch",
        "print_line",
        "indentation",
        "stain",
        "paper_loss",
        "foil_scratch",
    }
    assert set(DEFECT_CLASSES) == expected


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
        "card_name": f"Card {cert_id}",
        "set_name": "Test Set",
        "year": 2020,
        "card_number": str(cert_id),
        "front_image_path": str(front) if image_exists else None,
        "back_image_path": None,
        "source_url": f"https://x/{cert_id}",
        "scraped_at": "2026-04-29T00:00:00Z",
    }


def test_build_surface_samples_emits_one_per_record(tmp_path: Path) -> None:
    jsonl = tmp_path / "scraped.jsonl"
    _write_jsonl(jsonl, [
        _record(cert_id=1, tmp_path=tmp_path),
        _record(cert_id=2, tmp_path=tmp_path),
    ])
    samples = build_surface_samples(jsonl)
    assert len(samples) == 2
    assert {s.cert_id for s in samples} == {1, 2}
    assert all(isinstance(s, SurfaceSample) for s in samples)


def test_build_surface_samples_skips_records_without_front_image_path(tmp_path: Path) -> None:
    jsonl = tmp_path / "scraped.jsonl"
    _write_jsonl(jsonl, [
        _record(cert_id=1, tmp_path=tmp_path),
        _record(cert_id=2, image_exists=False, tmp_path=tmp_path),
    ])
    samples = build_surface_samples(jsonl)
    assert [s.cert_id for s in samples] == [1]


def test_build_surface_samples_skips_records_with_missing_image_file(tmp_path: Path) -> None:
    jsonl = tmp_path / "scraped.jsonl"
    rec = _record(cert_id=1, tmp_path=tmp_path)
    # Record claims a path but we delete the file.
    Path(rec["front_image_path"]).unlink()
    _write_jsonl(jsonl, [rec])
    samples = build_surface_samples(jsonl)
    assert samples == []


def test_build_surface_samples_filters_grades_outside_range(tmp_path: Path) -> None:
    jsonl = tmp_path / "scraped.jsonl"
    _write_jsonl(jsonl, [
        _record(cert_id=1, grade=0.5, tmp_path=tmp_path),  # below
        _record(cert_id=2, grade=5.0, tmp_path=tmp_path),  # ok
        _record(cert_id=3, grade=11.0, tmp_path=tmp_path),  # above
    ])
    samples = build_surface_samples(jsonl)
    assert [s.cert_id for s in samples] == [2]


def test_build_surface_samples_preserves_full_record_for_mask_loader(tmp_path: Path) -> None:
    """The whole JSONL row is kept on the SurfaceSample so a future
    mask_loader can use any field (set_name, year, etc.) without
    re-reading the manifest."""
    jsonl = tmp_path / "scraped.jsonl"
    rec = _record(cert_id=1, tmp_path=tmp_path)
    rec["custom_field"] = "future-mask-loader-input"
    _write_jsonl(jsonl, [rec])
    samples = build_surface_samples(jsonl)
    assert samples[0].record["custom_field"] == "future-mask-loader-input"


def test_build_surface_samples_tolerates_malformed_lines(tmp_path: Path) -> None:
    """A truncated trailing JSONL line (e.g. process killed mid-write)
    must not abort the build — the trainer should still see all the
    records that DID land cleanly."""
    jsonl = tmp_path / "scraped.jsonl"
    rec = _record(cert_id=1, tmp_path=tmp_path)
    jsonl.write_text(
        json.dumps(rec) + "\n"
        + '{"cert_id": 99, "grade": 9.0, "front_image_path": "x',  # truncated
        encoding="utf-8",
    )
    samples = build_surface_samples(jsonl)
    assert [s.cert_id for s in samples] == [1]


# --------------------------------------------------------------------------
# Split helper
# --------------------------------------------------------------------------


def _samples_for_certs(cert_ids: list[int], tmp_path: Path) -> list[SurfaceSample]:
    """Convenience: build SurfaceSamples from a list of cert_ids without
    going through JSONL."""
    out: list[SurfaceSample] = []
    for cid in cert_ids:
        path = tmp_path / f"img_{cid}.jpg"
        path.write_bytes(b"")
        out.append(SurfaceSample(
            cert_id=cid,
            image_path=path,
            grade=9.0,
            record={"cert_id": cid, "grade": 9.0},
        ))
    return out


def test_split_by_cert_keeps_card_intact(tmp_path: Path) -> None:
    """No cert_id appears in both the train and val sides — preserves the
    invariant the corners trainer's split relies on."""
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
# Mask loader contract
# --------------------------------------------------------------------------


def test_default_mask_loader_returns_none() -> None:
    """The skeleton's default loader returns None for every record so
    the dataset falls back to all-background. Documented contract; the
    trainer's smoke run depends on this."""
    assert all_background_mask_loader({"cert_id": 1, "grade": 9.0}) is None


# --------------------------------------------------------------------------
# Dataset (torch path — gracefully skipped when torch isn't installed)
# --------------------------------------------------------------------------


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="torch / Pillow not installed")
def test_dataset_yields_image_tensor_and_full_res_mask(tmp_path: Path) -> None:
    from training.datasets.psa_surface import PSASurfaceDataset

    img_path = tmp_path / "img.jpg"
    Image.new("RGB", (300, 420), color=(50, 100, 150)).save(img_path, "JPEG")
    sample = SurfaceSample(
        cert_id=1,
        image_path=img_path,
        grade=9.0,
        record={"cert_id": 1, "grade": 9.0},
    )

    ds = PSASurfaceDataset([sample], image_size=64, train=False)
    x, y = ds[0]
    assert x.shape == (3, 64, 64)
    assert y.shape == (64, 64)
    assert y.dtype == torch.int64
    # All-background placeholder by default.
    assert int(y.sum().item()) == 0


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="torch / Pillow not installed")
def test_dataset_uses_provided_mask_loader_for_real_labels(tmp_path: Path) -> None:
    from training.datasets.psa_surface import PSASurfaceDataset

    img_path = tmp_path / "img.jpg"
    Image.new("RGB", (300, 420), color=(50, 100, 150)).save(img_path, "JPEG")
    sample = SurfaceSample(
        cert_id=1,
        image_path=img_path,
        grade=9.0,
        record={"cert_id": 1, "grade": 9.0},
    )

    def loader(record: dict) -> "np.ndarray":
        # Return a mask with a single "scratch"-class square in the
        # top-left corner. Simulates a real labeled mask.
        m = np.zeros((64, 64), dtype=np.int64)
        m[:8, :8] = 1  # class index 1 = "scratch"
        return m

    ds = PSASurfaceDataset(
        [sample], image_size=64, train=False, mask_loader=loader,
    )
    _, y = ds[0]
    assert int(y[0, 0].item()) == 1  # the scratch
    assert int(y[63, 63].item()) == 0  # background elsewhere


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="torch / Pillow not installed")
def test_dataset_falls_back_when_loader_returns_wrong_shape(tmp_path: Path) -> None:
    """A loader returning a mask of the wrong shape is a defect-loader
    bug; the dataset shouldn't crash the run, it should warn and fall
    back to all-background so training continues."""
    from training.datasets.psa_surface import PSASurfaceDataset

    img_path = tmp_path / "img.jpg"
    Image.new("RGB", (200, 200), color=(0, 0, 0)).save(img_path, "JPEG")
    sample = SurfaceSample(
        cert_id=1,
        image_path=img_path,
        grade=9.0,
        record={},
    )

    def bad_loader(record: dict) -> "np.ndarray":
        return np.ones((32, 32), dtype=np.int64)  # wrong size

    ds = PSASurfaceDataset(
        [sample], image_size=64, train=False, mask_loader=bad_loader,
    )
    _, y = ds[0]
    assert y.shape == (64, 64)
    assert int(y.sum().item()) == 0  # all-bg fallback


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="torch / Pillow not installed")
def test_dataset_clips_out_of_range_class_indices(tmp_path: Path) -> None:
    """If a future loader returns an out-of-range class (because the
    ontology was extended without updating the loader), those pixels
    should be silently clipped to background rather than crashing the
    loss with index-out-of-range."""
    from training.datasets.psa_surface import PSASurfaceDataset

    img_path = tmp_path / "img.jpg"
    Image.new("RGB", (200, 200), color=(0, 0, 0)).save(img_path, "JPEG")
    sample = SurfaceSample(
        cert_id=1,
        image_path=img_path,
        grade=9.0,
        record={},
    )

    def loader(record: dict) -> "np.ndarray":
        m = np.zeros((64, 64), dtype=np.int64)
        m[:8, :8] = 999  # way out of range
        m[8:16, :8] = -1  # negative
        m[16:24, :8] = 2  # legitimate "print_line"
        return m

    ds = PSASurfaceDataset(
        [sample], image_size=64, train=False, mask_loader=loader,
    )
    _, y = ds[0]
    # Out-of-range and negative both clip to background.
    assert int(y[0, 0].item()) == BACKGROUND_CLASS
    assert int(y[8, 0].item()) == BACKGROUND_CLASS
    # Legitimate class index passes through untouched.
    assert int(y[16, 0].item()) == 2


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="torch / Pillow not installed")
def test_dataset_stacks_into_batch(tmp_path: Path) -> None:
    """A DataLoader's default collate must work — i.e. all (x, y) pairs
    should have stack-compatible shapes. This catches accidental shape
    drift between samples (e.g. if augmentation introduced randomness
    in output size)."""
    from torch.utils.data import DataLoader

    from training.datasets.psa_surface import PSASurfaceDataset

    samples: list[SurfaceSample] = []
    for i in range(3):
        p = tmp_path / f"img_{i}.jpg"
        Image.new("RGB", (200, 280), color=(i * 80, 40, 40)).save(p, "JPEG")
        samples.append(SurfaceSample(
            cert_id=i, image_path=p, grade=9.0, record={"cert_id": i},
        ))

    ds = PSASurfaceDataset(samples, image_size=64, train=False)
    loader = DataLoader(ds, batch_size=3, shuffle=False, num_workers=0)
    x, y = next(iter(loader))
    assert x.shape == (3, 3, 64, 64)
    assert y.shape == (3, 64, 64)
