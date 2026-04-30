"""Tests for the reference embedding pass + storage.

Uses a deterministic fake embedder seeded from image bytes so tests
don't pull DinoV2 weights. Covers:
  - Empty JSONL is tolerated (no archive written? Yes, we still
    create one so the consumer side can rely on the file existing).
  - A fresh embedding pass writes one entry per JSONL record.
  - Re-running without `force` is idempotent — counts go to
    skipped_already_present, not embedded.
  - `force=True` overwrites existing entries.
  - Missing / unreadable image paths are counted, not raised.
  - Embedder failures are counted, not raised.
  - `lookup_references` returns the right (1, d) shape on hit and
    None on miss.
  - `load_embeddings` round-trips keys through the
    `(manufacturer, variant_id)` tuple form.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.ingestion.reference_embeddings import (  # noqa: E402
    EmbeddingPassStats,
    embed_references,
    load_embeddings,
    lookup_references,
)


# --------------------------------------------------------------------------
# Fakes / fixtures
# --------------------------------------------------------------------------


@dataclass
class _FakeEmbedder:
    """Deterministic embedder for tests.

    For each call, hashes the image bytes and uses the digest to seed
    a numpy RNG that produces a `(dim,)` standard-normal vector. The
    result is identical for identical input images and stable across
    test runs."""

    dim: int = 64

    def encode(self, image: np.ndarray) -> np.ndarray:
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("fake embedder expects uint8 BGR image")
        digest = sha256(image.tobytes()).digest()
        seed = int.from_bytes(digest[:8], "big")
        rng = np.random.default_rng(seed)
        return rng.standard_normal(self.dim).astype(np.float32)


@dataclass
class _FlakyEmbedder:
    """Embedder that raises for a configured set of image paths.

    The reference-embedding pass tolerates per-record embedder
    failures; we use this to verify the counter increments without
    aborting the run."""

    dim: int = 64
    fail_after: int = 1
    _calls: int = 0

    def encode(self, image: np.ndarray) -> np.ndarray:
        self._calls += 1
        if self._calls > self.fail_after:
            raise RuntimeError("intentional embedder failure")
        digest = sha256(image.tobytes()).digest()
        seed = int.from_bytes(digest[:8], "big")
        rng = np.random.default_rng(seed)
        return rng.standard_normal(self.dim).astype(np.float32)


def _write_image(path: Path, color: tuple[int, int, int]) -> None:
    """Write a tiny solid-color JPEG so cv2.imread succeeds."""
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.full((32, 32, 3), color, dtype=np.uint8)
    assert cv2.imwrite(str(path), img), f"failed to write fixture image {path}"


def _write_record(
    fh,
    *,
    manufacturer: str,
    variant_id: str,
    front_image_path: str | None,
    name: str = "Test Card",
) -> None:
    payload = {
        "variant_id": variant_id,
        "manufacturer": manufacturer,
        "name": name,
        "set_code": "TST",
        "collector_number": "1",
        "rarity": "common",
        "front_image_path": front_image_path,
        "source_url": "https://example.test/card",
        "fetched_at": "2026-04-29T00:00:00Z",
    }
    fh.write(json.dumps(payload) + "\n")


def _seed_dataset(
    root: Path,
    records: list[tuple[str, str, tuple[int, int, int] | None]],
) -> Path:
    """Create a references.jsonl + image fixtures under `root`.
    Each record is (manufacturer, variant_id, color_or_None).
    color=None means the front_image_path is set but the file is
    missing, simulating a failed image fetch."""
    images_root = root / "images"
    jsonl_path = root / "references.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as fh:
        for manufacturer, variant_id, color in records:
            img_path = images_root / manufacturer / variant_id / "front.jpg"
            if color is not None:
                _write_image(img_path, color)
            _write_record(
                fh,
                manufacturer=manufacturer,
                variant_id=variant_id,
                front_image_path=str(img_path),
            )
    return jsonl_path


# --------------------------------------------------------------------------
# embed_references — happy path + idempotency
# --------------------------------------------------------------------------


def test_embed_references_empty_jsonl_writes_empty_archive(tmp_path: Path) -> None:
    jsonl = tmp_path / "references.jsonl"
    jsonl.write_text("", encoding="utf-8")
    store = tmp_path / "reference_embeddings.npz"

    stats = embed_references(jsonl, store, embedder=_FakeEmbedder())

    assert stats == EmbeddingPassStats()
    # Empty archive is still created — consumers can mmap it without
    # branching on existence.
    assert store.exists()
    assert load_embeddings(store) == {}


def test_embed_references_missing_jsonl_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        embed_references(
            tmp_path / "no_such.jsonl",
            tmp_path / "out.npz",
            embedder=_FakeEmbedder(),
        )


def test_embed_references_writes_one_entry_per_record(tmp_path: Path) -> None:
    jsonl = _seed_dataset(
        tmp_path,
        [
            ("mtg", "var-1", (10, 20, 30)),
            ("mtg", "var-2", (200, 100, 50)),
            ("pokemon", "base1-4", (255, 0, 0)),
        ],
    )
    store = tmp_path / "reference_embeddings.npz"
    embedder = _FakeEmbedder(dim=64)

    stats = embed_references(jsonl, store, embedder=embedder)

    assert stats.embedded == 3
    assert stats.skipped_already_present == 0
    assert stats.image_load_failures == 0
    assert stats.embedder_failures == 0

    embeddings = load_embeddings(store)
    assert set(embeddings.keys()) == {
        ("mtg", "var-1"),
        ("mtg", "var-2"),
        ("pokemon", "base1-4"),
    }
    for vec in embeddings.values():
        assert vec.dtype == np.float32
        assert vec.shape == (64,)


def test_embed_references_is_idempotent_without_force(tmp_path: Path) -> None:
    jsonl = _seed_dataset(
        tmp_path,
        [("mtg", "var-1", (10, 20, 30)), ("mtg", "var-2", (40, 50, 60))],
    )
    store = tmp_path / "reference_embeddings.npz"
    embedder = _FakeEmbedder(dim=64)

    embed_references(jsonl, store, embedder=embedder)
    snapshot = load_embeddings(store)
    second = embed_references(jsonl, store, embedder=embedder)

    assert second.embedded == 0
    assert second.skipped_already_present == 2
    # Vectors unchanged byte-for-byte.
    after = load_embeddings(store)
    assert set(after.keys()) == set(snapshot.keys())
    for k in snapshot:
        np.testing.assert_array_equal(after[k], snapshot[k])


def test_embed_references_force_overwrites(tmp_path: Path) -> None:
    jsonl = _seed_dataset(tmp_path, [("mtg", "var-1", (10, 20, 30))])
    store = tmp_path / "reference_embeddings.npz"

    embed_references(jsonl, store, embedder=_FakeEmbedder(dim=64))
    original = load_embeddings(store)[("mtg", "var-1")].copy()

    # Re-embed with a fake that returns a different shape signature
    # to ensure the entry is genuinely overwritten.
    class _ConstEmbedder:
        dim = 64

        def encode(self, image: np.ndarray) -> np.ndarray:
            return np.full((64,), 0.5, dtype=np.float32)

    stats = embed_references(jsonl, store, embedder=_ConstEmbedder(), force=True)
    assert stats.embedded == 1
    assert stats.skipped_already_present == 0

    new_vec = load_embeddings(store)[("mtg", "var-1")]
    assert not np.array_equal(new_vec, original)
    np.testing.assert_array_equal(new_vec, np.full((64,), 0.5, dtype=np.float32))


# --------------------------------------------------------------------------
# embed_references — tolerated failures
# --------------------------------------------------------------------------


def test_embed_references_tolerates_image_load_failures(tmp_path: Path) -> None:
    # Mix of usable images and a record whose image file doesn't exist.
    jsonl = _seed_dataset(
        tmp_path,
        [
            ("mtg", "var-good", (10, 20, 30)),
            ("mtg", "var-bad", None),  # path is set, file missing
        ],
    )
    store = tmp_path / "reference_embeddings.npz"

    stats = embed_references(jsonl, store, embedder=_FakeEmbedder(dim=64))

    assert stats.embedded == 1
    assert stats.image_load_failures == 1
    assert stats.embedder_failures == 0

    embeddings = load_embeddings(store)
    assert ("mtg", "var-good") in embeddings
    assert ("mtg", "var-bad") not in embeddings


def test_embed_references_tolerates_null_image_path(tmp_path: Path) -> None:
    """A record can have front_image_path=None when image fetch
    failed during ingest. The pass should treat it as an image-load
    failure, not crash."""
    images_root = tmp_path / "images"
    images_root.mkdir()
    jsonl = tmp_path / "references.jsonl"
    with jsonl.open("w", encoding="utf-8") as fh:
        _write_record(
            fh,
            manufacturer="mtg",
            variant_id="no-image",
            front_image_path=None,
        )
    store = tmp_path / "reference_embeddings.npz"

    stats = embed_references(jsonl, store, embedder=_FakeEmbedder())

    assert stats.embedded == 0
    assert stats.image_load_failures == 1


def test_embed_references_tolerates_embedder_failures(tmp_path: Path) -> None:
    jsonl = _seed_dataset(
        tmp_path,
        [
            ("mtg", "var-1", (10, 20, 30)),
            ("mtg", "var-2", (40, 50, 60)),
            ("mtg", "var-3", (70, 80, 90)),
        ],
    )
    store = tmp_path / "reference_embeddings.npz"

    stats = embed_references(
        jsonl, store, embedder=_FlakyEmbedder(dim=64, fail_after=1)
    )

    # Only the first call succeeded; the rest raised and were counted.
    assert stats.embedded == 1
    assert stats.embedder_failures == 2
    assert stats.skipped_already_present == 0


# --------------------------------------------------------------------------
# lookup_references / load_embeddings
# --------------------------------------------------------------------------


def test_lookup_references_returns_2d_shape_on_hit(tmp_path: Path) -> None:
    jsonl = _seed_dataset(tmp_path, [("mtg", "var-1", (10, 20, 30))])
    store = tmp_path / "reference_embeddings.npz"
    embed_references(jsonl, store, embedder=_FakeEmbedder(dim=64))

    refs = lookup_references(store, "mtg", "var-1")
    assert refs is not None
    assert refs.shape == (1, 64)
    assert refs.dtype == np.float32

    # Round-trip into measure_embedding_anomaly's expected shape: it
    # accepts (n_refs, d) and a (d,) submitted vector.
    flat = load_embeddings(store)[("mtg", "var-1")]
    np.testing.assert_array_equal(refs[0], flat)


def test_lookup_references_returns_none_on_miss(tmp_path: Path) -> None:
    jsonl = _seed_dataset(tmp_path, [("mtg", "var-1", (10, 20, 30))])
    store = tmp_path / "reference_embeddings.npz"
    embed_references(jsonl, store, embedder=_FakeEmbedder(dim=64))

    assert lookup_references(store, "mtg", "var-missing") is None
    assert lookup_references(store, "pokemon", "var-1") is None


def test_lookup_references_returns_none_when_store_missing(tmp_path: Path) -> None:
    """Inference path may run before the embedding pass has been
    executed for the corpus. Lookup should answer None, not raise."""
    assert (
        lookup_references(tmp_path / "no_store.npz", "mtg", "var-1") is None
    )


def test_load_embeddings_returns_empty_dict_when_store_missing(
    tmp_path: Path,
) -> None:
    assert load_embeddings(tmp_path / "no_store.npz") == {}
