"""Tests for the PSA exemplar accumulator.

Uses synthetic PSA scraped.jsonl rows + a deterministic fake embedder
+ an InMemoryCatalogIndex prepopulated with phash entries that match
the test images. Covers:

  - Idempotency by cert_id: re-running on the same JSONL is a no-op
    for already-consumed certs.
  - Records without an on-disk image are counted, not raised.
  - Records that don't identify to a confident variant are counted
    as skipped_unidentified.
  - Records that do identify get their embedding APPENDED to the
    matched variant's npz key — single-printing seed grows from
    `(d,)` to `(2, d)` after the first PSA-exemplar append.
  - Multiple PSA records mapping to the same variant stack into
    `(n_refs, d)` cleanly.
  - Empty PSA jsonl is tolerated (zero counts, no archive write).
  - Bad embedder shape is tolerated and counted.
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

from data.ingestion.psa_exemplars import (  # noqa: E402
    accumulate_psa_exemplars,
    ExemplarPassStats,
)
from data.ingestion.reference_embeddings import (  # noqa: E402
    load_embeddings,
    lookup_references,
)
from pipelines.identification.catalog import (  # noqa: E402
    CardCatalogEntry,
    InMemoryCatalogIndex,
)
from pipelines.identification.hashing import compute_phash  # noqa: E402


# --------------------------------------------------------------------------
# Fakes / fixtures
# --------------------------------------------------------------------------


@dataclass
class _FakeEmbedder:
    """Deterministic image-bytes-seeded embedder. Same shape as the one
    in test_reference_embeddings — copied here so tests don't cross-
    import each other's helpers."""

    dim: int = 64

    def encode(self, image: np.ndarray) -> np.ndarray:
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("fake embedder expects uint8 BGR image")
        digest = sha256(image.tobytes()).digest()
        seed = int.from_bytes(digest[:8], "big")
        rng = np.random.default_rng(seed)
        return rng.standard_normal(self.dim).astype(np.float32)


@dataclass
class _BadShapeEmbedder:
    """Embedder that returns the wrong shape, to verify the accumulator
    counts the failure rather than crashing."""

    dim: int = 64

    def encode(self, image: np.ndarray) -> np.ndarray:
        return np.zeros(7, dtype=np.float32)  # wrong size


def _write_test_image(path: Path, seed: int) -> None:
    """Write a tiny but valid PNG so cv2.imread returns a 3-channel
    uint8. Content varies per seed so identification can phash-match
    distinct cards."""
    rng = np.random.default_rng(seed)
    # 32x32 is below the canonical floor but cv2 + identify don't care
    # about size for the pure-phash short-circuit path.
    img = rng.integers(0, 255, size=(32, 32, 3), dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    path.write_bytes(buf.tobytes())


def _read_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    assert img is not None
    return img


def _write_psa_jsonl(jsonl_path: Path, records: list[dict]) -> None:
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")


def _build_catalog_for_image(
    image_path: Path,
    *,
    variant_id: str,
    game: str = "mtg",
    embedder: _FakeEmbedder | None = None,
) -> InMemoryCatalogIndex:
    """Add a catalog entry whose phash matches the on-disk image —
    identification's pHash short-circuit will pick it. Embedding is
    also populated so the embedding-side path also matches."""
    if embedder is None:
        embedder = _FakeEmbedder()
    img = _read_image(image_path)
    catalog = InMemoryCatalogIndex()
    catalog.add(
        CardCatalogEntry(
            variant_id=variant_id,
            name="Test Card",
            set_code="TST",
            card_number="1",
            game=game,
            canonical_phash=compute_phash(img),
            canonical_embedding=embedder.encode(img),
        )
    )
    return catalog


# --------------------------------------------------------------------------
# Identification → append happy path
# --------------------------------------------------------------------------


def test_appended_exemplar_lands_under_matched_variant_key(tmp_path: Path) -> None:
    """A PSA record that identifies to a known variant has its
    embedding appended under that variant's key in the npz."""
    img_path = tmp_path / "img_1.png"
    _write_test_image(img_path, seed=1)

    catalog = _build_catalog_for_image(img_path, variant_id="abc-123", game="mtg")
    psa_jsonl = tmp_path / "scraped.jsonl"
    _write_psa_jsonl(psa_jsonl, [
        {"cert_id": 100001, "front_image_path": str(img_path), "grade": 9.0},
    ])

    embeddings_npz = tmp_path / "reference_embeddings.npz"
    log_path = tmp_path / "psa_exemplars_log.jsonl"

    stats = accumulate_psa_exemplars(
        psa_jsonl_path=psa_jsonl,
        embeddings_store_path=embeddings_npz,
        log_path=log_path,
        catalog=catalog,
        embedder=_FakeEmbedder(),
    )

    assert stats.appended == 1
    assert stats.skipped_unidentified == 0
    assert stats.total_records == 1
    refs = lookup_references(embeddings_npz, "mtg", "abc-123")
    assert refs is not None
    assert refs.shape == (1, 64)


def test_append_promotes_seed_to_2d_and_stacks(tmp_path: Path) -> None:
    """If the variant already has a single-printing seed embedding
    (shape `(d,)` from `embed_references`), the first PSA append
    promotes it to `(2, d)` before stacking. The seed must not be
    lost."""
    img_path = tmp_path / "img_1.png"
    _write_test_image(img_path, seed=2)
    catalog = _build_catalog_for_image(img_path, variant_id="abc-123")

    # Seed the npz with a 1-D embedding under the same key (this is
    # what `embed_references` writes for a freshly-scraped variant).
    embeddings_npz = tmp_path / "reference_embeddings.npz"
    seed_vec = np.full(64, 7.0, dtype=np.float32)
    np.savez(str(embeddings_npz), **{"mtg/abc-123": seed_vec})

    psa_jsonl = tmp_path / "scraped.jsonl"
    _write_psa_jsonl(psa_jsonl, [
        {"cert_id": 100002, "front_image_path": str(img_path), "grade": 8.5},
    ])

    accumulate_psa_exemplars(
        psa_jsonl_path=psa_jsonl,
        embeddings_store_path=embeddings_npz,
        log_path=tmp_path / "log.jsonl",
        catalog=catalog,
        embedder=_FakeEmbedder(),
    )

    refs = lookup_references(embeddings_npz, "mtg", "abc-123")
    assert refs is not None
    assert refs.shape == (2, 64)
    # The seed survives at index 0; the new exemplar is at index 1.
    np.testing.assert_array_equal(refs[0], seed_vec)


def test_multiple_psa_exemplars_stack_into_n_refs_d(tmp_path: Path) -> None:
    """Three PSA records mapping to the same catalog variant produce
    a `(3, d)` reference array. n_references=3 so the embedding-anomaly
    detector's confidence ramp picks up signal."""
    img_paths = [tmp_path / f"img_{i}.png" for i in range(3)]
    # Use the SAME seed for all three so phash matches the catalog
    # entry — different seeds would diverge enough to drop below the
    # min-accept-confidence threshold.
    for p in img_paths:
        _write_test_image(p, seed=42)
    catalog = _build_catalog_for_image(img_paths[0], variant_id="abc-123")

    psa_jsonl = tmp_path / "scraped.jsonl"
    _write_psa_jsonl(psa_jsonl, [
        {"cert_id": 200001, "front_image_path": str(img_paths[0]), "grade": 9.0},
        {"cert_id": 200002, "front_image_path": str(img_paths[1]), "grade": 8.5},
        {"cert_id": 200003, "front_image_path": str(img_paths[2]), "grade": 9.5},
    ])

    embeddings_npz = tmp_path / "reference_embeddings.npz"
    stats = accumulate_psa_exemplars(
        psa_jsonl_path=psa_jsonl,
        embeddings_store_path=embeddings_npz,
        log_path=tmp_path / "log.jsonl",
        catalog=catalog,
        embedder=_FakeEmbedder(),
    )
    assert stats.appended == 3
    refs = lookup_references(embeddings_npz, "mtg", "abc-123")
    assert refs is not None
    assert refs.shape == (3, 64)


# --------------------------------------------------------------------------
# Idempotency
# --------------------------------------------------------------------------


def test_re_run_skips_already_consumed_certs(tmp_path: Path) -> None:
    """The sidecar log records consumed cert_ids; a second pass over
    the same JSONL yields zero appends and N skipped_already_consumed."""
    img_path = tmp_path / "img.png"
    _write_test_image(img_path, seed=1)
    catalog = _build_catalog_for_image(img_path, variant_id="abc-123")

    psa_jsonl = tmp_path / "scraped.jsonl"
    _write_psa_jsonl(psa_jsonl, [
        {"cert_id": 300001, "front_image_path": str(img_path), "grade": 9.0},
    ])

    embeddings_npz = tmp_path / "reference_embeddings.npz"
    log_path = tmp_path / "log.jsonl"

    first = accumulate_psa_exemplars(
        psa_jsonl_path=psa_jsonl, embeddings_store_path=embeddings_npz,
        log_path=log_path, catalog=catalog, embedder=_FakeEmbedder(),
    )
    assert first.appended == 1

    second = accumulate_psa_exemplars(
        psa_jsonl_path=psa_jsonl, embeddings_store_path=embeddings_npz,
        log_path=log_path, catalog=catalog, embedder=_FakeEmbedder(),
    )
    assert second.appended == 0
    assert second.skipped_already_consumed == 1


def test_log_contains_cert_id_and_variant_attribution(tmp_path: Path) -> None:
    img_path = tmp_path / "img.png"
    _write_test_image(img_path, seed=5)
    catalog = _build_catalog_for_image(img_path, variant_id="abc-123", game="mtg")

    psa_jsonl = tmp_path / "scraped.jsonl"
    _write_psa_jsonl(psa_jsonl, [
        {"cert_id": 400001, "front_image_path": str(img_path), "grade": 9.0},
    ])
    log_path = tmp_path / "log.jsonl"
    accumulate_psa_exemplars(
        psa_jsonl_path=psa_jsonl,
        embeddings_store_path=tmp_path / "ref.npz",
        log_path=log_path,
        catalog=catalog,
        embedder=_FakeEmbedder(),
    )
    line = log_path.read_text(encoding="utf-8").strip()
    obj = json.loads(line)
    assert obj["cert_id"] == 400001
    assert obj["manufacturer"] == "mtg"
    assert obj["variant_id"] == "abc-123"
    assert obj["added_at"].endswith("Z")


# --------------------------------------------------------------------------
# Skip / failure paths
# --------------------------------------------------------------------------


def test_record_without_image_path_is_skipped(tmp_path: Path) -> None:
    psa_jsonl = tmp_path / "scraped.jsonl"
    _write_psa_jsonl(psa_jsonl, [
        {"cert_id": 500001, "front_image_path": None, "grade": 9.0},
        {"cert_id": 500002, "front_image_path": "", "grade": 8.0},
    ])
    catalog = InMemoryCatalogIndex()  # empty
    stats = accumulate_psa_exemplars(
        psa_jsonl_path=psa_jsonl,
        embeddings_store_path=tmp_path / "ref.npz",
        log_path=tmp_path / "log.jsonl",
        catalog=catalog,
        embedder=_FakeEmbedder(),
    )
    assert stats.skipped_no_image == 2
    assert stats.appended == 0


def test_record_with_missing_image_file_is_skipped(tmp_path: Path) -> None:
    psa_jsonl = tmp_path / "scraped.jsonl"
    _write_psa_jsonl(psa_jsonl, [
        {
            "cert_id": 500003,
            "front_image_path": str(tmp_path / "doesnt_exist.png"),
            "grade": 9.0,
        },
    ])
    stats = accumulate_psa_exemplars(
        psa_jsonl_path=psa_jsonl,
        embeddings_store_path=tmp_path / "ref.npz",
        log_path=tmp_path / "log.jsonl",
        catalog=InMemoryCatalogIndex(),
        embedder=_FakeEmbedder(),
    )
    assert stats.skipped_no_image == 1
    assert stats.appended == 0


def test_unidentified_record_is_counted(tmp_path: Path) -> None:
    """If identification finds no confident variant (empty catalog),
    the record is counted as skipped_unidentified rather than appended."""
    img_path = tmp_path / "img.png"
    _write_test_image(img_path, seed=99)

    psa_jsonl = tmp_path / "scraped.jsonl"
    _write_psa_jsonl(psa_jsonl, [
        {"cert_id": 600001, "front_image_path": str(img_path), "grade": 9.0},
    ])
    stats = accumulate_psa_exemplars(
        psa_jsonl_path=psa_jsonl,
        embeddings_store_path=tmp_path / "ref.npz",
        log_path=tmp_path / "log.jsonl",
        catalog=InMemoryCatalogIndex(),  # empty → no match possible
        embedder=_FakeEmbedder(),
    )
    assert stats.skipped_unidentified == 1
    assert stats.appended == 0


def test_bad_embedder_shape_is_counted(tmp_path: Path) -> None:
    img_path = tmp_path / "img.png"
    _write_test_image(img_path, seed=1)
    # Catalog uses the GOOD embedder (so identification succeeds), but
    # the accumulator runs with the bad-shape embedder fallback path.
    catalog = _build_catalog_for_image(img_path, variant_id="abc-123")

    psa_jsonl = tmp_path / "scraped.jsonl"
    _write_psa_jsonl(psa_jsonl, [
        {"cert_id": 700001, "front_image_path": str(img_path), "grade": 9.0},
    ])
    # Note: identification short-circuits on phash exact match, so the
    # `submitted_embedding` on the result is None — the accumulator
    # will then call `embedder.encode` itself, which is where the bad
    # shape surfaces. (This is the path the test exercises.)
    stats = accumulate_psa_exemplars(
        psa_jsonl_path=psa_jsonl,
        embeddings_store_path=tmp_path / "ref.npz",
        log_path=tmp_path / "log.jsonl",
        catalog=catalog,
        embedder=_BadShapeEmbedder(dim=64),
    )
    assert stats.embedder_failures == 1
    assert stats.appended == 0


# --------------------------------------------------------------------------
# Empty input + max_records
# --------------------------------------------------------------------------


def test_empty_jsonl_is_a_no_op(tmp_path: Path) -> None:
    psa_jsonl = tmp_path / "scraped.jsonl"
    psa_jsonl.write_text("", encoding="utf-8")
    embeddings_npz = tmp_path / "ref.npz"

    stats = accumulate_psa_exemplars(
        psa_jsonl_path=psa_jsonl,
        embeddings_store_path=embeddings_npz,
        log_path=tmp_path / "log.jsonl",
        catalog=InMemoryCatalogIndex(),
        embedder=_FakeEmbedder(),
    )
    assert stats.total_records == 0
    assert stats.appended == 0
    # Nothing dirty → archive not written.
    assert not embeddings_npz.exists()


def test_missing_jsonl_raises_filenotfound(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        accumulate_psa_exemplars(
            psa_jsonl_path=tmp_path / "doesnt_exist.jsonl",
            embeddings_store_path=tmp_path / "ref.npz",
            log_path=tmp_path / "log.jsonl",
            catalog=InMemoryCatalogIndex(),
            embedder=_FakeEmbedder(),
        )


def test_max_records_caps_iteration(tmp_path: Path) -> None:
    img_path = tmp_path / "img.png"
    _write_test_image(img_path, seed=11)
    catalog = _build_catalog_for_image(img_path, variant_id="abc-123")
    psa_jsonl = tmp_path / "scraped.jsonl"
    _write_psa_jsonl(psa_jsonl, [
        {"cert_id": 800000 + i, "front_image_path": str(img_path), "grade": 9.0}
        for i in range(5)
    ])
    stats = accumulate_psa_exemplars(
        psa_jsonl_path=psa_jsonl,
        embeddings_store_path=tmp_path / "ref.npz",
        log_path=tmp_path / "log.jsonl",
        catalog=catalog,
        embedder=_FakeEmbedder(),
        max_records=2,
    )
    assert stats.total_records == 2
    assert stats.appended == 2
