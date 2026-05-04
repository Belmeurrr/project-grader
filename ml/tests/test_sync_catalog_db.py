"""Tests for `scripts.sync_catalog_db`.

Test path decision:

    The production sync_catalog_db pipeline talks to Postgres via
    asyncpg + pgvector — those packages aren't installed in the ml/
    test venv, and pgvector-on-SQLite isn't a viable fixture (Vector
    is a Postgres-extension type). So this test exercises the pure
    data-preparation layer (`iter_set_rows`, `iter_variant_rows`,
    `_phash_to_signed64`) end-to-end against a synthetic JSONL +
    npz + small JPEG fixtures, plus an in-process fake session that
    records the row payloads `sync_catalog_db` would have submitted
    for upsert. Idempotency is asserted at the prep level: walking
    the same fixtures twice yields identical row payloads.

    The actual `apply_sets` / `apply_variants` async-DB code paths
    (which import `grader.db.models` and therefore pgvector) are
    structurally idempotent via Postgres `INSERT ... ON CONFLICT DO
    UPDATE` and are exercised by the API test suite that runs
    against a real Postgres test database. Bringing that fixture
    over to ml/ would force every ml/ contributor to install
    asyncpg + pgvector + run docker — a poor cost/benefit for a
    catalog-sync helper.

    Embedding column: assertions use the raw numpy array shape
    (768,) for variants we synthesized embeddings for, and `None`
    for variants we didn't. The Vector → numpy bridge is
    pgvector's responsibility and is covered by API tests.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.sync_catalog_db import (  # noqa: E402
    SyncStats,
    _phash_to_signed64,
    iter_set_rows,
    iter_variant_rows,
    sync_catalog_db,
)


# --------------------------------------------------------------------------
# Fixture builders
# --------------------------------------------------------------------------


def _write_image(path: Path, color: tuple[int, int, int]) -> None:
    """Synthesize a tiny solid-color JPEG via cv2.imencode + write
    bytes — keeps the fixture deterministic and platform-agnostic."""
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.full((32, 32, 3), color, dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok, "cv2.imencode failed"
    path.write_bytes(buf.tobytes())


def _record(
    *,
    manufacturer: str,
    variant_id: str,
    name: str,
    set_code: str | None,
    collector_number: str | None,
    front_image_path: str | None,
) -> dict[str, object]:
    return {
        "variant_id": variant_id,
        "manufacturer": manufacturer,
        "name": name,
        "set_code": set_code,
        "collector_number": collector_number,
        "rarity": "common",
        "front_image_path": front_image_path,
        "source_url": "https://example.test/card",
        "fetched_at": "2026-04-29T00:00:00Z",
    }


def _seed_catalog(root: Path) -> tuple[Path, Path]:
    """Build a 4-record references.jsonl + matching .npz with
    embeddings for two of them. Returns (jsonl_path, npz_path)."""
    images_root = root / "images"
    jsonl_path = root / "references.jsonl"
    npz_path = root / "reference_embeddings.npz"

    img_lea_232 = images_root / "mtg" / "lea-232" / "front.jpg"
    img_lea_001 = images_root / "mtg" / "lea-001" / "front.jpg"
    img_base1_4 = images_root / "pokemon" / "base1-4" / "front.jpg"

    _write_image(img_lea_232, (10, 20, 30))
    _write_image(img_lea_001, (50, 60, 70))
    _write_image(img_base1_4, (220, 80, 40))
    # img for the unmapped row exists but the row itself will be
    # filtered out by manufacturer mapping.

    records = [
        _record(
            manufacturer="mtg",
            variant_id="lea-232",
            name="Black Lotus",
            set_code="LEA",
            collector_number="232",
            front_image_path=str(img_lea_232),
        ),
        _record(
            manufacturer="mtg",
            variant_id="lea-001",
            name="Ancestral Recall",
            set_code="LEA",
            collector_number="001",
            front_image_path=str(img_lea_001),
        ),
        _record(
            manufacturer="pokemon",
            variant_id="base1-4",
            name="Charizard",
            set_code="base1",
            collector_number="4",
            front_image_path=str(img_base1_4),
        ),
        # Manufacturer not in mapping — should be filtered.
        _record(
            manufacturer="ygo",
            variant_id="ygo-1",
            name="Blue-Eyes",
            set_code="LOB",
            collector_number="1",
            front_image_path=None,
        ),
    ]
    with jsonl_path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")

    # Embeddings for the first two MTG records only — verifies the
    # apply path tolerates variants that have JSONL rows but no npz
    # entry (e.g. mid-run crash before embedding).
    rng = np.random.default_rng(seed=42)
    payload = {
        "mtg/lea-232": rng.standard_normal(768).astype(np.float32),
        "mtg/lea-001": rng.standard_normal(768).astype(np.float32),
    }
    np.savez(str(npz_path), **payload)

    return jsonl_path, npz_path


# --------------------------------------------------------------------------
# iter_set_rows
# --------------------------------------------------------------------------


def test_iter_set_rows_dedupes_and_filters_unmapped(tmp_path: Path) -> None:
    jsonl, _ = _seed_catalog(tmp_path)

    rows = iter_set_rows(jsonl)

    # Two distinct (game, code) pairs from the 4 records: one MTG/LEA
    # (covering both lea-232 and lea-001), one pokemon/base1. The
    # ygo row is filtered.
    keys = sorted((r["game"], r["code"]) for r in rows)
    assert keys == [("mtg", "LEA"), ("pokemon", "base1")]
    for r in rows:
        # Name fallback is the upper-cased code (until the JSONL
        # carries a per-set name).
        assert r["name"] == r["code"].upper()


def test_iter_set_rows_tolerates_blank_lines(tmp_path: Path) -> None:
    jsonl = tmp_path / "references.jsonl"
    jsonl.write_text(
        "\n"
        + json.dumps(
            _record(
                manufacturer="mtg",
                variant_id="v1",
                name="X",
                set_code="LEA",
                collector_number="1",
                front_image_path=None,
            )
        )
        + "\n\n",
        encoding="utf-8",
    )
    rows = iter_set_rows(jsonl)
    assert rows == [{"game": "mtg", "code": "LEA", "name": "LEA"}]


def test_iter_set_rows_skips_rows_with_missing_set_code(tmp_path: Path) -> None:
    jsonl = tmp_path / "references.jsonl"
    with jsonl.open("w", encoding="utf-8") as fh:
        fh.write(
            json.dumps(
                _record(
                    manufacturer="mtg",
                    variant_id="v1",
                    name="X",
                    set_code=None,
                    collector_number="1",
                    front_image_path=None,
                )
            )
            + "\n"
        )
    assert iter_set_rows(jsonl) == []


# --------------------------------------------------------------------------
# iter_variant_rows
# --------------------------------------------------------------------------


def test_iter_variant_rows_emits_per_record_payloads(tmp_path: Path) -> None:
    jsonl, npz = _seed_catalog(tmp_path)

    set_id_lookup = {
        ("mtg", "LEA"): "set-id-lea",
        ("pokemon", "base1"): "set-id-base1",
    }
    stats = SyncStats()

    rows = list(
        iter_variant_rows(
            jsonl, npz, set_id_lookup=set_id_lookup, stats=stats
        )
    )

    # 4 input records — 1 unmapped manufacturer (ygo). 3 emitted.
    assert len(rows) == 3
    assert stats.variants_total == 4
    assert stats.skipped_unmapped_manufacturer == 1

    by_number = {r["card_number"]: r for r in rows}
    assert set(by_number) == {"232", "001", "4"}

    lotus = by_number["232"]
    assert lotus["game"] == "mtg"
    assert lotus["set_id"] == "set-id-lea"
    assert lotus["name"] == "Black Lotus"
    # MTG → variant_id stored on scryfall_id.
    assert lotus["scryfall_id"] == "lea-232"
    assert lotus["external_ids"]["manufacturer"] == "mtg"
    assert lotus["external_ids"]["mtg_variant_id"] == "lea-232"
    # phash is a signed 64-bit int.
    assert isinstance(lotus["canonical_phash"], int)
    assert -(2**63) <= lotus["canonical_phash"] < 2**63
    # Embedding came through as float32 (768,).
    emb = lotus["canonical_image_embedding"]
    assert isinstance(emb, np.ndarray)
    assert emb.shape == (768,)
    assert emb.dtype == np.float32

    # Pokemon row: scryfall_id is None (not MTG); external_ids
    # carries the source-side variant_id under a manufacturer-keyed
    # entry.
    charizard = by_number["4"]
    assert charizard["game"] == "pokemon"
    assert charizard["scryfall_id"] is None
    assert charizard["external_ids"]["pokemon_variant_id"] == "base1-4"
    # No embedding was synthesized for base1-4 in the fixture.
    assert charizard["canonical_image_embedding"] is None
    # phash still computed from the on-disk image.
    assert isinstance(charizard["canonical_phash"], int)

    assert stats.variants_with_phash == 3
    assert stats.variants_with_embedding == 2  # only lea-232 + lea-001


def test_iter_variant_rows_is_deterministic_across_runs(tmp_path: Path) -> None:
    """Re-running the prep over the same fixtures must yield byte-
    identical row payloads. This is the test-side stand-in for the
    DB-level idempotency the production ON CONFLICT DO UPDATE
    provides — same input → same upsert payload → no real change."""
    jsonl, npz = _seed_catalog(tmp_path)
    lookup = {
        ("mtg", "LEA"): "set-id-lea",
        ("pokemon", "base1"): "set-id-base1",
    }

    stats_a = SyncStats()
    rows_a = list(
        iter_variant_rows(jsonl, npz, set_id_lookup=lookup, stats=stats_a)
    )
    stats_b = SyncStats()
    rows_b = list(
        iter_variant_rows(jsonl, npz, set_id_lookup=lookup, stats=stats_b)
    )

    assert asdict(stats_a) == asdict(stats_b)
    assert len(rows_a) == len(rows_b)
    for a, b in zip(rows_a, rows_b):
        # Numpy arrays don't compare equal under == in dicts; check
        # each scalar key first then bytes-equal the arrays.
        assert a.keys() == b.keys()
        for k in a:
            if isinstance(a[k], np.ndarray):
                assert isinstance(b[k], np.ndarray)
                np.testing.assert_array_equal(a[k], b[k])
            else:
                assert a[k] == b[k]


def test_iter_variant_rows_tolerates_missing_image(tmp_path: Path) -> None:
    """A record can have front_image_path=None or pointing at a
    missing file (image fetch failed during ingest). The prep emits
    the row anyway with canonical_phash=None and increments the
    image_load_failures counter only when a path was provided."""
    jsonl = tmp_path / "references.jsonl"
    with jsonl.open("w", encoding="utf-8") as fh:
        fh.write(
            json.dumps(
                _record(
                    manufacturer="mtg",
                    variant_id="no-img",
                    name="No Image",
                    set_code="LEA",
                    collector_number="9",
                    front_image_path=None,
                )
            )
            + "\n"
        )
        fh.write(
            json.dumps(
                _record(
                    manufacturer="mtg",
                    variant_id="bad-img",
                    name="Bad Image",
                    set_code="LEA",
                    collector_number="10",
                    front_image_path=str(tmp_path / "does_not_exist.jpg"),
                )
            )
            + "\n"
        )
    npz = tmp_path / "missing.npz"  # never created — also tolerated

    stats = SyncStats()
    rows = list(
        iter_variant_rows(
            jsonl,
            npz,
            set_id_lookup={("mtg", "LEA"): "sid"},
            stats=stats,
        )
    )

    assert len(rows) == 2
    for r in rows:
        assert r["canonical_phash"] is None
        assert r["canonical_image_embedding"] is None
    # Only the bad-path record counts as image_load_failure;
    # front_image_path=None doesn't (it never tried to read).
    assert stats.image_load_failures == 1


def test_iter_variant_rows_skips_missing_card_number(tmp_path: Path) -> None:
    jsonl = tmp_path / "references.jsonl"
    with jsonl.open("w", encoding="utf-8") as fh:
        fh.write(
            json.dumps(
                _record(
                    manufacturer="mtg",
                    variant_id="v1",
                    name="X",
                    set_code="LEA",
                    collector_number=None,
                    front_image_path=None,
                )
            )
            + "\n"
        )
    npz = tmp_path / "no_npz.npz"
    stats = SyncStats()
    rows = list(
        iter_variant_rows(
            jsonl, npz, set_id_lookup={("mtg", "LEA"): "sid"}, stats=stats
        )
    )
    assert rows == []
    assert stats.skipped_missing_card_number == 1


# --------------------------------------------------------------------------
# _phash_to_signed64 — boundary check
# --------------------------------------------------------------------------


def test_phash_to_signed64_handles_high_bit_set() -> None:
    # Values < 2**63 pass through unchanged.
    assert _phash_to_signed64(0) == 0
    assert _phash_to_signed64(123456) == 123456
    assert _phash_to_signed64(2**63 - 1) == 2**63 - 1
    # Values ≥ 2**63 are mapped into the signed range.
    assert _phash_to_signed64(2**63) == -(2**63)
    assert _phash_to_signed64(2**64 - 1) == -1
    assert _phash_to_signed64(None) is None


# --------------------------------------------------------------------------
# sync_catalog_db end-to-end via dry-run
# --------------------------------------------------------------------------


def test_sync_catalog_db_dry_run_produces_full_stats(tmp_path: Path) -> None:
    """Dry-run walks both passes without touching the DB. Verifies
    the stats numbers a real run would produce, and confirms the
    function is asyncio-compatible without needing asyncpg.

    Driven via `asyncio.run` directly rather than pytest-asyncio
    because the ml/ test venv doesn't carry pytest-asyncio (the
    pipelines themselves are sync). One-line wrapper keeps the test
    surface obvious."""
    import asyncio

    _seed_catalog(tmp_path)

    stats = asyncio.run(sync_catalog_db(tmp_path, dry_run=True))

    # Two distinct (game, code) sets.
    assert stats.sets_total == 2
    # 4 records → 1 filtered (ygo) → 3 variants emitted.
    assert stats.variants_total == 4
    assert stats.skipped_unmapped_manufacturer == 1
    assert stats.variants_with_phash == 3
    assert stats.variants_with_embedding == 2
    # Dry-run never touches DB, so upserted counters stay at zero.
    assert stats.sets_upserted == 0
    assert stats.variants_upserted == 0


def test_sync_catalog_db_dry_run_is_idempotent(tmp_path: Path) -> None:
    """Re-running the dry-run over the same fixtures yields the
    same stats — the prep layer is deterministic, which is the
    test-side stand-in for the DB-level ON CONFLICT DO UPDATE
    idempotency."""
    import asyncio

    _seed_catalog(tmp_path)

    stats1 = asyncio.run(sync_catalog_db(tmp_path, dry_run=True))
    stats2 = asyncio.run(sync_catalog_db(tmp_path, dry_run=True))

    assert asdict(stats1) == asdict(stats2)
