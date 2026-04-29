"""Tests for the manufacturer reference storage layer.

Pure I/O — no network. Validates:
  - Roundtrip write/read of ReferenceRecord through the JSONL
  - Idempotency lookups (has_record, iter_existing_variant_ids)
  - Image bytes are written to the correct relative path
  - Cache sees writes from this process AND from a fresh load
  - Half-written trailing JSONL line (simulating a crash) is tolerated
  - reference_image_relative_key handles unknown / missing content-types
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Ensure ml/ is on sys.path when running tests directly.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.ingestion.references_storage import (
    LocalReferenceStore,
    ReferenceRecord,
    reference_image_relative_key,
)


def _sample_record(
    variant_id: str = "uuid-1",
    manufacturer: str = "mtg",
    name: str = "Black Lotus",
) -> ReferenceRecord:
    return ReferenceRecord(
        variant_id=variant_id,
        manufacturer=manufacturer,
        name=name,
        set_code="LEA",
        collector_number="232",
        rarity="rare",
        front_image_path=None,
        source_url="https://api.scryfall.com/cards/uuid-1",
        fetched_at="2026-04-29T00:00:00Z",
    )


# --------------------------------------------------------------------------
# reference_image_relative_key — pure helper
# --------------------------------------------------------------------------


def test_image_relative_key_uses_jpg_for_jpeg() -> None:
    key = reference_image_relative_key("mtg", "uuid-1", "front", "image/jpeg")
    assert key == "images/mtg/uuid-1/front.jpg"


def test_image_relative_key_normalizes_charset_suffix() -> None:
    # PokemonTCG.io has been seen returning "image/jpeg; charset=binary"; we
    # should still recognize the primary type and not fall back to "bin".
    key = reference_image_relative_key("pokemon", "base1-4", "front", "image/jpeg; charset=binary")
    assert key == "images/pokemon/base1-4/front.jpg"


def test_image_relative_key_falls_back_to_bin_for_unknown() -> None:
    key = reference_image_relative_key("mtg", "x", "front", "application/octet-stream")
    assert key == "images/mtg/x/front.bin"


def test_image_relative_key_handles_empty_content_type() -> None:
    key = reference_image_relative_key("mtg", "x", "front", "")
    assert key == "images/mtg/x/front.bin"


# --------------------------------------------------------------------------
# LocalReferenceStore — write / read / idempotency
# --------------------------------------------------------------------------


def test_write_record_appends_to_jsonl(tmp_path: Path) -> None:
    store = LocalReferenceStore(tmp_path)
    store.write_record(_sample_record(variant_id="a"))
    store.write_record(_sample_record(variant_id="b", name="Mox Sapphire"))

    lines = (tmp_path / "references.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    rows = [json.loads(line) for line in lines]
    assert rows[0]["variant_id"] == "a"
    assert rows[1]["variant_id"] == "b"
    assert rows[1]["name"] == "Mox Sapphire"


def test_has_record_picks_up_in_process_writes(tmp_path: Path) -> None:
    store = LocalReferenceStore(tmp_path)
    assert store.has_record("mtg", "a") is False
    store.write_record(_sample_record(variant_id="a"))
    assert store.has_record("mtg", "a") is True
    # Different manufacturer with same variant_id is a different record.
    assert store.has_record("pokemon", "a") is False


def test_has_record_picks_up_writes_from_a_prior_process(tmp_path: Path) -> None:
    """A fresh `LocalReferenceStore` instance pointed at a directory that
    already has a populated JSONL must see those records on first call to
    has_record. This is the resume contract."""
    first = LocalReferenceStore(tmp_path)
    first.write_record(_sample_record(variant_id="a"))
    first.write_record(_sample_record(variant_id="b"))

    # Brand-new instance — simulates re-running the script after a crash.
    second = LocalReferenceStore(tmp_path)
    assert second.has_record("mtg", "a") is True
    assert second.has_record("mtg", "b") is True
    assert second.has_record("mtg", "c") is False


def test_iter_existing_variant_ids_returns_full_set(tmp_path: Path) -> None:
    store = LocalReferenceStore(tmp_path)
    store.write_record(_sample_record(variant_id="a"))
    store.write_record(_sample_record(variant_id="b", manufacturer="pokemon"))

    keys = store.iter_existing_variant_ids()
    assert keys == {("mtg", "a"), ("pokemon", "b")}


def test_partial_jsonl_line_is_tolerated(tmp_path: Path) -> None:
    """If a previous process crashed mid-write, the JSONL may end with a
    truncated line. The store must skip it on resume rather than crash."""
    records_path = tmp_path / "references.jsonl"
    records_path.write_text(
        json.dumps({
            "variant_id": "good",
            "manufacturer": "mtg",
            "name": "Ok",
            "set_code": "LEA",
            "collector_number": "1",
            "rarity": "rare",
            "front_image_path": None,
            "source_url": "x",
            "fetched_at": "2026-04-29T00:00:00Z",
        }) + "\n"
        + '{"variant_id": "broken", "manufactur',  # truncated mid-key
        encoding="utf-8",
    )
    store = LocalReferenceStore(tmp_path)
    assert store.has_record("mtg", "good") is True
    # The corrupt line is silently skipped — its variant_id never makes
    # it into the cache, so a re-fetch will retry it. Better than an
    # outright crash on resume.
    assert store.has_record("mtg", "broken") is False


def test_write_image_creates_nested_path_and_returns_storage_path(tmp_path: Path) -> None:
    store = LocalReferenceStore(tmp_path)
    written = store.write_image(
        "mtg", "uuid-x", "front", "image/jpeg", b"fake-jpeg-bytes",
    )
    expected = tmp_path / "images" / "mtg" / "uuid-x" / "front.jpg"
    assert Path(written) == expected
    assert expected.read_bytes() == b"fake-jpeg-bytes"


def test_write_image_overwrites_existing_atomically(tmp_path: Path) -> None:
    """Re-fetching an image should overwrite the prior bytes via the
    write-then-rename atomic-replace pattern. We don't test the rename
    semantics directly (filesystem-dependent), just that the bytes
    update without leaving a stray .part file."""
    store = LocalReferenceStore(tmp_path)
    store.write_image("mtg", "x", "front", "image/jpeg", b"v1")
    store.write_image("mtg", "x", "front", "image/jpeg", b"v2")

    target = tmp_path / "images" / "mtg" / "x" / "front.jpg"
    assert target.read_bytes() == b"v2"
    # No leftover .part file.
    assert not (target.with_suffix(".jpg.part")).exists()


def test_records_path_property(tmp_path: Path) -> None:
    store = LocalReferenceStore(tmp_path)
    assert store.records_path == tmp_path / "references.jsonl"


def test_records_directory_is_created_lazily(tmp_path: Path) -> None:
    """Initializing the store on a non-existent dir is fine — it gets
    created. Initializing on an existing populated dir is also fine."""
    new_root = tmp_path / "does" / "not" / "exist"
    store = LocalReferenceStore(new_root)
    assert new_root.is_dir()
    assert (new_root / "images").is_dir()
    # And the store is usable immediately.
    store.write_record(_sample_record(variant_id="z"))
    assert store.has_record("mtg", "z") is True


@pytest.mark.parametrize(
    "manufacturer,variant_id",
    [
        ("mtg", "uuid-with-dashes-and-numbers-12345"),
        ("pokemon", "base1-4"),
        ("ygo", "RA01-EN001"),  # hypothetical future source
    ],
)
def test_image_relative_key_round_trips_variant_id(manufacturer: str, variant_id: str) -> None:
    key = reference_image_relative_key(manufacturer, variant_id, "front", "image/jpeg")
    assert manufacturer in key
    assert variant_id in key
    assert key.endswith("/front.jpg")
