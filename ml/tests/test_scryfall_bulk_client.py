"""Scryfall bulk-data ingest client tests.

All HTTP is mocked via httpx.MockTransport — we never touch real
api.scryfall.com or data.scryfall.io. Tests cover:
    - _select_bulk_entry picks by `type` field, returns None on miss
    - download_bulk_file streams chunks to disk and atomically renames
    - iter_cards_from_bulk yields card dicts; rejects non-list top level
    - ingest_bulk: full happy path (metadata → download → ingest)
    - ingest_bulk: respects max_cards
    - ingest_bulk: raises if requested bulk_type is missing
    - ingest_bulk: keep_cache controls bulk file retention
    - ingest_bulk: 5xx on the bulk file download propagates as error
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.ingestion.references_storage import LocalReferenceStore
from data.ingestion.scryfall import (
    SCRYFALL_API_BASE_URL,
    ScryfallIngestStats,
)
from data.ingestion.scryfall_bulk import (
    DEFAULT_BULK_TYPE,
    _select_bulk_entry,
    download_bulk_file,
    ingest_bulk,
    iter_cards_from_bulk,
)


# --------------------------------------------------------------------------
# Fixture helpers (shape mirrors test_scryfall_client._card)
# --------------------------------------------------------------------------


def _card(
    *,
    variant_id: str = "uuid-1",
    name: str = "Black Lotus",
    set_code: str = "lea",
    collector_number: str = "232",
    rarity: str = "rare",
) -> dict:
    return {
        "id": variant_id,
        "name": name,
        "set": set_code,
        "collector_number": collector_number,
        "rarity": rarity,
        "lang": "en",
        "layout": "normal",
        "scryfall_uri": f"https://scryfall.com/card/{set_code}/{collector_number}",
        "image_uris": {
            "small": f"https://cards.scryfall.io/small/{variant_id}.jpg",
            "normal": f"https://cards.scryfall.io/normal/{variant_id}.jpg",
            "large": f"https://cards.scryfall.io/large/{variant_id}.jpg",
            "png": f"https://cards.scryfall.io/png/{variant_id}.png",
        },
    }


def _bulk_metadata_payload(
    *,
    download_uri: str,
    bulk_type: str = DEFAULT_BULK_TYPE,
    extra_types: tuple[str, ...] = ("oracle_cards", "all_cards"),
) -> dict:
    """Synthesize a /bulk-data response with `bulk_type` plus a few siblings.

    Sibling entries exist so _select_bulk_entry has to actually filter
    rather than just take entries[0]."""
    entries = [
        {
            "object": "bulk_data",
            "type": t,
            "name": f"{t.replace('_', ' ').title()}",
            "download_uri": f"https://data.scryfall.io/{t}/{t}.json",
            "size": 12345,
        }
        for t in extra_types
    ]
    entries.insert(
        len(entries) // 2,
        {
            "object": "bulk_data",
            "type": bulk_type,
            "name": bulk_type.replace("_", " ").title(),
            "download_uri": download_uri,
            "size": 67890,
        },
    )
    return {"object": "list", "has_more": False, "data": entries}


def _png_1x1_bytes() -> bytes:
    """Smallest valid PNG (1x1, transparent). Matches the helper used in
    test_scryfall_client.py so the fixture image is always decodable."""
    return bytes.fromhex(
        "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4"
        "890000000d49444154789c63000100000005000146cea2270000000049454e44"
        "ae426082"
    )


# --------------------------------------------------------------------------
# _select_bulk_entry
# --------------------------------------------------------------------------


def test_select_bulk_entry_finds_matching_type() -> None:
    entries = [
        {"type": "oracle_cards", "download_uri": "x"},
        {"type": "default_cards", "download_uri": "y"},
        {"type": "all_cards", "download_uri": "z"},
    ]
    entry = _select_bulk_entry(entries, "default_cards")
    assert entry is not None
    assert entry["download_uri"] == "y"


def test_select_bulk_entry_returns_none_on_miss() -> None:
    entries = [{"type": "oracle_cards"}]
    assert _select_bulk_entry(entries, "default_cards") is None


def test_select_bulk_entry_skips_non_dict_entries() -> None:
    entries = ["not a dict", {"type": "default_cards", "download_uri": "u"}]  # type: ignore[list-item]
    entry = _select_bulk_entry(entries, "default_cards")  # type: ignore[arg-type]
    assert entry is not None
    assert entry["download_uri"] == "u"


# --------------------------------------------------------------------------
# download_bulk_file
# --------------------------------------------------------------------------


def test_download_bulk_file_writes_full_payload(tmp_path: Path) -> None:
    payload = b"x" * (3 * 1024 * 1024 + 17)  # 3 MB + a tail to exercise multiple chunks

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "data.scryfall.io"
        return httpx.Response(200, content=payload)

    dest = tmp_path / "bulk" / "default_cards.json"
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        size = download_bulk_file(
            client,
            "https://data.scryfall.io/default_cards/default_cards.json",
            dest,
        )
    assert size == len(payload)
    assert dest.exists()
    assert dest.read_bytes() == payload
    # The .part sidecar should have been renamed away.
    assert not dest.with_suffix(dest.suffix + ".part").exists()


def test_download_bulk_file_raises_on_non_200(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    dest = tmp_path / "bulk.json"
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(Exception):
            download_bulk_file(
                client,
                "https://data.scryfall.io/x.json",
                dest,
            )
    assert not dest.exists()


# --------------------------------------------------------------------------
# iter_cards_from_bulk
# --------------------------------------------------------------------------


def test_iter_cards_from_bulk_yields_each_card(tmp_path: Path) -> None:
    cards = [_card(variant_id=f"u-{i}") for i in range(5)]
    path = tmp_path / "default_cards.json"
    path.write_text(json.dumps(cards), encoding="utf-8")

    out = list(iter_cards_from_bulk(path))
    assert len(out) == 5
    assert [c["id"] for c in out] == [f"u-{i}" for i in range(5)]


def test_iter_cards_from_bulk_rejects_non_list_top_level(tmp_path: Path) -> None:
    path = tmp_path / "wrong.json"
    path.write_text(json.dumps({"data": []}), encoding="utf-8")
    with pytest.raises(ValueError):
        list(iter_cards_from_bulk(path))


def test_iter_cards_from_bulk_skips_non_dict_entries(tmp_path: Path) -> None:
    """Mixed-type top-level array. Strings/ints are skipped silently —
    same fail-soft contract the rest of the ingestion layer uses."""
    path = tmp_path / "mixed.json"
    path.write_text(
        json.dumps([_card(variant_id="ok-1"), "junk", 42, _card(variant_id="ok-2")]),
        encoding="utf-8",
    )
    out = list(iter_cards_from_bulk(path))
    assert [c["id"] for c in out] == ["ok-1", "ok-2"]


# --------------------------------------------------------------------------
# ingest_bulk — end-to-end
# --------------------------------------------------------------------------


def _ingest_bulk_handler(
    *,
    download_uri: str,
    cards: list[dict],
    image_bytes: bytes,
    metadata_status: int = 200,
    download_status: int = 200,
):
    """Build an httpx request handler for the full bulk flow.

    Three URL families are routed:
        - api.scryfall.com/bulk-data         → metadata payload
        - download_uri (any host)             → bulk file payload (cards list as JSON)
        - cards.scryfall.io/...               → image bytes
    """
    metadata = _bulk_metadata_payload(download_uri=download_uri)
    bulk_body = json.dumps(cards).encode("utf-8")

    def handler(request: httpx.Request) -> httpx.Response:
        url_str = str(request.url)
        if url_str.endswith("/bulk-data"):
            return httpx.Response(metadata_status, json=metadata)
        if url_str == download_uri:
            return httpx.Response(download_status, content=bulk_body)
        if request.url.host == "cards.scryfall.io":
            return httpx.Response(
                200, content=image_bytes, headers={"Content-Type": "image/jpeg"}
            )
        return httpx.Response(404)

    return handler


def _patched_client(handler):
    """Context-manager-style helper that monkeypatches httpx.Client to
    use the given mock transport. Mirrors the pattern used in
    test_scryfall_client to keep ingest_query's signature unchanged."""

    class _Patcher:
        def __enter__(self):
            self._real = httpx.Client.__init__

            def patched_init(s, *args, **kwargs):
                kwargs.setdefault("transport", httpx.MockTransport(handler))
                return self._real(s, *args, **kwargs)

            httpx.Client.__init__ = patched_init  # type: ignore[method-assign]
            return self

        def __exit__(self, exc_type, exc, tb):
            httpx.Client.__init__ = self._real  # type: ignore[method-assign]
            return False

    return _Patcher()


def test_ingest_bulk_writes_records_and_images(tmp_path: Path) -> None:
    download_uri = "https://data.scryfall.io/default_cards/default_cards-2026.json"
    cards = [_card(variant_id=f"u-{i}", collector_number=str(i)) for i in range(3)]
    handler = _ingest_bulk_handler(
        download_uri=download_uri, cards=cards, image_bytes=_png_1x1_bytes()
    )

    store = LocalReferenceStore(tmp_path / "store")
    cache_dir = tmp_path / "cache"

    with _patched_client(handler):
        stats = ingest_bulk(
            store=store,
            cache_dir=cache_dir,
            request_delay_s=0.0,
        )

    assert stats.successful == 3
    assert stats.cards_seen == 3
    assert stats.image_failures == 0
    # All three records on disk.
    lines = (tmp_path / "store" / "references.jsonl").read_text(
        encoding="utf-8"
    ).strip().splitlines()
    assert len(lines) == 3
    image_files = list((tmp_path / "store" / "images" / "mtg").rglob("front.jpg"))
    assert len(image_files) == 3
    # Cache file removed by default.
    assert not (cache_dir / "default_cards.json").exists()


def test_ingest_bulk_respects_max_cards(tmp_path: Path) -> None:
    download_uri = "https://data.scryfall.io/default_cards/x.json"
    cards = [_card(variant_id=f"u-{i}") for i in range(10)]
    handler = _ingest_bulk_handler(
        download_uri=download_uri, cards=cards, image_bytes=_png_1x1_bytes()
    )

    store = LocalReferenceStore(tmp_path / "store")
    with _patched_client(handler):
        stats = ingest_bulk(
            store=store,
            cache_dir=tmp_path / "cache",
            request_delay_s=0.0,
            max_cards=4,
        )
    assert stats.successful == 4


def test_ingest_bulk_raises_when_bulk_type_missing(tmp_path: Path) -> None:
    """If /bulk-data doesn't list the requested type, surface as RuntimeError
    so the CLI logs a clean error rather than silently doing nothing."""
    download_uri = "https://data.scryfall.io/default_cards/x.json"
    metadata = _bulk_metadata_payload(
        download_uri=download_uri,
        # Nothing here matches "default_cards".
        bulk_type="oracle_cards",
        extra_types=("all_cards",),
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url).endswith("/bulk-data"):
            return httpx.Response(200, json=metadata)
        return httpx.Response(404)

    store = LocalReferenceStore(tmp_path / "store")
    with _patched_client(handler):
        with pytest.raises(RuntimeError, match="default_cards"):
            ingest_bulk(
                store=store,
                cache_dir=tmp_path / "cache",
                request_delay_s=0.0,
            )


def test_ingest_bulk_keep_cache_retains_file(tmp_path: Path) -> None:
    download_uri = "https://data.scryfall.io/default_cards/x.json"
    cards = [_card(variant_id="u-1")]
    handler = _ingest_bulk_handler(
        download_uri=download_uri, cards=cards, image_bytes=_png_1x1_bytes()
    )

    cache_dir = tmp_path / "cache"
    store = LocalReferenceStore(tmp_path / "store")
    with _patched_client(handler):
        ingest_bulk(
            store=store,
            cache_dir=cache_dir,
            keep_cache=True,
            request_delay_s=0.0,
        )
    assert (cache_dir / "default_cards.json").exists()


def test_ingest_bulk_propagates_download_failure(tmp_path: Path) -> None:
    """A 5xx on the bulk-file download is fatal — the run can't proceed
    without the cards. Should bump network_errors and raise."""
    download_uri = "https://data.scryfall.io/default_cards/x.json"
    cards = [_card(variant_id="u-1")]
    handler = _ingest_bulk_handler(
        download_uri=download_uri,
        cards=cards,
        image_bytes=_png_1x1_bytes(),
        download_status=503,
    )

    store = LocalReferenceStore(tmp_path / "store")
    stats = ScryfallIngestStats()
    with _patched_client(handler):
        with pytest.raises(RuntimeError):
            ingest_bulk(
                store=store,
                cache_dir=tmp_path / "cache",
                request_delay_s=0.0,
                stats=stats,
            )
    assert stats.network_errors == 1
    assert stats.successful == 0


def test_ingest_bulk_skips_already_present(tmp_path: Path) -> None:
    """Idempotency: cards whose (manufacturer, variant_id) is already in
    the store should bump skipped_already_present, never network-fetch
    the image."""
    download_uri = "https://data.scryfall.io/default_cards/x.json"
    cards = [_card(variant_id="u-1"), _card(variant_id="u-2")]

    image_calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url).endswith("/bulk-data"):
            return httpx.Response(
                200, json=_bulk_metadata_payload(download_uri=download_uri)
            )
        if str(request.url) == download_uri:
            return httpx.Response(200, content=json.dumps(cards).encode("utf-8"))
        if request.url.host == "cards.scryfall.io":
            image_calls.append(str(request.url))
            return httpx.Response(
                200,
                content=_png_1x1_bytes(),
                headers={"Content-Type": "image/jpeg"},
            )
        return httpx.Response(404)

    # Pre-populate u-1 so only u-2 should hit the image endpoint.
    from data.ingestion.references_storage import ReferenceRecord

    store = LocalReferenceStore(tmp_path / "store")
    store.write_record(
        ReferenceRecord(
            variant_id="u-1",
            manufacturer="mtg",
            name="Already There",
            set_code="LEA",
            collector_number="1",
            rarity="rare",
            front_image_path=None,
            source_url="x",
            fetched_at="2026-04-30T00:00:00Z",
        )
    )

    with _patched_client(handler):
        stats = ingest_bulk(
            store=store,
            cache_dir=tmp_path / "cache",
            request_delay_s=0.0,
        )
    assert stats.skipped_already_present == 1
    assert stats.successful == 1
    # Only u-2 should have fetched its image.
    assert len(image_calls) == 1
    assert "u-2" in image_calls[0]
