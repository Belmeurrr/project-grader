"""PokemonTCG.io reference-ingest client tests.

All HTTP is mocked via httpx.MockTransport — we never touch real
api.pokemontcg.io. Tests cover:
    - _extract_image_url (large preferred, small fallback, absent)
    - _build_record validates required fields, falls back on optional ones
    - iter_cards_for_query paginates correctly via page+totalCount
    - ingest_card skips already-present, persists records and images
    - empty page on first request -> not_found
    - 429 propagates so caller can back off
    - Image-download failure leaves front_image_path=None
    - X-Api-Key passthrough when provided
"""

from __future__ import annotations

import sys
from pathlib import Path

import httpx
import pytest

# Ensure ml/ root is on sys.path when running this test directly.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.ingestion.pokemontcg import (
    MANUFACTURER,
    POKEMONTCG_API_BASE_URL,
    PokemonTCGIngestStats,
    _build_record,
    _extract_image_url,
    ingest_card,
    ingest_query,
    iter_cards_for_query,
)
from data.ingestion.references_storage import LocalReferenceStore, ReferenceRecord


# --------------------------------------------------------------------------
# Fixture helpers
# --------------------------------------------------------------------------


_OMITTED = object()


def _card(
    *,
    variant_id: str = "base1-1",
    name: str = "Alakazam",
    set_id: str = "base1",
    set_name: str = "Base",
    number: str = "1",
    rarity: str = "Rare Holo",
    images=_OMITTED,
) -> dict:
    """Build a synthetic PokemonTCG.io v2 card row.

    Default fills `images` with both `large` and `small`. Pass
    `images=None` explicitly to omit the field entirely (tests the
    "no images" path)."""
    if images is _OMITTED:
        images = {
            "small": "https://images.pokemontcg.io/base1/1.png",
            "large": "https://images.pokemontcg.io/base1/1_hires.png",
        }
    out: dict = {
        "id": variant_id,
        "name": name,
        "number": number,
        "rarity": rarity,
        "set": {"id": set_id, "name": set_name, "ptcgoCode": "BS"},
    }
    if images is not None:
        out["images"] = images
    return out


def _search_payload(
    cards: list[dict],
    *,
    page: int = 1,
    page_size: int = 250,
    total_count: int | None = None,
) -> dict:
    if total_count is None:
        total_count = len(cards)
    return {
        "data": cards,
        "page": page,
        "pageSize": page_size,
        "count": len(cards),
        "totalCount": total_count,
    }


def _png_1x1_bytes() -> bytes:
    """Smallest valid PNG. Used as fake image bytes."""
    return bytes.fromhex(
        "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4"
        "890000000d49444154789c63000100000005000146cea2270000000049454e44"
        "ae426082"
    )


def _mock_transport(handler) -> httpx.MockTransport:
    return httpx.MockTransport(handler)


# --------------------------------------------------------------------------
# Pure parsers
# --------------------------------------------------------------------------


def test_extract_image_url_prefers_large() -> None:
    url = _extract_image_url(_card())
    assert url == "https://images.pokemontcg.io/base1/1_hires.png"


def test_extract_image_url_falls_back_to_small_when_no_large() -> None:
    card = _card(images={"small": "https://x/small.png"})
    assert _extract_image_url(card) == "https://x/small.png"


def test_extract_image_url_returns_none_when_no_images() -> None:
    assert _extract_image_url(_card(images=None)) is None


def test_extract_image_url_skips_non_https_urls() -> None:
    card = _card(images={"large": "ftp://x/large.png", "small": "https://x/small.png"})
    assert _extract_image_url(card) == "https://x/small.png"


def test_build_record_succeeds_on_full_card() -> None:
    record = _build_record(_card(), front_image_path="/tmp/x.png")
    assert record is not None
    assert record.variant_id == "base1-1"
    assert record.manufacturer == MANUFACTURER
    assert record.name == "Alakazam"
    assert record.set_code == "BASE1"  # uppercased
    assert record.collector_number == "1"
    assert record.rarity == "Rare Holo"
    assert record.front_image_path == "/tmp/x.png"
    assert record.source_url == "https://pokemontcg.io/cards/base1-1"
    assert record.fetched_at.endswith("Z")


def test_build_record_rejects_missing_id() -> None:
    card = _card()
    del card["id"]
    assert _build_record(card, front_image_path=None) is None


def test_build_record_rejects_empty_name() -> None:
    card = _card(name="")
    assert _build_record(card, front_image_path=None) is None


def test_build_record_handles_missing_optional_fields() -> None:
    record = _build_record(
        {"id": "swsh1-99", "name": "Mystery Card"},
        front_image_path=None,
    )
    assert record is not None
    assert record.set_code is None
    assert record.collector_number is None
    assert record.rarity is None
    # Source URL is synthesized from the id.
    assert record.source_url == "https://pokemontcg.io/cards/swsh1-99"


def test_build_record_handles_set_with_no_id_field() -> None:
    """Some old PokemonTCG.io entries have a `set` block without an
    `id` key. We accept the record but leave set_code as None."""
    card = _card()
    card["set"] = {"name": "Mystery Set"}  # no id
    record = _build_record(card, front_image_path=None)
    assert record is not None
    assert record.set_code is None


# --------------------------------------------------------------------------
# Mock transport — pagination
# --------------------------------------------------------------------------


def test_iter_cards_for_query_handles_single_page() -> None:
    cards = [_card(variant_id=f"base1-{i}", number=str(i)) for i in range(1, 4)]

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v2/cards"
        # PokemonTCG.io uses Lucene `field:value` syntax, e.g. "set.id:base1".
        # Quote-encoded form: "set.id%3Abase1".
        assert "set.id%3Abase1" in str(request.url) or "set.id:base1" in str(request.url)
        return httpx.Response(200, json=_search_payload(cards, total_count=3))

    with httpx.Client(transport=_mock_transport(handler)) as client:
        out = list(iter_cards_for_query(
            "set.id:base1",
            client=client,
            base_url=POKEMONTCG_API_BASE_URL,
            page_size=250,
            request_delay_s=0.0,
        ))
    assert len(out) == 3
    assert [c["id"] for c in out] == ["base1-1", "base1-2", "base1-3"]


def test_iter_cards_for_query_paginates_until_total_count_reached() -> None:
    """Pagination uses page * pageSize >= totalCount as the stop signal,
    NOT a `has_more` flag. Two pages of 2 cards each, totalCount=4."""
    page1 = [_card(variant_id="base1-1"), _card(variant_id="base1-2")]
    page2 = [_card(variant_id="base1-3"), _card(variant_id="base1-4")]
    pages_seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        url_str = str(request.url)
        pages_seen.append(url_str)
        if "page=2" in url_str:
            return httpx.Response(
                200, json=_search_payload(page2, page=2, page_size=2, total_count=4)
            )
        return httpx.Response(
            200, json=_search_payload(page1, page=1, page_size=2, total_count=4)
        )

    stats = PokemonTCGIngestStats()
    with httpx.Client(transport=_mock_transport(handler)) as client:
        out = list(iter_cards_for_query(
            "set.id:base1",
            client=client,
            base_url=POKEMONTCG_API_BASE_URL,
            page_size=2,
            request_delay_s=0.0,
            stats=stats,
        ))
    assert len(out) == 4
    assert stats.pages_fetched == 2
    assert stats.cards_seen == 4
    assert len(pages_seen) == 2


def test_iter_cards_for_query_stops_on_short_page() -> None:
    """If the server returns fewer items than pageSize before totalCount
    is reached (stale total, race condition), bail rather than loop
    forever."""
    page1 = [_card(variant_id="base1-1")]  # only 1 card

    def handler(request: httpx.Request) -> httpx.Response:
        # totalCount lies — claims 5 but we only return 1
        return httpx.Response(
            200, json=_search_payload(page1, page=1, page_size=2, total_count=5)
        )

    stats = PokemonTCGIngestStats()
    with httpx.Client(transport=_mock_transport(handler)) as client:
        out = list(iter_cards_for_query(
            "set.id:base1",
            client=client,
            base_url=POKEMONTCG_API_BASE_URL,
            page_size=2,
            request_delay_s=0.0,
            stats=stats,
        ))
    assert len(out) == 1
    assert stats.pages_fetched == 1


def test_iter_cards_for_query_yields_nothing_on_empty_first_page() -> None:
    """Unlike Scryfall (which 404s for empty queries), PokemonTCG.io
    returns 200 with an empty data array. Treat as not_found."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json=_search_payload([], total_count=0)
        )

    stats = PokemonTCGIngestStats()
    with httpx.Client(transport=_mock_transport(handler)) as client:
        out = list(iter_cards_for_query(
            "set.id:doesnotexist",
            client=client,
            base_url=POKEMONTCG_API_BASE_URL,
            request_delay_s=0.0,
            stats=stats,
        ))
    assert out == []
    assert stats.not_found == 1


def test_iter_cards_for_query_propagates_429() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429)

    stats = PokemonTCGIngestStats()
    with httpx.Client(transport=_mock_transport(handler)) as client:
        with pytest.raises(Exception):
            list(iter_cards_for_query(
                "set.id:base1",
                client=client,
                base_url=POKEMONTCG_API_BASE_URL,
                request_delay_s=0.0,
                stats=stats,
            ))
    assert stats.rate_limited == 1


def test_iter_cards_for_query_sends_api_key_header_when_provided() -> None:
    """X-Api-Key passthrough: if the caller supplies an api_key, it
    should appear on every request."""
    seen_keys: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_keys.append(request.headers.get("x-api-key"))
        return httpx.Response(
            200, json=_search_payload([_card()], total_count=1)
        )

    with httpx.Client(transport=_mock_transport(handler)) as client:
        list(iter_cards_for_query(
            "set.id:base1",
            client=client,
            base_url=POKEMONTCG_API_BASE_URL,
            api_key="test-key-abc",
            request_delay_s=0.0,
        ))
    assert seen_keys == ["test-key-abc"]


def test_iter_cards_for_query_omits_api_key_when_absent() -> None:
    seen_keys: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_keys.append(request.headers.get("x-api-key"))
        return httpx.Response(
            200, json=_search_payload([_card()], total_count=1)
        )

    with httpx.Client(transport=_mock_transport(handler)) as client:
        list(iter_cards_for_query(
            "set.id:base1",
            client=client,
            base_url=POKEMONTCG_API_BASE_URL,
            api_key=None,
            request_delay_s=0.0,
        ))
    assert seen_keys == [None]


# --------------------------------------------------------------------------
# ingest_card
# --------------------------------------------------------------------------


def test_ingest_card_persists_record_and_image(tmp_path: Path) -> None:
    image_bytes = _png_1x1_bytes()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "images.pokemontcg.io":
            return httpx.Response(
                200, content=image_bytes, headers={"Content-Type": "image/png"}
            )
        return httpx.Response(404)

    store = LocalReferenceStore(tmp_path)
    stats = PokemonTCGIngestStats()
    with httpx.Client(transport=_mock_transport(handler)) as client:
        record = ingest_card(
            _card(),
            client=client,
            store=store,
            request_delay_s=0.0,
            stats=stats,
        )
    assert record is not None
    assert record.variant_id == "base1-1"
    assert record.manufacturer == "pokemon"
    assert record.front_image_path is not None
    assert Path(record.front_image_path).exists()
    assert Path(record.front_image_path).read_bytes() == image_bytes
    # Default content-type → png extension
    assert record.front_image_path.endswith(".png")
    assert stats.successful == 1


def test_ingest_card_skips_already_present(tmp_path: Path) -> None:
    store = LocalReferenceStore(tmp_path)
    store.write_record(ReferenceRecord(
        variant_id="base1-1",
        manufacturer=MANUFACTURER,
        name="Already There",
        set_code="BASE1",
        collector_number="1",
        rarity="Rare",
        front_image_path=None,
        source_url="x",
        fetched_at="2026-04-29T00:00:00Z",
    ))

    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"unexpected HTTP call: {request.url}")

    stats = PokemonTCGIngestStats()
    with httpx.Client(transport=_mock_transport(handler)) as client:
        result = ingest_card(
            _card(variant_id="base1-1"),
            client=client,
            store=store,
            request_delay_s=0.0,
            stats=stats,
        )
    assert result is None
    assert stats.skipped_already_present == 1
    assert stats.successful == 0


def test_ingest_card_persists_record_when_image_download_fails(tmp_path: Path) -> None:
    """Image GET 5xx → record still written with front_image_path=None."""
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "images.pokemontcg.io":
            return httpx.Response(503)
        return httpx.Response(404)

    store = LocalReferenceStore(tmp_path)
    stats = PokemonTCGIngestStats()
    with httpx.Client(transport=_mock_transport(handler)) as client:
        record = ingest_card(
            _card(),
            client=client,
            store=store,
            request_delay_s=0.0,
            stats=stats,
        )
    assert record is not None
    assert record.front_image_path is None
    assert stats.successful == 1
    assert stats.image_failures == 1


def test_ingest_card_records_parse_error_when_no_id() -> None:
    """A card row without an `id` is unusable — bump parse_errors and
    don't write anything."""
    class _Sentinel:
        def has_record(self, m, v):
            return False

        def write_record(self, r):
            raise AssertionError(f"should not write {r!r}")

        def write_image(self, *a, **k):
            raise AssertionError("should not write image")

    store = _Sentinel()
    stats = PokemonTCGIngestStats()
    bad = _card()
    del bad["id"]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    with httpx.Client(transport=_mock_transport(handler)) as client:
        record = ingest_card(
            bad,
            client=client,
            store=store,  # type: ignore[arg-type]
            request_delay_s=0.0,
            stats=stats,
        )
    assert record is None
    assert stats.parse_errors == 1
    assert stats.successful == 0


# --------------------------------------------------------------------------
# ingest_query — end-to-end with paging + image fetch
# --------------------------------------------------------------------------


def test_ingest_query_writes_records_and_images_through_paging(tmp_path: Path) -> None:
    image_bytes = _png_1x1_bytes()
    page1 = [_card(variant_id="base1-1"), _card(variant_id="base1-2")]
    page2 = [_card(variant_id="base1-3")]

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "images.pokemontcg.io":
            return httpx.Response(
                200, content=image_bytes, headers={"Content-Type": "image/png"}
            )
        url_str = str(request.url)
        if "page=2" in url_str:
            return httpx.Response(
                200, json=_search_payload(page2, page=2, page_size=2, total_count=3)
            )
        return httpx.Response(
            200, json=_search_payload(page1, page=1, page_size=2, total_count=3)
        )

    store = LocalReferenceStore(tmp_path)

    real_init = httpx.Client.__init__

    def patched_init(self, *args, **kwargs):
        kwargs.setdefault("transport", _mock_transport(handler))
        return real_init(self, *args, **kwargs)

    try:
        httpx.Client.__init__ = patched_init  # type: ignore[method-assign]
        stats = ingest_query(
            "set.id:base1",
            store=store,
            page_size=2,
            request_delay_s=0.0,
        )
    finally:
        httpx.Client.__init__ = real_init  # type: ignore[method-assign]

    assert stats.successful == 3
    assert stats.pages_fetched == 2
    lines = (tmp_path / "references.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3
    image_files = list((tmp_path / "images" / "pokemon").rglob("front.png"))
    assert len(image_files) == 3


def test_ingest_query_respects_max_cards(tmp_path: Path) -> None:
    image_bytes = _png_1x1_bytes()
    cards = [_card(variant_id=f"base1-{i}") for i in range(1, 6)]

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "images.pokemontcg.io":
            return httpx.Response(
                200, content=image_bytes, headers={"Content-Type": "image/png"}
            )
        return httpx.Response(
            200, json=_search_payload(cards, page=1, page_size=250, total_count=5)
        )

    real_init = httpx.Client.__init__

    def patched_init(self, *args, **kwargs):
        kwargs.setdefault("transport", _mock_transport(handler))
        return real_init(self, *args, **kwargs)

    store = LocalReferenceStore(tmp_path)
    try:
        httpx.Client.__init__ = patched_init  # type: ignore[method-assign]
        stats = ingest_query(
            "set.id:base1",
            store=store,
            request_delay_s=0.0,
            max_cards=2,
        )
    finally:
        httpx.Client.__init__ = real_init  # type: ignore[method-assign]

    assert stats.successful == 2


def test_ingest_query_picks_up_api_key_from_env(monkeypatch, tmp_path: Path) -> None:
    """If api_key is not passed but PTCG_API_KEY is set, the env value
    is used. Useful for daemons / cron jobs that get config via env."""
    image_bytes = _png_1x1_bytes()
    seen_keys: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "images.pokemontcg.io":
            return httpx.Response(
                200, content=image_bytes, headers={"Content-Type": "image/png"}
            )
        seen_keys.append(request.headers.get("x-api-key"))
        return httpx.Response(
            200, json=_search_payload([_card()], total_count=1)
        )

    real_init = httpx.Client.__init__

    def patched_init(self, *args, **kwargs):
        kwargs.setdefault("transport", _mock_transport(handler))
        return real_init(self, *args, **kwargs)

    monkeypatch.setenv("PTCG_API_KEY", "env-supplied-key")
    store = LocalReferenceStore(tmp_path)
    try:
        httpx.Client.__init__ = patched_init  # type: ignore[method-assign]
        ingest_query(
            "set.id:base1",
            store=store,
            request_delay_s=0.0,
        )
    finally:
        httpx.Client.__init__ = real_init  # type: ignore[method-assign]

    assert seen_keys == ["env-supplied-key"]
