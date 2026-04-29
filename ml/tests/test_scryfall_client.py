"""Scryfall reference-ingest client tests.

All HTTP is mocked via httpx.MockTransport — we never touch real
api.scryfall.com. Tests cover:
    - _extract_image_url handles single-faced, double-faced, and missing
    - _build_record validates required fields, falls back on optional ones
    - iter_cards_for_query paginates correctly via has_more / next_page
    - ingest_card skips already-present, persists records and images
    - 404 yields empty (matches Scryfall's "no matches" semantics)
    - 429 propagates so caller can back off
    - Image-download failure leaves front_image_path=None but writes the record
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import httpx
import pytest

# Ensure ml/ root is on sys.path when running this test directly.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.ingestion.references_storage import LocalReferenceStore
from data.ingestion.scryfall import (
    MANUFACTURER,
    SCRYFALL_API_BASE_URL,
    ScryfallIngestStats,
    _build_record,
    _extract_image_url,
    ingest_card,
    ingest_query,
    iter_cards_for_query,
)


# --------------------------------------------------------------------------
# Fixture helpers
# --------------------------------------------------------------------------


_OMITTED = object()  # sentinel: distinguish "not specified" from explicit None


def _card(
    *,
    variant_id: str = "uuid-1",
    name: str = "Black Lotus",
    set_code: str = "lea",
    collector_number: str = "232",
    rarity: str = "rare",
    image_uris=_OMITTED,
    layout: str = "normal",
    card_faces=_OMITTED,
    scryfall_uri: str = "https://scryfall.com/card/lea/232/black-lotus",
) -> dict:
    """Build a synthetic Scryfall card row.

    Default behavior fills `image_uris` with all four sizes. To test the
    "no images" path pass `image_uris=None` explicitly — that omits the
    field entirely (different from the default-applied case)."""
    if image_uris is _OMITTED and card_faces is _OMITTED:
        image_uris = {
            "small": "https://cards.scryfall.io/small/front/u/u/uuid-1.jpg",
            "normal": "https://cards.scryfall.io/normal/front/u/u/uuid-1.jpg",
            "large": "https://cards.scryfall.io/large/front/u/u/uuid-1.jpg",
            "png": "https://cards.scryfall.io/png/front/u/u/uuid-1.png",
        }
    out: dict = {
        "id": variant_id,
        "name": name,
        "set": set_code,
        "collector_number": collector_number,
        "rarity": rarity,
        "lang": "en",
        "layout": layout,
        "scryfall_uri": scryfall_uri,
    }
    if image_uris is not _OMITTED and image_uris is not None:
        out["image_uris"] = image_uris
    if card_faces is not _OMITTED and card_faces is not None:
        out["card_faces"] = card_faces
    return out


def _search_payload(cards: list[dict], *, has_more: bool = False, next_page: str | None = None) -> dict:
    payload: dict = {"object": "list", "data": cards, "has_more": has_more}
    if next_page is not None:
        payload["next_page"] = next_page
    return payload


# --------------------------------------------------------------------------
# Pure parsers
# --------------------------------------------------------------------------


def test_extract_image_url_prefers_large() -> None:
    url = _extract_image_url(_card())
    assert url is not None
    assert url.endswith("/large/front/u/u/uuid-1.jpg")


def test_extract_image_url_falls_back_to_normal_when_no_large() -> None:
    card = _card(image_uris={
        "small": "https://x/small.jpg",
        "normal": "https://x/normal.jpg",
    })
    url = _extract_image_url(card)
    assert url == "https://x/normal.jpg"


def test_extract_image_url_uses_card_faces_for_double_faced() -> None:
    """Double-faced layouts (transform, modal_dfc) put image_uris on each
    face, not the top level. We use the front face."""
    card = _card(
        image_uris=None,
        layout="transform",
        card_faces=[
            {
                "name": "Bloodbraid Marauder",
                "image_uris": {
                    "small": "https://x/small.jpg",
                    "normal": "https://x/normal.jpg",
                    "large": "https://x/large.jpg",
                },
            },
            {
                "name": "Chittering Witch",
                "image_uris": {
                    "small": "https://x/back-small.jpg",
                    "large": "https://x/back-large.jpg",
                },
            },
        ],
    )
    url = _extract_image_url(card)
    assert url == "https://x/large.jpg"


def test_extract_image_url_returns_none_when_no_images() -> None:
    card = _card(image_uris=None)
    assert _extract_image_url(card) is None


def test_extract_image_url_skips_non_https_urls() -> None:
    card = _card(image_uris={
        "large": "ftp://x/large.jpg",
        "normal": "https://x/normal.jpg",
    })
    # large fails the https check; we fall back to normal.
    url = _extract_image_url(card)
    assert url == "https://x/normal.jpg"


def test_build_record_succeeds_on_full_card() -> None:
    record = _build_record(_card(), front_image_path="/tmp/x.jpg")
    assert record is not None
    assert record.variant_id == "uuid-1"
    assert record.manufacturer == MANUFACTURER
    assert record.name == "Black Lotus"
    assert record.set_code == "LEA"  # uppercased
    assert record.collector_number == "232"
    assert record.rarity == "rare"
    assert record.front_image_path == "/tmp/x.jpg"
    assert record.source_url.startswith("https://scryfall.com/card/")
    assert record.fetched_at.endswith("Z")


def test_build_record_rejects_missing_id() -> None:
    card = _card()
    del card["id"]
    assert _build_record(card, front_image_path=None) is None


def test_build_record_rejects_missing_name() -> None:
    card = _card()
    card["name"] = ""
    assert _build_record(card, front_image_path=None) is None


def test_build_record_handles_missing_optional_fields() -> None:
    card = {"id": "uuid-x", "name": "Nameless"}
    record = _build_record(card, front_image_path=None)
    assert record is not None
    assert record.set_code is None
    assert record.collector_number is None
    assert record.rarity is None
    # Falls back to a synthetic source URL.
    assert "uuid-x" in record.source_url


# --------------------------------------------------------------------------
# Mock transport — pagination
# --------------------------------------------------------------------------


def _mock_transport(handler) -> httpx.MockTransport:
    return httpx.MockTransport(handler)


def test_iter_cards_for_query_handles_single_page() -> None:
    cards = [_card(variant_id=f"u-{i}", collector_number=str(i)) for i in range(3)]

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/cards/search"
        assert "set%3Alea" in str(request.url) or "set:lea" in str(request.url)
        return httpx.Response(200, json=_search_payload(cards, has_more=False))

    with httpx.Client(transport=_mock_transport(handler)) as client:
        out = list(iter_cards_for_query(
            "set:lea",
            client=client,
            base_url=SCRYFALL_API_BASE_URL,
            request_delay_s=0.0,
        ))
    assert len(out) == 3
    assert [c["id"] for c in out] == ["u-0", "u-1", "u-2"]


def test_iter_cards_for_query_follows_next_page() -> None:
    page1 = [_card(variant_id="u-1"), _card(variant_id="u-2")]
    page2 = [_card(variant_id="u-3")]
    next_page_url = "https://api.scryfall.com/cards/search?page=2&q=set:lea&unique=prints"

    pages_seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        pages_seen.append(str(request.url))
        if "page=2" in str(request.url):
            return httpx.Response(200, json=_search_payload(page2, has_more=False))
        return httpx.Response(
            200, json=_search_payload(page1, has_more=True, next_page=next_page_url)
        )

    stats = ScryfallIngestStats()
    with httpx.Client(transport=_mock_transport(handler)) as client:
        out = list(iter_cards_for_query(
            "set:lea",
            client=client,
            base_url=SCRYFALL_API_BASE_URL,
            request_delay_s=0.0,
            stats=stats,
        ))
    assert len(out) == 3
    assert stats.pages_fetched == 2
    assert stats.cards_seen == 3
    assert len(pages_seen) == 2


def test_iter_cards_for_query_yields_nothing_on_404() -> None:
    """A 404 from /cards/search means the query matched no cards. That's
    not an error — Scryfall returns 404 for empty result sets."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"object": "error", "code": "not_found"})

    stats = ScryfallIngestStats()
    with httpx.Client(transport=_mock_transport(handler)) as client:
        out = list(iter_cards_for_query(
            "set:doesnotexist",
            client=client,
            base_url=SCRYFALL_API_BASE_URL,
            request_delay_s=0.0,
            stats=stats,
        ))
    assert out == []
    assert stats.not_found == 1


def test_iter_cards_for_query_propagates_429() -> None:
    """Rate-limited responses should propagate so the caller can back off
    rather than swallow the error."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429)

    stats = ScryfallIngestStats()
    with httpx.Client(transport=_mock_transport(handler)) as client:
        with pytest.raises(Exception):
            list(iter_cards_for_query(
                "set:lea",
                client=client,
                base_url=SCRYFALL_API_BASE_URL,
                request_delay_s=0.0,
                stats=stats,
            ))
    assert stats.rate_limited == 1


# --------------------------------------------------------------------------
# ingest_card — full record + image flow
# --------------------------------------------------------------------------


def _png_1x1_bytes() -> bytes:
    """Smallest valid PNG (1x1, transparent). Used as fake image bytes
    so anything that tries to decode it actually succeeds."""
    return bytes.fromhex(
        "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4"
        "890000000d49444154789c63000100000005000146cea2270000000049454e44"
        "ae426082"
    )


def test_ingest_card_persists_record_and_image(tmp_path: Path) -> None:
    image_bytes = _png_1x1_bytes()

    def handler(request: httpx.Request) -> httpx.Response:
        # Image GET (CDN-style URL).
        if request.url.host == "cards.scryfall.io":
            return httpx.Response(200, content=image_bytes, headers={"Content-Type": "image/jpeg"})
        return httpx.Response(404)

    store = LocalReferenceStore(tmp_path)
    stats = ScryfallIngestStats()
    with httpx.Client(transport=_mock_transport(handler)) as client:
        record = ingest_card(
            _card(),
            client=client,
            store=store,
            request_delay_s=0.0,
            stats=stats,
        )
    assert record is not None
    assert record.variant_id == "uuid-1"
    assert record.front_image_path is not None
    assert Path(record.front_image_path).exists()
    assert Path(record.front_image_path).read_bytes() == image_bytes
    assert stats.successful == 1
    assert stats.image_failures == 0


def test_ingest_card_skips_already_present(tmp_path: Path) -> None:
    """Pre-populate the store with a record for uuid-1, then call
    ingest_card on a card with the same id — the function should hit the
    skip path before ever touching the HTTP transport."""
    from data.ingestion.references_storage import ReferenceRecord

    store = LocalReferenceStore(tmp_path)
    store.write_record(ReferenceRecord(
        variant_id="uuid-1",
        manufacturer=MANUFACTURER,
        name="Already There",
        set_code="LEA",
        collector_number="1",
        rarity="rare",
        front_image_path=None,
        source_url="x",
        fetched_at="2026-04-29T00:00:00Z",
    ))

    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"unexpected HTTP call: {request.url}")

    stats = ScryfallIngestStats()
    with httpx.Client(transport=_mock_transport(handler)) as client:
        result = ingest_card(
            _card(variant_id="uuid-1"),
            client=client,
            store=store,
            request_delay_s=0.0,
            stats=stats,
        )
    assert result is None
    assert stats.skipped_already_present == 1
    assert stats.successful == 0


def test_ingest_card_persists_record_when_image_download_fails(tmp_path: Path) -> None:
    """If the image GET returns 5xx the record should still be written
    with front_image_path=None — we know the variant exists even if we
    couldn't grab bytes for it."""
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "cards.scryfall.io":
            return httpx.Response(503)
        return httpx.Response(404)

    store = LocalReferenceStore(tmp_path)
    stats = ScryfallIngestStats()
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
    """A card row without an `id` is unusable. Should bump parse_errors
    and not write anything."""
    store = type("_Sentinel", (), {
        "has_record": lambda self, m, v: False,
        "write_record": lambda self, r: pytest.fail(f"should not write {r!r}"),
        "write_image": lambda self, *a, **k: pytest.fail("should not write image"),
    })()
    stats = ScryfallIngestStats()
    bad = _card()
    del bad["id"]

    def handler(request: httpx.Request) -> httpx.Response:
        # A real client wouldn't reach here for a parse failure (no id =
        # no image url to fetch since extract returns None first), but
        # we keep the transport in case behavior changes.
        return httpx.Response(404)

    with httpx.Client(transport=_mock_transport(handler)) as client:
        record = ingest_card(
            bad,
            client=client,
            store=store,
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
    page1 = [_card(variant_id="u-1"), _card(variant_id="u-2")]
    page2 = [_card(variant_id="u-3")]
    next_page_url = (
        "https://api.scryfall.com/cards/search?page=2&q=set:lea&unique=prints"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "cards.scryfall.io":
            return httpx.Response(
                200, content=image_bytes, headers={"Content-Type": "image/jpeg"}
            )
        if "page=2" in str(request.url):
            return httpx.Response(200, json=_search_payload(page2, has_more=False))
        return httpx.Response(
            200, json=_search_payload(page1, has_more=True, next_page=next_page_url)
        )

    store = LocalReferenceStore(tmp_path)

    # Patch httpx.Client so ingest_query uses our mock transport. The
    # function constructs its own client internally; passing transport
    # via context-manager monkeypatch keeps ingest_query's signature
    # unchanged while still routing through the mock.
    real_init = httpx.Client.__init__

    def patched_init(self, *args, **kwargs):
        kwargs.setdefault("transport", _mock_transport(handler))
        return real_init(self, *args, **kwargs)

    try:
        httpx.Client.__init__ = patched_init  # type: ignore[method-assign]
        stats = ingest_query(
            "set:lea",
            store=store,
            request_delay_s=0.0,
        )
    finally:
        httpx.Client.__init__ = real_init  # type: ignore[method-assign]

    assert stats.successful == 3
    assert stats.pages_fetched == 2
    # Three records persisted, three images on disk.
    lines = (tmp_path / "references.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3
    image_files = list((tmp_path / "images" / "mtg").rglob("front.jpg"))
    assert len(image_files) == 3


def test_ingest_query_respects_max_cards(tmp_path: Path) -> None:
    image_bytes = _png_1x1_bytes()
    cards = [_card(variant_id=f"u-{i}") for i in range(5)]

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "cards.scryfall.io":
            return httpx.Response(
                200, content=image_bytes, headers={"Content-Type": "image/jpeg"}
            )
        return httpx.Response(200, json=_search_payload(cards, has_more=False))

    real_init = httpx.Client.__init__

    def patched_init(self, *args, **kwargs):
        kwargs.setdefault("transport", _mock_transport(handler))
        return real_init(self, *args, **kwargs)

    store = LocalReferenceStore(tmp_path)
    try:
        httpx.Client.__init__ = patched_init  # type: ignore[method-assign]
        stats = ingest_query(
            "set:lea",
            store=store,
            request_delay_s=0.0,
            max_cards=2,
        )
    finally:
        httpx.Client.__init__ = real_init  # type: ignore[method-assign]

    assert stats.successful == 2
