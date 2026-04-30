"""PokemonTCG.io API client — manufacturer reference image ingest for Pokemon.

Why PokemonTCG.io:
    PokemonTCG.io is the de-facto authoritative catalog for Pokemon
    Trading Card Game — covers every modern set (~16k printings as of
    2026), exposes a public REST API at api.pokemontcg.io/v2, and
    serves card images on a CDN. Their data is the right "known
    authentic" reference for the embedding-anomaly detector.

Endpoints used:
    GET /v2/cards?q=<lucene-query>&page=<n>&pageSize=<m>
        Paginated search with a Lucene-syntax query. Each result is
        per-printing (the `id` field is e.g. "base1-1", unique per
        printing; the same Pokemon character across reprints has
        different ids). That's what we want — the embedding-anomaly
        detector needs per-printing reference embeddings, not per-
        species, because reprints have different art / borders /
        finishes.
    Image hosts: images.pokemontcg.io (CloudFront-style CDN). No
        auth, no API budget impact.

Rate limiting:
    PokemonTCG.io's free tier is 1000 calls/day with no API key (much
    higher with one). The docs don't pin a per-second ceiling but ask
    for "reasonable" usage. We use 100 ms post-call to be polite, same
    as the Scryfall client.

Required headers:
    User-Agent: a stable identifier with a project URL (so they can
        reach us if our traffic looks problematic).
    Accept: "application/json".
    X-Api-Key: optional. Set via env PTCG_API_KEY for the higher rate
        limit; absent on the free tier and that's fine.

Why we reuse the references storage layer:
    `ReferenceRecord` and the `*ReferenceStore` Protocol live in
    `references_storage.py` — backend-agnostic and source-agnostic.
    The Scryfall client writes records there with manufacturer="mtg";
    we write the same shape with manufacturer="pokemon", so resume/
    idempotency semantics carry over (each (manufacturer, variant_id)
    tuple is independent).

Error taxonomy (mapped to PokemonTCGIngestStats; mirrors Scryfall's):
    successful      — record persisted (with image)
    not_found       — empty page result (search returned no cards)
    parse_errors    — 200 returned but no usable image URL extracted
    image_failures  — record persisted but image download failed
    rate_limited    — 429 from server
    network_errors  — connect/read timeouts, 5xx after retries

Pagination differs from Scryfall:
    Scryfall returns `has_more` + `next_page` URL; PokemonTCG.io
    returns `count`, `totalCount`, `page`, `pageSize` and we compute
    "another page exists" client-side as
    `page * pageSize < totalCount`. That's the only structural
    difference between the two clients; everything else is parallel.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Iterator
from urllib.parse import quote

import httpx

if TYPE_CHECKING:
    from data.ingestion.references_storage import ReferenceRecordStore

from data.ingestion.references_storage import ReferenceRecord


_logger = logging.getLogger("pokemontcg_ingest")


# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

POKEMONTCG_API_BASE_URL = "https://api.pokemontcg.io"

# Manufacturer short-name used in storage paths and ReferenceRecord.manufacturer.
MANUFACTURER = "pokemon"

# 100 ms ≈ 10 req/s. PokemonTCG.io's free-tier cap is daily (1000 req/day),
# not per-second, but the same politeness delay we use for Scryfall keeps
# both clients consistent and well under any plausible per-second limit.
DEFAULT_REQUEST_DELAY_S: float = 0.1

# 250 is the v2 API's documented max page size; larger numbers are silently
# capped server-side. Picking 250 minimizes the number of paged requests.
DEFAULT_PAGE_SIZE: int = 250

DEFAULT_USER_AGENT: str = (
    "project-grader/0.1 (https://github.com/Belmeurrr/project-grader)"
)

# Image-resolution preference order. PokemonTCG.io exposes only `large` and
# `small`; `large` is a high-res PNG (~734×1024) close to our 750×1050
# canonical, so it's always preferred when available.
_IMAGE_SIZE_PREFERENCE: tuple[str, ...] = ("large", "small")


# --------------------------------------------------------------------------
# Stats
# --------------------------------------------------------------------------


@dataclass
class PokemonTCGIngestStats:
    """Counters for one ingest run.

    Same shape as `ScryfallIngestStats` — same dashboard / log-line
    consumers can read either source. Field names parallel the
    Scryfall stats so a future combined-source dashboard transfers
    directly."""

    pages_fetched: int = 0
    cards_seen: int = 0
    skipped_already_present: int = 0
    successful: int = 0
    not_found: int = 0
    parse_errors: int = 0
    image_failures: int = 0
    rate_limited: int = 0
    network_errors: int = 0


# --------------------------------------------------------------------------
# Pure parsers
# --------------------------------------------------------------------------


def _is_https_url(value: Any) -> bool:
    """A URL is acceptable if it parses as https with a netloc. No
    further validation — we trust PokemonTCG.io's CDN to serve images
    at their published URLs."""
    if not isinstance(value, str):
        return False
    from urllib.parse import urlparse

    parsed = urlparse(value)
    return parsed.scheme == "https" and bool(parsed.netloc)


def _extract_image_url(card: dict[str, Any]) -> str | None:
    """Return the best front-image URL from a PokemonTCG.io card record.

    Pure function. PokemonTCG.io exposes `images.large` (high-res PNG)
    and `images.small`; we prefer `large`. Returns None if neither
    yields a usable URL — caller maps that to `parse_errors`.
    """
    images = card.get("images")
    if not isinstance(images, dict):
        return None
    for size in _IMAGE_SIZE_PREFERENCE:
        url = images.get(size)
        if _is_https_url(url):
            return url
    return None


def _build_record(
    card: dict[str, Any],
    front_image_path: str | None,
) -> ReferenceRecord | None:
    """Assemble a ReferenceRecord from a PokemonTCG.io card row.

    Returns None if minimum required fields (id + name) are absent.
    Other fields fall back to None. Mirrors `scryfall._build_record`'s
    contract so the two sources produce shape-compatible records.
    """
    variant_id = card.get("id")
    name = card.get("name")
    if not isinstance(variant_id, str) or not variant_id:
        return None
    if not isinstance(name, str) or not name:
        return None

    set_obj = card.get("set") or {}
    set_id = set_obj.get("id") if isinstance(set_obj, dict) else None
    set_code: str | None
    if isinstance(set_id, str) and set_id:
        set_code = set_id.upper()
    else:
        set_code = None

    number = card.get("number")
    if not isinstance(number, str) or not number:
        number = None

    rarity = card.get("rarity")
    if not isinstance(rarity, str) or not rarity:
        rarity = None

    # PokemonTCG.io doesn't ship a per-card permalink in the API, but
    # the canonical web URL is deterministic from the id (e.g.
    # "base1-1" → "https://pokemontcg.io/cards/base1-1"). We stash
    # that for forensics + provenance.
    source_url = f"https://pokemontcg.io/cards/{variant_id}"

    return ReferenceRecord(
        variant_id=variant_id,
        manufacturer=MANUFACTURER,
        name=name,
        set_code=set_code,
        collector_number=number,
        rarity=rarity,
        front_image_path=front_image_path,
        source_url=source_url,
        fetched_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    )


# --------------------------------------------------------------------------
# HTTP layer
# --------------------------------------------------------------------------


class _RateLimitedError(RuntimeError):
    """429 — back off and retry later."""


class _ServerError(RuntimeError):
    """5xx after retries exhausted."""


def _request_json(
    client: httpx.Client,
    url: str,
    *,
    user_agent: str,
    api_key: str | None,
    timeout_s: float = 30.0,
) -> Any:
    """One GET; map status codes to typed exceptions. Caller handles
    counters + retries.

    PokemonTCG.io doesn't 404 on empty searches (it returns a 200 with
    an empty `data` list), so there's no _NotFoundError shape here —
    callers detect "no matches" by looking at totalCount == 0.
    """
    headers: dict[str, str] = {
        "User-Agent": user_agent,
        "Accept": "application/json",
    }
    if api_key:
        headers["X-Api-Key"] = api_key
    resp = client.get(url, headers=headers, timeout=timeout_s)
    if resp.status_code == 200:
        try:
            return resp.json()
        except ValueError as e:
            raise _ServerError(f"non-JSON 200 from {url}: {e}") from e
    if resp.status_code == 429:
        raise _RateLimitedError(url)
    if 500 <= resp.status_code < 600:
        raise _ServerError(f"{resp.status_code} on {url}")
    raise _ServerError(f"unexpected {resp.status_code} on {url}")


def _download_image(
    client: httpx.Client,
    url: str,
    *,
    user_agent: str,
    timeout_s: float = 30.0,
) -> tuple[bytes, str] | None:
    """Fetch image bytes; return (bytes, content_type) or None on failure.

    The CDN doesn't require auth and isn't gated by the API rate limit.
    Failure is non-fatal at the call-site: the record is still
    persisted with front_image_path=None so consumers know the variant
    exists."""
    try:
        resp = client.get(url, headers={"User-Agent": user_agent}, timeout=timeout_s)
        if resp.status_code != 200:
            _logger.warning("image %s -> HTTP %d", url, resp.status_code)
            return None
        # PokemonTCG.io serves PNG by default. Content-Type may be
        # missing on the CDN response — default to image/png so the
        # storage layer picks the right extension.
        content_type = resp.headers.get("Content-Type", "image/png")
        return (resp.content, content_type)
    except httpx.HTTPError as e:
        _logger.warning("image fetch failed %s err=%s", url, e)
        return None


# --------------------------------------------------------------------------
# Page iteration + ingest
# --------------------------------------------------------------------------


def iter_cards_for_query(
    query: str,
    *,
    client: httpx.Client,
    base_url: str = POKEMONTCG_API_BASE_URL,
    user_agent: str = DEFAULT_USER_AGENT,
    api_key: str | None = None,
    page_size: int = DEFAULT_PAGE_SIZE,
    request_delay_s: float = DEFAULT_REQUEST_DELAY_S,
    stats: PokemonTCGIngestStats | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield each card dict matching `query`, paginating through
    /v2/cards.

    PokemonTCG.io's pagination uses page numbers, not cursors:
    each response carries `page`, `pageSize`, `count` (rows in this
    page), and `totalCount` (overall match count). We continue while
    `page * pageSize < totalCount` — same termination semantics as
    Scryfall's `has_more` flag, just expressed differently.
    """
    if stats is None:
        stats = PokemonTCGIngestStats()

    page = 1
    while True:
        url = (
            f"{base_url}/v2/cards"
            f"?q={quote(query)}&page={page}&pageSize={page_size}"
        )
        try:
            payload = _request_json(
                client, url, user_agent=user_agent, api_key=api_key
            )
        except _RateLimitedError:
            stats.rate_limited += 1
            raise
        except (_ServerError, httpx.HTTPError) as e:
            stats.network_errors += 1
            raise RuntimeError(f"page fetch failed for {url}: {e}") from e

        stats.pages_fetched += 1
        cards = payload.get("data") or []
        # An empty page on page=1 means the search returned nothing.
        if page == 1 and not cards:
            stats.not_found += 1
            return

        for card in cards:
            stats.cards_seen += 1
            yield card

        total_count = int(payload.get("totalCount") or 0)
        page_size_used = int(payload.get("pageSize") or page_size)
        # page_size_used can differ from our request if the server
        # capped us; honor whichever was actually used to compute
        # the next-page condition.
        seen_so_far = page * page_size_used
        if seen_so_far >= total_count:
            return
        # Defensive: if the server returned fewer items than pageSize
        # AND seen_so_far < totalCount, totalCount is stale; bail
        # rather than loop forever.
        if len(cards) < page_size_used:
            return

        page += 1
        if request_delay_s > 0:
            time.sleep(request_delay_s)


def ingest_card(
    card: dict[str, Any],
    *,
    client: httpx.Client,
    store: "ReferenceRecordStore",
    user_agent: str = DEFAULT_USER_AGENT,
    request_delay_s: float = DEFAULT_REQUEST_DELAY_S,
    stats: PokemonTCGIngestStats | None = None,
) -> ReferenceRecord | None:
    """Persist one card row + its image. Returns the record on success,
    None if skipped (already present) or unparseable.

    Side effects on `stats` if provided. Image-download failure is
    non-fatal: the record is still written with front_image_path=None.
    """
    if stats is None:
        stats = PokemonTCGIngestStats()

    variant_id = card.get("id")
    if not isinstance(variant_id, str) or not variant_id:
        stats.parse_errors += 1
        return None

    if store.has_record(MANUFACTURER, variant_id):
        stats.skipped_already_present += 1
        return None

    image_url = _extract_image_url(card)
    front_image_path: str | None = None
    if image_url:
        downloaded = _download_image(client, image_url, user_agent=user_agent)
        if downloaded is None:
            stats.image_failures += 1
        else:
            data, content_type = downloaded
            front_image_path = store.write_image(
                MANUFACTURER, variant_id, "front", content_type, data
            )
        if request_delay_s > 0:
            time.sleep(request_delay_s)
    else:
        stats.image_failures += 1

    record = _build_record(card, front_image_path)
    if record is None:
        stats.parse_errors += 1
        return None

    store.write_record(record)
    stats.successful += 1
    return record


def ingest_query(
    query: str,
    *,
    store: "ReferenceRecordStore",
    base_url: str = POKEMONTCG_API_BASE_URL,
    user_agent: str = DEFAULT_USER_AGENT,
    api_key: str | None = None,
    page_size: int = DEFAULT_PAGE_SIZE,
    request_delay_s: float = DEFAULT_REQUEST_DELAY_S,
    max_cards: int | None = None,
    stats: PokemonTCGIngestStats | None = None,
) -> PokemonTCGIngestStats:
    """Top-level driver: iterate pages for `query` and ingest each card.

    Args:
        query: PokemonTCG.io Lucene-syntax search (e.g. "set.id:base1",
            "rarity:rare").
        store: where to write records + images.
        base_url: override for tests against a fixture server.
        user_agent: identifier sent with every request.
        api_key: optional X-Api-Key for higher rate limits. Falls back
            to env PTCG_API_KEY when None.
        page_size: PokemonTCG.io accepts up to 250.
        request_delay_s: politeness delay; defaults to 100 ms.
        max_cards: optional ceiling, for smoke runs.
        stats: pre-allocated counters; created fresh if not provided.

    Returns the (possibly mutated) `stats` for logging.
    """
    if stats is None:
        stats = PokemonTCGIngestStats()
    if api_key is None:
        api_key = os.environ.get("PTCG_API_KEY") or None

    with httpx.Client() as client:
        for card in iter_cards_for_query(
            query,
            client=client,
            base_url=base_url,
            user_agent=user_agent,
            api_key=api_key,
            page_size=page_size,
            request_delay_s=request_delay_s,
            stats=stats,
        ):
            ingest_card(
                card,
                client=client,
                store=store,
                user_agent=user_agent,
                request_delay_s=request_delay_s,
                stats=stats,
            )
            if max_cards is not None and stats.successful >= max_cards:
                break

    return stats


__all__ = [
    "MANUFACTURER",
    "POKEMONTCG_API_BASE_URL",
    "DEFAULT_USER_AGENT",
    "DEFAULT_REQUEST_DELAY_S",
    "DEFAULT_PAGE_SIZE",
    "PokemonTCGIngestStats",
    "iter_cards_for_query",
    "ingest_card",
    "ingest_query",
]
