"""Scryfall API client — manufacturer reference image ingest for MTG.

Why Scryfall:
    Scryfall is the de-facto authoritative catalog for Magic: the
    Gathering — covers every printing of every card across the game's
    history (~25k printings as of 2026), exposes a free, well-rate-
    limited public API, and serves high-resolution scans on a CDN.
    Their data is the right "known authentic" reference for the
    embedding-anomaly detector and (eventually) the siamese reference
    detector.

Endpoints used:
    GET /cards/search?q=<query>&unique=prints&page=<n>
        Paginated search, 175 results per page. `unique=prints` means
        each printing of the same card is a distinct result — that's
        what we want, because different printings have different art
        / borders / finishes and the embedding-anomaly detector needs
        per-printing reference embeddings.
    Image hosts: cards.scryfall.io (CloudFront-style CDN). Don't
        require auth and don't count against the API rate limit.

Rate limiting:
    Scryfall asks for 50-100 ms between requests in their docs (~10
    req/s ceiling). We use 100 ms post-call to stay comfortably below
    that. Image downloads also count against politeness if not the
    API budget; same delay applies.

Required headers:
    User-Agent: Scryfall reject requests without one. We send a stable
        identifier with a project URL so they can reach us if our
        traffic looks problematic.
    Accept: "application/json" — Scryfall sometimes returns 415 if
        absent on certain endpoints.

Why we reuse the references storage layer:
    `ReferenceRecord` and the `*ReferenceStore` Protocol live in
    `references_storage.py` — backend-agnostic and source-agnostic.
    A future PokemonTCG.io client will write to the same store using
    the same Protocol surface, so resume/idempotency semantics carry
    over.

Error taxonomy (mapped to IngestStats):
    successful      — record persisted (with image)
    not_found       — 404 (the search query matched no cards)
    parse_errors    — 200 returned but no usable image URL extracted
    image_failures  — record persisted but image download failed
    rate_limited    — 429 (rare; we should never hit it at 10 req/s)
    network_errors  — connect/read timeouts, 5xx after retries

Design choice — paginated search vs bulk download:
    Scryfall offers a bulk-data endpoint (~172 MB JSON for oracle_cards)
    that lists every card in one shot. For corpus-scale ingest that's
    the right call. For first-session validation and most realistic
    set-by-set workflows (e.g. "ingest all of LEA"), the paginated
    /cards/search endpoint is simpler and uses no temp storage. We
    start with paginated; bulk-download is a follow-on optimization
    if/when we need all 25k printings.

Design choice — strict per-printing or per-oracle:
    `unique=prints` (per-printing) is what the detectors want.
    `unique=cards` (per-oracle, default) would collapse e.g. all
    "Lightning Bolt" printings to one row and lose the per-set art
    variation that's exactly what the embedding-anomaly detector is
    supposed to recognize.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Iterator
from urllib.parse import urlparse

import httpx

if TYPE_CHECKING:
    from data.ingestion.references_storage import ReferenceRecordStore

from data.ingestion.references_storage import ReferenceRecord


_logger = logging.getLogger("scryfall_ingest")


# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

SCRYFALL_API_BASE_URL = "https://api.scryfall.com"

# Manufacturer short-name used in storage paths and ReferenceRecord.manufacturer.
MANUFACTURER = "mtg"

# 100 ms ≈ 10 req/s, comfortably under Scryfall's stated 10 req/s ceiling.
DEFAULT_REQUEST_DELAY_S: float = 0.1

# Scryfall requires a User-Agent. Stable string with a project URL so they
# can reach us if our traffic looks problematic. Update the URL when the
# repo moves.
DEFAULT_USER_AGENT: str = (
    "project-grader/0.1 (https://github.com/Belmeurrr/project-grader)"
)

# Image-resolution preference order. We want the highest fidelity that
# loads quickly. `large` (672x936 jpg) is closest to our 750x1050
# canonical without going to png-sized files.
_IMAGE_SIZE_PREFERENCE: tuple[str, ...] = ("large", "normal", "png")


# --------------------------------------------------------------------------
# Stats
# --------------------------------------------------------------------------


@dataclass
class ScryfallIngestStats:
    """Counters for one ingest run.

    Same shape as the PSA `IngestStats`, narrowed to the categories
    Scryfall actually surfaces. The CLI driver writes this to its run
    log so we can compare across days/queries."""

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


def _extract_image_url(card: dict[str, Any]) -> str | None:
    """Return the best front-image URL on a Scryfall card record.

    Pure function. Tries top-level `image_uris` first (single-faced
    cards, ~95% of MTG); falls back to `card_faces[0].image_uris` for
    double-faced layouts ("transform", "modal_dfc", etc.). Returns
    None if neither yields a usable URL — caller maps that to
    `parse_errors`.

    Resolution preference: large > normal > png. We avoid `png` first
    because it's much larger (~3 MB vs ~150 KB) and the extra bit-
    depth doesn't help downstream embedders.
    """
    candidates: list[dict[str, Any]] = []
    top = card.get("image_uris")
    if isinstance(top, dict):
        candidates.append(top)
    faces = card.get("card_faces")
    if isinstance(faces, list) and faces:
        face = faces[0]
        if isinstance(face, dict):
            face_uris = face.get("image_uris")
            if isinstance(face_uris, dict):
                candidates.append(face_uris)

    for uris in candidates:
        for size in _IMAGE_SIZE_PREFERENCE:
            url = uris.get(size)
            if isinstance(url, str) and _is_https_url(url):
                return url
    return None


def _is_https_url(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    parsed = urlparse(value)
    return parsed.scheme == "https" and bool(parsed.netloc)


def _build_record(
    card: dict[str, Any],
    front_image_path: str | None,
) -> ReferenceRecord | None:
    """Assemble a ReferenceRecord from a Scryfall card row.

    Returns None if minimum required fields (id + name) are absent —
    same fail-soft contract as the PSA scraper's `parse_cert_html`.
    Other fields fall back to None.
    """
    variant_id = card.get("id")
    name = card.get("name")
    if not isinstance(variant_id, str) or not variant_id:
        return None
    if not isinstance(name, str) or not name:
        return None

    set_code = card.get("set")
    if isinstance(set_code, str):
        set_code = set_code.upper() or None
    else:
        set_code = None

    collector_number = card.get("collector_number")
    if not isinstance(collector_number, str) or not collector_number:
        collector_number = None

    rarity = card.get("rarity")
    if not isinstance(rarity, str) or not rarity:
        rarity = None

    source_url = card.get("scryfall_uri")
    if not isinstance(source_url, str) or not source_url:
        # Fallback: synthesize a stable URL from the variant_id.
        source_url = f"{SCRYFALL_API_BASE_URL}/cards/{variant_id}"

    return ReferenceRecord(
        variant_id=variant_id,
        manufacturer=MANUFACTURER,
        name=name,
        set_code=set_code,
        collector_number=collector_number,
        rarity=rarity,
        front_image_path=front_image_path,
        source_url=source_url,
        fetched_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    )


# --------------------------------------------------------------------------
# HTTP layer
# --------------------------------------------------------------------------


class _NotFoundError(RuntimeError):
    """404 — search returned no cards."""


class _RateLimitedError(RuntimeError):
    """429 — back off and retry later."""


class _ServerError(RuntimeError):
    """5xx after retries exhausted."""


def _request_json(
    client: httpx.Client,
    url: str,
    *,
    user_agent: str,
    timeout_s: float = 30.0,
) -> Any:
    """One GET; map status codes to typed exceptions. Caller handles
    counters + retries. Mirrors the shape of psa_public_api._request_json."""
    headers = {
        "User-Agent": user_agent,
        "Accept": "application/json",
    }
    resp = client.get(url, headers=headers, timeout=timeout_s)
    if resp.status_code == 200:
        try:
            return resp.json()
        except ValueError as e:
            raise _ServerError(f"non-JSON 200 from {url}: {e}") from e
    if resp.status_code == 404:
        raise _NotFoundError(url)
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

    Image hosts are Scryfall's CDN — no auth, no API budget. Failure is
    non-fatal at the call-site: the record is still persisted with
    front_image_path=None so consumers know the variant exists."""
    try:
        resp = client.get(url, headers={"User-Agent": user_agent}, timeout=timeout_s)
        if resp.status_code != 200:
            _logger.warning("image %s -> HTTP %d", url, resp.status_code)
            return None
        content_type = resp.headers.get("Content-Type", "image/jpeg")
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
    base_url: str = SCRYFALL_API_BASE_URL,
    user_agent: str = DEFAULT_USER_AGENT,
    request_delay_s: float = DEFAULT_REQUEST_DELAY_S,
    stats: ScryfallIngestStats | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield each card dict matching `query`, paginating through Scryfall's
    /cards/search endpoint with `unique=prints`.

    Stops cleanly on 404 (no matches) by yielding nothing. Raises on any
    other error so the caller can decide whether to retry the run.
    Stats counters are updated in-place if `stats` is provided.
    """
    if stats is None:
        stats = ScryfallIngestStats()

    from urllib.parse import quote

    # Bootstrap the first page URL. After that, we follow the `next_page`
    # field returned by Scryfall (which already contains the right cursor
    # state, so we don't reconstruct page numbers ourselves).
    next_url: str | None = (
        f"{base_url}/cards/search?q={quote(query)}&unique=prints"
    )

    while next_url:
        try:
            payload = _request_json(client, next_url, user_agent=user_agent)
        except _NotFoundError:
            stats.not_found += 1
            return
        except _RateLimitedError:
            stats.rate_limited += 1
            raise
        except (_ServerError, httpx.HTTPError) as e:
            stats.network_errors += 1
            raise RuntimeError(f"page fetch failed for {next_url}: {e}") from e

        stats.pages_fetched += 1
        cards = payload.get("data") or []
        for card in cards:
            stats.cards_seen += 1
            yield card

        if payload.get("has_more"):
            next_url = payload.get("next_page")
            if not isinstance(next_url, str) or not next_url:
                next_url = None
        else:
            next_url = None

        # Politeness delay between page fetches.
        if next_url and request_delay_s > 0:
            time.sleep(request_delay_s)


def ingest_card(
    card: dict[str, Any],
    *,
    client: httpx.Client,
    store: "ReferenceRecordStore",
    user_agent: str = DEFAULT_USER_AGENT,
    request_delay_s: float = DEFAULT_REQUEST_DELAY_S,
    stats: ScryfallIngestStats | None = None,
) -> ReferenceRecord | None:
    """Persist one card row + its image. Returns the record on success,
    None if skipped (already present) or unparseable.

    Side effects on `stats` if provided. Image-download failure is non-
    fatal: the record is still written with front_image_path=None.
    """
    if stats is None:
        stats = ScryfallIngestStats()

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
    base_url: str = SCRYFALL_API_BASE_URL,
    user_agent: str = DEFAULT_USER_AGENT,
    request_delay_s: float = DEFAULT_REQUEST_DELAY_S,
    max_cards: int | None = None,
    stats: ScryfallIngestStats | None = None,
) -> ScryfallIngestStats:
    """Top-level driver: iterate pages for `query` and ingest each card.

    Args:
        query: Scryfall search syntax (e.g. "set:lea", "set:m21 r:rare").
        store: where to write records + images.
        base_url: override for tests against a fixture server.
        user_agent: identifier sent with every request.
        request_delay_s: politeness delay; defaults to 100 ms.
        max_cards: optional ceiling, for smoke runs.
        stats: pre-allocated counters; created fresh if not provided.

    Returns the (possibly mutated) `stats` for logging.
    """
    if stats is None:
        stats = ScryfallIngestStats()

    with httpx.Client() as client:
        for card in iter_cards_for_query(
            query,
            client=client,
            base_url=base_url,
            user_agent=user_agent,
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
    "SCRYFALL_API_BASE_URL",
    "DEFAULT_USER_AGENT",
    "DEFAULT_REQUEST_DELAY_S",
    "ScryfallIngestStats",
    "iter_cards_for_query",
    "ingest_card",
    "ingest_query",
]
