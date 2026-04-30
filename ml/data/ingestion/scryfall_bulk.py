"""Scryfall bulk-data ingest — corpus-scale alternative to /cards/search.

Why this exists:
    The paginated `/cards/search` path in `scryfall.py` is capped at
    ~1750 results (Scryfall's documented soft limit on search). That's
    fine for set-by-set queries like "set:lea" but unworkable for the
    full ~85k-printing default catalog. Scryfall's `/bulk-data`
    endpoint exposes a daily-refreshed JSON dump of every printing in
    one HTTP call — that's the right tool for whole-catalog ingest.

How it works:
    1. GET /bulk-data → list of bulk-data entries. We pick the one
       whose `type` matches `bulk_type` (default: "default_cards" =
       every printing once, English preferred — same per-printing
       semantics as `unique=prints` on the search endpoint).
    2. GET the entry's `download_uri` (served from data.scryfall.io,
       same CDN-style host family as the image hosts). Streamed
       chunked-write to disk so the download itself stays at constant
       memory regardless of file size.
    3. Parse the cached file with `json.load` and iterate cards. Each
       card flows through the same `ingest_card()` machinery as the
       paginated path: per-card idempotency by (manufacturer,
       variant_id), per-card image fetch against cards.scryfall.io,
       per-card record write to the `LocalReferenceStore`.

What stays the same vs `scryfall.ingest_query`:
    - `LocalReferenceStore` is the persistence layer. Re-ingest of a
      card already on disk is a no-op (cheap idempotency check).
    - Image fetch is per-card against cards.scryfall.io with the same
      politeness delay (`request_delay_s`).
    - `ScryfallIngestStats` is the same dataclass — the field
      `pages_fetched` is unused in the bulk path (stays 0); other
      counters mean exactly what they mean for the paginated path.
    - Error taxonomy is shared. Image download failures are non-fatal:
      the record is still persisted with `front_image_path=None`.

What's different:
    - There's no `query` argument: bulk pulls everything, then the
      per-card filter is just "is this a kind we already have?". A
      future optional `predicate: Callable[[dict], bool]` can be added
      if we want server-side-style filtering.
    - One large download instead of N paginated fetches. Cached on
      disk so re-running this driver in the same day doesn't re-download
      (caller controls cache lifetime via `cache_dir` / `keep_cache`).

Memory trade-off:
    `json.load` reads the full file into memory before iteration. For
    `default_cards` (~300 MB on disk) the parsed Python objects peak
    around ~1-1.5 GB transient. Adequate on any dev machine with
    16+ GB. If we later need true streaming (e.g. to ingest from a
    constrained worker), drop in `ijson.items(f, "item")` — the
    `iter_cards_from_bulk` surface is designed so the swap is local.

Why not stdlib-only:
    Same as `scryfall.py` — bulk needs `httpx` for the streaming
    download. The module follows the same lazy-import contract: it
    doesn't load until the CLI or a caller asks for `ingest_bulk`.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator

import httpx

if TYPE_CHECKING:
    from data.ingestion.references_storage import ReferenceRecordStore

from data.ingestion.scryfall import (
    DEFAULT_REQUEST_DELAY_S,
    DEFAULT_USER_AGENT,
    SCRYFALL_API_BASE_URL,
    ScryfallIngestStats,
    _NotFoundError,
    _RateLimitedError,
    _ServerError,
    _request_json,
    ingest_card,
)


_logger = logging.getLogger("scryfall_bulk_ingest")


# Default bulk type. "default_cards" is every printing once with English
# preferred — the per-printing semantics the embedding-anomaly detector
# wants. Other valid values: "all_cards" (every printing in every
# language, ~1.6 GB), "oracle_cards" (one per Oracle ID, collapses
# printings — wrong for our use case), "unique_artwork" (one per art).
DEFAULT_BULK_TYPE: str = "default_cards"


# --------------------------------------------------------------------------
# Bulk metadata helpers
# --------------------------------------------------------------------------


def _select_bulk_entry(entries: list[dict[str, Any]], bulk_type: str) -> dict[str, Any] | None:
    """Pick the bulk-data entry whose `type` field matches `bulk_type`.

    Pure helper. Returns the matching dict or None if no match. Scryfall
    returns entries in an unspecified order, so do not assume position.
    """
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if entry.get("type") == bulk_type:
            return entry
    return None


# --------------------------------------------------------------------------
# Streamed download
# --------------------------------------------------------------------------


def download_bulk_file(
    client: httpx.Client,
    url: str,
    dest: Path,
    *,
    user_agent: str = DEFAULT_USER_AGENT,
    chunk_size: int = 1024 * 1024,
    timeout_s: float = 600.0,
) -> int:
    """Stream-download `url` to `dest`. Returns total bytes written.

    Chunked write keeps memory bounded regardless of file size; the
    300+ MB bulk file would blow up a naive `resp.content` load. We
    write to a `.part` sidecar and atomically rename on success so a
    crash mid-download leaves an obvious partial file rather than a
    valid-looking truncated cache.
    """
    headers = {"User-Agent": user_agent}
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    bytes_written = 0
    with client.stream("GET", url, headers=headers, timeout=timeout_s) as resp:
        if resp.status_code != 200:
            raise _ServerError(f"bulk download {url} -> HTTP {resp.status_code}")
        with tmp.open("wb") as fh:
            for chunk in resp.iter_bytes(chunk_size=chunk_size):
                fh.write(chunk)
                bytes_written += len(chunk)
            fh.flush()
            os.fsync(fh.fileno())
    os.replace(tmp, dest)
    return bytes_written


# --------------------------------------------------------------------------
# Card iteration
# --------------------------------------------------------------------------


def iter_cards_from_bulk(path: Path) -> Iterator[dict[str, Any]]:
    """Yield each card dict from a downloaded Scryfall bulk JSON file.

    Uses `json.load` — see module docstring for the memory trade-off
    and the planned ijson swap-in. Skips entries that aren't dicts so
    a malformed line doesn't abort the run; that matches the fail-soft
    contract the rest of the ingestion layer follows.
    """
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError(
            f"expected list at top level of {path}, got {type(data).__name__}"
        )
    for card in data:
        if isinstance(card, dict):
            yield card


# --------------------------------------------------------------------------
# Top-level driver
# --------------------------------------------------------------------------


def ingest_bulk(
    *,
    store: "ReferenceRecordStore",
    base_url: str = SCRYFALL_API_BASE_URL,
    bulk_type: str = DEFAULT_BULK_TYPE,
    cache_dir: Path | str | None = None,
    keep_cache: bool = False,
    user_agent: str = DEFAULT_USER_AGENT,
    request_delay_s: float = DEFAULT_REQUEST_DELAY_S,
    max_cards: int | None = None,
    stats: ScryfallIngestStats | None = None,
) -> ScryfallIngestStats:
    """Top-level driver: fetch bulk metadata, download bulk file, ingest each card.

    Args:
        store: where to write records + images.
        base_url: override for tests against a fixture server.
        bulk_type: which Scryfall bulk type to fetch (default
            "default_cards"). See module docstring for the others.
        cache_dir: directory for the downloaded bulk file. Defaults to
            a per-process subdir under the system temp dir.
        keep_cache: if True, leaves the downloaded bulk file in place
            after ingest (useful for inspection / re-runs). If False
            (default), the file is removed when ingest completes —
            success or failure.
        user_agent: identifier sent with every request.
        request_delay_s: politeness delay applied to per-card image
            fetches; defaults to 100 ms.
        max_cards: optional ceiling on successful records, for smoke
            runs. Iteration stops cleanly when reached.
        stats: pre-allocated counters; created fresh if not provided.

    Returns the (possibly mutated) `stats` for logging.

    Raises:
        RuntimeError: bulk metadata fetch failed, the requested
            `bulk_type` wasn't present in the metadata, or the bulk
            file download failed.
    """
    if stats is None:
        stats = ScryfallIngestStats()

    if cache_dir is None:
        cache_root = Path(tempfile.gettempdir()) / "scryfall_bulk"
    else:
        cache_root = Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    bulk_path = cache_root / f"{bulk_type}.json"

    with httpx.Client() as client:
        # Step 1: fetch /bulk-data metadata. Small JSON, single GET.
        metadata_url = f"{base_url}/bulk-data"
        try:
            payload = _request_json(client, metadata_url, user_agent=user_agent)
        except _NotFoundError as e:
            stats.network_errors += 1
            raise RuntimeError(f"bulk metadata 404 at {metadata_url}") from e
        except _RateLimitedError as e:
            stats.rate_limited += 1
            raise RuntimeError(f"bulk metadata 429 at {metadata_url}") from e
        except (_ServerError, httpx.HTTPError) as e:
            stats.network_errors += 1
            raise RuntimeError(f"bulk metadata fetch failed: {e}") from e

        entries = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(entries, list):
            raise RuntimeError(
                f"unexpected /bulk-data response shape: missing 'data' list"
            )
        entry = _select_bulk_entry(entries, bulk_type)
        if entry is None:
            available = sorted(
                e.get("type") for e in entries if isinstance(e, dict) and e.get("type")
            )
            raise RuntimeError(
                f"bulk type {bulk_type!r} not in /bulk-data response; available: {available}"
            )
        download_uri = entry.get("download_uri")
        if not isinstance(download_uri, str) or not download_uri:
            raise RuntimeError(
                f"bulk entry {bulk_type!r} has no usable download_uri"
            )

        # Step 2: stream-download the bulk file to a local cache.
        try:
            size = download_bulk_file(
                client, download_uri, bulk_path, user_agent=user_agent
            )
        except (_ServerError, httpx.HTTPError) as e:
            stats.network_errors += 1
            raise RuntimeError(f"bulk file download failed: {e}") from e
        _logger.info("bulk %s downloaded: %d bytes -> %s", bulk_type, size, bulk_path)

        # Step 3: iterate cards from disk, dispatch to ingest_card per card.
        try:
            for card in iter_cards_from_bulk(bulk_path):
                stats.cards_seen += 1
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
        finally:
            if not keep_cache:
                try:
                    bulk_path.unlink(missing_ok=True)
                except OSError:
                    _logger.warning(
                        "failed to remove bulk cache file %s", bulk_path
                    )

    return stats


__all__ = [
    "DEFAULT_BULK_TYPE",
    "download_bulk_file",
    "iter_cards_from_bulk",
    "ingest_bulk",
]
