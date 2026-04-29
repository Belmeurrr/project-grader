"""PSA Public API client — sanctioned replacement for the deprecated scraper.

Why this exists:
    PSA's public cert pages are gated by Cloudflare's "Just a moment..." JS
    challenge (verified 2026-04-28: `cf-mitigated: challenge` header on
    every response to non-browser clients). The httpx-based pop-report
    scraper at `psa_pop_scraper.py` cannot pass it, and bypass tooling is
    ToS-gray for a paid product. PSA's official Public API at
    `api.psacard.com/publicapi` is the sanctioned source for the same
    data.

    Endpoints verified against `https://api.psacard.com/publicapi/swagger.json`
    on 2026-04-28:

        GET /cert/GetByCertNumber/{certNumber}     → PublicCertificationModel
        GET /cert/GetImagesByCertNumber/{certNumber} → opaque object (image URLs)
        GET /pop/GetPSASpecPopulation/{specID}      → PSASpecPopulationModel

    Auth:  `Authorization: bearer <token>` (apiKey scheme, not OAuth2)
    Token: self-serve at psacard.com/publicapi (free)
    Limit: 100 calls/day on the free tier; paid tier via webcert@collectors.com

Why we reuse the scraper's storage layer:
    `ScrapedRecord` and the `*ScrapedRecordStore` Protocol are
    backend-agnostic — they don't care whether records came from HTML
    parsing or a JSON API. The shape (cert_id, grade, card_name, set_name,
    year, card_number, front/back image paths, source_url, scraped_at)
    fits both. Reusing the storage layer means resume/idempotency
    semantics are preserved unchanged.

Per-cert API budget:
    Each cert costs 2 API calls — one metadata (`GetByCertNumber`), one
    images (`GetImagesByCertNumber`). On the free 100-calls/day tier that
    yields ~50 cert triples/day. The `DailyBudget` class is the budget
    gate; it raises `BudgetExhausted` when a call would exceed the daily
    cap so the caller can checkpoint and resume the next day.

Image-URL field shape (verified against a real call on 2026-04-29):
    `GetImagesByCertNumber` returns a top-level JSON ARRAY (the Swagger
    spec types it as opaque `{"type":"object"}`, but reality is an array
    of dicts). Each element is shaped:
        {"IsFrontImage": <bool>, "ImageURL": "<https cloudfront jpg>"}
    Two elements per cert (front + back). Image URLs are hot-linkable
    CloudFront jpgs and don't require auth or count against the API
    budget. The `_extract_image_urls` parser handles this exact shape
    first; the defensive tree walk remains as a fallback in case PSA
    changes the schema.

Error taxonomy (mapped to IngestStats, mirrors ScrapeStats):
    successful      — record persisted (with or without images)
    not_found       — 404 (cert never existed or was retired)
    auth_errors     — 401 / 403 (token revoked, expired, or wrong scope)
    rate_limited    — 429 from server, OR DailyBudget exhausted client-side
    parse_errors    — 200 returned but we couldn't extract a grade
    network_errors  — connect/read timeouts, 5xx after exhausted retries

A 401 or 403 is treated as "your credentials are bad" and aborts the run
by raising `IngestAborted` — same fail-fast posture as the scraper's 403
handling.
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

import httpx

if TYPE_CHECKING:
    from data.ingestion.storage import ScrapedRecordStore

# ScrapedRecord lives in the storage module so both ingest sources (the
# deprecated httpx scraper and this API client) can share the dataclass
# without cross-importing each other.
from data.ingestion.storage import ScrapedRecord


class IngestAborted(RuntimeError):
    """Raised when the API ingest hits a fatal condition (auth error,
    revoked token) that requires human attention rather than retry.

    The daily script catches this and exits non-zero WITHOUT advancing
    the state cursor — a token rotation doesn't lose its place in the
    corpus. Distinct from the deprecated scraper's ScrapeAborted: same
    semantics, but the two sources are entirely separate now and
    catching one shouldn't accidentally swallow the other."""


_logger = logging.getLogger("psa_public_api")


# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

PSA_PUBLIC_API_BASE_URL = "https://api.psacard.com/publicapi"

# Free tier cap. Paid tier overrides this via DailyBudget(daily_cap=...).
DEFAULT_DAILY_CALL_CAP = 100

# Image content-types we accept. Anything else gets stored as "bin" by the
# storage layer; we don't try to be clever about format conversion here.
_IMAGE_EXTENSIONS: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp")


# --------------------------------------------------------------------------
# Stats + budget
# --------------------------------------------------------------------------


@dataclass
class IngestStats:
    """Counters mirroring ScrapeStats for the API ingestion track.

    Field names match ScrapeStats where they overlap so dashboards built
    for the scraper transfer over. The API track adds two counters that
    don't exist in the HTML scraper:

      non_gradeable_skipped — cert exists and was authenticated but has
        no numeric grade. Two known shapes:
          (a) PSA/DNA dual certs — `IsDualCert: true` or `IsPSADNA: true`
              (verified on cert 90554074 — Bonds autograph)
          (b) Authentication-only cards — `IsDualCert: false`,
              `IsPSADNA: false`, but CardGrade like "N0: AUTHENTIC"
              (verified on cert 80000002 — 2022 Paolo Banchero auto/mem 1/1)
        These are not errors; they're out of scope for the grading data
        flywheel. The early-skip avoids burning a second budget unit on
        the images call.

      images_missing — cert exists but `GetImagesByCertNumber` returned
        an empty array. PSA only began storing cert images in Oct 2021,
        so a large fraction of older certs are legitimately imageless.
        Verified on cert 28887000 (1966 Topps Mantle).
    """

    total_attempted: int = 0
    successful: int = 0
    not_found: int = 0
    parse_errors: int = 0
    network_errors: int = 0
    rate_limited: int = 0
    auth_errors: int = 0
    non_gradeable_skipped: int = 0
    images_missing: int = 0


class BudgetExhausted(RuntimeError):
    """Raised when a call would exceed the daily API budget."""


@dataclass
class DailyBudget:
    """Client-side daily call counter.

    Keeps track of calls spent in the current UTC day. Resets on day
    rollover. Thread-safe. Doesn't try to coordinate across processes —
    if a paid tier is used in parallel, share the budget via a Redis
    counter or accept that each process gets its own slice of the cap.
    """

    daily_cap: int = DEFAULT_DAILY_CALL_CAP
    _calls_today: int = 0
    _day_key: str = ""
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def _today(self) -> str:
        return datetime.now(timezone.utc).date().isoformat()

    def consume(self, n: int = 1) -> None:
        """Reserve `n` calls. Raises BudgetExhausted if it would overflow."""
        with self._lock:
            today = self._today()
            if today != self._day_key:
                self._day_key = today
                self._calls_today = 0
            if self._calls_today + n > self.daily_cap:
                raise BudgetExhausted(
                    f"daily budget {self.daily_cap} would be exceeded "
                    f"(spent {self._calls_today}, requested {n})"
                )
            self._calls_today += n

    @property
    def remaining(self) -> int:
        with self._lock:
            today = self._today()
            if today != self._day_key:
                return self.daily_cap
            return self.daily_cap - self._calls_today


# --------------------------------------------------------------------------
# Pure parsers — no I/O, fully unit-testable on synthetic JSON
# --------------------------------------------------------------------------

# Both CardGrade and GradeDescription are descriptive strings ("MINT 9",
# "GEM MT 10", "NM-MT+ 8.5"), verified empirically against the API. They
# carry the same content. We extract the numeric grade with a single
# regex; CardGrade is checked first only because it's the canonical
# field name, not because it's numeric.
_GRADE_NUMERIC_RE = re.compile(r"\b(\d{1,2}(?:\.5)?)\b")


def _parse_grade(metadata: dict[str, Any]) -> float | None:
    """Pull a numeric grade out of a PublicCertificationModel.

    Pure function. Returns None if no grade can be parsed — caller maps
    that to `parse_errors` so the cert is not persisted with a bogus
    grade."""
    psa_cert = metadata.get("PSACert") or {}
    for field in ("CardGrade", "GradeDescription"):
        raw = psa_cert.get(field)
        if not raw:
            continue
        m = _GRADE_NUMERIC_RE.search(str(raw))
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                continue
    return None


def _parse_year(metadata: dict[str, Any]) -> int | None:
    psa_cert = metadata.get("PSACert") or {}
    raw = psa_cert.get("Year") or ""
    m = re.search(r"\b(1[89]\d{2}|20\d{2}|21\d{2})\b", str(raw))
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def _build_record(
    cert_id: int,
    metadata: dict[str, Any],
    front_path: str | None,
    back_path: str | None,
    source_url: str,
) -> ScrapedRecord | None:
    """Assemble a ScrapedRecord from a verified-shape metadata response.

    Returns None if the minimum required fields (grade + a card name)
    are absent — same contract as the scraper's `parse_cert_html`."""
    grade = _parse_grade(metadata)
    if grade is None:
        return None
    psa_cert = metadata.get("PSACert") or {}
    subject = psa_cert.get("Subject") or ""
    brand = psa_cert.get("Brand") or ""
    # "card_name" maps to subject in PSA's vocabulary; brand+year+subject
    # is what shows on the slab label.
    card_name = subject.strip() or brand.strip()
    if not card_name:
        return None
    set_name = brand.strip()
    return ScrapedRecord(
        cert_id=cert_id,
        grade=grade,
        card_name=card_name,
        set_name=set_name,
        year=_parse_year(metadata),
        card_number=(psa_cert.get("CardNumber") or None),
        front_image_path=front_path,
        back_image_path=back_path,
        source_url=source_url,
        scraped_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    )


def _is_image_url(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    parsed = urlparse(value)
    if parsed.scheme not in ("http", "https"):
        return False
    path = (parsed.path or "").lower()
    return any(path.endswith(ext) for ext in _IMAGE_EXTENSIONS)


def _extract_image_urls(images_response: Any) -> tuple[str | None, str | None]:
    """Extract (front_url, back_url) from a GetImagesByCertNumber response.

    Verified shape (2026-04-29): top-level JSON array of
        {"IsFrontImage": <bool>, "ImageURL": "<https url>"}
    elements. Two elements per cert. We handle this exact shape first.

    If PSA changes the schema (defensive fallback): walk the tree, find
    any image-URL-shaped strings, and infer side from the containing
    key path or declaration order.
    """
    front_url: str | None = None
    back_url: str | None = None

    # Fast path: verified shape from a real call.
    if isinstance(images_response, list):
        for el in images_response:
            if not isinstance(el, dict):
                continue
            url = el.get("ImageURL")
            if not _is_image_url(url):
                continue
            if el.get("IsFrontImage") is True and front_url is None:
                front_url = url
            elif el.get("IsFrontImage") is False and back_url is None:
                back_url = url
        if front_url or back_url:
            return (front_url, back_url)

    # Fallback: defensive tree walk for any future schema drift.
    found: list[tuple[str, str]] = []  # (lowercase-path, url)

    def walk(node: Any, path: str) -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                walk(v, f"{path}.{k}".lower())
        elif isinstance(node, list):
            for i, v in enumerate(node):
                walk(v, f"{path}[{i}]")
        elif _is_image_url(node):
            found.append((path, node))

    walk(images_response, "")

    if not found:
        return (None, None)

    for path, url in found:
        if "front" in path and front_url is None:
            front_url = url
        elif "back" in path and back_url is None:
            back_url = url

    if front_url is None and back_url is None:
        front_url = found[0][1]
        if len(found) >= 2:
            back_url = found[1][1]

    return (front_url, back_url)


# --------------------------------------------------------------------------
# HTTP layer
# --------------------------------------------------------------------------


class _AuthError(RuntimeError):
    """401 / 403 — abort the run."""


class _NotFoundError(RuntimeError):
    """404 — cert doesn't exist."""


class _RateLimitedError(RuntimeError):
    """429 — back off and try again later."""


class _ServerError(RuntimeError):
    """5xx after retries exhausted — record as network_error."""


def _request_json(
    client: httpx.Client,
    url: str,
    token: str,
    timeout_s: float = 30.0,
) -> Any:
    """One GET; map status codes to typed exceptions. Caller handles
    counters + retries."""
    headers = {
        "Authorization": f"bearer {token}",
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
    if resp.status_code in (401, 403):
        raise _AuthError(f"{resp.status_code} on {url}: {resp.text[:200]}")
    if resp.status_code == 429:
        raise _RateLimitedError(url)
    if 500 <= resp.status_code < 600:
        raise _ServerError(f"{resp.status_code} on {url}")
    raise _ServerError(f"unexpected {resp.status_code} on {url}")


# --------------------------------------------------------------------------
# Per-cert ingestion
# --------------------------------------------------------------------------


def ingest_cert(
    cert_id: int,
    *,
    token: str,
    client: httpx.Client,
    store: "ScrapedRecordStore",
    budget: DailyBudget,
    stats: IngestStats,
    base_url: str = PSA_PUBLIC_API_BASE_URL,
) -> ScrapedRecord | None:
    """Fetch one cert (metadata + images), persist via `store`, return record.

    Returns None if the cert was skipped (already persisted), not found,
    or unparseable. Side effects on `stats`. Raises `IngestAborted` on
    auth errors and `BudgetExhausted` if the daily cap is hit.
    """
    if store.has_record(cert_id):
        return None

    stats.total_attempted += 1
    metadata_url = f"{base_url}/cert/GetByCertNumber/{cert_id}"
    images_url = f"{base_url}/cert/GetImagesByCertNumber/{cert_id}"

    # Stage 1: metadata. Consume one budget unit up-front; we may bail
    # before the second call if the cert is a dual/autograph cert (which
    # has no numeric grade and is out of scope for the grading flywheel).
    try:
        budget.consume(1)
    except BudgetExhausted:
        stats.rate_limited += 1
        raise

    try:
        metadata = _request_json(client, metadata_url, token)
    except _NotFoundError:
        stats.not_found += 1
        return None
    except _AuthError as e:
        stats.auth_errors += 1
        raise IngestAborted(f"auth failed: {e}") from e
    except _RateLimitedError:
        stats.rate_limited += 1
        return None
    except (_ServerError, httpx.HTTPError) as e:
        stats.network_errors += 1
        _logger.warning("metadata fetch failed cert_id=%d err=%s", cert_id, e)
        return None

    psa_cert = (metadata or {}).get("PSACert") or {}
    # Skip any cert without an extractable numeric grade BEFORE consuming
    # a second budget unit on the images call. Two known cases trigger
    # this — the IsDualCert/IsPSADNA flags catch most autographs, but
    # some authentication-only cards (e.g. cert 80000002 — autograph/mem
    # 1/1) have both flags false yet still carry "N0: AUTHENTIC" as
    # CardGrade. The grade parser is the authoritative test.
    if (
        psa_cert.get("IsDualCert")
        or psa_cert.get("IsPSADNA")
        or _parse_grade(metadata) is None
    ):
        stats.non_gradeable_skipped += 1
        return None

    # Stage 2: images. Now reserve the second budget unit.
    try:
        budget.consume(1)
    except BudgetExhausted:
        stats.rate_limited += 1
        raise

    try:
        images_payload = _request_json(client, images_url, token)
    except _NotFoundError:
        images_payload = []
    except _AuthError as e:
        stats.auth_errors += 1
        raise IngestAborted(f"auth failed: {e}") from e
    except _RateLimitedError:
        stats.rate_limited += 1
        images_payload = []
    except (_ServerError, httpx.HTTPError) as e:
        _logger.warning("images fetch failed cert_id=%d err=%s", cert_id, e)
        images_payload = []

    # Empty array is the canonical "this cert has no images" response —
    # common for pre-Oct-2021 certs. Track separately from fetch failures.
    if isinstance(images_payload, list) and len(images_payload) == 0:
        stats.images_missing += 1

    front_url, back_url = _extract_image_urls(images_payload)
    if not front_url and not back_url and images_payload and not (
        isinstance(images_payload, list) and len(images_payload) == 0
    ):
        # Non-empty payload we couldn't parse — log the shape so we can
        # tighten the parser if PSA introduces a new variant.
        _logger.warning(
            "no image URLs extracted cert_id=%d shape=%s",
            cert_id,
            type(images_payload).__name__,
        )

    front_path = _download_and_store_image(client, store, cert_id, "front", front_url)
    back_path = _download_and_store_image(client, store, cert_id, "back", back_url)

    record = _build_record(
        cert_id=cert_id,
        metadata=metadata,
        front_path=front_path,
        back_path=back_path,
        source_url=metadata_url,
    )
    if record is None:
        stats.parse_errors += 1
        return None

    store.write_record(record)
    stats.successful += 1
    return record


def _download_and_store_image(
    client: httpx.Client,
    store: "ScrapedRecordStore",
    cert_id: int,
    kind: str,
    url: str | None,
) -> str | None:
    """Fetch image bytes from `url` and persist via `store`.

    Image hosts are PSA's own CDN (verified empirically on real cert pages)
    — they don't count against the API call budget and don't require auth.
    Failure is non-fatal: the record is still persisted with image_path=None.
    """
    if not url:
        return None
    try:
        resp = client.get(url, timeout=30.0)
        if resp.status_code != 200:
            return None
        content_type = resp.headers.get("Content-Type", "image/jpeg")
        return store.write_image(cert_id, kind, content_type, resp.content)
    except httpx.HTTPError as e:
        _logger.warning("image download failed cert_id=%d kind=%s err=%s", cert_id, kind, e)
        return None


# --------------------------------------------------------------------------
# Range driver
# --------------------------------------------------------------------------


def ingest_range(
    low: int,
    high: int,
    *,
    token: str | None = None,
    store: "ScrapedRecordStore",
    budget: DailyBudget | None = None,
    base_url: str = PSA_PUBLIC_API_BASE_URL,
    stats: IngestStats | None = None,
) -> IngestStats:
    """Sequentially ingest cert IDs in [low, high). Resumable.

    `token` defaults to env `PSA_PUBLIC_API_TOKEN`. Raises ValueError if
    no token is available. Stops gracefully on `BudgetExhausted` —
    the caller can re-invoke the next UTC day to continue.
    """
    if token is None:
        token = os.environ.get("PSA_PUBLIC_API_TOKEN")
    if not token:
        raise ValueError("PSA_PUBLIC_API_TOKEN not set and no token argument given")
    if budget is None:
        budget = DailyBudget()
    if stats is None:
        stats = IngestStats()

    with httpx.Client() as client:
        for cert_id in range(low, high):
            try:
                ingest_cert(
                    cert_id,
                    token=token,
                    client=client,
                    store=store,
                    budget=budget,
                    stats=stats,
                    base_url=base_url,
                )
            except BudgetExhausted:
                _logger.info(
                    "daily budget exhausted at cert_id=%d; stopping for today",
                    cert_id,
                )
                break
            except IngestAborted:
                _logger.error("aborting run on auth error at cert_id=%d", cert_id)
                raise
    return stats


__all__ = [
    "DailyBudget",
    "BudgetExhausted",
    "IngestAborted",
    "IngestStats",
    "PSA_PUBLIC_API_BASE_URL",
    "ingest_cert",
    "ingest_range",
]
