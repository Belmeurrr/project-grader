"""PSA pop-report scraper.

Why this exists:
    Per the architecture plan, "labels are the bottleneck." PSA's public
    cert-lookup pages collectively cover ~80M graded cards with public
    images and an authoritative final grade. Each successful scrape is a
    `(front_image, back_image, grade)` triple — the seed corpus for our
    corners / surface / edges grading models. Even at 1 req/sec we
    accumulate enough labeled data to bootstrap the grading pipeline
    inside a single calendar quarter.

Why polite-by-default:
    PSA's pop-report is public but not officially API'd. We treat the rate
    limit as a contract: 1 req/sec default, jittered, honoring 429s and
    bailing on 403. Two reasons — (1) we don't want our IP rotated out of
    the public web, and (2) the data is going to underwrite a paid
    grading product so we can't be cavalier about how it was acquired.

Why the parser is `re`+`html.parser` rather than BeautifulSoup:
    No new deps (per the project constraint). The parsing problem is
    small enough — extract one grade, one card name, two image URLs, a
    handful of metadata fields — that a careful HTMLParser subclass plus
    a few regexes is cleaner than pulling in lxml/BeautifulSoup. The
    parser is exposed as a pure function `parse_cert_html(html, cert_id,
    base_url)` so it's trivially unit-testable on synthetic HTML.

HTML assumptions (v1 — verify against real PSA HTML before running):
    PSA's cert page (https://www.psacard.com/cert/<n>) renders:
        <h1 class="...item-title...">{card_name}</h1>
        <span class="grade-num">9</span>      OR
        <div class="cert-grade ...">PSA 9</div>
        <img class="...front..." src="...">
        <img class="...back..." src="...">
        <dl> with <dt>Year</dt><dd>1999</dd>
                  <dt>Set</dt><dd>Pokemon Game</dd>
                  <dt>Card Number</dt><dd>4</dd>
        </dl>
    These selectors are a best-effort guess. If PSA's real HTML differs,
    update `parse_cert_html` and the regex patterns below — the parser
    is intentionally local to one function so the diff stays tiny.

Resume protocol:
    `scrape_range(a, b)` calls `store.has_record(cert_id)` per cert; certs
    already in the JSONL are skipped. A run can be killed mid-range and
    resumed cleanly.

Error taxonomy (mapped to ScrapeStats):
    network_errors  — connect/read timeouts, 5xx after exhausted retries
    not_found       — 404 (cert never existed or was retired)
    parse_errors    — page returned 200 but we couldn't extract a grade
    rate_limited    — server returned 429 at least once during this cert
    successful      — record persisted (with or without images)

A 403 is treated as "you're blocked" and aborts the run by raising
`ScrapeAborted`. Continuing into a hard-block is worse than failing fast.
"""

from __future__ import annotations

import io
import logging
import random
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from html.parser import HTMLParser
from typing import TYPE_CHECKING, Any
from urllib.parse import urljoin, urlparse

import httpx

if TYPE_CHECKING:
    from data.ingestion.storage import ScrapedRecordStore


# --- logging --- structlog if available, plain logging fallback otherwise.
# We always wrap in a thin shim so call sites can use kwargs uniformly even
# when structlog isn't installed (stdlib Logger raises on unknown kwargs).
try:
    import structlog

    _structlog_logger = structlog.get_logger("psa_pop_scraper")
    _USING_STRUCTLOG = True
except ImportError:  # pragma: no cover — exercised in envs w/o structlog
    _structlog_logger = None
    _USING_STRUCTLOG = False

_stdlib_logger = logging.getLogger("psa_pop_scraper")


class _LoggerShim:
    """Forward kwargs to structlog; flatten to a string for stdlib."""

    @staticmethod
    def _format(event: str, kwargs: dict) -> str:
        if not kwargs:
            return event
        kv = " ".join(f"{k}={v!r}" for k, v in kwargs.items())
        return f"{event} {kv}"

    def info(self, event: str, **kw) -> None:
        if _USING_STRUCTLOG:
            _structlog_logger.info(event, **kw)
        else:
            _stdlib_logger.info(self._format(event, kw))

    def warning(self, event: str, **kw) -> None:
        if _USING_STRUCTLOG:
            _structlog_logger.warning(event, **kw)
        else:
            _stdlib_logger.warning(self._format(event, kw))

    def error(self, event: str, **kw) -> None:
        if _USING_STRUCTLOG:
            _structlog_logger.error(event, **kw)
        else:
            _stdlib_logger.error(self._format(event, kw))


_logger = _LoggerShim()


PSA_CERT_BASE_URL = "https://www.psacard.com/cert/"


class ScrapeAborted(RuntimeError):
    """Raised when the scraper hits a fatal condition (e.g. 403 block)."""


@dataclass(frozen=True)
class ScrapedRecord:
    """Single successfully-parsed cert. Persisted as one JSONL line."""

    cert_id: int
    grade: float
    card_name: str
    set_name: str
    year: int | None
    card_number: str | None
    front_image_path: str | None
    back_image_path: str | None
    source_url: str
    scraped_at: str  # ISO8601 UTC, includes "Z"


@dataclass
class ScrapeStats:
    total_attempted: int = 0
    successful: int = 0
    not_found: int = 0
    parse_errors: int = 0
    network_errors: int = 0
    rate_limited: int = 0


# --------------------------------------------------------------------------
# HTML parsing
# --------------------------------------------------------------------------


_GRADE_PATTERNS: tuple[re.Pattern[str], ...] = (
    # "PSA 9" or "PSA 9.5" inside a cert-grade div
    re.compile(
        r"<[^>]*class=\"[^\"]*(?:cert-grade|grade)[^\"]*\"[^>]*>\s*"
        r"(?:PSA\s+)?(\d{1,2}(?:\.5)?)\s*<",
        re.IGNORECASE | re.DOTALL,
    ),
    # span/div with class="grade-num">9</span>
    re.compile(
        r"<(?:span|div)[^>]*class=\"[^\"]*grade-num[^\"]*\"[^>]*>\s*"
        r"(\d{1,2}(?:\.5)?)\s*</",
        re.IGNORECASE,
    ),
    # data-grade attribute fallback
    re.compile(r"data-grade=\"(\d{1,2}(?:\.5)?)\"", re.IGNORECASE),
)

_CARD_NAME_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"<h1[^>]*class=\"[^\"]*(?:item-title|cert-title|card-title)[^\"]*\"[^>]*>"
        r"(.*?)</h1>",
        re.IGNORECASE | re.DOTALL,
    ),
    # bare h1
    re.compile(r"<h1[^>]*>(.*?)</h1>", re.IGNORECASE | re.DOTALL),
)

_IMG_FRONT_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"<img[^>]*class=\"[^\"]*\bfront\b[^\"]*\"[^>]*src=\"([^\"]+)\"",
        re.IGNORECASE,
    ),
    re.compile(
        r"<img[^>]*src=\"([^\"]+)\"[^>]*class=\"[^\"]*\bfront\b[^\"]*\"",
        re.IGNORECASE,
    ),
)

_IMG_BACK_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"<img[^>]*class=\"[^\"]*\bback\b[^\"]*\"[^>]*src=\"([^\"]+)\"",
        re.IGNORECASE,
    ),
    re.compile(
        r"<img[^>]*src=\"([^\"]+)\"[^>]*class=\"[^\"]*\bback\b[^\"]*\"",
        re.IGNORECASE,
    ),
)

_DL_PAIR_PATTERN = re.compile(
    r"<dt[^>]*>(.*?)</dt>\s*<dd[^>]*>(.*?)</dd>",
    re.IGNORECASE | re.DOTALL,
)

_TAG_STRIP = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")


def _strip_tags(s: str) -> str:
    """Remove tags + collapse whitespace + decode common entities."""
    if not s:
        return ""
    no_tags = _TAG_STRIP.sub(" ", s)
    no_tags = (
        no_tags.replace("&nbsp;", " ")
        .replace("&amp;", "&")
        .replace("&quot;", '"')
        .replace("&#39;", "'")
        .replace("&apos;", "'")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
    )
    return _WS.sub(" ", no_tags).strip()


def _first_match(patterns: tuple[re.Pattern[str], ...], html: str) -> str | None:
    for p in patterns:
        m = p.search(html)
        if m:
            return m.group(1)
    return None


def _extract_metadata(html: str) -> dict[str, str]:
    """Pull out (label -> value) from any <dl><dt/><dd/></dl> blocks.

    PSA's metadata renders as a definition list; this is also a common
    pattern for card spec tables on partner sites, so the helper stays
    useful even if PSA's markup shifts."""
    out: dict[str, str] = {}
    for dt, dd in _DL_PAIR_PATTERN.findall(html):
        label = _strip_tags(dt).rstrip(":").lower()
        value = _strip_tags(dd)
        if label and value and label not in out:
            out[label] = value
    return out


def _coerce_year(raw: str | None) -> int | None:
    if not raw:
        return None
    m = re.search(r"\b(1[89]\d{2}|20\d{2}|21\d{2})\b", raw)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def _absolute_url(src: str | None, base_url: str) -> str | None:
    if not src:
        return None
    src = src.strip()
    if not src:
        return None
    if src.startswith("data:"):
        return None
    return urljoin(base_url, src)


def parse_cert_html(html: str, cert_id: int, base_url: str) -> ScrapedRecord | None:
    """Parse a PSA cert page into a ScrapedRecord.

    Pure function — does no I/O. Returns `None` if the minimum required
    fields (grade + card name) are absent. The caller is expected to bump
    `parse_errors` on `None`.

    Image *URLs* are returned via `front_image_path` / `back_image_path`
    in the *interim* return value; the scraper then downloads them and
    overwrites those fields with storage paths before persisting. (We
    keep this funnel narrow rather than returning a richer
    intermediate type — this is a v1 scraper, optimize for diff size.)
    """
    grade_str = _first_match(_GRADE_PATTERNS, html)
    if grade_str is None:
        return None
    try:
        grade = float(grade_str)
    except ValueError:
        return None
    if not (1.0 <= grade <= 10.0):
        return None

    card_name_raw = _first_match(_CARD_NAME_PATTERNS, html)
    card_name = _strip_tags(card_name_raw or "")
    if not card_name:
        return None

    front_url = _absolute_url(_first_match(_IMG_FRONT_PATTERNS, html), base_url)
    back_url = _absolute_url(_first_match(_IMG_BACK_PATTERNS, html), base_url)

    meta = _extract_metadata(html)
    set_name = meta.get("set") or meta.get("set name") or meta.get("brand") or ""
    year = _coerce_year(meta.get("year") or meta.get("year/season"))
    card_number = meta.get("card number") or meta.get("card #") or meta.get("number")

    return ScrapedRecord(
        cert_id=cert_id,
        grade=grade,
        card_name=card_name,
        set_name=set_name,
        year=year,
        card_number=card_number,
        front_image_path=front_url,
        back_image_path=back_url,
        source_url=base_url,
        scraped_at=_now_iso(),
    )


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --------------------------------------------------------------------------
# Rate limiting
# --------------------------------------------------------------------------


class _RateLimiter:
    """Minimum-interval limiter with optional jitter.

    A token bucket would be overkill for ~1 req/sec; the simpler
    "sleep until next slot" loop is exactly equivalent at this rate
    and is trivially testable by patching `time.monotonic` and `time.sleep`.
    Thread-safe so a future parallelization isn't immediately blocked.
    """

    def __init__(
        self,
        rate_per_sec: float,
        jitter_range: tuple[float, float] = (0.0, 0.0),
    ) -> None:
        if rate_per_sec <= 0:
            raise ValueError("rate_per_sec must be positive")
        self._min_interval = 1.0 / rate_per_sec
        self._jitter_range = jitter_range
        self._last_call: float | None = None
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            if self._last_call is None:
                self._last_call = now
                return
            elapsed = now - self._last_call
            target = self._min_interval
            jitter_lo, jitter_hi = self._jitter_range
            if jitter_hi > jitter_lo:
                target += random.uniform(jitter_lo, jitter_hi)
            if elapsed < target:
                time.sleep(target - elapsed)
            self._last_call = time.monotonic()

    def force_backoff(self, seconds: float) -> None:
        """Bump the next-call time by `seconds` (e.g. after a 429)."""
        seconds = max(0.0, seconds)
        with self._lock:
            self._last_call = time.monotonic() + seconds - self._min_interval


# --------------------------------------------------------------------------
# Scraper
# --------------------------------------------------------------------------


@dataclass
class PSAPopScraper:
    """Walks PSA cert IDs, parses cert pages, persists to a backing store.

    Design:
        - I/O is via `httpx.Client` (synchronous; the bottleneck is the
          remote-server-friendly rate limit, not local concurrency).
        - The HTTP client is constructed lazily on first request so a
          test can swap it via the `http_client` field before scraping.
        - Image download failure is non-fatal: we persist the record
          with that side's `_image_path = None`. A later pass can
          backfill missing images.

    Constructor args:
        store                       backend (LocalScrapedRecordStore / S3...)
        rate_limit_per_sec          target req/sec; default 1.0 (polite)
        user_agent                  identifying UA string
        timeout_seconds             per-request HTTP timeout
        max_retries_per_cert        retries on 5xx / timeout (default 2)
        request_jitter_seconds      (lo, hi) added to the minimum
                                    inter-request interval
    """

    store: "ScrapedRecordStore"
    rate_limit_per_sec: float = 1.0
    user_agent: str = "ProjectGrader-Bot/0.1 (research; bots@projectgrader.ai)"
    timeout_seconds: float = 15.0
    max_retries_per_cert: int = 2
    request_jitter_seconds: tuple[float, float] = (0.0, 0.5)
    base_url: str = PSA_CERT_BASE_URL
    http_client: httpx.Client | None = None
    _limiter: _RateLimiter = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._limiter = _RateLimiter(
            self.rate_limit_per_sec, self.request_jitter_seconds
        )

    # -- HTTP plumbing -----------------------------------------------------

    def _client(self) -> httpx.Client:
        if self.http_client is None:
            self.http_client = httpx.Client(
                timeout=self.timeout_seconds,
                headers={"User-Agent": self.user_agent},
                follow_redirects=True,
            )
        return self.http_client

    def _request_with_retries(
        self,
        url: str,
        cert_id: int,
        stats: ScrapeStats,
    ) -> tuple[httpx.Response | None, str]:
        """Fetch `url` honoring rate limits and the retry policy.

        Returns (response, outcome) where outcome is one of:
            "ok"            response is 2xx
            "not_found"     response is 404
            "rate_limited"  exhausted retries on 429
            "network"       exhausted retries on 5xx / timeout
            "blocked"       caller should abort (403)
        """
        client = self._client()
        attempt = 0
        saw_429 = False
        last_response: httpx.Response | None = None
        while True:
            self._limiter.wait()
            try:
                resp = client.get(url)
            except httpx.TimeoutException:
                if attempt >= self.max_retries_per_cert:
                    return None, "network"
                attempt += 1
                self._limiter.force_backoff(_backoff_seconds(attempt))
                continue
            except httpx.HTTPError:
                if attempt >= self.max_retries_per_cert:
                    return None, "network"
                attempt += 1
                self._limiter.force_backoff(_backoff_seconds(attempt))
                continue

            last_response = resp
            status = resp.status_code

            if 200 <= status < 300:
                if saw_429:
                    stats.rate_limited += 1
                return resp, "ok"
            if status == 404:
                return resp, "not_found"
            if status == 403:
                return resp, "blocked"
            if status == 429:
                saw_429 = True
                retry_after = _parse_retry_after(resp.headers.get("Retry-After"))
                self._limiter.force_backoff(retry_after)
                if attempt >= self.max_retries_per_cert:
                    stats.rate_limited += 1
                    return resp, "rate_limited"
                attempt += 1
                continue
            if 500 <= status < 600:
                if attempt >= self.max_retries_per_cert:
                    return resp, "network"
                attempt += 1
                self._limiter.force_backoff(_backoff_seconds(attempt))
                continue

            # Anything else (3xx that wasn't followed, weird 4xx) — don't
            # retry, treat as a parse-context error and let the caller see
            # the response so it can decide.
            return resp, "ok"

    # -- per-cert scrape ---------------------------------------------------

    def _cert_url(self, cert_id: int) -> str:
        return f"{self.base_url.rstrip('/')}/{cert_id}"

    def scrape_cert(self, cert_id: int) -> ScrapedRecord | None:
        """Scrape a single cert, persist it, return the record (or None).

        Side effects:
            - Writes images to `self.store` if successful.
            - Writes a record to `self.store` if successful.
        Returns:
            The record if successful, else None. The caller is responsible
            for bumping the right counters via `_log_outcome` / scrape_range.
        """
        # We don't take stats here — _scrape_one_with_stats is the
        # internal entry point. Public method wraps with a fresh stats so
        # callers can use it ad-hoc.
        stats = ScrapeStats()
        record = self._scrape_one_with_stats(cert_id, stats)
        return record

    def _scrape_one_with_stats(
        self,
        cert_id: int,
        stats: ScrapeStats,
    ) -> ScrapedRecord | None:
        url = self._cert_url(cert_id)
        stats.total_attempted += 1
        resp, outcome = self._request_with_retries(url, cert_id, stats)

        if outcome == "blocked":
            _logger.error("psa_scraper.blocked", cert_id=cert_id, url=url)
            raise ScrapeAborted(f"403 from PSA on cert {cert_id}; aborting run")
        if outcome == "not_found":
            stats.not_found += 1
            _logger.info("psa_scraper.not_found", cert_id=cert_id)
            return None
        if outcome == "rate_limited":
            _logger.warning("psa_scraper.rate_limited", cert_id=cert_id)
            return None
        if outcome == "network":
            stats.network_errors += 1
            _logger.warning("psa_scraper.network_error", cert_id=cert_id)
            return None
        if resp is None:
            stats.network_errors += 1
            return None

        try:
            html = resp.text
        except Exception:  # decoding error
            stats.parse_errors += 1
            _logger.warning("psa_scraper.decode_error", cert_id=cert_id)
            return None

        parsed = parse_cert_html(html, cert_id=cert_id, base_url=url)
        if parsed is None:
            stats.parse_errors += 1
            _logger.warning("psa_scraper.parse_error", cert_id=cert_id)
            return None

        front_path = self._maybe_download_image(
            cert_id, "front", parsed.front_image_path
        )
        back_path = self._maybe_download_image(
            cert_id, "back", parsed.back_image_path
        )

        record = ScrapedRecord(
            cert_id=parsed.cert_id,
            grade=parsed.grade,
            card_name=parsed.card_name,
            set_name=parsed.set_name,
            year=parsed.year,
            card_number=parsed.card_number,
            front_image_path=front_path,
            back_image_path=back_path,
            source_url=parsed.source_url,
            scraped_at=parsed.scraped_at,
        )
        self.store.write_record(record)
        stats.successful += 1
        _logger.info(
            "psa_scraper.success",
            cert_id=cert_id,
            grade=record.grade,
            has_front=front_path is not None,
            has_back=back_path is not None,
        )
        return record

    def _maybe_download_image(
        self,
        cert_id: int,
        kind: str,
        url: str | None,
    ) -> str | None:
        """Download `url` via the rate-limited HTTP client and persist.

        Failures are deliberately swallowed — a missing image is not a
        reason to drop the record. The grade label still has training
        value with one side. We just record the path as None and move on.
        """
        if not url:
            return None
        try:
            self._limiter.wait()
            resp = self._client().get(url)
            if resp.status_code != 200:
                _logger.warning(
                    "psa_scraper.image_http_error",
                    cert_id=cert_id,
                    kind=kind,
                    status=resp.status_code,
                )
                return None
            content_type = resp.headers.get("Content-Type", "")
            if not content_type:
                # best-effort sniff from extension
                ext = urlparse(url).path.rsplit(".", 1)[-1].lower()
                content_type = {
                    "jpg": "image/jpeg",
                    "jpeg": "image/jpeg",
                    "png": "image/png",
                    "webp": "image/webp",
                }.get(ext, "")
            return self.store.write_image(cert_id, kind, content_type, resp.content)
        except httpx.HTTPError as e:
            _logger.warning(
                "psa_scraper.image_network_error",
                cert_id=cert_id,
                kind=kind,
                error=str(e),
            )
            return None
        except Exception as e:  # storage error
            _logger.warning(
                "psa_scraper.image_storage_error",
                cert_id=cert_id,
                kind=kind,
                error=str(e),
            )
            return None

    # -- range walk --------------------------------------------------------

    def scrape_range(
        self,
        start_cert: int,
        end_cert: int,
        *,
        stop_after: int | None = None,
    ) -> ScrapeStats:
        """Walk cert IDs in [start_cert, end_cert].

        `stop_after` caps total *attempts* (skipped resume-hits don't
        count) — useful for short test runs.

        Resumes by skipping IDs already in `store.iter_existing_cert_ids()`.
        """
        if start_cert > end_cert:
            raise ValueError("start_cert must be <= end_cert")
        stats = ScrapeStats()
        # Capture resume state once at the start; per-cert has_record
        # call gives us the current-process incremental view too.
        existing = set(self.store.iter_existing_cert_ids())
        _logger.info(
            "psa_scraper.range_start",
            start=start_cert,
            end=end_cert,
            existing=len(existing),
            rate=self.rate_limit_per_sec,
        )
        attempted = 0
        for cert_id in range(start_cert, end_cert + 1):
            if cert_id in existing or self.store.has_record(cert_id):
                continue
            try:
                self._scrape_one_with_stats(cert_id, stats)
            except ScrapeAborted:
                _logger.error(
                    "psa_scraper.range_aborted",
                    cert_id=cert_id,
                    stats=stats.__dict__,
                )
                raise
            attempted += 1
            if attempted % 100 == 0:
                _logger.info("psa_scraper.progress", attempted=attempted, **stats.__dict__)
            if stop_after is not None and attempted >= stop_after:
                break
        _logger.info("psa_scraper.range_done", **stats.__dict__)
        return stats


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _backoff_seconds(attempt: int) -> float:
    """Exponential backoff with a small floor.

    attempt=1 -> ~1s, attempt=2 -> ~2s, attempt=3 -> ~4s. Capped so a
    single bad cert can't stall the run beyond a minute."""
    return min(60.0, 2 ** (attempt - 1))


def _parse_retry_after(value: str | None) -> float:
    """Parse Retry-After: HTTP-date OR delta-seconds.

    We tolerate either. Default to 5s if the header is missing or
    unparseable — better than spinning back into a 429 immediately."""
    if not value:
        return 5.0
    value = value.strip()
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    # HTTP-date — parse with email.utils per RFC 7231
    try:
        from email.utils import parsedate_to_datetime

        dt = parsedate_to_datetime(value)
        if dt is None:
            return 5.0
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        delta = (dt - datetime.now(timezone.utc)).total_seconds()
        return max(0.0, delta)
    except Exception:
        return 5.0


__all__ = [
    "PSA_CERT_BASE_URL",
    "PSAPopScraper",
    "ScrapeAborted",
    "ScrapeStats",
    "ScrapedRecord",
    "parse_cert_html",
]
