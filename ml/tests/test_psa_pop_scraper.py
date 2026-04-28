"""PSA pop-report scraper tests.

All HTTP is mocked — we never touch real PSA. Tests cover:
    - parse_cert_html on synthetic HTML
    - 404, 5xx-with-retry, 429 with Retry-After, 403 abort
    - resume support
    - rate limiting timing
    - image download failure tolerance
    - parse failure recording
    - LocalScrapedRecordStore (tmp_path) and S3ScrapedRecordStore (moto, optional)
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import pytest

# Ensure ml/ root is on sys.path when this file runs from CI directly.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.ingestion import (
    LocalScrapedRecordStore,
    PSAPopScraper,
    S3ScrapedRecordStore,
    ScrapedRecord,
    parse_cert_html,
)
from data.ingestion.psa_pop_scraper import ScrapeAborted


# --------------------------------------------------------------------------
# HTML fixtures
# --------------------------------------------------------------------------


def _cert_html(
    *,
    grade: str = "9",
    card_name: str = "Pikachu Illustrator",
    front_url: str = "/images/cert/12345/front.jpg",
    back_url: str = "/images/cert/12345/back.jpg",
    year: str = "1999",
    set_name: str = "Pokemon Game",
    card_number: str = "4",
) -> str:
    return f"""
    <html>
      <head><title>PSA Cert</title></head>
      <body>
        <h1 class="item-title">{card_name}</h1>
        <div class="cert-grade large">PSA {grade}</div>
        <img class="card-image front" src="{front_url}" />
        <img class="card-image back" src="{back_url}" />
        <dl class="cert-meta">
          <dt>Year</dt><dd>{year}</dd>
          <dt>Set</dt><dd>{set_name}</dd>
          <dt>Card Number</dt><dd>{card_number}</dd>
        </dl>
      </body>
    </html>
    """


def _broken_html() -> str:
    return "<html><body><p>nothing useful here</p></body></html>"


# --------------------------------------------------------------------------
# Mock httpx response helpers
# --------------------------------------------------------------------------


class _MockResponse:
    """Minimal httpx.Response stand-in for our usage surface."""

    def __init__(
        self,
        status_code: int = 200,
        text: str = "",
        content: bytes = b"",
        headers: dict[str, str] | None = None,
    ) -> None:
        self.status_code = status_code
        self.text = text
        self.content = content or text.encode("utf-8")
        self.headers = headers or {}


def _ok(text: str = "", **kw) -> _MockResponse:
    return _MockResponse(status_code=200, text=text, **kw)


def _img(content: bytes = b"\xff\xd8imgbytes") -> _MockResponse:
    return _MockResponse(
        status_code=200, content=content, headers={"Content-Type": "image/jpeg"}
    )


# --------------------------------------------------------------------------
# parse_cert_html
# --------------------------------------------------------------------------


class TestParseCertHtml:
    def test_extracts_all_fields(self):
        html = _cert_html()
        rec = parse_cert_html(html, cert_id=12345, base_url="https://www.psacard.com/cert/12345")
        assert rec is not None
        assert rec.cert_id == 12345
        assert rec.grade == 9.0
        assert rec.card_name == "Pikachu Illustrator"
        assert rec.set_name == "Pokemon Game"
        assert rec.year == 1999
        assert rec.card_number == "4"
        assert rec.front_image_path == "https://www.psacard.com/images/cert/12345/front.jpg"
        assert rec.back_image_path == "https://www.psacard.com/images/cert/12345/back.jpg"
        assert rec.source_url == "https://www.psacard.com/cert/12345"
        assert "T" in rec.scraped_at

    def test_half_grades(self):
        html = _cert_html(grade="9.5")
        rec = parse_cert_html(html, cert_id=1, base_url="https://x")
        assert rec is not None
        assert rec.grade == 9.5

    def test_returns_none_on_missing_grade(self):
        html = _cert_html().replace('<div class="cert-grade large">PSA 9</div>', "")
        rec = parse_cert_html(html, cert_id=1, base_url="https://x")
        assert rec is None

    def test_returns_none_on_missing_name(self):
        html = _cert_html().replace(
            '<h1 class="item-title">Pikachu Illustrator</h1>', ""
        )
        rec = parse_cert_html(html, cert_id=1, base_url="https://x")
        assert rec is None

    def test_returns_none_on_garbage(self):
        rec = parse_cert_html(_broken_html(), cert_id=1, base_url="https://x")
        assert rec is None

    def test_grade_num_span_alternative(self):
        html = """
        <h1>Charizard Holo</h1>
        <span class="grade-num">10</span>
        """
        rec = parse_cert_html(html, cert_id=2, base_url="https://x")
        assert rec is not None
        assert rec.grade == 10.0
        assert rec.card_name == "Charizard Holo"

    def test_year_extracted_from_messy_value(self):
        html = _cert_html(year="1999 Wizards")
        rec = parse_cert_html(html, cert_id=1, base_url="https://x")
        assert rec is not None
        assert rec.year == 1999

    def test_handles_relative_image_urls(self):
        html = _cert_html(front_url="/img/a.jpg")
        rec = parse_cert_html(html, cert_id=1, base_url="https://www.psacard.com/cert/1")
        assert rec is not None
        assert rec.front_image_path == "https://www.psacard.com/img/a.jpg"


# --------------------------------------------------------------------------
# Helpers for end-to-end testing
# --------------------------------------------------------------------------


class _ScriptedClient:
    """Drop-in for httpx.Client with a scripted sequence of responses.

    Calls are matched in order. Image fetches and cert fetches share the
    same queue — tests must enqueue them in the order the scraper calls
    them (cert page first, then front image, then back image)."""

    def __init__(self, responses: list[_MockResponse | Exception]) -> None:
        self._responses = list(responses)
        self.requested_urls: list[str] = []

    def get(self, url: str, *args, **kwargs):
        self.requested_urls.append(url)
        if not self._responses:
            raise AssertionError(f"unexpected extra GET: {url}")
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _make_scraper(
    store,
    responses: list[Any],
    *,
    rate: float = 1000.0,
    jitter: tuple[float, float] = (0.0, 0.0),
) -> tuple[PSAPopScraper, _ScriptedClient]:
    client = _ScriptedClient(responses)
    scraper = PSAPopScraper(
        store=store,
        rate_limit_per_sec=rate,
        request_jitter_seconds=jitter,
        max_retries_per_cert=2,
        http_client=client,  # type: ignore[arg-type]
    )
    return scraper, client


# --------------------------------------------------------------------------
# End-to-end / store tests
# --------------------------------------------------------------------------


class TestScraperHappyPath:
    def test_successful_scrape_persists_record_and_images(self, tmp_path: Path):
        store = LocalScrapedRecordStore(tmp_path)
        responses = [_ok(_cert_html()), _img(b"front-bytes"), _img(b"back-bytes")]
        scraper, client = _make_scraper(store, responses)

        rec = scraper.scrape_cert(12345)

        assert rec is not None
        assert rec.cert_id == 12345
        assert rec.grade == 9.0
        assert rec.front_image_path is not None
        assert rec.back_image_path is not None
        # Records JSONL written
        records = (tmp_path / "scraped.jsonl").read_text().strip().splitlines()
        assert len(records) == 1
        on_disk = json.loads(records[0])
        assert on_disk["cert_id"] == 12345
        assert on_disk["grade"] == 9.0
        # Images written
        front = Path(rec.front_image_path)
        back = Path(rec.back_image_path)
        assert front.read_bytes() == b"front-bytes"
        assert back.read_bytes() == b"back-bytes"


class TestScraperErrorPaths:
    def test_404_returns_none_and_increments_not_found(self, tmp_path: Path):
        store = LocalScrapedRecordStore(tmp_path)
        scraper, _ = _make_scraper(store, [_MockResponse(status_code=404)])
        stats = scraper.scrape_range(1, 1)
        assert stats.not_found == 1
        assert stats.successful == 0
        assert not (tmp_path / "scraped.jsonl").exists() or (
            (tmp_path / "scraped.jsonl").read_text().strip() == ""
        )

    def test_5xx_retries_then_counted_as_network_error(self, tmp_path: Path):
        store = LocalScrapedRecordStore(tmp_path)
        # max_retries_per_cert=2 means 1 + 2 = 3 attempts total.
        responses = [
            _MockResponse(status_code=500),
            _MockResponse(status_code=500),
            _MockResponse(status_code=503),
        ]
        scraper, client = _make_scraper(store, responses)
        stats = scraper.scrape_range(7, 7)
        assert stats.network_errors == 1
        assert stats.successful == 0
        assert len(client.requested_urls) == 3

    def test_429_with_retry_after_respected(self, tmp_path: Path):
        store = LocalScrapedRecordStore(tmp_path)
        # 429 then success; we want stats.rate_limited bumped
        responses = [
            _MockResponse(status_code=429, headers={"Retry-After": "0"}),
            _ok(_cert_html()),
            _img(b"f"),
            _img(b"b"),
        ]
        scraper, _ = _make_scraper(store, responses)
        stats = scraper.scrape_range(11, 11)
        assert stats.rate_limited == 1
        assert stats.successful == 1

    def test_403_aborts_run(self, tmp_path: Path):
        store = LocalScrapedRecordStore(tmp_path)
        responses = [_MockResponse(status_code=403)]
        scraper, _ = _make_scraper(store, responses)
        with pytest.raises(ScrapeAborted):
            scraper.scrape_range(1, 5)

    def test_timeout_retries_then_network_error(self, tmp_path: Path):
        store = LocalScrapedRecordStore(tmp_path)
        responses = [
            httpx.ReadTimeout("slow"),
            httpx.ReadTimeout("slow"),
            httpx.ReadTimeout("slow"),
        ]
        scraper, _ = _make_scraper(store, responses)
        stats = scraper.scrape_range(1, 1)
        assert stats.network_errors == 1

    def test_parse_failure_bumps_parse_errors_and_skips_record(self, tmp_path: Path):
        store = LocalScrapedRecordStore(tmp_path)
        responses = [_ok(_broken_html())]
        scraper, _ = _make_scraper(store, responses)
        stats = scraper.scrape_range(99, 99)
        assert stats.parse_errors == 1
        assert stats.successful == 0
        assert not (tmp_path / "scraped.jsonl").exists() or (
            (tmp_path / "scraped.jsonl").read_text().strip() == ""
        )

    def test_image_download_failure_keeps_record(self, tmp_path: Path):
        store = LocalScrapedRecordStore(tmp_path)
        responses = [
            _ok(_cert_html()),
            _MockResponse(status_code=500),  # front fails
            _img(b"back-bytes"),
        ]
        scraper, _ = _make_scraper(store, responses)
        rec = scraper.scrape_cert(12345)
        assert rec is not None
        assert rec.front_image_path is None
        assert rec.back_image_path is not None


class TestScraperResume:
    def test_resume_skips_existing_cert(self, tmp_path: Path):
        store = LocalScrapedRecordStore(tmp_path)
        # Pre-populate cert 100
        seed = ScrapedRecord(
            cert_id=100,
            grade=8.0,
            card_name="Seed",
            set_name="x",
            year=None,
            card_number=None,
            front_image_path=None,
            back_image_path=None,
            source_url="https://x",
            scraped_at="2024-01-01T00:00:00Z",
        )
        store.write_record(seed)
        # Range 99-101 should only fetch 99 and 101 (2 cert pages, 2 images each)
        responses = [
            _ok(_cert_html()), _img(b"f"), _img(b"b"),  # 99
            _ok(_cert_html()), _img(b"f"), _img(b"b"),  # 101
        ]
        scraper, client = _make_scraper(store, responses)
        stats = scraper.scrape_range(99, 101)
        assert stats.successful == 2
        # Three records total now (100 was pre-existing)
        on_disk = (tmp_path / "scraped.jsonl").read_text().strip().splitlines()
        assert len(on_disk) == 3
        # cert URLs hit: 99 and 101 only
        cert_urls = [u for u in client.requested_urls if u.endswith(("/99", "/101", "/100"))]
        assert any(u.endswith("/99") for u in cert_urls)
        assert any(u.endswith("/101") for u in cert_urls)
        assert not any(u.endswith("/100") for u in cert_urls)


class TestRateLimit:
    def test_rate_limited_run_takes_at_least_min_interval(self, tmp_path: Path):
        store = LocalScrapedRecordStore(tmp_path)
        # 5 successful certs at 5 req/sec (no images to keep request count
        # predictable: HTML has no image tags).
        plain_html = """
        <html><body>
          <h1>Card</h1>
          <span class="grade-num">8</span>
        </body></html>
        """
        responses = [_ok(plain_html) for _ in range(5)]
        scraper, _ = _make_scraper(store, responses, rate=5.0, jitter=(0.0, 0.0))
        t0 = time.monotonic()
        stats = scraper.scrape_range(1, 5)
        elapsed = time.monotonic() - t0
        assert stats.successful == 5
        # 5 reqs at 5 req/sec is ~0.8s minimum (4 inter-request waits of 0.2s).
        assert elapsed >= 0.7, f"too fast: {elapsed:.3f}s"


# --------------------------------------------------------------------------
# S3 store via moto (optional)
# --------------------------------------------------------------------------


def _has_moto() -> bool:
    try:
        import moto  # noqa: F401

        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _has_moto(), reason="moto not installed")
class TestS3StoreWithMoto:
    def test_write_and_read_record(self):
        from moto import mock_aws  # type: ignore
        import boto3

        with mock_aws():
            s3 = boto3.client("s3", region_name="us-east-1")
            s3.create_bucket(Bucket="test-bucket")
            store = S3ScrapedRecordStore(
                bucket="test-bucket", prefix="psa", s3_client=s3
            )
            rec = ScrapedRecord(
                cert_id=42,
                grade=9.5,
                card_name="X",
                set_name="Y",
                year=2020,
                card_number="1",
                front_image_path=None,
                back_image_path=None,
                source_url="https://x/cert/42",
                scraped_at="2024-01-01T00:00:00Z",
            )
            store.write_record(rec)
            assert store.has_record(42)
            assert 42 in store.iter_existing_cert_ids()

            # Image
            uri = store.write_image(42, "front", "image/jpeg", b"abc")
            assert uri == "s3://test-bucket/psa/images/42/front.jpg"
            obj = s3.get_object(Bucket="test-bucket", Key="psa/images/42/front.jpg")
            assert obj["Body"].read() == b"abc"

    def test_scraper_with_s3_store(self):
        from moto import mock_aws  # type: ignore
        import boto3

        with mock_aws():
            s3 = boto3.client("s3", region_name="us-east-1")
            s3.create_bucket(Bucket="test-bucket")
            store = S3ScrapedRecordStore(
                bucket="test-bucket", prefix="psa", s3_client=s3
            )
            responses = [_ok(_cert_html()), _img(b"front"), _img(b"back")]
            scraper, _ = _make_scraper(store, responses)
            rec = scraper.scrape_cert(12345)
            assert rec is not None
            assert rec.front_image_path == "s3://test-bucket/psa/images/12345/front.jpg"
            front = s3.get_object(
                Bucket="test-bucket", Key="psa/images/12345/front.jpg"
            )
            assert front["Body"].read() == b"front"
