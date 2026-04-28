"""Inline test runner for the PSA pop-report scraper.

Mirrors `_run_detection_tests.py` style. No pytest dependency, no moto —
S3 store is exercised against a mock client built with unittest.mock.
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
import traceback
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import httpx  # required at runtime — fail loudly if missing

from data.ingestion import (
    LocalScrapedRecordStore,
    PSAPopScraper,
    S3ScrapedRecordStore,
    ScrapedRecord,
    parse_cert_html,
)
from data.ingestion.psa_pop_scraper import ScrapeAborted


ran, passed, failed = 0, 0, 0


def case(name, fn):
    global ran, passed, failed
    ran += 1
    try:
        fn()
        print(f"PASS {name}")
        passed += 1
    except AssertionError as e:
        print(f"FAIL {name}: AssertionError: {e}")
        failed += 1
    except Exception as e:
        print(f"FAIL {name}: {type(e).__name__}: {e}")
        traceback.print_exc()
        failed += 1


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


def cert_html(grade="9", name="Pikachu Illustrator"):
    return f"""
    <html><body>
      <h1 class="item-title">{name}</h1>
      <div class="cert-grade">PSA {grade}</div>
      <img class="card-image front" src="/img/front.jpg" />
      <img class="card-image back" src="/img/back.jpg" />
      <dl>
        <dt>Year</dt><dd>1999</dd>
        <dt>Set</dt><dd>Pokemon Game</dd>
        <dt>Card Number</dt><dd>4</dd>
      </dl>
    </body></html>
    """


def broken_html():
    return "<html><body>nothing here</body></html>"


class MockResponse:
    def __init__(self, status_code=200, text="", content=b"", headers=None):
        self.status_code = status_code
        self.text = text
        self.content = content or text.encode("utf-8")
        self.headers = headers or {}


def ok(text=""):
    return MockResponse(200, text)


def img(content=b"img-bytes"):
    return MockResponse(200, "", content, {"Content-Type": "image/jpeg"})


class ScriptedClient:
    def __init__(self, responses):
        self._responses = list(responses)
        self.requested_urls = []

    def get(self, url, *args, **kwargs):
        self.requested_urls.append(url)
        if not self._responses:
            raise AssertionError(f"unexpected GET: {url}")
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def make_scraper(store, responses, rate=1000.0, jitter=(0.0, 0.0)):
    client = ScriptedClient(responses)
    s = PSAPopScraper(
        store=store,
        rate_limit_per_sec=rate,
        request_jitter_seconds=jitter,
        max_retries_per_cert=2,
        http_client=client,
    )
    return s, client


# --------------------------------------------------------------------------
# parse_cert_html
# --------------------------------------------------------------------------


def p1():
    rec = parse_cert_html(cert_html(), 12345, "https://www.psacard.com/cert/12345")
    assert rec is not None
    assert rec.cert_id == 12345
    assert rec.grade == 9.0
    assert rec.card_name == "Pikachu Illustrator"
    assert rec.set_name == "Pokemon Game"
    assert rec.year == 1999
    assert rec.card_number == "4"
    assert rec.front_image_path == "https://www.psacard.com/img/front.jpg"
    assert rec.back_image_path == "https://www.psacard.com/img/back.jpg"
case("parse_extracts_all_fields", p1)


def p2():
    rec = parse_cert_html(cert_html(grade="9.5"), 1, "https://x")
    assert rec is not None and rec.grade == 9.5
case("parse_half_grades", p2)


def p3():
    html = cert_html().replace('<div class="cert-grade">PSA 9</div>', "")
    rec = parse_cert_html(html, 1, "https://x")
    assert rec is None
case("parse_missing_grade_returns_none", p3)


def p4():
    html = cert_html().replace(
        '<h1 class="item-title">Pikachu Illustrator</h1>', ""
    )
    rec = parse_cert_html(html, 1, "https://x")
    assert rec is None
case("parse_missing_name_returns_none", p4)


def p5():
    rec = parse_cert_html(broken_html(), 1, "https://x")
    assert rec is None
case("parse_garbage_returns_none", p5)


def p6():
    html = """
    <h1>Charizard Holo</h1>
    <span class="grade-num">10</span>
    """
    rec = parse_cert_html(html, 2, "https://x")
    assert rec is not None and rec.grade == 10.0 and rec.card_name == "Charizard Holo"
case("parse_grade_num_alternative", p6)


# --------------------------------------------------------------------------
# Local store happy / error paths
# --------------------------------------------------------------------------


def s_happy():
    with tempfile.TemporaryDirectory() as td:
        store = LocalScrapedRecordStore(Path(td))
        responses = [ok(cert_html()), img(b"front"), img(b"back")]
        scraper, _ = make_scraper(store, responses)
        rec = scraper.scrape_cert(12345)
        assert rec is not None and rec.grade == 9.0
        records = (Path(td) / "scraped.jsonl").read_text().strip().splitlines()
        assert len(records) == 1
        on_disk = json.loads(records[0])
        assert on_disk["cert_id"] == 12345
        front_bytes = Path(rec.front_image_path).read_bytes()
        back_bytes = Path(rec.back_image_path).read_bytes()
        assert front_bytes == b"front"
        assert back_bytes == b"back"
case("local_store_happy_path", s_happy)


def s_404():
    with tempfile.TemporaryDirectory() as td:
        store = LocalScrapedRecordStore(Path(td))
        scraper, _ = make_scraper(store, [MockResponse(404)])
        stats = scraper.scrape_range(1, 1)
        assert stats.not_found == 1
        assert stats.successful == 0
case("404_increments_not_found", s_404)


def s_5xx_retry():
    with tempfile.TemporaryDirectory() as td:
        store = LocalScrapedRecordStore(Path(td))
        responses = [MockResponse(500), MockResponse(500), MockResponse(503)]
        scraper, client = make_scraper(store, responses)
        stats = scraper.scrape_range(7, 7)
        assert stats.network_errors == 1
        assert len(client.requested_urls) == 3
case("5xx_retried_then_network_error", s_5xx_retry)


def s_429():
    with tempfile.TemporaryDirectory() as td:
        store = LocalScrapedRecordStore(Path(td))
        responses = [
            MockResponse(429, headers={"Retry-After": "0"}),
            ok(cert_html()),
            img(b"f"),
            img(b"b"),
        ]
        scraper, _ = make_scraper(store, responses)
        stats = scraper.scrape_range(11, 11)
        assert stats.rate_limited == 1
        assert stats.successful == 1
case("429_retry_after_respected", s_429)


def s_403():
    with tempfile.TemporaryDirectory() as td:
        store = LocalScrapedRecordStore(Path(td))
        scraper, _ = make_scraper(store, [MockResponse(403)])
        try:
            scraper.scrape_range(1, 5)
        except ScrapeAborted:
            return
        raise AssertionError("expected ScrapeAborted")
case("403_aborts_run", s_403)


def s_timeout():
    with tempfile.TemporaryDirectory() as td:
        store = LocalScrapedRecordStore(Path(td))
        responses = [
            httpx.ReadTimeout("slow"),
            httpx.ReadTimeout("slow"),
            httpx.ReadTimeout("slow"),
        ]
        scraper, _ = make_scraper(store, responses)
        stats = scraper.scrape_range(1, 1)
        assert stats.network_errors == 1
case("timeout_counted_as_network_error", s_timeout)


def s_parse_error():
    with tempfile.TemporaryDirectory() as td:
        store = LocalScrapedRecordStore(Path(td))
        scraper, _ = make_scraper(store, [ok(broken_html())])
        stats = scraper.scrape_range(99, 99)
        assert stats.parse_errors == 1
        assert stats.successful == 0
        path = Path(td) / "scraped.jsonl"
        assert (not path.exists()) or path.read_text().strip() == ""
case("parse_error_no_record_persisted", s_parse_error)


def s_image_failure():
    with tempfile.TemporaryDirectory() as td:
        store = LocalScrapedRecordStore(Path(td))
        responses = [
            ok(cert_html()),
            MockResponse(500),  # front fails
            img(b"back-bytes"),
        ]
        scraper, _ = make_scraper(store, responses)
        rec = scraper.scrape_cert(12345)
        assert rec is not None
        assert rec.front_image_path is None
        assert rec.back_image_path is not None
case("image_failure_keeps_record_with_none_path", s_image_failure)


def s_resume():
    with tempfile.TemporaryDirectory() as td:
        store = LocalScrapedRecordStore(Path(td))
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
        responses = [
            ok(cert_html()), img(b"f"), img(b"b"),  # 99
            ok(cert_html()), img(b"f"), img(b"b"),  # 101
        ]
        scraper, client = make_scraper(store, responses)
        stats = scraper.scrape_range(99, 101)
        assert stats.successful == 2
        records = (Path(td) / "scraped.jsonl").read_text().strip().splitlines()
        assert len(records) == 3
        # Cert URLs hit: 99 and 101, not 100
        cert_hits = [u for u in client.requested_urls if "/99" in u or "/100" in u or "/101" in u]
        assert any(u.endswith("/99") for u in cert_hits)
        assert any(u.endswith("/101") for u in cert_hits)
        assert not any(u.endswith("/100") for u in cert_hits)
case("resume_skips_existing_cert", s_resume)


def s_rate_limit_timing():
    with tempfile.TemporaryDirectory() as td:
        store = LocalScrapedRecordStore(Path(td))
        plain_html = """
        <html><body><h1>Card</h1><span class="grade-num">8</span></body></html>
        """
        responses = [ok(plain_html) for _ in range(5)]
        scraper, _ = make_scraper(store, responses, rate=5.0, jitter=(0.0, 0.0))
        t0 = time.monotonic()
        stats = scraper.scrape_range(1, 5)
        elapsed = time.monotonic() - t0
        assert stats.successful == 5
        # 5 reqs at 5 req/sec ≥ ~0.7s of inter-request gaps
        assert elapsed >= 0.7, f"too fast: {elapsed:.3f}s"
case("rate_limit_5_at_5rps_takes_real_time", s_rate_limit_timing)


# --------------------------------------------------------------------------
# S3 store via mock client (no moto required)
# --------------------------------------------------------------------------


def _make_in_memory_s3():
    """Tiny in-memory stand-in for boto3 s3 client.

    Implements just enough surface for our store (get_object, put_object).
    Errors mimic boto3's ClientError shape so the store's NoSuchKey
    branch fires when the key is missing."""
    storage = {}

    class _NoSuchKey(Exception):
        def __init__(self):
            super().__init__("NoSuchKey")
            self.response = {"Error": {"Code": "NoSuchKey"}}

    class _Body:
        def __init__(self, data):
            self._data = data

        def read(self):
            return self._data

    client = MagicMock()

    def get_object(Bucket, Key):
        if (Bucket, Key) not in storage:
            raise _NoSuchKey()
        data, ct = storage[(Bucket, Key)]
        return {"Body": _Body(data), "ContentType": ct}

    def put_object(Bucket, Key, Body, ContentType=None, **kw):
        storage[(Bucket, Key)] = (
            Body if isinstance(Body, (bytes, bytearray)) else Body.encode("utf-8"),
            ContentType or "application/octet-stream",
        )
        return {}

    client.get_object.side_effect = get_object
    client.put_object.side_effect = put_object
    return client, storage


def s3_write_and_resume():
    client, storage = _make_in_memory_s3()
    store = S3ScrapedRecordStore(bucket="b", prefix="psa", s3_client=client)
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
    # New store instance over the same backing — must see the record
    store2 = S3ScrapedRecordStore(bucket="b", prefix="psa", s3_client=client)
    assert 42 in store2.iter_existing_cert_ids()
    # write_image returns an s3:// URI
    uri = store.write_image(42, "front", "image/jpeg", b"abc")
    assert uri == "s3://b/psa/images/42/front.jpg"
    assert storage[("b", "psa/images/42/front.jpg")][0] == b"abc"
case("s3_store_write_record_and_resume", s3_write_and_resume)


def s3_full_scrape():
    client, storage = _make_in_memory_s3()
    store = S3ScrapedRecordStore(bucket="b", prefix="psa", s3_client=client)
    responses = [ok(cert_html()), img(b"front"), img(b"back")]
    scraper, _ = make_scraper(store, responses)
    rec = scraper.scrape_cert(12345)
    assert rec is not None
    assert rec.front_image_path == "s3://b/psa/images/12345/front.jpg"
    assert storage[("b", "psa/images/12345/front.jpg")][0] == b"front"
    assert storage[("b", "psa/images/12345/back.jpg")][0] == b"back"
case("s3_store_full_scrape_persists_images", s3_full_scrape)


print()
print(f"{passed}/{ran} passed, {failed} failed")
sys.exit(0 if failed == 0 else 1)
