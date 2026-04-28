# Data Ingestion: PSA Pop-Report Scraper

## What this is

A polite, resumable scraper for PSA's public cert-lookup pages
(`https://www.psacard.com/cert/<cert_id>`). Each successful scrape produces a
`(front_image, back_image, final_grade)` tuple plus card metadata (player /
character name, set, year, card number).

PSA's pop-report covers ~80M graded cards. This is the highest-yield seed
corpus we have for the corners / surface / edges grading models — see the
"Data Strategy" section of the architecture plan.

## Legal posture (important)

These are **fair-use ML training inputs only**. Do **NOT** redistribute the
scraped images or republish PSA's data. The records and image cache are for
internal model training; downstream artifacts (trained weights, derived
features) are fine. If you ever need to publish anything derived from this,
get legal sign-off first.

The scraper identifies itself in its `User-Agent` and rate-limits to 1 req/sec
by default. Don't lower this number to be clever — being a polite citizen of
the web is part of the legal posture.

## Run locally

```bash
cd ml
python -c "
from pathlib import Path
from data.ingestion import LocalScrapedRecordStore, PSAPopScraper

store = LocalScrapedRecordStore(Path('./.scrape_out'))
scraper = PSAPopScraper(store=store, rate_limit_per_sec=1.0)
stats = scraper.scrape_range(1, 100, stop_after=10)
print(stats)
"
```

Output layout under `./.scrape_out/`:

```
scraped.jsonl                    one ScrapedRecord per line
images/<cert_id>/front.jpg
images/<cert_id>/back.jpg
```

## Run against S3

```python
import boto3
from data.ingestion import PSAPopScraper, S3ScrapedRecordStore

store = S3ScrapedRecordStore(
    bucket="grader-pop-report",
    prefix="psa",
    s3_client=boto3.client("s3"),
)
scraper = PSAPopScraper(store=store, rate_limit_per_sec=1.0)
scraper.scrape_range(1_000_000, 1_001_000)
```

Records land at `s3://grader-pop-report/psa/scraped.jsonl`, images at
`s3://grader-pop-report/psa/images/<cert_id>/<kind>.<ext>`.

## Resume protocol

Every cert ID written to `scraped.jsonl` is treated as "done." The scraper
calls `store.iter_existing_cert_ids()` once at the start of `scrape_range` and
also `store.has_record(cert_id)` per cert (cheap, in-memory). Killing the
process mid-range and restarting with the same range is safe — already-done
certs are skipped.

If a cert page parses fine but image download fails, the record is persisted
with `front_image_path=None` and/or `back_image_path=None`. A future
backfill pass can re-download those images without re-scraping the page.

## Default rate limits and tuning

| Field | Default | Notes |
|-------|---------|-------|
| `rate_limit_per_sec` | `1.0` | 1 request per second — be polite |
| `request_jitter_seconds` | `(0.0, 0.5)` | Random extra delay added to each request |
| `timeout_seconds` | `15.0` | Per-request HTTP timeout |
| `max_retries_per_cert` | `2` | Retries on 5xx / timeout |
| `user_agent` | `ProjectGrader-Bot/0.1 (research; bots@projectgrader.ai)` | Identify ourselves |

`Retry-After` on 429 is honored. A 403 aborts the run (we treat it as "you
are blocked, stop now").

## HTML parsing assumptions

The parser (`parse_cert_html`) makes best-effort assumptions about PSA's
markup — see the docstring in `psa_pop_scraper.py` for the reference HTML
shape. **Before kicking off a real scrape, fetch one or two cert pages by
hand and verify the regexes still match.** The parser is self-contained and
takes ~20 minutes to update if PSA changes their template.

## Stats tracked

`ScrapeStats` records, per run:

- `total_attempted` — non-skipped cert fetches
- `successful` — record persisted
- `not_found` — 404
- `parse_errors` — page returned but couldn't extract grade/name
- `network_errors` — exhausted retries on 5xx / timeout
- `rate_limited` — at least one 429 during this cert's fetch

## When to actually run

This is the **calendar-time leverage piece** — once it's running it
accumulates data while we work on other features. Open a long-running
process (tmux / nohup / EC2 t3.small) pointed at S3 and let it walk a few
hundred thousand cert IDs. Don't try to scrape the entire 80M space in one
shot — start small, validate the data, then scale.
