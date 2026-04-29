"""Data ingestion pipelines.

Public API for the PSA Public API client (primary), the deprecated
pop-report scraper (kept for the storage layer it shares), and the
GitHub seed-corpus ingestor. Importing this package does NOT pull in
boto3 — the S3 store imports it lazily on first use — but it DOES
import httpx, since both PSA paths require it.

The github_seed module is stdlib-only, but importing it via this
package's `__init__` will still pull httpx through psa_pop_scraper.
A dev who needs ONLY github_seed in an httpx-free environment can
import its file directly via importlib."""

from __future__ import annotations

from data.ingestion.psa_pop_scraper import (
    PSAPopScraper,
    ScrapedRecord,
    ScrapeStats,
    parse_cert_html,
)
from data.ingestion.psa_public_api import (
    DailyBudget,
    BudgetExhausted,
    IngestAborted,
    IngestStats,
    PSA_PUBLIC_API_BASE_URL,
    ingest_cert,
    ingest_range,
)
from data.ingestion.github_seed import (
    SeedRecord,
    iter_seed_records,
    write_manifest,
)
from data.ingestion.storage import (
    LocalScrapedRecordStore,
    S3ScrapedRecordStore,
    ScrapedRecordStore,
)

__all__ = [
    # Storage
    "LocalScrapedRecordStore",
    "S3ScrapedRecordStore",
    "ScrapedRecordStore",
    # Shared record / stats
    "ScrapedRecord",
    "ScrapeStats",
    # PSA Public API client (primary track)
    "DailyBudget",
    "BudgetExhausted",
    "IngestAborted",
    "IngestStats",
    "PSA_PUBLIC_API_BASE_URL",
    "ingest_cert",
    "ingest_range",
    # GitHub seed corpus
    "SeedRecord",
    "iter_seed_records",
    "write_manifest",
    # Deprecated httpx scraper (storage layer reused; HTML parser kept for tests)
    "PSAPopScraper",
    "parse_cert_html",
]
