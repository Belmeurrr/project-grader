"""Data ingestion pipelines.

Public API for the PSA pop-report scraper and its storage backends. Importing
this package does NOT pull in boto3 — the S3 store imports it lazily on first
use, so a dev who only needs the local store doesn't pay for the dep."""

from __future__ import annotations

from data.ingestion.psa_pop_scraper import (
    PSAPopScraper,
    ScrapedRecord,
    ScrapeStats,
    parse_cert_html,
)
from data.ingestion.storage import (
    LocalScrapedRecordStore,
    S3ScrapedRecordStore,
    ScrapedRecordStore,
)

__all__ = [
    "LocalScrapedRecordStore",
    "PSAPopScraper",
    "S3ScrapedRecordStore",
    "ScrapedRecord",
    "ScrapedRecordStore",
    "ScrapeStats",
    "parse_cert_html",
]
