"""Data ingestion pipelines.

Public API for the PSA Public API client (primary), the deprecated
pop-report scraper (kept for tests + storage compatibility), and the
GitHub seed-corpus ingestor.

Eager-loaded surface (no httpx required):
  - Storage classes / Protocol / shared dataclass:
      LocalScrapedRecordStore, S3ScrapedRecordStore, ScrapedRecordStore,
      ScrapedRecord
  - GitHub seed corpus (stdlib-only):
      SeedRecord, iter_seed_records, write_manifest

Lazy-loaded surface (require httpx, resolved on first attribute access
via PEP 562 __getattr__):
  - PSA Public API client:
      DailyBudget, BudgetExhausted, IngestAborted, IngestStats,
      PSA_PUBLIC_API_BASE_URL, ingest_cert, ingest_range
  - Deprecated httpx scraper (kept for tests + history):
      PSAPopScraper, ScrapeStats, parse_cert_html

This split lets a stdlib-only consumer (e.g. the github_seed manifest
writer running in an environment without httpx) import the package
without paying for httpx, while production consumers see the full
public surface unchanged."""

from __future__ import annotations

from data.ingestion.github_seed import (
    SeedRecord,
    iter_seed_records,
    write_manifest,
)
from data.ingestion.storage import (
    LocalScrapedRecordStore,
    S3ScrapedRecordStore,
    ScrapedRecord,
    ScrapedRecordStore,
)


# Module path for each lazy-loadable name. Triggered the first time
# a caller does `from data.ingestion import X` for one of these, at
# which point we import the underlying module (which pulls httpx) and
# return the attribute. Subsequent accesses are free — Python caches
# the module import in sys.modules.
_LAZY_NAMES: dict[str, str] = {
    # PSA Public API client (primary track)
    "DailyBudget": "data.ingestion.psa_public_api",
    "BudgetExhausted": "data.ingestion.psa_public_api",
    "IngestAborted": "data.ingestion.psa_public_api",
    "IngestStats": "data.ingestion.psa_public_api",
    "PSA_PUBLIC_API_BASE_URL": "data.ingestion.psa_public_api",
    "ingest_cert": "data.ingestion.psa_public_api",
    "ingest_range": "data.ingestion.psa_public_api",
    # Deprecated httpx scraper (kept for tests + history)
    "PSAPopScraper": "data.ingestion.psa_pop_scraper",
    "ScrapeStats": "data.ingestion.psa_pop_scraper",
    "parse_cert_html": "data.ingestion.psa_pop_scraper",
}


def __getattr__(name: str):
    """PEP 562 lazy attribute resolver. Imports the underlying module
    on first access for any name in `_LAZY_NAMES`; raises AttributeError
    for anything else (matches default behavior)."""
    target = _LAZY_NAMES.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module = importlib.import_module(target)
    return getattr(module, name)


__all__ = [
    # Storage (eager)
    "LocalScrapedRecordStore",
    "S3ScrapedRecordStore",
    "ScrapedRecord",
    "ScrapedRecordStore",
    # GitHub seed corpus (eager, stdlib-only)
    "SeedRecord",
    "iter_seed_records",
    "write_manifest",
    # PSA Public API client (lazy)
    "DailyBudget",
    "BudgetExhausted",
    "IngestAborted",
    "IngestStats",
    "PSA_PUBLIC_API_BASE_URL",
    "ingest_cert",
    "ingest_range",
    # Deprecated scraper (lazy — kept for tests / history)
    "PSAPopScraper",
    "ScrapeStats",
    "parse_cert_html",
]
