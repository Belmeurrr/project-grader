"""Data ingestion pipelines.

Public API for the PSA Public API client (primary), the deprecated
pop-report scraper (kept for tests + storage compatibility), the
GitHub seed-corpus ingestor, and the manufacturer reference image
ingest (Scryfall today, PokemonTCG.io next).

Eager-loaded surface (no httpx required):
  - PSA storage classes / Protocol / shared dataclass:
      LocalScrapedRecordStore, S3ScrapedRecordStore, ScrapedRecordStore,
      ScrapedRecord
  - Manufacturer reference storage:
      LocalReferenceStore, ReferenceRecordStore, ReferenceRecord,
      reference_image_relative_key
  - GitHub seed corpus (stdlib-only):
      SeedRecord, iter_seed_records, write_manifest

Lazy-loaded surface (require httpx, resolved on first attribute access
via PEP 562 __getattr__):
  - PSA Public API client:
      DailyBudget, BudgetExhausted, IngestAborted, IngestStats,
      PSA_PUBLIC_API_BASE_URL, ingest_cert, ingest_range
  - Deprecated httpx scraper (kept for tests + history):
      PSAPopScraper, ScrapeStats, parse_cert_html
  - Scryfall reference ingest:
      SCRYFALL_API_BASE_URL, ScryfallIngestStats,
      iter_cards_for_query, ingest_query
      (these names are the Scryfall variant — for the PokemonTCG.io
      variant import the submodule directly:
       `from data.ingestion import pokemontcg`)
  - Scryfall bulk-data ingest (corpus-scale alternative to ingest_query):
      DEFAULT_BULK_TYPE, ingest_bulk
  - PokemonTCG.io reference ingest (top-level surface):
      POKEMONTCG_API_BASE_URL, PokemonTCGIngestStats

This split lets a stdlib-only consumer (e.g. the github_seed manifest
writer running in an environment without httpx) import the package
without paying for httpx, while production consumers see the full
public surface unchanged. Function-name overlaps between Scryfall and
PokemonTCG.io (both have an `ingest_query`) are resolved by exposing
only the Scryfall functions at the top level — the PokemonTCG.io
variant is reachable via direct submodule import, which is what the
CLI driver does."""

from __future__ import annotations

from data.ingestion.github_seed import (
    SeedRecord,
    iter_seed_records,
    write_manifest,
)
from data.ingestion.references_storage import (
    LocalReferenceStore,
    ReferenceRecord,
    ReferenceRecordStore,
    reference_image_relative_key,
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
    # Scryfall reference ingest
    "SCRYFALL_API_BASE_URL": "data.ingestion.scryfall",
    "ScryfallIngestStats": "data.ingestion.scryfall",
    "iter_cards_for_query": "data.ingestion.scryfall",
    "ingest_query": "data.ingestion.scryfall",
    # Scryfall bulk-data ingest (corpus-scale)
    "DEFAULT_BULK_TYPE": "data.ingestion.scryfall_bulk",
    "ingest_bulk": "data.ingestion.scryfall_bulk",
    # PokemonTCG.io reference ingest (constants + stats only — the
    # function names overlap with Scryfall, so direct submodule import
    # is the right pattern for PokemonTCG.io's `ingest_query` etc.)
    "POKEMONTCG_API_BASE_URL": "data.ingestion.pokemontcg",
    "PokemonTCGIngestStats": "data.ingestion.pokemontcg",
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
    # PSA storage (eager)
    "LocalScrapedRecordStore",
    "S3ScrapedRecordStore",
    "ScrapedRecord",
    "ScrapedRecordStore",
    # Manufacturer reference storage (eager)
    "LocalReferenceStore",
    "ReferenceRecord",
    "ReferenceRecordStore",
    "reference_image_relative_key",
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
    # Scryfall reference ingest (lazy — needs httpx)
    "SCRYFALL_API_BASE_URL",
    "ScryfallIngestStats",
    "iter_cards_for_query",
    "ingest_query",
    # Scryfall bulk-data ingest (lazy — needs httpx)
    "DEFAULT_BULK_TYPE",
    "ingest_bulk",
    # PokemonTCG.io reference ingest (lazy — needs httpx)
    "POKEMONTCG_API_BASE_URL",
    "PokemonTCGIngestStats",
]
