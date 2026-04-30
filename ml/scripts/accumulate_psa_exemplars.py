"""Accumulate PSA-graded exemplars into the reference-embeddings npz.

One-shot driver, idempotent by cert_id (re-running the same PSA
JSONL is a no-op for already-consumed certs). Run after each daily
PSA ingest cycle so newly-graded cards that match known variants
contribute to embedding-anomaly's centroid coverage.

Usage:
    # Build a catalog from the on-disk references and accumulate
    # against today's PSA records.
    python -m scripts.accumulate_psa_exemplars \\
        --psa-data-dir ~/psa_data \\
        --refs-data-dir ~/manufacturer_refs \\
        --max-records 200

Args:
    --psa-data-dir       Where the PSA `scraped.jsonl` lives. Default
                         $PSA_INGEST_DATA_DIR or ~/psa_data.
    --refs-data-dir      Where references.jsonl + reference_embeddings.npz
                         live. Default $MANUFACTURER_REFS_DATA_DIR or
                         ~/manufacturer_refs.
    --log-path           Where the consumed-cert sidecar log lives.
                         Default <refs-data-dir>/psa_exemplars_log.jsonl.
    --max-records N      Stop after iterating N PSA rows. Useful for
                         smoke runs.

Catalog source:
    The catalog used for identification is built from the on-disk
    references.jsonl + reference_embeddings.npz — the same data the
    inference path consults. PSA records will only match variants the
    manufacturer scrapes have populated, which is the right
    correctness-vs-coverage trade for this script.

Embedder source:
    Same as `embed_references` — `pipelines.identification.embedding.
    get_embedder()`. Pick via `GRADER_EMBEDDER` env (`simple` for the
    fast deterministic embedder used in dev / tests, `dinov2` for the
    real backbone). Must match the embedder used at submission time
    in production.

Exit codes:
    0  success (or zero exemplars appended — no records matched)
    1  unexpected exception (file I/O, embedder load, etc.)
    2  bad CLI invocation
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

# Defensive sys.path: when invoked as a script (rather than `python -m`),
# ensure ml/ is importable.
_ML_ROOT = Path(__file__).resolve().parents[1]
if str(_ML_ROOT) not in sys.path:
    sys.path.insert(0, str(_ML_ROOT))


_logger = logging.getLogger("accumulate_psa_exemplars")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _build_catalog_from_references(refs_data_dir: Path):
    """Build an InMemoryCatalogIndex from references.jsonl +
    reference_embeddings.npz.

    Only variants that have BOTH a JSONL row (for the metadata) AND
    an embedding (for the lookup) become catalog entries. JSONL rows
    without embeddings are skipped — they wouldn't be findable by the
    embedding side of `identify` anyway."""
    from pipelines.identification.catalog import (
        CardCatalogEntry,
        InMemoryCatalogIndex,
    )
    from data.ingestion.reference_embeddings import load_embeddings

    refs_jsonl = refs_data_dir / "references.jsonl"
    embeddings_npz = refs_data_dir / "reference_embeddings.npz"

    if not refs_jsonl.exists():
        raise FileNotFoundError(
            f"references.jsonl not found at {refs_jsonl} — has the manufacturer scrape run?"
        )

    embeddings = load_embeddings(embeddings_npz)  # {} if file missing
    catalog = InMemoryCatalogIndex()

    with refs_jsonl.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            manufacturer = obj.get("manufacturer")
            variant_id = obj.get("variant_id")
            name = obj.get("name")
            if not (
                isinstance(manufacturer, str)
                and isinstance(variant_id, str)
                and isinstance(name, str)
            ):
                continue
            vec = embeddings.get((manufacturer, variant_id))
            if vec is None:
                continue
            # Use the first reference embedding as the catalog's
            # canonical_embedding (matches what the seed scrape
            # produces; subsequent appends from this script don't
            # change the catalog entry's embedding, just the npz).
            ref_vec = vec[0] if vec.ndim == 2 else vec

            set_code = obj.get("set_code") or ""
            card_number = obj.get("collector_number") or ""

            catalog.add(
                CardCatalogEntry(
                    variant_id=variant_id,
                    name=name,
                    set_code=str(set_code),
                    card_number=str(card_number),
                    game=manufacturer,
                    canonical_phash=None,  # phash not stored in references.jsonl
                    canonical_embedding=ref_vec,
                )
            )
    return catalog


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="accumulate_psa_exemplars",
        description=(
            "Append PSA-graded card embeddings to the reference store "
            "as additional authentic exemplars for matched variants."
        ),
    )
    p.add_argument(
        "--psa-data-dir",
        default=os.environ.get("PSA_INGEST_DATA_DIR")
        or str(Path.home() / "psa_data"),
    )
    p.add_argument(
        "--refs-data-dir",
        default=os.environ.get("MANUFACTURER_REFS_DATA_DIR")
        or str(Path.home() / "manufacturer_refs"),
    )
    p.add_argument(
        "--log-path",
        default=None,
        help="Override path to the consumed-cert sidecar log. "
        "Default: <refs-data-dir>/psa_exemplars_log.jsonl",
    )
    p.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Stop after iterating N PSA rows (smoke runs).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    psa_data_dir = Path(args.psa_data_dir)
    refs_data_dir = Path(args.refs_data_dir)
    log_path = Path(args.log_path) if args.log_path else (
        refs_data_dir / "psa_exemplars_log.jsonl"
    )

    psa_jsonl = psa_data_dir / "scraped.jsonl"
    embeddings_npz = refs_data_dir / "reference_embeddings.npz"

    _logger.info(
        "starting psa_data_dir=%s refs_data_dir=%s log_path=%s max_records=%s",
        psa_data_dir,
        refs_data_dir,
        log_path,
        args.max_records,
    )

    try:
        catalog = _build_catalog_from_references(refs_data_dir)
    except FileNotFoundError as e:
        _logger.error("catalog build failed: %s", e)
        return 1

    _logger.info("catalog size = %d entries", len(catalog))

    from data.ingestion.psa_exemplars import accumulate_psa_exemplars
    from pipelines.identification.embedding import get_embedder

    try:
        embedder = get_embedder()
    except Exception as e:  # noqa: BLE001
        _logger.exception("embedder construction failed: %s", e)
        return 1

    try:
        stats = accumulate_psa_exemplars(
            psa_jsonl_path=psa_jsonl,
            embeddings_store_path=embeddings_npz,
            log_path=log_path,
            catalog=catalog,
            embedder=embedder,
            max_records=args.max_records,
        )
    except Exception as e:  # noqa: BLE001
        _logger.exception("accumulator failed: %s", e)
        return 1

    _logger.info(
        "done timestamp=%s stats=%s",
        _now_iso(),
        asdict(stats),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
