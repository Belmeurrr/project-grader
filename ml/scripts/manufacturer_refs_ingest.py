"""Manufacturer reference image ingest CLI.

One-shot driver — manufacturer catalogs aren't continuously updated like
PSA's grading queue, so this isn't a daemon. Re-run when you want a new
set or to refresh after a catalog release. Safe to re-run on the same
data directory: idempotency is by (manufacturer, variant_id) tuple,
already-fetched cards are skipped without an API call.

Sources:
    --source mtg     → Scryfall (api.scryfall.com), default
    --source pokemon → PokemonTCG.io (api.pokemontcg.io)

Each source exposes a different query syntax (Scryfall uses its own
fluent search syntax; PokemonTCG.io uses Lucene). The --query string
is passed through verbatim.

Usage:
    python -m scripts.manufacturer_refs_ingest --query "set:lea"
    python -m scripts.manufacturer_refs_ingest --source pokemon --query "set.id:base1"
    python -m scripts.manufacturer_refs_ingest --source pokemon --query "set.id:swsh1" --max-cards 10

Configuration:
    --source STR          One of {mtg, pokemon}. Default: mtg.
    --query STR           Source-specific search syntax.
    --data-dir PATH       Where to write references.jsonl + images.
                          Default: $MANUFACTURER_REFS_DATA_DIR or
                          ~/manufacturer_refs
    --max-cards N         Stop after N successful records. Useful for
                          smoke runs.
    --user-agent STR      Override the default User-Agent header
                          (rarely needed; default identifies the
                          project + repo URL).

Exit codes:
    0  success (or no matches — `not_found` is not an error)
    1  unexpected exception (network, parse, etc.)
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

# Path tweak: this script is expected to run with the project's `ml/`
# dir on sys.path (set by `python -m scripts.manufacturer_refs_ingest`
# from inside `ml/`, or by a wrapper that prepends it). Defensive
# fallback for direct `python ml/scripts/manufacturer_refs_ingest.py`.
_ML_ROOT = Path(__file__).resolve().parents[1]
if str(_ML_ROOT) not in sys.path:
    sys.path.insert(0, str(_ML_ROOT))

from data.ingestion import LocalReferenceStore  # noqa: E402


_SOURCES = ("mtg", "pokemon")


_logger = logging.getLogger("manufacturer_refs_ingest")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _append_log(log_path: Path, entry: dict) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="manufacturer_refs_ingest",
        description="Ingest manufacturer reference card images.",
    )
    p.add_argument(
        "--source",
        choices=_SOURCES,
        default="mtg",
        help=(
            "Catalog source: 'mtg' for Scryfall (default), 'pokemon' for "
            "PokemonTCG.io. Each source has its own query syntax."
        ),
    )
    p.add_argument(
        "--query",
        required=True,
        help=(
            "Source-specific search syntax. Scryfall: e.g. 'set:lea', "
            "'set:m21 r:rare'. PokemonTCG.io (Lucene): e.g. 'set.id:base1', "
            "'rarity:Rare'."
        ),
    )
    p.add_argument(
        "--data-dir",
        default=os.environ.get("MANUFACTURER_REFS_DATA_DIR")
        or str(Path.home() / "manufacturer_refs"),
        help="Where to write references.jsonl + images. "
        "Default: $MANUFACTURER_REFS_DATA_DIR or ~/manufacturer_refs",
    )
    p.add_argument(
        "--max-cards",
        type=int,
        default=None,
        help="Stop after N successful records (smoke runs).",
    )
    p.add_argument(
        "--user-agent",
        default=None,
        help="Override the default User-Agent header.",
    )
    return p


def _dispatch_ingest(source: str):
    """Resolve the (ingest_query_callable, stats_factory) pair for a
    source. Submodule imports are deferred to here so the CLI --help
    output doesn't pay for httpx."""
    if source == "mtg":
        from data.ingestion import scryfall

        return scryfall.ingest_query, scryfall.ScryfallIngestStats
    if source == "pokemon":
        from data.ingestion import pokemontcg

        return pokemontcg.ingest_query, pokemontcg.PokemonTCGIngestStats
    raise ValueError(f"unknown source: {source!r}")  # unreachable: argparse gates this


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    data_dir = Path(args.data_dir)
    log_path = data_dir / "ingest_log.jsonl"

    _logger.info(
        "starting ingest source=%s data_dir=%s query=%r max_cards=%s",
        args.source,
        data_dir,
        args.query,
        args.max_cards,
    )

    store = LocalReferenceStore(data_dir)
    ingest_fn, _stats_cls = _dispatch_ingest(args.source)

    kwargs: dict = {}
    if args.user_agent:
        kwargs["user_agent"] = args.user_agent

    try:
        stats = ingest_fn(
            args.query,
            store=store,
            max_cards=args.max_cards,
            **kwargs,
        )
    except Exception as e:  # noqa: BLE001 — log and exit 1 on anything
        _append_log(
            log_path,
            {
                "timestamp": _now_iso(),
                "source": args.source,
                "query": args.query,
                "outcome": "error",
                "error": str(e),
                "error_type": type(e).__name__,
            },
        )
        _logger.exception("ingest failed: %s", e)
        return 1

    _append_log(
        log_path,
        {
            "timestamp": _now_iso(),
            "source": args.source,
            "query": args.query,
            "outcome": "ok",
            "stats": asdict(stats),
        },
    )

    _logger.info(
        "ingest done source=%s query=%r stats=%s",
        args.source,
        args.query,
        asdict(stats),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
