"""Manufacturer reference image ingest CLI.

One-shot driver — manufacturer catalogs aren't continuously updated like
PSA's grading queue, so this isn't a daemon. Re-run when you want a new
set or to refresh after a Scryfall release. Safe to re-run on the same
data directory: idempotency is by (manufacturer, variant_id) tuple,
already-fetched cards are skipped without an API call.

Usage:
    python -m scripts.manufacturer_refs_ingest --query "set:lea"
    python -m scripts.manufacturer_refs_ingest --query "set:m21" --max-cards 10
    python -m scripts.manufacturer_refs_ingest --query "set:lea" --data-dir /tmp/refs

Configuration:
    --query STR           Scryfall search syntax (required for now;
                          when we add more sources, --source will pick
                          between them)
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

from data.ingestion import (  # noqa: E402
    LocalReferenceStore,
    ScryfallIngestStats,
    ingest_query,
)


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
        "--query",
        required=True,
        help="Scryfall search syntax (e.g. 'set:lea', 'set:m21 r:rare').",
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
        "starting ingest data_dir=%s query=%r max_cards=%s",
        data_dir,
        args.query,
        args.max_cards,
    )

    store = LocalReferenceStore(data_dir)

    # Lazy import here so the CLI's --help doesn't pay for httpx.
    kwargs: dict = {}
    if args.user_agent:
        kwargs["user_agent"] = args.user_agent

    stats: ScryfallIngestStats
    try:
        stats = ingest_query(
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
            "query": args.query,
            "outcome": "ok",
            "stats": asdict(stats),
        },
    )

    _logger.info(
        "ingest done query=%r stats=%s",
        args.query,
        asdict(stats),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
