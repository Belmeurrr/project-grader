"""Reference-embedding pass CLI.

Reads `<data-dir>/references.jsonl` (produced by
`scripts.manufacturer_refs_ingest`) and writes embeddings to
`<data-dir>/reference_embeddings.npz` keyed by
`"<manufacturer>/<variant_id>"`. The counterfeit-detection
embedding-anomaly stage looks them up at inference time.

The embedder is whatever `pipelines.identification.embedding.get_embedder`
returns — controlled by `GRADER_EMBEDDER` (default: SimpleEmbedder; set
to "dinov2" with GRADER_DINOV2_WEIGHTS pointing at a checkpoint to use
the production model). This keeps the embedder choice in one place
rather than duplicating env handling here.

Usage:
    python -m scripts.embed_references --data-dir /path/to/refs
    python -m scripts.embed_references --data-dir /path/to/refs --max-cards 3
    python -m scripts.embed_references --data-dir /path/to/refs --force

Configuration:
    --data-dir PATH       Directory containing `references.jsonl`.
                          Output: `reference_embeddings.npz` in the
                          same directory.
    --max-cards N         Truncate the JSONL at N records before
                          embedding. Useful for smoke runs that don't
                          want to wait for the whole catalog. Note:
                          truncation is by JSONL row order, not by
                          variant id, so re-running with a different
                          --max-cards may miss / cover a different
                          subset.
    --force               Re-embed every record, overwriting any
                          existing entries in the npz.

Exit codes:
    0  success
    1  unexpected exception (I/O, embedder load, etc.)
    2  bad CLI invocation
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

# Mirror the path tweak used by `scripts.manufacturer_refs_ingest` so
# `python ml/scripts/embed_references.py` (no -m) also works.
_ML_ROOT = Path(__file__).resolve().parents[1]
if str(_ML_ROOT) not in sys.path:
    sys.path.insert(0, str(_ML_ROOT))

from data.ingestion.reference_embeddings import embed_references  # noqa: E402

_logger = logging.getLogger("embed_references")


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="embed_references",
        description="Embed manufacturer reference images and persist to .npz.",
    )
    p.add_argument(
        "--data-dir",
        default=os.environ.get("MANUFACTURER_REFS_DATA_DIR")
        or str(Path.home() / "manufacturer_refs"),
        help="Directory containing references.jsonl. "
        "Default: $MANUFACTURER_REFS_DATA_DIR or ~/manufacturer_refs",
    )
    p.add_argument(
        "--max-cards",
        type=int,
        default=None,
        help="Embed only the first N records from the JSONL (smoke runs).",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-embed everything, overwriting existing entries.",
    )
    return p


def _truncated_jsonl(src: Path, n: int) -> Path:
    """Materialize the first `n` lines of `src` into a temp file so
    the embedding pass sees a smaller corpus without us touching the
    real JSONL. Returns the temp path (caller cleans up)."""
    tmp = Path(tempfile.mkstemp(prefix="references_", suffix=".jsonl")[1])
    written = 0
    with src.open("r", encoding="utf-8") as fh_in, tmp.open(
        "w", encoding="utf-8"
    ) as fh_out:
        for line in fh_in:
            if written >= n:
                break
            stripped = line.strip()
            if not stripped:
                continue
            fh_out.write(line if line.endswith("\n") else line + "\n")
            written += 1
    return tmp


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    data_dir = Path(args.data_dir)
    jsonl_path = data_dir / "references.jsonl"
    store_path = data_dir / "reference_embeddings.npz"

    if not jsonl_path.exists():
        _logger.error("references.jsonl not found at %s", jsonl_path)
        return 1

    # Defer the embedder import: pulling the identification pipeline
    # (and DinoV2 if configured) is potentially slow and unwanted for
    # `--help`.
    from pipelines.identification.embedding import get_embedder

    _logger.info("loading embedder")
    embedder = get_embedder()
    _logger.info("embedder=%s dim=%d", type(embedder).__name__, embedder.dim)

    truncated: Path | None = None
    target_jsonl = jsonl_path
    if args.max_cards is not None:
        if args.max_cards < 0:
            _logger.error("--max-cards must be non-negative")
            return 2
        truncated = _truncated_jsonl(jsonl_path, args.max_cards)
        target_jsonl = truncated
        _logger.info(
            "truncated jsonl to %d rows at %s", args.max_cards, truncated
        )

    try:
        stats = embed_references(
            target_jsonl,
            store_path,
            embedder=embedder,
            force=args.force,
        )
    except Exception as e:  # noqa: BLE001 — log and surface as exit 1
        _logger.exception("embedding pass failed: %s", e)
        return 1
    finally:
        if truncated is not None:
            try:
                truncated.unlink()
            except OSError:
                pass

    _logger.info(
        "embedding pass done store=%s stats=%s",
        store_path,
        asdict(stats),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
