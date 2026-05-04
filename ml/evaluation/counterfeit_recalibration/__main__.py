"""CLI entry point for the counterfeit-detector recalibration tool.

Run from ml/:

    # Use the synthetic benchmark corpus (smoke / sanity):
    python -m evaluation.counterfeit_recalibration --use-default-corpus

    # Real labeled corpus from a CSV (image_path,ground_truth):
    python -m evaluation.counterfeit_recalibration --csv labels.csv

    # PSA-only authentics (no fakes yet — recalibrates AUTHENTIC side
    # of the threshold band, leaves COUNTERFEIT side alone):
    python -m evaluation.counterfeit_recalibration \\
        --psa-authentics ~/psa_data/scraped.jsonl \\
        --max-records 1000

CSV and PSA-authentics inputs may be combined (CSV provides
counterfeits, PSA jsonl provides authentics).

The tool reports recommended thresholds + a drop-in patch block; it
does NOT modify ensemble.py itself. Threshold updates remain a
deliberate human commit so they're reviewable and revertable.

Exit codes:
    0  success — recommendations written
    1  unexpected exception
    2  bad CLI invocation (missing/conflicting args)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Defensive sys.path bootstrap (matches sibling tools).
_ML_ROOT = Path(__file__).resolve().parents[2]
if str(_ML_ROOT) not in sys.path:
    sys.path.insert(0, str(_ML_ROOT))

from evaluation.counterfeit_benchmark import build_default_corpus, run_benchmark
from evaluation.counterfeit_recalibration.corpus import (
    LabeledRow,
    LoadStats,
    load_csv,
    load_psa_authentics,
    rows_to_samples,
)
from evaluation.counterfeit_recalibration.report import (
    render_console,
    render_markdown,
    to_json_dict,
)
from evaluation.counterfeit_recalibration.sweep import (
    collect_scores,
    recommend_all,
)


_logger = logging.getLogger("counterfeit_recalibration")


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="counterfeit_recalibration",
        description=(
            "Recommend thresholds for the counterfeit-detector ensemble "
            "from a labeled corpus. Does not modify ensemble.py — "
            "outputs a drop-in patch block for review."
        ),
    )
    p.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Labels CSV with columns image_path,ground_truth[,sample_id].",
    )
    p.add_argument(
        "--psa-authentics",
        type=Path,
        default=None,
        help="PSA scraped.jsonl. Every record with a front_image_path "
        "is treated as an authentic exemplar.",
    )
    p.add_argument(
        "--use-default-corpus",
        action="store_true",
        help="Use the synthetic counterfeit_benchmark corpus. Useful for "
        "smoke testing; meaningless for real recalibration.",
    )
    p.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Cap rows pulled from --psa-authentics (smoke runs).",
    )
    p.add_argument(
        "--target-authentic-recall",
        type=float,
        default=0.95,
        help="Fraction of authentics we commit to keeping above the "
        "AUTHENTIC threshold (default 0.95).",
    )
    p.add_argument(
        "--counterfeit-fpr-budget",
        type=float,
        default=0.005,
        help="Max fraction of confident authentics flagged "
        "LIKELY_COUNTERFEIT by a single detector (default 0.005).",
    )
    p.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Write structured JSON report to this path.",
    )
    p.add_argument(
        "--markdown",
        type=Path,
        default=None,
        help="Write Markdown report to this path.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    sources_chosen = sum(
        bool(x) for x in (args.csv, args.psa_authentics, args.use_default_corpus)
    )
    if sources_chosen == 0:
        parser.error(
            "must specify at least one of --csv / --psa-authentics / "
            "--use-default-corpus"
        )

    rows: list[LabeledRow] = []
    samples = []
    load_stats: LoadStats | None = None

    if args.use_default_corpus:
        # Default corpus is in-memory image arrays, not on-disk paths.
        # Use it directly without going through the CSV path.
        if args.csv or args.psa_authentics:
            parser.error(
                "--use-default-corpus is mutually exclusive with "
                "--csv / --psa-authentics"
            )
        samples = list(build_default_corpus())
        load_stats = LoadStats(
            requested=len(samples),
            loaded=len(samples),
            skipped_missing_file=0,
            skipped_unreadable=0,
        )
    else:
        if args.csv:
            rows.extend(load_csv(args.csv))
        if args.psa_authentics:
            rows.extend(
                load_psa_authentics(
                    args.psa_authentics, max_records=args.max_records
                )
            )
        samples, load_stats = rows_to_samples(rows)
        _logger.info(
            "loaded %d/%d samples (missing=%d, unreadable=%d)",
            load_stats.loaded,
            load_stats.requested,
            load_stats.skipped_missing_file,
            load_stats.skipped_unreadable,
        )
        if not samples:
            _logger.error("no samples loaded — nothing to recalibrate against")
            return 1

    _logger.info("running %d samples through detectors", len(samples))
    results = run_benchmark(samples)

    by_detector = collect_scores(results)
    recommendations = recommend_all(
        by_detector,
        target_authentic_recall=args.target_authentic_recall,
        counterfeit_fpr_budget=args.counterfeit_fpr_budget,
    )

    print(render_console(recommendations, load_stats=load_stats))

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(
                to_json_dict(recommendations, load_stats=load_stats),
                indent=2,
                ensure_ascii=False,
            )
        )
        print(f"\nJSON report written: {args.json}")
    if args.markdown is not None:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(
            render_markdown(recommendations, load_stats=load_stats)
        )
        print(f"Markdown report written: {args.markdown}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
