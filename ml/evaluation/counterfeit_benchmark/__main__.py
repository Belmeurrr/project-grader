"""CLI entry point for the counterfeit-detector benchmark.

Run from ml/:
    python -m evaluation.counterfeit_benchmark            # console only
    python -m evaluation.counterfeit_benchmark --json out.json
    python -m evaluation.counterfeit_benchmark --markdown out.md

Both report files are optional — without them, only the console
summary table prints. Returns exit 0 always (this is a measurement
tool, not a CI gate; gating gets layered on once the corpus
is ≥ 50% real images).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure ml/ root is on sys.path when invoked as `python -m`.
_ML_ROOT = Path(__file__).resolve().parents[2]
if str(_ML_ROOT) not in sys.path:
    sys.path.insert(0, str(_ML_ROOT))

from evaluation.counterfeit_benchmark import (  # noqa: E402
    build_default_corpus,
    compute_metrics,
    render_console,
    render_markdown,
    run_benchmark,
    to_json_dict,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the counterfeit-detector benchmark against the "
        "default synthetic corpus and print metrics."
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Write the structured JSON report to this path.",
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=None,
        help="Write the Markdown report to this path.",
    )
    args = parser.parse_args()

    corpus = build_default_corpus()
    results = run_benchmark(corpus)
    metrics = compute_metrics(results)

    print(render_console(metrics))

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(to_json_dict(metrics, results), indent=2, ensure_ascii=False)
        )
        print(f"\nJSON report written: {args.json}")
    if args.markdown is not None:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(render_markdown(metrics, results))
        print(f"Markdown report written: {args.markdown}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
