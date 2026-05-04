"""Counterfeit-detector threshold recalibration tool.

Recommends AUTHENTIC + COUNTERFEIT thresholds for the ensemble's
detectors based on a labeled corpus. Operates as a sibling to
`counterfeit_benchmark`: shares its runner so detectors run identically
to production, but consumes a real-image labeled corpus instead of the
synthetic-fixture corpus.

Two operating modes per detector, picked automatically based on
corpus contents:

  - **two_sided** (corpus has authentics + counterfeits): both
    thresholds recommended via Youden's-J + FPR-budget rules.
  - **authentic_only** (corpus has only authentics, the expected
    starting state once PSA accumulator data accumulates): only the
    AUTHENTIC threshold is recommended; COUNTERFEIT side is left for
    when fakes appear.

The tool DOES NOT modify `ml/pipelines/counterfeit/ensemble.py`
itself. It emits a drop-in patch block; threshold updates remain a
deliberate human commit so they're reviewable + revertable.

Run from ml/:

    python -m evaluation.counterfeit_recalibration --csv labels.csv
    python -m evaluation.counterfeit_recalibration --psa-authentics ~/psa_data/scraped.jsonl
    python -m evaluation.counterfeit_recalibration --use-default-corpus  # smoke
"""

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
    CURRENT_THRESHOLDS,
    DetectorScores,
    MODE_AUTHENTIC_ONLY,
    MODE_INSUFFICIENT,
    MODE_TWO_SIDED,
    RECALIBRATABLE_DETECTORS,
    ThresholdRecommendation,
    collect_scores,
    recommend_all,
    recommend_thresholds,
)

__all__ = [
    # corpus
    "LabeledRow",
    "LoadStats",
    "load_csv",
    "load_psa_authentics",
    "rows_to_samples",
    # sweep
    "CURRENT_THRESHOLDS",
    "DetectorScores",
    "MODE_AUTHENTIC_ONLY",
    "MODE_INSUFFICIENT",
    "MODE_TWO_SIDED",
    "RECALIBRATABLE_DETECTORS",
    "ThresholdRecommendation",
    "collect_scores",
    "recommend_all",
    "recommend_thresholds",
    # report
    "render_console",
    "render_markdown",
    "to_json_dict",
]
