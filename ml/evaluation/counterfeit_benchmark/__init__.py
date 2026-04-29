"""Counterfeit-detection benchmark.

Runs the rosette + color-profile detectors against a labeled corpus
of synthetic + (eventually) real-card samples. Reports per-detector
TPR/FPR/accuracy and the ensemble's combined verdict accuracy.

Operational hygiene per the project plan:
    "Counterfeit benchmark — curated test set (50 known authentics +
    50 known counterfeits), refreshed quarterly. CI gate on
    recall ≥ 90% / FPR ≤ 2%."

v1 corpus is entirely synthetic — built from the existing fixtures in
ml/tests/fixtures.py with parameterized generators for variety. As
real images accumulate (PSA Public API drip + manufacturer reference
scrape), they swap in alongside; the benchmark consumes whatever's
labeled in the corpus.

NOT included in v1:
  - Embedding-anomaly detector. It needs per-variant authentic
    reference embeddings, and that store doesn't exist yet (matches
    why it isn't wired into the pipeline runner). Adding it to the
    benchmark before adding it to production would create the same
    threshold-drift problem this whole module exists to prevent.
  - A hard CI gate. v1 prints metrics; thresholds in TODO ("recall
    ≥ 90% / FPR ≤ 2%") are for production data. Synthetic fixtures
    give artificial scores; gating against them produces either a
    no-op (if thresholds always pass) or false alarms (if they don't).
    Gate gets added once the corpus is ≥ 50% real images."""

from evaluation.counterfeit_benchmark.corpus import (
    BenchmarkSample,
    GROUND_TRUTH_AUTHENTIC,
    GROUND_TRUTH_COUNTERFEIT,
    build_default_corpus,
)
from evaluation.counterfeit_benchmark.runner import (
    DetectorOutcome,
    SampleResult,
    run_benchmark,
)
from evaluation.counterfeit_benchmark.report import (
    DetectorMetrics,
    compute_metrics,
    render_console,
    render_markdown,
    to_json_dict,
)

__all__ = [
    "BenchmarkSample",
    "GROUND_TRUTH_AUTHENTIC",
    "GROUND_TRUTH_COUNTERFEIT",
    "build_default_corpus",
    "DetectorOutcome",
    "SampleResult",
    "run_benchmark",
    "DetectorMetrics",
    "compute_metrics",
    "render_console",
    "render_markdown",
    "to_json_dict",
]
