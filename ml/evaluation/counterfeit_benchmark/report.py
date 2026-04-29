"""Metric computation + report rendering for the counterfeit benchmark.

Three render targets:
  - render_console: TTY-friendly summary table
  - render_markdown: report-friendly Markdown for PRs / dashboards
  - to_json_dict: structured payload for CI pipelines / longitudinal
    tracking

Per-detector metrics:
  - n_samples, n_abstained
  - true_positive_rate (TPR)  — recall on counterfeits
  - false_positive_rate (FPR) — fraction of authentics misflagged
  - accuracy                  — fraction of confident verdicts correct

A detector "predicts counterfeit" when its verdict is LIKELY_COUNTERFEIT
or SUSPICIOUS. UNVERIFIED outcomes count as abstentions and are
excluded from TPR/FPR/accuracy denominators (so a detector that
abstains doesn't artificially deflate its own metrics).

The ensemble row uses the SAME predicate (LIKELY_COUNTERFEIT or
SUSPICIOUS counts as "predicts counterfeit") so the combined number
is comparable to the per-detector ones.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Sequence

from pipelines.counterfeit import ensemble

from evaluation.counterfeit_benchmark.corpus import (
    GROUND_TRUTH_AUTHENTIC,
    GROUND_TRUTH_COUNTERFEIT,
    BenchmarkSample,
)
from evaluation.counterfeit_benchmark.runner import (
    DETECTOR_COLOR,
    DETECTOR_ENSEMBLE,
    DETECTOR_ROSETTE,
    DetectorOutcome,
    SampleResult,
)


# Verdicts that "predict counterfeit" for metric purposes. SUSPICIOUS
# is included because in production the cert page surfaces it for
# human review; treating it as a positive in the benchmark matches
# the operational outcome.
_PREDICTS_COUNTERFEIT = frozenset(
    {ensemble.VERDICT_LIKELY_COUNTERFEIT, ensemble.VERDICT_SUSPICIOUS}
)


@dataclass(frozen=True)
class DetectorMetrics:
    """Per-detector accuracy on the labeled corpus.

    Fields are floats in [0, 1] (or None when the denominator is
    empty, e.g. true_positive_rate when no counterfeits were
    presented to a detector that abstained on all of them)."""

    detector: str
    n_samples: int
    n_abstained: int  # verdict == UNVERIFIED
    true_positives: int  # ground=counterfeit AND verdict in _PREDICTS_COUNTERFEIT
    false_positives: int  # ground=authentic AND verdict in _PREDICTS_COUNTERFEIT
    true_negatives: int
    false_negatives: int
    true_positive_rate: float | None
    false_positive_rate: float | None
    accuracy: float | None


def compute_metrics(
    results: Sequence[SampleResult],
) -> dict[str, DetectorMetrics]:
    """Per-detector metrics over the result set, keyed by detector name."""
    by_detector: dict[str, list[tuple[BenchmarkSample, DetectorOutcome]]] = {}
    for r in results:
        for o in r.outcomes:
            by_detector.setdefault(o.detector, []).append((r.sample, o))

    out: dict[str, DetectorMetrics] = {}
    for detector, pairs in by_detector.items():
        out[detector] = _metrics_for_detector(detector, pairs)
    return out


def _metrics_for_detector(
    detector: str,
    pairs: Sequence[tuple[BenchmarkSample, DetectorOutcome]],
) -> DetectorMetrics:
    n_samples = len(pairs)
    n_abstained = 0
    tp = fp = tn = fn = 0
    n_authentic = n_counterfeit = 0

    for sample, outcome in pairs:
        if outcome.verdict == ensemble.VERDICT_UNVERIFIED:
            n_abstained += 1
            continue
        predicts_counterfeit = outcome.verdict in _PREDICTS_COUNTERFEIT
        if sample.ground_truth == GROUND_TRUTH_COUNTERFEIT:
            n_counterfeit += 1
            if predicts_counterfeit:
                tp += 1
            else:
                fn += 1
        elif sample.ground_truth == GROUND_TRUTH_AUTHENTIC:
            n_authentic += 1
            if predicts_counterfeit:
                fp += 1
            else:
                tn += 1

    tpr = (tp / n_counterfeit) if n_counterfeit else None
    fpr = (fp / n_authentic) if n_authentic else None
    n_confident = tp + fp + tn + fn
    accuracy = ((tp + tn) / n_confident) if n_confident else None

    return DetectorMetrics(
        detector=detector,
        n_samples=n_samples,
        n_abstained=n_abstained,
        true_positives=tp,
        false_positives=fp,
        true_negatives=tn,
        false_negatives=fn,
        true_positive_rate=tpr,
        false_positive_rate=fpr,
        accuracy=accuracy,
    )


# --------------------------------------------------------------------------
# Renderers
# --------------------------------------------------------------------------


def _fmt_pct(v: float | None) -> str:
    return f"{v * 100:5.1f}%" if v is not None else "  n/a"


def render_console(metrics: dict[str, DetectorMetrics]) -> str:
    """TTY-friendly summary. ASCII only — no terminal escape codes,
    so log captures + CI scrollback render readably."""
    detectors_in_order = [DETECTOR_ROSETTE, DETECTOR_COLOR, DETECTOR_ENSEMBLE]
    seen = set(metrics.keys())
    extras = sorted(seen - set(detectors_in_order))
    detectors_in_order = [d for d in detectors_in_order if d in seen] + extras

    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("Counterfeit-detector benchmark")
    lines.append("=" * 72)
    lines.append(
        f"{'detector':<12} {'n':>4} {'abst':>4} {'TPR':>7} {'FPR':>7} {'acc':>7}  TP/FP/TN/FN"
    )
    lines.append("-" * 72)
    for d in detectors_in_order:
        m = metrics[d]
        lines.append(
            f"{m.detector:<12} {m.n_samples:>4} {m.n_abstained:>4} "
            f"{_fmt_pct(m.true_positive_rate):>7} "
            f"{_fmt_pct(m.false_positive_rate):>7} "
            f"{_fmt_pct(m.accuracy):>7}  "
            f"{m.true_positives}/{m.false_positives}/{m.true_negatives}/{m.false_negatives}"
        )
    lines.append("=" * 72)
    return "\n".join(lines)


def render_markdown(
    metrics: dict[str, DetectorMetrics],
    results: Sequence[SampleResult],
) -> str:
    """Markdown report for PR comments / dashboard pages."""
    detectors_in_order = [DETECTOR_ROSETTE, DETECTOR_COLOR, DETECTOR_ENSEMBLE]
    seen = set(metrics.keys())
    detectors_in_order = [d for d in detectors_in_order if d in seen]

    n_samples = len(results)
    n_authentic = sum(
        1 for r in results if r.sample.ground_truth == GROUND_TRUTH_AUTHENTIC
    )
    n_counterfeit = sum(
        1 for r in results if r.sample.ground_truth == GROUND_TRUTH_COUNTERFEIT
    )

    lines: list[str] = []
    lines.append("# Counterfeit-detector benchmark")
    lines.append("")
    lines.append(
        f"_{n_samples} samples ({n_authentic} authentic + "
        f"{n_counterfeit} counterfeit) • run at "
        f"{datetime.now(timezone.utc).isoformat()}_"
    )
    lines.append("")
    lines.append("| detector | n | abst. | TPR | FPR | accuracy | TP/FP/TN/FN |")
    lines.append("|---|---:|---:|---:|---:|---:|---|")
    for d in detectors_in_order:
        m = metrics[d]
        lines.append(
            f"| `{m.detector}` | {m.n_samples} | {m.n_abstained} | "
            f"{_fmt_pct(m.true_positive_rate).strip()} | "
            f"{_fmt_pct(m.false_positive_rate).strip()} | "
            f"{_fmt_pct(m.accuracy).strip()} | "
            f"{m.true_positives}/{m.false_positives}/{m.true_negatives}/{m.false_negatives} |"
        )
    lines.append("")
    lines.append(
        "> TPR = recall on counterfeits • FPR = authentics misflagged • "
        "abst. = detector returned UNVERIFIED (excluded from TPR/FPR)"
    )
    return "\n".join(lines) + "\n"


def to_json_dict(
    metrics: dict[str, DetectorMetrics],
    results: Sequence[SampleResult],
) -> dict[str, Any]:
    """Structured payload for CI / longitudinal tracking."""
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "corpus_size": len(results),
        "ground_truth_counts": {
            GROUND_TRUTH_AUTHENTIC: sum(
                1 for r in results if r.sample.ground_truth == GROUND_TRUTH_AUTHENTIC
            ),
            GROUND_TRUTH_COUNTERFEIT: sum(
                1 for r in results if r.sample.ground_truth == GROUND_TRUTH_COUNTERFEIT
            ),
        },
        "metrics": {k: asdict(v) for k, v in metrics.items()},
        "samples": [
            {
                "sample_id": r.sample.sample_id,
                "ground_truth": r.sample.ground_truth,
                "variant": r.sample.variant,
                "elapsed_seconds": r.elapsed_seconds,
                "outcomes": [
                    {
                        "detector": o.detector,
                        "score": o.score,
                        "confidence": o.confidence,
                        "verdict": o.verdict,
                        "error": o.error,
                    }
                    for o in r.outcomes
                ],
            }
            for r in results
        ],
    }


__all__ = [
    "DetectorMetrics",
    "compute_metrics",
    "render_console",
    "render_markdown",
    "to_json_dict",
]
