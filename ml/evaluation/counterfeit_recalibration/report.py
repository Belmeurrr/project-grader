"""Render `ThresholdRecommendation`s as console / Markdown / JSON.

The console + Markdown renderers also include a "drop-in patch" block
— literal text the user can paste over `pipelines.counterfeit.ensemble`
constants to apply the recommendation. This makes the loop:

    1. python -m evaluation.counterfeit_recalibration --csv ...
    2. read the recommendation
    3. paste the patch block
    4. run pytest

…instead of "now hand-edit four constants based on this report."

The patch block intentionally only includes detectors whose mode is
TWO_SIDED — partial recommendations from authentic-only mode are
surfaced in the rendered tables but NOT in the patch, so the user
isn't tempted to half-apply them and forget the counterfeit threshold
later.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from evaluation.counterfeit_recalibration.corpus import LoadStats
from evaluation.counterfeit_recalibration.sweep import (
    MODE_AUTHENTIC_ONLY,
    MODE_INSUFFICIENT,
    MODE_TWO_SIDED,
    ThresholdRecommendation,
)


# Detector → ensemble.py constant names. Used to render the drop-in
# patch in the same identifier shape the source file uses.
_PATCH_CONSTANT_NAMES: dict[str, tuple[str, str]] = {
    "rosette": (
        "ROSETTE_AUTHENTIC_THRESHOLD",
        "ROSETTE_COUNTERFEIT_THRESHOLD",
    ),
    "color": (
        "COLOR_AUTHENTIC_THRESHOLD",
        "COLOR_COUNTERFEIT_THRESHOLD",
    ),
    "embedding_anomaly": (
        "EMBEDDING_AUTHENTIC_THRESHOLD",
        "EMBEDDING_COUNTERFEIT_THRESHOLD",
    ),
    "typography": (
        "TYPOGRAPHY_AUTHENTIC_THRESHOLD",
        "TYPOGRAPHY_COUNTERFEIT_THRESHOLD",
    ),
    "holographic": (
        "HOLOGRAPHIC_AUTHENTIC_THRESHOLD",
        "HOLOGRAPHIC_COUNTERFEIT_THRESHOLD",
    ),
    "knn_reference": (
        "KNN_REFERENCE_AUTHENTIC_THRESHOLD",
        "KNN_REFERENCE_COUNTERFEIT_THRESHOLD",
    ),
    "substrate": (
        "SUBSTRATE_AUTHENTIC_THRESHOLD",
        "SUBSTRATE_COUNTERFEIT_THRESHOLD",
    ),
}


def _fmt_threshold(v: float | None) -> str:
    return f"{v:.4f}" if v is not None else "  n/a "


def _fmt_pct(v: float | None) -> str:
    return f"{v * 100:5.1f}%" if v is not None else "  n/a"


def _fmt_delta(current: float, recommended: float | None) -> str:
    if recommended is None:
        return "  n/a "
    delta = recommended - current
    sign = "+" if delta >= 0 else "-"
    return f"{sign}{abs(delta):.4f}"


def _detector_order(by_det: dict[str, ThresholdRecommendation]) -> list[str]:
    """Render order matching the benchmark/console convention."""
    fixed = [
        "rosette",
        "color",
        "embedding_anomaly",
        "typography",
        "holographic",
        "knn_reference",
        "substrate",
    ]
    seen = set(by_det.keys())
    return [d for d in fixed if d in seen] + sorted(seen - set(fixed))


# --------------------------------------------------------------------------
# Console
# --------------------------------------------------------------------------


def render_console(
    recommendations: dict[str, ThresholdRecommendation],
    *,
    load_stats: LoadStats | None = None,
) -> str:
    """TTY-friendly summary table + per-detector notes + patch block."""
    lines: list[str] = []
    lines.append("=" * 78)
    lines.append("Counterfeit-detector threshold recalibration")
    lines.append("=" * 78)

    if load_stats is not None:
        lines.append(
            f"corpus: {load_stats.loaded}/{load_stats.requested} loaded "
            f"(missing={load_stats.skipped_missing_file}, "
            f"unreadable={load_stats.skipped_unreadable})"
        )
        lines.append("-" * 78)

    lines.append(
        f"{'detector':<20} {'mode':<16} "
        f"{'cur_a':>7} {'rec_a':>7} {'Δa':>8}  "
        f"{'cur_c':>7} {'rec_c':>7} {'Δc':>8}"
    )
    lines.append("-" * 78)

    for det in _detector_order(recommendations):
        r = recommendations[det]
        lines.append(
            f"{r.detector:<20} {r.mode:<16} "
            f"{r.current_authentic_threshold:>7.4f} "
            f"{_fmt_threshold(r.recommended_authentic_threshold):>7} "
            f"{_fmt_delta(r.current_authentic_threshold, r.recommended_authentic_threshold):>8}  "
            f"{r.current_counterfeit_threshold:>7.4f} "
            f"{_fmt_threshold(r.recommended_counterfeit_threshold):>7} "
            f"{_fmt_delta(r.current_counterfeit_threshold, r.recommended_counterfeit_threshold):>8}"
        )

    lines.append("-" * 78)
    lines.append("Achieved metrics @ recommended thresholds:")
    lines.append(
        f"{'detector':<20} {'n_a':>5} {'n_c':>5} {'abst':>5}  "
        f"{'auth_recall':>12} {'cf_recall':>10} {'auth_FPR':>10}"
    )
    for det in _detector_order(recommendations):
        r = recommendations[det]
        lines.append(
            f"{r.detector:<20} {r.n_authentic_confident:>5} "
            f"{r.n_counterfeit_confident:>5} {r.n_abstained:>5}  "
            f"{_fmt_pct(r.achieved_authentic_recall):>12} "
            f"{_fmt_pct(r.achieved_counterfeit_recall):>10} "
            f"{_fmt_pct(r.achieved_authentic_fpr):>10}"
        )
    lines.append("=" * 78)

    # Per-detector notes
    any_notes = any(r.notes for r in recommendations.values())
    if any_notes:
        lines.append("Notes:")
        for det in _detector_order(recommendations):
            r = recommendations[det]
            for note in r.notes:
                lines.append(f"  [{det}] {note}")
        lines.append("=" * 78)

    # Drop-in patch (TWO_SIDED detectors only)
    patch = _render_patch_block(recommendations)
    if patch:
        lines.append("Drop-in patch for ml/pipelines/counterfeit/ensemble.py:")
        lines.append("")
        lines.append(patch)
        lines.append("=" * 78)
    else:
        lines.append(
            "(No two-sided recommendations — drop-in patch omitted. "
            "Authentic-side numbers are above for manual review.)"
        )
        lines.append("=" * 78)

    return "\n".join(lines)


# --------------------------------------------------------------------------
# Markdown
# --------------------------------------------------------------------------


def render_markdown(
    recommendations: dict[str, ThresholdRecommendation],
    *,
    load_stats: LoadStats | None = None,
) -> str:
    """Markdown report — for PR comments / running notes."""
    lines: list[str] = []
    lines.append("# Counterfeit-detector threshold recalibration")
    lines.append("")
    lines.append(f"_Generated: {datetime.now(timezone.utc).isoformat()}_")
    if load_stats is not None:
        lines.append("")
        lines.append(
            f"**Corpus:** {load_stats.loaded}/{load_stats.requested} samples "
            f"loaded "
            f"(missing files: {load_stats.skipped_missing_file}, "
            f"unreadable: {load_stats.skipped_unreadable})"
        )
    lines.append("")

    lines.append("## Recommendations")
    lines.append("")
    lines.append(
        "| detector | mode | n_a | n_c | abst | "
        "current auth | rec auth | Δ | "
        "current cf | rec cf | Δ |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for det in _detector_order(recommendations):
        r = recommendations[det]
        lines.append(
            f"| `{r.detector}` | {r.mode} | "
            f"{r.n_authentic_confident} | {r.n_counterfeit_confident} | {r.n_abstained} | "
            f"{r.current_authentic_threshold:.4f} | "
            f"{_fmt_threshold(r.recommended_authentic_threshold).strip()} | "
            f"{_fmt_delta(r.current_authentic_threshold, r.recommended_authentic_threshold).strip()} | "
            f"{r.current_counterfeit_threshold:.4f} | "
            f"{_fmt_threshold(r.recommended_counterfeit_threshold).strip()} | "
            f"{_fmt_delta(r.current_counterfeit_threshold, r.recommended_counterfeit_threshold).strip()} |"
        )
    lines.append("")

    lines.append("## Achieved metrics at recommended thresholds")
    lines.append("")
    lines.append(
        "| detector | authentic recall | counterfeit recall | authentic FPR |"
    )
    lines.append("|---|---:|---:|---:|")
    for det in _detector_order(recommendations):
        r = recommendations[det]
        lines.append(
            f"| `{r.detector}` | "
            f"{_fmt_pct(r.achieved_authentic_recall).strip()} | "
            f"{_fmt_pct(r.achieved_counterfeit_recall).strip()} | "
            f"{_fmt_pct(r.achieved_authentic_fpr).strip()} |"
        )
    lines.append("")

    # Score distributions — useful for eyeballing
    lines.append("## Score distributions")
    lines.append("")
    for det in _detector_order(recommendations):
        r = recommendations[det]
        lines.append(f"### `{det}`")
        if r.score_distribution_authentic:
            d = r.score_distribution_authentic
            lines.append(
                f"- **Authentic** (n={d['n']}): "
                f"min={d['min']:.3f}, p5={d['p5']:.3f}, p25={d['p25']:.3f}, "
                f"p50={d['p50']:.3f}, p75={d['p75']:.3f}, p95={d['p95']:.3f}, "
                f"max={d['max']:.3f}, mean={d['mean']:.3f}±{d['stddev']:.3f}"
            )
        if r.score_distribution_counterfeit:
            d = r.score_distribution_counterfeit
            lines.append(
                f"- **Counterfeit** (n={d['n']}): "
                f"min={d['min']:.3f}, p5={d['p5']:.3f}, p25={d['p25']:.3f}, "
                f"p50={d['p50']:.3f}, p75={d['p75']:.3f}, p95={d['p95']:.3f}, "
                f"max={d['max']:.3f}, mean={d['mean']:.3f}±{d['stddev']:.3f}"
            )
        for note in r.notes:
            lines.append(f"- _Note:_ {note}")
        lines.append("")

    patch = _render_patch_block(recommendations)
    if patch:
        lines.append("## Drop-in patch for `ml/pipelines/counterfeit/ensemble.py`")
        lines.append("")
        lines.append("```python")
        lines.append(patch)
        lines.append("```")

    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# JSON
# --------------------------------------------------------------------------


def to_json_dict(
    recommendations: dict[str, ThresholdRecommendation],
    *,
    load_stats: LoadStats | None = None,
) -> dict[str, Any]:
    """Structured payload for CI / longitudinal tracking."""
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "load_stats": asdict(load_stats) if load_stats is not None else None,
        "recommendations": {det: asdict(r) for det, r in recommendations.items()},
    }


# --------------------------------------------------------------------------
# Drop-in patch
# --------------------------------------------------------------------------


def _render_patch_block(
    recommendations: dict[str, ThresholdRecommendation],
) -> str:
    """Emit a Python snippet replacing the relevant ensemble.py constants
    for any TWO_SIDED detector. Lines for AUTHENTIC_ONLY / INSUFFICIENT
    detectors are intentionally omitted — half-apply'd thresholds are
    a footgun.
    """
    parts: list[str] = []
    parts.append(
        f"# Recalibrated {datetime.now(timezone.utc).isoformat()} via "
        "evaluation.counterfeit_recalibration"
    )
    any_emitted = False
    for det in _detector_order(recommendations):
        r = recommendations[det]
        if r.mode != MODE_TWO_SIDED:
            continue
        if det not in _PATCH_CONSTANT_NAMES:
            continue
        a_name, c_name = _PATCH_CONSTANT_NAMES[det]
        a_val = r.recommended_authentic_threshold
        c_val = r.recommended_counterfeit_threshold
        if a_val is None or c_val is None:
            continue
        parts.append(f"{a_name}: float = {a_val:.4f}")
        parts.append(f"{c_name}: float = {c_val:.4f}")
        any_emitted = True

    return "\n".join(parts) if any_emitted else ""


__all__ = [
    "render_console",
    "render_markdown",
    "to_json_dict",
]
