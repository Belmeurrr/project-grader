"""Threshold-sweep math for the counterfeit-detector recalibration tool.

Pure functions; no I/O. Consumes `SampleResult`s from the existing
benchmark runner so the scores fed to the sweep are exactly the ones
production sees (same detectors, same `confidence` gate). Recalibration
recommendations are derived from those scores under two modes:

  - **two_sided**: the corpus has both authentic AND counterfeit
    samples. Both thresholds are recommended.
  - **authentic_only**: the corpus has only authentic samples (the
    expected first-step state once the PSA accumulator has been running
    a while — every PSA-graded card is an authentic by definition, but
    counterfeit samples have to be sourced separately). Only the
    AUTHENTIC threshold is recommended; the COUNTERFEIT threshold is
    left as `None` and the caller surfaces a "needs counterfeit data"
    note.

Why these two modes rather than one universal sweep:
  Recalibrating the AUTHENTIC threshold needs only authentic data
  (we're picking the score below which we're willing to lose authentics
  to false-flag). Recalibrating the COUNTERFEIT threshold needs
  authentic data (to bound the false-alarm rate of the strong-counterfeit
  signal) and is much more useful with counterfeit data (to confirm
  counterfeits actually score below it). Splitting the modes lets the
  tool give half-the-answer when half the data is available, instead of
  refusing.

The detector contract:
  Each detector returns score ∈ [0, 1] where higher = more authentic.
  AUTHENTIC threshold: score ≥ a → AUTHENTIC.
  COUNTERFEIT threshold: score < c → LIKELY_COUNTERFEIT.
  (a > c by convention; equal would collapse the SUSPICIOUS band.)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from pipelines.counterfeit import ensemble

from evaluation.counterfeit_benchmark.corpus import (
    GROUND_TRUTH_AUTHENTIC,
    GROUND_TRUTH_COUNTERFEIT,
)
from evaluation.counterfeit_benchmark.runner import SampleResult


MODE_TWO_SIDED = "two_sided"
MODE_AUTHENTIC_ONLY = "authentic_only"
MODE_INSUFFICIENT = "insufficient"


# Per-detector current thresholds, indexed by the detector identifier
# the runner emits. Kept here (not in ensemble.py) so the recalibration
# tool's "current vs recommended" diff has a single lookup table.
CURRENT_THRESHOLDS: dict[str, tuple[float, float]] = {
    "rosette": (
        ensemble.ROSETTE_AUTHENTIC_THRESHOLD,
        ensemble.ROSETTE_COUNTERFEIT_THRESHOLD,
    ),
    "color": (
        ensemble.COLOR_AUTHENTIC_THRESHOLD,
        ensemble.COLOR_COUNTERFEIT_THRESHOLD,
    ),
    "embedding_anomaly": (
        ensemble.EMBEDDING_AUTHENTIC_THRESHOLD,
        ensemble.EMBEDDING_COUNTERFEIT_THRESHOLD,
    ),
    "typography": (
        ensemble.TYPOGRAPHY_AUTHENTIC_THRESHOLD,
        ensemble.TYPOGRAPHY_COUNTERFEIT_THRESHOLD,
    ),
    "holographic": (
        ensemble.HOLOGRAPHIC_AUTHENTIC_THRESHOLD,
        ensemble.HOLOGRAPHIC_COUNTERFEIT_THRESHOLD,
    ),
}


# Detectors the recalibration tool can recommend for. The ensemble row
# from the benchmark runner is excluded — it has no scalar score to
# threshold against (it's a verdict combiner).
RECALIBRATABLE_DETECTORS = frozenset(CURRENT_THRESHOLDS.keys())


@dataclass(frozen=True)
class DetectorScores:
    """Confident scores for one detector, split by ground-truth label.

    A score is "confident" if the detector's verdict on that sample was
    NOT UNVERIFIED — i.e., the detector's confidence cleared its
    own MIN_CONFIDENCE gate. UNVERIFIED outcomes are not included in
    either array (they're counted in `n_abstained`)."""

    detector: str
    authentic_scores: NDArray[np.float64]  # shape (n_a,)
    counterfeit_scores: NDArray[np.float64]  # shape (n_c,)
    n_abstained: int


@dataclass(frozen=True)
class ThresholdRecommendation:
    """Recommended thresholds for one detector + supporting context.

    `mode` declares which thresholds are filled in:
      - MODE_TWO_SIDED: both thresholds are populated; achieved metrics
        reference the labeled corpus.
      - MODE_AUTHENTIC_ONLY: only `recommended_authentic_threshold` is
        populated; counterfeit threshold is None and has a note.
      - MODE_INSUFFICIENT: too little data for either; both None and
        the notes list explains.

    `current_*` carry the in-source thresholds at recalibration time so
    the rendered diff is unambiguous when the source-of-truth changes.
    `score_distribution_*` carry quartile-style summaries — useful in
    the report for human eyeballing of "is the recommendation
    reasonable?".
    """

    detector: str
    mode: str

    n_authentic_confident: int
    n_counterfeit_confident: int
    n_abstained: int

    recommended_authentic_threshold: float | None
    recommended_counterfeit_threshold: float | None

    achieved_authentic_recall: float | None
    achieved_counterfeit_recall: float | None
    achieved_authentic_fpr: float | None

    current_authentic_threshold: float
    current_counterfeit_threshold: float

    score_distribution_authentic: dict[str, float] | None
    score_distribution_counterfeit: dict[str, float] | None

    notes: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------
# Score collection
# --------------------------------------------------------------------------


def collect_scores(results: Sequence[SampleResult]) -> dict[str, DetectorScores]:
    """Group the benchmark runner's outputs by detector, splitting
    confident scores by ground-truth label.

    Scores from samples whose detector verdict was UNVERIFIED, or whose
    score is None (detector errored), are excluded — they carry no
    threshold-relevant signal. The abstained count is preserved per
    detector so the report can show "we lost N samples to abstention"
    alongside the recommendations.
    """
    by_det: dict[str, dict[str, list[float]]] = {}
    abstained: dict[str, int] = {}

    for r in results:
        for o in r.outcomes:
            if o.detector not in RECALIBRATABLE_DETECTORS:
                continue
            slot = by_det.setdefault(
                o.detector, {GROUND_TRUTH_AUTHENTIC: [], GROUND_TRUTH_COUNTERFEIT: []}
            )
            abstained.setdefault(o.detector, 0)
            if o.verdict == ensemble.VERDICT_UNVERIFIED or o.score is None:
                abstained[o.detector] += 1
                continue
            label = r.sample.ground_truth
            if label not in slot:
                # Unknown label — ignore. (This can happen if a caller
                # extends the corpus with custom labels; the tool stays
                # focused on the binary task.)
                continue
            slot[label].append(float(o.score))

    out: dict[str, DetectorScores] = {}
    for det, splits in by_det.items():
        out[det] = DetectorScores(
            detector=det,
            authentic_scores=np.asarray(splits[GROUND_TRUTH_AUTHENTIC], dtype=np.float64),
            counterfeit_scores=np.asarray(
                splits[GROUND_TRUTH_COUNTERFEIT], dtype=np.float64
            ),
            n_abstained=abstained.get(det, 0),
        )
    return out


# --------------------------------------------------------------------------
# Distribution summary
# --------------------------------------------------------------------------


def _summarize_distribution(scores: NDArray[np.float64]) -> dict[str, float] | None:
    """min / p5 / p25 / median / p75 / p95 / max. None on empty input."""
    if scores.size == 0:
        return None
    return {
        "n": int(scores.size),
        "min": float(scores.min()),
        "p5": float(np.percentile(scores, 5)),
        "p25": float(np.percentile(scores, 25)),
        "p50": float(np.percentile(scores, 50)),
        "p75": float(np.percentile(scores, 75)),
        "p95": float(np.percentile(scores, 95)),
        "max": float(scores.max()),
        "mean": float(scores.mean()),
        "stddev": float(scores.std(ddof=0)),
    }


# --------------------------------------------------------------------------
# Threshold pickers
# --------------------------------------------------------------------------


def _candidate_thresholds(
    authentic_scores: NDArray[np.float64],
    counterfeit_scores: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Threshold candidates for the sweep — midpoints of consecutive
    sorted unique scores, plus bookends slightly outside the observed
    range so "all authentic" and "all counterfeit" extremes are
    explored.

    Why midpoints (not the scores themselves): with the strict-less-than
    convention (`s < a` → predicted-counterfeit), placing `a` exactly
    at an observed score makes the prediction at that score depend on
    floating-point equality. Midpoints sit cleanly *between*
    observations and exercise the same TPR/FPR plateau without the
    boundary fragility. They also let the sweep see the full J-plateau
    in bimodal data — observed-score-only candidates skip across the
    gap between clusters and miss the optimum.
    """
    pts = np.unique(np.concatenate([authentic_scores, counterfeit_scores]))
    if pts.size == 0:
        return np.array([0.5])
    if pts.size == 1:
        # Degenerate — one score across the whole corpus. Threshold can't
        # do anything useful; return a single neutral bookend.
        return np.array([float(pts[0])])
    midpoints = (pts[:-1] + pts[1:]) / 2.0
    bookend_lo = max(0.0, float(pts[0]) - 1e-3)
    bookend_hi = min(1.0, float(pts[-1]) + 1e-3)
    return np.concatenate([[bookend_lo], midpoints, [bookend_hi]])


def _pick_authentic_threshold_two_sided(
    authentic_scores: NDArray[np.float64],
    counterfeit_scores: NDArray[np.float64],
) -> float:
    """Maximize Youden's J (TPR_counterfeit − FPR_authentic) over the
    midpoint-candidate set.

    For each candidate `a`:
      TPR = #counterfeits with score < a / n_counterfeits  (caught)
      FPR = #authentics  with score < a / n_authentics     (lost)
      J   = TPR − FPR

    When multiple candidates tie at max J (the ROC-optimal "plateau"),
    the **median of the plateau** is returned. In bimodal data, this
    naturally lands midway between max(counterfeit) and min(authentic),
    which is the maximally robust choice — picking either edge would
    misclassify the next near-boundary sample.
    """
    candidates = _candidate_thresholds(authentic_scores, counterfeit_scores)
    n_a = float(authentic_scores.size)
    n_c = float(counterfeit_scores.size)

    j_values = np.empty(candidates.shape, dtype=np.float64)
    for i, a in enumerate(candidates):
        tpr = float((counterfeit_scores < a).sum()) / n_c
        fpr = float((authentic_scores < a).sum()) / n_a
        j_values[i] = tpr - fpr

    max_j = float(j_values.max())
    plateau = candidates[j_values >= max_j - 1e-12]
    return float(np.median(plateau))


def _pick_authentic_threshold_one_sided(
    authentic_scores: NDArray[np.float64],
    target_recall: float,
) -> float:
    """Pick threshold = (1-target_recall) percentile of authentic scores.

    `target_recall = 0.95` => threshold = 5th percentile. By
    construction, target_recall fraction of confident authentics will
    score >= this threshold.
    """
    pct = max(0.0, min(100.0, (1.0 - target_recall) * 100.0))
    return float(np.percentile(authentic_scores, pct))


def _pick_counterfeit_threshold(
    authentic_scores: NDArray[np.float64],
    counterfeit_scores: NDArray[np.float64],
    *,
    fpr_budget: float,
    authentic_threshold: float,
) -> float:
    """Pick `c` for the LIKELY_COUNTERFEIT threshold under three caps:

      (1) FPR-budget cap: at most `fpr_budget` fraction of authentics
          may fall below `c`. A single detector firing LIKELY_COUNTERFEIT
          triggers the ensemble to fail; we bound that false-alarm
          rate hard.
      (2) Counterfeit-territory cap: `c ≤ max(counterfeit_scores)`. We
          shouldn't fire on scores higher than ANY counterfeit in the
          corpus — that's not "definitely counterfeit territory."
      (3) Suspicious-band cap: `c ≤ authentic_threshold − 0.01`. The
          SUSPICIOUS band exists for borderline cases; collapsing it
          (c = a) breaks the human-review path.

    The recommendation is the MIN of the three caps (most conservative —
    only fire when score is genuinely low).
    """
    n_a = authentic_scores.size
    if n_a == 0:
        return 0.0

    # (1) FPR-budget cap
    sorted_a = np.sort(authentic_scores)
    allowed_below = int(np.floor(fpr_budget * n_a))
    if allowed_below == 0:
        # Even one authentic below would exceed budget — stay below the
        # smallest authentic.
        c_fpr = max(0.0, float(sorted_a[0]) - 1e-3)
    elif allowed_below >= n_a:
        c_fpr = float(sorted_a[-1])
    else:
        # Place c just below sorted_a[allowed_below] so exactly
        # `allowed_below` authentics fall strictly below.
        c_fpr = float(
            (sorted_a[allowed_below - 1] + sorted_a[allowed_below]) / 2.0
        )

    # (2) Counterfeit-territory cap. If no counterfeits in corpus,
    # leave this cap unconstrained (caller should be in authentic-only
    # mode, which doesn't call this function — defensive default).
    c_cf = float(counterfeit_scores.max()) if counterfeit_scores.size else 1.0

    # (3) Suspicious-band cap
    c_band = max(0.0, authentic_threshold - 0.01)

    return max(0.0, min(c_fpr, c_cf, c_band))


# --------------------------------------------------------------------------
# Public sweep entrypoint
# --------------------------------------------------------------------------


def recommend_thresholds(
    scores: DetectorScores,
    *,
    target_authentic_recall: float = 0.95,
    counterfeit_fpr_budget: float = 0.005,
    current_authentic_threshold: float | None = None,
    current_counterfeit_threshold: float | None = None,
) -> ThresholdRecommendation:
    """Recommend (authentic, counterfeit) thresholds for one detector.

    Args:
        scores: confident-scores split as produced by `collect_scores`.
        target_authentic_recall: fraction of authentics we commit to
            keeping above the AUTHENTIC threshold. Used in the
            authentic-only mode and as a sanity check in two-sided mode.
        counterfeit_fpr_budget: maximum fraction of confident authentics
            we'll allow to fall below the COUNTERFEIT threshold (i.e.,
            be flagged LIKELY_COUNTERFEIT by this detector alone).
        current_*_threshold: the in-source values, for the report's
            current-vs-recommended diff. Defaults to the values in
            `CURRENT_THRESHOLDS[detector]` when omitted.

    Returns:
        A `ThresholdRecommendation`. When the corpus is too small,
        `mode == MODE_INSUFFICIENT` and both recommended thresholds
        are None; the caller surfaces the notes.
    """
    n_a = int(scores.authentic_scores.size)
    n_c = int(scores.counterfeit_scores.size)
    notes: list[str] = []

    cur_a, cur_c = CURRENT_THRESHOLDS.get(scores.detector, (0.65, 0.35))
    if current_authentic_threshold is None:
        current_authentic_threshold = cur_a
    if current_counterfeit_threshold is None:
        current_counterfeit_threshold = cur_c

    dist_a = _summarize_distribution(scores.authentic_scores)
    dist_c = _summarize_distribution(scores.counterfeit_scores)

    # Hard floor: no confident authentics → can't even pick the
    # authentic-side threshold. Bail out with insufficient mode.
    if n_a == 0:
        notes.append(
            "no confident authentic samples — cannot recommend any thresholds. "
            "Either supply authentic samples in the corpus, or relax detector "
            "MIN_CONFIDENCE gates (the latter has knock-on effects, prefer the "
            "former)."
        )
        return ThresholdRecommendation(
            detector=scores.detector,
            mode=MODE_INSUFFICIENT,
            n_authentic_confident=n_a,
            n_counterfeit_confident=n_c,
            n_abstained=scores.n_abstained,
            recommended_authentic_threshold=None,
            recommended_counterfeit_threshold=None,
            achieved_authentic_recall=None,
            achieved_counterfeit_recall=None,
            achieved_authentic_fpr=None,
            current_authentic_threshold=current_authentic_threshold,
            current_counterfeit_threshold=current_counterfeit_threshold,
            score_distribution_authentic=dist_a,
            score_distribution_counterfeit=dist_c,
            notes=notes,
        )

    # Sample-size warnings — recommendations from <30 samples are
    # statistically wobbly. We still emit them but flag the noise.
    if n_a < 30:
        notes.append(
            f"only {n_a} confident authentic samples — recommendation is "
            "noisy; aim for ≥ 30 before locking the threshold."
        )

    if n_c == 0:
        # ----------------- AUTHENTIC-ONLY MODE -----------------
        a = _pick_authentic_threshold_one_sided(
            scores.authentic_scores, target_authentic_recall
        )
        recall_a = float((scores.authentic_scores >= a).sum()) / float(n_a)
        notes.append(
            "no counterfeit samples in corpus → only AUTHENTIC threshold "
            "recalibrated. COUNTERFEIT threshold left at current value; "
            "needs counterfeit data to recalibrate."
        )
        return ThresholdRecommendation(
            detector=scores.detector,
            mode=MODE_AUTHENTIC_ONLY,
            n_authentic_confident=n_a,
            n_counterfeit_confident=0,
            n_abstained=scores.n_abstained,
            recommended_authentic_threshold=a,
            recommended_counterfeit_threshold=None,
            achieved_authentic_recall=recall_a,
            achieved_counterfeit_recall=None,
            achieved_authentic_fpr=None,
            current_authentic_threshold=current_authentic_threshold,
            current_counterfeit_threshold=current_counterfeit_threshold,
            score_distribution_authentic=dist_a,
            score_distribution_counterfeit=None,
            notes=notes,
        )

    if n_c < 10:
        notes.append(
            f"only {n_c} confident counterfeit samples — counterfeit-side "
            "recommendation is highly noisy; aim for ≥ 30."
        )

    # ----------------- TWO-SIDED MODE -----------------
    a = _pick_authentic_threshold_two_sided(
        scores.authentic_scores, scores.counterfeit_scores
    )
    c = _pick_counterfeit_threshold(
        scores.authentic_scores,
        scores.counterfeit_scores,
        fpr_budget=counterfeit_fpr_budget,
        authentic_threshold=a,
    )

    recall_a = float((scores.authentic_scores >= a).sum()) / float(n_a)
    recall_c = float((scores.counterfeit_scores < a).sum()) / float(n_c)
    fpr_a = float((scores.authentic_scores < a).sum()) / float(n_a)

    return ThresholdRecommendation(
        detector=scores.detector,
        mode=MODE_TWO_SIDED,
        n_authentic_confident=n_a,
        n_counterfeit_confident=n_c,
        n_abstained=scores.n_abstained,
        recommended_authentic_threshold=a,
        recommended_counterfeit_threshold=c,
        achieved_authentic_recall=recall_a,
        achieved_counterfeit_recall=recall_c,
        achieved_authentic_fpr=fpr_a,
        current_authentic_threshold=current_authentic_threshold,
        current_counterfeit_threshold=current_counterfeit_threshold,
        score_distribution_authentic=dist_a,
        score_distribution_counterfeit=dist_c,
        notes=notes,
    )


def recommend_all(
    by_detector: dict[str, DetectorScores],
    *,
    target_authentic_recall: float = 0.95,
    counterfeit_fpr_budget: float = 0.005,
) -> dict[str, ThresholdRecommendation]:
    """Run `recommend_thresholds` for every detector in the dict."""
    return {
        det: recommend_thresholds(
            scores,
            target_authentic_recall=target_authentic_recall,
            counterfeit_fpr_budget=counterfeit_fpr_budget,
        )
        for det, scores in by_detector.items()
    }


__all__ = [
    "MODE_TWO_SIDED",
    "MODE_AUTHENTIC_ONLY",
    "MODE_INSUFFICIENT",
    "CURRENT_THRESHOLDS",
    "RECALIBRATABLE_DETECTORS",
    "DetectorScores",
    "ThresholdRecommendation",
    "collect_scores",
    "recommend_thresholds",
    "recommend_all",
]
