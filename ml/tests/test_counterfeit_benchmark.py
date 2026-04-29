"""Counterfeit-detector benchmark smoke tests.

The benchmark is a measurement tool, not a CI gate, so these tests
verify SHAPE rather than score thresholds:
  - corpus has the expected size + ground-truth balance
  - runner produces an outcome per detector per sample
  - metrics dataclass has all expected fields populated
  - JSON/Markdown/console renderers produce non-empty output
  - per-detector confusion matrices sum to the correct totals

There ARE two soft "the detectors aren't broken" assertions:
  - ensemble row's true_positive_rate > 0.5 (the ensemble must actually
    detect SOME of the synthetic counterfeits we built it to detect)
  - ensemble row's false_positive_rate < 0.5 (it must not mis-flag
    most of the synthetic authentics)
These are loose enough to survive seed jitter; they tighten as the
corpus grows.
"""

from __future__ import annotations

import json

import pytest

from evaluation.counterfeit_benchmark import (
    GROUND_TRUTH_AUTHENTIC,
    GROUND_TRUTH_COUNTERFEIT,
    build_default_corpus,
    compute_metrics,
    render_console,
    render_markdown,
    run_benchmark,
    to_json_dict,
)
from evaluation.counterfeit_benchmark.runner import (
    DETECTOR_COLOR,
    DETECTOR_ENSEMBLE,
    DETECTOR_ROSETTE,
)


# -----------------------------
# Corpus shape
# -----------------------------


def test_default_corpus_has_balanced_size() -> None:
    corpus = build_default_corpus()
    assert len(corpus) == 50, f"corpus changed size: got {len(corpus)} (expected 50)"
    n_auth = sum(1 for s in corpus if s.ground_truth == GROUND_TRUTH_AUTHENTIC)
    n_fake = sum(1 for s in corpus if s.ground_truth == GROUND_TRUTH_COUNTERFEIT)
    assert n_auth == 25
    assert n_fake == 25


def test_corpus_sample_ids_unique() -> None:
    corpus = build_default_corpus()
    ids = [s.sample_id for s in corpus]
    assert len(set(ids)) == len(ids), "duplicate sample_id in corpus"


def test_corpus_images_are_canonical_size() -> None:
    """Every sample's image must have the same shape so the detectors
    can be applied uniformly without per-sample shape branches."""
    corpus = build_default_corpus()
    shapes = {tuple(s.image.shape) for s in corpus}
    assert len(shapes) == 1, f"corpus has multiple image shapes: {shapes}"


# -----------------------------
# Runner contract
# -----------------------------


@pytest.fixture(scope="module")
def benchmark_results():
    """Run the full benchmark once; share results across tests since
    the run takes 1-2 seconds and the result data is read-only."""
    return run_benchmark(build_default_corpus())


def test_runner_produces_three_outcomes_per_sample(benchmark_results) -> None:
    """rosette + color + ensemble = 3 outcomes per sample."""
    for r in benchmark_results:
        assert len(r.outcomes) == 3, (
            f"sample {r.sample.sample_id}: expected 3 outcomes, got {len(r.outcomes)}"
        )
        detectors = {o.detector for o in r.outcomes}
        assert detectors == {DETECTOR_ROSETTE, DETECTOR_COLOR, DETECTOR_ENSEMBLE}


def test_runner_records_elapsed_time(benchmark_results) -> None:
    for r in benchmark_results:
        assert r.elapsed_seconds > 0
        # Loose upper bound — a single sample shouldn't take more
        # than 10 seconds; tighter bounds are flaky on shared CI.
        assert r.elapsed_seconds < 10.0


def test_runner_assigns_valid_verdicts(benchmark_results) -> None:
    """Every outcome's verdict must be one of the four canonical values."""
    valid = {"authentic", "suspicious", "likely_counterfeit", "unverified"}
    for r in benchmark_results:
        for o in r.outcomes:
            assert o.verdict in valid


# -----------------------------
# Metrics
# -----------------------------


def test_metrics_keyed_by_detector(benchmark_results) -> None:
    metrics = compute_metrics(benchmark_results)
    assert set(metrics.keys()) == {
        DETECTOR_ROSETTE,
        DETECTOR_COLOR,
        DETECTOR_ENSEMBLE,
    }


def test_metrics_confusion_matrix_consistent(benchmark_results) -> None:
    """tp + fp + tn + fn + n_abstained == n_samples for every detector."""
    metrics = compute_metrics(benchmark_results)
    for m in metrics.values():
        total = (
            m.true_positives
            + m.false_positives
            + m.true_negatives
            + m.false_negatives
            + m.n_abstained
        )
        assert total == m.n_samples, (
            f"{m.detector}: tp+fp+tn+fn+abst={total} != n_samples={m.n_samples}"
        )


def test_ensemble_detects_some_counterfeits(benchmark_results) -> None:
    """Soft sanity: the ensemble must catch SOME of the synthetic
    counterfeits we deliberately built to be catchable. Loose
    threshold (>50%) survives seed jitter; tightens as corpus grows."""
    metrics = compute_metrics(benchmark_results)
    tpr = metrics[DETECTOR_ENSEMBLE].true_positive_rate
    assert tpr is not None and tpr > 0.5, (
        f"ensemble TPR collapsed to {tpr} — synthetic counterfeits should be catchable"
    )


def test_ensemble_doesnt_misflag_most_authentics(benchmark_results) -> None:
    """Soft sanity: the ensemble must not mis-flag most of the
    synthetic authentics. Loose threshold (<50%)."""
    metrics = compute_metrics(benchmark_results)
    fpr = metrics[DETECTOR_ENSEMBLE].false_positive_rate
    assert fpr is not None and fpr < 0.5, (
        f"ensemble FPR ballooned to {fpr} — synthetic authentics should mostly pass"
    )


# -----------------------------
# Renderers
# -----------------------------


def test_render_console_contains_each_detector(benchmark_results) -> None:
    metrics = compute_metrics(benchmark_results)
    out = render_console(metrics)
    for d in (DETECTOR_ROSETTE, DETECTOR_COLOR, DETECTOR_ENSEMBLE):
        assert d in out


def test_render_markdown_contains_table_header(benchmark_results) -> None:
    metrics = compute_metrics(benchmark_results)
    out = render_markdown(metrics, benchmark_results)
    assert "# Counterfeit-detector benchmark" in out
    assert "| detector |" in out
    assert "TPR" in out
    assert "FPR" in out


def test_to_json_dict_is_serializable(benchmark_results) -> None:
    metrics = compute_metrics(benchmark_results)
    payload = to_json_dict(metrics, benchmark_results)
    # Round-trip via JSON to catch any non-serializable types
    encoded = json.dumps(payload)
    decoded = json.loads(encoded)
    assert decoded["schema_version"] == 1
    assert decoded["corpus_size"] == 50
    assert "metrics" in decoded
    assert "samples" in decoded
    assert len(decoded["samples"]) == 50
