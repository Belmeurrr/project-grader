"""Tests for the counterfeit-detector recalibration tool.

Three layers:
  - Pure-function sweep tests (synthetic score arrays, no images).
    Cover the three modes (two_sided / authentic_only / insufficient),
    the FPR budget, and degenerate corpora.
  - Corpus-loader tests (CSV + PSA jsonl). Cover happy paths, missing
    files, bad labels, max_records.
  - End-to-end smoke: the synthetic counterfeit_benchmark corpus
    passes through the full pipeline and produces a sensible
    recommendation that still picks the current thresholds (or close
    enough on a 50-sample synthetic corpus).
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.counterfeit_benchmark import (  # noqa: E402
    build_default_corpus,
    run_benchmark,
)
from evaluation.counterfeit_benchmark.corpus import (  # noqa: E402
    GROUND_TRUTH_AUTHENTIC,
    GROUND_TRUTH_COUNTERFEIT,
)
from evaluation.counterfeit_recalibration import (  # noqa: E402
    LabeledRow,
    MODE_AUTHENTIC_ONLY,
    MODE_INSUFFICIENT,
    MODE_TWO_SIDED,
    ThresholdRecommendation,
    collect_scores,
    load_csv,
    load_psa_authentics,
    recommend_all,
    recommend_thresholds,
    render_console,
    render_markdown,
    rows_to_samples,
    to_json_dict,
)
from evaluation.counterfeit_recalibration.sweep import DetectorScores  # noqa: E402


# --------------------------------------------------------------------------
# Pure sweep — two-sided
# --------------------------------------------------------------------------


def _scores(detector: str, authentic: list[float], counterfeit: list[float], abst: int = 0) -> DetectorScores:
    return DetectorScores(
        detector=detector,
        authentic_scores=np.asarray(authentic, dtype=np.float64),
        counterfeit_scores=np.asarray(counterfeit, dtype=np.float64),
        n_abstained=abst,
    )


def test_two_sided_bimodal_picks_threshold_between_clusters() -> None:
    """Authentics clustered at 0.9, counterfeits at 0.1 → recommended
    AUTHENTIC threshold falls strictly between the clusters."""
    rng = np.random.default_rng(0)
    authentic = list(rng.normal(0.9, 0.02, size=100).clip(0, 1))
    counterfeit = list(rng.normal(0.1, 0.02, size=100).clip(0, 1))
    rec = recommend_thresholds(_scores("rosette", authentic, counterfeit))

    assert rec.mode == MODE_TWO_SIDED
    assert rec.recommended_authentic_threshold is not None
    assert rec.recommended_counterfeit_threshold is not None
    # Threshold falls in the gap between the modes.
    assert 0.2 < rec.recommended_authentic_threshold < 0.85
    # Counterfeit threshold is strictly below authentic.
    assert rec.recommended_counterfeit_threshold < rec.recommended_authentic_threshold
    # Achieved metrics — bimodal data should give near-perfect separation.
    assert rec.achieved_authentic_recall is not None and rec.achieved_authentic_recall >= 0.95
    assert rec.achieved_counterfeit_recall is not None and rec.achieved_counterfeit_recall >= 0.95
    assert rec.achieved_authentic_fpr is not None and rec.achieved_authentic_fpr <= 0.05


def test_two_sided_overlapping_distributions_still_returns_thresholds() -> None:
    """Heavily overlapping clusters → tool still emits thresholds (the
    Youden's-J pick is well-defined even when separability is low),
    but the user-facing achieved recall numbers reflect the overlap."""
    rng = np.random.default_rng(1)
    authentic = list(rng.normal(0.55, 0.10, size=100).clip(0, 1))
    counterfeit = list(rng.normal(0.45, 0.10, size=100).clip(0, 1))
    rec = recommend_thresholds(_scores("color", authentic, counterfeit))

    assert rec.mode == MODE_TWO_SIDED
    assert rec.recommended_authentic_threshold is not None
    # Achieved metrics are bounded by separability — they will not be 0.95.
    assert rec.achieved_authentic_fpr is not None and rec.achieved_authentic_fpr < 0.6


# --------------------------------------------------------------------------
# Pure sweep — authentic-only
# --------------------------------------------------------------------------


def test_authentic_only_recommends_authentic_side_at_target_percentile() -> None:
    """target_authentic_recall=0.95 → threshold is the 5th percentile
    of authentic scores. About 5% of authentics fall strictly below."""
    rng = np.random.default_rng(2)
    authentic = list(rng.normal(0.85, 0.05, size=200).clip(0, 1))
    rec = recommend_thresholds(_scores("rosette", authentic, []), target_authentic_recall=0.95)

    assert rec.mode == MODE_AUTHENTIC_ONLY
    assert rec.recommended_authentic_threshold is not None
    assert rec.recommended_counterfeit_threshold is None
    # Recall should be approximately the target (>=0.94 with 200 samples).
    assert rec.achieved_authentic_recall is not None
    assert rec.achieved_authentic_recall >= 0.94
    # Sanity: threshold is roughly at p5 of the input distribution.
    p5 = float(np.percentile(np.asarray(authentic), 5))
    assert abs(rec.recommended_authentic_threshold - p5) < 1e-9
    # Note about counterfeit data is surfaced.
    assert any("counterfeit" in n.lower() for n in rec.notes)


def test_authentic_only_target_recall_99_picks_lower_threshold() -> None:
    """target_authentic_recall=0.99 → threshold is p1, lower than p5."""
    rng = np.random.default_rng(3)
    authentic = list(rng.normal(0.85, 0.05, size=500).clip(0, 1))
    rec_95 = recommend_thresholds(_scores("rosette", authentic, []), target_authentic_recall=0.95)
    rec_99 = recommend_thresholds(_scores("rosette", authentic, []), target_authentic_recall=0.99)
    assert rec_99.recommended_authentic_threshold < rec_95.recommended_authentic_threshold


# --------------------------------------------------------------------------
# Pure sweep — insufficient + edge
# --------------------------------------------------------------------------


def test_insufficient_when_no_authentic_samples() -> None:
    rec = recommend_thresholds(_scores("color", [], [0.1, 0.2, 0.3]))
    assert rec.mode == MODE_INSUFFICIENT
    assert rec.recommended_authentic_threshold is None
    assert rec.recommended_counterfeit_threshold is None
    assert any("authentic" in n.lower() for n in rec.notes)


def test_small_sample_warning_in_notes() -> None:
    """Fewer than 30 confident authentics → a warning note is emitted
    so the human knows the recommendation is wobbly."""
    rec = recommend_thresholds(_scores("rosette", [0.9] * 5, [0.1] * 5))
    assert rec.mode == MODE_TWO_SIDED
    # Has both small-sample notes (n_a < 30 AND n_c < 10).
    assert any("noisy" in n.lower() for n in rec.notes)


def test_counterfeit_threshold_clipped_when_above_authentic() -> None:
    """If the corpus is so tangled that the recommended `c` ≥ `a`, we
    clip and surface a note. Avoids producing a degenerate (a, c) pair
    that collapses the SUSPICIOUS band."""
    # Authentics clustered LOW, counterfeits clustered HIGH — pathological.
    rng = np.random.default_rng(4)
    authentic = list(rng.normal(0.2, 0.05, size=50).clip(0, 1))
    counterfeit = list(rng.normal(0.8, 0.05, size=50).clip(0, 1))
    rec = recommend_thresholds(_scores("rosette", authentic, counterfeit))
    assert rec.mode == MODE_TWO_SIDED
    assert rec.recommended_authentic_threshold is not None
    assert rec.recommended_counterfeit_threshold is not None
    assert rec.recommended_counterfeit_threshold < rec.recommended_authentic_threshold


def test_fpr_budget_respected_in_counterfeit_threshold() -> None:
    """The COUNTERFEIT threshold must keep authentic-FPR ≤ budget."""
    rng = np.random.default_rng(5)
    authentic = list(rng.normal(0.85, 0.05, size=1000).clip(0, 1))
    counterfeit = list(rng.normal(0.1, 0.05, size=200).clip(0, 1))
    rec = recommend_thresholds(
        _scores("rosette", authentic, counterfeit),
        counterfeit_fpr_budget=0.005,
    )
    assert rec.mode == MODE_TWO_SIDED
    cf = rec.recommended_counterfeit_threshold
    assert cf is not None
    a_arr = np.asarray(authentic)
    actual_fpr = float((a_arr < cf).sum()) / float(a_arr.size)
    assert actual_fpr <= 0.005 + 1e-9


# --------------------------------------------------------------------------
# CSV loader
# --------------------------------------------------------------------------


def _make_image_file(p: Path, *, color: tuple[int, int, int] = (0, 0, 255)) -> None:
    """A 32x32 PNG at the given path. Tiny but cv2-loadable."""
    img = np.full((32, 32, 3), color, dtype=np.uint8)
    p.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    p.write_bytes(buf.tobytes())


def test_load_csv_happy_path(tmp_path: Path) -> None:
    img1 = tmp_path / "a.png"
    img2 = tmp_path / "b.png"
    _make_image_file(img1)
    _make_image_file(img2)
    csv = tmp_path / "labels.csv"
    csv.write_text(
        "image_path,ground_truth,sample_id\n"
        f"{img1},authentic,first\n"
        f"{img2},counterfeit,second\n",
        encoding="utf-8",
    )
    rows = load_csv(csv)
    assert len(rows) == 2
    assert rows[0].ground_truth == GROUND_TRUTH_AUTHENTIC
    assert rows[0].sample_id == "first"
    assert rows[1].ground_truth == GROUND_TRUTH_COUNTERFEIT
    assert rows[1].sample_id == "second"


def test_load_csv_missing_required_column(tmp_path: Path) -> None:
    csv = tmp_path / "bad.csv"
    csv.write_text("image_path\nfoo.png\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing required"):
        load_csv(csv)


def test_load_csv_bad_label(tmp_path: Path) -> None:
    csv = tmp_path / "bad.csv"
    csv.write_text(
        "image_path,ground_truth\nfoo.png,real\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="ground_truth"):
        load_csv(csv)


def test_load_csv_default_sample_id_is_filename_stem(tmp_path: Path) -> None:
    csv = tmp_path / "labels.csv"
    csv.write_text(
        "image_path,ground_truth\n"
        "/abs/path/foo.png,authentic\n",
        encoding="utf-8",
    )
    rows = load_csv(csv)
    assert rows[0].sample_id == "foo"


def test_load_csv_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_csv(tmp_path / "doesnt_exist.csv")


# --------------------------------------------------------------------------
# PSA jsonl loader
# --------------------------------------------------------------------------


def test_load_psa_authentics_happy_path(tmp_path: Path) -> None:
    img = tmp_path / "img.png"
    _make_image_file(img)
    jsonl = tmp_path / "scraped.jsonl"
    jsonl.write_text(
        "\n".join(
            json.dumps({"cert_id": cid, "front_image_path": str(img), "grade": 9.0})
            for cid in (1, 2, 3)
        )
        + "\n",
        encoding="utf-8",
    )
    rows = load_psa_authentics(jsonl)
    assert len(rows) == 3
    assert all(r.ground_truth == GROUND_TRUTH_AUTHENTIC for r in rows)
    assert rows[0].sample_id == "psa-1"


def test_load_psa_authentics_skips_rows_without_front_image(tmp_path: Path) -> None:
    jsonl = tmp_path / "scraped.jsonl"
    jsonl.write_text(
        json.dumps({"cert_id": 1, "front_image_path": None, "grade": 9.0}) + "\n"
        + json.dumps({"cert_id": 2, "front_image_path": "", "grade": 8.0}) + "\n"
        + json.dumps({"cert_id": 3, "front_image_path": "/some/path.jpg", "grade": 7.0}) + "\n",
        encoding="utf-8",
    )
    rows = load_psa_authentics(jsonl)
    assert len(rows) == 1
    assert rows[0].sample_id == "psa-3"


def test_load_psa_authentics_max_records(tmp_path: Path) -> None:
    jsonl = tmp_path / "scraped.jsonl"
    jsonl.write_text(
        "\n".join(
            json.dumps({"cert_id": i, "front_image_path": "/x.jpg", "grade": 9.0})
            for i in range(10)
        ),
        encoding="utf-8",
    )
    rows = load_psa_authentics(jsonl, max_records=4)
    assert len(rows) == 4


def test_load_psa_authentics_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_psa_authentics(tmp_path / "no.jsonl")


# --------------------------------------------------------------------------
# rows_to_samples
# --------------------------------------------------------------------------


def test_rows_to_samples_counts_missing_and_unreadable(tmp_path: Path) -> None:
    img = tmp_path / "good.png"
    _make_image_file(img)
    bad = tmp_path / "broken.png"
    bad.write_bytes(b"not an image at all")
    missing = tmp_path / "doesnt_exist.png"

    rows = [
        LabeledRow(image_path=img, ground_truth=GROUND_TRUTH_AUTHENTIC, sample_id="ok"),
        LabeledRow(image_path=bad, ground_truth=GROUND_TRUTH_COUNTERFEIT, sample_id="bad"),
        LabeledRow(image_path=missing, ground_truth=GROUND_TRUTH_AUTHENTIC, sample_id="gone"),
    ]
    samples, stats = rows_to_samples(rows)
    assert stats.requested == 3
    assert stats.loaded == 1
    assert stats.skipped_missing_file == 1
    assert stats.skipped_unreadable == 1
    assert len(samples) == 1
    assert samples[0].sample_id == "ok"


# --------------------------------------------------------------------------
# End-to-end smoke against the synthetic benchmark corpus
# --------------------------------------------------------------------------


def test_end_to_end_default_corpus_produces_sensible_recommendations() -> None:
    """The synthetic 50-sample benchmark corpus should pass cleanly
    through every layer and produce TWO_SIDED recommendations for both
    rosette + color, with achieved authentic-recall + counterfeit-recall
    above 0.8 (the synthetic corpus is well-separated)."""
    samples = list(build_default_corpus())
    results = run_benchmark(samples)
    by_det = collect_scores(results)
    recs = recommend_all(by_det)

    assert "rosette" in recs
    assert "color" in recs
    for det in ("rosette", "color"):
        r = recs[det]
        assert r.mode == MODE_TWO_SIDED, f"{det}: mode={r.mode}, notes={r.notes}"
        assert r.recommended_authentic_threshold is not None
        assert r.recommended_counterfeit_threshold is not None
        # On a well-separated synthetic corpus, achieved recall should be high.
        assert (r.achieved_authentic_recall or 0) >= 0.8
        # The recommended counterfeit threshold should keep authentic FPR
        # at or below the default budget (0.5%).
        assert (r.achieved_authentic_fpr or 0) <= 0.05


def test_end_to_end_renderers_produce_nonempty_output() -> None:
    samples = list(build_default_corpus())
    results = run_benchmark(samples)
    by_det = collect_scores(results)
    recs = recommend_all(by_det)

    console = render_console(recs)
    assert "Counterfeit-detector threshold recalibration" in console
    assert "rosette" in console
    assert "color" in console
    # Drop-in patch should appear (TWO_SIDED detectors present).
    assert "ROSETTE_AUTHENTIC_THRESHOLD" in console
    assert "COLOR_AUTHENTIC_THRESHOLD" in console

    md = render_markdown(recs)
    assert "# Counterfeit-detector threshold recalibration" in md
    assert "Drop-in patch" in md

    payload = to_json_dict(recs)
    assert payload["schema_version"] == 1
    assert "rosette" in payload["recommendations"]
    assert "color" in payload["recommendations"]


def test_end_to_end_authentic_only_corpus_skips_patch_for_one_sided() -> None:
    """When the corpus has only authentics (e.g. PSA shortcut), the
    drop-in patch block stays empty for affected detectors — we don't
    half-apply thresholds."""
    samples = [s for s in build_default_corpus() if s.ground_truth == GROUND_TRUTH_AUTHENTIC]
    results = run_benchmark(samples)
    by_det = collect_scores(results)
    recs = recommend_all(by_det)

    for det, r in recs.items():
        # Only authentics in the corpus → either AUTHENTIC_ONLY or, if
        # all samples got abstained, INSUFFICIENT.
        assert r.mode in (MODE_AUTHENTIC_ONLY, MODE_INSUFFICIENT), f"{det}: {r.mode}"

    console = render_console(recs)
    # Patch block omitted (no TWO_SIDED detectors).
    assert "ROSETTE_AUTHENTIC_THRESHOLD" not in console
    assert "drop-in patch omitted" in console.lower()
