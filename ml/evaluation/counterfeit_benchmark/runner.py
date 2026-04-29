"""Benchmark runner — applies the rosette + color detectors to each
sample in a corpus, returning the full result matrix.

Pure compute (no I/O); the corpus is held in memory and the pure-
function detectors run synchronously. ~50 samples × 2 detectors = 100
detector calls; on a modern laptop the run completes in 1-2 seconds.
The metric computation lives in `report.py` so multiple report formats
(console, JSON, Markdown) can consume the same raw data.

Embedding-anomaly detector is intentionally not in this run — it
needs per-variant authentic reference embeddings that don't yet exist
in the corpus, and it isn't wired into production either. Adding it
to the benchmark would test something production doesn't actually do.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Sequence

from pipelines.counterfeit import ensemble
from pipelines.counterfeit.color import (
    ColorProfileMeasurement,
    measure_color_profile,
)
from pipelines.counterfeit.rosette import (
    RosetteMeasurement,
    measure_rosette,
)

from evaluation.counterfeit_benchmark.corpus import BenchmarkSample


# Detector identifiers, matching the keys used in production's
# AuthenticityResult.detector_scores blob.
DETECTOR_ROSETTE = "rosette"
DETECTOR_COLOR = "color"
DETECTOR_ENSEMBLE = "ensemble"


@dataclass(frozen=True)
class DetectorOutcome:
    """One detector's view of one sample."""

    detector: str  # one of DETECTOR_*
    score: float | None  # None when the detector abstained / not applicable
    confidence: float
    verdict: str  # "authentic" | "suspicious" | "likely_counterfeit" | "unverified"
    raw: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass(frozen=True)
class SampleResult:
    """All detector outcomes for one labeled sample, plus elapsed time."""

    sample: BenchmarkSample
    outcomes: list[DetectorOutcome]
    elapsed_seconds: float


def _outcome_for_rosette(measurement: RosetteMeasurement) -> DetectorOutcome:
    return DetectorOutcome(
        detector=DETECTOR_ROSETTE,
        score=float(measurement.rosette_score),
        confidence=float(measurement.confidence),
        verdict=ensemble.verdict_from_rosette(measurement),
        raw={
            "peak_strength": float(measurement.peak_strength),
            "analyzed_patches": int(measurement.analyzed_patches),
        },
    )


def _outcome_for_color(measurement: ColorProfileMeasurement) -> DetectorOutcome:
    return DetectorOutcome(
        detector=DETECTOR_COLOR,
        score=float(measurement.color_score),
        confidence=float(measurement.confidence),
        verdict=ensemble.verdict_from_color_profile(measurement),
        raw={
            "p95_chroma": float(measurement.p95_chroma),
            "border_stddev": float(measurement.border_stddev),
        },
    )


def _outcome_for_error(detector: str, e: Exception) -> DetectorOutcome:
    return DetectorOutcome(
        detector=detector,
        score=None,
        confidence=0.0,
        verdict=ensemble.VERDICT_UNVERIFIED,
        error=f"{type(e).__name__}: {e}",
    )


def _run_one(sample: BenchmarkSample) -> SampleResult:
    """Run all detectors against one sample. Each detector runs in its
    own try/except so a crash in one doesn't kill the whole run; the
    failed detector lands as an UNVERIFIED outcome with the error
    string captured."""
    t0 = time.monotonic()
    outcomes: list[DetectorOutcome] = []

    # Rosette
    try:
        rosette_m = measure_rosette(sample.image)
        outcomes.append(_outcome_for_rosette(rosette_m))
    except Exception as e:  # noqa: BLE001 — benchmark should not abort on detector crashes
        outcomes.append(_outcome_for_error(DETECTOR_ROSETTE, e))

    # Color profile
    try:
        color_m = measure_color_profile(sample.image)
        outcomes.append(_outcome_for_color(color_m))
    except Exception as e:  # noqa: BLE001
        outcomes.append(_outcome_for_error(DETECTOR_COLOR, e))

    # Combined ensemble verdict — derives from the per-detector
    # verdicts above using the same logic production uses.
    combined_verdict = ensemble.combine_verdicts(o.verdict for o in outcomes)
    # Combined confidence = min across detector confidences (mirrors
    # apps/api's persist_authenticity_result).
    combined_confidence = (
        min(o.confidence for o in outcomes) if outcomes else 0.0
    )
    outcomes.append(
        DetectorOutcome(
            detector=DETECTOR_ENSEMBLE,
            score=None,  # ensemble has no scalar score; verdict is the output
            confidence=combined_confidence,
            verdict=combined_verdict,
            raw={"per_detector_verdicts": [o.verdict for o in outcomes]},
        )
    )

    return SampleResult(
        sample=sample,
        outcomes=outcomes,
        elapsed_seconds=time.monotonic() - t0,
    )


def run_benchmark(corpus: Sequence[BenchmarkSample]) -> list[SampleResult]:
    """Run all detectors against every sample. Returns the result
    matrix; metric computation is in `report.py`."""
    return [_run_one(s) for s in corpus]


__all__ = [
    "DETECTOR_ROSETTE",
    "DETECTOR_COLOR",
    "DETECTOR_ENSEMBLE",
    "DetectorOutcome",
    "SampleResult",
    "run_benchmark",
]
