"""Counterfeit-detection ensemble logic — shared between apps/api and
the ml/evaluation benchmark.

The thresholds + verdict-combining logic live here (not inside
apps/api/grader/services/counterfeit.py) so the benchmark can run the
exact same decision boundaries as production. Otherwise the benchmark
drifts from what callers actually see and stops being useful.

Verdict values are STRINGS, not an enum, so this module has zero
dependency on the apps/api SQLAlchemy enum class. apps/api wraps these
strings into its own AuthenticityVerdict enum at the persistence
boundary; the benchmark consumes the strings directly.

Thresholds are calibrated against synthetic fixtures (the
ml/tests/fixtures synth_* generators); recalibration against real PSA
images is tracked under "Counterfeit confidence calibration" in TODO.
"""

from __future__ import annotations

from typing import Iterable

from pipelines.counterfeit.color import ColorProfileMeasurement
from pipelines.counterfeit.embedding_anomaly import EmbeddingAnomalyMeasurement
from pipelines.counterfeit.rosette import RosetteMeasurement


# String values for verdicts. Match the apps/api SQLAlchemy enum's
# `.value` exactly so the apps/api wrapper can `AuthenticityVerdict(s)`
# on the string and round-trip cleanly.
VERDICT_AUTHENTIC = "authentic"
VERDICT_SUSPICIOUS = "suspicious"
VERDICT_LIKELY_COUNTERFEIT = "likely_counterfeit"
VERDICT_UNVERIFIED = "unverified"

ALL_VERDICTS = frozenset(
    {
        VERDICT_AUTHENTIC,
        VERDICT_SUSPICIOUS,
        VERDICT_LIKELY_COUNTERFEIT,
        VERDICT_UNVERIFIED,
    }
)


# --------------------------------------------------------------------------
# Per-detector thresholds (canonical home — apps/api re-imports these)
# --------------------------------------------------------------------------

# Rosette FFT detector. The score is a logistic of the per-patch FFT-peak
# prominence (see ml/pipelines/counterfeit/rosette). Continuous-tone
# counterfeits cluster around 0.05-0.15; clean offset-printed authentics
# cluster around 0.95+. The middle band is genuine uncertainty.
ROSETTE_AUTHENTIC_THRESHOLD: float = 0.65
ROSETTE_COUNTERFEIT_THRESHOLD: float = 0.35
ROSETTE_MIN_CONFIDENCE: float = 0.4
ROSETTE_MODEL_VERSION: str = "fft-v1"


# Color-profile detector. The score is a logistic of the inner-area p95
# CIELAB chroma after white-balance calibration. Saturated authentic
# prints score near 1.0; desaturated inkjet copies cluster around 0.05-0.20.
COLOR_AUTHENTIC_THRESHOLD: float = 0.65
COLOR_COUNTERFEIT_THRESHOLD: float = 0.20
COLOR_MIN_CONFIDENCE: float = 0.4
COLOR_MODEL_VERSION: str = "cielab-chroma-v1"


# Embedding-anomaly detector. Score is a logistic of the cosine distance
# between the submitted embedding and the centroid of the reference
# embeddings for the same variant (lower distance → higher score). The
# detector only has signal when at least one reference is on file —
# anything below `_MIN_CONFIDENCE` (which corresponds to having very few
# refs, see `_CONFIDENCE_BY_NREFS` in measure.py) is UNVERIFIED.
EMBEDDING_AUTHENTIC_THRESHOLD: float = 0.65
EMBEDDING_COUNTERFEIT_THRESHOLD: float = 0.35
EMBEDDING_MIN_CONFIDENCE: float = 0.4
EMBEDDING_MODEL_VERSION: str = "centroid-cosine-v1"


# --------------------------------------------------------------------------
# Per-detector verdict mappers
# --------------------------------------------------------------------------


def verdict_from_rosette(m: RosetteMeasurement) -> str:
    """Tri-state verdict for the rosette detector.

    UNVERIFIED when fewer than the required number of flat patches
    were found — the FFT signal is unreliable. Above the AUTHENTIC
    threshold → AUTHENTIC. Below the COUNTERFEIT threshold →
    LIKELY_COUNTERFEIT. Between → SUSPICIOUS."""
    if m.confidence < ROSETTE_MIN_CONFIDENCE:
        return VERDICT_UNVERIFIED
    if m.rosette_score >= ROSETTE_AUTHENTIC_THRESHOLD:
        return VERDICT_AUTHENTIC
    if m.rosette_score < ROSETTE_COUNTERFEIT_THRESHOLD:
        return VERDICT_LIKELY_COUNTERFEIT
    return VERDICT_SUSPICIOUS


def verdict_from_color_profile(m: ColorProfileMeasurement) -> str:
    """Tri-state verdict for the color-profile detector.

    UNVERIFIED when the white-balance calibration was unreliable
    (border too tinted, too noisy, or too dark) — the chroma signal
    can't be trusted. Otherwise applies the threshold band."""
    if m.confidence < COLOR_MIN_CONFIDENCE:
        return VERDICT_UNVERIFIED
    if m.color_score >= COLOR_AUTHENTIC_THRESHOLD:
        return VERDICT_AUTHENTIC
    if m.color_score < COLOR_COUNTERFEIT_THRESHOLD:
        return VERDICT_LIKELY_COUNTERFEIT
    return VERDICT_SUSPICIOUS


def verdict_from_embedding_anomaly(m: EmbeddingAnomalyMeasurement) -> str:
    """Tri-state verdict for the embedding-anomaly detector.

    UNVERIFIED when no references were available (n_references == 0,
    which forces confidence to 0) or when the available references are
    too few to trust the centroid distance. The signal scales with how
    many authentic exemplars we have for the variant; the detector
    abstains for variants we haven't seen at all."""
    if m.confidence < EMBEDDING_MIN_CONFIDENCE:
        return VERDICT_UNVERIFIED
    if m.embedding_score >= EMBEDDING_AUTHENTIC_THRESHOLD:
        return VERDICT_AUTHENTIC
    if m.embedding_score < EMBEDDING_COUNTERFEIT_THRESHOLD:
        return VERDICT_LIKELY_COUNTERFEIT
    return VERDICT_SUSPICIOUS


# --------------------------------------------------------------------------
# Ensemble combiner
# --------------------------------------------------------------------------


def combine_verdicts(verdicts: Iterable[str]) -> str:
    """Conservative ensemble combiner.

    Ranking, applied in order:
      1. ANY detector with sufficient confidence flags LIKELY_COUNTERFEIT
         → combined LIKELY_COUNTERFEIT. Counterfeit detection is high-
         precision-on-true-positives by design; we'd rather flag a real
         card for human review than pass a fake.
      2. Else, if every confident detector says AUTHENTIC → AUTHENTIC
         (consensus required for an authentic verdict).
      3. Else, if any confident detector said SUSPICIOUS → SUSPICIOUS.
      4. Else (all detectors abstained / no detectors ran) → UNVERIFIED.

    Same policy as apps/api used to implement inline; consolidated here
    so the benchmark and production stay aligned.
    """
    verdicts_list = list(verdicts)
    if not verdicts_list:
        return VERDICT_UNVERIFIED
    for v in verdicts_list:
        if v not in ALL_VERDICTS:
            raise ValueError(f"unknown verdict {v!r}; expected one of {ALL_VERDICTS}")
    if any(v == VERDICT_LIKELY_COUNTERFEIT for v in verdicts_list):
        return VERDICT_LIKELY_COUNTERFEIT
    confident = [v for v in verdicts_list if v != VERDICT_UNVERIFIED]
    if not confident:
        return VERDICT_UNVERIFIED
    if all(v == VERDICT_AUTHENTIC for v in confident):
        return VERDICT_AUTHENTIC
    return VERDICT_SUSPICIOUS


__all__ = [
    # Verdict string constants
    "VERDICT_AUTHENTIC",
    "VERDICT_SUSPICIOUS",
    "VERDICT_LIKELY_COUNTERFEIT",
    "VERDICT_UNVERIFIED",
    "ALL_VERDICTS",
    # Per-detector thresholds (canonical home)
    "ROSETTE_AUTHENTIC_THRESHOLD",
    "ROSETTE_COUNTERFEIT_THRESHOLD",
    "ROSETTE_MIN_CONFIDENCE",
    "ROSETTE_MODEL_VERSION",
    "COLOR_AUTHENTIC_THRESHOLD",
    "COLOR_COUNTERFEIT_THRESHOLD",
    "COLOR_MIN_CONFIDENCE",
    "COLOR_MODEL_VERSION",
    "EMBEDDING_AUTHENTIC_THRESHOLD",
    "EMBEDDING_COUNTERFEIT_THRESHOLD",
    "EMBEDDING_MIN_CONFIDENCE",
    "EMBEDDING_MODEL_VERSION",
    # Logic
    "verdict_from_rosette",
    "verdict_from_color_profile",
    "verdict_from_embedding_anomaly",
    "combine_verdicts",
]
