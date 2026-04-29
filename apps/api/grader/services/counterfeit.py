"""Counterfeit-detection service.

Wraps the ML rosette detector behind an S3-aware API so the pipeline
worker (Celery) can analyze a canonical card image straight from object
storage without having to know about cv2 / numpy.

This is the first detector in the planned 7-detector counterfeit-
authenticity ensemble (see docs/plans). v1 only implements the print-
rosette FFT detector — additional detectors (color profile, typography,
embedding anomaly, etc.) will land as siblings in this module.

Persistence lives in `persist_authenticity_result`: turns a raw rosette
measurement into a verdict and writes/updates the AuthenticityResult row
keyed by submission_id. The orchestrator (`grader.workers.pipeline_runner`)
calls `analyze_rosette` then `persist_authenticity_result`."""

from __future__ import annotations

import sys
import uuid
from pathlib import Path

import cv2
import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from grader.db.models import AuthenticityResult, AuthenticityVerdict
from grader.services import storage

_ML_ROOT = Path(__file__).resolve().parents[4] / "ml"
if str(_ML_ROOT) not in sys.path:
    sys.path.insert(0, str(_ML_ROOT))

from pipelines.counterfeit.rosette import (  # noqa: E402
    RosetteMeasurement,
    measure_rosette,
)


# Verdict thresholds for the rosette detector. The score is a logistic of
# the per-patch FFT-peak prominence (see ml/pipelines/counterfeit/rosette).
# Continuous-tone counterfeits cluster around 0.05-0.15; clean offset-printed
# authentics cluster around 0.95+. The middle band is genuine uncertainty —
# we don't fabricate a verdict there.
#
# These v1 thresholds are calibrated against synthetic halftone fixtures.
# Real cards may shift the operating point; recalibration is tracked under
# "Counterfeit confidence calibration thresholds" in TODO.md.
ROSETTE_AUTHENTIC_THRESHOLD: float = 0.65
ROSETTE_COUNTERFEIT_THRESHOLD: float = 0.35

# Detector confidence (= analyzed_patches / requested) below which we won't
# stake a verdict — too few flat patches were found to trust the FFT signal.
ROSETTE_MIN_CONFIDENCE: float = 0.4

ROSETTE_MODEL_VERSION: str = "fft-v1"


class CounterfeitFailedError(Exception):
    """Raised when the counterfeit-analysis pipeline cannot complete.

    Distinct from a "looks-counterfeit" result — that's a low rosette_score,
    not an exception. This error means the image couldn't be loaded /
    decoded / analyzed at all (bad bytes, missing key, image too small)."""


def _load_canonical_bgr(s3_key: str) -> np.ndarray:
    """Load and decode a canonical card image from S3.

    Duplicated from grader.services.grading._load_canonical_bgr to avoid
    cross-service coupling — both modules want the same thing but neither
    should be a transitive dependency of the other. If a third service
    needs this, the function will be moved to grader.services.storage.
    Until then, the duplication is cheap."""
    raw = storage.get_shot_bytes(s3_key)
    arr = np.frombuffer(raw, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise CounterfeitFailedError(f"could not decode canonical at {s3_key}")
    return image


def analyze_rosette(canonical_s3_key: str) -> RosetteMeasurement:
    """Run the print-rosette FFT counterfeit detector on a canonical image.

    Args:
        canonical_s3_key: S3 key of the dewarped 750x1050 BGR canonical
            image produced by Stage 2 (perspective correction).

    Returns:
        RosetteMeasurement with a score in [0, 1] (higher = more likely
        authentic) and an analyzed-patch confidence in [0, 1].

    Raises:
        CounterfeitFailedError: if the canonical can't be loaded/decoded
            or the image fails the detector's input-validation gate.
    """
    image = _load_canonical_bgr(canonical_s3_key)
    try:
        return measure_rosette(image)
    except ValueError as e:
        # measure_rosette raises ValueError for shape/dtype/size issues. We
        # surface those as service-level failures so the worker can route
        # them the same as load errors.
        raise CounterfeitFailedError(
            f"rosette analysis failed for {canonical_s3_key}: {e}"
        ) from e


def _verdict_from_rosette(m: RosetteMeasurement) -> AuthenticityVerdict:
    if m.confidence < ROSETTE_MIN_CONFIDENCE:
        return AuthenticityVerdict.UNVERIFIED
    if m.rosette_score >= ROSETTE_AUTHENTIC_THRESHOLD:
        return AuthenticityVerdict.AUTHENTIC
    if m.rosette_score < ROSETTE_COUNTERFEIT_THRESHOLD:
        return AuthenticityVerdict.LIKELY_COUNTERFEIT
    return AuthenticityVerdict.SUSPICIOUS


def _reasons_from_rosette(
    m: RosetteMeasurement, verdict: AuthenticityVerdict
) -> list[str]:
    reasons: list[str] = []
    if verdict == AuthenticityVerdict.UNVERIFIED:
        reasons.append(
            f"insufficient flat regions for FFT analysis "
            f"(analyzed_patches={m.analyzed_patches}, confidence={m.confidence:.2f})"
        )
        return reasons
    if verdict == AuthenticityVerdict.AUTHENTIC:
        reasons.append(
            f"halftone rosette pattern detected "
            f"(rosette_score={m.rosette_score:.2f}, peak_strength={m.peak_strength:.2f})"
        )
    elif verdict == AuthenticityVerdict.LIKELY_COUNTERFEIT:
        reasons.append(
            f"no halftone rosette pattern in expected band "
            f"(rosette_score={m.rosette_score:.2f}, peak_strength={m.peak_strength:.2f})"
        )
    else:  # SUSPICIOUS
        reasons.append(
            f"weak rosette signal — borderline halftone evidence "
            f"(rosette_score={m.rosette_score:.2f}, peak_strength={m.peak_strength:.2f})"
        )
    return reasons


async def persist_authenticity_result(
    submission_id: uuid.UUID,
    result: RosetteMeasurement,
    db: AsyncSession,
) -> AuthenticityResult:
    """Insert or update the AuthenticityResult row for a submission.

    A submission has at most one authenticity row (unique index on
    submission_id), so re-runs update in place rather than colliding on
    the unique constraint. The verdict is derived from the rosette score
    and detector confidence per the ROSETTE_* thresholds defined above.

    The detector_scores blob carries the raw rosette outputs verbatim so
    a future re-calibration can recompute verdicts from history without
    re-running the FFT."""
    verdict = _verdict_from_rosette(result)
    detector_scores = {
        "rosette": {
            "score": float(result.rosette_score),
            "peak_strength": float(result.peak_strength),
            "analyzed_patches": int(result.analyzed_patches),
            "confidence": float(result.confidence),
            "manufacturer_profile": result.manufacturer_profile,
        }
    }
    reasons = _reasons_from_rosette(result, verdict)
    model_versions = {
        "rosette": ROSETTE_MODEL_VERSION,
        "thresholds": {
            "authentic": ROSETTE_AUTHENTIC_THRESHOLD,
            "counterfeit": ROSETTE_COUNTERFEIT_THRESHOLD,
            "min_confidence": ROSETTE_MIN_CONFIDENCE,
        },
    }

    existing = await db.scalar(
        select(AuthenticityResult).where(
            AuthenticityResult.submission_id == submission_id
        )
    )
    if existing is None:
        row = AuthenticityResult(
            submission_id=submission_id,
            verdict=verdict,
            confidence=float(result.confidence),
            detector_scores=detector_scores,
            reasons=reasons,
            model_versions=model_versions,
        )
        db.add(row)
        await db.flush()
        return row

    existing.verdict = verdict
    existing.confidence = float(result.confidence)
    existing.detector_scores = detector_scores
    existing.reasons = reasons
    existing.model_versions = model_versions
    await db.flush()
    return existing
