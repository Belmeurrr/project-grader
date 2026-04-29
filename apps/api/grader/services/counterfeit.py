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

from pipelines.counterfeit.color import (  # noqa: E402
    ColorProfileMeasurement,
    measure_color_profile,
)
from pipelines.counterfeit.rosette import (  # noqa: E402
    RosetteMeasurement,
    measure_rosette,
)
from pipelines.counterfeit import ensemble  # noqa: E402

# Re-export thresholds + verdict logic from the ml-side ensemble module
# so existing imports (tests, etc.) keep working without reaching into
# pipelines.counterfeit.ensemble. Canonical home is the ml side; this
# is the persistence-layer's local view.
ROSETTE_AUTHENTIC_THRESHOLD = ensemble.ROSETTE_AUTHENTIC_THRESHOLD
ROSETTE_COUNTERFEIT_THRESHOLD = ensemble.ROSETTE_COUNTERFEIT_THRESHOLD
ROSETTE_MIN_CONFIDENCE = ensemble.ROSETTE_MIN_CONFIDENCE
ROSETTE_MODEL_VERSION = ensemble.ROSETTE_MODEL_VERSION
COLOR_AUTHENTIC_THRESHOLD = ensemble.COLOR_AUTHENTIC_THRESHOLD
COLOR_COUNTERFEIT_THRESHOLD = ensemble.COLOR_COUNTERFEIT_THRESHOLD
COLOR_MIN_CONFIDENCE = ensemble.COLOR_MIN_CONFIDENCE
COLOR_MODEL_VERSION = ensemble.COLOR_MODEL_VERSION


class CounterfeitFailedError(Exception):
    """Raised when the counterfeit-analysis pipeline cannot complete.

    Distinct from a "looks-counterfeit" result — that's a low rosette_score,
    not an exception. This error means the image couldn't be loaded /
    decoded / analyzed at all (bad bytes, missing key, image too small)."""


def _load_canonical_bgr(s3_key: str) -> np.ndarray:
    """Service-typed wrapper around storage.load_canonical_bgr.

    Catches the storage-level CanonicalLoadError and re-raises as a
    CounterfeitFailedError so the worker can route load failures the
    same as detector failures."""
    try:
        return storage.load_canonical_bgr(s3_key)
    except storage.CanonicalLoadError as e:
        raise CounterfeitFailedError(str(e)) from e


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


def analyze_color_profile(canonical_s3_key: str) -> ColorProfileMeasurement:
    """Run the CIELAB color-profile counterfeit detector on a canonical image.

    Same shape as `analyze_rosette` so the pipeline runner can call both
    in sequence and combine their measurements via the ensemble logic.
    Authentic offset prints reach high CIELAB chroma in saturated art
    areas; consumer-printer counterfeits clip at lower chroma. Border-
    sampled white-balance calibration removes lighting cast as a
    confounder.

    Args:
        canonical_s3_key: S3 key of the dewarped 750x1050 BGR canonical.

    Returns:
        ColorProfileMeasurement with a score in [0, 1] (higher = more
        likely authentic) and a calibration-quality confidence in [0, 1].

    Raises:
        CounterfeitFailedError: if the canonical can't be loaded/decoded
            or the image fails the detector's input-validation gate.
    """
    image = _load_canonical_bgr(canonical_s3_key)
    try:
        return measure_color_profile(image)
    except ValueError as e:
        raise CounterfeitFailedError(
            f"color-profile analysis failed for {canonical_s3_key}: {e}"
        ) from e


def _verdict_from_rosette(m: RosetteMeasurement) -> AuthenticityVerdict:
    """Translate the ml-side string verdict into the SQLAlchemy enum.

    The decision logic + thresholds live in ml/pipelines/counterfeit/
    ensemble.py (single home, shared with the benchmark). This wrapper
    just maps "authentic" → AuthenticityVerdict.AUTHENTIC etc."""
    return AuthenticityVerdict(ensemble.verdict_from_rosette(m))


def _verdict_from_color_profile(m: ColorProfileMeasurement) -> AuthenticityVerdict:
    """Same shape as `_verdict_from_rosette` against the color-detector
    thresholds. Decision logic in ml/pipelines/counterfeit/ensemble.py."""
    return AuthenticityVerdict(ensemble.verdict_from_color_profile(m))


def _reasons_from_color_profile(
    m: ColorProfileMeasurement, verdict: AuthenticityVerdict
) -> list[str]:
    reasons: list[str] = []
    if verdict == AuthenticityVerdict.UNVERIFIED:
        reasons.append(
            f"unreliable white-balance calibration "
            f"(border_stddev={m.border_stddev:.1f}, confidence={m.confidence:.2f})"
        )
        return reasons
    if verdict == AuthenticityVerdict.AUTHENTIC:
        reasons.append(
            f"high chroma consistent with offset print "
            f"(color_score={m.color_score:.2f}, p95_chroma={m.p95_chroma:.1f})"
        )
    elif verdict == AuthenticityVerdict.LIKELY_COUNTERFEIT:
        reasons.append(
            f"low chroma — gamut clipping consistent with consumer printer "
            f"(color_score={m.color_score:.2f}, p95_chroma={m.p95_chroma:.1f})"
        )
    else:  # SUSPICIOUS
        reasons.append(
            f"borderline chroma — neither clearly offset nor inkjet "
            f"(color_score={m.color_score:.2f}, p95_chroma={m.p95_chroma:.1f})"
        )
    return reasons


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


def _combine_verdicts(verdicts: list[AuthenticityVerdict]) -> AuthenticityVerdict:
    """Conservative ensemble combiner. Decision logic in
    ml/pipelines/counterfeit/ensemble.py (shared with the benchmark);
    this wrapper handles the SQLAlchemy enum boundary."""
    return AuthenticityVerdict(
        ensemble.combine_verdicts(v.value for v in verdicts)
    )


async def persist_authenticity_result(
    submission_id: uuid.UUID,
    *,
    rosette: RosetteMeasurement,
    color: ColorProfileMeasurement,
    db: AsyncSession,
) -> AuthenticityResult:
    """Insert or update the AuthenticityResult row for a submission.

    A submission has at most one authenticity row (unique index on
    submission_id), so re-runs update in place rather than colliding on
    the unique constraint.

    Per-detector verdicts are computed by `_verdict_from_rosette` and
    `_verdict_from_color_profile`; the combined row-level verdict comes
    from `_combine_verdicts`. The combined `confidence` field is the
    MIN of per-detector confidences — a single high-confidence detector
    can flag a fake even if others abstain, but the row's confidence
    only reflects "confident overall" when both detectors are confident.

    The detector_scores blob carries each detector's raw outputs
    verbatim so a future re-calibration can recompute verdicts from
    history without re-running the math."""
    rosette_verdict = _verdict_from_rosette(rosette)
    color_verdict = _verdict_from_color_profile(color)
    verdict = _combine_verdicts([rosette_verdict, color_verdict])

    detector_scores = {
        "rosette": {
            "score": float(rosette.rosette_score),
            "peak_strength": float(rosette.peak_strength),
            "analyzed_patches": int(rosette.analyzed_patches),
            "confidence": float(rosette.confidence),
            "manufacturer_profile": rosette.manufacturer_profile,
            "verdict": rosette_verdict.value,
        },
        "color": {
            "score": float(color.color_score),
            "p95_chroma": float(color.p95_chroma),
            "border_white_bgr": list(color.border_white_bgr),
            "border_stddev": float(color.border_stddev),
            "gain_applied": list(color.gain_applied),
            "confidence": float(color.confidence),
            "manufacturer_profile": color.manufacturer_profile,
            "verdict": color_verdict.value,
        },
    }
    reasons: list[str] = []
    reasons.extend(_reasons_from_rosette(rosette, rosette_verdict))
    reasons.extend(_reasons_from_color_profile(color, color_verdict))

    model_versions = {
        "rosette": ROSETTE_MODEL_VERSION,
        "color": COLOR_MODEL_VERSION,
        "thresholds": {
            "rosette_authentic": ROSETTE_AUTHENTIC_THRESHOLD,
            "rosette_counterfeit": ROSETTE_COUNTERFEIT_THRESHOLD,
            "rosette_min_confidence": ROSETTE_MIN_CONFIDENCE,
            "color_authentic": COLOR_AUTHENTIC_THRESHOLD,
            "color_counterfeit": COLOR_COUNTERFEIT_THRESHOLD,
            "color_min_confidence": COLOR_MIN_CONFIDENCE,
        },
    }

    # Combined confidence: min across detectors. The row's "confidence"
    # is what a downstream UI shows as "we're sure"; that should only
    # be high when both detectors agree they have signal.
    combined_confidence = float(min(rosette.confidence, color.confidence))

    existing = await db.scalar(
        select(AuthenticityResult).where(
            AuthenticityResult.submission_id == submission_id
        )
    )
    if existing is None:
        row = AuthenticityResult(
            submission_id=submission_id,
            verdict=verdict,
            confidence=combined_confidence,
            detector_scores=detector_scores,
            reasons=reasons,
            model_versions=model_versions,
        )
        db.add(row)
        await db.flush()
        return row

    existing.verdict = verdict
    existing.confidence = combined_confidence
    existing.detector_scores = detector_scores
    existing.reasons = reasons
    existing.model_versions = model_versions
    await db.flush()
    return existing
