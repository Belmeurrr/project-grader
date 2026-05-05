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

import uuid
from pathlib import Path

import cv2
import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from data.ingestion.reference_embeddings import lookup_references
from grader.db.models import AuthenticityResult, AuthenticityVerdict
from grader.services import storage
from pipelines.counterfeit import ensemble
from pipelines.counterfeit.color import (
    ColorProfileMeasurement,
    measure_color_profile,
)
from pipelines.counterfeit.embedding_anomaly import (
    EmbeddingAnomalyMeasurement,
    measure_embedding_anomaly,
)
from pipelines.counterfeit.rosette import (
    RosetteMeasurement,
    measure_rosette,
)
from pipelines.counterfeit.typography import (
    TypographyResult,
    analyze_typography,
)

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
EMBEDDING_AUTHENTIC_THRESHOLD = ensemble.EMBEDDING_AUTHENTIC_THRESHOLD
EMBEDDING_COUNTERFEIT_THRESHOLD = ensemble.EMBEDDING_COUNTERFEIT_THRESHOLD
EMBEDDING_MIN_CONFIDENCE = ensemble.EMBEDDING_MIN_CONFIDENCE
EMBEDDING_MODEL_VERSION = ensemble.EMBEDDING_MODEL_VERSION
TYPOGRAPHY_AUTHENTIC_THRESHOLD = ensemble.TYPOGRAPHY_AUTHENTIC_THRESHOLD
TYPOGRAPHY_COUNTERFEIT_THRESHOLD = ensemble.TYPOGRAPHY_COUNTERFEIT_THRESHOLD
TYPOGRAPHY_MIN_CONFIDENCE = ensemble.TYPOGRAPHY_MIN_CONFIDENCE
TYPOGRAPHY_MODEL_VERSION = ensemble.TYPOGRAPHY_MODEL_VERSION


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


def analyze_embedding_anomaly(
    submitted_embedding: np.ndarray | None,
    *,
    manufacturer: str | None,
    variant_id: str | None,
    references_store_path: str | Path,
) -> EmbeddingAnomalyMeasurement:
    """Run the embedding-anomaly counterfeit detector.

    Unlike the rosette and color detectors (which operate on the
    canonical image directly), this detector compares the *submitted
    card's identification embedding* to a centroid of authentic
    reference embeddings for the same variant. It therefore depends
    on (a) identification having produced an embedding and (b) at
    least one reference embedding being on file for the identified
    variant.

    The detector returns a no-signal `EmbeddingAnomalyMeasurement`
    (n_references=0, confidence=0, score=0.5) — which the ensemble's
    `verdict_from_embedding_anomaly` maps to UNVERIFIED — when:
      - no submitted embedding is available (e.g. identification
        short-circuited on a confident pHash exact match and never
        computed an embedding), OR
      - the card was not identified to a known variant, OR
      - the identified variant has no reference embeddings stored
        (uncommon variants, fresh sets, or a stale store).

    This is by design: the detector abstains gracefully so the
    surrounding ensemble (rosette + color) still produces a verdict.

    Args:
        submitted_embedding: float32 (d,) embedding of the submitted
            canonical, or None if unavailable.
        manufacturer: short-name from the catalog entry's `game`
            field ("mtg", "pokemon", ...). None if unidentified.
        variant_id: per-printing string id matching the references
            store keys (Scryfall UUID for MTG, "<set>-<num>" for
            Pokemon). None if unidentified.
        references_store_path: path to the npz archive produced by
            `ml/data/ingestion/reference_embeddings.embed_references`.

    Returns:
        EmbeddingAnomalyMeasurement. Always returns a measurement
        (never raises CounterfeitFailedError) — the abstain path is
        a measurement with confidence=0, not an exception.
    """
    if (
        submitted_embedding is None
        or manufacturer is None
        or variant_id is None
    ):
        return _no_signal_embedding_measurement(reason="unidentified")

    references = lookup_references(
        Path(references_store_path), manufacturer, variant_id
    )
    if references is None:
        return _no_signal_embedding_measurement(reason="no_references")

    return measure_embedding_anomaly(
        np.asarray(submitted_embedding, dtype=np.float32),
        np.asarray(references, dtype=np.float32),
    )


def _no_signal_embedding_measurement(*, reason: str) -> EmbeddingAnomalyMeasurement:
    """Construct an embedding-anomaly measurement that the ensemble will
    map to UNVERIFIED. Used when the detector can't run for non-error
    reasons — unidentified card, no references, etc."""
    return EmbeddingAnomalyMeasurement(
        embedding_score=0.5,
        distance_from_centroid=0.0,
        n_references=0,
        confidence=0.0,
        manufacturer_profile="generic",
        metadata={"reason": reason},
    )


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


def analyze_typography_service(
    canonical_s3_key: str,
    identified_card_name: str | None,
) -> TypographyResult:
    """Run the typography (OCR + Levenshtein vs identified name) detector.

    Same envelope as the existing service wrappers but with one extra
    abstain path: if `identified_card_name` is None (e.g. identification
    didn't match the card to a known variant) the detector self-abstains
    with reason='no_expected_text' rather than running the OCR — there's
    nothing to compare against.

    Never raises. The `analyze_typography` core function already encodes
    every failure mode as an abstain measurement (confidence=0); this
    wrapper additionally swallows storage / load errors so a bad canonical
    can't take down the counterfeit ensemble. The verdict mapper in
    ensemble.py turns confidence=0 into UNVERIFIED.

    Args:
        canonical_s3_key: S3 key of the dewarped 750x1050 BGR canonical.
        identified_card_name: name of the matched card from the
            identification stage, or None if unidentified.

    Returns:
        TypographyResult. Always populated; abstain encoded as
        confidence=0 with `abstain_reason` set.
    """
    if identified_card_name is None or not str(identified_card_name).strip():
        return TypographyResult(
            score=0.5,
            confidence=0.0,
            extracted_text=None,
            expected_text=None,
            levenshtein_distance=None,
            abstain_reason="no_expected_text",
            manufacturer_profile="generic",
            metadata={"reason": "unidentified"},
        )
    try:
        image = _load_canonical_bgr(canonical_s3_key)
    except CounterfeitFailedError as e:
        return TypographyResult(
            score=0.5,
            confidence=0.0,
            extracted_text=None,
            expected_text=identified_card_name,
            levenshtein_distance=None,
            abstain_reason="invalid_image",
            manufacturer_profile="generic",
            metadata={"reason": "load_failed", "error": str(e)},
        )
    try:
        return analyze_typography(image, identified_card_name)
    except Exception as e:  # extremely defensive — analyze_typography is no-raise by contract
        return TypographyResult(
            score=0.5,
            confidence=0.0,
            extracted_text=None,
            expected_text=identified_card_name,
            levenshtein_distance=None,
            abstain_reason="ocr_unavailable",
            manufacturer_profile="generic",
            metadata={"reason": "detector_failed", "error": str(e)},
        )


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


def _verdict_from_embedding_anomaly(
    m: EmbeddingAnomalyMeasurement,
) -> AuthenticityVerdict:
    """Same shape as the others against the embedding-anomaly thresholds.
    The detector self-abstains (UNVERIFIED) when no references are
    available — see `analyze_embedding_anomaly` for the abstain paths."""
    return AuthenticityVerdict(ensemble.verdict_from_embedding_anomaly(m))


def _verdict_from_typography(m: TypographyResult) -> AuthenticityVerdict:
    """Same shape as the others against the typography thresholds. The
    detector self-abstains (UNVERIFIED) when RapidOCR is missing, no
    expected name was supplied, OCR failed on the input, or the image
    is invalid — see `analyze_typography_service` for the abstain paths."""
    return AuthenticityVerdict(
        ensemble.verdict_from_typography(m.score, m.confidence)
    )


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


def _reasons_from_embedding_anomaly(
    m: EmbeddingAnomalyMeasurement, verdict: AuthenticityVerdict
) -> list[str]:
    reasons: list[str] = []
    if verdict == AuthenticityVerdict.UNVERIFIED:
        # The metadata "reason" tag distinguishes unidentified-card from
        # no-references-on-file from genuinely-too-few-references.
        why = m.metadata.get("reason") if isinstance(m.metadata, dict) else None
        if why == "unidentified":
            reasons.append("card not identified — no variant to compare against")
        elif why == "no_references":
            reasons.append(
                "no authentic reference embeddings on file for this variant"
            )
        else:
            reasons.append(
                f"insufficient references for centroid analysis "
                f"(n_references={m.n_references}, confidence={m.confidence:.2f})"
            )
        return reasons
    if verdict == AuthenticityVerdict.AUTHENTIC:
        reasons.append(
            f"submitted embedding consistent with authentic exemplars "
            f"(embedding_score={m.embedding_score:.2f}, "
            f"distance={m.distance_from_centroid:.3f}, "
            f"n_refs={m.n_references})"
        )
    elif verdict == AuthenticityVerdict.LIKELY_COUNTERFEIT:
        reasons.append(
            f"submitted embedding far from authentic centroid "
            f"(embedding_score={m.embedding_score:.2f}, "
            f"distance={m.distance_from_centroid:.3f}, "
            f"n_refs={m.n_references})"
        )
    else:  # SUSPICIOUS
        reasons.append(
            f"borderline embedding distance — neither clearly authentic "
            f"nor counterfeit (embedding_score={m.embedding_score:.2f}, "
            f"distance={m.distance_from_centroid:.3f}, "
            f"n_refs={m.n_references})"
        )
    return reasons


def _reasons_from_typography(
    m: TypographyResult, verdict: AuthenticityVerdict
) -> list[str]:
    reasons: list[str] = []
    if verdict == AuthenticityVerdict.UNVERIFIED:
        why = m.abstain_reason or (
            m.metadata.get("reason") if isinstance(m.metadata, dict) else None
        )
        if why == "no_expected_text":
            reasons.append(
                "card not identified — no expected card name to OCR-compare"
            )
        elif why == "ocr_unavailable":
            reasons.append(
                "OCR engine unavailable — typography signal not collected"
            )
        elif why == "invalid_image":
            reasons.append(
                "title-region image invalid for OCR — typography signal skipped"
            )
        else:
            reasons.append(
                f"insufficient OCR confidence "
                f"(confidence={m.confidence:.2f}, reason={why or 'unknown'})"
            )
        return reasons
    if verdict == AuthenticityVerdict.AUTHENTIC:
        reasons.append(
            f"OCR'd title matches identified card name "
            f"(typography_score={m.score:.2f}, "
            f"levenshtein={m.levenshtein_distance})"
        )
    elif verdict == AuthenticityVerdict.LIKELY_COUNTERFEIT:
        reasons.append(
            f"OCR'd title diverges from identified card name "
            f"(typography_score={m.score:.2f}, "
            f"levenshtein={m.levenshtein_distance}, "
            f"extracted={m.extracted_text!r})"
        )
    else:  # SUSPICIOUS
        reasons.append(
            f"OCR'd title partially matches identified card name "
            f"(typography_score={m.score:.2f}, "
            f"levenshtein={m.levenshtein_distance})"
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


def _typography_no_signal(reason: str = "no_expected_text") -> TypographyResult:
    """Default-shaped abstaining TypographyResult for callers that don't
    have a real measurement to pass in (legacy callers, tests). Mapped
    to UNVERIFIED by the verdict mapper."""
    return TypographyResult(
        score=0.5,
        confidence=0.0,
        extracted_text=None,
        expected_text=None,
        levenshtein_distance=None,
        abstain_reason=reason,
        manufacturer_profile="generic",
        metadata={"reason": reason},
    )


async def persist_authenticity_result(
    submission_id: uuid.UUID,
    *,
    rosette: RosetteMeasurement,
    color: ColorProfileMeasurement,
    embedding: EmbeddingAnomalyMeasurement,
    db: AsyncSession,
    typography: TypographyResult | None = None,
) -> AuthenticityResult:
    """Insert or update the AuthenticityResult row for a submission.

    A submission has at most one authenticity row (unique index on
    submission_id), so re-runs update in place rather than colliding on
    the unique constraint.

    Per-detector verdicts come from the `_verdict_from_*` mappers; the
    combined row-level verdict is `_combine_verdicts(...)`. The combined
    `confidence` field is the MIN over CONFIDENT detectors only —
    abstaining detectors (confidence < their MIN_CONFIDENCE threshold)
    are excluded from the min so their abstain doesn't drag the row
    confidence to zero. If every detector abstains, combined confidence
    is 0.

    `typography` is optional for source compatibility with callers that
    haven't been updated yet — when omitted, a no-signal abstain is
    used so the row still includes a `typography` slot in
    `detector_scores` (for forensic completeness) but it doesn't
    influence the combined verdict.

    The detector_scores blob carries each detector's raw outputs
    verbatim so a future re-calibration can recompute verdicts from
    history without re-running the math."""
    if typography is None:
        typography = _typography_no_signal()
    rosette_verdict = _verdict_from_rosette(rosette)
    color_verdict = _verdict_from_color_profile(color)
    embedding_verdict = _verdict_from_embedding_anomaly(embedding)
    typography_verdict = _verdict_from_typography(typography)
    verdict = _combine_verdicts(
        [rosette_verdict, color_verdict, embedding_verdict, typography_verdict]
    )

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
        "embedding_anomaly": {
            "score": float(embedding.embedding_score),
            "distance_from_centroid": float(embedding.distance_from_centroid),
            "n_references": int(embedding.n_references),
            "confidence": float(embedding.confidence),
            "manufacturer_profile": embedding.manufacturer_profile,
            "verdict": embedding_verdict.value,
            "abstain_reason": (
                embedding.metadata.get("reason")
                if isinstance(embedding.metadata, dict)
                else None
            ),
        },
        "typography": {
            "score": float(typography.score),
            "confidence": float(typography.confidence),
            "extracted_text": typography.extracted_text,
            "expected_text": typography.expected_text,
            "levenshtein_distance": (
                int(typography.levenshtein_distance)
                if typography.levenshtein_distance is not None
                else None
            ),
            "manufacturer_profile": typography.manufacturer_profile,
            "verdict": typography_verdict.value,
            "abstain_reason": typography.abstain_reason,
        },
    }
    reasons: list[str] = []
    reasons.extend(_reasons_from_rosette(rosette, rosette_verdict))
    reasons.extend(_reasons_from_color_profile(color, color_verdict))
    reasons.extend(_reasons_from_embedding_anomaly(embedding, embedding_verdict))
    reasons.extend(_reasons_from_typography(typography, typography_verdict))

    model_versions = {
        "rosette": ROSETTE_MODEL_VERSION,
        "color": COLOR_MODEL_VERSION,
        "embedding_anomaly": EMBEDDING_MODEL_VERSION,
        "typography": TYPOGRAPHY_MODEL_VERSION,
        "thresholds": {
            "rosette_authentic": ROSETTE_AUTHENTIC_THRESHOLD,
            "rosette_counterfeit": ROSETTE_COUNTERFEIT_THRESHOLD,
            "rosette_min_confidence": ROSETTE_MIN_CONFIDENCE,
            "color_authentic": COLOR_AUTHENTIC_THRESHOLD,
            "color_counterfeit": COLOR_COUNTERFEIT_THRESHOLD,
            "color_min_confidence": COLOR_MIN_CONFIDENCE,
            "embedding_authentic": EMBEDDING_AUTHENTIC_THRESHOLD,
            "embedding_counterfeit": EMBEDDING_COUNTERFEIT_THRESHOLD,
            "embedding_min_confidence": EMBEDDING_MIN_CONFIDENCE,
            "typography_authentic": TYPOGRAPHY_AUTHENTIC_THRESHOLD,
            "typography_counterfeit": TYPOGRAPHY_COUNTERFEIT_THRESHOLD,
            "typography_min_confidence": TYPOGRAPHY_MIN_CONFIDENCE,
        },
    }

    # Combined confidence: min across CONFIDENT detectors (those with
    # confidence ≥ their per-detector min-confidence threshold). An
    # abstaining detector contributes UNVERIFIED but doesn't drag the
    # row confidence to zero — embedding-anomaly will abstain on every
    # uncatalogued card, and we don't want that to mask the rosette/
    # color signal that DID fire.
    confident: list[float] = []
    if rosette.confidence >= ROSETTE_MIN_CONFIDENCE:
        confident.append(float(rosette.confidence))
    if color.confidence >= COLOR_MIN_CONFIDENCE:
        confident.append(float(color.confidence))
    if embedding.confidence >= EMBEDDING_MIN_CONFIDENCE:
        confident.append(float(embedding.confidence))
    if typography.confidence >= TYPOGRAPHY_MIN_CONFIDENCE:
        confident.append(float(typography.confidence))
    combined_confidence = min(confident) if confident else 0.0

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
