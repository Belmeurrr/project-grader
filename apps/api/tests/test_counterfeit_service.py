"""Tests for grader.services.counterfeit — the bridge between S3 storage
and the ml/pipelines/counterfeit/rosette module."""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import boto3
import cv2
import numpy as np
import pytest
from moto import mock_aws

from grader.db.models import AuthenticityVerdict
from grader.services import counterfeit, storage
from grader.settings import get_settings
from pipelines.counterfeit.color import (
    ColorProfileMeasurement,
)
from pipelines.counterfeit.rosette import (
    RosetteMeasurement,
)
from tests.fixtures import (
    synth_card,
    synth_continuous_tone_card,
    synth_halftone_card,
)


@pytest.fixture(autouse=True)
def _aws_creds() -> Iterator[None]:
    prior = {k: os.environ.get(k) for k in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY")}
    os.environ["AWS_ACCESS_KEY_ID"] = "test"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "test"
    storage.reset_s3_client_cache()
    yield
    for k, v in prior.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    storage.reset_s3_client_cache()


@pytest.fixture
def s3_bucket() -> Iterator[str]:
    with mock_aws():
        bucket = get_settings().s3_bucket
        boto3.client("s3", region_name="us-east-1").create_bucket(Bucket=bucket)
        storage.reset_s3_client_cache()
        yield bucket


def _put_canonical(bucket: str, key: str, image: np.ndarray) -> None:
    ok, buf = cv2.imencode(".png", image)
    assert ok
    boto3.client("s3", region_name="us-east-1").put_object(
        Bucket=bucket, Key=key, Body=buf.tobytes(), ContentType="image/png"
    )


# -----------------------------
# analyze_rosette
# -----------------------------


def test_analyze_rosette_halftone_canonical_scores_high(s3_bucket: str) -> None:
    key = "test/canonical_halftone.png"
    _put_canonical(s3_bucket, key, synth_halftone_card(cell_size=6))

    result = counterfeit.analyze_rosette(key)
    assert result.rosette_score >= 0.7
    assert result.confidence > 0.0
    assert result.manufacturer_profile == "generic"


def test_analyze_rosette_continuous_tone_canonical_scores_low(s3_bucket: str) -> None:
    key = "test/canonical_continuous.png"
    _put_canonical(s3_bucket, key, synth_continuous_tone_card())

    result = counterfeit.analyze_rosette(key)
    assert result.rosette_score <= 0.3, (
        f"continuous canonical got score {result.rosette_score}; "
        f"peak={result.peak_strength}, patches={result.analyzed_patches}"
    )


def test_analyze_rosette_halftone_strictly_beats_continuous(s3_bucket: str) -> None:
    """Both canonicals are valid inputs; the halftone one must always
    out-score the continuous-tone one."""
    ht_key = "test/halftone.png"
    ct_key = "test/continuous.png"
    _put_canonical(s3_bucket, ht_key, synth_halftone_card(cell_size=6))
    _put_canonical(s3_bucket, ct_key, synth_continuous_tone_card())

    ht = counterfeit.analyze_rosette(ht_key)
    ct = counterfeit.analyze_rosette(ct_key)
    assert ht.rosette_score > ct.rosette_score


def test_analyze_rosette_raises_on_missing_key(s3_bucket: str) -> None:
    with pytest.raises(Exception):
        # Either ClientError from boto3 or CounterfeitFailedError — the
        # important thing is "missing key" is NOT silently turned into a
        # successful measurement. Different boto3 versions raise different
        # exception types; we accept any.
        counterfeit.analyze_rosette("test/does-not-exist.png")


def test_analyze_rosette_raises_on_corrupt_bytes(s3_bucket: str) -> None:
    key = "test/bad.png"
    boto3.client("s3", region_name="us-east-1").put_object(
        Bucket=s3_bucket, Key=key, Body=b"not a png", ContentType="image/png"
    )
    with pytest.raises(counterfeit.CounterfeitFailedError, match="decode"):
        counterfeit.analyze_rosette(key)


def test_analyze_rosette_raises_on_too_small_canonical(s3_bucket: str) -> None:
    """If something upstream produced a tiny canonical (shouldn't happen,
    but if it does), we surface a CounterfeitFailedError rather than a
    raw ValueError leak."""
    key = "test/tiny.png"
    tiny = np.zeros((50, 50, 3), dtype=np.uint8)
    _put_canonical(s3_bucket, key, tiny)
    with pytest.raises(counterfeit.CounterfeitFailedError):
        counterfeit.analyze_rosette(key)


# -----------------------------
# analyze_color_profile
# -----------------------------


def test_analyze_color_profile_saturated_canonical_scores_high(s3_bucket: str) -> None:
    key = "test/canonical_saturated.png"
    _put_canonical(s3_bucket, key, synth_card(image_color=(0, 0, 255)))

    result = counterfeit.analyze_color_profile(key)
    assert result.color_score >= 0.85
    assert result.confidence > 0.0
    assert result.manufacturer_profile == "generic"


def test_analyze_color_profile_desaturated_canonical_scores_low(s3_bucket: str) -> None:
    key = "test/canonical_desaturated.png"
    _put_canonical(s3_bucket, key, synth_card(image_color=(140, 130, 120)))

    result = counterfeit.analyze_color_profile(key)
    assert result.color_score <= 0.20, (
        f"desaturated canonical got score {result.color_score}; "
        f"p95_chroma={result.p95_chroma}, conf={result.confidence}"
    )


def test_analyze_color_profile_raises_on_corrupt_bytes(s3_bucket: str) -> None:
    key = "test/bad-color.png"
    boto3.client("s3", region_name="us-east-1").put_object(
        Bucket=s3_bucket, Key=key, Body=b"not a png", ContentType="image/png"
    )
    with pytest.raises(counterfeit.CounterfeitFailedError, match="decode"):
        counterfeit.analyze_color_profile(key)


def test_analyze_color_profile_raises_on_too_small_canonical(s3_bucket: str) -> None:
    key = "test/tiny-color.png"
    tiny = np.zeros((50, 50, 3), dtype=np.uint8)
    _put_canonical(s3_bucket, key, tiny)
    with pytest.raises(counterfeit.CounterfeitFailedError):
        counterfeit.analyze_color_profile(key)


# -----------------------------
# analyze_typography_service — abstain paths and image-load envelope
# -----------------------------


def test_analyze_typography_service_abstains_on_missing_card_name(
    s3_bucket: str,
) -> None:
    """No identified card name → no comparison possible → abstain
    (UNVERIFIED). Documents the design that the typography detector
    never raises; abstain is encoded as confidence=0."""
    key = "test/canonical_typography_unidentified.png"
    _put_canonical(s3_bucket, key, synth_card())
    r = counterfeit.analyze_typography_service(key, identified_card_name=None)
    assert r.confidence == 0.0
    assert r.abstain_reason == "no_expected_text"
    # Verdict mapper turns confidence=0 → UNVERIFIED.
    assert (
        counterfeit._verdict_from_typography(r) == AuthenticityVerdict.UNVERIFIED
    )


def test_analyze_typography_service_abstains_on_blank_card_name(
    s3_bucket: str,
) -> None:
    key = "test/canonical_typography_blank.png"
    _put_canonical(s3_bucket, key, synth_card())
    r = counterfeit.analyze_typography_service(key, identified_card_name="   ")
    assert r.confidence == 0.0
    assert r.abstain_reason == "no_expected_text"


def test_analyze_typography_service_does_not_raise_on_corrupt_bytes(
    s3_bucket: str,
) -> None:
    """The typography service swallows S3/load errors and abstains —
    a bad canonical should never take down the counterfeit ensemble.
    Distinct from `analyze_rosette` which raises CounterfeitFailedError
    (the orchestrator's stage-3.5 try/except catches that)."""
    key = "test/bad-typography.png"
    boto3.client("s3", region_name="us-east-1").put_object(
        Bucket=s3_bucket, Key=key, Body=b"not a png", ContentType="image/png"
    )
    r = counterfeit.analyze_typography_service(key, identified_card_name="Lightning Bolt")
    assert r.confidence == 0.0
    assert r.abstain_reason == "invalid_image"


def test_analyze_typography_service_happy_path_returns_populated_result(
    s3_bucket: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When OCR is available and the image loads cleanly, the service
    returns a populated TypographyResult (non-zero confidence, non-None
    extracted_text). Stub the OCR loader to avoid pulling in onnxruntime
    at test time — we're testing the service envelope, not the OCR
    engine itself."""
    from pipelines.counterfeit.typography import detector as ty_detector

    key = "test/canonical_typography_happy.png"
    _put_canonical(s3_bucket, key, synth_card())

    # Pretend RapidOCR is installed and reads the title cleanly.
    monkeypatch.setattr(
        ty_detector,
        "_try_load_ocr",
        lambda: (lambda img: ["Lightning Bolt"]),
    )

    r = counterfeit.analyze_typography_service(
        key, identified_card_name="Lightning Bolt"
    )
    assert r.abstain_reason is None
    assert r.confidence > 0.5
    assert r.extracted_text == "Lightning Bolt"
    assert r.expected_text == "Lightning Bolt"
    assert r.levenshtein_distance == 0
    # Exact match → ~0.985 per the detector's sigmoid (midpoint=0.35,
    # slope=12). Above the AUTHENTIC threshold (0.65) with margin.
    assert r.score >= 0.98
    # Verdict maps to AUTHENTIC at the default thresholds.
    assert (
        counterfeit._verdict_from_typography(r) == AuthenticityVerdict.AUTHENTIC
    )


# -----------------------------
# analyze_holographic_service — abstain paths and image-load envelope
# -----------------------------


def _put_synth_holo_canonical(
    bucket: str,
    key: str,
    *,
    foil_block: tuple[int, int, int] = (50, 200, 240),
    shift_dx: int = 0,
) -> None:
    """Build a synthetic 1050x750 BGR canonical with a foil-like
    saturated rectangle in the middle. shift_dx > 0 displaces the
    block horizontally — used to simulate the parallax difference
    between a front and tilt shot."""
    img = np.full((1050, 750, 3), 200, dtype=np.uint8)
    y0, y1 = 200, 600
    x0, x1 = 150 + shift_dx, 600 + shift_dx
    x0 = max(0, x0)
    x1 = min(750, x1)
    img[y0:y1, x0:x1] = foil_block
    # Add a small gradient so the optical-flow estimator can lock on.
    yy, xx = np.mgrid[y0:y1, x0:x1]
    mod = ((xx + yy) % 16).astype(np.uint8) * 8
    img[y0:y1, x0:x1, 0] = np.clip(
        img[y0:y1, x0:x1, 0].astype(np.int16) + mod - 64, 0, 255
    ).astype(np.uint8)
    img[y0:y1, x0:x1, 2] = np.clip(
        img[y0:y1, x0:x1, 2].astype(np.int16) - mod + 64, 0, 255
    ).astype(np.uint8)
    _put_canonical(bucket, key, img)


def test_analyze_holographic_service_abstains_when_tilt_missing(
    s3_bucket: str,
) -> None:
    """No tilt canonical key (tilt_30 not captured) → abstain
    (UNVERIFIED) with reason='tilt_not_captured'. Documents the
    optional-shot graceful-degradation path."""
    front_key = "test/canonical_holo_front.png"
    _put_synth_holo_canonical(s3_bucket, front_key, shift_dx=0)
    r = counterfeit.analyze_holographic_service(front_key, None)
    assert r.confidence == 0.0
    assert r.abstain_reason == "tilt_not_captured"
    assert (
        counterfeit._verdict_from_holographic(r) == AuthenticityVerdict.UNVERIFIED
    )


def test_analyze_holographic_service_does_not_raise_on_corrupt_front(
    s3_bucket: str,
) -> None:
    """Corrupt front canonical → swallow + abstain. The holographic
    service never raises, even on unreadable inputs."""
    front_key = "test/bad-holo-front.png"
    tilt_key = "test/canonical_holo_tilt.png"
    _put_synth_holo_canonical(s3_bucket, tilt_key, shift_dx=6)
    boto3.client("s3", region_name="us-east-1").put_object(
        Bucket=s3_bucket,
        Key=front_key,
        Body=b"not a png",
        ContentType="image/png",
    )
    r = counterfeit.analyze_holographic_service(front_key, tilt_key)
    assert r.confidence == 0.0
    assert r.abstain_reason == "invalid_image"


def test_analyze_holographic_service_does_not_raise_on_corrupt_tilt(
    s3_bucket: str,
) -> None:
    front_key = "test/canonical_holo_front2.png"
    tilt_key = "test/bad-holo-tilt.png"
    _put_synth_holo_canonical(s3_bucket, front_key, shift_dx=0)
    boto3.client("s3", region_name="us-east-1").put_object(
        Bucket=s3_bucket,
        Key=tilt_key,
        Body=b"not a png",
        ContentType="image/png",
    )
    r = counterfeit.analyze_holographic_service(front_key, tilt_key)
    assert r.confidence == 0.0
    assert r.abstain_reason == "invalid_image"


def test_analyze_holographic_service_happy_path(s3_bucket: str) -> None:
    """Front + tilt canonicals both present, foil region shifted on the
    tilt → flow ratio fires, score lands authentic-side, verdict
    AUTHENTIC."""
    front_key = "test/canonical_holo_front_happy.png"
    tilt_key = "test/canonical_holo_tilt_happy.png"
    _put_synth_holo_canonical(s3_bucket, front_key, shift_dx=0)
    _put_synth_holo_canonical(s3_bucket, tilt_key, shift_dx=6)

    r = counterfeit.analyze_holographic_service(front_key, tilt_key)
    assert r.abstain_reason is None
    assert r.confidence > 0.5
    assert r.flow_ratio is not None
    assert r.flow_ratio > 2.0
    assert r.score >= 0.85
    assert (
        counterfeit._verdict_from_holographic(r) == AuthenticityVerdict.AUTHENTIC
    )


def test_analyze_holographic_service_flat_foil_scores_counterfeit(
    s3_bucket: str,
) -> None:
    """Same foil region, no shift → ratio ≈ 1, score counterfeit-side."""
    front_key = "test/canonical_holo_front_flat.png"
    tilt_key = "test/canonical_holo_tilt_flat.png"
    _put_synth_holo_canonical(s3_bucket, front_key, shift_dx=0)
    _put_synth_holo_canonical(s3_bucket, tilt_key, shift_dx=0)

    r = counterfeit.analyze_holographic_service(front_key, tilt_key)
    assert r.abstain_reason is None
    assert r.confidence > 0.5
    assert r.score <= 0.20
    assert (
        counterfeit._verdict_from_holographic(r)
        == AuthenticityVerdict.LIKELY_COUNTERFEIT
    )


def test_verdict_from_holographic_maps_to_enum() -> None:
    """The string verdicts from ml/pipelines/counterfeit/ensemble round-
    trip cleanly into the SQLAlchemy AuthenticityVerdict enum for
    holographic. Mirrors the matching tests for typography + embedding."""
    from pipelines.counterfeit.holographic import HolographicResult

    AV = AuthenticityVerdict
    abstain = HolographicResult(
        score=0.5,
        confidence=0.0,
        flow_ratio=None,
        holo_mask_fraction=None,
        abstain_reason="tilt_not_captured",
        manufacturer_profile="generic",
    )
    assert counterfeit._verdict_from_holographic(abstain) == AV.UNVERIFIED

    authentic = HolographicResult(
        score=0.92,
        confidence=0.85,
        flow_ratio=3.5,
        holo_mask_fraction=0.12,
        abstain_reason=None,
        manufacturer_profile="generic",
    )
    assert counterfeit._verdict_from_holographic(authentic) == AV.AUTHENTIC

    fake = HolographicResult(
        score=0.05,
        confidence=0.85,
        flow_ratio=1.0,
        holo_mask_fraction=0.12,
        abstain_reason=None,
        manufacturer_profile="generic",
    )
    assert counterfeit._verdict_from_holographic(fake) == AV.LIKELY_COUNTERFEIT


def test_verdict_from_typography_maps_to_enum() -> None:
    """The string verdicts from ml/pipelines/counterfeit/ensemble round-
    trip cleanly into the SQLAlchemy AuthenticityVerdict enum for
    typography. Mirrors the matching test for embedding-anomaly."""
    from pipelines.counterfeit.typography import TypographyResult

    AV = AuthenticityVerdict
    abstain = TypographyResult(
        score=0.5,
        confidence=0.0,
        extracted_text=None,
        expected_text=None,
        levenshtein_distance=None,
        abstain_reason="no_expected_text",
        manufacturer_profile="generic",
    )
    assert counterfeit._verdict_from_typography(abstain) == AV.UNVERIFIED

    authentic = TypographyResult(
        score=0.95,
        confidence=0.85,
        extracted_text="Lightning Bolt",
        expected_text="Lightning Bolt",
        levenshtein_distance=0,
        abstain_reason=None,
        manufacturer_profile="generic",
    )
    assert counterfeit._verdict_from_typography(authentic) == AV.AUTHENTIC

    fake = TypographyResult(
        score=0.05,
        confidence=0.85,
        extracted_text="Pikachu",
        expected_text="Lightning Bolt",
        levenshtein_distance=10,
        abstain_reason=None,
        manufacturer_profile="generic",
    )
    assert counterfeit._verdict_from_typography(fake) == AV.LIKELY_COUNTERFEIT


# -----------------------------
# _combine_verdicts ensemble logic (pure function — no S3 needed)
# -----------------------------


def test_combine_verdicts_any_likely_counterfeit_wins() -> None:
    """Conservative ensemble: a single LIKELY_COUNTERFEIT decides the
    combined verdict regardless of what the other detectors say."""
    AV = AuthenticityVerdict
    # Even three AUTHENTICs can't override one LIKELY_COUNTERFEIT.
    assert counterfeit._combine_verdicts(
        [AV.LIKELY_COUNTERFEIT, AV.AUTHENTIC]
    ) == AV.LIKELY_COUNTERFEIT
    assert counterfeit._combine_verdicts(
        [AV.AUTHENTIC, AV.AUTHENTIC, AV.LIKELY_COUNTERFEIT]
    ) == AV.LIKELY_COUNTERFEIT


def test_combine_verdicts_authentic_requires_consensus() -> None:
    """AUTHENTIC only when ALL confident detectors agree on AUTHENTIC."""
    AV = AuthenticityVerdict
    assert counterfeit._combine_verdicts([AV.AUTHENTIC, AV.AUTHENTIC]) == AV.AUTHENTIC
    # An UNVERIFIED detector should not block an authentic verdict if
    # the other detectors are confident — the ensemble degrades.
    assert counterfeit._combine_verdicts(
        [AV.AUTHENTIC, AV.UNVERIFIED]
    ) == AV.AUTHENTIC


def test_combine_verdicts_suspicious_when_disagree() -> None:
    AV = AuthenticityVerdict
    assert counterfeit._combine_verdicts([AV.AUTHENTIC, AV.SUSPICIOUS]) == AV.SUSPICIOUS


def test_combine_verdicts_all_unverified_returns_unverified() -> None:
    AV = AuthenticityVerdict
    assert counterfeit._combine_verdicts(
        [AV.UNVERIFIED, AV.UNVERIFIED]
    ) == AV.UNVERIFIED


def test_combine_verdicts_empty_returns_unverified() -> None:
    """Defensive: an empty verdict list (no detectors ran) is UNVERIFIED."""
    assert counterfeit._combine_verdicts([]) == AuthenticityVerdict.UNVERIFIED


# -----------------------------
# analyze_embedding_anomaly — abstain paths and detector path
# -----------------------------


def test_analyze_embedding_anomaly_abstains_when_submitted_embedding_missing(
    tmp_path: Path,
) -> None:
    """No submitted embedding (e.g. identification short-circuited on a
    pHash exact match) → no-signal measurement with reason='unidentified'.
    Documents the design that embedding-anomaly never raises for non-
    error reasons; abstain is encoded in confidence=0, n_references=0."""
    m = counterfeit.analyze_embedding_anomaly(
        submitted_embedding=None,
        manufacturer="mtg",
        variant_id="some-variant",
        references_store_path=tmp_path / "doesnt_exist.npz",
    )
    assert m.n_references == 0
    assert m.confidence == 0.0
    assert m.metadata.get("reason") == "unidentified"


def test_analyze_embedding_anomaly_abstains_when_unidentified(
    tmp_path: Path,
) -> None:
    m = counterfeit.analyze_embedding_anomaly(
        submitted_embedding=np.random.default_rng(0).standard_normal(8).astype(np.float32),
        manufacturer=None,
        variant_id=None,
        references_store_path=tmp_path / "doesnt_exist.npz",
    )
    assert m.n_references == 0
    assert m.confidence == 0.0
    assert m.metadata.get("reason") == "unidentified"


def test_analyze_embedding_anomaly_abstains_when_store_missing(
    tmp_path: Path,
) -> None:
    """Identified card BUT references npz doesn't exist → no_references
    abstain. The detector should never crash on a missing store."""
    m = counterfeit.analyze_embedding_anomaly(
        submitted_embedding=np.random.default_rng(0).standard_normal(8).astype(np.float32),
        manufacturer="mtg",
        variant_id="abc-123",
        references_store_path=tmp_path / "doesnt_exist.npz",
    )
    assert m.n_references == 0
    assert m.confidence == 0.0
    assert m.metadata.get("reason") == "no_references"


def test_analyze_embedding_anomaly_abstains_when_variant_not_in_store(
    tmp_path: Path,
) -> None:
    """References npz exists but doesn't have an entry for the requested
    variant → no_references abstain."""
    npz = tmp_path / "ref.npz"
    np.savez(str(npz), **{"mtg/other-variant": np.zeros(8, dtype=np.float32)})

    m = counterfeit.analyze_embedding_anomaly(
        submitted_embedding=np.zeros(8, dtype=np.float32),
        manufacturer="mtg",
        variant_id="missing-variant",
        references_store_path=npz,
    )
    assert m.n_references == 0
    assert m.metadata.get("reason") == "no_references"


def test_analyze_embedding_anomaly_runs_detector_when_references_present(
    tmp_path: Path,
) -> None:
    """Identified card + matching reference → detector runs, returns a
    real (non-abstain) measurement. Submitted == reference embedding
    means cosine distance 0 and a high authenticity score."""
    npz = tmp_path / "ref.npz"
    ref = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    np.savez(str(npz), **{"mtg/abc-123": ref})

    m = counterfeit.analyze_embedding_anomaly(
        submitted_embedding=ref.copy(),
        manufacturer="mtg",
        variant_id="abc-123",
        references_store_path=npz,
    )
    assert m.n_references == 1
    assert m.confidence > 0.0
    # Same embedding → cosine distance ≈ 0 → score saturates near 1.
    assert m.embedding_score >= 0.9
    assert m.distance_from_centroid < 1e-3
    assert m.metadata.get("reason") is None


# -----------------------------
# analyze_knn_reference_service — abstain paths and detector path
# -----------------------------


def test_analyze_knn_reference_service_abstains_when_submitted_embedding_missing(
    tmp_path: Path,
) -> None:
    """No submitted embedding (e.g. identification short-circuited on a
    pHash exact match) → no-signal measurement with
    abstain_reason='no_submitted_embedding'. Mirrors embedding-anomaly's
    abstain shape — both detectors share the same submitted-embedding
    source, so they share this failure mode."""
    m = counterfeit.analyze_knn_reference_service(
        submitted_embedding=None,
        manufacturer="mtg",
        variant_id="some-variant",
        references_store_path=tmp_path / "doesnt_exist.npz",
    )
    assert m.n_references_used == 0
    assert m.confidence == 0.0
    assert m.abstain_reason == "no_submitted_embedding"
    assert (
        counterfeit._verdict_from_knn_reference(m) == AuthenticityVerdict.UNVERIFIED
    )


def test_analyze_knn_reference_service_abstains_when_unidentified(
    tmp_path: Path,
) -> None:
    """Submitted embedding present but card not identified → no variant
    to look up → 'insufficient_references' abstain (per the service's
    documented mapping for unidentified-but-have-embedding)."""
    m = counterfeit.analyze_knn_reference_service(
        submitted_embedding=np.random.default_rng(0).standard_normal(8).astype(np.float32),
        manufacturer=None,
        variant_id=None,
        references_store_path=tmp_path / "doesnt_exist.npz",
    )
    assert m.n_references_used == 0
    assert m.confidence == 0.0
    assert m.abstain_reason == "insufficient_references"


def test_analyze_knn_reference_service_abstains_when_too_few_references(
    tmp_path: Path,
) -> None:
    """References npz exists with a single reference for the variant —
    fewer than k=3 — so the detector abstains. The whole point of
    top-k is that it requires AT LEAST k references; below that it
    silently degenerates into nearest-neighbor on a tiny set."""
    npz = tmp_path / "ref.npz"
    ref = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    np.savez(str(npz), **{"mtg/abc-123": ref})

    m = counterfeit.analyze_knn_reference_service(
        submitted_embedding=ref.copy(),
        manufacturer="mtg",
        variant_id="abc-123",
        references_store_path=npz,
    )
    assert m.n_references_used == 1  # one ref on file but < k=3
    assert m.abstain_reason == "insufficient_references"
    assert (
        counterfeit._verdict_from_knn_reference(m) == AuthenticityVerdict.UNVERIFIED
    )


def test_analyze_knn_reference_service_abstains_when_variant_not_in_store(
    tmp_path: Path,
) -> None:
    """References npz exists but doesn't have an entry for the requested
    variant → no_references abstain via the underlying lookup_references
    returning None → 'insufficient_references' from the detector."""
    npz = tmp_path / "ref.npz"
    np.savez(
        str(npz),
        **{"mtg/other-variant": np.zeros((3, 8), dtype=np.float32)},
    )

    m = counterfeit.analyze_knn_reference_service(
        submitted_embedding=np.zeros(8, dtype=np.float32),
        manufacturer="mtg",
        variant_id="missing-variant",
        references_store_path=npz,
    )
    assert m.n_references_used == 0
    assert m.abstain_reason == "insufficient_references"


def test_analyze_knn_reference_service_runs_detector_when_enough_references(
    tmp_path: Path,
) -> None:
    """Identified variant + 3+ reference exemplars → detector runs and
    returns a real (non-abstain) measurement. With submitted == one of
    the references, top-k mean is small → high authenticity score."""
    npz = tmp_path / "ref.npz"
    rng = np.random.default_rng(0xC4FF)
    base = rng.standard_normal(8).astype(np.float32)
    base /= float(np.linalg.norm(base))
    # 5 same-variant references in a tight cluster around `base`.
    refs = []
    for _ in range(5):
        v = base + rng.standard_normal(8).astype(np.float32) * 0.05
        v /= float(np.linalg.norm(v))
        refs.append(v)
    refs_arr = np.stack(refs).astype(np.float32)
    np.savez(str(npz), **{"mtg/abc-123": refs_arr})

    m = counterfeit.analyze_knn_reference_service(
        submitted_embedding=base.copy(),
        manufacturer="mtg",
        variant_id="abc-123",
        references_store_path=npz,
    )
    assert m.abstain_reason is None
    assert m.n_references_used == 5
    assert m.k == 3
    assert m.confidence >= 0.4
    # Submitted == base; references are tightly clustered near base.
    # Mean top-3 cosine distance is very small → score saturates near 1.
    assert m.score >= 0.9
    assert m.mean_topk_distance < 0.05


def test_verdict_from_knn_reference_maps_to_enum() -> None:
    """The string verdicts from ml/pipelines/counterfeit/ensemble round-
    trip cleanly into the SQLAlchemy AuthenticityVerdict enum for k-NN
    reference. Mirrors the matching tests for the other 5 detectors."""
    from pipelines.counterfeit.knn_reference import KnnReferenceResult

    AV = AuthenticityVerdict
    abstain = KnnReferenceResult(
        score=0.5,
        confidence=0.0,
        mean_topk_distance=0.0,
        n_references_used=0,
        k=3,
        abstain_reason="insufficient_references",
        manufacturer_profile="generic",
    )
    assert counterfeit._verdict_from_knn_reference(abstain) == AV.UNVERIFIED

    authentic = KnnReferenceResult(
        score=0.95,
        confidence=0.85,
        mean_topk_distance=0.05,
        n_references_used=10,
        k=3,
        abstain_reason=None,
        manufacturer_profile="generic",
    )
    assert counterfeit._verdict_from_knn_reference(authentic) == AV.AUTHENTIC

    fake = KnnReferenceResult(
        score=0.05,
        confidence=0.85,
        mean_topk_distance=0.50,
        n_references_used=10,
        k=3,
        abstain_reason=None,
        manufacturer_profile="generic",
    )
    assert counterfeit._verdict_from_knn_reference(fake) == AV.LIKELY_COUNTERFEIT


# -----------------------------
# analyze_substrate_service — abstain paths and image-load envelope
# -----------------------------


def _put_substrate_pair(
    bucket: str,
    front_key: str,
    flash_key: str | None,
    *,
    front_color: tuple[int, int, int] = (230, 230, 230),
    flash_color: tuple[int, int, int] = (230, 230, 230),
) -> None:
    """Build a paired (front, flash) canonical pair and put both at the
    given keys. flash_key=None skips the flash upload (used by the
    'flash_not_captured' abstain path test)."""
    front_img = np.full((1050, 750, 3), front_color, dtype=np.uint8)
    _put_canonical(bucket, front_key, front_img)
    if flash_key is not None:
        flash_img = np.full((1050, 750, 3), flash_color, dtype=np.uint8)
        _put_canonical(bucket, flash_key, flash_img)


def test_analyze_substrate_service_abstains_when_flash_missing(
    s3_bucket: str,
) -> None:
    """No flash canonical key (front_full_flash not captured) → abstain
    UNVERIFIED with reason='flash_not_captured'. Documents the
    optional-shot graceful-degradation path. Mirrors the holographic
    service's tilt_not_captured abstain shape."""
    front_key = "test/canonical_substrate_front.png"
    _put_substrate_pair(s3_bucket, front_key, None)
    r = counterfeit.analyze_substrate_service(front_key, None)
    assert r.confidence == 0.0
    assert r.abstain_reason == "flash_not_captured"
    assert (
        counterfeit._verdict_from_substrate(r) == AuthenticityVerdict.UNVERIFIED
    )


def test_analyze_substrate_service_does_not_raise_on_corrupt_front(
    s3_bucket: str,
) -> None:
    """Corrupt front canonical → swallow + abstain. Like the holographic
    + typography service wrappers, the substrate service never raises."""
    front_key = "test/bad-substrate-front.png"
    flash_key = "test/canonical_substrate_flash.png"
    _put_canonical(
        s3_bucket,
        flash_key,
        np.full((1050, 750, 3), 230, dtype=np.uint8),
    )
    boto3.client("s3", region_name="us-east-1").put_object(
        Bucket=s3_bucket,
        Key=front_key,
        Body=b"not a png",
        ContentType="image/png",
    )
    r = counterfeit.analyze_substrate_service(front_key, flash_key)
    assert r.confidence == 0.0
    assert r.abstain_reason == "invalid_image"


def test_analyze_substrate_service_does_not_raise_on_corrupt_flash(
    s3_bucket: str,
) -> None:
    front_key = "test/canonical_substrate_front2.png"
    flash_key = "test/bad-substrate-flash.png"
    _put_canonical(
        s3_bucket,
        front_key,
        np.full((1050, 750, 3), 230, dtype=np.uint8),
    )
    boto3.client("s3", region_name="us-east-1").put_object(
        Bucket=s3_bucket,
        Key=flash_key,
        Body=b"not a png",
        ContentType="image/png",
    )
    r = counterfeit.analyze_substrate_service(front_key, flash_key)
    assert r.confidence == 0.0
    assert r.abstain_reason == "invalid_image"


def test_analyze_substrate_service_happy_path_authentic(s3_bucket: str) -> None:
    """Front + flash with similar near-neutral border b* → score lands
    authentic-side. Documents the happy-path contract end-to-end through
    the S3-load envelope."""
    front_key = "test/canonical_substrate_happy_front.png"
    flash_key = "test/canonical_substrate_happy_flash.png"
    _put_substrate_pair(
        s3_bucket,
        front_key,
        flash_key,
        front_color=(230, 230, 230),
        flash_color=(230, 230, 230),  # no fluorescence → delta_b ≈ 0
    )
    r = counterfeit.analyze_substrate_service(front_key, flash_key)
    assert r.abstain_reason is None
    assert r.confidence > 0.5
    # delta_b should be ~0 on identical inputs (within rounding noise).
    assert r.delta_b is not None
    assert abs(r.delta_b) < 1.0
    # Above the AUTHENTIC threshold (0.65) at default settings.
    assert r.score >= 0.70
    assert (
        counterfeit._verdict_from_substrate(r) == AuthenticityVerdict.AUTHENTIC
    )


def test_verdict_from_substrate_maps_to_enum() -> None:
    """The string verdicts from ml/pipelines/counterfeit/ensemble round-
    trip cleanly into the SQLAlchemy AuthenticityVerdict enum for
    substrate. Mirrors the matching tests for the other 6 detectors."""
    from pipelines.counterfeit.substrate import SubstrateResult

    AV = AuthenticityVerdict
    abstain = SubstrateResult(
        score=0.5,
        confidence=0.0,
        delta_b=None,
        border_mad=None,
        n_border_pixels=0,
        abstain_reason="flash_not_captured",
        manufacturer_profile="generic",
    )
    assert counterfeit._verdict_from_substrate(abstain) == AV.UNVERIFIED

    authentic = SubstrateResult(
        score=0.85,
        confidence=0.7,
        delta_b=0.0,
        border_mad=0.5,
        n_border_pixels=70_000,
        abstain_reason=None,
        manufacturer_profile="generic",
    )
    assert counterfeit._verdict_from_substrate(authentic) == AV.AUTHENTIC

    fake = SubstrateResult(
        score=0.05,
        confidence=0.7,
        delta_b=-10.0,
        border_mad=0.5,
        n_border_pixels=70_000,
        abstain_reason=None,
        manufacturer_profile="generic",
    )
    assert counterfeit._verdict_from_substrate(fake) == AV.LIKELY_COUNTERFEIT


def test_verdict_from_embedding_anomaly_maps_to_enum() -> None:
    """The string verdicts from ml/pipelines/counterfeit/ensemble round-
    trip cleanly into the SQLAlchemy AuthenticityVerdict enum."""
    from pipelines.counterfeit.embedding_anomaly import EmbeddingAnomalyMeasurement

    AV = AuthenticityVerdict
    # n_refs=0 → confidence=0 → UNVERIFIED.
    abstain = EmbeddingAnomalyMeasurement(
        embedding_score=0.5,
        distance_from_centroid=0.0,
        n_references=0,
        confidence=0.0,
        manufacturer_profile="generic",
    )
    assert counterfeit._verdict_from_embedding_anomaly(abstain) == AV.UNVERIFIED

    # High score + sufficient confidence → AUTHENTIC.
    authentic = EmbeddingAnomalyMeasurement(
        embedding_score=0.95,
        distance_from_centroid=0.05,
        n_references=3,
        confidence=0.7,
        manufacturer_profile="generic",
    )
    assert counterfeit._verdict_from_embedding_anomaly(authentic) == AV.AUTHENTIC

    # Low score + sufficient confidence → LIKELY_COUNTERFEIT.
    fake = EmbeddingAnomalyMeasurement(
        embedding_score=0.10,
        distance_from_centroid=0.50,
        n_references=3,
        confidence=0.7,
        manufacturer_profile="generic",
    )
    assert counterfeit._verdict_from_embedding_anomaly(fake) == AV.LIKELY_COUNTERFEIT
