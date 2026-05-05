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
    assert r.score >= 0.99
    # Verdict maps to AUTHENTIC at the default thresholds.
    assert (
        counterfeit._verdict_from_typography(r) == AuthenticityVerdict.AUTHENTIC
    )


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
