"""Tests for grader.services.counterfeit — the bridge between S3 storage
and the ml/pipelines/counterfeit/rosette module."""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator
from pathlib import Path

import boto3
import cv2
import numpy as np
import pytest
from moto import mock_aws

from grader.services import counterfeit, storage
from grader.settings import get_settings

_ML_ROOT = Path(__file__).resolve().parents[3] / "ml"
if str(_ML_ROOT) not in sys.path:
    sys.path.insert(0, str(_ML_ROOT))

from tests.fixtures import (  # noqa: E402
    synth_card,
    synth_continuous_tone_card,
    synth_halftone_card,
)
from grader.db.models import AuthenticityVerdict  # noqa: E402
from pipelines.counterfeit.color import (  # noqa: E402
    ColorProfileMeasurement,
)
from pipelines.counterfeit.rosette import (  # noqa: E402
    RosetteMeasurement,
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
