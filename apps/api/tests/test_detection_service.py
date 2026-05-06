"""Tests for grader.services.detection — the bridge between S3 storage and
the ml/pipelines/detection module."""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator

import boto3
import cv2
import numpy as np
import pytest
from moto import mock_aws

from grader.db.models import ShotKind
from grader.services import detection, storage
from grader.settings import get_settings
from tests.fixtures import card_in_scene, encode_jpeg


@pytest.fixture(autouse=True)
def _aws_creds_for_moto() -> Iterator[None]:
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


def _put_shot(bucket: str, submission_id: uuid.UUID, image: np.ndarray) -> tuple[str, ShotKind]:
    shot_id = uuid.uuid4()
    kind = ShotKind.FRONT_FULL
    key = f"submissions/{submission_id}/shots/{shot_id}/{kind.value}.jpg"
    boto3.client("s3", region_name="us-east-1").put_object(
        Bucket=bucket, Key=key, Body=encode_jpeg(image), ContentType="image/jpeg"
    )
    return key, kind


def test_detect_and_dewarp_clean_card_succeeds(s3_bucket: str) -> None:
    sid = uuid.uuid4()
    key, kind = _put_shot(s3_bucket, sid, card_in_scene(fill=0.55))

    result = detection.detect_and_dewarp_shot(key, kind)
    assert result.detection.confidence > 0
    assert result.dewarp.canonical.shape == (1050, 750, 3)
    assert result.dewarp.irregularity < 0.15
    assert result.canonical_s3_key == f"submissions/{sid}/canonical/front_full.png"


def test_detect_and_dewarp_persists_canonical_to_s3(s3_bucket: str) -> None:
    sid = uuid.uuid4()
    key, kind = _put_shot(s3_bucket, sid, card_in_scene(fill=0.55))

    result = detection.detect_and_dewarp_shot(key, kind)
    head = storage.head_shot(result.canonical_s3_key)
    assert head is not None
    assert head["content_type"] == "image/png"
    assert head["content_length"] > 0


def test_detect_and_dewarp_skip_persist(s3_bucket: str) -> None:
    sid = uuid.uuid4()
    key, kind = _put_shot(s3_bucket, sid, card_in_scene(fill=0.55))

    result = detection.detect_and_dewarp_shot(key, kind, persist_canonical=False)
    head = storage.head_shot(result.canonical_s3_key)
    assert head is None  # nothing was written


def test_detect_and_dewarp_raises_when_no_card(s3_bucket: str) -> None:
    sid = uuid.uuid4()
    blank = np.full((1000, 1000, 3), 35, dtype=np.uint8)
    key, kind = _put_shot(s3_bucket, sid, blank)
    with pytest.raises(detection.DetectionFailedError, match="no card"):
        detection.detect_and_dewarp_shot(key, kind)


def test_detect_and_dewarp_rejects_corrupt_bytes(s3_bucket: str) -> None:
    sid = uuid.uuid4()
    shot_id = uuid.uuid4()
    kind = ShotKind.FRONT_FULL
    key = f"submissions/{sid}/shots/{shot_id}/{kind.value}.jpg"
    boto3.client("s3", region_name="us-east-1").put_object(
        Bucket=s3_bucket, Key=key, Body=b"not an image", ContentType="image/jpeg"
    )
    with pytest.raises(detection.DetectionFailedError, match="decode"):
        detection.detect_and_dewarp_shot(key, kind)


def test_detect_and_dewarp_rejects_bent_card(s3_bucket: str) -> None:
    """A heavily skewed quad should fail the irregularity gate."""
    sid = uuid.uuid4()
    skewed = card_in_scene(fill=0.55, perspective_skew_px=200)
    key, kind = _put_shot(s3_bucket, sid, skewed)
    with pytest.raises(detection.DetectionFailedError, match="irregular|too steep|bent"):
        detection.detect_and_dewarp_shot(key, kind, max_irregularity=0.05)


def test_submission_id_from_s3_key_extracts_uuid() -> None:
    """The regex must accept the canonical layout produced by
    ``storage.shot_s3_key``."""
    sid = uuid.uuid4()
    key = f"submissions/{sid}/shots/{uuid.uuid4()}/front_full.jpg"
    assert detection._submission_id_from_s3_key(key) == str(sid)


def test_submission_id_from_s3_key_rejects_malformed() -> None:
    """A key that doesn't start with ``submissions/<uuid>/`` must
    raise rather than silently mis-parse via index."""
    with pytest.raises(detection.DetectionFailedError, match="layout"):
        detection._submission_id_from_s3_key("not-a-prefix/abc/foo.jpg")
    with pytest.raises(detection.DetectionFailedError, match="layout"):
        detection._submission_id_from_s3_key("submissions/not-a-uuid/shots/x/front_full.jpg")
