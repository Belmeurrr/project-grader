"""Storage service tests against a moto-mocked S3.

We avoid real AWS — moto provides an in-process S3 implementation. The tests
exercise key derivation, presigned URL params, and the head/get round-trip."""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator

import boto3
import pytest
from moto import mock_aws

from grader.services import storage
from grader.settings import get_settings


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


def test_shot_s3_key_uses_extension_from_content_type() -> None:
    sub = uuid.UUID("11111111-1111-1111-1111-111111111111")
    shot = uuid.UUID("22222222-2222-2222-2222-222222222222")
    key = storage.shot_s3_key(sub, shot, "front_full", "image/jpeg")
    assert key == f"submissions/{sub}/shots/{shot}/front_full.jpg"

    key = storage.shot_s3_key(sub, shot, "front_full", "image/png")
    assert key.endswith("front_full.png")

    key = storage.shot_s3_key(sub, shot, "front_full", "image/webp")
    assert key.endswith("front_full.webp")


def test_shot_s3_key_rejects_unknown_content_type() -> None:
    sub = uuid.uuid4()
    shot = uuid.uuid4()
    with pytest.raises(ValueError, match="unsupported"):
        storage.shot_s3_key(sub, shot, "front_full", "image/tiff")


def test_presigned_put_for_shot_round_trip(s3_bucket: str) -> None:
    sub = uuid.uuid4()
    shot = uuid.uuid4()
    presigned = storage.presigned_put_for_shot(sub, shot, "front_full", "image/jpeg")
    assert presigned.s3_key.startswith(f"submissions/{sub}/shots/{shot}/")
    assert presigned.upload_url.startswith("http")
    assert presigned.required_headers == {"Content-Type": "image/jpeg"}
    # The signed URL targets the right bucket and key.
    assert s3_bucket in presigned.upload_url or "/" + s3_bucket + "/" in presigned.upload_url
    assert presigned.s3_key in presigned.upload_url


def test_head_shot_returns_none_for_missing_object(s3_bucket: str) -> None:
    assert storage.head_shot("submissions/missing/shots/missing/front_full.jpg") is None


def test_head_and_get_after_put(s3_bucket: str) -> None:
    sub = uuid.uuid4()
    shot = uuid.uuid4()
    key = storage.shot_s3_key(sub, shot, "front_full", "image/jpeg")
    payload = b"\xff\xd8\xff\xe0fake jpeg"

    boto3.client("s3", region_name="us-east-1").put_object(
        Bucket=s3_bucket, Key=key, Body=payload, ContentType="image/jpeg"
    )

    head = storage.head_shot(key)
    assert head is not None
    assert head["content_length"] == len(payload)
    assert head["content_type"] == "image/jpeg"

    assert storage.get_shot_bytes(key) == payload
