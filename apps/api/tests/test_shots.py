"""End-to-end shot upload + register flow.

Walks the full client journey: create submission → request upload URL →
PUT bytes to S3 (moto) → POST /shots → assert quality result persisted on
the submission_shots row."""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator

import boto3
import httpx
import pytest
from moto import mock_aws

from grader.services import storage
from grader.settings import get_settings
from tests.fixtures import (
    blurry,
    card_in_scene,
    encode_jpeg,
    with_glare,
)

# Every test in this module drives the FastAPI client + submission_shots
# rows through the `client`/`db_session` fixtures.
pytestmark = pytest.mark.requires_postgres


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


async def _create_submission(client: httpx.AsyncClient, headers: dict[str, str]) -> uuid.UUID:
    r = await client.post("/submissions", headers=headers, json={})
    assert r.status_code == 201
    return uuid.UUID(r.json()["id"])


async def _request_upload(
    client: httpx.AsyncClient,
    headers: dict[str, str],
    submission_id: uuid.UUID,
    kind: str = "front_full",
) -> dict[str, str]:
    r = await client.post(
        f"/submissions/{submission_id}/shots/upload-url",
        headers=headers,
        json={"kind": kind, "content_type": "image/jpeg"},
    )
    assert r.status_code == 201, r.text
    return r.json()


def _put_to_s3(bucket: str, key: str, body: bytes, content_type: str = "image/jpeg") -> None:
    boto3.client("s3", region_name="us-east-1").put_object(
        Bucket=bucket, Key=key, Body=body, ContentType=content_type
    )


@pytest.mark.asyncio
async def test_upload_url_request_requires_auth(
    client: httpx.AsyncClient, s3_bucket: str
) -> None:
    sid = uuid.uuid4()
    r = await client.post(
        f"/submissions/{sid}/shots/upload-url",
        json={"kind": "front_full", "content_type": "image/jpeg"},
    )
    assert r.status_code == 401


@pytest.mark.asyncio
async def test_upload_url_404_for_other_users_submission(
    client: httpx.AsyncClient, auth_headers: dict[str, str], s3_bucket: str
) -> None:
    sid = await _create_submission(client, auth_headers)
    other = {"Authorization": "Dev someone-else"}
    r = await client.post(
        f"/submissions/{sid}/shots/upload-url",
        headers=other,
        json={"kind": "front_full", "content_type": "image/jpeg"},
    )
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_upload_url_response_is_presigned_post_form(
    client: httpx.AsyncClient, auth_headers: dict[str, str], s3_bucket: str
) -> None:
    """The endpoint must return the presigned-POST shape (``url`` +
    ``fields``) rather than the legacy presigned-PUT URL. The
    ``fields`` dict must contain a signed policy whose
    ``content-length-range`` condition matches our configured cap."""
    import base64
    import json as _json

    from grader.settings import get_settings

    sid = await _create_submission(client, auth_headers)
    upload = await _request_upload(client, auth_headers, sid)

    # New POST-shaped payload.
    assert "url" in upload
    assert "fields" in upload
    assert "shot_id" in upload
    assert "s3_key" in upload
    # Legacy fields removed.
    assert "upload_url" not in upload
    assert "required_headers" not in upload

    fields = upload["fields"]
    assert isinstance(fields, dict)
    assert fields["Content-Type"] == "image/jpeg"
    assert fields["key"] == upload["s3_key"]
    assert "policy" in fields
    assert "x-amz-signature" in fields

    # The size cap is in the signed policy.
    policy = _json.loads(base64.b64decode(fields["policy"]).decode("utf-8"))
    range_cond = next(
        (c for c in policy["conditions"]
         if isinstance(c, list) and c and c[0] == "content-length-range"),
        None,
    )
    assert range_cond is not None
    assert range_cond[1] == 1
    assert range_cond[2] == get_settings().submission_max_image_bytes


@pytest.mark.asyncio
async def test_register_413_when_object_exceeds_size_cap(
    client: httpx.AsyncClient,
    auth_headers: dict[str, str],
    s3_bucket: str,
) -> None:
    """If a client somehow lands an object larger than the configured
    cap (e.g. weak MinIO policy enforcement, an attacker bypassing
    the form), the server-side HEAD check on register must reject it
    with 413 before quality decoding spends CPU on a multi-GB blob."""
    from unittest.mock import patch

    sid = await _create_submission(client, auth_headers)
    upload = await _request_upload(client, auth_headers, sid)
    body = encode_jpeg(card_in_scene(fill=0.55))
    _put_to_s3(s3_bucket, upload["s3_key"], body)

    # Patch storage.head_shot to claim the object is way too big.
    over_limit = 50 * 1024 * 1024 * 1024  # 50 GiB
    with patch(
        "grader.routers.submissions.storage.head_shot",
        return_value={
            "content_length": over_limit,
            "content_type": "image/jpeg",
            "etag": "fake-etag",
        },
    ):
        r = await client.post(
            f"/submissions/{sid}/shots",
            headers=auth_headers,
            json={"shot_id": upload["shot_id"], "s3_key": upload["s3_key"]},
        )
    assert r.status_code == 413, r.text
    detail = r.json()["detail"]
    assert detail["reason"] == "upload_too_large"
    assert detail["got_bytes"] == over_limit
    assert detail["max_bytes"] > 0


@pytest.mark.asyncio
async def test_full_shot_flow_passes_quality(
    client: httpx.AsyncClient,
    auth_headers: dict[str, str],
    s3_bucket: str,
) -> None:
    sid = await _create_submission(client, auth_headers)
    upload = await _request_upload(client, auth_headers, sid)
    body = encode_jpeg(card_in_scene(fill=0.55))
    _put_to_s3(s3_bucket, upload["s3_key"], body)

    r = await client.post(
        f"/submissions/{sid}/shots",
        headers=auth_headers,
        json={"shot_id": upload["shot_id"], "s3_key": upload["s3_key"]},
    )
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["quality_passed"] is True
    assert body["kind"] == "front_full"
    assert body["width_px"] > 0 and body["height_px"] > 0


@pytest.mark.asyncio
async def test_blurry_shot_records_failure(
    client: httpx.AsyncClient,
    auth_headers: dict[str, str],
    s3_bucket: str,
) -> None:
    sid = await _create_submission(client, auth_headers)
    upload = await _request_upload(client, auth_headers, sid)
    body = encode_jpeg(blurry(card_in_scene(fill=0.55), k=35))
    _put_to_s3(s3_bucket, upload["s3_key"], body)

    r = await client.post(
        f"/submissions/{sid}/shots",
        headers=auth_headers,
        json={"shot_id": upload["shot_id"], "s3_key": upload["s3_key"]},
    )
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["quality_passed"] is False
    reasons = body["quality_metadata"]["reasons"]
    assert any("blurry" in r for r in reasons)


@pytest.mark.asyncio
async def test_glary_shot_records_failure(
    client: httpx.AsyncClient,
    auth_headers: dict[str, str],
    s3_bucket: str,
) -> None:
    sid = await _create_submission(client, auth_headers)
    upload = await _request_upload(client, auth_headers, sid)
    body = encode_jpeg(with_glare(card_in_scene(fill=0.55), fraction=0.05))
    _put_to_s3(s3_bucket, upload["s3_key"], body)

    r = await client.post(
        f"/submissions/{sid}/shots",
        headers=auth_headers,
        json={"shot_id": upload["shot_id"], "s3_key": upload["s3_key"]},
    )
    assert r.status_code == 201
    assert r.json()["quality_passed"] is False
    assert any("glare" in r for r in r.json()["quality_metadata"]["reasons"])


@pytest.mark.asyncio
async def test_register_400_when_s3_object_missing(
    client: httpx.AsyncClient,
    auth_headers: dict[str, str],
    s3_bucket: str,
) -> None:
    sid = await _create_submission(client, auth_headers)
    upload = await _request_upload(client, auth_headers, sid)
    # Skip the PUT to S3 entirely.
    r = await client.post(
        f"/submissions/{sid}/shots",
        headers=auth_headers,
        json={"shot_id": upload["shot_id"], "s3_key": upload["s3_key"]},
    )
    assert r.status_code == 400
    assert "not found" in r.json()["detail"].lower()


@pytest.mark.asyncio
async def test_register_400_when_bytes_are_not_an_image(
    client: httpx.AsyncClient,
    auth_headers: dict[str, str],
    s3_bucket: str,
) -> None:
    sid = await _create_submission(client, auth_headers)
    upload = await _request_upload(client, auth_headers, sid)
    _put_to_s3(s3_bucket, upload["s3_key"], b"this is not an image")
    r = await client.post(
        f"/submissions/{sid}/shots",
        headers=auth_headers,
        json={"shot_id": upload["shot_id"], "s3_key": upload["s3_key"]},
    )
    assert r.status_code == 400
    assert "decodable" in r.json()["detail"].lower()


@pytest.mark.asyncio
async def test_register_400_when_s3_key_does_not_match_ids(
    client: httpx.AsyncClient,
    auth_headers: dict[str, str],
    s3_bucket: str,
) -> None:
    sid = await _create_submission(client, auth_headers)
    upload = await _request_upload(client, auth_headers, sid)
    body = encode_jpeg(card_in_scene())
    _put_to_s3(s3_bucket, upload["s3_key"], body)

    forged_key = upload["s3_key"].replace(upload["shot_id"], str(uuid.uuid4()))
    r = await client.post(
        f"/submissions/{sid}/shots",
        headers=auth_headers,
        json={"shot_id": upload["shot_id"], "s3_key": forged_key},
    )
    assert r.status_code == 400


@pytest.mark.asyncio
async def test_register_409_on_duplicate_shot_id(
    client: httpx.AsyncClient,
    auth_headers: dict[str, str],
    s3_bucket: str,
) -> None:
    sid = await _create_submission(client, auth_headers)
    upload = await _request_upload(client, auth_headers, sid)
    body = encode_jpeg(card_in_scene(fill=0.55))
    _put_to_s3(s3_bucket, upload["s3_key"], body)

    first = await client.post(
        f"/submissions/{sid}/shots",
        headers=auth_headers,
        json={"shot_id": upload["shot_id"], "s3_key": upload["s3_key"]},
    )
    assert first.status_code == 201

    second = await client.post(
        f"/submissions/{sid}/shots",
        headers=auth_headers,
        json={"shot_id": upload["shot_id"], "s3_key": upload["s3_key"]},
    )
    assert second.status_code == 409


@pytest.mark.asyncio
async def test_first_shot_advances_submission_to_capturing(
    client: httpx.AsyncClient,
    auth_headers: dict[str, str],
    s3_bucket: str,
) -> None:
    sid = await _create_submission(client, auth_headers)
    sub = (await client.get(f"/submissions/{sid}", headers=auth_headers)).json()
    assert sub["status"] == "pending"

    upload = await _request_upload(client, auth_headers, sid)
    body = encode_jpeg(card_in_scene(fill=0.55))
    _put_to_s3(s3_bucket, upload["s3_key"], body)
    await client.post(
        f"/submissions/{sid}/shots",
        headers=auth_headers,
        json={"shot_id": upload["shot_id"], "s3_key": upload["s3_key"]},
    )

    sub_after = (await client.get(f"/submissions/{sid}", headers=auth_headers)).json()
    assert sub_after["status"] == "capturing"
