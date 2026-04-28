"""S3 storage service.

Issues presigned PUT URLs that the client uses to upload shot images directly
to S3 (or MinIO in dev). The API never proxies image bytes — that would burn
expensive Fargate CPU on data movement and add a synchronous round-trip to
every capture step.

Key layout:
    submissions/<submission_id>/shots/<shot_id>/<kind>.<ext>

Putting <shot_id> in the key (rather than just <kind>) means a re-take of the
same shot kind doesn't overwrite the previous capture — useful for forensic
audit when a user retakes a borderline shot."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from functools import lru_cache

import boto3
from botocore.client import Config
from botocore.exceptions import ClientError

from grader.settings import get_settings


@dataclass(frozen=True)
class PresignedPut:
    upload_url: str
    s3_key: str
    expires_at: datetime
    required_headers: dict[str, str]


_CONTENT_TYPE_TO_EXT = {
    "image/jpeg": "jpg",
    "image/png": "png",
    "image/webp": "webp",
}


def _ext_from_content_type(content_type: str) -> str:
    try:
        return _CONTENT_TYPE_TO_EXT[content_type]
    except KeyError as e:
        raise ValueError(f"unsupported content-type: {content_type}") from e


def shot_s3_key(
    submission_id: uuid.UUID,
    shot_id: uuid.UUID,
    kind: str,
    content_type: str,
) -> str:
    ext = _ext_from_content_type(content_type)
    return f"submissions/{submission_id}/shots/{shot_id}/{kind}.{ext}"


@lru_cache(maxsize=1)
def _s3_client():
    s = get_settings()
    return boto3.client(
        "s3",
        region_name=s.s3_region,
        endpoint_url=s.s3_endpoint_url,
        aws_access_key_id=s.s3_access_key_id,
        aws_secret_access_key=s.s3_secret_access_key,
        config=Config(
            signature_version="s3v4",
            s3={"addressing_style": "path" if s.s3_force_path_style else "auto"},
        ),
    )


def reset_s3_client_cache() -> None:
    """Clear the cached client. Used by tests that override settings."""
    _s3_client.cache_clear()


def presigned_put_for_shot(
    submission_id: uuid.UUID,
    shot_id: uuid.UUID,
    kind: str,
    content_type: str,
) -> PresignedPut:
    """Generate a presigned PUT URL for direct client → S3 upload.

    The returned `required_headers` MUST be set verbatim by the client on the
    PUT — they're part of the signature."""
    settings = get_settings()
    key = shot_s3_key(submission_id, shot_id, kind, content_type)
    ttl = settings.s3_presigned_url_ttl_seconds

    url = _s3_client().generate_presigned_url(
        ClientMethod="put_object",
        Params={
            "Bucket": settings.s3_bucket,
            "Key": key,
            "ContentType": content_type,
        },
        ExpiresIn=ttl,
        HttpMethod="PUT",
    )
    return PresignedPut(
        upload_url=url,
        s3_key=key,
        expires_at=datetime.now(timezone.utc) + timedelta(seconds=ttl),
        required_headers={"Content-Type": content_type},
    )


def head_shot(s3_key: str) -> dict[str, str | int] | None:
    """Confirm an object exists at the given key. Returns metadata or None."""
    settings = get_settings()
    try:
        head = _s3_client().head_object(Bucket=settings.s3_bucket, Key=s3_key)
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code")
        if code in {"404", "NoSuchKey", "NotFound"}:
            return None
        raise
    return {
        "content_length": int(head.get("ContentLength", 0)),
        "content_type": str(head.get("ContentType", "")),
        "etag": str(head.get("ETag", "")).strip('"'),
    }


def get_shot_bytes(s3_key: str) -> bytes:
    settings = get_settings()
    obj = _s3_client().get_object(Bucket=settings.s3_bucket, Key=s3_key)
    return obj["Body"].read()
