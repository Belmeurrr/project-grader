"""S3 storage service.

Issues presigned POST forms that the client uses to upload shot images directly
to S3 (or MinIO in dev). The API never proxies image bytes — that would burn
expensive Fargate CPU on data movement and add a synchronous round-trip to
every capture step.

POST (vs PUT) is used because the presigned-POST policy lets us pin a
``content-length-range`` condition so a malicious client can't push a 5 GB
upload through what we intended to be a ~25 MiB phone-photo. The signature
covers the policy document, so S3 enforces the size cap server-side regardless
of what the client claims.

Key layout:
    submissions/<submission_id>/shots/<shot_id>/<kind>.<ext>

Putting <shot_id> in the key (rather than just <kind>) means a re-take of the
same shot kind doesn't overwrite the previous capture — useful for forensic
audit when a user retakes a borderline shot."""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from functools import lru_cache

import boto3
from botocore.client import Config
from botocore.exceptions import BotoCoreError, ClientError

from grader.settings import get_settings

logger = logging.getLogger(__name__)


class StoragePresignError(Exception):
    """Raised when presigned-URL generation fails because the S3 client
    itself is unhealthy (bad credentials, missing bucket, network).

    Distinct from a successful presign whose URL later 403s on PUT —
    that's a bucket policy / CORS issue, not something we can detect
    here. Route handlers should map this to 503 with a retry hint, the
    same shape as broker-failure handling in submit_submission."""


@dataclass(frozen=True)
class PresignedPost:
    """Presigned POST form payload for direct browser → S3 uploads.

    ``url`` is the bucket endpoint; ``fields`` is the multipart-form
    dictionary the client must POST verbatim, with the file blob added
    last under the field name ``file``. The signature covers the
    embedded policy document, which pins the ``content-length-range``
    bounds — S3 rejects oversized uploads server-side."""

    url: str
    fields: dict[str, str]
    s3_key: str
    expires_at: datetime


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


def presigned_post_for_shot(
    submission_id: uuid.UUID,
    shot_id: uuid.UUID,
    kind: str,
    content_type: str,
) -> PresignedPost:
    """Generate a presigned POST form for direct client → S3 upload.

    A ``content-length-range`` condition is baked into the policy and
    signed alongside it: S3 rejects any upload smaller than 1 byte or
    larger than ``settings.submission_max_image_bytes``. Without this
    a client holding a presigned URL could PUT an arbitrarily large
    object to our bucket. ``register_shot`` applies a defense-in-depth
    HEAD check on the same setting after the upload completes."""
    settings = get_settings()
    key = shot_s3_key(submission_id, shot_id, kind, content_type)
    ttl = settings.s3_presigned_url_ttl_seconds

    # ``generate_presigned_post`` happily returns a URL even when AWS
    # credentials are wrong or the bucket is gone — the breakage only
    # surfaces on the client's PUT as an opaque 403. Wrapping the call
    # surfaces that earlier as a 503 with a retry hint.
    try:
        response = _s3_client().generate_presigned_post(
            Bucket=settings.s3_bucket,
            Key=key,
            Fields={"Content-Type": content_type},
            Conditions=[
                ["content-length-range", 1, settings.submission_max_image_bytes],
                {"Content-Type": content_type},
            ],
            ExpiresIn=ttl,
        )
    except (ClientError, BotoCoreError) as e:
        error_code = (
            e.response.get("Error", {}).get("Code") if isinstance(e, ClientError) else None
        )
        logger.error(
            "presigned_post_for_shot failed",
            extra={
                "operation": "generate_presigned_post",
                "bucket": settings.s3_bucket,
                "s3_key": key,
                "error_code": error_code,
            },
        )
        raise StoragePresignError(
            f"could not presign upload for {key}: {e}"
        ) from e
    return PresignedPost(
        url=response["url"],
        fields=response["fields"],
        s3_key=key,
        expires_at=datetime.now(timezone.utc) + timedelta(seconds=ttl),
    )


def presigned_get_for_canonical(
    s3_key: str,
    *,
    expires_in_seconds: int = 3600,  # 1 hour
) -> str | None:
    """Return a presigned GET URL for a canonical image, or None if
    generation fails.

    Public-facing — used by the cert page to render the actual card scan
    with bounded blast radius. Unlike ``presigned_post_for_shot`` (which
    raises a typed error for the route handler to map to 503), this
    helper is best-effort: a missing S3 client / bad creds should NOT
    503 the public cert page just because we can't display the image.
    The cert page degrades gracefully to a placeholder when the URL is
    None.

    Expiry defaults to 1 hour; the cert endpoint pairs this with a
    ``max-age`` short enough to leave a comfortable margin before the
    URL would become stale (see ``_PUBLIC_CACHE_HEADER`` in the cert
    router for the trade-off)."""
    settings = get_settings()
    try:
        url = _s3_client().generate_presigned_url(
            "get_object",
            Params={"Bucket": settings.s3_bucket, "Key": s3_key},
            ExpiresIn=expires_in_seconds,
        )
    except (ClientError, BotoCoreError) as e:
        error_code = (
            e.response.get("Error", {}).get("Code") if isinstance(e, ClientError) else None
        )
        logger.warning(
            "presigned_get_for_canonical failed",
            extra={
                "operation": "generate_presigned_url",
                "bucket": settings.s3_bucket,
                "s3_key": s3_key,
                "error_code": error_code,
            },
        )
        return None
    return url


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


class CanonicalLoadError(Exception):
    """Raised when a canonical card image can't be loaded or decoded.

    Distinct from any service-level error (e.g. GradingFailedError,
    CounterfeitFailedError); each calling service should catch this
    and re-raise as their own typed error so their callers can route
    failures uniformly without learning about storage internals."""


def load_canonical_bgr(s3_key: str):
    """Load and decode a canonical card image from S3 → BGR ndarray.

    Single home for what used to live duplicated in
    grader.services.{grading,counterfeit} as `_load_canonical_bgr`.
    Now there are three callers (grading + 2 counterfeit detectors)
    and the duplication is no longer cheap.

    cv2 + numpy are imported lazily so storage.py stays light when
    callers only need the presigned-URL / bytes-fetch helpers."""
    import cv2
    import numpy as np

    raw = get_shot_bytes(s3_key)
    arr = np.frombuffer(raw, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise CanonicalLoadError(f"could not decode canonical at {s3_key}")
    return image
