"""Quality gating service.

After a client uploads a shot to S3 via a presigned URL, it calls
POST /submissions/{id}/shots to register the shot. This service:
  1. Confirms the upload landed in S3.
  2. Pulls the bytes, decodes the image.
  3. Runs the quality pipeline (blur, glare, card detection, perspective).
  4. Records the result on submission_shots.

If quality fails, the shot is still recorded but `quality_passed=False` and
the reasons are saved in `quality_metadata`. The client UI shows the reasons
and prompts the user to retake.

The grading pipeline only runs when *every* required shot has
`quality_passed=True`.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from grader.db.models import ShotKind
from grader.services import storage

# Pipelines are inside ml/, which is a separate package outside apps/api/.
# In dev/test we add ml/ to sys.path; in prod the worker container bundles
# ml/ via Docker COPY.
_ML_ROOT = Path(__file__).resolve().parents[4] / "ml"
if str(_ML_ROOT) not in sys.path:
    sys.path.insert(0, str(_ML_ROOT))

from pipelines.quality import QualityReport, evaluate_shot  # noqa: E402


@dataclass(frozen=True)
class ShotIngestionResult:
    quality: QualityReport
    width_px: int
    height_px: int
    s3_etag: str | None


class ShotNotFoundInS3Error(Exception):
    """The client claimed it uploaded but the S3 object isn't there."""


class ShotDecodeError(Exception):
    """The bytes at the S3 key are not a decodable image."""


def ingest_shot(s3_key: str, shot_kind: ShotKind) -> ShotIngestionResult:
    head = storage.head_shot(s3_key)
    if head is None:
        raise ShotNotFoundInS3Error(s3_key)

    raw = storage.get_shot_bytes(s3_key)
    arr = np.frombuffer(raw, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise ShotDecodeError(s3_key)

    report = evaluate_shot(image, shot_kind.value)
    h, w = image.shape[:2]
    return ShotIngestionResult(
        quality=report,
        width_px=int(w),
        height_px=int(h),
        s3_etag=str(head.get("etag")) if head.get("etag") else None,
    )
