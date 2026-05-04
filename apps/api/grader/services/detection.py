"""Detection + dewarp service.

Bridges S3 storage and the ml/pipelines/detection module. The grading worker
(Celery, not yet wired) calls `detect_and_dewarp_shot` for each
quality-passed shot before any per-criterion grading runs.

We persist the canonical (dewarped) image back to S3 under
``submissions/<sub>/canonical/<shot_kind>.png`` so the per-criterion models
can pull it directly without re-running detection. The PNG format avoids any
JPEG re-encoding artifacts on a downstream pipeline that already trained on
JPEGs from the original capture.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from grader.db.models import ShotKind
from grader.services import storage
from grader.settings import get_settings
from pipelines.detection import (
    DetectedCard,
    DewarpResult,
    detect_card,
    dewarp_to_canonical,
)


@dataclass(frozen=True)
class DetectAndDewarpResult:
    detection: DetectedCard
    dewarp: DewarpResult
    canonical_s3_key: str


class DetectionFailedError(Exception):
    """No card found, or fill / irregularity outside acceptable range."""


def _canonical_s3_key(submission_id: str, shot_kind: ShotKind) -> str:
    return f"submissions/{submission_id}/canonical/{shot_kind.value}.png"


def detect_and_dewarp_shot(
    s3_key: str,
    shot_kind: ShotKind,
    *,
    max_irregularity: float = 0.15,
    persist_canonical: bool = True,
) -> DetectAndDewarpResult:
    """Run Stage 1 + Stage 2 on a previously-uploaded shot.

    Args:
        s3_key: location of the original capture.
        shot_kind: used to derive the canonical key on persist.
        max_irregularity: reject if the input quad is too non-rectangular
            (bent card / steep angle). 0 = flat overhead; 1 = degenerate.
        persist_canonical: when True, upload the dewarped image back to S3.
            Disable for ad-hoc analysis where you don't want side effects.

    Raises:
        DetectionFailedError: no card found or irregularity too high.
    """
    raw = storage.get_shot_bytes(s3_key)
    arr = np.frombuffer(raw, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise DetectionFailedError(f"could not decode {s3_key}")

    detection = detect_card(image)
    if detection is None:
        raise DetectionFailedError("no card detected")

    dewarp = dewarp_to_canonical(image, detection.quad)
    if dewarp.irregularity > max_irregularity:
        raise DetectionFailedError(
            f"quad too irregular ({dewarp.irregularity:.2f} > {max_irregularity}); "
            "card likely bent or shot at too steep an angle"
        )

    settings = get_settings()
    submission_id = s3_key.split("/")[1]
    canonical_key = _canonical_s3_key(submission_id, shot_kind)

    if persist_canonical:
        ok, buf = cv2.imencode(".png", dewarp.canonical)
        if not ok:
            raise DetectionFailedError("failed to encode canonical PNG")
        # Use the same S3 client config as the storage module.
        storage._s3_client().put_object(  # noqa: SLF001 — internal coupling is fine here
            Bucket=settings.s3_bucket,
            Key=canonical_key,
            Body=buf.tobytes(),
            ContentType="image/png",
        )

    return DetectAndDewarpResult(
        detection=detection,
        dewarp=dewarp,
        canonical_s3_key=canonical_key,
    )
