"""Counterfeit-detection service.

Wraps the ML rosette detector behind an S3-aware API so the pipeline
worker (Celery) can analyze a canonical card image straight from object
storage without having to know about cv2 / numpy.

This is the first detector in the planned 7-detector counterfeit-
authenticity ensemble (see docs/plans). v1 only implements the print-
rosette FFT detector — additional detectors (color profile, typography,
embedding anomaly, etc.) will land as siblings in this module.

Persistence is intentionally NOT done here: the orchestrator
(`grader.workers.pipeline_runner`) decides how to persist counterfeit
results and is owned by another agent. This module only computes."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

from grader.services import storage

_ML_ROOT = Path(__file__).resolve().parents[3] / "ml"
if str(_ML_ROOT) not in sys.path:
    sys.path.insert(0, str(_ML_ROOT))

from pipelines.counterfeit.rosette import (  # noqa: E402
    RosetteMeasurement,
    measure_rosette,
)


class CounterfeitFailedError(Exception):
    """Raised when the counterfeit-analysis pipeline cannot complete.

    Distinct from a "looks-counterfeit" result — that's a low rosette_score,
    not an exception. This error means the image couldn't be loaded /
    decoded / analyzed at all (bad bytes, missing key, image too small)."""


def _load_canonical_bgr(s3_key: str) -> np.ndarray:
    """Load and decode a canonical card image from S3.

    Duplicated from grader.services.grading._load_canonical_bgr to avoid
    cross-service coupling — both modules want the same thing but neither
    should be a transitive dependency of the other. If a third service
    needs this, the function will be moved to grader.services.storage.
    Until then, the duplication is cheap."""
    raw = storage.get_shot_bytes(s3_key)
    arr = np.frombuffer(raw, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise CounterfeitFailedError(f"could not decode canonical at {s3_key}")
    return image


def analyze_rosette(canonical_s3_key: str) -> RosetteMeasurement:
    """Run the print-rosette FFT counterfeit detector on a canonical image.

    Args:
        canonical_s3_key: S3 key of the dewarped 750x1050 BGR canonical
            image produced by Stage 2 (perspective correction).

    Returns:
        RosetteMeasurement with a score in [0, 1] (higher = more likely
        authentic) and an analyzed-patch confidence in [0, 1].

    Raises:
        CounterfeitFailedError: if the canonical can't be loaded/decoded
            or the image fails the detector's input-validation gate.
    """
    image = _load_canonical_bgr(canonical_s3_key)
    try:
        return measure_rosette(image)
    except ValueError as e:
        # measure_rosette raises ValueError for shape/dtype/size issues. We
        # surface those as service-level failures so the worker can route
        # them the same as load errors.
        raise CounterfeitFailedError(
            f"rosette analysis failed for {canonical_s3_key}: {e}"
        ) from e
