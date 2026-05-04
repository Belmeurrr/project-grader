"""Labeled-corpus loader for the recalibration tool.

Two input shapes:
  - **CSV**: `image_path,ground_truth[,sample_id]`. The general path —
    works for any source, including hand-curated counterfeit lists.
  - **PSA scraped.jsonl**: every row with a `front_image_path` is a
    real authentic sample (PSA's grading is the ground-truth label).
    Convenience shortcut for the "I have authentic data, no fakes yet"
    starting state — you'd typically pass `--psa-authentics` and skip
    the CSV until counterfeit samples arrive.

Both shapes produce `BenchmarkSample`s the existing `run_benchmark`
runner accepts — so the recalibration tool runs the SAME detectors on
the SAME inputs that production sees, and recommended thresholds
correspond to actual production behavior at submission time.

Out-of-scope:
  - Embedding-anomaly inputs (would need a per-sample manufacturer +
    variant_id and a refs npz lookup; deferred until the existing
    `counterfeit_benchmark` corpus also supports it, to keep one
    detector-set across both tools).
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from evaluation.counterfeit_benchmark.corpus import (
    BenchmarkSample,
    GROUND_TRUTH_AUTHENTIC,
    GROUND_TRUTH_COUNTERFEIT,
)


_logger = logging.getLogger(__name__)


_VALID_LABELS = frozenset({GROUND_TRUTH_AUTHENTIC, GROUND_TRUTH_COUNTERFEIT})


@dataclass(frozen=True)
class LabeledRow:
    """Pre-image-load row description. Image is loaded later by
    `rows_to_samples` so corpus assembly can fail cheaply on
    metadata before paying I/O on potentially-thousands of files."""

    image_path: Path
    ground_truth: str  # one of _VALID_LABELS
    sample_id: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LoadStats:
    """Accounting for `rows_to_samples`. Surfaced in the report so the
    user knows when the corpus they thought they were running was
    silently truncated by load failures."""

    requested: int
    loaded: int
    skipped_missing_file: int
    skipped_unreadable: int


# --------------------------------------------------------------------------
# CSV loader
# --------------------------------------------------------------------------


def load_csv(path: Path) -> list[LabeledRow]:
    """Load a labels CSV.

    Required columns: `image_path`, `ground_truth`.
    Optional column: `sample_id` (defaults to filename stem).

    Header is required (uses csv.DictReader). Comments aren't supported;
    keep the file label-only.

    Raises:
        FileNotFoundError: if `path` doesn't exist.
        ValueError: if required columns are missing or any row's
            ground_truth is not 'authentic'/'counterfeit'.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"labels CSV not found: {path}")

    rows: list[LabeledRow] = []
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError(f"empty CSV: {path}")
        missing = {"image_path", "ground_truth"} - set(reader.fieldnames)
        if missing:
            raise ValueError(
                f"CSV {path} missing required column(s): {sorted(missing)}"
            )
        for line_idx, row in enumerate(reader, start=2):  # line 1 = header
            label = (row.get("ground_truth") or "").strip().lower()
            if label not in _VALID_LABELS:
                raise ValueError(
                    f"CSV {path} line {line_idx}: ground_truth={label!r} "
                    f"not in {sorted(_VALID_LABELS)}"
                )
            img_path = (row.get("image_path") or "").strip()
            if not img_path:
                raise ValueError(
                    f"CSV {path} line {line_idx}: empty image_path"
                )
            sid = (row.get("sample_id") or "").strip() or Path(img_path).stem
            rows.append(
                LabeledRow(
                    image_path=Path(img_path),
                    ground_truth=label,
                    sample_id=sid,
                    metadata={"source": "csv", "csv_line": line_idx},
                )
            )
    return rows


# --------------------------------------------------------------------------
# PSA authentics shortcut
# --------------------------------------------------------------------------


def load_psa_authentics(
    jsonl_path: Path,
    *,
    max_records: int | None = None,
) -> list[LabeledRow]:
    """Every PSA-graded record with a `front_image_path` is by
    definition an authentic exemplar (PSA's grade is the ground truth).
    This loader emits one `LabeledRow` per such record — no
    counterfeits.

    Useful for the "we have lots of authentics from the daily ingest,
    no counterfeits yet" starting state. Result: authentic-only mode
    in the sweep, which can recalibrate the AUTHENTIC threshold but
    not the COUNTERFEIT threshold.

    Args:
        jsonl_path: PSA `scraped.jsonl` from the daily ingest.
        max_records: optional ceiling for smoke runs.

    Tolerates a half-written trailing line (matches LocalScrapedRecordStore).

    Returns:
        LabeledRows for every parseable record with an on-disk
        `front_image_path`. Records with missing/empty
        front_image_path are silently skipped (they're typically
        non_gradeable / images_missing rows, not relevant here).
    """
    jsonl_path = Path(jsonl_path)
    if not jsonl_path.exists():
        raise FileNotFoundError(f"PSA scraped.jsonl not found: {jsonl_path}")

    rows: list[LabeledRow] = []
    with jsonl_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            cert_id = obj.get("cert_id")
            front = obj.get("front_image_path")
            if not isinstance(cert_id, int):
                continue
            if not isinstance(front, str) or not front:
                continue
            rows.append(
                LabeledRow(
                    image_path=Path(front),
                    ground_truth=GROUND_TRUTH_AUTHENTIC,
                    sample_id=f"psa-{cert_id}",
                    metadata={
                        "source": "psa_jsonl",
                        "cert_id": cert_id,
                        "grade": obj.get("grade"),
                    },
                )
            )
            if max_records is not None and len(rows) >= max_records:
                break
    return rows


# --------------------------------------------------------------------------
# Image loader → BenchmarkSample
# --------------------------------------------------------------------------


def _read_image(path: Path) -> NDArray[np.uint8] | None:
    """Load BGR uint8 (H,W,3). Returns None on any failure so the
    caller can route to skipped_unreadable."""
    if not path.exists():
        return None
    try:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    except Exception:  # noqa: BLE001
        return None
    if img is None:
        return None
    if img.dtype != np.uint8 or img.ndim != 3 or img.shape[2] != 3:
        return None
    return img


def rows_to_samples(
    rows: list[LabeledRow],
) -> tuple[list[BenchmarkSample], LoadStats]:
    """Read images off disk, return BenchmarkSamples + load accounting.

    Failed loads are dropped from the output and counted in `LoadStats`
    rather than raised; a single broken image shouldn't abort an
    overnight recalibration on a 1000-sample corpus. The skipped totals
    surface in the report so the user notices systemic problems
    (e.g. wrong --psa-data-dir).
    """
    samples: list[BenchmarkSample] = []
    skipped_missing = 0
    skipped_unreadable = 0

    for row in rows:
        if not row.image_path.exists():
            skipped_missing += 1
            continue
        img = _read_image(row.image_path)
        if img is None:
            skipped_unreadable += 1
            _logger.debug("unreadable image %s", row.image_path)
            continue
        samples.append(
            BenchmarkSample(
                sample_id=row.sample_id,
                ground_truth=row.ground_truth,
                variant=row.metadata.get("source", "labeled"),
                seed=0,
                image=img,
                metadata=row.metadata,
            )
        )

    stats = LoadStats(
        requested=len(rows),
        loaded=len(samples),
        skipped_missing_file=skipped_missing,
        skipped_unreadable=skipped_unreadable,
    )
    return samples, stats


__all__ = [
    "LabeledRow",
    "LoadStats",
    "load_csv",
    "load_psa_authentics",
    "rows_to_samples",
]
