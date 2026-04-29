"""GitHub seed-corpus ingestor — cold-start training data from a public repo.

Why this exists:
    Until the PSA Public API drip (free tier ~50 cert triples/day) builds
    enough volume to train on, we need *something* to validate the
    grading-model trainer end-to-end. The samsilverman/PSA-Baseball-Grades
    repo (https://github.com/samsilverman/PSA-Baseball-Grades) ships
    11,500 PSA-graded baseball card images already paired with grade
    labels — 1,150 per integer grade 1..10. That's enough to smoke-test
    the training loop, the data loaders, and the model's coarse
    grade-regression behavior before any real production data lands.

What this is NOT:
    Production training data. The README's own "Known Issues" section
    admits dupes, label noise, and irrelevant images (hands, tables,
    grading cases). Images are 150×200 px — too low-res for our
    canonical 750×1050 grading pipeline. There is no front/back
    distinction; each image is just one face of one card. Integer grades
    only — no PSA half-grades. Use this corpus for SMOKE-TESTING only.
    Do not let it leak into evaluation sets.

Layout we ingest (verified 2026-04-28):
    psa1/psa1_<n>.jpg, psa2/psa2_<n>.jpg, ..., psa10/psa10_<n>.jpg

Output format:
    A JSONL manifest at `<out_root>/seed_manifest.jsonl`, one row per
    image:
        {"image_path": "<abs path>", "grade": 7, "source": "github_seed"}

    We deliberately do NOT emit `ScrapedRecord` rows. ScrapedRecord
    requires a card name, set, and front/back image distinction, none of
    which this corpus has. Forcing the data through that schema would
    require synthesizing fake metadata, which would silently corrupt any
    downstream code that distinguishes seed from API records.

License posture:
    The source repo has no LICENSE file (`license: None` per the GitHub
    API as of 2026-04-28). Treat as research/local-train only — do not
    redistribute. The `--allow-redistribute` knob is a tripwire: it
    forces the user to acknowledge they've checked the license posture
    themselves before any code path that would copy the data into a
    bucket / dataset card / public release.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

_logger = logging.getLogger("github_seed")


# Filename pattern: psa<grade>_<n>.<ext>; case-insensitive on extension.
_FILENAME_RE = re.compile(r"^psa(\d{1,2})_\d+\.(jpe?g|png|webp)$", re.IGNORECASE)

# Folder name pattern: psa<grade>
_FOLDER_RE = re.compile(r"^psa(\d{1,2})$", re.IGNORECASE)


@dataclass(frozen=True)
class SeedRecord:
    """One image from the seed corpus."""

    image_path: str
    grade: int
    source: str = "github_seed"


def iter_seed_records(repo_root: Path | str) -> Iterator[SeedRecord]:
    """Yield SeedRecord for every valid image in the repo.

    Pure function — no I/O beyond filesystem reads. Skips files that
    don't match the expected naming convention (the repo also contains
    .DS_Store, README.md, etc.). Logs but does not raise on
    folder/filename grade mismatch — those are the kind of bad-label
    case the README warns about, and we want to surface them in counts
    rather than abort.
    """
    repo_root = Path(repo_root)
    if not repo_root.is_dir():
        raise FileNotFoundError(f"repo_root does not exist: {repo_root}")

    for entry in sorted(repo_root.iterdir()):
        if not entry.is_dir():
            continue
        m = _FOLDER_RE.match(entry.name)
        if not m:
            continue
        folder_grade = int(m.group(1))
        if not (1 <= folder_grade <= 10):
            continue

        for img in sorted(entry.iterdir()):
            if not img.is_file():
                continue
            fm = _FILENAME_RE.match(img.name)
            if not fm:
                continue
            file_grade = int(fm.group(1))
            if file_grade != folder_grade:
                _logger.warning(
                    "grade mismatch: folder=%d file=%s — using folder grade",
                    folder_grade,
                    img.name,
                )
            yield SeedRecord(
                image_path=str(img.resolve()),
                grade=folder_grade,
            )


def write_manifest(
    repo_root: Path | str,
    out_root: Path | str,
    *,
    allow_redistribute: bool = False,
) -> Path:
    """Walk the seed repo and emit `seed_manifest.jsonl`.

    Returns the manifest path. Raises if `allow_redistribute=True` is
    passed without an explicit acknowledgement env var
    (`PSA_SEED_LICENSE_ACK=1`) — see the module docstring.
    """
    if allow_redistribute and os.environ.get("PSA_SEED_LICENSE_ACK") != "1":
        raise PermissionError(
            "allow_redistribute=True requires PSA_SEED_LICENSE_ACK=1 in the "
            "environment as an acknowledgement that the seed repo's license "
            "posture has been independently verified for your use case"
        )

    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    manifest_path = out_root / "seed_manifest.jsonl"

    count_per_grade: dict[int, int] = {}
    total = 0
    with manifest_path.open("w", encoding="utf-8") as fh:
        for rec in iter_seed_records(repo_root):
            payload = json.dumps(
                {"image_path": rec.image_path, "grade": rec.grade, "source": rec.source},
                ensure_ascii=False,
            )
            fh.write(payload + "\n")
            count_per_grade[rec.grade] = count_per_grade.get(rec.grade, 0) + 1
            total += 1

    _logger.info(
        "seed manifest written path=%s total=%d per_grade=%s",
        manifest_path,
        total,
        dict(sorted(count_per_grade.items())),
    )
    return manifest_path


__all__ = [
    "SeedRecord",
    "iter_seed_records",
    "write_manifest",
]
