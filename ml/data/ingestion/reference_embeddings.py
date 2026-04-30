"""Manufacturer reference embedding pass + storage.

Walks the `references.jsonl` produced by `LocalReferenceStore`, embeds
each variant's front image with the identification embedder, and
persists the result to a single numpy archive keyed by
`"<manufacturer>/<variant_id>"`. The counterfeit-detection
embedding-anomaly stage looks up reference embeddings here at
inference time so submitted-card embeddings and reference embeddings
live in the same space.

Storage choice — single .npz archive:
    `<data_dir>/reference_embeddings.npz` is one file shipping a
    mapping of key → float32 vector. `np.load(path, mmap_mode='r')`
    opens it lazily so a per-variant lookup is O(1) without
    materializing the whole table. Keyed access matches the
    inference-time call shape (the caller already knows the variant
    id from identification). This sidesteps the "what database"
    question until corpus size justifies one — when reference counts
    cross ~50k records and lookup latency starts to matter, swap to
    FAISS or pgvector behind the same `load_embeddings` /
    `lookup_references` surface, no caller changes required.

Atomicity:
    The npz is rewritten via write-temp-then-rename, so a crashed
    embedding pass leaves either the previous archive or the new one,
    never a half-written file. Idempotency is by archive key:
    re-running the pass over the same JSONL is a no-op unless
    `force=True` or new records have been appended since.

Embedder injection:
    `embed_references` takes an embedder callable directly rather
    than instantiating one — the production CLI passes the real
    DinoV2-backed embedder; the test suite passes a deterministic
    fake. This avoids pulling torch into pytest and keeps the
    embedder choice (Simple vs DinoV2) where it already lives, in
    `pipelines.identification.embedding.get_embedder`.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np
from numpy.typing import NDArray


_logger = logging.getLogger(__name__)


class _EmbedderLike(Protocol):
    """Minimal duck type matching `pipelines.identification.embedding.ImageEmbedder`.

    Anything with `.dim` and `.encode(uint8 BGR HxWx3) -> float32 (d,)`
    works. Declared locally to avoid eager-importing the identification
    pipeline (and through it, cv2-heavy modules) just to grab a type."""

    @property
    def dim(self) -> int: ...

    def encode(self, image: NDArray[np.uint8]) -> NDArray[np.float32]: ...


@dataclass
class EmbeddingPassStats:
    """Counters from a single `embed_references` pass.

    Fields:
      embedded: number of records freshly embedded this run.
      skipped_already_present: records skipped because the archive
        already had an embedding for that key (and force=False).
      image_load_failures: records whose `front_image_path` was
        missing, unreadable, or not decodable as an image. Counted,
        not raised — one bad image shouldn't abort a 5000-card pass.
      embedder_failures: records whose embedder call raised. Same
        tolerance policy as image-load failures.
    """

    embedded: int = 0
    skipped_already_present: int = 0
    image_load_failures: int = 0
    embedder_failures: int = 0


# JSONL → records helpers --------------------------------------------------


def _iter_jsonl_records(jsonl_path: Path):
    """Yield (manufacturer, variant_id, front_image_path) for each
    record in the JSONL. Tolerates blank/half-written trailing lines
    the same way `LocalReferenceStore._scan_variant_ids` does — a
    crash mid-write should not block the embedding pass."""
    if not jsonl_path.exists():
        return
    with jsonl_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            manufacturer = obj.get("manufacturer")
            variant_id = obj.get("variant_id")
            front = obj.get("front_image_path")
            if not isinstance(manufacturer, str) or not isinstance(variant_id, str):
                continue
            yield manufacturer, variant_id, front


def _archive_key(manufacturer: str, variant_id: str) -> str:
    """The npz key for a (manufacturer, variant_id) pair.

    np.savez accepts arbitrary string keys; we use a forward-slash
    join because it's human-readable when inspecting an archive and
    matches the on-disk image layout."""
    return f"{manufacturer}/{variant_id}"


def _parse_archive_key(key: str) -> tuple[str, str] | None:
    """Inverse of `_archive_key`. Returns None for malformed keys
    (defensive — an archive built by an older version with a
    different scheme shouldn't crash inference)."""
    if "/" not in key:
        return None
    manufacturer, _, variant_id = key.partition("/")
    if not manufacturer or not variant_id:
        return None
    return manufacturer, variant_id


# Archive read / write -----------------------------------------------------


def _load_existing_archive(store_path: Path) -> dict[str, NDArray[np.float32]]:
    """Load an existing npz into a plain dict. Returns an empty dict
    if the file doesn't exist. We materialize fully (vs mmap) because
    the next step is rewriting the archive with new entries appended,
    and np.savez requires concrete arrays."""
    if not store_path.exists():
        return {}
    with np.load(store_path, allow_pickle=False) as archive:
        return {k: archive[k].astype(np.float32, copy=False) for k in archive.files}


def _atomic_savez(store_path: Path, payload: dict[str, NDArray[np.float32]]) -> None:
    """Write the archive via temp-file + rename for crash safety.

    Implementation detail: `np.savez` auto-appends `.npz` to a target
    that doesn't already end in `.npz`. We name the temp file with a
    `.part.npz` extension so np.savez doesn't surprise-rename it, then
    swap atomically into place with `os.replace`."""
    store_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = store_path.with_name(store_path.name + ".part.npz")
    np.savez(str(tmp), **payload)
    os.replace(tmp, store_path)


def _read_image(path: str | None) -> NDArray[np.uint8] | None:
    """Load an image off disk as BGR uint8. Returns None on any
    failure (missing path, unreadable file, non-image bytes). Cv2's
    `imread` already returns None on failure; we wrap to also catch
    OSError-ish stuff from the path stat."""
    if not path:
        return None
    try:
        if not Path(path).exists():
            return None
        img = cv2.imread(path, cv2.IMREAD_COLOR)
    except Exception:  # noqa: BLE001 — image-load failures are tolerated
        return None
    if img is None:
        return None
    if img.dtype != np.uint8 or img.ndim != 3 or img.shape[2] != 3:
        return None
    return img


# Public surface ----------------------------------------------------------


def embed_references(
    jsonl_path: Path | str,
    store_path: Path | str,
    *,
    embedder: _EmbedderLike,
    force: bool = False,
) -> EmbeddingPassStats:
    """Embed every reference in `jsonl_path`, writing results to
    `store_path` (a `.npz` archive).

    Args:
        jsonl_path: path to a `references.jsonl` produced by
            `LocalReferenceStore`.
        store_path: path to the `.npz` archive to read+write.
            Created if missing; overwritten atomically on each call.
        embedder: anything with `.dim: int` and
            `.encode(uint8 HxWx3) -> float32 (d,)`. Production passes
            the DinoV2 embedder; tests pass a deterministic fake.
        force: if True, re-embed all records and overwrite existing
            entries. Default False — already-present keys are skipped.

    Returns:
        EmbeddingPassStats with per-outcome counters.

    Raises:
        FileNotFoundError: if `jsonl_path` doesn't exist. (An empty
            JSONL is fine — we just produce an empty archive.)
    """
    jsonl_path = Path(jsonl_path)
    store_path = Path(store_path)

    if not jsonl_path.exists():
        raise FileNotFoundError(f"references jsonl not found: {jsonl_path}")

    archive = _load_existing_archive(store_path)
    stats = EmbeddingPassStats()
    expected_dim = int(embedder.dim)

    dirty = False
    for manufacturer, variant_id, front in _iter_jsonl_records(jsonl_path):
        key = _archive_key(manufacturer, variant_id)
        if not force and key in archive:
            stats.skipped_already_present += 1
            continue

        image = _read_image(front)
        if image is None:
            stats.image_load_failures += 1
            _logger.warning(
                "image load failed key=%s path=%s", key, front
            )
            continue

        try:
            vec = embedder.encode(image)
        except Exception as e:  # noqa: BLE001 — embedder failures tolerated
            stats.embedder_failures += 1
            _logger.warning("embedder failed key=%s err=%s", key, e)
            continue

        vec = np.asarray(vec, dtype=np.float32)
        if vec.ndim != 1 or vec.shape[0] != expected_dim:
            stats.embedder_failures += 1
            _logger.warning(
                "embedder returned bad shape key=%s shape=%s expected=(%d,)",
                key,
                vec.shape,
                expected_dim,
            )
            continue

        archive[key] = vec
        stats.embedded += 1
        dirty = True

    # Always write at least once so callers can rely on the file
    # existing after a successful call (even if every record was
    # already present, an empty archive is still "the current state").
    if dirty or not store_path.exists():
        _atomic_savez(store_path, archive)

    return stats


def load_embeddings(
    store_path: Path | str,
) -> dict[tuple[str, str], NDArray[np.float32]]:
    """Load the entire archive into an in-memory dict keyed by
    `(manufacturer, variant_id)`.

    Suitable for offline analyses and tests. For inference paths that
    only need one variant, prefer `lookup_references` which avoids
    materializing the whole table."""
    store_path = Path(store_path)
    if not store_path.exists():
        return {}
    out: dict[tuple[str, str], NDArray[np.float32]] = {}
    with np.load(store_path, allow_pickle=False) as archive:
        for key in archive.files:
            parsed = _parse_archive_key(key)
            if parsed is None:
                continue
            out[parsed] = archive[key].astype(np.float32, copy=False)
    return out


def lookup_references(
    store_path: Path | str,
    manufacturer: str,
    variant_id: str,
) -> NDArray[np.float32] | None:
    """Return the reference embeddings for a single variant as a
    `(n_refs, d)` float32 array, or None if the variant has no
    embedding stored.

    Currently each variant has exactly one reference (the front-face
    artwork from the manufacturer catalog), so the returned array is
    `(1, d)`. The shape and the `n_refs`-first convention matches
    what `measure_embedding_anomaly` expects — once we accumulate
    multiple authentic exemplars per variant (e.g. from PSA-graded
    submissions cross-keyed to a variant), the storage layer can grow
    a `(k, d)` array under the same key and the consumer surface is
    unchanged."""
    store_path = Path(store_path)
    if not store_path.exists():
        return None
    key = _archive_key(manufacturer, variant_id)
    with np.load(store_path, allow_pickle=False) as archive:
        if key not in archive.files:
            return None
        vec = archive[key].astype(np.float32, copy=False)
    if vec.ndim == 1:
        return vec.reshape(1, -1)
    return vec


__all__ = [
    "EmbeddingPassStats",
    "embed_references",
    "load_embeddings",
    "lookup_references",
]
