"""Accumulate PSA-graded card images as additional authentic
exemplars in the reference-embeddings store.

Why this exists:
    Embedding-anomaly's signal scales with how many reference embeddings
    it has per variant (`_CONFIDENCE_BY_NREFS` in the detector). The
    manufacturer reference scrapes seed each variant with one canonical
    print from Scryfall / PokemonTCG.io. Every PSA-graded submission
    that we identify to a known variant is, by definition, another
    authentic exemplar — appending those embeddings broadens the
    centroid coverage and bumps detector confidence over time without
    any new scraping.

The flow:
    1. Iterate `scraped.jsonl` from the PSA daily ingest.
    2. For each record with an on-disk `front_image_path`, load the
       image, run the standard identification pipeline against the
       provided catalog.
    3. If identification picks a confident variant, encode the image
       (the embedder used here is the SAME one identification uses, so
       the embedding lives in the same space the inference path looks
       up) and append it under that variant's key in the npz archive.
    4. Record the consumed cert_id in a sidecar log so re-running the
       script on the same JSONL is a no-op for already-seen certs.

Storage shape:
    The reference-embeddings npz stores either a `(d,)` array (the
    single-printing seed from a manufacturer scrape) or a `(n_refs, d)`
    stack. `lookup_references` already accepts either shape; we expand
    `(d,)` to `(1, d)` on first append so the archive only stores 2-D
    arrays after that.

Idempotency:
    `psa_exemplars_log.jsonl` is the source of truth for "this cert
    contributed an exemplar." Before processing a record we check the
    log; we never re-add. Crash-safe via the same write-temp-rename
    pattern used by `LocalReferenceStore.write_image`.

Why not just write the embedding back into the manufacturer
references.jsonl?
    references.jsonl is a manufacturer catalog manifest — one row per
    Scryfall / PokemonTCG.io printing, with provenance fields (source
    URL, fetched_at, set/rarity). PSA-graded submissions don't fit
    that shape. The npz, by contrast, is the operational embedding
    store the inference path consults; it doesn't care where the
    embedding came from. Keeping the two stores aligned by `(manufacturer,
    variant_id)` key is the right composition.

Embedder note:
    The script accepts an embedder callable directly (Protocol-typed,
    not a concrete class) so the production CLI passes the real
    DinoV2-backed embedder while tests pass a deterministic fake. The
    embedder MUST be the same one identification uses at submission
    time, otherwise PSA exemplars and submitted-card embeddings would
    live in different spaces and the cosine distance would be
    meaningless.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol, TYPE_CHECKING

import cv2
import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from pipelines.identification.catalog import CatalogIndex

from data.ingestion.reference_embeddings import (
    _archive_key,
    _atomic_savez,
    _load_existing_archive,
)


_logger = logging.getLogger(__name__)


class _EmbedderLike(Protocol):
    """Same duck type used by reference_embeddings._EmbedderLike. Local
    redeclaration avoids a public-surface coupling."""

    @property
    def dim(self) -> int: ...

    def encode(self, image: NDArray[np.uint8]) -> NDArray[np.float32]: ...


@dataclass
class ExemplarPassStats:
    """Counters from one accumulator run.

    Fields:
      total_records: rows iterated in the PSA JSONL (modulo
        max_records).
      skipped_already_consumed: cert already in the log; never re-added.
      skipped_no_image: front_image_path missing, file gone, or
        undecodable as an image.
      skipped_unidentified: identification didn't pick a confident
        variant. Common — most PSA records won't match the
        manufacturer-reference catalog until catalog coverage grows.
      identification_failures: identification raised (e.g. shape /
        dtype error from the canonical loader). Rare; counted
        separately so a flood signals a pipeline bug.
      embedder_failures: embedder.encode raised.
      appended: exemplars freshly appended to the npz.
    """

    total_records: int = 0
    skipped_already_consumed: int = 0
    skipped_no_image: int = 0
    skipped_unidentified: int = 0
    identification_failures: int = 0
    embedder_failures: int = 0
    appended: int = 0


# --- Log helpers ----------------------------------------------------------


def _read_consumed_cert_ids(log_path: Path) -> set[int]:
    """One-shot scan of the sidecar log. Tolerates a half-written
    trailing line the same way LocalScrapedRecordStore does."""
    if not log_path.exists():
        return set()
    out: set[int] = set()
    with log_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            cert_id = obj.get("cert_id")
            if isinstance(cert_id, int):
                out.add(cert_id)
    return out


def _append_log_entry(log_path: Path, entry: dict) -> None:
    """Append a single JSONL line, fsync'd. Crash-safe: the worst case
    is a half-written trailing line that the next run skips on parse."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


# --- Image / record helpers -----------------------------------------------


def _read_image(path: str | None) -> NDArray[np.uint8] | None:
    """Load image off disk as BGR uint8. Returns None on any failure
    so the caller can route to skipped_no_image. Mirrors the helper in
    reference_embeddings but inlined to avoid reaching into a private
    helper."""
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


def _iter_psa_records(jsonl_path: Path):
    """Yield (cert_id, front_image_path) pairs for each parseable PSA
    record. Tolerates a half-written trailing line."""
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
            cert_id = obj.get("cert_id")
            front = obj.get("front_image_path")
            if not isinstance(cert_id, int):
                continue
            yield cert_id, front


# --- Append helper --------------------------------------------------------


def _append_exemplar(
    archive: dict[str, NDArray[np.float32]],
    key: str,
    new_vec: NDArray[np.float32],
) -> None:
    """Append `new_vec` (shape (d,)) under `key`. Promotes a 1-D entry
    to (1, d) before stacking so the archive only carries 2-D arrays
    after the first append."""
    new_2d = new_vec.reshape(1, -1).astype(np.float32, copy=False)
    existing = archive.get(key)
    if existing is None:
        archive[key] = new_2d
        return
    if existing.ndim == 1:
        existing = existing.reshape(1, -1)
    archive[key] = np.vstack([existing.astype(np.float32, copy=False), new_2d])


# --- Public entrypoint ----------------------------------------------------


def accumulate_psa_exemplars(
    psa_jsonl_path: Path | str,
    embeddings_store_path: Path | str,
    log_path: Path | str,
    *,
    catalog: "CatalogIndex",
    embedder: _EmbedderLike,
    max_records: int | None = None,
) -> ExemplarPassStats:
    """Walk a PSA scraped.jsonl, identify each record, and append its
    embedding to the reference-embeddings npz when identification
    picks a confident variant.

    Args:
        psa_jsonl_path: path to the JSONL produced by
            `data.ingestion.psa_public_api.ingest_range` /
            `LocalScrapedRecordStore`.
        embeddings_store_path: path to the npz archive maintained by
            `reference_embeddings.embed_references`. Created if missing.
        log_path: sidecar JSONL recording consumed cert_ids. Used as
            the idempotency cursor.
        catalog: identification catalog (any `CatalogIndex`).
            Production callers build this from card_variants + the
            on-disk reference_embeddings.npz; tests pass an
            `InMemoryCatalogIndex` directly.
        embedder: object exposing `.dim` and
            `.encode(uint8 BGR (H,W,3)) -> float32 (d,)`. MUST be the
            same embedder identification uses at submission time.
        max_records: optional ceiling for smoke runs.

    Returns:
        ExemplarPassStats with per-outcome counters.

    Raises:
        FileNotFoundError: if `psa_jsonl_path` doesn't exist.

    Side effects:
        - Rewrites the npz atomically once per call (only if any
          exemplar was appended; a no-op pass leaves the file alone).
        - Appends one line per appended exemplar to `log_path`.
    """
    psa_jsonl_path = Path(psa_jsonl_path)
    embeddings_store_path = Path(embeddings_store_path)
    log_path = Path(log_path)

    if not psa_jsonl_path.exists():
        raise FileNotFoundError(f"PSA scraped jsonl not found: {psa_jsonl_path}")

    # Defer the import so `data.ingestion.psa_exemplars` stays
    # importable in environments without the identification pipeline
    # available (e.g. ad-hoc stats scripts that don't call
    # `accumulate_psa_exemplars`). The function-level import keeps the
    # public surface lean.
    from pipelines.identification import identify

    consumed = _read_consumed_cert_ids(log_path)
    archive = _load_existing_archive(embeddings_store_path)
    stats = ExemplarPassStats()
    expected_dim = int(embedder.dim)

    dirty = False
    for cert_id, front in _iter_psa_records(psa_jsonl_path):
        stats.total_records += 1
        if max_records is not None and stats.total_records > max_records:
            stats.total_records -= 1  # don't count the row we won't process
            break

        if cert_id in consumed:
            stats.skipped_already_consumed += 1
            continue

        image = _read_image(front)
        if image is None:
            stats.skipped_no_image += 1
            continue

        try:
            result = identify(image, catalog=catalog, embedder=embedder)
        except Exception as e:  # noqa: BLE001 — identification failure tolerated
            stats.identification_failures += 1
            _logger.warning("identify failed cert_id=%d err=%s", cert_id, e)
            continue

        if result.chosen is None:
            stats.skipped_unidentified += 1
            continue

        # Reuse the embedding identification already computed when
        # available — saves a round-trip through the embedder. Falls
        # back to encoding here when identification short-circuited
        # on a confident pHash exact match (no embedding was computed).
        vec = result.submitted_embedding
        if vec is None:
            try:
                vec = embedder.encode(image)
            except Exception as e:  # noqa: BLE001
                stats.embedder_failures += 1
                _logger.warning("embedder failed cert_id=%d err=%s", cert_id, e)
                continue

        vec = np.asarray(vec, dtype=np.float32)
        if vec.ndim != 1 or vec.shape[0] != expected_dim:
            stats.embedder_failures += 1
            _logger.warning(
                "bad embedding shape cert_id=%d shape=%s expected=(%d,)",
                cert_id,
                vec.shape,
                expected_dim,
            )
            continue

        manufacturer = result.chosen.entry.game
        variant_id = result.chosen.entry.variant_id
        key = _archive_key(manufacturer, variant_id)
        _append_exemplar(archive, key, vec)
        consumed.add(cert_id)
        stats.appended += 1
        dirty = True

        _append_log_entry(
            log_path,
            {
                "cert_id": cert_id,
                "manufacturer": manufacturer,
                "variant_id": variant_id,
                "added_at": datetime.now(timezone.utc)
                .isoformat()
                .replace("+00:00", "Z"),
            },
        )

    if dirty:
        _atomic_savez(embeddings_store_path, archive)

    return stats


__all__ = [
    "ExemplarPassStats",
    "accumulate_psa_exemplars",
]
