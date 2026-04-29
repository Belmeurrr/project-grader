"""Storage backends for manufacturer reference card images.

A "reference" is a known-authentic catalog entry from the manufacturer's
own data source (Scryfall for MTG, PokemonTCG.io for Pokemon, etc.).
Distinct from a `ScrapedRecord` (which represents a PSA grading event):
references describe the platonic card variant — the artwork, set,
collector number — not a graded slab. Two separate dataclasses on
purpose: the field shapes don't overlap meaningfully, and pretending
otherwise would force unhelpful conditionals on consumers.

Why JSONL + filesystem:
    Same trade-off as the PSA store. Append-only, line-oriented, crash-
    safe. Resume is "scan the file once, build a set of variant_ids
    already done." Pandas / DuckDB / DVC consume it without ceremony.

Image key layout:
    <prefix>/images/<manufacturer>/<variant_id>/<kind>.<ext>
    where:
      manufacturer in {"mtg", "pokemon", ...} — the source brand
      variant_id  is the source-stable per-printing id (Scryfall UUID,
                  PokemonTCG.io "<set>-<num>", etc.). Stable means
                  re-running the ingest doesn't relocate the bytes.
      kind        is "front" almost always. Manufacturer catalogs
                  rarely expose card backs (they're shared per game),
                  so back is reserved for completeness, not in active
                  use.

Resume protocol:
    `iter_existing_variant_ids()` returns the (manufacturer, variant_id)
    tuples already persisted. The ingest unions this with the cards
    returned by the source query and skips those already done.
    `has_record(manufacturer, variant_id)` is the cheap per-card check
    used inside the ingest loop.

Why stdlib-only:
    Mirrors the eager-load policy in `data.ingestion.__init__`: a
    consumer that just wants the storage classes shouldn't have to
    pay for httpx. The Scryfall / PokemonTCG.io clients (which DO
    use httpx) live in sibling modules and are lazy-loaded.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class ReferenceRecord:
    """Single successfully-fetched manufacturer reference. Persisted as
    one JSONL line.

    Fields:
      variant_id: source-stable per-printing identifier. Scryfall returns
        a UUID (string form); PokemonTCG.io returns "<set_id>-<number>".
        Unique within `manufacturer` but not across manufacturers, so the
        full key is the (manufacturer, variant_id) tuple.
      manufacturer: brand short-name. Currently "mtg" or "pokemon"; new
        sources land as new short-names (e.g. "ygo" for Yu-Gi-Oh).
      name: human-readable card name ("Black Lotus").
      set_code: short set identifier ("LEA"). May be None for unusual
        catalog entries (token cards, art series, alters); keep the
        record but don't fail.
      collector_number: per-set collector number ("232", "232a", "T1").
        Strings, not ints — Scryfall and PokemonTCG.io both use
        suffixes for variants/promos.
      rarity: "common", "uncommon", "rare", "mythic", etc. Useful for
        embedding-anomaly cluster diversity (we want rare + common
        printings to span the appearance space). May be None if the
        source doesn't report rarity.
      front_image_path: storage URI of the downloaded front image, or
        None if image fetch failed (the record is still persisted —
        we know the variant exists even if we couldn't grab bytes).
      source_url: the API permalink the metadata came from. Used for
        debug + provenance (e.g., re-fetching when a parse changes).
      fetched_at: ISO8601 UTC, includes "Z" suffix.
    """

    variant_id: str
    manufacturer: str
    name: str
    set_code: str | None
    collector_number: str | None
    rarity: str | None
    front_image_path: str | None
    source_url: str
    fetched_at: str


_CONTENT_TYPE_TO_EXT = {
    "image/jpeg": "jpg",
    "image/jpg": "jpg",
    "image/png": "png",
    "image/webp": "webp",
}


def _ext_from_content_type(content_type: str) -> str:
    """Map an HTTP Content-Type to a file extension. Falls back to
    "bin" rather than raising; an unknown content-type is not a reason
    to abort an entire ingest run."""
    if not content_type:
        return "bin"
    primary = content_type.split(";", 1)[0].strip().lower()
    return _CONTENT_TYPE_TO_EXT.get(primary, "bin")


def reference_image_relative_key(
    manufacturer: str,
    variant_id: str,
    kind: str,
    content_type: str,
) -> str:
    """Return the relative storage path for an image of a given variant.

    Pure helper so callers can format URIs without instantiating a
    store (e.g. when planning a manifest or running dry-run analyses)."""
    ext = _ext_from_content_type(content_type)
    return f"images/{manufacturer}/{variant_id}/{kind}.{ext}"


@runtime_checkable
class ReferenceRecordStore(Protocol):
    """Storage backend Protocol for manufacturer reference records.

    Implementations MUST be safe for single-process appends. Multi-
    process coordination is out of scope at v1 — run one ingest per
    backend. (If we later need parallelism, the obvious split is by
    manufacturer or by set, with each shard writing its own JSONL.)
    """

    def write_record(self, record: ReferenceRecord) -> None:
        """Append `record` durably. Must be atomic per-record."""

    def write_image(
        self,
        manufacturer: str,
        variant_id: str,
        kind: str,
        content_type: str,
        data: bytes,
    ) -> str:
        """Persist image bytes; return the storage URI/key written."""

    def has_record(self, manufacturer: str, variant_id: str) -> bool:
        """Cheap idempotency check used by the ingest loop."""

    def iter_existing_variant_ids(self) -> set[tuple[str, str]]:
        """One-shot scan of all already-persisted (manufacturer, variant_id)
        tuples. Used for resume."""


class LocalReferenceStore:
    """Filesystem-backed reference store. Default for dev and CI fixtures.

    Layout under `root`:
        references.jsonl                                  one ReferenceRecord per line
        images/<manufacturer>/<variant_id>/<kind>.<ext>   image bytes

    `has_record` is implemented by maintaining an in-memory set of
    (manufacturer, variant_id) tuples, populated lazily on first call
    from the JSONL. The set is updated on each `write_record` so
    subsequent `has_record` calls reflect the current process's writes.
    """

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "images").mkdir(parents=True, exist_ok=True)
        self._records_path = self.root / "references.jsonl"
        self._lock = threading.Lock()
        self._cache: set[tuple[str, str]] | None = None

    @property
    def records_path(self) -> Path:
        return self._records_path

    def _ensure_cache(self) -> set[tuple[str, str]]:
        if self._cache is None:
            self._cache = self._scan_variant_ids()
        return self._cache

    def _scan_variant_ids(self) -> set[tuple[str, str]]:
        keys: set[tuple[str, str]] = set()
        if not self._records_path.exists():
            return keys
        with self._records_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    # Tolerate a half-written trailing line from a crash.
                    continue
                m = obj.get("manufacturer")
                v = obj.get("variant_id")
                if isinstance(m, str) and isinstance(v, str):
                    keys.add((m, v))
        return keys

    def write_record(self, record: ReferenceRecord) -> None:
        payload = json.dumps(asdict(record), ensure_ascii=False)
        with self._lock:
            with self._records_path.open("a", encoding="utf-8") as fh:
                fh.write(payload + "\n")
                fh.flush()
                os.fsync(fh.fileno())
            cache = self._ensure_cache()
            cache.add((record.manufacturer, record.variant_id))

    def write_image(
        self,
        manufacturer: str,
        variant_id: str,
        kind: str,
        content_type: str,
        data: bytes,
    ) -> str:
        rel = reference_image_relative_key(manufacturer, variant_id, kind, content_type)
        target = self.root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        # write-then-rename for atomicity vs partial writes during crash
        tmp = target.with_suffix(target.suffix + ".part")
        with tmp.open("wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, target)
        return str(target)

    def has_record(self, manufacturer: str, variant_id: str) -> bool:
        return (manufacturer, variant_id) in self._ensure_cache()

    def iter_existing_variant_ids(self) -> set[tuple[str, str]]:
        # Force a fresh scan; resume is a once-per-run operation so the
        # cost is acceptable, and this guarantees we see records written
        # by a previous process.
        self._cache = self._scan_variant_ids()
        return set(self._cache)


__all__ = [
    "LocalReferenceStore",
    "ReferenceRecord",
    "ReferenceRecordStore",
    "reference_image_relative_key",
]
