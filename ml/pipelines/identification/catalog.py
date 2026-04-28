"""Catalog index abstraction.

The catalog is the searchable store of known card variants — name, set,
canonical pHash, canonical embedding, etc. The production backend is
pgvector (see grader.services.catalog_repo); this module defines the
Protocol the identifier depends on, plus an in-memory implementation
used by tests.

The Protocol is intentionally synchronous. The pgvector adapter wraps
async SQLAlchemy calls behind a sync façade for tasks that run inside
Celery workers, where the surrounding code is sync."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from pipelines.identification.hashing import HASH_BITS, hamming_distance


@dataclass(frozen=True)
class CardCatalogEntry:
    """A row in the card_variants catalog. Mirrors the relevant subset of
    grader.db.models.CardVariant — kept here to avoid a cross-package
    dependency from ml/ → apps/api/."""

    variant_id: str
    name: str
    set_code: str
    card_number: str
    game: str
    canonical_phash: int | None = None
    canonical_embedding: NDArray[np.float32] | None = None
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class CatalogHit:
    """One candidate returned by a catalog query."""

    entry: CardCatalogEntry
    distance: float           # method-specific scalar (Hamming for pHash, cosine for embedding)
    method: str               # "phash" | "embedding"


class CatalogIndex(Protocol):
    def find_by_phash(self, phash: int, max_distance: int = 6, limit: int = 50) -> list[CatalogHit]: ...
    def nearest_by_embedding(self, embedding: NDArray[np.float32], top_k: int = 5) -> list[CatalogHit]: ...
    def get(self, variant_id: str) -> CardCatalogEntry | None: ...


# -----------------------------
# In-memory implementation
# -----------------------------


@dataclass
class InMemoryCatalogIndex:
    """Linear-scan catalog suitable for unit tests and small dev fixtures.

    Production lookups go through pgvector HNSW; this implementation only
    needs to be correct, not fast."""

    entries: list[CardCatalogEntry] = field(default_factory=list)

    def add(self, entry: CardCatalogEntry) -> None:
        self.entries.append(entry)

    def get(self, variant_id: str) -> CardCatalogEntry | None:
        for e in self.entries:
            if e.variant_id == variant_id:
                return e
        return None

    def find_by_phash(self, phash: int, max_distance: int = 6, limit: int = 50) -> list[CatalogHit]:
        if phash < 0:
            raise ValueError("phash must be non-negative")
        hits: list[CatalogHit] = []
        for e in self.entries:
            if e.canonical_phash is None:
                continue
            d = hamming_distance(phash, e.canonical_phash)
            if d <= max_distance:
                hits.append(CatalogHit(entry=e, distance=float(d), method="phash"))
        hits.sort(key=lambda h: h.distance)
        return hits[:limit]

    def nearest_by_embedding(self, embedding: NDArray[np.float32], top_k: int = 5) -> list[CatalogHit]:
        if embedding.ndim != 1:
            raise ValueError(f"embedding must be 1-D, got shape {embedding.shape}")
        scored: list[tuple[float, CardCatalogEntry]] = []
        norm_q = float(np.linalg.norm(embedding)) or 1.0
        q = embedding / norm_q
        for e in self.entries:
            if e.canonical_embedding is None:
                continue
            v = e.canonical_embedding
            if v.shape != q.shape:
                continue
            norm_v = float(np.linalg.norm(v)) or 1.0
            cos_sim = float(np.dot(q, v) / norm_v)
            cos_dist = 1.0 - cos_sim
            scored.append((cos_dist, e))
        scored.sort(key=lambda x: x[0])
        return [
            CatalogHit(entry=e, distance=d, method="embedding") for d, e in scored[:top_k]
        ]

    def __len__(self) -> int:
        return len(self.entries)


def assert_hash_in_range(h: int) -> None:
    """Defensive check for callers building catalog entries from integers."""
    if not (0 <= h < (1 << HASH_BITS)):
        raise ValueError(f"hash {h} outside 0..2^{HASH_BITS}")
