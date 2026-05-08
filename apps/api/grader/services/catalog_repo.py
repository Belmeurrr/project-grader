"""Catalog repository — pgvector-backed implementation of `CatalogIndex`.

Two query patterns:

  1. pHash pre-filter — Hamming distance via Postgres `bit_count(a # b)`
     where `a` and `b` are bigints. Postgres bigint is signed 64-bit, so
     hashes >= 2^63 are stored as their two's-complement signed form;
     `bit_count` operates on the bit pattern regardless of sign, which
     gives the correct Hamming distance.

  2. Embedding nearest-neighbor — pgvector `<=>` (cosine distance)
     against the HNSW index on `card_variants.canonical_image_embedding`.

The repo exposes a sync façade (`PgVectorCatalogIndex`) suitable for
calling from a Celery worker. Internally it bridges to async SQLAlchemy
via `asyncio.run` per call (workers run one card at a time, so the
performance cost is negligible compared to the actual model inference)."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sqlalchemy import bindparam, select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from grader.db.models import CardVariant
from pipelines.identification.catalog import (
    CardCatalogEntry,
    CatalogHit,
)
from pipelines.identification.hashing import HASH_BITS

_SIGNED_64_OFFSET = 1 << 64
_SIGNED_64_MAX = (1 << 63) - 1


def unsigned_to_signed_64(u: int) -> int:
    """Convert a 64-bit unsigned hash to a Postgres-bigint-friendly signed int."""
    if not (0 <= u < (1 << HASH_BITS)):
        raise ValueError(f"value {u} out of unsigned 64-bit range")
    return u if u <= _SIGNED_64_MAX else u - _SIGNED_64_OFFSET


def signed_to_unsigned_64(s: int) -> int:
    if s < 0:
        return s + _SIGNED_64_OFFSET
    return s


@dataclass
class PgVectorCatalogIndex:
    """Sync façade around async SQLAlchemy catalog queries.

    The session_factory is the application's async_sessionmaker; we open
    a fresh session per call so connections aren't held across the long
    inference work that happens between catalog hits."""

    session_factory: async_sessionmaker[AsyncSession]

    # ------------------------------------------------------------------
    # Sync API (matches CatalogIndex Protocol)
    # ------------------------------------------------------------------

    def find_by_phash(
        self, phash: int, max_distance: int = 6, limit: int = 50
    ) -> list[CatalogHit]:
        return _run(self._find_by_phash_async(phash, max_distance, limit))

    def nearest_by_embedding(
        self, embedding: NDArray[np.float32], top_k: int = 5
    ) -> list[CatalogHit]:
        return _run(self._nearest_by_embedding_async(embedding, top_k))

    def get(self, variant_id: str) -> CardCatalogEntry | None:
        return _run(self._get_async(variant_id))

    # ------------------------------------------------------------------
    # Async implementations
    # ------------------------------------------------------------------

    async def _find_by_phash_async(
        self, phash: int, max_distance: int, limit: int
    ) -> list[CatalogHit]:
        if phash < 0 or phash >= (1 << HASH_BITS):
            raise ValueError(f"phash {phash} out of unsigned 64-bit range")
        signed = unsigned_to_signed_64(phash)

        # Hamming distance via bit_count on the XOR. Postgres 14+ ships
        # bit_count(bytea) and bit_count(bit) — there is NO
        # bit_count(bigint), so we have to cast the XOR result. We
        # bitand-mask to 64 bits first because Postgres's `#` operator
        # returns a signed bigint that the `bit(64)` cast won't accept
        # negative values for.
        sql = text(
            """
            SELECT
              cv.id, cv.name, cv.card_number, cv.game,
              cs.code AS set_code,
              cv.canonical_phash,
              cv.canonical_image_embedding,
              cv.metadata,
              bit_count(
                ((cv.canonical_phash # :probe)::bit(64))::varbit
              ) AS distance
            FROM card_variants cv
            JOIN card_sets cs ON cs.id = cv.set_id
            WHERE cv.canonical_phash IS NOT NULL
              AND bit_count(
                ((cv.canonical_phash # :probe)::bit(64))::varbit
              ) <= :max_distance
            ORDER BY distance ASC
            LIMIT :limit
            """
        ).bindparams(
            bindparam("probe", signed),
            bindparam("max_distance", max_distance),
            bindparam("limit", limit),
        )

        async with self.session_factory() as session:
            rows = (await session.execute(sql)).mappings().all()

        return [
            CatalogHit(
                entry=_row_to_entry(row),
                distance=float(row["distance"]),
                method="phash",
            )
            for row in rows
        ]

    async def _nearest_by_embedding_async(
        self, embedding: NDArray[np.float32], top_k: int
    ) -> list[CatalogHit]:
        if embedding.ndim != 1:
            raise ValueError(f"embedding must be 1-D, got shape {embedding.shape}")
        vec = embedding.astype(np.float32).tolist()

        sql = text(
            """
            SELECT
              cv.id, cv.name, cv.card_number, cv.game,
              cs.code AS set_code,
              cv.canonical_phash,
              cv.canonical_image_embedding,
              cv.metadata,
              cv.canonical_image_embedding <=> CAST(:probe AS vector) AS distance
            FROM card_variants cv
            JOIN card_sets cs ON cs.id = cv.set_id
            WHERE cv.canonical_image_embedding IS NOT NULL
            ORDER BY cv.canonical_image_embedding <=> CAST(:probe AS vector)
            LIMIT :top_k
            """
        ).bindparams(
            bindparam("probe", str(vec)),
            bindparam("top_k", top_k),
        )

        async with self.session_factory() as session:
            rows = (await session.execute(sql)).mappings().all()

        return [
            CatalogHit(
                entry=_row_to_entry(row),
                distance=float(row["distance"]),
                method="embedding",
            )
            for row in rows
        ]

    async def _get_async(self, variant_id: str) -> CardCatalogEntry | None:
        async with self.session_factory() as session:
            stmt = select(CardVariant).where(CardVariant.id == variant_id)
            row = (await session.execute(stmt)).scalar_one_or_none()
            if row is None:
                return None
            return CardCatalogEntry(
                variant_id=str(row.id),
                name=row.name,
                set_code="",
                card_number=row.card_number,
                game=row.game.value if hasattr(row.game, "value") else str(row.game),
                canonical_phash=(
                    signed_to_unsigned_64(row.canonical_phash)
                    if row.canonical_phash is not None
                    else None
                ),
                canonical_embedding=(
                    np.asarray(row.canonical_image_embedding, dtype=np.float32)
                    if row.canonical_image_embedding is not None
                    else None
                ),
                metadata=dict(row.metadata_ or {}),
            )


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _row_to_entry(row) -> CardCatalogEntry:
    phash = row["canonical_phash"]
    embedding = row["canonical_image_embedding"]
    return CardCatalogEntry(
        variant_id=str(row["id"]),
        name=row["name"],
        set_code=row["set_code"] or "",
        card_number=row["card_number"],
        game=row["game"].value if hasattr(row["game"], "value") else str(row["game"]),
        canonical_phash=(signed_to_unsigned_64(phash) if phash is not None else None),
        canonical_embedding=(
            np.asarray(embedding, dtype=np.float32) if embedding is not None else None
        ),
        metadata=dict(row["metadata"] or {}),
    )


def _run(coro):
    """Run an async coroutine from sync context.

    If we're already inside a running loop (rare in workers, possible in
    tests), schedule and wait. Otherwise, asyncio.run."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    # If there's a running loop, we can't asyncio.run; in tests this means
    # callers should use the async API directly. Raise a helpful error.
    raise RuntimeError(
        "PgVectorCatalogIndex sync API cannot be called from inside a running "
        "event loop; use the underlying async methods directly"
    ) from None
