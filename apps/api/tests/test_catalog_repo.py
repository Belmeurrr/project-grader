"""Tests for grader.services.catalog_repo against the real Postgres test DB.

Exercises pHash bit_count Hamming distance, pgvector cosine NN, and the
two's-complement bigint round-trip helpers."""

from __future__ import annotations

import uuid

import numpy as np
import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker

from grader.db.models import CardSet, CardVariant, Game
from grader.services.catalog_repo import (
    PgVectorCatalogIndex,
    signed_to_unsigned_64,
    unsigned_to_signed_64,
)
from pipelines.identification.hashing import HASH_BITS


# -----------------------------
# Pure unit: 64-bit signed/unsigned round-trip
# -----------------------------


def test_unsigned_to_signed_below_2_63() -> None:
    assert unsigned_to_signed_64(0) == 0
    assert unsigned_to_signed_64(123) == 123
    assert unsigned_to_signed_64((1 << 63) - 1) == (1 << 63) - 1


def test_unsigned_to_signed_above_2_63() -> None:
    """Values with the top bit set should map to negative signed bigint."""
    assert unsigned_to_signed_64(1 << 63) == -(1 << 63)
    assert unsigned_to_signed_64((1 << 64) - 1) == -1


def test_unsigned_signed_round_trip_at_boundary() -> None:
    for v in [0, 1, (1 << 63) - 1, 1 << 63, (1 << 64) - 1, 0xDEADBEEFFACECAFE]:
        assert signed_to_unsigned_64(unsigned_to_signed_64(v)) == v


def test_unsigned_to_signed_rejects_overflow() -> None:
    with pytest.raises(ValueError):
        unsigned_to_signed_64(1 << HASH_BITS)


# -----------------------------
# Integration with Postgres + pgvector
# -----------------------------


@pytest.fixture
def session_factory(_engine):
    return async_sessionmaker(_engine, expire_on_commit=False)


def _add_variant(
    db_session,
    *,
    name: str,
    phash: int | None,
    embedding: np.ndarray | None,
    set_id,
) -> CardVariant:
    cv = CardVariant(
        game=Game.MTG,
        set_id=set_id,
        card_number="1",
        name=name,
        canonical_phash=unsigned_to_signed_64(phash) if phash is not None else None,
        canonical_image_embedding=(
            embedding.tolist() if embedding is not None else None
        ),
    )
    db_session.add(cv)
    return cv


@pytest.mark.requires_postgres
@pytest.mark.asyncio
async def test_find_by_phash_returns_exact_match(db_session, session_factory) -> None:
    cs = CardSet(game=Game.MTG, code="UTS", name="UnitTestSet")
    db_session.add(cs)
    await db_session.flush()
    target = _add_variant(
        db_session, name="target", phash=0xABCDEF0123456789, embedding=None, set_id=cs.id
    )
    _add_variant(
        db_session,
        name="distractor",
        phash=0x0000_0000_0000_0000,
        embedding=None,
        set_id=cs.id,
    )
    await db_session.commit()

    repo = PgVectorCatalogIndex(session_factory=session_factory)
    hits = await repo._find_by_phash_async(0xABCDEF0123456789, max_distance=0, limit=10)
    assert len(hits) == 1
    assert hits[0].entry.variant_id == str(target.id)
    assert hits[0].distance == 0


@pytest.mark.requires_postgres
@pytest.mark.asyncio
async def test_find_by_phash_within_distance(db_session, session_factory) -> None:
    cs = CardSet(game=Game.MTG, code="UTS", name="UnitTestSet")
    db_session.add(cs)
    await db_session.flush()
    base = 0xFFFF_0000_FFFF_0000
    near = base ^ 0b111  # 3 bits flipped
    far = base ^ 0xFFFF_FFFF_FFFF_FFFF  # 64 bits flipped

    n_var = _add_variant(db_session, name="near", phash=near, embedding=None, set_id=cs.id)
    _add_variant(db_session, name="far", phash=far, embedding=None, set_id=cs.id)
    await db_session.commit()

    repo = PgVectorCatalogIndex(session_factory=session_factory)
    hits = await repo._find_by_phash_async(base, max_distance=8, limit=10)
    assert len(hits) == 1
    assert hits[0].entry.variant_id == str(n_var.id)
    assert hits[0].distance == 3


@pytest.mark.requires_postgres
@pytest.mark.asyncio
async def test_find_by_phash_high_bit_round_trip(db_session, session_factory) -> None:
    """Bigint two's-complement: a hash >= 2^63 must still match itself."""
    cs = CardSet(game=Game.MTG, code="UTS", name="UnitTestSet")
    db_session.add(cs)
    await db_session.flush()
    high = (1 << 63) | 0xDEAD_BEEF
    var = _add_variant(db_session, name="high", phash=high, embedding=None, set_id=cs.id)
    await db_session.commit()

    repo = PgVectorCatalogIndex(session_factory=session_factory)
    hits = await repo._find_by_phash_async(high, max_distance=0, limit=10)
    assert len(hits) == 1
    assert hits[0].entry.variant_id == str(var.id)


@pytest.mark.requires_postgres
@pytest.mark.asyncio
async def test_nearest_by_embedding_orders_by_cosine_distance(
    db_session, session_factory
) -> None:
    cs = CardSet(game=Game.MTG, code="UTS", name="UnitTestSet")
    db_session.add(cs)
    await db_session.flush()
    rng = np.random.default_rng(0)
    base = rng.standard_normal(768).astype(np.float32)
    base = base / np.linalg.norm(base)
    near = base + 0.05 * rng.standard_normal(768).astype(np.float32)
    near = near / np.linalg.norm(near)
    far = -base
    far = far / np.linalg.norm(far)

    near_var = _add_variant(db_session, name="near", phash=None, embedding=near, set_id=cs.id)
    far_var = _add_variant(db_session, name="far", phash=None, embedding=far, set_id=cs.id)
    await db_session.commit()

    repo = PgVectorCatalogIndex(session_factory=session_factory)
    hits = await repo._nearest_by_embedding_async(base, top_k=2)
    ids = [h.entry.variant_id for h in hits]
    assert ids[0] == str(near_var.id)
    assert ids[1] == str(far_var.id)
    assert hits[0].distance < hits[1].distance


@pytest.mark.requires_postgres
@pytest.mark.asyncio
async def test_nearest_by_embedding_skips_null_embeddings(
    db_session, session_factory
) -> None:
    cs = CardSet(game=Game.MTG, code="UTS", name="UnitTestSet")
    db_session.add(cs)
    await db_session.flush()
    rng = np.random.default_rng(1)
    only = rng.standard_normal(768).astype(np.float32)
    only = only / np.linalg.norm(only)

    _add_variant(db_session, name="no-emb", phash=None, embedding=None, set_id=cs.id)
    only_var = _add_variant(
        db_session, name="only", phash=None, embedding=only, set_id=cs.id
    )
    await db_session.commit()

    repo = PgVectorCatalogIndex(session_factory=session_factory)
    hits = await repo._nearest_by_embedding_async(only, top_k=5)
    ids = [h.entry.variant_id for h in hits]
    assert ids == [str(only_var.id)]
