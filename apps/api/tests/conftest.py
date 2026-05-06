"""Shared fixtures for the API test suite.

We run unit tests against a Postgres test database brought up via the local
docker-compose. SQLite is not viable because the schema uses pgvector,
JSONB, and named ENUM types — all Postgres-specific. CI runs the same
fixture against an ephemeral Postgres service container.

Required env (defaults assume the local docker-compose Postgres):
  TEST_DATABASE_URL=postgresql+asyncpg://grader:grader@localhost:5432/grader_test
"""

from __future__ import annotations

import os
import uuid
from collections.abc import AsyncIterator

import httpx
import pytest
import pytest_asyncio
from fastapi import FastAPI
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from grader.db import get_db
from grader.db.models import Base
from grader.main import create_app


def _test_database_url() -> str:
    return os.environ.get(
        "TEST_DATABASE_URL",
        "postgresql+asyncpg://grader:grader@localhost:5432/grader_test",
    )


# pytest-asyncio 1.x manages the event loop itself when
# `asyncio_default_*_loop_scope` is configured in pyproject.toml. We no longer
# define a custom `event_loop` fixture (which was the pytest-asyncio 0.x
# pattern and is deprecated in 1.x).


@pytest_asyncio.fixture(scope="session")
async def _engine():
    engine = create_async_engine(_test_database_url(), pool_pre_ping=True)
    async with engine.begin() as conn:
        # The pgvector extension must exist for Vector columns; uuid-ossp is
        # not strictly required since we generate uuids in Python, but we keep
        # parity with the production migration.
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest_asyncio.fixture
async def db_session(_engine) -> AsyncIterator[AsyncSession]:
    """Per-test session wrapped in a transaction that rolls back at the end.

    This keeps tests independent without paying for create_all/drop_all per
    test. Each test sees an empty database state from its own perspective."""
    maker = async_sessionmaker(_engine, expire_on_commit=False, class_=AsyncSession)
    async with _engine.connect() as conn:
        trans = await conn.begin()
        async with maker(bind=conn) as session:
            try:
                yield session
            finally:
                await trans.rollback()


@pytest_asyncio.fixture
async def app(db_session: AsyncSession) -> AsyncIterator[FastAPI]:
    """FastAPI app with the DB dependency overridden to use the test session.

    Also resets the slowapi limiter's in-memory storage between tests
    so a noisy test doesn't poison the per-user / per-IP buckets for
    its successor. The limiter is a module-level singleton; tests
    that don't explicitly assert on rate limits should never see 429s
    from prior runs."""
    from grader.services.rate_limit import limiter

    limiter.reset()
    application = create_app()

    async def _override_get_db() -> AsyncIterator[AsyncSession]:
        # Expire identity-map state at the start of each request so handlers
        # don't see stale snapshots of rows that were written by test setup
        # (or a previous request in the same test) via the same session.
        #
        # Why this matters: the test fixture shares one AsyncSession between
        # the test body's setup writes and the request handlers (we override
        # `get_db` to yield this session). After a flush/commit, rows live
        # in the session's identity map. When a request handler then calls
        # `db.get(Model, id, options=[selectinload(...)])`, the identity-map
        # cache hit returns the in-memory instance and the eager-load options
        # are silently dropped. Production code in `_to_out` then accesses a
        # relationship that was never loaded, triggering a sync lazy-load
        # under asyncpg → `sqlalchemy.exc.MissingGreenlet`.
        #
        # `expire_all()` marks every persistent instance as needing a refresh,
        # which combined with `populate_existing=True` on the production
        # `db.get` calls makes selectinload options actually apply on
        # identity-map hits.
        db_session.expire_all()
        yield db_session

    application.dependency_overrides[get_db] = _override_get_db
    yield application
    application.dependency_overrides.clear()


@pytest_asyncio.fixture
async def client(app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.fixture
def dev_token() -> str:
    """A unique-per-test Clerk-style dev token; the dev auth service auto-creates
    a matching User row on first use."""
    return f"dev-{uuid.uuid4().hex[:12]}"


@pytest.fixture
def auth_headers(dev_token: str) -> dict[str, str]:
    return {"Authorization": f"Dev {dev_token}"}
