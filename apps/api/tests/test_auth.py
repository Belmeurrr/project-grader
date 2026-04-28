"""Tests for `grader.services.auth`.

Covers both modes of `get_current_user`:

* Dev mode: ``Authorization: Dev <token>`` (the existing stub behavior).
* Prod mode: ``Authorization: Bearer <jwt>``, JWT verified against a mocked
  JWKS endpoint.

We mint test JWTs with a freshly-generated RSA key so the tests are entirely
self-contained — they never reach the real Clerk JWKS endpoint. The httpx
client used for JWKS retrieval is mocked at the cache boundary
(``JWKSCache._fetch_jwks``) so we can assert hit counts.
"""

from __future__ import annotations

import time
import uuid
from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
import jwt
import pytest
import pytest_asyncio
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from grader.db.models import User
from grader.services import auth as auth_service
from grader.settings import get_settings


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _make_rsa_keypair() -> tuple[rsa.RSAPrivateKey, str]:
    """Generate an RSA keypair, return (private_key, public_pem)."""
    private = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public_pem = private.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    ).decode("utf-8")
    return private, public_pem


def _sign_jwt(
    private: rsa.RSAPrivateKey,
    *,
    kid: str,
    issuer: str,
    sub: str,
    audience: str | None = None,
    extra: dict[str, Any] | None = None,
    exp_offset: int = 300,
    nbf_offset: int = -10,
    iat_offset: int = -10,
) -> str:
    now = int(time.time())
    claims: dict[str, Any] = {
        "iss": issuer,
        "sub": sub,
        "iat": now + iat_offset,
        "nbf": now + nbf_offset,
        "exp": now + exp_offset,
    }
    if audience is not None:
        claims["aud"] = audience
    if extra:
        claims.update(extra)
    return jwt.encode(claims, private, algorithm="RS256", headers={"kid": kid})


def _patch_jwks_cache_fetch(
    mock_fetch: AsyncMock,
) -> Any:
    """Patch JWKSCache._fetch_jwks so tests don't talk to the network."""
    return patch.object(auth_service.JWKSCache, "_fetch_jwks", mock_fetch)


@pytest.fixture(autouse=True)
def _reset_cache_and_settings_between_tests() -> Any:
    """Clear settings cache and JWKS cache before/after each test.

    Settings are read via ``get_settings()`` which is ``lru_cache``'d, so a
    test that mutates env-derived behavior (toggling dev_auth_enabled, etc.)
    must invalidate it. We do this on every test for safety.
    """
    get_settings.cache_clear()
    auth_service._reset_jwks_cache()
    yield
    get_settings.cache_clear()
    auth_service._reset_jwks_cache()


@pytest_asyncio.fixture
async def prod_settings(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Configure the app in prod-mode (Clerk JWT verification) for one test.

    Sets the env-vars pydantic-settings reads on construction, then yields a
    fresh Settings instance. The autouse fixture above clears the cache so
    `get_settings()` re-reads.
    """
    issuer = "https://test.clerk.example"
    monkeypatch.setenv("ENV", "prod")
    monkeypatch.setenv("DEV_AUTH_ENABLED", "false")
    monkeypatch.setenv("CLERK_JWKS_URL", f"{issuer}/.well-known/jwks.json")
    monkeypatch.setenv("CLERK_ISSUER", issuer)
    # No audience by default; tests that need one re-monkeypatch.

    get_settings.cache_clear()
    yield get_settings()
    get_settings.cache_clear()


# --------------------------------------------------------------------------
# Dev mode
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dev_mode_accepts_dev_scheme_and_creates_user(
    client: httpx.AsyncClient,
    db_session: AsyncSession,
    auth_headers: dict[str, str],
) -> None:
    # Hits a real authenticated endpoint to exercise the FastAPI dependency.
    r = await client.post("/submissions", headers=auth_headers, json={})
    assert r.status_code == 201

    token = auth_headers["Authorization"].split(" ", 1)[1]
    rows = await db_session.execute(select(User).where(User.clerk_id == token))
    user = rows.scalar_one()
    assert user.email == f"{token}@dev.local"


@pytest.mark.asyncio
async def test_dev_mode_missing_auth_returns_401(client: httpx.AsyncClient) -> None:
    r = await client.post("/submissions", json={})
    assert r.status_code == 401
    assert r.json()["detail"] == "missing auth"


@pytest.mark.asyncio
async def test_dev_mode_bad_scheme_returns_401(client: httpx.AsyncClient) -> None:
    r = await client.post(
        "/submissions",
        headers={"Authorization": "Bearer not-a-real-jwt"},
        json={},
    )
    assert r.status_code == 401
    assert r.json()["detail"] == "bad auth scheme"


@pytest.mark.asyncio
async def test_dev_mode_empty_token_returns_401(client: httpx.AsyncClient) -> None:
    r = await client.post(
        "/submissions",
        headers={"Authorization": "Dev"},
        json={},
    )
    assert r.status_code == 401


# --------------------------------------------------------------------------
# Prod mode (mocked JWKS)
# --------------------------------------------------------------------------


def _jwks_response(public_pem: str, kid: str) -> dict[str, Any]:
    """Build a JWKS JSON document containing one RS256 key.

    We re-derive the JWK from the public PEM via PyJWK so this matches the
    exact shape `jwt.PyJWK` will accept downstream.
    """
    from jwt.algorithms import RSAAlgorithm

    public_key = serialization.load_pem_public_key(public_pem.encode("utf-8"))
    jwk = RSAAlgorithm.to_jwk(public_key, as_dict=True)
    jwk["kid"] = kid
    jwk["alg"] = "RS256"
    jwk["use"] = "sig"
    return {"keys": [jwk]}


@pytest.mark.asyncio
async def test_prod_mode_valid_jwt_authenticates(
    client: httpx.AsyncClient,
    db_session: AsyncSession,
    prod_settings: Any,
) -> None:
    private, public_pem = _make_rsa_keypair()
    kid = "test-kid-1"
    token = _sign_jwt(
        private,
        kid=kid,
        issuer=prod_settings.clerk_issuer,
        sub="user_abc123",
        extra={"email": "real@user.example"},
    )
    fetch = AsyncMock(return_value=_jwks_response(public_pem, kid))
    with _patch_jwks_cache_fetch(fetch):
        r = await client.post(
            "/submissions",
            headers={"Authorization": f"Bearer {token}"},
            json={},
        )

    assert r.status_code == 201, r.text
    rows = await db_session.execute(select(User).where(User.clerk_id == "user_abc123"))
    user = rows.scalar_one()
    assert user.email == "real@user.example"


@pytest.mark.asyncio
async def test_prod_mode_email_falls_back_to_placeholder(
    client: httpx.AsyncClient,
    db_session: AsyncSession,
    prod_settings: Any,
) -> None:
    private, public_pem = _make_rsa_keypair()
    kid = "test-kid-2"
    token = _sign_jwt(
        private,
        kid=kid,
        issuer=prod_settings.clerk_issuer,
        sub="user_no_email",
    )
    fetch = AsyncMock(return_value=_jwks_response(public_pem, kid))
    with _patch_jwks_cache_fetch(fetch):
        r = await client.post(
            "/submissions",
            headers={"Authorization": f"Bearer {token}"},
            json={},
        )

    assert r.status_code == 201, r.text
    rows = await db_session.execute(select(User).where(User.clerk_id == "user_no_email"))
    user = rows.scalar_one()
    assert user.email == "user_no_email@user.clerk"


@pytest.mark.asyncio
async def test_prod_mode_existing_user_is_reused(
    client: httpx.AsyncClient,
    db_session: AsyncSession,
    prod_settings: Any,
) -> None:
    # Pre-create a user with the same clerk_id we'll mint a JWT for.
    existing = User(clerk_id="user_already_here", email="prior@example.com")
    db_session.add(existing)
    await db_session.commit()

    private, public_pem = _make_rsa_keypair()
    kid = "test-kid-3"
    token = _sign_jwt(
        private,
        kid=kid,
        issuer=prod_settings.clerk_issuer,
        sub="user_already_here",
        extra={"email": "different@example.com"},
    )
    fetch = AsyncMock(return_value=_jwks_response(public_pem, kid))
    with _patch_jwks_cache_fetch(fetch):
        r = await client.post(
            "/submissions",
            headers={"Authorization": f"Bearer {token}"},
            json={},
        )
    assert r.status_code == 201

    rows = await db_session.execute(
        select(User).where(User.clerk_id == "user_already_here")
    )
    users = rows.scalars().all()
    assert len(users) == 1
    # Email is NOT updated — we keep whatever the row already held.
    assert users[0].email == "prior@example.com"


@pytest.mark.asyncio
async def test_prod_mode_expired_jwt_returns_401(
    client: httpx.AsyncClient, prod_settings: Any
) -> None:
    private, public_pem = _make_rsa_keypair()
    kid = "test-kid-exp"
    token = _sign_jwt(
        private,
        kid=kid,
        issuer=prod_settings.clerk_issuer,
        sub="user_expired",
        exp_offset=-60,
        iat_offset=-300,
        nbf_offset=-300,
    )
    fetch = AsyncMock(return_value=_jwks_response(public_pem, kid))
    with _patch_jwks_cache_fetch(fetch):
        r = await client.post(
            "/submissions",
            headers={"Authorization": f"Bearer {token}"},
            json={},
        )
    assert r.status_code == 401
    assert "expired" in r.json()["detail"].lower()


@pytest.mark.asyncio
async def test_prod_mode_wrong_issuer_returns_401(
    client: httpx.AsyncClient, prod_settings: Any
) -> None:
    private, public_pem = _make_rsa_keypair()
    kid = "test-kid-iss"
    token = _sign_jwt(
        private,
        kid=kid,
        issuer="https://attacker.example",
        sub="user_evil",
    )
    fetch = AsyncMock(return_value=_jwks_response(public_pem, kid))
    with _patch_jwks_cache_fetch(fetch):
        r = await client.post(
            "/submissions",
            headers={"Authorization": f"Bearer {token}"},
            json={},
        )
    assert r.status_code == 401
    assert "issuer" in r.json()["detail"].lower()


@pytest.mark.asyncio
async def test_prod_mode_malformed_jwt_returns_401(
    client: httpx.AsyncClient, prod_settings: Any
) -> None:
    fetch = AsyncMock(return_value={"keys": []})
    with _patch_jwks_cache_fetch(fetch):
        r = await client.post(
            "/submissions",
            headers={"Authorization": "Bearer not.a.jwt"},
            json={},
        )
    assert r.status_code == 401
    assert "malformed" in r.json()["detail"].lower()


@pytest.mark.asyncio
async def test_prod_mode_unknown_kid_returns_401(
    client: httpx.AsyncClient, prod_settings: Any
) -> None:
    private, _ = _make_rsa_keypair()
    # Server's JWKS happens to return *no* keys, so `unknown-kid` will not
    # be present after refresh.
    token = _sign_jwt(
        private,
        kid="unknown-kid",
        issuer=prod_settings.clerk_issuer,
        sub="user_x",
    )
    fetch = AsyncMock(return_value={"keys": []})
    with _patch_jwks_cache_fetch(fetch):
        r = await client.post(
            "/submissions",
            headers={"Authorization": f"Bearer {token}"},
            json={},
        )
    assert r.status_code == 401
    assert "unknown signing key" in r.json()["detail"].lower()


@pytest.mark.asyncio
async def test_prod_mode_bad_scheme_returns_401(
    client: httpx.AsyncClient, prod_settings: Any
) -> None:
    r = await client.post(
        "/submissions",
        headers={"Authorization": "Dev fallback-attempt"},
        json={},
    )
    assert r.status_code == 401
    assert r.json()["detail"] == "bad auth scheme"


@pytest.mark.asyncio
async def test_prod_mode_audience_enforced_when_configured(
    client: httpx.AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
    prod_settings: Any,
) -> None:
    monkeypatch.setenv("CLERK_AUDIENCE", "expected-audience")
    get_settings.cache_clear()
    settings = get_settings()
    assert settings.clerk_audience == "expected-audience"

    private, public_pem = _make_rsa_keypair()
    kid = "test-kid-aud"
    bad_token = _sign_jwt(
        private,
        kid=kid,
        issuer=settings.clerk_issuer,
        sub="user_aud",
        audience="WRONG-audience",
    )
    fetch = AsyncMock(return_value=_jwks_response(public_pem, kid))
    with _patch_jwks_cache_fetch(fetch):
        r = await client.post(
            "/submissions",
            headers={"Authorization": f"Bearer {bad_token}"},
            json={},
        )
    assert r.status_code == 401
    assert "audience" in r.json()["detail"].lower()


# --------------------------------------------------------------------------
# JWKS cache behavior
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_jwks_cache_reuses_within_ttl(
    client: httpx.AsyncClient, prod_settings: Any
) -> None:
    private, public_pem = _make_rsa_keypair()
    kid = "test-kid-cache"
    fetch = AsyncMock(return_value=_jwks_response(public_pem, kid))

    with _patch_jwks_cache_fetch(fetch):
        for _ in range(3):
            token = _sign_jwt(
                private,
                kid=kid,
                issuer=prod_settings.clerk_issuer,
                sub=f"user_{uuid.uuid4().hex[:6]}",
            )
            r = await client.post(
                "/submissions",
                headers={"Authorization": f"Bearer {token}"},
                json={},
            )
            assert r.status_code == 201, r.text

    # 3 successful requests, exactly 1 JWKS fetch -> cache works.
    assert fetch.await_count == 1


@pytest.mark.asyncio
async def test_jwks_cache_refetches_after_ttl_expiry(prod_settings: Any) -> None:
    """Direct unit test on the JWKSCache without going through HTTP."""
    private, public_pem = _make_rsa_keypair()
    kid = "test-kid-ttl"
    cache = auth_service.JWKSCache(
        jwks_url=prod_settings.clerk_jwks_url, ttl_seconds=300
    )

    fetch = AsyncMock(return_value=_jwks_response(public_pem, kid))
    with patch.object(auth_service.JWKSCache, "_fetch_jwks", fetch):
        await cache.get_signing_key(kid)
        await cache.get_signing_key(kid)
        assert fetch.await_count == 1

        # Force expiration by rewinding the cached entry.
        cache._keys[kid] = (cache._keys[kid][0], 0.0)
        await cache.get_signing_key(kid)
        assert fetch.await_count == 2


@pytest.mark.asyncio
async def test_jwks_cache_unconfigured_raises(prod_settings: Any) -> None:
    """If clerk_jwks_url is None, the cache must raise a useful 401."""
    cache = auth_service.JWKSCache(jwks_url=None, ttl_seconds=300)
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as ctx:
        await cache.get_signing_key("any-kid")
    assert ctx.value.status_code == 401
    assert "clerk_jwks_url" in ctx.value.detail


# --------------------------------------------------------------------------
# Settings: dev_auth_enabled defaults
# --------------------------------------------------------------------------


def test_dev_auth_default_true_for_env_dev(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ENV", "dev")
    monkeypatch.delenv("DEV_AUTH_ENABLED", raising=False)
    get_settings.cache_clear()
    assert get_settings().dev_auth_enabled is True


def test_dev_auth_default_false_for_env_prod(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ENV", "prod")
    monkeypatch.delenv("DEV_AUTH_ENABLED", raising=False)
    get_settings.cache_clear()
    assert get_settings().dev_auth_enabled is False


def test_dev_auth_default_false_for_env_staging(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ENV", "staging")
    monkeypatch.delenv("DEV_AUTH_ENABLED", raising=False)
    get_settings.cache_clear()
    assert get_settings().dev_auth_enabled is False


def test_dev_auth_explicit_override_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ENV", "prod")
    monkeypatch.setenv("DEV_AUTH_ENABLED", "true")
    get_settings.cache_clear()
    assert get_settings().dev_auth_enabled is True
