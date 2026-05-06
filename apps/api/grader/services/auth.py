"""Authentication service.

Two modes, switched by ``settings.dev_auth_enabled``:

* **Dev mode** — accepts ``Authorization: Dev <clerk_id>`` and auto-creates a
  matching User. This keeps tests hermetic and lets local dev work without
  Clerk credentials.
* **Production mode** — accepts ``Authorization: Bearer <jwt>``, verifies the
  signature against Clerk's JWKS, validates ``iss`` / ``aud`` / ``exp`` /
  ``nbf`` / ``iat``, and resolves the User by ``sub`` (Clerk user id).

The FastAPI dependency contract — ``get_current_user(authorization, db)`` —
is unchanged so callers in ``grader.routers`` keep working.
"""

from __future__ import annotations

import time
from typing import Any

import httpx
import jwt
from cryptography.hazmat.primitives import serialization
from fastapi import Depends, Header, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from grader.db import get_db
from grader.db.models import User
from grader.settings import Settings, get_settings


# --------------------------------------------------------------------------
# JWKS cache
# --------------------------------------------------------------------------


class JWKSCache:
    """Tiny in-memory JWKS cache keyed on `kid`.

    JWKS endpoints are rate-limited and Clerk rotates keys rarely, so a TTL'd
    in-memory cache is the right shape. We deliberately do *not* share state
    across processes — each gunicorn worker re-fetches once per TTL window,
    which is cheap.

    The fetcher is pluggable so tests can patch it without monkeypatching
    httpx globally.
    """

    def __init__(self, jwks_url: str | None, ttl_seconds: int = 300) -> None:
        self._jwks_url = jwks_url
        self._ttl = ttl_seconds
        # kid -> (PEM-encoded public key string, expires_at_monotonic)
        self._keys: dict[str, tuple[str, float]] = {}
        # Track JWKS fetches so tests can assert the cache is actually doing
        # its job. Public attribute on purpose.
        self.fetch_count: int = 0

    def clear(self) -> None:
        self._keys.clear()

    def _now(self) -> float:
        return time.monotonic()

    async def _fetch_jwks(self) -> dict[str, Any]:
        if not self._jwks_url:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="auth not configured: clerk_jwks_url is unset",
            )
        self.fetch_count += 1
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(self._jwks_url)
            resp.raise_for_status()
            return resp.json()

    async def get_signing_key(self, kid: str) -> str:
        """Return a PEM-encoded RSA public key for the given JWT `kid`.

        Raises HTTP 401 if the JWKS fetch fails or the kid is unknown after
        a refresh.
        """
        cached = self._keys.get(kid)
        if cached is not None and cached[1] > self._now():
            return cached[0]

        try:
            jwks = await self._fetch_jwks()
        except httpx.HTTPError as exc:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"could not fetch jwks: {exc}",
            ) from exc

        expires_at = self._now() + self._ttl
        # Replace the cache wholesale so removed keys disappear too.
        new_keys: dict[str, tuple[str, float]] = {}
        for jwk_dict in jwks.get("keys", []):
            jwk_kid = jwk_dict.get("kid")
            if not jwk_kid:
                continue
            try:
                pyjwk = jwt.PyJWK(jwk_dict)
                # `pyjwk.key` is a cryptography RSAPublicKey; serialize to PEM
                # so we can hand it directly to jwt.decode().
                pem = pyjwk.key.public_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PublicFormat.SubjectPublicKeyInfo,
                ).decode("utf-8")
            except Exception:
                # Skip malformed JWKs rather than fail the whole refresh —
                # one bad key shouldn't take auth down.
                continue
            new_keys[jwk_kid] = (pem, expires_at)
        self._keys = new_keys

        if kid not in self._keys:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"unknown signing key kid={kid}",
            )
        return self._keys[kid][0]


# Module-level cache instance. Reset by tests via `_reset_jwks_cache()`.
_jwks_cache: JWKSCache | None = None


def get_jwks_cache(settings: Settings | None = None) -> JWKSCache:
    """Return the process-wide JWKS cache, lazily constructed from settings."""
    global _jwks_cache
    settings = settings or get_settings()
    if _jwks_cache is None:
        _jwks_cache = JWKSCache(
            jwks_url=settings.clerk_jwks_url,
            ttl_seconds=settings.clerk_jwks_cache_ttl_seconds,
        )
    return _jwks_cache


def _reset_jwks_cache() -> None:
    """Drop the cached JWKSCache instance. Tests use this between cases."""
    global _jwks_cache
    _jwks_cache = None


# --------------------------------------------------------------------------
# JWT verification
# --------------------------------------------------------------------------


async def verify_clerk_jwt(token: str, settings: Settings | None = None) -> dict[str, Any]:
    """Verify a Clerk-issued JWT and return its claims.

    Raises HTTP 401 with a helpful detail on every failure path. The detail
    string is safe to expose: it never includes signing-key material.
    """
    settings = settings or get_settings()

    if not settings.clerk_issuer:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="auth not configured: clerk_issuer is unset",
        )

    try:
        unverified_header = jwt.get_unverified_header(token)
    except jwt.InvalidTokenError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"malformed jwt: {exc}",
        ) from exc

    kid = unverified_header.get("kid")
    if not kid:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="malformed jwt: missing kid",
        )

    alg = unverified_header.get("alg")
    if alg != "RS256":
        # Clerk uses RS256. Anything else (notably `none` or HS256 with a
        # public key as secret) is a known JWT confusion attack — reject.
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"unsupported alg: {alg}",
        )

    cache = get_jwks_cache(settings)
    public_key = await cache.get_signing_key(kid)

    decode_kwargs: dict[str, Any] = {
        "algorithms": ["RS256"],
        "issuer": settings.clerk_issuer,
        "options": {
            "require": ["exp", "iat", "iss", "sub"],
            "verify_aud": settings.clerk_audience is not None,
        },
    }
    if settings.clerk_audience is not None:
        decode_kwargs["audience"] = settings.clerk_audience

    try:
        claims = jwt.decode(token, public_key, **decode_kwargs)
    except jwt.ExpiredSignatureError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="jwt expired",
        ) from exc
    except jwt.ImmatureSignatureError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="jwt not yet valid (nbf)",
        ) from exc
    except jwt.InvalidIssuerError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="jwt has wrong issuer",
        ) from exc
    except jwt.InvalidAudienceError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="jwt has wrong audience",
        ) from exc
    except jwt.MissingRequiredClaimError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"jwt missing required claim: {exc.claim}",
        ) from exc
    except jwt.InvalidTokenError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"jwt invalid: {exc}",
        ) from exc

    return claims


# --------------------------------------------------------------------------
# User resolution helpers
# --------------------------------------------------------------------------


async def _get_or_create_user(db: AsyncSession, clerk_id: str, email: str) -> User:
    result = await db.execute(select(User).where(User.clerk_id == clerk_id))
    user = result.scalar_one_or_none()
    if user is None:
        user = User(clerk_id=clerk_id, email=email)
        db.add(user)
        await db.commit()
        await db.refresh(user)
    return user


async def _authenticate_dev(token: str, db: AsyncSession) -> User:
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="bad auth scheme",
        )
    return await _get_or_create_user(db, clerk_id=token, email=f"{token}@dev.local")


async def _authenticate_clerk(
    jwt_token: str, db: AsyncSession, settings: Settings | None = None
) -> User:
    if not jwt_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="bad auth scheme",
        )
    claims = await verify_clerk_jwt(jwt_token, settings=settings)
    sub = claims.get("sub")
    if not sub:
        # Defensive: `require` should have caught this, but PyJWT semantics
        # mean an empty string can sneak through.
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="jwt missing sub",
        )
    # Block self-issued / unverified Clerk accounts: a user that hasn't
    # confirmed their email never gets a User row, never auto-creates,
    # and can't submit jobs. The dev path (_authenticate_dev) is
    # intentionally exempt — it's gated behind dev_auth_enabled.
    if claims.get("email_verified") is not True:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="email_not_verified",
        )
    email = claims.get("email") or f"{sub}@user.clerk"
    return await _get_or_create_user(db, clerk_id=sub, email=email)


# --------------------------------------------------------------------------
# Public FastAPI dependency
# --------------------------------------------------------------------------


async def get_current_user(
    authorization: str | None = Header(default=None),
    db: AsyncSession = Depends(get_db),
) -> User:
    """Resolve the current user from the Authorization header.

    Routes a request to either the dev stub or full Clerk JWT verification
    based on ``settings.dev_auth_enabled``. The dependency signature (and
    thus the FastAPI contract) is unchanged from the original stub.
    """
    if authorization is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="missing auth",
        )

    settings = get_settings()
    scheme, _, token = authorization.partition(" ")
    scheme_lower = scheme.lower()

    if settings.dev_auth_enabled:
        if scheme_lower != "dev":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="bad auth scheme",
            )
        return await _authenticate_dev(token, db)

    if scheme_lower != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="bad auth scheme",
        )
    return await _authenticate_clerk(token, db, settings=settings)
