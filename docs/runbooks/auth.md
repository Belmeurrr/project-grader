# Runbook — Authentication

Project Grader uses Clerk for end-user identity. The FastAPI service exposes
a single dependency, `grader.services.auth.get_current_user`, which resolves
the caller's `User` row from the `Authorization` header. It has two modes,
selected by configuration.

## Modes at a glance

| Mode | Header                       | Use                       |
|------|------------------------------|---------------------------|
| Dev  | `Authorization: Dev <token>` | Local dev, tests          |
| Prod | `Authorization: Bearer <jwt>`| Staging, prod, real users |

The mode is chosen by `Settings.dev_auth_enabled`:

* `env=dev` -> defaults to `True` (dev mode).
* `env=staging` / `env=prod` -> defaults to `False` (Clerk JWT mode).
* The default is overridable per environment with `DEV_AUTH_ENABLED=true|false`.

## Dev mode

The dev stub accepts any opaque `<token>` in `Authorization: Dev <token>`,
and on first sight creates a `User` row with:

```
clerk_id = <token>
email    = <token>@dev.local
```

Subsequent requests with the same token resolve to the same user. This
matches the `dev_token` / `auth_headers` pytest fixtures so existing route
tests do not need any changes.

Bad scheme (`Bearer ...` while in dev mode), missing `Authorization`, or an
empty token after `Dev ` all return HTTP 401.

## Production mode (Clerk JWT verification)

In prod mode the service:

1. Splits the `Authorization` header on whitespace; the scheme must be
   `Bearer` (case-insensitive).
2. Reads the JWT header to extract `kid` and `alg`. Only `RS256` is
   accepted — anything else is rejected (defense against alg-confusion
   attacks).
3. Looks up the signing key by `kid` in an in-memory JWKS cache (5-minute
   TTL by default). On a miss, the cache fetches the JWKS document from
   `Settings.clerk_jwks_url` and replaces its key set.
4. Verifies the signature, then validates standard claims:
   - `iss` must equal `Settings.clerk_issuer`.
   - `aud` must equal `Settings.clerk_audience` if it is configured;
     ignored otherwise (Clerk often omits `aud`).
   - `exp` must be in the future, `nbf` (if present) in the past, `iat`
     present.
   - `sub` must be present.
5. Resolves `User.clerk_id == claims["sub"]`. If the user does not yet
   exist, creates one with `email = claims.get("email") or f"{sub}@user.clerk"`.

Any failure on the path above returns HTTP 401 with a useful (but
non-leaky) `detail` string — `jwt expired`, `jwt has wrong issuer`,
`unknown signing key kid=<kid>`, `malformed jwt: <reason>`, etc.

## Configuring Clerk

In the Clerk dashboard:

1. Open **API Keys** -> the JWKS endpoint URL is shown there. Copy it; it
   looks like `https://<your-instance>.clerk.accounts.dev/.well-known/jwks.json`
   for development instances or `https://clerk.<your-domain>.com/.well-known/jwks.json`
   in production.
2. The **issuer** is the same URL minus `/.well-known/jwks.json` —
   `https://<your-instance>.clerk.accounts.dev`. Confirm by decoding any
   issued JWT (e.g. with `jwt.io`) and reading `iss`.
3. The **audience** field is typically empty in Clerk JWTs. Leave
   `CLERK_AUDIENCE` unset unless you have a specific reason to enforce one
   (e.g. multi-tenant separation). When unset, the audience check is
   skipped.

Set the env vars on the service:

```env
ENV=prod
DEV_AUTH_ENABLED=false                # explicit, recommended
CLERK_JWKS_URL=https://...clerk.accounts.dev/.well-known/jwks.json
CLERK_ISSUER=https://...clerk.accounts.dev
# CLERK_AUDIENCE=                    # only if you actually issue audience claims
CLERK_JWKS_CACHE_TTL_SECONDS=300
```

## JWKS caching

The cache is process-local (one dict per gunicorn worker) keyed on `kid`,
TTL 300s by default. Each worker performs at most one JWKS fetch per TTL
window, so even at high RPS the load on Clerk's JWKS endpoint is trivial.

A JWKS refresh replaces the entire keyset, so a key that was removed
upstream disappears on the next refresh. There is a small window
(<= TTL) where a freshly rotated key is unknown to a given worker; the
worker will refresh on the first 401 attempt for the new `kid`.

### Emergency: clear the cache

If Clerk emergency-rotates a key and you need to drop the cached set
without restarting workers, the simplest options are:

* Roll the deployment (each new worker comes up with an empty cache).
* Reduce `CLERK_JWKS_CACHE_TTL_SECONDS` to e.g. `30`, redeploy, then
  raise it back. Faster than a full restart in some environments.
* In a pinch, exec into a worker and call:

  ```python
  from grader.services.auth import _reset_jwks_cache
  _reset_jwks_cache()
  ```

  But this is per-worker, so prefer a redeploy.

## Testing JWT verification locally

The test suite (`apps/api/tests/test_auth.py`) is fully self-contained: it
generates an RSA keypair per test, signs JWTs with it, mocks the JWKS
fetch to return the matching public key, and exercises every failure
branch. Run it with:

```bash
cd apps/api
uv run pytest tests/test_auth.py -v
```

To smoke-test against a real Clerk instance, set the env vars above and
mint a token from the Clerk dashboard's **Sessions** -> **JWT Templates**
page. Then:

```bash
curl -H "Authorization: Bearer <jwt>" http://localhost:8000/submissions
```

## Failure modes you'll see in logs

| 401 detail                          | Meaning                                              |
|-------------------------------------|------------------------------------------------------|
| `missing auth`                      | No `Authorization` header at all.                    |
| `bad auth scheme`                   | Wrong scheme for the current mode.                   |
| `malformed jwt: ...`                | Token structure is broken.                           |
| `unsupported alg: ...`              | Token signed with non-RS256 alg.                     |
| `unknown signing key kid=<kid>`     | `kid` not in JWKS even after refresh.                |
| `jwt expired`                       | `exp` is in the past.                                |
| `jwt not yet valid (nbf)`           | `nbf` is in the future. Clock skew?                  |
| `jwt has wrong issuer`              | `iss` does not match `CLERK_ISSUER`.                 |
| `jwt has wrong audience`            | `aud` does not match `CLERK_AUDIENCE`.               |
| `auth not configured: ...`          | Prod mode is on but `CLERK_*` env vars are missing.  |
| `could not fetch jwks: ...`         | Network/HTTP error reaching the JWKS endpoint.       |
