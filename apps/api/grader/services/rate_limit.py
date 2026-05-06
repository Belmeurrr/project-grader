"""Rate limiting via slowapi.

Two key functions are exposed:

* :func:`user_or_ip_key` — primary key for write endpoints. Resolves
  to the authenticated Clerk user id when ``request.state.user`` is
  populated by ``get_current_user``. Falls back to the remote IP when
  the user dependency hasn't run yet (e.g. malformed Authorization
  header — slowapi runs before the dep), which keeps the limiter
  honest for unauth'd probes too.

* The default :func:`slowapi.util.get_remote_address` — used directly
  on the public ``GET /cert/{id}`` endpoint where there is no user
  context.

Storage is in-memory on purpose. We're single-process FastAPI for the
launch; once we go multi-instance we'll swap the limiter's ``storage_uri``
for a Redis URL. That migration is a Phase-2 concern called out in
the production-hardening doc — don't add Redis as a dep for limits
right now.

We also expose a tiny adapter ``rate_limit_exceeded_handler`` that
forwards to slowapi's built-in handler. Re-exported here so callers
import a single name from this module instead of remembering the
underscore-prefixed slowapi import.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

if TYPE_CHECKING:
    from fastapi import Request


def user_or_ip_key(request: "Request") -> str:
    """Key function for per-user rate limits.

    Returns ``user:<clerk_id>`` when the auth dependency populated
    ``request.state.user``; falls back to ``ip:<addr>`` otherwise.
    Prefixes prevent collisions between an attacker spoofing
    ``X-Forwarded-For`` and a legitimate user id that happened to look
    like an IP.

    Note: slowapi evaluates the key function during route execution,
    *after* FastAPI has resolved dependencies — so by the time this
    runs, ``get_current_user`` has either populated ``request.state.user``
    or raised 401 (in which case the request never reaches us).
    """
    user = getattr(request.state, "user", None)
    if user is not None and getattr(user, "clerk_id", None):
        return f"user:{user.clerk_id}"
    return f"ip:{get_remote_address(request)}"


# A single Limiter instance is shared across the app. The default
# ``key_func`` is the IP-based one — per-route decorators override it
# with ``user_or_ip_key`` for authenticated endpoints.
limiter = Limiter(key_func=get_remote_address)


# Re-export the handler under a stable name. slowapi's handler is
# private-named upstream (`_rate_limit_exceeded_handler`) but is the
# documented one; we keep a public alias so consumers don't reach
# through the underscore.
def rate_limit_exceeded_handler(request, exc):  # type: ignore[no-untyped-def]
    return _rate_limit_exceeded_handler(request, exc)


__all__ = [
    "RateLimitExceeded",
    "get_remote_address",
    "limiter",
    "rate_limit_exceeded_handler",
    "user_or_ip_key",
]
