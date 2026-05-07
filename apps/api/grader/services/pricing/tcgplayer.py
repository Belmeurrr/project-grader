"""TCGplayer Developer API client.

Tiny: just enough to OAuth, look up a product by ``(name, set)``, and
fetch its current pricing tiers. Intentionally NOT a full SDK — we
hit three endpoints and bail out on anything unexpected.

Auth: client_credentials OAuth. POST to ``/token`` with
``grant_type=client_credentials&client_id={pub}&client_secret={priv}``
returns a bearer token with a ~14-day TTL. We cache it in-process
for ``min(server-issued ttl, 13 days)`` to leave a 1-day margin.

Failure mode: every public function returns None on any error and
logs at WARNING. The cert endpoint must never fail because TCGplayer
is having a bad day.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_BASE_URL = "https://api.tcgplayer.com"
_TOKEN_TTL_MARGIN_SECONDS = 86_400  # 1 day below the issued TTL
_HTTP_TIMEOUT_SECONDS = 5.0


@dataclass(frozen=True)
class ProductPricing:
    """Subset of the TCGplayer pricing payload we surface on the cert.

    All numeric fields are nullable: TCGplayer themselves return null
    when no listings exist in that price tier (a card with no Direct
    listings has ``direct_low_price=None`` while ``low_price`` is
    still populated). The cert page renders nullable fields as "—".
    """

    market_price: float | None
    low_price: float | None
    median_price: float | None
    direct_low_price: float | None
    product_url: str | None


class TcgplayerClient:
    """In-process client. Construct once per app lifecycle.

    Token caching lives on the instance, not module-global, so test
    fixtures can stub a fresh client per test without monkey-patching
    a global.
    """

    def __init__(self, public_key: str, private_key: str) -> None:
        self._public_key = public_key
        self._private_key = private_key
        self._token: str | None = None
        self._token_expires_at: float = 0.0

    async def _get_token(self, client: httpx.AsyncClient) -> str | None:
        """Returns a bearer token, refreshing if cached one is stale.

        Returns None on any failure — caller treats this as "TCGplayer
        unavailable, render cert without pricing"."""
        now = time.time()
        if self._token is not None and now < self._token_expires_at:
            return self._token
        try:
            resp = await client.post(
                f"{_BASE_URL}/token",
                data={
                    "grant_type": "client_credentials",
                    "client_id": self._public_key,
                    "client_secret": self._private_key,
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=_HTTP_TIMEOUT_SECONDS,
            )
        except httpx.HTTPError as exc:
            logger.warning("tcgplayer token fetch failed", extra={"error": str(exc)})
            return None
        if resp.status_code != 200:
            logger.warning(
                "tcgplayer token non-200",
                extra={"status": resp.status_code, "body": resp.text[:200]},
            )
            return None
        payload = resp.json()
        token = payload.get("access_token")
        # TCGplayer returns expires_in as a string (per their docs); be
        # defensive and accept either.
        try:
            ttl = int(payload.get("expires_in", 0))
        except (TypeError, ValueError):
            ttl = 0
        if not token or ttl <= 0:
            logger.warning("tcgplayer token payload malformed")
            return None
        self._token = token
        self._token_expires_at = now + max(ttl - _TOKEN_TTL_MARGIN_SECONDS, 60)
        return token

    async def lookup_pricing(
        self, *, name: str, set_name: str, card_number: str | None = None
    ) -> ProductPricing | None:
        """Find a product matching ``(name, set_name)`` and return its
        current pricing tiers. Returns None on any failure.

        ``card_number`` is currently unused for the lookup but reserved
        for future tie-breaking (TCGplayer sometimes returns multiple
        products for a single name+set when reprints exist).
        """
        async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT_SECONDS) as client:
            token = await self._get_token(client)
            if token is None:
                return None
            headers = {"Authorization": f"Bearer {token}"}
            try:
                # Step 1: catalog/products keyed on (productName, groupName).
                cat = await client.get(
                    f"{_BASE_URL}/catalog/products",
                    headers=headers,
                    params={
                        "productName": name,
                        "groupName": set_name,
                        "limit": 5,
                    },
                )
            except httpx.HTTPError as exc:
                logger.warning("tcgplayer catalog fetch failed", extra={"error": str(exc)})
                return None
            if cat.status_code != 200:
                logger.info(
                    "tcgplayer catalog non-200",
                    extra={"status": cat.status_code, "name": name, "set": set_name},
                )
                return None
            results = cat.json().get("results") or []
            if not results:
                return None
            product = results[0]
            product_id = product.get("productId")
            if product_id is None:
                return None
            product_url = product.get("url")  # "/product/<id>/<slug>"
            if product_url and product_url.startswith("/"):
                product_url = f"https://www.tcgplayer.com{product_url}"

            # Step 2: pricing/product/{id} → tier rows.
            try:
                pricing = await client.get(
                    f"{_BASE_URL}/pricing/product/{product_id}",
                    headers=headers,
                )
            except httpx.HTTPError as exc:
                logger.warning("tcgplayer pricing fetch failed", extra={"error": str(exc)})
                return None
            if pricing.status_code != 200:
                logger.info(
                    "tcgplayer pricing non-200",
                    extra={"status": pricing.status_code, "product_id": product_id},
                )
                return None
            tiers = pricing.json().get("results") or []
            if not tiers:
                return None
            # Pick the "Normal" subTypeName when present; otherwise first row.
            row = next((t for t in tiers if t.get("subTypeName") == "Normal"), tiers[0])
            return ProductPricing(
                market_price=_as_float(row.get("marketPrice")),
                low_price=_as_float(row.get("lowPrice")),
                median_price=_as_float(row.get("midPrice")),
                direct_low_price=_as_float(row.get("directLowPrice")),
                product_url=product_url,
            )


def _as_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None
