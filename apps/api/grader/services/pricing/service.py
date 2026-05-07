"""Top-level pricing service — what ``routers/cert.py`` calls.

Composes a singleton ``TcgplayerClient`` with a 24-hour TTL cache
keyed on ``(name, set_code, card_number)``. Personal-use scope: the
identified-card population is small enough that an in-process LRU
caps fine; we'd revisit this if the cert page got hit at scale.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

from cachetools import TTLCache

from grader.schemas.submissions import IdentifiedCard, PricingPublic
from grader.services.pricing.tcgplayer import ProductPricing, TcgplayerClient
from grader.settings import get_settings

logger = logging.getLogger(__name__)

_CACHE_TTL_SECONDS = 24 * 60 * 60
_CACHE_MAX_ENTRIES = 1024
_LOOKUP_TIMEOUT_SECONDS = 3.0


_cache: TTLCache[tuple[str, str, str], tuple[ProductPricing | None, datetime]] = TTLCache(
    maxsize=_CACHE_MAX_ENTRIES, ttl=_CACHE_TTL_SECONDS
)
_client: TcgplayerClient | None = None


def _get_client() -> TcgplayerClient | None:
    """Build (or return cached) TCGplayer client. Returns None when
    keys aren't configured — pricing service goes silent in that case
    and the cert payload omits the pricing block."""
    global _client
    settings = get_settings()
    if not (settings.tcgplayer_public_key and settings.tcgplayer_private_key):
        return None
    if _client is None:
        _client = TcgplayerClient(
            public_key=settings.tcgplayer_public_key,
            private_key=settings.tcgplayer_private_key,
        )
    return _client


async def fetch_pricing(card: IdentifiedCard) -> PricingPublic | None:
    """Fetch market-price comps for an identified card.

    Returns ``None`` when:
      - the operator hasn't configured TCGplayer keys
      - the cache holds a previous None for this card (lookup miss
        rememberd to avoid re-pinging TCGplayer for cards they don't
        catalog — e.g. promos / one-of-ones)
      - the lookup itself fails (timeout, HTTP error, no product match)

    Cache key is ``(name, set_code, card_number)``. card_number is
    included so reprints with identical name+set don't collide; it's
    not currently passed to the TCGplayer query but reserved for
    future tie-breaking.
    """
    client = _get_client()
    if client is None:
        return None

    key = (card.name, card.set_code, card.card_number)
    if key in _cache:
        product, fetched_at = _cache[key]
    else:
        try:
            product = await asyncio.wait_for(
                client.lookup_pricing(
                    name=card.name,
                    set_name=card.set_code,
                    card_number=card.card_number,
                ),
                timeout=_LOOKUP_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            logger.info(
                "tcgplayer pricing timeout",
                extra={"name": card.name, "set": card.set_code},
            )
            product = None
        except Exception as exc:  # noqa: BLE001 — service must not raise
            logger.warning(
                "tcgplayer pricing unexpected error",
                extra={"error": str(exc), "name": card.name},
            )
            product = None
        fetched_at = datetime.now(timezone.utc)
        # Cache misses (None) too — saves repeated lookups for cards
        # not in their catalog. 24-hour TTL means we'll retry tomorrow.
        _cache[key] = (product, fetched_at)

    if product is None:
        return None
    return PricingPublic(
        source="tcgplayer",
        fetched_at=fetched_at,
        market_price_usd=product.market_price,
        low_price_usd=product.low_price,
        median_price_usd=product.median_price,
        direct_low_price_usd=product.direct_low_price,
        product_url=product.product_url,
    )
