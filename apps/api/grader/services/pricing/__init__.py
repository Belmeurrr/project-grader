"""TCGplayer market-price comps for the public cert page.

Public surface: ``fetch_pricing(identified_card)`` from ``service``.
Returns ``PricingPublic | None`` — None when the operator hasn't
configured a TCGplayer key (graceful no-op for personal-use without
an account) or any lookup step failed (graceful degrade so a flaky
TCGplayer API never blocks the cert page).
"""

from grader.services.pricing.service import fetch_pricing

__all__ = ["fetch_pricing"]
