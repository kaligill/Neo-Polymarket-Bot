"""
Fetches active markets/events from Polymarket's public Gamma API.

Gamma API docs: https://docs.polymarket.com/
This module intentionally has zero trading-side effects; it is read-only.
"""
from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Optional

import httpx

from config import settings
from logger import get_logger
from models import Market, MarketCategory

log = get_logger(__name__)

CATEGORY_KEYWORDS: dict[MarketCategory, list[str]] = {
    MarketCategory.POLITICS: ["election", "president", "senate", "congress", "governor", "vote", "policy"],
    MarketCategory.CRYPTO: ["bitcoin", "btc", "ethereum", "eth", "crypto", "solana", "token"],
    MarketCategory.SPORTS: ["nfl", "nba", "mlb", "nhl", "soccer", "football", "match", "championship", "olympics"],
    MarketCategory.ECONOMICS: ["fed", "inflation", "rate", "gdp", "recession", "jobs report", "cpi"],
    MarketCategory.WEATHER: ["hurricane", "temperature", "storm", "weather", "climate"],
    MarketCategory.POP_CULTURE: ["oscar", "grammy", "movie", "celebrity", "album"],
    MarketCategory.SCIENCE_TECH: ["ai", "spacex", "launch", "openai", "nasa"],
}


def infer_category(question: str, tags: list[str]) -> MarketCategory:
    text = (question + " " + " ".join(tags)).lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(kw in text for kw in keywords):
            return category
    return MarketCategory.OTHER


class MarketFetcher:
    """Async client for pulling the current universe of active markets."""

    def __init__(self, base_url: Optional[str] = None, timeout: float = 15.0):
        self.base_url = base_url or settings.POLYMARKET_GAMMA_API_URL
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=timeout)

    async def close(self):
        await self._client.aclose()

    async def fetch_active_markets(self, limit: int = 500, offset: int = 0) -> list[Market]:
        """
        Pull active, non-closed markets. Paginates until it has retrieved
        `limit` markets or the API returns an empty page.
        """
        markets: list[Market] = []
        page_offset = offset
        page_size = min(limit, 100)

        while len(markets) < limit:
            try:
                resp = await self._client.get(
                    "/markets",
                    params={
                        "active": "true",
                        "closed": "false",
                        "limit": page_size,
                        "offset": page_offset,
                    },
                )
                resp.raise_for_status()
                page = resp.json()
            except httpx.HTTPError as e:
                log.error("market_fetch_failed", error=str(e), offset=page_offset)
                break

            if not page:
                break

            for raw in page:
                market = self._parse_market(raw)
                if market:
                    markets.append(market)

            page_offset += page_size
            if len(page) < page_size:
                break

        return markets[:limit]

    def _parse_market(self, raw: dict) -> Optional[Market]:
        try:
            outcome_prices = raw.get("outcomePrices")
            if isinstance(outcome_prices, str):
                import json
                outcome_prices = json.loads(outcome_prices)
            yes_price = float(outcome_prices[0]) if outcome_prices else 0.5
            no_price = float(outcome_prices[1]) if outcome_prices and len(outcome_prices) > 1 else 1 - yes_price

            tags = raw.get("tags") or []
            if isinstance(tags, str):
                tags = [t.strip() for t in tags.split(",") if t.strip()]

            question = raw.get("question", "")

            return Market(
                market_id=str(raw.get("id") or raw.get("conditionId") or raw.get("slug")),
                condition_id=raw.get("conditionId"),
                question=question,
                slug=raw.get("slug"),
                category=infer_category(question, tags),
                yes_price=yes_price,
                no_price=no_price,
                volume_24h_usd=float(raw.get("volume24hr") or 0.0),
                liquidity_usd=float(raw.get("liquidity") or 0.0),
                end_date=self._parse_date(raw.get("endDate")),
                active=bool(raw.get("active", True)),
                closed=bool(raw.get("closed", False)),
                tags=tags,
                description=raw.get("description"),
            )
        except (KeyError, ValueError, TypeError, IndexError) as e:
            log.warning("market_parse_failed", error=str(e), raw_id=raw.get("id"))
            return None

    @staticmethod
    def _parse_date(value: Optional[str]) -> Optional[datetime]:
        if not value:
            return None
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None


async def scan_loop(fetcher: MarketFetcher, on_markets, interval_seconds: Optional[int] = None):
    """Continuously poll for active markets and hand the batch to a callback."""
    interval = interval_seconds or settings.MARKET_SCAN_INTERVAL_SECONDS
    while True:
        markets = await fetcher.fetch_active_markets()
        log.info("market_scan_complete", count=len(markets))
        await on_markets(markets)
        await asyncio.sleep(interval)
