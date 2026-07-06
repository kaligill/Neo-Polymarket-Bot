"""
Fetches live order books from Polymarket's CLOB REST API.

Used to measure liquidity/spread before allowing a trade signal to fire,
and to determine realistic fill prices for execution.
"""
from __future__ import annotations

from typing import Optional

import httpx

from config import settings
from logger import get_logger
from models import OrderBook, OrderBookLevel, Outcome

log = get_logger(__name__)


class OrderBookClient:
    def __init__(self, base_url: Optional[str] = None, timeout: float = 10.0):
        self.base_url = base_url or settings.POLYMARKET_API_URL
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=timeout)

    async def close(self):
        await self._client.aclose()

    async def fetch_order_book(self, token_id: str, market_id: str, outcome: Outcome) -> Optional[OrderBook]:
        """token_id = the CLOB token id for the specific outcome (YES/NO) side."""
        try:
            resp = await self._client.get("/book", params={"token_id": token_id})
            resp.raise_for_status()
            raw = resp.json()
        except httpx.HTTPError as e:
            log.error("orderbook_fetch_failed", error=str(e), token_id=token_id)
            return None

        bids = sorted(
            (OrderBookLevel(price=float(b["price"]), size=float(b["size"])) for b in raw.get("bids", [])),
            key=lambda l: -l.price,
        )
        asks = sorted(
            (OrderBookLevel(price=float(a["price"]), size=float(a["size"])) for a in raw.get("asks", [])),
            key=lambda l: l.price,
        )

        return OrderBook(market_id=market_id, outcome=outcome, bids=bids, asks=asks)

    async def is_liquid_enough(self, book: OrderBook, min_liquidity_usd: Optional[float] = None,
                                max_spread_pct: Optional[float] = None) -> bool:
        min_liquidity_usd = min_liquidity_usd if min_liquidity_usd is not None else settings.MIN_LIQUIDITY_USD
        max_spread_pct = max_spread_pct if max_spread_pct is not None else settings.MAX_SPREAD_PCT

        depth = book.depth_within_pct(0.02)
        spread = book.spread_pct
        if spread is None:
            return False
        return depth >= min_liquidity_usd and spread <= max_spread_pct
