"""
Turns a ProbabilityEstimate + market data into a TradeSignal, but only if
every gate passes:
  - |edge| exceeds MIN_EDGE_THRESHOLD
  - confidence exceeds MIN_CONFIDENCE_THRESHOLD
  - liquidity/spread checks pass (via OrderBook)
  - 24h volume exceeds MIN_VOLUME_24H_USD

This module has no side effects — it does not size positions (strategy/kelly.py)
or manage portfolio constraints (strategy/portfolio.py); it just decides
"is this market a real, tradeable opportunity."
"""
from __future__ import annotations

from typing import Optional

from config import settings
from logger import get_logger
from models import Market, OrderBook, Outcome, OrderSide, ProbabilityEstimate, TradeSignal

log = get_logger(__name__)


class EdgeDetector:
    def __init__(
        self,
        min_edge: Optional[float] = None,
        min_confidence: Optional[float] = None,
        min_volume_24h: Optional[float] = None,
    ):
        self.min_edge = min_edge if min_edge is not None else settings.MIN_EDGE_THRESHOLD
        self.min_confidence = min_confidence if min_confidence is not None else settings.MIN_CONFIDENCE_THRESHOLD
        self.min_volume_24h = min_volume_24h if min_volume_24h is not None else settings.MIN_VOLUME_24H_USD

    def evaluate(
        self,
        market: Market,
        estimate: ProbabilityEstimate,
        order_book: Optional[OrderBook] = None,
    ) -> Optional[TradeSignal]:
        reasons_failed = []

        if abs(estimate.edge) < self.min_edge:
            reasons_failed.append(f"edge {estimate.edge:.2%} below threshold {self.min_edge:.2%}")

        if estimate.confidence < self.min_confidence:
            reasons_failed.append(f"confidence {estimate.confidence:.2f} below threshold {self.min_confidence:.2f}")

        if market.volume_24h_usd < self.min_volume_24h:
            reasons_failed.append(f"24h volume ${market.volume_24h_usd:,.0f} below minimum")

        if order_book is not None:
            spread = order_book.spread_pct
            depth = order_book.depth_within_pct(0.02)
            if spread is None or spread > settings.MAX_SPREAD_PCT:
                reasons_failed.append(f"spread too wide or unavailable ({spread})")
            if depth < settings.MIN_LIQUIDITY_USD:
                reasons_failed.append(f"insufficient depth ${depth:,.0f}")
        else:
            reasons_failed.append("no order book data — cannot verify liquidity")

        if reasons_failed:
            log.debug("opportunity_rejected", market_id=market.market_id, reasons=reasons_failed)
            return None

        # Positive edge -> AI thinks YES is underpriced -> buy YES.
        # Negative edge -> AI thinks YES is overpriced -> buy NO.
        if estimate.edge > 0:
            outcome, limit_price = Outcome.YES, market.yes_price
        else:
            outcome, limit_price = Outcome.NO, market.no_price

        signal = TradeSignal(
            market_id=market.market_id,
            outcome=outcome,
            side=OrderSide.BUY,
            edge=estimate.edge,
            confidence=estimate.confidence,
            suggested_size_usd=0.0,  # filled in by strategy/kelly.py + portfolio.py
            limit_price=limit_price,
            reasoning=estimate.reasoning,
        )
        log.info(
            "opportunity_found",
            market_id=market.market_id,
            question=market.question[:80],
            edge=round(estimate.edge, 4),
            confidence=round(estimate.confidence, 4),
            outcome=outcome.value,
        )
        return signal
