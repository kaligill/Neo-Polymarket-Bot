"""
Order execution layer.

SAFETY MODEL:
  - settings.PAPER_TRADING defaults to True. In this mode, ALL orders are
    simulated against live order book prices but no funds move. This is the
    correct mode for development, backtridging live behavior, and building
    confidence in the model before risking capital.
  - Live execution (PAPER_TRADING=False) requires POLYMARKET_API_KEY/SECRET/
    PASSPHRASE and a signing wallet, and depends on Polymarket's official
    `py-clob-client` package (not bundled here — add it to requirements.txt
    yourself once you've read their docs and are ready to go live:
    https://docs.polymarket.com/). This module defines the interface so
    swapping the simulated client for the real one is a one-line change.

Regardless of mode, every fill (simulated or real) is persisted via
database/trade_log.py so backtesting and the dashboard see identical data
shapes.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional, Protocol

from config import settings
from logger import get_logger
from models import OrderBook, OrderSide, Outcome, TradeSignal

log = get_logger(__name__)


class ExecutionResult:
    def __init__(self, order_id: str, filled: bool, fill_price: float, filled_size_usd: float, message: str = ""):
        self.order_id = order_id
        self.filled = filled
        self.fill_price = fill_price
        self.filled_size_usd = filled_size_usd
        self.message = message
        self.timestamp = datetime.now(timezone.utc)


class ExecutionClient(Protocol):
    async def place_order(self, market_id: str, outcome: Outcome, side: OrderSide,
                           size_usd: float, limit_price: float, order_book: Optional[OrderBook]) -> ExecutionResult:
        ...


class PaperExecutionClient:
    """Simulates fills against the current order book with a conservative
    slippage model: walks the book consuming liquidity level by level."""

    def __init__(self, slippage_buffer_pct: float = 0.005):
        self.slippage_buffer_pct = slippage_buffer_pct

    async def place_order(self, market_id: str, outcome: Outcome, side: OrderSide,
                           size_usd: float, limit_price: float, order_book: Optional[OrderBook]) -> ExecutionResult:
        order_id = f"paper-{uuid.uuid4().hex[:12]}"

        if order_book is None:
            # No book data: assume fill at limit_price plus a slippage buffer
            fill_price = limit_price * (1 + self.slippage_buffer_pct)
            log.warning("paper_fill_no_book", market_id=market_id, order_id=order_id)
            return ExecutionResult(order_id, True, fill_price, size_usd, "filled without book data (estimated)")

        levels = order_book.asks if side == OrderSide.BUY else order_book.bids
        if not levels:
            return ExecutionResult(order_id, False, 0.0, 0.0, "no liquidity on relevant side of book")

        remaining_usd = size_usd
        cost_usd = 0.0
        shares_filled = 0.0

        for level in levels:
            if remaining_usd <= 0:
                break
            level_value_usd = level.price * level.size
            take_usd = min(remaining_usd, level_value_usd)
            take_shares = take_usd / level.price if level.price > 0 else 0
            cost_usd += take_usd
            shares_filled += take_shares
            remaining_usd -= take_usd

        if shares_filled == 0:
            return ExecutionResult(order_id, False, 0.0, 0.0, "unable to fill any size")

        avg_fill_price = cost_usd / shares_filled
        # Respect the limit price as a worst-case guard
        worst_acceptable = limit_price * (1 + self.slippage_buffer_pct)
        if side == OrderSide.BUY and avg_fill_price > worst_acceptable:
            return ExecutionResult(order_id, False, 0.0, 0.0,
                                    f"avg fill price {avg_fill_price:.4f} exceeds limit+slippage {worst_acceptable:.4f}")

        filled_usd = cost_usd
        if filled_usd < size_usd * 0.5:
            log.info("paper_partial_fill", market_id=market_id, requested=size_usd, filled=filled_usd)

        return ExecutionResult(order_id, True, round(avg_fill_price, 4), round(filled_usd, 2), "filled (simulated)")


class LiveExecutionClient:
    """
    Placeholder for real Polymarket CLOB execution. Intentionally raises until
    wired up — this stops the bot from ever silently "faking" a live trade.

    To implement: install `py-clob-client`, initialize it with
    settings.POLYMARKET_API_KEY/SECRET/PASSPHRASE and a signer built from
    settings.POLYMARKET_WALLET_PRIVATE_KEY, then translate place_order calls
    into their `create_order` / `post_order` calls. Test exhaustively on
    Polymarket's Mumbai/testnet environment (if available) before mainnet.
    """

    async def place_order(self, market_id: str, outcome: Outcome, side: OrderSide,
                           size_usd: float, limit_price: float, order_book: Optional[OrderBook]) -> ExecutionResult:
        raise NotImplementedError(
            "Live execution is not implemented. Set PAPER_TRADING=True, or implement "
            "LiveExecutionClient using Polymarket's py-clob-client before enabling live trading."
        )


def get_execution_client() -> ExecutionClient:
    if settings.PAPER_TRADING:
        return PaperExecutionClient()
    log.warning("live_trading_enabled", message="PAPER_TRADING is False — real capital is at risk.")
    return LiveExecutionClient()


async def execute_signal(client: ExecutionClient, signal: TradeSignal, size_usd: float,
                          order_book: Optional[OrderBook]) -> ExecutionResult:
    log.info("executing_order", market_id=signal.market_id, outcome=signal.outcome.value,
              side=signal.side.value, size_usd=size_usd, paper=settings.PAPER_TRADING)
    return await client.place_order(
        market_id=signal.market_id,
        outcome=signal.outcome,
        side=signal.side,
        size_usd=size_usd,
        limit_price=signal.limit_price,
        order_book=order_book,
    )
