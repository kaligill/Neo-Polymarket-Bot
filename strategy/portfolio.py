"""
Portfolio & risk manager. This is the single gatekeeper that decides whether
a sized TradeSignal is actually allowed to execute, given:
  - max daily loss (kill switch for the rest of the trading day)
  - max total exposure (% of bankroll deployed across all open positions)
  - max concurrent open positions
  - per-category diversification cap
  - per-position size cap

It also tracks realized/unrealized PnL and exposes the numbers the
dashboard needs (win rate, Sharpe, drawdown are computed in dashboard/metrics.py
from the position/trade history this class maintains).
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from config import settings
from logger import get_logger
from models import MarketCategory, Position, TradeSignal

log = get_logger(__name__)


class RiskLimitBreached(Exception):
    pass


class PortfolioManager:
    def __init__(self, starting_bankroll: Optional[float] = None):
        self.bankroll_usd: float = starting_bankroll if starting_bankroll is not None else settings.STARTING_BANKROLL_USD
        self.starting_bankroll_usd: float = self.bankroll_usd
        self.open_positions: dict[str, Position] = {}
        self.closed_positions: list[Position] = []
        self._daily_pnl_usd: float = 0.0
        self._daily_pnl_date: datetime = datetime.now(timezone.utc).date()

    # ---------- accounting helpers ----------

    def _reset_daily_pnl_if_new_day(self):
        today = datetime.now(timezone.utc).date()
        if today != self._daily_pnl_date:
            self._daily_pnl_date = today
            self._daily_pnl_usd = 0.0

    @property
    def total_exposure_usd(self) -> float:
        return sum(p.size_usd for p in self.open_positions.values())

    @property
    def total_equity_usd(self) -> float:
        # Simplified: assumes open positions marked at cost. A live system should
        # mark-to-market using current order book mid prices.
        return self.bankroll_usd + self.total_exposure_usd

    def positions_in_category(self, category: MarketCategory) -> int:
        return sum(1 for p in self.open_positions.values() if p.category == category)

    # ---------- risk gate ----------

    def can_open_position(self, signal: TradeSignal, size_usd: float, category: MarketCategory) -> tuple[bool, str]:
        self._reset_daily_pnl_if_new_day()

        if self._daily_pnl_usd <= -settings.MAX_DAILY_LOSS_PCT * self.starting_bankroll_usd:
            return False, "daily loss limit reached — trading halted for today"

        if len(self.open_positions) >= settings.MAX_OPEN_POSITIONS:
            return False, "max open positions reached"

        if self.positions_in_category(category) >= settings.MAX_POSITIONS_PER_CATEGORY:
            return False, f"max positions in category '{category.value}' reached"

        projected_exposure = self.total_exposure_usd + size_usd
        if projected_exposure > settings.MAX_EXPOSURE_PCT_OF_BANKROLL * self.total_equity_usd:
            return False, "max portfolio exposure reached"

        if size_usd > settings.MAX_POSITION_PCT_OF_BANKROLL * self.total_equity_usd:
            return False, "position size exceeds max per-position cap"

        if size_usd <= 0:
            return False, "computed position size is zero or negative — no edge after risk adjustment"

        if size_usd > self.bankroll_usd:
            return False, "insufficient free bankroll"

        return True, "ok"

    # ---------- position lifecycle ----------

    def open_position(self, signal: TradeSignal, size_usd: float, category: MarketCategory,
                       fill_price: float) -> Position:
        allowed, reason = self.can_open_position(signal, size_usd, category)
        if not allowed:
            raise RiskLimitBreached(reason)

        shares = size_usd / fill_price if fill_price > 0 else 0.0
        position = Position(
            position_id=f"{signal.market_id}:{signal.outcome.value}:{datetime.now(timezone.utc).timestamp()}",
            market_id=signal.market_id,
            outcome=signal.outcome,
            entry_price=fill_price,
            size_shares=shares,
            size_usd=size_usd,
            category=category,
            stop_loss_price=fill_price * (1 - settings.STOP_LOSS_PCT),
            profit_target_price=fill_price * (1 + settings.PROFIT_TARGET_PCT),
        )
        self.bankroll_usd -= size_usd
        self.open_positions[position.position_id] = position
        log.info("position_opened", position_id=position.position_id, market_id=signal.market_id,
                  size_usd=size_usd, entry_price=fill_price)
        return position

    def close_position(self, position_id: str, exit_price: float, reason: str) -> Position:
        position = self.open_positions.pop(position_id)
        proceeds = position.size_shares * exit_price
        pnl = proceeds - position.size_usd

        position.closed = True
        position.closed_at = datetime.now(timezone.utc)
        position.exit_price = exit_price
        position.realized_pnl_usd = pnl
        position.close_reason = reason

        self.bankroll_usd += proceeds
        self._daily_pnl_usd += pnl
        self.closed_positions.append(position)

        log.info("position_closed", position_id=position_id, pnl_usd=round(pnl, 2), reason=reason)
        return position

    # ---------- reporting ----------

    def summary(self) -> dict:
        realized_pnl = sum(p.realized_pnl_usd or 0.0 for p in self.closed_positions)
        wins = sum(1 for p in self.closed_positions if (p.realized_pnl_usd or 0) > 0)
        total_closed = len(self.closed_positions)
        win_rate = wins / total_closed if total_closed else 0.0

        return {
            "bankroll_usd": round(self.bankroll_usd, 2),
            "total_equity_usd": round(self.total_equity_usd, 2),
            "open_positions": len(self.open_positions),
            "total_exposure_usd": round(self.total_exposure_usd, 2),
            "realized_pnl_usd": round(realized_pnl, 2),
            "win_rate": round(win_rate, 4),
            "total_closed_trades": total_closed,
            "daily_pnl_usd": round(self._daily_pnl_usd, 2),
        }
