"""
Replays historical market price series through the strategy pipeline to
measure how the edge-detection + sizing + exit rules would have performed,
without needing the LLM in the loop (backtests use the quant model or a
supplied historical "ai_probability" series — running the live LLM over
years of history is slow/expensive, so plug in either:
  (a) a trained quant model (backtest/train_model.py), or
  (b) pre-computed AI probability snapshots you've logged in production
      (see database/trade_log.py -> probability_estimates table) and are
      now replaying to validate the strategy layer in isolation.

This keeps the backtester fast and deterministic while still exercising
the exact same sizing/risk/exit code paths used live.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from config import settings
from logger import get_logger
from strategy.kelly import compute_position_size
from strategy.stoploss import check_exit, update_trailing_stop
from models import Outcome, Position

log = get_logger(__name__)


@dataclass
class BacktestResult:
    trades: list[dict] = field(default_factory=list)
    equity_curve: list[float] = field(default_factory=list)

    @property
    def total_trades(self) -> int:
        return len(self.trades)

    @property
    def win_rate(self) -> float:
        if not self.trades:
            return 0.0
        wins = sum(1 for t in self.trades if t["pnl_usd"] > 0)
        return wins / len(self.trades)

    @property
    def roi(self) -> float:
        if len(self.equity_curve) < 2:
            return 0.0
        return (self.equity_curve[-1] - self.equity_curve[0]) / self.equity_curve[0]

    @property
    def max_drawdown(self) -> float:
        if not self.equity_curve:
            return 0.0
        peak = self.equity_curve[0]
        max_dd = 0.0
        for equity in self.equity_curve:
            peak = max(peak, equity)
            dd = (peak - equity) / peak if peak > 0 else 0.0
            max_dd = max(max_dd, dd)
        return max_dd

    @property
    def avg_profit(self) -> float:
        profits = [t["pnl_usd"] for t in self.trades if t["pnl_usd"] > 0]
        return sum(profits) / len(profits) if profits else 0.0

    @property
    def avg_loss(self) -> float:
        losses = [t["pnl_usd"] for t in self.trades if t["pnl_usd"] <= 0]
        return sum(losses) / len(losses) if losses else 0.0

    def summary(self) -> dict:
        return {
            "total_trades": self.total_trades,
            "win_rate": round(self.win_rate, 4),
            "roi": round(self.roi, 4),
            "max_drawdown": round(self.max_drawdown, 4),
            "avg_profit_usd": round(self.avg_profit, 2),
            "avg_loss_usd": round(self.avg_loss, 2),
        }


def run_backtest(
    price_series: pd.DataFrame,
    ai_probabilities: pd.Series,
    market_id: str,
    starting_bankroll: float | None = None,
) -> BacktestResult:
    """
    price_series: DataFrame indexed by timestamp with a `yes_price` column (market price history).
    ai_probabilities: Series aligned to the same index, giving the AI's estimated true probability
                      at each timestamp (from quant model or logged historical estimates).
    """
    bankroll = starting_bankroll if starting_bankroll is not None else settings.STARTING_BANKROLL_USD
    result = BacktestResult(equity_curve=[bankroll])

    open_position: Position | None = None

    for ts, row in price_series.iterrows():
        price = float(row["yes_price"])
        ai_prob = float(ai_probabilities.get(ts, price))
        edge = ai_prob - price

        if open_position is None:
            if abs(edge) >= settings.MIN_EDGE_THRESHOLD:
                outcome = Outcome.YES if edge > 0 else Outcome.NO
                entry_price = price if outcome == Outcome.YES else 1 - price
                size_usd = compute_position_size(bankroll, ai_prob if outcome == Outcome.YES else 1 - ai_prob,
                                                  entry_price)
                if size_usd > 0:
                    shares = size_usd / entry_price if entry_price > 0 else 0
                    open_position = Position(
                        position_id=f"bt-{market_id}-{ts}",
                        market_id=market_id,
                        outcome=outcome,
                        entry_price=entry_price,
                        size_shares=shares,
                        size_usd=size_usd,
                        opened_at=ts,
                        stop_loss_price=entry_price * (1 - settings.STOP_LOSS_PCT),
                        profit_target_price=entry_price * (1 + settings.PROFIT_TARGET_PCT),
                    )
                    bankroll -= size_usd
        else:
            current_price = price if open_position.outcome == Outcome.YES else 1 - price
            update_trailing_stop(open_position, current_price)
            reason = check_exit(open_position, current_price)
            if reason:
                proceeds = open_position.size_shares * current_price
                pnl = proceeds - open_position.size_usd
                bankroll += proceeds
                result.trades.append({
                    "market_id": market_id,
                    "opened_at": open_position.opened_at,
                    "closed_at": ts,
                    "entry_price": open_position.entry_price,
                    "exit_price": current_price,
                    "size_usd": open_position.size_usd,
                    "pnl_usd": pnl,
                    "reason": reason,
                })
                open_position = None

        result.equity_curve.append(bankroll + (open_position.size_usd if open_position else 0))

    # Force-close anything still open at the end of the series at last known price
    if open_position is not None:
        last_price = float(price_series["yes_price"].iloc[-1])
        current_price = last_price if open_position.outcome == Outcome.YES else 1 - last_price
        proceeds = open_position.size_shares * current_price
        pnl = proceeds - open_position.size_usd
        bankroll += proceeds
        result.trades.append({
            "market_id": market_id,
            "opened_at": open_position.opened_at,
            "closed_at": price_series.index[-1],
            "entry_price": open_position.entry_price,
            "exit_price": current_price,
            "size_usd": open_position.size_usd,
            "pnl_usd": pnl,
            "reason": "series_end",
        })
        result.equity_curve.append(bankroll)

    return result
