"""
Position sizing models.

Kelly Criterion for a binary bet with price `p_market` (cost per share, 0-1)
and estimated true probability `p_true`:

    b = (1 - p_market) / p_market       # net odds received per $1 staked
    kelly_fraction = (b * p_true - (1 - p_true)) / b

We apply a fractional Kelly (config.KELLY_FRACTION) as a safety margin since
the true probability is itself an estimate with error, and cap the result
at MAX_POSITION_PCT_OF_BANKROLL regardless of what Kelly suggests.
"""
from __future__ import annotations

from config import settings
from logger import get_logger

log = get_logger(__name__)


def kelly_fraction(p_true: float, price: float) -> float:
    """Returns the raw (full) Kelly fraction of bankroll to stake. Can be negative
    (meaning no edge / negative edge at this price) — caller should floor at 0."""
    price = min(max(price, 1e-4), 1 - 1e-4)
    b = (1 - price) / price
    f = (b * p_true - (1 - p_true)) / b
    return f


def position_size_kelly(bankroll_usd: float, p_true: float, price: float,
                         kelly_multiplier: float | None = None) -> float:
    kelly_multiplier = kelly_multiplier if kelly_multiplier is not None else settings.KELLY_FRACTION
    f = kelly_fraction(p_true, price)
    f = max(f, 0.0) * kelly_multiplier
    f = min(f, settings.MAX_POSITION_PCT_OF_BANKROLL)
    return round(bankroll_usd * f, 2)


def position_size_fixed(bankroll_usd: float) -> float:
    size = bankroll_usd * settings.FIXED_RISK_PCT_PER_TRADE
    size = min(size, bankroll_usd * settings.MAX_POSITION_PCT_OF_BANKROLL)
    return round(size, 2)


def compute_position_size(bankroll_usd: float, p_true: float, price: float) -> float:
    if settings.RISK_MODEL == "kelly":
        return position_size_kelly(bankroll_usd, p_true, price)
    return position_size_fixed(bankroll_usd)
