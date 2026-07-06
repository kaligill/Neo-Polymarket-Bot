"""
Evaluates every open position each cycle against exit rules:
  - profit target
  - stop loss
  - trailing stop (ratchets up as price improves, exits on pullback)
  - max hold duration (time-based safety net)

Returns a list of (position_id, exit_reason) for positions that should be
closed this cycle; strategy/execution.py handles actually placing the exit.
"""
from __future__ import annotations

from datetime import datetime, timezone

from config import settings
from logger import get_logger
from models import Position

log = get_logger(__name__)


def update_trailing_stop(position: Position, current_price: float) -> Position:
    if not settings.TRAILING_STOP_ENABLED:
        return position

    candidate = current_price * (1 - settings.TRAILING_STOP_PCT)
    if position.trailing_stop_price is None or candidate > position.trailing_stop_price:
        # Only ratchet upward, and only once position is in profit territory
        if current_price > position.entry_price:
            position.trailing_stop_price = candidate
    return position


def check_exit(position: Position, current_price: float) -> str | None:
    """Returns an exit reason string if the position should be closed, else None."""
    now = datetime.now(timezone.utc)
    hold_days = (now - position.opened_at).total_seconds() / 86400

    if position.profit_target_price is not None and current_price >= position.profit_target_price:
        return "profit_target"

    if position.stop_loss_price is not None and current_price <= position.stop_loss_price:
        return "stop_loss"

    if (
        settings.TRAILING_STOP_ENABLED
        and position.trailing_stop_price is not None
        and current_price <= position.trailing_stop_price
    ):
        return "trailing_stop"

    if hold_days >= settings.MAX_HOLD_DAYS:
        return "max_hold_time"

    return None


def evaluate_positions(positions: dict[str, Position], current_prices: dict[str, float]) -> list[tuple[str, str]]:
    """current_prices maps position_id -> latest mark price for that position's outcome side."""
    to_close = []
    for position_id, position in positions.items():
        price = current_prices.get(position_id)
        if price is None:
            continue
        update_trailing_stop(position, price)
        reason = check_exit(position, price)
        if reason:
            to_close.append((position_id, reason))
            log.info("exit_triggered", position_id=position_id, reason=reason, price=price)
    return to_close
