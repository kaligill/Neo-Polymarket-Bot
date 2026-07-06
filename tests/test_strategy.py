"""
Lightweight unit tests for the pieces that don't require network/LLM access —
sizing math and risk gating. Run with: pytest tests/
"""
from datetime import datetime, timezone

import pytest

from models import MarketCategory, Outcome, OrderSide, TradeSignal
from strategy.kelly import kelly_fraction, position_size_kelly, position_size_fixed
from strategy.portfolio import PortfolioManager, RiskLimitBreached
from strategy.stoploss import check_exit
from models import Position


def test_kelly_fraction_positive_edge():
    # Market price 0.50, true prob 0.70 -> should suggest a positive stake
    f = kelly_fraction(p_true=0.70, price=0.50)
    assert f > 0


def test_kelly_fraction_negative_edge():
    # Market price 0.70, true prob 0.50 -> no edge for buying at 0.70, negative Kelly
    f = kelly_fraction(p_true=0.50, price=0.70)
    assert f < 0


def test_position_size_kelly_capped():
    size = position_size_kelly(bankroll_usd=10_000, p_true=0.99, price=0.10)
    # Even a huge edge should never exceed the configured max position % cap
    from config import settings
    assert size <= 10_000 * settings.MAX_POSITION_PCT_OF_BANKROLL + 1e-6


def test_position_size_fixed():
    size = position_size_fixed(bankroll_usd=10_000)
    assert size > 0


def test_portfolio_blocks_when_daily_loss_exceeded():
    pm = PortfolioManager(starting_bankroll=1000)
    pm._daily_pnl_usd = -100  # simulate a 10% daily loss with default 5% cap
    signal = TradeSignal(
        market_id="m1", outcome=Outcome.YES, side=OrderSide.BUY,
        edge=0.1, confidence=0.8, suggested_size_usd=50, limit_price=0.5, reasoning="test",
    )
    allowed, reason = pm.can_open_position(signal, size_usd=50, category=MarketCategory.OTHER)
    assert not allowed
    assert "daily loss" in reason


def test_portfolio_opens_and_closes_position():
    pm = PortfolioManager(starting_bankroll=1000)
    signal = TradeSignal(
        market_id="m1", outcome=Outcome.YES, side=OrderSide.BUY,
        edge=0.1, confidence=0.8, suggested_size_usd=50, limit_price=0.5, reasoning="test",
    )
    position = pm.open_position(signal, size_usd=50, category=MarketCategory.OTHER, fill_price=0.5)
    assert position.size_shares == 100
    assert pm.bankroll_usd == 950

    closed = pm.close_position(position.position_id, exit_price=0.6, reason="profit_target")
    assert closed.realized_pnl_usd == pytest.approx(10.0)
    assert pm.bankroll_usd == pytest.approx(950 + 60)


def test_stoploss_triggers():
    position = Position(
        position_id="p1", market_id="m1", outcome=Outcome.YES,
        entry_price=0.5, size_shares=100, size_usd=50,
        stop_loss_price=0.4, profit_target_price=0.7,
    )
    assert check_exit(position, current_price=0.35) == "stop_loss"
    assert check_exit(position, current_price=0.75) == "profit_target"
    assert check_exit(position, current_price=0.5) is None
