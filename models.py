"""
Shared dataclasses / pydantic models used across data, ai, and strategy layers.
Keeping these in one place avoids circular imports between modules.
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


def _utcnow() -> datetime:
    """Timezone-aware UTC now, used as a default_factory everywhere so datetime
    arithmetic elsewhere in the codebase never mixes naive/aware values."""
    return datetime.now(timezone.utc)


class MarketCategory(str, Enum):
    POLITICS = "politics"
    CRYPTO = "crypto"
    SPORTS = "sports"
    ECONOMICS = "economics"
    WEATHER = "weather"
    POP_CULTURE = "pop_culture"
    SCIENCE_TECH = "science_tech"
    OTHER = "other"


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class Outcome(str, Enum):
    YES = "yes"
    NO = "no"


class Market(BaseModel):
    """A single Polymarket binary (or multi-outcome) market/event."""

    market_id: str
    condition_id: Optional[str] = None
    question: str
    slug: Optional[str] = None
    category: MarketCategory = MarketCategory.OTHER
    yes_price: float = Field(ge=0, le=1)
    no_price: float = Field(ge=0, le=1)
    volume_24h_usd: float = 0.0
    liquidity_usd: float = 0.0
    spread_pct: float = 0.0
    end_date: Optional[datetime] = None
    active: bool = True
    closed: bool = False
    tags: list[str] = Field(default_factory=list)
    description: Optional[str] = None
    updated_at: datetime = Field(default_factory=_utcnow)


class OrderBookLevel(BaseModel):
    price: float
    size: float


class OrderBook(BaseModel):
    market_id: str
    outcome: Outcome
    bids: list[OrderBookLevel] = Field(default_factory=list)
    asks: list[OrderBookLevel] = Field(default_factory=list)
    fetched_at: datetime = Field(default_factory=_utcnow)

    @property
    def best_bid(self) -> Optional[float]:
        return self.bids[0].price if self.bids else None

    @property
    def best_ask(self) -> Optional[float]:
        return self.asks[0].price if self.asks else None

    @property
    def mid_price(self) -> Optional[float]:
        if self.best_bid is None or self.best_ask is None:
            return None
        return (self.best_bid + self.best_ask) / 2

    @property
    def spread_pct(self) -> Optional[float]:
        if self.best_bid is None or self.best_ask is None or self.mid_price in (None, 0):
            return None
        return (self.best_ask - self.best_bid) / self.mid_price

    def depth_within_pct(self, pct: float) -> float:
        """USD liquidity resting within `pct` of mid on both sides combined."""
        mid = self.mid_price
        if mid is None:
            return 0.0
        total = 0.0
        for lvl in self.bids:
            if mid - lvl.price <= mid * pct:
                total += lvl.price * lvl.size
        for lvl in self.asks:
            if lvl.price - mid <= mid * pct:
                total += lvl.price * lvl.size
        return total


class NewsItem(BaseModel):
    source: str
    title: str
    url: str
    published_at: datetime
    summary: Optional[str] = None
    related_keywords: list[str] = Field(default_factory=list)


class SocialSignal(BaseModel):
    platform: str  # "twitter" | "reddit" | "telegram"
    keyword: str
    bullish_score: float = Field(ge=-1, le=1)   # -1 fully bearish, +1 fully bullish
    bearish_score: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)
    fake_news_probability: float = Field(ge=0, le=1)
    sample_size: int = 0
    fetched_at: datetime = Field(default_factory=_utcnow)


class ProbabilityEstimate(BaseModel):
    market_id: str
    ai_probability_yes: float = Field(ge=0, le=1)
    market_probability_yes: float = Field(ge=0, le=1)
    edge: float  # ai_probability_yes - market_probability_yes
    confidence: float = Field(ge=0, le=1)
    reasoning: str
    supporting_news: list[str] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=_utcnow)


class TradeSignal(BaseModel):
    market_id: str
    outcome: Outcome
    side: OrderSide
    edge: float
    confidence: float
    suggested_size_usd: float
    limit_price: float
    reasoning: str


class Position(BaseModel):
    position_id: str
    market_id: str
    outcome: Outcome
    entry_price: float
    size_shares: float
    size_usd: float
    opened_at: datetime = Field(default_factory=_utcnow)
    category: MarketCategory = MarketCategory.OTHER
    stop_loss_price: Optional[float] = None
    profit_target_price: Optional[float] = None
    trailing_stop_price: Optional[float] = None
    closed: bool = False
    closed_at: Optional[datetime] = None
    exit_price: Optional[float] = None
    realized_pnl_usd: Optional[float] = None
    close_reason: Optional[str] = None


class TradeLogEntry(BaseModel):
    timestamp: datetime = Field(default_factory=_utcnow)
    market_id: str
    outcome: Outcome
    side: OrderSide
    price: float
    size_usd: float
    ai_confidence: float
    ai_probability: float
    market_probability: float
    edge: float
    reasoning: str
    pnl_usd: Optional[float] = None
