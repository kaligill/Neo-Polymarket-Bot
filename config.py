"""
Central configuration for Neo Prediction Engine.

All tunables live here so strategy behavior can be adjusted without
touching business logic. Values are loaded from environment variables
(.env file supported) with sane defaults for a paper-trading setup.
"""
from __future__ import annotations

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # ---- Runtime mode ----
    ENV: str = "development"
    PAPER_TRADING: bool = True  # NEVER flip to False without understanding the execution module fully

    # ---- Polymarket ----
    POLYMARKET_API_URL: str = "https://clob.polymarket.com"
    POLYMARKET_GAMMA_API_URL: str = "https://gamma-api.polymarket.com"
    POLYMARKET_WS_URL: str = "wss://ws-subscriptions-clob.polymarket.com/ws"
    POLYMARKET_API_KEY: str = ""
    POLYMARKET_API_SECRET: str = ""
    POLYMARKET_API_PASSPHRASE: str = ""
    POLYMARKET_WALLET_PRIVATE_KEY: str = ""  # required only for live execution signing
    POLYMARKET_FUNDER_ADDRESS: str = ""

    # ---- Database ----
    DATABASE_URL: str = "postgresql+asyncpg://neo:neo@localhost:5432/neo_polymarket"
    REDIS_URL: str = "redis://localhost:6379/0"

    # ---- LLM providers ----
    LLM_PROVIDER: str = "anthropic"  # "anthropic" | "openai" | "openrouter"
    LLM_MODEL: str = "deepseek/deepseek-chat-v3.1"  # used for openrouter
    ANTHROPIC_API_KEY: str = ""
    ANTHROPIC_MODEL: str = "claude-sonnet-4-6"
    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "gpt-4o"
    OPENROUTER_API_KEY: str = ""
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"

    # ---- News / Social sources ----
    NEWSAPI_KEY: str = ""
    TWITTER_BEARER_TOKEN: str = ""
    REDDIT_CLIENT_ID: str = ""
    REDDIT_CLIENT_SECRET: str = ""
    REDDIT_USER_AGENT: str = "neo-polymarket-bot/0.1"

    # ---- Scanner cadence ----
    MARKET_SCAN_INTERVAL_SECONDS: int = 15
    ORDERBOOK_REFRESH_SECONDS: int = 5
    NEWS_REFRESH_SECONDS: int = 120
    SOCIAL_REFRESH_SECONDS: int = 90

    # ---- Edge / opportunity thresholds ----
    MIN_EDGE_THRESHOLD: float = 0.07          # AI probability must diverge from market by 7%+
    MIN_CONFIDENCE_THRESHOLD: float = 0.65    # AI confidence score (0-1) required to act
    MIN_LIQUIDITY_USD: float = 500.0          # minimum resting liquidity near touch
    MAX_SPREAD_PCT: float = 0.05              # reject markets with >5% bid/ask spread
    MIN_VOLUME_24H_USD: float = 1000.0

    # ---- Risk management ----
    RISK_MODEL: str = "kelly"                 # "fixed" | "kelly"
    KELLY_FRACTION: float = 0.25              # fractional Kelly (safety multiplier)
    FIXED_RISK_PCT_PER_TRADE: float = 0.01     # used if RISK_MODEL == fixed
    MAX_POSITION_PCT_OF_BANKROLL: float = 0.05
    MAX_EXPOSURE_PCT_OF_BANKROLL: float = 0.40  # total capital deployed across all open positions
    MAX_DAILY_LOSS_PCT: float = 0.05           # halts trading for the day if breached
    MAX_OPEN_POSITIONS: int = 15
    MAX_POSITIONS_PER_CATEGORY: int = 4         # diversification cap (e.g. politics, sports, crypto)

    # ---- Exit rules ----
    PROFIT_TARGET_PCT: float = 0.35            # take profit once position gains this much
    STOP_LOSS_PCT: float = 0.20                # cut loss once position drops this much
    MAX_HOLD_DAYS: int = 30                    # time-based exit safety net
    TRAILING_STOP_ENABLED: bool = True
    TRAILING_STOP_PCT: float = 0.10

    # ---- Starting bankroll (paper trading) ----
    STARTING_BANKROLL_USD: float = 10_000.0

    # ---- Logging ----
    LOG_LEVEL: str = "INFO"
    LOG_DIR: str = "logs"


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
