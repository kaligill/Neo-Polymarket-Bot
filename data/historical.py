"""
Historical price/volume series for closed and active markets, used by the
backtesting engine and by the probability engine for feature generation
(e.g. momentum, realized volatility).
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

import httpx
import pandas as pd

from config import settings
from logger import get_logger

log = get_logger(__name__)


class HistoricalDataClient:
    def __init__(self, base_url: Optional[str] = None, timeout: float = 15.0):
        self.base_url = base_url or settings.POLYMARKET_GAMMA_API_URL
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=timeout)

    async def close(self):
        await self._client.aclose()

    async def fetch_price_history(self, market_id: str, interval: str = "1h") -> pd.DataFrame:
        """
        Returns a DataFrame indexed by timestamp with columns: yes_price, volume.
        `interval` maps to Polymarket's timeseries resolution parameter.
        """
        try:
            resp = await self._client.get(
                "/prices-history",
                params={"market": market_id, "interval": interval, "fidelity": 10},
            )
            resp.raise_for_status()
            raw = resp.json()
        except httpx.HTTPError as e:
            log.error("history_fetch_failed", error=str(e), market_id=market_id)
            return pd.DataFrame(columns=["yes_price", "volume"])

        points = raw.get("history", raw if isinstance(raw, list) else [])
        if not points:
            return pd.DataFrame(columns=["yes_price", "volume"])

        df = pd.DataFrame(points)
        if "t" in df.columns:
            df["timestamp"] = pd.to_datetime(df["t"], unit="s")
        elif "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.rename(columns={"p": "yes_price", "v": "volume"})
        df = df.set_index("timestamp")
        for col in ("yes_price", "volume"):
            if col not in df.columns:
                df[col] = None
        return df[["yes_price", "volume"]].sort_index()

    @staticmethod
    def realized_volatility(df: pd.DataFrame, window: int = 24) -> float:
        if df.empty or "yes_price" not in df:
            return 0.0
        returns = df["yes_price"].astype(float).diff().dropna()
        if len(returns) < 2:
            return 0.0
        return float(returns.rolling(window=min(window, len(returns))).std().iloc[-1] or 0.0)

    @staticmethod
    def momentum(df: pd.DataFrame, lookback: int = 12) -> float:
        """Simple price momentum over the last `lookback` bars, as a fraction."""
        if df.empty or len(df) < lookback + 1:
            return 0.0
        recent = df["yes_price"].astype(float)
        return float(recent.iloc[-1] - recent.iloc[-lookback - 1])
