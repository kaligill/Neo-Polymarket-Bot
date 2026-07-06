"""
Twitter/X data fetcher using the official API v2 (requires TWITTER_BEARER_TOKEN).

This module only fetches raw recent tweets matching a keyword/topic; scoring
(bullish/bearish/confidence/fake-news probability) happens in ai/sentiment.py
so this stays a thin, swappable data source.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import httpx

from config import settings
from logger import get_logger

log = get_logger(__name__)

TWITTER_RECENT_SEARCH_URL = "https://api.twitter.com/2/tweets/search/recent"


class TwitterFetcher:
    def __init__(self, timeout: float = 15.0):
        self._client = httpx.AsyncClient(timeout=timeout)

    async def close(self):
        await self._client.aclose()

    async def fetch_recent(self, query: str, max_results: int = 50) -> list[dict]:
        """Returns raw tweet dicts: {id, text, created_at, public_metrics, author_id}."""
        if not settings.TWITTER_BEARER_TOKEN:
            log.debug("twitter_skipped_no_token")
            return []

        params = {
            "query": f"{query} -is:retweet lang:en",
            "max_results": min(max_results, 100),
            "tweet.fields": "created_at,public_metrics,author_id",
        }
        headers = {"Authorization": f"Bearer {settings.TWITTER_BEARER_TOKEN}"}

        try:
            resp = await self._client.get(TWITTER_RECENT_SEARCH_URL, params=params, headers=headers)
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPError as e:
            log.error("twitter_fetch_failed", query=query, error=str(e))
            return []

        return data.get("data", [])

    @staticmethod
    def engagement_weight(tweet: dict) -> float:
        """Weight a tweet by engagement so viral posts count more in sentiment scoring."""
        metrics = tweet.get("public_metrics", {})
        return 1.0 + metrics.get("like_count", 0) * 0.01 + metrics.get("retweet_count", 0) * 0.03
