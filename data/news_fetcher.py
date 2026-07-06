"""
News intelligence layer.

Pulls from two sources:
1. Public RSS feeds of major outlets (no key required) — Reuters, AP, BBC,
   CNBC, Bloomberg (where available), plus a slot for government feeds.
2. NewsAPI.org for keyword-targeted search when a market's topic needs
   deeper coverage (requires NEWSAPI_KEY).

Both paths return a normalized list[NewsItem] so the rest of the pipeline
doesn't care where an article came from.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import feedparser
import httpx

from config import settings
from logger import get_logger
from models import NewsItem

log = get_logger(__name__)

# Public RSS feeds. Swap/add URLs as outlets change theirs.
RSS_FEEDS: dict[str, str] = {
    "Reuters World": "https://www.reutersagency.com/feed/?best-topics=top-news",
    "AP Top News": "https://apnews.com/apf-topnews",
    "BBC World": "https://feeds.bbci.co.uk/news/world/rss.xml",
    "CNBC Top News": "https://www.cnbc.com/id/100003114/device/rss/rss.html",
    "Bloomberg Markets": "https://feeds.bloomberg.com/markets/news.rss",
}


class NewsFetcher:
    def __init__(self, timeout: float = 15.0):
        self._client = httpx.AsyncClient(timeout=timeout)

    async def close(self):
        await self._client.aclose()

    async def fetch_rss_all(self) -> list[NewsItem]:
        items: list[NewsItem] = []
        for source, url in RSS_FEEDS.items():
            items.extend(await self._fetch_rss_feed(source, url))
        return items

    async def _fetch_rss_feed(self, source: str, url: str) -> list[NewsItem]:
        try:
            resp = await self._client.get(url, headers={"User-Agent": "neo-polymarket-bot/0.1"})
            resp.raise_for_status()
        except httpx.HTTPError as e:
            log.warning("rss_fetch_failed", source=source, error=str(e))
            return []

        parsed = feedparser.parse(resp.text)
        items = []
        for entry in parsed.entries[:50]:
            published = self._parse_entry_date(entry)
            items.append(
                NewsItem(
                    source=source,
                    title=entry.get("title", ""),
                    url=entry.get("link", ""),
                    published_at=published,
                    summary=entry.get("summary"),
                )
            )
        return items

    async def search_newsapi(self, query: str, page_size: int = 20) -> list[NewsItem]:
        if not settings.NEWSAPI_KEY:
            log.debug("newsapi_skipped_no_key")
            return []
        try:
            resp = await self._client.get(
                "https://newsapi.org/v2/everything",
                params={
                    "q": query,
                    "pageSize": page_size,
                    "sortBy": "publishedAt",
                    "language": "en",
                    "apiKey": settings.NEWSAPI_KEY,
                },
            )
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPError as e:
            log.error("newsapi_search_failed", query=query, error=str(e))
            return []

        items = []
        for article in data.get("articles", []):
            items.append(
                NewsItem(
                    source=article.get("source", {}).get("name", "NewsAPI"),
                    title=article.get("title", ""),
                    url=article.get("url", ""),
                    published_at=self._safe_parse_iso(article.get("publishedAt")),
                    summary=article.get("description"),
                    related_keywords=[query],
                )
            )
        return items

    @staticmethod
    def _parse_entry_date(entry) -> datetime:
        for field in ("published_parsed", "updated_parsed"):
            struct = getattr(entry, field, None)
            if struct:
                return datetime(*struct[:6], tzinfo=timezone.utc)
        return datetime.now(timezone.utc)

    @staticmethod
    def _safe_parse_iso(value: Optional[str]) -> datetime:
        if not value:
            return datetime.now(timezone.utc)
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return datetime.now(timezone.utc)


def keywords_for_market(question: str) -> list[str]:
    """Cheap keyword extraction — pulls capitalized/quoted tokens as search seeds.
    For production, replace with an LLM-based keyword extraction call."""
    import re
    tokens = re.findall(r"[A-Z][a-zA-Z]{2,}(?:\s[A-Z][a-zA-Z]{2,})*", question)
    seen, out = set(), []
    for t in tokens:
        if t.lower() not in seen:
            seen.add(t.lower())
            out.append(t)
    return out[:5] or [question[:60]]
