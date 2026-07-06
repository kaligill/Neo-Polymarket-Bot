"""
Reddit data fetcher using PRAW (requires REDDIT_CLIENT_ID/SECRET).

PRAW's client is synchronous, so calls are offloaded to a thread pool via
asyncio.to_thread to keep the rest of the pipeline async-friendly.
"""
from __future__ import annotations

import asyncio
from typing import Optional

import praw

from config import settings
from logger import get_logger

log = get_logger(__name__)

DEFAULT_SUBREDDITS = [
    "politics", "wallstreetbets", "cryptocurrency", "sports",
    "worldnews", "economics", "geopolitics",
]


class RedditFetcher:
    def __init__(self):
        self._reddit: Optional[praw.Reddit] = None
        if settings.REDDIT_CLIENT_ID and settings.REDDIT_CLIENT_SECRET:
            self._reddit = praw.Reddit(
                client_id=settings.REDDIT_CLIENT_ID,
                client_secret=settings.REDDIT_CLIENT_SECRET,
                user_agent=settings.REDDIT_USER_AGENT,
            )

    async def search(self, query: str, subreddits: Optional[list[str]] = None, limit: int = 30) -> list[dict]:
        if self._reddit is None:
            log.debug("reddit_skipped_no_credentials")
            return []
        subreddits = subreddits or DEFAULT_SUBREDDITS
        return await asyncio.to_thread(self._search_sync, query, subreddits, limit)

    def _search_sync(self, query: str, subreddits: list[str], limit: int) -> list[dict]:
        results = []
        try:
            subreddit = self._reddit.subreddit("+".join(subreddits))
            for submission in subreddit.search(query, limit=limit, sort="new"):
                results.append({
                    "id": submission.id,
                    "title": submission.title,
                    "selftext": submission.selftext,
                    "score": submission.score,
                    "num_comments": submission.num_comments,
                    "created_utc": submission.created_utc,
                    "subreddit": str(submission.subreddit),
                    "url": submission.url,
                })
        except Exception as e:  # praw raises assorted prawcore exceptions
            log.error("reddit_search_failed", query=query, error=str(e))
        return results

    @staticmethod
    def engagement_weight(post: dict) -> float:
        return 1.0 + post.get("score", 0) * 0.02 + post.get("num_comments", 0) * 0.05
