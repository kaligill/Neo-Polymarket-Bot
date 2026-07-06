"""
Aggregates raw social posts into SocialSignal objects using an LLM classifier
in batches (cheaper + more consistent than one call per post). Falls back to
a lightweight lexicon scorer if no LLM key is configured, so the system still
runs end-to-end in a degraded mode.
"""
from __future__ import annotations

import json
from typing import Optional

from config import settings
from logger import get_logger
from models import SocialSignal

log = get_logger(__name__)

_BULLISH_WORDS = {"win", "confirmed", "surge", "approved", "victory", "bullish", "yes", "pass", "beat"}
_BEARISH_WORDS = {"lose", "denied", "crash", "rejected", "loss", "bearish", "no", "fail", "miss"}
_FAKE_MARKERS = {"unconfirmed", "rumor", "alleged", "according to sources", "breaking (unverified)"}


class SentimentAnalyzer:
    def __init__(self, llm_client=None):
        # llm_client is injected (see ai/reasoning.py's get_llm_client) to avoid
        # duplicating provider setup logic in every module.
        self._llm = llm_client

    async def score_posts(self, platform: str, keyword: str, posts: list[dict]) -> SocialSignal:
        if not posts:
            return SocialSignal(
                platform=platform, keyword=keyword, bullish_score=0.0,
                bearish_score=0.0, confidence=0.0, fake_news_probability=0.0, sample_size=0,
            )

        if self._llm is not None:
            try:
                return await self._score_with_llm(platform, keyword, posts)
            except Exception as e:
                log.warning("llm_sentiment_failed_falling_back", error=str(e))

        return self._score_with_lexicon(platform, keyword, posts)

    async def _score_with_llm(self, platform: str, keyword: str, posts: list[dict]) -> SocialSignal:
        text_samples = "\n---\n".join(
            (p.get("text") or p.get("title") or "")[:280] for p in posts[:40]
        )
        prompt = f"""You are a social-sentiment classifier for prediction-market trading.
Topic/keyword: "{keyword}"
Below are {len(posts)} recent {platform} posts about this topic. Analyze overall sentiment.

POSTS:
{text_samples}

Respond ONLY with JSON, no prose, matching this schema:
{{"bullish_score": float between -1 and 1, "confidence": float 0-1, "fake_news_probability": float 0-1}}
bullish_score: -1 fully bearish/negative for the topic resolving YES, +1 fully bullish/positive.
confidence: how consistent/clear the signal is across posts (low if contradictory or sparse).
fake_news_probability: likelihood the dominant narrative is based on unverified/rumor content."""

        raw = await self._llm.complete_json(prompt)
        bullish = float(raw.get("bullish_score", 0.0))
        confidence = float(raw.get("confidence", 0.3))
        fake_prob = float(raw.get("fake_news_probability", 0.2))

        return SocialSignal(
            platform=platform,
            keyword=keyword,
            bullish_score=max(-1.0, min(1.0, bullish)),
            bearish_score=max(0.0, -min(0.0, bullish)),
            confidence=max(0.0, min(1.0, confidence)),
            fake_news_probability=max(0.0, min(1.0, fake_prob)),
            sample_size=len(posts),
        )

    def _score_with_lexicon(self, platform: str, keyword: str, posts: list[dict]) -> SocialSignal:
        """Deterministic fallback: simple keyword-count scoring, weighted by engagement."""
        total_weight = 0.0
        bull_weight = 0.0
        bear_weight = 0.0
        fake_hits = 0

        for p in posts:
            text = (p.get("text") or p.get("title") or "").lower()
            weight = 1.0 + p.get("score", 0) * 0.01 + p.get("num_comments", 0) * 0.02
            total_weight += weight
            bull_weight += weight * sum(1 for w in _BULLISH_WORDS if w in text)
            bear_weight += weight * sum(1 for w in _BEARISH_WORDS if w in text)
            if any(m in text for m in _FAKE_MARKERS):
                fake_hits += 1

        denom = max(bull_weight + bear_weight, 1.0)
        bullish_score = (bull_weight - bear_weight) / denom
        confidence = min(1.0, total_weight / (len(posts) * 3)) if posts else 0.0
        fake_prob = min(1.0, fake_hits / max(len(posts), 1))

        return SocialSignal(
            platform=platform,
            keyword=keyword,
            bullish_score=max(-1.0, min(1.0, bullish_score)),
            bearish_score=max(0.0, -min(0.0, bullish_score)),
            confidence=confidence,
            fake_news_probability=fake_prob,
            sample_size=len(posts),
        )


def combine_signals(signals: list[SocialSignal]) -> Optional[SocialSignal]:
    """Combine multiple platform signals into one aggregate, weighting by confidence*sample_size."""
    if not signals:
        return None
    weights = [max(s.confidence * (1 + s.sample_size), 0.01) for s in signals]
    total_weight = sum(weights)

    bullish = sum(s.bullish_score * w for s, w in zip(signals, weights)) / total_weight
    fake_prob = sum(s.fake_news_probability * w for s, w in zip(signals, weights)) / total_weight
    confidence = sum(s.confidence * w for s, w in zip(signals, weights)) / total_weight
    sample_size = sum(s.sample_size for s in signals)

    return SocialSignal(
        platform="combined",
        keyword=signals[0].keyword,
        bullish_score=bullish,
        bearish_score=max(0.0, -bullish),
        confidence=confidence,
        fake_news_probability=fake_prob,
        sample_size=sample_size,
    )
