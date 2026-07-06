"""
The core "fair value" model: given a market question, current market-implied
probability, relevant news, and social sentiment, produce an independent
AI probability estimate with reasoning.

Design: LLM does the qualitative reasoning (it can read news/context no
statistical model can), while an XGBoost model (optional, trained via
backtest/train_model.py once enough logged outcomes exist) provides a
quantitative prior based on structural features (time-to-resolution,
momentum, volatility, volume). The two are blended; early on, when no
trained model exists yet, the system runs LLM-only.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ai.confidence_score import ConfidenceInputs, compute_confidence
from ai.reasoning import LLMClient, get_llm_client
from config import settings
from logger import get_logger
from models import Market, NewsItem, ProbabilityEstimate, SocialSignal

log = get_logger(__name__)


class ProbabilityEngine:
    def __init__(self, llm_client: Optional[LLMClient] = None, quant_model=None):
        self._llm = llm_client or get_llm_client()
        self._quant_model = quant_model  # optional trained sklearn/xgboost model, see backtest/train_model.py

    async def estimate(
        self,
        market: Market,
        news: list[NewsItem],
        social: Optional[SocialSignal],
        quant_features: Optional[dict] = None,
    ) -> ProbabilityEstimate:
        llm_prob, llm_confidence, reasoning, cited_news = await self._llm_estimate(market, news, social)

        quant_prob = None
        if self._quant_model is not None and quant_features is not None:
            quant_prob = self._quant_estimate(quant_features)

        final_prob = llm_prob if quant_prob is None else self._blend(llm_prob, quant_prob)

        confidence = compute_confidence(ConfidenceInputs(
            llm_self_confidence=llm_confidence,
            ai_probability_yes=final_prob,
            social_signal=social,
            news_item_count=len(news),
            liquidity_usd=market.liquidity_usd,
            min_liquidity_usd=settings.MIN_LIQUIDITY_USD,
        ))

        edge = final_prob - market.yes_price

        return ProbabilityEstimate(
            market_id=market.market_id,
            ai_probability_yes=final_prob,
            market_probability_yes=market.yes_price,
            edge=edge,
            confidence=confidence,
            reasoning=reasoning,
            supporting_news=cited_news,
        )

    async def _llm_estimate(
        self, market: Market, news: list[NewsItem], social: Optional[SocialSignal]
    ) -> tuple[float, float, str, list[str]]:
        news_block = "\n".join(
            f"- [{n.source}] {n.title} ({n.published_at.isoformat()})" for n in news[:12]
        ) or "No recent relevant news found."

        social_block = "No social sentiment data available."
        if social is not None and social.sample_size > 0:
            lean = "bullish (favoring YES)" if social.bullish_score > 0 else "bearish (favoring NO)"
            social_block = (
                f"Aggregated social sentiment ({social.sample_size} posts): {lean}, "
                f"strength={abs(social.bullish_score):.2f}, confidence={social.confidence:.2f}, "
                f"estimated fake-news/rumor probability={social.fake_news_probability:.2f}."
            )

        prompt = f"""You are a professional quantitative forecaster estimating the TRUE probability
of a prediction-market question resolving YES, independent of the current market price.

QUESTION: {market.question}
MARKET CATEGORY: {market.category.value}
CURRENT MARKET PRICE (implied probability of YES): {market.yes_price:.2%}
MARKET ENDS: {market.end_date.isoformat() if market.end_date else "unknown"}
24H VOLUME: ${market.volume_24h_usd:,.0f}

RECENT NEWS:
{news_block}

SOCIAL SENTIMENT:
{social_block}

Think about base rates, the strength/reliability of the evidence above, and how much time remains
until resolution. Do NOT anchor on the current market price — estimate independently, then the
market price will be compared to your estimate to find mispricing.

Respond ONLY with JSON matching this schema:
{{
  "probability_yes": float between 0 and 1,
  "confidence": float between 0 and 1 (how confident YOU are in this estimate, not the market),
  "reasoning": "2-4 sentence explanation citing the specific evidence that drove your estimate",
  "key_news_titles": ["titles of the 1-3 most decision-relevant news items, or empty list"]
}}"""

        result = await self._llm.complete_json(prompt, max_tokens=600)
        if not result:
            # Degraded fallback: no signal, so best estimate is the market price itself
            # with low confidence, which will naturally fail the edge/confidence gates.
            return market.yes_price, 0.1, "LLM estimation failed; defaulting to market price.", []

        prob = max(0.0, min(1.0, float(result.get("probability_yes", market.yes_price))))
        conf = max(0.0, min(1.0, float(result.get("confidence", 0.3))))
        reasoning = result.get("reasoning", "")
        cited = result.get("key_news_titles", []) or []
        return prob, conf, reasoning, cited

    def _quant_estimate(self, features: dict) -> float:
        """features expected keys: momentum, volatility, days_to_resolution, volume_24h, liquidity."""
        vector = np.array([[
            features.get("momentum", 0.0),
            features.get("volatility", 0.0),
            features.get("days_to_resolution", 30.0),
            features.get("volume_24h", 0.0),
            features.get("liquidity", 0.0),
        ]])
        pred = self._quant_model.predict_proba(vector)[0][1]
        return float(pred)

    @staticmethod
    def _blend(llm_prob: float, quant_prob: float, llm_weight: float = 0.7) -> float:
        return llm_weight * llm_prob + (1 - llm_weight) * quant_prob
