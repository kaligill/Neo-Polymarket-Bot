"""
Combines several independent quality signals into one final confidence score
(0-1) used to gate trades. Kept separate from probability_engine.py so the
weighting scheme can be tuned/backtested independently of the probability
model itself.

Inputs considered:
  - LLM's own self-reported confidence in its probability estimate
  - Agreement between LLM estimate and social sentiment direction
  - Freshness/volume of supporting news
  - Fake-news probability from social layer (penalizes confidence)
  - Market liquidity (thin markets get a confidence haircut — noisy pricing)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from models import SocialSignal


@dataclass
class ConfidenceInputs:
    llm_self_confidence: float          # 0-1, reported by the probability engine
    ai_probability_yes: float           # 0-1
    social_signal: Optional[SocialSignal]
    news_item_count: int
    liquidity_usd: float
    min_liquidity_usd: float


def compute_confidence(inputs: ConfidenceInputs) -> float:
    score = inputs.llm_self_confidence

    # Agreement bonus/penalty: does social sentiment direction match the AI's lean?
    if inputs.social_signal is not None and inputs.social_signal.sample_size > 0:
        ai_lean = 1.0 if inputs.ai_probability_yes >= 0.5 else -1.0
        agreement = ai_lean * inputs.social_signal.bullish_score  # positive if aligned
        agreement_adj = 0.10 * agreement * inputs.social_signal.confidence
        score += agreement_adj

        # Fake-news penalty scales with how much weight the social signal carries
        score -= 0.15 * inputs.social_signal.fake_news_probability * inputs.social_signal.confidence

    # News coverage bonus: more corroborating articles -> more confidence, capped
    news_bonus = min(inputs.news_item_count, 5) * 0.02
    score += news_bonus

    # Liquidity haircut: thin books make both pricing and fills unreliable
    if inputs.min_liquidity_usd > 0:
        liquidity_ratio = min(inputs.liquidity_usd / inputs.min_liquidity_usd, 2.0)
        if liquidity_ratio < 1.0:
            score *= 0.5 + 0.5 * liquidity_ratio  # scales down toward 0.5x at zero liquidity

    return max(0.0, min(1.0, score))
