"""
Persists every trade/estimate to Postgres and also appends a JSONL audit
trail to disk (logs/trades.jsonl) so nothing is lost even if the DB write
fails — trading systems should never lose a record of what they did.
"""
from __future__ import annotations

import json
import os
from datetime import datetime

from config import settings
from database.db import PositionRecord, ProbabilityEstimateRecord, TradeRecord, get_session
from logger import get_logger
from models import Position, ProbabilityEstimate, TradeLogEntry

log = get_logger(__name__)

_JSONL_PATH = os.path.join(settings.LOG_DIR, "trades.jsonl")


def _append_jsonl(entry: dict):
    os.makedirs(settings.LOG_DIR, exist_ok=True)
    with open(_JSONL_PATH, "a") as f:
        f.write(json.dumps(entry, default=str) + "\n")


async def log_trade(entry: TradeLogEntry, question: str = "", paper_trade: bool = True):
    _append_jsonl(entry.model_dump())
    try:
        async with get_session() as session:
            record = TradeRecord(
                timestamp=entry.timestamp,
                market_id=entry.market_id,
                question=question,
                outcome=entry.outcome.value,
                side=entry.side.value,
                price=entry.price,
                size_usd=entry.size_usd,
                ai_confidence=entry.ai_confidence,
                ai_probability=entry.ai_probability,
                market_probability=entry.market_probability,
                edge=entry.edge,
                reasoning=entry.reasoning,
                pnl_usd=entry.pnl_usd,
                paper_trade=paper_trade,
            )
            session.add(record)
            await session.commit()
    except Exception as e:
        log.error("trade_log_db_write_failed", error=str(e))


async def log_probability_estimate(estimate: ProbabilityEstimate, traded: bool = False):
    try:
        async with get_session() as session:
            record = ProbabilityEstimateRecord(
                market_id=estimate.market_id,
                generated_at=estimate.generated_at,
                ai_probability_yes=estimate.ai_probability_yes,
                market_probability_yes=estimate.market_probability_yes,
                edge=estimate.edge,
                confidence=estimate.confidence,
                reasoning=estimate.reasoning,
                traded=traded,
            )
            session.add(record)
            await session.commit()
    except Exception as e:
        log.error("estimate_log_db_write_failed", error=str(e))


async def upsert_position(position: Position):
    try:
        async with get_session() as session:
            record = PositionRecord(
                position_id=position.position_id,
                market_id=position.market_id,
                outcome=position.outcome.value,
                entry_price=position.entry_price,
                size_shares=position.size_shares,
                size_usd=position.size_usd,
                category=position.category.value,
                opened_at=position.opened_at,
                closed=position.closed,
                closed_at=position.closed_at,
                exit_price=position.exit_price,
                realized_pnl_usd=position.realized_pnl_usd,
                close_reason=position.close_reason,
            )
            await session.merge(record)
            await session.commit()
    except Exception as e:
        log.error("position_upsert_failed", error=str(e))
