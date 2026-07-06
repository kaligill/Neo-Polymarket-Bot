"""
Neo Prediction Engine — main entrypoint.

Runs a FastAPI app (for health checks / manual control / dashboard data) and
launches the async trading loop as a background task on startup.

Run with:  uvicorn main:app --reload --port 8000
Or:        python main.py   (runs uvicorn programmatically)
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI

from ai.probability_engine import ProbabilityEngine
from ai.sentiment import SentimentAnalyzer, combine_signals
from config import settings
from data.market_fetcher import MarketFetcher
from data.news_fetcher import NewsFetcher, keywords_for_market
from data.orderbook import OrderBookClient
from data.reddit_fetcher import RedditFetcher
from data.twitter_fetcher import TwitterFetcher
from database.db import init_db
from database.trade_log import log_probability_estimate, log_trade
from logger import get_logger
from models import Outcome, OrderSide, TradeLogEntry
from strategy.edge_detector import EdgeDetector
from strategy.execution import execute_signal, get_execution_client
from strategy.kelly import compute_position_size
from strategy.portfolio import PortfolioManager, RiskLimitBreached
from strategy.stoploss import evaluate_positions

log = get_logger(__name__)

# ---- shared singletons (constructed once, reused across the trading loop) ----
market_fetcher = MarketFetcher()
orderbook_client = OrderBookClient()
news_fetcher = NewsFetcher()
twitter_fetcher = TwitterFetcher()
reddit_fetcher = RedditFetcher()
probability_engine = ProbabilityEngine()
sentiment_analyzer = SentimentAnalyzer()
edge_detector = EdgeDetector()
portfolio = PortfolioManager()
execution_client = get_execution_client()

_trading_task: asyncio.Task | None = None
_shutdown = asyncio.Event()


async def process_market(market, news_cache: list):
    """Full pipeline for one market: gather context -> estimate -> gate -> size -> execute."""
    try:
        keywords = keywords_for_market(market.question)
        relevant_news = [
            n for n in news_cache
            if any(kw.lower() in (n.title or "").lower() for kw in keywords)
        ][:10]

        social_signals = []
        query = " ".join(keywords[:3])
        tweets = await twitter_fetcher.fetch_recent(query)
        if tweets:
            social_signals.append(await sentiment_analyzer.score_posts("twitter", query, tweets))
        reddit_posts = await reddit_fetcher.search(query)
        if reddit_posts:
            social_signals.append(await sentiment_analyzer.score_posts("reddit", query, reddit_posts))
        combined_social = combine_signals(social_signals)

        estimate = await probability_engine.estimate(market, relevant_news, combined_social)
        await log_probability_estimate(estimate)

        # Fetch order book only for markets that already clear the edge/confidence
        # bar, to avoid burning API calls on markets we won't trade anyway.
        order_book = None
        if abs(estimate.edge) >= settings.MIN_EDGE_THRESHOLD and estimate.confidence >= settings.MIN_CONFIDENCE_THRESHOLD:
            token_id = market.condition_id or market.market_id
            outcome = Outcome.YES if estimate.edge > 0 else Outcome.NO
            order_book = await orderbook_client.fetch_order_book(token_id, market.market_id, outcome)

        signal = edge_detector.evaluate(market, estimate, order_book)
        if signal is None:
            return

        true_prob = estimate.ai_probability_yes if signal.outcome == Outcome.YES else 1 - estimate.ai_probability_yes
        size_usd = compute_position_size(portfolio.total_equity_usd, true_prob, signal.limit_price)
        if size_usd <= 0:
            return

        result = await execute_signal(execution_client, signal, size_usd, order_book)
        if not result.filled:
            log.info("order_not_filled", market_id=market.market_id, message=result.message)
            return

        try:
            position = portfolio.open_position(signal, result.filled_size_usd, market.category, result.fill_price)
        except RiskLimitBreached as e:
            log.warning("risk_limit_blocked_trade", market_id=market.market_id, reason=str(e))
            return

        await log_trade(
            TradeLogEntry(
                market_id=market.market_id,
                outcome=signal.outcome,
                side=OrderSide.BUY,
                price=result.fill_price,
                size_usd=result.filled_size_usd,
                ai_confidence=estimate.confidence,
                ai_probability=estimate.ai_probability_yes,
                market_probability=estimate.market_probability_yes,
                edge=estimate.edge,
                reasoning=estimate.reasoning,
            ),
            question=market.question,
            paper_trade=settings.PAPER_TRADING,
        )

    except Exception as e:
        log.error("process_market_failed", market_id=market.market_id, error=str(e))


async def manage_open_positions(latest_markets: dict):
    """Checks all open positions against stop-loss/profit-target/trailing-stop/time exits.
    `latest_markets` maps market_id -> Market, populated from the current scan cycle."""
    current_prices = {}
    for position_id, position in portfolio.open_positions.items():
        market = latest_markets.get(position.market_id)
        if market is None:
            current_prices[position_id] = position.entry_price  # stale fallback; market left the active set
            continue
        current_prices[position_id] = market.yes_price if position.outcome == Outcome.YES else market.no_price

    to_close = evaluate_positions(portfolio.open_positions, current_prices)
    for position_id, reason in to_close:
        position = portfolio.open_positions.get(position_id)
        if position is None:
            continue
        exit_price = current_prices.get(position_id, position.entry_price)
        portfolio.close_position(position_id, exit_price, reason)


async def trading_loop():
    try:
        await asyncio.wait_for(init_db(), timeout=10.0)
    except asyncio.TimeoutError:
        log.warning("database_init_timeout")
    except Exception as e:
        log.warning("database_init_error", error=str(e))
    
    log.info("trading_loop_starting", paper_trading=settings.PAPER_TRADING)

    while not _shutdown.is_set():
        try:
            try:
                news_cache = await asyncio.wait_for(news_fetcher.fetch_rss_all(), timeout=30.0)
            except asyncio.TimeoutError:
                log.warning("news_fetch_timeout")
                news_cache = []
            except Exception as e:
                log.warning("news_fetch_error", error=str(e))
                news_cache = []
            
            try:
                markets = await asyncio.wait_for(market_fetcher.fetch_active_markets(), timeout=30.0)
            except asyncio.TimeoutError:
                log.warning("markets_fetch_timeout")
                markets = []
            except Exception as e:
                log.warning("markets_fetch_error", error=str(e))
                markets = []
            
            log.info("scan_cycle", markets_found=len(markets), news_items=len(news_cache))

            # Process markets concurrently but bounded, to avoid hammering the LLM/API rate limits.
            semaphore = asyncio.Semaphore(5)

            async def bounded(m):
                async with semaphore:
                    await process_market(m, news_cache)

            await asyncio.gather(*(bounded(m) for m in markets))
            latest_markets = {m.market_id: m for m in markets}
            await manage_open_positions(latest_markets)

            log.info("portfolio_summary", **portfolio.summary())

        except Exception as e:
            log.error("trading_loop_error", error=str(e))

        await asyncio.sleep(settings.MARKET_SCAN_INTERVAL_SECONDS)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _trading_task
    _trading_task = asyncio.create_task(trading_loop())
    yield
    _shutdown.set()
    if _trading_task:
        _trading_task.cancel()
    for client in (market_fetcher, orderbook_client, news_fetcher, twitter_fetcher):
        await client.close()


app = FastAPI(title="Neo Prediction Engine", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok", "paper_trading": settings.PAPER_TRADING}


@app.get("/portfolio")
async def get_portfolio():
    return portfolio.summary()


@app.get("/positions")
async def get_positions():
    return {
        "open": [p.model_dump() for p in portfolio.open_positions.values()],
        "closed": [p.model_dump() for p in portfolio.closed_positions[-50:]],
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
