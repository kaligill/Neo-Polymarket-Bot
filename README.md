# Neo Prediction Engine

An AI-powered quantitative research and trading system for Polymarket. It scans
active markets, estimates a "fair" probability for each using an LLM (informed
by news + social sentiment) plus an optional trained quant model, flags
mispriced markets, sizes positions with Kelly Criterion or fixed-risk sizing,
and manages risk/exits automatically — all in **paper trading mode by default**.

## ⚠️ Read before enabling live trading

`PAPER_TRADING=True` is the default and the only mode that's actually
implemented. Live execution (`strategy/execution.py::LiveExecutionClient`) is
a deliberate stub that raises `NotImplementedError`. This is not an oversight —
wiring a bot up to sign and submit real on-chain orders is the highest-stakes
part of this system, and it deserves its own careful implementation, testing
against Polymarket's `py-clob-client`, and a long paper-trading track record
before any real capital touches it. This is not financial advice, and past
backtest performance does not guarantee future results.

## Architecture

```
neo_polymarket/
├── main.py                  # FastAPI app + async trading loop orchestration
├── config.py                # All tunables (thresholds, risk limits, API keys)
├── models.py                 # Shared pydantic models (Market, Position, etc.)
├── logger.py                  # Structured JSON logging
│
├── data/                     # Read-only data ingestion, no trading side effects
│   ├── market_fetcher.py      # Polymarket Gamma API — active markets
│   ├── orderbook.py            # Polymarket CLOB API — live order books
│   ├── historical.py            # Historical price series for backtesting
│   ├── news_fetcher.py           # RSS (Reuters/AP/BBC/CNBC/Bloomberg) + NewsAPI
│   ├── twitter_fetcher.py         # X/Twitter recent search API
│   └── reddit_fetcher.py           # Reddit search via PRAW
│
├── ai/                       # Probability estimation
│   ├── probability_engine.py  # LLM + optional quant model blend -> ProbabilityEstimate
│   ├── sentiment.py            # Social post scoring (LLM or lexicon fallback)
│   ├── reasoning.py             # Provider-agnostic LLM client (Anthropic/OpenAI)
│   └── confidence_score.py       # Combines signal quality into one confidence score
│
├── strategy/                 # Decision + execution
│   ├── edge_detector.py       # Gates: edge, confidence, liquidity, spread, volume
│   ├── kelly.py                 # Kelly Criterion + fixed-risk position sizing
│   ├── portfolio.py               # Risk manager: exposure, diversification, daily loss
│   ├── stoploss.py                 # Profit target / stop loss / trailing stop / time exit
│   └── execution.py                 # Paper execution (simulated fills) + live stub
│
├── backtest/
│   ├── engine.py               # Replays historical series through the same strategy code
│   └── train_model.py           # Trains the optional XGBoost quant prior
│
├── database/
│   ├── db.py                   # SQLAlchemy async models (trades, positions, estimates)
│   └── trade_log.py              # Persists every trade + JSONL audit trail
│
├── dashboard/
│   └── app.py                   # Streamlit live dashboard (reads from the FastAPI API)
│
└── tests/
    └── test_strategy.py          # Unit tests for sizing/risk/exit logic (no network needed)
```

## How a trade decision gets made

1. `data/market_fetcher.py` pulls active markets from Polymarket.
2. For each market, `data/news_fetcher.py` and `data/twitter_fetcher.py` /
   `data/reddit_fetcher.py` gather relevant context.
3. `ai/probability_engine.py` asks the LLM (Claude/GPT) to independently
   estimate the true probability of YES, given that context — without
   anchoring on the current market price — and blends it with an optional
   quant model.
4. `ai/confidence_score.py` combines the LLM's self-reported confidence with
   signal agreement, news coverage, fake-news risk, and liquidity into one
   number.
5. `strategy/edge_detector.py` only produces a `TradeSignal` if edge,
   confidence, liquidity, spread, and volume all clear their thresholds
   (`config.py`).
6. `strategy/kelly.py` sizes the position (fractional Kelly by default).
7. `strategy/portfolio.py` checks the trade against exposure, per-category
   diversification, max open positions, and daily loss limits before
   allowing it.
8. `strategy/execution.py` fills the order (simulated in paper mode).
9. `strategy/stoploss.py` manages exits every cycle: profit target, stop
   loss, trailing stop, or max hold time.
10. Every step is logged to Postgres + `logs/trades.jsonl` via
    `database/trade_log.py`.

## Setup

```bash
cp .env.example .env        # fill in your API keys
pip install -r requirements.txt

# Local Postgres/Redis via Docker
docker compose up postgres redis -d

# Run the trading engine (paper mode)
uvicorn main:app --reload --port 8000

# Run the dashboard (separate terminal)
streamlit run dashboard/app.py
```

Or run everything containerized:

```bash
docker compose up --build
```

- API: http://localhost:8000 (`/health`, `/portfolio`, `/positions`)
- Dashboard: http://localhost:8501

## Required API keys

| Key | Needed for | Required? |
|---|---|---|
| `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` | Probability estimation, sentiment | Yes |
| `NEWSAPI_KEY` | Keyword-targeted news search (RSS works without it) | Optional |
| `TWITTER_BEARER_TOKEN` | Social sentiment from X | Optional |
| `REDDIT_CLIENT_ID` / `REDDIT_CLIENT_SECRET` | Social sentiment from Reddit | Optional |
| `POLYMARKET_API_*` / wallet key | Live execution only (not implemented) | No (paper mode ignores these) |

The system degrades gracefully without the optional keys — it just loses that
signal source rather than crashing (e.g. no Twitter token means Twitter
sentiment is simply skipped).

## Backtesting

```bash
python -m backtest.train_model --data logs/resolved_markets.csv --out models/quant_model.pkl
```

Then use `backtest/engine.py::run_backtest()` with a historical price series
(`data/historical.py::fetch_price_history`) and either the trained model's
predictions or your own logged AI probability estimates
(`probability_estimates` table) to measure ROI, drawdown, win rate, and
average win/loss before ever risking capital.

## Tuning risk

All thresholds live in `config.py` / `.env`:
- `MIN_EDGE_THRESHOLD`, `MIN_CONFIDENCE_THRESHOLD` — opportunity gates
- `RISK_MODEL`, `KELLY_FRACTION`, `FIXED_RISK_PCT_PER_TRADE` — sizing
- `MAX_EXPOSURE_PCT_OF_BANKROLL`, `MAX_DAILY_LOSS_PCT`, `MAX_OPEN_POSITIONS`,
  `MAX_POSITIONS_PER_CATEGORY` — portfolio-level risk caps
- `PROFIT_TARGET_PCT`, `STOP_LOSS_PCT`, `TRAILING_STOP_PCT`, `MAX_HOLD_DAYS` — exits

Start conservative (small `KELLY_FRACTION`, high `MIN_CONFIDENCE_THRESHOLD`)
and loosen only after a solid paper-trading track record.

## Extending (matches the "Future Upgrades" roadmap)

- **Multi-agent collaboration**: run several LLM "analyst" prompts with
  different personas (contrarian, momentum, fundamentals) in
  `ai/probability_engine.py` and ensemble their outputs before blending with
  the quant model.
- **Election / sports / weather specialist models**: add category-specific
  prompts or dedicated quant features (e.g. polling averages, ELO ratings,
  forecast model outputs) keyed off `MarketCategory`.
- **Discord/Telegram alerts**: add a notifier in a new `alerts/` package that
  subscribes to the same events `database/trade_log.py` logs.
- **Self-learning loop**: once markets resolve, join `positions` against
  resolution outcomes to build the labeled dataset `backtest/train_model.py`
  expects, and retrain periodically.
