#!/usr/bin/env python
"""Quick startup test"""
import sys
import asyncio
print("1. Python started", flush=True)

print("2. Importing config...", flush=True)
from config import settings
print(f"   - LLM_PROVIDER: {settings.LLM_PROVIDER}", flush=True)
print(f"   - DATABASE_URL: {settings.DATABASE_URL[:50]}...", flush=True)

print("3. Testing logger...", flush=True)
from logger import get_logger
log = get_logger(__name__)
log.info("logger_test")

print("4. Testing market fetcher...", flush=True)
from data.market_fetcher import MarketFetcher
mf = MarketFetcher(timeout=5.0)
print("   - MarketFetcher created", flush=True)

print("5. Testing probability engine...", flush=True)
from ai.probability_engine import ProbabilityEngine
pe = ProbabilityEngine()
print("   - ProbabilityEngine created", flush=True)

print("6. All imports successful!", flush=True)

async def test_api_call():
    print("7. Testing market fetch (5 second timeout)...", flush=True)
    try:
        markets = await asyncio.wait_for(mf.fetch_active_markets(limit=1), timeout=5.0)
        print(f"   - Fetched {len(markets)} markets", flush=True)
    except asyncio.TimeoutError:
        print("   - Timeout (expected if no internet)", flush=True)
    except Exception as e:
        print(f"   - Error: {e}", flush=True)

asyncio.run(test_api_call())
print("Done!", flush=True)
