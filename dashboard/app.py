"""
Neo Prediction Engine — quant-terminal styled dashboard.

Run with:  streamlit run dashboard/app.py
Requires:  the main.py FastAPI server running (default http://localhost:8000)

Visual design note: the "agent network" panel at the bottom is DECORATIVE
ONLY (dashboard/theme.py::decorative_network_figure) — it does not represent
real model internals or live computation. It exists purely for the terminal
aesthetic. Everything else on this page (PnL, positions, win rate, etc.) is
real data pulled live from the FastAPI backend.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone

import httpx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.theme import (
    GREEN, RED, TEXT_DIM,
    decorative_network_figure, execution_cycle, inject_theme, mode_pill, stat_card,
)

API_BASE = os.environ.get("NEO_API_BASE", "http://localhost:8000")

st.set_page_config(page_title="Neo Prediction Engine", layout="wide", initial_sidebar_state="collapsed")
inject_theme()


@st.cache_data(ttl=5)
def fetch_json(path: str) -> dict:
    try:
        resp = httpx.get(f"{API_BASE}{path}", timeout=5.0)
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPError:
        return {}


health = fetch_json("/health")
portfolio = fetch_json("/portfolio")
positions = fetch_json("/positions")

# ---------------- Header ----------------
h1, h2 = st.columns([3, 1])
with h1:
    st.markdown(
        f"""
        <div style="display:flex; align-items:center; gap:14px;">
            <span style="font-size:28px; font-weight:800; color:white;">◆ NEO</span>
            <span style="font-size:14px; color:{TEXT_DIM}; letter-spacing:1px;">QUANT · KELLY · SELF-LEARN</span>
            {mode_pill(health.get("paper_trading", True))}
        </div>
        """,
        unsafe_allow_html=True,
    )
with h2:
    st.markdown(
        f"<div style='text-align:right; color:{TEXT_DIM}; font-family:monospace;'>"
        f"{datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}</div>",
        unsafe_allow_html=True,
    )

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

# ---------------- Top stat row ----------------
realized_pnl = portfolio.get("realized_pnl_usd", 0.0)
pnl_color = GREEN if realized_pnl >= 0 else RED
win_rate = portfolio.get("win_rate", 0.0)
daily_pnl = portfolio.get("daily_pnl_usd", 0.0)
daily_color = GREEN if daily_pnl >= 0 else RED

c1, c2, c3, c4, c5, c6 = st.columns(6)
with c1:
    stat_card("Total Equity", f"${portfolio.get('total_equity_usd', 0):,.0f}", "bankroll + open exposure")
with c2:
    stat_card("Realized PnL", f"${realized_pnl:,.0f}", f"{portfolio.get('total_closed_trades', 0)} closed trades", pnl_color)
with c3:
    stat_card("Win Rate", f"{win_rate:.1%}", "", GREEN if win_rate >= 0.5 else RED)
with c4:
    stat_card("Open Positions", str(portfolio.get("open_positions", 0)), "active right now")
with c5:
    stat_card("Exposure", f"${portfolio.get('total_exposure_usd', 0):,.0f}", "capital deployed")
with c6:
    stat_card("Daily PnL", f"${daily_pnl:,.0f}", "resets at UTC midnight", daily_color)

st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

# ---------------- Execution cycle indicator (decorative pulse over real stages) ----------------
st.markdown(f"<div class='neo-label'>EXECUTION CYCLE</div>", unsafe_allow_html=True)
cycle_step = int(datetime.now(timezone.utc).timestamp() // 3) % 6  # cosmetic rotation
execution_cycle(cycle_step)

st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

# ---------------- Positions + PnL curve ----------------
left, right = st.columns([1.3, 1])

with left:
    st.markdown("<div class='neo-label'>ACTIVE POSITIONS</div>", unsafe_allow_html=True)
    open_positions = positions.get("open", [])
    if open_positions:
        df_open = pd.DataFrame(open_positions)
        df_open = df_open[["market_id", "outcome", "entry_price", "size_usd", "opened_at", "category"]]
        st.dataframe(df_open, use_container_width=True, height=260, hide_index=True)
    else:
        st.markdown(
            f"<div class='neo-card' style='color:{TEXT_DIM}; text-align:center; padding:40px;'>"
            "No open positions right now.</div>",
            unsafe_allow_html=True,
        )

with right:
    st.markdown("<div class='neo-label'>CUMULATIVE REALIZED PnL</div>", unsafe_allow_html=True)
    closed_positions = positions.get("closed", [])
    if closed_positions:
        df_closed = pd.DataFrame(closed_positions)
        df_closed["closed_at"] = pd.to_datetime(df_closed["closed_at"])
        df_closed = df_closed.sort_values("closed_at")
        df_closed["cumulative_pnl"] = df_closed["realized_pnl_usd"].cumsum()

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_closed["closed_at"], y=df_closed["cumulative_pnl"],
            mode="lines", name="Cumulative PnL",
            line=dict(color=GREEN, width=2),
            fill="tozeroy", fillcolor="rgba(0,230,118,0.08)",
        ))
        fig.update_layout(
            plot_bgcolor="#0b0e11", paper_bgcolor="#0b0e11",
            font=dict(color=TEXT_DIM, family="Courier New"),
            xaxis=dict(gridcolor="#1c2128"), yaxis=dict(gridcolor="#1c2128", title="USD"),
            margin=dict(l=10, r=10, t=10, b=10), height=260,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.markdown(
            f"<div class='neo-card' style='color:{TEXT_DIM}; text-align:center; padding:40px;'>"
            "No closed trades yet.</div>",
            unsafe_allow_html=True,
        )

st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

# ---------------- Closed trades table ----------------
st.markdown("<div class='neo-label'>RECENT CLOSED TRADES</div>", unsafe_allow_html=True)
if closed_positions:
    df_closed_view = pd.DataFrame(closed_positions)[
        ["market_id", "outcome", "entry_price", "exit_price", "realized_pnl_usd", "close_reason"]
    ].sort_values("realized_pnl_usd", ascending=False)
    st.dataframe(df_closed_view, use_container_width=True, hide_index=True)
else:
    st.caption("Nothing to show yet.")

st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

# ---------------- Decorative agent-network panel ----------------
with st.expander("AGENT SWARM — network visualization (decorative)", expanded=False):
    st.plotly_chart(decorative_network_figure(), use_container_width=True)
    st.caption(
        "Cosmetic visualization only — not a live model diagram or real computation. "
        "Actual probability estimation happens in ai/probability_engine.py via LLM calls."
    )

st.divider()
st.caption(
    f"Connected to API at `{API_BASE}` · Set `NEO_API_BASE` to point elsewhere · "
    f"Refreshes every ~5s · Mode: {'Paper Trading' if health.get('paper_trading', True) else 'LIVE'}"
)
