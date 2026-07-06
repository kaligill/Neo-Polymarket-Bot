"""
Shared visual theme for the Streamlit dashboard: dark quant-terminal look,
neon accent colors, stat-card CSS, and a purely decorative neural-net style
plotly figure (no real computation — just visual flavor, clearly labeled as
decorative in the UI so nobody mistakes it for a live model diagram).
"""
from __future__ import annotations

import random

import plotly.graph_objects as go
import streamlit as st

GREEN = "#00e676"
RED = "#ff5252"
AMBER = "#ffb300"
BG = "#0b0e11"
CARD_BG = "#14181d"
TEXT_DIM = "#8a929b"


def inject_theme():
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-color: {BG};
        }}
        .neo-card {{
            background: {CARD_BG};
            border: 1px solid #232830;
            border-radius: 10px;
            padding: 14px 16px;
            margin-bottom: 8px;
        }}
        .neo-label {{
            color: {TEXT_DIM};
            font-size: 12px;
            letter-spacing: 0.5px;
            text-transform: uppercase;
            margin-bottom: 4px;
        }}
        .neo-value {{
            font-size: 26px;
            font-weight: 700;
            font-family: 'Courier New', monospace;
        }}
        .neo-sub {{
            font-size: 12px;
            color: {TEXT_DIM};
            margin-top: 2px;
        }}
        .neo-pill {{
            display: inline-block;
            padding: 3px 10px;
            border-radius: 20px;
            font-size: 11px;
            font-weight: 700;
            letter-spacing: 0.5px;
        }}
        .neo-pill-live {{
            background: rgba(0, 230, 118, 0.15);
            color: {GREEN};
            border: 1px solid {GREEN};
        }}
        .neo-pill-paper {{
            background: rgba(255, 179, 0, 0.15);
            color: {AMBER};
            border: 1px solid {AMBER};
        }}
        .neo-cycle-step {{
            text-align: center;
            padding: 8px 4px;
            border-radius: 8px;
            background: {CARD_BG};
            border: 1px solid #232830;
            font-size: 12px;
            color: {TEXT_DIM};
        }}
        .neo-cycle-step.active {{
            border: 1px solid {GREEN};
            color: {GREEN};
            box-shadow: 0 0 12px rgba(0,230,118,0.25);
        }}
        div[data-testid="stMetricValue"] {{
            font-family: 'Courier New', monospace;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def stat_card(label: str, value: str, sub: str = "", color: str = "#ffffff"):
    st.markdown(
        f"""
        <div class="neo-card">
            <div class="neo-label">{label}</div>
            <div class="neo-value" style="color:{color};">{value}</div>
            <div class="neo-sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def mode_pill(paper_trading: bool) -> str:
    if paper_trading:
        return '<span class="neo-pill neo-pill-paper">● PAPER TRADING</span>'
    return '<span class="neo-pill neo-pill-live">● LIVE</span>'


def execution_cycle(active_step: int = 0):
    """active_step: index 0-5 of which stage is currently highlighted (decorative)."""
    steps = ["Scan", "Detect", "Validate", "Size", "Fill", "Settle"]
    cols = st.columns(len(steps))
    for i, (col, step) in enumerate(zip(cols, steps)):
        cls = "neo-cycle-step active" if i == active_step else "neo-cycle-step"
        col.markdown(f'<div class="{cls}">{step}</div>', unsafe_allow_html=True)


def decorative_network_figure(seed: int = 42) -> go.Figure:
    """
    Purely cosmetic 'agent swarm' style layered-network visualization.
    THIS IS DECORATIVE ONLY — it does not represent a real model architecture
    or live computation. Labeled as such in the dashboard caption.
    """
    rng = random.Random(seed)
    layer_sizes = [6, 10, 10, 8, 4]
    layer_x = [0, 1, 2, 3, 4]

    node_x, node_y, node_color = [], [], []
    layer_nodes = []
    for li, size in enumerate(layer_sizes):
        ys = [ (i - (size - 1) / 2) for i in range(size) ]
        layer_nodes.append(list(zip([layer_x[li]] * size, ys)))
        for y in ys:
            node_x.append(layer_x[li])
            node_y.append(y)
            node_color.append(rng.choice([GREEN, RED, "#4fc3f7", AMBER]))

    edge_x, edge_y = [], []
    for li in range(len(layer_nodes) - 1):
        for (x0, y0) in layer_nodes[li]:
            sample = rng.sample(layer_nodes[li + 1], k=min(3, len(layer_nodes[li + 1])))
            for (x1, y1) in sample:
                edge_x += [x0, x1, None]
                edge_y += [y0, y1, None]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y, mode="lines",
        line=dict(color="rgba(0,230,118,0.15)", width=1),
        hoverinfo="skip", showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y, mode="markers",
        marker=dict(size=10, color=node_color, line=dict(color="#0b0e11", width=1)),
        hoverinfo="skip", showlegend=False,
    ))
    fig.update_layout(
        plot_bgcolor=BG, paper_bgcolor=BG,
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        margin=dict(l=0, r=0, t=0, b=0), height=220,
    )
    return fig
