from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


REGIME_COLORS = {
    "Bull Trend": "rgba(0, 180, 0, 0.14)",
    "Bear Trend": "rgba(220, 0, 0, 0.14)",
    "High Volatility / Risk-Off": "rgba(255, 165, 0, 0.18)",
    "Sideways / Mean Reversion": "rgba(0, 120, 255, 0.12)",
}


def _base_regime_name(name: str) -> str:
    # Remove duplicate suffixes like " #2" for color lookup.
    return str(name).split(" #")[0]


def make_regime_chart(df: pd.DataFrame, ticker: str = "Market") -> go.Figure:
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.58, 0.20, 0.22],
        subplot_titles=(f"{ticker} price with detected regimes", "Rolling volatility", "Strategy vs benchmark"),
    )

    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Close", mode="lines"), row=1, col=1)

    if "rolling_vol_20" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["rolling_vol_20"], name="20D vol", mode="lines"), row=2, col=1)

    if "benchmark_equity" in df.columns and "strategy_equity" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["benchmark_equity"], name="Buy & Hold", mode="lines"), row=3, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["strategy_equity"], name="Regime Strategy", mode="lines"), row=3, col=1)

    # Draw regime backgrounds as contiguous rectangles.
    # We build the shapes list directly instead of calling fig.add_vrect in a loop,
    # which is much faster when a rule-based model changes regimes often.
    regimes = df["regime"].astype(str)
    starts = regimes.ne(regimes.shift()).cumsum()
    shapes = []
    annotations = []
    min_label_block = max(20, int(len(df) * 0.04))
    for _, block in df.groupby(starts):
        regime = block["regime"].iloc[0]
        base = _base_regime_name(regime)
        color = REGIME_COLORS.get(base, "rgba(150, 150, 150, 0.10)")
        shapes.append(
            dict(
                type="rect",
                xref="x",
                yref="paper",
                x0=block.index[0],
                x1=block.index[-1],
                y0=0,
                y1=1,
                fillcolor=color,
                opacity=1,
                layer="below",
                line_width=0,
            )
        )
        if len(block) >= min_label_block and len(annotations) < 12:
            annotations.append(
                dict(
                    x=block.index[0],
                    y=1.02,
                    xref="x",
                    yref="paper",
                    text=regime,
                    showarrow=False,
                    align="left",
                    font=dict(size=10),
                )
            )

    fig.update_layout(
        height=850,
        hovermode="x unified",
        legend_orientation="h",
        margin=dict(l=40, r=20, t=70, b=40),
        shapes=shapes,
        annotations=list(fig.layout.annotations) + annotations,
    )
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Vol", row=2, col=1)
    fig.update_yaxes(title_text="Equity", row=3, col=1)
    return fig


def save_chart_html(fig: go.Figure, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(path)
    return path
