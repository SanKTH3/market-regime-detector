from __future__ import annotations

import numpy as np
import pandas as pd


def build_strategy_returns(regime_df: pd.DataFrame, allow_short: bool = False) -> pd.DataFrame:
    """Create simple benchmark and regime-aware strategy returns.

    This is deliberately simple: the goal is to validate whether the detected
    regimes have useful information, not to claim a production trading edge.
    """
    df = regime_df.copy()
    df["asset_return"] = df["Close"].pct_change().fillna(0)

    # Signal is decided at close, used next bar to avoid look-ahead bias.
    signal = pd.Series(0.0, index=df.index)
    regime = df["regime"].astype(str)
    signal.loc[regime.str.contains("Bull", case=False)] = 1.0
    signal.loc[regime.str.contains("Sideways", case=False) & (df["bb_z_20"] < -0.75)] = 1.0
    signal.loc[regime.str.contains("Sideways", case=False) & (df["bb_z_20"] > 0.75)] = -1.0 if allow_short else 0.0
    signal.loc[regime.str.contains("Bear", case=False)] = -1.0 if allow_short else 0.0
    signal.loc[regime.str.contains("High Volatility", case=False)] = 0.0

    df["signal"] = signal.shift(1).fillna(0)
    df["strategy_return"] = df["signal"] * df["asset_return"]
    df["benchmark_equity"] = (1 + df["asset_return"]).cumprod()
    df["strategy_equity"] = (1 + df["strategy_return"]).cumprod()
    return df


def performance_metrics(backtest_df: pd.DataFrame, periods_per_year: int = 252) -> pd.DataFrame:
    rows = []
    for name, col in [("Buy & Hold", "asset_return"), ("Regime Strategy", "strategy_return")]:
        r = backtest_df[col].dropna()
        if r.empty:
            continue
        total_return = (1 + r).prod() - 1
        years = max(len(r) / periods_per_year, 1 / periods_per_year)
        cagr = (1 + total_return) ** (1 / years) - 1
        vol = r.std() * np.sqrt(periods_per_year)
        sharpe = np.nan if vol == 0 else (r.mean() * periods_per_year) / vol
        equity = (1 + r).cumprod()
        drawdown = equity / equity.cummax() - 1
        rows.append(
            {
                "strategy": name,
                "total_return": total_return,
                "cagr": cagr,
                "annualized_vol": vol,
                "sharpe_no_rf": sharpe,
                "max_drawdown": drawdown.min(),
            }
        )
    return pd.DataFrame(rows)
