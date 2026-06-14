from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_FEATURE_COLUMNS = [
    "return_1d",
    "log_return_1d",
    "rolling_vol_20",
    "rolling_vol_60",
    "momentum_10",
    "momentum_20",
    "ma_slope_20",
    "ma_slope_50",
    "rsi_14",
    "atr_pct_14",
    "autocorr_20",
    "bb_z_20",
    "volume_z_20",
]


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()


def _rolling_autocorr(series: pd.Series, window: int = 20, lag: int = 1) -> pd.Series:
    return series.rolling(window).corr(series.shift(lag))


def _zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    return (series - mean) / std.replace(0, np.nan)


def build_features(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Add market-regime features to an OHLCV dataframe.

    The goal is to convert raw prices into behavior descriptors that can be
    used by a regime model. These features describe trend, volatility,
    mean reversion, and unusual trading volume.
    """
    df = ohlcv.copy()
    close = df["Close"]

    df["return_1d"] = close.pct_change()
    df["log_return_1d"] = np.log(close).diff()

    # Volatility: annualized standard deviation of daily returns.
    df["rolling_vol_20"] = df["log_return_1d"].rolling(20).std() * np.sqrt(252)
    df["rolling_vol_60"] = df["log_return_1d"].rolling(60).std() * np.sqrt(252)

    # Momentum and trend.
    df["momentum_10"] = close.pct_change(10)
    df["momentum_20"] = close.pct_change(20)
    ma20 = close.rolling(20).mean()
    ma50 = close.rolling(50).mean()
    df["ma_slope_20"] = ma20.pct_change(5)
    df["ma_slope_50"] = ma50.pct_change(10)

    # Mean reversion and range/volatility features.
    df["rsi_14"] = _rsi(close, 14)
    df["atr_pct_14"] = _atr(df, 14) / close
    df["autocorr_20"] = _rolling_autocorr(df["return_1d"], 20, 1)
    df["bb_z_20"] = (close - ma20) / close.rolling(20).std().replace(0, np.nan)
    df["volume_z_20"] = _zscore(df["Volume"].astype(float), 20)

    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    if df.empty:
        raise ValueError("Not enough rows to compute features. Try more history or a longer period.")
    return df


def select_feature_matrix(feature_df: pd.DataFrame, columns: Iterable[str] | None = None) -> pd.DataFrame:
    cols = list(columns or DEFAULT_FEATURE_COLUMNS)
    missing = [c for c in cols if c not in feature_df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    return feature_df.loc[:, cols].copy()
