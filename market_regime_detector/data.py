from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {"Open", "High", "Low", "Close", "Volume"}


@dataclass(frozen=True)
class MarketDataRequest:
    ticker: str = "SPY"
    period: str = "5y"
    interval: str = "1d"
    synthetic: bool = False
    seed: int = 42


def _normalize_ohlcv_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a clean OHLCV dataframe with a DatetimeIndex.

    yfinance may return MultiIndex columns for some calls. This function
    makes downloaded and generated data consistent.
    """
    if isinstance(df.columns, pd.MultiIndex):
        # yfinance sometimes returns columns like (Price, Ticker). Keep level 0.
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

    rename_map = {}
    for col in df.columns:
        cleaned = str(col).strip().lower().replace("adj close", "adj_close")
        if cleaned in {"open", "high", "low", "close", "volume"}:
            rename_map[col] = cleaned.title()
        elif cleaned == "adj_close":
            rename_map[col] = "Adj Close"
    df = df.rename(columns=rename_map)

    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except Exception as exc:  # pragma: no cover - defensive
            raise ValueError("Data must have a datetime-like index or Date column.") from exc

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required OHLCV columns: {sorted(missing)}")

    df = df.sort_index()
    df = df.loc[:, ["Open", "High", "Low", "Close", "Volume"]].copy()
    df = df.apply(pd.to_numeric, errors="coerce").dropna()
    if df.empty:
        raise ValueError("No usable OHLCV data after cleaning.")
    return df


def load_market_data(request: MarketDataRequest) -> pd.DataFrame:
    """Load OHLCV data from yfinance or the synthetic demo generator."""
    if request.synthetic:
        return generate_synthetic_ohlcv(seed=request.seed)

    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError(
            "yfinance is required for live Yahoo Finance downloads. "
            "Install it with `pip install yfinance`, or use --synthetic."
        ) from exc

    df = yf.download(
        request.ticker,
        period=request.period,
        interval=request.interval,
        auto_adjust=False,
        progress=False,
    )
    if df is None or df.empty:
        raise ValueError(f"No data returned for ticker={request.ticker!r}.")
    return _normalize_ohlcv_columns(df)


def generate_synthetic_ohlcv(seed: int = 42, periods: int = 900) -> pd.DataFrame:
    """Create synthetic OHLCV data with changing market regimes.

    This lets the project run without internet access. The generator simulates
    bull-trend, high-volatility, sideways, and bear-trend sections so the rest
    of the pipeline can be tested end to end.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=periods)

    regimes = np.repeat([0, 1, 2, 3, 0, 2], repeats=periods // 6)
    if len(regimes) < periods:
        regimes = np.pad(regimes, (0, periods - len(regimes)), mode="edge")

    drift = np.select(
        [regimes == 0, regimes == 1, regimes == 2, regimes == 3],
        [0.00075, 0.00005, 0.00005, -0.00065],
    )
    vol = np.select(
        [regimes == 0, regimes == 1, regimes == 2, regimes == 3],
        [0.009, 0.026, 0.006, 0.014],
    )
    shocks = rng.normal(drift, vol)

    close = 100 * np.exp(np.cumsum(shocks))
    open_ = close * (1 + rng.normal(0, 0.002, periods))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0.003, 0.003, periods)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0.003, 0.003, periods)))
    volume = rng.lognormal(mean=13.5, sigma=0.25, size=periods)
    volume = volume * np.where(regimes == 1, 1.8, 1.0)

    return pd.DataFrame(
        {
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume.astype(int),
        },
        index=dates,
    )
