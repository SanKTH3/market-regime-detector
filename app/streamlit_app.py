from __future__ import annotations

import sys
from pathlib import Path

# Allow `streamlit run app/streamlit_app.py` from the project root.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from market_regime_detector.backtest import build_strategy_returns, performance_metrics
from market_regime_detector.data import MarketDataRequest, load_market_data
from market_regime_detector.features import build_features
from market_regime_detector.regimes import RegimeDetector, available_methods
from market_regime_detector.visualization import make_regime_chart


st.set_page_config(page_title="Market Regime Detector", layout="wide")
st.title("Market Regime Detector")
st.caption("Analyze market conditions using OHLCV data, engineered features, regime detection, and a validation backtest.")


def section_title(text: str) -> None:
    """Render a standard Streamlit section heading."""
    st.subheader(text)


METHOD_HELP_TEXT = """
- **rule**: Beginner-friendly. Uses transparent if/else thresholds.

- **kmeans**: First machine-learning baseline. Groups similar market days into hard clusters.

- **gmm**: Recommended default. Uses probabilistic clustering, which works better when regimes overlap.

- **hmm**: Advanced option. Models hidden states and regime transitions over time. Uses Hidden Markov Models through the hmmlearn library..
"""

with st.sidebar:
    st.header("Configuration")
    source = st.radio("Data source", ["Yahoo Finance", "Synthetic demo"], index=0)
    ticker = st.text_input("Ticker", value="SPY")
    period = st.selectbox("Period", ["1y", "2y", "5y", "10y", "max"], index=2)
    interval = st.selectbox("Interval", ["1d", "1h", "1wk", "1mo"], index=0)
    method = st.selectbox(
        "Regime method",
        list(available_methods().keys()),
        index=2,
        help=METHOD_HELP_TEXT,
    )
    n_regimes = st.slider("Number of regimes", 2, 6, 4)
    allow_short = st.checkbox("Allow short positions in validation backtest", value=False)
    run = st.button("Run detector", type="primary")

    st.sidebar.markdown("---")
    st.sidebar.caption(
    "Built by San Kaneskan · Portfolio project"
)

st.markdown(
    """
This dashboard classifies market conditions into regimes such as **Bull Trend**,
**Bear Trend**, **High Volatility / Risk-Off**, and **Sideways / Mean Reversion**.

The project is educational and research-oriented. It is not financial advice and
should not be treated as a production trading system.
"""
)

with st.expander("How the detector works", expanded=False):
    st.markdown(
        """
1. **Load OHLCV data**: open, high, low, close, and volume.
2. **Create features**: returns, volatility, momentum, RSI, ATR, autocorrelation, Bollinger z-score, and volume z-score.
3. **Detect regimes**: apply the selected method to assign each date to a market state.
4. **Validate**: compare a simple regime-aware strategy against buy-and-hold.

The validation strategy uses the previous row's signal on the next row's return.
That design helps avoid look-ahead bias, which happens when a backtest uses information
that would not have been available at the time.
"""
    )

if run:
    try:
        synthetic = source == "Synthetic demo"

        with st.spinner("Loading data, building features, and detecting regimes..."):
            ohlcv = load_market_data(
                MarketDataRequest(
                    ticker=ticker,
                    period=period,
                    interval=interval,
                    synthetic=synthetic,
                )
            )
            feature_df = build_features(ohlcv)
            result = RegimeDetector(method=method, n_regimes=n_regimes).fit_predict(feature_df)
            backtest_df = build_strategy_returns(result.data, allow_short=allow_short)
            metrics = performance_metrics(backtest_df)

        section_title("Regime chart")
        chart_title = ticker if not synthetic else "Synthetic Demo"
        st.plotly_chart(make_regime_chart(backtest_df, ticker=chart_title), use_container_width=True)
        st.markdown(
            """
            **Color guide:**  
            **Bull trend** — Green market is generally moving upward.  
            **Bear trend** — Red market is generally moving downward.  
            **High volatility** — Yellow market is moving sharply or unpredictably.  
            **Sideways / mean-reverting** — Blue market has no strong direction and may be moving around an average.
            """
        )

        c1, c2 = st.columns(2)
        with c1:
            section_title("Regime profiles")
            st.dataframe(result.profiles.style.format(precision=4), use_container_width=True)
        with c2:
            section_title("Backtest metrics")
            st.dataframe(metrics.style.format(precision=4), use_container_width=True)

        section_title("Latest rows")
        st.dataframe(backtest_df.tail(25), use_container_width=True)

        section_title("Column guide")
        st.markdown(
            """
**Regime profiles**

- Each row summarizes the average behavior of one detected cluster.
- `count` shows how many rows were assigned to that cluster.
- `avg_forward_5d_return` shows the average five-row forward return after dates in that cluster.

**Backtest metrics**

- `total_return`: total percentage gain or loss over the test window.
- `cagr`: annualized growth rate.
- `annualized_vol`: annualized volatility of returns.
- `sharpe_no_rf`: return divided by risk, without subtracting a risk-free rate.
- `max_drawdown`: worst peak-to-trough decline during the test.

**Latest rows**

- `return_1d`: one-period percentage change in the close price.
- `log_return_1d`: logarithmic version of the one-period return; useful for financial modeling.
- `rolling_vol_20` / `rolling_vol_60`: annualized volatility over the last 20 or 60 rows.
- `momentum_10` / `momentum_20`: percentage price change over the last 10 or 20 rows.
- `ma_slope_20` / `ma_slope_50`: recent slope of the 20-row or 50-row moving average.
- `rsi_14`: Relative Strength Index; higher values suggest stronger recent upward movement.
- `atr_pct_14`: Average True Range divided by price; a range-based volatility measure.
- `autocorr_20`: checks whether recent returns tend to continue or reverse.
- `bb_z_20`: distance from the 20-row moving average measured in standard-deviation units.
- `volume_z_20`: how unusual the latest volume is compared with recent volume.
- `cluster_id`: numeric group assigned by the selected regime model.
- `regime`: readable market-state label created from the cluster profile.
- `asset_return`: raw return of the asset for the row.
- `signal`: position used by the validation strategy on that row: `1` long, `0` flat, `-1` short.
- `strategy_return`: asset return multiplied by the strategy signal.
- `benchmark_equity`: growth of a simple buy-and-hold benchmark.
- `strategy_equity`: growth of the regime-aware validation strategy.


"""
        )

    except Exception as exc:
        st.error(str(exc))
        st.info("Try Synthetic demo first. For Yahoo Finance, make sure yfinance is installed and you have internet access.")
else:
    st.info("Choose options in the sidebar and click **Run detector**. Start with Synthetic demo + GMM if you want a no-internet test.")

    st.markdown("---")
    st.caption(
    "Built by San Kaneskan as a portfolio project in machine learning and financial data analysis."
)
