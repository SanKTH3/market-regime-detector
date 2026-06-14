from __future__ import annotations

import argparse
from pathlib import Path

from .backtest import build_strategy_returns, performance_metrics
from .data import MarketDataRequest, load_market_data
from .features import build_features
from .regimes import RegimeDetector, available_methods
from .visualization import make_regime_chart, save_chart_html


def parse_args() -> argparse.Namespace:
    methods = list(available_methods().keys())
    parser = argparse.ArgumentParser(description="Detect market regimes from OHLCV data.")
    parser.add_argument("--ticker", default="SPY", help="Ticker for yfinance download, e.g. SPY, AAPL, BTC-USD.")
    parser.add_argument("--period", default="5y", help="yfinance period, e.g. 1y, 5y, 10y, max.")
    parser.add_argument("--interval", default="1d", help="yfinance interval, e.g. 1d, 1h.")
    parser.add_argument("--synthetic", action="store_true", help="Use built-in fake OHLCV data; no internet needed.")
    parser.add_argument("--method", choices=methods, default="gmm", help="Regime detection method.")
    parser.add_argument("--n-regimes", type=int, default=4, help="Number of regimes for ML methods.")
    parser.add_argument("--allow-short", action="store_true", help="Allow short positions in the validation backtest.")
    parser.add_argument("--out", default="outputs", help="Output directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    ohlcv = load_market_data(
        MarketDataRequest(
            ticker=args.ticker,
            period=args.period,
            interval=args.interval,
            synthetic=args.synthetic,
        )
    )
    features = build_features(ohlcv)
    detector = RegimeDetector(method=args.method, n_regimes=args.n_regimes)
    result = detector.fit_predict(features)
    bt = build_strategy_returns(result.data, allow_short=args.allow_short)
    metrics = performance_metrics(bt)

    data_path = out_dir / "regimes.csv"
    profiles_path = out_dir / "regime_profiles.csv"
    metrics_path = out_dir / "performance_metrics.csv"
    html_path = out_dir / "regime_dashboard.html"

    bt.to_csv(data_path)
    result.profiles.to_csv(profiles_path)
    metrics.to_csv(metrics_path, index=False)
    fig = make_regime_chart(bt, ticker=args.ticker if not args.synthetic else "Synthetic Demo")
    save_chart_html(fig, html_path)

    print("\nMarket Regime Detector complete")
    print(f"Method: {args.method}")
    print(f"Rows analyzed: {len(bt):,}")
    print("\nRegime profiles:")
    print(result.profiles.round(4).to_string())
    print("\nPerformance metrics:")
    print(metrics.round(4).to_string(index=False))
    print("\nFiles written:")
    print(f"- {data_path}")
    print(f"- {profiles_path}")
    print(f"- {metrics_path}")
    print(f"- {html_path}")


if __name__ == "__main__":
    main()
