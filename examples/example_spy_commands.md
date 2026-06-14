# Example commands

Run these from the project root after installing dependencies.

## No internet demo

```bash
python -m market_regime_detector.cli --synthetic --method rule --out outputs/demo_rule
python -m market_regime_detector.cli --synthetic --method kmeans --out outputs/demo_kmeans
python -m market_regime_detector.cli --synthetic --method gmm --out outputs/demo_gmm
```

## SPY examples

```bash
python -m market_regime_detector.cli --ticker SPY --period 5y --method rule --out outputs/spy_rule
python -m market_regime_detector.cli --ticker SPY --period 5y --method kmeans --n-regimes 4 --out outputs/spy_kmeans
python -m market_regime_detector.cli --ticker SPY --period 5y --method gmm --n-regimes 4 --out outputs/spy_gmm
python -m market_regime_detector.cli --ticker SPY --period 5y --method hmm --n-regimes 4 --out outputs/spy_hmm
```

## Crypto example

```bash
python -m market_regime_detector.cli --ticker BTC-USD --period 5y --method gmm --out outputs/btc_gmm
```

## Dashboard

```bash
streamlit run app/streamlit_app.py
```
