# Market Regime Detector

A Python and Streamlit project for detecting financial market regimes using financial time-series data, feature engineering, unsupervised machine learning, and simple backtest validation.

The project classifies market conditions into regimes such as **bull trend**, **bear trend**, **high volatility**, and **sideways / mean-reverting conditions**. Instead of predicting the exact next price, it focuses on identifying the current type of market environment.

Built by **San Kaneskan** as a portfolio project exploring machine learning, applied mathematics, and financial time-series analysis.

---

## Live Demo

Try the deployed app here:

https://san-market-regime.streamlit.app/

---

## Screenshots

### Dashboard

![Dashboard screenshot](images/dashboard.png)

### Regime Chart

![Regime chart screenshot](images/regime-chart.png)

### Backtest Metrics

![Backtest metrics screenshot](images/backtest-metrics.png)

---

## Project Overview

Financial markets behave differently over time. Some periods trend upward, some trend downward, some become highly volatile, and others move sideways without a clear direction.

This project follows this pipeline:

```text
Market data
    ↓
Feature engineering
    ↓
Regime detection model
    ↓
Regime labels
    ↓
Visualization
    ↓
Backtest validation
```

---

## Features

* Load real market data from Yahoo Finance using `yfinance`
* Generate synthetic demo data for testing and offline use
* Create financial features from OHLCV data
* Detect regimes using rule-based logic, KMeans, GMM, or HMM
* Visualize regimes with interactive Plotly charts
* Explore results in a Streamlit dashboard
* Run the project from the dashboard or command line
* Run smoke tests with `pytest`

---

## Tech Stack

| Area             | Tools                  |
| ---------------- | ---------------------- |
| Language         | Python                 |
| Data processing  | NumPy, Pandas          |
| Machine learning | scikit-learn, hmmlearn |
| Market data      | yfinance               |
| Visualization    | Plotly                 |
| Dashboard        | Streamlit              |
| Testing          | pytest                 |

---

## Project Structure

```text
market-regime-detector/
│
├── app/
│   └── streamlit_app.py
│
├── market_regime_detector/
│   ├── __init__.py
│   ├── backtest.py
│   ├── cli.py
│   ├── data.py
│   ├── features.py
│   ├── regimes.py
│   └── visualization.py
│
├── tests/
│   └── test_smoke.py
│
├── images/
│   ├── dashboard.png
│   ├── regime-chart.png
│   └── backtest-metrics.png
│
├── README.md
├── LEARNING_PATH.md
├── requirements.txt
├── pyproject.toml
└── .gitignore
```

---

## How It Works

### 1. Data Loading

The app supports:

* **Yahoo Finance data** for real historical market data
* **Synthetic demo data** for testing and demonstration

Main file:

```text
market_regime_detector/data.py
```

---

### 2. Feature Engineering

The raw market data contains OHLCV columns:

```text
Open, High, Low, Close, Volume
```

The project transforms these into financial features such as:

| Feature            | Meaning                                 |
| ------------------ | --------------------------------------- |
| `return_1d`        | One-period percentage price change      |
| `log_return_1d`    | Logarithmic price return                |
| rolling volatility | Recent size of price movements          |
| momentum           | Recent trend direction                  |
| RSI                | Overbought / oversold indicator         |
| ATR                | Average daily price range               |
| Bollinger z-score  | Distance from recent average price      |
| volume z-score     | Whether volume is unusually high or low |

Main file:

```text
market_regime_detector/features.py
```

---

### 3. Regime Detection

The project supports four regime detection methods:

| Method     | Description                                    | Best for                    |
| ---------- | ---------------------------------------------- | --------------------------- |
| Rule-based | Uses transparent if/else thresholds            | Learning and explainability |
| KMeans     | Groups similar periods into hard clusters      | Simple ML baseline          |
| GMM        | Uses probabilistic clustering                  | Recommended default         |
| HMM        | Models hidden states and transitions over time | Advanced sequence modelling |

Main file:

```text
market_regime_detector/regimes.py
```

---

### 4. Visualization and Backtesting

The dashboard shows:

* market price over time
* detected regimes
* latest feature rows
* regime profiles
* backtest metrics
* explanations for model outputs

The backtest is used as a simple validation step to compare a regime-aware strategy against a buy-and-hold benchmark. It is not intended to represent a production trading system.

Main files:

```text
market_regime_detector/visualization.py
market_regime_detector/backtest.py
app/streamlit_app.py
```

---

## Why I Chose This Approach

I chose regime detection because I wanted to build a project that combines machine learning, applied mathematics, and financial data analysis without making unrealistic claims about exact price prediction.

Predicting exact future prices is difficult and often misleading. Regime detection asks a more practical question:

```text
What type of market environment are we currently in?
```

This still involves important machine learning concepts, but it is easier to explain, evaluate, and visualize.

---

## Technical Decisions

### Plotly for visualization

I used Plotly because the dashboard is interactive. Users can zoom, hover, inspect individual points, and explore regime changes visually.

### Synthetic data

Synthetic data is included so the project can still run for testing and demonstration even if Yahoo Finance data is unavailable.

### GMM as the recommended default

GMM is the recommended default because it is more flexible than KMeans but easier to explain than HMM. It handles overlapping regimes better than hard clustering.

### HMM as an advanced option

HMM is included because market regimes are sequential. A market can remain in one state for multiple periods before transitioning to another.

---

## Limitations

This project is educational and should not be treated as a production trading system.

Current limitations:

* The backtest does not include trading fees, slippage, taxes, or realistic execution constraints.
* The validation is simple and does not include full walk-forward testing.
* More complex models, especially HMM, can overfit historical data.
* Regime labels are statistical interpretations and may not always match real economic regimes.
* Yahoo Finance data can sometimes contain missing values or adjusted price differences.
* The model currently uses only price and volume data.

---

## What I Learned

While building this project, I learned how to:

* Structure a Python data science project into reusable modules
* Work with OHLCV financial market data
* Create features such as returns, volatility, momentum, RSI, ATR, and Bollinger z-score
* Apply unsupervised learning to time-series data
* Compare rule-based logic, KMeans, GMM, and HMM
* Build an interactive dashboard with Streamlit and Plotly
* Add a command-line interface for reproducible experiments
* Use basic tests to check that the main pipeline works
* Think about validation issues such as look-ahead bias and overfitting

---

## Future Improvements

Future improvements I would like to add:

1. Walk-forward validation
2. Transaction costs and slippage in the backtest
3. Support for comparing multiple tickers
4. A dashboard page for comparing regime detection methods
5. More tests for missing data and failed downloads
6. More stable regime naming across different assets
7. Additional data sources such as macro indicators or volatility indexes

A longer-term idea is to expand this into a broader Quant ML Research Dashboard with additional modules.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/SanKTH3/market-regime-detector.git
cd market-regime-detector
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Activate it on macOS or Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the App Locally

Start the Streamlit dashboard:

```bash
streamlit run app/streamlit_app.py
```

Then open the local URL shown in the terminal.

---

## Command Line Usage

Run with synthetic demo data:

```bash
python -m market_regime_detector.cli --synthetic --method gmm --out outputs/demo_gmm
```

Run with Yahoo Finance data:

```bash
python -m market_regime_detector.cli --ticker SPY --period 5y --method gmm --out outputs/spy_gmm
```

Available methods:

```text
rule
kmeans
gmm
hmm
```

The command-line version can generate local result files inside an `outputs/` folder. This folder is not included in the repository because it is generated automatically.

---

## Running Tests

Run the smoke tests with:

```bash
pytest
```

---

## Deployment

The app is deployed on Streamlit Community Cloud:

https://san-market.streamlit.app/

Streamlit installs the dependencies from `requirements.txt` during deployment.

---

## Disclaimer

This project is for educational and portfolio purposes only.

It is not financial advice, investment advice, or a production trading system. Historical backtest results do not guarantee future performance.
