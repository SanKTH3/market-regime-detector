from market_regime_detector.backtest import build_strategy_returns, performance_metrics
from market_regime_detector.data import MarketDataRequest, load_market_data
from market_regime_detector.features import build_features
from market_regime_detector.regimes import RegimeDetector


def test_synthetic_gmm_smoke():
    ohlcv = load_market_data(MarketDataRequest(synthetic=True))
    features = build_features(ohlcv)
    result = RegimeDetector(method="gmm", n_regimes=4).fit_predict(features)
    bt = build_strategy_returns(result.data)
    metrics = performance_metrics(bt)
    assert not result.data.empty
    assert "regime" in result.data.columns
    assert len(result.profiles) >= 2
    assert not metrics.empty


def test_rule_smoke():
    ohlcv = load_market_data(MarketDataRequest(synthetic=True))
    features = build_features(ohlcv)
    result = RegimeDetector(method="rule", n_regimes=4).fit_predict(features)
    assert "regime" in result.data.columns
