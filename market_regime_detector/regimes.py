from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Literal, Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from .features import DEFAULT_FEATURE_COLUMNS, select_feature_matrix

RegimeMethod = Literal["rule", "kmeans", "gmm", "hmm"]


@dataclass
class RegimeResult:
    data: pd.DataFrame
    profiles: pd.DataFrame
    method: str
    feature_columns: list[str]


class RegimeDetector:
    """Fit a market-regime detector.

    Methods:
      - rule: transparent thresholds, best for learning/debugging.
      - kmeans: beginner-friendly unsupervised clustering.
      - gmm: probabilistic clustering, recommended default.
      - hmm: sequential probabilistic regimes that model state transitions.
    """

    def __init__(
        self,
        method: RegimeMethod = "gmm",
        n_regimes: int = 4,
        feature_columns: Optional[Iterable[str]] = None,
        random_state: int = 42,
    ) -> None:
        self.method = method
        self.n_regimes = n_regimes
        self.feature_columns = list(feature_columns or DEFAULT_FEATURE_COLUMNS)
        self.random_state = random_state
        self.scaler: Optional[StandardScaler] = None
        self.model = None
        self.cluster_name_map: Dict[int, str] = {}

    def fit_predict(self, feature_df: pd.DataFrame) -> RegimeResult:
        if self.method == "rule":
            labels = self._rule_based_labels(feature_df)
        else:
            x = select_feature_matrix(feature_df, self.feature_columns)
            self.scaler = StandardScaler()
            x_scaled = self.scaler.fit_transform(x)
            raw_labels = self._fit_unsupervised(x_scaled)
            labels = pd.Series(raw_labels, index=feature_df.index, name="cluster_id")

        out = feature_df.copy()
        out["cluster_id"] = labels.astype(int)
        profiles = _build_profiles(out, self.feature_columns)
        self.cluster_name_map = _name_clusters(profiles)
        out["regime"] = out["cluster_id"].map(self.cluster_name_map)
        profiles["regime"] = profiles.index.map(self.cluster_name_map)
        return RegimeResult(out, profiles, self.method, self.feature_columns)

    def _fit_unsupervised(self, x_scaled: np.ndarray) -> np.ndarray:
        if self.method == "kmeans":
            self.model = KMeans(n_clusters=self.n_regimes, n_init="auto", random_state=self.random_state)
            return self.model.fit_predict(x_scaled)

        if self.method == "gmm":
            self.model = GaussianMixture(
                n_components=self.n_regimes,
                covariance_type="full",
                n_init=5,
                random_state=self.random_state,
            )
            return self.model.fit_predict(x_scaled)

        if self.method == "hmm":
            try:
                from hmmlearn.hmm import GaussianHMM
            except ImportError as exc:
                raise ImportError(
                    "hmmlearn is required for --method hmm. Install it with "
                    "`pip install hmmlearn`, or use --method gmm/kmeans/rule."
                ) from exc
            self.model = GaussianHMM(
                n_components=self.n_regimes,
                covariance_type="full",
                n_iter=500,
                random_state=self.random_state,
            )
            self.model.fit(x_scaled)
            return self.model.predict(x_scaled)

        raise ValueError(f"Unknown method: {self.method}")

    def _rule_based_labels(self, df: pd.DataFrame) -> pd.Series:
        """Simple transparent rules based on volatility, slope, and momentum."""
        vol_hi = df["rolling_vol_20"].quantile(0.70)
        vol_lo = df["rolling_vol_20"].quantile(0.35)
        slope_hi = df["ma_slope_20"].quantile(0.60)
        slope_lo = df["ma_slope_20"].quantile(0.40)
        mom_hi = df["momentum_20"].quantile(0.60)
        mom_lo = df["momentum_20"].quantile(0.40)

        labels = pd.Series(index=df.index, dtype=int)
        labels.loc[df["rolling_vol_20"] >= vol_hi] = 0  # high volatility
        labels.loc[(df["rolling_vol_20"] < vol_hi) & (df["ma_slope_20"] > slope_hi) & (df["momentum_20"] > mom_hi)] = 1
        labels.loc[(df["rolling_vol_20"] < vol_hi) & (df["ma_slope_20"] < slope_lo) & (df["momentum_20"] < mom_lo)] = 2
        labels.loc[(df["rolling_vol_20"] <= vol_lo) & (df["momentum_20"].abs() <= df["momentum_20"].abs().quantile(0.60))] = 3
        labels = labels.fillna(3)
        return labels.astype(int)


def _build_profiles(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    profile_cols = [
        "return_1d",
        "rolling_vol_20",
        "momentum_20",
        "ma_slope_20",
        "rsi_14",
        "autocorr_20",
        "bb_z_20",
    ]
    profile_cols = [c for c in profile_cols if c in df.columns]
    profiles = df.groupby("cluster_id")[profile_cols].mean()
    profiles["count"] = df.groupby("cluster_id").size()
    profiles["avg_forward_5d_return"] = df["Close"].pct_change(5).shift(-5).groupby(df["cluster_id"]).mean()
    return profiles.sort_index()


def _name_clusters(profiles: pd.DataFrame) -> Dict[int, str]:
    """Convert anonymous cluster IDs into finance-readable labels."""
    vol = profiles["rolling_vol_20"]
    mom = profiles["momentum_20"]
    slope = profiles["ma_slope_20"]
    rsi = profiles.get("rsi_14", pd.Series(50, index=profiles.index))

    vol_hi = vol.quantile(0.70)
    mom_hi = mom.quantile(0.55)
    mom_lo = mom.quantile(0.45)
    slope_hi = slope.quantile(0.55)
    slope_lo = slope.quantile(0.45)

    names: Dict[int, str] = {}
    used_counts: Dict[str, int] = {}
    for cluster_id in profiles.index:
        if vol.loc[cluster_id] >= vol_hi:
            base = "High Volatility / Risk-Off"
        elif mom.loc[cluster_id] >= mom_hi and slope.loc[cluster_id] >= slope_hi and rsi.loc[cluster_id] >= 50:
            base = "Bull Trend"
        elif mom.loc[cluster_id] <= mom_lo and slope.loc[cluster_id] <= slope_lo:
            base = "Bear Trend"
        else:
            base = "Sideways / Mean Reversion"

        # If two clusters share a name, make the label unique but still readable.
        count = used_counts.get(base, 0) + 1
        used_counts[base] = count
        names[int(cluster_id)] = base if count == 1 else f"{base} #{count}"
    return names


def available_methods() -> dict[str, dict[str, str]]:
    return {
        "rule": {
            "level": "beginner",
            "summary": "Transparent thresholds; easiest to understand and debug.",
        },
        "kmeans": {
            "level": "recommended first ML version",
            "summary": "Fast unsupervised clustering; good baseline for portfolio comparisons.",
        },
        "gmm": {
            "level": "recommended default",
            "summary": "Probabilistic clustering; handles soft, overlapping regimes better than KMeans.",
        },
        "hmm": {
            "level": "advanced",
            "summary": "Hidden Markov Model; models regime transitions over time.",
        },
    }
