import pandas as pd
import numpy as np
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from src.utils.app_logger import AppLogger
from src.utils.custom_stats import linregress, LinregressResult


class FeatureAnalyzer:
    def __init__(self, df: pd.DataFrame, logger: AppLogger):
        self.df = df
        self.logger = logger

        # Define all available candidate features
        self.feature_cols = [
            "log_ret_1m",
            "log_ret_5m",
            "mom_5m",
            "mom_30m",
            "RV_20",
            "EWMA_vol",
            "parkinson_vol",
            "vol_of_vol",
            "rel_vol",
            "vol_imbalance",
            "amihud_illiq",
            "spread_hl",
            "spread_roll",
            "dist_sma_20",
            "rsi_14",
        ]
        # Filter to those actually present
        self.feature_cols = [c for c in self.feature_cols if c in self.df.columns]

    def run_quality_assessment(self) -> dict:
        self.logger.info("Running Feature Quality Assessment")
        return {
            "autocorrelation": self._calc_return_autocorrelation(),
            "asymmetric_response": self._test_asymmetric_response(),  # 3.2.2
            "correlation_matrix": self._calc_feature_correlation(),
            "pca_analysis": self._run_pca(),
            "predictive_power": self._test_predictive_power(),
            "feature_stability": self._analyze_feature_stability(),  # 3.7.2
        }

    def _calc_return_autocorrelation(self):
        """3.2.1 Autocorrelation of Returns"""
        horizons = [1, 5, 15, 30, 60]
        lags = [1, 5, 15, 30, 60]
        results = {}

        for h in horizons:
            col = f"log_ret_{h}m"
            if col in self.df.columns:
                ac_values = []
                for lag in lags:
                    # Average AC across symbols
                    ac = (
                        self.df.groupby("symbol")[col]
                        .apply(lambda x: x.autocorr(lag=lag))
                        .mean()
                    )
                    ac_values.append(ac)
                results[f"{h}m Horizon"] = ac_values

        return pd.DataFrame(results, index=[f"Lag {l}" for l in lags])

    def _test_asymmetric_response(self):
        """3.2.2 Asymmetric Response (Chi-Squared Test)"""
        # Create contingency table: Current Direction vs Next Direction
        # 1 = Up, -1 = Down
        current_dir = np.sign(self.df["log_ret_1m"])
        # Next bar return direction
        next_dir = np.sign(self.df.groupby("symbol")["log_ret_1m"].shift(-1))

        # Remove zeros (flat) for stricter Up/Down test or treat 0 as noise
        mask = (current_dir != 0) & (next_dir != 0) & (~np.isnan(next_dir))

        if mask.sum() == 0:
            return {"chi2": 0, "p_value": 1.0, "contingency_table": None}

        contingency = pd.crosstab(
            current_dir[mask], next_dir[mask], rownames=["Current"], colnames=["Next"]
        )
        chi2, p, dof, expected = stats.chi2_contingency(contingency)

        self.logger.info(f"Asymmetric Response Chi2: {chi2:.2f}, p-value: {p:.4f}")
        return {"chi2": chi2, "p_value": p, "contingency_table": contingency}

    def _calc_feature_correlation(self):
        """3.6.1 Correlation Matrix"""
        return self.df[self.feature_cols].corr()

    def _run_pca(self):
        """3.6.2 PCA"""
        self.logger.info("Running PCA on features")
        # Handle Infs/NaNs before PCA just in case
        df_clean = (
            self.df[self.feature_cols].replace([np.inf, -np.inf], np.nan).dropna()
        )

        if df_clean.empty:
            return {}

        x = StandardScaler().fit_transform(df_clean.values)
        pca = PCA(n_components=min(len(self.feature_cols), 10))
        pca.fit(x)

        return {
            "explained_variance": pca.explained_variance_ratio_,
            "cumulative_variance": np.cumsum(pca.explained_variance_ratio_),
            "loadings": pd.DataFrame(
                pca.components_.T,
                columns=[f"PC{i+1}" for i in range(len(pca.explained_variance_ratio_))],
                index=self.feature_cols,
            ),
        }

    def _test_predictive_power(self):
        """3.7.1 Predictive Power (Univariate Regression)"""
        target_col = "target_ret_1m"
        self.df[target_col] = self.df.groupby("symbol")["log_ret_1m"].shift(-1)

        results = []
        for feature in self.feature_cols:
            # Drop NaNs for this specific pair
            valid = (
                self.df[[feature, target_col]]
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
            )

            if len(valid) < 100:
                continue

            x = valid[feature].values

            # For Volatility features, we predict next-bar Volatility |r|, not Direction r
            # Heuristic: if feature name implies volatility, target is abs(return)
            if any(x in feature.lower() for x in ["vol", "rv", "spread", "illiq"]):
                y = np.abs(valid[target_col].values)
            else:
                y = valid[target_col].values

            res = linregress(x, y)
            results.append(
                {
                    "feature": feature,
                    "r_squared": res.rvalue**2,
                    "t_stat": res.slope / res.stderr if res.stderr != 0 else 0,
                    "p_value": res.pvalue,
                }
            )

        return pd.DataFrame(results).sort_values("r_squared", ascending=False)

    def _analyze_feature_stability(self):
        """3.7.2 Feature Stability (Rolling Correlation)"""
        # Calculate rolling correlation of key features against next return (or vol)
        # We'll pick one representative feature per group to save time/space
        key_features = ["RV_20", "mom_5m", "spread_hl"]
        stability_results = {}

        target_ret = self.df.groupby("symbol")["log_ret_1m"].shift(-1)
        target_vol = target_ret.abs()

        for feat in key_features:
            if feat not in self.df.columns:
                continue

            target = target_vol if "RV" in feat or "spread" in feat else target_ret

            # Rolling correlation (window=1000 bars approx 2-3 days of minutes)
            rolling_corr = self.df.groupby("symbol").apply(
                lambda x: x[feat].rolling(1000).corr(target.loc[x.index])
            )

            # Reset index to keep timestamp for plotting
            # Note: groupby apply might result in multiindex
            stability_results[feat] = rolling_corr

        return stability_results
