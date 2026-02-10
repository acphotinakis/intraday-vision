import pandas as pd
import numpy as np
from scipy import stats
from src.utils.custom_stats import linregress, LinregressResult
from src.utils.app_logger import AppLogger
from src.config import AppConfig


class DescriptiveAnalyzer:
    def __init__(self, df: pd.DataFrame, config: AppConfig, logger: AppLogger):
        self.df = df
        self.config = config
        self.logger = logger

        self.logger.info("Initializing DescriptiveAnalyzer")
        self.logger.info(f"Input DataFrame shape: {self.df.shape}")

        # Pre-compute returns (log returns are preferred for additivity, but simple for readability)
        self.df = self.df.sort_values(["symbol", "timestamp"])
        self.logger.info("Sorted DataFrame by symbol and timestamp")

        # self.df["log_ret"] = self.df.groupby("symbol")["close"].apply(
        #     lambda x: np.log(x / x.shift(1))
        # )
        # self.df["simple_ret"] = self.df.groupby("symbol")["close"].pct_change()
        self.df["log_ret"] = self.df.groupby("symbol")["close"].transform(
            lambda x: np.log(x / x.shift(1))
        )
        self.df["simple_ret"] = self.df.groupby("symbol")["close"].transform(
            lambda x: x.pct_change()
        )
        self.logger.info(
            "Computed log returns and simple returns. "
            f"Sample log_ret:\n{self.df['log_ret'].head(5)}\n"
            f"Sample simple_ret:\n{self.df['simple_ret'].head(5)}"
        )

        # Drop first row of NaN
        self.df = self.df.dropna(subset=["log_ret"])
        self.logger.info(
            f"Dropped rows with NaN in log_ret. Remaining rows: {len(self.df)}"
        )

    def run_all_analyses(self) -> dict:
        self.logger.info("Starting Stage 2: Descriptive Analysis")
        result = {
            "price_dynamics": {
                "cumulative_returns": self._calc_cumulative_returns(),
                "intraday_patterns": self._calc_intraday_patterns(),
                "distribution_stats": self._calc_distribution_stats(),
            },
            "volatility": {
                "realized_vol": self._calc_realized_volatility(),
                "vol_clustering": self._calc_vol_clustering(),
                "intraday_vol": self._calc_intraday_volatility(),
            },
            "volume": {
                "intraday_profile": self._calc_volume_profile(),
                "vol_vol_relation": self._calc_vol_vol_relationship(),
            },
            "cross_sectional": {
                "correlation": self._calc_correlations(),
                "betas": self._calc_rolling_betas(),
            },
        }

        self.logger.info("Stage 2 Descriptive Analysis completed.")
        self.logger.info(f"Result: {result}")
        return result

    # --- 2.2 Price Dynamics ---
    def _calc_cumulative_returns(self):
        """2.2.1 Normalized Price Paths"""
        self.logger.info("Calculating cumulative returns")
        pivot = self.df.pivot_table(index="timestamp", columns="symbol", values="close")
        self.logger.info(f"Pivot table shape: {pivot.shape}")
        normalized = (pivot / pivot.iloc[0]) * 100
        self.logger.info(
            "Normalized cumulative returns (first 5 rows):\n"
            + normalized.head(5).to_string()
        )
        return normalized

    def _calc_intraday_patterns(self):
        """2.2.2 Intraday Seasonality (U-Shape)"""
        self.logger.info("Calculating intraday price patterns")
        df_local = self.df.copy()
        df_local["timestamp"] = pd.to_datetime(df_local["timestamp"])
        df_local["hour"] = pd.DatetimeIndex(df_local["timestamp"]).hour
        df_local["time"] = pd.DatetimeIndex(df_local["timestamp"]).time

        # Mean return by minute
        intraday = df_local.groupby("time")["simple_ret"].mean() * 100  # in bps
        self.logger.info(
            "Intraday mean returns by minute (first 10):\n"
            + intraday.head(10).to_string()
        )

        self.logger.info(
            "df_local - First 10 rows of merged data:\n" + df_local.head(10).to_string()
        )

        self.logger.info(
            "intraday - First 10 rows of merged data:\n" + intraday.head(10).to_string()
        )

        # T-Test: First hour vs Last hour
        # Note: This is simplified; robust test requires careful handling of overnight gaps
        morning = df_local[df_local["timestamp"] == 9]["simple_ret"]
        evening = df_local[df_local["timestamp"] == 15]["simple_ret"]

        self.logger.info(
            f"Morning returns sample:\n{morning.head(10).to_string()}\n"
            f"Evening returns sample:\n{evening.head(10).to_string()}"
        )

        t_stat, p_val = stats.ttest_ind(
            morning.dropna(), evening.dropna(), equal_var=False
        )
        self.logger.info(f"T-test results: t_stat={t_stat}, p_value={p_val}")

        return {"profile": intraday, "stats": {"t_stat": t_stat, "p_value": p_val}}

    def _calc_distribution_stats(self):
        """2.2.3 Return Distribution (Moments)"""
        self.logger.info("Calculating return distribution moments per symbol")

        results = []
        for sym in self.df["symbol"].unique():
            rets = self.df[self.df["symbol"] == sym]["log_ret"]
            results.append(
                {
                    "symbol": sym,
                    "mean": rets.mean(),
                    "std": rets.std(),
                    "skew": rets.skew(),
                    "kurtosis": rets.kurtosis(),  # Excess kurtosis (Fisher)
                }
            )
            self.logger.info(
                f"{sym}: mean={rets.mean()}, std={rets.std()}, skew={rets.skew()}, kurtosis={rets.kurtosis()}"
            )

        df_stats = pd.DataFrame(results).set_index("symbol")
        self.logger.info("Completed distribution stats calculation")
        return df_stats

    # --- 2.3 Volatility ---
    def _calc_realized_volatility(self, window=21):
        """2.3.1 Rolling Realized Volatility (Annualized)"""
        self.logger.info(
            f"Calculating rolling realized volatility with window={window}"
        )
        SCALE_FACTOR = np.sqrt(252 * 390)
        rolling_vol = (
            self.df.groupby("symbol")["log_ret"].rolling(window=window * 390).std()
            * SCALE_FACTOR
        )
        self.logger.info(
            "Sample rolling volatility:\n" + rolling_vol.head(10).to_string()
        )
        return rolling_vol.dropna()

    def _calc_vol_clustering(self):
        """2.3.2 Volatility Autocorrelation (ACF)"""
        self.logger.info("Calculating volatility clustering (ACF)")
        acf_results = {}
        for sym in self.df["symbol"].unique():
            abs_rets = self.df[self.df["symbol"] == sym]["log_ret"].abs()
            # Calculate ACF for lags 1, 5, 10, 100
            acf_vals = [abs_rets.autocorr(lag=i) for i in [1, 5, 10, 100]]
            acf_results[sym] = acf_vals
            self.logger.info(f"{sym} ACF values: {acf_vals}")
        return pd.DataFrame(acf_results, index=[1, 5, 10, 100])

    def _calc_intraday_volatility(self):
        """2.3.3 Intraday Volatility Surface"""
        self.logger.info("Calculating intraday volatility surface")
        df_local = self.df.copy()
        df_local["time"] = pd.DatetimeIndex(df_local["timestamp"]).time

        # Std dev of returns by minute and symbol
        vol_surface = df_local.groupby(["symbol", "time"])["log_ret"].std().unstack()
        # Annualize
        vol_surface = vol_surface * np.sqrt(252 * 390)
        self.logger.info(
            "Intraday volatility surface (first 5 rows):\n"
            + vol_surface.head(5).to_string()
        )

        return vol_surface

    # --- 2.4 Volume ---
    def _calc_volume_profile(self):
        """2.4.1 Intraday Volume Profile"""
        self.logger.info("Calculating intraday volume profile")
        df_local = self.df.copy()
        df_local["time"] = pd.DatetimeIndex(df_local["timestamp"]).time

        # Mean volume by minute, normalized by symbol's average volume
        # (To prevent high volume stocks dominating the chart)
        avg_daily_vol = df_local.groupby("symbol")["volume"].transform("mean")
        df_local["norm_volume"] = df_local["volume"] / avg_daily_vol

        profile = df_local.groupby("time")["norm_volume"].mean()
        self.logger.info(
            "Intraday normalized volume profile (first 10 rows):\n"
            + profile.head(10).to_string()
        )

        return profile

    def _calc_vol_vol_relationship(self):
        """2.4.2 Volume-Volatility Regression"""
        self.logger.info("Calculating Volume-Volatility regression")

        # Filter out zero or negative volume
        df_clean = self.df[self.df["volume"] > 0].copy()

        # Compute log(volume)
        x = pd.Series(np.log(df_clean["volume"]), index=df_clean.index)

        # Compute log(abs(log_ret)), drop zeros to avoid log(0)
        y = pd.Series(df_clean["log_ret"].abs(), index=df_clean.index)
        y = y[y > 0]
        y = pd.Series(np.log(y), index=y.index)

        # Align indices
        common_idx = x.index.intersection(y.index)
        x = x.loc[common_idx]
        y = y.loc[common_idx]

        # Perform linear regression
        res: LinregressResult = linregress(x, y)

        self.logger.info(f"LinregressResult: {res}")

        return {
            "slope": res.slope,
            "intercept": res.intercept,
            "r_squared": res.rvalue**2,
            "p_value": res.pvalue,
            "std_err": res.stderr,
            "intercept_stderr": res.intercept_stderr,
        }

    # --- 2.5 Cross-Sectional ---
    def _calc_correlations(self):
        """2.5.1 Correlation Matrix"""
        self.logger.info("Calculating correlation matrix")
        pivot_rets = self.df.pivot_table(
            index="timestamp", columns="symbol", values="simple_ret"
        )
        corr_matrix = pivot_rets.corr()
        self.logger.info(
            "Correlation matrix (first 5 x 5 block):\n"
            + corr_matrix.iloc[:5, :5].to_string()
        )
        return corr_matrix

    def _calc_rolling_betas(self):
        """2.5.2 Rolling Beta to Benchmark (SPY proxy)"""
        self.logger.info("Calculating rolling betas relative to benchmark")
        pivot_rets = self.df.pivot_table(
            index="timestamp", columns="symbol", values="simple_ret"
        )

        if "SPY" in pivot_rets.columns:
            mkt = pivot_rets["SPY"]
            self.logger.info("Using SPY as market benchmark")
        else:
            self.logger.warning(
                "SPY not found. Using Equal Weighted Universe as Market Proxy."
            )
            mkt = pivot_rets.mean(axis=1)

        rolling_cov = pivot_rets.rolling(window=390 * 20).cov(mkt)
        rolling_var = mkt.rolling(window=390 * 20).var()
        betas = rolling_cov.div(rolling_var, axis=0)
        self.logger.info(
            "Sample rolling betas (first 10 rows):\n" + betas.head(10).to_string()
        )
        return betas
