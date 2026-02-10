import numpy as np
import pandas as pd


class FeatureBuilder:
    def __init__(self, df: pd.DataFrame, config, logger):
        self.config = config
        self.logger = logger

        # Sort once, reset index once
        self.df = (
            df.sort_values(["symbol", "timestamp"])
              .reset_index(drop=True)
        )

        # Reduce memory pressure
        float_cols = ["open", "high", "low", "close", "volume"]
        self.df[float_cols] = self.df[float_cols].astype("float32")

    def generate_features(self) -> pd.DataFrame:
        self.logger.info("Starting optimized feature generation...")

        self._returns()
        self._momentum()
        self._volatility()
        self._liquidity()
        self._technicals()

        before = len(self.df)
        self.df.dropna(inplace=True)
        self.logger.info(f"Dropped {before - len(self.df)} rows (lags / rolling)")

        return self.df

    # ------------------------------------------------------------------
    # RETURNS
    # ------------------------------------------------------------------
    def _returns(self):
        log_close = np.log(self.df["close"])
        grp = log_close.groupby(self.df["symbol"])

        # Base log returns (fast C-path)
        self.df["log_ret_1m"] = grp.diff(1)

        for h in (5, 15, 30, 60):
            self.df[f"log_ret_{h}m"] = grp.diff(h)

        self.df["ret_pos"] = self.df["log_ret_1m"].clip(lower=0)
        self.df["ret_neg"] = self.df["log_ret_1m"].clip(upper=0)

    # ------------------------------------------------------------------
    # MOMENTUM / REVERSAL
    # ------------------------------------------------------------------
    def _momentum(self):
        grp_ret = self.df.groupby("symbol")["log_ret_1m"]

        self.df["mom_5m"] = self.df["log_ret_5m"]
        self.df["mom_30m"] = grp_ret.transform(lambda x: x.rolling(30).sum())

        rolling_std = grp_ret.transform(lambda x: x.rolling(60).std())
        r = self.df["log_ret_1m"]

        self.df["reversal_signal"] = np.where(
            r.abs() > 2 * rolling_std,
            -np.sign(r),
            0,
        )

    # ------------------------------------------------------------------
    # VOLATILITY
    # ------------------------------------------------------------------
    def _volatility(self):
        grp_ret = self.df.groupby("symbol")["log_ret_1m"]

        # Realized volatility
        self.df["RV_20"] = grp_ret.transform(
            lambda x: np.sqrt((x ** 2).rolling(20).sum())
        )

        # EWMA volatility
        self.df["EWMA_vol"] = grp_ret.transform(
            lambda x: np.sqrt((x ** 2).ewm(alpha=0.06, adjust=False).mean())
        )

        # Parkinson volatility
        hl = np.log(self.df["high"] / self.df["low"])
        self.df["parkinson_vol"] = np.sqrt((hl ** 2) / (4 * np.log(2)))

        self.df["vol_of_vol"] = (
            self.df.groupby("symbol")["RV_20"]
            .transform(lambda x: x.rolling(60).std())
        )

    # ------------------------------------------------------------------
    # LIQUIDITY
    # ------------------------------------------------------------------
    def _liquidity(self):
        grp = self.df.groupby("symbol")

        # Rolling relative volume (instead of TOD grouping)
        self.df["rel_vol"] = (
            self.df["volume"] /
            grp["volume"].transform(
                lambda x: x.rolling(390, min_periods=30).mean()
            )
        )

        # Volume imbalance
        rng = (self.df["high"] - self.df["low"]).replace(0, np.nan)
        self.df["vol_imbalance"] = (
            ((self.df["close"] - self.df["open"]) / rng) * self.df["volume"]
        ).fillna(0)

        # Amihud
        dollar_vol = self.df["close"] * self.df["volume"]
        self.df["amihud_illiq"] = self.df["log_ret_1m"].abs() / (dollar_vol + 1e-9)

        # Spread proxies
        self.df["spread_hl"] = (self.df["high"] - self.df["low"]) / self.df["close"]

        def roll_spread(x):
            cov = x.rolling(20).cov(x.shift(1))
            return 2 * np.sqrt(-cov.where(cov < 0, 0))

        self.df["spread_roll"] = grp["log_ret_1m"].transform(roll_spread)

    # ------------------------------------------------------------------
    # TECHNICALS
    # ------------------------------------------------------------------
    def _technicals(self):
        grp = self.df.groupby("symbol")["close"]

        sma_20 = grp.transform(lambda x: x.rolling(20).mean())
        self.df["SMA_60"] = grp.transform(lambda x: x.rolling(60).mean())
        self.df["EMA_20"] = grp.transform(lambda x: x.ewm(span=20, adjust=False).mean())

        self.df["dist_sma_20"] = (self.df["close"] - sma_20) / sma_20

        def rsi(x, n=14):
            d = x.diff()
            gain = d.clip(lower=0).rolling(n).mean()
            loss = (-d.clip(upper=0)).rolling(n).mean()
            rs = gain / loss
            return 100 - (100 / (1 + rs))

        self.df["rsi_14"] = grp.transform(rsi)

# import pandas as pd
# import numpy as np
# from src.utils.app_logger import AppLogger
# from src.config import AppConfig


# class FeatureBuilder:
#     def __init__(self, df: pd.DataFrame, config: AppConfig, logger: AppLogger):
#         self.df = df.copy()
#         self.config = config
#         self.logger = logger

#         # Ensure sorted and time-indexed
#         self.df = self.df.sort_values(["symbol", "timestamp"])

#     def generate_features(self) -> pd.DataFrame:
#         self.logger.info("Starting feature generation...")

#         # 3.2 Return-Based Features
#         self._add_multi_horizon_returns()
#         self._add_signed_returns()
#         self._add_momentum_features()

#         # 3.3 Volatility Features
#         self._add_volatility_features()

#         # 3.4 Liquidity Features
#         self._add_liquidity_features()

#         # 3.5 Technical Indicators
#         self._add_technical_indicators()

#         # Cleanup
#         initial_rows = len(self.df)
#         self.df = self.df.dropna()
#         dropped_rows = initial_rows - len(self.df)
#         self.logger.info(
#             f"Feature generation complete. Dropped {dropped_rows} rows due to NaN (lags/rolling)."
#         )

#         return self.df

#     def _add_multi_horizon_returns(self):
#         grp = self.df.groupby("symbol")["close"]

#         # Base 1m log return
#         self.df["log_ret_1m"] = np.log(grp.transform("last") / grp.shift(1))

#         # Derived horizons via shift of log returns
#         for h in [5, 15, 30, 60]:
#             self.df[f"log_ret_{h}m"] = grp.transform(lambda x: np.log(x / x.shift(h)))

#     # def _add_multi_horizon_returns(self):
#     #     """3.2.1 Multi-Horizon Log Returns"""
#     #     horizons = [1, 5, 15, 30, 60]

#     #     for h in horizons:
#     #         self.df[f"log_ret_{h}m"] = self.df.groupby("symbol")["close"].transform(
#     #             lambda x: np.log(x / x.shift(h))
#     #         )

#     #     # for h in horizons:
#     #     #     # Log returns: ln(P_t / P_{t-h})
#     #     #     self.df[f"log_ret_{h}m"] = self.df.groupby("symbol")["close"].apply(
#     #     #         lambda x: np.log(x / x.shift(h))
#     #     #     )

#     def _add_signed_returns(self):
#         """3.2.2 Signed Returns (Asymmetric Response)"""
#         # Based on 1-minute return
#         if "log_ret_1m" not in self.df.columns:
#             self._add_multi_horizon_returns()

#         self.df["ret_pos"] = self.df["log_ret_1m"].clip(lower=0)
#         self.df["ret_neg"] = self.df["log_ret_1m"].clip(upper=0)

#     def _add_momentum_features(self):
#         """3.2.3 Momentum vs Reversal"""
#         # 1. Short-term momentum (5m)
#         self.df["mom_5m"] = np.sign(self.df["log_ret_5m"]) * self.df["log_ret_5m"].abs()

#         # 2. Medium-term momentum (30m sum of returns)
#         # Note: log_ret_30m is simple return over 30m, but doc asks for "sum of last 30 returns"
#         # which acts as a path-dependent metric.
#         self.df["mom_30m"] = self.df.groupby("symbol")["log_ret_1m"].transform(
#             lambda x: x.rolling(30).sum()
#         )

#         # 3. Reversal Signal (> 2 std dev)
#         rolling_std = self.df.groupby("symbol")["log_ret_1m"].transform(
#             lambda x: x.rolling(60).std()
#         )
#         self.df["reversal_signal"] = np.where(
#             self.df["log_ret_1m"].abs() > (2 * rolling_std),
#             -np.sign(self.df["log_ret_1m"]),
#             0,
#         )

#     def _add_volatility_features(self):
#         """3.3 Volatility Features"""
#         # 1. Realized Volatility (Rolling 20 bars)
#         # self.df["RV_20"] = self.df.groupby("symbol")["log_ret_1m"].transform(
#         #     lambda x: x.rolling(20).apply(lambda r: np.sqrt(np.sum(r**2)))
#         # )
#         self.df["RV_20"] = self.df.groupby("symbol")["log_ret_1m"].transform(
#             lambda x: np.sqrt((x**2).rolling(20).sum())
#         )

#         # 2. EWMA Volatility (lambda = 0.94)
#         # Variance_t = lambda * Var_{t-1} + (1-lambda) * r_t^2
#         # In pandas ewm, alpha = 1 - lambda. So alpha = 0.06.
#         lambda_param = 0.94
#         self.df["EWMA_vol"] = self.df.groupby("symbol")["log_ret_1m"].transform(
#             lambda x: np.sqrt((x**2).ewm(alpha=1 - lambda_param, adjust=False).mean())
#         )

#         # 3. Parkinson Estimator
#         const_factor = 1.0 / (4 * np.log(2))
#         log_hl = np.log(self.df["high"] / self.df["low"])
#         self.df["parkinson_vol"] = np.sqrt(const_factor * log_hl**2)

#         # 4. Volatility of Volatility (Std of RV over 60 bars)
#         self.df["vol_of_vol"] = self.df.groupby("symbol")["RV_20"].transform(
#             lambda x: x.rolling(60).std()
#         )

#     def _add_liquidity_features(self):
#         """3.4 Liquidity Features"""
#         # 1. Relative Volume (Time of Day Average)
#         # Extract time for grouping
#         self.df["time"] = pd.DatetimeIndex(self.df["timestamp"]).time
#         avg_vol_tod = self.df.groupby(["symbol", "time"])["volume"].transform("mean")
#         self.df["rel_vol"] = self.df["volume"] / avg_vol_tod.replace(
#             0, 1
#         )  # Avoid div/0

#         # 2. Volume Imbalance
#         # (Close - Open) / (High - Low) * Volume
#         range_hl = (self.df["high"] - self.df["low"]).replace(0, np.nan)  # Avoid div/0
#         self.df["vol_imbalance"] = (
#             (self.df["close"] - self.df["open"]) / range_hl
#         ) * self.df["volume"]
#         self.df["vol_imbalance"] = self.df["vol_imbalance"].fillna(0)

#         # 3. Amihud Illiquidity
#         dollar_vol = self.df["close"] * self.df["volume"]
#         self.df["amihud_illiq"] = self.df["log_ret_1m"].abs() / (dollar_vol + 1e-9)

#         # 4. Spread Proxies
#         # a. High-Low Spread
#         self.df["spread_hl"] = (self.df["high"] - self.df["low"]) / self.df["close"]

#         # b. Roll's Spread Estimator: 2 * sqrt(-Cov(r_t, r_{t-1}))
#         def calc_roll_spread(series):
#             # Rolling covariance of returns with lagged returns
#             cov = series.rolling(20).cov(series.shift(1))
#             # Only valid where covariance is negative
#             return 2 * np.sqrt(-cov.where(cov < 0, 0))

#         self.df["spread_roll"] = self.df.groupby("symbol")["log_ret_1m"].transform(
#             calc_roll_spread
#         )

#     def _add_technical_indicators(self):
#         """3.5 Technical Indicator Features"""
#         grp = self.df.groupby("symbol")["close"]

#         # Moving Averages (20, 60)
#         sma_20 = grp.transform(lambda x: x.rolling(20).mean())
#         self.df["SMA_60"] = grp.transform(lambda x: x.rolling(60).mean())

#         # EMA 20
#         self.df["EMA_20"] = grp.transform(lambda x: x.ewm(span=20, adjust=False).mean())

#         # Distance from SMA
#         self.df["dist_sma_20"] = (self.df["close"] - sma_20) / sma_20

#         # RSI 14
#         def calc_rsi(series, period=14):
#             delta = series.diff()
#             gain = (delta.where(delta > 0, 0)).rolling(period).mean()
#             loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
#             rs = gain / loss
#             return 100 - (100 / (1 + rs))

#         self.df["rsi_14"] = grp.transform(calc_rsi)
