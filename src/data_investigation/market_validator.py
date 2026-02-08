import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas_market_calendars as mcal
from datetime import datetime
from src.utils.app_logger import *
from src.config import AppConfig


class MarketDataValidator:
    def __init__(
        self,
        df: pd.DataFrame,
        config: AppConfig,
        logger: AppLogger,
        plots_dir: Path,
    ):
        """
        df: DataFrame with ['timestamp','symbol','open','high','low','close','volume']
        plots_dir: optional directory to save visualizations
        """
        self.logger = logger
        self.df = df.copy()
        self.df["timestamp"] = pd.to_datetime(self.df["timestamp"])
        self.symbols = self.df["symbol"].unique()
        self.config = config
        self.plots_dir = plots_dir
        if self.plots_dir:
            self.plots_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Validator initialized for {len(self.symbols)} symbols")

    # --- 1.2 Coverage and Completeness ---

    def check_temporal_coverage(self, start_date, end_date):
        """1.2.1 Verifies dates against NYSE calendar"""
        self.logger.info(
            f"Checking temporal coverage against NYSE calendar: {start_date} to {end_date}"
        )

        nyse = mcal.get_calendar("NYSE")
        schedule = nyse.schedule(start_date=start_date, end_date=end_date)
        expected_dates = schedule.index.to_series().apply(lambda x: x.date()).values

        self.logger.debug(
            f"NYSE calendar expects {len(expected_dates)} trading days in this range"
        )

        coverage_report = []
        for symbol in self.symbols:
            actual_dates = self.df[self.df["symbol"] == symbol][
                "timestamp"
            ].dt.date.unique()
            missing = set(expected_dates) - set(actual_dates)
            coverage = len(actual_dates) / len(expected_dates)

            if coverage < 1.0:
                self.logger.warning(
                    f"Incomplete coverage for {symbol}",
                    extra={
                        "symbol": symbol,
                        "coverage": f"{coverage:.2%}",
                        "missing_days_count": len(missing),
                    },
                )

            coverage_report.append(
                {"symbol": symbol, "coverage": coverage, "missing_count": len(missing)}
            )

        report_df = pd.DataFrame(coverage_report)
        self.logger.info(
            "Temporal coverage check completed",
            extra={
                "avg_coverage": f"{report_df['coverage'].mean():.2%}",
                "total_missing_bars": int(report_df["missing_count"].sum()),
            },
        )
        return report_df

    def analyze_intraday_completeness(self):
        """1.2.2 Analyzes bar counts per day (Target: 390 for NYSE)"""
        self.logger.info("Starting intraday bar count analysis (Target: 390 bars/day)")

        # Explicitly create a 'date' column without triggering Pylance warning
        self.df["date"] = self.df["timestamp"].apply(
            lambda x: x.date() if pd.notnull(x) else None
        )

        counts = (
            self.df.groupby(["symbol", "date"]).size().reset_index(name="bar_count")
        )

        # Flagging logic
        counts["status"] = "normal"
        counts.loc[counts["bar_count"] == 210, "status"] = "early_close"
        counts.loc[
            (counts["bar_count"] < 390) & (counts["bar_count"] != 210), "status"
        ] = "investigate_missing"
        counts.loc[counts["bar_count"] > 390, "status"] = "investigate_extra"

        status_summary = counts["status"].value_counts().to_dict()
        self.logger.info(
            "Intraday completeness analysis finished", extra=status_summary
        )

        if status_summary.get("investigate_missing", 0) > 0:
            self.logger.error(
                f"Found {status_summary['investigate_missing']} days with missing intraday bars!"
            )

        return counts

    # --- 1.3 Schema and Structural Validation ---

    def validate_ohlc_integrity(self):
        """1.3.1 Mathematical OHLC constraints"""
        self.logger.info("Validating OHLC mathematical integrity constraints")

        errors = self.df[
            (self.df["high"] < self.df["low"])
            | (self.df["high"] < self.df["open"])
            | (self.df["high"] < self.df["close"])
            | (self.df["low"] > self.df["open"])
            | (self.df["low"] > self.df["close"])
            | (self.df["open"] <= 0)
            | (self.df["volume"] < 0)
        ]

        if not errors.empty:
            self.logger.error(
                f"OHLC Integrity Violation: {len(errors)} invalid rows detected!",
                extra={"violated_symbols": list(errors["symbol"].unique()[:5])},
            )
        else:
            self.logger.info(
                "OHLC integrity check passed: No mathematical violations found"
            )

        return errors

    def check_timestamp_integrity(self):
        """1.3.2 Monotonicity and Duplicates"""
        self.logger.info("Checking timestamp monotonicity and identifying duplicates")

        report = []
        for symbol in self.symbols:
            sym_data = self.df[self.df["symbol"] == symbol].sort_values("timestamp")

            # Check for duplicates
            dupes = sym_data.duplicated(subset=["timestamp"]).sum()

            # Check for monotonicity
            is_monotonic = sym_data["timestamp"].is_monotonic_increasing

            if dupes > 0 or not is_monotonic:
                self.logger.warning(
                    f"Timestamp issue detected for {symbol}",
                    extra={
                        "symbol": symbol,
                        "duplicate_count": int(dupes),
                        "is_monotonic": is_monotonic,
                    },
                )

            report.append(
                {"symbol": symbol, "duplicates": dupes, "is_monotonic": is_monotonic}
            )
        return pd.DataFrame(report)

    # --- 1.4 Distributional Sanity Checks ---

    def detect_price_anomalies(self, threshold=0.5):
        """1.4.1 Detects sudden shifts (potential unadjusted splits)"""
        self.logger.info(f"Detecting price anomalies with threshold: {threshold*100}%")

        self.df = self.df.sort_values(["symbol", "timestamp"])
        self.df["returns"] = self.df.groupby("symbol")["close"].pct_change()

        anomalies = self.df[abs(self.df["returns"]) > threshold]

        if not anomalies.empty:
            self.logger.warning(
                f"Detected {len(anomalies)} price anomalies/spikes",
                extra={
                    "max_return": f"{anomalies['returns'].max():.2%}",
                    "min_return": f"{anomalies['returns'].min():.2%}",
                },
            )

        return anomalies

    def detect_volume_outliers(self, sigma=10):
        """1.4.2 Volume spikes > N sigma from mean log volume"""
        self.logger.info(f"Scanning for volume outliers using {sigma}-sigma threshold")

        self.df["log_vol"] = np.log1p(self.df["volume"])
        stats = self.df.groupby("symbol")["log_vol"].agg(["mean", "std"]).reset_index()

        merged = self.df.merge(stats, on="symbol")
        outliers = merged[merged["log_vol"] > (merged["mean"] + sigma * merged["std"])]

        self.logger.info(
            f"Volume outlier scan complete. Found {len(outliers)} bars exceeding threshold"
        )
        return outliers

    # --- 1.5 Cross-Sectional Consistency ---

    def calculate_market_concordance(self):
        """1.5.2 % of symbols moving in the same direction per bar"""
        self.logger.info("Calculating market cross-sectional concordance")

        self.df["direction"] = np.sign(self.df["close"] - self.df["open"])
        concordance = (
            self.df.groupby("timestamp")["direction"]
            .value_counts(normalize=True)
            .unstack()
            .fillna(0)
        )

        self.logger.debug(
            "Market concordance calculation finished",
            extra={"timesteps_analyzed": len(concordance)},
        )
        return concordance

    # --- Visualization Helpers ---

    def plot_coverage_heatmap(self, coverage_df):
        self.logger.info("Generating coverage heatmap visualization")
        # Pivot for heatmap: rows=symbols, columns=dates (simplified to month/year for display)
        plt.figure(figsize=(12, 6))
        sns.heatmap(coverage_df.pivot("symbol", "date", "bar_count"), cmap="RdYlGn")
        plt.title("Trading Bar Coverage Heatmap")
        plt.show()

    def run_all_checks_for_symbol(
        self, symbol: str, start_date: datetime, end_date: datetime
    ):
        """
        Runs all validation functions for a single symbol.
        Returns a dictionary containing all reports/results.
        """
        self.logger.info(f"Running full validation pipeline for symbol: {symbol}")

        # Filter to just this symbol
        df_symbol = self.df[self.df["symbol"] == symbol].copy()
        if df_symbol.empty:
            self.logger.warning(f"No data found for symbol: {symbol}")
            return {}

        # Temporarily replace self.df with filtered data
        original_df = self.df
        self.df = df_symbol

        try:
            # 1. Temporal coverage
            coverage_report = self.check_temporal_coverage(
                start_date=start_date, end_date=end_date
            )

            # 2. Intraday completeness
            intraday_report = self.analyze_intraday_completeness()

            # 3. OHLC integrity
            ohlc_errors = self.validate_ohlc_integrity()

            # 4. Timestamp integrity
            timestamp_report = self.check_timestamp_integrity()

            # 5. Price anomalies
            price_anomalies = self.detect_price_anomalies()

            # 6. Volume outliers
            volume_outliers = self.detect_volume_outliers()

            # 7. Market concordance (optional if comparing multiple symbols, will still run)
            concordance = self.calculate_market_concordance()

            results = {
                "symbol": symbol,
                "temporal_coverage": coverage_report,
                "intraday_completeness": intraday_report,
                "ohlc_errors": ohlc_errors,
                "timestamp_integrity": timestamp_report,
                "price_anomalies": price_anomalies,
                "volume_outliers": volume_outliers,
                "market_concordance": concordance,
            }

            self.logger.info(f"Completed full validation for {symbol}")

            return results

        finally:
            # Restore original dataframe
            self.df = original_df
