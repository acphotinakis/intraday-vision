import pandas as pd
import numpy as np
import pandas_market_calendars as mcal
from pathlib import Path
from src.utils.app_logger import AppLogger
from src.config import AppConfig


class MarketDataQualityAssessment:
    def __init__(
        self,
        df: pd.DataFrame,
        symbols: list[str],
        config: AppConfig,
        logger: AppLogger,
    ):
        self.df = df
        self.symbols = symbols
        self.config = config
        self.logger = logger

        self.logger.info(
            f"Initializing MarketDataQualityAssessment with {len(df)} rows and {len(symbols)} symbols."
        )

        # # Ensure timestamp is datetime and sorted
        # if not pd.api.types.is_datetime64_any_dtype(self.df["timestamp"]):
        #     self.logger.info("Converting 'timestamp' column to datetime with UTC")
        #     self.df["timestamp"] = pd.to_datetime(self.df["timestamp"], utc=True)

        # self.df = self.df.sort_values("timestamp")
        # self.logger.info(
        #     f"Data sorted by timestamp, range: {self.df['timestamp'].min()} to {self.df['timestamp'].max()}"
        # )

        # # Type-safe date column
        # self.df["date"] = pd.DatetimeIndex(self.df["timestamp"]).floor("D")
        # self.logger.info(
        #     f"Added 'date' column. Sample dates: {self.df['date'].head(3).tolist()}"
        # )

    def run_full_assessment(self) -> dict:
        self.logger.info("Starting Stage 1: Full Data Quality Assessment")
        self.logger.info(
            f"Data contains {len(self.df)} rows, columns: {self.df.columns.tolist()}"
        )

        results = {
            "temporal_coverage": self._check_temporal_coverage(),
            "intraday_completeness": self._analyze_intraday_completeness(),
            "ohlc_integrity": self._validate_ohlc_integrity(),
            "timestamp_integrity": self._check_timestamp_integrity(),
            "price_anomalies": self._detect_price_anomalies(),
            "volume_stats": self._analyze_volume(),
            "market_concordance": self._analyze_market_concordance(),
            "synchronized_availability": self._analyze_synchronized_availability(),
        }

        self.logger.info("Stage 1 Assessment completed.")
        return results

    # --- 1.2 Coverage and Completeness ---

    def _check_temporal_coverage(self):
        self.logger.info("Running 1.2.1 Temporal Coverage Verification")

        start_date = pd.Timestamp(self.df["timestamp"].min()).date()
        end_date = pd.Timestamp(self.df["timestamp"].max()).date()
        self.logger.info(f"Temporal coverage: start={start_date}, end={end_date}")

        nyse = mcal.get_calendar("NYSE")
        schedule = nyse.schedule(start_date=start_date, end_date=end_date)
        expected_days = len(schedule)
        expected_dates = set(pd.DatetimeIndex(schedule.index).date)
        self.logger.info(f"Expected trading days: {expected_days}")

        coverage_stats = []
        for symbol in self.symbols:
            sym_df = self.df[self.df["symbol"] == symbol]
            actual_dates = set(sym_df["date"].unique())
            missing_dates = expected_dates - actual_dates
            coverage_pct = len(actual_dates) / expected_days if expected_days > 0 else 0

            self.logger.info(
                f"Symbol {symbol}: actual_days={len(actual_dates)}, missing_days={len(missing_dates)}, coverage={coverage_pct:.2%}"
            )

            coverage_stats.append(
                {
                    "symbol": symbol,
                    "expected_days": expected_days,
                    "actual_days": len(actual_dates),
                    "coverage_pct": coverage_pct,
                    "missing_dates": sorted(list(missing_dates)),
                }
            )

        df_stats = pd.DataFrame(coverage_stats)
        self.logger.info(f"Temporal coverage DataFrame shape: {df_stats.shape}")
        return df_stats

    def _analyze_intraday_completeness(self):
        self.logger.info("Running 1.2.2 Intraday Completeness Analysis")

        daily_counts = (
            self.df.groupby(["symbol", "date"]).size().reset_index(name="bar_count")
        )
        self.logger.info(f"Grouped bars count per symbol/day:")
        self.logger.info(f"{daily_counts.head(len(self.symbols))}")

        daily_counts["completeness_score"] = daily_counts["bar_count"] / 390.0

        conditions = [
            (daily_counts["bar_count"] == 390),
            (daily_counts["bar_count"] == 210),
            (daily_counts["bar_count"] < 390),
            (daily_counts["bar_count"] > 390),
        ]
        choices = [
            "Normal",
            "Early Close",
            "Investigate (Missing)",
            "Investigate (Extra)",
        ]
        daily_counts["status"] = np.select(conditions, choices, default="Unknown")

        self.logger.info(
            f"Intraday completeness sample:"
        )
        self.logger.info(f"{daily_counts[['symbol', 'date', 'bar_count', 'status']].head(len(self.symbols))}")
        return daily_counts

    def _validate_ohlc_integrity(self):
        self.logger.info("Running 1.3.1 OHLC Integrity Check")
        total_bars = len(self.df)
        self.logger.info(f"Total bars to check: {total_bars}")

        violations = {
            "High < Close": (self.df["high"] < self.df["close"]),
            "Low > Open": (self.df["low"] > self.df["open"]),
            "High < Low": (self.df["high"] < self.df["low"]),
            "High < Open": (self.df["high"] < self.df["open"]),
            "Low > Close": (self.df["low"] > self.df["close"]),
            "Negative Price": (self.df[["open", "high", "low", "close"]] <= 0).any(
                axis=1
            ),
            "Negative Volume": (self.df["volume"] < 0),
        }

        summary = []
        for v_type, mask in violations.items():
            count = mask.sum()
            pct_total = (count / total_bars) * 100
            self.logger.info(
                f"Violation '{v_type}': count={count}, pct={pct_total:.2f}%"
            )
            summary.append(
                {"violation_type": v_type, "count": count, "pct_total": pct_total}
            )

        return pd.DataFrame(summary)

    def _check_timestamp_integrity(self):
        self.logger.info("Running 1.3.2 Timestamp Integrity Check")

        duplicates = self.df[
            self.df.duplicated(subset=["symbol", "timestamp"], keep=False)
        ]
        self.logger.info(f"Found {len(duplicates)} duplicate rows")

        if not duplicates.empty:
            self.logger.error(f"Duplicate timestamps detected: {len(duplicates)} rows")
            return (
                duplicates.groupby(["symbol", "timestamp"])
                .size()
                .reset_index(name="count")
            )
        return pd.DataFrame()

    def _detect_price_anomalies(self):
        self.logger.info("Running 1.4.1 Price Anomaly Detection")

        stats = self.df.groupby("symbol")["close"].agg(["min", "max"]).reset_index()
        stats["range"] = stats["max"] - stats["min"]
        self.logger.info(f"Price range per symbol:")
        self.logger.info(f"{stats.head(len(self.symbols))}")

        daily_closes = self.df.groupby(["symbol", "date"])["close"].last().reset_index()
        daily_closes["pct_change"] = daily_closes.groupby("symbol")[
            "close"
        ].pct_change()
        suspicious_shifts = daily_closes[abs(daily_closes["pct_change"]) > 0.5]
        self.logger.info(
            f"Suspicious daily shifts (>50%): {len(suspicious_shifts)} rows"
        )

        return {"price_ranges": stats, "suspicious_shifts": suspicious_shifts}

    def _analyze_volume(self):
        self.logger.info("Running 1.4.2 Volume Sanity Check")

        stats = (
            self.df.groupby("symbol")
            .agg(
                mean_volume=("volume", "mean"),
                std_volume=("volume", "std"),
                max_volume=("volume", "max"),
                zero_volume_count=("volume", lambda x: (x == 0).sum()),
            )
            .reset_index()
        )
        stats["pct_zero_volume"] = (
            stats["zero_volume_count"] / self.df.groupby("symbol").size().values
        ) * 100
        self.logger.info(f"Volume stats sample:")
        self.logger.info(f"{stats.head(len(self.symbols))}")

        return stats

    def _analyze_synchronized_availability(self):
        self.logger.info("Running 1.5.1 Synchronized Availability Analysis")

        daily_presence = (
            self.df.groupby(["date", "symbol"]).size().unstack(fill_value=0)
        )
        daily_presence = (daily_presence > 0).astype(int)
        universe_size = len(self.symbols)
        daily_presence["available_symbols"] = daily_presence.sum(axis=1)
        daily_presence["coverage_pct"] = (
            daily_presence["available_symbols"] / universe_size
        ) * 100

        self.logger.info(f"Synchronized availability head:")
        self.logger.info(f"{daily_presence.head(len(self.symbols))}")

        return daily_presence[["available_symbols", "coverage_pct"]]

    def _analyze_market_concordance(self):
        self.logger.info("Running 1.5.2 Market Concordance")

        pivot_close = self.df.pivot_table(
            index="timestamp", columns="symbol", values="close"
        )
        returns = pivot_close.pct_change()
        dirs = np.sign(returns)
        dirs_df = pd.DataFrame(dirs, index=returns.index, columns=returns.columns)

        concordance = pd.DataFrame()
        concordance["pct_up"] = (dirs_df == 1).sum(axis=1) / dirs_df.notna().sum(axis=1)
        concordance["pct_down"] = (dirs_df == -1).sum(axis=1) / dirs_df.notna().sum(
            axis=1
        )
        concordance["pct_flat"] = (dirs_df == 0).sum(axis=1) / dirs_df.notna().sum(
            axis=1
        )

        concordance.index = pd.to_datetime(concordance.index)
        concordance["time"] = pd.DatetimeIndex(concordance.index).time

        concordance_agg = concordance.groupby("time").mean()
        self.logger.info(f"Market concordance aggregated by time sample:")
        self.logger.info(f"{concordance_agg.head(len(self.symbols))}")

        return concordance_agg
