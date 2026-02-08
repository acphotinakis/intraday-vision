import pandas as pd
import numpy as np
import pandas_market_calendars as mcal
from scipy.stats.mstats import winsorize
from src.utils.app_logger import AppLogger
from src.config import AppConfig
from typing import Optional
import sys
from src.cleaning.utils_data import *


class MarketDataCleaner:
    def __init__(
        self, symbol: str, logger: AppLogger, config: AppConfig, timeframe: str = "1min"
    ):
        self.symbol = symbol
        self.logger = logger
        self.config = config
        self.timeframe = timeframe

        self.exchange = self.config.data_cleaning.exchange
        self.ffill_limit = self.config.data_cleaning.ffill_limit

        self.outlier_threshold = self.config.data_cleaning.outlier_threshold

        self.calendar = mcal.get_calendar(self.exchange)
        self.quality_report: dict = {}
        self.freq_alias = self._map_timeframe_to_freq(timeframe)

    def _map_timeframe_to_freq(self, tf: str) -> str:
        """Helper to map config timeframe strings to pandas aliases."""
        mapping = self.config.global_settings.resample_map
        if tf not in mapping:
            self.logger.warning(
                f"Timeframe {tf} not recognized in resample_map. Defaulting to 1min."
            )
        return mapping.get(tf, "1min")

    def dataframe_report(
        self,
        df: pd.DataFrame,
        name: Optional[str] = None,
    ):
        """
        Print logistics about a DataFrame (typically merged parquet data).

        Args:
            df (pd.DataFrame): DataFrame to inspect.
            name (str, optional): Identifier for logging (e.g. symbol_timeframe).
        """
        if df is None or df.empty:
            self.logger.info("DataFrame is empty or None.")
            return

        n_rows, n_cols = df.shape
        mem_usage = df.memory_usage(deep=True).sum() / (1024**2)  # MB

        self.logger.info(f"DataFrame Report: {name or 'Unnamed'}")
        self.logger.info(f"  Rows: {n_rows}")
        self.logger.info(f"  Columns: {n_cols} -> {list(df.columns)}")
        self.logger.info(f"  Index type: {type(df.index)}")
        self.logger.info(f"  Memory usage: {mem_usage:.2f} MB")

        # Timestamp handling (index-first, column fallback)
        if isinstance(df.index, pd.DatetimeIndex):
            self.logger.info(f"  First timestamp (index): {df.index.min()}")
            self.logger.info(f"  Last timestamp  (index): {df.index.max()}")
        elif "timestamp" in df.columns:
            ts = pd.to_datetime(df["timestamp"], errors="coerce").dropna()
            if not ts.empty:
                self.logger.info(f"  First timestamp (column): {ts.min()}")
                self.logger.info(f"  Last timestamp  (column): {ts.max()}")
            else:
                self.logger.info("  Timestamp column present but all values invalid.")

        self.logger.info(f"First 5 rows:\n{df.head()}")

        self.logger.info("-" * 100)

    def clean_pipeline(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Executes the full cleaning suite on a raw DataFrame.
        Expected index: DatetimeIndex (UTC)
        """
        self.logger.info(f"Starting cleaning pipeline for {self.symbol}...")
        self.quality_report = {}  # Reset report

        df = normalize_symbol_and_timestamp(df)

        self.dataframe_report(df, name=self.symbol)

        # _check_ohlcv_integrity
        df = self._check_ohlcv_integrity(df, symbol=self.symbol)

        sys.exit(0)

        # 1. Structural Check
        df = self._remove_duplicates(df)

        self.logger.info("Cleaning complete.")
        return df

    def _check_ohlcv_integrity(
        self, df: pd.DataFrame, symbol: str = "UNKNOWN"
    ) -> pd.DataFrame:
        """
        Perform data integrity & sanity checks on an OHLCV DataFrame.
        """

        df = df.copy()

        if "timestamp" not in df.columns:
            raise ValueError("DataFrame must contain a 'timestamp' column.")

        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df.sort_values("timestamp", inplace=True)
        df.reset_index(drop=True, inplace=True)

        issues = pd.DataFrame(index=df.index)

        issues["missing_data"] = df.isna().any(axis=1)
        issues["duplicate_row"] = df.duplicated(keep=False)
        issues["invalid_prices"] = (df[["open", "high", "low", "close"]] < 0).any(
            axis=1
        )
        issues["invalid_volume"] = df["volume"] < 0
        issues["low_gt_high"] = df["low"] > df["high"]
        issues["open_out_of_bounds"] = ~df["open"].between(df["low"], df["high"])
        issues["close_out_of_bounds"] = ~df["close"].between(df["low"], df["high"])

        df["timestamp_diff"] = df["timestamp"].diff()
        median_diff = df["timestamp_diff"].median()
        issues["timestamp_gap"] = df["timestamp_diff"] > 1.5 * median_diff

        # Logging summary
        self.logger.info(f"Data Integrity Report for {symbol}:")
        for col in issues.columns:
            count = issues[col].sum()
            if count > 0:
                self.logger.info(f"  {col}: {count} rows")

        # Split clean vs corrupt
        corrupt_mask = issues.any(axis=1)
        df_corrupt = df[corrupt_mask]
        df_clean = df[~corrupt_mask]

        # Save corrupt rows
        save_corrupt_rows(
            df_corrupt,
            reason="ohlcv_integrity",
            config=self.config,
            symbol=symbol,
            timeframe=self.timeframe,
            logger=self.logger,
        )

        self.quality_report["corrupt_rows"] = len(df_corrupt)

        return df_clean

    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        original_len = len(df)

        # AGGRESSIVE LOGGING: Identify duplicates before dropping
        duplicates = df[df.index.duplicated(keep="first")]
        if not duplicates.empty:
            self.logger.info(
                f"First 5 duplicate timestamps found: {duplicates.index[:5].tolist()}"
            )

        df = df[~df.index.duplicated(keep="first")]
        removed_count = original_len - len(df)
        self.quality_report["duplicates_removed"] = removed_count

        if removed_count > 0:
            self.logger.warning(f"Removed {removed_count} total duplicate timestamps.")
        return df

    def get_report(self):
        return self.quality_report
