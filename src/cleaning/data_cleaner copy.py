# import pandas as pd
# import numpy as np
# import pandas_market_calendars as mcal
# from scipy.stats.mstats import winsorize
# from src.utils.app_logger import AppLogger
# from src.config import AppConfig
# from typing import Optional
# import sys


# class MarketDataCleaner:
#     def __init__(
#         self, symbol: str, logger: AppLogger, config: AppConfig, timeframe: str = "1min"
#     ):
#         self.symbol = symbol
#         self.logger = logger
#         self.config = config
#         self.timeframe = timeframe

#         self.exchange = self.config.data_cleaning.exchange
#         self.ffill_limit = self.config.data_cleaning.ffill_limit

#         self.outlier_threshold = self.config.data_cleaning.outlier_threshold

#         self.calendar = mcal.get_calendar(self.exchange)
#         self.quality_report: dict = {}
#         self.freq_alias = self._map_timeframe_to_freq(timeframe)

#     def _map_timeframe_to_freq(self, tf: str) -> str:
#         """Helper to map config timeframe strings to pandas aliases."""
#         mapping = self.config.global_settings.resample_map
#         if tf not in mapping:
#             self.logger.warning(
#                 f"Timeframe {tf} not recognized in resample_map. Defaulting to 1min."
#             )
#         return mapping.get(tf, "1min")

#     def normalize_symbol_and_timestamp(
#         self,
#         df: pd.DataFrame,
#         drop_symbol: bool = True,
#         timestamp_col: str = "timestamp",
#     ) -> pd.DataFrame:
#         """
#         Normalize market data DataFrame by:
#         - Dropping the 'symbol' column (if present)
#         - Converting DatetimeIndex to a 'timestamp' column

#         Args:
#             df (pd.DataFrame): Input DataFrame
#             drop_symbol (bool): Whether to drop 'symbol' column
#             timestamp_col (str): Name of timestamp column to create

#         Returns:
#             pd.DataFrame: Normalized DataFrame
#         """
#         if df is None or df.empty:
#             raise ValueError("Input DataFrame is empty or None.")

#         df = df.copy()

#         # Drop symbol column if present
#         if drop_symbol and "symbol" in df.columns:
#             df.drop(columns=["symbol"], inplace=True)

#         # Move DatetimeIndex to timestamp column
#         if isinstance(df.index, pd.DatetimeIndex):
#             df[timestamp_col] = df.index
#             df.reset_index(drop=True, inplace=True)
#         elif timestamp_col not in df.columns:
#             raise ValueError(
#                 "DataFrame must contain a DatetimeIndex or a 'timestamp' column."
#             )

#         # Final validation
#         if timestamp_col not in df.columns:
#             raise ValueError("Failed to create timestamp column.")

#         return df

#     def dataframe_report(
#         self,
#         df: pd.DataFrame,
#         name: Optional[str] = None,
#     ):
#         """
#         Print logistics about a DataFrame (typically merged parquet data).

#         Args:
#             df (pd.DataFrame): DataFrame to inspect.
#             name (str, optional): Identifier for logging (e.g. symbol_timeframe).
#         """
#         if df is None or df.empty:
#             self.logger.info("DataFrame is empty or None.")
#             return

#         n_rows, n_cols = df.shape
#         mem_usage = df.memory_usage(deep=True).sum() / (1024**2)  # MB

#         self.logger.info(f"DataFrame Report: {name or 'Unnamed'}")
#         self.logger.info(f"  Rows: {n_rows}")
#         self.logger.info(f"  Columns: {n_cols} -> {list(df.columns)}")
#         self.logger.info(f"  Index type: {type(df.index)}")
#         self.logger.info(f"  Memory usage: {mem_usage:.2f} MB")

#         # Timestamp handling (index-first, column fallback)
#         if isinstance(df.index, pd.DatetimeIndex):
#             self.logger.info(f"  First timestamp (index): {df.index.min()}")
#             self.logger.info(f"  Last timestamp  (index): {df.index.max()}")
#         elif "timestamp" in df.columns:
#             ts = pd.to_datetime(df["timestamp"], errors="coerce").dropna()
#             if not ts.empty:
#                 self.logger.info(f"  First timestamp (column): {ts.min()}")
#                 self.logger.info(f"  Last timestamp  (column): {ts.max()}")
#             else:
#                 self.logger.info("  Timestamp column present but all values invalid.")

#         self.logger.info(f"First 5 rows:\n{df.head()}")

#         self.logger.info("-" * 100)

#     def clean_pipeline(self, df: pd.DataFrame) -> pd.DataFrame:
#         """
#         Executes the full cleaning suite on a raw DataFrame.
#         Expected index: DatetimeIndex (UTC)
#         """
#         self.logger.info(f"Starting cleaning pipeline for {self.symbol}...")
#         self.quality_report = {}  # Reset report

#         df = self.normalize_symbol_and_timestamp(df)

#         self.dataframe_report(df, name=self.symbol)

#         # _check_ohlcv_integrity
#         df = self._check_ohlcv_integrity(df, symbol=self.symbol)

#         sys.exit(0)

#         # 1. Structural Check
#         df = self._remove_duplicates(df)

#         # 2. Market Hour Alignment (Gap Detection)
#         df = self._align_to_market_hours(df)

#         # 3. Value Validation (Logic & Zero Checks)
#         df = self._validate_ohlcv_logic(df)

#         # 4. Outlier Handling
#         df = self._handle_outliers(df, threshold=self.outlier_threshold)

#         # 5. Statistical Sanity Check
#         self._perform_sanity_checks(df)

#         self.logger.info("Cleaning complete.")
#         return df

#     def _check_ohlcv_integrity(
#         self, df: pd.DataFrame, symbol: str = "UNKNOWN"
#     ) -> pd.DataFrame:
#         """
#         Perform data integrity & sanity checks on an OHLCV DataFrame.

#         Expected columns: ['timestamp', 'open', 'high', 'low', 'close', 'volume']
#         Returns: DataFrame with issues flagged, self.logger.infos summary of problems.
#         """

#         df = df.copy()

#         # Ensure 'timestamp' exists and is datetime
#         if "timestamp" not in df.columns:
#             raise ValueError("DataFrame must contain a 'timestamp' column.")

#         df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
#         df.sort_values("timestamp", inplace=True)
#         df.reset_index(drop=True, inplace=True)

#         issues = pd.DataFrame(index=df.index)
#         issues["missing_data"] = df.isna().any(axis=1)
#         issues["duplicate_row"] = df.duplicated(keep=False)

#         # Negative prices or zero volume
#         issues["invalid_prices"] = (df[["open", "high", "low", "close"]] < 0).any(
#             axis=1
#         )
#         issues["invalid_volume"] = df["volume"] < 0

#         # Price consistency: low <= open/close <= high
#         issues["low_gt_high"] = df["low"] > df["high"]
#         issues["open_out_of_bounds"] = ~df["open"].between(df["low"], df["high"])
#         issues["close_out_of_bounds"] = ~df["close"].between(df["low"], df["high"])

#         # Missing timestamps (detect gaps larger than expected interval)
#         df["timestamp_diff"] = df["timestamp"].diff()
#         # Estimate expected frequency
#         median_diff = df["timestamp_diff"].median()
#         issues["timestamp_gap"] = df["timestamp_diff"] > 1.5 * median_diff

#         # self.logger.info summary
#         self.logger.info(f"Data Integrity Report for {symbol}:")
#         for col in issues.columns:
#             n_issues = issues[col].sum()
#             if n_issues > 0:
#                 self.logger.info(f"  {col}: {n_issues} rows affected")

#         # Optionally, return all problematic rows
#         problematic_rows = df[issues.any(axis=1)]

#         self.logger.info(f"_check_ohlcv_integrity - first 5 rows:\n{df.head()}")

#         return problematic_rows

#     def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
#         original_len = len(df)

#         # AGGRESSIVE LOGGING: Identify duplicates before dropping
#         duplicates = df[df.index.duplicated(keep="first")]
#         if not duplicates.empty:
#             self.logger.info(
#                 f"First 5 duplicate timestamps found: {duplicates.index[:5].tolist()}"
#             )

#         df = df[~df.index.duplicated(keep="first")]
#         removed_count = original_len - len(df)
#         self.quality_report["duplicates_removed"] = removed_count

#         if removed_count > 0:
#             self.logger.warning(f"Removed {removed_count} total duplicate timestamps.")
#         return df

#     def _align_to_market_hours(self, df: pd.DataFrame) -> pd.DataFrame:
#         # Determine schedule based on the data's range
#         if df.empty:
#             self.logger.warning("Dataframe is empty, skipping alignment.")
#             return df

#         schedule = self.calendar.schedule(
#             start_date=df.index.min(), end_date=df.index.max()
#         )

#         if self.timeframe == "1D":
#             expected_index = pd.DatetimeIndex(
#                 mcal.date_range(schedule, frequency="1D")
#             ).normalize()

#             df.index = pd.DatetimeIndex(df.index).normalize()
#         else:
#             expected_index = pd.DatetimeIndex(
#                 mcal.date_range(schedule, frequency=self.freq_alias)
#             )

#         missing_timestamps = expected_index.difference(pd.DatetimeIndex(df.index))

#         missing_count = len(missing_timestamps)
#         self.quality_report["missing_bars_filled"] = missing_count

#         if missing_count > 0:
#             self.logger.info(f"Detected {missing_count} missing bars. Imputing...")
#             self.logger.info(
#                 f"First 5 missing intervals: {missing_timestamps[:5].tolist()}"
#             )

#         df = df.reindex(expected_index)

#         df[["open", "high", "low", "close"]] = df[
#             ["open", "high", "low", "close"]
#         ].ffill(limit=self.ffill_limit)
#         df["volume"] = df["volume"].fillna(0)

#         if df["close"].isna().all():
#             self.logger.error(
#                 "CRITICAL: Reindexing dropped ALL data. Check timestamp alignment."
#             )
#             return df.dropna(how="all")

#         nan_rows = df[df["close"].isna()]
#         dropped_nans = len(nan_rows)

#         if dropped_nans > 0:
#             self.logger.info(
#                 f"First 5 rows dropped (unfillable NaNs): {nan_rows.index[:5].tolist()}"
#             )

#             df.dropna(subset=["close"], inplace=True)
#             self.quality_report["dropped_unfillable_nans"] = int(dropped_nans)

#         return df

#     def _validate_ohlcv_logic(self, df):
#         # AGGRESSIVE LOGGING: Check for Logic Errors (High < Low, etc) before fixing
#         # Logic: High must be >= Low, Open, Close. Low must be <= Open, Close.
#         bad_logic_mask = (
#             (df["high"] < df["low"])
#             | (df["high"] < df["open"])
#             | (df["high"] < df["close"])
#             | (df["low"] > df["open"])
#             | (df["low"] > df["close"])
#         )
#         if bad_logic_mask.any():
#             self.logger.info(
#                 f"First 5 rows with invalid OHLC logic (fixed automatically): {df[bad_logic_mask].index[:5].tolist()}"
#             )

#         # 1. Ensure High is highest and Low is lowest (Fixing the errors detected above)
#         df["high"] = df[["open", "high", "low", "close"]].max(axis=1)
#         df["low"] = df[["open", "high", "low", "close"]].min(axis=1)

#         # 2. Non-positive price check
#         bad_price_mask = (df[["open", "high", "low", "close"]] <= 0).any(axis=1)
#         non_positive = bad_price_mask.sum()

#         if non_positive > 0:
#             # AGGRESSIVE LOGGING: self.logger.info first 5 negative price rows
#             self.logger.warning(
#                 f"Found {non_positive} rows with <= 0 prices. Removing."
#             )
#             self.logger.info(
#                 f"First 5 non-positive rows: {df[bad_price_mask].index[:5].tolist()}"
#             )

#             df = df[df["open"] > 0]  # Filter out bad rows
#             self.quality_report["non_positive_rows_removed"] = int(non_positive)

#         return df

#     def _handle_outliers(self, df, threshold=0.05):
#         if df.empty:
#             return df

#         # Identify extreme jumps (log returns > 5% in 1 minute)
#         # We use a shift to compare current Close to previous Close
#         log_ret = np.log(df["close"] / df["close"].shift(1))
#         outliers = log_ret.abs() > threshold
#         outlier_count = outliers.sum()

#         self.quality_report["extreme_spikes_detected"] = int(outlier_count)

#         if outlier_count > 0:
#             self.logger.info(
#                 f"Detected {outlier_count} extreme price spikes (> {threshold*100}%)."
#             )
#             # AGGRESSIVE LOGGING: self.logger.info first 5 spikes
#             spike_indices = log_ret[outliers].index[:5]
#             spike_values = log_ret[outliers].head(5).values
#             self.logger.info(f"First 5 spike timestamps: {spike_indices.tolist()}")
#             self.logger.info(f"First 5 spike values (log ret): {spike_values}")

#         # Winsorize: Cap the top/bottom 0.1% of prices to reduce impact of "flash crashes"
#         # Note: We winsorize the raw prices here for stability, though usually winsorizing returns is preferred.
#         # For simplicity in this cleaning pipeline, we are capping the specific values.
#         # df["close"] = winsorize(df["close"], limits=[0.001, 0.001])
#         # df["open"] = winsorize(df["open"], limits=[0.001, 0.001])
#         # df["high"] = winsorize(df["high"], limits=[0.001, 0.001])
#         # df["low"] = winsorize(df["low"], limits=[0.001, 0.001])

#         return df

#     def _perform_sanity_checks(self, df):
#         # Zero Variance Check (Dead Instrument)
#         if df["close"].std() == 0:
#             self.logger.error(
#                 f"CRITICAL: Zero variance detected for {self.symbol}. Data may be flatlined."
#             )
#             self.quality_report["status"] = "CRITICAL_ZERO_VARIANCE"
#         else:
#             self.quality_report["status"] = "PASS"

#     def resample_data(self, df, timeframe="5min"):
#         """
#         Aggregates 1-min data into higher timeframes.
#         """
#         logic = {
#             "open": "first",
#             "high": "max",
#             "low": "min",
#             "close": "last",
#             "volume": "sum",
#         }
#         # Resample and drop rows that don't have enough data (e.g. incomplete last bar)
#         return df.resample(timeframe).agg(logic).dropna()

#     def get_report(self):
#         return self.quality_report
