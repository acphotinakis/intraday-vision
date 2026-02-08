import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas_market_calendars as mcal
from pathlib import Path
from datetime import datetime
from src.utils.app_logger import *
from src.config import AppConfig
from src.data_investigation.market_validator import MarketDataValidator
import re
import os
import sys


class DataInvestigationPipeline:
    def __init__(self, logger: AppLogger, config: AppConfig):
        self.config = config
        self.logger = logger
        self.output_dir = Path(self.config.project.data_investigation)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.timeframes = self.config.global_settings.timeframes
        self.symbols = self.config.global_settings.symbols

        # Store processed reports
        self.all_reports = {}

    # ────────────── Core Symbol Processing ──────────────
    def process_all_symbols(self):
        for symbol in self.symbols:
            self.logger.info(f"=== Processing symbol: {symbol} ===")
            self.all_reports[symbol] = {}

            for timeframe in self.timeframes:
                self.logger.info(f"Processing timeframe: {timeframe}")

                results = self.process_symbol_timeframe(symbol, timeframe)
                if results:
                    self.all_reports[symbol][timeframe] = results

        self.logger.info("All symbols processed!")

    def process_symbol_timeframe(self, symbol: str, timeframe: str):
        raw_dir = Path(self.config.dirs["raw_data"]) / symbol / timeframe
        if not raw_dir.exists():
            self.logger.error(
                f"No data directory for {symbol} / {timeframe}: {raw_dir}"
            )
            return None

        df = self._load_and_merge_timeframe_files(raw_dir)
        if df.empty:
            self.logger.warning(f"No data found for {symbol} / {timeframe}")
            return None

        # Initialize validator
        validator = MarketDataValidator(
            df=df,
            config=self.config,
            logger=self.logger,
            plots_dir=Path(self.config.project.data_investigation),
        )

        # Determine date range for temporal coverage
        start_date = df["timestamp"].min().date()
        end_date = df["timestamp"].max().date()

        # Run all validation checks
        results = validator.run_all_checks_for_symbol(
            symbol,
            start_date=start_date,
            end_date=end_date,
        )

        # ────────────── Save Reports ──────────────
        out_symbol_dir = self.output_dir / symbol / timeframe
        out_symbol_dir.mkdir(parents=True, exist_ok=True)

        # Save Data Overview
        df.reset_index().to_csv(out_symbol_dir / "merged_data.csv", index=False)
        self.logger.info(f"Merged data CSV saved: {out_symbol_dir / 'merged_data.csv'}")

        # Save validation reports
        for key, report in results.items():
            if isinstance(report, pd.DataFrame) and not report.empty:
                report.to_csv(out_symbol_dir / f"{key}.csv", index=False)
                self.logger.info(f"{key} report saved: {out_symbol_dir / f'{key}.csv'}")

        # ────────────── Generate Plots ──────────────
        # 1. Temporal Coverage Heatmap
        if "temporal_coverage" in results and not results["temporal_coverage"].empty:
            plt.figure(figsize=(12, 6))
            cov_df = results["temporal_coverage"].copy()
            cov_df["date"] = cov_df.get("date", [start_date] * len(cov_df))
            sns.heatmap(
                cov_df.pivot(index="symbol", columns="date", values="coverage").fillna(
                    0
                ),
                cmap="RdYlGn",
            )
            plt.title(f"{symbol} Temporal Coverage Heatmap")
            plt.tight_layout()
            plt.savefig(out_symbol_dir / "coverage_heatmap.png")
            plt.close()
            self.logger.info(f"Coverage heatmap saved")

        # 2. Intraday Bar Count Distribution
        if (
            "intraday_completeness" in results
            and not results["intraday_completeness"].empty
        ):
            plt.figure(figsize=(12, 6))
            sns.histplot(
                results["intraday_completeness"]["bar_count"],
                bins=50,
                kde=False,
                color="skyblue",
            )
            plt.title(f"{symbol} Intraday Bar Count Distribution")
            plt.xlabel("Bars per Day")
            plt.ylabel("Frequency")
            plt.tight_layout()
            plt.savefig(out_symbol_dir / "intraday_bar_count.png")
            plt.close()
            self.logger.info("Intraday bar count plot saved")

        # 3. Price Anomalies Over Time
        if "price_anomalies" in results and not results["price_anomalies"].empty:
            anomalies = results["price_anomalies"].copy()
            anomalies.reset_index(inplace=True)
            plt.figure(figsize=(12, 6))
            plt.plot(df.index, df["close"], label="Close Price", alpha=0.7)
            plt.scatter(
                anomalies["timestamp"],
                anomalies["close"],
                color="red",
                label="Anomalies",
            )
            plt.title(f"{symbol} Price Anomalies Over Time")
            plt.xlabel("Time")
            plt.ylabel("Price")
            plt.legend()
            plt.tight_layout()
            plt.savefig(out_symbol_dir / "price_anomalies.png")
            plt.close()
            self.logger.info("Price anomalies plot saved")

        # 4. Volume Spikes Over Time
        if "volume_outliers" in results and not results["volume_outliers"].empty:
            volume_out = results["volume_outliers"].copy()
            volume_out.reset_index(inplace=True)
            plt.figure(figsize=(12, 6))
            plt.plot(df.index, df["volume"], label="Volume", alpha=0.7)
            plt.scatter(
                volume_out["timestamp"],
                volume_out["volume"],
                color="orange",
                label="Volume Spikes",
            )
            plt.title(f"{symbol} Volume Spikes Over Time")
            plt.xlabel("Time")
            plt.ylabel("Volume")
            plt.legend()
            plt.tight_layout()
            plt.savefig(out_symbol_dir / "volume_spikes.png")
            plt.close()
            self.logger.info("Volume spikes plot saved")

        self.logger.info(f"Completed processing {symbol} / {timeframe}")
        return results

    # ────────────── Helper Functions ──────────────
    def _load_and_merge_timeframe_files(self, raw_dir: Path) -> pd.DataFrame:
        """
        Load all parquet files in a timeframe directory into a single DataFrame.
        Fixes timestamp column (ms → datetime) and drops unnecessary index columns.
        Prints the first 10 rows of the merged data.
        """
        messages = []
        block_msg_header = "LOADING AND MERGING TIMEFRAME FILES"

        def extract_start_date(file_path: Path):
            match = re.search(r"_(\d{4}-\d{2}-\d{2})_to_", file_path.name)
            if match:
                return pd.to_datetime(match.group(1))
            return pd.Timestamp.min

        files = list(raw_dir.glob("*.parquet"))
        if not files:
            return pd.DataFrame()

        files.sort(key=extract_start_date)

        messages.append(f"Merging {len(files)} parquet files from {raw_dir}")

        dfs = []
        for file_path in files:
            try:
                df = pd.read_parquet(file_path)

                # Drop unwanted index columns from previous exports
                df = df.drop(
                    columns=[c for c in df.columns if "__index" in c], errors="ignore"
                )

                # Ensure timestamp exists
                if "timestamp" not in df.columns:
                    raise ValueError(f"No 'timestamp' column in {file_path.name}")

                # Convert milliseconds → UTC datetime
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
                messages.append("Convert milliseconds → UTC datetime")
                dfs.append(df)
            except Exception as e:
                messages.append(f"Failed reading {file_path.name}: {e}")

        if not dfs:
            self.logger.warning(f"No valid data loaded from {raw_dir}")
            return pd.DataFrame()

        # Concatenate all dataframes
        df_all = pd.concat(dfs, ignore_index=True)

        # Sort by timestamp and drop duplicates
        df_all = df_all.sort_values("timestamp").loc[
            ~df_all["timestamp"].duplicated(keep="first")
        ]

        # Log first 10 rows
        self.logger.info(
            "First 10 rows of merged data:\n" + df_all.head(10).to_string()
        )

        messages.append("First 10 rows of merged data:\n" + df_all.head(10).to_string())

        messages.append(
            f"Merged DataFrame shape: {df_all.shape}, "
            f"time range: {df_all['timestamp'].min()} → {df_all['timestamp'].max()}"
        )

        self.logger.info_block(
            messages, block_msg_header, levels=["info", "warning", "error"]
        )

        return df_all


if __name__ == "__main__":
    config = AppConfig.load_from_yaml()

    logger = AppLogger(
        name="Data Processing",
        log_dir=config.dirs["logs"],
        filename="process_data.log",
        json_format=False,
    )

    logger.info_block(
        ["Step 1 completed", "Step 2 warning", "Step 3 failed"],
        header="PROCESS START",
        footer="PROCESS END",
        levels=["info", "warning", "error"],
        extra={"user": "Andrew", "task_id": 42},
    )

    sys.exit(0)

    data_processor = DataInvestigationPipeline(logger=logger, config=config)

    data_processor.process_all_symbols()

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas_market_calendars as mcal
# from pathlib import Path
# from datetime import datetime
# from src.utils.app_logger import *
# from src.config import AppConfig
# from src.data_investigation.market_validator import MarketDataValidator
# import re
# import os
# import sys


# class DataInvestigationPipeline:
#     def __init__(self, logger: AppLogger, config: AppConfig):
#         self.config = config
#         self.logger = logger
#         self.output_dir = Path(self.config.project.data_investigation)
#         self.output_dir.mkdir(parents=True, exist_ok=True)

#         self.timeframes = self.config.global_settings.timeframes
#         self.symbols = self.config.global_settings.symbols

#         # Store processed reports
#         self.all_reports = {}

#     # ────────────── Core Symbol Processing ──────────────
#     def process_all_symbols(self):
#         for symbol in self.symbols:
#             self.logger.info(f"=== Processing symbol: {symbol} ===")
#             self.all_reports[symbol] = {}

#             for timeframe in self.timeframes:
#                 self.logger.info(f"Processing timeframe: {timeframe}")

#                 results = self.process_symbol_timeframe(symbol, timeframe)
#                 if results:
#                     self.all_reports[symbol][timeframe] = results

#         self.logger.info("All symbols processed!")

#     def process_symbol_timeframe(self, symbol: str, timeframe: str):
#         raw_dir = Path(self.config.dirs["raw_data"]) / symbol / timeframe
#         if not raw_dir.exists():
#             self.logger.error(
#                 f"No data directory for {symbol} / {timeframe}: {raw_dir}"
#             )
#             return None

#         df = self._load_and_merge_timeframe_files(raw_dir)
#         if df.empty:
#             self.logger.warning(f"No data found for {symbol} / {timeframe}")
#             return None

#         # Initialize validator
#         validator = MarketDataValidator(
#             df=df,
#             config=self.config,
#             logger=self.logger,
#             plots_dir=Path(self.config.project.data_investigation),
#         )

#         # Determine date range for temporal coverage
#         start_date = df["timestamp"].min().date()
#         end_date = df["timestamp"].max().date()

#         # Run all validation checks
#         results = validator.run_all_checks_for_symbol(
#             symbol,
#             start_date=start_date,
#             end_date=end_date,
#         )

#         # ────────────── Save Reports ──────────────
#         out_symbol_dir = self.output_dir / symbol / timeframe
#         out_symbol_dir.mkdir(parents=True, exist_ok=True)

#         # Save Data Overview
#         df.reset_index().to_csv(out_symbol_dir / "merged_data.csv", index=False)
#         self.logger.info(f"Merged data CSV saved: {out_symbol_dir / 'merged_data.csv'}")

#         # Save validation reports
#         for key, report in results.items():
#             if isinstance(report, pd.DataFrame) and not report.empty:
#                 report.to_csv(out_symbol_dir / f"{key}.csv", index=False)
#                 self.logger.info(f"{key} report saved: {out_symbol_dir / f'{key}.csv'}")

#         # ────────────── Generate Plots ──────────────
#         # 1. Temporal Coverage Heatmap
#         if "temporal_coverage" in results and not results["temporal_coverage"].empty:
#             plt.figure(figsize=(12, 6))
#             cov_df = results["temporal_coverage"].copy()
#             cov_df["date"] = cov_df.get("date", [start_date] * len(cov_df))
#             sns.heatmap(
#                 cov_df.pivot(index="symbol", columns="date", values="coverage").fillna(
#                     0
#                 ),
#                 cmap="RdYlGn",
#             )
#             plt.title(f"{symbol} Temporal Coverage Heatmap")
#             plt.tight_layout()
#             plt.savefig(out_symbol_dir / "coverage_heatmap.png")
#             plt.close()
#             self.logger.info(f"Coverage heatmap saved")

#         # 2. Intraday Bar Count Distribution
#         if (
#             "intraday_completeness" in results
#             and not results["intraday_completeness"].empty
#         ):
#             plt.figure(figsize=(12, 6))
#             sns.histplot(
#                 results["intraday_completeness"]["bar_count"],
#                 bins=50,
#                 kde=False,
#                 color="skyblue",
#             )
#             plt.title(f"{symbol} Intraday Bar Count Distribution")
#             plt.xlabel("Bars per Day")
#             plt.ylabel("Frequency")
#             plt.tight_layout()
#             plt.savefig(out_symbol_dir / "intraday_bar_count.png")
#             plt.close()
#             self.logger.info("Intraday bar count plot saved")

#         # 3. Price Anomalies Over Time
#         if "price_anomalies" in results and not results["price_anomalies"].empty:
#             anomalies = results["price_anomalies"].copy()
#             anomalies.reset_index(inplace=True)
#             plt.figure(figsize=(12, 6))
#             plt.plot(df.index, df["close"], label="Close Price", alpha=0.7)
#             plt.scatter(
#                 anomalies["timestamp"],
#                 anomalies["close"],
#                 color="red",
#                 label="Anomalies",
#             )
#             plt.title(f"{symbol} Price Anomalies Over Time")
#             plt.xlabel("Time")
#             plt.ylabel("Price")
#             plt.legend()
#             plt.tight_layout()
#             plt.savefig(out_symbol_dir / "price_anomalies.png")
#             plt.close()
#             self.logger.info("Price anomalies plot saved")

#         # 4. Volume Spikes Over Time
#         if "volume_outliers" in results and not results["volume_outliers"].empty:
#             volume_out = results["volume_outliers"].copy()
#             volume_out.reset_index(inplace=True)
#             plt.figure(figsize=(12, 6))
#             plt.plot(df.index, df["volume"], label="Volume", alpha=0.7)
#             plt.scatter(
#                 volume_out["timestamp"],
#                 volume_out["volume"],
#                 color="orange",
#                 label="Volume Spikes",
#             )
#             plt.title(f"{symbol} Volume Spikes Over Time")
#             plt.xlabel("Time")
#             plt.ylabel("Volume")
#             plt.legend()
#             plt.tight_layout()
#             plt.savefig(out_symbol_dir / "volume_spikes.png")
#             plt.close()
#             self.logger.info("Volume spikes plot saved")

#         self.logger.info(f"Completed processing {symbol} / {timeframe}")
#         return results

#     # ────────────── Helper Functions ──────────────
#     def _load_and_merge_timeframe_files(self, raw_dir: Path) -> pd.DataFrame:
#         """
#         Load all parquet files in a timeframe directory into a single DataFrame.
#         Fixes timestamp column (ms → datetime) and drops unnecessary index columns.
#         Prints the first 10 rows of the merged data.
#         """
#         messages = []
#         block_msg_header = "LOADING AND MERGING TIMEFRAME FILES"

#         def extract_start_date(file_path: Path):
#             match = re.search(r"_(\d{4}-\d{2}-\d{2})_to_", file_path.name)
#             if match:
#                 return pd.to_datetime(match.group(1))
#             return pd.Timestamp.min

#         files = list(raw_dir.glob("*.parquet"))
#         if not files:
#             return pd.DataFrame()

#         files.sort(key=extract_start_date)

#         self.logger.info(f"Merging {len(files)} parquet files from {raw_dir}")

#         dfs = []
#         for file_path in files:
#             try:
#                 df = pd.read_parquet(file_path)

#                 # Drop unwanted index columns from previous exports
#                 df = df.drop(
#                     columns=[c for c in df.columns if "__index" in c], errors="ignore"
#                 )

#                 # Ensure timestamp exists
#                 if "timestamp" not in df.columns:
#                     raise ValueError(f"No 'timestamp' column in {file_path.name}")

#                 # Convert milliseconds → UTC datetime
#                 df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)

#                 dfs.append(df)
#             except Exception as e:
#                 self.logger.error(f"Failed reading {file_path.name}: {e}")

#         if not dfs:
#             self.logger.warning(f"No valid data loaded from {raw_dir}")
#             return pd.DataFrame()

#         # Concatenate all dataframes
#         df_all = pd.concat(dfs, ignore_index=True)

#         # Sort by timestamp and drop duplicates
#         df_all = df_all.sort_values("timestamp").loc[
#             ~df_all["timestamp"].duplicated(keep="first")
#         ]

#         # Log first 10 rows
#         self.logger.info(
#             "First 10 rows of merged data:\n" + df_all.head(10).to_string()
#         )

#         self.logger.info(
#             f"Merged DataFrame shape: {df_all.shape}, "
#             f"time range: {df_all['timestamp'].min()} → {df_all['timestamp'].max()}"
#         )

#         return df_all


# if __name__ == "__main__":
#     config = AppConfig.load_from_yaml()

#     logger = AppLogger(
#         name="Data Processing",
#         log_dir=config.dirs["logs"],
#         filename="process_data.log",
#         json_format=False,
#     )

#     data_processor = DataInvestigationPipeline(logger=logger, config=config)

#     data_processor.process_all_symbols()
