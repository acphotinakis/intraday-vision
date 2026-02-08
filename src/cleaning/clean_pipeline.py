import pandas as pd
from pathlib import Path
from src.utils.app_logger import *
from src.config import AppConfig
from typing import List
import re
import sys
from rich.table import Table
from rich.console import Console

# Import your classes
from src.cleaning.data_cleaner import MarketDataCleaner
from src.cleaning.persistence import DataPersistenceManager


class DataProcessPipeline:
    def __init__(self, logger: AppLogger, config: AppConfig):
        self.config = config
        self.logger = logger
        self.timeframes = self.config.global_settings.timeframes
        self.symbols = self.config.global_settings.symbols
        self.all_processed_files = {}

    def process_all_symbol_data(self):
        # Process all symbols and all timeframes
        for symbol in self.symbols:
            self.parquet_summary_table(symbol)

            self.all_processed_files[symbol] = {}
            for timeframe in self.timeframes:
                results = self.process_symbol_data(
                    symbol=symbol,
                    input_root=self.config.dirs["raw_data"],
                    timeframe=timeframe,
                    cleaner_cls=MarketDataCleaner,
                    persistence_cls=DataPersistenceManager,
                )
                self.all_processed_files[symbol][timeframe] = results
                sys.exit(0)

        # Example cross-symbol alignment check
        # for timeframe in self.timeframes:
        #     spy_tf = self.all_processed_files.get("SPY", {}).get(timeframe, [])
        #     vix_tf = self.all_processed_files.get("VIXY", {}).get(timeframe, [])
        #     if spy_tf and vix_tf:
        #         self.check_cross_symbol_alignment(spy_tf, vix_tf)

    def process_symbol_data(
        self,
        symbol: str,
        input_root: Path,
        timeframe: str = "1min",
        cleaner_cls=MarketDataCleaner,
        persistence_cls=DataPersistenceManager,
    ):
        """
        Load ALL raw parquet files for a (symbol, timeframe),
        merge into one DataFrame, clean once, save once.
        """
        cleaner = cleaner_cls(
            symbol=symbol,
            logger=self.logger,
            config=self.config,
            timeframe=timeframe,
        )
        persistence = persistence_cls(self.config, self.logger)

        raw_dir = Path(input_root) / symbol / timeframe
        if not raw_dir.exists():
            self.logger.error(f"Directory not found: {raw_dir}")
            return None

        self.logger.info(
            f"--- Processing {symbol} | {timeframe} (merged dataframe) ---"
        )

        try:
            # 1. Load + merge
            df_raw = self._load_and_merge_timeframe_files(raw_dir)

            if df_raw.empty:
                self.logger.warning(f"No data found for {symbol} at {timeframe}")
                return None

            # 2. Clean
            df_cleaned = cleaner.clean_pipeline(df_raw)
            quality_report = cleaner.get_report()

            # 3. Save once
            saved_path = persistence.save_cleaned_data(
                df=df_cleaned,
                symbol=symbol,
                timeframe=timeframe,
                original_filename=f"{symbol}_{timeframe}_MERGED.parquet",
                report=quality_report,
            )

            self.logger.info(f"Saved merged cleaned data → {saved_path}")

            return {
                "symbol": symbol,
                "timeframe": timeframe,
                "rows": len(df_cleaned),
                "path": saved_path,
            }

        except Exception as e:
            self.logger.error(f"FAILED processing {symbol} {timeframe}: {e}")
            return None

    def check_cross_symbol_alignment(self, spy_files, vix_files):
        self.logger.info("--- Performing Cross-Symbol Alignment Check ---")
        if not spy_files or not vix_files:
            self.logger.warning(
                "Skipping alignment check: Insufficient processed files."
            )
            return

        spy_df = spy_files[0][0]
        vix_df = vix_files[0][0]

        common_index = spy_df.index.intersection(vix_df.index)
        spy_unique = len(spy_df.index.difference(vix_df.index))
        vix_unique = len(vix_df.index.difference(spy_df.index))

        self.logger.info(f"Alignment Report:")
        self.logger.info(f"Common Timestamps: {len(common_index)}")
        self.logger.info(f"SPY-only timestamps: {spy_unique}")
        self.logger.info(f"VIXY-only timestamps: {vix_unique}")

        if spy_unique > 100 or vix_unique > 100:
            self.logger.warning("Significant misalignment detected.")
        else:
            self.logger.info("Alignment looks healthy.")

    def parquet_file_report(self, file_or_dir: Path):
        """
        Print logistics about Parquet files.

        Args:
            file_or_dir (Path): Path to a single Parquet file or a directory containing Parquet files.
        """
        if not isinstance(file_or_dir, Path):
            file_or_dir = Path(file_or_dir)

        # Collect all parquet files
        if file_or_dir.is_file() and file_or_dir.suffix == ".parquet":
            files = [file_or_dir]
        elif file_or_dir.is_dir():
            files = list(file_or_dir.glob("*.parquet"))
        else:
            self.logger.info(f"No parquet files found at {file_or_dir}")
            return

        self.logger.info(f"Found {len(files)} parquet file(s) at {file_or_dir}\n")

        for file_path in files:
            try:
                df = pd.read_parquet(file_path)
                n_rows, n_cols = df.shape
                mem_usage = df.memory_usage(deep=True).sum() / (1024**2)  # MB

                self.logger.info(f"File: {file_path.name}")
                self.logger.info(f"  Rows: {n_rows}")
                self.logger.info(f"  Columns: {n_cols} -> {list(df.columns)}")
                self.logger.info(f"  Index type: {type(df.index)}")
                self.logger.info(f"  Memory usage: {mem_usage:.2f} MB")

                # If timestamp exists, self.logger.info first/last
                if "timestamp" in df.columns:
                    ts = pd.to_datetime(df["timestamp"], errors="coerce")
                    ts = ts.dropna()
                    if not ts.empty:
                        self.logger.info(f"  First timestamp: {ts.min()}")
                        self.logger.info(f"  Last timestamp: {ts.max()}")
                    else:
                        self.logger.info(
                            f"  Timestamp column present but all values invalid."
                        )

                self.logger.info("-" * 50)

            except Exception as e:
                self.logger.info(f"Failed to read {file_path}: {e}")

    def parquet_summary_table(self, symbol: str):
        """
        Print a Rich table summarizing all parquet files for a given stock ticker
        across all timeframes, including total bars per timeframe.
        """
        console = Console()
        table = Table(title=f"Parquet Logistics Summary: {symbol}")

        # Columns
        table.add_column("Timeframe", style="cyan", no_wrap=True)
        table.add_column("File Name", style="magenta")
        table.add_column("Rows", justify="right")
        table.add_column("Columns", justify="right")
        table.add_column("Index Type")
        table.add_column("Memory (MB)", justify="right")
        table.add_column("First Timestamp")
        table.add_column("Last Timestamp")

        symbol_dir = self.config.dirs["raw_data"] / symbol

        if not symbol_dir.exists():
            self.logger.warning(f"No directory found for symbol {symbol}: {symbol_dir}")
            return

        for timeframe in self.timeframes:
            tf_dir = symbol_dir / timeframe
            if not tf_dir.exists():
                self.logger.warning(f"No data for {symbol} at {timeframe}")
                continue

            total_rows = 0
            total_mem = 0.0
            tf_first_ts = None
            tf_last_ts = None

            for file_path in sorted(tf_dir.glob("*.parquet")):
                try:
                    df = pd.read_parquet(file_path)

                    n_rows, n_cols = df.shape
                    mem_usage = df.memory_usage(deep=True).sum() / (1024**2)  # MB
                    index_type = type(df.index).__name__

                    total_rows += n_rows
                    total_mem += mem_usage

                    first_ts, last_ts = None, None
                    if "timestamp" in df.columns:
                        ts = pd.to_datetime(df["timestamp"], errors="coerce").dropna()
                        if not ts.empty:
                            first_ts = ts.min()
                            last_ts = ts.max()

                            tf_first_ts = (
                                first_ts
                                if tf_first_ts is None
                                else min(tf_first_ts, first_ts)
                            )
                            tf_last_ts = (
                                last_ts
                                if tf_last_ts is None
                                else max(tf_last_ts, last_ts)
                            )

                    table.add_row(
                        timeframe,
                        file_path.name,
                        str(n_rows),
                        str(n_cols),
                        index_type,
                        f"{mem_usage:.2f}",
                        str(first_ts) if first_ts is not None else "-",
                        str(last_ts) if last_ts is not None else "-",
                    )

                except Exception as e:
                    self.logger.error(f"Failed to read {file_path}: {e}")

            # ─────────────────────────────────────────────────────────────
            # Timeframe TOTAL row
            # ─────────────────────────────────────────────────────────────
            table.add_row(
                timeframe,
                "[bold]TOTAL[/bold]",
                f"[bold]{total_rows:,}[/bold]",
                "-",
                "-",
                f"[bold]{total_mem:.2f}[/bold]",
                str(tf_first_ts) if tf_first_ts is not None else "-",
                str(tf_last_ts) if tf_last_ts is not None else "-",
            )

        # Print table to console
        console.print(table)
        self.logger.info(f"Completed parquet summary table for {symbol}")

    # def parquet_summary_table(self, symbol: str):
    #     """
    #     Print a Rich table summarizing all parquet files for a given stock ticker
    #     across all timeframes.
    #     """
    #     console = Console()
    #     table = Table(title=f"Parquet Logistics Summary: {symbol}")

    #     # Columns
    #     table.add_column("Timeframe", style="cyan", no_wrap=True)
    #     table.add_column("File Name", style="magenta")
    #     table.add_column("Rows", justify="right")
    #     table.add_column("Columns", justify="right")
    #     table.add_column("Index Type")
    #     table.add_column("Memory (MB)", justify="right")
    #     table.add_column("First Timestamp")
    #     table.add_column("Last Timestamp")

    #     symbol_dir = self.config.dirs["raw_data"] / symbol

    #     if not symbol_dir.exists():
    #         self.logger.warning(f"No directory found for symbol {symbol}: {symbol_dir}")
    #         return

    #     for timeframe in self.timeframes:
    #         tf_dir = symbol_dir / timeframe
    #         if not tf_dir.exists():
    #             self.logger.warning(f"No data for {symbol} at {timeframe}")
    #             continue

    #         for file_path in sorted(tf_dir.glob("*.parquet")):
    #             try:
    #                 df = pd.read_parquet(file_path)
    #                 n_rows, n_cols = df.shape
    #                 mem_usage = df.memory_usage(deep=True).sum() / (1024**2)  # MB
    #                 index_type = str(type(df.index))

    #                 first_ts, last_ts = None, None
    #                 if "timestamp" in df.columns:
    #                     ts = pd.to_datetime(df["timestamp"], errors="coerce").dropna()
    #                     if not ts.empty:
    #                         first_ts = str(ts.min())
    #                         last_ts = str(ts.max())

    #                 table.add_row(
    #                     timeframe,
    #                     file_path.name,
    #                     str(n_rows),
    #                     str(n_cols),
    #                     index_type,
    #                     f"{mem_usage:.2f}",
    #                     first_ts or "-",
    #                     last_ts or "-",
    #                 )

    #             except Exception as e:
    #                 self.logger.error(f"Failed to read {file_path}: {e}")

    #     # Print table to console
    #     console.print(table)
    #     self.logger.info(f"Completed parquet summary table for {symbol}")

    def _load_and_merge_timeframe_files(self, raw_dir: Path) -> pd.DataFrame:
        """
        Load all parquet files in a timeframe directory into a single DataFrame.
        """

        def extract_start_date(file_path: Path):
            match = re.search(r"_(\d{4}-\d{2}-\d{2})_to_", file_path.name)
            if match:
                return pd.to_datetime(match.group(1))
            return pd.Timestamp.min

        files = list(raw_dir.glob("*.parquet"))
        if not files:
            return pd.DataFrame()

        files.sort(key=extract_start_date)
        self.logger.info(f"Merging {len(files)} parquet files from {raw_dir}")

        dfs = []
        for file_path in files:
            try:
                df = pd.read_parquet(file_path)
                dfs.append(df)
            except Exception as e:
                self.logger.error(f"Failed reading {file_path.name}: {e}")

        if not dfs:
            return pd.DataFrame()

        df_all = pd.concat(dfs, axis=0, ignore_index=True)

        if "timestamp" in df_all.columns:
            df_all["timestamp"] = pd.to_datetime(df_all["timestamp"], utc=True)
            df_all = df_all.set_index("timestamp")

        df_all = df_all.sort_index().loc[~df_all.index.duplicated(keep="first")]

        self.logger.info(
            f"Merged dataframe shape: {df_all.shape}, "
            f"time range: {df_all.index.min()} → {df_all.index.max()}"
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

    data_processor = DataProcessPipeline(logger=logger, config=config)

    data_processor.process_all_symbol_data()
