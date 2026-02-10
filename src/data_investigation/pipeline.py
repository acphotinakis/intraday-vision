import sys
import pandas as pd
from pathlib import Path
from src.utils.app_logger import AppLogger
from src.config import AppConfig
from src.data_investigation.market_quality_analyzer import MarketDataQualityAssessment
from src.data_investigation.quality_plots import QualityAssessmentPlots
import re
from src.data_investigation.descriptive_analyzer import DescriptiveAnalyzer
from src.data_investigation.descriptive_plots import DescriptivePlots


class DataInvestigationPipeline:
    def __init__(self, logger: AppLogger, config: AppConfig):
        self.config = config
        self.logger = logger
        self.output_dir = Path(self.config.project.data_investigation)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.symbols = self.config.global_settings.symbols

        self.logger.info(
            f"DataInvestigationPipeline initialized with {len(self.symbols)} symbols, "
            f"output directory: {self.output_dir}"
        )

    def run(self):
        """Main execution loop"""
        for timeframe in self.config.global_settings.timeframes:
            self.logger.info(f"--- Processing Timeframe: {timeframe} ---")

            # 1. Load Data for ALL symbols in this timeframe
            self.logger.info(f"Starting universe data load for timeframe: {timeframe}")
            df_universe = self._load_universe_data(timeframe)
            self.logger.info(
                f"Universe data load completed for {timeframe}: "
                f"{len(df_universe)} rows, {len(df_universe.columns)} columns"
            )

            if df_universe.empty:
                self.logger.warning(f"No data loaded for {timeframe}. Skipping.")
                continue

            # STAGE 1: Data Overview & Quality Assessment
            quality_results = self._run_data_overview_and_quality_assessment(
                df_universe, timeframe
            )

            # STAGE 2: Descriptive Analysis
            desc_results = self._run_descriptive_analysis(df_universe)

            plot_subdir = self.output_dir / timeframe
            plot_subdir.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Generating visualizations in {plot_subdir}")
            self._execute_plots(plot_subdir, timeframe, quality_results, desc_results)

            self.logger.info(f"--- Completed Timeframe: {timeframe} ---\n")

    def _execute_plots(self, plot_subdir, timeframe, quality_results, desc_results):
        # STAGE 1: Data Overview & Quality Assessment
        plotter = QualityAssessmentPlots(plots_dir=plot_subdir, logger=self.logger)
        plotter.generate_all_plots(quality_results)
        self.logger.info(
            f"Visualization generation completed for timeframe: {timeframe}"
        )

        # STAGE 2: Descriptive Analysis
        desc_plotter = DescriptivePlots(plots_dir=plot_subdir, logger=self.logger)
        desc_plotter.generate_all_plots(desc_results)

    def _run_data_overview_and_quality_assessment(self, df_universe, timeframe):
        # STAGE 1: Data Overview & Quality Assessment
        self.logger.info("--- Starting Stage 1: Data Overview & Quality Assessment ---")
        # Run Analysis

        self.logger.info(
            f"Running full quality assessment on {len(df_universe['symbol'].unique())} symbols"
        )

        self.logger.info(
            "First 10 rows of merged data:\n" + df_universe.head(10).to_string()
        )
        assessor = MarketDataQualityAssessment(
            df=df_universe,
            symbols=self.symbols,
            config=self.config,
            logger=self.logger,
        )
        results = assessor.run_full_assessment()
        self.logger.info(f"Quality assessment completed for timeframe: {timeframe}")

        return results

    def _run_descriptive_analysis(self, df_universe):
        self.logger.info("--- Starting Stage 2: Descriptive Statistics ---")

        analyzer = DescriptiveAnalyzer(df_universe, self.config, self.logger)
        desc_results = analyzer.run_all_analyses()

        return desc_results

    def _load_universe_data(self, timeframe: str) -> pd.DataFrame:
        """Loads and merges data for all symbols into one large DataFrame with caching."""
        cache_file = self.output_dir / f"universe_{timeframe}.parquet"

        # If cached file exists, load it
        if cache_file.exists():
            self.logger.info(
                f"Loading cached universe file for {timeframe}: {cache_file}"
            )
            df_universe = pd.read_parquet(cache_file)
            self.logger.info(
                f"Loaded cached universe: {df_universe.shape[0]} rows, {df_universe.shape[1]} columns"
            )
            return df_universe

        # Otherwise, merge individual files
        all_dfs = []
        raw_root = Path(self.config.dirs["raw_data"])
        self.logger.info(f"Raw data root: {raw_root}")

        for symbol in self.symbols:
            symbol_dir = raw_root / symbol / timeframe
            if not symbol_dir.exists():
                self.logger.warning(
                    f"Missing directory for symbol {symbol}: {symbol_dir}"
                )
                continue

            files = list(symbol_dir.glob("*.parquet"))
            if not files:
                self.logger.warning(
                    f"No parquet files found for {symbol} in {symbol_dir}"
                )
                continue

            def get_date(f):
                m = re.search(r"_(\d{4}-\d{2}-\d{2})_to_", f.name)
                return m.group(1) if m else f.name

            files.sort(key=get_date)

            sym_dfs = []
            for f in files:
                try:
                    df = pd.read_parquet(
                        f,
                        columns=["timestamp", "open", "high", "low", "close", "volume"],
                    )
                    df["timestamp"] = pd.to_datetime(
                        df["timestamp"], unit="ms", utc=True
                    ).dt.tz_convert("America/New_York")
                    sym_dfs.append(df)
                except Exception as e:
                    self.logger.warning(f"Error reading {f}: {e}")

            if sym_dfs:
                df_sym = pd.concat(sym_dfs, ignore_index=True)
                df_sym["symbol"] = symbol
                df_sym = df_sym.drop_duplicates(subset=["timestamp"]).reset_index(
                    drop=True
                )

                # --- Keep only bars during market hours 9:30-16:00 ---
                before_filter = len(df_sym)
                self.logger.info(
                    f"{symbol}: Filtering market hours, before: {before_filter} rows"
                )

                df_sym = df_sym[
                    (df_sym["timestamp"].dt.time >= pd.to_datetime("09:30").time())
                    & (df_sym["timestamp"].dt.time <= pd.to_datetime("16:00").time())
                ].copy()
                after_filter = len(df_sym)
                self.logger.info(
                    f"{symbol}: Filtered market hours: {before_filter} -> {after_filter} rows"
                )

                all_dfs.append(df_sym)

        if not all_dfs:
            self.logger.warning(
                f"No data loaded for any symbols in timeframe {timeframe}"
            )
            return pd.DataFrame()

        df_universe = pd.concat(all_dfs, ignore_index=True)
        self.logger.info(f"Universe DataFrame shape: {df_universe.shape}")

        # Ensure timestamp is datetime and sorted
        if not pd.api.types.is_datetime64_any_dtype(df_universe["timestamp"]):
            self.logger.info("Converting 'timestamp' column to datetime with UTC")
            df_universe["timestamp"] = pd.to_datetime(
                df_universe["timestamp"], utc=True
            )

        df_universe = df_universe.sort_values("timestamp")
        self.logger.info(
            f"Data sorted by timestamp, range: {df_universe['timestamp'].min()} to {df_universe['timestamp'].max()}"
        )

        # Type-safe date column
        df_universe["date"] = pd.DatetimeIndex(df_universe["timestamp"]).floor("D")
        self.logger.info(
            f"Added 'date' column. Sample dates: {df_universe['date'].head(3).tolist()}"
        )

        # Save merged DataFrame to cache for next time
        try:
            df_universe.to_parquet(cache_file, index=False)
            self.logger.info(f"Cached universe saved to {cache_file}")
        except Exception as e:
            self.logger.warning(f"Failed to save cached universe: {e}")

        return df_universe


if __name__ == "__main__":
    config = AppConfig.load_from_yaml()
    logger = AppLogger(
        name="DataQA",
        log_dir=config.dirs["logs"],
        filename="data_qa.log",
        json_format=False,
    )

    pipeline = DataInvestigationPipeline(logger=logger, config=config)
    pipeline.run()
