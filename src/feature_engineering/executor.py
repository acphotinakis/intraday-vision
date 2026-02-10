import sys
import pandas as pd
from pathlib import Path
from src.utils.app_logger import AppLogger
from src.config import AppConfig
import re
from src.feature_engineering.feature_analyzer import FeatureAnalyzer
from src.feature_engineering.feature_builder import FeatureBuilder
from src.feature_engineering.feature_plots import FeaturePlots


class FeatureEngineeringPipeline:
    def __init__(self, logger: AppLogger, config: AppConfig):
        self.config = config
        self.logger = logger
        self.output_dir = Path(self.config.project.feature_data)
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

            # --- STAGE 3: Feature Exploration ---
            if not df_universe.empty:
                self.logger.info("--- Starting Stage 3: Feature Exploration ---")

                # 3.1 Build Features
                builder = FeatureBuilder(df_universe, self.config, self.logger)
                df_features = builder.generate_features()

                if not df_features.empty:
                    # 3.2 Analyze Features
                    feat_analyzer = FeatureAnalyzer(df_features, self.logger)
                    quality_report = feat_analyzer.run_quality_assessment()

                    plot_subdir = self.output_dir / timeframe
                    plot_subdir.mkdir(parents=True, exist_ok=True)
                    self.logger.info(f"Generating visualizations in {plot_subdir}")

                    # 3.3 Visualize
                    feat_plotter = FeaturePlots(
                        plots_dir=plot_subdir, logger=self.logger
                    )
                    feat_plotter.generate_all_plots(quality_report)

                    self.logger.info("Stage 3 Complete.")

            self.logger.info(f"--- Completed Timeframe: {timeframe} ---\n")

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
        name="featre_engineering",
        log_dir=config.dirs["logs"],
        filename="fea_engine.log",
        json_format=False,
    )

    pipeline = FeatureEngineeringPipeline(logger=logger, config=config)
    pipeline.run()
