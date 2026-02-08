import pandas as pd
from finvizfinance.screener.overview import Overview
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from src.config import AppConfig
from src.utils.app_logger import AppLogger
from src.ingestion.alpaca_data import HistoricalDataDownloader


class DataIngestion:
    def __init__(self, config: AppConfig, logger: AppLogger, top_n: int = 0):
        self.config = config
        self.logger = logger
        if top_n <= 0:
            raise ValueError("top_n must be a positive integer.")
        self.top_n = top_n
        self.tickers = []

    # -----------------------------
    # Step 1: Screen Top Liquid Stocks
    # -----------------------------
    def screen_top_liquid_stocks(self):
        self.logger.info(f"Screening top {self.top_n} liquid large-cap stocks...")
        f_overview = Overview()

        filters_dict = {
            "Market Cap.": "Mega ($200bln and more)",
            "Average Volume": "Over 1M",
        }
        f_overview.set_filter(filters_dict=filters_dict)

        df_screener = f_overview.screener_view()
        if df_screener is None or df_screener.empty:
            raise ValueError("Finviz screener returned no data. Check your filters.")

        # Clean Volume column
        df_screener["Volume"] = pd.to_numeric(
            df_screener["Volume"].astype(str).str.replace(",", ""), errors="coerce"
        )
        df_screener = df_screener.dropna(subset=["Volume"])

        # Sort by Volume descending and take top N
        top_stocks = df_screener.sort_values(by="Volume", ascending=False).head(
            self.top_n
        )
        self.tickers = top_stocks["Ticker"].tolist()
        self.logger.info(f"Top tickers found: {self.tickers}")

        self.logger.info(f"Initial Tickers")
        self.logger.info(f"{self.config.global_settings.symbols}")

        # Update config symbols
        self.config.update_symbols(self.tickers)

        self.logger.info(f"Updated Tickers")
        self.logger.info(f"{self.config.global_settings.symbols}")

    # -----------------------------
    # Step 2: Fetch Alpaca Historical Data
    # -----------------------------
    def fetch_alpaca_data(self):
        self.logger.info("Fetching historical OHLCV data from Alpaca...")
        raw_client = StockHistoricalDataClient(
            config.alpaca_api_key, config.alpaca_secret_key
        )

        downloader = HistoricalDataDownloader(
            client=raw_client, logger=logger, config=config
        )
        downloader.download_all()

    # -----------------------------
    # Full Pipeline
    # -----------------------------
    def run(self):
        self.screen_top_liquid_stocks()
        self.fetch_alpaca_data()


# -----------------------------
# --- EXECUTION ---
# -----------------------------
if __name__ == "__main__":
    # Load config and initialize logger
    config = AppConfig.load_from_yaml()
    logger = AppLogger(
        name="alpaca_downloads", log_dir=config.dirs["logs"], json_format=False
    )
    try:
        config.pretty_print()

        # Run ingestion pipeline
        ingestion = DataIngestion(
            config=config, logger=logger, top_n=config.data_ingestion.n_stocks
        )
        ingestion.run()

    except Exception as e:
        logger.info(f"An error occurred: {e}")
