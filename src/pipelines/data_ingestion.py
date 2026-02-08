from src.config import config  # Import the pre-initialized singleton
from src.utils.app_logger import AppLogger
from src.utils.rate_limit_api_client import RateLimitedApiClient
from src.ingestion.alpaca_data import HistoricalDataDownloader
from alpaca.data.historical import StockHistoricalDataClient


def run() -> None:
    """
    Entry point for the data ingestion pipeline.
    """
    # 1. Logger Setup
    # Uses config.dirs['logs'] which we defined in the AppConfig property
    logger = AppLogger(
        name="data_ingestion",
        log_dir=config.dirs["logs"],
        clear_existing=False,
        json_format=True,
    )
    raw_client = StockHistoricalDataClient(
        api_key=config.alpaca_api_key,
        secret_key=config.alpaca_secret_key,
    )
    # 2. Rate Limited Client Setup
    # Note: We pass the raw_client as None or a dummy if the downloader
    # now manages its own Alpaca client, OR we initialize it here.
    # To keep your RateLimitedApiClient working as intended:
    client = RateLimitedApiClient(
        raw_client,
        max_requests_per_minute=config.data_ingestion["max_requests_per_minute"],
        logger=logger,
    )

    # 3. Downloader Execution
    downloader = HistoricalDataDownloader(
        client=client,
        logger=logger,
        config=config,
    )

    logger.info("Starting historical data ingestion pipeline...")
    downloader.download_all()
    logger.info("Pipeline execution finished.")


if __name__ == "__main__":
    run()
