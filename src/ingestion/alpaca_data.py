import hashlib
import sys
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from src.utils.app_logger import AppLogger
from src.utils.rate_limit_api_client import RateLimitedApiClient
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.enums import Adjustment, DataFeed
from alpaca.data.timeframe import TimeFrame as AlpacaTimeFrame, TimeFrameUnit
from src.config import AppConfig
from pandas.core.api import DataFrame
from datetime import timezone
from typing import Optional
from pandas.tseries.offsets import BDay


class HistoricalDataDownloader:
    def __init__(
        self, client: StockHistoricalDataClient, logger: AppLogger, config: AppConfig
    ):
        self.client = client
        self.logger = logger
        self.config = config
        # Initialize the Alpaca internal client once
        # self.data_client = StockHistoricalDataClient(
        #     self.config.alpaca_api_key, self.config.alpaca_secret_key
        # )
        # Moved from old config.py
        self.timeframe_map = {
            "1min": AlpacaTimeFrame(1, TimeFrameUnit("Min")),
            "5min": AlpacaTimeFrame(5, TimeFrameUnit("Min")),
            "15min": AlpacaTimeFrame(15, TimeFrameUnit("Min")),
            "1h": AlpacaTimeFrame(1, TimeFrameUnit("Hour")),
            "1D": AlpacaTimeFrame(1, TimeFrameUnit("Day")),
        }

        # Dynamic date calculation using dot-notation
        ingestion = self.config.data_ingestion
        self.start_date = datetime.now() - timedelta(days=ingestion.years_back * 365)
        self.end_date = datetime.now()
        self.chunk_days = ingestion.chunk_days

    @staticmethod
    def generate_checksum(df: DataFrame) -> str:
        """Generate SHA-256 hash for a DataFrame's content"""
        hash_bytes = pd.util.hash_pandas_object(df).to_numpy().tobytes()
        return hashlib.sha256(hash_bytes).hexdigest()

    def save_parquet_if_changed(self, df: DataFrame, file_path: Path):
        """Save DataFrame only if contents have changed"""
        if file_path.exists():
            existing_df = pd.read_parquet(file_path)
            existing_checksum = self.generate_checksum(existing_df)
            new_checksum = self.generate_checksum(df)
            if existing_checksum == new_checksum:
                self.logger.info(
                    f"File unchanged, skipping save: {file_path}",
                    extra={"rows": len(df)},
                )
                return False
            else:
                self.logger.info(
                    f"File exists but changed. Overwriting: {file_path}",
                    extra={"rows": len(df)},
                )
        df.reset_index(inplace=True)
        self.logger.info(f"First 5 rows saved:\n{df.head()}")
        df.to_parquet(file_path, engine="pyarrow", index=True, compression="snappy")
        return True

    def parse_data_feed(self, value: str) -> DataFeed:
        try:
            return DataFeed(value)
        except ValueError:
            raise ValueError(
                f"Invalid data feed '{value}'. "
                f"Valid options: {[f.value for f in DataFeed]}"
            )

    def parse_data_adjustment(self, value: str) -> Adjustment:
        try:
            return Adjustment(value)
        except ValueError:
            raise ValueError(
                f"Invalid data adjustment type  '{value}'. "
                f"Valid options: {[f.value for f in Adjustment]}"
            )

    def download_symbol(self, symbol: str, timeframe_str: str):
        """Download all historical bars for a single symbol and timeframe, handling pagination."""

        timeframe_obj = self.timeframe_map[timeframe_str]
        freq = self.config.global_settings.resample_map[timeframe_str]

        data_adjustment = self.parse_data_adjustment(
            self.config.data_ingestion.adjustment
        )
        data_feed = self.parse_data_feed(self.config.data_ingestion.data_feed)

        output_dir = self.config.dirs["raw_data"] / symbol / timeframe_str
        output_dir.mkdir(parents=True, exist_ok=True)

        # Use UTC-aware datetimes for Alpaca
        current_start = self.start_date.replace(tzinfo=timezone.utc)
        global_end = datetime.now(timezone.utc) - pd.Timedelta(
            minutes=16
        )  # avoid incomplete last bars

        while current_start < global_end:
            current_chunk_end = min(
                current_start + pd.Timedelta(days=self.chunk_days), global_end
            )

            file_path = (
                output_dir
                / f"{symbol}_{timeframe_str}_{current_start.date()}_to_{current_chunk_end.date()}.parquet"
            )
            self.logger.info(
                f"Processing chunk: {current_start.date()} to {current_chunk_end.date()} -> {file_path}"
            )

            all_pages = []
            page_token: Optional[str] = None
            self.estimate_rows(symbol, timeframe_str)

            while True:
                request_params = StockBarsRequest(
                    symbol_or_symbols=symbol,
                    timeframe=timeframe_obj,
                    start=current_start,
                    end=current_chunk_end,
                    adjustment=data_adjustment,
                    feed=data_feed,
                    # limit=10000,
                    # page_token=page_token,
                )

                self.logger.info(f"Request params: {request_params}")
                try:

                    response = self.client.get_stock_bars(request_params)
                    # self.logger.info(f"Response: {response}")
                    # Log only metadata, not OHLCV data
                    response_info = {
                        "symbol": symbol,
                        "timeframe": timeframe_str,
                        "start": current_start.isoformat(),
                        "end": current_chunk_end.isoformat(),
                        "page_token": page_token,
                        "next_page_token": getattr(response, "next_page_token", None),
                        "row_count": len(response.df) if hasattr(response, "df") else 0,
                    }
                    self.logger.info(f"Response metadata: {response_info}")

                    df = response.df

                    if df.empty:
                        self.logger.warning(
                            f"No data returned for page_token={page_token}"
                        )
                        break

                    self.logger.info(f"Page first 5 rows:\n{df.head()}")
                    all_pages.append(df)

                    # Check for more pages
                    page_token = getattr(response, "next_page_token", None)
                    if not page_token:
                        break

                except Exception as e:
                    self.logger.error(
                        f"Error fetching page for {symbol}: {e}",
                        extra={"page_token": page_token},
                    )
                    break  # stop this chunk if error occurs
            if all_pages:
                df = pd.concat(all_pages)
                self.save_parquet_if_changed(df, file_path)
            else:
                self.logger.warning(
                    f"No data downloaded for chunk {current_start.date()} to {current_chunk_end.date()}"
                )

            # Move to next chunk
            current_start = current_chunk_end + pd.Timedelta(minutes=1)

    def download_all(self):
        symbols = self.config.global_settings.symbols
        timeframes = self.config.global_settings.timeframes
        for symbol in symbols:
            for timeframe_str in timeframes:
                self.logger.info(f"Downloading {symbol} at {timeframe_str} timeframe")
                self.download_symbol(symbol, timeframe_str)
        self.logger.info("All downloads completed.")

    def estimate_rows(self, symbol: str, timeframe_str: str) -> int:
        """
        Estimate how many rows we should get for a symbol and timeframe
        before downloading data.
        """
        tf = timeframe_str.lower()
        start = self.start_date
        end = self.end_date

        # Approximate trading days (assuming weekdays only)
        total_days = pd.date_range(start=start, end=end, freq=BDay()).size

        # Map timeframe to bars per day
        # Assuming US market hours 6.5 hours/day -> 390 minutes/day
        bars_per_day = {
            "1min": 390,
            "5min": 78,  # 390 / 5
            "15min": 26,  # 390 / 15
            "1h": 6,  # 6.5 hours rounded down
            "1d": 1,
        }

        bars = bars_per_day.get(tf, 0) * total_days
        self.logger.info(
            f"Estimated rows for {symbol} at {timeframe_str} from "
            f"{start.date()} to {end.date()}: ~{bars} rows"
        )
        return bars


if __name__ == "__main__":
    # Initialize config, logger, and API client
    config = AppConfig.load_from_yaml()
    config.pretty_print()

    logger = AppLogger(
        name="alpaca_downloads", log_dir=config.dirs["logs"], json_format=False
    )
    raw_client = StockHistoricalDataClient(
        config.alpaca_api_key, config.alpaca_secret_key
    )

    downloader = HistoricalDataDownloader(
        client=raw_client, logger=logger, config=config
    )
    downloader.download_all()
