import hashlib
import json
import os
import pandas as pd
from datetime import datetime
from pathlib import Path
from src.config import AppConfig
from src.utils.app_logger import AppLogger


class DataPersistenceManager:
    def __init__(self, config: AppConfig, logger: AppLogger):
        self.config = config
        self.logger = logger
        # Define directory structure
        self.processed_path = self.config.dirs["processed_data"]
        self.metadata_path = self.config.dirs["metadata"]

        # Ensure directories exist
        os.makedirs(self.processed_path, exist_ok=True)
        os.makedirs(self.metadata_path, exist_ok=True)

    def generate_checksum(self, df):
        """Creates a unique SHA-256 hash based on the dataframe content."""
        hash_bytes = pd.util.hash_pandas_object(df).to_numpy().tobytes()
        return hashlib.sha256(hash_bytes).hexdigest()

    def save_cleaned_data(self, df, symbol, timeframe, original_filename, report):
        """
        Saves data as Parquet and logs the version metadata.
        """
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        checksum = self.generate_checksum(df)

        # Construct output paths
        symbol_dir = self.processed_path / symbol / timeframe
        meta_dir = self.metadata_path / symbol / timeframe

        os.makedirs(symbol_dir, exist_ok=True)
        os.makedirs(meta_dir, exist_ok=True)

        # Filename: Keep original name but append 'cleaned'
        # e.g., SPY_1min_2021... -> SPY_1min_2021..._cleaned.parquet
        safe_filename = Path(original_filename).stem + "_cleaned.parquet"
        file_path = symbol_dir / safe_filename

        # Save the dataset
        df.to_parquet(file_path, compression="snappy")

        # Create Metadata Entry
        version_info = {
            "original_file": str(original_filename),
            "processed_timestamp": timestamp_str,
            "row_count": len(df),
            "checksum_sha256": checksum,
            "quality_metrics": report,
            "storage_path": str(file_path),
        }

        self.logger.info(f"Version Info: {version_info}")

        # Save metadata sidecar
        meta_filename = safe_filename.replace(".parquet", "_metadata.json")
        with open(meta_dir / meta_filename, "w") as f:
            json.dump(version_info, f, indent=4)

        return str(file_path)
