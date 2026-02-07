import pandas as pd
import numpy as np
import pandas_market_calendars as mcal
from scipy.stats.mstats import winsorize
from src.utils.app_logger import AppLogger
from src.config import AppConfig
from typing import Optional
import sys
from pathlib import Path
from datetime import datetime
from src.config import AppConfig
from src.utils.app_logger import AppLogger
import matplotlib.pyplot as plt


def normalize_symbol_and_timestamp(
    df: pd.DataFrame,
    drop_symbol: bool = True,
    timestamp_col: str = "timestamp",
) -> pd.DataFrame:
    """
    Normalize market data DataFrame by:
    - Dropping the 'symbol' column (if present)
    - Converting DatetimeIndex to a 'timestamp' column

    Args:
        df (pd.DataFrame): Input DataFrame
        drop_symbol (bool): Whether to drop 'symbol' column
        timestamp_col (str): Name of timestamp column to create

    Returns:
        pd.DataFrame: Normalized DataFrame
    """
    if df is None or df.empty:
        raise ValueError("Input DataFrame is empty or None.")

    df = df.copy()

    # Drop symbol column if present
    if drop_symbol and "symbol" in df.columns:
        df.drop(columns=["symbol"], inplace=True)

    # Move DatetimeIndex to timestamp column
    if isinstance(df.index, pd.DatetimeIndex):
        df[timestamp_col] = df.index
        df.reset_index(drop=True, inplace=True)
    elif timestamp_col not in df.columns:
        raise ValueError(
            "DataFrame must contain a DatetimeIndex or a 'timestamp' column."
        )

    # Final validation
    if timestamp_col not in df.columns:
        raise ValueError("Failed to create timestamp column.")

    return df


def save_corrupt_rows(
    df_corrupt: pd.DataFrame,
    reason: str,
    config: AppConfig,
    symbol: str,
    timeframe: str,
    logger: AppLogger,
):
    """
    Save corrupt rows to CSV for later inspection.

    Args:
        df_corrupt (pd.DataFrame): Rows flagged as corrupt
        reason (str): Short reason tag (e.g. 'ohlcv_integrity')
    """
    if df_corrupt.empty:
        return

    base_dir = config.dirs["data"] / "corrupt_data"
    symbol_dir = base_dir / symbol / timeframe
    symbol_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filename = f"{symbol}_{timeframe}_{reason}_{timestamp}.csv"
    path = symbol_dir / filename

    df_corrupt.to_csv(path, index=False)

    logger.warning(f"Saved {len(df_corrupt)} corrupt rows → {path}")


def visualize_market_data(
    df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    output_dir: Path,
):
    output_dir = output_dir / symbol / timeframe
    output_dir.mkdir(parents=True, exist_ok=True)

    df = df.sort_values("timestamp")

    returns = np.log(df["close"]).diff()

    plots = [
        (
            "price_vwap.png",
            lambda: (
                plt.plot(df["timestamp"], df["close"], label="Close"),
                plt.plot(df["timestamp"], df["vwap"], label="VWAP"),
                plt.legend(),
                plt.title("Price & VWAP"),
            ),
        ),
        (
            "returns.png",
            lambda: (
                plt.plot(df["timestamp"], returns),
                plt.title("Log Returns"),
            ),
        ),
        (
            "returns_dist.png",
            lambda: (
                plt.hist(returns.dropna(), bins=200, density=True),
                plt.title("Returns Distribution"),
            ),
        ),
        (
            "volume.png",
            lambda: (
                plt.plot(df["timestamp"], df["volume"]),
                plt.title("Volume"),
            ),
        ),
        (
            "volatility.png",
            lambda: (
                plt.plot(df["timestamp"], returns.rolling(30).std()),
                plt.title("Rolling Volatility"),
            ),
        ),
    ]

    for fname, plot_fn in plots:
        plt.figure(figsize=(12, 4))
        plot_fn()
        plt.tight_layout()
        plt.savefig(output_dir / fname)
        plt.close()
