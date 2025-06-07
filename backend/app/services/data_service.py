import pandas as pd
import numpy as np
import os
import logging
from typing import List, Dict, Optional, Union
from datetime import datetime, timedelta

from app.utils.data_fetchers import get_data_fetcher
from app.schemas.data import DataInfo
from app.core.config import settings

logger = logging.getLogger(__name__)

class DataService:
    """Service for handling cryptocurrency data"""

    def __init__(self):
        """Initialize data service"""
        self.data_dir = settings.DATA_DIR
        os.makedirs(self.data_dir, exist_ok=True)

    def fetch_data(self, symbol: str, source: str = "coinmarketcap", days: int = 365) -> DataInfo:
        """
        Fetch cryptocurrency data from external source

        Args:
            symbol: Cryptocurrency symbol (e.g., BTC, ETH)
            source: Data source (coinmarketcap, binance)
            days: Number of days of historical data to fetch

        Returns:
            DataInfo object with information about the fetched data
        """
        try:
            # Try to use real data fetchers first
            # Get data fetcher
            fetcher = get_data_fetcher(source)

            # Fetch data
            df = fetcher.load_data(symbol=symbol, source=source, days=days)

            # Verify we have real data
            if df is None or df.empty:
                logger.error(f"No data returned from fetcher for {symbol} from {source}")
                raise ValueError(f"No data available for {symbol} from {source}. Please try another source.")

            # Save the data
            self.save_data(df, symbol, source)

            # Create data info
            info = self._create_data_info(df, symbol, source)

            return info
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            raise

    def save_data(self, df: pd.DataFrame, symbol: str, source: str) -> DataInfo:
        """
        Save cryptocurrency data

        Args:
            df: DataFrame with cryptocurrency data
            symbol: Cryptocurrency symbol (e.g., BTC, ETH)
            source: Data source name

        Returns:
            DataInfo object with information about the saved data
        """
        # Ensure timestamp column exists
        if 'timestamp' not in df.columns:
            raise ValueError("DataFrame must have a 'timestamp' column")

        # Convert timestamp to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Save data
        filename = f"{symbol.lower()}_{source}_{datetime.now().strftime('%Y%m%d')}.csv"
        filepath = os.path.join(self.data_dir, filename)
        df.to_csv(filepath, index=False)

        logger.info(f"Saved data to {filepath}")

        # Create data info
        info = self._create_data_info(df, symbol, source)

        return info

    def load_data(self, symbol: str, source: str = None) -> pd.DataFrame:
        """
        Load cryptocurrency data

        Args:
            symbol: Cryptocurrency symbol (e.g., BTC, ETH)
            source: Data source name

        Returns:
            DataFrame with cryptocurrency data
        """
        try:
            # Find the latest file for the symbol and source
            files = os.listdir(self.data_dir)

            if source:
                # Filter by symbol and source
                files = [f for f in files if f.startswith(f"{symbol.lower()}_{source}") and f.endswith('.csv')]
            else:
                # Filter by symbol only
                files = [f for f in files if f.startswith(f"{symbol.lower()}_") and f.endswith('.csv')]

            if not files:
                # If no local data found, try to fetch it
                logger.info(f"No local data found for {symbol}, fetching from API...")
                # Fetch data from API
                info = self.fetch_data(symbol=symbol, source=source or "coinmarketcap", days=365)
                # Load the newly fetched data
                return self.load_data(symbol=symbol, source=source)

            # If we still don't have files after trying to fetch, raise an error
            if not files:
                logger.error(f"No data available for {symbol}")
                raise ValueError(f"No data available for {symbol}. Please try fetching data first.")

            # Sort by date (newest first)
            files.sort(reverse=True)

            # Load the latest file
            filepath = os.path.join(self.data_dir, files[0])
            df = pd.read_csv(filepath)

            # Convert timestamp to datetime
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])

            return df

        except FileNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Error loading data for {symbol}: {e}")
            raise

    def get_data_info(self, symbol: Optional[str] = None) -> List[DataInfo]:
        """
        Get information about available data

        Args:
            symbol: Cryptocurrency symbol (e.g., BTC, ETH)

        Returns:
            List of DataInfo objects
        """
        # List all CSV files in data directory
        files = [f for f in os.listdir(self.data_dir) if f.endswith('.csv')]

        # Filter by symbol if provided
        if symbol:
            files = [f for f in files if f.startswith(f"{symbol.lower()}_")]

        # Create data info for each file
        info_list = []
        for file in files:
            # Parse symbol and source from filename
            parts = file.split('_')
            if len(parts) >= 2:
                file_symbol = parts[0]
                file_source = parts[1]

                # Load data
                filepath = os.path.join(self.data_dir, file)
                df = pd.read_csv(filepath)

                # Convert timestamp to datetime if needed
                if 'timestamp' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                    df['timestamp'] = pd.to_datetime(df['timestamp'])

                # Create data info
                info = self._create_data_info(df, file_symbol, file_source)
                info_list.append(info)

        return info_list

    def _generate_synthetic_data(self, symbol: str, days: int = 365) -> pd.DataFrame:
        """
        This method has been disabled to ensure only real data is used
        """
        logger.error(f"Synthetic data generation is disabled. Only real data can be used.")
        raise ValueError(f"Synthetic data generation is disabled. Please use real data for {symbol}.")

    def _create_data_info(self, df: pd.DataFrame, symbol: str, source: str) -> DataInfo:
        """
        Create DataInfo object from DataFrame

        Args:
            df: DataFrame with cryptocurrency data
            symbol: Cryptocurrency symbol (e.g., BTC, ETH)
            source: Data source name

        Returns:
            DataInfo object
        """
        # Ensure timestamp column exists
        if 'timestamp' not in df.columns:
            raise ValueError("DataFrame must have a 'timestamp' column")

        # Convert timestamp to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Get start and end dates
        start_date = df['timestamp'].min()
        end_date = df['timestamp'].max()

        # Create data info
        info = DataInfo(
            symbol=symbol,
            source=source,
            start_date=start_date,
            end_date=end_date,
            rows=len(df),
            columns=df.columns.tolist()
        )

        return info
