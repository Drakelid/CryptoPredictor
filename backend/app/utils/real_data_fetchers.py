import pandas as pd
import numpy as np
import requests
import logging
import os
import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Union, Any
import time

from app.core.config import settings

logger = logging.getLogger(__name__)

class RealDataFetcher:
    """Base class for real cryptocurrency data fetchers"""

    def __init__(self):
        """Initialize data fetcher"""
        self.cache_dir = os.path.join(settings.DATA_DIR, 'cache')
        os.makedirs(self.cache_dir, exist_ok=True)

    def load_data(self, symbol: str, source: str, days: int = 365) -> pd.DataFrame:
        """
        Load cryptocurrency data

        Args:
            symbol: Cryptocurrency symbol (e.g., BTC, ETH)
            source: Data source name
            days: Number of days of historical data to fetch

        Returns:
            DataFrame with cryptocurrency data
        """
        # Try to load cached data first
        cached_data = self._load_cached_data(symbol, source, days)
        if cached_data is not None:
            return cached_data

        # If no cached data, fetch from source
        data = self._fetch_data(symbol, days)

        # Cache the data
        self._cache_data(data, symbol, source)

        return data

    def _load_cached_data(self, symbol: str, source: str, days: int) -> Optional[pd.DataFrame]:
        """
        Load cached data

        Args:
            symbol: Cryptocurrency symbol
            source: Data source name
            days: Number of days of data needed

        Returns:
            DataFrame with cryptocurrency data or None if no suitable cache exists
        """
        # Find the latest cache file for this symbol and source
        cache_files = [f for f in os.listdir(self.cache_dir)
                      if f.startswith(f"{symbol.lower()}_{source}_") and f.endswith('.csv')]

        if not cache_files:
            return None

        # Sort by date (newest first)
        cache_files.sort(reverse=True)
        latest_file = os.path.join(self.cache_dir, cache_files[0])

        # Check if the cache is recent enough
        file_date_str = cache_files[0].split('_')[-1].split('.')[0]
        file_date = datetime.strptime(file_date_str, '%Y%m%d')

        if datetime.now() - file_date > timedelta(days=1):
            # Cache is too old, need to refresh
            return None

        # Load the cache
        df = pd.read_csv(latest_file)

        # Convert timestamp to datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Check if we have enough days of data
        if len(df) < days:
            return None

        # Filter to the requested number of days
        return df.tail(days)

    def _cache_data(self, df: pd.DataFrame, symbol: str, source: str):
        """
        Cache data

        Args:
            df: DataFrame with cryptocurrency data
            symbol: Cryptocurrency symbol
            source: Data source name
        """
        filename = f"{symbol.lower()}_{source}_{datetime.now().strftime('%Y%m%d')}.csv"
        filepath = os.path.join(self.cache_dir, filename)
        df.to_csv(filepath, index=False)
        logger.info(f"Cached data to {filepath}")

    def _fetch_data(self, symbol: str, days: int) -> pd.DataFrame:
        """
        Fetch data from source

        Args:
            symbol: Cryptocurrency symbol
            days: Number of days of data to fetch

        Returns:
            DataFrame with cryptocurrency data
        """
        raise NotImplementedError("Subclasses must implement _fetch_data method")


class RealCoinMarketCapFetcher(RealDataFetcher):
    """Data fetcher for CoinMarketCap API"""

    def __init__(self):
        """Initialize CoinMarketCap fetcher"""
        super().__init__()
        self.base_url = "https://pro-api.coinmarketcap.com"
        self.api_key = settings.COINMARKETCAP_API_KEY

    def _fetch_data(self, symbol: str, days: int) -> pd.DataFrame:
        """
        Fetch data from CoinMarketCap

        Args:
            symbol: Cryptocurrency symbol
            days: Number of days of data to fetch

        Returns:
            DataFrame with cryptocurrency data
        """
        symbol = symbol.upper()
        # Fetch market data
        try:
            url = f"{self.base_url}/v2/cryptocurrency/ohlcv/historical"
            headers = {"X-CMC_PRO_API_KEY": self.api_key} if self.api_key else {}
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=days)
            params = {
                'symbol': symbol,
                'convert': 'USD',
                'time_start': start_date.strftime('%Y-%m-%d'),
                'time_end': end_date.strftime('%Y-%m-%d')
            }

            response = requests.get(url, headers=headers, params=params, timeout=10)
            data = response.json()

            quotes = data.get('data', {}).get('quotes', [])
            records = []
            for q in quotes:
                usd = q.get('quote', {}).get('USD', {})
                records.append({
                    'timestamp': q.get('time_open'),
                    'open': usd.get('open'),
                    'high': usd.get('high'),
                    'low': usd.get('low'),
                    'close': usd.get('close'),
                    'volume': usd.get('volume')
                })

            df = pd.DataFrame(records)

            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df['symbol'] = symbol
                df = df.sort_values('timestamp')
            return df

        except Exception as e:
            logger.error(f"Error fetching data from CoinMarketCap: {e}")

            # Raise an error if API fails
            raise ValueError(f"Failed to fetch data from CoinMarketCap for {symbol}: {e}")

    # Synthetic data generation has been removed to ensure only real data is used


class RealBinanceFetcher(RealDataFetcher):
    """Data fetcher for Binance API"""

    def __init__(self):
        """Initialize Binance fetcher"""
        super().__init__()
        self.base_url = "https://api.binance.com/api/v3"

    def _fetch_data(self, symbol: str, days: int) -> pd.DataFrame:
        """
        Fetch data from Binance

        Args:
            symbol: Cryptocurrency symbol
            days: Number of days of data to fetch

        Returns:
            DataFrame with cryptocurrency data
        """
        symbol = symbol.upper()

        # Add USDT suffix if not present
        if not symbol.endswith('USDT'):
            market_symbol = f"{symbol}USDT"
        else:
            market_symbol = symbol

        try:
            # Calculate start time
            end_time = int(time.time() * 1000)  # Current time in milliseconds
            start_time = end_time - (days * 24 * 60 * 60 * 1000)  # days ago in milliseconds

            # Fetch klines (candlestick) data
            url = f"{self.base_url}/klines"
            params = {
                'symbol': market_symbol,
                'interval': '1d',  # Daily data
                'startTime': start_time,
                'endTime': end_time,
                'limit': 1000  # Maximum limit
            }

            response = requests.get(url, params=params)
            data = response.json()

            # Check if we got valid data
            if not isinstance(data, list) or len(data) == 0:
                logger.warning(f"No data returned from Binance for {market_symbol}")
                raise ValueError(f"No data returned from Binance for {market_symbol}")

            # Create DataFrame
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])

            # Convert timestamp from milliseconds to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

            # Convert numeric columns
            numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'quote_asset_volume']
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col])

            # Add symbol column
            df['symbol'] = symbol

            # Add price column (same as close)
            df['price'] = df['close']

            # Add market cap (not available from Binance, use volume as proxy)
            df['market_cap'] = df['price'] * df['volume']

            # Sort by timestamp
            df = df.sort_values('timestamp')

            # Select relevant columns
            df = df[['timestamp', 'symbol', 'price', 'open', 'high', 'low', 'close', 'volume', 'market_cap']]

            return df

        except Exception as e:
            logger.error(f"Error fetching data from Binance: {e}")

            # Raise an error if API fails
            raise ValueError(f"Failed to fetch data from Binance for {symbol}: {e}")

    # Synthetic data generation has been removed to ensure only real data is used


def get_real_data_fetcher(source: str) -> RealDataFetcher:
    """
    Get real data fetcher for a source

    Args:
        source: Data source name

    Returns:
        RealDataFetcher instance
    """
    if source.lower() == 'coinmarketcap':
        return RealCoinMarketCapFetcher()
    elif source.lower() == 'binance':
        return RealBinanceFetcher()
    else:
        raise ValueError(f"Unknown data source: {source}")
