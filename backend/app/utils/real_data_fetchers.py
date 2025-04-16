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


class RealCoinGeckoFetcher(RealDataFetcher):
    """Data fetcher for CoinGecko API"""

    def __init__(self):
        """Initialize CoinGecko fetcher"""
        super().__init__()
        self.base_url = "https://api.coingecko.com/api/v3"
        self.coin_ids = {
            'btc': 'bitcoin',
            'eth': 'ethereum',
            'bnb': 'binancecoin',
            'xrp': 'ripple',
            'ada': 'cardano',
            'sol': 'solana',
            'doge': 'dogecoin',
            'dot': 'polkadot',
            'avax': 'avalanche-2',
            'matic': 'matic-network'
        }

    def _fetch_data(self, symbol: str, days: int) -> pd.DataFrame:
        """
        Fetch data from CoinGecko

        Args:
            symbol: Cryptocurrency symbol
            days: Number of days of data to fetch

        Returns:
            DataFrame with cryptocurrency data
        """
        symbol = symbol.lower()

        # Get coin ID
        coin_id = self.coin_ids.get(symbol)
        if coin_id is None:
            # Try to find the coin ID
            try:
                coins_list = requests.get(f"{self.base_url}/coins/list").json()
                for coin in coins_list:
                    if coin['symbol'].lower() == symbol:
                        coin_id = coin['id']
                        break
            except Exception as e:
                logger.error(f"Error fetching coin list: {e}")

            if coin_id is None:
                raise ValueError(f"Unknown coin symbol: {symbol}")

        # Fetch market data
        try:
            # CoinGecko has rate limits, so we need to be careful
            url = f"{self.base_url}/coins/{coin_id}/market_chart"
            params = {
                'vs_currency': 'usd',
                'days': days,
                'interval': 'daily'
            }

            response = requests.get(url, params=params)
            data = response.json()

            # Extract price, volume, and market cap data
            prices = data.get('prices', [])
            volumes = data.get('total_volumes', [])
            market_caps = data.get('market_caps', [])

            # Create DataFrame
            df_prices = pd.DataFrame(prices, columns=['timestamp', 'price'])
            df_volumes = pd.DataFrame(volumes, columns=['timestamp', 'volume'])
            df_market_caps = pd.DataFrame(market_caps, columns=['timestamp', 'market_cap'])

            # Convert timestamp from milliseconds to datetime for all DataFrames
            df_prices['timestamp'] = pd.to_datetime(df_prices['timestamp'], unit='ms')
            df_volumes['timestamp'] = pd.to_datetime(df_volumes['timestamp'], unit='ms')
            df_market_caps['timestamp'] = pd.to_datetime(df_market_caps['timestamp'], unit='ms')

            # Merge DataFrames
            df = df_prices.merge(df_volumes, on='timestamp', how='left')
            df = df.merge(df_market_caps, on='timestamp', how='left')

            # Add symbol column
            df['symbol'] = symbol.upper()

            # Add OHLC columns (CoinGecko only provides daily close prices)
            df['close'] = df['price']
            df['open'] = df['price'].shift(1)
            df['high'] = df['price']
            df['low'] = df['price']

            # Fill missing values
            df['open'] = df['open'].fillna(df['close'])

            # Sort by timestamp
            df = df.sort_values('timestamp')

            return df

        except Exception as e:
            logger.error(f"Error fetching data from CoinGecko: {e}")

            # Raise an error if API fails
            raise ValueError(f"Failed to fetch data from CoinGecko for {symbol}: {e}")

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
    if source.lower() == 'coingecko':
        return RealCoinGeckoFetcher()
    elif source.lower() == 'binance':
        return RealBinanceFetcher()
    else:
        raise ValueError(f"Unknown data source: {source}")
