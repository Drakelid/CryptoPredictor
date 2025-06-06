import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional
import os
import logging

from app.core.config import settings
from app.utils.real_data_fetchers import get_real_data_fetcher

logger = logging.getLogger(__name__)

class CryptoDataFetcher:
    """Base class for fetching cryptocurrency data"""
    def __init__(self):
        self.data_dir = settings.DATA_DIR
        os.makedirs(self.data_dir, exist_ok=True)

    def _save_data(self, data: pd.DataFrame, symbol: str, source: str) -> str:
        """Save data to CSV file"""
        filename = f"{symbol.lower()}_{source}_{datetime.now().strftime('%Y%m%d')}.csv"
        filepath = os.path.join(self.data_dir, filename)
        data.to_csv(filepath, index=False)
        logger.info(f"Saved data to {filepath}")
        return filepath

    def load_data(self, symbol: str, source: str = None, days: int = 365) -> pd.DataFrame:
        """Load data from local storage or fetch if not available"""
        raise NotImplementedError("Subclasses must implement this method")


import requests

class CoinMarketCapFetcher(CryptoDataFetcher):
    """Fetcher for CoinMarketCap API"""

    def __init__(self, api_key: Optional[str] = None):
        super().__init__()
        self.api_key = api_key
        self.base_url = "https://pro-api.coinmarketcap.com"

    def load_data(self, symbol: str, source: str = "coinmarketcap", days: int = 365) -> pd.DataFrame:
        """Load data from CoinMarketCap API"""
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)

        url = f"{self.base_url}/v2/cryptocurrency/ohlcv/historical"
        headers = {"X-CMC_PRO_API_KEY": self.api_key} if self.api_key else {}
        params = {
            "symbol": symbol.upper(),
            "convert": "USD",
            "time_start": start_date.strftime('%Y-%m-%d'),
            "time_end": end_date.strftime('%Y-%m-%d')
        }

        response = requests.get(url, headers=headers, params=params, timeout=10)
        data = response.json()
        quotes = data.get("data", {}).get("quotes", [])

        records = []
        for q in quotes:
            usd = q.get("quote", {}).get("USD", {})
            records.append({
                "timestamp": q.get("time_open"),
                "open": usd.get("open"),
                "high": usd.get("high"),
                "low": usd.get("low"),
                "close": usd.get("close"),
                "volume": usd.get("volume")
            })

        df = pd.DataFrame(records)
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            self._save_data(df, symbol, source)
        return df


from binance.client import Client

class BinanceFetcher(CryptoDataFetcher):
    """Fetcher for Binance API"""
    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None):
        super().__init__()
        self.client = Client(api_key, api_secret)

    def load_data(self, symbol: str, source: str = "binance", days: int = 365) -> pd.DataFrame:
        """Load data from Binance API"""
        try:
            # Format symbol for Binance
            formatted_symbol = f"{symbol.upper()}USDT"

            # Calculate start time
            start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)

            # Get klines (candlestick data)
            klines = self.client.get_historical_klines(
                formatted_symbol,
                Client.KLINE_INTERVAL_1HOUR,
                start_time
            )

            # Create DataFrame
            columns = [
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ]
            df = pd.DataFrame(klines, columns=columns)

            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

            # Convert numeric columns
            numeric_columns = ['open', 'high', 'low', 'close', 'volume',
                              'quote_asset_volume', 'taker_buy_base_asset_volume',
                              'taker_buy_quote_asset_volume']
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col])

            # Drop unnecessary columns
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

            # Save data
            filepath = self._save_data(df, symbol, source)

            return df

        except Exception as e:
            logger.error(f"Error fetching data from Binance: {e}")
            raise


def get_data_fetcher(source: str = "coinmarketcap"):
    """Factory function to get the appropriate data fetcher"""
    # Use real data fetchers
    try:
        return get_real_data_fetcher(source)
    except ValueError as e:
        # If the source is not supported, raise the error
        raise e
    except Exception as e:
        logger.warning(f"Error getting real data fetcher: {e}. Falling back to synthetic data.")

        # Fall back to existing fetchers
        if source.lower() == "coinmarketcap":
            return CoinMarketCapFetcher(api_key=settings.COINMARKETCAP_API_KEY)
        elif source.lower() == "binance":
            return BinanceFetcher(
                api_key=settings.BINANCE_API_KEY,
                api_secret=settings.BINANCE_API_SECRET
            )
        else:
            raise ValueError(f"Unsupported data source: {source}")
