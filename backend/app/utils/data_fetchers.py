import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional
import os
import logging
import requests

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


from pycoingecko import CoinGeckoAPI

class CoinGeckoFetcher(CryptoDataFetcher):
    """Fetcher for CoinGecko API"""
    def __init__(self, api_key: Optional[str] = None):
        super().__init__()
        self.cg = CoinGeckoAPI(api_key=api_key)
        self.coin_list = None
        self._init_coin_list()

    def _init_coin_list(self):
        """Initialize coin list for ID mapping"""
        try:
            self.coin_list = self.cg.get_coins_list()
            logger.info(f"Initialized CoinGecko coin list with {len(self.coin_list)} coins")
        except Exception as e:
            logger.error(f"Failed to initialize CoinGecko coin list: {e}")
            self.coin_list = []

    def _get_coin_id(self, symbol: str) -> str:
        """Get coin ID from symbol"""
        if not self.coin_list:
            self._init_coin_list()

        # Find exact match first
        for coin in self.coin_list:
            if coin['symbol'].lower() == symbol.lower():
                return coin['id']

        # If no exact match, use common mappings
        common_mappings = {
            'btc': 'bitcoin',
            'eth': 'ethereum',
            'bnb': 'binancecoin',
            'xrp': 'ripple',
            'ada': 'cardano',
            'sol': 'solana',
            'doge': 'dogecoin'
        }

        return common_mappings.get(symbol.lower())

    def load_data(self, symbol: str, source: str = "coingecko", days: int = 365) -> pd.DataFrame:
        """Load data from CoinGecko API"""
        coin_id = self._get_coin_id(symbol)
        if not coin_id:
            raise ValueError(f"Could not find coin ID for symbol {symbol}")

        try:
            # Get market data
            market_data = self.cg.get_coin_market_chart_by_id(
                id=coin_id,
                vs_currency='usd',
                days=days
            )

            # Process price data
            prices = market_data['prices']
            volumes = market_data['total_volumes']
            market_caps = market_data['market_caps']

            df = pd.DataFrame(prices, columns=['timestamp', 'price'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)

            # Add volume and market cap
            df['volume'] = [v[1] for v in volumes]
            df['market_cap'] = [m[1] for m in market_caps]

            # Resample to hourly data to ensure consistency
            df = df.resample('1H').mean()
            df.dropna(inplace=True)

            # Save data
            df.reset_index(inplace=True)
            filepath = self._save_data(df, symbol, source)

            return df

        except Exception as e:
            logger.error(f"Error fetching data from CoinGecko: {e}")
            raise


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


class CoinMarketCapFetcher(CryptoDataFetcher):
    """Fetcher for CoinMarketCap API"""

    def __init__(self, api_key: Optional[str] = None):
        super().__init__()
        self.api_key = api_key
        self.base_url = "https://pro-api.coinmarketcap.com/v1"

    def load_data(
        self, symbol: str, source: str = "coinmarketcap", days: int = 365
    ) -> pd.DataFrame:
        """Load data from CoinMarketCap API"""
        try:
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(days=days)
            url = f"{self.base_url}/cryptocurrency/ohlcv/historical"
            params = {
                "symbol": symbol.upper(),
                "time_start": start_time.strftime("%Y-%m-%d"),
                "time_end": end_time.strftime("%Y-%m-%d"),
                "interval": "daily",
                "convert": "USD",
            }

            headers = {}
            if self.api_key:
                headers["X-CMC_PRO_API_KEY"] = self.api_key

            response = requests.get(url, params=params, headers=headers)
            data = response.json()

            quotes = data.get("data", {}).get("quotes", [])
            if not quotes:
                raise ValueError("No data returned from CoinMarketCap")

            records = []
            for q in quotes:
                quote = q.get("quote", {}).get("USD", {})
                records.append(
                    {
                        "timestamp": q.get("time_open"),
                        "open": quote.get("open"),
                        "high": quote.get("high"),
                        "low": quote.get("low"),
                        "close": quote.get("close"),
                        "volume": quote.get("volume"),
                        "market_cap": quote.get("market_cap"),
                    }
                )

            df = pd.DataFrame(records)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df["price"] = df["close"]

            self._save_data(df, symbol, source)
            return df

        except Exception as e:
            logger.error(f"Error fetching data from CoinMarketCap: {e}")
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
        elif source.lower() == "coingecko":
            return CoinGeckoFetcher(api_key=settings.COINGECKO_API_KEY)
        elif source.lower() == "binance":
            return BinanceFetcher(
                api_key=settings.BINANCE_API_KEY,
                api_secret=settings.BINANCE_API_SECRET
            )
        else:
            raise ValueError(f"Unsupported data source: {source}")
