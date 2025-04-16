import pandas as pd
import numpy as np
import requests
import logging
import os
import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Union, Any
import re
import time

from app.core.config import settings

logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    """Class for analyzing cryptocurrency market sentiment"""

    def __init__(self):
        """Initialize sentiment analyzer"""
        self.sentiment_dir = os.path.join(settings.DATA_DIR, 'sentiment')
        os.makedirs(self.sentiment_dir, exist_ok=True)

    def get_sentiment_data(self, symbol: str, days: int = 30) -> pd.DataFrame:
        """
        Get sentiment data for a cryptocurrency

        Args:
            symbol: Cryptocurrency symbol (e.g., BTC, ETH)
            days: Number of days of historical data to fetch

        Returns:
            DataFrame with sentiment data
        """
        # Try to load cached data first
        cached_data = self._load_cached_data(symbol, days)
        if cached_data is not None:
            return cached_data

        # If no cached data, fetch from sources
        sentiment_data = self._fetch_sentiment_data(symbol, days)

        # Cache the data
        self._cache_sentiment_data(sentiment_data, symbol)

        return sentiment_data

    def _load_cached_data(self, symbol: str, days: int) -> Optional[pd.DataFrame]:
        """
        Load cached sentiment data

        Args:
            symbol: Cryptocurrency symbol
            days: Number of days of data needed

        Returns:
            DataFrame with sentiment data or None if no suitable cache exists
        """
        # Find the latest cache file for this symbol
        cache_files = [f for f in os.listdir(self.sentiment_dir)
                      if f.startswith(f"{symbol.lower()}_sentiment_") and f.endswith('.csv')]

        if not cache_files:
            return None

        # Sort by date (newest first)
        cache_files.sort(reverse=True)
        latest_file = os.path.join(self.sentiment_dir, cache_files[0])

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

    def _cache_sentiment_data(self, df: pd.DataFrame, symbol: str):
        """
        Cache sentiment data

        Args:
            df: DataFrame with sentiment data
            symbol: Cryptocurrency symbol
        """
        filename = f"{symbol.lower()}_sentiment_{datetime.now().strftime('%Y%m%d')}.csv"
        filepath = os.path.join(self.sentiment_dir, filename)
        df.to_csv(filepath, index=False)
        logger.info(f"Cached sentiment data to {filepath}")

    def _fetch_sentiment_data(self, symbol: str, days: int) -> pd.DataFrame:
        """
        Fetch sentiment data from various sources

        Args:
            symbol: Cryptocurrency symbol
            days: Number of days of data to fetch

        Returns:
            DataFrame with sentiment data
        """
        # In a real implementation, this would call APIs like:
        # - CryptoCompare News API
        # - Twitter/X API
        # - Reddit API
        # - Santiment API
        # - Alternative.me Fear & Greed Index

        # For now, we'll return a minimal DataFrame with a neutral sentiment
        # This is not synthetic data, just a placeholder until real APIs are integrated
        logger.warning(f"No real sentiment data available for {symbol}. Using neutral sentiment.")

        # Generate dates for the requested period
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')

        # Create a minimal DataFrame with neutral sentiment
        data = []
        for date in date_range:
            data.append({
                'timestamp': date,
                'sentiment_score': 0.5,  # Neutral sentiment
                'mention_volume': 0,     # No mentions
                'news_sentiment': 0.5,    # Neutral news sentiment
                'social_sentiment': 0.5,  # Neutral social sentiment
                'fear_greed_index': 50,   # Neutral fear/greed
                'bullish_percentage': 50, # Neutral market sentiment
                'bearish_percentage': 50  # Neutral market sentiment
            })

        return pd.DataFrame(data)

    def analyze_text_sentiment(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment of a text

        Args:
            text: Text to analyze

        Returns:
            Dictionary with sentiment scores
        """
        # In a real implementation, this would use NLP models like:
        # - VADER sentiment analysis
        # - Hugging Face transformers
        # - Custom trained models for crypto sentiment

        # For now, use a simple rule-based approach
        text = text.lower()

        # Define sentiment dictionaries
        positive_words = ['bullish', 'buy', 'moon', 'pump', 'gain', 'profit', 'growth', 'positive', 'up', 'rise']
        negative_words = ['bearish', 'sell', 'dump', 'crash', 'loss', 'negative', 'down', 'fall', 'decrease']

        # Count occurrences
        positive_count = sum(1 for word in positive_words if word in text)
        negative_count = sum(1 for word in negative_words if word in text)

        # Calculate sentiment score
        total_count = positive_count + negative_count
        if total_count == 0:
            sentiment_score = 0.5  # Neutral
        else:
            sentiment_score = positive_count / total_count

        return {
            'sentiment_score': sentiment_score,
            'positive_count': positive_count,
            'negative_count': negative_count,
            'is_positive': sentiment_score > 0.6,
            'is_negative': sentiment_score < 0.4,
            'is_neutral': 0.4 <= sentiment_score <= 0.6
        }


class CryptoFearGreedIndex:
    """Class for fetching the Crypto Fear & Greed Index"""

    def __init__(self):
        """Initialize Fear & Greed Index fetcher"""
        self.cache_dir = os.path.join(settings.DATA_DIR, 'sentiment')
        os.makedirs(self.cache_dir, exist_ok=True)

    def get_fear_greed_data(self, days: int = 30) -> pd.DataFrame:
        """
        Get Fear & Greed Index data

        Args:
            days: Number of days of historical data to fetch

        Returns:
            DataFrame with Fear & Greed Index data
        """
        # Try to load cached data first
        cached_data = self._load_cached_data(days)
        if cached_data is not None:
            return cached_data

        # If no cached data, fetch from source
        fg_data = self._fetch_fear_greed_data(days)

        # Cache the data
        self._cache_fear_greed_data(fg_data)

        return fg_data

    def _load_cached_data(self, days: int) -> Optional[pd.DataFrame]:
        """
        Load cached Fear & Greed Index data

        Args:
            days: Number of days of data needed

        Returns:
            DataFrame with Fear & Greed Index data or None if no suitable cache exists
        """
        # Find the latest cache file
        cache_files = [f for f in os.listdir(self.cache_dir)
                      if f.startswith('fear_greed_') and f.endswith('.csv')]

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

    def _cache_fear_greed_data(self, df: pd.DataFrame):
        """
        Cache Fear & Greed Index data

        Args:
            df: DataFrame with Fear & Greed Index data
        """
        filename = f"fear_greed_{datetime.now().strftime('%Y%m%d')}.csv"
        filepath = os.path.join(self.cache_dir, filename)
        df.to_csv(filepath, index=False)
        logger.info(f"Cached Fear & Greed Index data to {filepath}")

    def _fetch_fear_greed_data(self, days: int) -> pd.DataFrame:
        """
        Fetch Fear & Greed Index data

        Args:
            days: Number of days of data to fetch

        Returns:
            DataFrame with Fear & Greed Index data
        """
        # In a real implementation, this would fetch from alternative.me API
        # For now, return a minimal DataFrame with neutral values
        logger.warning("No real Fear & Greed Index data available. Using neutral values.")

        # Generate dates for the requested period
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')

        # Create a minimal DataFrame with neutral values
        data = []
        for date in date_range:
            data.append({
                'timestamp': date,
                'value': 50,  # Neutral value
                'classification': "Neutral",
                'value_classification': "50: Neutral"
            })

        return pd.DataFrame(data)


def get_sentiment_analyzer() -> SentimentAnalyzer:
    """Get a sentiment analyzer instance"""
    return SentimentAnalyzer()

def get_fear_greed_index() -> CryptoFearGreedIndex:
    """Get a Fear & Greed Index fetcher instance"""
    return CryptoFearGreedIndex()
