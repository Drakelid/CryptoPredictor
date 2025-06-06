import logging
import random
import os
import json
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
from app.schemas.sentiment import SentimentResponse, SocialSentiment, NewsSentiment, FearGreedIndex, MarketIndicators, HistoricalSentiment
from app.core.config import settings

logger = logging.getLogger(__name__)

class SentimentService:
    """Service for sentiment analysis and market sentiment data using real-world APIs"""

    def __init__(self):
        """Initialize the sentiment service"""
        self.sentiment_data = {}  # Cache for sentiment data
        self.cache_dir = os.path.join(settings.DATA_DIR, 'sentiment_cache')
        os.makedirs(self.cache_dir, exist_ok=True)

        # API endpoints and keys
        self.fear_greed_api = "https://api.alternative.me/fng/"
        self.news_api = "https://cryptopanic.com/api/v1/posts/"
        self.news_api_key = settings.NEWS_API_KEY if hasattr(settings, 'NEWS_API_KEY') else None
        self.twitter_api = "https://api.senticrypt.com/v1/twitter.json"
        self.reddit_api = "https://api.senticrypt.com/v1/reddit.json"

        # Load cached data if available
        self._load_cached_data()

    def _load_cached_data(self):
        """Load cached sentiment data from disk"""
        try:
            # Check if cache directory exists
            if not os.path.exists(self.cache_dir):
                logger.info("Cache directory does not exist. Creating it.")
                os.makedirs(self.cache_dir, exist_ok=True)
                return

            # Load cached data for each symbol
            for filename in os.listdir(self.cache_dir):
                if filename.endswith('.json'):
                    symbol = filename.split('_')[0].upper()
                    file_path = os.path.join(self.cache_dir, filename)

                    try:
                        with open(file_path, 'r') as f:
                            data = json.load(f)

                        # Convert string dates back to datetime objects
                        if 'last_updated' in data and data['last_updated']:
                            data['last_updated'] = datetime.fromisoformat(data['last_updated'])

                        self.sentiment_data[symbol] = data
                        logger.info(f"Loaded cached sentiment data for {symbol}")
                    except Exception as e:
                        logger.error(f"Error loading cached data for {symbol}: {str(e)}")
        except Exception as e:
            logger.error(f"Error loading cached sentiment data: {str(e)}")

    def _save_cached_data(self, symbol: str):
        """Save sentiment data to disk"""
        try:
            if symbol not in self.sentiment_data:
                logger.warning(f"No sentiment data to cache for {symbol}")
                return

            # Create cache directory if it doesn't exist
            os.makedirs(self.cache_dir, exist_ok=True)

            # Prepare data for serialization
            data = self.sentiment_data[symbol].copy()

            # Convert datetime objects to strings
            if 'last_updated' in data and isinstance(data['last_updated'], datetime):
                data['last_updated'] = data['last_updated'].isoformat()

            # Save to file
            file_path = os.path.join(self.cache_dir, f"{symbol.lower()}_sentiment.json")
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)

            logger.info(f"Cached sentiment data for {symbol}")
        except Exception as e:
            logger.error(f"Error caching sentiment data for {symbol}: {str(e)}")

    def _fetch_fear_greed_index(self):
        """Fetch the Fear & Greed Index from alternative.me"""
        try:
            response = requests.get(self.fear_greed_api)
            if response.status_code == 200:
                data = response.json()
                if 'data' in data and len(data['data']) > 0:
                    current = data['data'][0]
                    yesterday = data['data'][1] if len(data['data']) > 1 else None

                    value = int(current['value'])
                    classification = current['value_classification']
                    change_24h = int(current['value']) - int(yesterday['value']) if yesterday else 0

                    return {
                        "value": value,
                        "classification": classification,
                        "change_24h": change_24h
                    }

            logger.warning("Failed to fetch Fear & Greed Index. Using neutral values.")
            return {
                "value": 50,
                "classification": "Neutral",
                "change_24h": 0
            }
        except Exception as e:
            logger.error(f"Error fetching Fear & Greed Index: {str(e)}")
            return {
                "value": 50,
                "classification": "Neutral",
                "change_24h": 0
            }

    def _fetch_news_sentiment(self, symbol: str):
        """Fetch news sentiment for a cryptocurrency"""
        try:
            params = {
                'currencies': symbol,
                'public': 'true'
            }

            if self.news_api_key:
                params['auth_token'] = self.news_api_key

            response = requests.get(self.news_api, params=params)
            if response.status_code == 200:
                data = response.json()
                if 'results' in data:
                    articles = data['results']

                    # Count positive, negative, and neutral articles
                    positive = sum(1 for a in articles if a.get('sentiment') == 'positive')
                    negative = sum(1 for a in articles if a.get('sentiment') == 'negative')
                    neutral = sum(1 for a in articles if a.get('sentiment') == 'neutral' or a.get('sentiment') is None)

                    total = positive + negative + neutral
                    if total > 0:
                        overall = positive / total
                    else:
                        overall = 0.5

                    # Get previous sentiment if available
                    prev_overall = 0.5
                    if symbol in self.sentiment_data and 'news_sentiment' in self.sentiment_data[symbol]:
                        prev_overall = self.sentiment_data[symbol]['news_sentiment'].get('overall', 0.5)

                    change_24h = overall - prev_overall

                    return {
                        "positive_articles": positive,
                        "negative_articles": negative,
                        "neutral_articles": neutral,
                        "overall": overall,
                        "change_24h": change_24h
                    }

            logger.warning(f"No real news sentiment data available for {symbol}. Using neutral values.")
            return {
                "positive_articles": 10,
                "negative_articles": 10,
                "neutral_articles": 10,
                "overall": 0.5,
                "change_24h": 0.0
            }
        except Exception as e:
            logger.error(f"Error fetching news sentiment for {symbol}: {str(e)}")
            return {
                "positive_articles": 10,
                "negative_articles": 10,
                "neutral_articles": 10,
                "overall": 0.5,
                "change_24h": 0.0
            }

    def _fetch_social_sentiment(self, symbol: str):
        """Fetch social media sentiment for a cryptocurrency"""
        try:
            # Try to fetch Twitter sentiment
            twitter_sentiment = 0.5
            try:
                response = requests.get(f"{self.twitter_api}?symbol={symbol}")
                if response.status_code == 200:
                    data = response.json()
                    if 'sentiment' in data:
                        twitter_sentiment = float(data['sentiment'])
            except Exception as e:
                logger.error(f"Error fetching Twitter sentiment for {symbol}: {str(e)}")

            # Try to fetch Reddit sentiment
            reddit_sentiment = 0.5
            try:
                response = requests.get(f"{self.reddit_api}?symbol={symbol}")
                if response.status_code == 200:
                    data = response.json()
                    if 'sentiment' in data:
                        reddit_sentiment = float(data['sentiment'])
            except Exception as e:
                logger.error(f"Error fetching Reddit sentiment for {symbol}: {str(e)}")

            # Calculate overall sentiment
            overall = (twitter_sentiment + reddit_sentiment) / 2

            # Get previous sentiment if available
            prev_overall = 0.5
            if symbol in self.sentiment_data and 'social_sentiment' in self.sentiment_data[symbol]:
                prev_overall = self.sentiment_data[symbol]['social_sentiment'].get('overall', 0.5)

            change_24h = overall - prev_overall

            return {
                "twitter": twitter_sentiment,
                "reddit": reddit_sentiment,
                "overall": overall,
                "change_24h": change_24h
            }
        except Exception as e:
            logger.error(f"Error fetching social sentiment for {symbol}: {str(e)}")
            return {
                "twitter": 0.5,
                "reddit": 0.5,
                "overall": 0.5,
                "change_24h": 0.0
            }

    def _fetch_market_indicators(self, symbol: str):
        """Fetch market indicators for a cryptocurrency"""
        try:
            # Use CoinMarketCap API to get market data
            url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/latest"
            headers = {"X-CMC_PRO_API_KEY": settings.COINMARKETCAP_API_KEY} if settings.COINMARKETCAP_API_KEY else {}
            response = requests.get(url, headers=headers, params={"symbol": symbol.upper()})
            if response.status_code == 200:
                data = response.json()
                market_data = data.get('data', {}).get(symbol.upper(), {}).get('quote', {}).get('USD', {})

                # Extract relevant market indicators
                volume_change = market_data.get('volume_change_24h', 0) / 100
                price_change = market_data.get('percent_change_24h', 0) / 100
                volatility = abs(price_change) * 2  # Simple volatility estimate

                # Calculate momentum (based on price changes over different time periods)
                momentum_factors = [
                    market_data.get('price_change_percentage_24h', 0),
                    market_data.get('price_change_percentage_7d', 0),
                    market_data.get('price_change_percentage_14d', 0),
                    market_data.get('price_change_percentage_30d', 0)
                ]

                # Convert to values between 0 and 1 (0.5 is neutral)
                momentum_values = [0.5 + (x / 200) for x in momentum_factors if x is not None]
                momentum = sum(momentum_values) / len(momentum_values) if momentum_values else 0.5

                return {
                    "volume_change": volume_change,
                    "volatility": volatility,
                    "momentum": momentum
                }

            logger.warning(f"No real market data available for {symbol}. Using neutral values.")
            return {
                "volume_change": 0.0,
                "volatility": 0.05,
                "momentum": 0.5
            }
        except Exception as e:
            logger.error(f"Error fetching market indicators for {symbol}: {str(e)}")
            return {
                "volume_change": 0.0,
                "volatility": 0.05,
                "momentum": 0.5
            }

    def _fetch_historical_sentiment(self, symbol: str):
        """Fetch historical sentiment data for a cryptocurrency"""
        try:
            # Get current sentiment data
            fear_greed_index = self._fetch_fear_greed_index()
            social_sentiment = self._fetch_social_sentiment(symbol)
            news_sentiment = self._fetch_news_sentiment(symbol)

            # Get previous historical data if available
            historical = []
            if symbol in self.sentiment_data and 'historical' in self.sentiment_data[symbol]:
                historical = self.sentiment_data[symbol]['historical']

            # Add today's data if not already present
            today = datetime.now().strftime("%Y-%m-%d")
            today_entry = None

            for entry in historical:
                if entry['date'] == today:
                    today_entry = entry
                    break

            if today_entry:
                # Update today's entry
                today_entry['social'] = social_sentiment['overall']
                today_entry['news'] = news_sentiment['overall']
                today_entry['fear_greed'] = fear_greed_index['value']
            else:
                # Add new entry for today
                historical.append({
                    "date": today,
                    "social": social_sentiment['overall'],
                    "news": news_sentiment['overall'],
                    "fear_greed": fear_greed_index['value']
                })

            # Keep only the last 30 days
            if len(historical) > 30:
                historical = historical[-30:]

            # Sort by date
            historical.sort(key=lambda x: x['date'])

            return historical
        except Exception as e:
            logger.error(f"Error fetching historical sentiment for {symbol}: {str(e)}")
            # Return empty list or minimal historical data
            return [
                {
                    "date": datetime.now().strftime("%Y-%m-%d"),
                    "social": 0.5,
                    "news": 0.5,
                    "fear_greed": 50
                }
            ]

    def get_sentiment(self, symbol: str) -> SentimentResponse:
        """Get sentiment data for a cryptocurrency"""
        symbol = symbol.upper()

        # Check if we have recent data in cache (less than 2 hours old)
        if symbol in self.sentiment_data:
            last_updated = self.sentiment_data[symbol].get('last_updated')
            if last_updated and (datetime.now() - last_updated).total_seconds() < 7200:  # 2 hours
                logger.info(f"Using cached sentiment data for {symbol}")
                return SentimentResponse(**self.sentiment_data[symbol])

        # Fetch fresh sentiment data
        try:
            # Fetch sentiment data from various sources
            fear_greed_index = self._fetch_fear_greed_index()
            social_sentiment = self._fetch_social_sentiment(symbol)
            news_sentiment = self._fetch_news_sentiment(symbol)
            market_indicators = self._fetch_market_indicators(symbol)
            historical = self._fetch_historical_sentiment(symbol)

            # Combine all data
            sentiment_data = {
                "symbol": symbol,
                "social_sentiment": social_sentiment,
                "news_sentiment": news_sentiment,
                "fear_greed_index": fear_greed_index,
                "market_indicators": market_indicators,
                "historical": historical,
                "last_updated": datetime.now()
            }

            # Cache the data
            self.sentiment_data[symbol] = sentiment_data
            self._save_cached_data(symbol)

            return SentimentResponse(**sentiment_data)
        except Exception as e:
            logger.error(f"Error getting sentiment data for {symbol}: {str(e)}")

            # If we have cached data, use it even if it's old
            if symbol in self.sentiment_data:
                logger.info(f"Using old cached sentiment data for {symbol}")
                return SentimentResponse(**self.sentiment_data[symbol])

            # Otherwise, return neutral sentiment
            logger.warning(f"No sentiment data available for {symbol}. Using neutral values.")
            return SentimentResponse(
                symbol=symbol,
                social_sentiment={
                    "twitter": 0.5,
                    "reddit": 0.5,
                    "overall": 0.5,
                    "change_24h": 0.0,
                },
                news_sentiment={
                    "positive_articles": 10,
                    "negative_articles": 10,
                    "neutral_articles": 10,
                    "overall": 0.5,
                    "change_24h": 0.0,
                },
                fear_greed_index={
                    "value": 50,
                    "classification": "Neutral",
                    "change_24h": 0,
                },
                market_indicators={
                    "volume_change": 0.0,
                    "volatility": 0.05,
                    "momentum": 0.5,
                },
                historical=[
                    {
                        "date": datetime.now().strftime("%Y-%m-%d"),
                        "social": 0.5,
                        "news": 0.5,
                        "fear_greed": 50
                    }
                ],
                last_updated=datetime.now()
            )

    def refresh_sentiment(self, symbol: str) -> SentimentResponse:
        """Refresh sentiment data for a cryptocurrency"""
        symbol = symbol.upper()

        # Force a refresh by fetching fresh data from APIs
        try:
            # Fetch sentiment data from various sources
            fear_greed_index = self._fetch_fear_greed_index()
            social_sentiment = self._fetch_social_sentiment(symbol)
            news_sentiment = self._fetch_news_sentiment(symbol)
            market_indicators = self._fetch_market_indicators(symbol)
            historical = self._fetch_historical_sentiment(symbol)

            # Combine all data
            sentiment_data = {
                "symbol": symbol,
                "social_sentiment": social_sentiment,
                "news_sentiment": news_sentiment,
                "fear_greed_index": fear_greed_index,
                "market_indicators": market_indicators,
                "historical": historical,
                "last_updated": datetime.now()
            }

            # Cache the data
            self.sentiment_data[symbol] = sentiment_data
            self._save_cached_data(symbol)

            logger.info(f"Refreshed sentiment data for {symbol}")
            return SentimentResponse(**sentiment_data)
        except Exception as e:
            logger.error(f"Error refreshing sentiment data for {symbol}: {str(e)}")

            # If we have cached data, use it even if it's old
            if symbol in self.sentiment_data:
                logger.info(f"Using cached sentiment data for {symbol} after refresh failure")
                return SentimentResponse(**self.sentiment_data[symbol])

            # Otherwise, return neutral sentiment
            logger.warning(f"No sentiment data available for {symbol}. Using neutral values.")
            return SentimentResponse(
                symbol=symbol,
                social_sentiment={
                    "twitter": 0.5,
                    "reddit": 0.5,
                    "overall": 0.5,
                    "change_24h": 0.0,
                },
                news_sentiment={
                    "positive_articles": 10,
                    "negative_articles": 10,
                    "neutral_articles": 10,
                    "overall": 0.5,
                    "change_24h": 0.0,
                },
                fear_greed_index={
                    "value": 50,
                    "classification": "Neutral",
                    "change_24h": 0,
                },
                market_indicators={
                    "volume_change": 0.0,
                    "volatility": 0.05,
                    "momentum": 0.5,
                },
                historical=[
                    {
                        "date": datetime.now().strftime("%Y-%m-%d"),
                        "social": 0.5,
                        "news": 0.5,
                        "fear_greed": 50
                    }
                ],
                last_updated=datetime.now()
            )

    def analyze_text(self, text: str) -> float:
        """
        Analyze text for sentiment using NLP techniques
        """
        try:
            # Use TextBlob for sentiment analysis
            from textblob import TextBlob
            analysis = TextBlob(text)

            # TextBlob polarity is between -1 (negative) and 1 (positive)
            # Convert to 0-1 scale
            sentiment_score = (analysis.sentiment.polarity + 1) / 2

            return round(sentiment_score, 2)
        except Exception as e:
            logger.error(f"Error analyzing text sentiment: {str(e)}")

            # Fallback to simple keyword-based analysis
            positive_words = ["bullish", "up", "gain", "profit", "growth", "positive", "good", "great", "excellent", "increase"]
            negative_words = ["bearish", "down", "loss", "crash", "decline", "negative", "bad", "poor", "terrible", "decrease"]

            text = text.lower()

            positive_count = sum(1 for word in positive_words if word in text)
            negative_count = sum(1 for word in negative_words if word in text)

            total_count = positive_count + negative_count

            if total_count == 0:
                # No sentiment words found, return neutral
                return 0.5

            # Calculate sentiment score (0 to 1)
            sentiment_score = positive_count / total_count

            return round(sentiment_score, 2)

    def get_sentiment_for_prediction(self, symbol: str) -> Dict[str, float]:
        """
        Get sentiment data formatted for use in price predictions

        Returns a dictionary with sentiment features that can be used in prediction models
        """
        symbol = symbol.upper()

        # Get sentiment data
        try:
            # Try to get fresh or cached sentiment data
            sentiment_response = self.get_sentiment(symbol)

            # Extract the relevant sentiment features for prediction
            sentiment_features = {
                "social_sentiment": sentiment_response.social_sentiment.overall,
                "news_sentiment": sentiment_response.news_sentiment.overall,
                "fear_greed_index": sentiment_response.fear_greed_index.value / 100.0,  # Normalize to 0-1
                "market_momentum": sentiment_response.market_indicators.momentum,
                "market_volatility": sentiment_response.market_indicators.volatility
            }

            logger.info(f"Using real sentiment data for {symbol} prediction")
            return sentiment_features
        except Exception as e:
            logger.error(f"Error getting sentiment for prediction for {symbol}: {str(e)}")
            logger.warning(f"No real sentiment data available for {symbol}. Using neutral sentiment.")

            # Return neutral sentiment values as fallback
            return {
                "social_sentiment": 0.5,
                "news_sentiment": 0.5,
                "fear_greed_index": 0.5,  # Already normalized
                "market_momentum": 0.5,
                "market_volatility": 0.05
            }
