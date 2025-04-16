from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from datetime import datetime

class SocialSentiment(BaseModel):
    twitter: float
    reddit: float
    overall: float
    change_24h: float

class NewsSentiment(BaseModel):
    positive_articles: int
    negative_articles: int
    neutral_articles: int
    overall: float
    change_24h: float

class FearGreedIndex(BaseModel):
    value: int
    classification: str
    change_24h: int

class MarketIndicators(BaseModel):
    volume_change: float
    volatility: float
    momentum: float

class HistoricalSentiment(BaseModel):
    date: str
    social: float
    news: float
    fear_greed: int

class SentimentResponse(BaseModel):
    symbol: str
    social_sentiment: SocialSentiment
    news_sentiment: NewsSentiment
    fear_greed_index: FearGreedIndex
    market_indicators: MarketIndicators
    historical: List[HistoricalSentiment]
    last_updated: Optional[datetime] = None

class SentimentRequest(BaseModel):
    text: str
