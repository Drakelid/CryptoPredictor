from pydantic import BaseModel, Field
from typing import Optional

class APIKeyUpdate(BaseModel):
    """Incoming API key values"""
    coingecko_api_key: Optional[str] = Field(None, description="CoinGecko API key")
    binance_api_key: Optional[str] = Field(None, description="Binance API key")
    binance_api_secret: Optional[str] = Field(None, description="Binance API secret")
    news_api_key: Optional[str] = Field(None, description="News API key")

class APIKeyStatus(BaseModel):
    """Whether each key is set"""
    coingecko_api_key: bool
    binance_api_key: bool
    binance_api_secret: bool
    news_api_key: bool
