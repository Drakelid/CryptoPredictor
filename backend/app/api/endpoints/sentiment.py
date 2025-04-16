from fastapi import APIRouter, HTTPException, Depends, Query
from typing import List, Dict, Any, Optional
from app.schemas.sentiment import SentimentResponse, SentimentRequest
from app.services.sentiment_service import SentimentService
from app.core.config import settings
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

sentiment_service = SentimentService()

@router.get("/{symbol}", response_model=SentimentResponse)
async def get_sentiment(symbol: str):
    """
    Get sentiment data for a specific cryptocurrency
    """
    try:
        sentiment_data = sentiment_service.get_sentiment(symbol)
        return sentiment_data
    except Exception as e:
        logger.error(f"Error getting sentiment data for {symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting sentiment data: {str(e)}")

@router.post("/refresh/{symbol}", response_model=SentimentResponse)
async def refresh_sentiment(symbol: str):
    """
    Refresh sentiment data for a specific cryptocurrency
    """
    try:
        sentiment_data = sentiment_service.refresh_sentiment(symbol)
        return sentiment_data
    except Exception as e:
        logger.error(f"Error refreshing sentiment data for {symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error refreshing sentiment data: {str(e)}")

@router.post("/analyze", response_model=Dict[str, float])
async def analyze_sentiment(request: SentimentRequest):
    """
    Analyze text for sentiment
    """
    try:
        sentiment_score = sentiment_service.analyze_text(request.text)
        return {"score": sentiment_score}
    except Exception as e:
        logger.error(f"Error analyzing sentiment: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error analyzing sentiment: {str(e)}")
