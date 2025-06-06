from fastapi import APIRouter, HTTPException
import logging

from app.schemas.api_key import APIKeyUpdate, APIKeyStatus
from app.services.api_key_service import api_key_service

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("", response_model=APIKeyStatus)
async def update_keys(keys: APIKeyUpdate):
    """Set API keys used by the application"""
    try:
        return api_key_service.set_keys(
            coinmarketcap_api_key=keys.coinmarketcap_api_key,
            binance_api_key=keys.binance_api_key,
            binance_api_secret=keys.binance_api_secret,
            news_api_key=keys.news_api_key,
        )
    except Exception as e:
        logger.error(f"Error setting API keys: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("", response_model=APIKeyStatus)
async def get_key_status():
    """Retrieve which API keys have been set"""
    try:
        return api_key_service.get_status()
    except Exception as e:
        logger.error(f"Error getting API key status: {e}")
        raise HTTPException(status_code=500, detail=str(e))
