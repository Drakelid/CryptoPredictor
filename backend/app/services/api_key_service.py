import logging
from app.core.config import settings

logger = logging.getLogger(__name__)

class APIKeyService:
    """Simple service to store API keys in memory"""

    def set_keys(
        self,
        coinmarketcap_api_key: str | None = None,
        binance_api_key: str | None = None,
        binance_api_secret: str | None = None,
        news_api_key: str | None = None,
    ):
        if coinmarketcap_api_key is not None:
            settings.COINMARKETCAP_API_KEY = coinmarketcap_api_key
        if binance_api_key is not None:
            settings.BINANCE_API_KEY = binance_api_key
        if binance_api_secret is not None:
            settings.BINANCE_API_SECRET = binance_api_secret
        if news_api_key is not None:
            settings.NEWS_API_KEY = news_api_key
        return self.get_status()

    def get_status(self):
        return {
            "coinmarketcap_api_key": bool(settings.COINMARKETCAP_API_KEY),
            "binance_api_key": bool(settings.BINANCE_API_KEY),
            "binance_api_secret": bool(settings.BINANCE_API_SECRET),
            "news_api_key": bool(settings.NEWS_API_KEY),
        }

api_key_service = APIKeyService()
