import os
from pydantic import BaseModel
from typing import List, Dict, Optional, Union

class Settings(BaseModel):
    API_V1_STR: str = "/api"
    PROJECT_NAME: str = "CryptoPricer"

    # Data paths
    DATA_DIR: str = os.path.join(os.getcwd(), "..", "data")
    MODELS_DIR: str = os.path.join(os.getcwd(), "..", "models")

    # API Keys (in production, use environment variables)
    COINGECKO_API_KEY: Optional[str] = None
    BINANCE_API_KEY: Optional[str] = None
    BINANCE_API_SECRET: Optional[str] = None
    NEWS_API_KEY: Optional[str] = None  # For CryptoPanic API
    REDDIT_CLIENT_ID: Optional[str] = None
    REDDIT_CLIENT_SECRET: Optional[str] = None
    REDDIT_USER_AGENT: Optional[str] = "CryptoPricerBot/0.1"

    # Model settings
    DEFAULT_LOOKBACK_WINDOW: int = 30  # Days for historical data
    PREDICTION_HORIZON: int = 7  # Days to predict ahead
    SUPPORTED_CRYPTOCURRENCIES: List[str] = ["BTC", "ETH", "BNB", "XRP", "ADA"]
    SUPPORTED_MODELS: List[str] = [
        # Basic models
        "lstm", "gru", "xgboost", "lightgbm",
        # Simple linear regression fallback
        "linear",
        # Advanced models
        "bidirectional_lstm",
        "attention_lstm",
        "cnn_lstm",
        "transformer",
        "dual_attention"
        # Additional ensemble methods could be added here
    ]

    # Control whether prediction results should be randomly adjusted
    USE_RANDOM_PRICE_OVERRIDE: bool = False

    # Feature engineering
    TECHNICAL_INDICATORS: List[str] = [
        "rsi", "macd", "bollinger_bands", "sma", "ema", "atr", "obv"
    ]

    # Sentiment analysis
    INCLUDE_SENTIMENT: bool = True
    SENTIMENT_SOURCES: List[str] = ["news", "social", "fear_greed"]
    SENTIMENT_WEIGHT: float = 0.3  # Weight of sentiment in predictions

    # Training parameters
    DEFAULT_TRAIN_TEST_SPLIT: float = 0.8
    DEFAULT_BATCH_SIZE: int = 32
    DEFAULT_EPOCHS: int = 100
    DEFAULT_PATIENCE: int = 10  # For early stopping

    # Advanced training parameters
    USE_ADVANCED_FEATURES: bool = True
    USE_ENSEMBLE: bool = True
    USE_FEATURE_SELECTION: bool = True
    USE_ANOMALY_DETECTION: bool = True
    ANOMALY_DETECTION_METHOD: str = "isolation_forest"
    FEATURE_SELECTION_METHOD: str = "importance"
    HYPEROPT_N_TRIALS: int = 30
    TIME_CV_SPLITS: int = 5

    # Backtesting
    DEFAULT_INITIAL_CAPITAL: float = 10000.0
    DEFAULT_POSITION_SIZE: float = 0.1  # 10% of capital per trade

settings = Settings()
