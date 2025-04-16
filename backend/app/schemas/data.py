from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from datetime import datetime

class DataFetchInput(BaseModel):
    """Schema for data fetch input"""
    symbol: str = Field(..., description="Cryptocurrency symbol (e.g., BTC, ETH)")
    source: str = Field("coingecko", description="Data source (coingecko, binance)")
    days: int = Field(365, description="Number of days of historical data to fetch")

    model_config = {
        'protected_namespaces': (),
        'json_schema_extra': {
            "example": {
                "symbol": "BTC",
                "source": "coingecko",
                "days": 365
            }
        }
    }

class DataUploadInput(BaseModel):
    """Schema for data upload input"""
    symbol: str = Field(..., description="Cryptocurrency symbol (e.g., BTC, ETH)")
    source: str = Field(..., description="Data source name")

    model_config = {
        'protected_namespaces': (),
        'json_schema_extra': {
            "example": {
                "symbol": "BTC",
                "source": "custom"
            }
        }
    }

class DataInfo(BaseModel):
    """Schema for data information"""
    symbol: str
    source: str
    start_date: datetime
    end_date: datetime
    rows: int
    columns: List[str]

    model_config = {
        'protected_namespaces': (),
        'json_schema_extra': {
            "example": {
                "symbol": "BTC",
                "source": "coingecko",
                "start_date": "2022-01-01T00:00:00",
                "end_date": "2023-01-01T00:00:00",
                "rows": 365,
                "columns": ["timestamp", "price", "volume", "market_cap"]
            }
        }
    }

class TrainingInput(BaseModel):
    """Schema for training input"""
    symbol: str = Field(..., description="Cryptocurrency symbol (e.g., BTC, ETH)")
    model_type: str = Field(..., description="Model type (lstm, gru, xgboost, lightgbm)")
    lookback: int = Field(30, description="Number of days to look back")
    horizon: int = Field(7, description="Number of days to predict ahead")
    epochs: Optional[int] = Field(100, description="Number of epochs for DL models")
    batch_size: Optional[int] = Field(32, description="Batch size for DL models")
    hyperparameter_tuning: bool = Field(False, description="Whether to perform hyperparameter tuning")

    model_config = {
        'protected_namespaces': (),
        'json_schema_extra': {
            "example": {
                "symbol": "BTC",
                "model_type": "lstm",
                "lookback": 30,
                "horizon": 7,
                "epochs": 100,
                "batch_size": 32,
                "hyperparameter_tuning": False
            }
        }
    }

class TrainingResult(BaseModel):
    """Schema for training result"""
    symbol: str
    model_type: str
    training_date: datetime
    metrics: Dict[str, float]
    model_path: str

    model_config = {
        'protected_namespaces': (),
        'json_schema_extra': {
            "example": {
                "symbol": "BTC",
                "model_type": "lstm",
                "training_date": "2023-01-01T00:00:00",
                "metrics": {
                    "train_loss": 0.001,
                    "val_loss": 0.002,
                    "train_rmse": 100.0,
                    "val_rmse": 150.0
                },
                "model_path": "models/btc_lstm_20230101.h5"
            }
        }
    }
