from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Union
from datetime import datetime

class PredictionInput(BaseModel):
    """Schema for prediction input"""
    symbol: str = Field(..., description="Cryptocurrency symbol (e.g., BTC, ETH)")
    model_type: str = Field(..., description="Model type (lstm, gru, xgboost, lightgbm)")
    horizon: int = Field(7, description="Number of days to predict ahead")
    confidence_interval: bool = Field(False, description="Whether to include confidence intervals")
    use_ensemble: bool = Field(False, description="Whether to use an ensemble of available models")

    model_config = {
        'protected_namespaces': (),
        'json_schema_extra': {
            "example": {
                "symbol": "BTC",
                "model_type": "lstm",
                "horizon": 7,
                "confidence_interval": True,
                "use_ensemble": False
            }
        }
    }

class PredictionResult(BaseModel):
    """Schema for prediction result"""
    symbol: str
    model_type: str
    prediction_date: datetime
    horizon: int
    values: List[float]
    timestamps: List[datetime]
    confidence_lower: Optional[List[float]] = None
    confidence_upper: Optional[List[float]] = None

    model_config = {
        'protected_namespaces': (),
        'json_schema_extra': {
            "example": {
                "symbol": "BTC",
                "model_type": "lstm",
                "prediction_date": "2023-01-01T00:00:00",
                "horizon": 7,
                "values": [50000.0, 51000.0, 52000.0, 51500.0, 52500.0, 53000.0, 52000.0],
                "timestamps": [
                    "2023-01-02T00:00:00",
                    "2023-01-03T00:00:00",
                    "2023-01-04T00:00:00",
                    "2023-01-05T00:00:00",
                    "2023-01-06T00:00:00",
                    "2023-01-07T00:00:00",
                    "2023-01-08T00:00:00"
                ],
                "confidence_lower": [49000.0, 49500.0, 50000.0, 49500.0, 50500.0, 51000.0, 50000.0],
                "confidence_upper": [51000.0, 52500.0, 54000.0, 53500.0, 54500.0, 55000.0, 54000.0]
            }
        }
    }

class ExplanationInput(BaseModel):
    """Schema for explanation input"""
    symbol: str = Field(..., description="Cryptocurrency symbol (e.g., BTC, ETH)")
    model_type: str = Field(..., description="Model type (lstm, gru, xgboost, lightgbm)")
    prediction_id: Optional[str] = Field(None, description="ID of prediction to explain")

    model_config = {
        'protected_namespaces': (),
        'json_schema_extra': {
            "example": {
                "symbol": "BTC",
                "model_type": "xgboost",
                "prediction_id": "123456"
            }
        }
    }

class FeatureImportance(BaseModel):
    """Schema for feature importance"""
    feature: str
    importance: float

    model_config = {
        'protected_namespaces': (),
        'json_schema_extra': {
            "example": {
                "feature": "rsi_14",
                "importance": 0.25
            }
        }
    }

class ExplanationResult(BaseModel):
    """Schema for explanation result"""
    symbol: str
    model_type: str
    prediction_date: datetime
    feature_importance: List[FeatureImportance]
    shap_values: Optional[Dict[str, List[float]]] = None

    model_config = {
        'protected_namespaces': (),
        'json_schema_extra': {
            "example": {
                "symbol": "BTC",
                "model_type": "xgboost",
                "prediction_date": "2023-01-01T00:00:00",
                "feature_importance": [
                    {"feature": "rsi_14", "importance": 0.25},
                    {"feature": "macd_line", "importance": 0.20},
                    {"feature": "bb_percent_b", "importance": 0.15},
                    {"feature": "sma_50_200_cross", "importance": 0.10},
                    {"feature": "volume", "importance": 0.05}
                ],
                "shap_values": {
                    "rsi_14": [0.15, 0.25, 0.30, 0.20, 0.10, 0.05, 0.15],
                    "macd_line": [0.10, 0.15, 0.25, 0.20, 0.15, 0.10, 0.05]
                }
            }
        }
    }
