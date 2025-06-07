from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
import logging

from app.schemas.prediction import PredictionInput, PredictionResult, ExplanationInput, ExplanationResult
from app.services.prediction_service import PredictionService
from app.services.explanation_service import ExplanationService

router = APIRouter()
logger = logging.getLogger(__name__)

# Create service instances
prediction_service = PredictionService()
explanation_service = ExplanationService()

@router.post("/predict", response_model=PredictionResult)
async def predict(
    input_data: PredictionInput
):
    """
    Get price prediction for a cryptocurrency
    """
    try:
        # Debug logging
        logger.info(f"Prediction request: symbol={input_data.symbol}, model_type={input_data.model_type}, horizon={input_data.horizon}")

        # Force model_type to lowercase
        model_type = input_data.model_type.lower() if input_data.model_type else 'lstm'

        # Special handling for XGBoost
        if model_type == 'xgboost':
            logger.info("XGBoost model detected in API endpoint. Using fixed feature count: 1610")

        # Only use real trained models, no mock data
        result = prediction_service.predict(
            symbol=input_data.symbol,
            model_type=model_type,  # Use the lowercase model_type
            horizon=input_data.horizon,
            confidence_interval=input_data.confidence_interval,
            use_ensemble=input_data.use_ensemble
        )
        return result
    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/history", response_model=List[PredictionResult])
async def get_prediction_history(
    symbol: str = Query(..., description="Cryptocurrency symbol (e.g., BTC, ETH)"),
    model_type: Optional[str] = Query(None, description="Model type (lstm, gru, xgboost, lightgbm)"),
    limit: int = Query(10, description="Number of predictions to return")
):
    """
    Get historical predictions for a cryptocurrency
    """
    try:
        results = prediction_service.get_prediction_history(
            symbol=symbol,
            model_type=model_type,
            limit=limit
        )
        return results
    except Exception as e:
        logger.error(f"Error getting prediction history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/explain", response_model=ExplanationResult)
async def explain_prediction(
    input_data: ExplanationInput
):
    """
    Get explanation for a prediction
    """
    try:
        result = explanation_service.explain(
            symbol=input_data.symbol,
            model_type=input_data.model_type,
            prediction_id=input_data.prediction_id
        )
        return result
    except Exception as e:
        logger.error(f"Error explaining prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))
