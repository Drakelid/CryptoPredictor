from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
import logging

from app.schemas.data import TrainingInput, TrainingResult
from app.services.training_service import TrainingService

router = APIRouter()
logger = logging.getLogger(__name__)

# Create service instance
training_service = TrainingService()

@router.post("/train", response_model=TrainingResult)
async def train_model(
    input_data: TrainingInput
):
    """
    Train a new model
    """
    try:
        # Log the training request
        logger.info(f"Training model request received: {input_data.dict()}")

        # Validate input parameters
        if input_data.lookback <= 0:
            raise ValueError(f"lookback must be positive, got {input_data.lookback}")
        if input_data.horizon <= 0:
            raise ValueError(f"horizon must be positive, got {input_data.horizon}")
        if input_data.epochs <= 0:
            raise ValueError(f"epochs must be positive, got {input_data.epochs}")
        if input_data.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {input_data.batch_size}")

        # Call the training service
        result = training_service.train_model(
            symbol=input_data.symbol,
            model_type=input_data.model_type,
            lookback=input_data.lookback,
            horizon=input_data.horizon,
            epochs=input_data.epochs,
            batch_size=input_data.batch_size,
            hyperparameter_tuning=input_data.hyperparameter_tuning
        )

        # Log success
        logger.info(f"Model training successful for {input_data.symbol} using {input_data.model_type}")
        return result
    except ValueError as e:
        # Handle validation errors
        logger.error(f"Validation error in training model: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Handle other errors
        logger.error(f"Error training model: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/info", response_model=List[TrainingResult])
async def get_model_info(
    symbol: Optional[str] = Query(None, description="Cryptocurrency symbol (e.g., BTC, ETH)"),
    model_type: Optional[str] = Query(None, description="Model type (lstm, gru, xgboost, lightgbm)")
):
    """
    Get information about available models
    """
    try:
        results = training_service.get_model_info(
            symbol=symbol,
            model_type=model_type
        )
        return results
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{symbol}/{model_type}")
async def delete_model(
    symbol: str,
    model_type: str
):
    """
    Delete a model
    """
    try:
        success = training_service.delete_model(
            symbol=symbol,
            model_type=model_type
        )
        if success:
            return {"message": f"Model {model_type} for {symbol} deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail=f"Model {model_type} for {symbol} not found")
    except Exception as e:
        logger.error(f"Error deleting model: {e}")
        raise HTTPException(status_code=500, detail=str(e))
