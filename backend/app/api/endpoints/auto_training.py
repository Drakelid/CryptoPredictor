from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Optional
from app.services.auto_training_service import auto_training_service
from app.utils.json_utils import json_dumps
import logging
import traceback

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/start")
async def start_auto_training(background_tasks: BackgroundTasks):
    """
    Start the automatic training process
    """
    try:
        # Start in background to avoid blocking the API
        background_tasks.add_task(auto_training_service.start_auto_training)
        return {"message": "Automatic training process started"}
    except Exception as e:
        logger.error(f"Error starting auto-training: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error starting auto-training: {str(e)}")

@router.post("/stop")
async def stop_auto_training():
    """
    Stop the automatic training process
    """
    try:
        auto_training_service.stop_auto_training()
        return {"message": "Automatic training process stopped"}
    except Exception as e:
        logger.error(f"Error stopping auto-training: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error stopping auto-training: {str(e)}")

@router.get("/status")
async def get_training_status(symbol: Optional[str] = None, model_type: Optional[str] = None):
    """
    Get the status of the automatic training process
    """
    try:
        status = auto_training_service.get_training_status(symbol, model_type)

        # Ensure the response is JSON serializable
        try:
            # Test JSON serialization with our custom encoder
            json_dumps(status)
        except (TypeError, ValueError) as json_error:
            logger.error(f"JSON serialization error: {str(json_error)}")
            # Return a simplified status if there's a serialization error
            return {"error": "Could not serialize full status", "message": str(json_error)}

        return status
    except Exception as e:
        logger.error(f"Error getting training status: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error getting training status: {str(e)}")

@router.get("/performance")
async def get_model_performance(symbol: Optional[str] = None, model_type: Optional[str] = None):
    """
    Get performance metrics for models
    """
    try:
        performance = auto_training_service.get_model_performance(symbol, model_type)

        # Ensure the response is JSON serializable
        try:
            # Test JSON serialization with our custom encoder
            json_dumps(performance)
        except (TypeError, ValueError) as json_error:
            logger.error(f"JSON serialization error: {str(json_error)}")
            # Return a simplified status if there's a serialization error
            return {"error": "Could not serialize full performance data", "message": str(json_error)}

        return performance
    except Exception as e:
        logger.error(f"Error getting model performance: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error getting model performance: {str(e)}")

@router.post("/force-train/{symbol}/{model_type}")
async def force_train_model(symbol: str, model_type: str, background_tasks: BackgroundTasks):
    """
    Force training of a specific model
    """
    try:
        # Run in background to avoid blocking the API
        background_tasks.add_task(auto_training_service.force_train_model, symbol, model_type)
        return {"message": f"Training of {symbol} model using {model_type} started"}
    except Exception as e:
        logger.error(f"Error forcing model training: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error forcing model training: {str(e)}")

@router.post("/force-evaluate/{symbol}/{model_type}")
async def force_evaluate_model(symbol: str, model_type: str, background_tasks: BackgroundTasks):
    """
    Force evaluation of a specific model
    """
    try:
        # Run in background to avoid blocking the API
        background_tasks.add_task(auto_training_service.force_evaluate_model, symbol, model_type)
        return {"message": f"Evaluation of {symbol} model using {model_type} started"}
    except Exception as e:
        logger.error(f"Error forcing model evaluation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error forcing model evaluation: {str(e)}")

@router.post("/force-update/{symbol}/{model_type}")
async def force_update_model(symbol: str, model_type: str, background_tasks: BackgroundTasks):
    """
    Force update of a specific model with new data
    """
    try:
        # Run in background to avoid blocking the API
        background_tasks.add_task(auto_training_service.force_update_model, symbol, model_type)
        return {"message": f"Update of {symbol} model using {model_type} started"}
    except Exception as e:
        logger.error(f"Error forcing model update: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error forcing model update: {str(e)}")
