from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Optional, List, Dict, Any
from app.utils.prediction_feedback import prediction_feedback_system
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/metrics")
async def get_prediction_metrics(symbol: Optional[str] = None, model_type: Optional[str] = None):
    """
    Get prediction metrics
    
    Args:
        symbol: Cryptocurrency symbol (optional)
        model_type: Type of model (optional)
    """
    try:
        metrics = prediction_feedback_system.get_prediction_metrics(symbol, model_type)
        return metrics
    except Exception as e:
        logger.error(f"Error getting prediction metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting prediction metrics: {str(e)}")

@router.get("/history/{symbol}")
async def get_prediction_history(symbol: str, model_type: Optional[str] = None, limit: int = 10):
    """
    Get prediction history for a symbol
    
    Args:
        symbol: Cryptocurrency symbol
        model_type: Type of model (optional)
        limit: Maximum number of predictions to return
    """
    try:
        history = prediction_feedback_system.get_prediction_history(symbol, model_type, limit)
        return history
    except Exception as e:
        logger.error(f"Error getting prediction history: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting prediction history: {str(e)}")

@router.post("/process")
async def process_feedback(background_tasks: BackgroundTasks, max_days_old: int = 30):
    """
    Process feedback for all stored predictions
    
    Args:
        max_days_old: Maximum age of predictions to process (in days)
    """
    try:
        # Run in background to avoid blocking the API
        background_tasks.add_task(prediction_feedback_system.process_feedback, max_days_old)
        return {"message": "Feedback processing started"}
    except Exception as e:
        logger.error(f"Error processing feedback: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing feedback: {str(e)}")

@router.get("/status")
async def get_feedback_status():
    """
    Get feedback system status
    """
    try:
        # Count predictions by symbol and model type
        feedback_dir = prediction_feedback_system.feedback_dir
        import os
        
        status = {
            "metrics_available": os.path.exists(prediction_feedback_system.metrics_file),
            "symbols": {},
            "total_predictions": 0,
            "processed_predictions": 0
        }
        
        # Check if feedback directory exists
        if not os.path.exists(feedback_dir):
            return status
        
        # Count predictions by symbol
        for symbol_dir in os.listdir(feedback_dir):
            symbol_path = os.path.join(feedback_dir, symbol_dir)
            
            # Skip if not a directory or is the metrics file
            if not os.path.isdir(symbol_path) or symbol_dir == 'prediction_metrics.json':
                continue
            
            symbol = symbol_dir.upper()
            status["symbols"][symbol] = {
                "total_predictions": 0,
                "processed_predictions": 0,
                "models": {}
            }
            
            # Count predictions by model type
            for prediction_file in os.listdir(symbol_path):
                if not prediction_file.endswith('.json'):
                    continue
                
                prediction_path = os.path.join(symbol_path, prediction_file)
                
                try:
                    # Load prediction record
                    import json
                    with open(prediction_path, 'r') as f:
                        prediction_record = json.load(f)
                    
                    model_type = prediction_record.get('model_type', 'unknown')
                    processed = prediction_record.get('feedback_processed', False)
                    
                    # Update counts
                    status["total_predictions"] += 1
                    status["symbols"][symbol]["total_predictions"] += 1
                    
                    if processed:
                        status["processed_predictions"] += 1
                        status["symbols"][symbol]["processed_predictions"] += 1
                    
                    # Update model type counts
                    if model_type not in status["symbols"][symbol]["models"]:
                        status["symbols"][symbol]["models"][model_type] = {
                            "total_predictions": 0,
                            "processed_predictions": 0
                        }
                    
                    status["symbols"][symbol]["models"][model_type]["total_predictions"] += 1
                    
                    if processed:
                        status["symbols"][symbol]["models"][model_type]["processed_predictions"] += 1
                        
                except Exception as e:
                    logger.error(f"Error loading prediction file {prediction_file}: {str(e)}")
        
        return status
    except Exception as e:
        logger.error(f"Error getting feedback status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting feedback status: {str(e)}")
