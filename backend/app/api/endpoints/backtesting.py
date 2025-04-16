from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
import logging

from app.services.backtesting_service import BacktestingService

router = APIRouter()
logger = logging.getLogger(__name__)

# Create service instance
backtesting_service = BacktestingService()

@router.post("/run")
async def run_backtest(
    symbol: str = Query(..., description="Cryptocurrency symbol (e.g., BTC, ETH)"),
    model_type: str = Query(..., description="Model type (lstm, gru, xgboost, lightgbm)"),
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
    initial_capital: float = Query(10000.0, description="Initial capital"),
    position_size: float = Query(0.1, description="Position size (0-1)")
):
    """
    Run a backtest for a trading strategy based on model predictions
    """
    try:
        result = backtesting_service.run_backtest(
            symbol=symbol,
            model_type=model_type,
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
            position_size=position_size
        )
        return result
    except Exception as e:
        logger.error(f"Error running backtest: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/strategies")
async def get_strategies():
    """
    Get available backtesting strategies
    """
    try:
        strategies = backtesting_service.get_strategies()
        return strategies
    except Exception as e:
        logger.error(f"Error getting strategies: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/results")
@router.get("/history")  # Add an alias for the frontend
async def get_backtest_results(
    symbol: Optional[str] = Query(None, description="Cryptocurrency symbol (e.g., BTC, ETH)"),
    model_type: Optional[str] = Query(None, description="Model type (lstm, gru, xgboost, lightgbm)"),
    limit: int = Query(10, description="Number of results to return")
):
    """
    Get historical backtest results
    """
    try:
        results = backtesting_service.get_backtest_results(
            symbol=symbol,
            model_type=model_type,
            limit=limit
        )
        return results
    except Exception as e:
        logger.error(f"Error getting backtest results: {e}")
        raise HTTPException(status_code=500, detail=str(e))
