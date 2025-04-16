from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

class BacktestInput(BaseModel):
    """Input for backtesting"""
    symbol: str = Field(..., description="Cryptocurrency symbol (e.g., BTC, ETH)")
    model_type: str = Field(..., description="Model type (lstm, gru, xgboost, lightgbm)")
    start_date: Optional[str] = Field(None, description="Start date (YYYY-MM-DD)")
    end_date: Optional[str] = Field(None, description="End date (YYYY-MM-DD)")
    initial_capital: float = Field(10000.0, description="Initial capital")
    position_size: float = Field(0.1, description="Position size (0-1)")

class BacktestResult(BaseModel):
    """Result of backtesting"""
    symbol: str
    model_type: str
    start_date: str
    end_date: str
    initial_capital: float
    final_value: float
    roi: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    trades: int

class Strategy(BaseModel):
    """Backtesting strategy"""
    name: str
    description: str
