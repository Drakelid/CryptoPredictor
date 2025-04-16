from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from datetime import datetime

class ModelPerformance(BaseModel):
    rmse: float
    mae: float
    r2: float
    accuracy_trend: List[float] = []
    last_updated: Optional[datetime] = None

class ModelStatus(BaseModel):
    model_exists: bool
    last_training: Optional[datetime] = None
    last_evaluation: Optional[datetime] = None
    last_update: Optional[datetime] = None
    performance: Dict[str, Any] = {}

class TrainingStatus(BaseModel):
    auto_training_enabled: bool
    training_interval_hours: int
    evaluation_interval_hours: int
    continuous_learning_interval_hours: int
    models: Dict[str, Dict[str, ModelStatus]]
