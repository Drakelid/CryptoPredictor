import numpy as np
import pandas as pd
import os
import pickle
import logging
from typing import Dict, List, Tuple, Optional, Union, Any

from river import linear_model, preprocessing, compose, metrics, ensemble
import faiss
import numpy as np

from app.core.config import settings

logger = logging.getLogger(__name__)

class OnlineLearningModel:
    """Class for online learning models using River"""
    
    def __init__(
        self,
        model_type: str = "linear",
        model_path: Optional[str] = None
    ):
        """
        Initialize online learning model
        
        Args:
            model_type: Type of model ('linear', 'tree', 'ensemble')
            model_path: Path to saved model (if loading existing model)
        """
        self.model_type = model_type.lower()
        self.model_path = model_path
        self.model = None
        self.scaler = None
        self.metrics = metrics.RegressionMetrics()
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            self._build_model()
    
    def _build_model(self):
        """Build the online learning model"""
        # Create scaler
        self.scaler = preprocessing.StandardScaler()
        
        # Create model based on type
        if self.model_type == "linear":
            model = linear_model.LinearRegression(
                optimizer=linear_model.optimizers.SGD(0.01),
                loss=linear_model.losses.SquaredError(),
                l2=0.01
            )
        elif self.model_type == "tree":
            model = ensemble.AdaptiveRandomForestRegressor(
                n_models=10,
                seed=42
            )
        elif self.model_type == "ensemble":
            model = ensemble.SoftImpute()
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        # Create pipeline
        self.model = compose.Pipeline(
            ('scale', self.scaler),
            ('model', model)
        )
        
        logger.info(f"Built online learning model of type {self.model_type}")
    
    def update(self, x: Dict[str, float], y: float) -> Dict[str, float]:
        """
        Update the model with a single observation
        
        Args:
            x: Dictionary of features
            y: Target value
        
        Returns:
            Dictionary with updated metrics
        """
        if self.model is None:
            self._build_model()
        
        # Make prediction before learning
        y_pred = self.model.predict_one(x)
        
        # Update metrics if prediction is not None
        if y_pred is not None:
            self.metrics.update(y, y_pred)
        
        # Learn from the observation
        self.model.learn_one(x, y)
        
        # Return current metrics
        return {
            'MAE': self.metrics.MAE.get(),
            'RMSE': self.metrics.RMSE.get(),
            'R2': self.metrics.R2.get()
        }
    
    def predict(self, x: Dict[str, float]) -> float:
        """
        Make a prediction for a single observation
        
        Args:
            x: Dictionary of features
        
        Returns:
            Predicted value
        """
        if self.model is None:
            raise ValueError("Model not initialized. Call update() or load_model() first.")
        
        return self.model.predict_one(x)
    
    def save_model(self, path: str):
        """
        Save model to disk
        
        Args:
            path: Path to save the model
        """
        if self.model is None:
            raise ValueError("Model not initialized. Call update() first.")
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save model
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)
        
        # Save metrics
        metrics_path = os.path.join(os.path.dirname(path), f"{os.path.basename(path)}_metrics.pkl")
        with open(metrics_path, 'wb') as f:
            pickle.dump(self.metrics, f)
        
        self.model_path = path
        logger.info(f"Saved model to {path} and metrics to {metrics_path}")
    
    def load_model(self, path: str):
        """
        Load model from disk
        
        Args:
            path: Path to load the model from
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        
        # Load model
        with open(path, 'rb') as f:
            self.model = pickle.load(f)
        
        # Load metrics if available
        metrics_path = os.path.join(os.path.dirname(path), f"{os.path.basename(path)}_metrics.pkl")
        if os.path.exists(metrics_path):
            with open(metrics_path, 'rb') as f:
                self.metrics = pickle.load(f)
        
        self.model_path = path
        logger.info(f"Loaded model from {path}")


class VectorStore:
    """Class for storing and retrieving similar patterns using FAISS"""
    
    def __init__(
        self,
        dimension: int = 30,
        store_path: Optional[str] = None
    ):
        """
        Initialize vector store
        
        Args:
            dimension: Dimension of vectors to store
            store_path: Path to saved vector store (if loading existing store)
        """
        self.dimension = dimension
        self.store_path = store_path
        self.index = None
        self.patterns = []
        self.targets = []
        
        if store_path and os.path.exists(store_path):
            self.load_store(store_path)
        else:
            self._build_store()
    
    def _build_store(self):
        """Build the vector store"""
        # Create FAISS index
        self.index = faiss.IndexFlatL2(self.dimension)
        logger.info(f"Built vector store with dimension {self.dimension}")
    
    def add_pattern(self, pattern: np.ndarray, target: np.ndarray):
        """
        Add a pattern to the vector store
        
        Args:
            pattern: Pattern vector (features)
            target: Target vector (values to predict)
        """
        if self.index is None:
            self._build_store()
        
        # Ensure pattern is 2D
        if len(pattern.shape) == 1:
            pattern = pattern.reshape(1, -1)
        
        # Add to index
        self.index.add(pattern.astype(np.float32))
        
        # Store pattern and target
        self.patterns.append(pattern)
        self.targets.append(target)
        
        logger.debug(f"Added pattern to vector store. Total patterns: {len(self.patterns)}")
    
    def find_similar_patterns(self, pattern: np.ndarray, k: int = 5) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Find similar patterns in the vector store
        
        Args:
            pattern: Pattern vector to find similar patterns for
            k: Number of similar patterns to return
        
        Returns:
            Tuple of (similar patterns, corresponding targets)
        """
        if self.index is None or self.index.ntotal == 0:
            logger.warning("Vector store is empty. No similar patterns found.")
            return [], []
        
        # Ensure pattern is 2D
        if len(pattern.shape) == 1:
            pattern = pattern.reshape(1, -1)
        
        # Search for similar patterns
        distances, indices = self.index.search(pattern.astype(np.float32), min(k, self.index.ntotal))
        
        # Get corresponding patterns and targets
        similar_patterns = [self.patterns[i] for i in indices[0]]
        similar_targets = [self.targets[i] for i in indices[0]]
        
        return similar_patterns, similar_targets
    
    def save_store(self, path: str):
        """
        Save vector store to disk
        
        Args:
            path: Path to save the vector store
        """
        if self.index is None:
            raise ValueError("Vector store not initialized.")
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, path)
        
        # Save patterns and targets
        data_path = os.path.join(os.path.dirname(path), f"{os.path.basename(path)}_data.pkl")
        with open(data_path, 'wb') as f:
            pickle.dump((self.patterns, self.targets), f)
        
        self.store_path = path
        logger.info(f"Saved vector store to {path} and data to {data_path}")
    
    def load_store(self, path: str):
        """
        Load vector store from disk
        
        Args:
            path: Path to load the vector store from
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Vector store file not found: {path}")
        
        # Load FAISS index
        self.index = faiss.read_index(path)
        self.dimension = self.index.d
        
        # Load patterns and targets
        data_path = os.path.join(os.path.dirname(path), f"{os.path.basename(path)}_data.pkl")
        if os.path.exists(data_path):
            with open(data_path, 'rb') as f:
                self.patterns, self.targets = pickle.load(f)
        
        self.store_path = path
        logger.info(f"Loaded vector store from {path} with {self.index.ntotal} patterns")
