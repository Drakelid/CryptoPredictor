import numpy as np
import pandas as pd
import os
import pickle
import json
import logging
from typing import List, Dict, Any, Tuple, Optional, Union
from sklearn.ensemble import VotingRegressor, StackingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb
from tensorflow.keras.models import load_model
import tensorflow as tf

from app.models.ml_models import XGBoostModel, LightGBMModel
from app.models.dl_models import LSTMModel, GRUModel
from app.models.advanced_models import AdvancedModelFactory

logger = logging.getLogger(__name__)

class KerasWrapper:
    """Wrapper for Keras models to use in scikit-learn ensembles"""

    def __init__(self, model_path: str, scaler_path: str):
        """
        Initialize wrapper

        Args:
            model_path: Path to Keras model
            scaler_path: Path to scaler
        """
        self.model = load_model(model_path)
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)

        # Get metadata
        metadata_path = f"{os.path.splitext(model_path)[0]}_metadata.json"
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)

        self.lookback = self.metadata.get('lookback', 30)
        self.horizon = self.metadata.get('horizon', 7)
        self.is_sequence_model = True

    def fit(self, X, y):
        """Dummy fit method for compatibility"""
        return self

    def predict(self, X):
        """
        Make predictions

        Args:
            X: Input features

        Returns:
            Predictions
        """
        # Reshape input for sequence models if needed
        if self.is_sequence_model:
            if len(X.shape) == 2:
                # Reshape to (samples, lookback, features)
                n_features = X.shape[1]
                X = X.reshape(-1, self.lookback, n_features // self.lookback)

        # Make predictions
        preds = self.model.predict(X)

        # Inverse transform predictions
        if hasattr(self.scaler, 'inverse_transform'):
            preds = self.scaler.inverse_transform(preds.reshape(-1, 1)).flatten()

        return preds


class EnsembleModel:
    """Ensemble model combining multiple base models"""

    def __init__(self, ensemble_type: str = 'voting'):
        """
        Initialize ensemble model

        Args:
            ensemble_type: Type of ensemble ('voting', 'stacking')
        """
        self.ensemble_type = ensemble_type
        self.model = None
        self.base_models = []
        self.feature_importance = None

    def add_model(self, model_name: str, model: Any):
        """
        Add a model to the ensemble

        Args:
            model_name: Name of the model
            model: Model instance
        """
        self.base_models.append((model_name, model))

    def build(self, meta_model=None):
        """
        Build the ensemble model

        Args:
            meta_model: Meta-model for stacking (optional)
        """
        if self.ensemble_type == 'voting':
            self.model = VotingRegressor(
                estimators=self.base_models,
                weights=None  # Equal weights initially
            )
        elif self.ensemble_type == 'stacking':
            if meta_model is None:
                meta_model = Ridge()

            self.model = StackingRegressor(
                estimators=self.base_models,
                final_estimator=meta_model,
                cv=TimeSeriesSplit(n_splits=5)
            )
        else:
            raise ValueError(f"Unknown ensemble type: {self.ensemble_type}")

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the ensemble model

        Args:
            X: Input features
            y: Target values
        """
        if self.model is None:
            self.build()

        self.model.fit(X, y)

        # Extract feature importance if available
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = self.model.feature_importances_
        elif self.ensemble_type == 'stacking' and hasattr(self.model.final_estimator_, 'coef_'):
            # For stacking, use meta-model coefficients as proxy for importance
            self.feature_importance = np.abs(self.model.final_estimator_.coef_)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions

        Args:
            X: Input features

        Returns:
            Predictions
        """
        if self.model is None:
            raise ValueError("Model not built. Call build() first.")

        return self.model.predict(X)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the model

        Args:
            X: Input features
            y: Target values

        Returns:
            Dictionary with evaluation metrics
        """
        y_pred = self.predict(X)

        return {
            'mse': mean_squared_error(y, y_pred),
            'rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'mae': mean_absolute_error(y, y_pred),
            'r2': r2_score(y, y_pred)
        }

    def save(self, path: str):
        """
        Save the model

        Args:
            path: Path to save the model
        """
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)

        # Save feature importance if available
        if self.feature_importance is not None:
            importance_path = f"{os.path.splitext(path)[0]}_importance.pkl"
            with open(importance_path, 'wb') as f:
                pickle.dump(self.feature_importance, f)

    def load(self, path: str):
        """
        Load the model

        Args:
            path: Path to load the model from
        """
        with open(path, 'rb') as f:
            self.model = pickle.load(f)

        # Load feature importance if available
        importance_path = f"{os.path.splitext(path)[0]}_importance.pkl"
        if os.path.exists(importance_path):
            with open(importance_path, 'rb') as f:
                self.feature_importance = pickle.load(f)


class TimeSeriesEnsemble:
    """Specialized ensemble for time series forecasting"""

    def __init__(self, models_config: List[Dict[str, Any]]):
        """
        Initialize time series ensemble

        Args:
            models_config: List of model configurations
        """
        self.models_config = models_config
        self.models = []
        self.weights = None
        self.feature_importance = None

    def build_models(self, input_shape: Tuple[int, int], output_shape: int):
        """
        Build all models in the ensemble

        Args:
            input_shape: Shape of input data
            output_shape: Number of output units
        """
        for config in self.models_config:
            model_type = config['type']
            hyperparams = config.get('hyperparams', {})

            if model_type in ['lstm', 'gru']:
                if model_type == 'lstm':
                    model = LSTMModel(input_shape, output_shape, hyperparams)
                else:
                    model = GRUModel(input_shape, output_shape, hyperparams)
            elif model_type in ['xgboost', 'lightgbm']:
                if model_type == 'xgboost':
                    model = XGBoostModel(hyperparams)
                else:
                    model = LightGBMModel(hyperparams)
            elif model_type in ['attention_lstm', 'bidirectional_lstm', 'cnn_lstm', 'transformer', 'dual_attention']:
                # Advanced models
                model = AdvancedModelFactory.create_model(model_type, input_shape, output_shape, hyperparams)
            else:
                raise ValueError(f"Unknown model type: {model_type}")

            self.models.append({
                'type': model_type,
                'model': model,
                'weight': config.get('weight', 1.0)
            })

    def fit(self, X: np.ndarray, y: np.ndarray, validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None):
        """
        Fit all models in the ensemble

        Args:
            X: Input features
            y: Target values
            validation_data: Validation data (optional)
        """
        for model_info in self.models:
            model_type = model_info['type']
            model = model_info['model']

            logger.info(f"Training {model_type} model...")

            if model_type in ['lstm', 'gru', 'attention_lstm', 'bidirectional_lstm', 'cnn_lstm', 'transformer', 'dual_attention']:
                # Deep learning models
                callbacks = AdvancedModelFactory.get_callbacks()
                model.fit(X, y, validation_data=validation_data, callbacks=callbacks)
            else:
                # ML models
                model.fit(X, y)

            logger.info(f"Finished training {model_type} model")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using weighted ensemble

        Args:
            X: Input features

        Returns:
            Weighted ensemble predictions
        """
        predictions = []
        weights = []

        for model_info in self.models:
            model = model_info['model']
            weight = model_info['weight']

            pred = model.predict(X)
            predictions.append(pred)
            weights.append(weight)

        # Normalize weights
        weights = np.array(weights) / sum(weights)

        # Weighted average of predictions
        ensemble_pred = np.zeros_like(predictions[0])
        for i, pred in enumerate(predictions):
            ensemble_pred += pred * weights[i]

        return ensemble_pred

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Dict[str, float]]:
        """
        Evaluate all models and the ensemble

        Args:
            X: Input features
            y: Target values

        Returns:
            Dictionary with evaluation metrics for each model and the ensemble
        """
        results = {}

        # Evaluate individual models
        for model_info in self.models:
            model_type = model_info['type']
            model = model_info['model']

            y_pred = model.predict(X)

            results[model_type] = {
                'mse': mean_squared_error(y, y_pred),
                'rmse': np.sqrt(mean_squared_error(y, y_pred)),
                'mae': mean_absolute_error(y, y_pred),
                'r2': r2_score(y, y_pred)
            }

        # Evaluate ensemble
        ensemble_pred = self.predict(X)

        results['ensemble'] = {
            'mse': mean_squared_error(y, ensemble_pred),
            'rmse': np.sqrt(mean_squared_error(y, ensemble_pred)),
            'mae': mean_absolute_error(y, ensemble_pred),
            'r2': r2_score(y, ensemble_pred)
        }

        return results

    def optimize_weights(self, X: np.ndarray, y: np.ndarray):
        """
        Optimize model weights based on validation performance

        Args:
            X: Input features
            y: Target values
        """
        # Get predictions from each model
        predictions = []
        for model_info in self.models:
            model = model_info['model']
            pred = model.predict(X)
            predictions.append(pred)

        # Convert to numpy arrays
        predictions = np.array(predictions)

        # Simple weight optimization based on inverse MSE
        mse_values = []
        for pred in predictions:
            mse = mean_squared_error(y, pred)
            mse_values.append(mse)

        # Inverse MSE (higher weight for lower error)
        inv_mse = 1.0 / np.array(mse_values)

        # Normalize weights
        weights = inv_mse / np.sum(inv_mse)

        # Update model weights
        for i, model_info in enumerate(self.models):
            model_info['weight'] = float(weights[i])

        logger.info(f"Optimized weights: {[model_info['weight'] for model_info in self.models]}")

    def save_models(self, base_path: str, symbol: str):
        """
        Save all models

        Args:
            base_path: Base path to save models
            symbol: Cryptocurrency symbol
        """
        os.makedirs(base_path, exist_ok=True)

        for model_info in self.models:
            model_type = model_info['type']
            model = model_info['model']
            weight = model_info['weight']

            # Save model
            model_path = os.path.join(base_path, f"{symbol.lower()}_{model_type}.model")

            if model_type in ['lstm', 'gru', 'attention_lstm', 'bidirectional_lstm', 'cnn_lstm', 'transformer', 'dual_attention']:
                # Save Keras model
                # Convert path to .keras format if it's .h5
                if model_path.endswith('.h5'):
                    new_path = model_path.replace('.h5', '.keras')
                    logger.info(f"Changed model save format from HDF5 (.h5) to native Keras format (.keras)")
                    model.save(new_path)
                    model_path = new_path
                else:
                    model.save(model_path)
            else:
                # Save ML model
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)

            # Save metadata
            metadata = {
                'type': model_type,
                'weight': weight
            }

            metadata_path = os.path.join(base_path, f"{symbol.lower()}_{model_type}_metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f)

    def load_models(self, base_path: str, symbol: str):
        """
        Load all models

        Args:
            base_path: Base path to load models from
            symbol: Cryptocurrency symbol
        """
        self.models = []

        for config in self.models_config:
            model_type = config['type']

            # Load model
            model_path = os.path.join(base_path, f"{symbol.lower()}_{model_type}.model")

            if not os.path.exists(model_path):
                logger.warning(f"Model not found: {model_path}")
                continue

            if model_type in ['lstm', 'gru', 'attention_lstm', 'bidirectional_lstm', 'cnn_lstm', 'transformer', 'dual_attention']:
                # Load Keras model
                model = load_model(model_path)
            else:
                # Load ML model
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)

            # Load metadata
            metadata_path = os.path.join(base_path, f"{symbol.lower()}_{model_type}_metadata.json")
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

            self.models.append({
                'type': model_type,
                'model': model,
                'weight': metadata.get('weight', 1.0)
            })
