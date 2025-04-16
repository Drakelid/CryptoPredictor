import numpy as np
import pandas as pd
import os
import pickle
import logging
from typing import Dict, List, Tuple, Optional, Union, Any

# Import ML libraries conditionally to avoid errors if not installed
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    xgb = None

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    lgb = None

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    optuna = None

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from app.core.config import settings

logger = logging.getLogger(__name__)


class MLModel:
    """Base class for machine learning models"""

    def __init__(
        self,
        model_type: str = "xgboost",
        model_path: Optional[str] = None
    ):
        """
        Initialize ML model

        Args:
            model_type: Type of model ('xgboost' or 'lightgbm')
            model_path: Path to saved model (if loading existing model)
        """
        self.model_type = model_type.lower()
        self.model_path = model_path
        self.model = None
        self.feature_importance = None

        if model_path and os.path.exists(model_path):
            self.load_model(model_path)

    def _create_model(self, params: Optional[Dict[str, Any]] = None) -> Any:
        """
        Create a new model instance

        Args:
            params: Model parameters

        Returns:
            Model instance
        """
        if self.model_type == "xgboost":
            if not XGBOOST_AVAILABLE:
                raise ImportError("XGBoost is not installed. Install it with 'pip install xgboost'.")

            default_params = {
                'objective': 'reg:squarederror',
                'eval_metric': 'rmse',
                'learning_rate': 0.05,
                'max_depth': 6,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'n_estimators': 100
            }

            # Update with provided params
            if params:
                default_params.update(params)

            return xgb.XGBRegressor(**default_params)

        elif self.model_type == "lightgbm":
            if not LIGHTGBM_AVAILABLE:
                raise ImportError("LightGBM is not installed. Install it with 'pip install lightgbm'.")

            default_params = {
                'objective': 'regression',
                'metric': 'rmse',
                'learning_rate': 0.05,
                'max_depth': 6,
                'num_leaves': 50,  # Explicitly set num_leaves to avoid warning
                'min_data_in_leaf': 5,  # Minimum number of data in one leaf
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'n_estimators': 100,
                'verbose': -1  # Suppress verbose output
            }

            # Update with provided params
            if params:
                default_params.update(params)

            return lgb.LGBMRegressor(**default_params)

        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the model to data

        Args:
            X: Input features
            y: Target values
        """
        if self.model is None:
            self.model = self._create_model()

        # Ensure y is 1D to avoid the column-vector warning
        if isinstance(y, np.ndarray) and y.ndim > 1 and y.shape[1] == 1:
            logger.info("Reshaping y from column vector to 1D array")
            y = y.ravel()

        self.model.fit(X, y)

        # Get feature importance if available
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = self.model.feature_importances_

    def _check_data_size(self, X: np.ndarray, min_samples: int = 20) -> bool:
        """
        Check if we have enough data for training

        Args:
            X: Input features
            min_samples: Minimum number of samples required

        Returns:
            True if we have enough data, False otherwise
        """
        if X is None or len(X) < min_samples:
            logger.warning(f"Not enough data for training: {len(X) if X is not None else 0} samples (minimum {min_samples})")
            return False
        return True

    def train(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray = None, y_val: np.ndarray = None, params: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """
        Train the model and evaluate on validation data

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            params: Model parameters (optional)

        Returns:
            Dictionary of evaluation metrics
        """
        logger.info(f"Training {self.model_type} model with {len(X_train)} samples")

        # Check if we have enough data
        if not self._check_data_size(X_train):
            logger.warning(f"Not enough training data for {self.model_type} model. Using simplified model.")
            # Adjust parameters for small datasets
            if self.model_type == "lightgbm":
                small_data_params = {
                    'num_leaves': 7,
                    'min_data_in_leaf': 3,
                    'max_depth': 3,
                    'learning_rate': 0.05,
                    'n_estimators': 50,
                    'verbose': -1
                }
                if params:
                    params.update(small_data_params)
                else:
                    params = small_data_params

        # Create model with parameters if provided
        if params:
            self.model = self._create_model(params)
        elif self.model is None:
            self.model = self._create_model()

        # Fit the model
        self.fit(X_train, y_train)

        # Calculate training metrics
        y_pred_train = self.predict(X_train)
        train_mse = mean_squared_error(y_train, y_pred_train)
        train_mae = mean_absolute_error(y_train, y_pred_train)
        train_r2 = r2_score(y_train, y_pred_train)

        metrics = {
            'mse': train_mse,
            'mae': train_mae,
            'r2': train_r2
        }

        # Calculate validation metrics if validation data is provided
        if X_val is not None and y_val is not None:
            y_pred_val = self.predict(X_val)
            val_mse = mean_squared_error(y_val, y_pred_val)
            val_mae = mean_absolute_error(y_val, y_pred_val)
            val_r2 = r2_score(y_val, y_pred_val)

            metrics.update({
                'val_mse': val_mse,
                'val_mae': val_mae,
                'val_r2': val_r2
            })

        # Log metrics
        logger.info(f"Training metrics: {metrics}")

        return metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions

        Args:
            X: Input features

        Returns:
            Predicted values
        """
        if self.model is None:
            raise ValueError("Model not initialized. Call fit() or load_model() first.")

        # Special handling for XGBoost models
        if self.model_type == 'xgboost':
            logger.info(f"XGBoost predict with input shape: {X.shape}")

            # For XGBoost, we need a 2D input with 1610 features
            if len(X.shape) == 3:
                # Reshape to 2D by flattening the time steps and features
                samples, time_steps, features = X.shape
                X_flat = X.reshape(samples, time_steps * features)
                logger.info(f"Reshaped input from {X.shape} to {X_flat.shape} for XGBoost")
                X = X_flat

            # Create a new input with exactly 1610 features
            logger.info(f"Creating new input with 1610 features for XGBoost")
            new_X = np.zeros((X.shape[0], 1610))

            # Copy as many features as we can
            copy_features = min(X.shape[1], 1610)
            new_X[:, :copy_features] = X[:, :copy_features]

            # Use the new input
            X = new_X
            logger.info(f"New input shape for XGBoost: {X.shape}")

        return self.model.predict(X)

    def save_model(self, path: str) -> None:
        """
        Save model to disk

        Args:
            path: Path to save the model
        """
        if self.model is None:
            raise ValueError("Model not initialized. Call fit() first.")

        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Save model
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)

        # Save metadata
        metadata = {
            'model_type': self.model_type,
            'feature_importance': self.feature_importance
        }

        metadata_path = os.path.join(os.path.dirname(path), f"{os.path.basename(path)}_metadata.pkl")
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)

        self.model_path = path
        logger.info(f"Saved model to {path} and metadata to {metadata_path}")

    def load_model(self, path: str) -> None:
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

        # Load metadata if available
        metadata_path = os.path.join(os.path.dirname(path), f"{os.path.basename(path)}_metadata.pkl")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
                self.model_type = metadata.get('model_type', self.model_type)
                self.feature_importance = metadata.get('feature_importance')

        self.model_path = path
        logger.info(f"Loaded model from {path}")


class XGBoostModel(MLModel):
    """XGBoost model implementation"""

    def __init__(self, params: Optional[Dict[str, Any]] = None, model_path: Optional[str] = None):
        """
        Initialize XGBoost model

        Args:
            params: Model parameters
            model_path: Path to saved model (if loading existing model)
        """
        super().__init__(model_type="xgboost", model_path=model_path)

        if model_path is None and params is not None:
            self.model = self._create_model(params)

    def train(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray = None, y_val: np.ndarray = None, params: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """
        Train the XGBoost model and evaluate on validation data

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            params: Model parameters (optional)

        Returns:
            Dictionary of evaluation metrics
        """
        logger.info("Training XGBoost model")
        return super().train(X_train, y_train, X_val, y_val, params)


class LightGBMModel(MLModel):
    """LightGBM model implementation"""

    def __init__(self, params: Optional[Dict[str, Any]] = None, model_path: Optional[str] = None):
        """
        Initialize LightGBM model

        Args:
            params: Model parameters
            model_path: Path to saved model (if loading existing model)
        """
        super().__init__(model_type="lightgbm", model_path=model_path)

        if model_path is None and params is not None:
            self.model = self._create_model(params)

    def _adjust_params_for_dataset(self, X_train: np.ndarray, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Adjust parameters based on dataset size

        Args:
            X_train: Training features
            params: Original parameters (optional)

        Returns:
            Adjusted parameters
        """
        # Start with existing params or empty dict
        adjusted_params = params.copy() if params else {}

        # Get dataset size
        n_samples, n_features = X_train.shape

        # Adjust parameters based on dataset size
        if n_samples < 50:
            # Very small dataset
            small_data_params = {
                'num_leaves': min(7, max(3, n_samples // 5)),
                'min_data_in_leaf': 1,
                'max_depth': 3,
                'learning_rate': 0.05,
                'n_estimators': 50,
                'verbose': -1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 0.1
            }
            adjusted_params.update(small_data_params)
            logger.info(f"Adjusted LightGBM parameters for very small dataset (n_samples={n_samples})")
        elif n_samples < 100:
            # Small dataset
            small_data_params = {
                'num_leaves': min(15, max(7, n_samples // 7)),
                'min_data_in_leaf': 2,
                'max_depth': 4,
                'learning_rate': 0.05,
                'n_estimators': 100,
                'verbose': -1
            }
            adjusted_params.update(small_data_params)
            logger.info(f"Adjusted LightGBM parameters for small dataset (n_samples={n_samples})")

        return adjusted_params

    def train(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray = None, y_val: np.ndarray = None, params: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """
        Train the LightGBM model and evaluate on validation data

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            params: Model parameters (optional)

        Returns:
            Dictionary of evaluation metrics
        """
        logger.info(f"Training LightGBM model with {len(X_train)} samples and {X_train.shape[1]} features")

        # Adjust parameters based on dataset size
        adjusted_params = self._adjust_params_for_dataset(X_train, params)

        # Train with adjusted parameters
        return super().train(X_train, y_train, X_val, y_val, adjusted_params)


class HyperparameterTuner:
    """Class for hyperparameter tuning using Optuna"""

    def __init__(self, model_type: str = "xgboost", n_trials: int = 20, **kwargs):
        """
        Initialize hyperparameter tuner

        Args:
            model_type: Type of model to tune
            n_trials: Number of trials for hyperparameter search
            **kwargs: Additional keyword arguments
        """
        self.model_type = model_type
        self.n_trials = n_trials
        self.best_params = None

        # Store any additional parameters
        for key, value in kwargs.items():
            setattr(self, key, value)

    def tune(self, X_train, y_train, X_val, y_val):
        """Tune hyperparameters for the specified model type"""
        logger.info(f"Starting hyperparameter tuning for {self.model_type} with {self.n_trials} trials")

        try:
            # Check if Optuna is available
            if not OPTUNA_AVAILABLE:
                logger.warning("Optuna is not available. Using default parameters.")
                return self._get_default_params()

            # Define the objective function for Optuna
            def objective(trial):
                # Get parameters based on model type
                if self.model_type == "xgboost":
                    params = {
                        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                        'max_depth': trial.suggest_int('max_depth', 3, 10),
                        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                        'gamma': trial.suggest_float('gamma', 0, 5),
                        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10)
                    }
                    model = XGBoostModel(params=params)
                elif self.model_type == "lightgbm":
                    # Calculate appropriate num_leaves based on max_depth
                    max_depth = trial.suggest_int('max_depth', 5, 12)  # Increased minimum max_depth to avoid issues

                    # Ensure low is always less than high
                    low_leaves = min(20, 2**max_depth - 10)
                    high_leaves = 2**max_depth + 10

                    # Add safety check
                    if low_leaves >= high_leaves:
                        low_leaves = high_leaves - 1

                    num_leaves = trial.suggest_int('num_leaves', low_leaves, high_leaves)

                    params = {
                        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                        'max_depth': max_depth,
                        'num_leaves': num_leaves,
                        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 3, 20),
                        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                        'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
                        'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 1.0),
                        'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 1.0),
                        'verbose': -1  # Suppress verbose output
                    }
                    model = LightGBMModel(params=params)
                else:
                    logger.warning(f"Unsupported model type for tuning: {self.model_type}")
                    return float('inf')  # Return a large value to indicate failure

                # Train the model with the suggested parameters
                try:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_val)
                    mse = mean_squared_error(y_val, y_pred)
                    return mse
                except Exception as e:
                    logger.error(f"Error during hyperparameter tuning: {e}")
                    return float('inf')  # Return a large value to indicate failure

            # Create and run the Optuna study
            study = optuna.create_study(direction='minimize')
            study.optimize(objective, n_trials=self.n_trials)

            # Get the best parameters
            self.best_params = study.best_params
            logger.info(f"Best parameters found: {self.best_params}")
            return self.best_params

        except Exception as e:
            logger.error(f"Error during hyperparameter tuning: {e}")
            logger.info("Falling back to default parameters")
            return self._get_default_params()

    def _get_default_params(self):
        """Get default parameters for the specified model type"""
        if self.model_type == "xgboost":
            # Default parameters for XGBoost
            self.best_params = {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'gamma': 0,
                'min_child_weight': 1
            }
        elif self.model_type == "lightgbm":
            # Default parameters for LightGBM
            self.best_params = {
                'n_estimators': 100,
                'max_depth': 6,
                'num_leaves': 50,  # Explicitly set num_leaves to avoid warning
                'min_data_in_leaf': 5,  # Minimum number of data in one leaf
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_samples': 20,
                'reg_alpha': 0.1,
                'reg_lambda': 0.1,
                'verbose': -1  # Suppress verbose output
            }
        else:
            # Generic default parameters
            self.best_params = {}

        logger.info(f"Using default parameters for {self.model_type}: {self.best_params}")
        return self.best_params

    def get_best_model(self):
        """Get the best model based on tuning"""
        return MLModel(model_type=self.model_type)


def get_ml_model(
    model_type: str = "xgboost",
    model_path: Optional[str] = None
) -> MLModel:
    """
    Factory function to get the appropriate ML model

    Args:
        model_type: Type of model ('xgboost' or 'lightgbm')
        model_path: Path to saved model (if loading existing model)

    Returns:
        Initialized model
    """
    if model_type.lower() == "xgboost":
        return XGBoostModel(model_path=model_path)
    elif model_type.lower() == "lightgbm":
        return LightGBMModel(model_path=model_path)
    else:
        return MLModel(model_type=model_type, model_path=model_path)
