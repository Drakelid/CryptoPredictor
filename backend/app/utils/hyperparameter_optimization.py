import numpy as np
import pandas as pd
import logging
from typing import List, Dict, Tuple, Optional, Union, Any, Callable
import os
import json
import pickle
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Import local modules
from app.models.dl_models import LSTMModel, GRUModel
from app.models.ml_models import XGBoostModel, LightGBMModel
from app.models.advanced_models import AdvancedModelFactory

# Import optuna conditionally to avoid errors if not installed
try:
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    # Create dummy classes for type hints
    class MedianPruner: pass
    class TPESampler: pass

logger = logging.getLogger(__name__)

class HyperparameterOptimizer:
    """Base class for hyperparameter optimization"""

    def __init__(
        self,
        model_type: str,
        param_space: Dict[str, Any],
        n_trials: int = 50,
        timeout: Optional[int] = None,
        metric: str = 'rmse',
        direction: str = 'minimize',
        cv: int = 5,
        study_name: Optional[str] = None
    ):
        """
        Initialize hyperparameter optimizer

        Args:
            model_type: Type of model to optimize
            param_space: Parameter space to search
            n_trials: Number of trials
            timeout: Timeout in seconds (optional)
            metric: Metric to optimize
            direction: Direction of optimization ('minimize' or 'maximize')
            cv: Number of cross-validation folds
            study_name: Name of the study (optional)
        """
        # Check if Optuna is available
        if not OPTUNA_AVAILABLE:
            raise ImportError(
                "Optuna is required for hyperparameter optimization. "
                "Install it with 'pip install optuna'."
            )

        self.model_type = model_type
        self.param_space = param_space
        self.n_trials = n_trials
        self.timeout = timeout
        self.metric = metric
        self.direction = direction
        self.cv = cv
        self.study_name = study_name or f"{model_type}_optimization"

        # Create Optuna study
        self.study = optuna.create_study(
            study_name=self.study_name,
            direction=self.direction,
            sampler=TPESampler(),
            pruner=MedianPruner()
        )

        self.best_params = None
        self.best_score = None
        self.best_model = None

    def optimize(self, X: np.ndarray, y: np.ndarray):
        """
        Run hyperparameter optimization

        Args:
            X: Input features
            y: Target values
        """
        raise NotImplementedError("Subclasses must implement optimize method")

    def get_best_params(self) -> Dict[str, Any]:
        """
        Get best hyperparameters

        Returns:
            Dictionary of best hyperparameters
        """
        if self.best_params is None:
            raise ValueError("No optimization results available. Call optimize() first.")

        return self.best_params

    def get_best_score(self) -> float:
        """
        Get best score

        Returns:
            Best score
        """
        if self.best_score is None:
            raise ValueError("No optimization results available. Call optimize() first.")

        return self.best_score

    def get_best_model(self) -> Any:
        """
        Get best model

        Returns:
            Best model
        """
        if self.best_model is None:
            raise ValueError("No optimization results available. Call optimize() first.")

        return self.best_model

    def save_results(self, path: str):
        """
        Save optimization results

        Args:
            path: Path to save results
        """
        if self.best_params is None:
            raise ValueError("No optimization results available. Call optimize() first.")

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Save results
        results = {
            'model_type': self.model_type,
            'best_params': self.best_params,
            'best_score': self.best_score,
            'study_name': self.study_name,
            'n_trials': self.n_trials,
            'metric': self.metric,
            'direction': self.direction
        }

        with open(path, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"Saved optimization results to {path}")

    def load_results(self, path: str):
        """
        Load optimization results

        Args:
            path: Path to load results from
        """
        with open(path, 'r') as f:
            results = json.load(f)

        self.model_type = results['model_type']
        self.best_params = results['best_params']
        self.best_score = results['best_score']
        self.study_name = results['study_name']
        self.n_trials = results['n_trials']
        self.metric = results['metric']
        self.direction = results['direction']

        logger.info(f"Loaded optimization results from {path}")

    def _calculate_metric(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate metric

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            Metric value
        """
        if self.metric == 'rmse':
            return np.sqrt(mean_squared_error(y_true, y_pred))
        elif self.metric == 'mse':
            return mean_squared_error(y_true, y_pred)
        elif self.metric == 'mae':
            return mean_absolute_error(y_true, y_pred)
        elif self.metric == 'r2':
            return r2_score(y_true, y_pred)
        else:
            raise ValueError(f"Unknown metric: {self.metric}")


class MLHyperparameterOptimizer(HyperparameterOptimizer):
    """Hyperparameter optimizer for machine learning models"""

    def __init__(
        self,
        model_type: str,
        param_space: Optional[Dict[str, Any]] = None,
        n_trials: int = 50,
        timeout: Optional[int] = None,
        metric: str = 'rmse',
        direction: str = 'minimize',
        cv: int = 5,
        study_name: Optional[str] = None
    ):
        """
        Initialize ML hyperparameter optimizer

        Args:
            model_type: Type of model to optimize ('xgboost' or 'lightgbm')
            param_space: Parameter space to search (optional)
            n_trials: Number of trials
            timeout: Timeout in seconds (optional)
            metric: Metric to optimize
            direction: Direction of optimization ('minimize' or 'maximize')
            cv: Number of cross-validation folds
            study_name: Name of the study (optional)
        """
        # Set default parameter space if not provided
        if param_space is None:
            if model_type == 'xgboost':
                param_space = {
                    'n_estimators': (50, 500),
                    'max_depth': (3, 10),
                    'learning_rate': (0.01, 0.3),
                    'subsample': (0.6, 1.0),
                    'colsample_bytree': (0.6, 1.0),
                    'min_child_weight': (1, 10),
                    'gamma': (0, 5),
                    'reg_alpha': (0, 5),
                    'reg_lambda': (0, 5)
                }
            elif model_type == 'lightgbm':
                param_space = {
                    'n_estimators': (50, 500),
                    'max_depth': (3, 10),
                    'learning_rate': (0.01, 0.3),
                    'num_leaves': (20, 100),
                    'subsample': (0.6, 1.0),
                    'colsample_bytree': (0.6, 1.0),
                    'min_child_samples': (5, 50),
                    'reg_alpha': (0, 5),
                    'reg_lambda': (0, 5)
                }
            else:
                raise ValueError(f"Unknown model type: {model_type}")

        super().__init__(model_type, param_space, n_trials, timeout, metric, direction, cv, study_name)
        self.cv_splits = TimeSeriesSplit(n_splits=cv)

    def optimize(self, X: np.ndarray, y: np.ndarray):
        """
        Run hyperparameter optimization for ML models

        Args:
            X: Input features
            y: Target values
        """
        def objective(trial):
            # Sample hyperparameters
            params = {}
            for param_name, param_range in self.param_space.items():
                if isinstance(param_range, tuple) and len(param_range) == 2:
                    # Continuous parameter
                    if isinstance(param_range[0], int) and isinstance(param_range[1], int):
                        params[param_name] = trial.suggest_int(param_name, param_range[0], param_range[1])
                    else:
                        params[param_name] = trial.suggest_float(param_name, param_range[0], param_range[1])
                elif isinstance(param_range, list):
                    # Categorical parameter
                    params[param_name] = trial.suggest_categorical(param_name, param_range)
                else:
                    # Fixed parameter
                    params[param_name] = param_range

            # Create model
            if self.model_type == 'xgboost':
                model = XGBoostModel(params)
            elif self.model_type == 'lightgbm':
                model = LightGBMModel(params)
            else:
                raise ValueError(f"Unknown model type: {self.model_type}")

            # Cross-validation
            cv_scores = []
            for train_idx, val_idx in self.cv_splits.split(X):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                # Train model
                model.fit(X_train, y_train)

                # Evaluate model
                y_pred = model.predict(X_val)
                score = self._calculate_metric(y_val, y_pred)
                cv_scores.append(score)

            # Return mean score
            return np.mean(cv_scores)

        # Run optimization
        self.study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout)

        # Get best parameters
        self.best_params = self.study.best_params
        self.best_score = self.study.best_value

        # Train best model on all data
        if self.model_type == 'xgboost':
            self.best_model = XGBoostModel(self.best_params)
        elif self.model_type == 'lightgbm':
            self.best_model = LightGBMModel(self.best_params)

        self.best_model.fit(X, y)

        logger.info(f"Best {self.model_type} parameters: {self.best_params}")
        logger.info(f"Best {self.model_type} score ({self.metric}): {self.best_score}")


class DLHyperparameterOptimizer(HyperparameterOptimizer):
    """Hyperparameter optimizer for deep learning models"""

    def __init__(
        self,
        model_type: str,
        input_shape: Tuple[int, int],
        output_shape: int,
        param_space: Optional[Dict[str, Any]] = None,
        n_trials: int = 30,
        timeout: Optional[int] = None,
        metric: str = 'rmse',
        direction: str = 'minimize',
        cv: int = 3,
        study_name: Optional[str] = None
    ):
        """
        Initialize DL hyperparameter optimizer

        Args:
            model_type: Type of model to optimize ('lstm', 'gru', 'attention_lstm', etc.)
            input_shape: Shape of input data (sequence_length, features)
            output_shape: Number of output units
            param_space: Parameter space to search (optional)
            n_trials: Number of trials
            timeout: Timeout in seconds (optional)
            metric: Metric to optimize
            direction: Direction of optimization ('minimize' or 'maximize')
            cv: Number of cross-validation folds
            study_name: Name of the study (optional)
        """
        # Set default parameter space if not provided
        if param_space is None:
            if model_type in ['lstm', 'gru']:
                param_space = {
                    'units': (32, 256),
                    'layers': (1, 3),
                    'dropout_rate': (0.1, 0.5),
                    'recurrent_dropout': (0.0, 0.3),
                    'learning_rate': (0.0001, 0.01),
                    'batch_size': [16, 32, 64, 128],
                    'activation': ['relu', 'tanh']
                }
            elif model_type in ['attention_lstm', 'bidirectional_lstm']:
                param_space = {
                    'units': (64, 256),
                    'dropout_rate': (0.1, 0.5),
                    'learning_rate': (0.0001, 0.01),
                    'batch_size': [16, 32, 64, 128]
                }
            elif model_type == 'cnn_lstm':
                param_space = {
                    'lstm_units': (32, 256),
                    'cnn_filters': (32, 128),
                    'kernel_size': [3, 5, 7],
                    'dropout_rate': (0.1, 0.5),
                    'learning_rate': (0.0001, 0.01),
                    'batch_size': [16, 32, 64, 128]
                }
            elif model_type == 'transformer':
                param_space = {
                    'embed_dim': (64, 256),
                    'num_heads': [2, 4, 8],
                    'ff_dim': (128, 512),
                    'dropout_rate': (0.1, 0.5),
                    'learning_rate': (0.0001, 0.01),
                    'batch_size': [16, 32, 64, 128],
                    'num_transformer_blocks': [1, 2, 3]
                }
            elif model_type == 'dual_attention':
                param_space = {
                    'units': (64, 256),
                    'dropout_rate': (0.1, 0.5),
                    'learning_rate': (0.0001, 0.01),
                    'batch_size': [16, 32, 64, 128]
                }
            else:
                raise ValueError(f"Unknown model type: {model_type}")

        super().__init__(model_type, param_space, n_trials, timeout, metric, direction, cv, study_name)
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.cv_splits = TimeSeriesSplit(n_splits=cv)

    def optimize(self, X: np.ndarray, y: np.ndarray):
        """
        Run hyperparameter optimization for DL models

        Args:
            X: Input features
            y: Target values
        """
        def objective(trial):
            # Sample hyperparameters
            params = {}
            for param_name, param_range in self.param_space.items():
                if isinstance(param_range, tuple) and len(param_range) == 2:
                    # Continuous parameter
                    if isinstance(param_range[0], int) and isinstance(param_range[1], int):
                        params[param_name] = trial.suggest_int(param_name, param_range[0], param_range[1])
                    else:
                        params[param_name] = trial.suggest_float(param_name, param_range[0], param_range[1])
                elif isinstance(param_range, list):
                    # Categorical parameter
                    params[param_name] = trial.suggest_categorical(param_name, param_range)
                else:
                    # Fixed parameter
                    params[param_name] = param_range

            # Create model
            if self.model_type == 'lstm':
                model = LSTMModel(self.input_shape, self.output_shape, params)
            elif self.model_type == 'gru':
                model = GRUModel(self.input_shape, self.output_shape, params)
            elif self.model_type in ['attention_lstm', 'bidirectional_lstm', 'cnn_lstm', 'transformer', 'dual_attention']:
                # Advanced models
                model = AdvancedModelFactory.create_model(self.model_type, self.input_shape, self.output_shape, params)
            else:
                raise ValueError(f"Unknown model type: {self.model_type}")

            # Get batch size
            batch_size = params.get('batch_size', 32)

            # Cross-validation
            cv_scores = []
            for train_idx, val_idx in self.cv_splits.split(X):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                # Early stopping callback
                early_stopping = EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True
                )

                # Train model
                model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    batch_size=batch_size,
                    epochs=100,  # Use early stopping to determine actual epochs
                    callbacks=[early_stopping],
                    verbose=0
                )

                # Evaluate model
                y_pred = model.predict(X_val)
                score = self._calculate_metric(y_val, y_pred)
                cv_scores.append(score)

                # Clean up to avoid memory leaks
                tf.keras.backend.clear_session()

            # Return mean score
            return np.mean(cv_scores)

        # Run optimization
        self.study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout)

        # Get best parameters
        self.best_params = self.study.best_params
        self.best_score = self.study.best_value

        # Train best model on all data
        if self.model_type == 'lstm':
            self.best_model = LSTMModel(self.input_shape, self.output_shape, self.best_params)
        elif self.model_type == 'gru':
            self.best_model = GRUModel(self.input_shape, self.output_shape, self.best_params)
        elif self.model_type in ['attention_lstm', 'bidirectional_lstm', 'cnn_lstm', 'transformer', 'dual_attention']:
            # Advanced models
            self.best_model = AdvancedModelFactory.create_model(
                self.model_type, self.input_shape, self.output_shape, self.best_params
            )

        # Early stopping callback
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True
        )

        # Learning rate reduction callback
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-6
        )

        # Split data for validation
        train_size = int(len(X) * 0.8)
        X_train, X_val = X[:train_size], X[train_size:]
        y_train, y_val = y[:train_size], y[train_size:]

        # Train best model
        self.best_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            batch_size=self.best_params.get('batch_size', 32),
            epochs=200,  # Use early stopping to determine actual epochs
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )

        logger.info(f"Best {self.model_type} parameters: {self.best_params}")
        logger.info(f"Best {self.model_type} score ({self.metric}): {self.best_score}")


class AdvancedModelOptimizer(DLHyperparameterOptimizer):
    """Hyperparameter optimizer for advanced models"""

    def __init__(
        self,
        model_type: str,
        input_shape: Tuple[int, int],
        output_shape: int,
        param_space: Optional[Dict[str, Any]] = None,
        n_trials: int = 20,
        timeout: Optional[int] = None,
        metric: str = 'rmse',
        direction: str = 'minimize',
        cv: int = 3,
        study_name: Optional[str] = None
    ):
        """
        Initialize advanced model optimizer

        Args:
            model_type: Type of model to optimize
            input_shape: Shape of input data
            output_shape: Number of output units
            param_space: Parameter space to search (optional)
            n_trials: Number of trials
            timeout: Timeout in seconds (optional)
            metric: Metric to optimize
            direction: Direction of optimization
            cv: Number of cross-validation folds
            study_name: Name of the study (optional)
        """
        super().__init__(
            model_type, input_shape, output_shape, param_space,
            n_trials, timeout, metric, direction, cv, study_name
        )
