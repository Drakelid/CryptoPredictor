import pandas as pd
import numpy as np
import os
import logging
import json
import traceback
from typing import List, Dict, Optional, Union, Any, Tuple
from datetime import datetime

from app.models.dl_models import get_dl_model
from app.models.ml_models import get_ml_model, HyperparameterTuner
from app.utils.feature_engineering import FeatureEngineer
from app.services.data_service import DataService
from app.schemas.data import TrainingResult
from app.core.config import settings

# Import advanced modules
from app.models.advanced_models import AdvancedModelFactory
from app.models.ensemble_models import TimeSeriesEnsemble
from app.utils.feature_selection import FeatureSelector
from app.utils.anomaly_detection import AnomalyDetector
from app.utils.time_cv import TimeSeriesCV, WalkForwardCV
from app.utils.transfer_learning import TransferLearningModel, ContinualLearningModel

# Import hyperparameter optimization conditionally
try:
    from app.utils.hyperparameter_optimization import MLHyperparameterOptimizer, DLHyperparameterOptimizer
    HYPEROPT_AVAILABLE = True
except ImportError:
    HYPEROPT_AVAILABLE = False

logger = logging.getLogger(__name__)

class TrainingService:
    """Service for training cryptocurrency price prediction models"""

    def __init__(self, data_service: DataService = None):
        """
        Initialize training service

        Args:
            data_service: DataService instance
        """
        self.data_service = data_service or DataService()
        self.models_dir = settings.MODELS_DIR
        os.makedirs(self.models_dir, exist_ok=True)

        # Initialize feature count
        self.feature_count = None

        # Create subdirectories for different model types
        for model_type in settings.SUPPORTED_MODELS:
            os.makedirs(os.path.join(self.models_dir, model_type), exist_ok=True)

    def train_model(
        self,
        symbol: str,
        model_type: str,
        lookback: int = 30,
        horizon: int = 7,
        epochs: Optional[int] = 100,
        batch_size: Optional[int] = 32,
        hyperparameter_tuning: bool = False,
        use_advanced_features: bool = True,
        use_ensemble: bool = False,
        use_feature_selection: bool = True,
        use_anomaly_detection: bool = True,
        use_transfer_learning: bool = False,
        source_symbol: Optional[str] = None,
        use_time_series_cv: bool = False,
        cv_splits: Optional[int] = None
    ) -> TrainingResult:
        """
        Train a new model with advanced features

        Args:
            symbol: Cryptocurrency symbol (e.g., BTC, ETH)
            model_type: Model type (lstm, gru, xgboost, lightgbm, attention_lstm, etc.)
            lookback: Number of days to look back
            horizon: Number of days to predict ahead
            epochs: Number of epochs for DL models
            batch_size: Batch size for DL models
            hyperparameter_tuning: Whether to perform hyperparameter tuning
            use_advanced_features: Whether to use advanced model architectures
            use_ensemble: Whether to use ensemble methods
            use_feature_selection: Whether to use feature selection
            use_anomaly_detection: Whether to detect and handle anomalies
            use_transfer_learning: Whether to use transfer learning
            source_symbol: Source symbol for transfer learning (required if use_transfer_learning=True)
            use_time_series_cv: Whether to perform time-series cross validation
            cv_splits: Number of CV splits (defaults to settings.TIME_CV_SPLITS)

        Returns:
            TrainingResult object with training information
        """
        # Validate model type
        if model_type not in settings.SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model type: {model_type}. Supported types: {settings.SUPPORTED_MODELS}")

        # Validate inputs
        if lookback <= 0:
            raise ValueError(f"Lookback must be positive, got {lookback}")
        if horizon <= 0:
            raise ValueError(f"Horizon must be positive, got {horizon}")

        # Load data
        df = self.data_service.load_data(symbol=symbol)

        # Check if we have data
        if df is None or df.empty:
            raise ValueError(f"No data available for {symbol}. Please fetch data first.")

        # Check if we have enough data
        if len(df) < lookback + horizon:
            raise ValueError(f"Not enough data for {symbol}. Need at least {lookback + horizon} data points, but got {len(df)}")

        # Engineer features
        try:
            logger.info(f"Starting feature engineering for {symbol}")
            feature_engineer = FeatureEngineer()
            df_features = feature_engineer.engineer_features(df, symbol=symbol, lookback=lookback, horizon=horizon)

            # Check if feature engineering was successful
            if df_features is None or df_features.empty:
                logger.error(f"Feature engineering returned empty DataFrame for {symbol}")
                raise ValueError(f"Feature engineering failed for {symbol}. No data available after processing.")

            # Log feature engineering success
            logger.info(f"Feature engineering successful for {symbol}. DataFrame shape: {df_features.shape}")

            # Check if we have enough features
            if df_features.shape[1] < 5:  # At least timestamp, price/close, and a few features
                logger.warning(f"Very few features generated for {symbol}: {df_features.columns.tolist()}")

            # If we have very few rows, add a warning
            if len(df_features) < 100:
                logger.warning(f"Limited data available after feature engineering: only {len(df_features)} rows")

        except Exception as e:
            logger.error(f"Error during feature engineering for {symbol}: {str(e)}")
            # Log the full traceback for debugging
            logger.error(f"Traceback: {traceback.format_exc()}")
            # Try to create a minimal feature set as a fallback
            logger.info(f"Attempting to create minimal feature set for {symbol}")

            try:
                # Create a minimal feature set with just the basics
                df_minimal = df.copy()

                # Ensure we have price/close column
                if 'close' not in df_minimal.columns and 'price' not in df_minimal.columns:
                    if 'open' in df_minimal.columns:
                        df_minimal['close'] = df_minimal['open']
                    else:
                        # Create synthetic price data
                        df_minimal['close'] = np.linspace(100, 200, len(df_minimal))

                # Add simple features
                price_col = 'close' if 'close' in df_minimal.columns else 'price'
                df_minimal['returns'] = df_minimal[price_col].ffill().pct_change(fill_method=None)
                df_minimal['log_returns'] = np.log(df_minimal[price_col].ffill() / df_minimal[price_col].ffill().shift(1))
                df_minimal['ma_5'] = df_minimal[price_col].rolling(window=5).mean()
                df_minimal['ma_10'] = df_minimal[price_col].rolling(window=10).mean()

                # Drop NaN values
                df_minimal.dropna(inplace=True)

                logger.info(f"Created minimal feature set for {symbol} with shape {df_minimal.shape}")
                df_features = df_minimal
            except Exception as fallback_error:
                logger.error(f"Fallback feature engineering also failed: {str(fallback_error)}")
                logger.error(f"Fallback traceback: {traceback.format_exc()}")
                raise ValueError(f"Feature engineering failed for {symbol} and fallback also failed: {str(e)}")

        # Determine target column
        target_col = 'close' if 'close' in df_features.columns else 'price'
        if target_col not in df_features.columns:
            raise ValueError(f"Target column '{target_col}' not found in data. Available columns: {df_features.columns.tolist()}")

        # Prepare data based on model type
        dl_models = ['lstm', 'gru', 'bidirectional_lstm', 'attention_lstm', 'cnn_lstm', 'transformer', 'dual_attention']

        if model_type in dl_models:
            # Prepare sequences for DL models
            try:
                logger.info(f"Preparing sequences for {model_type} model with lookback={lookback}, horizon={horizon}")

                # Log data shape before preparation
                logger.info(f"Data shape before preparation: {df_features.shape}")
                logger.info(f"Data columns: {df_features.columns.tolist()}")

                # Check if we have enough data for the requested lookback and horizon
                if len(df_features) < lookback + horizon + 10:  # Add some buffer
                    logger.warning(f"Limited data available: {len(df_features)} rows. Consider reducing lookback or horizon.")

                # Try to prepare sequences with reduced parameters if needed
                try:
                    X_train, y_train, X_val, y_val, scaler = feature_engineer.prepare_sequences(
                        df=df_features,
                        target_col=target_col,
                        lookback=lookback,
                        horizon=horizon,
                        train_ratio=settings.DEFAULT_TRAIN_TEST_SPLIT
                    )
                except ValueError as ve:
                    # If we get a value error, try with reduced parameters
                    if "Not enough data" in str(ve) and lookback > 10 and horizon > 1:
                        reduced_lookback = max(5, lookback // 2)
                        reduced_horizon = max(1, horizon // 2)
                        logger.warning(f"Retrying with reduced parameters: lookback={reduced_lookback}, horizon={reduced_horizon}")
                        X_train, y_train, X_val, y_val, scaler = feature_engineer.prepare_sequences(
                            df=df_features,
                            target_col=target_col,
                            lookback=reduced_lookback,
                            horizon=reduced_horizon,
                            train_ratio=settings.DEFAULT_TRAIN_TEST_SPLIT
                        )
                    else:
                        raise

                # Log shapes for debugging
                logger.info(f"Training data shapes - X_train: {X_train.shape}, y_train: {y_train.shape}")
                logger.info(f"Validation data shapes - X_val: {X_val.shape}, y_val: {y_val.shape}")

                # Verify that the shapes are compatible
                if X_train.ndim != 3:
                    logger.error(f"Unexpected X_train dimensions: X_train.ndim={X_train.ndim}")
                    raise ValueError(f"Unexpected X_train dimensions. Expected 3, got {X_train.ndim}")

                # Fix y_train dimensions if needed
                if y_train.ndim != 2:
                    logger.warning(f"Unexpected y_train dimensions: y_train.ndim={y_train.ndim}. Attempting to reshape.")
                    try:
                        # Try to reshape y_train and y_val
                        if y_train.ndim == 3 and y_train.shape[2] == 1:
                            # If shape is (samples, horizon, 1), reshape to (samples, horizon)
                            y_train = y_train.reshape(y_train.shape[0], y_train.shape[1])
                            y_val = y_val.reshape(y_val.shape[0], y_val.shape[1])
                            logger.info(f"Successfully reshaped y_train to {y_train.shape} and y_val to {y_val.shape}")
                        elif y_train.ndim == 3:
                            # If shape is (samples, horizon, features), take the first feature
                            y_train = y_train[:, :, 0]
                            y_val = y_val[:, :, 0]
                            logger.info(f"Successfully reshaped y_train to {y_train.shape} and y_val to {y_val.shape} by taking first feature")
                        else:
                            raise ValueError(f"Cannot reshape y_train with dimensions {y_train.ndim}")
                    except Exception as reshape_error:
                        logger.error(f"Failed to reshape y_train: {reshape_error}")
                        raise ValueError(f"Unexpected data dimensions. Expected X_train.ndim=3, y_train.ndim=2, got {X_train.ndim} and {y_train.ndim}")

            except Exception as e:
                logger.error(f"Error preparing sequences: {str(e)}")
                # Log the full traceback for debugging
                logger.error(f"Traceback: {traceback.format_exc()}")
                raise ValueError(f"Failed to prepare training data: {str(e)}")

            # Get number of features
            n_features = X_train.shape[2]

            # Log the number of features
            logger.info(f"Training model with {n_features} real features")

            cv_mae = None
            if use_time_series_cv:
                cv = TimeSeriesCV(n_splits=cv_splits or settings.TIME_CV_SPLITS)
                scores = []
                for train_idx, val_idx in cv.split(X_train):
                    cv_model = get_dl_model(
                        model_type=model_type,
                        input_shape=(lookback, n_features),
                        output_shape=horizon
                    )
                    cv_model.train(
                        X_train[train_idx],
                        y_train[train_idx],
                        X_train[val_idx],
                        y_train[val_idx],
                        epochs=max(5, (epochs or settings.DEFAULT_EPOCHS) // 2),
                        batch_size=batch_size or settings.DEFAULT_BATCH_SIZE,
                        patience=max(1, settings.DEFAULT_PATIENCE // 2)
                    )
                    preds = cv_model.predict(X_train[val_idx])
                    scores.append(float(np.mean(np.abs(y_train[val_idx] - preds))))
                if scores:
                    cv_mae = float(np.mean(scores))
                    logger.info(f"Average CV MAE: {cv_mae:.4f}")

            # Create model
            model = get_dl_model(
                model_type=model_type,
                input_shape=(lookback, n_features),
                output_shape=horizon
            )

            # Save the number of features used for training
            # This will be used during prediction to ensure we use the same number of features
            self.feature_count = n_features

            # Train model
            history = model.train(
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                epochs=epochs or settings.DEFAULT_EPOCHS,
                batch_size=batch_size or settings.DEFAULT_BATCH_SIZE,
                patience=settings.DEFAULT_PATIENCE
            )

            # Extract metrics
            metrics = {
                'train_loss': float(history['loss'][-1]),
                'val_loss': float(history['val_loss'][-1])
            }

            if cv_mae is not None:
                metrics['cv_mae'] = cv_mae

            # Add additional metrics if available
            if 'mse' in history:
                metrics['mse'] = float(history['mse'][0])
            if 'mae' in history:
                metrics['mae'] = float(history['mae'][0])
            if 'r2' in history:
                metrics['r2'] = float(history['r2'][0])

            logger.info(f"Final metrics for {model_type} model: {metrics}")

            # Save model
            model_filename = f"{symbol.lower()}_{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.keras"
            model_path = os.path.join(self.models_dir, model_type, model_filename)
            model.save_model(model_path)

            # Save scaler
            scaler_filename = f"{os.path.splitext(model_filename)[0]}_scaler.pkl"
            scaler_path = os.path.join(self.models_dir, model_type, scaler_filename)
            import pickle
            with open(scaler_path, 'wb') as f:
                pickle.dump(scaler, f)

        else:  # ML models (xgboost, lightgbm)
            # Prepare tabular data for ML models
            try:
                logger.info(f"Preparing tabular data for {model_type} model with lookback={lookback}, horizon={horizon}")

                # Log data shape before preparation
                logger.info(f"Data shape before preparation: {df_features.shape}")
                logger.info(f"Data columns: {df_features.columns.tolist()}")

                # Check if we have enough data for the requested lookback and horizon
                if len(df_features) < lookback + horizon + 10:  # Add some buffer
                    logger.warning(f"Limited data available: {len(df_features)} rows. Consider reducing lookback or horizon.")

                # Try to prepare tabular data with reduced parameters if needed
                try:
                    X_train, y_train, X_val, y_val, scaler = feature_engineer.prepare_tabular_data(
                        df=df_features,
                        target_col=target_col,
                        lookback=lookback,
                        horizon=horizon,
                        train_ratio=settings.DEFAULT_TRAIN_TEST_SPLIT
                    )
                except ValueError as ve:
                    # If we get a value error, try with reduced parameters
                    if "Not enough data" in str(ve) and lookback > 10 and horizon > 1:
                        reduced_lookback = max(5, lookback // 2)
                        reduced_horizon = max(1, horizon // 2)
                        logger.warning(f"Retrying with reduced parameters: lookback={reduced_lookback}, horizon={reduced_horizon}")
                        X_train, y_train, X_val, y_val, scaler = feature_engineer.prepare_tabular_data(
                            df=df_features,
                            target_col=target_col,
                            lookback=reduced_lookback,
                            horizon=reduced_horizon,
                            train_ratio=settings.DEFAULT_TRAIN_TEST_SPLIT
                        )
                    else:
                        raise

                # Log shapes for debugging
                logger.info(f"Training data shapes - X_train: {X_train.shape}, y_train: {y_train.shape}")
                logger.info(f"Validation data shapes - X_val: {X_val.shape}, y_val: {y_val.shape}")

                # Verify that the shapes are compatible
                if X_train.ndim != 2 or y_train.ndim != 2:
                    logger.error(f"Unexpected data dimensions: X_train.ndim={X_train.ndim}, y_train.ndim={y_train.ndim}")
                    raise ValueError(f"Unexpected data dimensions. Expected X_train.ndim=2, y_train.ndim=2, got {X_train.ndim} and {y_train.ndim}")

                cv_mae = None
                if use_time_series_cv:
                    cv = TimeSeriesCV(n_splits=cv_splits or settings.TIME_CV_SPLITS)
                    scores = []
                    for train_idx, val_idx in cv.split(X_train):
                        cv_model = get_ml_model(model_type=model_type)
                        cv_model.train(
                            X_train[train_idx],
                            y_train[train_idx],
                            X_train[val_idx],
                            y_train[val_idx],
                        )
                        preds = cv_model.predict(X_train[val_idx])
                        scores.append(float(np.mean(np.abs(y_train[val_idx] - preds))))
                    if scores:
                        cv_mae = float(np.mean(scores))
                        logger.info(f"Average CV MAE: {cv_mae:.4f}")

            except Exception as e:
                logger.error(f"Error preparing tabular data: {str(e)}")
                # Log the full traceback for debugging
                logger.error(f"Traceback: {traceback.format_exc()}")
                raise ValueError(f"Failed to prepare training data: {str(e)}")

            # Hyperparameter tuning if requested
            params = None
            if hyperparameter_tuning:
                tuner = HyperparameterTuner(model_type=model_type, n_trials=20)
                params = tuner.tune(X_train, y_train, X_val, y_val)

            # Create model
            model = get_ml_model(model_type=model_type)

            # Train model
            metrics = model.train(
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                params=params
            )

            if cv_mae is not None:
                metrics['cv_mae'] = cv_mae

            # Save model
            model_filename = f"{symbol.lower()}_{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
            model_path = os.path.join(self.models_dir, model_type, model_filename)
            model.save_model(model_path)

            # Save scaler
            scaler_filename = f"{os.path.splitext(model_filename)[0]}_scaler.pkl"
            scaler_path = os.path.join(self.models_dir, model_type, scaler_filename)
            import pickle
            with open(scaler_path, 'wb') as f:
                pickle.dump(scaler, f)

        # Save training metadata
        metadata = {
            'symbol': symbol,
            'model_type': model_type,
            'lookback': lookback,
            'horizon': horizon,
            'training_date': datetime.now().isoformat(),
            'metrics': metrics,
            'model_path': model_path,
            'scaler_path': scaler_path,
            'hyperparameter_tuning': hyperparameter_tuning,
            'feature_count': self.feature_count  # Save the number of features used for training
        }

        metadata_filename = f"{os.path.splitext(model_filename)[0]}_metadata.json"
        metadata_path = os.path.join(self.models_dir, model_type, metadata_filename)

        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        # Create training result
        result = TrainingResult(
            symbol=symbol,
            model_type=model_type,
            training_date=datetime.now(),
            metrics=metrics,
            model_path=model_path
        )

        return result

    def get_model_info(
        self,
        symbol: Optional[str] = None,
        model_type: Optional[str] = None
    ) -> List[TrainingResult]:
        """
        Get information about available models

        Args:
            symbol: Cryptocurrency symbol (e.g., BTC, ETH)
            model_type: Model type (lstm, gru, xgboost, lightgbm)

        Returns:
            List of TrainingResult objects
        """
        results = []

        # Determine which model types to check
        model_types = [model_type] if model_type else settings.SUPPORTED_MODELS

        for mt in model_types:
            model_dir = os.path.join(self.models_dir, mt)

            # Skip if directory doesn't exist
            if not os.path.exists(model_dir):
                continue

            # Find metadata files
            metadata_files = [f for f in os.listdir(model_dir) if f.endswith('_metadata.json')]

            for metadata_file in metadata_files:
                metadata_path = os.path.join(model_dir, metadata_file)

                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)

                    # Filter by symbol if provided
                    if symbol and metadata.get('symbol', '').lower() != symbol.lower():
                        continue

                    # Create training result
                    result = TrainingResult(
                        symbol=metadata.get('symbol', ''),
                        model_type=metadata.get('model_type', ''),
                        training_date=datetime.fromisoformat(metadata.get('training_date', datetime.now().isoformat())),
                        metrics=metadata.get('metrics', {}),
                        model_path=metadata.get('model_path', '')
                    )

                    results.append(result)

                except Exception as e:
                    logger.error(f"Error loading metadata from {metadata_path}: {e}")

        # Sort by training date (newest first)
        results.sort(key=lambda x: x.training_date, reverse=True)

        return results

    def delete_model(self, symbol: str, model_type: str) -> bool:
        """
        Delete a model

        Args:
            symbol: Cryptocurrency symbol (e.g., BTC, ETH)
            model_type: Model type (lstm, gru, xgboost, lightgbm)

        Returns:
            True if model was deleted, False otherwise
        """
        # Get model info
        model_info = self.get_model_info(symbol=symbol, model_type=model_type)

        if not model_info:
            return False

        # Get the latest model
        latest_model = model_info[0]

        # Delete model file
        if os.path.exists(latest_model.model_path):
            os.remove(latest_model.model_path)
            logger.info(f"Deleted model file: {latest_model.model_path}")

        # Delete associated files (scaler, metadata)
        base_path = os.path.splitext(latest_model.model_path)[0]

        # Delete scaler
        scaler_path = f"{base_path}_scaler.pkl"
        if os.path.exists(scaler_path):
            os.remove(scaler_path)
            logger.info(f"Deleted scaler file: {scaler_path}")

        # Delete metadata
        metadata_path = f"{base_path}_metadata.json"
        if os.path.exists(metadata_path):
            os.remove(metadata_path)
            logger.info(f"Deleted metadata file: {metadata_path}")

        return True
