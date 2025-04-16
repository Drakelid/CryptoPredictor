import pandas as pd
import numpy as np
import os
import logging
import json
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from app.models.dl_models import get_dl_model
from app.models.ml_models import get_ml_model
from app.utils.feature_engineering import FeatureEngineer
from app.core.config import settings

logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Class for evaluating model performance"""

    def __init__(self):
        """Initialize the model evaluator"""
        self.models_dir = settings.MODELS_DIR
        self.feature_engineer = FeatureEngineer()

    def evaluate_model(
        self,
        symbol: str,
        model_type: str,
        test_data: pd.DataFrame,
        lookback: int = 30,
        horizon: int = 7
    ) -> Dict[str, Any]:
        """
        Evaluate a model on test data

        Args:
            symbol: Cryptocurrency symbol
            model_type: Type of model to evaluate
            test_data: Test data for evaluation
            lookback: Number of days to look back for features
            horizon: Number of days to predict ahead

        Returns:
            Dictionary of evaluation metrics
        """
        try:
            # Find the latest model file
            model_dir = os.path.join(self.models_dir, model_type)
            if not os.path.exists(model_dir):
                logger.error(f"Model directory not found: {model_dir}")
                return {}

            # Find the latest model file for this symbol
            model_files = [f for f in os.listdir(model_dir) if f.startswith(f"{symbol.lower()}_") and f.endswith((".h5", ".keras", ".pkl"))]
            if not model_files:
                logger.error(f"No model files found for {symbol} in {model_dir}")
                return {}

            # Sort by date (assuming filename format includes date)
            model_files.sort(reverse=True)
            model_path = os.path.join(model_dir, model_files[0])

            # Load model metadata
            metadata_path = model_path.replace(".h5", ".json").replace(".pkl", ".json")
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
            else:
                metadata = {"lookback": lookback, "horizon": horizon}

            # Engineer features
            df_features = self.feature_engineer.engineer_features(
                test_data,
                symbol=symbol,
                lookback=metadata.get("lookback", lookback),
                horizon=metadata.get("horizon", horizon)
            )

            if df_features.empty:
                logger.error("No data available after feature engineering")
                return {}

            # Prepare data for prediction
            target_col = 'close' if 'close' in df_features.columns else 'price'

            # Prepare sequences
            X, y, _, _, _ = self.feature_engineer.prepare_sequences(
                df=df_features,
                target_col=target_col,
                lookback=metadata.get("lookback", lookback),
                horizon=metadata.get("horizon", horizon),
                train_ratio=1.0  # Use all data for evaluation
            )

            # Load the model
            if model_type in ['lstm', 'gru', 'bidirectional_lstm']:
                # Deep learning model
                import tensorflow as tf
                try:
                    # Try loading the model directly
                    logger.info(f"Attempting to load model from {model_path}")
                    model = tf.keras.models.load_model(model_path)
                    logger.info(f"Successfully loaded model from {model_path}")
                except Exception as e:
                    # If loading fails, try alternative formats
                    logger.warning(f"Error loading model from {model_path}: {str(e)}")

                    # Try alternative file formats
                    alt_paths = []

                    # If path is .h5, try .keras
                    if model_path.endswith('.h5'):
                        alt_paths.append(model_path.replace('.h5', '.keras'))

                    # If path is .keras, try .h5
                    if model_path.endswith('.keras'):
                        alt_paths.append(model_path.replace('.keras', '.h5'))

                    # Try loading from alternative paths
                    for alt_path in alt_paths:
                        if os.path.exists(alt_path):
                            try:
                                logger.info(f"Attempting to load model from alternative path: {alt_path}")
                                model = tf.keras.models.load_model(alt_path)
                                logger.info(f"Successfully loaded model from alternative path: {alt_path}")
                                # Update the model path to the working path
                                model_path = alt_path
                                break
                            except Exception as alt_e:
                                logger.warning(f"Error loading model from alternative path {alt_path}: {str(alt_e)}")

                    # If all attempts fail, raise an error
                    if not 'model' in locals():
                        raise ValueError(f"Failed to load model from {model_path} or any alternative paths")

                y_pred = model.predict(X)
            else:
                # Machine learning model
                if model_type == 'xgboost':
                    import xgboost as xgb
                    model = xgb.Booster()
                    model.load_model(model_path)

                    # Reshape for XGBoost if needed
                    if len(X.shape) == 3:
                        samples, time_steps, features = X.shape
                        X = X.reshape(samples, time_steps * features)

                    # Ensure we have the right number of features
                    if hasattr(model, 'feature_names') and model.feature_names is not None:
                        expected_features = len(model.feature_names)
                        if X.shape[1] != expected_features:
                            # Pad or truncate features
                            if X.shape[1] < expected_features:
                                padding = np.zeros((X.shape[0], expected_features - X.shape[1]))
                                X = np.hstack([X, padding])
                            else:
                                X = X[:, :expected_features]

                    # Convert to DMatrix for prediction
                    dmatrix = xgb.DMatrix(X)
                    y_pred = model.predict(dmatrix)
                    y_pred = y_pred.reshape(-1, 1) if len(y_pred.shape) == 1 else y_pred

                elif model_type == 'lightgbm':
                    import lightgbm as lgb
                    model = lgb.Booster(model_file=model_path)

                    # Reshape for LightGBM if needed
                    if len(X.shape) == 3:
                        samples, time_steps, features = X.shape
                        X = X.reshape(samples, time_steps * features)

                    y_pred = model.predict(X)
                    y_pred = y_pred.reshape(-1, 1) if len(y_pred.shape) == 1 else y_pred
                else:
                    # Other ML models
                    import pickle
                    with open(model_path, 'rb') as f:
                        model = pickle.load(f)

                    # Reshape for ML models if needed
                    if len(X.shape) == 3:
                        samples, time_steps, features = X.shape
                        X = X.reshape(samples, time_steps * features)

                    y_pred = model.predict(X)
                    y_pred = y_pred.reshape(-1, 1) if len(y_pred.shape) == 1 else y_pred

            # Calculate metrics
            metrics = self._calculate_metrics(y, y_pred)

            # Add additional information
            metrics['symbol'] = symbol
            metrics['model_type'] = model_type
            metrics['evaluation_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            metrics['data_points'] = len(y)

            return metrics

        except Exception as e:
            logger.error(f"Error evaluating model: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return {}

    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate evaluation metrics

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            Dictionary of metrics
        """
        # Ensure shapes match
        if y_true.shape != y_pred.shape:
            # If predicting multiple steps ahead, take the first step for simplicity
            if len(y_true.shape) > 1 and y_true.shape[1] > 1 and len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
                y_true = y_true[:, 0].reshape(-1, 1)
                y_pred = y_pred[:, 0].reshape(-1, 1)

        # Flatten arrays if needed
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()

        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_true_flat, y_pred_flat))
        mae = mean_absolute_error(y_true_flat, y_pred_flat)
        r2 = r2_score(y_true_flat, y_pred_flat)

        # Calculate MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((y_true_flat - y_pred_flat) / (y_true_flat + 1e-10))) * 100

        # Calculate directional accuracy (how often the model correctly predicts the direction of change)
        direction_true = np.diff(y_true_flat)
        direction_pred = np.diff(y_pred_flat)
        directional_accuracy = np.mean((direction_true > 0) == (direction_pred > 0)) * 100

        return {
            'rmse': float(rmse),
            'mae': float(mae),
            'r2': float(r2),
            'mape': float(mape),
            'directional_accuracy': float(directional_accuracy)
        }
