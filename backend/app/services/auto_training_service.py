import pandas as pd
import numpy as np
import pandas.api.types
import os
import logging
import json
import traceback
import time
import threading

# Try to import schedule, but provide a fallback if it's not available
try:
    import schedule
    SCHEDULE_AVAILABLE = True
except ImportError:
    SCHEDULE_AVAILABLE = False
    # Create a simple mock for schedule to avoid errors
    class ScheduleMock:
        def __init__(self):
            pass

        def every(self, interval):
            return self

        def hours(self):
            return self

        def do(self, func):
            return func

        def run_pending(self):
            pass

    schedule = ScheduleMock()
from typing import List, Dict, Optional, Union, Any, Tuple
from datetime import datetime, timedelta

from app.services.training_service import TrainingService
from app.services.data_service import DataService
from app.services.sentiment_service import SentimentService
from app.utils.feature_engineering import FeatureEngineer
from app.utils.transfer_learning import ContinualLearningModel
from app.utils.model_evaluation import ModelEvaluator
from app.core.config import settings

logger = logging.getLogger(__name__)

class AutoTrainingService:
    """Service for automatic model training, continuous learning, and accuracy improvement"""

    def __init__(
        self,
        data_service: DataService = None,
        training_service: TrainingService = None,
        sentiment_service: SentimentService = None
    ):
        """
        Initialize auto training service

        Args:
            data_service: DataService instance
            training_service: TrainingService instance
            sentiment_service: SentimentService instance
        """
        self.data_service = data_service or DataService()
        self.training_service = training_service or TrainingService(data_service)
        self.sentiment_service = sentiment_service or SentimentService()
        self.models_dir = settings.MODELS_DIR
        self.auto_training_enabled = True
        self.training_thread = None
        self.training_interval_hours = 24  # Train models once a day by default
        self.evaluation_interval_hours = 12  # Evaluate models twice a day
        self.continuous_learning_interval_hours = 6  # Update models with new data every 6 hours
        self.cryptocurrencies = settings.SUPPORTED_CRYPTOCURRENCIES
        self.model_types = ['lstm', 'gru', 'xgboost', 'lightgbm']  # Default model types to train
        self.last_training_time = {}  # Track last training time for each model
        self.last_evaluation_time = {}  # Track last evaluation time for each model
        self.last_continuous_learning_time = {}  # Track last continuous learning time for each model
        self.model_performance = {}  # Track model performance metrics

        # Initialize performance tracking for each cryptocurrency and model type
        for symbol in self.cryptocurrencies:
            self.model_performance[symbol] = {}
            self.last_training_time[symbol] = {}
            self.last_evaluation_time[symbol] = {}
            self.last_continuous_learning_time[symbol] = {}

            for model_type in self.model_types:
                self.model_performance[symbol][model_type] = {
                    'rmse': None,  # Use None instead of float('inf') for JSON serialization
                    'mae': None,   # Use None instead of float('inf') for JSON serialization
                    'r2': 0.0,
                    'accuracy_trend': [],
                    'last_updated': None
                }
                self.last_training_time[symbol][model_type] = None
                self.last_evaluation_time[symbol][model_type] = None
                self.last_continuous_learning_time[symbol][model_type] = None

    def start_auto_training(self):
        """Start the automatic training process in a background thread"""
        if self.training_thread is not None and self.training_thread.is_alive():
            logger.warning("Auto-training is already running")
            return

        self.auto_training_enabled = True
        self.training_thread = threading.Thread(target=self._auto_training_loop, daemon=True)
        self.training_thread.start()
        logger.info("Auto-training process started")

    def stop_auto_training(self):
        """Stop the automatic training process"""
        self.auto_training_enabled = False
        if self.training_thread is not None and self.training_thread.is_alive():
            self.training_thread.join(timeout=5)
        logger.info("Auto-training process stopped")

    def _auto_training_loop(self):
        """Main loop for automatic training"""
        logger.info("Starting auto-training loop")

        if SCHEDULE_AVAILABLE:
            # Schedule tasks
            schedule.every(self.training_interval_hours).hours.do(self._train_all_models)
            schedule.every(self.evaluation_interval_hours).hours.do(self._evaluate_all_models)
            schedule.every(self.continuous_learning_interval_hours).hours.do(self._update_all_models)

            # Run initial training for models that don't exist yet
            self._initial_model_check()

            # Main loop
            while self.auto_training_enabled:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
        else:
            logger.warning("Schedule package not available. Auto-training will run once and then exit.")
            # Run initial training for models that don't exist yet
            self._initial_model_check()

            # Run all tasks once
            self._train_all_models()
            self._evaluate_all_models()
            self._update_all_models()

    def _initial_model_check(self):
        """Check for existing models and train if needed"""
        logger.info("Performing initial model check")

        for symbol in self.cryptocurrencies:
            for model_type in self.model_types:
                # Check if model exists
                model_exists = self._check_model_exists(symbol, model_type)

                if not model_exists:
                    logger.info(f"No existing model found for {symbol} using {model_type}. Training new model.")
                    try:
                        self._train_model(symbol, model_type)
                    except Exception as e:
                        logger.error(f"Error training initial model for {symbol} using {model_type}: {str(e)}")

    def _check_model_exists(self, symbol: str, model_type: str) -> bool:
        """Check if a model exists for the given symbol and model type"""
        model_dir = os.path.join(self.models_dir, model_type)
        if not os.path.exists(model_dir):
            return False

        # Check for model files matching the symbol
        for filename in os.listdir(model_dir):
            if filename.startswith(f"{symbol.lower()}_") and filename.endswith((".h5", ".pkl")):
                return True

        return False

    def _train_all_models(self):
        """Train models for all cryptocurrencies"""
        logger.info("Starting scheduled training for all models")

        for symbol in self.cryptocurrencies:
            for model_type in self.model_types:
                try:
                    # Check if it's time to train this model
                    last_training = self.last_training_time[symbol].get(model_type)
                    if last_training is None or (datetime.now() - last_training).total_seconds() >= self.training_interval_hours * 3600:
                        logger.info(f"Training model for {symbol} using {model_type}")
                        self._train_model(symbol, model_type)
                except Exception as e:
                    logger.error(f"Error training model for {symbol} using {model_type}: {str(e)}")

    def _train_model(self, symbol: str, model_type: str):
        """Train a model for a specific cryptocurrency and model type"""
        try:
            # Train the model
            training_result = self.training_service.train_model(
                symbol=symbol,
                model_type=model_type,
                lookback=30,
                horizon=7,
                hyperparameter_tuning=True,  # Enable hyperparameter tuning for better performance
                use_feature_selection=True,
                use_anomaly_detection=True
            )

            # Update tracking
            self.last_training_time[symbol][model_type] = datetime.now()

            # Update performance metrics
            if training_result and hasattr(training_result, 'metrics'):
                metrics = training_result.metrics
                self.model_performance[symbol][model_type] = {
                    'rmse': metrics.get('rmse', float('inf')),
                    'mae': metrics.get('mae', float('inf')),
                    'r2': metrics.get('r2', 0.0),
                    'accuracy_trend': self.model_performance[symbol][model_type].get('accuracy_trend', []) + [metrics.get('r2', 0.0)],
                    'last_updated': datetime.now()
                }

            logger.info(f"Successfully trained model for {symbol} using {model_type}")
            return training_result
        except Exception as e:
            logger.error(f"Error in _train_model for {symbol} using {model_type}: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def _evaluate_all_models(self):
        """Evaluate all models to track performance"""
        logger.info("Starting scheduled evaluation for all models")

        for symbol in self.cryptocurrencies:
            for model_type in self.model_types:
                try:
                    # Check if it's time to evaluate this model
                    last_eval = self.last_evaluation_time[symbol].get(model_type)
                    if last_eval is None or (datetime.now() - last_eval).total_seconds() >= self.evaluation_interval_hours * 3600:
                        logger.info(f"Evaluating model for {symbol} using {model_type}")
                        self._evaluate_model(symbol, model_type)
                except Exception as e:
                    logger.error(f"Error evaluating model for {symbol} using {model_type}: {str(e)}")

    def _evaluate_model(self, symbol: str, model_type: str):
        """Evaluate a model for a specific cryptocurrency and model type"""
        try:
            # Get the latest data for evaluation
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)  # Use last 30 days for evaluation

            # Get price data using the load_data method
            try:
                df = self.data_service.load_data(symbol=symbol)

                # Filter data by date if needed
                if 'timestamp' in df.columns:
                    # Convert timestamp to datetime if needed
                    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                        df['timestamp'] = pd.to_datetime(df['timestamp'])

                    # Filter by date range
                    start_date_str = start_date.strftime('%Y-%m-%d')
                    end_date_str = end_date.strftime('%Y-%m-%d')
                    df = df[(df['timestamp'] >= start_date_str) & (df['timestamp'] <= end_date_str)]
                else:
                    logger.warning(f"No timestamp column in data for {symbol}, using all available data")
            except Exception as e:
                logger.error(f"Error loading data for {symbol}: {str(e)}")
                # Try to fetch data if loading fails
                try:
                    logger.info(f"Attempting to fetch data for {symbol}")
                    self.data_service.fetch_data(symbol=symbol, days=365)
                    df = self.data_service.load_data(symbol=symbol)
                except Exception as fetch_error:
                    logger.error(f"Error fetching data for {symbol}: {str(fetch_error)}")
                    return None

            if df.empty:
                logger.warning(f"No data available for evaluating {symbol} model")
                return

            # Create evaluator
            evaluator = ModelEvaluator()

            # Evaluate the model
            metrics = evaluator.evaluate_model(
                symbol=symbol,
                model_type=model_type,
                test_data=df,
                lookback=30,
                horizon=7
            )

            # Update tracking
            self.last_evaluation_time[symbol][model_type] = datetime.now()

            # Update performance metrics
            if metrics:
                current_metrics = self.model_performance[symbol][model_type]
                current_metrics['rmse'] = metrics.get('rmse', current_metrics['rmse'])
                current_metrics['mae'] = metrics.get('mae', current_metrics['mae'])
                current_metrics['r2'] = metrics.get('r2', current_metrics['r2'])
                current_metrics['accuracy_trend'].append(metrics.get('r2', 0.0))
                current_metrics['last_updated'] = datetime.now()

                # Keep only the last 10 accuracy measurements
                if len(current_metrics['accuracy_trend']) > 10:
                    current_metrics['accuracy_trend'] = current_metrics['accuracy_trend'][-10:]

                self.model_performance[symbol][model_type] = current_metrics

            logger.info(f"Successfully evaluated model for {symbol} using {model_type}")
            return metrics
        except Exception as e:
            logger.error(f"Error in _evaluate_model for {symbol} using {model_type}: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def _update_all_models(self):
        """Update all models with new data (continuous learning)"""
        logger.info("Starting scheduled continuous learning for all models")

        for symbol in self.cryptocurrencies:
            for model_type in self.model_types:
                try:
                    # Check if it's time to update this model
                    last_update = self.last_continuous_learning_time[symbol].get(model_type)
                    if last_update is None or (datetime.now() - last_update).total_seconds() >= self.continuous_learning_interval_hours * 3600:
                        logger.info(f"Updating model for {symbol} using {model_type} with new data")
                        self._update_model(symbol, model_type)
                except Exception as e:
                    logger.error(f"Error updating model for {symbol} using {model_type}: {str(e)}")

    def _update_model(self, symbol: str, model_type: str):
        """Update a model with new data (continuous learning)"""
        try:
            # Get the latest data for continuous learning
            end_date = datetime.now()
            start_date = end_date - timedelta(days=7)  # Use last 7 days for updating

            # Get price data using the load_data method
            try:
                df = self.data_service.load_data(symbol=symbol)

                # Filter data by date if needed
                if 'timestamp' in df.columns:
                    # Convert timestamp to datetime if needed
                    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                        df['timestamp'] = pd.to_datetime(df['timestamp'])

                    # Filter by date range
                    start_date_str = start_date.strftime('%Y-%m-%d')
                    end_date_str = end_date.strftime('%Y-%m-%d')
                    df = df[(df['timestamp'] >= start_date_str) & (df['timestamp'] <= end_date_str)]
                else:
                    logger.warning(f"No timestamp column in data for {symbol}, using all available data")
            except Exception as e:
                logger.error(f"Error loading data for {symbol}: {str(e)}")
                # Try to fetch data if loading fails
                try:
                    logger.info(f"Attempting to fetch data for {symbol}")
                    self.data_service.fetch_data(symbol=symbol, days=30)
                    df = self.data_service.load_data(symbol=symbol)
                except Exception as fetch_error:
                    logger.error(f"Error fetching data for {symbol}: {str(fetch_error)}")
                    return None

            if df.empty:
                logger.warning(f"No new data available for updating {symbol} model")
                return

            # Find the latest model file
            model_dir = os.path.join(self.models_dir, model_type)
            if not os.path.exists(model_dir):
                logger.error(f"Model directory not found: {model_dir}")
                return False

            # Find the latest model file for this symbol
            model_files = [f for f in os.listdir(model_dir) if f.startswith(f"{symbol.lower()}_") and f.endswith((".h5", ".keras", ".pkl"))]
            if not model_files:
                logger.error(f"No model files found for {symbol} in {model_dir}")
                return False

            # Sort by date (assuming filename format includes date)
            model_files.sort(reverse=True)
            model_path = os.path.join(model_dir, model_files[0])

            # Create continuous learning model
            try:
                cl_model = ContinualLearningModel(base_model_path=model_path)
            except Exception as e:
                logger.error(f"Error creating ContinualLearningModel: {str(e)}")
                # For now, let's just log the error and return False
                return False

            # Update the model with new data
            updated = cl_model.update_model(
                symbol=symbol,
                new_data=df,
                lookback=30,
                horizon=7
            )

            # Update tracking
            if updated:
                self.last_continuous_learning_time[symbol][model_type] = datetime.now()
                logger.info(f"Successfully updated model for {symbol} using {model_type}")
            else:
                logger.warning(f"Model update not performed for {symbol} using {model_type}")

            return updated
        except Exception as e:
            logger.error(f"Error in _update_model for {symbol} using {model_type}: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def get_model_performance(self, symbol: str = None, model_type: str = None):
        """Get performance metrics for models"""
        if symbol and model_type:
            return self._get_serializable_performance(symbol, model_type)
        elif symbol:
            result = {}
            for mt in self.model_types:
                result[mt] = self._get_serializable_performance(symbol, mt)
            return result
        else:
            result = {}
            for sym in self.cryptocurrencies:
                result[sym] = {}
                for mt in self.model_types:
                    result[sym][mt] = self._get_serializable_performance(sym, mt)
            return result

    def get_training_status(self, symbol: str = None, model_type: str = None):
        """Get training status for models"""
        status = {}

        if symbol and model_type:
            symbols = [symbol]
            model_types = [model_type]
        elif symbol:
            symbols = [symbol]
            model_types = self.model_types
        else:
            symbols = self.cryptocurrencies
            model_types = self.model_types

        for sym in symbols:
            status[sym] = {}
            for mt in model_types:
                # Get performance metrics and ensure they're JSON serializable
                performance = self._get_serializable_performance(sym, mt)

                # Convert datetime objects to ISO format strings
                last_training = self.last_training_time.get(sym, {}).get(mt)
                last_evaluation = self.last_evaluation_time.get(sym, {}).get(mt)
                last_update = self.last_continuous_learning_time.get(sym, {}).get(mt)

                status[sym][mt] = {
                    'model_exists': self._check_model_exists(sym, mt),
                    'last_training': last_training.isoformat() if last_training else None,
                    'last_evaluation': last_evaluation.isoformat() if last_evaluation else None,
                    'last_update': last_update.isoformat() if last_update else None,
                    'performance': performance
                }

        return status

    def _get_serializable_performance(self, symbol: str, model_type: str):
        """Get performance metrics in a JSON-serializable format"""
        performance = self.model_performance.get(symbol, {}).get(model_type, {})
        result = {}

        # Convert all values to ensure they're JSON serializable
        for key, value in performance.items():
            if key == 'accuracy_trend':
                # Ensure all values in the list are serializable
                result[key] = [float(v) if v != float('inf') and v != float('-inf') else None for v in value]
            elif isinstance(value, float):
                # Replace infinity with None
                if value == float('inf') or value == float('-inf'):
                    result[key] = None
                else:
                    result[key] = float(value)
            elif isinstance(value, (int, str, bool, type(None))):
                result[key] = value
            elif hasattr(value, 'isoformat'):  # For datetime objects
                result[key] = value.isoformat()
            else:
                # For other types, convert to string
                result[key] = str(value)

        return result

    def force_train_model(self, symbol: str, model_type: str):
        """Force training of a specific model"""
        logger.info(f"Force training model for {symbol} using {model_type}")
        return self._train_model(symbol, model_type)

    def force_evaluate_model(self, symbol: str, model_type: str):
        """Force evaluation of a specific model"""
        logger.info(f"Force evaluating model for {symbol} using {model_type}")
        return self._evaluate_model(symbol, model_type)

    def force_update_model(self, symbol: str, model_type: str):
        """Force update of a specific model with new data"""
        logger.info(f"Force updating model for {symbol} using {model_type}")
        return self._update_model(symbol, model_type)

# Create a singleton instance
auto_training_service = AutoTrainingService()
