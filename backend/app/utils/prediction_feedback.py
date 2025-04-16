import pandas as pd
import numpy as np
import os
import logging
import json
import datetime
from typing import Dict, List, Optional, Any, Tuple
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from app.utils.transfer_learning import ContinualLearningModel
from app.core.config import settings

logger = logging.getLogger(__name__)

class PredictionFeedbackSystem:
    """
    System for collecting feedback on predictions and using it to improve models
    
    This system:
    1. Stores predictions with their timestamps
    2. Compares predictions with actual values when they become available
    3. Calculates prediction accuracy metrics
    4. Uses feedback to improve models through continuous learning
    """
    
    def __init__(self):
        """Initialize the prediction feedback system"""
        self.models_dir = settings.MODELS_DIR
        self.feedback_dir = os.path.join(self.models_dir, 'feedback')
        os.makedirs(self.feedback_dir, exist_ok=True)
        
        # Initialize metrics tracking
        self.metrics_file = os.path.join(self.feedback_dir, 'prediction_metrics.json')
        self.metrics = self._load_metrics()
    
    def _load_metrics(self) -> Dict:
        """Load prediction metrics from file"""
        if os.path.exists(self.metrics_file):
            try:
                with open(self.metrics_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading prediction metrics: {str(e)}")
                return {}
        return {}
    
    def _save_metrics(self):
        """Save prediction metrics to file"""
        try:
            with open(self.metrics_file, 'w') as f:
                json.dump(self.metrics, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving prediction metrics: {str(e)}")
    
    def store_prediction(
        self,
        symbol: str,
        model_type: str,
        prediction_values: List[float],
        prediction_timestamps: List[str],
        confidence_lower: Optional[List[float]] = None,
        confidence_upper: Optional[List[float]] = None
    ):
        """
        Store a prediction for later comparison with actual values
        
        Args:
            symbol: Cryptocurrency symbol
            model_type: Type of model used for prediction
            prediction_values: List of predicted values
            prediction_timestamps: List of timestamps for predictions
            confidence_lower: Lower confidence interval (optional)
            confidence_upper: Upper confidence interval (optional)
        """
        try:
            # Create prediction record
            prediction_record = {
                'symbol': symbol,
                'model_type': model_type,
                'prediction_values': prediction_values,
                'prediction_timestamps': prediction_timestamps,
                'confidence_lower': confidence_lower,
                'confidence_upper': confidence_upper,
                'created_at': datetime.datetime.now().isoformat(),
                'feedback_processed': False
            }
            
            # Create directory for symbol if it doesn't exist
            symbol_dir = os.path.join(self.feedback_dir, symbol.lower())
            os.makedirs(symbol_dir, exist_ok=True)
            
            # Save prediction record
            prediction_id = f"{symbol.lower()}_{model_type}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
            prediction_file = os.path.join(symbol_dir, f"{prediction_id}.json")
            
            with open(prediction_file, 'w') as f:
                json.dump(prediction_record, f, indent=2)
            
            logger.info(f"Stored prediction {prediction_id} for later feedback")
            return prediction_id
            
        except Exception as e:
            logger.error(f"Error storing prediction: {str(e)}")
            return None
    
    def process_feedback(self, max_days_old: int = 30):
        """
        Process feedback for all stored predictions
        
        This method:
        1. Finds all stored predictions that haven't been processed
        2. For each prediction, checks if actual values are now available
        3. If actual values are available, calculates accuracy metrics
        4. Updates models based on the feedback
        
        Args:
            max_days_old: Maximum age of predictions to process (in days)
        """
        try:
            # Get current date
            current_date = datetime.datetime.now()
            
            # Find all symbol directories
            for symbol_dir in os.listdir(self.feedback_dir):
                symbol_path = os.path.join(self.feedback_dir, symbol_dir)
                
                # Skip if not a directory or is the metrics file
                if not os.path.isdir(symbol_path) or symbol_dir == 'prediction_metrics.json':
                    continue
                
                symbol = symbol_dir.upper()
                logger.info(f"Processing feedback for {symbol}")
                
                # Find all prediction files
                for prediction_file in os.listdir(symbol_path):
                    if not prediction_file.endswith('.json'):
                        continue
                    
                    prediction_path = os.path.join(symbol_path, prediction_file)
                    
                    try:
                        # Load prediction record
                        with open(prediction_path, 'r') as f:
                            prediction_record = json.load(f)
                        
                        # Skip if already processed
                        if prediction_record.get('feedback_processed', False):
                            continue
                        
                        # Check if prediction is too old
                        created_at = datetime.datetime.fromisoformat(prediction_record['created_at'])
                        if (current_date - created_at).days > max_days_old:
                            logger.info(f"Skipping old prediction {prediction_file}")
                            continue
                        
                        # Process feedback for this prediction
                        self._process_prediction_feedback(prediction_record, prediction_path)
                        
                    except Exception as e:
                        logger.error(f"Error processing prediction file {prediction_file}: {str(e)}")
            
            # Save updated metrics
            self._save_metrics()
            
        except Exception as e:
            logger.error(f"Error processing feedback: {str(e)}")
    
    def _process_prediction_feedback(self, prediction_record: Dict, prediction_path: str):
        """
        Process feedback for a single prediction
        
        Args:
            prediction_record: Prediction record
            prediction_path: Path to prediction file
        """
        try:
            symbol = prediction_record['symbol']
            model_type = prediction_record['model_type']
            prediction_values = prediction_record['prediction_values']
            prediction_timestamps = prediction_record['prediction_timestamps']
            
            # Convert timestamps to datetime objects
            prediction_dates = [datetime.datetime.fromisoformat(ts) for ts in prediction_timestamps]
            
            # Check which predictions now have actual values available
            actual_values = []
            actual_dates = []
            missing_dates = []
            
            # Get actual price data
            from app.services.data_service import DataService
            data_service = DataService()
            
            try:
                # Load the latest data
                df = data_service.load_data(symbol=symbol)
                
                if df is not None and not df.empty:
                    # Ensure timestamp column is datetime
                    if 'timestamp' in df.columns:
                        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                            df['timestamp'] = pd.to_datetime(df['timestamp'])
                        
                        # Get price column (close or price)
                        price_col = 'close' if 'close' in df.columns else 'price'
                        
                        # Match predictions with actual values
                        for i, pred_date in enumerate(prediction_dates):
                            # Find the closest actual date
                            closest_idx = None
                            min_diff = datetime.timedelta(days=1)  # Maximum 1 day difference
                            
                            for j, actual_date in enumerate(df['timestamp']):
                                diff = abs(pred_date - actual_date)
                                if diff < min_diff:
                                    min_diff = diff
                                    closest_idx = j
                            
                            if closest_idx is not None:
                                actual_values.append(float(df.iloc[closest_idx][price_col]))
                                actual_dates.append(df.iloc[closest_idx]['timestamp'])
                            else:
                                missing_dates.append(pred_date)
                    else:
                        logger.warning(f"No timestamp column in data for {symbol}")
                else:
                    logger.warning(f"No data available for {symbol}")
            except Exception as e:
                logger.error(f"Error getting actual values for {symbol}: {str(e)}")
            
            # If we have at least some actual values, calculate metrics
            if actual_values:
                # Get matching predictions
                matched_predictions = [prediction_values[i] for i in range(len(prediction_values)) 
                                      if prediction_dates[i] not in missing_dates]
                
                # Calculate metrics
                if len(matched_predictions) == len(actual_values):
                    metrics = self._calculate_metrics(actual_values, matched_predictions)
                    
                    # Update metrics tracking
                    if symbol not in self.metrics:
                        self.metrics[symbol] = {}
                    
                    if model_type not in self.metrics[symbol]:
                        self.metrics[symbol][model_type] = {
                            'rmse': [],
                            'mae': [],
                            'r2': [],
                            'directional_accuracy': [],
                            'last_updated': None
                        }
                    
                    # Add new metrics
                    self.metrics[symbol][model_type]['rmse'].append(metrics['rmse'])
                    self.metrics[symbol][model_type]['mae'].append(metrics['mae'])
                    self.metrics[symbol][model_type]['r2'].append(metrics['r2'])
                    self.metrics[symbol][model_type]['directional_accuracy'].append(metrics['directional_accuracy'])
                    self.metrics[symbol][model_type]['last_updated'] = datetime.datetime.now().isoformat()
                    
                    # Keep only the last 10 metrics
                    for key in ['rmse', 'mae', 'r2', 'directional_accuracy']:
                        if len(self.metrics[symbol][model_type][key]) > 10:
                            self.metrics[symbol][model_type][key] = self.metrics[symbol][model_type][key][-10:]
                    
                    # Update prediction record
                    prediction_record['actual_values'] = actual_values
                    prediction_record['actual_dates'] = [d.isoformat() for d in actual_dates]
                    prediction_record['metrics'] = metrics
                    prediction_record['feedback_processed'] = True
                    prediction_record['processed_at'] = datetime.datetime.now().isoformat()
                    
                    # Save updated prediction record
                    with open(prediction_path, 'w') as f:
                        json.dump(prediction_record, f, indent=2)
                    
                    # Update model with feedback
                    self._update_model_with_feedback(symbol, model_type, actual_values, matched_predictions, actual_dates)
                    
                    logger.info(f"Processed feedback for {symbol} {model_type} prediction: RMSE={metrics['rmse']:.4f}, R²={metrics['r2']:.4f}")
                else:
                    logger.warning(f"Mismatched lengths: predictions={len(matched_predictions)}, actual={len(actual_values)}")
            else:
                logger.info(f"No actual values available yet for {symbol} prediction")
                
        except Exception as e:
            logger.error(f"Error processing prediction feedback: {str(e)}")
    
    def _calculate_metrics(self, actual_values: List[float], predicted_values: List[float]) -> Dict:
        """
        Calculate prediction accuracy metrics
        
        Args:
            actual_values: List of actual values
            predicted_values: List of predicted values
            
        Returns:
            Dictionary of metrics
        """
        # Convert to numpy arrays
        y_true = np.array(actual_values)
        y_pred = np.array(predicted_values)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        
        # Handle edge case for R²
        if len(y_true) <= 1 or np.all(y_true == y_true[0]):
            r2 = 0.0  # R² is undefined when all actual values are the same
        else:
            r2 = r2_score(y_true, y_pred)
        
        # Calculate directional accuracy
        if len(y_true) > 1:
            direction_true = np.diff(y_true) > 0
            direction_pred = np.diff(y_pred) > 0
            directional_accuracy = np.mean(direction_true == direction_pred) * 100
        else:
            directional_accuracy = 0.0
        
        return {
            'rmse': float(rmse),
            'mae': float(mae),
            'r2': float(r2),
            'directional_accuracy': float(directional_accuracy),
            'sample_size': len(y_true)
        }
    
    def _update_model_with_feedback(
        self,
        symbol: str,
        model_type: str,
        actual_values: List[float],
        predicted_values: List[float],
        actual_dates: List[datetime.datetime]
    ):
        """
        Update model with feedback from predictions
        
        Args:
            symbol: Cryptocurrency symbol
            model_type: Type of model
            actual_values: List of actual values
            predicted_values: List of predicted values
            actual_dates: List of dates for actual values
        """
        try:
            # Find the latest model file
            model_dir = os.path.join(self.models_dir, model_type)
            if not os.path.exists(model_dir):
                logger.error(f"Model directory not found: {model_dir}")
                return
            
            # Find the latest model file for this symbol
            model_files = [f for f in os.listdir(model_dir) if f.startswith(f"{symbol.lower()}_") and f.endswith((".h5", ".pkl"))]
            if not model_files:
                logger.error(f"No model files found for {symbol} in {model_dir}")
                return
            
            # Sort by date (assuming filename format includes date)
            model_files.sort(reverse=True)
            model_path = os.path.join(model_dir, model_files[0])
            
            # Create a dataframe with the actual values
            df = pd.DataFrame({
                'timestamp': actual_dates,
                'close': actual_values,
                'predicted': predicted_values
            })
            
            # Sort by timestamp
            df = df.sort_values('timestamp')
            
            # Create continuous learning model
            try:
                cl_model = ContinualLearningModel(base_model_path=model_path)
                
                # Update the model with new data
                updated = cl_model.update_model(
                    symbol=symbol,
                    new_data=df,
                    lookback=30,
                    horizon=7
                )
                
                if updated:
                    logger.info(f"Updated {symbol} {model_type} model with prediction feedback")
                else:
                    logger.warning(f"Failed to update {symbol} {model_type} model with prediction feedback")
                
            except Exception as e:
                logger.error(f"Error updating model with feedback: {str(e)}")
            
        except Exception as e:
            logger.error(f"Error in _update_model_with_feedback: {str(e)}")
    
    def get_prediction_metrics(self, symbol: Optional[str] = None, model_type: Optional[str] = None) -> Dict:
        """
        Get prediction metrics
        
        Args:
            symbol: Cryptocurrency symbol (optional)
            model_type: Type of model (optional)
            
        Returns:
            Dictionary of metrics
        """
        if symbol and model_type:
            return self.metrics.get(symbol, {}).get(model_type, {})
        elif symbol:
            return self.metrics.get(symbol, {})
        else:
            return self.metrics
    
    def get_prediction_history(self, symbol: str, model_type: Optional[str] = None, limit: int = 10) -> List[Dict]:
        """
        Get prediction history for a symbol
        
        Args:
            symbol: Cryptocurrency symbol
            model_type: Type of model (optional)
            limit: Maximum number of predictions to return
            
        Returns:
            List of prediction records
        """
        try:
            symbol_dir = os.path.join(self.feedback_dir, symbol.lower())
            if not os.path.exists(symbol_dir):
                return []
            
            # Find all prediction files
            prediction_files = [f for f in os.listdir(symbol_dir) if f.endswith('.json')]
            if not prediction_files:
                return []
            
            # Filter by model type if specified
            if model_type:
                prediction_files = [f for f in prediction_files if f.split('_')[1] == model_type]
            
            # Sort by creation date (newest first)
            prediction_files.sort(reverse=True)
            
            # Load prediction records
            predictions = []
            for prediction_file in prediction_files[:limit]:
                prediction_path = os.path.join(symbol_dir, prediction_file)
                try:
                    with open(prediction_path, 'r') as f:
                        prediction_record = json.load(f)
                    predictions.append(prediction_record)
                except Exception as e:
                    logger.error(f"Error loading prediction file {prediction_file}: {str(e)}")
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error getting prediction history: {str(e)}")
            return []

# Create a singleton instance
prediction_feedback_system = PredictionFeedbackSystem()
