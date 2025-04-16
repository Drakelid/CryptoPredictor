import pandas as pd
import numpy as np
import os
import logging
import json
import pickle
from typing import List, Dict, Optional, Union, Any
from datetime import datetime
import shap

from app.models.dl_models import get_dl_model
from app.models.ml_models import get_ml_model
from app.utils.feature_engineering import FeatureEngineer
from app.services.data_service import DataService
from app.services.training_service import TrainingService
from app.schemas.prediction import ExplanationResult, FeatureImportance
from app.core.config import settings

logger = logging.getLogger(__name__)

class ExplanationService:
    """Service for explaining cryptocurrency price predictions"""

    def __init__(
        self,
        data_service: DataService = None,
        training_service: TrainingService = None
    ):
        """
        Initialize explanation service

        Args:
            data_service: DataService instance
            training_service: TrainingService instance
        """
        self.data_service = data_service or DataService()
        self.training_service = training_service or TrainingService(data_service)
        self.models_dir = settings.MODELS_DIR
        self.predictions_dir = os.path.join(self.models_dir, 'predictions')

    def explain(
        self,
        symbol: str,
        model_type: str,
        prediction_id: Optional[str] = None
    ) -> ExplanationResult:
        """
        Explain a prediction

        Args:
            symbol: Cryptocurrency symbol (e.g., BTC, ETH)
            model_type: Model type (lstm, gru, xgboost, lightgbm)
            prediction_id: ID of prediction to explain

        Returns:
            ExplanationResult object with explanation information
        """
        # Validate model type
        if model_type not in settings.SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model type: {model_type}. Supported types: {settings.SUPPORTED_MODELS}")

        # Get model info
        model_info = self.training_service.get_model_info(symbol=symbol, model_type=model_type)

        if not model_info:
            raise ValueError(f"No trained model found for {symbol} with type {model_type}")

        # Get the latest model
        latest_model = model_info[0]
        model_path = latest_model.model_path

        # Load data
        df = self.data_service.load_data(symbol=symbol)

        # Engineer features
        feature_engineer = FeatureEngineer()
        df_features = feature_engineer.engineer_features(df, symbol=symbol)

        # Determine target column
        target_col = 'close' if 'close' in df_features.columns else 'price'

        # Get prediction data if prediction_id is provided
        prediction_data = None
        if prediction_id:
            prediction_data = self._load_prediction(symbol, model_type, prediction_id)

        # Generate explanation based on model type
        if model_type in ['xgboost', 'lightgbm']:
            # ML models are easier to explain with SHAP
            explanation = self._explain_ml_model(
                df_features=df_features,
                target_col=target_col,
                model_path=model_path,
                model_type=model_type
            )
        else:
            # DL models require a different approach
            explanation = self._explain_dl_model(
                df_features=df_features,
                target_col=target_col,
                model_path=model_path,
                model_type=model_type
            )

        # Create explanation result
        result = ExplanationResult(
            symbol=symbol,
            model_type=model_type,
            prediction_date=prediction_data.get('prediction_date', datetime.now()) if prediction_data else datetime.now(),
            feature_importance=explanation['feature_importance'],
            shap_values=explanation.get('shap_values')
        )

        return result

    def _explain_ml_model(
        self,
        df_features: pd.DataFrame,
        target_col: str,
        model_path: str,
        model_type: str
    ) -> Dict[str, Any]:
        """
        Explain ML model prediction

        Args:
            df_features: DataFrame with engineered features
            target_col: Target column to predict
            model_path: Path to model file
            model_type: Model type (xgboost, lightgbm)

        Returns:
            Dictionary with explanation information
        """
        # Get model parameters from metadata
        metadata_path = f"{os.path.splitext(model_path)[0]}_metadata.json"
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        lookback = metadata.get('lookback', 30)
        horizon = metadata.get('horizon', 7)

        # Prepare input features
        feature_engineer = FeatureEngineer()
        X, _, _, _, _ = feature_engineer.prepare_tabular_data(
            df=df_features,
            target_col=target_col,
            lookback=lookback,
            horizon=horizon,
            train_ratio=1.0  # Use all data for explanation
        )

        # Get feature names
        feature_cols = [col for col in df_features.columns if col not in ['timestamp', target_col]]
        for col in feature_cols:
            for lag in range(1, lookback + 1):
                feature_cols.append(f"{col}_lag_{lag}")

        # Load model
        model = get_ml_model(
            model_type=model_type,
            model_path=model_path
        )

        # Get feature importance
        feature_importance = []
        if model.feature_importance is not None:
            # Sort features by importance
            indices = np.argsort(model.feature_importance)[::-1]

            # Create feature importance list
            for i in indices:
                if i < len(feature_cols):
                    feature_importance.append(
                        FeatureImportance(
                            feature=feature_cols[i],
                            importance=float(model.feature_importance[i])
                        )
                    )

        # Calculate SHAP values
        try:
            # Create explainer
            explainer = shap.Explainer(model.model)

            # Calculate SHAP values
            shap_values = explainer(X)

            # Convert to dictionary
            shap_dict = {}
            for i, feature in enumerate(feature_cols[:X.shape[1]]):
                shap_dict[feature] = shap_values.values[:, i].tolist()

            return {
                'feature_importance': feature_importance,
                'shap_values': shap_dict
            }

        except Exception as e:
            logger.error(f"Error calculating SHAP values: {e}")

            # Return feature importance only
            return {
                'feature_importance': feature_importance
            }

    def _explain_dl_model(
        self,
        df_features: pd.DataFrame,
        target_col: str,
        model_path: str,
        model_type: str
    ) -> Dict[str, Any]:
        """
        Explain DL model prediction

        Args:
            df_features: DataFrame with engineered features
            target_col: Target column to predict
            model_path: Path to model file
            model_type: Model type (lstm, gru)

        Returns:
            Dictionary with explanation information
        """
        # Get model parameters from metadata
        metadata_path = f"{os.path.splitext(model_path)[0]}_metadata.json"
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        lookback = metadata.get('lookback', 30)
        horizon = metadata.get('horizon', 7)

        # Prepare input sequence
        feature_engineer = FeatureEngineer()
        X, _, _, _, _ = feature_engineer.prepare_sequences(
            df=df_features,
            target_col=target_col,
            lookback=lookback,
            horizon=horizon,
            train_ratio=1.0  # Use all data for explanation
        )

        # Get feature names
        feature_cols = [col for col in df_features.columns if col not in ['timestamp', target_col]]

        # For DL models, we don't have direct feature importance
        # We can use integrated gradients or other methods, but for simplicity
        # we'll just use a heuristic approach based on feature correlation with target

        # Calculate correlation with target
        correlations = df_features.corr()[target_col].abs().sort_values(ascending=False)

        # Create feature importance list
        feature_importance = []
        for feature, corr in correlations.items():
            if feature != target_col:
                feature_importance.append(
                    FeatureImportance(
                        feature=feature,
                        importance=float(corr)
                    )
                )

        # For DL models, SHAP values are more complex to calculate
        # We'll skip them for now

        return {
            'feature_importance': feature_importance
        }

    def _load_prediction(self, symbol: str, model_type: str, prediction_id: str) -> Optional[Dict[str, Any]]:
        """
        Load prediction data

        Args:
            symbol: Cryptocurrency symbol (e.g., BTC, ETH)
            model_type: Model type (lstm, gru, xgboost, lightgbm)
            prediction_id: ID of prediction to load

        Returns:
            Dictionary with prediction data or None if not found
        """
        # Find prediction file
        file_path = os.path.join(
            self.predictions_dir,
            f"{symbol.lower()}_{model_type}_{prediction_id}.json"
        )

        if not os.path.exists(file_path):
            logger.warning(f"Prediction file not found: {file_path}")
            return None

        # Load prediction data
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)

            # Convert prediction date to datetime
            if 'prediction_date' in data:
                data['prediction_date'] = datetime.fromisoformat(data['prediction_date'])

            return data

        except Exception as e:
            logger.error(f"Error loading prediction data: {e}")
            return None
