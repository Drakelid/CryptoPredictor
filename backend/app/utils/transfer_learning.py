import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Input
import os
import logging
import json
import pickle
from typing import List, Dict, Tuple, Optional, Union, Any
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

class TransferLearningModel:
    """Class for transfer learning with pre-trained models"""

    def __init__(
        self,
        base_model_path: str,
        target_layers: Optional[List[int]] = None,
        fine_tune_layers: Optional[List[int]] = None,
        learning_rate: float = 0.001
    ):
        """
        Initialize transfer learning model

        Args:
            base_model_path: Path to pre-trained model
            target_layers: Indices of layers to use as target (None = use all)
            fine_tune_layers: Indices of layers to fine-tune (None = freeze all)
            learning_rate: Learning rate for fine-tuning
        """
        self.base_model_path = base_model_path
        self.target_layers = target_layers
        self.fine_tune_layers = fine_tune_layers
        self.learning_rate = learning_rate

        # Load base model
        self.base_model = load_model(base_model_path)

        # Load metadata
        metadata_path = f"{os.path.splitext(base_model_path)[0]}_metadata.json"
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}

        # Load scaler if available
        scaler_path = f"{os.path.splitext(base_model_path)[0]}_scaler.pkl"
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
        else:
            self.scaler = StandardScaler()

        # Create transfer model
        self.model = self._create_transfer_model()

    def _create_transfer_model(self) -> Model:
        """
        Create transfer learning model

        Returns:
            Transfer learning model
        """
        # Get base model layers
        base_layers = self.base_model.layers

        # Determine target layers
        if self.target_layers is None:
            # Use all layers except the last one
            target_layers = base_layers[:-1]
        else:
            # Use specified layers
            target_layers = [base_layers[i] for i in self.target_layers]

        # Create input layer
        input_shape = base_layers[0].input_shape[0][1:]
        inputs = Input(shape=input_shape)

        # Build model up to target layers
        x = inputs
        for layer in target_layers:
            x = layer(x)

        # Add new output layer
        outputs = Dense(1, activation='linear')(x)

        # Create model
        model = Model(inputs=inputs, outputs=outputs)

        # Freeze base layers
        for layer in target_layers:
            layer.trainable = False

        # Unfreeze fine-tune layers if specified
        if self.fine_tune_layers is not None:
            for i in self.fine_tune_layers:
                if i < len(target_layers):
                    target_layers[i].trainable = True

        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae']
        )

        return model

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        batch_size: int = 32,
        epochs: int = 100,
        callbacks: Optional[List[tf.keras.callbacks.Callback]] = None,
        verbose: int = 1
    ) -> tf.keras.callbacks.History:
        """
        Fit transfer learning model

        Args:
            X: Input features
            y: Target values
            validation_data: Validation data (optional)
            batch_size: Batch size
            epochs: Number of epochs
            callbacks: Callbacks (optional)
            verbose: Verbosity level

        Returns:
            Training history
        """
        # Scale data if scaler is available
        if hasattr(self.scaler, 'transform'):
            X = self.scaler.transform(X)

        # Fit model
        history = self.model.fit(
            X, y,
            validation_data=validation_data,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks,
            verbose=verbose
        )

        return history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions

        Args:
            X: Input features

        Returns:
            Predictions
        """
        # Scale data if scaler is available
        if hasattr(self.scaler, 'transform'):
            X = self.scaler.transform(X)

        # Make predictions
        return self.model.predict(X)

    def save(self, path: str):
        """
        Save model

        Args:
            path: Path to save model
        """
        # Convert path to .keras format if it's .h5
        if path.endswith('.h5'):
            path = path.replace('.h5', '.keras')
            logger.info(f"Changed model save format from HDF5 (.h5) to native Keras format (.keras)")

        # Save model
        self.model.save(path)

        # Save metadata
        metadata_path = f"{os.path.splitext(path)[0]}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f)

        # Save scaler
        scaler_path = f"{os.path.splitext(path)[0]}_scaler.pkl"
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)


class CrossAssetTransferLearning:
    """Class for transfer learning across different assets"""

    def __init__(
        self,
        base_models: Dict[str, str],
        target_symbol: str,
        ensemble_weights: Optional[Dict[str, float]] = None,
        learning_rate: float = 0.001
    ):
        """
        Initialize cross-asset transfer learning

        Args:
            base_models: Dictionary mapping symbols to model paths
            target_symbol: Target symbol
            ensemble_weights: Dictionary mapping symbols to weights (optional)
            learning_rate: Learning rate for fine-tuning
        """
        self.base_models = base_models
        self.target_symbol = target_symbol
        self.learning_rate = learning_rate

        # Set default weights if not provided
        if ensemble_weights is None:
            # Equal weights
            self.ensemble_weights = {symbol: 1.0 / len(base_models) for symbol in base_models}
        else:
            # Normalize weights
            total_weight = sum(ensemble_weights.values())
            self.ensemble_weights = {symbol: weight / total_weight for symbol, weight in ensemble_weights.items()}

        # Load transfer models
        self.transfer_models = {}
        for symbol, model_path in base_models.items():
            self.transfer_models[symbol] = TransferLearningModel(
                model_path,
                fine_tune_layers=[-2, -3],  # Fine-tune last two layers
                learning_rate=learning_rate
            )

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        batch_size: int = 32,
        epochs: int = 100,
        callbacks: Optional[List[tf.keras.callbacks.Callback]] = None,
        verbose: int = 1
    ):
        """
        Fit all transfer learning models

        Args:
            X: Input features
            y: Target values
            validation_data: Validation data (optional)
            batch_size: Batch size
            epochs: Number of epochs
            callbacks: Callbacks (optional)
            verbose: Verbosity level
        """
        for symbol, model in self.transfer_models.items():
            logger.info(f"Training transfer model for {symbol} -> {self.target_symbol}")
            model.fit(
                X, y,
                validation_data=validation_data,
                batch_size=batch_size,
                epochs=epochs,
                callbacks=callbacks,
                verbose=verbose
            )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make ensemble predictions

        Args:
            X: Input features

        Returns:
            Weighted ensemble predictions
        """
        # Get predictions from each model
        predictions = {}
        for symbol, model in self.transfer_models.items():
            predictions[symbol] = model.predict(X)

        # Compute weighted ensemble
        ensemble_pred = np.zeros_like(predictions[list(predictions.keys())[0]])
        for symbol, pred in predictions.items():
            ensemble_pred += pred * self.ensemble_weights[symbol]

        return ensemble_pred

    def save_models(self, base_path: str):
        """
        Save all models

        Args:
            base_path: Base path to save models
        """
        os.makedirs(base_path, exist_ok=True)

        for symbol, model in self.transfer_models.items():
            model_path = os.path.join(base_path, f"{symbol}_to_{self.target_symbol}.keras")
            model.save(model_path)
            logger.info(f"Saved transfer model from {symbol} to {self.target_symbol} using native Keras format (.keras)")

        # Save ensemble weights
        weights_path = os.path.join(base_path, f"{self.target_symbol}_ensemble_weights.json")
        with open(weights_path, 'w') as f:
            json.dump(self.ensemble_weights, f)


class ContinualLearningModel:
    """Class for continual learning with pre-trained models"""

    def __init__(
        self,
        base_model_path: str,
        window_size: int = 30,
        update_frequency: int = 7,
        learning_rate: float = 0.001
    ):
        """
        Initialize continual learning model

        Args:
            base_model_path: Path to pre-trained model
            window_size: Size of sliding window for training
            update_frequency: Frequency of model updates (in days)
            learning_rate: Learning rate for updates
        """
        self.base_model_path = base_model_path
        self.window_size = window_size
        self.update_frequency = update_frequency
        self.learning_rate = learning_rate

        # Load base model
        try:
            # Try loading the model directly
            logger.info(f"Attempting to load model from {base_model_path}")
            self.model = load_model(base_model_path)
            logger.info(f"Successfully loaded model from {base_model_path}")
        except Exception as e:
            # If loading fails, try alternative formats
            logger.warning(f"Error loading model from {base_model_path}: {str(e)}")

            # Try alternative file formats
            alt_paths = []

            # If path is .h5, try .keras
            if base_model_path.endswith('.h5'):
                alt_paths.append(base_model_path.replace('.h5', '.keras'))

            # If path is .keras, try .h5
            if base_model_path.endswith('.keras'):
                alt_paths.append(base_model_path.replace('.keras', '.h5'))

            # Try loading from alternative paths
            for alt_path in alt_paths:
                if os.path.exists(alt_path):
                    try:
                        logger.info(f"Attempting to load model from alternative path: {alt_path}")
                        self.model = load_model(alt_path)
                        logger.info(f"Successfully loaded model from alternative path: {alt_path}")
                        # Update the base model path to the working path
                        self.base_model_path = alt_path
                        break
                    except Exception as alt_e:
                        logger.warning(f"Error loading model from alternative path {alt_path}: {str(alt_e)}")

            # If all attempts fail, raise an error
            if not hasattr(self, 'model'):
                raise ValueError(f"Failed to load model from {base_model_path} or any alternative paths")

        # Load metadata
        metadata_path = f"{os.path.splitext(base_model_path)[0]}_metadata.json"
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}

        # Load scaler if available
        scaler_path = f"{os.path.splitext(base_model_path)[0]}_scaler.pkl"
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
        else:
            self.scaler = StandardScaler()

        # Initialize history
        self.update_history = []

    def update(
        self,
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int = 32,
        epochs: int = 10,
        callbacks: Optional[List[tf.keras.callbacks.Callback]] = None,
        verbose: int = 0
    ) -> Dict[str, Any]:
        """
        Update model with new data

        Args:
            X: Input features
            y: Target values
            batch_size: Batch size
            epochs: Number of epochs
            callbacks: Callbacks (optional)
            verbose: Verbosity level

        Returns:
            Update metrics
        """
        # Scale data if scaler is available
        if hasattr(self.scaler, 'transform'):
            X = self.scaler.transform(X)

        # Update model
        history = self.model.fit(
            X, y,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks,
            verbose=verbose
        )

        # Get metrics
        metrics = {
            'loss': history.history['loss'][-1],
            'mae': history.history['mae'][-1] if 'mae' in history.history else None
        }

        # Add to update history
        self.update_history.append({
            'samples': len(X),
            'epochs': epochs,
            'metrics': metrics
        })

        return metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions

        Args:
            X: Input features

        Returns:
            Predictions
        """
        # Scale data if scaler is available
        if hasattr(self.scaler, 'transform'):
            X = self.scaler.transform(X)

        # Make predictions
        return self.model.predict(X)

    def save(self, path: str):
        """
        Save model

        Args:
            path: Path to save model
        """
        # Convert path to .keras format if it's .h5
        if path.endswith('.h5'):
            path = path.replace('.h5', '.keras')
            logger.info(f"Changed model save format from HDF5 (.h5) to native Keras format (.keras)")

        # Save model
        self.model.save(path)

        # Update metadata with update history
        self.metadata['update_history'] = self.update_history

        # Save metadata
        metadata_path = f"{os.path.splitext(path)[0]}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f)

        # Save scaler
        scaler_path = f"{os.path.splitext(path)[0]}_scaler.pkl"
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)

    def should_update(self, current_time: pd.Timestamp, last_update_time: Optional[pd.Timestamp] = None) -> bool:
        """
        Check if model should be updated

        Args:
            current_time: Current time
            last_update_time: Last update time (optional)

        Returns:
            True if model should be updated, False otherwise
        """
        if last_update_time is None:
            # No previous update, should update
            return True

        # Calculate days since last update
        days_since_update = (current_time - last_update_time).days

        # Check if update is due
        return days_since_update >= self.update_frequency
