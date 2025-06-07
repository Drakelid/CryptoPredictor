import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, LSTM, GRU, Dropout, Input, Bidirectional, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Import advanced factory for additional architectures
from app.models.advanced_models import AdvancedModelFactory
import numpy as np
import os
import logging
import traceback
from typing import Tuple, List, Dict, Optional, Union, Any
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from app.core.config import settings

logger = logging.getLogger(__name__)

class DeepLearningModel:
    """Base class for deep learning models"""

    def __init__(
        self,
        model_type: str = "lstm",
        input_shape: Tuple[int, int] = (30, 10),
        output_shape: int = 7,
        learning_rate: float = 0.001,
        model_path: Optional[str] = None
    ):
        """
        Initialize deep learning model

        Args:
            model_type: Type of model ('lstm' or 'gru')
            input_shape: Shape of input data (lookback, features)
            output_shape: Number of time steps to predict
            learning_rate: Learning rate for optimizer
            model_path: Path to saved model (if loading existing model)
        """
        self.model_type = model_type.lower()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.learning_rate = learning_rate
        self.model_path = model_path
        self.model = None

        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            self._build_model()

    def _build_model(self):
        """Build the deep learning model"""
        if self.model_type == "lstm":
            self._build_lstm_model()
        elif self.model_type == "gru":
            self._build_gru_model()
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def _build_lstm_model(self):
        """Build LSTM model"""
        model = Sequential()

        # First LSTM layer with return sequences
        model.add(LSTM(
            units=64,
            return_sequences=True,
            input_shape=self.input_shape
        ))
        model.add(Dropout(0.2))

        # Second LSTM layer
        model.add(LSTM(units=64, return_sequences=False))
        model.add(Dropout(0.2))

        # Dense layers
        model.add(Dense(units=32, activation='relu'))
        model.add(Dense(units=self.output_shape))

        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='mean_squared_error'
        )

        self.model = model
        logger.info(f"Built LSTM model with input shape {self.input_shape} and output shape {self.output_shape}")

    def _build_gru_model(self):
        """Build GRU model"""
        model = Sequential()

        # First GRU layer with return sequences
        model.add(GRU(
            units=64,
            return_sequences=True,
            input_shape=self.input_shape
        ))
        model.add(Dropout(0.2))

        # Second GRU layer
        model.add(GRU(units=64, return_sequences=False))
        model.add(Dropout(0.2))

        # Dense layers
        model.add(Dense(units=32, activation='relu'))
        model.add(Dense(units=self.output_shape))

        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='mean_squared_error'
        )

        self.model = model
        logger.info(f"Built GRU model with input shape {self.input_shape} and output shape {self.output_shape}")

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        patience: int = 10,
        save_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Train the model

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            epochs: Number of epochs to train
            batch_size: Batch size for training
            patience: Patience for early stopping
            save_path: Path to save the model

        Returns:
            Dictionary with training history
        """
        if self.model is None:
            self._build_model()

        # Define callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True
            )
        ]

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            callbacks.append(
                ModelCheckpoint(
                    filepath=save_path,
                    monitor='val_loss',
                    save_best_only=True
                )
            )

        # Check and reshape y_train and y_val if needed
        logger.info(f"Before training - X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        logger.info(f"Before training - X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")

        # Ensure y_train and y_val have the correct shape (2D)
        if y_train.ndim == 3:
            logger.warning(f"Reshaping y_train from {y_train.shape} to 2D")
            if y_train.shape[2] == 1:
                # If shape is (samples, horizon, 1), reshape to (samples, horizon)
                y_train = y_train.reshape(y_train.shape[0], y_train.shape[1])
                y_val = y_val.reshape(y_val.shape[0], y_val.shape[1])
            else:
                # If shape is (samples, horizon, features), take the first feature
                y_train = y_train[:, :, 0]
                y_val = y_val[:, :, 0]
            logger.info(f"After reshaping - y_train shape: {y_train.shape}, y_val shape: {y_val.shape}")

        # Train model
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )

        # Calculate additional metrics that aren't included in history
        train_pred = self.model.predict(X_train)
        val_pred = self.model.predict(X_val)

        # Ensure predictions have the right shape
        if train_pred.ndim == 3 and train_pred.shape[2] == 1:
            train_pred = train_pred.reshape(train_pred.shape[0], train_pred.shape[1])
        if val_pred.ndim == 3 and val_pred.shape[2] == 1:
            val_pred = val_pred.reshape(val_pred.shape[0], val_pred.shape[1])

        # Calculate metrics
        try:
            train_mse = mean_squared_error(y_train, train_pred)
            train_mae = mean_absolute_error(y_train, train_pred)
            train_r2 = r2_score(y_train, train_pred)

            val_mse = mean_squared_error(y_val, val_pred)
            val_mae = mean_absolute_error(y_val, val_pred)
            val_r2 = r2_score(y_val, val_pred)

            # Add metrics to history
            history.history['mse'] = [train_mse]
            history.history['mae'] = [train_mae]
            history.history['r2'] = [train_r2]
            history.history['val_mse'] = [val_mse]
            history.history['val_mae'] = [val_mae]
            history.history['val_r2'] = [val_r2]

            logger.info(f"Training metrics - MSE: {train_mse:.4f}, MAE: {train_mae:.4f}, R²: {train_r2:.2f}")
            logger.info(f"Validation metrics - MSE: {val_mse:.4f}, MAE: {val_mae:.4f}, R²: {val_r2:.2f}")
        except Exception as e:
            logger.error(f"Error calculating additional metrics: {e}")
            logger.error(f"Shapes - y_train: {y_train.shape}, train_pred: {train_pred.shape}, y_val: {y_val.shape}, val_pred: {val_pred.shape}")

        # Save model path
        if save_path:
            self.model_path = save_path

        return history.history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions

        Args:
            X: Input features

        Returns:
            Predicted values
        """
        if self.model is None:
            raise ValueError("Model not initialized. Call train() or load_model() first.")

        return self.model.predict(X)

    def save_model(self, path: str):
        """
        Save model to disk

        Args:
            path: Path to save the model
        """
        if self.model is None:
            raise ValueError("Model not initialized. Call train() first.")

        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Check if the path ends with .h5 and change it to .keras
        if path.endswith('.h5'):
            new_path = path.replace('.h5', '.keras')
            logger.info(f"Changing model save format from HDF5 (.h5) to native Keras format (.keras)")
            self.model.save(new_path)
            self.model_path = new_path
            logger.info(f"Saved model to {new_path}")
        else:
            # Save with native Keras format
            self.model.save(path)
            self.model_path = path
            logger.info(f"Saved model to {path}")

    def load_model(self, path: str):
        """
        Load model from disk

        Args:
            path: Path to load the model from
        """
        # Check if the original path exists
        if os.path.exists(path):
            # Load model with reduced retracing to avoid TensorFlow warnings
            self.model = load_model(path, compile=False)  # Load without compiling to reduce retracing

            # Compile the model with reduced retracing
            self.model.compile(optimizer='adam', loss='mse', run_eagerly=False)

            # Get the model's input shape
            input_shape = self.model.input_shape
            if input_shape and len(input_shape) > 1:
                # Extract the expected dimensions
                batch_size = input_shape[0]  # Usually None
                timesteps = input_shape[1] if len(input_shape) > 2 else 30  # Default to 30 if not specified
                features = input_shape[2] if len(input_shape) > 2 else 50  # Default to 50 if not specified

                logger.info(f"Model expects input shape: {input_shape}")
                logger.info(f"Using timesteps={timesteps}, features={features} for dummy prediction")

                # Make a dummy prediction to initialize the model's predict function
                # This helps avoid retracing warnings during actual predictions
                dummy_input = np.zeros((1, timesteps, features))  # Use model's expected shape
                self.model.predict(dummy_input, verbose=0)
            else:
                logger.warning(f"Could not determine model input shape: {input_shape}")
                # Use a default shape as fallback
                dummy_input = np.zeros((1, 30, 50))  # Default shape (batch_size, timesteps, features)
                try:
                    self.model.predict(dummy_input, verbose=0)
                except Exception as e:
                    logger.warning(f"Failed to make dummy prediction with shape (1, 30, 50): {e}")

            self.model_path = path
            logger.info(f"Loaded model from {path} with reduced retracing")
            return

        # If not, try with .keras extension if the path ends with .h5
        if path.endswith('.h5'):
            keras_path = path.replace('.h5', '.keras')
            if os.path.exists(keras_path):
                # Load model with reduced retracing to avoid TensorFlow warnings
                self.model = load_model(keras_path, compile=False)  # Load without compiling to reduce retracing

                # Compile the model with reduced retracing
                self.model.compile(optimizer='adam', loss='mse', run_eagerly=False)

                # Get the model's input shape
                input_shape = self.model.input_shape
                if input_shape and len(input_shape) > 1:
                    # Extract the expected dimensions
                    batch_size = input_shape[0]  # Usually None
                    timesteps = input_shape[1] if len(input_shape) > 2 else 30  # Default to 30 if not specified
                    features = input_shape[2] if len(input_shape) > 2 else 50  # Default to 50 if not specified

                    logger.info(f"Model expects input shape: {input_shape}")
                    logger.info(f"Using timesteps={timesteps}, features={features} for dummy prediction")

                    # Make a dummy prediction to initialize the model's predict function
                    # This helps avoid retracing warnings during actual predictions
                    dummy_input = np.zeros((1, timesteps, features))  # Use model's expected shape
                    self.model.predict(dummy_input, verbose=0)
                else:
                    logger.warning(f"Could not determine model input shape: {input_shape}")
                    # Use a default shape as fallback
                    dummy_input = np.zeros((1, 30, 50))  # Default shape (batch_size, timesteps, features)
                    try:
                        self.model.predict(dummy_input, verbose=0)
                    except Exception as e:
                        logger.warning(f"Failed to make dummy prediction with shape (1, 30, 50): {e}")

                self.model_path = keras_path
                logger.info(f"Loaded model from {keras_path} (converted from .h5) with reduced retracing")
                return

        # If we get here, neither path exists
        raise FileNotFoundError(f"Model file not found: {path} or {path.replace('.h5', '.keras') if path.endswith('.h5') else path}")


class LSTMModel(DeepLearningModel):
    """LSTM model implementation"""

    def __init__(
        self,
        input_shape: Tuple[int, int] = (30, 10),
        output_shape: int = 7,
        params: Optional[Dict[str, Any]] = None,
        model_path: Optional[str] = None
    ):
        """Initialize LSTM model"""
        # Set default parameters
        if params is None:
            params = {}

        learning_rate = params.get('learning_rate', 0.001)

        super().__init__(
            model_type="lstm",
            input_shape=input_shape,
            output_shape=output_shape,
            learning_rate=learning_rate,
            model_path=model_path
        )


class GRUModel(DeepLearningModel):
    """GRU model implementation"""

    def __init__(
        self,
        input_shape: Tuple[int, int] = (30, 10),
        output_shape: int = 7,
        params: Optional[Dict[str, Any]] = None,
        model_path: Optional[str] = None
    ):
        """Initialize GRU model"""
        # Set default parameters
        if params is None:
            params = {}

        learning_rate = params.get('learning_rate', 0.001)

        super().__init__(
            model_type="gru",
            input_shape=input_shape,
            output_shape=output_shape,
            learning_rate=learning_rate,
            model_path=model_path
        )


class BidirectionalLSTM(DeepLearningModel):
    """Bidirectional LSTM model for improved sequence learning"""

    def _build_lstm_model(self):
        """Build Bidirectional LSTM model"""
        model = Sequential()

        # First Bidirectional LSTM layer
        model.add(Bidirectional(
            LSTM(
                units=64,
                return_sequences=True
            ),
            input_shape=self.input_shape
        ))
        model.add(Dropout(0.2))

        # Second Bidirectional LSTM layer
        model.add(Bidirectional(
            LSTM(units=64, return_sequences=False)
        ))
        model.add(Dropout(0.2))

        # Dense layers
        model.add(Dense(units=32, activation='relu'))
        model.add(Dense(units=self.output_shape))

        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='mean_squared_error'
        )

        self.model = model
        logger.info(f"Built Bidirectional LSTM model with input shape {self.input_shape} and output shape {self.output_shape}")


class AdvancedKerasModel(DeepLearningModel):
    """Wrapper for advanced architectures provided by ``AdvancedModelFactory``."""

    def __init__(
        self,
        model_type: str,
        input_shape: Tuple[int, int],
        output_shape: int,
        params: Optional[Dict[str, Any]] = None,
        model_path: Optional[str] = None,
    ) -> None:
        self._advanced_type = model_type
        self._params = params or {}
        super().__init__(
            model_type=model_type,
            input_shape=input_shape,
            output_shape=output_shape,
            learning_rate=self._params.get("learning_rate", 0.001),
            model_path=model_path,
        )

    def _build_model(self) -> None:  # type: ignore[override]
        self.model = AdvancedModelFactory.create_model(
            self._advanced_type, self.input_shape, self.output_shape, self._params
        )


def get_dl_model(
    model_type: str = "lstm",
    input_shape: Tuple[int, int] = (30, 10),
    output_shape: int = 7,
    learning_rate: float = 0.001,
    model_path: Optional[str] = None,
    bidirectional: bool = False
) -> DeepLearningModel:
    """
    Factory function to get the appropriate deep learning model

    Args:
        model_type: Type of model ('lstm' or 'gru')
        input_shape: Shape of input data (lookback, features)
        output_shape: Number of time steps to predict
        learning_rate: Learning rate for optimizer
        model_path: Path to saved model (if loading existing model)
        bidirectional: Whether to use bidirectional layers

    Returns:
        Initialized model
    """
    model_type_l = model_type.lower()

    if model_type_l == "bidirectional_lstm":
        bidirectional = True
        model_type_l = "lstm"

    if model_type_l in ["attention_lstm", "cnn_lstm", "transformer", "dual_attention"]:
        return AdvancedKerasModel(
            model_type=model_type_l,
            input_shape=input_shape,
            output_shape=output_shape,
            model_path=model_path,
            params={"learning_rate": learning_rate},
        )

    if bidirectional and model_type_l == "lstm":
        return BidirectionalLSTM(
            model_type=model_type_l,
            input_shape=input_shape,
            output_shape=output_shape,
            learning_rate=learning_rate,
            model_path=model_path
        )
    else:
        return DeepLearningModel(
            model_type=model_type_l,
            input_shape=input_shape,
            output_shape=output_shape,
            learning_rate=learning_rate,
            model_path=model_path
        )
