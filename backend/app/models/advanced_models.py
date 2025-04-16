import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Dense, LSTM, GRU, Dropout, BatchNormalization, 
    Input, Concatenate, Bidirectional, Conv1D, 
    MaxPooling1D, Attention, MultiHeadAttention, 
    LayerNormalization, GlobalAveragePooling1D
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import logging
from typing import Tuple, Dict, Any, List, Optional

logger = logging.getLogger(__name__)

class AdvancedModelFactory:
    """Factory for creating advanced deep learning models"""
    
    @staticmethod
    def create_model(
        model_type: str,
        input_shape: Tuple[int, int],
        output_shape: int,
        hyperparams: Optional[Dict[str, Any]] = None
    ) -> Model:
        """
        Create an advanced model
        
        Args:
            model_type: Type of model to create
            input_shape: Shape of input data (sequence_length, features)
            output_shape: Number of output units
            hyperparams: Optional hyperparameters
        
        Returns:
            Keras model
        """
        if hyperparams is None:
            hyperparams = {}
        
        if model_type == "attention_lstm":
            return AdvancedModelFactory._create_attention_lstm(input_shape, output_shape, hyperparams)
        elif model_type == "bidirectional_lstm":
            return AdvancedModelFactory._create_bidirectional_lstm(input_shape, output_shape, hyperparams)
        elif model_type == "cnn_lstm":
            return AdvancedModelFactory._create_cnn_lstm(input_shape, output_shape, hyperparams)
        elif model_type == "transformer":
            return AdvancedModelFactory._create_transformer(input_shape, output_shape, hyperparams)
        elif model_type == "dual_attention":
            return AdvancedModelFactory._create_dual_attention_model(input_shape, output_shape, hyperparams)
        else:
            raise ValueError(f"Unknown advanced model type: {model_type}")
    
    @staticmethod
    def _create_attention_lstm(
        input_shape: Tuple[int, int],
        output_shape: int,
        hyperparams: Dict[str, Any]
    ) -> Model:
        """Create an LSTM model with attention mechanism"""
        # Get hyperparameters with defaults
        units = hyperparams.get("units", 128)
        dropout_rate = hyperparams.get("dropout_rate", 0.2)
        recurrent_dropout = hyperparams.get("recurrent_dropout", 0.2)
        learning_rate = hyperparams.get("learning_rate", 0.001)
        
        # Define model
        inputs = Input(shape=input_shape)
        
        # LSTM layer with return sequences for attention
        lstm = LSTM(units, return_sequences=True, dropout=dropout_rate, 
                   recurrent_dropout=recurrent_dropout)(inputs)
        
        # Self-attention mechanism
        attention = MultiHeadAttention(
            key_dim=units // 4,  # Reduced dimension for efficiency
            num_heads=4,
            dropout=dropout_rate
        )(lstm, lstm)
        
        # Add & normalize (residual connection)
        attention = LayerNormalization()(attention + lstm)
        
        # Global pooling to reduce sequence dimension
        pooled = GlobalAveragePooling1D()(attention)
        
        # Output layers
        x = Dense(units // 2, activation="relu")(pooled)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)
        outputs = Dense(output_shape, activation="linear")(x)
        
        # Create and compile model
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss="mse",
            metrics=["mae"]
        )
        
        return model
    
    @staticmethod
    def _create_bidirectional_lstm(
        input_shape: Tuple[int, int],
        output_shape: int,
        hyperparams: Dict[str, Any]
    ) -> Model:
        """Create a bidirectional LSTM model"""
        # Get hyperparameters with defaults
        units = hyperparams.get("units", 128)
        dropout_rate = hyperparams.get("dropout_rate", 0.2)
        learning_rate = hyperparams.get("learning_rate", 0.001)
        
        # Define model
        model = Sequential()
        model.add(Input(shape=input_shape))
        
        # Bidirectional LSTM layers
        model.add(Bidirectional(LSTM(units, return_sequences=True, dropout=dropout_rate)))
        model.add(Bidirectional(LSTM(units // 2, dropout=dropout_rate)))
        
        # Output layers
        model.add(Dense(units // 2, activation="relu"))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        model.add(Dense(output_shape, activation="linear"))
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss="mse",
            metrics=["mae"]
        )
        
        return model
    
    @staticmethod
    def _create_cnn_lstm(
        input_shape: Tuple[int, int],
        output_shape: int,
        hyperparams: Dict[str, Any]
    ) -> Model:
        """Create a CNN-LSTM hybrid model"""
        # Get hyperparameters with defaults
        lstm_units = hyperparams.get("lstm_units", 128)
        cnn_filters = hyperparams.get("cnn_filters", 64)
        kernel_size = hyperparams.get("kernel_size", 3)
        dropout_rate = hyperparams.get("dropout_rate", 0.2)
        learning_rate = hyperparams.get("learning_rate", 0.001)
        
        # Define model
        inputs = Input(shape=input_shape)
        
        # CNN layers for feature extraction
        x = Conv1D(filters=cnn_filters, kernel_size=kernel_size, padding="same", activation="relu")(inputs)
        x = BatchNormalization()(x)
        x = MaxPooling1D(pool_size=2)(x)
        
        x = Conv1D(filters=cnn_filters*2, kernel_size=kernel_size, padding="same", activation="relu")(x)
        x = BatchNormalization()(x)
        x = MaxPooling1D(pool_size=2)(x)
        
        # LSTM layer for sequence modeling
        x = LSTM(lstm_units, dropout=dropout_rate)(x)
        
        # Output layers
        x = Dense(lstm_units // 2, activation="relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)
        outputs = Dense(output_shape, activation="linear")(x)
        
        # Create and compile model
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss="mse",
            metrics=["mae"]
        )
        
        return model
    
    @staticmethod
    def _create_transformer(
        input_shape: Tuple[int, int],
        output_shape: int,
        hyperparams: Dict[str, Any]
    ) -> Model:
        """Create a Transformer-based model"""
        # Get hyperparameters with defaults
        embed_dim = hyperparams.get("embed_dim", 128)
        num_heads = hyperparams.get("num_heads", 4)
        ff_dim = hyperparams.get("ff_dim", 256)
        dropout_rate = hyperparams.get("dropout_rate", 0.2)
        learning_rate = hyperparams.get("learning_rate", 0.001)
        num_transformer_blocks = hyperparams.get("num_transformer_blocks", 2)
        
        # Define model
        inputs = Input(shape=input_shape)
        
        # Initial projection to embed_dim
        x = Dense(embed_dim)(inputs)
        
        # Transformer blocks
        for _ in range(num_transformer_blocks):
            # Multi-head attention
            attention_output = MultiHeadAttention(
                key_dim=embed_dim // num_heads,
                num_heads=num_heads,
                dropout=dropout_rate
            )(x, x)
            
            # Add & normalize (residual connection)
            x = LayerNormalization(epsilon=1e-6)(attention_output + x)
            
            # Feed-forward network
            ffn_output = Sequential([
                Dense(ff_dim, activation="relu"),
                Dropout(dropout_rate),
                Dense(embed_dim)
            ])(x)
            
            # Add & normalize (residual connection)
            x = LayerNormalization(epsilon=1e-6)(ffn_output + x)
        
        # Global pooling
        x = GlobalAveragePooling1D()(x)
        
        # Output layers
        x = Dense(ff_dim // 2, activation="relu")(x)
        x = Dropout(dropout_rate)(x)
        outputs = Dense(output_shape, activation="linear")(x)
        
        # Create and compile model
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss="mse",
            metrics=["mae"]
        )
        
        return model
    
    @staticmethod
    def _create_dual_attention_model(
        input_shape: Tuple[int, int],
        output_shape: int,
        hyperparams: Dict[str, Any]
    ) -> Model:
        """Create a dual-path model with separate attention for technical and sentiment features"""
        # Get hyperparameters with defaults
        units = hyperparams.get("units", 128)
        dropout_rate = hyperparams.get("dropout_rate", 0.2)
        learning_rate = hyperparams.get("learning_rate", 0.001)
        
        # Assume the last 30% of features are sentiment-related
        tech_features = int(input_shape[1] * 0.7)
        sentiment_features = input_shape[1] - tech_features
        
        # Define model
        inputs = Input(shape=input_shape)
        
        # Split input into technical and sentiment features
        technical_input = tf.keras.layers.Lambda(
            lambda x: x[:, :, :tech_features]
        )(inputs)
        
        sentiment_input = tf.keras.layers.Lambda(
            lambda x: x[:, :, tech_features:]
        )(inputs)
        
        # Technical features path
        tech_lstm = LSTM(units, return_sequences=True, dropout=dropout_rate)(technical_input)
        tech_attention = MultiHeadAttention(
            key_dim=units // 4,
            num_heads=4,
            dropout=dropout_rate
        )(tech_lstm, tech_lstm)
        tech_norm = LayerNormalization()(tech_attention + tech_lstm)
        tech_pooled = GlobalAveragePooling1D()(tech_norm)
        
        # Sentiment features path
        sent_lstm = LSTM(units // 2, return_sequences=True, dropout=dropout_rate)(sentiment_input)
        sent_attention = MultiHeadAttention(
            key_dim=units // 8,
            num_heads=2,
            dropout=dropout_rate
        )(sent_lstm, sent_lstm)
        sent_norm = LayerNormalization()(sent_attention + sent_lstm)
        sent_pooled = GlobalAveragePooling1D()(sent_norm)
        
        # Combine paths
        combined = Concatenate()([tech_pooled, sent_pooled])
        
        # Output layers
        x = Dense(units, activation="relu")(combined)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)
        outputs = Dense(output_shape, activation="linear")(x)
        
        # Create and compile model
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss="mse",
            metrics=["mae"]
        )
        
        return model

    @staticmethod
    def get_callbacks(patience: int = 10) -> List[tf.keras.callbacks.Callback]:
        """Get standard callbacks for training"""
        return [
            EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=patience // 2,
                min_lr=1e-6
            )
        ]
