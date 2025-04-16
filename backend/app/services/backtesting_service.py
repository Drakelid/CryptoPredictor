import pandas as pd
import numpy as np
import os
import logging
import json
import pickle
from typing import List, Dict, Optional, Union, Any
from datetime import datetime, timedelta
import uuid
import backtrader as bt

from app.models.dl_models import get_dl_model
from app.models.ml_models import get_ml_model
from app.utils.feature_engineering import FeatureEngineer
from app.services.data_service import DataService
from app.services.training_service import TrainingService
from app.core.config import settings

logger = logging.getLogger(__name__)

class PredictionBasedStrategy(bt.Strategy):
    """Backtrader strategy based on model predictions"""

    params = (
        ('prediction_data', None),  # Prediction data to use
        ('position_size', 0.1),     # Position size (0-1)
    )

    def __init__(self):
        """Initialize strategy"""
        self.predictions = self.p.prediction_data
        self.position_size = self.p.position_size
        self.current_prediction = None
        self.last_prediction_date = None

    def next(self):
        """Called for each bar in the data"""
        # Get current date
        current_date = self.data.datetime.date()

        # Check if we have a prediction for this date
        if current_date in self.predictions:
            self.current_prediction = self.predictions[current_date]
            self.last_prediction_date = current_date

        # If we have a prediction, use it to make trading decisions
        if self.current_prediction is not None:
            # Simple strategy: if prediction is up, go long; if down, go short
            current_price = self.data.close[0]

            # Calculate predicted return
            predicted_return = (self.current_prediction / current_price) - 1

            # Determine position size based on confidence
            # Higher absolute return = more confidence
            size = self.position_size * self.broker.get_value()

            # If we have a position, close it
            if self.position:
                self.close()

            # Open new position based on prediction
            if predicted_return > 0.01:  # 1% threshold for going long
                self.buy(size=size / current_price)
            elif predicted_return < -0.01:  # -1% threshold for going short
                self.sell(size=size / current_price)


class MovingAveragePredictionStrategy(bt.Strategy):
    """Backtrader strategy combining moving averages with predictions"""

    params = (
        ('prediction_data', None),  # Prediction data to use
        ('position_size', 0.1),     # Position size (0-1)
        ('fast_period', 10),        # Fast moving average period
        ('slow_period', 30),        # Slow moving average period
    )

    def __init__(self):
        """Initialize strategy"""
        self.predictions = self.p.prediction_data
        self.position_size = self.p.position_size
        self.current_prediction = None
        self.last_prediction_date = None

        # Add moving average indicators
        self.fast_ma = bt.indicators.SMA(self.data.close, period=self.p.fast_period)
        self.slow_ma = bt.indicators.SMA(self.data.close, period=self.p.slow_period)

    def next(self):
        """Called for each bar in the data"""
        # Get current date
        current_date = self.data.datetime.date()

        # Check if we have a prediction for this date
        if current_date in self.predictions:
            self.current_prediction = self.predictions[current_date]
            self.last_prediction_date = current_date

        # If we have a prediction, use it to make trading decisions
        if self.current_prediction is not None:
            # Calculate predicted return
            current_price = self.data.close[0]
            predicted_return = (self.current_prediction / current_price) - 1

            # Determine position size
            size = self.position_size * self.broker.get_value()

            # If we have a position, close it
            if self.position:
                self.close()

            # Combine moving average signal with prediction
            ma_signal = self.fast_ma[0] > self.slow_ma[0]  # True if fast MA > slow MA

            # Open new position based on combined signals
            if ma_signal and predicted_return > 0:
                # Both signals are positive
                self.buy(size=size / current_price)
            elif not ma_signal and predicted_return < 0:
                # Both signals are negative
                self.sell(size=size / current_price)


class BacktestingService:
    """Service for backtesting trading strategies"""

    def __init__(
        self,
        data_service: DataService = None,
        training_service: TrainingService = None
    ):
        """
        Initialize backtesting service

        Args:
            data_service: DataService instance
            training_service: TrainingService instance
        """
        self.data_service = data_service or DataService()
        self.training_service = training_service or TrainingService(data_service)
        self.models_dir = settings.MODELS_DIR
        self.backtest_dir = os.path.join(self.models_dir, 'backtests')
        os.makedirs(self.backtest_dir, exist_ok=True)

    def run_backtest(
        self,
        symbol: str,
        model_type: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        initial_capital: float = 10000.0,
        position_size: float = 0.1
    ) -> Dict[str, Any]:
        """
        Run a backtest

        Args:
            symbol: Cryptocurrency symbol (e.g., BTC, ETH)
            model_type: Model type (lstm, gru, xgboost, lightgbm)
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            initial_capital: Initial capital
            position_size: Position size (0-1)

        Returns:
            Dictionary with backtest results
        """
        # Validate model type
        if model_type not in settings.SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model type: {model_type}. Supported types: {settings.SUPPORTED_MODELS}")

        # Get model info
        try:
            model_info = self.training_service.get_model_info(symbol=symbol, model_type=model_type)

            if not model_info:
                logger.error(f"No trained model found for {symbol} with type {model_type}")
                raise ValueError(f"No trained model found for {symbol} with type {model_type}. Please train a model first.")

            # Get the latest model
            latest_model = model_info[0]
            model_path = latest_model.model_path

            # Check if model file exists
            if not os.path.exists(model_path):
                logger.error(f"Model file not found: {model_path}")
                raise ValueError(f"Model file not found: {model_path}. Please train a model first.")

            logger.info(f"Using model: {model_path}")
        except Exception as e:
            logger.error(f"Error getting model info: {str(e)}")
            raise ValueError(f"Error getting model info: {str(e)}")

        # Load data
        try:
            df = self.data_service.load_data(symbol=symbol)

            if df is None or len(df) == 0:
                logger.error(f"No data found for {symbol}")
                raise ValueError(f"No data found for {symbol}. Please fetch data first.")

            # Filter data by date range if provided
            if start_date:
                start_date = pd.to_datetime(start_date)
                df = df[df['timestamp'] >= start_date]

            if end_date:
                end_date = pd.to_datetime(end_date)
                df = df[df['timestamp'] <= end_date]

            # Ensure we have enough data
            if len(df) < 100:
                logger.error(f"Not enough data for backtesting. Found {len(df)} data points, need at least 100.")
                raise ValueError(f"Not enough data for backtesting. Found {len(df)} data points, need at least 100.")

            logger.info(f"Loaded {len(df)} data points for {symbol} from {df['timestamp'].min()} to {df['timestamp'].max()}")
        except Exception as e:
            logger.error(f"Error loading data for {symbol}: {str(e)}")
            raise ValueError(f"Error loading data for {symbol}: {str(e)}")

        # Generate predictions for each day in the backtest period
        predictions = self._generate_backtest_predictions(
            df=df,
            symbol=symbol,
            model_type=model_type,
            model_path=model_path
        )

        # Create Backtrader cerebro
        cerebro = bt.Cerebro()

        # Add data
        data = self._create_backtrader_data(df)
        cerebro.adddata(data)

        # Add strategy
        cerebro.addstrategy(
            PredictionBasedStrategy,
            prediction_data=predictions,
            position_size=position_size
        )

        # Set initial capital
        cerebro.broker.setcash(initial_capital)

        # Set commission
        cerebro.broker.setcommission(commission=0.001)  # 0.1% commission

        # Add analyzers
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')

        # Run backtest
        results = cerebro.run()

        # Extract results
        strategy = results[0]

        # Calculate metrics
        final_value = cerebro.broker.getvalue()
        roi = (final_value / initial_capital - 1) * 100
        sharpe = strategy.analyzers.sharpe.get_analysis()['sharperatio']
        drawdown = strategy.analyzers.drawdown.get_analysis()
        returns = strategy.analyzers.returns.get_analysis()
        trades = strategy.analyzers.trades.get_analysis()

        # Create result dictionary
        result = {
            'symbol': symbol,
            'model_type': model_type,
            'start_date': df['timestamp'].min().strftime('%Y-%m-%d'),
            'end_date': df['timestamp'].max().strftime('%Y-%m-%d'),
            'initial_capital': initial_capital,
            'final_value': final_value,
            'roi': roi,
            'sharpe_ratio': sharpe,
            'max_drawdown': drawdown['max']['drawdown'],
            'max_drawdown_len': drawdown['max']['len'],
            'total_trades': trades.get('total', {}).get('total', 0),
            'win_trades': trades.get('won', {}).get('total', 0),
            'loss_trades': trades.get('lost', {}).get('total', 0),
            'win_rate': trades.get('won', {}).get('total', 0) / trades.get('total', {}).get('total', 1) * 100,
            'backtest_id': str(uuid.uuid4())
        }

        # Save backtest results
        self._save_backtest_results(result)

        return result

    def get_strategies(self) -> List[Dict[str, Any]]:
        """
        Get available backtesting strategies

        Returns:
            List of strategy information
        """
        strategies = [
            {
                'id': 'prediction_based',
                'name': 'Prediction Based Strategy',
                'description': 'Trading strategy based solely on model predictions'
            },
            {
                'id': 'ma_prediction',
                'name': 'Moving Average + Prediction Strategy',
                'description': 'Trading strategy combining moving averages with model predictions'
            }
        ]

        return strategies

    def get_backtest_results(
        self,
        symbol: Optional[str] = None,
        model_type: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get historical backtest results

        Args:
            symbol: Cryptocurrency symbol (e.g., BTC, ETH)
            model_type: Model type (lstm, gru, xgboost, lightgbm)
            limit: Number of results to return

        Returns:
            List of backtest results
        """
        results = []

        # List backtest files
        backtest_files = [f for f in os.listdir(self.backtest_dir) if f.endswith('.json')]

        for file in backtest_files:
            file_path = os.path.join(self.backtest_dir, file)

            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)

                # Filter by symbol
                if symbol and data.get('symbol', '').lower() != symbol.lower():
                    continue

                # Filter by model type
                if model_type and data.get('model_type', '').lower() != model_type.lower():
                    continue

                results.append(data)

            except Exception as e:
                logger.error(f"Error loading backtest results from {file_path}: {e}")

        # Sort by date (newest first)
        results.sort(key=lambda x: x.get('end_date', ''), reverse=True)

        # Limit results
        return results[:limit]

    def _generate_backtest_predictions(
        self,
        df: pd.DataFrame,
        symbol: str,
        model_type: str,
        model_path: str
    ) -> Dict[datetime.date, float]:
        """
        Generate predictions for backtesting

        Args:
            df: DataFrame with cryptocurrency data
            symbol: Cryptocurrency symbol (e.g., BTC, ETH)
            model_type: Model type (lstm, gru, xgboost, lightgbm)
            model_path: Path to model file

        Returns:
            Dictionary mapping dates to predicted prices
        """
        # Engineer features
        feature_engineer = FeatureEngineer()
        df_features = feature_engineer.engineer_features(df, symbol=symbol)

        # Determine target column
        target_col = 'close' if 'close' in df_features.columns else 'price'

        # Load scaler
        scaler_path = f"{os.path.splitext(model_path)[0]}_scaler.pkl"
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)

        # Get model parameters from metadata
        metadata_path = f"{os.path.splitext(model_path)[0]}_metadata.json"
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        lookback = metadata.get('lookback', 30)
        horizon = metadata.get('horizon', 7)

        # Initialize predictions dictionary
        predictions = {}

        # For backtesting, we need to simulate making predictions at each point in time
        # We'll use an expanding window approach
        min_samples = lookback + horizon

        if len(df_features) <= min_samples:
            raise ValueError(f"Not enough data for backtesting. Need at least {min_samples} samples.")

        # For each day in the backtest period (after initial lookback)
        for i in range(lookback, len(df_features) - horizon):
            # Get data up to this point
            df_window = df_features.iloc[:i+1].copy()

            # Make prediction based on model type
            if model_type in ['lstm', 'gru']:
                # Prepare input sequence
                X, _, _, _, _ = feature_engineer.prepare_sequences(
                    df=df_window,
                    target_col=target_col,
                    lookback=lookback,
                    horizon=horizon,
                    train_ratio=1.0  # Use all data for prediction
                )

                # Get the latest sequence
                X_pred = X[-1:].copy()

                # Load model
                input_shape = (lookback, X_pred.shape[2])
                model = get_dl_model(
                    model_type=model_type,
                    input_shape=input_shape,
                    output_shape=horizon,
                    model_path=model_path
                )

                # Make prediction
                y_pred = model.predict(X_pred)

                # Inverse transform prediction
                y_pred_inv = scaler.inverse_transform(y_pred[0].reshape(-1, 1)).flatten()

            else:  # ML models (xgboost, lightgbm)
                # Prepare input features
                X, _, _, _, _ = feature_engineer.prepare_tabular_data(
                    df=df_window,
                    target_col=target_col,
                    lookback=lookback,
                    horizon=horizon,
                    train_ratio=1.0  # Use all data for prediction
                )

                # Get the latest features
                X_pred = X[-1:].copy()

                # Load model
                model = get_ml_model(
                    model_type=model_type,
                    model_path=model_path
                )

                # Make prediction
                y_pred = model.predict(X_pred)

                # Inverse transform prediction
                y_pred_inv = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()

                # If we only have one prediction but need horizon days, repeat the value
                if len(y_pred_inv) == 1 and horizon > 1:
                    y_pred_inv = np.repeat(y_pred_inv, horizon)

            # Add predictions to dictionary
            for j in range(horizon):
                pred_date = df_features.iloc[i + j + 1]['timestamp'].date()
                predictions[pred_date] = y_pred_inv[j]

        return predictions

    def _create_backtrader_data(self, df: pd.DataFrame) -> bt.feeds.PandasData:
        """
        Create Backtrader data feed from DataFrame

        Args:
            df: DataFrame with cryptocurrency data

        Returns:
            Backtrader data feed
        """
        # Ensure we have OHLCV data
        if 'open' not in df.columns:
            # If we only have price data, use it for all OHLC values
            if 'price' in df.columns:
                df['open'] = df['price']
                df['high'] = df['price']
                df['low'] = df['price']
                df['close'] = df['price']
            else:
                # If we have close but not other OHLC, use close for all
                if 'close' in df.columns:
                    df['open'] = df['close']
                    df['high'] = df['close']
                    df['low'] = df['close']

        # Ensure we have volume data
        if 'volume' not in df.columns:
            df['volume'] = 0

        # Set timestamp as index
        df = df.set_index('timestamp')

        # Create Backtrader data feed
        data = bt.feeds.PandasData(
            dataname=df,
            open='open',
            high='high',
            low='low',
            close='close',
            volume='volume',
            openinterest=None
        )

        return data

    def _save_backtest_results(self, results: Dict[str, Any]):
        """
        Save backtest results to disk

        Args:
            results: Dictionary with backtest results
        """
        # Create filename
        filename = f"{results['symbol'].lower()}_{results['model_type']}_{results['backtest_id']}.json"
        file_path = os.path.join(self.backtest_dir, filename)

        # Save to file
        with open(file_path, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"Saved backtest results to {file_path}")
