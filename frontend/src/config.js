// Configuration for the application

// API configuration
export const API_BASE_URL = 'http://localhost:8000';

// Default settings
export const DEFAULT_SETTINGS = {
  // Data fetching defaults
  data: {
    defaultCrypto: 'BTC',
    defaultSource: 'coinmarketcap',
    defaultDays: 365
  },
  
  // Model training defaults
  model: {
    defaultType: 'lstm',
    defaultEpochs: 50,
    defaultBatchSize: 32,
    defaultLookback: 30
  },
  
  // Backtesting defaults
  backtest: {
    defaultStrategy: 'prediction_based',
    defaultInitialCapital: 10000,
    defaultPositionSize: 0.1
  }
};

// Available cryptocurrencies
export const AVAILABLE_CRYPTOCURRENCIES = [
  { value: 'BTC', label: 'Bitcoin (BTC)' },
  { value: 'ETH', label: 'Ethereum (ETH)' },
  { value: 'BNB', label: 'Binance Coin (BNB)' },
  { value: 'XRP', label: 'Ripple (XRP)' },
  { value: 'ADA', label: 'Cardano (ADA)' },
  { value: 'SOL', label: 'Solana (SOL)' },
  { value: 'DOGE', label: 'Dogecoin (DOGE)' },
  { value: 'DOT', label: 'Polkadot (DOT)' }
];

// Available data sources
export const AVAILABLE_DATA_SOURCES = [
  { value: 'coinmarketcap', label: 'CoinMarketCap' },
  { value: 'binance', label: 'Binance' }
];

// Available model types
export const AVAILABLE_MODEL_TYPES = [
  { value: 'lstm', label: 'LSTM', color: 'blue' },
  { value: 'gru', label: 'GRU', color: 'purple' },
  { value: 'xgboost', label: 'XGBoost', color: 'green' },
  { value: 'lightgbm', label: 'LightGBM', color: 'orange' }
];
