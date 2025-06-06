// Configuration for the application

// API configuration
// Allow overriding the backend URL via environment variables so Docker
// containers can point to the backend service when deployed.
// `process.env.VITE_API_BASE_URL` is used when building in Node (e.g. in Docker)
// and `import.meta.env.VITE_API_BASE_URL` works when running via Vite dev server
// or a regular production build. Both fall back to localhost for development.
  (typeof process !== 'undefined' && process.env.VITE_API_BASE_URL) ||
  import.meta.env.VITE_API_BASE_URL ||
  'http://localhost:8000';
    defaultSource: 'coinmarketcap',
  { value: 'coinmarketcap', label: 'CoinMarketCap' },
  // For debugging - log out to console which helps diagnose issues
  console.log('Window ENV:', window.__ENV);
  console.log('Import meta env:', import.meta.env);
  
  // IMPORTANT: In browser context, we ALWAYS use localhost:8000 directly
  // This ensures browser compatibility regardless of Docker network setup
  // We're simplifying to ensure it works, then can refine later
  return 'http://localhost:8000';
}

// Log the API URL on startup for debugging
const apiBaseUrl = getApiBaseUrl();
console.log('Using API base URL:', apiBaseUrl);

export const API_BASE_URL = apiBaseUrl;

// Default settings
export const DEFAULT_SETTINGS = {
  // Data fetching defaults
  data: {
    defaultCrypto: 'BTC',
    defaultSource: 'coingecko',
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
  { value: 'coingecko', label: 'CoinGecko' },
  { value: 'binance', label: 'Binance' }
];

// Available model types
export const AVAILABLE_MODEL_TYPES = [
  { value: 'lstm', label: 'LSTM', color: 'blue' },
  { value: 'gru', label: 'GRU', color: 'purple' },
  { value: 'xgboost', label: 'XGBoost', color: 'green' },
  { value: 'lightgbm', label: 'LightGBM', color: 'orange' }
];
