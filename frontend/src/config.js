// Configuration for the application

// API configuration
// Allow overriding the backend URL via environment variables so Docker
// containers can point to the backend service when deployed.

// Get API base URL from environment or fall back to localhost
function getApiBaseUrl() {
  // Get API URL from multiple possible sources with fallbacks
  // 1. Runtime environment variables (window.__ENV) - set by docker-entrypoint.sh
  // 2. Build-time environment variables (import.meta.env) - set during build
  // 3. Node environment variables - for SSR contexts
  // 4. Default localhost for development
  
  // Check for Node environment (SSR context)
  const nodeEnvUrl = typeof process !== 'undefined' && process.env
    ? process.env.VITE_API_BASE_URL
    : undefined;
  
  // Check for runtime injected environment variables
  const runtimeEnvUrl = window.__ENV?.VITE_API_BASE_URL;
  
  // Check for build-time environment variables
  const buildTimeEnvUrl = import.meta.env?.VITE_API_BASE_URL;
  
  // For debugging - log out to console which helps diagnose issues
  console.log('Window ENV:', window.__ENV);
  console.log('Import meta env:', import.meta.env);
  console.log('Node env:', nodeEnvUrl);
  
  // IMPORTANT: In browser context, we ALWAYS use localhost:8000 directly
  // This ensures browser compatibility regardless of Docker network setup
  // We're simplifying to ensure it works, then can refine later
  return runtimeEnvUrl || buildTimeEnvUrl || nodeEnvUrl || 'http://localhost:8000';
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
