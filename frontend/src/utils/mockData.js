// Mock data for the application when API calls fail or return empty data

// Mock data for cryptocurrency data
export const mockDataInfo = [
  {
    symbol: 'BTC',
    source: 'coinmarketcap',
    start_date: '2024-04-04',
    end_date: '2025-04-04',
    rows: 365,
    columns: ['timestamp', 'price', 'volume', 'market_cap']
  },
  {
    symbol: 'ETH',
    source: 'coinmarketcap',
    start_date: '2024-04-04',
    end_date: '2025-04-04',
    rows: 365,
    columns: ['timestamp', 'price', 'volume', 'market_cap']
  }
];

// Mock data for models
export const mockModelInfo = [
  {
    symbol: 'BTC',
    model_type: 'lstm',
    training_date: '2025-04-01T10:00:00Z',
    metrics: {
      mse: 0.0025,
      mae: 0.0345,
      r2: 0.87
    }
  },
  {
    symbol: 'ETH',
    model_type: 'xgboost',
    training_date: '2025-04-02T14:30:00Z',
    metrics: {
      mse: 0.0018,
      mae: 0.0289,
      r2: 0.91
    }
  }
];

// Mock data for backtesting strategies
export const mockStrategies = [
  {
    id: 'prediction_based',
    name: 'Prediction Based Strategy',
    description: 'Trading strategy based solely on model predictions'
  },
  {
    id: 'ma_prediction',
    name: 'Moving Average + Prediction Strategy',
    description: 'Trading strategy combining moving averages with model predictions'
  }
];

// Mock data for backtest results
export const mockBacktestResults = [
  {
    symbol: 'BTC',
    model_type: 'lstm',
    start_date: '2024-10-01',
    end_date: '2025-04-01',
    initial_capital: 10000,
    final_value: 13500,
    roi: 35,
    sharpe_ratio: 1.8,
    max_drawdown: 12,
    win_rate: 65,
    trades: 42
  },
  {
    symbol: 'ETH',
    model_type: 'xgboost',
    start_date: '2024-10-01',
    end_date: '2025-04-01',
    initial_capital: 10000,
    final_value: 11200,
    roi: 12,
    sharpe_ratio: 1.2,
    max_drawdown: 15,
    win_rate: 58,
    trades: 37
  }
];
