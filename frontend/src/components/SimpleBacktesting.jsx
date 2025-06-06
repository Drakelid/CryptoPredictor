import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';
import { API_BASE_URL } from '../config';

// Fallback history used if the API call fails
const mockBacktestHistory = [
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
    trades: 42,
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
    trades: 37,
  },
  {
    symbol: 'SOL',
    model_type: 'gru',
    start_date: '2024-10-01',
    end_date: '2025-04-01',
    initial_capital: 10000,
    final_value: 14200,
    roi: 42,
    sharpe_ratio: 1.9,
    max_drawdown: 18,
    win_rate: 62,
    trades: 45,
  },
];

const SimpleBacktesting = () => {
  const [activeTab, setActiveTab] = useState('run');
  const navigate = useNavigate();
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [backtestResults, setBacktestResults] = useState([]);
  const [currentResult, setCurrentResult] = useState(null);
  const [availableCryptos, setAvailableCryptos] = useState(['BTC', 'ETH', 'BNB', 'XRP', 'ADA']);
  const [availableModels, setAvailableModels] = useState(['lstm', 'gru', 'xgboost', 'lightgbm']);

  // Fetch backtest history on component mount
  useEffect(() => {
    fetchBacktestHistory();
  }, []);

  // Fetch backtest history
  const fetchBacktestHistory = async () => {
    try {
      setIsLoading(true);
      const response = await axios.get(`${API_BASE_URL}/api/backtesting/history`);
      setBacktestResults(response.data);
      setIsLoading(false);
    } catch (error) {
      console.error('Error fetching backtest history:', error);
      setError('Failed to fetch backtest history. Please try again.');
      setIsLoading(false);
      // Populate table with fallback data if API is unreachable
      setBacktestResults(mockBacktestHistory);
    }
  };

  // Run backtest
  const runBacktest = async (event) => {
    console.log('runBacktest called');
    event.preventDefault();

    // Get form values
    const symbol = document.getElementById('crypto-symbol')?.value || 'BTC';
    const modelType = document.getElementById('model-type')?.value || 'lstm';
    const startDate = document.getElementById('start-date')?.value || '2023-01-01';
    const endDate = document.getElementById('end-date')?.value || '2023-12-31';
    const initialCapital = parseFloat(document.getElementById('initial-capital')?.value || '10000');
    const positionSize = parseFloat(document.getElementById('position-size')?.value || '0.1');

    console.log('Form values:', { symbol, modelType, startDate, endDate, initialCapital, positionSize });

    try {
      setIsLoading(true);
      setError(null);

      // Make API call to run backtest
      console.log('Making API call to:', `${API_BASE_URL}/api/backtesting/run`);
      const response = await axios.post(`${API_BASE_URL}/api/backtesting/run`, null, {
        params: {
          symbol,
          model_type: modelType,
          start_date: startDate,
          end_date: endDate,
          initial_capital: initialCapital,
          position_size: positionSize
        }
      });
      console.log('API response:', response.data);

      // Set current result
      setCurrentResult(response.data);

      // Refresh backtest history
      fetchBacktestHistory();

      setIsLoading(false);
    } catch (error) {
      console.error('Error running backtest:', error);
      console.error('Error details:', error.response?.data || error.message);
      setError(`Failed to run backtest: ${error.response?.data?.detail || error.message || 'Unknown error'}`);
      setIsLoading(false);
    }
  };

  const mockStrategies = [
    {
      id: 'prediction_based',
      name: 'Prediction Based Strategy',
      description: 'Trading strategy based solely on model predictions'
    },
    {
      id: 'ma_prediction',
      name: 'Moving Average + Prediction Strategy',
      description: 'Trading strategy combining moving averages with model predictions'
    },
    {
      id: 'bollinger_bands',
      name: 'Bollinger Bands Strategy',
      description: 'Trading strategy using Bollinger Bands with model predictions'
    },
    {
      id: 'rsi_strategy',
      name: 'RSI Strategy',
      description: 'Trading strategy using RSI indicator with model predictions'
    }
  ];

  const containerStyle = {
    maxWidth: '1200px',
    margin: '0 auto',
    backgroundColor: 'white',
    padding: '20px',
    borderRadius: '8px',
    boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
  };

  const tabsStyle = {
    display: 'flex',
    borderBottom: '1px solid #e2e8f0',
    marginBottom: '20px'
  };

  const tabStyle = (tab) => ({
    padding: '10px 15px',
    cursor: 'pointer',
    borderBottom: activeTab === tab ? '2px solid #4299e1' : 'none',
    color: activeTab === tab ? '#4299e1' : 'inherit'
  });

  const cardStyle = {
    border: '1px solid #e2e8f0',
    borderRadius: '8px',
    padding: '20px',
    marginBottom: '20px'
  };

  const buttonStyle = {
    backgroundColor: '#4299e1',
    color: 'white',
    border: 'none',
    padding: '10px 15px',
    borderRadius: '4px',
    cursor: 'pointer',
    marginTop: '20px'
  };

  const tableStyle = {
    width: '100%',
    borderCollapse: 'collapse'
  };

  const thStyle = {
    padding: '10px',
    borderBottom: '1px solid #e2e8f0',
    backgroundColor: '#f7fafc',
    textAlign: 'left'
  };

  const tdStyle = {
    padding: '10px',
    borderBottom: '1px solid #e2e8f0'
  };

  const badgeStyle = (type) => ({
    display: 'inline-block',
    padding: '2px 8px',
    borderRadius: '4px',
    fontSize: '0.8em',
    fontWeight: 'bold',
    backgroundColor:
      type === 'lstm' ? '#bee3f8' :
      type === 'gru' ? '#d6bcfa' :
      type === 'xgboost' ? '#c6f6d5' : '#fefcbf',
    color:
      type === 'lstm' ? '#2b6cb0' :
      type === 'gru' ? '#6b46c1' :
      type === 'xgboost' ? '#2f855a' : '#975a16',
  });

  const statCardStyle = {
    border: '1px solid #e2e8f0',
    borderRadius: '8px',
    padding: '15px',
    marginBottom: '10px',
    backgroundColor: '#f7fafc'
  };

  const handleGoToModels = () => {
    navigate('/models');
  };

  return (
    <div style={containerStyle}>
      <h1 style={{ color: '#2c5282' }}>Backtesting</h1>

      <div style={tabsStyle}>
        <div
          style={tabStyle('run')}
          onClick={() => setActiveTab('run')}
        >
          Run Backtest
        </div>
        <div
          style={tabStyle('results')}
          onClick={() => setActiveTab('results')}
        >
          Backtest Results
        </div>
      </div>

      {activeTab === 'run' && (
        <div>
          <div style={cardStyle}>
            <h2 style={{ marginTop: 0, color: '#2c5282' }}>Backtest Configuration</h2>

            <div style={{ marginBottom: '20px', padding: '10px', backgroundColor: '#ebf8ff', borderRadius: '4px', borderLeft: '4px solid #4299e1' }}>
              <p style={{ margin: 0, color: '#2b6cb0' }}>
                Before running a backtest, make sure you have trained a model.
              </p>
              <button
                style={{ ...buttonStyle, marginTop: '10px', backgroundColor: '#3182ce' }}
                onClick={handleGoToModels}
              >
                Go to Model Training
              </button>
            </div>

            <form onSubmit={runBacktest}>
              {error && (
                <div style={{ color: 'red', marginBottom: '15px', padding: '10px', backgroundColor: '#FFF5F5', borderRadius: '4px' }}>
                  {error}
                </div>
              )}

              <div>
                <label htmlFor="crypto-symbol">Cryptocurrency:</label>
                <select id="crypto-symbol" style={{ marginLeft: '10px' }}>
                  {availableCryptos.map(crypto => (
                    <option key={crypto} value={crypto}>
                      {crypto === 'BTC' ? 'Bitcoin (BTC)' :
                       crypto === 'ETH' ? 'Ethereum (ETH)' :
                       crypto === 'BNB' ? 'Binance Coin (BNB)' :
                       crypto === 'XRP' ? 'Ripple (XRP)' :
                       crypto === 'ADA' ? 'Cardano (ADA)' : crypto}
                    </option>
                  ))}
                </select>
              </div>

              <div style={{ marginTop: '10px' }}>
                <label htmlFor="model-type">Model Type:</label>
                <select id="model-type" style={{ marginLeft: '10px' }}>
                  {availableModels.map(model => (
                    <option key={model} value={model}>
                      {model.toUpperCase()}
                    </option>
                  ))}
                </select>
              </div>

              <div style={{ marginTop: '10px' }}>
                <label htmlFor="strategy">Trading Strategy:</label>
                <select id="strategy" style={{ marginLeft: '10px' }}>
                  {mockStrategies.map(strategy => (
                    <option key={strategy.id} value={strategy.id}>
                      {strategy.name}
                    </option>
                  ))}
                </select>
              </div>

              <div style={{ marginTop: '10px' }}>
                <label htmlFor="start-date">Start Date:</label>
                <input
                  type="date"
                  id="start-date"
                  defaultValue={new Date(Date.now() - 180 * 24 * 60 * 60 * 1000).toISOString().split('T')[0]} // 180 days ago
                  style={{ marginLeft: '10px' }}
                />
              </div>

              <div style={{ marginTop: '10px' }}>
                <label htmlFor="end-date">End Date:</label>
                <input
                  type="date"
                  id="end-date"
                  defaultValue={new Date().toISOString().split('T')[0]} // Today
                  style={{ marginLeft: '10px' }}
                />
              </div>

              <div style={{ marginTop: '10px' }}>
                <label htmlFor="initial-capital">Initial Capital:</label>
                <input
                  type="number"
                  id="initial-capital"
                  defaultValue="10000"
                  min="100"
                  max="1000000"
                  style={{ marginLeft: '10px' }}
                />
              </div>

              <div style={{ marginTop: '10px' }}>
                <label htmlFor="position-size">Position Size (0-1):</label>
                <input
                  type="number"
                  id="position-size"
                  defaultValue="0.1"
                  min="0.01"
                  max="1"
                  step="0.01"
                  style={{ marginLeft: '10px' }}
                />
              </div>

              <button
                type="submit"
                style={buttonStyle}
                disabled={isLoading}
              >
                {isLoading ? 'Running...' : 'Run Backtest'}
              </button>
            </form>
          </div>

          <div style={cardStyle}>
            <h2 style={{ marginTop: 0, color: '#2c5282' }}>Current Backtest Result</h2>

            {isLoading ? (
              <div style={{ textAlign: 'center', padding: '20px' }}>
                <div style={{ fontSize: '2em', marginBottom: '10px' }}>⏳</div>
                <div>Running backtest...</div>
              </div>
            ) : currentResult ? (
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '15px' }}>
                <div style={{ ...statCardStyle, flex: '1 1 200px' }}>
                  <div style={{ fontSize: '0.9em', color: '#718096' }}>Final Value</div>
                  <div style={{ fontSize: '1.5em', fontWeight: 'bold', color: '#2d3748' }}>
                    ${currentResult.final_value.toFixed(2)}
                  </div>
                  <div style={{
                    color: currentResult.roi > 0 ? '#38a169' : '#e53e3e',
                    fontSize: '0.9em'
                  }}>
                    {currentResult.roi > 0 ? '+' : ''}{currentResult.roi.toFixed(2)}%
                  </div>
                </div>

                <div style={{ ...statCardStyle, flex: '1 1 200px' }}>
                  <div style={{ fontSize: '0.9em', color: '#718096' }}>Sharpe Ratio</div>
                  <div style={{ fontSize: '1.5em', fontWeight: 'bold', color: '#2d3748' }}>
                    {currentResult.sharpe_ratio.toFixed(2)}
                  </div>
                  <div style={{ fontSize: '0.9em', color: '#718096' }}>
                    {currentResult.sharpe_ratio > 1.5 ? 'Good' : currentResult.sharpe_ratio > 1 ? 'Decent' : 'Poor'} risk-adjusted return
                  </div>
                </div>

                <div style={{ ...statCardStyle, flex: '1 1 200px' }}>
                  <div style={{ fontSize: '0.9em', color: '#718096' }}>Max Drawdown</div>
                  <div style={{ fontSize: '1.5em', fontWeight: 'bold', color: '#2d3748' }}>
                    {currentResult.max_drawdown.toFixed(2)}%
                  </div>
                  <div style={{ fontSize: '0.9em', color: '#718096' }}>Maximum observed loss</div>
                </div>

                <div style={{ ...statCardStyle, flex: '1 1 200px' }}>
                  <div style={{ fontSize: '0.9em', color: '#718096' }}>Win Rate</div>
                  <div style={{ fontSize: '1.5em', fontWeight: 'bold', color: '#2d3748' }}>
                    {currentResult.win_rate.toFixed(2)}%
                  </div>
                  <div style={{ fontSize: '0.9em', color: '#718096' }}>
                    {currentResult.total_trades} trades
                  </div>
                </div>
              </div>
            ) : (
              <div style={{ textAlign: 'center', padding: '20px', color: '#718096' }}>
                <div>No backtest results yet. Run a backtest to see results here.</div>
              </div>
            )}
          </div>
        </div>
      )}

      {activeTab === 'results' && (
        <div style={cardStyle}>
          <h2 style={{ marginTop: 0, color: '#2c5282' }}>Historical Backtest Results</h2>

          {isLoading ? (
            <div style={{ textAlign: 'center', padding: '20px' }}>
              <div style={{ fontSize: '2em', marginBottom: '10px' }}>⏳</div>
              <div>Loading backtest history...</div>
            </div>
          ) : error ? (
            <div style={{ color: 'red', padding: '20px', backgroundColor: '#FFF5F5', borderRadius: '4px' }}>
              {error}
            </div>
          ) : backtestResults.length === 0 ? (
            <div style={{ textAlign: 'center', padding: '20px', color: '#718096' }}>
              <div>No backtest results found. Run a backtest to see results here.</div>
            </div>
          ) : (
            <div style={{ overflowX: 'auto' }}>
              <table style={tableStyle}>
                <thead>
                  <tr>
                    <th style={thStyle}>Symbol</th>
                    <th style={thStyle}>Model</th>
                    <th style={thStyle}>Period</th>
                    <th style={thStyle}>Initial Capital</th>
                    <th style={thStyle}>Final Value</th>
                    <th style={thStyle}>ROI</th>
                    <th style={thStyle}>Sharpe</th>
                    <th style={thStyle}>Max DD</th>
                    <th style={thStyle}>Win Rate</th>
                  </tr>
                </thead>
                <tbody>
                  {backtestResults.map((result, index) => (
                    <tr key={index}>
                      <td style={tdStyle}>{result.symbol}</td>
                      <td style={tdStyle}>
                        <span style={badgeStyle(result.model_type || 'unknown')}>
                          {result.model_type ? result.model_type.toUpperCase() : 'UNKNOWN'}
                        </span>
                      </td>
                      <td style={tdStyle}>
                        {result.start_date ? result.start_date : 'N/A'} to {result.end_date ? result.end_date : 'N/A'}
                      </td>
                      <td style={tdStyle}>
                        {typeof result.initial_capital === 'number' ? `$${result.initial_capital.toLocaleString()}` : 'N/A'}
                      </td>
                      <td style={tdStyle}>
                        {typeof result.final_value === 'number' ? `$${result.final_value.toLocaleString()}` : 'N/A'}
                      </td>
                      <td style={tdStyle}>
                        <span style={{ color: (result.roi && result.roi > 0) ? '#38a169' : '#e53e3e' }}>
                          {typeof result.roi === 'number' ? `${result.roi.toFixed(2)}%` : 'N/A'}
                        </span>
                      </td>
                      <td style={tdStyle}>
                        {typeof result.sharpe_ratio === 'number' ? result.sharpe_ratio.toFixed(2) : 'N/A'}
                      </td>
                      <td style={tdStyle}>
                        {typeof result.max_drawdown === 'number' ? `${result.max_drawdown.toFixed(1)}%` : 'N/A'}
                      </td>
                      <td style={tdStyle}>
                        {typeof result.win_rate === 'number' ? `${result.win_rate.toFixed(1)}%` : 'N/A'}
                        {typeof result.total_trades === 'number' ? ` (${result.total_trades} trades)` : ''}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>

              <div style={{ marginTop: '20px', textAlign: 'center' }}>
                <button
                  style={buttonStyle}
                  onClick={fetchBacktestHistory}
                  disabled={isLoading}
                >
                  {isLoading ? 'Refreshing...' : 'Refresh Results'}
                </button>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default SimpleBacktesting;
