import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';

const SimpleModelTraining = () => {
  const [activeTab, setActiveTab] = useState('train');
  const navigate = useNavigate();
  
  // Mock data
  const mockModelInfo = [
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
    },
    {
      symbol: 'SOL',
      model_type: 'gru',
      training_date: '2025-04-03T09:15:00Z',
      metrics: {
        mse: 0.0031,
        mae: 0.0412,
        r2: 0.83
      }
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

  const handleGoToData = () => {
    navigate('/data');
  };
  
  return (
    <div style={containerStyle}>
      <h1 style={{ color: '#2c5282' }}>Model Training</h1>
      
      <div style={tabsStyle}>
        <div 
          style={tabStyle('train')} 
          onClick={() => setActiveTab('train')}
        >
          Train Model
        </div>
        <div 
          style={tabStyle('models')} 
          onClick={() => setActiveTab('models')}
        >
          Trained Models
        </div>
      </div>
      
      {activeTab === 'train' && (
        <div style={cardStyle}>
          <h2 style={{ marginTop: 0, color: '#2c5282' }}>Train New Model</h2>
          
          <div style={{ marginBottom: '20px', padding: '10px', backgroundColor: '#ebf8ff', borderRadius: '4px', borderLeft: '4px solid #4299e1' }}>
            <p style={{ margin: 0, color: '#2b6cb0' }}>
              Before training a model, make sure you have fetched or uploaded cryptocurrency data.
            </p>
            <button 
              style={{ ...buttonStyle, marginTop: '10px', backgroundColor: '#3182ce' }}
              onClick={handleGoToData}
            >
              Go to Data Management
            </button>
          </div>
          
          <div>
            <label htmlFor="crypto">Cryptocurrency:</label>
            <select id="crypto" style={{ marginLeft: '10px' }}>
              <option value="BTC">Bitcoin (BTC)</option>
              <option value="ETH">Ethereum (ETH)</option>
              <option value="SOL">Solana (SOL)</option>
            </select>
          </div>
          
          <div style={{ marginTop: '10px' }}>
            <label htmlFor="model-type">Model Type:</label>
            <select id="model-type" style={{ marginLeft: '10px' }}>
              <option value="lstm">LSTM</option>
              <option value="gru">GRU</option>
              <option value="xgboost">XGBoost</option>
              <option value="lightgbm">LightGBM</option>
            </select>
          </div>
          
          <div style={{ marginTop: '10px' }}>
            <label htmlFor="epochs">Epochs:</label>
            <input 
              type="number" 
              id="epochs" 
              defaultValue="50" 
              min="1" 
              max="1000" 
              style={{ marginLeft: '10px' }}
            />
          </div>
          
          <div style={{ marginTop: '10px' }}>
            <label htmlFor="batch-size">Batch Size:</label>
            <input 
              type="number" 
              id="batch-size" 
              defaultValue="32" 
              min="1" 
              max="256" 
              style={{ marginLeft: '10px' }}
            />
          </div>
          
          <div style={{ marginTop: '10px' }}>
            <label htmlFor="lookback">Lookback Days:</label>
            <input 
              type="number" 
              id="lookback" 
              defaultValue="30" 
              min="1" 
              max="365" 
              style={{ marginLeft: '10px' }}
            />
          </div>
          
          <div style={{ marginTop: '10px' }}>
            <label htmlFor="sentiment">Include Sentiment Analysis:</label>
            <input 
              type="checkbox" 
              id="sentiment" 
              defaultChecked={true} 
              style={{ marginLeft: '10px' }}
            />
          </div>
          
          <button style={buttonStyle}>Train Model</button>
        </div>
      )}
      
      {activeTab === 'models' && (
        <div style={cardStyle}>
          <h2 style={{ marginTop: 0, color: '#2c5282' }}>Trained Models</h2>
          <table style={tableStyle}>
            <thead>
              <tr>
                <th style={thStyle}>Symbol</th>
                <th style={thStyle}>Model Type</th>
                <th style={thStyle}>Training Date</th>
                <th style={thStyle}>Metrics</th>
                <th style={thStyle}>Actions</th>
              </tr>
            </thead>
            <tbody>
              {mockModelInfo.map((model, index) => (
                <tr key={index}>
                  <td style={tdStyle}>{model.symbol}</td>
                  <td style={tdStyle}>
                    <span style={badgeStyle(model.model_type)}>
                      {model.model_type.toUpperCase()}
                    </span>
                  </td>
                  <td style={tdStyle}>{new Date(model.training_date).toLocaleString()}</td>
                  <td style={tdStyle}>
                    <div>MSE: {model.metrics.mse.toFixed(4)}</div>
                    <div>MAE: {model.metrics.mae.toFixed(4)}</div>
                    <div>R²: {model.metrics.r2.toFixed(2)}</div>
                  </td>
                  <td style={tdStyle}>
                    <button style={{ ...buttonStyle, marginTop: 0, padding: '5px 10px', backgroundColor: '#e53e3e' }}>
                      Delete
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
};

export default SimpleModelTraining;
