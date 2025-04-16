import React, { useState } from 'react';

const SimpleDataManagement = () => {
  const [activeTab, setActiveTab] = useState('fetch');
  
  // Mock data
  const mockDataInfo = [
    {
      symbol: 'BTC',
      source: 'coingecko',
      start_date: '2024-04-04',
      end_date: '2025-04-04',
      rows: 365,
      columns: ['timestamp', 'price', 'volume', 'market_cap']
    },
    {
      symbol: 'ETH',
      source: 'coingecko',
      start_date: '2024-04-04',
      end_date: '2025-04-04',
      rows: 365,
      columns: ['timestamp', 'price', 'volume', 'market_cap']
    },
    {
      symbol: 'SOL',
      source: 'coingecko',
      start_date: '2024-04-04',
      end_date: '2025-04-04',
      rows: 365,
      columns: ['timestamp', 'price', 'volume', 'market_cap']
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
  
  return (
    <div style={containerStyle}>
      <h1 style={{ color: '#2c5282' }}>Data Management</h1>
      
      <div style={tabsStyle}>
        <div 
          style={tabStyle('fetch')} 
          onClick={() => setActiveTab('fetch')}
        >
          Fetch Data
        </div>
        <div 
          style={tabStyle('upload')} 
          onClick={() => setActiveTab('upload')}
        >
          Upload Data
        </div>
        <div 
          style={tabStyle('available')} 
          onClick={() => setActiveTab('available')}
        >
          Available Data
        </div>
      </div>
      
      {activeTab === 'fetch' && (
        <div style={cardStyle}>
          <h2 style={{ marginTop: 0, color: '#2c5282' }}>Fetch Cryptocurrency Data</h2>
          <div>
            <label htmlFor="crypto">Cryptocurrency:</label>
            <select id="crypto" style={{ marginLeft: '10px' }}>
              <option value="BTC">Bitcoin (BTC)</option>
              <option value="ETH">Ethereum (ETH)</option>
              <option value="SOL">Solana (SOL)</option>
            </select>
          </div>
          <div style={{ marginTop: '10px' }}>
            <label htmlFor="source">Data Source:</label>
            <select id="source" style={{ marginLeft: '10px' }}>
              <option value="coingecko">CoinGecko</option>
              <option value="binance">Binance</option>
            </select>
          </div>
          <div style={{ marginTop: '10px' }}>
            <label htmlFor="days">Days of Historical Data:</label>
            <input 
              type="number" 
              id="days" 
              defaultValue="365" 
              min="1" 
              max="2000" 
              style={{ marginLeft: '10px' }}
            />
          </div>
          <button style={buttonStyle}>Fetch Data</button>
        </div>
      )}
      
      {activeTab === 'upload' && (
        <div style={cardStyle}>
          <h2 style={{ marginTop: 0, color: '#2c5282' }}>Upload Cryptocurrency Data</h2>
          <div>
            <label htmlFor="upload-crypto">Cryptocurrency:</label>
            <select id="upload-crypto" style={{ marginLeft: '10px' }}>
              <option value="BTC">Bitcoin (BTC)</option>
              <option value="ETH">Ethereum (ETH)</option>
              <option value="SOL">Solana (SOL)</option>
            </select>
          </div>
          <div style={{ marginTop: '10px' }}>
            <label htmlFor="custom-source">Data Source:</label>
            <input 
              type="text" 
              id="custom-source" 
              placeholder="Enter custom source name" 
              style={{ marginLeft: '10px' }}
            />
          </div>
          <div style={{ marginTop: '10px' }}>
            <label htmlFor="file">CSV File:</label>
            <input 
              type="file" 
              id="file" 
              accept=".csv" 
              style={{ marginLeft: '10px' }}
            />
            <p style={{ fontSize: '0.8em', color: '#718096' }}>
              CSV file should contain columns: timestamp, price, volume, etc.
            </p>
          </div>
          <button style={buttonStyle}>Upload Data</button>
        </div>
      )}
      
      {activeTab === 'available' && (
        <div style={cardStyle}>
          <h2 style={{ marginTop: 0, color: '#2c5282' }}>Available Data</h2>
          <table style={tableStyle}>
            <thead>
              <tr>
                <th style={thStyle}>Symbol</th>
                <th style={thStyle}>Source</th>
                <th style={thStyle}>Date Range</th>
                <th style={thStyle}>Rows</th>
                <th style={thStyle}>Columns</th>
              </tr>
            </thead>
            <tbody>
              {mockDataInfo.map((item, index) => (
                <tr key={index}>
                  <td style={tdStyle}>{item.symbol}</td>
                  <td style={tdStyle}>{item.source}</td>
                  <td style={tdStyle}>{item.start_date} to {item.end_date}</td>
                  <td style={tdStyle}>{item.rows}</td>
                  <td style={tdStyle}>{item.columns.join(', ')}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
};

export default SimpleDataManagement;
