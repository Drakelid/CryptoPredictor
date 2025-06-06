# CryptoPricer

A full-stack, AI-powered web application for predicting cryptocurrency prices (e.g., BTC, ETH) using deep learning and machine learning. Supports continuous learning, interpretability, and backtesting—all accessible via a modern web interface.

## 🚀 Features

- **Price Forecasting**: Predicts short-term crypto prices using LSTM/GRU and ML models.
- **Interactive Dashboard**: Visualizes predictions with candlestick charts, prediction overlays, and SHAP-based explanations.
- **Model Training**: Upload new data and retrain models via the web UI.
- **Backtesting**: Simulate trading performance with different strategies.
- **Continuous Learning**: Online model updates and memory-based recall using vector databases.
- **Ensemble Predictions**: Optionally combine multiple models for improved accuracy.

## 🧱 Tech Stack

| Layer | Tools/Frameworks |
|-------|-----------------|
| Frontend | React with Chakra UI |
| Backend | FastAPI (REST API), Pydantic |
| ML/DL | TensorFlow, XGBoost, LightGBM |
| Online Learning | River (for streaming model updates) |
| Tuning | Optuna |
| Backtesting | Backtrader |
| Interpretability | SHAP |
| Vector Store | FAISS |
| Containerization | Docker + docker-compose |

## 📊 Data Sources & Feature Engineering

- **Historical Data**: CoinGecko / Binance OHLCV prices
- **Engineered Features**: Technical indicators like MA, RSI, MACD, Bollinger Bands
- **Normalization**: All features scaled appropriately

## 🤖 Modeling Overview

### 🔷 Deep Learning Models
- **Type**: LSTM or GRU
- **Inputs**: Sequences of engineered features
- **Outputs**: Next time step price or log return
- **Framework**: TensorFlow/Keras
- **Loss**: MSE/MAE

### 🔶 Machine Learning Baselines
- **Models**: XGBoost & LightGBM
- **Inputs**: Tabular features (lags, volatility, sentiment)
- **Tuning**: Hyperparameter search via Optuna

## 🔍 Interpretability

- SHAP for feature attributions
- Dashboard visualization of top features per prediction

## 🔁 Continuous Learning

- **Online Models**: River for lightweight, per-batch model updates
- **Scheduled Retrains**: Fine-tune DL models with new data
- **Memory Store**: Embed historical patterns in FAISS for similar pattern recall

## 🧪 Backtesting

- **Framework**: Backtrader
- **Strategies**: Model-based long/short, Moving average + prediction combo
- **Metrics**: ROI, Sharpe, Drawdown, Trade stats

## 📦 Setup & Deployment

### Docker (recommended)

```bash
# Build containers
docker-compose build

# Run app
docker-compose up
```

The frontend expects the backend to be reachable at the URL defined by
`VITE_API_BASE_URL` during its build process. The provided Dockerfile sets
this to `http://backend:8000` so the frontend container can communicate with
the backend service defined in `docker-compose.yml`.

### Dev Setup

```bash
# Backend
cd backend
pip install -r requirements.txt
uvicorn main:app --reload

# Frontend
cd frontend
npm install
npm run dev
```

## 🧪 Testing

- Unit tests for data preprocessing, feature engineering, model I/O
- Integration tests for end-to-end data → model → prediction
- API tests using FastAPI TestClient

## 📚 Documentation

- FastAPI auto-generated docs at /docs
- README covers architecture, usage, and setup

## ✅ Future Enhancements

- Multi-step forecasting
- Reinforcement learning for trading policy
- Real-time streaming prediction
- Model registry (e.g., MLflow)
