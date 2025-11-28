# 📈 Stock Price Prediction System

> LSTM-based machine learning system for predicting stock prices using technical indicators and real-time data.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 🎯 What It Does

- Predicts tomorrow's stock price using deep learning (LSTM neural networks)
- Analyzes 21+ technical indicators (MACD, RSI, Bollinger Bands, etc.)
- Fetches real-time data from Yahoo Finance
- Provides confidence intervals and buy/sell recommendations
- Works with any stock ticker (TSLA, AAPL, GOOGL, etc.)

## 🚀 Quick Start

1. Clone and setup
```
git clone https://github.com/chinmaydhamgunde/Stock-Price-Prediction
cd stock-price-prediction-ml
```
2. Install dependencies
```
pip install -r requirements.txt
```
3. Train a model
```
python main.py
```
4. Get predictions
```
python predict_realtime.py
```


## 📦 Requirements
```
tensorflow==2.18.0
keras==3.6.0
yfinance==0.2.54
pandas==2.2.3
numpy==2.1.3
scikit-learn==1.6.0
matplotlib==3.9.2
plotly==5.24.1
```


## 💻 Usage

### Train a Model
```
python main.py

Enter stock ticker (e.g., TSLA)
Model saves to models/ folder
```

### Real-Time Prediction
```
python predict_realtime.py

Get tomorrow's price prediction
Includes 7-day forecast and confidence intervals
```

### Production Predictor (Most Reliable)
```
python src/production_predictor.py

Uses ensemble of 25+ technical indicators
Works at any price level
```

## 📊 Results
```

| Metric | Value |
|--------|-------|
| MAPE | 4.2% |
| R² Score | 0.89 |
| Direction Accuracy | 85% |

**Sample Output:**
Current Price: $429.16
Predicted Price: $432.69 (+0.82%)
Confidence Interval: $408.95 - $456.43
Recommendation: 🟢 BUY
```


## 🏗️ Project Structure
```
stock-price-prediction-ml/
├── src/ # Source code
├── models/ # Trained models
├── data/ # Historical stock data
├── outputs/ # Predictions & visualizations
└── main.py # Training pipeline

```

## 🎓 Key Features

**Three Prediction Methods:**
1. **LSTM Model** - Deep learning predictions (best for prices within training range)
2. **Hybrid Predictor** - Combines LSTM + technical analysis
3. **Production System** - Ensemble approach (most robust)

**Technical Indicators:**
- Moving Averages (SMA, EMA)
- MACD & Signal Line
- RSI (Relative Strength Index)
- Bollinger Bands
- Volatility & Volume Analysis

## ⚠️ Important Notes

- **Distribution Shift**: Model works best within training data range
- **Retraining**: Retrain periodically with fresh data for best accuracy
- **Not Financial Advice**: Educational project only - do your own research

## 🐛 Troubleshooting

**"Not enough data" error?**
```
Retrain with latest data
python main.py

```

**Unrealistic predictions?**
```
Use production predictor instead
python src/production_predictor.py
```


## 🤝 Contributing

Pull requests welcome! Feel free to:
- Add new technical indicators
- Improve model architecture
- Test on different stocks
- Enhance documentation



**⭐ Star this repo if you found it helpful!**

*Disclaimer: This is for educational purposes only. Stock predictions are uncertain - always do your own research.*