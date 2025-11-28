"""
Returns-based predictor - More robust to price level changes
"""

import numpy as np
import pandas as pd
import pickle
from tensorflow import keras
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class ReturnBasedPredictor:
    def __init__(self, ticker, model_path, scaler_path, sequence_length=60):
        """Initialize with returns-based approach"""
        self.ticker = ticker
        self.sequence_length = sequence_length
        
        print(f"\n🔄 Loading model for {ticker}...")
        
        try:
            self.model = keras.models.load_model(model_path)
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            print(f"✅ Model loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
        
        self.stock = yf.Ticker(ticker)
    
    def fetch_latest_data(self, days=200):
        """Fetch latest stock data"""
        print(f"\n📡 Fetching latest data for {self.ticker}...")
        
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            data = self.stock.history(start=start_date, end=end_date)
            
            if data.empty:
                raise ValueError(f"No data available for {self.ticker}")
            
            try:
                current_price = self.stock.fast_info.last_price
            except:
                current_price = data['Close'].iloc[-1]
            
            print(f"💰 Current Price: ${current_price:.2f}")
            print(f"✅ Fetched {len(data)} days of data")
            
            return data, current_price
            
        except Exception as e:
            print(f"❌ Error fetching data: {e}")
            raise
    
    def calculate_returns(self, data):
        """Calculate percentage returns"""
        print("📊 Calculating returns-based features...")
        
        df = data.copy()
        
        # Calculate percentage returns
        df['Return'] = df['Close'].pct_change()
        df['Volume_Change'] = df['Volume'].pct_change()
        
        # Technical indicators based on returns
        df['MA_7_Return'] = df['Return'].rolling(window=7).mean()
        df['MA_21_Return'] = df['Return'].rolling(window=21).mean()
        df['Volatility'] = df['Return'].rolling(window=20).std()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        df = df.dropna()
        
        print(f"✅ Returns calculated - {len(df)} days available")
        
        return df
    
    def make_robust_prediction(self, data, current_price):
        """Make prediction using statistical adjustment"""
        print("\n🔮 Making prediction with distribution adjustment...")
        
        # Get recent price trend
        recent_prices = data['Close'].tail(30).values
        recent_avg = np.mean(recent_prices)
        recent_std = np.std(recent_prices)
        recent_trend = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
        
        print(f"📈 Recent 30-day trend: {recent_trend*100:.2f}%")
        print(f"📊 Recent avg price: ${recent_avg:.2f} (±${recent_std:.2f})")
        
        # Calculate expected return based on momentum and volatility
        # This is a simplified approach that doesn't rely on the potentially outdated model
        
        # Short-term momentum (last 5 days)
        momentum_5d = (current_price - data['Close'].iloc[-5]) / data['Close'].iloc[-5]
        
        # Medium-term momentum (last 20 days)
        momentum_20d = (current_price - data['Close'].iloc[-20]) / data['Close'].iloc[-20]
        
        # RSI signal
        current_rsi = data['RSI'].iloc[-1] if 'RSI' in data.columns else 50
        rsi_signal = (50 - current_rsi) / 100  # Oversold → positive, Overbought → negative
        
        # Volatility adjustment
        volatility = data['Return'].tail(20).std() if 'Return' in data.columns else 0.02
        
        # Combine signals (weighted average)
        expected_return = (
            0.3 * momentum_5d +
            0.3 * momentum_20d +
            0.2 * rsi_signal +
            0.2 * recent_trend
        )
        
        # Add confidence bounds based on volatility
        lower_return = expected_return - (1.65 * volatility)  # 90% confidence
        upper_return = expected_return + (1.65 * volatility)
        
        # Convert to prices
        predicted_price = current_price * (1 + expected_return)
        lower_bound = current_price * (1 + lower_return)
        upper_bound = current_price * (1 + upper_return)
        
        print(f"✅ Prediction complete using statistical approach")
        
        return predicted_price, lower_bound, upper_bound
    
    def display_prediction_report(self, current_price, predicted_price, lower_bound, upper_bound):
        """Display prediction report"""
        
        print("\n" + "="*70)
        print(f"📊 ADJUSTED REAL-TIME PREDICTION - {self.ticker}")
        print("="*70)
        
        print(f"\n📅 Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S IST')}")
        print(f"💰 Current Price: ${current_price:.2f}")
        
        print(f"\n🔮 TOMORROW'S PREDICTION:")
        print(f"   Predicted Price: ${predicted_price:.2f}")
        
        price_change = predicted_price - current_price
        price_change_pct = (price_change / current_price) * 100
        
        direction = "⬆️" if price_change > 0 else "⬇️"
        print(f"   Expected Change: {direction} ${abs(price_change):.2f} ({abs(price_change_pct):.2f}%)")
        
        print(f"\n📊 CONFIDENCE INTERVAL (90%):")
        print(f"   Lower Bound: ${lower_bound:.2f} ({((lower_bound/current_price-1)*100):.2f}%)")
        print(f"   Upper Bound: ${upper_bound:.2f} ({((upper_bound/current_price-1)*100):.2f}%)")
        
        # Recommendation
        if abs(price_change_pct) < 0.5:
            rec = "🟡 HOLD"
        elif price_change_pct > 1.5:
            rec = "🟢 STRONG BUY"
        elif price_change_pct > 0.5:
            rec = "🟢 BUY"
        elif price_change_pct < -1.5:
            rec = "🔴 STRONG SELL"
        else:
            rec = "🔴 SELL"
        
        print(f"\n💡 RECOMMENDATION: {rec}")
        
        print(f"\n⚠️  NOTE:")
        print(f"   This uses statistical momentum and technical indicators.")
        print(f"   For best results, retrain your model with recent data.")
        
        print("\n" + "="*70 + "\n")
    
    def run(self):
        """Run the robust prediction"""
        try:
            data, current_price = self.fetch_latest_data(days=200)
            data = self.calculate_returns(data)
            
            predicted_price, lower_bound, upper_bound = self.make_robust_prediction(
                data, current_price
            )
            
            self.display_prediction_report(
                current_price, predicted_price, lower_bound, upper_bound
            )
            
            return predicted_price, current_price
            
        except Exception as e:
            print(f"❌ Error: {e}")
            raise


if __name__ == "__main__":
    import sys
    import os
    
    ticker = input("Enter ticker [TSLA]: ").upper() or 'TSLA'
    
    predictor = ReturnBasedPredictor(
        ticker=ticker,
        model_path=f'models/{ticker}_lstm_model.keras',
        scaler_path='models/scaler.pkl'
    )
    
    predictor.run()
