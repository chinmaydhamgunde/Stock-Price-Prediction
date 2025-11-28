"""
Improved Real-Time Predictor - Handles high prices better
"""

import numpy as np
import pandas as pd
import pickle
from tensorflow import keras
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class ImprovedRealtimePredictor:
    def __init__(self, ticker, model_path, scaler_path, sequence_length=60):
        """Initialize improved predictor"""
        self.ticker = ticker
        self.sequence_length = sequence_length
        
        print(f"\n🔄 Loading model for {ticker}...")
        
        try:
            self.model = keras.models.load_model(model_path)
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            print(f"✅ Model and scaler loaded!")
        except Exception as e:
            print(f"❌ Error: {e}")
            raise
        
        self.stock = yf.Ticker(ticker)
    
    def fetch_latest_data(self, days=200):
        """Fetch latest data"""
        print(f"\n📡 Fetching latest data for {self.ticker}...")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        data = self.stock.history(start=start_date, end=end_date)
        
        if data.empty:
            raise ValueError(f"No data available")
        
        try:
            current_price = self.stock.fast_info.last_price
        except:
            current_price = data['Close'].iloc[-1]
        
        print(f"💰 Current Price: ${current_price:.2f}")
        print(f"✅ Fetched {len(data)} days of data")
        
        return data, current_price
    
    def add_technical_indicators(self, data):
        """Add technical indicators"""
        print("🔧 Calculating indicators...")
        
        df = data.copy()
        
        # Moving Averages
        df['MA_7'] = df['Close'].rolling(window=7).mean()
        df['MA_21'] = df['Close'].rolling(window=21).mean()
        df['MA_50'] = df['Close'].rolling(window=50).mean()
        
        # EMA
        df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
        
        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        
        # Volatility
        df['Daily_Return'] = df['Close'].pct_change()
        df['Volatility'] = df['Daily_Return'].rolling(window=20).std()
        
        # Volume
        df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
        df['High_Low_Diff'] = df['High'] - df['Low']
        df['Open_Close_Diff'] = df['Open'] - df['Close']
        
        df = df.dropna()
        
        print(f"✅ Indicators calculated")
        return df
    
    def hybrid_prediction(self, data, current_price):
        """Hybrid approach: Model + Technical Analysis"""
        print("\n🔮 Making hybrid prediction...")
        
        # Prepare features
        feature_columns = [
            'Close', 'Open', 'High', 'Low', 'Volume',
            'MA_7', 'MA_21', 'MA_50', 'EMA_12', 'EMA_26',
            'MACD', 'Signal_Line', 'RSI', 
            'BB_Middle', 'BB_Upper', 'BB_Lower',
            'Daily_Return', 'Volatility', 'Volume_MA',
            'High_Low_Diff', 'Open_Close_Diff'
        ]
        
        features = data[feature_columns].values
        
        # Check if current price is extreme
        historical_max = self.scaler.data_max_[0]
        price_ratio = current_price / historical_max
        
        print(f"📊 Price Analysis:")
        print(f"   Historical max in training: ${historical_max:.2f}")
        print(f"   Current price ratio: {price_ratio:.2f}x")
        
        if price_ratio > 1.1:  # Current price is 10% above training max
            print(f"   ⚠️  Price above training range - using hybrid approach")
            use_hybrid = True
        else:
            print(f"   ✅ Price within training range - using model")
            use_hybrid = False
        
        # Get model prediction
        normalized_features = self.scaler.transform(features)
        last_sequence = normalized_features[-self.sequence_length:]
        input_data = last_sequence.reshape(1, self.sequence_length, last_sequence.shape[1])
        
        pred_normalized = self.model.predict(input_data, verbose=0)
        
        # Inverse transform
        dummy = np.zeros((1, self.scaler.n_features_in_))
        dummy[0, 0] = pred_normalized[0, 0]
        model_prediction = self.scaler.inverse_transform(dummy)[0, 0]
        
        if not use_hybrid:
            # Trust the model
            final_prediction = model_prediction
        else:
            # Hybrid: Combine model with technical analysis
            
            # Technical signals
            recent_return = data['Daily_Return'].tail(5).mean()
            momentum_20d = (current_price - data['Close'].iloc[-20]) / data['Close'].iloc[-20]
            current_rsi = data['RSI'].iloc[-1]
            
            # RSI signal
            if current_rsi > 70:
                rsi_bias = -0.01  # Overbought
            elif current_rsi < 30:
                rsi_bias = 0.01  # Oversold
            else:
                rsi_bias = 0
            
            # Combine signals
            technical_return = (
                0.4 * recent_return +
                0.3 * (momentum_20d / 20) +  # Daily equivalent
                0.3 * rsi_bias
            )
            
            technical_prediction = current_price * (1 + technical_return)
            
            # Weighted average (more weight on technical for extreme prices)
            model_weight = 0.3 if price_ratio > 1.2 else 0.5
            technical_weight = 1 - model_weight
            
            final_prediction = (model_weight * model_prediction + 
                              technical_weight * technical_prediction)
            
            print(f"\n🔄 Hybrid Combination:")
            print(f"   Model prediction: ${model_prediction:.2f} (weight: {model_weight:.0%})")
            print(f"   Technical prediction: ${technical_prediction:.2f} (weight: {technical_weight:.0%})")
        
        # Calculate confidence interval based on recent volatility
        volatility = data['Daily_Return'].tail(20).std()
        lower_bound = final_prediction * (1 - 1.65 * volatility)
        upper_bound = final_prediction * (1 + 1.65 * volatility)
        
        return final_prediction, lower_bound, upper_bound
    
    def display_report(self, current_price, predicted_price, lower_bound, upper_bound):
        """Display prediction report"""
        
        print("\n" + "="*70)
        print(f"📊 IMPROVED REAL-TIME PREDICTION - {self.ticker}")
        print("="*70)
        
        print(f"\n📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S IST')}")
        print(f"💰 Current Price: ${current_price:.2f}")
        
        print(f"\n🔮 TOMORROW'S PREDICTION:")
        print(f"   Predicted Price: ${predicted_price:.2f}")
        
        change = predicted_price - current_price
        change_pct = (change / current_price) * 100
        
        direction = "⬆️" if change > 0 else "⬇️"
        print(f"   Expected Change: {direction} ${abs(change):.2f} ({abs(change_pct):.2f}%)")
        
        print(f"\n📊 CONFIDENCE INTERVAL (90%):")
        print(f"   Lower Bound: ${lower_bound:.2f}")
        print(f"   Upper Bound: ${upper_bound:.2f}")
        
        # Recommendation
        if abs(change_pct) < 1:
            rec = "🟡 HOLD"
            reason = "Minimal expected change"
        elif change_pct > 2:
            rec = "🟢 STRONG BUY"
            reason = f"Strong upward momentum expected"
        elif change_pct > 0.5:
            rec = "🟢 BUY"
            reason = f"Positive trend expected"
        elif change_pct < -2:
            rec = "🔴 STRONG SELL"
            reason = f"Strong downward pressure"
        else:
            rec = "🔴 SELL"
            reason = f"Negative trend expected"
        
        print(f"\n💡 RECOMMENDATION: {rec}")
        print(f"   {reason}")
        
        print(f"\n⚠️  DISCLAIMER:")
        print(f"   Uses hybrid model + technical analysis for high price levels")
        print(f"   Not financial advice - for educational purposes only")
        
        print("\n" + "="*70 + "\n")
    
    def run(self):
        """Run improved prediction"""
        try:
            data, current_price = self.fetch_latest_data(days=200)
            data = self.add_technical_indicators(data)
            
            predicted_price, lower_bound, upper_bound = self.hybrid_prediction(
                data, current_price
            )
            
            self.display_report(current_price, predicted_price, lower_bound, upper_bound)
            
            return predicted_price, current_price
            
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            raise


if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    
    ticker = input("Enter ticker [TSLA]: ").upper() or 'TSLA'
    
    predictor = ImprovedRealtimePredictor(
        ticker=ticker,
        model_path=f'models/{ticker}_lstm_model.keras',
        scaler_path='models/scaler.pkl'
    )
    
    predictor.run()
