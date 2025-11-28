"""
Production-Ready Stock Predictor - Returns-based approach
Handles any price level by predicting percentage changes
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class ProductionStockPredictor:
    def __init__(self, ticker):
        """Initialize production predictor"""
        self.ticker = ticker
        self.stock = yf.Ticker(ticker)
        
    def fetch_latest_data(self, days=200):
        """Fetch latest market data"""
        print(f"\n📡 Fetching latest data for {self.ticker}...")
        
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
        print(f"✅ Fetched {len(data)} trading days")
        
        return data, current_price
    
    def calculate_technical_indicators(self, data):
        """Calculate comprehensive technical indicators"""
        print("🔧 Calculating technical indicators...")
        
        df = data.copy()
        
        # Price-based indicators
        df['SMA_5'] = df['Close'].rolling(window=5).mean()
        df['SMA_10'] = df['Close'].rolling(window=10).mean()
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        
        # EMA
        df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
        
        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
        
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
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
        
        # Returns and volatility
        df['Daily_Return'] = df['Close'].pct_change()
        df['Volatility_5'] = df['Daily_Return'].rolling(window=5).std()
        df['Volatility_20'] = df['Daily_Return'].rolling(window=20).std()
        
        # Volume indicators
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        
        # Momentum indicators
        df['ROC_5'] = ((df['Close'] - df['Close'].shift(5)) / df['Close'].shift(5)) * 100
        df['ROC_10'] = ((df['Close'] - df['Close'].shift(10)) / df['Close'].shift(10)) * 100
        
        # Price position relative to moving averages
        df['Price_to_SMA20'] = (df['Close'] - df['SMA_20']) / df['SMA_20']
        df['Price_to_SMA50'] = (df['Close'] - df['SMA_50']) / df['SMA_50']
        
        df = df.dropna()
        
        print(f"✅ Calculated 25+ technical indicators")
        
        return df
    
    def generate_signals(self, data):
        """Generate trading signals from technical indicators"""
        print("\n📊 Analyzing market signals...")
        
        latest = data.iloc[-1]
        
        signals = {}
        
        # 1. Trend signals
        if latest['Close'] > latest['SMA_20'] > latest['SMA_50']:
            signals['trend'] = {'signal': 'bullish', 'strength': 0.8, 'description': 'Strong uptrend'}
        elif latest['Close'] < latest['SMA_20'] < latest['SMA_50']:
            signals['trend'] = {'signal': 'bearish', 'strength': 0.8, 'description': 'Strong downtrend'}
        else:
            signals['trend'] = {'signal': 'neutral', 'strength': 0.3, 'description': 'Mixed trend'}
        
        # 2. Momentum signals
        if latest['RSI'] > 70:
            signals['momentum'] = {'signal': 'overbought', 'strength': -0.6, 'description': 'Overbought (RSI > 70)'}
        elif latest['RSI'] < 30:
            signals['momentum'] = {'signal': 'oversold', 'strength': 0.6, 'description': 'Oversold (RSI < 30)'}
        elif 45 <= latest['RSI'] <= 55:
            signals['momentum'] = {'signal': 'neutral', 'strength': 0.0, 'description': 'Neutral momentum'}
        else:
            rsi_signal = (50 - latest['RSI']) / 50  # Normalize around 50
            signals['momentum'] = {'signal': 'moderate', 'strength': rsi_signal * 0.4, 'description': f'RSI at {latest["RSI"]:.1f}'}
        
        # 3. MACD signals
        if latest['MACD'] > latest['MACD_Signal'] and latest['MACD_Hist'] > 0:
            signals['macd'] = {'signal': 'bullish', 'strength': 0.5, 'description': 'MACD bullish crossover'}
        elif latest['MACD'] < latest['MACD_Signal'] and latest['MACD_Hist'] < 0:
            signals['macd'] = {'signal': 'bearish', 'strength': -0.5, 'description': 'MACD bearish crossover'}
        else:
            signals['macd'] = {'signal': 'neutral', 'strength': 0.0, 'description': 'MACD neutral'}
        
        # 4. Volatility signals
        if latest['Volatility_20'] > data['Volatility_20'].mean() * 1.5:
            signals['volatility'] = {'signal': 'high', 'strength': -0.2, 'description': 'High volatility - caution'}
        else:
            signals['volatility'] = {'signal': 'normal', 'strength': 0.0, 'description': 'Normal volatility'}
        
        # 5. Volume signals
        if latest['Volume_Ratio'] > 1.5:
            signals['volume'] = {'signal': 'high', 'strength': 0.3, 'description': 'Above-average volume'}
        elif latest['Volume_Ratio'] < 0.5:
            signals['volume'] = {'signal': 'low', 'strength': -0.2, 'description': 'Below-average volume'}
        else:
            signals['volume'] = {'signal': 'normal', 'strength': 0.0, 'description': 'Normal volume'}
        
        # 6. Bollinger Bands
        price_position = (latest['Close'] - latest['BB_Lower']) / (latest['BB_Upper'] - latest['BB_Lower'])
        if price_position > 0.9:
            signals['bollinger'] = {'signal': 'overbought', 'strength': -0.4, 'description': 'Near upper Bollinger Band'}
        elif price_position < 0.1:
            signals['bollinger'] = {'signal': 'oversold', 'strength': 0.4, 'description': 'Near lower Bollinger Band'}
        else:
            signals['bollinger'] = {'signal': 'normal', 'strength': 0.0, 'description': 'Within Bollinger Bands'}
        
        return signals
    
    def calculate_prediction(self, data, signals):
        """Calculate next-day prediction using signal ensemble"""
        print("🔮 Generating prediction...")
        
        latest = data.iloc[-1]
        current_price = latest['Close']
        
        # Weight each signal
        weights = {
            'trend': 0.25,
            'momentum': 0.20,
            'macd': 0.15,
            'volatility': 0.10,
            'volume': 0.15,
            'bollinger': 0.15
        }
        
        # Calculate weighted signal
        total_signal = sum(signals[key]['strength'] * weights[key] for key in weights.keys())
        
        # Recent momentum
        recent_return = data['Daily_Return'].tail(5).mean()
        
        # Combine signals with recent momentum
        expected_return = 0.6 * total_signal * 0.02 + 0.4 * recent_return
        
        # Cap extreme predictions
        expected_return = np.clip(expected_return, -0.05, 0.05)  # Max ±5% per day
        
        # Calculate prediction
        predicted_price = current_price * (1 + expected_return)
        
        # Confidence interval using recent volatility
        volatility = data['Daily_Return'].tail(20).std()
        lower_bound = current_price * (1 + expected_return - 1.65 * volatility)
        upper_bound = current_price * (1 + expected_return + 1.65 * volatility)
        
        return predicted_price, lower_bound, upper_bound, expected_return, total_signal
    
    def display_comprehensive_report(self, data, current_price, predicted_price, 
                                     lower_bound, upper_bound, expected_return, 
                                     total_signal, signals):
        """Display detailed prediction report"""
        
        print("\n" + "="*80)
        print(" " * 20 + f"📊 PRODUCTION PREDICTION REPORT - {self.ticker}")
        print("="*80)
        
        # Current status
        print(f"\n📅 Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S IST')}")
        print(f"💰 Current Price: ${current_price:.2f}")
        
        # Market context
        latest = data.iloc[-1]
        week_change = ((latest['Close'] - data['Close'].iloc[-5]) / data['Close'].iloc[-5]) * 100
        month_change = ((latest['Close'] - data['Close'].iloc[-20]) / data['Close'].iloc[-20]) * 100
        
        print(f"\n📈 Recent Performance:")
        print(f"   Last 5 days: {week_change:+.2f}%")
        print(f"   Last 20 days: {month_change:+.2f}%")
        print(f"   Current RSI: {latest['RSI']:.1f}")
        print(f"   20-day Volatility: {latest['Volatility_20']*100:.2f}%")
        
        # Prediction
        print(f"\n🔮 TOMORROW'S PREDICTION:")
        print(f"   Predicted Price: ${predicted_price:.2f}")
        
        change = predicted_price - current_price
        change_pct = expected_return * 100
        
        direction = "⬆️" if change > 0 else "⬇️"
        print(f"   Expected Change: {direction} ${abs(change):.2f} ({abs(change_pct):.2f}%)")
        
        # Confidence interval
        print(f"\n📊 CONFIDENCE INTERVAL (90%):")
        print(f"   Lower Bound: ${lower_bound:.2f} ({((lower_bound/current_price-1)*100):.2f}%)")
        print(f"   Predicted: ${predicted_price:.2f}")
        print(f"   Upper Bound: ${upper_bound:.2f} ({((upper_bound/current_price-1)*100):.2f}%)")
        
        # Signal breakdown
        print(f"\n🎯 SIGNAL ANALYSIS (Overall Signal: {total_signal:+.2f}):")
        for key, signal in signals.items():
            strength_bar = "█" * int(abs(signal['strength']) * 10)
            print(f"   {key.capitalize():12s} [{signal['signal']:10s}] {strength_bar:10s} {signal['description']}")
        
        # Recommendation
        print(f"\n💡 RECOMMENDATION:")
        if abs(change_pct) < 0.5:
            rec = "🟡 HOLD"
            reason = "Minimal expected movement - wait for clearer signals"
        elif change_pct > 2:
            rec = "🟢 STRONG BUY"
            reason = "Strong bullish signals across multiple indicators"
        elif change_pct > 0.5:
            rec = "🟢 BUY"
            reason = "Positive momentum with supportive technical indicators"
        elif change_pct < -2:
            rec = "🔴 STRONG SELL"
            reason = "Strong bearish signals - consider reducing position"
        elif change_pct < -0.5:
            rec = "🔴 SELL"
            reason = "Negative momentum - caution advised"
        else:
            rec = "🟡 HOLD"
            reason = "Mixed signals - maintain current position"
        
        print(f"   {rec}")
        print(f"   {reason}")
        
        # Confidence level
        signal_strength = abs(total_signal)
        if signal_strength > 0.5:
            confidence = "HIGH"
        elif signal_strength > 0.2:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"
        
        print(f"\n📌 PREDICTION CONFIDENCE: {confidence}")
        
        # Risk factors
        print(f"\n⚠️  RISK FACTORS:")
        if latest['Volatility_20'] > 0.03:
            print(f"   • High volatility detected - increased risk")
        if latest['RSI'] > 70 or latest['RSI'] < 30:
            print(f"   • Extreme RSI levels - potential reversal")
        if latest['Volume_Ratio'] < 0.5:
            print(f"   • Low volume - less reliable price action")
        
        print(f"\n⚠️  DISCLAIMER:")
        print(f"   This analysis uses 25+ technical indicators and statistical methods.")
        print(f"   Stock predictions are inherently uncertain - this is NOT financial advice.")
        print(f"   Always do your own research and consider your risk tolerance.")
        
        print("\n" + "="*80 + "\n")
    
    def run(self):
        """Execute complete prediction pipeline"""
        print("\n" + "="*80)
        print(" " * 20 + "🚀 PRODUCTION STOCK PREDICTION SYSTEM")
        print("="*80)
        
        try:
            # Fetch data
            data, current_price = self.fetch_latest_data(days=200)
            
            # Calculate indicators
            data = self.calculate_technical_indicators(data)
            
            # Generate signals
            signals = self.generate_signals(data)
            
            # Calculate prediction
            predicted_price, lower_bound, upper_bound, expected_return, total_signal = \
                self.calculate_prediction(data, signals)
            
            # Display report
            self.display_comprehensive_report(
                data, current_price, predicted_price, lower_bound, upper_bound,
                expected_return, total_signal, signals
            )
            
            # Save to CSV
            result = {
                'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'Ticker': self.ticker,
                'Current_Price': current_price,
                'Predicted_Price': predicted_price,
                'Expected_Change_Pct': expected_return * 100,
                'Lower_Bound': lower_bound,
                'Upper_Bound': upper_bound,
                'Signal_Strength': total_signal
            }
            
            df = pd.DataFrame([result])
            filename = f'outputs/production_predictions_{self.ticker}.csv'
            
            try:
                existing = pd.read_csv(filename)
                df = pd.concat([existing, df], ignore_index=True)
            except FileNotFoundError:
                pass
            
            df.to_csv(filename, index=False)
            print(f"💾 Results saved to: {filename}\n")
            
            return predicted_price, current_price
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    ticker = input("Enter stock ticker [TSLA]: ").upper() or 'TSLA'
    
    predictor = ProductionStockPredictor(ticker)
    predictor.run()
