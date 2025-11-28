"""
Real-Time Stock Price Predictor - Predict tomorrow's price using live data
"""

import numpy as np
import pandas as pd
import pickle
from tensorflow import keras
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class RealtimeStockPredictor:
    def __init__(self, ticker, model_path, scaler_path, sequence_length=60):
        """
        Initialize real-time predictor
        
        Args:
            ticker (str): Stock ticker symbol
            model_path (str): Path to trained model
            scaler_path (str): Path to scaler
            sequence_length (int): Lookback period
        """
        self.ticker = ticker
        self.sequence_length = sequence_length
        
        print(f"\n🔄 Loading model for {ticker}...")
        
        # Load model and scaler
        try:
            self.model = keras.models.load_model(model_path)
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            print(f"✅ Model and scaler loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
        
        self.stock = yf.Ticker(ticker)
        
    def fetch_latest_data(self, days=200):  # Changed from 120 to 200
        """
        Fetch latest stock data from Yahoo Finance
    
        Args:
            days (int): Number of days to fetch (need extra for indicators)
        """
        print(f"\n📡 Fetching latest data for {self.ticker}...")
    
        try:
            # Get historical data - fetch more to account for indicator calculation
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
        
            data = self.stock.history(start=start_date, end=end_date)
        
            if data.empty:
                raise ValueError(f"No data available for {self.ticker}")
        
            # Get current/latest price using fast_info
            try:
                current_price = self.stock.fast_info.last_price
                print(f"💰 Current Price: ${current_price:.2f}")
            except:
                current_price = data['Close'].iloc[-1]
                print(f"💰 Last Close Price: ${current_price:.2f}")
        
            print(f"✅ Fetched {len(data)} days of raw data")
            print(f"📅 Date range: {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}")
        
            return data, current_price
        
        except Exception as e:
            print(f"❌ Error fetching data: {e}")
            raise
    
    def add_technical_indicators(self, data):
        """Add the same technical indicators used in training"""
        print("🔧 Calculating technical indicators...")
        
        df = data.copy()
        
        # Moving Averages
        df['MA_7'] = df['Close'].rolling(window=7).mean()
        df['MA_21'] = df['Close'].rolling(window=21).mean()
        df['MA_50'] = df['Close'].rolling(window=50).mean()
        
        # Exponential Moving Average
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
        
        # Volume indicators
        df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
        
        # Price differences
        df['High_Low_Diff'] = df['High'] - df['Low']
        df['Open_Close_Diff'] = df['Open'] - df['Close']
        
        # Drop NaN values
        df = df.dropna()
        
        print(f"✅ Technical indicators calculated")
        
        return df
    
    def prepare_prediction_data(self, data):
        """Prepare data for prediction"""
        print("📊 Preparing data for prediction...")
    
        # Feature columns (same as training)
        feature_columns = [
            'Close', 'Open', 'High', 'Low', 'Volume',
            'MA_7', 'MA_21', 'MA_50', 'EMA_12', 'EMA_26',
            'MACD', 'Signal_Line', 'RSI', 
            'BB_Middle', 'BB_Upper', 'BB_Lower',
            'Daily_Return', 'Volatility', 'Volume_MA',
            'High_Low_Diff', 'Open_Close_Diff'
        ]
    
        # Extract features
        features = data[feature_columns].values
    
        print(f"📏 Data after indicators: {len(features)} days")
    
        # Normalize
        normalized_features = self.scaler.transform(features)
    
        # Get last sequence
        if len(normalized_features) < self.sequence_length:
            raise ValueError(
                f"Not enough data after technical indicators. "
                f"Need {self.sequence_length} days, got {len(normalized_features)}. "
                f"Try running the model training again or check your data source."
            )
    
        last_sequence = normalized_features[-self.sequence_length:]
    
        print(f"✅ Data prepared - using last {self.sequence_length} days from {len(normalized_features)} available")
    
        return last_sequence, normalized_features

    def predict_next_day(self, last_sequence):
        """Predict next day's closing price"""
        print("\n🔮 Predicting tomorrow's price...")
        
        # Reshape for model input
        input_data = last_sequence.reshape(1, self.sequence_length, last_sequence.shape[1])
        
        # Make prediction
        prediction_normalized = self.model.predict(input_data, verbose=0)
        
        # Inverse transform to get actual price
        dummy = np.zeros((1, self.scaler.n_features_in_))
        dummy[0, 0] = prediction_normalized[0, 0]
        predicted_price = self.scaler.inverse_transform(dummy)[0, 0]
        
        return predicted_price
    
    def predict_multiple_days(self, last_sequence, n_days=7):
        """Predict multiple days ahead"""
        print(f"\n🔮 Predicting next {n_days} days...")
        
        predictions = []
        current_sequence = last_sequence.copy()
        
        for day in range(n_days):
            # Reshape for prediction
            input_data = current_sequence.reshape(1, self.sequence_length, current_sequence.shape[1])
            
            # Predict
            pred_normalized = self.model.predict(input_data, verbose=0)
            predictions.append(pred_normalized[0, 0])
            
            # Update sequence for next prediction
            new_row = current_sequence[-1].copy()
            new_row[0] = pred_normalized[0, 0]
            current_sequence = np.vstack([current_sequence[1:], new_row])
        
        # Inverse transform all predictions
        dummy = np.zeros((len(predictions), self.scaler.n_features_in_))
        dummy[:, 0] = predictions
        predicted_prices = self.scaler.inverse_transform(dummy)[:, 0]
        
        return predicted_prices
    
    def calculate_confidence_interval(self, last_sequence, n_simulations=100):
        """Calculate prediction confidence interval using Monte Carlo"""
        print("📈 Calculating confidence intervals...")
        
        predictions = []
        
        for _ in range(n_simulations):
            # Add small random noise to simulate uncertainty
            noisy_sequence = last_sequence + np.random.normal(0, 0.001, last_sequence.shape)
            input_data = noisy_sequence.reshape(1, self.sequence_length, noisy_sequence.shape[1])
            
            pred_normalized = self.model.predict(input_data, verbose=0)
            
            # Inverse transform
            dummy = np.zeros((1, self.scaler.n_features_in_))
            dummy[0, 0] = pred_normalized[0, 0]
            predicted_price = self.scaler.inverse_transform(dummy)[0, 0]
            
            predictions.append(predicted_price)
        
        predictions = np.array(predictions)
        
        # Calculate statistics
        mean_pred = np.mean(predictions)
        std_pred = np.std(predictions)
        lower_bound = np.percentile(predictions, 5)  # 5th percentile
        upper_bound = np.percentile(predictions, 95)  # 95th percentile
        
        return mean_pred, lower_bound, upper_bound, std_pred
    
    def get_recommendation(self, current_price, predicted_price):
        """Generate trading recommendation"""
        price_change = predicted_price - current_price
        price_change_pct = (price_change / current_price) * 100
        
        if price_change_pct > 2:
            recommendation = "🟢 STRONG BUY"
            reason = f"Expected to rise by ${price_change:.2f} ({price_change_pct:.2f}%)"
        elif price_change_pct > 0.5:
            recommendation = "🟢 BUY"
            reason = f"Expected to rise by ${price_change:.2f} ({price_change_pct:.2f}%)"
        elif price_change_pct > -0.5:
            recommendation = "🟡 HOLD"
            reason = f"Expected minimal change of ${price_change:.2f} ({price_change_pct:.2f}%)"
        elif price_change_pct > -2:
            recommendation = "🔴 SELL"
            reason = f"Expected to fall by ${abs(price_change):.2f} ({abs(price_change_pct):.2f}%)"
        else:
            recommendation = "🔴 STRONG SELL"
            reason = f"Expected to fall by ${abs(price_change):.2f} ({abs(price_change_pct):.2f}%)"
        
        return recommendation, reason
    
    def display_prediction_report(self, current_price, predicted_price, 
                                  lower_bound, upper_bound, future_predictions=None):
        """Display comprehensive prediction report"""
        
        print("\n" + "="*70)
        print(f"📊 REAL-TIME PREDICTION REPORT - {self.ticker}")
        print("="*70)
        
        # Current info
        print(f"\n📅 Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S IST')}")
        print(f"💰 Current Price: ${current_price:.2f}")
        
        # Tomorrow's prediction
        print(f"\n🔮 TOMORROW'S PREDICTION:")
        print(f"   Predicted Price: ${predicted_price:.2f}")
        
        price_change = predicted_price - current_price
        price_change_pct = (price_change / current_price) * 100
        
        direction = "⬆️" if price_change > 0 else "⬇️"
        print(f"   Expected Change: {direction} ${abs(price_change):.2f} ({abs(price_change_pct):.2f}%)")
        
        # Confidence interval
        print(f"\n📊 CONFIDENCE INTERVAL (90%):")
        print(f"   Lower Bound: ${lower_bound:.2f}")
        print(f"   Upper Bound: ${upper_bound:.2f}")
        print(f"   Range: ${upper_bound - lower_bound:.2f}")
        
        # Recommendation
        recommendation, reason = self.get_recommendation(current_price, predicted_price)
        print(f"\n💡 RECOMMENDATION: {recommendation}")
        print(f"   {reason}")
        
        # Future predictions
        if future_predictions is not None and len(future_predictions) > 1:
            print(f"\n📈 NEXT 7 DAYS FORECAST:")
            for i, price in enumerate(future_predictions, 1):
                change = price - current_price
                change_pct = (change / current_price) * 100
                direction = "⬆️" if change > 0 else "⬇️"
                future_date = (datetime.now() + timedelta(days=i)).strftime('%Y-%m-%d')
                print(f"   Day {i} ({future_date}): ${price:.2f} {direction} {abs(change_pct):.2f}%")
        
        # Risk warning
        print(f"\n⚠️  DISCLAIMER:")
        print(f"   This prediction is based on historical patterns and technical indicators.")
        print(f"   Stock markets are volatile. This is NOT financial advice.")
        print(f"   Always do your own research before making investment decisions.")
        
        print("\n" + "="*70 + "\n")
    
    def save_prediction(self, current_price, predicted_price, lower_bound, upper_bound):
        """Save prediction to CSV"""
        
        prediction_data = {
            'Timestamp': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
            'Ticker': [self.ticker],
            'Current_Price': [current_price],
            'Predicted_Price': [predicted_price],
            'Lower_Bound': [lower_bound],
            'Upper_Bound': [upper_bound],
            'Expected_Change': [predicted_price - current_price],
            'Expected_Change_Pct': [(predicted_price - current_price) / current_price * 100]
        }
        
        df = pd.DataFrame(prediction_data)
        
        # Append to file if exists, create new if not
        filename = f'outputs/realtime_predictions_{self.ticker}.csv'
        try:
            existing_df = pd.read_csv(filename)
            df = pd.concat([existing_df, df], ignore_index=True)
        except FileNotFoundError:
            pass
        
        df.to_csv(filename, index=False)
        print(f"💾 Prediction saved to {filename}")
    
    def run_realtime_prediction(self, save_results=True, predict_multiple=True):
        """Run complete real-time prediction pipeline"""
        
        print("\n" + "="*70)
        print(f"🚀 REAL-TIME PREDICTION PIPELINE - {self.ticker}")
        print("="*70)
        
        try:
            # Step 1: Fetch latest data
            data, current_price = self.fetch_latest_data(days=200)
            
            # Step 2: Add technical indicators
            data = self.add_technical_indicators(data)
            
            # Step 3: Prepare data
            last_sequence, _ = self.prepare_prediction_data(data)
            
            # Step 4: Predict tomorrow's price
            predicted_price = self.predict_next_day(last_sequence)
            
            # Step 5: Calculate confidence interval
            mean_pred, lower_bound, upper_bound, std = self.calculate_confidence_interval(last_sequence)
            
            # Use mean from confidence interval for better estimate
            predicted_price = mean_pred
            
            # Step 6: Predict multiple days (optional)
            future_predictions = None
            if predict_multiple:
                future_predictions = self.predict_multiple_days(last_sequence, n_days=7)
            
            # Step 7: Display report
            self.display_prediction_report(
                current_price, 
                predicted_price, 
                lower_bound, 
                upper_bound,
                future_predictions
            )
            
            # Step 8: Save results
            if save_results:
                self.save_prediction(current_price, predicted_price, lower_bound, upper_bound)
            
            return predicted_price, current_price, lower_bound, upper_bound
            
        except Exception as e:
            print(f"\n❌ Error in prediction pipeline: {e}")
            import traceback
            traceback.print_exc()
            raise


# Example usage
if __name__ == "__main__":
    ticker = 'TSLA'
    
    predictor = RealtimeStockPredictor(
        ticker=ticker,
        model_path=f'models/{ticker}_lstm_model.keras',
        scaler_path='models/scaler.pkl',
        sequence_length=60
    )
    
    predictor.run_realtime_prediction(save_results=True, predict_multiple=True)
