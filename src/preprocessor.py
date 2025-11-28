"""
Data Preprocessor - Handles data cleaning, feature engineering, and normalization
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import pickle

class StockDataPreprocessor:
    def __init__(self, data, sequence_length=60):
        """
        Initialize preprocessor
        
        Args:
            data (pd.DataFrame): Stock data
            sequence_length (int): Number of days to look back for predictions
        """
        self.data = data.copy()
        self.sequence_length = sequence_length
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.feature_columns = None
        
    def add_technical_indicators(self):
        """Add technical indicators as features"""
        print("🔧 Adding technical indicators...")
        
        df = self.data
        
        # Moving Averages
        df['MA_7'] = df['Close'].rolling(window=7).mean()
        df['MA_21'] = df['Close'].rolling(window=21).mean()
        df['MA_50'] = df['Close'].rolling(window=50).mean()
        
        # Exponential Moving Average
        df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
        
        # MACD (Moving Average Convergence Divergence)
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # RSI (Relative Strength Index)
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
        
        self.data = df
        print("✅ Technical indicators added")
        
    def clean_data(self):
        """Clean data by handling missing values"""
        print("🧹 Cleaning data...")
        
        # Drop rows with NaN values (from indicator calculations)
        initial_rows = len(self.data)
        self.data = self.data.dropna()
        dropped_rows = initial_rows - len(self.data)
        
        print(f"✅ Cleaned data - Dropped {dropped_rows} rows with missing values")
        print(f"📊 Remaining rows: {len(self.data)}")
        
    def prepare_features(self, target_column='Close', additional_features=None):
        """
        Prepare feature set for training
        
        Args:
            target_column (str): Column to predict
            additional_features (list): Additional columns to use as features
        """
        print("🎯 Preparing features...")
        
        if additional_features is None:
            # Use all technical indicators
            self.feature_columns = [
                'Close', 'Open', 'High', 'Low', 'Volume',
                'MA_7', 'MA_21', 'MA_50', 'EMA_12', 'EMA_26',
                'MACD', 'Signal_Line', 'RSI', 
                'BB_Middle', 'BB_Upper', 'BB_Lower',
                'Daily_Return', 'Volatility', 'Volume_MA',
                'High_Low_Diff', 'Open_Close_Diff'
            ]
        else:
            self.feature_columns = additional_features
        
        # Extract features
        self.features = self.data[self.feature_columns].values
        
        print(f"✅ Feature set prepared with {len(self.feature_columns)} features")
        print(f"📋 Features: {self.feature_columns}")
        
    def normalize_data(self):
        """Normalize features using MinMaxScaler"""
        print("📏 Normalizing data...")
        
        self.normalized_features = self.scaler.fit_transform(self.features)
        
        print("✅ Data normalized to range [0, 1]")
        
    def create_sequences(self, train_split=0.8):
        """
        Create sequences for LSTM training
        
        Args:
            train_split (float): Ratio of training data
            
        Returns:
            tuple: (X_train, y_train, X_test, y_test)
        """
        print(f"🔄 Creating sequences with lookback = {self.sequence_length} days...")
        
        X, y = [], []
        
        for i in range(self.sequence_length, len(self.normalized_features)):
            X.append(self.normalized_features[i-self.sequence_length:i])
            y.append(self.normalized_features[i, 0])  # Predict 'Close' price
        
        X, y = np.array(X), np.array(y)
        
        # Split into train and test
        split_idx = int(len(X) * train_split)
        
        X_train = X[:split_idx]
        y_train = y[:split_idx]
        X_test = X[split_idx:]
        y_test = y[split_idx:]
        
        print(f"✅ Sequences created")
        print(f"📊 Training samples: {len(X_train)}")
        print(f"📊 Testing samples: {len(X_test)}")
        print(f"📐 Sequence shape: {X_train.shape}")
        
        return X_train, y_train, X_test, y_test
    
    def save_scaler(self, filename='models/scaler.pkl'):
        """Save the scaler for later use"""
        import os
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"💾 Scaler saved to {filename}")
    
    def inverse_transform(self, predictions):
        """Convert normalized predictions back to original scale"""
        # Create dummy array with same shape as original features
        dummy = np.zeros((len(predictions), len(self.feature_columns)))
        dummy[:, 0] = predictions.flatten()
        
        # Inverse transform
        original_scale = self.scaler.inverse_transform(dummy)
        
        return original_scale[:, 0]


# Example usage
if __name__ == "__main__":
    import pandas as pd
    
    # Load data
    data = pd.read_csv('data/TSLA_stock_data.csv', index_col='Date', parse_dates=True)
    
    # Initialize preprocessor
    preprocessor = StockDataPreprocessor(data, sequence_length=60)
    
    # Add technical indicators
    preprocessor.add_technical_indicators()
    
    # Clean data
    preprocessor.clean_data()
    
    # Prepare features
    preprocessor.prepare_features()
    
    # Normalize
    preprocessor.normalize_data()
    
    # Create sequences
    X_train, y_train, X_test, y_test = preprocessor.create_sequences(train_split=0.8)
    
    # Save scaler
    preprocessor.save_scaler()
    
    print("\n✅ Preprocessing complete!")
