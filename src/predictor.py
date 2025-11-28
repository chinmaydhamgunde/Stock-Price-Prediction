"""
Predictor Module - Make future price predictions
"""

import numpy as np
import pandas as pd
import pickle
from tensorflow import keras
import matplotlib.pyplot as plt

class FuturePricePredictor:
    def __init__(self, ticker, model_path, scaler_path):
        """
        Initialize predictor
        
        Args:
            ticker (str): Stock ticker symbol
            model_path (str): Path to saved model
            scaler_path (str): Path to saved scaler
        """
        self.ticker = ticker
        self.model = keras.models.load_model(model_path)
        
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        
        print(f"✅ Model and scaler loaded for {ticker}")
        
    def predict_next_days(self, last_sequence, n_days=30):
        """
        Predict next N days
        
        Args:
            last_sequence (array): Last sequence of normalized data
            n_days (int): Number of days to predict
            
        Returns:
            array: Predicted prices
        """
        print(f"🔮 Predicting next {n_days} days...")
        
        predictions = []
        current_sequence = last_sequence.copy()
        
        for i in range(n_days):
            # Reshape for prediction
            current_input = current_sequence.reshape(1, current_sequence.shape[0], current_sequence.shape[1])
            
            # Predict next value
            next_pred = self.model.predict(current_input, verbose=0)
            
            # Store prediction
            predictions.append(next_pred[0, 0])
            
            # Update sequence
            # Create new row with predicted value
            new_row = current_sequence[-1].copy()
            new_row[0] = next_pred[0, 0]  # Update Close price
            
            # Append new row and remove first row
            current_sequence = np.vstack([current_sequence[1:], new_row])
            
            if (i + 1) % 10 == 0:
                print(f"   Predicted {i + 1}/{n_days} days...")
        
        # Inverse transform predictions
        dummy = np.zeros((len(predictions), self.scaler.n_features_in_))
        dummy[:, 0] = predictions
        predictions_original = self.scaler.inverse_transform(dummy)[:, 0]
        
        print(f"✅ Prediction complete!")
        
        return predictions_original
    
    def plot_future_predictions(self, historical_prices, future_predictions, 
                               save_path='outputs/future_predictions.png'):
        """
        Plot historical and future predicted prices
        
        Args:
            historical_prices (array): Historical prices
            future_predictions (array): Future predicted prices
            save_path (str): Path to save plot
        """
        print("📊 Creating future predictions plot...")
        
        plt.figure(figsize=(16, 8))
        
        # Plot historical prices
        hist_days = range(len(historical_prices))
        plt.plot(hist_days, historical_prices, label='Historical Prices', 
                color='#2E86AB', linewidth=2, alpha=0.8)
        
        # Plot future predictions
        future_days = range(len(historical_prices), len(historical_prices) + len(future_predictions))
        plt.plot(future_days, future_predictions, label='Future Predictions', 
                color='#A23B72', linewidth=2, alpha=0.8, linestyle='--')
        
        # Add connection line
        plt.plot([hist_days[-1], future_days[0]], 
                [historical_prices[-1], future_predictions[0]], 
                color='gray', linewidth=2, alpha=0.5)
        
        # Shade future prediction area
        plt.axvspan(len(historical_prices), len(historical_prices) + len(future_predictions), 
                   alpha=0.2, color='purple', label='Prediction Zone')
        
        plt.title(f'{self.ticker} Stock Price: Historical & Future Predictions', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Days', fontsize=12)
        plt.ylabel('Stock Price ($)', fontsize=12)
        plt.legend(loc='upper left', fontsize=11)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Plot saved to {save_path}")
        plt.show()
        
    def save_predictions_to_csv(self, future_predictions, save_path='outputs/future_predictions.csv'):
        """
        Save future predictions to CSV
        
        Args:
            future_predictions (array): Future predicted prices
            save_path (str): Path to save CSV
        """
        from datetime import datetime, timedelta
        
        # Generate future dates
        today = datetime.now()
        future_dates = [today + timedelta(days=i+1) for i in range(len(future_predictions))]
        
        # Create DataFrame
        predictions_df = pd.DataFrame({
            'Date': future_dates,
            'Predicted_Price': future_predictions
        })
        
        predictions_df.to_csv(save_path, index=False)
        print(f"💾 Future predictions saved to {save_path}")
        
        return predictions_df


# Example usage
if __name__ == "__main__":
    ticker = 'TSLA'
    
    # Load preprocessed data to get last sequence
    data = pd.read_csv(f'data/{ticker}_stock_data.csv', index_col='Date', parse_dates=True)
    
    # You need to preprocess this data same way as training
    # For simplicity, loading the normalized last sequence from training
    # In practice, you would run preprocessor again
    
    # Initialize predictor
    predictor = FuturePricePredictor(
        ticker=ticker,
        model_path=f'models/{ticker}_lstm_model.keras',
        scaler_path='models/scaler.pkl'
    )
    
    print("\n📝 Note: You need to provide the last_sequence from your preprocessed data")
    print("   Run the complete pipeline first to generate this data\n")
