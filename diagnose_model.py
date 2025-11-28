"""
Diagnostic tool - Check if current price is within training range
"""

import pandas as pd
import pickle
import numpy as np

def diagnose_model(ticker='TSLA'):
    """Check model training data distribution"""
    
    print("\n" + "="*70)
    print(f"🔍 MODEL DIAGNOSTIC - {ticker}")
    print("="*70 + "\n")
    
    # Load training data
    try:
        data = pd.read_csv(f'data/{ticker}_stock_data.csv', index_col='Date', parse_dates=True)
        print(f"✅ Loaded training data: {len(data)} days")
        print(f"📅 Date range: {data.index[0].date()} to {data.index[-1].date()}")
        
        # Analyze price distribution
        close_prices = data['Close'].values
        
        print(f"\n📊 TRAINING DATA STATISTICS:")
        print(f"   Min Price: ${np.min(close_prices):.2f}")
        print(f"   Max Price: ${np.max(close_prices):.2f}")
        print(f"   Mean Price: ${np.mean(close_prices):.2f}")
        print(f"   Median Price: ${np.median(close_prices):.2f}")
        print(f"   Std Dev: ${np.std(close_prices):.2f}")
        
        # Check recent vs historical
        recent_prices = close_prices[-30:]
        historical_prices = close_prices[:-30]
        
        print(f"\n📈 RECENT (Last 30 Days) vs HISTORICAL:")
        print(f"   Recent Mean: ${np.mean(recent_prices):.2f}")
        print(f"   Historical Mean: ${np.mean(historical_prices):.2f}")
        print(f"   Difference: {((np.mean(recent_prices)/np.mean(historical_prices))-1)*100:.1f}%")
        
        # Current price
        current_price = close_prices[-1]
        print(f"\n💰 CURRENT PRICE: ${current_price:.2f}")
        
        # Check percentile
        percentile = (np.sum(close_prices < current_price) / len(close_prices)) * 100
        print(f"   Percentile in training data: {percentile:.1f}%")
        
        if percentile > 95:
            print(f"   ⚠️  WARNING: Current price is in top 5% of training data!")
            print(f"   This explains unrealistic predictions (extrapolation issue)")
        elif percentile > 90:
            print(f"   ⚠️  CAUTION: Current price is in top 10% of training data")
        else:
            print(f"   ✅ Current price is within normal training range")
        
        # Load scaler
        with open('models/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        
        print(f"\n🔧 SCALER INFORMATION:")
        print(f"   Scaler Min (Close): {scaler.data_min_[0]:.2f}")
        print(f"   Scaler Max (Close): {scaler.data_max_[0]:.2f}")
        print(f"   Scaler Range: ${scaler.data_max_[0] - scaler.data_min_[0]:.2f}")
        
        # Normalized value
        normalized_current = (current_price - scaler.data_min_[0]) / (scaler.data_max_[0] - scaler.data_min_[0])
        print(f"   Current price normalized: {normalized_current:.4f}")
        
        if normalized_current > 0.95:
            print(f"   ⚠️  Current price near upper bound of scaler!")
        
        print("\n" + "="*70)
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        
        if percentile > 90:
            print(f"   1. ⭐ Use percentage returns instead of absolute prices")
            print(f"   2. Consider StandardScaler instead of MinMaxScaler")
            print(f"   3. Retrain with more recent high-price data")
            print(f"   4. Use ensemble with technical indicators")
        else:
            print(f"   ✅ Model should work well - training range is appropriate")
        
        print("\n")
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("Make sure you've trained the model first!")

if __name__ == "__main__":
    import sys
    ticker = sys.argv[1] if len(sys.argv) > 1 else 'TSLA'
    diagnose_model(ticker)
