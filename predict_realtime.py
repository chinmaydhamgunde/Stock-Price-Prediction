"""
Real-Time Stock Price Prediction CLI Tool
Run this script to get live predictions for any stock
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from realtime_predictor import RealtimeStockPredictor
import warnings
warnings.filterwarnings('ignore')

def main():
    """Main function for real-time prediction"""
    
    print("\n" + "="*80)
    print(" " * 25 + "🔮 REAL-TIME STOCK PREDICTOR")
    print(" " * 20 + "AI-Powered Next-Day Price Prediction")
    print("="*80 + "\n")
    
    # Get ticker from user
    print("📊 Available trained models:")
    
    # Check for trained models
    models_dir = 'models'
    if os.path.exists(models_dir):
        model_files = [f for f in os.listdir(models_dir) if f.endswith('_lstm_model.keras')]
        if model_files:
            trained_tickers = [f.replace('_lstm_model.keras', '') for f in model_files]
            print(f"   Trained: {', '.join(trained_tickers)}")
        else:
            print("   No trained models found. Run main.py first to train a model.")
            return
    
    ticker = input("\nEnter stock ticker for prediction: ").upper() or 'TSLA'
    
    # Check if model exists
    model_path = f'models/{ticker}_lstm_model.keras'
    if not os.path.exists(model_path):
        print(f"\n❌ No trained model found for {ticker}")
        print(f"   Please train the model first by running: python main.py")
        return
    
    print(f"\n✅ Selected: {ticker}")
    
    # Ask for prediction options
    print("\n📋 Prediction Options:")
    print("   1. Tomorrow only (fast)")
    print("   2. Next 7 days (detailed)")
    
    choice = input("\nSelect option (1-2) [default: 2]: ").strip() or '2'
    predict_multiple = (choice == '2')
    
    print("\n" + "="*80)
    
    try:
        # Initialize predictor
        predictor = RealtimeStockPredictor(
            ticker=ticker,
            model_path=model_path,
            scaler_path='models/scaler.pkl',
            sequence_length=60
        )
        
        # Run prediction
        predicted_price, current_price, lower_bound, upper_bound = predictor.run_realtime_prediction(
            save_results=True,
            predict_multiple=predict_multiple
        )
        
        print("✅ Prediction complete!")
        print(f"\n📁 Results saved to: outputs/realtime_predictions_{ticker}.csv")
        
        # Ask to predict another stock
        another = input("\n🔄 Predict another stock? (y/n) [default: n]: ").lower()
        if another == 'y':
            main()  # Recursive call
        else:
            print("\n👋 Thanks for using Real-Time Stock Predictor!")
            print("💡 Tip: Run this script anytime to get fresh predictions!\n")
        
    except Exception as e:
        print(f"\n❌ Prediction failed: {e}")
        print("\n💡 Troubleshooting:")
        print("   1. Make sure you have an internet connection (for live data)")
        print("   2. Verify the model was trained for this stock")
        print("   3. Check if the stock ticker is valid")


if __name__ == "__main__":
    main()
