"""
Main Execution Script - Run the complete stock prediction project
"""

import warnings
warnings.filterwarnings('ignore')
import os
import sys

# Add src to path so we can import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Now import will work
from trainer import StockPredictionTrainer
from visualizer import StockVisualizer
import pandas as pd

def main():
    """Main execution function"""
    
    print("\n" + "="*80)
    print(" " * 20 + "STOCK PRICE PREDICTION USING LSTM")
    print(" " * 25 + "Machine Learning Project")
    print("="*80 + "\n")
    
    # =====================================================================
    # CONFIGURATION
    # =====================================================================
    
    # Stock selection
    print("Popular stocks: TSLA, AAPL, GOOGL, MSFT, AMZN, META, NVDA")
    TICKER = input("Enter stock ticker: ").upper() or 'TSLA'
    
    print(f"\n✅ Selected stock: {TICKER}")
    
    # Data parameters
    SEQUENCE_LENGTH = 60
    START_DATE = '2018-01-01'
    TRAIN_SPLIT = 0.8
    
    # Model hyperparameters
    UNITS = [128, 64, 32]
    DROPOUT_RATE = 0.3
    LEARNING_RATE = 0.001
    EPOCHS = 50  # Increase to 100 for better results
    BATCH_SIZE = 32
    
    print("\n📋 Configuration:")
    print(f"   • Sequence Length: {SEQUENCE_LENGTH} days")
    print(f"   • Start Date: {START_DATE}")
    print(f"   • Train-Test Split: {int(TRAIN_SPLIT*100)}% - {int((1-TRAIN_SPLIT)*100)}%")
    print(f"   • LSTM Units: {UNITS}")
    print(f"   • Dropout Rate: {DROPOUT_RATE}")
    print(f"   • Learning Rate: {LEARNING_RATE}")
    print(f"   • Epochs: {EPOCHS}")
    print(f"   • Batch Size: {BATCH_SIZE}\n")
    
    input("⏎ Press Enter to start training...")
    
    # =====================================================================
    # STEP 1: TRAINING
    # =====================================================================
    
    print("\n🚀 Starting training pipeline...\n")
    
    trainer = StockPredictionTrainer(
        ticker=TICKER,
        sequence_length=SEQUENCE_LENGTH,
        start_date=START_DATE
    )
    
    predictions, actuals = trainer.run_complete_pipeline(
        train_split=TRAIN_SPLIT,
        units=UNITS,
        dropout_rate=DROPOUT_RATE,
        learning_rate=LEARNING_RATE,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE
    )
    
    # =====================================================================
    # STEP 2: VISUALIZATION
    # =====================================================================
    
    print("\n🎨 Creating visualizations...\n")
    
    # Load data
    data = pd.read_csv(f'data/{TICKER}_stock_data.csv', index_col='Date', parse_dates=True)
    
    # Create visualizer
    visualizer = StockVisualizer(ticker=TICKER)
    
    # Create comprehensive report
    visualizer.create_comprehensive_report(data, actuals, predictions)
    
    # =====================================================================
    # COMPLETION
    # =====================================================================
    
    print("\n" + "="*80)
    print("🎉 PROJECT COMPLETED SUCCESSFULLY!")
    print("="*80 + "\n")
    
    print("📂 Generated Files:")
    print(f"   ✓ Model: models/{TICKER}_lstm_model.keras")
    print(f"   ✓ Scaler: models/scaler.pkl")
    print(f"   ✓ Data: data/{TICKER}_stock_data.csv")
    print(f"   ✓ Predictions: outputs/{TICKER}_predictions.csv")
    print(f"   ✓ Training History: outputs/training_history.png")
    print(f"   ✓ Predictions Plot: outputs/predictions_comparison.png")
    print(f"   ✓ Error Analysis: outputs/error_distribution.png")
    print(f"   ✓ Interactive Chart: outputs/interactive_chart.html")
    
    print("\n💡 Next Steps:")
    print("   • Open outputs/interactive_chart.html in your browser")
    print("   • Review all plots in the outputs folder")
    print("   • Try different stocks by running main.py again")
    print("   • Experiment with hyperparameters for better results\n")


if __name__ == "__main__":
    main()
