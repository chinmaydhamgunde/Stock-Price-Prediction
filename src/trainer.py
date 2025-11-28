"""
Training Pipeline - Complete training workflow
"""

import numpy as np
import pandas as pd
import os
import sys

# Add src directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Use absolute imports (no dots)
from data_loader import StockDataLoader
from preprocessor import StockDataPreprocessor
from model import StockPriceLSTM

import warnings
warnings.filterwarnings('ignore')

class StockPredictionTrainer:
    def __init__(self, ticker='TSLA', sequence_length=60, start_date='2018-01-01'):
        """
        Initialize trainer
        
        Args:
            ticker (str): Stock ticker symbol
            sequence_length (int): Number of days to look back
            start_date (str): Start date for data collection
        """
        self.ticker = ticker
        self.sequence_length = sequence_length
        self.start_date = start_date
        self.preprocessor = None
        self.model = None
        
        # Create necessary directories
        os.makedirs('data', exist_ok=True)
        os.makedirs('models', exist_ok=True)
        os.makedirs('outputs', exist_ok=True)
        
    def load_and_prepare_data(self):
        """Load and prepare data for training"""
        print("\n" + "="*70)
        print("📊 STEP 1: DATA LOADING AND PREPARATION")
        print("="*70 + "\n")
        
        # Load data
        loader = StockDataLoader(ticker=self.ticker, start_date=self.start_date)
        data = loader.download_data()
        
        if data is None:
            raise ValueError("Failed to download data")
        
        # Save data
        loader.save_data()
        loader.get_basic_info()
        
        # Initialize preprocessor
        self.preprocessor = StockDataPreprocessor(data, sequence_length=self.sequence_length)
        
        # Add technical indicators
        self.preprocessor.add_technical_indicators()
        
        # Clean data
        self.preprocessor.clean_data()
        
        # Prepare features
        self.preprocessor.prepare_features()
        
        # Normalize data
        self.preprocessor.normalize_data()
        
        # Save scaler
        self.preprocessor.save_scaler()
        
        print("\n✅ Data preparation complete!\n")
        
    def create_train_test_split(self, train_split=0.8):
        """Create training and testing sequences"""
        print("\n" + "="*70)
        print("📊 STEP 2: CREATING TRAIN-TEST SPLIT")
        print("="*70 + "\n")
        
        self.X_train, self.y_train, self.X_test, self.y_test = \
            self.preprocessor.create_sequences(train_split=train_split)
        
        print(f"\n✅ Train-test split complete!")
        print(f"📐 Training data shape: {self.X_train.shape}")
        print(f"📐 Testing data shape: {self.X_test.shape}\n")
        
    def build_and_train_model(self, units=[128, 64, 32], dropout_rate=0.3, 
                              learning_rate=0.001, epochs=100, batch_size=32):
        """Build and train the LSTM model"""
        print("\n" + "="*70)
        print("🧠 STEP 3: BUILDING AND TRAINING MODEL")
        print("="*70 + "\n")
        
        # Initialize model
        n_features = self.X_train.shape[2]
        self.model = StockPriceLSTM(
            sequence_length=self.sequence_length,
            n_features=n_features
        )
        
        # Build model
        self.model.build_model(
            units=units,
            dropout_rate=dropout_rate,
            learning_rate=learning_rate
        )
        
        # Train model
        self.model.train_model(
            self.X_train,
            self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1
        )
        
        # Plot training history
        self.model.plot_training_history()
        
        # Save model
        self.model.save_model(f'models/{self.ticker}_lstm_model.keras')
        
        print("\n✅ Model training complete!\n")
        
    def evaluate_and_save_results(self):
        """Evaluate model and save results"""
        print("\n" + "="*70)
        print("📈 STEP 4: MODEL EVALUATION")
        print("="*70 + "\n")
        
        # Evaluate model
        predictions, actuals = self.model.evaluate_model(
            self.X_test,
            self.y_test,
            self.preprocessor.scaler
        )
        
        # Save predictions to CSV
        results_df = pd.DataFrame({
            'Actual_Price': actuals,
            'Predicted_Price': predictions,
            'Difference': actuals - predictions,
            'Percentage_Error': np.abs((actuals - predictions) / actuals) * 100
        })
        
        results_df.to_csv(f'outputs/{self.ticker}_predictions.csv', index=False)
        print(f"💾 Predictions saved to outputs/{self.ticker}_predictions.csv\n")
        
        return predictions, actuals
    
    def run_complete_pipeline(self, train_split=0.8, units=[128, 64, 32], 
                             dropout_rate=0.3, learning_rate=0.001, 
                             epochs=100, batch_size=32):
        """Run the complete training pipeline"""
        print("\n" + "="*70)
        print(f"🚀 STARTING COMPLETE TRAINING PIPELINE FOR {self.ticker}")
        print("="*70 + "\n")
        
        try:
            # Step 1: Load and prepare data
            self.load_and_prepare_data()
            
            # Step 2: Create train-test split
            self.create_train_test_split(train_split=train_split)
            
            # Step 3: Build and train model
            self.build_and_train_model(
                units=units,
                dropout_rate=dropout_rate,
                learning_rate=learning_rate,
                epochs=epochs,
                batch_size=batch_size
            )
            
            # Step 4: Evaluate and save results
            predictions, actuals = self.evaluate_and_save_results()
            
            print("\n" + "="*70)
            print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
            print("="*70 + "\n")
            
            print("📁 Output Files:")
            print(f"   • Model: models/{self.ticker}_lstm_model.keras")
            print(f"   • Scaler: models/scaler.pkl")
            print(f"   • Data: data/{self.ticker}_stock_data.csv")
            print(f"   • Predictions: outputs/{self.ticker}_predictions.csv")
            print(f"   • Training Plot: outputs/training_history.png\n")
            
            return predictions, actuals
            
        except Exception as e:
            print(f"\n❌ ERROR: {str(e)}")
            import traceback
            traceback.print_exc()
            raise


# Example usage
if __name__ == "__main__":
    # Configuration
    TICKER = 'TSLA'
    SEQUENCE_LENGTH = 60
    START_DATE = '2018-01-01'
    
    # Hyperparameters
    TRAIN_SPLIT = 0.8
    UNITS = [128, 64, 32]
    DROPOUT_RATE = 0.3
    LEARNING_RATE = 0.001
    EPOCHS = 50
    BATCH_SIZE = 32
    
    # Initialize trainer
    trainer = StockPredictionTrainer(
        ticker=TICKER,
        sequence_length=SEQUENCE_LENGTH,
        start_date=START_DATE
    )
    
    # Run complete pipeline
    predictions, actuals = trainer.run_complete_pipeline(
        train_split=TRAIN_SPLIT,
        units=UNITS,
        dropout_rate=DROPOUT_RATE,
        learning_rate=LEARNING_RATE,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE
    )
