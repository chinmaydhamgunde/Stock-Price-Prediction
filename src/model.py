"""
LSTM Model - Build, train, and evaluate stock prediction model
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import math

class StockPriceLSTM:
    def __init__(self, sequence_length, n_features):
        """
        Initialize LSTM model
        
        Args:
            sequence_length (int): Number of time steps
            n_features (int): Number of features
        """
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.model = None
        self.history = None
        
    def build_model(self, units=[128, 64, 32], dropout_rate=0.3, learning_rate=0.001):
        """
        Build LSTM model with multiple layers
        
        Args:
            units (list): Number of units in each LSTM layer
            dropout_rate (float): Dropout rate for regularization
            learning_rate (float): Learning rate for optimizer
        """
        print("🏗️ Building LSTM model...")
        
        self.model = Sequential()
        
        # First Bidirectional LSTM layer
        self.model.add(Bidirectional(
            LSTM(units[0], return_sequences=True, activation='tanh'),
            input_shape=(self.sequence_length, self.n_features)
        ))
        self.model.add(Dropout(dropout_rate))
        self.model.add(BatchNormalization())
        
        # Second LSTM layer
        self.model.add(LSTM(units[1], return_sequences=True, activation='tanh'))
        self.model.add(Dropout(dropout_rate))
        self.model.add(BatchNormalization())
        
        # Third LSTM layer
        self.model.add(LSTM(units[2], return_sequences=False, activation='tanh'))
        self.model.add(Dropout(dropout_rate))
        self.model.add(BatchNormalization())
        
        # Dense layers
        self.model.add(Dense(32, activation='relu'))
        self.model.add(Dropout(dropout_rate * 0.5))
        
        self.model.add(Dense(16, activation='relu'))
        
        # Output layer
        self.model.add(Dense(1))
        
        # Compile model with Adam optimizer
        optimizer = Adam(learning_rate=learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss='mean_squared_error',
            metrics=['mae', 'mse']
        )
        
        print("✅ Model built successfully!")
        print(f"\n{self.model.summary()}\n")
        
    def train_model(self, X_train, y_train, epochs=100, batch_size=32, validation_split=0.1):
        """
        Train the LSTM model
        
        Args:
            X_train: Training features
            y_train: Training targets
            epochs (int): Number of training epochs
            batch_size (int): Batch size
            validation_split (float): Validation data ratio
        """
        print(f"🚀 Training model for {epochs} epochs...")
        
        # Callbacks
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        )
        
        checkpoint = ModelCheckpoint(
            'models/best_model.keras',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.00001,
            verbose=1
        )
        
        # Train model
        self.history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stop, checkpoint, reduce_lr],
            verbose=1
        )
        
        print("✅ Training complete!")
        
    def plot_training_history(self):
        """Plot training and validation loss"""
        if self.history is None:
            print("❌ No training history available")
            return
        
        plt.figure(figsize=(14, 5))
        
        # Loss plot
        plt.subplot(1, 2, 1)
        plt.plot(self.history.history['loss'], label='Training Loss', linewidth=2)
        plt.plot(self.history.history['val_loss'], label='Validation Loss', linewidth=2)
        plt.title('Model Loss During Training', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (MSE)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # MAE plot
        plt.subplot(1, 2, 2)
        plt.plot(self.history.history['mae'], label='Training MAE', linewidth=2)
        plt.plot(self.history.history['val_mae'], label='Validation MAE', linewidth=2)
        plt.title('Mean Absolute Error During Training', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('outputs/training_history.png', dpi=300, bbox_inches='tight')
        print("💾 Training history plot saved to outputs/training_history.png")
        plt.show()
        
    def evaluate_model(self, X_test, y_test, scaler):
        """
        Evaluate model performance
        
        Args:
            X_test: Test features
            y_test: Test targets
            scaler: Scaler object for inverse transformation
        """
        print("📊 Evaluating model...")
        
        # Make predictions
        predictions = self.model.predict(X_test)
        
        # Calculate metrics on normalized data
        mse_norm = mean_squared_error(y_test, predictions)
        rmse_norm = math.sqrt(mse_norm)
        mae_norm = mean_absolute_error(y_test, predictions)
        
        # Inverse transform to original scale
        y_test_original = self._inverse_transform_single(y_test, scaler)
        predictions_original = self._inverse_transform_single(predictions.flatten(), scaler)
        
        # Calculate metrics on original scale
        mse = mean_squared_error(y_test_original, predictions_original)
        rmse = math.sqrt(mse)
        mae = mean_absolute_error(y_test_original, predictions_original)
        r2 = r2_score(y_test_original, predictions_original)
        
        # Calculate MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((y_test_original - predictions_original) / y_test_original)) * 100
        
        print("\n" + "="*60)
        print("📈 MODEL PERFORMANCE METRICS")
        print("="*60)
        print(f"🎯 Root Mean Squared Error (RMSE): ${rmse:.2f}")
        print(f"📏 Mean Absolute Error (MAE): ${mae:.2f}")
        print(f"📊 R² Score: {r2:.4f}")
        print(f"📉 Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
        print("="*60 + "\n")
        
        return predictions_original, y_test_original
        
    def _inverse_transform_single(self, data, scaler):
        """Helper function to inverse transform single feature"""
        dummy = np.zeros((len(data), scaler.n_features_in_))
        dummy[:, 0] = data.flatten()
        original = scaler.inverse_transform(dummy)
        return original[:, 0]
    
    def save_model(self, filename='models/lstm_model.keras'):
        """Save the trained model"""
        self.model.save(filename)
        print(f"💾 Model saved to {filename}")
    
    def load_model(self, filename='models/lstm_model.keras'):
        """Load a trained model"""
        self.model = keras.models.load_model(filename)
        print(f"📂 Model loaded from {filename}")
