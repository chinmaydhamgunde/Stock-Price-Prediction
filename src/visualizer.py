"""
Visualization Module - Create beautiful and interactive visualizations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (16, 8)
plt.rcParams['font.size'] = 11

class StockVisualizer:
    def __init__(self, ticker):
        """
        Initialize visualizer
        
        Args:
            ticker (str): Stock ticker symbol
        """
        self.ticker = ticker
        
    def plot_predictions_vs_actual(self, actuals, predictions, save_path='outputs/predictions_comparison.png'):
        """
        Plot actual vs predicted prices with matplotlib
        
        Args:
            actuals (array): Actual prices
            predictions (array): Predicted prices
            save_path (str): Path to save the plot
        """
        print("📊 Creating predictions comparison plot...")
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
        
        # Plot 1: Actual vs Predicted
        ax1.plot(actuals, label='Actual Price', color='#2E86AB', linewidth=2, alpha=0.8)
        ax1.plot(predictions, label='Predicted Price', color='#A23B72', linewidth=2, alpha=0.8)
        ax1.fill_between(range(len(actuals)), actuals, predictions, alpha=0.2, color='gray')
        ax1.set_title(f'{self.ticker} Stock Price: Actual vs Predicted', 
                     fontsize=16, fontweight='bold', pad=20)
        ax1.set_xlabel('Time Steps (Days)', fontsize=12)
        ax1.set_ylabel('Stock Price ($)', fontsize=12)
        ax1.legend(loc='upper left', fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Prediction Error
        error = actuals - predictions
        ax2.plot(error, color='#F18F01', linewidth=2, alpha=0.8)
        ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)
        ax2.fill_between(range(len(error)), error, 0, alpha=0.3, color='#F18F01')
        ax2.set_title('Prediction Error Over Time', fontsize=16, fontweight='bold', pad=20)
        ax2.set_xlabel('Time Steps (Days)', fontsize=12)
        ax2.set_ylabel('Error ($)', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Plot saved to {save_path}")
        plt.show()
        
    def plot_error_distribution(self, actuals, predictions, save_path='outputs/error_distribution.png'):
        """
        Plot error distribution and analysis
        
        Args:
            actuals (array): Actual prices
            predictions (array): Predicted prices
            save_path (str): Path to save the plot
        """
        print("📊 Creating error distribution plot...")
        
        error = actuals - predictions
        percentage_error = (error / actuals) * 100
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        
        # Plot 1: Error distribution histogram
        axes[0, 0].hist(error, bins=50, color='#2E86AB', alpha=0.7, edgecolor='black')
        axes[0, 0].axvline(x=np.mean(error), color='red', linestyle='--', 
                          linewidth=2, label=f'Mean: ${np.mean(error):.2f}')
        axes[0, 0].set_title('Error Distribution', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Error ($)', fontsize=11)
        axes[0, 0].set_ylabel('Frequency', fontsize=11)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Percentage error distribution
        axes[0, 1].hist(percentage_error, bins=50, color='#A23B72', alpha=0.7, edgecolor='black')
        axes[0, 1].axvline(x=np.mean(percentage_error), color='red', linestyle='--', 
                          linewidth=2, label=f'Mean: {np.mean(percentage_error):.2f}%')
        axes[0, 1].set_title('Percentage Error Distribution', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Percentage Error (%)', fontsize=11)
        axes[0, 1].set_ylabel('Frequency', fontsize=11)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Scatter plot - Actual vs Predicted
        axes[1, 0].scatter(actuals, predictions, alpha=0.5, color='#2E86AB', s=20)
        min_val = min(actuals.min(), predictions.min())
        max_val = max(actuals.max(), predictions.max())
        axes[1, 0].plot([min_val, max_val], [min_val, max_val], 
                       'r--', linewidth=2, label='Perfect Prediction')
        axes[1, 0].set_title('Actual vs Predicted Prices', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Actual Price ($)', fontsize=11)
        axes[1, 0].set_ylabel('Predicted Price ($)', fontsize=11)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Error over time with moving average
        axes[1, 1].plot(error, alpha=0.5, color='gray', linewidth=1, label='Error')
        ma_error = pd.Series(error).rolling(window=20).mean()
        axes[1, 1].plot(ma_error, color='#F18F01', linewidth=2, label='20-Day MA Error')
        axes[1, 1].axhline(y=0, color='black', linestyle='--', linewidth=1)
        axes[1, 1].set_title('Error Trend with Moving Average', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Time Steps (Days)', fontsize=11)
        axes[1, 1].set_ylabel('Error ($)', fontsize=11)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Plot saved to {save_path}")
        plt.show()
        
    def plot_interactive_candlestick(self, data, predictions=None, save_path='outputs/interactive_chart.html'):
        """
        Create interactive candlestick chart with Plotly
        
        Args:
            data (pd.DataFrame): Stock data with OHLC
            predictions (array): Optional prediction data
            save_path (str): Path to save the HTML file
        """
        print("📊 Creating interactive candlestick chart...")
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=(f'{self.ticker} Stock Price', 'Volume'),
            row_heights=[0.7, 0.3]
        )
        
        # Candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name='OHLC',
                increasing_line_color='#26A69A',
                decreasing_line_color='#EF5350'
            ),
            row=1, col=1
        )
        
        # Add moving averages
        if 'MA_21' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['MA_21'],
                    mode='lines',
                    name='MA 21',
                    line=dict(color='orange', width=1.5)
                ),
                row=1, col=1
            )
        
        if 'MA_50' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['MA_50'],
                    mode='lines',
                    name='MA 50',
                    line=dict(color='blue', width=1.5)
                ),
                row=1, col=1
            )
        
        # Add predictions if provided
        if predictions is not None:
            pred_dates = data.index[-len(predictions):]
            fig.add_trace(
                go.Scatter(
                    x=pred_dates,
                    y=predictions,
                    mode='lines',
                    name='Predictions',
                    line=dict(color='purple', width=2, dash='dash')
                ),
                row=1, col=1
            )
        
        # Volume bar chart
        colors = ['red' if data['Close'].iloc[i] < data['Open'].iloc[i] 
                 else 'green' for i in range(len(data))]
        
        fig.add_trace(
            go.Bar(
                x=data.index,
                y=data['Volume'],
                name='Volume',
                marker_color=colors,
                showlegend=False
            ),
            row=2, col=1
        )
        
        # Update layout
        fig.update_layout(
            title=f'{self.ticker} Stock Analysis - Interactive Chart',
            xaxis_rangeslider_visible=False,
            height=800,
            template='plotly_white',
            hovermode='x unified'
        )
        
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        
        # Save to HTML
        fig.write_html(save_path)
        print(f"✅ Interactive chart saved to {save_path}")
        
        # Show plot
        fig.show()
        
    def create_comprehensive_report(self, data, actuals, predictions):
        """
        Create comprehensive visualization report
        
        Args:
            data (pd.DataFrame): Stock data
            actuals (array): Actual prices
            predictions (array): Predicted prices
        """
        print("\n" + "="*70)
        print("📊 CREATING COMPREHENSIVE VISUALIZATION REPORT")
        print("="*70 + "\n")
        
        # Plot 1: Predictions vs Actual
        self.plot_predictions_vs_actual(actuals, predictions)
        
        # Plot 2: Error analysis
        self.plot_error_distribution(actuals, predictions)
        
        # Plot 3: Interactive candlestick
        self.plot_interactive_candlestick(data, predictions)
        
        print("\n✅ All visualizations created successfully!\n")


# Example usage
if __name__ == "__main__":
    import sys
    import os
    # Add parent directory to path when running standalone
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Load results
    ticker = 'TSLA'
    
    # Load data
    data = pd.read_csv(f'data/{ticker}_stock_data.csv', index_col='Date', parse_dates=True)
    
    # Load predictions
    results = pd.read_csv(f'outputs/{ticker}_predictions.csv')
    actuals = results['Actual_Price'].values
    predictions = results['Predicted_Price'].values
    
    # Create visualizer
    visualizer = StockVisualizer(ticker=ticker)
    
    # Create comprehensive report
    visualizer.create_comprehensive_report(data, actuals, predictions)
