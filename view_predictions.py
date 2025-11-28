"""
View and analyze prediction history
"""

import pandas as pd
import matplotlib.pyplot as plt
import os

def view_prediction_history(ticker='TSLA'):
    """View all predictions made for a stock"""
    
    filename = f'outputs/realtime_predictions_{ticker}.csv'
    
    if not os.path.exists(filename):
        print(f"❌ No prediction history found for {ticker}")
        return
    
    df = pd.read_csv(filename)
    
    print("\n" + "="*80)
    print(f"📊 PREDICTION HISTORY - {ticker}")
    print("="*80 + "\n")
    
    print(df.to_string(index=False))
    
    print(f"\n📈 Summary Statistics:")
    print(f"   Total Predictions: {len(df)}")
    print(f"   Average Expected Change: {df['Expected_Change_Pct'].mean():.2f}%")
    print(f"   Max Expected Gain: {df['Expected_Change_Pct'].max():.2f}%")
    print(f"   Max Expected Loss: {df['Expected_Change_Pct'].min():.2f}%")
    
    # Plot prediction trend
    plt.figure(figsize=(14, 6))
    plt.plot(df['Timestamp'], df['Current_Price'], 'o-', label='Current Price', linewidth=2)
    plt.plot(df['Timestamp'], df['Predicted_Price'], 's-', label='Predicted Price', linewidth=2)
    plt.fill_between(range(len(df)), df['Lower_Bound'], df['Upper_Bound'], alpha=0.2)
    plt.xticks(rotation=45)
    plt.title(f'{ticker} - Prediction History', fontsize=14, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'outputs/prediction_history_{ticker}.png', dpi=300)
    print(f"\n💾 Chart saved to outputs/prediction_history_{ticker}.png")
    plt.show()

if __name__ == "__main__":
    ticker = input("Enter ticker to view history [TSLA]: ").upper() or 'TSLA'
    view_prediction_history(ticker)
