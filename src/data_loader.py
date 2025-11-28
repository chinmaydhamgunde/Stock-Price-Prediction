"""
Data Loader Module - Downloads stock data using latest yfinance API
"""

import yfinance as yf
import pandas as pd
import os
from datetime import datetime, timedelta

class StockDataLoader:
    def __init__(self, ticker, start_date=None, end_date=None):
        """
        Initialize the data loader
        
        Args:
            ticker (str): Stock ticker symbol (e.g., 'AAPL', 'TSLA', 'GOOGL')
            start_date (str): Start date in 'YYYY-MM-DD' format
            end_date (str): End date in 'YYYY-MM-DD' format
        """
        self.ticker = ticker
        self.start_date = start_date or (datetime.now() - timedelta(days=5*365)).strftime('%Y-%m-%d')
        self.end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        self.data = None
        
    def download_data(self):
        """Download stock data from Yahoo Finance"""
        print(f"📥 Downloading {self.ticker} data from {self.start_date} to {self.end_date}...")
        
        try:
            # Using latest yfinance 0.2.54 API
            stock = yf.Ticker(self.ticker)
            self.data = stock.history(start=self.start_date, end=self.end_date)
            
            if self.data.empty:
                raise ValueError(f"No data found for {self.ticker}")
            
            print(f"✅ Downloaded {len(self.data)} rows of data")
            print(f"📅 Date range: {self.data.index[0].date()} to {self.data.index[-1].date()}")
            
            return self.data
            
        except Exception as e:
            print(f"❌ Error downloading data: {e}")
            return None
    
    def save_data(self, filename=None):
        """Save data to CSV file"""
        if self.data is None:
            print("❌ No data to save. Download data first.")
            return
        
        if filename is None:
            filename = f"data/{self.ticker}_stock_data.csv"
        
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        self.data.to_csv(filename)
        print(f"💾 Data saved to {filename}")
        
    def load_data(self, filename):
        """Load data from CSV file"""
        self.data = pd.read_csv(filename, index_col='Date', parse_dates=True)
        print(f"📂 Loaded data from {filename}")
        return self.data
    
    def get_basic_info(self):
        """Display basic information about the data"""
        if self.data is None:
            print("❌ No data available")
            return
        
        print("\n" + "="*50)
        print(f"📊 STOCK DATA SUMMARY - {self.ticker}")
        print("="*50)
        print(f"\n📏 Shape: {self.data.shape}")
        print(f"\n📋 Columns: {list(self.data.columns)}")
        print(f"\n📈 First 5 rows:")
        print(self.data.head())
        print(f"\n📉 Last 5 rows:")
        print(self.data.tail())
        print(f"\n📊 Statistical Summary:")
        print(self.data.describe())
        print(f"\n🔍 Missing Values:")
        print(self.data.isnull().sum())
        print("="*50 + "\n")


# Example usage
if __name__ == "__main__":
    # Download Tesla stock data
    loader = StockDataLoader(ticker='TSLA', start_date='2018-01-01')
    data = loader.download_data()
    loader.save_data()
    loader.get_basic_info()
