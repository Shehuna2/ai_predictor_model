import ccxt
import pandas as pd
import os
from datetime import datetime

def fetch_binance_data(symbol="ETH/USDT", timeframe="1h", since=None, limit=1000):
    """
    Fetch historical data from Binance using CCXT.
    
    Args:
        symbol (str): Trading pair (default is ETH/USDT).
        timeframe (str): Timeframe for candles (default is 1h).
        since (int): Start time in milliseconds (default is None, fetch recent data).
        limit (int): Number of data points to fetch (default is 1000).
    
    Returns:
        pd.DataFrame: Historical data as a pandas DataFrame.
    """
    try:
        # Initialize Binance exchange
        exchange = ccxt.binance()
        
        # Fetch OHLCV data
        data = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
        
        # Convert data to DataFrame
        df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume"])
        
        # Convert timestamp to readable date
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        
        return df
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

def save_data_to_csv(df, file_path):
    """
    Save DataFrame to CSV.
    
    Args:
        df (pd.DataFrame): DataFrame to save.
        file_path (str): Path to save the file.
    """
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Save DataFrame to CSV
        df.to_csv(file_path, index=False)
        print(f"Data saved to {file_path}")
    except Exception as e:
        print(f"Error saving data: {e}")

if __name__ == "__main__":
    # Fetch data
    print("Fetching data...")
    data = fetch_binance_data(symbol="ETH/USDT", timeframe="1h", limit=500)
    
    if data is not None:
        # Save to CSV
        output_file = "../data/raw/eth_usdt_data.csv"
        save_data_to_csv(data, output_file)
