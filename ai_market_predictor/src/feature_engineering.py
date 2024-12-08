import pandas as pd
import numpy as np
import ta  # For technical indicators

def add_technical_indicators(data):
    """
    Add technical indicators to the dataset.
    """
    # Simple Moving Averages
    data['SMA_10'] = data['close'].rolling(window=10).mean()
    data['SMA_50'] = data['close'].rolling(window=50).mean()
    
    # Exponential Moving Averages
    data['EMA_10'] = data['close'].ewm(span=10, adjust=False).mean()
    data['EMA_50'] = data['close'].ewm(span=50, adjust=False).mean()
    
    # Bollinger Bands
    bollinger = ta.volatility.BollingerBands(close=data['close'], window=20, window_dev=2)
    data['Bollinger_High'] = bollinger.bollinger_hband()
    data['Bollinger_Low'] = bollinger.bollinger_lband()

    # Relative Strength Index (RSI)
    data['RSI'] = ta.momentum.RSIIndicator(close=data['close'], window=14).rsi()
    
    # Moving Average Convergence Divergence (MACD)
    macd = ta.trend.MACD(close=data['close'])
    data['MACD'] = macd.macd()
    data['MACD_Signal'] = macd.macd_signal()
    
    # Average True Range (ATR)
    data['ATR'] = ta.volatility.AverageTrueRange(high=data['high'], low=data['low'], close=data['close']).average_true_range()
    
    return data

def generate_features(input_file, output_file):
    """
    Load raw data, generate features, and save to a new file.
    """
    print("Loading data...")
    data = pd.read_csv(input_file)
    print("Data loaded.")
    
    print("Generating features...")
    
    # Add technical indicators
    data = add_technical_indicators(data)
    
    # Add lagging features
    for lag in range(1, 4):  # Add lag 1, 2, 3
        data[f'close_lag_{lag}'] = data['close'].shift(lag)
    
    # Add time-based features
    data['hour'] = pd.to_datetime(data['timestamp']).dt.hour
    data['day_of_week'] = pd.to_datetime(data['timestamp']).dt.dayofweek
    data['week_of_year'] = pd.to_datetime(data['timestamp']).dt.isocalendar().week
    
    # Normalize data (example for 'close' column)
    data['close_normalized'] = (data['close'] - data['close'].mean()) / data['close'].std()
    # Define target: Predicting if price increases or decreases
    data['target'] = (data['close'].shift(-1) > data['close']).astype(int)


    # Drop rows with NaN values due to lagging and rolling calculations
    data = data.dropna()
    
    print("Saving features to", output_file)
    data.to_csv(output_file, index=False)
    print("Features saved.")

if __name__ == "__main__":
    generate_features("./data/raw/eth_usdt_data.csv", "./data/features/eth_usdt_features.csv")
