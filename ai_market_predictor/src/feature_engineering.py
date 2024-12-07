import pandas as pd
import numpy as np
import os

# Define file paths
PREPROCESSED_FILE = "../data/processed/eth_usdt_preprocessed.csv"
FEATURES_FILE = "../data/features/eth_usdt_features.csv"

def generate_lag_features(df, column, lags):
    """
    Generate lag features for a given column.
    """
    for lag in lags:
        df[f"{column}_lag_{lag}"] = df[column].shift(lag)
    return df

def generate_rolling_features(df, column, windows):
    """
    Generate rolling mean and rolling std for a given column.
    """
    for window in windows:
        df[f"{column}_roll_mean_{window}"] = df[column].rolling(window).mean()
        df[f"{column}_roll_std_{window}"] = df[column].rolling(window).std()
    return df

def calculate_macd(df, short_window=12, long_window=26, signal_window=9):
    """
    Calculate the MACD (Moving Average Convergence Divergence) and Signal Line.
    """
    df['ema_short'] = df['close'].ewm(span=short_window, adjust=False).mean()
    df['ema_long'] = df['close'].ewm(span=long_window, adjust=False).mean()
    df['macd'] = df['ema_short'] - df['ema_long']
    df['macd_signal'] = df['macd'].ewm(span=signal_window, adjust=False).mean()
    return df

def calculate_rsi(df, window=14):
    """
    Calculate Relative Strength Index (RSI).
    """
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    return df

def main():
    print("Loading data...")
    if not os.path.exists(PREPROCESSED_FILE):
        print(f"Preprocessed file not found: {PREPROCESSED_FILE}")
        return
    
    df = pd.read_csv(PREPROCESSED_FILE)
    print("Data loaded.")

    print("Generating features...")
    # Generate lag features
    df = generate_lag_features(df, column='close', lags=[1, 2, 3])

    # Generate rolling features
    df = generate_rolling_features(df, column='close', windows=[5, 10, 20])

    # Calculate MACD
    df = calculate_macd(df)

    # Calculate RSI
    df = calculate_rsi(df)

    # Drop rows with NaN values (due to lagging/rolling calculations)
    df.dropna(inplace=True)

    # Save features
    print(f"Saving features to {FEATURES_FILE}...")
    os.makedirs(os.path.dirname(FEATURES_FILE), exist_ok=True)
    df.to_csv(FEATURES_FILE, index=False)
    print("Features saved.")

if __name__ == "__main__":
    main()
