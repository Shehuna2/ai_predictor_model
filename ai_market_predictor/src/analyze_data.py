import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# File paths
PREPROCESSED_DATA_PATH = "../data/processed/eth_usdt_preprocessed.csv"

def load_data(file_path):
    """Load preprocessed data."""
    print("Loading preprocessed data...")
    try:
        df = pd.read_csv(file_path, parse_dates=["timestamp"])
        print("Data loaded successfully!")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None

def plot_price_trend(df):
    """Plot the price trend over time."""
    print("Plotting price trend...")
    plt.figure(figsize=(12, 6))
    plt.plot(df["timestamp"], df["close"], label="Close Price", color="blue")
    plt.title("ETH/USDT Price Trend")
    plt.xlabel("Time")
    plt.ylabel("Price (USDT)")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("../results/price_trend.png")
    plt.show()

def plot_volatility(df):
    """Plot volatility over time."""
    print("Plotting volatility trend...")
    plt.figure(figsize=(12, 6))
    plt.plot(df["timestamp"], df["volatility"], label="Volatility", color="orange")
    plt.title("ETH/USDT Volatility Trend")
    plt.xlabel("Time")
    plt.ylabel("Volatility")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("../results/volatility_trend.png")
    plt.show()

def plot_price_change_distribution(df):
    """Plot distribution of price changes."""
    print("Plotting price change distribution...")
    plt.figure(figsize=(10, 6))
    sns.histplot(df["price_change"], bins=50, kde=True, color="green")
    plt.title("Distribution of Price Changes")
    plt.xlabel("Price Change (USDT)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig("../results/price_change_distribution.png")
    plt.show()

def generate_summary_statistics(df):
    """Print summary statistics for key columns."""
    print("Generating summary statistics...")
    summary = df[["close", "price_change", "volatility"]].describe()
    print(summary)
    return summary

if __name__ == "__main__":
    # Load data
    data = load_data(PREPROCESSED_DATA_PATH)
    
    if data is not None:
        # Perform analysis
        plot_price_trend(data)
        plot_volatility(data)
        plot_price_change_distribution(data)
        summary_stats = generate_summary_statistics(data)
        
        # Save summary statistics
        summary_stats.to_csv("../results/summary_statistics.csv")
        print("Summary statistics saved to ../results/summary_statistics.csv")
