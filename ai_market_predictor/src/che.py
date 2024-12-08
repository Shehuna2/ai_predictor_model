import pandas as pd

# Load the feature dataset
data_path = "./data/features/eth_usdt_features.csv"
df = pd.read_csv(data_path)

# Print the columns
print("Columns in the dataset:")
print(df.columns)
