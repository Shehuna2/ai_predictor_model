import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Define the file path for the processed data
FEATURES_FILE = "../data/features/eth_usdt_features.csv"

# Load the data
print("Loading data...")
df = pd.read_csv(FEATURES_FILE)
print("Data loaded.")

# Handle the timestamp column if it exists
if 'timestamp' in df.columns:
    # Option 1: Drop the 'timestamp' column if it's not needed
    df = df.drop(columns=['timestamp'])
    
    # Option 2: If you want to use timestamp, convert it to numeric (e.g., seconds since the start)
    # df['timestamp'] = pd.to_datetime(df['timestamp'])
    # df['timestamp'] = (df['timestamp'] - df['timestamp'].min()) / np.timedelta64(1, 's')

# Define the target column (assuming 'target' column exists in your features file)
target_column = 'target'  # Update this if your target column is named differently

# Split the dataset into features and target
X = df.drop(columns=[target_column])  # Features (drop target column)
y = df[target_column]  # Target column

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
print("Training the model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
print("Making predictions...")
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")

