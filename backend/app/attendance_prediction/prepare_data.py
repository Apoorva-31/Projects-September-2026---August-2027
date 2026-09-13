import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv("synthetic_event_data_v1.csv")

# Remove event_id (not a useful feature)
df = df.drop(columns=["event_id"])

# Create X (features) and y (target)
X = df[["event_category", "ticket_price", "venue_capacity", "promotion_days", "num_registrations"]]
y = df["actual_attendance"]

# Apply one-hot encoding to event_category
X = pd.get_dummies(X, columns=["event_category"])

# Split into training and testing data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Print shapes
print("Original dataset shape:", df.shape)
print("X shape:", X.shape)
print("y shape:", y.shape)
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

# Print final feature names after one-hot encoding
print("Final feature names:", list(X.columns))
