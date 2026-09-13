import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os

# Get the directory where staff.py is located
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, "events.csv")

# Load the file using the full path
df = pd.read_csv(csv_path)

df = pd.get_dummies(
    df,
    columns=["event_type", "venue_type"],
    drop_first=True
)

X = df.drop("staff_required", axis=1)
y = df["staff_required"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)
predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("MAE:", mae)
print("R2:", r2)

# Saves the model file inside your models directory
joblib.dump(model, os.path.join(script_dir, "staff_model.pkl"))
