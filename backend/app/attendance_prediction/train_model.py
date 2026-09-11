import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# ---- Step A: Load the data ----

df = pd.read_csv("synthetic_event_data_v1.csv")

# Remove event_id because it is only an identifier
df = df.drop(columns=["event_id"])

# Create features and target
X = df[
    [
        "event_category",
        "ticket_price",
        "venue_capacity",
        "promotion_days",
        "num_registrations"
    ]
]

y = df["actual_attendance"]


# ---- Step B: Split the data ----

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ---- Step C: Create preprocessing ----

preprocessor = ColumnTransformer(
    transformers=[
        (
            "category_encoder",
            OneHotEncoder(handle_unknown="ignore"),
            ["event_category"]
        )
    ],
    remainder="passthrough"
)


# ---- Step D: Create the complete pipeline ----

pipeline = Pipeline(
    steps=[
        ("preprocessing", preprocessor),
        ("model", RandomForestRegressor(random_state=42))
    ]
)


# ---- Step E: Train the pipeline ----

pipeline.fit(X_train, y_train)

print("Model training complete.")


# ---- Step F: Save the complete pipeline ----

joblib.dump(pipeline, "attendance_model.pkl")

print("Trained pipeline saved as attendance_model.pkl")


# ---- Step G: Test the model ----

predictions = pipeline.predict(X_test)

print("\nActual vs Predicted attendance (first 10 test events):")

for actual, predicted in list(zip(y_test, predictions))[:10]:
    print(f"  Actual: {actual}   Predicted: {predicted:.1f}")


# ---- Step H: Evaluate the model ----

mae = mean_absolute_error(y_test, predictions)

rmse = np.sqrt(mean_squared_error(y_test, predictions))

r2 = r2_score(y_test, predictions)

print("\nModel evaluation:")

print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.3f}")


# ---- Step I: Test a brand-new event ----

new_event = pd.DataFrame([
    {
        "event_category": "Concert",
        "ticket_price": 300,
        "venue_capacity": 1000,
        "promotion_days": 20,
        "num_registrations": 400
    }
])

new_prediction = pipeline.predict(new_event)

print("\nPrediction for new event:")

print(f"Predicted attendance: {new_prediction[0]:.1f}")