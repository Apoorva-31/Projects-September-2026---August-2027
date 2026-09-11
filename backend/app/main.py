from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import pandas as pd
from pathlib import Path


# Create the FastAPI application
app = FastAPI()


# Load the trained pipeline ONCE, when the server starts
# The pipeline contains both preprocessing and the ML model
MODEL_PATH = Path(__file__).parent / "attendance_prediction" / "attendance_model.pkl"
model = joblib.load(MODEL_PATH)


# Defines the exact shape of data our API expects
class EventInput(BaseModel):
    event_category: str
    ticket_price: float = Field(ge=0)
    venue_capacity: int = Field(gt=0)
    promotion_days: int = Field(ge=0)
    num_registrations: int = Field(ge=0)


# Simple test endpoint
@app.get("/")
def home():
    return {
        "message": "Event Attendance Prediction API is running"
    }


# The actual prediction endpoint
@app.post("/predict-attendance")
def predict_attendance(event: EventInput):

    # Convert incoming event data into a one-row DataFrame
    input_data = pd.DataFrame([{
        "event_category": event.event_category,
        "ticket_price": event.ticket_price,
        "venue_capacity": event.venue_capacity,
        "promotion_days": event.promotion_days,
        "num_registrations": event.num_registrations
    }])


    # Get prediction from the trained ML pipeline
    prediction = model.predict(input_data)

    predicted_attendance = round(float(prediction[0]), 1)


    # Calculate expected percentage of venue capacity
    capacity_utilization = round(
        (predicted_attendance / event.venue_capacity) * 100,
        1
    )


    # Determine attendance status
    if capacity_utilization < 30:
        attendance_status = "Low"

    elif capacity_utilization < 60:
        attendance_status = "Moderate"

    elif capacity_utilization < 80:
        attendance_status = "High"

    else:
        attendance_status = "Very High"


    # Determine crowd risk
    if capacity_utilization < 60:
        crowd_alert = "No crowd risk"

    elif capacity_utilization < 80:
        crowd_alert = "Moderate crowd risk"

    else:
        crowd_alert = "High crowd risk"


    # Return all prediction information
    return {
        "predicted_attendance": predicted_attendance,
        "capacity_utilization": capacity_utilization,
        "attendance_status": attendance_status,
        "crowd_alert": crowd_alert
    }