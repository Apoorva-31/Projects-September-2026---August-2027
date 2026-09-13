from fastapi import FastAPI
import joblib

app = FastAPI()

model = joblib.load("models/staff_model.pkl")
@app.get("/")
def home():
    return {
        "message": "AI Event Management API is running"
    }