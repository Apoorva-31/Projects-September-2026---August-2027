from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np
import os

app = FastAPI(title="AI Event Management - Staff Prediction API")

# 1. Dynamically locate and load your model relative to this file
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(BASE_DIR, "app", "models", "staff_model.pkl")

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    print(f"Error: Could not find model at {MODEL_PATH}")
    model = None

# 2. Define the exact shape of data the endpoint expects
class StaffInput(BaseModel):
    experience_years: float
    department_id: int
    role_level: int
    # 💡 IMPORTANT: Update these fields to match exactly what your model was trained on!

# 3. Create the POST endpoint
@app.post("/predict-staff")
def predict_staff(data: StaffInput):
    if model is None:
        raise HTTPException(status_code=500, detail="Prediction model is not loaded on the server.")
    
    try:
        # Convert incoming JSON data into a 2D numpy array format that scikit-learn expects
        input_data = np.array([[data.experience_years, data.department_id, data.role_level]])
        
        # Generate the prediction
        prediction = model.predict(input_data)
        
        # Return the result as JSON
        return {
            "status": "success",
            "prediction": float(prediction[0])  # Convert numpy float to native Python float
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")
