from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict,Any

from src.predict import Predictor 


CLASS_MAP = {
    0: "Healthy",
    1: "Pre-Diabetic",
    2: "Diabetic"
}


# FastAPI app

app = FastAPI(
    title ="Diabetes Prediction API",
    description ="Predict diabetes status",
    version="1.0"
)

# Load model once

Predictor = Predictor(preprocessor_path="/home/ayush/ishu/MLE-TRAINING/preprocessor.pkl",
                      model_path="/home/ayush/ishu/MLE-TRAINING/models/model.pkl")

# -------- Request Schema --------
class PredictRequest(BaseModel):
    input: Dict[str,Any]
    

# -------- Response Schema --------
class PredictResponse(BaseModel):
    predicted_class: str
    probabilities: Dict[str,float]
    
# -------- API Endpoint --------
@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):

    # 🔹 raw model output
    y_pred, raw_probs = Predictor.predict(request.input)

    # ✅ numpy → python int
    y_pred = int(y_pred)

    # ✅ class label
    predicted_class = CLASS_MAP[y_pred]

    # ✅ probabilities keys + values fix
    probabilities = {
        CLASS_MAP[int(k)]: float(v)
        for k, v in raw_probs.items()
    }

    return {
        "predicted_class": predicted_class,
        "probabilities": probabilities
    }
