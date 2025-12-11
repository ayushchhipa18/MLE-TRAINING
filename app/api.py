from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict,Any

from src.predict import Predictor 


CLASS_MAP = {
    0: "Healthy",
    1: "Pre-Diabetic",
    2: "Diabetic"
}

app = FastAPI(
    title ="Diabetes Prediction API",
    description ="Predict diabetes status",
    version="1.0"
)

# Load model once
model_predictor = Predictor(
    preprocessor_path="/home/ayush/ishu/MLE-TRAINING/preprocessor.pkl",
    model_path="/home/ayush/ishu/MLE-TRAINING/models/model.pkl"
)

class PredictRequest(BaseModel):
    HighBP: int
    HighChol: int
    CholCheck: int
    BMI: float
    Smoker: int
    Stroke: int
    HeartDiseaseorAttack: int
    PhysActivity: int
    Fruits: int
    Veggies: int
    HvyAlcoholConsump: int
    AnyHealthcare: int
    NoDocbcCost: int
    GenHlth: int
    MentHlth: int
    PhysHlth: int
    DiffWalk: int
    Sex: int
    Age: int
    Education: int
    Income: int

class PredictResponse(BaseModel):
    predicted_class: str
    probabilities: Dict[str,float]

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):

    input_data = request.model_dump()
    
    # ⬇️ FIXED
    y_pred, raw_probs = model_predictor.predict(input_data)

    y_pred = int(y_pred)
    predicted_class = CLASS_MAP[y_pred]

    probabilities = {
        CLASS_MAP[int(k)]: float(v)
        for k, v in raw_probs.items()
    }

    return {
        "predicted_class": predicted_class,
        "probabilities": probabilities
    }
