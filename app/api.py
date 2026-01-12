from fastapi import FastAPI
from pydantic import BaseModel
import os
from typing import Dict, Any

# -- CONSTANTS --
CLASS_MAP = {0: "Healthy", 1: "Pre-Diabetic", 2: "Diabetic"}
# -- APP INIT --
app = FastAPI(
    title="Diabetes Prediction API",
    description="Predict diabetes status",
    version="1.0",
    root_path="/api",
)
# -- Load model once --
_model_predictor = None


def get_predictor():
    global _model_predictor
    if _model_predictor is None:
        from src.predict import Predictor

        _model_predictor = Predictor(
            preprocessor_path=os.getenv("PREPROCESSOR_PATH", "models/preprocessor.pkl"),
            model_path=os.getenv("MODEL_PATH", "models/model.pkl"),
        )
    return _model_predictor


# -- SCHEMAS --
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
    probabilities: Dict[str, float]


# -- HEALTH CHECK --
@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/api/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    predictor = get_predictor()

    input_data = request.model_dump()

    y_pred, raw_probs = predictor.predict(input_data)

    y_pred = int(y_pred)
    predicted_class = CLASS_MAP[y_pred]

    probabilities = {CLASS_MAP[int(k)]: float(v) for k, v in raw_probs.items()}

    return {"predicted_class": predicted_class, "probabilities": probabilities}
