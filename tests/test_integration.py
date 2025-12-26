import json
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))) 

from fastapi.testclient import TestClient
from app.api import app

client = TestClient(app)


def test_predict():
    sample_playlod = {
        "HighBP": 1,
        "HighChol":1,
        "CholCheck": 1,
        "BMI": 35.5,
        "Smoker": 1,
        "Stroke": 0,
        "HeartDiseaseorAttack": 1,
        "PhysActivity": 0,
        "Fruits": 0,
        "Veggies": 0,
        "HvyAlcoholConsump": 0,
        "AnyHealthcare": 1,
        "NoDocbcCost": 0,
        "GenHlth": 4,
        "MentHlth": 5,
        "PhysHlth": 5,
        "DiffWalk": 1,
        "Sex": 1,
        "Age": 50,
        "Education": 4,
        "Income": 3
        
    }
    
    response = client.post("/predict",json=sample_playlod)
    
    assert response.status_code == 200
    
    data = response.json()
    
    assert "predicted_class" in data
    assert "probabilities" in data
    assert isinstance(data["probabilities"],dict)
    
