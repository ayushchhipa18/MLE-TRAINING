from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from app.api import app



@patch("app.api.predictor")
def test_predict(mock_predictor):
    mock_predictor = MagicMock()
    mock_predictor.predict.return_value = (
        1,
        {"0": 0.2, "1": 0.8}
    )
    mock_get_predictor.return_value = mock_predictor

    client = TestClient(app)
    
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
    
    
    assert data["predicted_class"] == "Pre-Diabetic"
    assert isinstance(data["probabilities"], dict)
    
