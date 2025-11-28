import pandas as pd
import numpy as np
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from predict import Predictor


def test_predictor_load():
    p = Predictor(
        preprocessor_path="/home/ayush/ishu/MLE-TRAINING/preprocessor.pkl",
        model_path="/home/ayush/ishu/MLE-TRAINING/model.pkl",
    )
    assert p is not None


def test_predict_single_row(monkeypatch):
    dummy_pred = np.array(["Healthy"])

    def mock_predict(_):
        return dummy_pred

    p = Predictor(
        preprocessor_path="/home/ayush/ishu/MLE-TRAINING/preprocessor.pkl",
        model_path="/home/ayush/ishu/MLE-TRAINING/model.pkl",
    )

    monkeypatch.setattr(p.model, "predict", mock_predict)

    df = pd.DataFrame([{"Age": 30, "BMI": 28, "HighBP": 0}])

    pred = p.predict(df)
    assert pred[0] == "Healthy"
