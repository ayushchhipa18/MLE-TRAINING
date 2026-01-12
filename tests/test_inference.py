import pandas as pd
import numpy as np
import sys, os
import pytest

if os.getenv("CI") == "true":
    pytest.skip(
        "Skipping inference tests in CI (model artifacts not available)",
        allow_module_level=True,
    )

import numpy as np
from src.predict import Predictor


def test_predictor_load():
    p = Predictor(
        preprocessor_path="models/preprocessor.pkl",
        model_path="models/model.pkl",
    )
    assert p.model is None
    assert p.preprocessor is None


def test_predict_single_row(monkeypatch):
    dummy_pred = np.array(["Healthy"])

    def mock_predict(_):
        return dummy_pred

    p = Predictor(
        preprocessor_path="/home/ayush/ishu/MLE-TRAINING/models/preprocessor.pkl",
        model_path="/home/ayush/ishu/MLE-TRAINING/models/model.pkl",
    )

    monkeypatch.setattr(p.model, "predict", mock_predict)

    df = pd.DataFrame([{"Age": 30, "BMI": 28, "HighBP": 0}])

    pred = p.predict(df)
    assert pred[0] == "Healthy"
