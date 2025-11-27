import pandas as pd
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from data_prep import clean_health_data, build_preprocessor


def test_clean_health_data():
    df = pd.DataFrame({"Age": [25, None], "BMI": [30.5, 22.1], "Diabetic": ["Yes", "No"]})
    cleaned = clean_health_data(df)

    assert cleaned.isnull().sum().sum() == 0

    assert "Age" in cleaned.columns
    assert "Diabetic" in cleaned.columns


def test_preprocessor():
    numerical = ["Age", "BMI"]
    categorical = ["Diabetic"]

    preprocessor = build_preprocessor(numerical, categorical)

    assert preprocessor is not None
