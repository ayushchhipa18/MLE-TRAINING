import joblib
import numpy as np
import pandas as pd


class Predictor:
    def __init__(
        self,
        preprocessor_path: str = "preprocessor.pkl",
        model_path: str = "model.pkl",
        load_verbose: bool = False,
    ):
        self.preprocessor_path = preprocessor_path
        self.model_path = model_path

        self.preprocessor = self._load_preprocessor(preprocessor_path)
        self.model = self._load_model(model_path)

    def _load_preprocessor(self, path):
        obj = joblib.load(path)
        if isinstance(obj, dict):
            return obj.get("preprocessor") or obj.get("Preprocessor")
        return obj

    def _load_model(self, path):
        obj = joblib.load(path)
        if isinstance(obj, dict):
            return obj.get("model") or obj.get("Model")
        return obj

    # ✅ Tests expect this function
    def predict(self, df: pd.DataFrame):
        X = self.preprocessor.transform(df)
        return self.model.predict(X)

    # Optional single-row function
    def predict_single_row(self, row: dict):
        df = pd.DataFrame([row])
        return self.predict(df)[0]

    def _align_columns(self, df):
        required_cols = list(self.preprocessor.feature_names_in_)
        for col in required_cols:
            if col not in df.columns:
                df[col] = 0
        return df[required_cols]

    def predict(self, df: pd.DataFrame):
        df = self._align_columns(df)
        X = self.preprocessor.transform(df)
        return self.model.predict(X)
