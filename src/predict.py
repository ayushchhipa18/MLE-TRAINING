import joblib
import numpy as np
import pandas as pd


class Predictor:
    def __init__(
        self,
        preprocessor_path: str ,
        model_path: str,
        load_verbose: bool = False,
    ):
        self.preprocessor_path = preprocessor_path
        self.model_path = model_path
        self.load_verbose = load_verbose

        self.preprocessor = None
        self.model = None

    def load(self):
        if self.preprocessor is None:
            obj = joblib.load(self.preprocessor_path)
            self.preprocessor = obj.get("preprocessor") if isinstance(obj, dict) else obj
        
        if self.model is None:
            obj = joblib.load(self.model_path)
            self.model = obj.get("model") if isinstance(obj, dict) else obj
        
    def _align_columns(self, df: pd.DataFrame):
        required_cols = list(self.preprocessor.feature_names_in_)
        for col in required_cols:
            if col not in df.columns:
                df[col] = 0
        return df[required_cols]

    def predict(self, input_data):
        """
        input_data:
        - pd.DataFrame (tests)
        - dict (FastAPI)
        """
        if isinstance(input_data,dict):
            df = pd.DataFrame([input_data])
        else:
            df = input_data.copy()
        
        df =self._align_columns(df)
        X = self.preprocessor.transform(df)
        
        preds = self.model.predict(X)
        
        if hasattr(self.model,"predict_proba"):
            probs = self.model.predict_proba(X)[0]
            probs_dict = dict(zip(self.model.classes_,probs))
            
        return preds[0],probs_dict