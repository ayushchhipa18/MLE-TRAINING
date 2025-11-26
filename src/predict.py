from typing import Dict, Tuple, Any, Optional, List
import os
import logging
import pickle
import joblib
import time
import pandas as pd
import numpy as np


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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
        self.classes_ = getattr(self.model, "classes_", None)

        if load_verbose:
            logger.info(f"Loaded preprocessor:{type(self.preprocessor)}")
            logger.info(f"Loaded model:{type(self.model)}")
            logger.info(f"classes:{self.classes_}")

    def _load_model(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model artifact  not found at: {path}")

        try:
            return joblib.load(path)
        except Exception:
            try:
                with open(path, "rb") as f:
                    return pickle.load(f)
            except Exception:
                raise RuntimeError(
                    "Failed to load model with joblib/pickle. "
                    "If using TensorFlow/Keras saved model, implement TF loader."
                )

    def _build_dataframe(self, input_dict: Dict[str, Any]) -> pd.DataFrame:
        """
        Convert single-row dict to DataFrame and align columns if preprocessor has feature_names_in_.
        """
        if not isinstance(input_dict, dict):
            raise ValueError("input_dict must be a dict of feature_name: value")

        df = pd.DataFrame([input_dict])

        # If the preprocessor saved the training feature order, enforce it.
        feature_names = getattr(self.preprocessor, "feature_names_in_", None)
        if feature_names is not None:
            missing = [c for c in feature_names if c not in df.columns]
            if missing:
                raise ValueError(f"Missing features in input: {missing}")

            df = df.loc[:, feature_names]
        return df

    # Prediction logic
    def predict(self, input_dict: Dict[str, Any]) -> Tuple[Any, Dict[str, float]]:
        """
        Predict single-row input.

        Args:
            input_dict: mapping feature->value for one sample.

        Returns:
            (predicted_label, probabilities_dict)
        """
        start = time.time()
        df = self._build_dataframe(input_dict)

        try:
            X_transformed = self.preprocessor.transform(df)
        except Exception as e:
            logger.exception("Preprocessor.transform failed")
            raise RuntimeError(f"preprocessor.transform failed:{e}")

        # ensure numpy array for model input
        if isinstance(X_transformed, (pd.DataFrame, pd.Series)):
            X_input = X_transformed.values
        else:
            X_input = X_transformed

        # Try predict_proba
        proba = None
        if hasattr(self.model, "predict_proba"):
            try:
                proba_arr = self.model.predict_proba(X_input)
                proba = proba_arr[0]
            except Exception:
                logger.exception("model.predict_proba failed")

        # fallback: decision_function -> softmax
        if proba is None and hasattr(self.model, "decision_function"):
            try:
                scores = self.model.decision_function(X_input)
                if scores.ndim == 1:
                    # binary case
                    scores = np.vstack([-scores, scores]).T
                exp = np.exp(scores - np.max(scores, axis=1, keepdims=True))
                proba_arr = exp / exp.sum(axis=1, keepdims=True)
                proba = proba_arr[0]
            except Exception:
                logger.exception("model.decision_function fallback failed")

        # last fallback: use predict and set prob 1.0
        if proba is None:
            y_pred = self.model.predict(X_input)
            pred_label = y_pred[0]
            probs = {str(pred_label): 1.0}
            elapsed = time.time() - start
            logger.info(f"Predicted{pred_label} (no proba).a time={elapsed:.3f}s")
            return pred_label, probs

        # Build probabilities dict
        if self.classes_ is not None:
            probs = {str(cls): float(proba[i]) for i, cls in enumerate(self.classes_)}
            pred_label = self.classes_[int(np.argmax(proba))]
        else:
            probs = {str(i): float(p) for i, p in enumerate(proba)}
            pred_label = max(probs, key=probs.get)

        elapsed = time.time() - start
        logger.info(f"Predicted{pred_label}. time={elapsed:.3f}s")
        return pred_label, probs

    # Optional: batch predict (list of dicts)
    def predict_batch(self, inputs: List[Dict[str, Any]]) -> List[Tuple[Any, Dict[str, float]]]:
        """
        Accept list of dicts and return list of (label, probs).
        """
        if not isinstance(input, list):
            raise ValueError("inputs must be a list of dicts")
        results = []
        for inp in inputs:
            results.append(self.predict(inp))
        return results
