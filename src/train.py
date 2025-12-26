import argparse
import logging
import os
import joblib

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
import mlflow
import mlflow.sklearn as mlflow_sklearn
import matplotlib.pyplot as plt
import seaborn as sns


# Experiment create
mlflow.set_experiment("diabetes_classification")


logging.basicConfig(level=logging.INFO, format="%(asctime)s-%(levelname)s-%(message)s")
logger = logging.getLogger(__name__)


def load_data(path: str) -> pd.DataFrame:
    logger.info("Loading data from %s", path)
    return pd.read_csv(path)


def main(args):
    # load data
    df = load_data(args.train_csv)

    if args.target not in df.columns:
        raise ValueError(
            f"Target columns'{args.target}'not found in data columns:{df.columns.to_list()}"
        )

    X = df.drop(columns=[args.target])
    y = df[args.target]

    # 2) train/test split
    if args.test_size and args.test_size > 0:
        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=args.test_size,
            random_state=args.random_seed,
            stratify=y if args.stratify else None,
        )
        logger.info("Train shape: %s, Val shape: %s", X_train.shape, X_val.shape)
    else:
        X_train, y_train = X, y
        X_val, y_val = None, None
        logger.info("Use data for training")

    # Load preprocessor

    logger.info("Load preprocessor from %s", args.preprocessor)
    preprocessor = joblib.load(args.preprocessor)

    # Transform feature
    logger.info("Transforming training feature with preprocessor")
    X_train_transformed = preprocessor.transform(X_train)
    if X_val is not None:
        X_val_transformed = preprocessor.transform(X_val)
    else:
        X_val_transformed = None

    # ML flow run
    with mlflow.start_run():
        # Log params
        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("max_depth", args.max_depth)
        mlflow.log_param("random_seed", args.random_seed)
        mlflow.log_param("cv", args.cv)
        mlflow.log_param("test_size", args.test_size)
        mlflow.log_param("stratify", args.stratify)
        mlflow.log_param("save_with_preprocessor", args.save_with_preprocessor)

        # Baseline model(Randomforest as ex)
        logger.info("Initializing baseline model (RandomForestClassifier)")
        model = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth if args.max_depth > 0 else None,
            random_state=args.random_seed,
            n_jobs=-1,
        )

        # cross-validation
        if args.cv and args.cv > 1:
            logger.info("Running %d-fold cross validation", args.cv)
            cv_score = cross_val_score(
                model, X_train_transformed, y_train, cv=args.cv, scoring="f1_weighted", n_jobs=-1
            )
            logger.info("CV scores: %s", np.round(cv_score, 4).tolist())
            logger.info("CV mean f1_weighted: %.4f", np.mean(cv_score))
            mlflow.log_metric("cv_mean_f1_weighted", float(np.mean(cv_score)))

        # Fit the model
        logger.info("Fitting model on training data")
        model.fit(X_train_transformed, y_train)

        # validate  data
        if X_val_transformed is not None:
            logger.info("Evaluating on validation set ")
            y_pred = model.predict(X_val_transformed)
            acc = accuracy_score(y_val, y_pred)
            logger.info("Validation accuracy: %.4f", acc)
            logger.info("Classification report:\n%s", classification_report(y_val, y_pred))

            # Weighted F1-score
            f1 = f1_score(y_val, y_pred, average="weighted")
            logger.info("Weighted F1 score: %.4f", f1)
            mlflow.log_metric("weighted_f1_score", float(f1))

            # Confusion matrix and log as artifact
            cm = confusion_matrix(y_val, y_pred)
            plt.figure(figsize=(6, 6))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.title("Confusion Matrix")
            cm_path = "confusion_matrix.png"
            plt.savefig(cm_path, bbox_inches="tight")
            plt.close()
            mlflow.log_artifact(cm_path)
            logger.info("confusion matrix saved and logged to MLflow as artifact: %s", cm_path)
        else:
            logger.info("No validation set to evaluate")

        # save model
        # os.makedirs(os.path.dirname(args.output), exist_ok=True)
        out_dir = os.path.dirname(args.output)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        if args.save_with_preprocessor:
            logger.info("Saving pipeline (preprocessor + model) to %s", args.output)
            joblib.dump({"Preprocessor": preprocessor, "model": model}, args.output)

            # log pipeline into mlflow as model artifact (optional)

            try:
                mlflow_sklearn.log_model(model, artifact_path="model")
                mlflow.log_artifact(args.preprocessor)

            except Exception as e:
                logger.warning("Could not log pipeline to MLflow: %s", e)

        else:
            logger.info("Saving model only to %s", args.output)
            joblib.dump(model, args.output)

            try:
                mlflow_sklearn.log_model(model, artifact_path="model")
            except Exception as e:
                logger.warning("Could not log model to MLflow: %s", e)

        logger.info("Training finished, save to %s", args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train baseline model script")
    parser.add_argument("--train-csv", required=True, help="Path to processed train CSV")
    parser.add_argument("--preprocessor", required=True, help="Path to preprocessor .pkl (joblib)")
    parser.add_argument("--output", required=True, help="Patht to save trained model(.pkl)")
    parser.add_argument("--target", required=True, help="Name of target column in CSV")
    parser.add_argument(
        "--test-size", type=float, default=0.2, help="Validation split fraction (0 to disable)"
    )
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=10)
    parser.add_argument("--cv", type=int, default=3, help="Number of CV folds (0 or 1 to disable)")
    parser.add_argument("--stratify", action="store_true", help="Use stratified split")
    parser.add_argument(
        "--save-with-preprocessor",
        action="store_true",
        help="If set, save both preprocessor and model together in output (dict)",
    )
    args = parser.parse_args()
    main(args)
