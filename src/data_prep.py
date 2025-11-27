import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
import joblib


def load_data(path):
    """Load  Dataset"""
    return pd.read_csv(path)


def clean_data(df):
    """CLEAN DATA = Drop duplicates & missing value"""
    df = df.drop_duplicates()
    df = df.dropna()
    return df


def encode_target(df, target_col):
    """Change into integer type"""
    df[target_col] = df[target_col].astype(int)
    return df


def create_preprocessor(df, target_col):
    """
    Prepare the preprocessing pipeline:
    - Separate features (X) and target (y)
    - Select numeric columns
    - Apply StandardScaler to numeric columns
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
    categorical_features = X.select_dtypes(include=["object", "category"]).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )
    return preprocessor, X, y


def save_preprocessor(preprocessor, save_path="preprocessor.pkl"):
    """Save the fitted preprocessing object to a .pkl file."""
    joblib.dump(preprocessor, save_path)
    print(f"Preprocessor saved at {save_path}")


def run_prep(data_path, target_col="Diabetes_012"):
    # load
    df = load_data(data_path)
    print("Loaded data Shape:", df.shape)

    # clean
    df = clean_data(df)
    print("After Cleaning Shape:", df.shape)

    # encoded target
    df = encode_target(df, target_col)

    # Create the preprocessing pipeline (scaling numeric features) & X,Y
    preprocessor, X, y = create_preprocessor(df, target_col)
    print("Feature columns:", X.columns.tolist())

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print("Shapes-> X_train:", X_train.shape, "X_test:", X_test.shape)

    # Fit the preprocessor on training data only
    preprocessor.fit(X_train)

    # Save the fitted preprocessor as a .pkl file
    save_preprocessor(preprocessor)


def clean_health_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic cleaning: remove duplicates, handle NaN, strip column names.
    """
    df = df.copy()
    df.columns = df.columns.str.strip()
    df = df.drop_duplicates()
    df = df.dropna()
    return df


def build_preprocessor(numerical_cols):
    """
    Returns a scikit-learn ColumnTransformer with StandardScaler.
    """
    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numerical_cols),
        ]
    )

    return preprocessor


if __name__ == "__main__":
    run_prep("/home/ayush/ishu/MLE-TRAINING/data/diabetes_cleaned.csv")
