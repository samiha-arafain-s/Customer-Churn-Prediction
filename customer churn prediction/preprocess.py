"""
Data preprocessing utilities for the Telco Customer Churn dataset.

Responsibilities:
- Load CSV data safely.
- Clean common Telco quirks (TotalCharges sometimes blank strings).
- Split features/target.
- Build a preprocessing ColumnTransformer that imputes, encodes, and scales.
"""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET_COLUMN = "Churn"


def load_dataset(csv_path: str) -> pd.DataFrame:
    """Read the dataset and normalize obvious issues."""
    df = pd.read_csv(csv_path)

    # Convert TotalCharges to numeric, coerce blanks to NaN first
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    return df


def split_features_target(df: pd.DataFrame):
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Dataset must contain '{TARGET_COLUMN}' column.")

    y = df[TARGET_COLUMN].map({"Yes": 1, "No": 0}).fillna(df[TARGET_COLUMN])
    X = df.drop(columns=[TARGET_COLUMN])
    return X, y


def build_preprocess_pipeline(
    X: pd.DataFrame,
    categorical_cols: list[str] | None = None,
    numeric_cols: list[str] | None = None,
) -> ColumnTransformer:
    """
    Create preprocessing transformer for numeric and categorical columns.

    The column lists can be supplied (to persist them for later predictions) or
    inferred from the dataframe.
    """
    if categorical_cols is None or numeric_cols is None:
        categorical_cols = [c for c in X.columns if X[c].dtype == "object"]
        numeric_cols = [c for c in X.columns if c not in categorical_cols]

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", categorical_pipeline, categorical_cols),
            ("numeric", numeric_pipeline, numeric_cols),
        ]
    )
    return preprocessor


def preprocess_features(df: pd.DataFrame):
    """Return X, y, preprocessing pipeline, and column metadata."""
    X, y = split_features_target(df)
    categorical_cols = [c for c in X.columns if X[c].dtype == "object"]
    numeric_cols = [c for c in X.columns if c not in categorical_cols]
    preprocessor = build_preprocess_pipeline(
        X, categorical_cols=categorical_cols, numeric_cols=numeric_cols
    )
    return X, y, preprocessor, categorical_cols, numeric_cols


