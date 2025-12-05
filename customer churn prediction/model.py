"""
Prediction utilities that wrap the saved joblib pipeline.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List

import joblib
import pandas as pd

MODEL_PATH = "churn_model.pkl"


def load_saved_model(model_path: str = MODEL_PATH):
    """Load the persisted model bundle."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No trained model found at {model_path}. Train it via /train."
        )
    bundle = joblib.load(model_path)
    if "model" not in bundle:
        raise ValueError("Saved model file is invalid. Please retrain the model.")
    return bundle


def _align_features(df: pd.DataFrame, feature_columns: List[str] | None) -> pd.DataFrame:
    """Ensure incoming data has all columns used during training."""
    if not feature_columns:
        return df

    for col in feature_columns:
        if col not in df.columns:
            df[col] = pd.NA

    # Extra columns are dropped to match the training set exactly
    return df[feature_columns]


def _coerce_numeric(df: pd.DataFrame, numeric_columns: List[str] | None) -> pd.DataFrame:
    """Convert known numeric columns to numeric dtype, coercing errors to NaN."""
    if not numeric_columns:
        return df
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def predict_single(payload: Dict[str, Any], model_path: str = MODEL_PATH) -> Dict[str, Any]:
    bundle = load_saved_model(model_path)
    model = bundle["model"]
    feature_columns = bundle.get("feature_columns")
    numeric_columns = bundle.get("numeric_columns")

    df = pd.DataFrame([payload])
    df = _coerce_numeric(df, numeric_columns)
    df = _align_features(df, feature_columns)

    prob = model.predict_proba(df)[0][1]
    pred = int(prob >= 0.5)
    return {"prediction": pred, "probability": round(float(prob), 4)}


def predict_bulk(csv_path: str, model_path: str = MODEL_PATH) -> pd.DataFrame:
    bundle = load_saved_model(model_path)
    model = bundle["model"]
    feature_columns = bundle.get("feature_columns")
    numeric_columns = bundle.get("numeric_columns")

    df = pd.read_csv(csv_path)
    if "Churn" in df.columns:
        df = df.drop(columns=["Churn"])

    df = _coerce_numeric(df, numeric_columns)
    df = _align_features(df, feature_columns)

    probs = model.predict_proba(df)[:, 1]
    preds = (probs >= 0.5).astype(int)
    df_out = df.copy()
    df_out["churn_probability"] = probs
    df_out["churn_prediction"] = preds
    return df_out


