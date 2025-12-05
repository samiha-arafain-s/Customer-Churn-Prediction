"""
Training utilities for churn models.
Two models are trained: Logistic Regression and Random Forest.
The best model by accuracy is persisted to disk with joblib.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Dict, Tuple

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from preprocess import load_dataset, preprocess_features


def train_models(
    data_path: str,
    model_path: str = "churn_model.pkl",
) -> Tuple[Dict[str, float], str]:
    """
    Train Logistic Regression and Random Forest, save the best model.

    Returns:
        metrics: dict containing accuracies per model and timestamp.
        best_model_name: which model was saved.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    df = load_dataset(data_path)
    X, y, preprocessor, categorical_cols, numeric_cols = preprocess_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, max_depth=None, random_state=42
        ),
    }

    accuracies: Dict[str, float] = {}
    best_model_name = None
    best_accuracy = -np.inf
    best_pipeline: Pipeline | None = None

    for name, model in models.items():
        pipeline = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)
        acc = accuracy_score(y_test, preds)
        accuracies[name] = round(float(acc), 4)

        if acc > best_accuracy:
            best_accuracy = acc
            best_model_name = name
            best_pipeline = pipeline

    if best_pipeline is None or best_model_name is None:
        raise RuntimeError("Model training failed.")

    joblib.dump(
        {
            "model": best_pipeline,
            "trained_at": datetime.utcnow().isoformat(),
            "feature_columns": list(X.columns),
            "categorical_columns": categorical_cols,
            "numeric_columns": numeric_cols,
            "best_model": best_model_name,
            "metrics": accuracies,
        },
        model_path,
    )

    return {
        **accuracies,
        "saved_model": best_model_name,
        "trained_at": datetime.utcnow().isoformat(),
    }, best_model_name


