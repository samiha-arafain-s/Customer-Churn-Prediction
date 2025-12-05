from __future__ import annotations

import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from flask import (
    Flask,
    flash,
    redirect,
    render_template,
    request,
    send_file,
    url_for,
)
import pandas as pd

from model import predict_bulk, predict_single
from train import train_models
from preprocess import load_dataset


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "dataset.csv"
DB_PATH = BASE_DIR / "data" / "churn.db"
MODEL_PATH = BASE_DIR / "churn_model.pkl"
TELCO_FEATURE_COLUMNS = [
    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
]
TELCO_TARGET_COLUMN = "Churn"
PREDICT_FIELDS = [
    {"name": "customerID", "label": "Customer ID", "type": "text", "placeholder": "0001-BG"},
    {"name": "gender", "label": "Gender", "type": "select", "options": ["Female", "Male"]},
    {
        "name": "SeniorCitizen",
        "label": "Senior Citizen (0 = No, 1 = Yes)",
        "type": "number",
        "placeholder": "0",
    },
    {"name": "Partner", "label": "Partner", "type": "select", "options": ["Yes", "No"]},
    {"name": "Dependents", "label": "Dependents", "type": "select", "options": ["Yes", "No"]},
    {"name": "tenure", "label": "Tenure (months)", "type": "number", "placeholder": "12"},
    {"name": "PhoneService", "label": "Phone Service", "type": "select", "options": ["Yes", "No"]},
    {
        "name": "MultipleLines",
        "label": "Multiple Lines",
        "type": "select",
        "options": ["No phone service", "No", "Yes"],
    },
    {
        "name": "InternetService",
        "label": "Internet Service",
        "type": "select",
        "options": ["DSL", "Fiber optic", "No"],
    },
    {
        "name": "OnlineSecurity",
        "label": "Online Security",
        "type": "select",
        "options": ["No internet service", "No", "Yes"],
    },
    {
        "name": "OnlineBackup",
        "label": "Online Backup",
        "type": "select",
        "options": ["No internet service", "No", "Yes"],
    },
    {
        "name": "DeviceProtection",
        "label": "Device Protection",
        "type": "select",
        "options": ["No internet service", "No", "Yes"],
    },
    {
        "name": "TechSupport",
        "label": "Tech Support",
        "type": "select",
        "options": ["No internet service", "No", "Yes"],
    },
    {
        "name": "StreamingTV",
        "label": "Streaming TV",
        "type": "select",
        "options": ["No internet service", "No", "Yes"],
    },
    {
        "name": "StreamingMovies",
        "label": "Streaming Movies",
        "type": "select",
        "options": ["No internet service", "No", "Yes"],
    },
    {
        "name": "Contract",
        "label": "Contract",
        "type": "select",
        "options": ["Month-to-month", "One year", "Two year"],
    },
    {
        "name": "PaperlessBilling",
        "label": "Paperless Billing",
        "type": "select",
        "options": ["Yes", "No"],
    },
    {
        "name": "PaymentMethod",
        "label": "Payment Method",
        "type": "select",
        "options": [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
    },
    {"name": "MonthlyCharges", "label": "Monthly Charges", "type": "number", "placeholder": "70.35"},
    {"name": "TotalCharges", "label": "Total Charges", "type": "number", "placeholder": "3500.50"},
]

app = Flask(__name__, static_folder="static", template_folder="templates")
app.secret_key = "super-secret-key"  # for demo purposes only


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    DB_PATH.parent.mkdir(exist_ok=True)
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS training_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_type TEXT,
            accuracy REAL,
            trained_at TEXT
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS counters (
            name TEXT PRIMARY KEY,
            value INTEGER
        )
        """
    )
    conn.commit()
    conn.close()

# Initialize database tables on import (covers both `flask run` and `python app.py`)
init_db()


def get_prediction_count() -> int:
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT value FROM counters WHERE name='prediction_count'")
    row = cur.fetchone()
    conn.close()
    return row["value"] if row else 0


def increment_prediction_count(amount: int = 1):
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO counters(name, value) VALUES('prediction_count', ?)
        ON CONFLICT(name) DO UPDATE SET value = value + ?
        """,
        (amount, amount),
    )
    conn.commit()
    conn.close()


def store_training_metric(model_type: str, accuracy: float):
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO training_metrics(model_type, accuracy, trained_at) VALUES (?, ?, ?)",
        (model_type, accuracy, datetime.utcnow().isoformat()),
    )
    conn.commit()
    conn.close()


def latest_training_metric() -> Optional[Dict]:
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        "SELECT model_type, accuracy, trained_at FROM training_metrics ORDER BY id DESC LIMIT 1"
    )
    row = cur.fetchone()
    conn.close()
    if row:
        return dict(row)
    return None


def dataset_stats(csv_path: Path) -> Dict[str, float]:
    if not csv_path.exists():
        return {"total": 0, "churned": 0, "churn_pct": 0.0}
    df = load_dataset(csv_path)
    total = len(df)
    churned = int(df["Churn"].map({"Yes": 1, "No": 0}).sum()) if "Churn" in df else 0
    churn_pct = round((churned / total) * 100, 2) if total else 0.0
    return {"total": total, "churned": churned, "churn_pct": churn_pct}


@app.route("/")
def home():
    stats = dataset_stats(DATA_PATH)
    last_metric = latest_training_metric()
    return render_template(
        "home.html",
        stats=stats,
        last_metric=last_metric,
        model_exists=MODEL_PATH.exists(),
    )


@app.route("/upload", methods=["GET", "POST"])
def upload():
    stats = {}
    if request.method == "POST":
        file = request.files.get("dataset")
        if not file or file.filename == "":
            flash("Please choose a CSV file to upload.", "warning")
            return redirect(request.url)
        DATA_PATH.parent.mkdir(exist_ok=True)
        file.save(DATA_PATH)
        stats = dataset_stats(DATA_PATH)
        flash("Dataset uploaded successfully!", "success")
    else:
        stats = dataset_stats(DATA_PATH)
    return render_template(
        "upload.html",
        stats=stats,
        data_path=DATA_PATH,
        expected_columns=TELCO_FEATURE_COLUMNS + [TELCO_TARGET_COLUMN],
    )


@app.route("/train", methods=["GET", "POST"])
def train_route():
    metrics = latest_training_metric()
    if request.method == "POST":
        if not DATA_PATH.exists():
            flash("Upload a dataset first.", "danger")
            return redirect(url_for("upload"))
        try:
            results, best_model = train_models(str(DATA_PATH), str(MODEL_PATH))
            store_training_metric(best_model, results.get(best_model, 0))
            metrics = results
            flash(
                f"Training complete! Best model: {best_model} with accuracy {results.get(best_model)}",
                "success",
            )
        except Exception as exc:  # noqa: BLE001
            flash(f"Training failed: {exc}", "danger")
    return render_template(
        "train.html",
        metrics=metrics,
        data_path=DATA_PATH,
        expected_columns=TELCO_FEATURE_COLUMNS + [TELCO_TARGET_COLUMN],
    )


def extract_single_payload(form_data) -> Dict[str, str]:
    """Collect form fields into a payload dict for prediction."""
    payload = {}
    for key, value in form_data.items():
        if key in ("prediction_type",):
            continue
        payload[key] = value
    return payload


@app.route("/predict", methods=["GET", "POST"])
def predict_route():
    single_result = None
    bulk_result_path = None

    if request.method == "POST":
        if request.form.get("prediction_type") == "single":
            payload = extract_single_payload(request.form)
            try:
                single_result = predict_single(payload, str(MODEL_PATH))
                increment_prediction_count(1)
                flash("Prediction generated.", "success")
            except Exception as exc:  # noqa: BLE001
                flash(f"Prediction failed: {exc}", "danger")
        else:
            file = request.files.get("file")
            if not file or file.filename == "":
                flash("Please upload a CSV file for bulk prediction.", "warning")
                return redirect(request.url)
            input_path = DATA_PATH.parent / "bulk_input.csv"
            output_path = DATA_PATH.parent / "bulk_predictions.csv"
            file.save(input_path)
            try:
                df_pred = predict_bulk(str(input_path), str(MODEL_PATH))
                df_pred.to_csv(output_path, index=False)
                bulk_result_path = output_path
                increment_prediction_count(len(df_pred))
                flash("Bulk predictions complete. Download the results below.", "success")
            except Exception as exc:  # noqa: BLE001
                flash(f"Bulk prediction failed: {exc}", "danger")

    return render_template(
        "predict.html",
        single_result=single_result,
        bulk_result_path=bulk_result_path,
        predict_fields=PREDICT_FIELDS,
        expected_columns=TELCO_FEATURE_COLUMNS,
    )


@app.route("/download_predictions")
def download_predictions():
    output_path = DATA_PATH.parent / "bulk_predictions.csv"
    if not output_path.exists():
        flash("No predictions file found. Run a bulk prediction first.", "warning")
        return redirect(url_for("predict_route"))
    return send_file(output_path, as_attachment=True)


@app.route("/dashboard")
def dashboard():
    stats = dataset_stats(DATA_PATH)
    prediction_count = get_prediction_count()
    last_metric = latest_training_metric()
    return render_template(
        "dashboard.html",
        stats=stats,
        prediction_count=prediction_count,
        last_metric=last_metric,
    )


if __name__ == "__main__":
    init_db()
    app.run(debug=True)


