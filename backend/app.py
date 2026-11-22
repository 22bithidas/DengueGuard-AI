"""DengueGuard AI API and local frontend server."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory
from scipy import sparse

from feature_engineering import FEATURES, build_features, apply_weather_scenario

ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "frontend"
DATA_PATH = ROOT / "data" / "hackathon_preli_dataset.xlsx"
MODEL_DIR = ROOT / "models"
app = Flask(__name__)


@lru_cache(maxsize=1)
def feature_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    source = pd.read_excel(DATA_PATH)
    engineered = build_features(source)
    return engineered.dropna(subset=FEATURES).copy()


@lru_cache(maxsize=1)
def artifacts() -> dict:
    required = [
        "catboost_2week_advanced.pkl", "lightgbm_4week_advanced.pkl",
        "preprocessor_2week_advanced.pkl", "preprocessor_4week_advanced.pkl",
        "risk_thresholds.json", "model_report.json",
    ]
    missing = [name for name in required if not (MODEL_DIR / name).exists()]
    if missing:
        raise FileNotFoundError("Trained model files missing. Run: python backend/train_models.py")
    with (MODEL_DIR / "risk_thresholds.json").open(encoding="utf-8") as f:
        thresholds = json.load(f)
    with (MODEL_DIR / "model_report.json").open(encoding="utf-8") as f:
        report = json.load(f)
    return {
        "cat2": joblib.load(MODEL_DIR / required[0]),
        "lgb4": joblib.load(MODEL_DIR / required[1]),
        "pre2": joblib.load(MODEL_DIR / required[2]),
        "pre4": joblib.load(MODEL_DIR / required[3]),
        "thresholds": thresholds,
        "report": report,
    }


def _model_and_preprocessor(horizon: int, fitted: dict):
    if horizon == 2:
        return fitted["cat2"], fitted["pre2"], "Advanced CatBoost"
    return fitted["lgb4"], fitted["pre4"], "Advanced LightGBM"


def predict_frame(rows: pd.DataFrame, horizon: int, fitted: dict) -> np.ndarray:
    model, preprocessor, model_name = _model_and_preprocessor(horizon, fitted)
    transformed = preprocessor.transform(rows[FEATURES])
    if model_name == "Advanced CatBoost" and sparse.issparse(transformed):
        transformed = transformed.toarray()
    return np.maximum(np.asarray(model.predict(transformed), dtype=float).reshape(-1), 0.0)


def risk_tier(value: float, thresholds: dict) -> str:
    # Preserve the notebook's P75/P90 cutoffs (Moderate/High) and add a
    # P95 Critical tier to satisfy the hackathon's four-tier display.
    if value < thresholds["medium_from"]:
        return "Low"
    if value < thresholds["high_from"]:
        return "Medium"
    if value < thresholds["critical_from"]:
        return "High"
    return "Critical"


def importance_rows(model, preprocessor, top_n: int = 7) -> list[dict]:
    importance = getattr(model, "feature_importances_", None)
    if importance is None or len(importance) == 0:
        return []
    names = preprocessor.get_feature_names_out()
    total = float(np.sum(importance)) or 1.0
    order = np.argsort(importance)[::-1][:top_n]
    output = []
    for i in order:
        raw = str(names[i]).replace("categorical__", "").replace("numeric__", "")
        output.append({"feature": raw, "importance_pct": round(float(importance[i]) / total * 100, 1)})
    return output


@app.get("/")
def home():
    response = send_from_directory(FRONTEND, "index.html")
    response.headers["Cache-Control"] = "no-store, max-age=0"
    return response


@app.get("/frontend/<path:filename>")
def frontend_file(filename: str):
    response = send_from_directory(FRONTEND, filename)
    response.headers["Cache-Control"] = "no-store, max-age=0"
    return response


@app.get("/api/health")
def health():
    try:
        fitted = artifacts()
        return jsonify({"status": "ready", "models": fitted["report"]["selected_models"]})
    except FileNotFoundError as exc:
        return jsonify({"status": "setup_required", "message": str(exc)}), 503


@app.get("/api/options")
def options():
    try:
        df, fitted = feature_data(), artifacts()
    except (FileNotFoundError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 503
    cities = sorted(df["city"].dropna().astype(str).unique().tolist())
    city = request.args.get("city", cities[0] if cities else "")
    city_rows = df[df.city.astype(str) == city]
    wards = sorted(city_rows["ward_id"].dropna().astype(str).unique().tolist())
    ward = request.args.get("ward_id", wards[0] if wards else "")
    selected = city_rows[city_rows.ward_id.astype(str) == ward].sort_values("date")
    dates = selected.date.dt.strftime("%Y-%m-%d").tolist()
    report = fitted["report"]
    return jsonify({
        "cities": cities, "wards": wards, "dates": dates,
        "default_city": city, "default_ward": ward,
        "default_date": dates[-1] if dates else None,
        "horizons": [2, 4],
        "dataset_rows": report["dataset_rows"], "ward_count": report["ward_count"],
        "latest_date": report["latest_input_date"], "selected_models": report["selected_models"],
        "test_metrics": report["test_metrics"], "validation_metrics": report["validation_metrics"],
        "thresholds": fitted["thresholds"],
    })


@app.post("/api/predict")
def predict():
    body = request.get_json(silent=True) or {}
    try:
        df, fitted = feature_data(), artifacts()
    except (FileNotFoundError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 503
    try:
        horizon = int(body.get("horizon_weeks", 2))
    except (TypeError, ValueError):
        return jsonify({"error": "Forecast horizon must be 2 or 4 weeks."}), 400
    if horizon not in (2, 4):
        return jsonify({"error": "The notebook trained 2-week and 4-week models only."}), 400

    city, ward, date_value = str(body.get("city", "")), str(body.get("ward_id", "")), str(body.get("date", ""))
    city_rows = df[df.city.astype(str) == city]
    rows = city_rows[city_rows.ward_id.astype(str) == ward].sort_values("date")
    if rows.empty:
        return jsonify({"error": "Selected ward is unavailable in the prepared dataset."}), 404
    selected = rows[rows.date.dt.strftime("%Y-%m-%d") == date_value]
    row = (selected.iloc[-1] if not selected.empty else rows.iloc[-1]).copy()
    updates = body.get("weather", {})
    allowed = {key: updates[key] for key in ("avg_temp_c", "rainfall_mm", "relative_humidity_pct") if key in updates}
    try:
        row = apply_weather_scenario(row, allowed)
    except (TypeError, ValueError):
        return jsonify({"error": "Weather inputs must be numeric."}), 400

    model, preprocessor, model_name = _model_and_preprocessor(horizon, fitted)
    predicted = float(predict_frame(pd.DataFrame([row]), horizon, fitted)[0])
    tier_key = str(horizon)
    tier = risk_tier(predicted, fitted["thresholds"][tier_key])

    latest_by_ward = city_rows.groupby("ward_id", as_index=False, sort=False).tail(1).copy()
    # Score latest observations for every ward in the selected city in one model call.
    scores = predict_frame(latest_by_ward, horizon, fitted)
    bounds = fitted["thresholds"][tier_key]
    watchlist = []
    for (_, ward_row), score in zip(latest_by_ward.iterrows(), scores):
        watchlist.append({
            "ward_id": str(ward_row.ward_id), "predicted_cases": round(float(score), 2),
            "risk_tier": risk_tier(float(score), bounds), "date": ward_row.date.strftime("%Y-%m-%d"),
        })
    watchlist.sort(key=lambda item: item["predicted_cases"], reverse=True)
    selected_rank = next((i for i, item in enumerate(watchlist, 1) if item["ward_id"] == ward), None)
    history = rows[rows.date <= row.date].tail(26)
    test_metric = fitted["report"]["test_metrics"][f"{horizon}_week"]
    validation_metric = fitted["report"]["validation_metrics"][f"{horizon}_week"]

    return jsonify({
        "city": city, "ward_id": ward, "date": row.date.strftime("%Y-%m-%d"),
        "horizon_weeks": horizon, "model_name": model_name,
        "predicted_cases": round(predicted, 2), "risk_tier": tier,
        "weather": {key: float(row[key]) for key in ("avg_temp_c", "rainfall_mm", "relative_humidity_pct")},
        "drivers": importance_rows(model, preprocessor),
        "history": [{"date": r.date.strftime("%Y-%m-%d"), "cases": int(r.dengue_cases)} for r in history.itertuples()],
        "watchlist": watchlist[:10], "selected_rank": selected_rank,
        "test_metrics": test_metric, "validation_metrics": validation_metric,
        "thresholds": bounds,
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
