"""DengueGuard AI interactive dashboard, powered by the notebook-selected models."""
from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from scipy import sparse

ROOT = Path(__file__).resolve().parent
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from feature_engineering import FEATURES, apply_weather_scenario, build_features  # noqa: E402

MODEL_DIR = ROOT / "models"
DATA_PATH = ROOT / "data" / "hackathon_preli_dataset.xlsx"

st.set_page_config(page_title="DengueGuard AI", page_icon="🦟", layout="wide")

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.8rem; padding-bottom: 2rem; max-width: 1500px;}
      [data-testid="stMetric"] {background:#f1f7f4; border:1px solid #dce9e3; padding:14px 16px; border-radius:12px;}
      [data-testid="stMetricLabel"] {color:#49645b;}
      h1, h2, h3 {color:#123b31;}
      div[data-testid="stExpander"] {border-color:#dce9e3; border-radius:10px;}
      .small-note {color:#62736f; font-size:0.88rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner="Preparing the ward dataset...")
def load_feature_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset file not found: {DATA_PATH}")
    raw = pd.read_excel(DATA_PATH)
    prepared = build_features(raw)
    return prepared.dropna(subset=FEATURES).copy()


@st.cache_resource(show_spinner="Loading the saved dengue models...")
def load_artifacts() -> dict:
    names = {
        "cat2": "catboost_2week_advanced.pkl",
        "lgb4": "lightgbm_4week_advanced.pkl",
        "pre2": "preprocessor_2week_advanced.pkl",
        "pre4": "preprocessor_4week_advanced.pkl",
        "thresholds": "risk_thresholds.json",
        "report": "model_report.json",
    }
    missing = [filename for filename in names.values() if not (MODEL_DIR / filename).exists()]
    if missing:
        raise FileNotFoundError("Model files are missing from models/: " + ", ".join(missing))
    loaded = {}
    for key, filename in names.items():
        path = MODEL_DIR / filename
        if filename.endswith(".pkl"):
            loaded[key] = joblib.load(path)
        else:
            loaded[key] = json.loads(path.read_text(encoding="utf-8"))
    return loaded


def selected_pipeline(horizon: int, fitted: dict):
    if horizon == 2:
        return fitted["cat2"], fitted["pre2"], "Advanced CatBoost"
    return fitted["lgb4"], fitted["pre4"], "Advanced LightGBM"


def predict_rows(rows: pd.DataFrame, horizon: int, fitted: dict) -> np.ndarray:
    model, preprocessor, _ = selected_pipeline(horizon, fitted)
    transformed = preprocessor.transform(rows[FEATURES])
    if horizon == 2 and sparse.issparse(transformed):
        transformed = transformed.toarray()
    return np.maximum(np.asarray(model.predict(transformed), dtype=float).reshape(-1), 0.0)


def risk_tier(value: float, thresholds: dict) -> str:
    if value < thresholds["medium_from"]:
        return "Low"
    if value < thresholds["high_from"]:
        return "Medium"
    if value < thresholds["critical_from"]:
        return "High"
    return "Critical"


def feature_importance(model, preprocessor, limit: int = 8) -> pd.DataFrame:
    values = getattr(model, "feature_importances_", None)
    if values is None or not len(values):
        return pd.DataFrame(columns=["Feature", "Importance (%)"])
    names = preprocessor.get_feature_names_out()
    total = float(np.sum(values)) or 1.0
    indices = np.argsort(values)[::-1][:limit]
    return pd.DataFrame({
        "Feature": [str(names[i]).replace("categorical__", "").replace("numeric__", "").replace("_", " ").title() for i in indices],
        "Importance (%)": [round(float(values[i]) / total * 100, 1) for i in indices],
    })


def make_watchlist(city_rows: pd.DataFrame, horizon: int, fitted: dict) -> pd.DataFrame:
    latest = city_rows.groupby("ward_id", sort=False).tail(1).copy()
    predictions = predict_rows(latest, horizon, fitted)
    thresholds = fitted["thresholds"][str(horizon)]
    result = pd.DataFrame({
        "Ward": latest["ward_id"].astype(str).to_numpy(),
        "Forecast cases": np.round(predictions, 2),
        "Risk tier": [risk_tier(float(value), thresholds) for value in predictions],
        "Forecast week": (latest["date"] + pd.to_timedelta(horizon, unit="W")).dt.strftime("%Y-%m-%d").to_numpy(),
    })
    return result.sort_values("Forecast cases", ascending=False).reset_index(drop=True)


st.title("DengueGuard AI")
st.caption("Ward-level dengue early warning | Historical-snapshot scenario dashboard")

try:
    data = load_feature_data()
    fitted = load_artifacts()
except Exception as exc:
    st.error(f"The app could not load its data or model files: {exc}")
    st.info("Confirm that data/hackathon_preli_dataset.xlsx and all saved files under models/ are included in the repository.")
    st.stop()

report = fitted["report"]
cities = sorted(data["city"].dropna().astype(str).unique())
if not cities:
    st.error("The prepared dataset contains no usable cities.")
    st.stop()

with st.sidebar:
    st.header("Forecast settings")
    city = st.selectbox("City", cities)
    city_data = data[data["city"].astype(str) == city]
    wards = sorted(city_data["ward_id"].dropna().astype(str).unique())
    ward = st.selectbox("Ward", wards)
    ward_data = city_data[city_data["ward_id"].astype(str) == ward].sort_values("date")
    date_values = ward_data["date"].dt.strftime("%Y-%m-%d").tolist()[::-1]
    date_value = st.selectbox("Observation week", date_values)
    horizon = st.selectbox("Forecast lead time", [2, 4], format_func=lambda x: f"{x} weeks ahead")

    selected_rows = ward_data[ward_data["date"].dt.strftime("%Y-%m-%d") == date_value]
    row = (selected_rows.iloc[-1] if not selected_rows.empty else ward_data.iloc[-1]).copy()
    context = f"{city}|{ward}|{date_value}"
    weather_keys = {"avg_temp_c": "scenario_temp", "rainfall_mm": "scenario_rain", "relative_humidity_pct": "scenario_humidity"}
    if st.session_state.get("scenario_context") != context:
        for column, key in weather_keys.items():
            st.session_state[key] = float(row[column])
        st.session_state["scenario_context"] = context

    with st.expander("What-if weather inputs", expanded=False):
        st.caption("Adjust the selected week's weather and the same saved model will recalculate the outlook.")
        scenario_weather = {
            "avg_temp_c": st.number_input("Average temperature (C)", key="scenario_temp", step=0.1),
            "rainfall_mm": st.number_input("Rainfall (mm)", key="scenario_rain", min_value=0.0, step=1.0),
            "relative_humidity_pct": st.number_input("Relative humidity (%)", key="scenario_humidity", min_value=0.0, max_value=100.0, step=1.0),
        }
    st.divider()
    st.caption(f"{report['dataset_rows']:,} workbook rows | {report['ward_count']} wards")
    st.caption(f"Workbook through {report['latest_input_date']}")

try:
    scenario_row = apply_weather_scenario(row, scenario_weather)
    model, preprocessor, model_name = selected_pipeline(horizon, fitted)
    predicted = float(predict_rows(pd.DataFrame([scenario_row]), horizon, fitted)[0])
    thresholds = fitted["thresholds"][str(horizon)]
    tier = risk_tier(predicted, thresholds)
    history = ward_data[ward_data["date"] <= row["date"]].tail(26)
    watchlist = make_watchlist(city_data, horizon, fitted)
    rank_matches = watchlist.index[watchlist["Ward"] == str(ward)].tolist()
    selected_rank = rank_matches[0] + 1 if rank_matches else None
    test_metrics = report["test_metrics"][f"{horizon}_week"]
    validation_metrics = report["validation_metrics"][f"{horizon}_week"]
    drivers = feature_importance(model, preprocessor)
except Exception as exc:
    st.error(f"Could not generate the forecast: {exc}")
    st.stop()

st.markdown(f"### {city} / {ward}")
st.caption(f"Observation week: {date_value} | Model: {model_name} | Forecast lead: {horizon} weeks")

tab_outlook, tab_performance, tab_about = st.tabs(["Forecast overview", "Model performance", "About this prototype"])

with tab_outlook:
    left, right = st.columns([1.2, 1])
    with left:
        st.metric(f"Predicted dengue cases ({horizon} weeks)", f"{predicted:.2f}", help="Estimated weekly case count for the selected ward and observation week.")
        tier_color = {"Low": "🟢", "Medium": "🟡", "High": "🟠", "Critical": "🔴"}.get(tier, "")
        st.markdown(f"#### {tier_color} {tier} risk")
        st.caption(f"Model risk bands for this horizon: Low < {thresholds['medium_from']:.0f}; Medium < {thresholds['high_from']:.0f}; High < {thresholds['critical_from']:.0f}; Critical from {thresholds['critical_from']:.0f} cases.")
        st.markdown("**Model test results**")
        m1, m2, m3 = st.columns(3)
        m1.metric("Test MAE", f"{test_metrics['mae']:.2f}")
        m2.metric("Test RMSE", f"{test_metrics['rmse']:.2f}")
        m3.metric("Test R2", f"{test_metrics['r2']:.3f}")
        st.caption(f"Chronological holdout: {test_metrics['samples']:,} test rows. These scores describe past test data, not guaranteed future accuracy.")
    with right:
        st.markdown("#### Reported weekly dengue cases")
        chart_data = history[["date", "dengue_cases"]].rename(columns={"date": "Week", "dengue_cases": "Cases"}).set_index("Week")
        st.line_chart(chart_data, color="#20B99A", height=245)
        st.caption("Observed history for the selected ward, up to the selected observation week.")

    c1, c2 = st.columns([0.9, 1.1])
    with c1:
        st.markdown("#### Prediction drivers")
        if drivers.empty:
            st.info("Feature importance is not available for this model artifact.")
        else:
            st.bar_chart(drivers.set_index("Feature"), color="#20B99A", horizontal=True)
    with c2:
        st.markdown("#### Ward watchlist")
        st.caption(f"Top wards in {city} by {horizon}-week predicted cases. Selected ward rank: {f'#{selected_rank}' if selected_rank else 'not ranked'}.")
        forecast_weeks = watchlist["Forecast week"].dropna().unique().tolist()
        if len(forecast_weeks) == 1:
            st.caption(f"Forecast week: {forecast_weeks[0]}. All listed wards share this week in the source data.")
        else:
            st.caption("Forecast week is calculated per ward from its latest available observation.")
        st.dataframe(watchlist.drop(columns=["Forecast week"]).head(10), width="stretch", hide_index=True)

with tab_performance:
    st.markdown(f"### {horizon}-week model: {model_name}")
    metrics_rows = []
    for split in ["validation_metrics", "test_metrics"]:
        result = report[split][f"{horizon}_week"]
        metrics_rows.append({"Split": split.replace("_metrics", "").title(), "Rows": result["samples"], "MAE": result["mae"], "RMSE": result["rmse"], "R2": result["r2"]})
    st.dataframe(pd.DataFrame(metrics_rows).set_index("Split").style.format({"MAE": "{:.3f}", "RMSE": "{:.3f}", "R2": "{:.3f}"}), width="stretch")
    st.caption(f"Training split: {report['split']}. Train end: {report['train_end'][f'{horizon}_week']}; validation end: {report['validation_end'][f'{horizon}_week']}.")
    st.markdown("### Risk thresholds")
    st.write({"Medium begins": thresholds["medium_from"], "High begins": thresholds["high_from"], "Critical begins": thresholds["critical_from"]})
    st.caption("The notebook supplied the P75/P90 case cutoffs. The Critical band uses the extra high-end cutoff used by this prototype.")

with tab_about:
    st.markdown("### Data and model")
    st.write("This app uses the existing saved notebook-selected models: Advanced CatBoost for 2 weeks and Advanced LightGBM for 4 weeks. The dataset is historical and ends on 29 December 2025.")
    st.markdown("### Limitations")
    st.warning("This is a hackathon prototype for scenario exploration. It does not use a live weather feed or live case reporting and must not be treated as a clinical tool or official public-health alert.")

st.divider()
st.caption("DengueGuard AI | Historical-snapshot decision-support prototype | No external prediction API")

