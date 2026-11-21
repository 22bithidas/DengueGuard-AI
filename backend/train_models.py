"""Train the notebook-selected advanced CatBoost and LightGBM models."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder

from feature_engineering import FEATURES, TARGETS, build_features

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "hackathon_preli_dataset.xlsx"
MODEL_DIR = ROOT / "models"
CATEGORICAL = ["city", "ward_id"]


def chronological_split(model_data: pd.DataFrame, horizon: int):
    data = model_data.copy()
    data["target_date"] = data["date"] + pd.to_timedelta(horizon, unit="W")
    unique_dates = np.sort(data["date"].unique())
    train_idx = int(len(unique_dates) * 0.70)
    val_idx = int(len(unique_dates) * 0.85)
    train_end = pd.Timestamp(unique_dates[train_idx - 1])
    val_end = pd.Timestamp(unique_dates[val_idx - 1])

    train = data[(data.date <= train_end) & (data.target_date <= train_end)].copy()
    validation = data[
        (data.date > train_end) & (data.date <= val_end) & (data.target_date <= val_end)
    ].copy()
    test = data[data.date > val_end].copy()
    for frame in (train, validation, test):
        frame.drop(columns="target_date", inplace=True)
        frame.reset_index(drop=True, inplace=True)
    if not (train.date.max() < validation.date.min() < test.date.min()):
        raise ValueError(f"Invalid chronological split for {horizon}-week target")
    return train, validation, test, train_end, val_end


def make_preprocessor() -> ColumnTransformer:
    numeric = [name for name in FEATURES if name not in CATEGORICAL]
    return ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
            ("numeric", "passthrough", numeric),
        ]
    )


def as_dense(matrix):
    """CatBoost needs a dense matrix; sklearn may already return one."""
    return matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)


def metrics(y_true, y_pred) -> dict:
    return {
        "samples": int(len(y_true)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def main() -> dict:
    raw = pd.read_excel(DATA_PATH)
    engineered = build_features(raw)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    splits = {}
    for horizon, target in TARGETS.items():
        model_data = engineered.dropna(subset=FEATURES + [target]).copy()
        splits[horizon] = chronological_split(model_data, horizon)

    train2, val2, test2, train_end2, val_end2 = splits[2]
    train4, val4, test4, train_end4, val_end4 = splits[4]

    pre2, pre4 = make_preprocessor(), make_preprocessor()
    x_train2 = as_dense(pre2.fit_transform(train2[FEATURES]))
    x_val2 = as_dense(pre2.transform(val2[FEATURES]))
    x_test2 = as_dense(pre2.transform(test2[FEATURES]))
    x_train4 = pre4.fit_transform(train4[FEATURES])
    x_val4 = pre4.transform(val4[FEATURES])
    x_test4 = pre4.transform(test4[FEATURES])
    y_train2, y_val2, y_test2 = train2[TARGETS[2]].to_numpy(), val2[TARGETS[2]].to_numpy(), test2[TARGETS[2]].to_numpy()
    y_train4, y_val4, y_test4 = train4[TARGETS[4]].to_numpy(), val4[TARGETS[4]].to_numpy(), test4[TARGETS[4]].to_numpy()

    # Exact selected estimators and primary parameters from notebook cells 84 and 82.
    cat2 = CatBoostRegressor(
        iterations=500, depth=6, learning_rate=0.05, loss_function="RMSE",
        l2_leaf_reg=3, random_seed=42, verbose=False,
        allow_writing_files=False, thread_count=-1,
    )
    cat2.fit(x_train2, y_train2, verbose=False)
    lgb4 = LGBMRegressor(
        n_estimators=500, learning_rate=0.05, num_leaves=31, max_depth=-1,
        min_child_samples=20, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1,
        objective="regression", random_state=42, n_jobs=-1, verbosity=-1,
    )
    lgb4.fit(x_train4, y_train4)

    val_pred2 = np.maximum(cat2.predict(x_val2), 0)
    val_pred4 = np.maximum(lgb4.predict(x_val4), 0)
    test_pred2 = np.maximum(cat2.predict(x_test2), 0)
    test_pred4 = np.maximum(lgb4.predict(x_test4), 0)

    joblib.dump(cat2, MODEL_DIR / "catboost_2week_advanced.pkl")
    joblib.dump(lgb4, MODEL_DIR / "lightgbm_4week_advanced.pkl")
    joblib.dump(pre2, MODEL_DIR / "preprocessor_2week_advanced.pkl")
    joblib.dump(pre4, MODEL_DIR / "preprocessor_4week_advanced.pkl")

    # Preserve the notebook's q75/q90 case cutoffs, and derive q95 for the
    # extra Critical tier required by the hackathon brief.
    supplied_thresholds_path = ROOT / "risk_classification_thresholds.json"
    supplied = json.loads(supplied_thresholds_path.read_text(encoding="utf-8")) if supplied_thresholds_path.exists() else {}
    thresholds = {}
    for horizon, y_train in [(2, y_train2), (4, y_train4)]:
        key = f"{horizon}_week"
        base = supplied.get(key, {})
        moderate = float(base.get("moderate_threshold", np.percentile(y_train, 75)))
        high = float(base.get("high_threshold", np.percentile(y_train, 90)))
        critical = float(np.percentile(y_train, 95))
        thresholds[str(horizon)] = {
            "low_below": moderate,
            "medium_from": moderate,
            "high_from": high,
            "critical_from": max(high, critical),
            "notebook_moderate_threshold": moderate,
            "notebook_high_threshold": high,
        }
    (MODEL_DIR / "risk_thresholds.json").write_text(json.dumps(thresholds, indent=2), encoding="utf-8")

    report = {
        "selected_models": {"2_week": "Advanced CatBoost", "4_week": "Advanced LightGBM"},
        "feature_count": len(FEATURES),
        "features": FEATURES,
        "split": "chronological 70/15/15 with target-date boundary protection",
        "train_end": {"2_week": train_end2.strftime("%Y-%m-%d"), "4_week": train_end4.strftime("%Y-%m-%d")},
        "validation_end": {"2_week": val_end2.strftime("%Y-%m-%d"), "4_week": val_end4.strftime("%Y-%m-%d")},
        "validation_metrics": {"2_week": metrics(y_val2, val_pred2), "4_week": metrics(y_val4, val_pred4)},
        "test_metrics": {"2_week": metrics(y_test2, test_pred2), "4_week": metrics(y_test4, test_pred4)},
        "dataset_rows": int(len(raw)),
        "ward_count": int(raw["ward_id"].nunique()),
        "latest_input_date": pd.to_datetime(raw["date"]).max().strftime("%Y-%m-%d"),
    }
    (MODEL_DIR / "model_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"selected_models": report["selected_models"], "test_metrics": report["test_metrics"], "models_saved_to": str(MODEL_DIR)}, indent=2))
    return report


if __name__ == "__main__":
    main()
