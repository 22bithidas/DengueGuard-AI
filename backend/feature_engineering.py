"""Notebook-aligned feature engineering for DengueGuard AI."""
from __future__ import annotations

import numpy as np
import pandas as pd

GROUPS = ["city", "ward_id"]
TARGETS = {2: "target_2week", 4: "target_4week"}

# Kept in the same order as the notebook's feature_columns and advanced additions.
ORIGINAL_FEATURES = [
    "city", "ward_id", "population_density_sqkm", "water_logging_risk_index",
    "sanitation_score", "avg_temp_c", "max_temp_c", "min_temp_c", "rainfall_mm",
    "relative_humidity_pct", "dengue_cases", "dengue_lag_1", "dengue_lag_2",
    "dengue_lag_4", "dengue_lag_8", "rainfall_sum_2w", "rainfall_sum_4w",
    "avg_temp_c_lag_1", "avg_temp_c_lag_2", "relative_humidity_pct_lag_1",
    "relative_humidity_pct_lag_2", "rainfall_mm_lag_1", "rainfall_mm_lag_2",
    "month", "week_of_year",
]
ADVANCED_ADDITIONS = [
    "dengue_lag_12", "dengue_lag_52", "dengue_roll_4", "dengue_roll_8",
    "dengue_roll_12", "week_sin", "week_cos", "rainfall_roll_4",
    "rainfall_roll_8", "temp_humidity_interaction", "rainfall_humidity_interaction",
    "temp_rainfall_interaction", "temp_humidity_rainfall",
]
FEATURES = ORIGINAL_FEATURES + ADVANCED_ADDITIONS


def build_features(source: pd.DataFrame) -> pd.DataFrame:
    """Reproduce the notebook's grouped feature engineering without Colab paths."""
    df = source.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "city", "ward_id"])
    numeric = [
        "population_density_sqkm", "water_logging_risk_index", "sanitation_score",
        "avg_temp_c", "max_temp_c", "min_temp_c", "rainfall_mm",
        "relative_humidity_pct", "dengue_cases",
    ]
    for column in numeric:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.drop_duplicates(subset=["city", "ward_id", "date"], keep="last")
    df = df.sort_values(GROUPS + ["date"]).reset_index(drop=True)
    grouped = df.groupby(GROUPS, sort=False)

    for horizon, target in TARGETS.items():
        df[target] = grouped["dengue_cases"].shift(-horizon)

    for lag in [1, 2, 4, 8]:
        df[f"dengue_lag_{lag}"] = grouped["dengue_cases"].shift(lag)
    # Notebook rolling sums include the current observation week.
    for window in [2, 4]:
        df[f"rainfall_sum_{window}w"] = grouped["rainfall_mm"].transform(
            lambda values, n=window: values.rolling(window=n, min_periods=n).sum()
        )
    for column in ["avg_temp_c", "relative_humidity_pct", "rainfall_mm"]:
        for lag in [1, 2]:
            df[f"{column}_lag_{lag}"] = grouped[column].shift(lag)

    df["month"] = df["date"].dt.month
    week = df["date"].dt.isocalendar().week.astype(int)
    df["week_of_year"] = week

    df["dengue_lag_12"] = grouped["dengue_cases"].shift(12)
    df["dengue_lag_52"] = grouped["dengue_cases"].shift(52)
    for window in [4, 8, 12]:
        df[f"dengue_roll_{window}"] = grouped["dengue_cases"].transform(
            lambda values, n=window: values.shift(1).rolling(window=n, min_periods=n).mean()
        )
    df["week_sin"] = np.sin(2 * np.pi * week / 52.18)
    df["week_cos"] = np.cos(2 * np.pi * week / 52.18)

    rainfall = df.groupby(GROUPS, sort=False)["rainfall_mm"]
    for window in [4, 8]:
        df[f"rainfall_roll_{window}"] = rainfall.transform(
            lambda values, n=window: values.shift(1).rolling(window=n, min_periods=n).mean()
        )
    df["temp_humidity_interaction"] = df["avg_temp_c"] * df["relative_humidity_pct"]
    df["rainfall_humidity_interaction"] = df["rainfall_mm"] * df["relative_humidity_pct"]
    df["temp_rainfall_interaction"] = df["avg_temp_c"] * df["rainfall_mm"]
    df["temp_humidity_rainfall"] = (
        df["avg_temp_c"] * df["relative_humidity_pct"] * df["rainfall_mm"]
    )
    return df.sort_values(["date", "city", "ward_id"]).reset_index(drop=True)


def apply_weather_scenario(row: pd.Series, updates: dict) -> pd.Series:
    """Update the notebook-derived same-week interaction features consistently."""
    result = row.copy()
    prior_rain = float(result["rainfall_mm"])
    for key in ["avg_temp_c", "rainfall_mm", "relative_humidity_pct"]:
        if key in updates:
            result[key] = float(updates[key])
    rain_delta = float(result["rainfall_mm"]) - prior_rain
    result["rainfall_sum_2w"] = float(result["rainfall_sum_2w"]) + rain_delta
    result["rainfall_sum_4w"] = float(result["rainfall_sum_4w"]) + rain_delta
    result["temp_humidity_interaction"] = result["avg_temp_c"] * result["relative_humidity_pct"]
    result["rainfall_humidity_interaction"] = result["rainfall_mm"] * result["relative_humidity_pct"]
    result["temp_rainfall_interaction"] = result["avg_temp_c"] * result["rainfall_mm"]
    result["temp_humidity_rainfall"] = (
        result["avg_temp_c"] * result["relative_humidity_pct"] * result["rainfall_mm"]
    )
    return result
