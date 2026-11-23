"""Export model-scored dashboard data for an always-available static site.

The forecast values come from the same trained CatBoost and LightGBM artifacts
used by the Flask API. The exporter runs during a static-site build.
"""
from __future__ import annotations

import json
from pathlib import Path

from app import artifacts, feature_data, importance_rows, predict_frame, risk_tier

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "frontend" / "dashboard-data.json"


def main() -> None:
    frame = feature_data().copy()
    fitted = artifacts()
    report = fitted["report"]

    for horizon in (2, 4):
        scores = predict_frame(frame, horizon, fitted)
        frame[f"forecast_{horizon}"] = scores
        bounds = fitted["thresholds"][str(horizon)]
        frame[f"tier_{horizon}"] = [risk_tier(float(score), bounds) for score in scores]

    cities: dict[str, dict] = {}
    for (city, ward), rows in frame.groupby(["city", "ward_id"], sort=True):
        series = []
        for row in rows.sort_values("date").itertuples():
            series.append({
                "date": row.date.strftime("%Y-%m-%d"),
                "cases": int(row.dengue_cases),
                "weather": {
                    "avg_temp_c": float(row.avg_temp_c),
                    "rainfall_mm": float(row.rainfall_mm),
                    "relative_humidity_pct": float(row.relative_humidity_pct),
                },
                "forecast_2": round(float(row.forecast_2), 2),
                "tier_2": row.tier_2,
                "forecast_4": round(float(row.forecast_4), 2),
                "tier_4": row.tier_4,
            })
            cities.setdefault(str(city), {"wards": {}})["wards"][str(ward)] = series

    watchlists: dict[str, dict[str, list[dict]]] = {}
    ranks: dict[str, dict[str, dict[str, int]]] = {}
    for city, city_rows in frame.groupby("city", sort=True):
        latest = city_rows.sort_values("date").groupby("ward_id", sort=True).tail(1).copy()
        watchlists[str(city)] = {}
        ranks[str(city)] = {}
        for horizon in (2, 4):
            ordered = latest.sort_values(f"forecast_{horizon}", ascending=False)
            watchlists[str(city)][str(horizon)] = [
                {
                    "ward_id": str(row.ward_id),
                    "predicted_cases": round(float(getattr(row, f"forecast_{horizon}")), 2),
                    "risk_tier": str(getattr(row, f"tier_{horizon}")),
                    "date": row.date.strftime("%Y-%m-%d"),
                }
                for row in ordered.head(10).itertuples()
            ]
            ranks[str(city)][str(horizon)] = {
                str(row.ward_id): rank
                for rank, row in enumerate(ordered.itertuples(), start=1)
            }

    payload = {
        "generated_from": "trained repository model artifacts",
        "cities": cities,
        "dataset_rows": int(report["dataset_rows"]),
        "ward_count": int(report["ward_count"]),
        "latest_date": str(report["latest_input_date"]),
        "selected_models": report["selected_models"],
        "test_metrics": report["test_metrics"],
        "validation_metrics": report["validation_metrics"],
        "drivers": {
            "2": importance_rows(fitted["cat2"], fitted["pre2"]),
            "4": importance_rows(fitted["lgb4"], fitted["pre4"]),
        },
        "thresholds": fitted["thresholds"],
        "watchlists": watchlists,
        "ranks": ranks,
    }
    OUTPUT.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    print(f"Exported static dashboard data: {OUTPUT} ({OUTPUT.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
