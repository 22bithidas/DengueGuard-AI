# DengueGuard-AI




## Web Dashboard

This web app adds a browser frontend and Flask API to the supplied notebook project. Its forecast implementations follow the notebook's selected **Advanced CatBoost 2-week model** and **Advanced LightGBM 4-week model**, including the notebook's feature engineering, one-hot preprocessing, and chronological split. The original notebooks and result files remain as supplied.

## Run locally

Python 3.12 is recommended. The CatBoost dependency in this project does not install cleanly under Python 3.15 in the current Windows setup. Run these commands from the repository root in PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
# Run once if model files are missing, or after replacing the dataset
.\.venv\Scripts\python.exe backend\train_models.py
.\.venv\Scripts\python.exe backend\app.py
```

Open `http://127.0.0.1:5000`. The first training command saves the selected estimators and preprocessors under `models/`; the API loads them once per process and caches the engineered workbook. If you replace the workbook, retrain before starting the app.

## Dashboard

Choose a city, ward, observation week, and 2- or 4-week horizon. The dashboard shows the predicted weekly case count, a four-tier risk classification, model feature importance, trailing ward cases, city watchlist and validation/test metrics. The what-if form edits current-week temperature, rainfall and humidity and recalculates the notebook's dependent interaction features.

The source dataset ends on 29 December 2025. This prototype gives historical-snapshot scenarios, not live weather forecasts or official public-health alerts. No external prediction or LLM API is called.

## Model and source alignment

`backend/feature_engineering.py` reproduces the notebook's 25 original inputs plus 13 advanced features: dengue lags 12/52, shifted dengue rolling means (4/8/12 weeks), seasonal sine/cosine, shifted rainfall means (4/8 weeks), and four temperature/rainfall/humidity interactions. The 2-week horizon uses the selected advanced CatBoost model; 4 weeks uses selected advanced LightGBM. Estimator parameters are in `backend/train_models.py`.

The repository includes the saved estimator and preprocessor artifacts under `models/`, along with the notebook-reported metrics and risk thresholds. To retrain them after changing the dataset, run `python backend/train_models.py`.

## AI and external service disclosure

- AI assistant used for the web integration: OpenAI Codex.
- Model notebook: supplied by the team; add any AI assistants used to create it to this disclosure before submission.
- Application APIs: none; all inference runs locally.

## Three-minute demo checklist

Record one uncut screen capture (maximum 3:00), with voiceover and no slides: start the localhost app, select a ward and horizon, explain the model tier and feature drivers, edit weather assumptions, and show chronological validation metrics.
## Deploy to Render

The repository includes `render.yaml` with the Flask web service configuration. From Render, create a new **Blueprint** and connect this repository; Render will read the manifest and configure the Python build, Gunicorn start command, and `/api/health` check. The service must include the `backend`, `frontend`, `data`, and `models` directories together at the repository root because the Flask app loads them via relative project paths.

Before creating the service, commit and push these local deployment changes and any other intended project files to your GitHub branch. This project pins the Python model-library versions to the environment used to load the checked-in model artifacts; `.python-version` selects Python 3.12. Do not delete the model artifacts or dataset. After Render reports the service as live, open its `onrender.com` URL and confirm `/api/health` returns `"status":"ready"`, then test a prediction in the dashboard.


