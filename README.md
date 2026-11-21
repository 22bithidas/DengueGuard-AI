# DengueGuard-AI



## Web Dashboard
This web app adds a browser frontend and Flask API to the supplied notebook project. Its forecast implementations follow the notebook's selected **Advanced CatBoost 2-week model** and **Advanced LightGBM 4-week model**, including the notebook's feature engineering, one-hot preprocessing, and chronological split. The original notebooks and result files remain as supplied.

## Run locally
```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
# Run once if model files are missing, or after replacing the dataset
.\.venv\Scripts\python.exe backend\train_models.py
.\.venv\Scripts\python.exe backend\app.py
```

## Dashboard
Choose a city, ward, observation week, and 2- or 4-week horizon. The dashboard shows the predicted weekly case count, a four-tier risk classification, model feature importance, trailing ward cases, city watchlist and validation/test metrics. The what-if form edits current-week temperature, rainfall and humidity and recalculates the notebook's dependent interaction features.


