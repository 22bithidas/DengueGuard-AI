# DengueGuard-AI

A Streamlit dashboard for ward-level dengue early-warning scenarios. It keeps the notebook-selected models: Advanced CatBoost for the 2-week horizon and Advanced LightGBM for the 4-week horizon. The existing model artifacts, preprocessing pipelines, thresholds, and model report are used directly; launching the dashboard does not retrain or replace the models.

## Run locally on Windows

Use Python 3.12 from the repository root in PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

Open the Local URL printed by Streamlit, usually `http://localhost:8501`.

The saved artifacts are in `models/`; the source workbook is `data/hackathon_preli_dataset.xlsx`. If you replace the workbook or need to retrain, run `python backend/train_models.py` before launching the app.

## Deploy with Streamlit Community Cloud

1. Commit and push the local Streamlit changes and required `data/` and `models/` files to your GitHub branch. Nothing is pushed automatically from this local folder.
2. Sign in at <https://share.streamlit.io/> with GitHub and choose **Create app**.
3. Select repository `22bithidas/DengueGuard-AI`, branch `main` (or the branch you pushed), and entrypoint `streamlit_app.py`.
4. In **Advanced settings**, choose Python 3.12, then deploy.

Community Cloud installs packages listed in `requirements.txt`. Keep the workbook and all files under `models/` in the repository because the app loads them at runtime.

## Dashboard features

- Select city, ward, observation week, and a 2- or 4-week forecast horizon.
- Explore an optional weather what-if using the same saved model.
- View predicted cases, risk tier, case history, feature importance, ward watchlist, and validation/test metrics.

## Model implementation

`backend/feature_engineering.py` reproduces the notebook's 25 original inputs plus 13 advanced features. `backend/train_models.py` trains and saves the selected estimators and preprocessors. `streamlit_app.py` loads the saved CatBoost, LightGBM, preprocessors, risk thresholds, and model report.

The dataset ends on 29 December 2025. The app does not use live weather or case feeds and is not an official public-health alert or clinical tool.

## Retained Flask implementation

The earlier Flask files and frontend are retained in the repository. The Streamlit entrypoint is now `streamlit_app.py`; Community Cloud should use that file.

