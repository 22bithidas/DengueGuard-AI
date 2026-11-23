# DengueGuard-AI

## 👥 Team Contributions

DengueGuard AI is developed collaboratively by a three-member team, with responsibilities covering Machine Learning, Backend Development, and Frontend Dashboard Development.

### Member 1 (Bithi Das) - Machine Learning Developer
**Responsibility: ML Model Training & Evaluation**

- Dataset exploration, preprocessing, and feature engineering.
- Training and hyperparameter tuning of CatBoost and LightGBM models.
- Model comparison and final model selection.
- 2-week and 4-week dengue forecasting.
- Ward-wise dengue prediction generation.
- Dengue risk classification and Top-10 ward watchlist generation.
- Model evaluation and performance reporting.
- Preparation of ML models and supporting files for backend integration.

### Member 2 (Rakesh Das) — Backend Developer
**Responsibility: Backend Development & API Implementation**

- Backend architecture and development.
- API development and endpoint implementation.
- Prediction request and response handling.
- Backend-side data processing.
- HTML-based dashboard implementation.
- Support for machine learning model integration.

### Member 3 (Riddhi Chowdhury) — Full-Stack Developer
**Responsibility: Flask Dashboard & Backend Development**

- Backend development and data processing.
- flask based interactive dashboard development.
- Dashboard layout and user interface.
- Data visualization using charts and graphs.
- Ward-wise dengue prediction display.
- Dengue risk-level visualization.
- Backend and dashboard integration support.

Deployment Site Link: https://dengueguard-ai.onrender.com/
### API link: https://dengueguard-ai.onrender.com/api/options

### Collaborative Development

All three team members contribute to building an integrated DengueGuard AI system by combining machine learning predictions, backend services, and interactive data visualization.

## Web Dashboard
This web app adds a browser frontend and Flask API to the supplied notebook project. Its forecast implementations follow the notebook's selected **Advanced CatBoost 2-week model** and **Advanced LightGBM 4-week model**, including the notebook's feature engineering, one-hot preprocessing, and chronological split. The original notebooks and result files remain as supplied.

## Run locally
```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
# Run once if model files are missing, or after replacing the dataset
.\.venv\Scripts\python.exe backend\train_models.py
.\.venv\Scripts\python.exe backend\export_static_dashboard.py
.\.venv\Scripts\python.exe backend\app.py
```

## Dashboard
Choose a city, ward, observation week, and 2- or 4-week horizon. The dashboard shows the predicted weekly case count, a four-tier risk classification, model feature importance, trailing ward cases, city watchlist and validation/test metrics. The static dashboard's forecasts are precomputed by the same saved CatBoost and LightGBM models from the supplied dataset snapshot. This keeps the results available when the Flask API is asleep. The API remains available for live inference and what-if requests.

## Always-available static dashboard on Render
Render's free web service can sleep when idle. To give reviewers a dashboard that does not depend on the API waking up, deploy the static dashboard as a separate Render Static Site:

1. In Render, choose **New → Static Site** and connect this GitHub repository on the `main` branch.
2. Leave **Root Directory** blank.
3. Set **Build Command** to `pip install -r requirements.txt && python backend/export_static_dashboard.py`.
4. Set **Publish Directory** to `frontend`.
5. Deploy and submit the new static site's URL to reviewers. The existing Flask web service and API can remain available separately.

The build runs both saved model artifacts over the prepared dataset and writes `frontend/dashboard-data.json`. The static page then reads that file directly from Render's CDN. When the dataset or model artifacts change, a new static-site deploy regenerates the snapshot. The static dashboard does not provide interactive weather what-if scoring; that remains in the Flask API.

For Blueprint deployments, `render.yaml` also defines the separate static site and a build step for generating its dashboard data.


