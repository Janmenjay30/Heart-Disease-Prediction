# Heart Disease Prediction Project

A dual-pipeline machine learning project for predicting heart disease using two clinical datasets with five shared ML models.

## Project Overview

| | Pipeline 1: Cleveland | Pipeline 2: Framingham |
|---|---|---|
| **Goal** | Predict existing heart disease | Predict 10-year CHD risk |
| **Dataset** | UCI Cleveland (303 rows) | Framingham Heart Study (~4,200 rows) |
| **Target** | `target` (0/1) | `TenYearCHD` (0/1) |
| **Class Balance** | ~54% / 46% (balanced) | ~85% / 15% (imbalanced) |

**Shared across both pipelines:**
- **Models:** Logistic Regression, KNN, SVC, Random Forest, XGBoost
- **Metrics:** Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **Visualizations:** Accuracy bars, metrics comparison, confusion matrices, ROC curves

## Project Structure

```
Projectprototype/
├── src/                          # Source code
│   ├── shared.py                 # Shared models, evaluation, plots
│   ├── data_utils.py             # Dataset loaders (Cleveland + Framingham)
│   ├── train.py                  # Pipeline 1 — Cleveland training
│   ├── train_framingham.py       # Pipeline 2 — Framingham training
│   └── evaluate_models.py        # Evaluation script (supports both)
├── data/                         # Dataset files
│   └── processed.cleveland.data  # Cleveland dataset
├── results/
│   ├── cleveland/                # Pipeline 1 outputs
│   │   ├── models/               # Trained models
│   │   ├── model_metrics.csv
│   │   ├── evaluation_summary.json
│   │   └── figures/              # Plots (accuracy, metrics, confusion, ROC)
│   └── framingham/               # Pipeline 2 outputs
│       ├── models/               # Trained models
│       ├── model_metrics.csv
│       ├── evaluation_summary.json
│       └── figures/              # Plots (accuracy, metrics, confusion, ROC)
├── notebooks/                    # Jupyter notebooks for EDA
├── frontend/                     # HTML/CSS/JS web frontend
├── api_fastapi.py                # FastAPI prediction API
├── app_streamlit.py              # Streamlit web app
├── ModelReport.md                # Detailed model analysis report
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Setup

### 1. Create and Activate Virtual Environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install Dependencies

```powershell
pip install -r requirements.txt
```

### 3. Configure Kaggle Credentials (for Framingham dataset)

Copy your `kaggle.json` to `C:\Users\<YourUsername>\.kaggle\kaggle.json`. This is required for automatic download of the Framingham dataset.

## Running the Pipelines

### Pipeline 1 — Cleveland Dataset

Train the models:
```powershell
python src/train.py
```

Evaluate and generate plots:
```powershell
python src/evaluate_models.py
```

Outputs:
- Trained models → `results/cleveland/models/`
- Metrics & plots → `results/cleveland/`

### Pipeline 2 — Framingham Dataset

Train the models (downloads dataset automatically):
```powershell
python src/train_framingham.py
```

Evaluate and generate plots:
```powershell
python src/evaluate_models.py --pipeline framingham
```

Outputs:
- Trained models → `results/framingham/models/`
- Metrics & plots → `results/framingham/`

## Running the Web Applications

### FastAPI Inference API

```powershell
uvicorn api_fastapi:app --reload
```

- Health check: http://127.0.0.1:8000/health
- Predict endpoint: `POST http://127.0.0.1:8000/predict`
- Swagger docs: http://127.0.0.1:8000/docs

### Streamlit App

```powershell
streamlit run app_streamlit.py
```

Opens at http://localhost:8501

### HTML Frontend

```powershell
python -m http.server 5500 --directory frontend
```

Opens at http://127.0.0.1:5500 (requires FastAPI running on port 8000)

## Models Trained

1. **Logistic Regression** — Linear baseline, highly interpretable
2. **K-Nearest Neighbors** — Non-parametric, instance-based
3. **Support Vector Classifier** — Finds optimal decision boundary
4. **Random Forest** — Ensemble of 100 decision trees
5. **XGBoost** — Gradient boosting (requires `xgboost` package)

## Dataset Sources

- **Cleveland:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/heart+disease) — `processed.cleveland.data`
- **Framingham:** [Kaggle](https://www.kaggle.com/datasets/dileep070/heart-disease-prediction-using-logistic-regression) — Downloaded automatically via `kagglehub`

## License

Educational project for learning purposes.
