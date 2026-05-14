# Heart Disease Prediction Project

A comprehensive machine learning project for predicting heart disease using the UCI Heart Disease Dataset.

## Project Structure

```
Projectprototype/
├── data/                    # Dataset files
├── models/                  # Saved trained models
├── notebooks/              # Jupyter notebooks
│   └── heart_disease_analysis.ipynb
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Setup Instructions

### 1. Create Virtual Environment

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 2. Install Dependencies

```powershell
pip install -r requirements.txt
```

### 3. Run the Analysis

```powershell
jupyter notebook notebooks/heart_disease_analysis.ipynb
```

### 4. Train Models From Script

```powershell
python src/train.py
```

This generates:
- `models/best_pipeline.joblib`
- `models/*_pipeline.joblib` (all trained model pipelines)
- `models/metrics.json` (includes Recall, F1, Confusion Matrix, ROC-AUC, and imbalance analysis)

### 4.1 Evaluate All Models + Generate Graphs

```powershell
python src/evaluate_models.py
```

This generates paper-ready artifacts in `results/`:
- `results/model_metrics.csv`
- `results/evaluation_summary.json`
- `results/figures/accuracy_comparison.png`
- `results/figures/metrics_comparison.png`
- `results/figures/confusion_matrices.png`
- `results/figures/roc_curves.png` (when probability scores are available)

### 4.2 Run Kaggle Cardio Dataset Experiment (Separate Folder)

```powershell
python src/run_cardio_kaggle_experiment.py
```

This downloads `sulianova/cardiovascular-disease-dataset` and stores all artifacts in:
- `experiments/cardio_kaggle/models/`
- `experiments/cardio_kaggle/results/`

Generated outputs include:
- per-model pipelines + `best_pipeline.joblib`
- `model_metrics.csv`
- `evaluation_summary.json`
- accuracy, metric, confusion matrix, and ROC plots under `results/figures/`

### 5. Run Streamlit App

```powershell
streamlit run app_streamlit.py
```

### 6. Run FastAPI Inference API

```powershell
uvicorn api_fastapi:app --reload
```

Endpoints:
- `GET /health`
- `POST /predict`

### 7. Run Web Frontend (HTML/CSS/JS)

From project root, serve the `frontend` folder in a separate terminal:

```powershell
python -m http.server 5500 --directory frontend
```

Then open:
- `http://127.0.0.1:5500`

Make sure FastAPI is already running at `http://127.0.0.1:8000`.

## Dataset

- **Source:** UCI Machine Learning Repository
- **File:** `processed.cleveland.data`
- **Features:** 13 clinical features + 1 target variable
- **Target:** Binary classification (presence/absence of heart disease)

## Models Trained

1. **Logistic Regression** (Baseline)
2. **K-Nearest Neighbors**
3. **Support Vector Machine**
4. **Random Forest**
5. **XGBoost** (if installed)

## Key Findings

Results will be documented after running the analysis.

## Dependencies

- Python 3.8+
- pandas, numpy, matplotlib, seaborn
- scikit-learn, joblib
- jupyter

## License

Educational project for learning purposes.
