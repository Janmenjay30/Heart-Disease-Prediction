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

## Dataset

- **Source:** UCI Machine Learning Repository
- **File:** `processed.cleveland.data`
- **Features:** 13 clinical features + 1 target variable
- **Target:** Binary classification (presence/absence of heart disease)

## Models Trained

1. **Logistic Regression** (Baseline)
2. **K-Nearest Neighbors**
3. **Support Vector Machine**
4. **Random Forest** (Best performing)

## Key Findings

Results will be documented after running the analysis.

## Dependencies

- Python 3.8+
- pandas, numpy, matplotlib, seaborn
- scikit-learn, joblib
- jupyter

## License

Educational project for learning purposes.
