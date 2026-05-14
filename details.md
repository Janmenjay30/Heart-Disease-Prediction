# Project Report: Heart Disease Prediction

## 1. Project Overview
**Objective:** To develop a machine learning model capable of predicting the presence of heart disease in patients based on standard clinical attributes. The project prioritizes **Recall (Sensitivity)** to minimize false negatives, ensuring that patients with heart disease are not missed.

**Dataset:** UCI Machine Learning Repository - Cleveland Heart Disease Dataset (`processed.cleveland.data`).
- **Samples:** 303 patients.
- **Features:** 13 clinical attributes (age, sex, chest pain type, etc.).
- **Target:** Presence (1) or absence (0) of heart disease.

**Tech Stack:**
- **Language:** Python 3.13
- **Libraries:** `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `joblib`
- **Web Framework:** `Streamlit`

---

## 2. Data Pipeline & Preprocessing
The project implements a robust, reproducible pipeline using `scikit-learn`.

### Data Loading (`src/data_utils.py`)
- **Source:** Reads the raw CSV file `processed.cleveland.data`.
- **Cleaning:**
  - Replaces missing values denoted by `?` with `NaN`.
  - **Target Transformation:** The original dataset has target values 0 (no disease) and 1-4 (levels of disease). These are mapped to a binary format: `0` remains `0`, and `1-4` are mapped to `1`.

### Feature Engineering (`src/train.py`)
The preprocessing logic is encapsulated in a `ColumnTransformer` to prevent data leakage.

1.  **Numerical Features:**
    - **Columns:** `age`, `trestbps` (resting bp), `chol` (cholesterol), `thalach` (max heart rate), `oldpeak` (ST depression).
    - **Imputation:** Missing values filled with the **median**.
    - **Scaling:** Standardized using `StandardScaler` (zero mean, unit variance).

2.  **Categorical Features:**
    - **Columns:** `sex`, `cp` (chest pain), `fbs` (fasting blood sugar), `restecg`, `exang` (exercise angina), `slope`, `ca` (vessels), `thal` (thalassemia).
    - **Imputation:** Missing values filled with the **most frequent** value (mode).
    - **Encoding:** Transformed using `OneHotEncoder` (creates binary columns for each category).

### Train-Test Split
- **Ratio:** 80% Training, 20% Testing.
- **Stratification:** Stratified by `target` to maintain the same class balance in both sets.
- **Random State:** Fixed at `42` for reproducibility.

---

## 3. Model Development
Four different classification algorithms were trained and evaluated.

1.  **Logistic Regression:** A linear model that estimates probabilities using the logistic function.
    - *Configuration:* `max_iter=1000`
2.  **K-Nearest Neighbors (KNN):** Classifies based on the majority class of the 5 nearest neighbors.
    - *Configuration:* `n_neighbors=5`
3.  **Support Vector Classifier (SVC):** Finds the optimal hyperplane to separate classes.
    - *Configuration:* `probability=True` (to allow probability output)
4.  **Random Forest:** An ensemble of 100 decision trees.
    - *Configuration:* `n_estimators=100`, `random_state=42`

---

## 4. Evaluation Results
The models were evaluated on the unseen test set (20% of data). The primary metric for selection was **Recall**.

| Model | Accuracy | Precision | Recall | F1-Score | False Negatives |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | **88.5%** | 83.9% | **92.9%** | 88.1% | **2** |
| **SVC** | 88.5% | 83.9% | **92.9%** | 88.1% | 2 |
| **Random Forest** | 86.9% | 81.3% | **92.9%** | 86.7% | 2 |
| **KNN (k=5)** | 88.5% | **86.2%** | 89.3% | 87.7% | 3 |

**Key Findings:**
- **Logistic Regression** was selected as the best model. It tied for the highest Recall (92.9%) and Accuracy (88.5%) but is simpler and more interpretable than SVC or Random Forest.
- The model only missed **2 positive cases** (False Negatives) out of 28 positive cases in the test set.

---

## 5. Deployment Application
A user-friendly web application was built to demonstrate the model.

- **File:** `app_streamlit.py`
- **Functionality:**
  - Loads the trained pipeline (`models/best_pipeline.joblib`).
  - Provides a form for users to input all 13 clinical features.
  - Preprocesses the input on-the-fly using the saved pipeline.
  - Displays the **predicted probability** (e.g., "85.4% chance") and the **final classification** (Disease/No Disease).

---

## 6. Project Structure & Files
- **`src/`**: Contains source code modules.
  - `data_utils.py`: Functions for loading and cleaning data.
  - `train.py`: Main script for training models and saving artifacts.
- **`models/`**: Stores trained model artifacts.
  - `best_pipeline.joblib`: The serialized final model pipeline.
  - `metrics.json`: JSON file containing detailed performance metrics for all models.
- **`data/`**: Contains the dataset `processed.cleveland.data`.
- **`notebooks/`**: Jupyter notebooks for EDA and experiments.
- **`requirements.txt`**: List of Python dependencies.
- **`REPORT.md` / `ModelReport.md`**: Documentation and reports.
