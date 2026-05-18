"""
Standalone test to reproduce & debug the LIME truncnorm error.
Run: python test_lime.py
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import lime
import lime.lime_tabular
import joblib

from src.data_utils import load_cleveland

# --- 1. Load training data ---
print("[1] Loading Cleveland data...")
df = load_cleveland(PROJECT_ROOT / "data" / "processed.cleveland.data")
X_train = df.drop(columns=["target"])
print(f"    Shape: {X_train.shape}")
print(f"    Dtypes:\n{X_train.dtypes}")
# CRITICAL: drop NaN rows first — LIME cannot handle NaN
X_train = X_train.dropna()
print(f"    After dropna: {X_train.shape}")
print(f"    Any NaN? {X_train.isna().any().any()}")
print(f"    Unique counts per column:")
for col in X_train.columns:
    print(f"      {col}: {sorted(X_train[col].dropna().unique())}")

# --- 2. Build LIME explainer with categorical features ---
print("\n[2] Building LIME explainer...")
categorical_names = {
    X_train.columns.get_loc("sex"):     {0: "Female",          1: "Male"},
    X_train.columns.get_loc("cp"):      {1: "Typical angina",  2: "Atypical angina",
                                          3: "Non-anginal pain", 4: "Asymptomatic"},
    X_train.columns.get_loc("fbs"):     {0: "No",  1: "Yes"},
    X_train.columns.get_loc("restecg"): {0: "Normal", 1: "ST-T abnormality",
                                          2: "LV hypertrophy"},
    X_train.columns.get_loc("exang"):   {0: "No",  1: "Yes"},
    X_train.columns.get_loc("slope"):   {1: "Upsloping", 2: "Flat", 3: "Downsloping"},
    X_train.columns.get_loc("ca"):      {0: "0", 1: "1", 2: "2", 3: "3"},
    X_train.columns.get_loc("thal"):    {3: "Normal", 6: "Fixed defect",
                                          7: "Reversable defect"},
}
categorical_feature_indices = list(categorical_names.keys())
print(f"    Categorical feature indices: {categorical_feature_indices}")

training_data = X_train.values.astype(float)
print(f"    Training data dtype: {training_data.dtype}")
print(f"    Any NaN in training data? {np.isnan(training_data).any()}")
print(f"    Any Inf in training data? {np.isinf(training_data).any()}")

explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train.values.astype(float),
    feature_names=X_train.columns.tolist(),
    class_names=["No Disease", "Disease"],
    categorical_features=categorical_feature_indices,
    categorical_names=categorical_names,
    mode="classification",
    discretize_continuous=True,
)
print("    Explainer created OK.")

# --- 3. Load model ---
print("\n[3] Loading best model...")
MODEL_PATH = PROJECT_ROOT / "results" / "cleveland" / "models" / "best_pipeline.joblib"
model = joblib.load(MODEL_PATH)
print(f"    Model loaded: {type(model).__name__}")

# --- 4. Create a sample input (same as what the form would produce) ---
sample_input = {
    "age": 55,
    "sex": 1,
    "cp": 2,     # 1-indexed in Cleveland: 2=Atypical angina
    "trestbps": 130,
    "chol": 250,
    "fbs": 0,
    "restecg": 0,
    "thalach": 150,
    "exang": 0,
    "oldpeak": 1.0,
    "slope": 2,  # 1-indexed in Cleveland: 2=Flat
    "ca": 0,
    "thal": 3,
}
X_user = pd.DataFrame([sample_input])
data_row = X_user.values[0].astype(float)
print(f"\n[4] User data row: {data_row}")
print(f"    Data row dtype: {data_row.dtype}")

# --- 5. Define predict_fn ---
def predict_fn(x_numpy):
    df_x = pd.DataFrame(x_numpy, columns=X_user.columns)
    return model.predict_proba(df_x)

# Quick sanity check on predict_fn
print("\n[5] Testing predict_fn with 3 rows...")
test_input = np.tile(data_row, (3, 1))
try:
    proba_out = predict_fn(test_input)
    print(f"    predict_fn output shape: {proba_out.shape} — OK")
except Exception as e:
    print(f"    predict_fn FAILED: {e}")
    sys.exit(1)

# --- 6. Run LIME explain_instance ---
print("\n[6] Running explain_instance...")
try:
    exp = explainer.explain_instance(
        data_row=data_row,
        predict_fn=predict_fn,
        num_features=10
    )
    print("    explain_instance OK!")
    print("    Top features:")
    for feat, weight in exp.as_list():
        print(f"      {feat}: {weight:.4f}")
except Exception as e:
    import traceback
    print(f"\n    FAILED with: {e}")
    traceback.print_exc()

print("\nDone.")
