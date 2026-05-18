"""
train_framingham.py — Framingham CHD prediction pipeline.

Architecture (zero data-leakage guarantee):
  Every model is wrapped in an imblearn.Pipeline:
    1. ColumnTransformer  – median impute + scale (numeric)
                         – most-frequent impute + OHE  (categorical)
    2. SMOTE(sampling_strategy=0.33)  ← synthetic samples ONLY on train folds
    3. Classifier (LR / KNN / SVC / RandomForest / XGBoost / ANN)

  RandomForest is additionally tuned via GridSearchCV (cv=5, scoring='f1')
  so that SMOTE never contaminates validation folds during search.
"""

import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False
    warnings.warn("xgboost not installed — XGBoost model will be skipped.")

try:
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.over_sampling import SMOTE
except ImportError:
    raise ImportError(
        "imbalanced-learn is required.  pip install imbalanced-learn"
    )

try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, Input
    from scikeras.wrappers import KerasClassifier
    _HAS_KERAS = True
except ImportError:
    _HAS_KERAS = False
    warnings.warn("tensorflow/scikeras not installed — ANN model will be skipped.")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data_utils import load_framingham
from shared import get_imbalance_info, save_results

EXPERIMENT_DIR = PROJECT_ROOT / "results" / "framingham"
MODELS_DIR     = EXPERIMENT_DIR / "models"
RESULTS_DIR    = EXPERIMENT_DIR
FIGURES_DIR    = RESULTS_DIR / "figures"

# ---------------------------------------------------------------------------
# Decision threshold for positive-class classification at inference time
# ---------------------------------------------------------------------------
DECISION_THRESHOLD = 0.35
SMOTE_RATIO        = 0.33


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def build_preprocessor(numeric_features: list, categorical_features: list) -> ColumnTransformer:
    """Median-impute + scale numerics; most-frequent-impute + OHE categoricals."""
    num_pipe = SklearnPipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])
    transformers = [("num", num_pipe, numeric_features)]

    if categorical_features:
        cat_pipe = SklearnPipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe",     OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        transformers.append(("cat", cat_pipe, categorical_features))

    return ColumnTransformer(transformers)


def build_imblearn_pipeline(preprocessor: ColumnTransformer, classifier) -> ImbPipeline:
    """
    Wrap any classifier in a leakage-free imblearn Pipeline:
      preprocessor → SMOTE(0.33) → classifier

    Because SMOTE is a pipeline step, GridSearchCV will ONLY apply
    oversampling to the n-1 training folds; validation folds are untouched.
    """
    return ImbPipeline([
        ("preprocessor", preprocessor),
        ("smote",        SMOTE(sampling_strategy=SMOTE_RATIO, random_state=42)),
        ("model",        classifier),
    ])


def build_keras_model(meta):
    model = Sequential([
        Input(shape=(meta["n_features_in_"],)),
        Dense(32, activation="relu"),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1,  activation="sigmoid"),
    ])
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model


def evaluate_at_threshold(pipeline, X_test, y_test, threshold: float = 0.5) -> dict:
    """Evaluate a fitted imblearn pipeline at a custom probability threshold."""
    try:
        y_proba = pipeline.predict_proba(X_test)[:, 1]
    except Exception:
        try:
            y_proba = pipeline.decision_function(X_test)
        except Exception:
            y_proba = None

    if y_proba is not None:
        y_pred = (y_proba >= threshold).astype(int)
    else:
        y_pred = pipeline.predict(X_test)

    metrics = {
        "accuracy":         float(accuracy_score(y_test, y_pred)),
        "precision":        float(precision_score(y_test, y_pred, zero_division=0)),
        "recall":           float(recall_score(y_test, y_pred, zero_division=0)),
        "f1":               float(f1_score(y_test, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "roc_auc":          float(roc_auc_score(y_test, y_proba)) if y_proba is not None else None,
        "y_pred":           y_pred,
        "y_proba":          y_proba,
    }
    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print("[1/4] Loading Framingham dataset via Kaggle...")
    df = load_framingham()
    X  = df.drop(columns=["TenYearCHD"])
    y  = df["TenYearCHD"]
    print(f"Dataset loaded: {len(df)} rows, {df.shape[1]} columns.")

    binary_cols          = [c for c in X.columns if X[c].dropna().isin([0, 1]).all()]
    numeric_features     = [c for c in X.columns if c not in binary_cols]
    categorical_features = binary_cols
    print(f"Numeric features: {len(numeric_features)}  |  "
          f"Categorical (binary): {len(categorical_features)}")

    # ------------------------------------------------------------------
    # 2. Stratified 80/20 split
    # ------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    imbalance_info = get_imbalance_info(y_train)
    print(f"[2/4] Class imbalance ratio: "
          f"{imbalance_info['minority_to_majority_ratio']:.3f}  "
          f"(SMOTE @ {SMOTE_RATIO} applied inside each pipeline — no leakage)")

    # ------------------------------------------------------------------
    # 3. Build shared preprocessor + model catalogue
    # ------------------------------------------------------------------
    preprocessor = build_preprocessor(numeric_features, categorical_features)

    classifiers = {
        "LogisticRegression": LogisticRegression(
            max_iter=1000, class_weight="balanced"
        ),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "SVC": SVC(probability=True, class_weight="balanced"),
        "RandomForest": RandomForestClassifier(
            class_weight="balanced", random_state=42
        ),
    }

    if _HAS_XGB:
        classifiers["XGBoost"] = XGBClassifier(
            n_estimators=250,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            scale_pos_weight=5.0,   # compensates class imbalance for XGB
            random_state=42,
            eval_metric="logloss",
        )

    if _HAS_KERAS:
        classifiers["ANN_Keras"] = KerasClassifier(
            model=build_keras_model,
            epochs=50,
            batch_size=32,
            verbose=0,
            class_weight="balanced",
        )

    # ------------------------------------------------------------------
    # 4. Train all models inside leakage-free imblearn pipelines
    #    RandomForest gets GridSearchCV tuning on top.
    # ------------------------------------------------------------------
    print(f"[3/4] Training {len(classifiers)} models with imblearn pipelines...")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    results    = {}
    best_f1    = -1.0
    best_name  = None
    best_pipeline_obj = None

    rf_grid_search = None   # keep reference for later CSV export

    for name, clf in classifiers.items():
        print(f"\n  >> {name}")
        pipeline = build_imblearn_pipeline(preprocessor, clf)

        # RandomForest: tune via GridSearchCV (SMOTE still inside pipeline)
        if name == "RandomForest":
            param_grid = {
                "model__n_estimators":      [100, 200, 300],
                "model__max_depth":         [None, 10, 20],
                "model__min_samples_split": [2, 5, 10],
            }
            print("     Running GridSearchCV (5-fold, F1) — this may take a minute...")
            rf_grid_search = GridSearchCV(
                estimator  = pipeline,
                param_grid = param_grid,
                cv         = 5,
                scoring    = "f1",
                n_jobs     = -1,
                verbose    = 0,
                refit      = True,
            )
            rf_grid_search.fit(X_train, y_train)
            fitted_pipeline = rf_grid_search.best_estimator_
            print(f"     Best CV F1 : {rf_grid_search.best_score_:.4f}")
            print(f"     Best params: {rf_grid_search.best_params_}")
        else:
            pipeline.fit(X_train, y_train)
            fitted_pipeline = pipeline

        metrics = evaluate_at_threshold(
            fitted_pipeline, X_test, y_test, threshold=DECISION_THRESHOLD
        )
        results[name] = metrics

        print(f"     accuracy={metrics['accuracy']:.4f}  "
              f"precision={metrics['precision']:.4f}  "
              f"recall={metrics['recall']:.4f}  "
              f"f1={metrics['f1']:.4f}  "
              f"roc_auc={metrics['roc_auc'] or 0.0:.4f}")

        # Persist each pipeline
        joblib.dump(fitted_pipeline, MODELS_DIR / f"{name}_pipeline.joblib")

        # Track best by F1
        if metrics["f1"] > best_f1:
            best_f1           = metrics["f1"]
            best_name         = name
            best_pipeline_obj = fitted_pipeline

    # Persist best model alias
    joblib.dump(best_pipeline_obj, MODELS_DIR / "best_pipeline.joblib")

    # Export GridSearch CV detail for RandomForest
    if rf_grid_search is not None:
        pd.DataFrame(rf_grid_search.cv_results_).to_csv(
            MODELS_DIR / "gridsearch_cv_results.csv", index=False
        )

    # ------------------------------------------------------------------
    # 5. Save results & plots
    # ------------------------------------------------------------------
    print(f"\n[4/4] Saving results to {RESULTS_DIR}...")
    save_results(
        results        = results,
        best_name      = best_name,
        y_test         = y_test,
        models_dir     = MODELS_DIR,
        results_dir    = RESULTS_DIR,
        figures_dir    = FIGURES_DIR,
        imbalance_info = imbalance_info,
        title_prefix   = "Framingham: ",
        extra_summary  = {
            "decision_threshold": DECISION_THRESHOLD,
            "smote_strategy":     SMOTE_RATIO,
            "pipeline_type":      "imblearn (zero data leakage, all models)",
            "rf_cv_best_f1":      float(rf_grid_search.best_score_) if rf_grid_search else None,
            "rf_best_params":     rf_grid_search.best_params_ if rf_grid_search else None,
        },
    )

    print("\nFramingham pipeline completed successfully!")


if __name__ == "__main__":
    main()
