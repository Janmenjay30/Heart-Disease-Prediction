from pathlib import Path
import sys
import json
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import pandas as pd
import numpy as np
from data_utils import load_cleveland


PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Ensure `src` package is importable when running the script from project root
sys.path.insert(0, str(PROJECT_ROOT / "src"))
DATA_PATH = PROJECT_ROOT / "data" / "processed.cleveland.data"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def build_preprocessor(numeric_features, categorical_features):
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipeline, numeric_features),
        ("cat", cat_pipeline, categorical_features),
    ])
    return preprocessor


def evaluate_model(pipeline, X_test, y_test):
    y_pred = pipeline.predict(X_test)
    y_proba = None
    try:
        y_proba = pipeline.predict_proba(X_test)[:, 1]
    except Exception:
        pass
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }
    return metrics, y_pred, y_proba


def main():
    df = load_cleveland(DATA_PATH)

    # Features
    X = df.drop(columns=["target"])
    y = df["target"]

    numeric_features = ["age", "trestbps", "chol", "thalach", "oldpeak"]
    categorical_features = [c for c in X.columns if c not in numeric_features]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = build_preprocessor(numeric_features, categorical_features)

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "SVC": SVC(probability=True),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    }

    results = {}
    best_by_recall = (None, -1.0)

    for name, model in models.items():
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model),
        ])
        pipeline.fit(X_train, y_train)
        metrics, y_pred, y_proba = evaluate_model(pipeline, X_test, y_test)
        results[name] = metrics
        if metrics["recall"] > best_by_recall[1]:
            best_by_recall = (name, metrics["recall"]) 
            best_pipeline = pipeline

    # Save best pipeline
    best_name = best_by_recall[0]
    joblib.dump(best_pipeline, MODELS_DIR / "best_pipeline.joblib")
    with open(MODELS_DIR / "metrics.json", "w") as f:
        json.dump({"results": results, "best": best_name}, f, indent=2)

    # Write a small report
    report_lines = []
    report_lines.append(f"Best model by Recall: {best_name}\n")
    report_lines.append("Model metrics:\n")
    for name, m in results.items():
        report_lines.append(f"- {name}: Accuracy={m['accuracy']:.3f}, Precision={m['precision']:.3f}, Recall={m['recall']:.3f}, F1={m['f1']:.3f}")
    report_text = "\n".join(report_lines)
    (PROJECT_ROOT / "REPORT.md").write_text(report_text)

    print("Training complete. Best model:", best_name)
    print("Metrics saved to", MODELS_DIR / "metrics.json")


if __name__ == "__main__":
    main()
