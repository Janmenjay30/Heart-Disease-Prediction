from pathlib import Path
import sys
import json
import joblib
import warnings
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
import pandas as pd
import numpy as np
from data_utils import load_cleveland

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None


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
        "roc_auc": float(roc_auc_score(y_test, y_proba)) if y_proba is not None else None,
    }
    return metrics, y_pred, y_proba


def get_imbalance_info(y: pd.Series) -> dict:
    """Compute basic class imbalance diagnostics for binary labels."""
    class_counts = y.value_counts().sort_index()
    if len(class_counts) != 2:
        return {
            "is_binary": False,
            "is_imbalanced": False,
            "minority_to_majority_ratio": None,
            "counts": class_counts.to_dict(),
        }

    minority = int(class_counts.min())
    majority = int(class_counts.max())
    ratio = minority / majority if majority else 1.0
    # Conservative threshold; below 0.80 we enable class-balancing strategy.
    is_imbalanced = ratio < 0.80
    return {
        "is_binary": True,
        "is_imbalanced": is_imbalanced,
        "minority_to_majority_ratio": float(ratio),
        "counts": class_counts.to_dict(),
    }


def build_models(use_class_weight: bool):
    class_weight = "balanced" if use_class_weight else None
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, class_weight=class_weight),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "SVC": SVC(probability=True, class_weight=class_weight),
        "RandomForest": RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight=class_weight,
        ),
    }

    if XGBClassifier is not None:
        # scale_pos_weight helps with imbalance when present.
        scale_pos_weight = 1.0
        if use_class_weight:
            scale_pos_weight = 1.2
        models["XGBoost"] = XGBClassifier(
            n_estimators=250,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
        )
    else:
        warnings.warn("xgboost is not installed. Skipping XGBoost model.")

    return models


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

    imbalance_info = get_imbalance_info(y_train)
    use_class_weight = imbalance_info["is_imbalanced"]
    models = build_models(use_class_weight=use_class_weight)

    results = {}
    best_by_recall = (None, -1.0)
    best_pipeline = None

    for name, model in models.items():
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model),
        ])
        pipeline.fit(X_train, y_train)
        metrics, y_pred, y_proba = evaluate_model(pipeline, X_test, y_test)
        results[name] = metrics

        # Persist each full pipeline so downstream analysis can re-evaluate all models.
        joblib.dump(pipeline, MODELS_DIR / f"{name}_pipeline.joblib")

        if metrics["recall"] > best_by_recall[1]:
            best_by_recall = (name, metrics["recall"])
            best_pipeline = pipeline
        elif metrics["recall"] == best_by_recall[1] and best_pipeline is not None:
            current_auc = metrics.get("roc_auc") or -1.0
            best_auc = results[best_by_recall[0]].get("roc_auc") or -1.0
            if current_auc > best_auc:
                best_by_recall = (name, metrics["recall"])
                best_pipeline = pipeline

    # Save best pipeline
    if best_pipeline is None:
        raise RuntimeError("No model was trained successfully; cannot save best pipeline.")

    best_name = best_by_recall[0]
    joblib.dump(best_pipeline, MODELS_DIR / "best_pipeline.joblib")
    with open(MODELS_DIR / "metrics.json", "w") as f:
        json.dump(
            {
                "results": results,
                "best": best_name,
                "imbalance_analysis": imbalance_info,
                "class_weight_enabled": use_class_weight,
            },
            f,
            indent=2,
        )

    # Write a small report
    report_lines = []
    report_lines.append(f"Best model by Recall: {best_name}\n")
    report_lines.append(
        f"Class imbalance detected (train split): {use_class_weight} "
        f"(minority/majority ratio={imbalance_info['minority_to_majority_ratio']})\n"
    )
    report_lines.append("Model metrics:\n")
    for name, m in results.items():
        auc_text = f", ROC-AUC={m['roc_auc']:.3f}" if m["roc_auc"] is not None else ""
        report_lines.append(
            f"- {name}: Accuracy={m['accuracy']:.3f}, Precision={m['precision']:.3f}, "
            f"Recall={m['recall']:.3f}, F1={m['f1']:.3f}{auc_text}"
        )
    report_text = "\n".join(report_lines)
    (PROJECT_ROOT / "REPORT.md").write_text(report_text)

    print("Training complete. Best model:", best_name)
    print("Metrics saved to", MODELS_DIR / "metrics.json")


if __name__ == "__main__":
    main()
