"""
shared.py — Shared ML models, evaluation metrics, and visualization functions
used by both the Cleveland and Framingham pipelines.
"""

from pathlib import Path
import json
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
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
    roc_curve,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None


# ---------------------------------------------------------------------------
# 1. Preprocessor builder
# ---------------------------------------------------------------------------
def build_preprocessor(numeric_features: list, categorical_features: list) -> ColumnTransformer:
    """Build a ColumnTransformer with median-impute + scale for numeric features
    and most-frequent-impute + one-hot for categorical features."""
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    transformers = [("num", num_pipeline, numeric_features)]

    if categorical_features:
        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore")),
        ])
        transformers.append(("cat", cat_pipeline, categorical_features))

    return ColumnTransformer(transformers)


# ---------------------------------------------------------------------------
# 2. Model factory
# ---------------------------------------------------------------------------
def build_models(use_class_weight: bool = False) -> dict:
    """Return a dict of {name: estimator} for the five standard classifiers."""
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
        scale_pos_weight = 1.2 if use_class_weight else 1.0
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


# ---------------------------------------------------------------------------
# 3. Imbalance analysis
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# 4. Evaluation
# ---------------------------------------------------------------------------
def evaluate_model(pipeline, X_test, y_test) -> tuple:
    """Evaluate a trained pipeline on the test set.

    Returns (metrics_dict, y_pred, y_proba).
    """
    y_pred = pipeline.predict(X_test)
    y_proba = None
    try:
        y_proba = pipeline.predict_proba(X_test)[:, 1]
    except Exception:
        try:
            y_proba = pipeline.decision_function(X_test)
        except Exception:
            y_proba = None

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "roc_auc": float(roc_auc_score(y_test, y_proba)) if y_proba is not None else None,
    }
    return metrics, y_pred, y_proba


# ---------------------------------------------------------------------------
# 5. Visualization functions
# ---------------------------------------------------------------------------
def plot_accuracy_bar(df_metrics: pd.DataFrame, figures_dir: Path, title_prefix: str = ""):
    """Bar chart comparing accuracy across models."""
    plt.figure(figsize=(9, 5))
    ax = sns.barplot(data=df_metrics, x="model", y="accuracy", hue="model",
                     palette="viridis", legend=False)
    title = f"{title_prefix}Accuracy Comparison" if title_prefix else "Model Accuracy Comparison"
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Model")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=25)

    for i, acc in enumerate(df_metrics["accuracy"].tolist()):
        ax.text(float(i), float(acc) + 0.01, f"{acc:.3f}", ha="center", fontsize=9)

    plt.tight_layout()
    out_path = figures_dir / "accuracy_comparison.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path


def plot_multi_metric_bars(df_metrics: pd.DataFrame, figures_dir: Path, title_prefix: str = ""):
    """Grouped bar chart for accuracy, precision, recall, f1."""
    metrics_long = df_metrics.melt(
        id_vars=["model"],
        value_vars=["accuracy", "precision", "recall", "f1"],
        var_name="metric",
        value_name="score",
    )

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=metrics_long, x="model", y="score", hue="metric", palette="Set2")
    title = f"{title_prefix}Performance Metrics" if title_prefix else "Model Performance Metrics"
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=25)

    plt.legend(title="Metric", loc="lower right")
    plt.tight_layout()
    out_path = figures_dir / "metrics_comparison.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path


def plot_confusion_matrices(results: dict, figures_dir: Path,
                            class_labels: list[str] | None = None):
    """Grid of confusion matrices for each model."""
    if class_labels is None:
        class_labels = ["No Disease", "Disease"]

    model_names = list(results.keys())
    n = len(model_names)
    cols = 2
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(12, 4.5 * rows))
    axes = np.atleast_1d(axes).ravel()

    for idx, name in enumerate(model_names):
        cm = np.array(results[name]["confusion_matrix"])
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            ax=axes[idx],
            xticklabels=class_labels,
            yticklabels=class_labels,
        )
        axes[idx].set_title(f"{name} Confusion Matrix", fontweight="bold")
        axes[idx].set_xlabel("Predicted")
        axes[idx].set_ylabel("Actual")

    for j in range(n, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    out_path = figures_dir / "confusion_matrices.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path


def plot_roc_curves(results: dict, y_test, figures_dir: Path, title_prefix: str = ""):
    """ROC curves for all models that have probability scores."""
    plt.figure(figsize=(8, 6))
    plotted = False

    for name, res in results.items():
        y_proba = res.get("y_proba")
        if y_proba is None:
            continue
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc_score = res["roc_auc"] if res["roc_auc"] is not None else np.nan
        plt.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC={auc_score:.3f})")
        plotted = True

    plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random baseline")
    title = f"{title_prefix}ROC Curves" if title_prefix else "ROC Curves"
    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.2)

    out_path = figures_dir / "roc_curves.png"
    if plotted:
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path if plotted else None


# ---------------------------------------------------------------------------
# 6. Training loop helper
# ---------------------------------------------------------------------------
def train_and_evaluate_all(
    X_train,
    X_test,
    y_train,
    y_test,
    preprocessor: ColumnTransformer,
    models: dict,
    models_dir: Path,
    select_best_by: str = "recall",
) -> tuple[dict, str, Pipeline]:
    """Train all models, evaluate, persist pipelines, and pick the best.

    Returns (results_dict, best_model_name, best_pipeline).
    """
    results = {}
    best_score = -1.0
    best_name = None
    best_pipeline = None

    for name, model in models.items():
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model),
        ])
        pipeline.fit(X_train, y_train)
        metrics, y_pred, y_proba = evaluate_model(pipeline, X_test, y_test)
        metrics["y_pred"] = y_pred
        metrics["y_proba"] = y_proba
        results[name] = metrics

        # Persist each full pipeline
        joblib.dump(pipeline, models_dir / f"{name}_pipeline.joblib")

        current_score = metrics.get(select_best_by) or -1.0
        if current_score > best_score:
            best_score = current_score
            best_name = name
            best_pipeline = pipeline
        elif current_score == best_score and best_pipeline is not None:
            # Tie-break: prefer higher AUC, then higher recall
            current_auc = metrics.get("roc_auc") or -1.0
            best_auc = results[best_name].get("roc_auc") or -1.0
            if current_auc > best_auc:
                best_name = name
                best_pipeline = pipeline

    if best_pipeline is None:
        raise RuntimeError("No model was trained successfully.")

    joblib.dump(best_pipeline, models_dir / "best_pipeline.joblib")
    return results, best_name, best_pipeline


# ---------------------------------------------------------------------------
# 7. Saving results helper
# ---------------------------------------------------------------------------
def save_results(
    results: dict,
    best_name: str,
    y_test,
    models_dir: Path,
    results_dir: Path,
    figures_dir: Path,
    imbalance_info: dict | None = None,
    class_labels: list[str] | None = None,
    title_prefix: str = "",
    extra_summary: dict | None = None,
):
    """Persist metrics JSON, CSV, and generate all evaluation plots."""
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Build metrics DataFrame
    rows = []
    for name, m in results.items():
        rows.append({
            "model": name,
            "accuracy": m["accuracy"],
            "precision": m["precision"],
            "recall": m["recall"],
            "f1": m["f1"],
            "roc_auc": m["roc_auc"],
        })
    metrics_df = pd.DataFrame(rows).sort_values("accuracy", ascending=False)
    metrics_df.to_csv(results_dir / "model_metrics.csv", index=False)

    # Plots
    fig_paths = {
        "accuracy_comparison": str(plot_accuracy_bar(metrics_df, figures_dir, title_prefix)),
        "metrics_comparison": str(plot_multi_metric_bars(metrics_df, figures_dir, title_prefix)),
        "confusion_matrices": str(plot_confusion_matrices(results, figures_dir, class_labels)),
        "roc_curves": None,
    }
    roc_path = plot_roc_curves(results, y_test, figures_dir, title_prefix)
    if roc_path is not None:
        fig_paths["roc_curves"] = str(roc_path)

    # Serializable metrics (strip numpy arrays)
    serializable = {
        model_name: {
            "accuracy": vals["accuracy"],
            "precision": vals["precision"],
            "recall": vals["recall"],
            "f1": vals["f1"],
            "roc_auc": vals["roc_auc"],
            "confusion_matrix": vals["confusion_matrix"],
        }
        for model_name, vals in results.items()
    }

    # Metrics JSON (train.py-compatible format)
    metrics_json = {
        "results": serializable,
        "best": best_name,
    }
    if imbalance_info is not None:
        metrics_json["imbalance_analysis"] = imbalance_info
        metrics_json["class_weight_enabled"] = imbalance_info.get("is_imbalanced", False)

    with open(models_dir / "metrics.json", "w") as f:
        json.dump(metrics_json, f, indent=2)

    # Evaluation summary JSON
    summary = {
        "best_model": best_name,
        "metrics": serializable,
        "figures": fig_paths,
    }
    if extra_summary:
        summary.update(extra_summary)
    with open(results_dir / "evaluation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Print console report
    print(f"\nBest model: {best_name}")
    print(metrics_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}" if pd.notna(x) else "N/A"
    ))
    for key, val in fig_paths.items():
        if val:
            print(f"  {key}: {val}")

    return metrics_df
