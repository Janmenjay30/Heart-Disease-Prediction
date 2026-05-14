from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from data_utils import load_cleveland


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed.cleveland.data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = OUTPUT_DIR / "figures"


def prepare_test_data():
    df = load_cleveland(DATA_PATH)
    X = df.drop(columns=["target"])
    y = df["target"]
    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    return X_test, y_test


def find_model_pipelines():
    pipelines = {}
    for p in MODELS_DIR.glob("*_pipeline.joblib"):
        if p.name == "best_pipeline.joblib":
            continue
        model_name = p.name.replace("_pipeline.joblib", "")
        pipelines[model_name] = p
    return dict(sorted(pipelines.items(), key=lambda x: x[0].lower()))


def evaluate_single_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = None
    if hasattr(model, "predict_proba"):
        try:
            y_proba = model.predict_proba(X_test)[:, 1]
        except Exception:
            y_proba = None

    result = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "roc_auc": float(roc_auc_score(y_test, y_proba)) if y_proba is not None else None,
        "y_pred": y_pred,
        "y_proba": y_proba,
    }
    return result


def plot_accuracy_bar(df_metrics: pd.DataFrame):
    plt.figure(figsize=(9, 5))
    ax = sns.barplot(data=df_metrics, x="model", y="accuracy", hue="model", palette="viridis", legend=False)
    ax.set_title("Model Accuracy Comparison", fontsize=14, fontweight="bold")
    ax.set_xlabel("Model")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=25)

    for i, acc in enumerate(df_metrics["accuracy"].tolist()):
        ax.text(float(i), float(acc) + 0.01, f"{acc:.3f}", ha="center", fontsize=9)

    plt.tight_layout()
    out_path = FIGURES_DIR / "accuracy_comparison.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path


def plot_multi_metric_bars(df_metrics: pd.DataFrame):
    metrics_long = df_metrics.melt(
        id_vars=["model"],
        value_vars=["accuracy", "precision", "recall", "f1"],
        var_name="metric",
        value_name="score",
    )

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=metrics_long, x="model", y="score", hue="metric", palette="Set2")
    ax.set_title("Model Performance Metrics", fontsize=14, fontweight="bold")
    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=25)

    plt.legend(title="Metric", loc="lower right")
    plt.tight_layout()
    out_path = FIGURES_DIR / "metrics_comparison.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path


def plot_confusion_matrices(results: dict):
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
            xticklabels=["No Disease", "Disease"],
            yticklabels=["No Disease", "Disease"],
        )
        axes[idx].set_title(f"{name} Confusion Matrix", fontweight="bold")
        axes[idx].set_xlabel("Predicted")
        axes[idx].set_ylabel("Actual")

    for j in range(n, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    out_path = FIGURES_DIR / "confusion_matrices.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path


def plot_roc_curves(results: dict, y_test):
    plt.figure(figsize=(8, 6))
    plotted = False

    for name, res in results.items():
        y_proba = res["y_proba"]
        if y_proba is None:
            continue
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc_score = res["roc_auc"] if res["roc_auc"] is not None else np.nan
        plt.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC={auc_score:.3f})")
        plotted = True

    plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random baseline")
    plt.title("ROC Curves", fontsize=14, fontweight="bold")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.2)

    out_path = FIGURES_DIR / "roc_curves.png"
    if plotted:
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path if plotted else None


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    pipelines = find_model_pipelines()
    if not pipelines:
        raise FileNotFoundError(
            "No model pipelines found. Run `python src/train.py` first to generate *_pipeline.joblib files."
        )

    X_test, y_test = prepare_test_data()

    results = {}
    rows = []

    for name, path in pipelines.items():
        model = joblib.load(path)
        eval_result = evaluate_single_model(model, X_test, y_test)
        results[name] = eval_result
        rows.append(
            {
                "model": name,
                "accuracy": eval_result["accuracy"],
                "precision": eval_result["precision"],
                "recall": eval_result["recall"],
                "f1": eval_result["f1"],
                "roc_auc": eval_result["roc_auc"],
            }
        )

    metrics_df = pd.DataFrame(rows).sort_values("accuracy", ascending=False)

    # Console output for quick paper-ready table values.
    print("\nModel Evaluation Summary")
    print(metrics_df.to_string(index=False, float_format=lambda x: f"{x:.4f}" if pd.notna(x) else "N/A"))

    # Save clean metrics table.
    table_path = OUTPUT_DIR / "model_metrics.csv"
    metrics_df.to_csv(table_path, index=False)

    fig_paths = {
        "accuracy_comparison": str(plot_accuracy_bar(metrics_df)),
        "metrics_comparison": str(plot_multi_metric_bars(metrics_df)),
        "confusion_matrices": str(plot_confusion_matrices(results)),
        "roc_curves": None,
    }
    roc_path = plot_roc_curves(results, y_test)
    if roc_path is not None:
        fig_paths["roc_curves"] = str(roc_path)

    serializable_results = {
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

    summary_path = OUTPUT_DIR / "evaluation_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"metrics": serializable_results, "figures": fig_paths}, f, indent=2)

    print("\nSaved outputs:")
    print(f"- Metrics CSV: {table_path}")
    print(f"- Summary JSON: {summary_path}")
    for key, val in fig_paths.items():
        if val:
            print(f"- {key}: {val}")


if __name__ == "__main__":
    main()
