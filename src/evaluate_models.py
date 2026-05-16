"""
Evaluate trained model pipelines and generate plots.

Supports both the Cleveland and Framingham pipelines via a CLI argument.

Usage:
    python src/evaluate_models.py                  # Evaluates Cleveland (default)
    python src/evaluate_models.py --pipeline framingham
"""

from pathlib import Path
import sys
import argparse
import json

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data_utils import load_cleveland, load_framingham
from shared import (
    evaluate_model,
    plot_accuracy_bar,
    plot_multi_metric_bars,
    plot_confusion_matrices,
    plot_roc_curves,
)


# ---------------------------------------------------------------------------
# Pipeline configurations
# ---------------------------------------------------------------------------
PIPELINES = {
    "cleveland": {
        "data_loader": lambda: load_cleveland(PROJECT_ROOT / "data" / "processed.cleveland.data"),
        "target_col": "target",
        "models_dir": PROJECT_ROOT / "results" / "cleveland" / "models",
        "results_dir": PROJECT_ROOT / "results" / "cleveland",
        "class_labels": ["No Disease", "Disease"],
        "title_prefix": "Cleveland: ",
    },
    "framingham": {
        "data_loader": lambda: load_framingham().drop(
            columns=["currentSmoker", "heartRate", "cigsPerDay", "prevalentStroke", "BMI"]
        ),
        "target_col": "TenYearCHD",
        "models_dir": PROJECT_ROOT / "results" / "framingham" / "models",
        "results_dir": PROJECT_ROOT / "results" / "framingham",
        "class_labels": ["No CHD", "CHD"],
        "title_prefix": "Framingham: ",
    },
}


def find_model_pipelines(models_dir: Path) -> dict:
    """Discover all *_pipeline.joblib files (except best_pipeline.joblib)."""
    pipelines = {}
    for p in models_dir.glob("*_pipeline.joblib"):
        if p.name == "best_pipeline.joblib":
            continue
        model_name = p.name.replace("_pipeline.joblib", "")
        pipelines[model_name] = p
    return dict(sorted(pipelines.items(), key=lambda x: x[0].lower()))


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained model pipelines.")
    parser.add_argument(
        "--pipeline",
        choices=list(PIPELINES.keys()),
        default="cleveland",
        help="Which pipeline to evaluate (default: cleveland).",
    )
    args = parser.parse_args()

    config = PIPELINES[args.pipeline]
    models_dir = config["models_dir"]
    results_dir = config["results_dir"]
    figures_dir = results_dir / "figures"

    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Find trained pipelines
    pipelines = find_model_pipelines(models_dir)
    if not pipelines:
        raise FileNotFoundError(
            f"No model pipelines found in {models_dir}. "
            f"Run the training script first."
        )

    # Prepare test data (same split as training)
    print(f"Evaluating pipeline: {args.pipeline}")
    df = config["data_loader"]()
    target_col = config["target_col"]
    X = df.drop(columns=[target_col])
    y = df[target_col]
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Evaluate each model
    results = {}
    rows = []

    for name, path in pipelines.items():
        model = joblib.load(path)
        metrics, y_pred, y_proba = evaluate_model(model, X_test, y_test)
        metrics["y_pred"] = y_pred
        metrics["y_proba"] = y_proba
        results[name] = metrics
        rows.append({
            "model": name,
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "roc_auc": metrics["roc_auc"],
        })

    metrics_df = pd.DataFrame(rows).sort_values("accuracy", ascending=False)

    # Console output
    print(f"\n{args.pipeline.title()} Model Evaluation Summary")
    print(metrics_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}" if pd.notna(x) else "N/A",
    ))

    # Save CSV
    table_path = results_dir / "model_metrics.csv"
    metrics_df.to_csv(table_path, index=False)

    # Generate plots
    title_prefix = config["title_prefix"]
    class_labels = config["class_labels"]

    fig_paths = {
        "accuracy_comparison": str(plot_accuracy_bar(metrics_df, figures_dir, title_prefix)),
        "metrics_comparison": str(plot_multi_metric_bars(metrics_df, figures_dir, title_prefix)),
        "confusion_matrices": str(plot_confusion_matrices(results, figures_dir, class_labels)),
        "roc_curves": None,
    }
    roc_path = plot_roc_curves(results, y_test, figures_dir, title_prefix)
    if roc_path is not None:
        fig_paths["roc_curves"] = str(roc_path)

    # Save summary JSON
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

    summary_path = results_dir / "evaluation_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"metrics": serializable, "figures": fig_paths}, f, indent=2)

    print(f"\nSaved outputs:")
    print(f"  Metrics CSV: {table_path}")
    print(f"  Summary JSON: {summary_path}")
    for key, val in fig_paths.items():
        if val:
            print(f"  {key}: {val}")


if __name__ == "__main__":
    main()
