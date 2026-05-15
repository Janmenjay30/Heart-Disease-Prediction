"""
Pipeline 1 — Cleveland Dataset
Predict existing heart disease using 5 ML models.

Usage:
    python src/train.py
"""

from pathlib import Path
import sys

from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data_utils import load_cleveland
from shared import (
    build_preprocessor,
    build_models,
    get_imbalance_info,
    train_and_evaluate_all,
    save_results,
)

DATA_PATH = PROJECT_ROOT / "data" / "processed.cleveland.data"
MODELS_DIR = PROJECT_ROOT / "results" / "cleveland" / "models"
RESULTS_DIR = PROJECT_ROOT / "results" / "cleveland"
FIGURES_DIR = RESULTS_DIR / "figures"


def main():
    print("[1/4] Loading Cleveland dataset...")
    df = load_cleveland(DATA_PATH)

    X = df.drop(columns=["target"])
    y = df["target"]

    print(f"Dataset loaded: {len(df)} rows, {df.shape[1]} columns.")

    numeric_features = ["age", "trestbps", "chol", "thalach", "oldpeak"]
    categorical_features = [c for c in X.columns if c not in numeric_features]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("[2/4] Preprocessing and checking class imbalance...")
    preprocessor = build_preprocessor(numeric_features, categorical_features)

    imbalance_info = get_imbalance_info(y_train)
    use_class_weight = imbalance_info["is_imbalanced"]
    if use_class_weight:
        print(f"-> Imbalance detected (ratio {imbalance_info['minority_to_majority_ratio']:.2f}). Enabling class_weight.")

    models = build_models(use_class_weight=use_class_weight)

    print(f"[3/4] Training models: {', '.join(models.keys())}")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    results, best_name, best_pipeline = train_and_evaluate_all(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        preprocessor=preprocessor,
        models=models,
        models_dir=MODELS_DIR,
        select_best_by="recall",
    )

    print(f"[4/4] Saving results to {RESULTS_DIR}...")
    save_results(
        results=results,
        best_name=best_name,
        y_test=y_test,
        models_dir=MODELS_DIR,
        results_dir=RESULTS_DIR,
        figures_dir=FIGURES_DIR,
        imbalance_info=imbalance_info,
        class_labels=["No Disease", "Disease"],
        title_prefix="Cleveland: ",
    )

    print("\nCleveland pipeline completed successfully!")


if __name__ == "__main__":
    main()
