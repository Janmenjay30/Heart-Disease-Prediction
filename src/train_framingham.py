import sys
from pathlib import Path
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data_utils import load_framingham
from shared import (
    build_preprocessor,
    build_models,
    get_imbalance_info,
    train_and_evaluate_all,
    save_results
)

EXPERIMENT_DIR = PROJECT_ROOT / "results" / "framingham"
MODELS_DIR = EXPERIMENT_DIR / "models"
RESULTS_DIR = EXPERIMENT_DIR
FIGURES_DIR = RESULTS_DIR / "figures"


def main():
    print("[1/4] Loading Framingham dataset via Kaggle...")
    df = load_framingham()

    # The target column is TenYearCHD
    X = df.drop(columns=["TenYearCHD"])
    y = df["TenYearCHD"]
    
    print(f"Dataset loaded: {len(df)} rows, {df.shape[1]} columns.")

    # Identify numeric vs categorical features
    # (Since Framingham has many binary features, we'll treat 0/1 as categorical
    # and continuous as numeric)
    binary_cols = [c for c in X.columns if X[c].dropna().isin([0, 1]).all()]
    numeric_features = [c for c in X.columns if c not in binary_cols]
    categorical_features = binary_cols

    print(f"Numeric features: {len(numeric_features)}")
    print(f"Categorical (binary) features: {len(categorical_features)}")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"[2/4] Preprocessing and handling class imbalance...")
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
        select_best_by="f1"  # F1 is better for highly imbalanced datasets
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
        title_prefix="Framingham: "
    )
    
    print("\nFramingham pipeline completed successfully!")


if __name__ == "__main__":
    main()
