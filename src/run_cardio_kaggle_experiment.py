from pathlib import Path
import json
import warnings

import joblib
import kagglehub
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
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_DIR = PROJECT_ROOT / "experiments" / "cardio_kaggle"
MODELS_DIR = EXPERIMENT_DIR / "models"
RESULTS_DIR = EXPERIMENT_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
SELECT_BEST_BY = "f1"  # Options: "f1" or "roc_auc"


def setup_dirs():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_cardio_dataset() -> pd.DataFrame:
    dataset_root = Path(kagglehub.dataset_download("sulianova/cardiovascular-disease-dataset"))
    csv_path = dataset_root / "cardio_train.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset CSV not found at {csv_path}")

    df = pd.read_csv(csv_path, sep=";")

    # Basic quality filtering for physiologically plausible values.
    before = len(df)
    df = df[(df["ap_hi"] >= 80) & (df["ap_hi"] <= 240)]
    df = df[(df["ap_lo"] >= 40) & (df["ap_lo"] <= 160)]
    df = df[df["ap_hi"] > df["ap_lo"]]
    df = df[(df["height"] >= 120) & (df["height"] <= 220)]
    df = df[(df["weight"] >= 35) & (df["weight"] <= 220)]
    df = df.drop_duplicates()

    # Feature engineering for cardiovascular risk signal.
    df["age_years"] = (df["age"] / 365.25).round(2)
    height_m = df["height"] / 100.0
    df["bmi"] = df["weight"] / (height_m ** 2)
    df["pulse_pressure"] = df["ap_hi"] - df["ap_lo"]
    df["map_pressure"] = ((2 * df["ap_lo"]) + df["ap_hi"]) / 3

    if "id" in df.columns:
        df = df.drop(columns=["id"])

    # Avoid redundant age representation; keep age_years for interpretability.
    if "age" in df.columns:
        df = df.drop(columns=["age"])

    print(f"Data cleaning: kept {len(df)} / {before} rows (dropped {before - len(df)}).")

    return df


def build_preprocessor(numeric_features, categorical_features):
    num_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    cat_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    transformers = [("num", num_pipeline, numeric_features)]
    if categorical_features:
        transformers.append(("cat", cat_pipeline, categorical_features))

    return ColumnTransformer(transformers)


def build_models():
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1500, C=1.0),
        "KNN": KNeighborsClassifier(n_neighbors=31, weights="distance"),
        "MLP": MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            learning_rate_init=1e-3,
            max_iter=120,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            random_state=42,
        ),
        # RBF SVC with probability=True is very slow on 70k rows; linear kernel keeps this tractable.
        "SVC": SVC(kernel="linear", probability=False, max_iter=10000),
        "RandomForest": RandomForestClassifier(
            n_estimators=400,
            max_depth=16,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
        ),
    }

    if XGBClassifier is not None:
        models["XGBoost"] = XGBClassifier(
            n_estimators=450,
            learning_rate=0.04,
            max_depth=4,
            min_child_weight=5,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.5,
            random_state=42,
            eval_metric="logloss",
            n_jobs=-1,
        )
    else:
        warnings.warn("xgboost is not installed; skipping XGBoost for this experiment.")

    return models


def evaluate(y_true, y_pred, y_score):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_score)) if y_score is not None else None,
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def find_best_threshold(y_true, y_score):
    """Tune binary decision threshold to maximize F1 on provided labels/scores."""
    thresholds = np.linspace(0.1, 0.9, 81)
    best_threshold = 0.5
    best_f1 = -1.0

    for thr in thresholds:
        y_pred = (y_score >= thr).astype(int)
        score = f1_score(y_true, y_pred, zero_division=0)
        if score > best_f1:
            best_f1 = float(score)
            best_threshold = float(thr)

    return best_threshold, best_f1


def plot_accuracy(df_metrics: pd.DataFrame):
    plt.figure(figsize=(9, 5))
    ax = sns.barplot(data=df_metrics, x="model", y="accuracy", hue="model", legend=False, palette="viridis")
    ax.set_title("Cardio Dataset: Accuracy Comparison", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=20)

    for i, acc in enumerate(df_metrics["accuracy"].tolist()):
        ax.text(float(i), float(acc) + 0.005, f"{acc:.3f}", ha="center", fontsize=9)

    out = FIGURES_DIR / "accuracy_comparison.png"
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close()


def plot_metrics(df_metrics: pd.DataFrame):
    long_df = df_metrics.melt(
        id_vars=["model"],
        value_vars=["accuracy", "precision", "recall", "f1"],
        var_name="metric",
        value_name="score",
    )
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=long_df, x="model", y="score", hue="metric", palette="Set2")
    ax.set_title("Cardio Dataset: Metrics Comparison", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=20)
    out = FIGURES_DIR / "metrics_comparison.png"
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close()


def plot_confusions(results: dict):
    names = list(results.keys())
    n = len(names)
    cols = 2
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(12, 4.5 * rows))
    axes = np.atleast_1d(axes).ravel()

    for idx, name in enumerate(names):
        cm = np.array(results[name]["confusion_matrix"])
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            ax=axes[idx],
            xticklabels=["No CVD", "CVD"],
            yticklabels=["No CVD", "CVD"],
        )
        axes[idx].set_title(f"{name} Confusion Matrix", fontweight="bold")
        axes[idx].set_xlabel("Predicted")
        axes[idx].set_ylabel("Actual")

    for idx in range(n, len(axes)):
        axes[idx].axis("off")

    out = FIGURES_DIR / "confusion_matrices.png"
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close()


def plot_roc_curves(y_test, proba_map: dict, auc_map: dict):
    plt.figure(figsize=(8, 6))
    plotted = False
    for name, y_proba in proba_map.items():
        if y_proba is None:
            continue
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc_val = auc_map[name]
        plt.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC={auc_val:.3f})")
        plotted = True

    plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
    plt.title("Cardio Dataset: ROC Curves", fontsize=13, fontweight="bold")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.grid(alpha=0.2)
    plt.legend(loc="lower right")

    if plotted:
        out = FIGURES_DIR / "roc_curves.png"
        plt.tight_layout()
        plt.savefig(out, dpi=300)
    plt.close()


def main():
    setup_dirs()
    print("[1/4] Loading Kaggle cardiovascular dataset...")

    df = load_cardio_dataset()
    X = df.drop(columns=["cardio"])
    y = df["cardio"].astype(int)
    print(f"Dataset loaded: rows={len(df)}, cols={df.shape[1]}")

    # This dataset uses integer-coded binary/ordinal risk factors; treating all as numeric
    # generally works better than one-hot expansion on this tabular setup.
    numeric_features = list(X.columns)
    categorical_features = []

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    print(f"Train/Test split: train={len(X_train)}, test={len(X_test)}")

    preprocessor = build_preprocessor(numeric_features, categorical_features)
    models = build_models()
    print(f"[2/4] Training models: {', '.join(models.keys())}")

    results = {}
    rows = []
    proba_map = {}
    threshold_tuning = {}

    best_by_metric = (None, -1.0)
    best_pipeline = None

    for name, model in models.items():
        print(f" -> Training {name}...")
        pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        y_score = None
        try:
            y_score = pipeline.predict_proba(X_test)[:, 1]
        except Exception:
            try:
                # Use margin scores for models without predict_proba (e.g., linear SVC).
                y_score = pipeline.decision_function(X_test)
            except Exception:
                y_score = None

        # Automatically tune XGBoost threshold (on train split) and apply to test prediction.
        if name == "XGBoost" and y_score is not None and hasattr(pipeline, "predict_proba"):
            train_score = pipeline.predict_proba(X_train)[:, 1]
            tuned_thr, tuned_train_f1 = find_best_threshold(y_train.values, train_score)
            y_pred = (y_score >= tuned_thr).astype(int)
            threshold_tuning[name] = {
                "threshold": tuned_thr,
                "train_f1_at_threshold": tuned_train_f1,
            }
            print(
                f"    Applied XGBoost threshold tuning: threshold={tuned_thr:.3f}, "
                f"train_f1={tuned_train_f1:.4f}"
            )

        metrics = evaluate(y_test, y_pred, y_score)
        results[name] = metrics
        proba_map[name] = y_score

        rows.append(
            {
                "model": name,
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "roc_auc": metrics["roc_auc"],
                "selection_score": metrics[SELECT_BEST_BY] if metrics[SELECT_BEST_BY] is not None else np.nan,
            }
        )

        joblib.dump(pipeline, MODELS_DIR / f"{name}_pipeline.joblib")
        print(
            f"    Done {name}: acc={metrics['accuracy']:.4f}, recall={metrics['recall']:.4f}, "
            f"auc={metrics['roc_auc']:.4f}"
        )

        current_primary = metrics[SELECT_BEST_BY] if metrics[SELECT_BEST_BY] is not None else -1.0
        if current_primary > best_by_metric[1]:
            best_by_metric = (name, current_primary)
            best_pipeline = pipeline
        elif current_primary == best_by_metric[1] and best_pipeline is not None:
            # Tie-break using AUC, then Recall for safety-first preference.
            current_auc = metrics["roc_auc"] if metrics["roc_auc"] is not None else -1.0
            current_recall = metrics["recall"] if metrics["recall"] is not None else -1.0
            best_name = best_by_metric[0]
            best_auc = results[best_name]["roc_auc"] if results[best_name]["roc_auc"] is not None else -1.0
            best_recall = results[best_name]["recall"] if results[best_name]["recall"] is not None else -1.0
            if current_auc > best_auc or (current_auc == best_auc and current_recall > best_recall):
                best_by_metric = (name, current_primary)
                best_pipeline = pipeline

    if best_pipeline is None:
        raise RuntimeError("No model trained successfully for cardio experiment.")

    best_name = best_by_metric[0]
    joblib.dump(best_pipeline, MODELS_DIR / "best_pipeline.joblib")

    metrics_df = pd.DataFrame(rows).sort_values("selection_score", ascending=False)
    metrics_df.to_csv(RESULTS_DIR / "model_metrics.csv", index=False)

    print("[3/4] Saving plots and metrics...")
    plot_accuracy(metrics_df)
    plot_metrics(metrics_df)
    plot_confusions(results)
    auc_map = {k: v["roc_auc"] for k, v in results.items()}
    plot_roc_curves(y_test, proba_map, auc_map)

    summary = {
        "dataset": "sulianova/cardiovascular-disease-dataset",
        "dataset_shape": [int(df.shape[0]), int(df.shape[1])],
        "target_distribution": {str(k): int(v) for k, v in y.value_counts().to_dict().items()},
        "selection_metric": SELECT_BEST_BY,
        "best_model": best_name,
        "metrics": results,
        "threshold_tuning": threshold_tuning,
        "artifacts": {
            "models_dir": str(MODELS_DIR),
            "results_dir": str(RESULTS_DIR),
        },
    }
    with open(RESULTS_DIR / "evaluation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("[4/4] Cardio experiment complete.")
    print(f"Best model by {SELECT_BEST_BY.upper()}: {best_name}")
    print(f"Models saved to: {MODELS_DIR}")
    print(f"Results saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
