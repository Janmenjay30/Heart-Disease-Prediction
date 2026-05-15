import pandas as pd
import numpy as np
from pathlib import Path

COLS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "target",
]


def load_cleveland(path: str | Path):
    """Load the processed Cleveland heart disease file into a pandas DataFrame.

    Replaces '?' with NaN and maps target>0 to 1 (disease present).
    """
    path = Path(path)
    df = pd.read_csv(path, header=None, names=COLS, na_values="?")
    # In this dataset target values 0 = no disease, 1-4 indicate disease.
    df["target"] = df["target"].fillna(0).astype(int)
    df["target"] = (df["target"] > 0).astype(int)
    return df


def load_framingham(path: str | Path | None = None) -> pd.DataFrame:
    """Load the Framingham Heart Study dataset for 10-year CHD risk prediction.

    If *path* is None the dataset is downloaded from Kaggle automatically
    using kagglehub (requires ``kaggle.json`` credentials).

    Preprocessing steps (different from Cleveland):
      - Drop 'education' (irrelevant to clinical prediction).
      - Impute missing values: median for continuous, mode for binary.
      - Create derived features: pulse_pressure, BMI (if absent).
      - Target column: ``TenYearCHD`` (0 = no CHD, 1 = CHD within 10 years).
    """
    if path is not None:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Framingham CSV not found at {path}")
        df = pd.read_csv(path)
    else:
        import kagglehub
        dataset_root = Path(
            kagglehub.dataset_download("dileep070/heart-disease-prediction-using-logistic-regression")
        )
        csv_path = dataset_root / "framingham.csv"
        if not csv_path.exists():
            # Some uploads use a subfolder
            candidates = list(dataset_root.rglob("framingham.csv"))
            if not candidates:
                raise FileNotFoundError(
                    f"framingham.csv not found under {dataset_root}"
                )
            csv_path = candidates[0]
        df = pd.read_csv(csv_path)

    # --- Preprocessing specific to Framingham ---
    # Drop education (not clinically relevant)
    if "education" in df.columns:
        df = df.drop(columns=["education"])

    # Binary / categorical columns — impute with mode
    binary_cols = [c for c in df.columns
                   if df[c].dropna().isin([0, 1]).all() and c != "TenYearCHD"]
    for col in binary_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].mode()[0])

    # Continuous columns — impute with median
    continuous_cols = [c for c in df.columns
                       if c not in binary_cols and c != "TenYearCHD"]
    for col in continuous_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    # Feature engineering
    if "sysBP" in df.columns and "diaBP" in df.columns:
        df["pulse_pressure"] = df["sysBP"] - df["diaBP"]

    # Ensure target is int
    df["TenYearCHD"] = df["TenYearCHD"].astype(int)

    return df


if __name__ == "__main__":
    import sys
    p = sys.argv[1] if len(sys.argv) > 1 else "../data/processed.cleveland.data"
    df = load_cleveland(p)
    print("Loaded Cleveland:", df.shape)
    print(df.head())

    print("\nLoading Framingham...")
    df2 = load_framingham()
    print("Loaded Framingham:", df2.shape)
    print(df2.head())
