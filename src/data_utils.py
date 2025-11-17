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


if __name__ == "__main__":
    import sys
    p = sys.argv[1] if len(sys.argv) > 1 else "../data/processed.cleveland.data"
    df = load_cleveland(p)
    print("Loaded", df.shape)
    print(df.head())
