import sys
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

PROJECT_ROOT = Path("d:/E Drive/programming/Heart_disease_prediction/Projectprototype")
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data_utils import load_cleveland, load_framingham

def generate_correlation(df, name, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Calculate correlation matrix
    corr = df.corr()
    
    # Save as CSV
    csv_path = out_dir / f"{name}_correlation.csv"
    corr.to_csv(csv_path)
    print(f"Saved {name} correlation CSV to {csv_path}")
    
    # Plot and save as PNG
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, square=True)
    plt.title(f"{name.capitalize()} Dataset Correlation Matrix")
    plt.tight_layout()
    
    png_path = out_dir / f"{name}_correlation.png"
    plt.savefig(png_path, dpi=300)
    plt.close()
    print(f"Saved {name} correlation plot to {png_path}")
    
    # Print target correlation sorted
    target_col = "target" if "target" in df.columns else "TenYearCHD"
    print(f"\nCorrelation with target ({target_col}) in {name}:")
    print(corr[target_col].sort_values(ascending=False))
    print("-" * 50)

if __name__ == "__main__":
    # Cleveland
    print("Processing Cleveland dataset...")
    df_cleveland = load_cleveland(PROJECT_ROOT / "data" / "processed.cleveland.data")
    generate_correlation(df_cleveland, "cleveland", PROJECT_ROOT / "results" / "cleveland" / "figures")
    
    # Framingham
    print("\nProcessing Framingham dataset...")
    df_framingham = load_framingham()
    generate_correlation(df_framingham, "framingham", PROJECT_ROOT / "results" / "framingham" / "figures")
