from src.data_utils import load_cleveland
from pathlib import Path
import sys

# Add src to path
sys.path.append('src')

# Load data
data_path = Path("data/processed.cleveland.data")
try:
    df = load_cleveland(data_path)

    # Check gender distribution (sex: 1 = male, 0 = female)
    gender_counts = df['sex'].value_counts()
    gender_percent = df['sex'].value_counts(normalize=True)

    print("Gender Counts:\n", gender_counts)
    print("\nGender Percentages:\n", gender_percent)

    # Check target distribution by sex
    print("\nTarget Distribution by Sex:")
    print(df.groupby('sex')['target'].value_counts(normalize=True))
    print("\nCounts:")
    print(df.groupby('sex')['target'].value_counts())
except Exception as e:
    print(f"Error: {e}")
