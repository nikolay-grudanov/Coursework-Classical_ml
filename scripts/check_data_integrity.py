"""Quick data integrity checker for main CSV artifacts.
Prints shape, total missing, top-5 columns by missing, duplicate rows count,
and `anomaly` distribution when present.
"""

import sys
from pathlib import Path

FILES = [
    "data/cleaned_data.csv",
    "data/cleaned_data_with_anomalies.csv",
    "data/preprocessed_data.csv",
    "data/feature_engineered_data.csv",
    "data/reduced_data.csv",
    "data/train_features.csv",
    "data/test_features.csv",
]

try:
    import pandas as pd
except Exception as e:
    print(
        "ERROR: pandas is not available. Install dependencies (e.g. pip install -r requirements.txt) or run inside the project's conda env."
    )
    print("ImportError:", e)
    sys.exit(2)


def summarize_file(path: Path):
    if not path.exists():
        print(f"MISSING: {path} (file not found)")
        return
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception as e:
        print(f"FAILED READ: {path} -> {e}")
        return

    n, m = df.shape
    total_missing = int(df.isna().sum().sum())
    missing_by_col = df.isna().sum().sort_values(ascending=False)
    top_missing = missing_by_col[missing_by_col > 0].head(5)
    dup_count = int(df.duplicated().sum())

    print(f"FILE: {path}")
    print(f"  shape: {n} rows x {m} cols")
    print(f"  total missing values: {total_missing}")
    if top_missing.shape[0] > 0:
        print("  top missing cols:")
        for col, cnt in top_missing.items():
            print(f"    - {col}: {int(cnt)}")
    else:
        print("  no missing values detected")
    print(f"  duplicate rows: {dup_count}")

    if "anomaly" in df.columns:
        print("  anomaly distribution:")
        print(df["anomaly"].value_counts(dropna=False).to_string())

    # show small sample of first row values for quick sanity
    try:
        print("  sample row:")
        print(df.iloc[0].to_dict())
    except Exception:
        pass


if __name__ == "__main__":
    for p in FILES:
        summarize_file(Path(p))
        print("-" * 60)
