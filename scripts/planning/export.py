"""
Export the planning dataset (test_final.parquet) to multiple scientific formats:
  CSV, JSON, YAML, HDF5 (.h5), NumPy (.npz), Excel (.xlsx), Feather, Pickle
"""

import os
import json
import pickle
import numpy as np
import pandas as pd

OUT_DIR = "outputs/data/planning"
SRC = "outputs/planning_dataset/test_final.parquet"

os.makedirs(OUT_DIR, exist_ok=True)

print(f"Loading {SRC} ...")
df = pd.read_parquet(SRC)
print(f"  {len(df)} rows x {len(df.columns)} columns")

# ── Numeric-only copy for numpy/hdf5 ──────────────────────────────────────────
num_cols = df.select_dtypes(include="number").columns.tolist()
df_num = df[num_cols]

# 1. CSV
p = f"{OUT_DIR}/planning_dataset.csv"
df.to_csv(p, index=False)
print(f"  [OK] CSV        {os.path.getsize(p)/1024:.1f} KB")

# 2. JSON (records orientation)
p = f"{OUT_DIR}/planning_dataset.json"
df.to_json(p, orient="records", indent=2, default_handler=str)
print(f"  [OK] JSON       {os.path.getsize(p)/1024:.1f} KB")

# 3. YAML
try:
    import yaml
    p = f"{OUT_DIR}/planning_dataset.yaml"
    records = json.loads(df.to_json(orient="records", default_handler=str))
    with open(p, "w") as fh:
        yaml.dump(records, fh, default_flow_style=False, allow_unicode=True)
    print(f"  [OK] YAML       {os.path.getsize(p)/1024:.1f} KB")
except ImportError:
    print("  [SKIP] YAML       (pyyaml not installed)")

# 4. HDF5
try:
    p = f"{OUT_DIR}/planning_dataset.h5"
    df_num.to_hdf(p, key="planning", mode="w", complevel=5)
    print(f"  [OK] HDF5       {os.path.getsize(p)/1024:.1f} KB")
except Exception as e:
    print(f"  [SKIP] HDF5       ({e})")

# 5. NumPy npz (numeric columns)
p = f"{OUT_DIR}/planning_dataset.npz"
arrays = {col: df_num[col].values for col in num_cols}
np.savez_compressed(p, **arrays)
print(f"  [OK] NumPy npz  {os.path.getsize(p)/1024:.1f} KB")

# 6. Excel
try:
    p = f"{OUT_DIR}/planning_dataset.xlsx"
    df.to_excel(p, index=False, engine="openpyxl")
    print(f"  [OK] Excel      {os.path.getsize(p)/1024:.1f} KB")
except ImportError:
    print("  [SKIP] Excel      (openpyxl not installed)")

# 7. Feather
try:
    p = f"{OUT_DIR}/planning_dataset.feather"
    df_num.reset_index(drop=True).to_feather(p)
    print(f"  [OK] Feather    {os.path.getsize(p)/1024:.1f} KB")
except Exception as e:
    print(f"  [SKIP] Feather    ({e})")

# 8. Pickle
p = f"{OUT_DIR}/planning_dataset.pkl"
with open(p, "wb") as fh:
    pickle.dump(df, fh, protocol=pickle.HIGHEST_PROTOCOL)
print(f"  [OK] Pickle     {os.path.getsize(p)/1024:.1f} KB")

print(f"\nAll exports written to {OUT_DIR}/")
