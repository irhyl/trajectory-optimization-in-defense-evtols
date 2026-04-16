#!/usr/bin/env python3
"""
Mumbai Metropolitan Region — Planning Dataset Runner
=====================================================

Generates 2,000 RRT*+NSGA-III planning trajectories over the Mumbai theatre
using the Mumbai perception dataset produced by scripts/perception/run_mumbai.py.

Prerequisites
-------------
  1. Run perception layer first:
       python scripts/perception/run_mumbai.py
     (outputs to outputs/perception_dataset_mumbai/)

  2. Then run this script:
       python scripts/planning/run_mumbai.py

Outputs
-------
  outputs/planning_dataset/planning_mumbai.parquet
  outputs/planning_dataset/planning_mumbai.csv

Mumbai SAM Emitters (distance-decay gradient, same sites as perception layer)
  - S-125_site_A  : 19.07°N 72.88°E, effective_range 25 km
  - Barak_8_site_B: 18.95°N 73.05°E, effective_range 35 km
  - Akash_site_C  : 19.20°N 72.95°E, effective_range 20 km
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "planning"))

import dataset as plan  # noqa: E402

# -----------------------------------------------------------------------
# Override SAM gradient emitters to match Mumbai theatre
# -----------------------------------------------------------------------
plan.SAM_EMITTERS_GRADIENT = [
    {
        "name":               "S-125_site_A",
        "lat":                19.07,
        "lon":                72.88,
        "effective_range_km": 25.0,
    },
    {
        "name":               "Barak_8_site_B",
        "lat":                18.95,
        "lon":                73.05,
        "effective_range_km": 35.0,
    },
    {
        "name":               "Akash_site_C",
        "lat":                19.20,
        "lon":                72.95,
        "effective_range_km": 20.0,
    },
]

# -----------------------------------------------------------------------
# Run via parse_args() so argparse defaults still apply
# -----------------------------------------------------------------------
if __name__ == "__main__":
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    perception_csv = (
        REPO_ROOT / "outputs" / "perception_dataset_mumbai" / "perception_full_dataset.csv"
    )
    output_path = REPO_ROOT / "outputs" / "planning_dataset" / "planning_mumbai.parquet"

    if not perception_csv.exists():
        print(f"ERROR: Mumbai perception CSV not found at:\n  {perception_csv}")
        print("Run 'python scripts/perception/run_mumbai.py' first.")
        sys.exit(1)

    # Inject CLI args so parse_args() picks them up
    sys.argv = [
        "run_mumbai.py",
        "--perception_csv", str(perception_csv),
        "--output",         str(output_path),
        "--n_trajectories", "2000",
        "--seed",           "43",   # different seed from Delhi (42)
    ]

    plan.main()
