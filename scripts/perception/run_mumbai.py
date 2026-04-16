#!/usr/bin/env python3
"""
Mumbai Metropolitan Region — Perception Dataset Runner
=======================================================

Second geographic region for the defense eVTOL generalization benchmark.

Theatre: Mumbai Metropolitan Region, India
  Bounds: 18.85–19.35°N, 72.75–73.35°E
  Area:   ~55 km (N-S) × ~60 km (E-W)
  Terrain: Coastal plain + Western Ghats foothills + Salsette Island

Contrast with Delhi NCR:
  - Coastal terrain (sea-level at west coast → 200 m hills in east)
  - Markedly different wind patterns (monsoon sea-breeze vs continental)
  - Higher obstacle density (offshore platforms, antenna farms)
  - Different threat scenario: naval SA-N / coastal defense mix

Run this script to generate Mumbai perception data in
  outputs/perception_dataset_mumbai/

Then run:
  python scripts/planning/dataset.py \
      --perception_csv outputs/perception_dataset_mumbai/perception_full_dataset.csv \
      --output outputs/planning_dataset/planning_mumbai.parquet \
      --n_trajectories 2000

Usage
-----
  python scripts/perception/run_mumbai.py

Outputs to: outputs/perception_dataset_mumbai/
"""

from __future__ import annotations

import sys
from pathlib import Path

# Patch constants before importing the dataset module
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "perception"))

import dataset as perc  # noqa: E402

# -----------------------------------------------------------------------
# Override theatre bounds
# -----------------------------------------------------------------------
perc.LAT_MIN = 18.85
perc.LAT_MAX = 19.35
perc.LON_MIN = 72.75
perc.LON_MAX = 73.35

# -----------------------------------------------------------------------
# Mumbai SAM threat scenario (coastal defense + naval shore-based systems)
# Public-domain open-source threat parameterization only.
# -----------------------------------------------------------------------
perc.SAM_SYSTEMS = [
    {
        "name":            "S-125_site_A",         # Pechora / SA-3 shore battery
        "lat":             19.07,
        "lon":             72.88,
        "max_range_km":    35.0,
        "radar_power_kw":  100.0,
        "radar_gain_db":   28.0,
        "freq_ghz":        5.5,                    # C-band
        "priority":        7,
        "lethal_radius_m": 2500.0,
    },
    {
        "name":            "Barak_8_site_B",        # Israeli/Indian LR-SAM
        "lat":             18.95,
        "lon":             73.05,
        "max_range_km":    70.0,
        "radar_power_kw":  200.0,
        "radar_gain_db":   31.0,
        "freq_ghz":        9.0,                    # X-band MF-STAR
        "priority":        9,
        "lethal_radius_m": 4000.0,
    },
    {
        "name":            "Akash_site_C",          # Indian Akash SR-SAM
        "lat":             19.20,
        "lon":             72.95,
        "max_range_km":    25.0,
        "radar_power_kw":  80.0,
        "radar_gain_db":   26.0,
        "freq_ghz":        3.5,                    # S-band
        "priority":        8,
        "lethal_radius_m": 2000.0,
    },
]

# -----------------------------------------------------------------------
# Redirect outputs to a parallel directory
# -----------------------------------------------------------------------
perc.OUTPUTS_DIR = REPO_ROOT / "outputs" / "perception_dataset_mumbai"
perc.CACHE_DIR   = perc.OUTPUTS_DIR / "cache"

# -----------------------------------------------------------------------
# Run
# -----------------------------------------------------------------------
if __name__ == "__main__":
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("perception_dataset").info(
        "Mumbai Metropolitan Region — Perception Dataset Generator"
    )
    perc.main()
