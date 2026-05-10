#!/usr/bin/env python3
"""
run_region.py — Generalized perception dataset runner for any defined region.

Fetches REAL geospatial data from live APIs:
  - SRTM terrain elevation  (Open-Meteo elevation API)
  - GFS/HRES wind forecasts (Open-Meteo forecast API)
  - Obstacle inventory      (OpenStreetMap Overpass API)
  - SAM threat field        (analytical, from region_configs.py)

Outputs land in:
  datasets/<region>/perception_dataset/
    perception_terrain.csv
    perception_obstacle.csv
    perception_full_dataset.csv
    osm_obstacle_inventory.csv
    perception_metadata.json

Usage
------------------------------------------------------------------------------------
  python scripts/perception/run_region.py --region srinagar
  python scripts/perception/run_region.py --region chennai
  python scripts/perception/run_region.py --region all   # run all 10 new regions

To regenerate any of the original six regions:
  python scripts/perception/run_region.py --region delhi
  python scripts/perception/run_region.py --region mumbai
  ... etc.

Each region takes ~5–20 minutes depending on API response times.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "perception"))
sys.path.insert(0, str(REPO_ROOT / "src"))

# Import region configs BEFORE dataset so we can patch constants
from region_configs import REGIONS  # noqa: E402
import dataset as perc              # noqa: E402


def _subdir_for(region: str) -> str:
    """Match the existing naming convention used by the original 6 regions."""
    return "perception" if region == "arunachal" else "perception_dataset"


def run_perception_for_region(region: str) -> None:
    cfg = REGIONS[region]

    # ── Patch theatre bounds ───────────────────────────────────────────────────
    perc.LAT_MIN = cfg["lat_min"]
    perc.LAT_MAX = cfg["lat_max"]
    perc.LON_MIN = cfg["lon_min"]
    perc.LON_MAX = cfg["lon_max"]

    # ── Patch eVTOL RCS ───────────────────────────────────────────────────────
    perc.EVTOL_RCS_SQM = cfg.get("evtol_rcs_sqm", 0.5)

    # ── Patch SAM systems from region config ──────────────────────────────────
    perc.SAM_SYSTEMS = cfg["sam_systems"]

    # ── Redirect output to datasets/<region>/perception_dataset/ ─────────────
    out_dir = REPO_ROOT / "datasets" / region / _subdir_for(region)
    perc.OUTPUTS_DIR = out_dir
    perc.CACHE_DIR   = out_dir / "cache"

    logging.getLogger("perception_dataset").info(
        "=== %s  |  lat %.2f-%.2f  lon %.2f-%.2f  |  output -> %s",
        region.upper(),
        cfg["lat_min"], cfg["lat_max"],
        cfg["lon_min"], cfg["lon_max"],
        out_dir,
    )

    perc.main()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate real perception data (SRTM/Open-Meteo/OSM) for any region."
    )
    p.add_argument(
        "--region",
        choices=list(REGIONS) + ["all"],
        required=True,
        help="Region name from region_configs.py, or 'all' to run every region.",
    )
    p.add_argument(
        "--only_new",
        action="store_true",
        help="When --region all, skip the original 6 regions (which already have data).",
    )
    return p.parse_args()


ORIGINAL_SIX = {"delhi", "mumbai", "bangalore", "arunachal", "odisha", "ladakh"}
NEW_TEN = {r for r in REGIONS if r not in ORIGINAL_SIX}


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )

    args = parse_args()

    if args.region == "all":
        targets = list(REGIONS)
        if args.only_new:
            targets = [r for r in targets if r in NEW_TEN]
    else:
        targets = [args.region]

    for region in targets:
        out_dir = REPO_ROOT / "datasets" / region / _subdir_for(region)
        terrain_csv = out_dir / "perception_terrain.csv"
        if terrain_csv.exists():
            logging.getLogger("perception_dataset").info(
                "[%s] perception_terrain.csv already exists — skipping "
                "(delete the folder to force regeneration)", region
            )
            continue
        logging.getLogger("perception_dataset").info(
            "Starting perception pipeline for: %s", region
        )
        try:
            run_perception_for_region(region)
        except Exception as exc:
            logging.getLogger("perception_dataset").error(
                "[%s] FAILED: %s — skipping, will retry on next run", region, exc
            )
            continue


if __name__ == "__main__":
    main()
