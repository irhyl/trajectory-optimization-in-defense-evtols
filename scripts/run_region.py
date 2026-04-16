#!/usr/bin/env python3
"""
run_region.py - Unified multi-region perception + planning pipeline runner.

Usage:
  python scripts/run_region.py --region mumbai
  python scripts/run_region.py --region bangalore --only perception
  python scripts/run_region.py --region ladakh --only planning
  python scripts/run_region.py --region arunachal --n_trajectories 2000

Each region writes to outputs/<region_name>/:
  perception_dataset/perception_full_dataset.csv
  planning_dataset/planning_<region>.parquet

Available regions: delhi, mumbai, bangalore, arunachal, odisha, ladakh
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
# Add repo src/ so evtol.perception.* is importable (used by perception/dataset.py)
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from region_configs import REGIONS  # noqa: E402


def _import_module(path: Path, name: str):
    """Import a Python file by absolute path with a unique module name."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def run_perception(region_cfg: dict) -> Path:
    """Import and run the perception dataset generator for the given region."""
    perc = _import_module(REPO_ROOT / "scripts" / "perception" / "dataset.py", "perc_dataset")

    perc.LAT_MIN = region_cfg["lat_min"]
    perc.LAT_MAX = region_cfg["lat_max"]
    perc.LON_MIN = region_cfg["lon_min"]
    perc.LON_MAX = region_cfg["lon_max"]
    perc.SAM_SYSTEMS = region_cfg["sam_systems"]
    perc.EVTOL_RCS_SQM = region_cfg["evtol_rcs_sqm"]

    out_dir = REPO_ROOT / "outputs" / region_cfg["outputs_subdir"] / "perception_dataset"
    out_dir.mkdir(parents=True, exist_ok=True)
    perc.OUTPUTS_DIR = out_dir
    perc.CACHE_DIR = out_dir / "cache"

    import logging
    logging.getLogger("perception_dataset").info(
        "Region: %s  lat=%.2f-%.2f  lon=%.2f-%.2f",
        region_cfg["name"],
        region_cfg["lat_min"], region_cfg["lat_max"],
        region_cfg["lon_min"], region_cfg["lon_max"],
    )
    perc.main()
    return out_dir / "perception_full_dataset.csv"


def run_planning(region_cfg: dict, perception_csv: Path, n_trajectories: int) -> Path:
    """Import and run the planning dataset generator for the given region."""
    plan = _import_module(REPO_ROOT / "scripts" / "planning" / "dataset.py", "plan_dataset")

    plan.SAM_EMITTERS_GRADIENT = region_cfg["sam_gradient_emitters"]

    out_dir = REPO_ROOT / "outputs" / region_cfg["outputs_subdir"] / "planning_dataset"
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / ("planning_" + region_cfg["name"] + ".parquet")

    # Override sys.argv so plan.parse_args() picks up correct arguments
    sys.argv = [
        "run_region.py",
        "--perception_csv", str(perception_csv),
        "--output",         str(output_path),
        "--n_trajectories", str(n_trajectories),
        "--seed",           str(region_cfg["planning_seed"]),
    ]
    plan.main()
    return output_path


def main():
    # Parse our own args BEFORE modifying sys.argv
    p = argparse.ArgumentParser(description="Multi-region perception + planning pipeline")
    p.add_argument("--region", required=True, choices=list(REGIONS), help="Region name")
    p.add_argument("--only", choices=["perception", "planning"], default=None,
                   help="Run only one stage (default: both)")
    p.add_argument("--n_trajectories", type=int, default=2000,
                   help="Number of planning trajectories (default: 2000)")
    args = p.parse_args()

    region_cfg = REGIONS[args.region]
    n_traj = args.n_trajectories
    only   = args.only

    print(f"=== {region_cfg['description']} ===")

    perception_csv = (
        REPO_ROOT / "outputs" / region_cfg["outputs_subdir"]
        / "perception_dataset" / "perception_full_dataset.csv"
    )

    if only != "planning":
        perception_csv = run_perception(region_cfg)

    if only != "perception":
        if not perception_csv.exists():
            print(f"ERROR: perception CSV not found at {perception_csv}")
            print("Run with --only perception first.")
            sys.exit(1)
        run_planning(region_cfg, perception_csv, n_traj)

    print(f"Done. Outputs in outputs/{region_cfg['outputs_subdir']}/ ")


if __name__ == "__main__":
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    main()
