"""
expand_dataset_10x.py
=====================
Expands the current defense eVTOL dataset to 10× its original size across
all six Indian geographic regions.

Strategy
--------
Current totals (before expansion):
  Delhi   : 10,000 planning / vehicle / control records
  Others  : 2,000 each × 5 regions = 10,000

10× target:
  Delhi   : 100,000 records  (generate 90,000 new)
  Others  : 20,000 each      (generate 18,000 new each)
  Grand total: 200,000 missions × 3 layers = 600,000 dataset rows

Generation approach (analytical fast-mode)
------------------------------------------
Rather than re-running the full RRT* + NSGA-III pipeline (which would take
50+ hours), this script uses a **physics-consistent analytical generator**:

  1.  Analytical path planner: generates random waypoint sequences within each
      region's bounding box, computing path metrics via Haversine geometry.
  2.  Threat model: identical SAM distance-decay formula used by the planning
      layer (from region_configs.py) — values are physically consistent.
  3.  Cost sampling: terrain/wind/obstacle costs drawn from empirical
      distributions fitted to the existing dataset for each region.
  4.  Vehicle simulation: imports and calls simulate_mission() from
      scripts/vehicle/dataset.py (full BEMT + 2-RC battery physics).
  5.  Control simulation: imports and calls simulate_mission() from
      scripts/control/dataset.py (50 Hz cascaded PID plant simulation).

All generated records are physically self-consistent, use the same column
schema as the originals, and are merged with the existing data.

Output
------
  datasets/<region>/planning/planning_<region>_10x.csv   (+ .parquet)
  datasets/<region>/vehicle/vehicle_dataset_10x.csv      (+ .parquet)
  datasets/<region>/control/control_dataset_10x.csv      (+ .parquet)

Usage
-----
  python scripts/expand_dataset_10x.py
  python scripts/expand_dataset_10x.py --region delhi --n_new 5000  # quick test
  python scripts/expand_dataset_10x.py --dry_run  # validate only, no I/O
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ── Repository root ───────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

# ── Region configuration ──────────────────────────────────────────────────────
from region_configs import REGIONS  # noqa: E402

# ── Physical constants (planning layer) ───────────────────────────────────────
EARTH_R     = 6_371_000.0   # m
V_CRUISE    = 25.0           # m/s  nominal cruise speed for time/energy estimate
MASS_KG     = 1350.0         # kg   vehicle mass (matches vehicle/dataset.py)
GRAVITY     = 9.81           # m/s²
ENERGY_PM   = 0.30           # Wh/m  rough energy per metre at cruise
RISK_THRESH = 0.55           # fused_cost threshold for risk_label

# ── Expansion targets per region ──────────────────────────────────────────────
TARGETS: Dict[str, int] = {
    "delhi":      90_000,   # 10k → 100k
    "mumbai":     18_000,   # 2k  → 20k
    "bangalore":  18_000,
    "arunachal":  18_000,
    "odisha":     18_000,
    "ladakh":     18_000,
}


# ══════════════════════════════════════════════════════════════════════════════
# Haversine helper
# ══════════════════════════════════════════════════════════════════════════════
def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat * 0.5) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(dlon * 0.5) ** 2)
    return EARTH_R * 2.0 * math.asin(min(1.0, math.sqrt(max(0.0, a))))


# ══════════════════════════════════════════════════════════════════════════════
# SAM distance-decay threat (identical to planning/dataset.py)
# ══════════════════════════════════════════════════════════════════════════════
def _threat_at(lat: float, lon: float, emitters: List[dict]) -> float:
    """Combined SAM detection probability using exponential decay."""
    prob_none = 1.0
    for e in emitters:
        r = haversine_m(lat, lon, e["lat"], e["lon"])
        r_eff = e["effective_range_km"] * 1000.0
        p_i = math.exp(-r / r_eff)
        prob_none *= (1.0 - p_i)
    return 1.0 - prob_none


# ══════════════════════════════════════════════════════════════════════════════
# Empirical distribution parameters from existing data
# ══════════════════════════════════════════════════════════════════════════════
def _fit_distributions(df: pd.DataFrame) -> dict:
    """Fit Gaussian parameters for cost columns from existing data."""
    cols = ["terrain_cost_mean", "wind_cost_mean", "obstacle_cost_mean", "fused_cost_mean"]
    params = {}
    for c in cols:
        if c in df.columns:
            v = df[c].dropna().values.astype(float)
            params[c] = (float(v.mean()), float(v.std()) + 1e-6, float(v.min()), float(v.max()))
        else:
            params[c] = (0.3, 0.15, 0.0, 1.0)
    return params


# ══════════════════════════════════════════════════════════════════════════════
# Analytical planning record generator
# ══════════════════════════════════════════════════════════════════════════════
def generate_planning_records(
    region_cfg: dict,
    dist_params: dict,
    n: int,
    seed: int,
) -> List[dict]:
    """
    Generate `n` analytically-computed planning records for a region.

    Uses:
    - Random start/goal pairs within bounding box
    - 3–8 intermediate waypoints along a perturbed straight-line path
    - Haversine path length
    - SAM distance-decay threat (same formula as planning layer)
    - Gaussian-sampled terrain/wind/obstacle costs from empirical distribution
    - Physics-based energy and time estimates
    """
    rng = np.random.default_rng(seed)
    emitters = region_cfg["sam_gradient_emitters"]

    lat_min, lat_max = region_cfg["lat_min"], region_cfg["lat_max"]
    lon_min, lon_max = region_cfg["lon_min"], region_cfg["lon_max"]
    lat_span = lat_max - lat_min
    lon_span = lon_max - lon_min

    records: List[dict] = []
    attempts = 0
    max_attempts = n * 4

    while len(records) < n and attempts < max_attempts:
        attempts += 1

        # ── Sample start and goal ─────────────────────────────────────────────
        s_lat = float(rng.uniform(lat_min + 0.05 * lat_span, lat_max - 0.05 * lat_span))
        s_lon = float(rng.uniform(lon_min + 0.05 * lon_span, lon_max - 0.05 * lon_span))
        g_lat = float(rng.uniform(lat_min + 0.05 * lat_span, lat_max - 0.05 * lat_span))
        g_lon = float(rng.uniform(lon_min + 0.05 * lon_span, lon_max - 0.05 * lon_span))
        s_alt = float(rng.uniform(100.0, 400.0))
        g_alt = float(rng.uniform(100.0, 400.0))

        # Distance filter: 2–40 km
        d_direct = haversine_m(s_lat, s_lon, g_lat, g_lon)
        if d_direct < 2_000 or d_direct > 40_000:
            continue

        # ── Generate intermediate waypoints ───────────────────────────────────
        n_wp = int(rng.integers(3, 9))   # 3–8 intermediates + start + goal
        fracs = np.sort(rng.uniform(0.1, 0.9, n_wp))
        lats = [s_lat] + [s_lat + f * (g_lat - s_lat) + rng.normal(0, 0.005) for f in fracs] + [g_lat]
        lons = [s_lon] + [s_lon + f * (g_lon - s_lon) + rng.normal(0, 0.005) for f in fracs] + [g_lon]
        alts = [s_alt] + [float(rng.uniform(100, 500)) for _ in fracs] + [g_alt]

        # Clamp to bounding box
        lats = [float(np.clip(v, lat_min, lat_max)) for v in lats]
        lons = [float(np.clip(v, lon_min, lon_max)) for v in lons]

        # ── Path length ───────────────────────────────────────────────────────
        segs = [haversine_m(lats[i], lons[i], lats[i+1], lons[i+1]) for i in range(len(lats)-1)]
        path_m = sum(segs)
        if path_m < 1_000:
            continue

        # ── Threat (averaged along path) ──────────────────────────────────────
        threats = [_threat_at(la, lo, emitters) for la, lo in zip(lats, lons)]
        threat_cost = float(np.mean(threats))
        max_threat  = float(np.max(threats))

        # ── Terrain/wind/obstacle costs from empirical distributions ──────────
        def _sample(key: str) -> float:
            mu, sigma, lo, hi = dist_params[key]
            return float(np.clip(rng.normal(mu, sigma), lo, hi))

        terrain_cost_mean  = _sample("terrain_cost_mean")
        wind_cost_mean     = _sample("wind_cost_mean")
        obstacle_cost_mean = _sample("obstacle_cost_mean")
        fused_cost_mean    = float(np.clip(
            0.35 * terrain_cost_mean + 0.20 * wind_cost_mean
            + 0.20 * obstacle_cost_mean + 0.25 * threat_cost, 0.0, 1.0
        ))

        # ── Time and energy ───────────────────────────────────────────────────
        v_cruise = float(np.clip(rng.normal(V_CRUISE, 5.0), 15.0, 60.0))
        time_s   = path_m / v_cruise
        alt_gains = sum(max(0, alts[i+1] - alts[i]) for i in range(len(alts)-1))
        energy_wh = path_m * ENERGY_PM + alt_gains * MASS_KG * GRAVITY / 3600.0

        # ── Feasibility and risk label ─────────────────────────────────────────
        alt_ok   = int(min(alts) >= 50.0)
        speed_ok = int(V_CRUISE >= 5.0)
        feasible = int(alt_ok and speed_ok)
        risk_label = int(fused_cost_mean >= RISK_THRESH)

        # ── RRT* planning metadata ────────────────────────────────────────────
        rrt_cost      = float(fused_cost_mean * path_m / 1000.0)
        planning_time = float(rng.uniform(0.05, 0.5))

        records.append({
            "start_lat":  s_lat, "start_lon": s_lon, "start_alt_m": s_alt,
            "goal_lat":   g_lat, "goal_lon":  g_lon, "goal_alt_m":  g_alt,
            "waypoint_lats": "|".join(f"{v:.6f}" for v in lats),
            "waypoint_lons": "|".join(f"{v:.6f}" for v in lons),
            "waypoint_alts": "|".join(f"{v:.1f}" for v in alts),
            "n_waypoints":   len(lats),
            "planning_time_s": planning_time,
            "rrt_cost":        rrt_cost,
            "path_length_m":   path_m,
            "time_cost_s":     time_s,
            "energy_cost_wh":  energy_wh,
            "threat_cost":     threat_cost,
            "terrain_cost_mean":  terrain_cost_mean,
            "wind_cost_mean":     wind_cost_mean,
            "obstacle_cost_mean": obstacle_cost_mean,
            "fused_cost_mean":    fused_cost_mean,
            "max_combined_threat": max_threat,
            "feasible":           feasible,
            "altitude_clearance_ok": alt_ok,
            "speed_ok":            speed_ok,
            "risk_label":          risk_label,
        })

    return records


# ══════════════════════════════════════════════════════════════════════════════
# Vehicle simulation wrapper (imports from scripts/vehicle/dataset.py)
# ══════════════════════════════════════════════════════════════════════════════
def _load_vehicle_module():
    """Import simulate_mission from scripts/vehicle/dataset.py."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "vehicle_dataset",
        ROOT / "scripts" / "vehicle" / "dataset.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def generate_vehicle_records(planning_df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Run physics-based vehicle simulation on all planning records."""
    veh_mod = _load_vehicle_module()
    simulate_fn = veh_mod.simulate_mission

    rows = []
    n = len(planning_df)
    t0 = time.perf_counter()
    for i, (_, row) in enumerate(planning_df.iterrows()):
        try:
            res = simulate_fn(row)
            # Carry forward key planning columns
            res["path_length_m"]  = row["path_length_m"]
            res["risk_label"]     = row["risk_label"]
            res["feasible"]       = row["feasible"]
            res["n_waypoints"]    = row["n_waypoints"]
            rows.append(res)
        except Exception as exc:
            if verbose and i < 5:
                print(f"  [warn] vehicle sim failed row {i}: {exc}")
        if verbose and (i+1) % 1000 == 0:
            elapsed = time.perf_counter() - t0
            rate = (i+1) / elapsed
            eta  = (n - i - 1) / rate
            print(f"  vehicle {i+1:>6}/{n}  {rate:.1f} rec/s  ETA {eta/60:.1f} min")
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# Control simulation wrapper (imports from scripts/control/dataset.py)
# ══════════════════════════════════════════════════════════════════════════════
def _load_control_module():
    """Import simulate_mission from scripts/control/dataset.py."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "control_dataset",
        ROOT / "scripts" / "control" / "dataset.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def generate_control_records(vehicle_df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Run 50 Hz cascaded-PID simulation on all vehicle records."""
    ctrl_mod = _load_control_module()
    simulate_fn = ctrl_mod.simulate_mission

    rows = []
    n = len(vehicle_df)
    t0 = time.perf_counter()
    for i, (_, row) in enumerate(vehicle_df.iterrows()):
        try:
            res = simulate_fn(row)
            rows.append(res)
        except Exception as exc:
            if verbose and i < 5:
                print(f"  [warn] control sim failed row {i}: {exc}")
        if verbose and (i+1) % 500 == 0:
            elapsed = time.perf_counter() - t0
            rate = (i+1) / elapsed
            eta  = (n - i - 1) / rate
            print(f"  control {i+1:>6}/{n}  {rate:.1f} rec/s  ETA {eta/60:.1f} min")
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# I/O helpers
# ══════════════════════════════════════════════════════════════════════════════
REGION_PLANNING_PATHS = {
    "delhi":     "datasets/delhi/planning_dataset/planning_dataset_10k.csv",
    "mumbai":    "datasets/mumbai/planning_dataset/planning_mumbai.csv",
    "bangalore": "datasets/bangalore/planning_dataset/planning_bangalore.csv",
    "arunachal": "datasets/arunachal/planning/planning_arunachal.csv",
    "odisha":    "datasets/odisha/planning_dataset/planning_odisha.csv",
    "ladakh":    "datasets/ladakh/planning_dataset/planning_ladakh.csv",
}
REGION_VEHICLE_PATHS = {
    "delhi":     "datasets/delhi/vehicle/vehicle_dataset.csv",
    "mumbai":    "datasets/mumbai/vehicle/vehicle_dataset.csv",
    "bangalore": "datasets/bangalore/vehicle/vehicle_dataset.csv",
    "arunachal": "datasets/arunachal/vehicle/vehicle_dataset.csv",
    "odisha":    "datasets/odisha/vehicle/vehicle_dataset.csv",
    "ladakh":    "datasets/ladakh/vehicle/vehicle_dataset.csv",
}
REGION_CONTROL_PATHS = {
    "delhi":     "datasets/delhi/control/control_dataset.csv",
    "mumbai":    "datasets/mumbai/control/control_dataset.csv",
    "bangalore": "datasets/bangalore/control/control_dataset.csv",
    "arunachal": "datasets/arunachal/control/control_dataset.csv",
    "odisha":    "datasets/odisha/control/control_dataset.csv",
    "ladakh":    "datasets/ladakh/control/control_dataset.csv",
}


def _out_dir(region: str) -> Path:
    return ROOT / "datasets" / region


def _save(df: pd.DataFrame, path: Path, tag: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    pq_path = path.with_suffix(".parquet")
    try:
        df.to_parquet(pq_path, index=False)
    except Exception:
        pass
    print(f"  [{tag}] saved {len(df):,} rows → {path.name}")


# ══════════════════════════════════════════════════════════════════════════════
# Per-region expansion
# ══════════════════════════════════════════════════════════════════════════════
def expand_region(
    region: str,
    n_new: Optional[int] = None,
    dry_run: bool = False,
    verbose: bool = True,
    batch_size: int = 5_000,
) -> None:
    """Expand one region's dataset by generating n_new additional records."""
    cfg = REGIONS[region]
    n_new = n_new if n_new is not None else TARGETS[region]

    print(f"\n{'='*64}")
    print(f"  Region: {region.upper()}  |  Generating {n_new:,} new records")
    print(f"{'='*64}")

    # ── Load existing planning data ────────────────────────────────────────────
    plan_path = ROOT / REGION_PLANNING_PATHS[region]
    if not plan_path.exists():
        print(f"  [ERROR] existing planning file not found: {plan_path}")
        return
    existing_plan = pd.read_csv(plan_path, low_memory=False)
    print(f"  Existing planning records: {len(existing_plan):,}")

    # ── Fit empirical distributions from existing data ─────────────────────────
    dist_params = _fit_distributions(existing_plan)
    print(f"  Fitted distributions: {dist_params}")

    # ── Generate new planning records in batches ───────────────────────────────
    all_new_plan: List[pd.DataFrame] = []
    generated = 0
    batch_num = 0
    t0 = time.perf_counter()

    while generated < n_new:
        batch_n = min(batch_size, n_new - generated)
        seed = hash(f"{region}_{batch_num}") % (2**31)
        recs = generate_planning_records(cfg, dist_params, batch_n, seed)
        if not recs:
            print(f"  [warn] batch {batch_num} produced 0 records, retrying")
            batch_num += 1
            continue
        all_new_plan.append(pd.DataFrame(recs))
        generated += len(recs)
        rate = generated / max(time.perf_counter() - t0, 0.01)
        print(f"  Planning batch {batch_num}: {len(recs):,} → total {generated:,}  ({rate:.0f} rec/s)")
        batch_num += 1

    new_plan_df = pd.concat(all_new_plan, ignore_index=True)
    combined_plan = pd.concat([existing_plan, new_plan_df], ignore_index=True)
    print(f"  Planning combined: {len(combined_plan):,} rows")

    if dry_run:
        print("  [dry_run] skipping vehicle/control simulation and file I/O")
        return

    # ── Save expanded planning dataset ─────────────────────────────────────────
    out = _out_dir(region)
    plan_subdir = "planning_dataset" if region != "arunachal" else "planning"
    _save(
        combined_plan,
        out / plan_subdir / f"planning_{region}_10x.csv",
        "planning",
    )

    # ── Vehicle simulation on new records ──────────────────────────────────────
    print(f"\n  Running vehicle simulation on {len(new_plan_df):,} new records …")
    new_veh_df = generate_vehicle_records(new_plan_df, verbose=verbose)

    existing_veh_path = ROOT / REGION_VEHICLE_PATHS[region]
    if existing_veh_path.exists():
        existing_veh = pd.read_csv(existing_veh_path, low_memory=False)
        combined_veh = pd.concat([existing_veh, new_veh_df], ignore_index=True)
    else:
        combined_veh = new_veh_df
    print(f"  Vehicle combined: {len(combined_veh):,} rows")
    _save(combined_veh, out / "vehicle" / f"vehicle_dataset_10x.csv", "vehicle")

    # ── Control simulation on new vehicle records ──────────────────────────────
    print(f"\n  Running control simulation on {len(new_veh_df):,} new records …")
    new_ctrl_df = generate_control_records(new_veh_df, verbose=verbose)

    existing_ctrl_path = ROOT / REGION_CONTROL_PATHS[region]
    if existing_ctrl_path.exists():
        existing_ctrl = pd.read_csv(existing_ctrl_path, low_memory=False)
        combined_ctrl = pd.concat([existing_ctrl, new_ctrl_df], ignore_index=True)
    else:
        combined_ctrl = new_ctrl_df
    print(f"  Control combined: {len(combined_ctrl):,} rows")
    _save(combined_ctrl, out / "control" / f"control_dataset_10x.csv", "control")

    elapsed = time.perf_counter() - t0
    print(f"\n  ✓ {region} complete in {elapsed/60:.1f} min")
    print(f"    Planning: {len(combined_plan):,} rows")
    print(f"    Vehicle:  {len(combined_veh):,} rows")
    print(f"    Control:  {len(combined_ctrl):,} rows")


# ══════════════════════════════════════════════════════════════════════════════
# Summary report
# ══════════════════════════════════════════════════════════════════════════════
def print_summary(regions_done: List[str]) -> None:
    print(f"\n{'='*64}")
    print("  EXPANSION SUMMARY")
    print(f"{'='*64}")
    total_plan = total_veh = total_ctrl = 0
    for r in regions_done:
        out = ROOT / "datasets" / r
        plan_subdir = "planning_dataset" if r != "arunachal" else "planning"
        pf = out / plan_subdir / f"planning_{r}_10x.csv"
        vf = out / "vehicle" / "vehicle_dataset_10x.csv"
        cf = out / "control" / "control_dataset_10x.csv"
        np_ = sum(1 for _ in open(pf)) - 1 if pf.exists() else 0
        nv  = sum(1 for _ in open(vf)) - 1 if vf.exists() else 0
        nc  = sum(1 for _ in open(cf)) - 1 if cf.exists() else 0
        print(f"  {r:<15}  plan={np_:>7,}  veh={nv:>7,}  ctrl={nc:>7,}")
        total_plan += np_; total_veh += nv; total_ctrl += nc
    print(f"  {'TOTAL':<15}  plan={total_plan:>7,}  veh={total_veh:>7,}  ctrl={total_ctrl:>7,}")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Expand defense eVTOL dataset to 10× size")
    p.add_argument("--region", choices=list(REGIONS) + ["all"], default="all",
                   help="Region to expand (default: all)")
    p.add_argument("--n_new", type=int, default=None,
                   help="Override number of new records to generate (default from TARGETS)")
    p.add_argument("--dry_run", action="store_true",
                   help="Generate planning records only, skip simulation and file I/O")
    p.add_argument("--batch_size", type=int, default=5_000,
                   help="Records per planning batch (default 5000)")
    p.add_argument("--quiet", action="store_true", help="Suppress per-record progress")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    regions = list(REGIONS) if args.region == "all" else [args.region]
    verbose = not args.quiet

    print("Defense eVTOL Dataset 10× Expansion")
    print(f"Target regions : {regions}")
    print(f"Dry run        : {args.dry_run}")
    print(f"Batch size     : {args.batch_size:,}")
    print()

    t_global = time.perf_counter()
    for r in regions:
        expand_region(
            region=r,
            n_new=args.n_new,
            dry_run=args.dry_run,
            verbose=verbose,
            batch_size=args.batch_size,
        )

    if not args.dry_run:
        print_summary(regions)

    elapsed = time.perf_counter() - t_global
    print(f"\nTotal elapsed: {elapsed/60:.1f} min ({elapsed/3600:.2f} h)")


if __name__ == "__main__":
    main()
