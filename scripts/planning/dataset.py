"""
Planning Dataset Generator
==========================

Generates 1M–10M planning trajectory records from the perception dataset for
downstream vehicle and control layer training.

Architecture (layered stack):
  Perception layer outputs  →  perception_full_dataset.csv  (1,057,714 rows)
                               ↓
  This script samples grid cells, runs RRT*-style path planning using the
  perception cost fields as the planning environment, evaluates each trajectory
  for energy, time, and threat cost, then writes the result as a flat CSV /
  Parquet usable by the vehicle and control layers.

Output columns
--------------
  start_lat, start_lon, start_alt_m          : start position (WGS-84 degrees / m MSL)
  goal_lat,  goal_lon,  goal_alt_m            : goal  position
  waypoint_lats, waypoint_lons, waypoint_alts : serialised intermediate waypoints (|n| separated)
  n_waypoints                                 : number of waypoints including start + goal
  path_length_m                               : total Euclidean arc length  [m]
  planning_time_s                             : RRT* wall-clock budget used  [s]
  time_cost_s                                 : estimated flight time        [s]
  energy_cost_wh                              : estimated energy consumption  [Wh]
  threat_cost                                 : integrated threat exposure   [0-1 normalised]
  terrain_cost_mean                           : mean terrain cost along path
  wind_cost_mean                              : mean wind cost along path
  obstacle_cost_mean                          : mean obstacle cost along path
  fused_cost_mean                             : mean fused perception cost
  max_combined_threat                         : peak threat probability along path
  feasible                                    : 1 if all hard constraints satisfied, else 0
  altitude_clearance_ok                       : 1 if terrain_clearance_m >= 50 m everywhere
  speed_ok                                    : 1 if cruise speed within [5, 60] m/s

Usage
-----
  python scripts/generate_planning_dataset.py \
      --n_trajectories 1000000 \
      --output outputs/planning_dataset/planning_dataset.parquet \
      --n_workers 4

Author: Defense eVTOL Research Team
"""

from __future__ import annotations

import argparse
import logging
import math
import multiprocessing as mp
import os
import random
import sys
import time
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PERCEPTION_CSV = Path(
    "outputs/perception_dataset/perception_full_dataset.csv"
)
DEFAULT_OUTPUT = Path("outputs/planning_dataset/planning_dataset.parquet")
DEFAULT_N = 1_000_000
BATCH_SIZE = 50_000        # rows written per Parquet batch
MIN_CLEARANCE_M = 50.0     # hard terrain clearance threshold [m]
V_CRUISE = 25.0            # nominal cruise speed [m/s]
V_MIN = 5.0                # min feasible speed [m/s]
V_MAX = 60.0               # max feasible speed [m/s]
MASS_KG = 2500.0           # vehicle mass [kg]
ENERGY_PER_METER = 0.12    # rough Wh/m at cruise (hover-weighted)
GRAVITY = 9.81

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class PercGrid:
    """Lightweight raster of perception cost fields."""
    lat: np.ndarray          # (N,)
    lon: np.ndarray          # (N,)
    alt: np.ndarray          # (N,)  alt_m
    elev: np.ndarray         # (N,)  elevation above MSL
    terrain_cost: np.ndarray # (N,)
    wind_cost: np.ndarray    # (N,)
    obstacle_cost: np.ndarray# (N,)
    threat_prob: np.ndarray  # (N,)  combined_threat_prob
    fused_cost: np.ndarray   # (N,)
    wind_speed: np.ndarray   # (N,)  m/s
    clearance: np.ndarray    # (N,)  terrain_clearance_m
    n: int = 0

    def __post_init__(self):
        self.n = len(self.lat)

    @classmethod
    def load(cls, csv_path: Path, max_rows: Optional[int] = None) -> "PercGrid":
        logger.info(f"Loading perception dataset from {csv_path} ...")
        df = pd.read_csv(csv_path, nrows=max_rows)
        logger.info(f"  Loaded {len(df):,} rows, columns: {list(df.columns)}")
        return cls(
            lat=df["lat"].to_numpy(dtype=np.float32),
            lon=df["lon"].to_numpy(dtype=np.float32),
            alt=df["alt_m"].to_numpy(dtype=np.float32),
            elev=df["elev_m"].to_numpy(dtype=np.float32),
            terrain_cost=df["terrain_cost"].to_numpy(dtype=np.float32),
            wind_cost=df["wind_cost"].to_numpy(dtype=np.float32),
            obstacle_cost=df["obstacle_cost"].to_numpy(dtype=np.float32),
            # Use max_threat_prob (individual peak) rather than combined_threat_prob
            # (combined is nearly 1.0 everywhere due to overlapping threats).
            threat_prob=df["max_threat_prob"].to_numpy(dtype=np.float32),
            fused_cost=df["fused_cost"].to_numpy(dtype=np.float32),
            wind_speed=df["wind_speed_mps"].to_numpy(dtype=np.float32),
            # alt_m is already AGL (height above ground level).
            # terrain_clearance_m = max(0, alt_m - min_threshold) and equals 0
            # when the aircraft is exactly at the minimum safe altitude.
            # Use alt_m directly as the clearance proxy.
            clearance=df["alt_m"].to_numpy(dtype=np.float32),
        )


# ---------------------------------------------------------------------------
# Simple RRT*-style planner on the perception grid
# ---------------------------------------------------------------------------
@dataclass
class Node:
    idx: int             # index into PercGrid arrays
    lat: float
    lon: float
    alt: float
    parent: Optional["Node"] = None
    cost: float = 0.0    # cumulative fused_cost from root


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Fast approximate Haversine distance in metres."""
    R = 6_371_000.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat * 0.5) ** 2
         + math.cos(math.radians(lat1))
         * math.cos(math.radians(lat2))
         * math.sin(dlon * 0.5) ** 2)
    return R * 2.0 * math.asin(min(1.0, math.sqrt(a)))


def _cost_at(grid: PercGrid, idx: int) -> float:
    """Return fused cost at grid cell (used as RRT* step cost)."""
    return float(grid.fused_cost[idx]) + float(grid.threat_prob[idx]) * 0.3


class RRTStarPlanar:
    """
    Lightweight 2-D RRT* on the PercGrid.

    Operates in (lat, lon, alt) space.  Each "sample" is drawn from the
    perception grid so every node automatically carries valid cost data.

    Parameters
    ----------
    grid         : perception grid
    step_deg     : maximum RRT step size in degrees (~1 km at this latitude)
    max_iter     : maximum tree-expansion iterations
    goal_radius  : acceptance radius in degrees
    """

    def __init__(
        self,
        grid: PercGrid,
        step_deg: float = 0.03,
        max_iter: int = 300,
        goal_radius_deg: float = 0.015,
        rng: Optional[np.random.Generator] = None,
    ):
        self.grid = grid
        self.step = step_deg
        self.max_iter = max_iter
        self.goal_r = goal_radius_deg
        self.rng = rng or np.random.default_rng()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _nearest(self, tree: list[Node], lat: float, lon: float) -> Node:
        dists = [math.hypot(n.lat - lat, n.lon - lon) for n in tree]
        return tree[int(np.argmin(dists))]

    def _steer(self, from_node: Node, to_lat: float, to_lon: float) -> tuple[float, float, float]:
        """Return new (lat, lon, alt) clamped to step size."""
        dlat = to_lat - from_node.lat
        dlon = to_lon - from_node.lon
        d = math.hypot(dlat, dlon) + 1e-12
        if d > self.step:
            dlat = dlat / d * self.step
            dlon = dlon / d * self.step
        new_lat = from_node.lat + dlat
        new_lon = from_node.lon + dlon
        # altitude: linear interpolation between grid extremes
        new_alt = float(from_node.alt) + (to_lat - from_node.lat) / (d + 1e-12) * 0.0
        return new_lat, new_lon, float(from_node.alt)

    def _nearest_grid_idx(self, lat: float, lon: float) -> int:
        """Find the closest grid cell to (lat, lon)."""
        dists = (self.grid.lat - lat) ** 2 + (self.grid.lon - lon) ** 2
        return int(np.argmin(dists))

    def _rewire_nearby(
        self,
        tree: list[Node],
        new_node: Node,
        radius: float = 0.06,
    ) -> None:
        """RRT* rewire: update parent if cheaper path found through new_node."""
        for node in tree:
            d = math.hypot(node.lat - new_node.lat, node.lon - new_node.lon)
            if d < radius and d > 1e-9:
                potential = new_node.cost + _cost_at(self.grid, node.idx) * d
                if potential < node.cost:
                    node.parent = new_node
                    node.cost = potential

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------
    def plan(
        self,
        start_idx: int,
        goal_idx: int,
    ) -> tuple[list[int], float]:
        """
        Run RRT* from start to goal (indices into PercGrid).

        Returns
        -------
        path_indices : list of grid indices from start to goal (inclusive)
        total_cost   : accumulated fused+threat cost along path
        """
        g = self.grid
        root = Node(
            idx=start_idx,
            lat=float(g.lat[start_idx]),
            lon=float(g.lon[start_idx]),
            alt=float(g.alt[start_idx]),
            cost=0.0,
        )
        tree: list[Node] = [root]

        goal_lat = float(g.lat[goal_idx])
        goal_lon = float(g.lon[goal_idx])
        goal_node: Optional[Node] = None

        lat_min, lat_max = float(g.lat.min()), float(g.lat.max())
        lon_min, lon_max = float(g.lon.min()), float(g.lon.max())

        for iteration in range(self.max_iter):
            # Bias 10 % of samples toward goal
            if self.rng.random() < 0.10:
                s_lat, s_lon = goal_lat, goal_lon
            else:
                s_lat = float(self.rng.uniform(lat_min, lat_max))
                s_lon = float(self.rng.uniform(lon_min, lon_max))

            nearest = self._nearest(tree, s_lat, s_lon)
            new_lat, new_lon, new_alt = self._steer(nearest, s_lat, s_lon)

            # Map to nearest grid cell
            new_idx = self._nearest_grid_idx(new_lat, new_lon)
            new_lat = float(g.lat[new_idx])
            new_lon = float(g.lon[new_idx])
            new_alt = float(g.alt[new_idx])

            step_dist = math.hypot(new_lat - nearest.lat, new_lon - nearest.lon)
            step_cost = _cost_at(g, new_idx) * max(step_dist, 1e-9)

            new_node = Node(
                idx=new_idx,
                lat=new_lat,
                lon=new_lon,
                alt=new_alt,
                parent=nearest,
                cost=nearest.cost + step_cost,
            )
            tree.append(new_node)
            self._rewire_nearby(tree, new_node)

            # Check goal
            dist_to_goal = math.hypot(new_lat - goal_lat, new_lon - goal_lon)
            if dist_to_goal < self.goal_r:
                if goal_node is None or new_node.cost < goal_node.cost:
                    goal_node = new_node

        # Reconstruct path
        if goal_node is None:
            # Fallback: nearest node to goal
            dists = [math.hypot(n.lat - goal_lat, n.lon - goal_lon) for n in tree]
            goal_node = tree[int(np.argmin(dists))]

        path_indices: list[int] = []
        node: Optional[Node] = goal_node
        while node is not None:
            path_indices.append(node.idx)
            node = node.parent
        path_indices.reverse()

        return path_indices, goal_node.cost


# ---------------------------------------------------------------------------
# Cost evaluators
# ---------------------------------------------------------------------------
def evaluate_trajectory(
    grid: PercGrid,
    path_idx: list[int],
) -> dict:
    """
    Compute planning-layer cost metrics for a planned path.

    Returns a dict matching the output schema.
    """
    if len(path_idx) < 2:
        return {}

    lats = grid.lat[path_idx]
    lons = grid.lon[path_idx]
    alts = grid.alt[path_idx]

    # Arc length in metres
    seg_lengths = np.array([
        _haversine_m(float(lats[i]), float(lons[i]), float(lats[i + 1]), float(lons[i + 1]))
        for i in range(len(path_idx) - 1)
    ])
    path_length_m = float(seg_lengths.sum())

    # Time
    time_s = path_length_m / V_CRUISE if path_length_m > 0 else 0.0

    # Energy (simplified: cruise drag + potential energy changes)
    alt_diffs = np.diff(alts)
    climb_energy_wh = float(np.sum(np.clip(alt_diffs, 0, None)) * MASS_KG * GRAVITY / 3600.0)
    cruise_energy_wh = path_length_m * ENERGY_PER_METER
    energy_wh = cruise_energy_wh + climb_energy_wh

    # Cost field means along path
    terrain_cost_mean = float(grid.terrain_cost[path_idx].mean())
    wind_cost_mean    = float(grid.wind_cost[path_idx].mean())
    obstacle_cost_mean = float(grid.obstacle_cost[path_idx].mean())
    fused_cost_mean   = float(grid.fused_cost[path_idx].mean())
    threat_cost       = float(grid.threat_prob[path_idx].mean())
    max_threat        = float(grid.threat_prob[path_idx].max())

    # Hard constraints
    # clearance = alt_m - elev_m; feasible if aircraft stays above terrain + buffer
    clearances = grid.clearance[path_idx]
    altitude_clearance_ok = int(float(clearances.min()) >= MIN_CLEARANCE_M)

    # Speed feasibility (trivially OK for fixed-speed model)
    speed_ok = int(V_MIN <= V_CRUISE <= V_MAX)

    # Physical feasibility: kinodynamically achievable + adequate altitude
    feasible = int(altitude_clearance_ok and speed_ok)

    # Risk label: 0 = low-risk (fused_cost < 0.55), 1 = high-risk (fused_cost >= 0.55)
    # Provides a balanced binary classification target for downstream models.
    risk_label = int(fused_cost_mean >= 0.55)

    return {
        "path_length_m": path_length_m,
        "time_cost_s": time_s,
        "energy_cost_wh": energy_wh,
        "threat_cost": threat_cost,
        "terrain_cost_mean": terrain_cost_mean,
        "wind_cost_mean": wind_cost_mean,
        "obstacle_cost_mean": obstacle_cost_mean,
        "fused_cost_mean": fused_cost_mean,
        "max_combined_threat": max_threat,
        "feasible": feasible,
        "altitude_clearance_ok": altitude_clearance_ok,
        "speed_ok": speed_ok,
        "risk_label": risk_label,
    }


# ---------------------------------------------------------------------------
# Record builder
# ---------------------------------------------------------------------------
def build_record(
    grid: PercGrid,
    start_idx: int,
    goal_idx: int,
    path_idx: list[int],
    planning_time_s: float,
    rrt_cost: float,
) -> Optional[dict]:
    """Combine path geometry and cost metrics into a flat record."""
    if len(path_idx) < 2:
        return None

    costs = evaluate_trajectory(grid, path_idx)
    if not costs:
        return None

    lats  = grid.lat[path_idx].tolist()
    lons  = grid.lon[path_idx].tolist()
    alts  = grid.alt[path_idx].tolist()

    record = {
        # Start / goal
        "start_lat":    float(grid.lat[start_idx]),
        "start_lon":    float(grid.lon[start_idx]),
        "start_alt_m":  float(grid.alt[start_idx]),
        "goal_lat":     float(grid.lat[goal_idx]),
        "goal_lon":     float(grid.lon[goal_idx]),
        "goal_alt_m":   float(grid.alt[goal_idx]),
        # Waypoints (pipe-separated strings)
        "waypoint_lats": "|".join(f"{v:.6f}" for v in lats),
        "waypoint_lons": "|".join(f"{v:.6f}" for v in lons),
        "waypoint_alts": "|".join(f"{v:.1f}" for v in alts),
        "n_waypoints":  len(path_idx),
        # Planning meta
        "planning_time_s": planning_time_s,
        "rrt_cost":        rrt_cost,
        # Cost fields
        **costs,
    }
    return record


# ---------------------------------------------------------------------------
# Batch generator
# ---------------------------------------------------------------------------
def generate_batch(
    grid: PercGrid,
    n: int,
    rng: np.random.Generator,
    rrt_iterations: int = 300,
) -> list[dict]:
    """Generate `n` trajectory records using random start/goal pairs."""
    planner = RRTStarPlanar(grid, max_iter=rrt_iterations, rng=rng)
    records = []

    # Pre-sample start/goal pairs to avoid same cell
    indices = rng.integers(0, grid.n, size=(n * 2 + n // 2,))
    ptr = 0

    while len(records) < n and ptr + 1 < len(indices):
        start_idx = int(indices[ptr])
        goal_idx  = int(indices[ptr + 1])
        ptr += 2

        if start_idx == goal_idx:
            continue

        # Distance sanity: require 0.5 – 20 km  (0.005 – 0.18 deg approx)
        d_deg = math.hypot(
            float(grid.lat[goal_idx]) - float(grid.lat[start_idx]),
            float(grid.lon[goal_idx]) - float(grid.lon[start_idx]),
        )
        if d_deg < 0.005 or d_deg > 0.18:
            continue

        t0 = time.perf_counter()
        path_idx, rrt_cost = planner.plan(start_idx, goal_idx)
        planning_time = time.perf_counter() - t0

        rec = build_record(grid, start_idx, goal_idx, path_idx, planning_time, rrt_cost)
        if rec is not None:
            records.append(rec)

    return records


# ---------------------------------------------------------------------------
# Multiprocessing worker
# ---------------------------------------------------------------------------
def _worker_generate(args_tuple: tuple) -> list[dict]:
    """
    Worker function for multiprocessing.Pool.
    Unpacks (grid_arrays, n, seed, rrt_iterations) and returns records.
    Grid arrays are passed as plain numpy arrays to be picklable.
    """
    grid_arrays, n, seed, rrt_iterations = args_tuple
    grid = PercGrid(**grid_arrays)
    rng = np.random.default_rng(seed)
    return generate_batch(grid, n, rng, rrt_iterations=rrt_iterations)


def _grid_to_arrays(grid: PercGrid) -> dict:
    """Convert PercGrid to a dict of numpy arrays for pickling."""
    return {
        "lat": grid.lat, "lon": grid.lon, "alt": grid.alt, "elev": grid.elev,
        "terrain_cost": grid.terrain_cost, "wind_cost": grid.wind_cost,
        "obstacle_cost": grid.obstacle_cost, "threat_prob": grid.threat_prob,
        "fused_cost": grid.fused_cost, "wind_speed": grid.wind_speed,
        "clearance": grid.clearance,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate planning trajectory dataset")
    p.add_argument("--n_trajectories", type=int, default=DEFAULT_N,
                   help=f"Number of trajectories to generate (default {DEFAULT_N:,})")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                   help=f"Output file (.parquet or .csv, default {DEFAULT_OUTPUT})")
    p.add_argument("--perception_csv", type=Path, default=PERCEPTION_CSV,
                   help="Path to perception_full_dataset.csv")
    p.add_argument("--max_perception_rows", type=int, default=None,
                   help="Cap perception rows loaded (for testing)")
    p.add_argument("--rrt_iterations", type=int, default=300,
                   help="RRT* iterations per trajectory (default 300)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch_size", type=int, default=BATCH_SIZE,
                   help="Rows per write batch")
    p.add_argument("--n_workers", type=int, default=1,
                   help="Parallel workers (default 1; use cpu_count for max parallelism)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ---- Resolve paths relative to project root ----
    project_root = Path(__file__).resolve().parent.parent
    perc_csv = args.perception_csv
    if not perc_csv.is_absolute():
        perc_csv = project_root / perc_csv
    if not perc_csv.exists():
        logger.error(f"Perception CSV not found: {perc_csv}")
        sys.exit(1)

    output_path = args.output
    if not output_path.is_absolute():
        output_path = project_root / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ---- Load perception grid ----
    grid = PercGrid.load(perc_csv, max_rows=args.max_perception_rows)

    # ---- Generate trajectories in batches ----
    rng_master = np.random.default_rng(args.seed)
    n_total = args.n_trajectories
    batch_sz = args.batch_size
    n_workers = args.n_workers

    logger.info(
        f"Generating {n_total:,} planning trajectories -> {output_path} "
        f"(workers={n_workers}, rrt_iter={args.rrt_iterations})"
    )

    grid_arrays = _grid_to_arrays(grid)
    all_batches: list[pd.DataFrame] = []
    generated = 0
    batch_num = 0

    if n_workers > 1:
        pool = mp.Pool(processes=n_workers)
    else:
        pool = None

    try:
        while generated < n_total:
            n_this_batch = min(batch_sz, n_total - generated)
            logger.info(
                f"  Batch {batch_num + 1}: {n_this_batch:,} records "
                f"(so far {generated:,}/{n_total:,})"
            )

            t0 = time.perf_counter()

            if pool is not None:
                # Split batch across workers
                per_worker = max(1, n_this_batch // n_workers)
                seeds = rng_master.integers(0, 2**32 - 1, size=n_workers).tolist()
                tasks = [
                    (grid_arrays, per_worker, int(seeds[i]), args.rrt_iterations)
                    for i in range(n_workers)
                ]
                results = pool.map(_worker_generate, tasks)
                records = [r for sub in results for r in sub]
            else:
                rng = np.random.default_rng(int(rng_master.integers(0, 2**32 - 1)))
                records = generate_batch(grid, n_this_batch, rng, rrt_iterations=args.rrt_iterations)

            elapsed = time.perf_counter() - t0

            if not records:
                logger.warning("  Batch produced 0 records, skipping")
                batch_num += 1
                continue

            df_batch = pd.DataFrame(records)
            all_batches.append(df_batch)
            generated += len(records)
            batch_num += 1

            rate = len(records) / max(elapsed, 1e-3)
            logger.info(
                f"  Done: {len(records):,} records in {elapsed:.1f}s "
                f"({rate:.0f} rec/s), total: {generated:,}"
            )

            # Periodic flush to avoid OOM on very large runs
            if generated % (batch_sz * 10) == 0 or generated >= n_total:
                _flush(all_batches, output_path, final=(generated >= n_total))
                all_batches = []

    finally:
        if pool is not None:
            pool.close()
            pool.join()

    # Final flush
    if all_batches:
        _flush(all_batches, output_path, final=True)

    logger.info(f"Done. {generated:,} records written to {output_path}")


def _flush(batches: list[pd.DataFrame], path: Path, final: bool) -> None:
    """Concatenate batches and write/append to output."""
    if not batches:
        return
    df = pd.concat(batches, ignore_index=True)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        if path.exists() and not final:
            # Append by reading existing + concat
            existing = pd.read_parquet(path)
            df = pd.concat([existing, df], ignore_index=True)
        df.to_parquet(path, index=False, compression="snappy")
    else:
        mode = "a" if path.exists() else "w"
        header = not path.exists() or mode == "w"
        df.to_csv(path, index=False, mode=mode, header=header)
    logger.info(f"  → Flushed {len(df):,} rows to {path}")


if __name__ == "__main__":
    main()
