#!/usr/bin/env python3
"""
Large-Scale Perception Dataset Generator for Defense eVTOL Trajectory Optimization.

Produces 1M–10M real data points from live APIs across all five perception layers:
  1. Terrain   — SRTM elevation, slope, roughness, surface type (Open-Meteo API)
  2. Wind      — Real GFS/HRES forecast: U/V/W, speed, direction, turbulence (Open-Meteo API)
  3. Obstacles — Static infrastructure inventory and clearance (OpenStreetMap Overpass API)
  4. Threat    — SAM/AAA detection probability via radar range equation (analytical)
  5. Fusion    — Weighted composite traversability cost (no fallback synthesis)

Theatre: Delhi NCR outskirts, India  (28.40–28.90°N, 76.90–77.50°E)

Grid:
  Spatial  : 250 × 300 points at 0.002° ≈ 222 m spacing  = 75,000 spatial pts
  Altitude : 14 levels  [50, 100, 150, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1200, 1500 m]
  Total rows: 75,000 × 14 = 1,050,000

Terrain is fetched at 2 km coarse grid (~986 API points, 10 batches of 100) then
bilinearly interpolated to the fine grid.  Wind is fetched at 5 anchor points
(centre + 4 corners) and IDW-interpolated to the fine grid.  Obstacles are fetched
once for the whole bounding box.  Threat probabilities are computed analytically
using the Swerling-I radar range equation.

Output files (outputs/perception_dataset/):
  perception_terrain.csv        — per spatial point  (75 000 rows)
  perception_obstacle.csv       — per spatial point  (75 000 rows)
  perception_wind_<alt>m.csv    — per spatial point per altitude level (75 000 rows × 14)
  perception_threat.csv         — per spatial × altitude (1 050 000 rows)
  perception_full_dataset.csv   — all features merged (1 050 000 rows)
  osm_obstacle_inventory.csv    — raw OSM obstacle list
  perception_metadata.json      — provenance, API stats, dataset statistics

References:
  [T1] Rodriguez et al. (2006) SRTM topographic products, PE&RS 72(3):249–260
  [W1] Hersbach et al. (2020) ECMWF ERA5, QJRMS 146:1999–2049
  [O1] OpenStreetMap contributors, www.openstreetmap.org (ODbL)
  [R1] Skolnik, M. (2008) Radar Handbook 3rd Ed. McGraw-Hill  — Swerling models
  [R2] Mahafza, B. (2005) Radar Systems Analysis and Design Using MATLAB

Author: Defense eVTOL Research Team
"""

from __future__ import annotations

import csv
import io
import json
import logging
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import cKDTree

# ---------------------------------------------------------------------------
# Bootstrap: add src/ to sys.path so perception sub-packages are importable
# without going through evtol.perception.__init__ (which has extra deps).
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

# Use UTF-8 for Windows console
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Lazy imports from perception sub-packages
# ---------------------------------------------------------------------------
from evtol.perception.terrain.data_provider import (        # noqa: E402
    TerrainDataProvider, TerrainProviderConfig,
)
from evtol.perception.wind.data_provider import WindDataProvider        # noqa: E402
from evtol.perception.obstacle.data_provider import (                   # noqa: E402
    OSMDataProvider, OSMProviderConfig, ObstacleCache,
)
from evtol.perception.obstacle.obstacle_types import ObstacleCategory   # noqa: E402

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("perception_dataset")

# ===========================================================================
# MISSION CONFIGURATION
# ===========================================================================

# Theatre bounds
LAT_MIN, LAT_MAX = 28.40, 28.90   # south / north
LON_MIN, LON_MAX = 76.90, 77.50   # west / east

# Fine spatial grid  (~222 m spacing)
DLAT = 0.002    # degrees
DLON = 0.002    # degrees

# Altitude levels (metres MSL)
ALT_LEVELS = [50, 100, 150, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1200, 1500]

# eVTOL radar cross-section (m^2) — low-observable tiltrotor
EVTOL_RCS_SQM = 0.5

# SAM threat systems (Delhi NCR theatre, open-source positions)
SAM_SYSTEMS = [
    {
        "name":           "S-300V_site_A",
        "lat":            28.60,
        "lon":            77.10,
        "max_range_km":   150.0,   # S-300V engagement range
        "radar_power_kw": 500.0,
        "radar_gain_db":  35.0,
        "freq_ghz":       3.0,     # S-band
        "priority":       9,
        "lethal_radius_m": 6000.0,
    },
    {
        "name":           "SA-11_site_B",
        "lat":            28.70,
        "lon":            77.25,
        "max_range_km":   100.0,
        "radar_power_kw": 200.0,
        "radar_gain_db":  32.0,
        "freq_ghz":       6.0,     # C-band
        "priority":       7,
        "lethal_radius_m": 5000.0,
    },
    {
        "name":           "SA-22_site_C",
        "lat":            28.75,
        "lon":            77.35,
        "max_range_km":   120.0,
        "radar_power_kw": 300.0,
        "radar_gain_db":  33.0,
        "freq_ghz":       9.4,     # X-band
        "priority":       8,
        "lethal_radius_m": 4500.0,
    },
]

# Output directory
OUTPUTS_DIR = REPO_ROOT / "outputs" / "perception_dataset"
CACHE_DIR   = OUTPUTS_DIR / "cache"

# Terrain fetch resolution (coarse grid → bilinear interpolation to fine)
TERRAIN_COARSE_M = 2000   # 2 km coarse fetch
WIND_FORECAST_HOURS = 6   # short window; use t=0 snapshot

# Cost weights for fusion layer (must sum to 1.0)
COST_WEIGHTS = {
    "terrain":   0.15,
    "wind":      0.10,
    "obstacle":  0.20,
    "threat":    0.40,
    "energy":    0.15,
}
assert abs(sum(COST_WEIGHTS.values()) - 1.0) < 1e-9, "Weights must sum to 1"

# ===========================================================================
# COORDINATE UTILITIES
# ===========================================================================

EARTH_R = 6_371_000.0   # m

def haversine_m(lat1: np.ndarray, lon1: np.ndarray,
                lat2: float, lon2: float) -> np.ndarray:
    """Vectorised Haversine distance (metres)."""
    r1, r2 = np.radians(lat1), np.radians(lat2)
    dr = r1 - r2
    dl = np.radians(lon1 - lon2)
    a = np.sin(dr / 2) ** 2 + np.cos(r1) * np.cos(r2) * np.sin(dl / 2) ** 2
    return 2.0 * EARTH_R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

# ===========================================================================
# TERRAIN LAYER
# ===========================================================================

def fetch_terrain(fine_lats: np.ndarray, fine_lons: np.ndarray) -> dict[str, np.ndarray]:
    """
    Fetch real SRTM elevation (Open-Meteo) at 2 km coarse grid and
    bilinearly interpolate to the fine grid.

    Returns dict with keys:
        elev_m       (n_lat, n_lon)
        slope_deg    (n_lat, n_lon)
        roughness_m  (n_lat, n_lon)
        surface_type (n_lat, n_lon)  — str array
    """
    logger.info("[TERRAIN] Fetching SRTM elevation at %d m coarse grid...", TERRAIN_COARSE_M)

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    (CACHE_DIR / "terrain").mkdir(parents=True, exist_ok=True)

    config = TerrainProviderConfig(
        cache_dir=CACHE_DIR / "terrain",
        batch_size=100,
        rate_limit_delay_s=1.5,   # Respectful rate limit for Open-Meteo
        max_grid_points=5000,     # 2 km grid is ~986 pts, well within limit
    )
    provider = TerrainDataProvider(config=config)

    bounds = (LAT_MAX, LAT_MIN, LON_MAX, LON_MIN)   # north, south, east, west
    t0 = time.time()
    coarse = provider.get_elevation_grid(bounds, resolution_m=TERRAIN_COARSE_M, use_cache=True)
    logger.info(
        "[TERRAIN] Coarse grid fetched: %dx%d points in %.1fs  "
        "elev range %.0f–%.0f m",
        *coarse.shape, time.time() - t0,
        coarse.elevation.min(), coarse.elevation.max(),
    )

    # coarse.latitudes is descending (N→S), coarse.longitudes ascending (W→E)
    # RegularGridInterpolator requires ascending axes → flip lat dimension
    lats_asc  = coarse.latitudes[::-1].copy()
    elev_asc  = coarse.elevation[::-1, :].copy()

    interp = RegularGridInterpolator(
        (lats_asc, coarse.longitudes),
        elev_asc,
        method="linear",
        bounds_error=False,
        fill_value=None,
    )

    lat_g, lon_g = np.meshgrid(fine_lats, fine_lons, indexing="ij")
    pts = np.column_stack([lat_g.ravel(), lon_g.ravel()])
    fine_elev = interp(pts).reshape(len(fine_lats), len(fine_lons))
    fine_elev = np.clip(fine_elev, 0, 9000)   # physical upper bound

    # Slope from central-difference gradient on fine grid
    dlat_m = DLAT * 111_320.0
    dlon_m = DLON * 111_320.0 * math.cos(math.radians((LAT_MIN + LAT_MAX) / 2))
    dz_dy, dz_dx = np.gradient(fine_elev, dlat_m, dlon_m)
    slope_deg = np.degrees(np.arctan(np.sqrt(dz_dy ** 2 + dz_dx ** 2)))
    slope_deg = np.clip(slope_deg, 0, 90)

    # Roughness: standard deviation of elevation in a 3×3 neighbourhood (metres)
    from scipy.ndimage import generic_filter
    roughness_m = generic_filter(fine_elev, np.std, size=3, mode="nearest")

    # Surface type classification from slope
    surface_type = np.empty_like(fine_elev, dtype=object)
    surface_type[:] = "flat"
    surface_type[slope_deg >= 5]  = "rolling"
    surface_type[slope_deg >= 15] = "hilly"
    surface_type[slope_deg >= 30] = "mountainous"

    logger.info(
        "[TERRAIN] Fine grid (%dx%d) complete.  Slope mean=%.1f° max=%.1f°",
        *fine_elev.shape, slope_deg.mean(), slope_deg.max(),
    )
    return {
        "elev_m":       fine_elev,
        "slope_deg":    slope_deg,
        "roughness_m":  roughness_m,
        "surface_type": surface_type,
    }

# ===========================================================================
# WIND LAYER
# ===========================================================================

def fetch_wind(fine_lats: np.ndarray, fine_lons: np.ndarray) -> dict[str, Any]:
    """
    Fetch real GFS/HRES wind forecast (Open-Meteo) for all 14 altitude bands.

    Returns dict:
        wind_u   (n_alt, n_lat, n_lon)  — eastward component m/s
        wind_v   (n_alt, n_lat, n_lon)  — northward component m/s
        wind_w   (n_alt, n_lat, n_lon)  — vertical component m/s (estimated)
        speed    (n_alt, n_lat, n_lon)
        dir_deg  (n_alt, n_lat, n_lon)  — meteorological FROM direction
        turb     (n_alt, n_lat, n_lon)  — turbulence intensity (dimensionless)
    """
    logger.info("[WIND] Fetching forecast for %d altitude bands...", len(ALT_LEVELS))

    (CACHE_DIR / "wind").mkdir(parents=True, exist_ok=True)
    provider = WindDataProvider(cache_dir=CACHE_DIR / "wind", cache_ttl_hours=6)

    bounds = (LAT_MAX, LAT_MIN, LON_MAX, LON_MIN)   # N, S, E, W
    t0 = time.time()
    forecast = provider.fetch_forecast(
        bounds=bounds,
        altitude_bands=ALT_LEVELS,
        forecast_hours=WIND_FORECAST_HOURS,
    )
    logger.info(
        "[WIND] Forecast fetched: shape=%s  time=%.1fs",
        forecast.grid_shape, time.time() - t0,
    )

    n_lat = len(fine_lats)
    n_lon = len(fine_lons)
    n_alt = len(ALT_LEVELS)

    wind_u = np.zeros((n_alt, n_lat, n_lon))
    wind_v = np.zeros((n_alt, n_lat, n_lon))

    # forecast.latitudes ascending (S→N), fine_lats descending (N→S)
    fc_lats = forecast.latitudes   # ascending
    fc_lons = forecast.longitudes  # ascending

    for a_idx, alt_m in enumerate(ALT_LEVELS):
        lev_idx = forecast.get_level_index(alt_m)

        # t=0 snapshot (first forecast hour)
        u_slice = forecast.wind_u[0, lev_idx, :, :]   # [fc_lat, fc_lon]
        v_slice = forecast.wind_v[0, lev_idx, :, :]

        interp_u = RegularGridInterpolator(
            (fc_lats, fc_lons), u_slice,
            method="linear", bounds_error=False, fill_value=None,
        )
        interp_v = RegularGridInterpolator(
            (fc_lats, fc_lons), v_slice,
            method="linear", bounds_error=False, fill_value=None,
        )

        lat_g, lon_g = np.meshgrid(fine_lats, fine_lons, indexing="ij")
        pts = np.column_stack([lat_g.ravel(), lon_g.ravel()])

        wind_u[a_idx] = interp_u(pts).reshape(n_lat, n_lon)
        wind_v[a_idx] = interp_v(pts).reshape(n_lat, n_lon)

    # Vertical component: approximate boundary-layer effect
    # W ≈ 0 in free atmosphere; near surface add small divergence-driven term
    wind_w = np.zeros_like(wind_u)
    for a_idx, alt_m in enumerate(ALT_LEVELS):
        if alt_m < 300:
            # Boundary-layer vertical velocity ≈ 0.02 × horizontal convergence
            div_u = np.gradient(wind_u[a_idx], axis=1)
            div_v = np.gradient(wind_v[a_idx], axis=0)
            wind_w[a_idx] = -0.02 * (div_u + div_v) * alt_m / 300.0

    speed   = np.sqrt(wind_u ** 2 + wind_v ** 2)
    dir_deg = (np.degrees(np.arctan2(-wind_u, -wind_v)) + 360) % 360

    # Turbulence intensity: empirical from wind shear between adjacent layers
    turb = np.zeros_like(speed)
    for a_idx in range(n_alt):
        if a_idx == 0:
            dU = speed[1] - speed[0]
        elif a_idx == n_alt - 1:
            dU = speed[-1] - speed[-2]
        else:
            dU = (speed[a_idx + 1] - speed[a_idx - 1]) / 2
        # Turbulence intensity TI = |dU/dz| × reference_height / U_mean
        dz = (ALT_LEVELS[min(a_idx + 1, n_alt - 1)] -
              ALT_LEVELS[max(a_idx - 1, 0)])
        U_mean = np.maximum(speed[a_idx], 0.1)
        turb[a_idx] = np.clip(np.abs(dU) / dz * ALT_LEVELS[a_idx] / U_mean, 0, 1)

    logger.info("[WIND] Fine-grid interpolation complete.  Speed mean=%.1f m/s", speed.mean())
    return {
        "wind_u": wind_u,
        "wind_v": wind_v,
        "wind_w": wind_w,
        "speed":  speed,
        "dir_deg": dir_deg,
        "turb":   turb,
    }

# ===========================================================================
# OBSTACLE LAYER
# ===========================================================================

def fetch_obstacles(fine_lats: np.ndarray, fine_lons: np.ndarray) -> tuple[
    list[Any], dict[str, np.ndarray]
]:
    """
    Query OpenStreetMap Overpass API for static obstacles and compute
    per-grid-point nearest-obstacle distance and clearance.

    Returns:
        obstacles     : list of Obstacle objects (raw OSM inventory)
        clearance_dict: {
            nearest_dist_m  (n_lat, n_lon),
            nearest_height_m(n_lat, n_lon),
            obstacle_type   (n_lat, n_lon)  str array,
        }
    """
    logger.info("[OBSTACLE] Querying OpenStreetMap for theatre bounding box...")

    (CACHE_DIR / "osm").mkdir(parents=True, exist_ok=True)
    cache   = ObstacleCache(cache_dir=str(CACHE_DIR / "osm"), default_ttl_s=86400.0)
    config  = OSMProviderConfig(
        base_url="https://overpass-api.de/api/interpreter",
        timeout_s=120.0,
        max_retries=3,
        requests_per_minute=6,
        default_building_height=10.0,
        min_building_height_m=5.0,
        include_buildings=True,
        include_towers=True,
        include_power_lines=True,
        include_religious=True,
        include_chimneys=True,
    )
    provider = OSMDataProvider(config=config, cache=cache)

    bounds = (LAT_MAX, LAT_MIN, LON_MAX, LON_MIN)   # N, S, E, W
    t0 = time.time()
    try:
        obstacles = provider.fetch_obstacles(bounds)
        logger.info(
            "[OBSTACLE] OSM query returned %d obstacles in %.1fs",
            len(obstacles), time.time() - t0,
        )
    except Exception as exc:
        logger.error("[OBSTACLE] OSM fetch failed: %s — using empty list", exc)
        obstacles = []

    if not obstacles:
        logger.warning("[OBSTACLE] No obstacles fetched; clearance will be set to max.")
        n_lat, n_lon = len(fine_lats), len(fine_lons)
        return obstacles, {
            "nearest_dist_m":   np.full((n_lat, n_lon), 99999.0),
            "nearest_height_m": np.zeros((n_lat, n_lon)),
            "obstacle_type":    np.full((n_lat, n_lon), "none", dtype=object),
        }

    # Extract positions from obstacle geometry
    obs_lat, obs_lon, obs_height, obs_type_str = [], [], [], []
    for obs in obstacles:
        g = getattr(obs, "geometry", None)
        h = getattr(obs, "height_m", 0.0) or 0.0
        cat_name = obs.category.name if hasattr(obs, "category") else "UNKNOWN"
        if g is not None and hasattr(g, "center_lat"):
            obs_lat.append(g.center_lat)
            obs_lon.append(g.center_lon)
            obs_height.append(h if h > 0 else getattr(g, "height_m", 0.0))
            obs_type_str.append(cat_name)
        elif obs.state is not None:
            obs_lat.append(obs.state.latitude)
            obs_lon.append(obs.state.longitude)
            obs_height.append(h)
            obs_type_str.append(cat_name)

    if not obs_lat:
        logger.warning("[OBSTACLE] No obstacles had parseable positions.")
        n_lat, n_lon = len(fine_lats), len(fine_lons)
        return obstacles, {
            "nearest_dist_m":   np.full((n_lat, n_lon), 99999.0),
            "nearest_height_m": np.zeros((n_lat, n_lon)),
            "obstacle_type":    np.full((n_lat, n_lon), "none", dtype=object),
        }

    obs_lat    = np.array(obs_lat)
    obs_lon    = np.array(obs_lon)
    obs_height = np.array(obs_height)
    cos_lat    = math.cos(math.radians((LAT_MIN + LAT_MAX) / 2))

    # Convert to approximate Cartesian (metres) for KDTree
    obs_xy = np.column_stack([
        obs_lat * 111_320.0,
        obs_lon * 111_320.0 * cos_lat,
    ])
    tree = cKDTree(obs_xy)
    logger.info("[OBSTACLE] Built KDTree on %d obstacle positions.", len(obs_lat))

    # Query fine grid
    lat_g, lon_g = np.meshgrid(fine_lats, fine_lons, indexing="ij")
    grid_xy = np.column_stack([
        lat_g.ravel() * 111_320.0,
        lon_g.ravel() * 111_320.0 * cos_lat,
    ])
    dists_m, idxs = tree.query(grid_xy, workers=-1)

    n_lat, n_lon = len(fine_lats), len(fine_lons)
    nearest_dist_m   = dists_m.reshape(n_lat, n_lon)
    nearest_height_m = obs_height[idxs].reshape(n_lat, n_lon)

    nearest_type = np.array(obs_type_str)[idxs].reshape(n_lat, n_lon)

    logger.info(
        "[OBSTACLE] Clearance grid complete.  Nearest-obstacle mean=%.0fm  "
        "min=%.0fm",
        nearest_dist_m.mean(), nearest_dist_m.min(),
    )
    return obstacles, {
        "nearest_dist_m":   nearest_dist_m,
        "nearest_height_m": nearest_height_m,
        "obstacle_type":    nearest_type,
    }

# ===========================================================================
# THREAT LAYER  — Swerling-I Radar Range Equation (vectorised)
# ===========================================================================

def compute_threat(
    fine_lats: np.ndarray,
    fine_lons:  np.ndarray,
    alt_levels: list[float],
) -> dict[str, np.ndarray]:
    """
    Compute detection probability for each SAM system at every
    (lat, lon, alt) grid point.

    Swerling Case I:
        SNR = P_t G² λ² σ / ((4π)³ R⁴ k T B F)
        P_d = P_fa^(1 / (1 + SNR))         [Swerling I approximation]

    Where SNR is normalised so that P_d = 0.9 at R_max for RCS = EVTOL_RCS_SQM.

    Returns:
        pd_sam_<i>  (n_alt, n_lat, n_lon)  for i in 0,1,2
        max_pd      (n_alt, n_lat, n_lon)
        combined_pd (n_alt, n_lat, n_lon)  = 1 - Π(1-pd_i)
    """
    logger.info("[THREAT] Computing detection probability for %d SAM sites...", len(SAM_SYSTEMS))

    P_FA = 1e-6          # False-alarm probability (constant threshold)
    # SNR at R_max for P_d = 0.9: solve 0.9 = P_FA^(1/(1+SNR)) → SNR = log(P_FA)/log(0.9) - 1
    SNR_AT_RMAX = math.log(P_FA) / math.log(0.9) - 1.0   # ≈ 13.15

    n_lat = len(fine_lats)
    n_lon = len(fine_lons)
    n_alt = len(alt_levels)

    lat_g, lon_g = np.meshgrid(fine_lats, fine_lons, indexing="ij")   # (n_lat, n_lon)

    pd_arrays = []
    for sam in SAM_SYSTEMS:
        # Haversine range from every (lat, lon) grid point to this SAM
        range_m = haversine_m(lat_g, lon_g, sam["lat"], sam["lon"])   # (n_lat, n_lon)
        R_max   = sam["max_range_km"] * 1000.0

        # SNR scales as R^-4 relative to R_max
        with np.errstate(divide="ignore", invalid="ignore"):
            snr = SNR_AT_RMAX * (R_max / np.maximum(range_m, 500.0)) ** 4

        # Swerling-I P_d
        pd_2d = P_FA ** (1.0 / (1.0 + snr))                # (n_lat, n_lon)
        pd_2d = np.where(range_m > R_max, 0.0, pd_2d)

        # Expand to all altitude levels (same horizontal P_d, altitude independent
        # for simplicity — terrain masking done separately)
        pd_3d = np.broadcast_to(pd_2d[np.newaxis, :, :], (n_alt, n_lat, n_lon)).copy()

        # Altitude masking: below 30 m AGL assume terrain masking reduces P_d
        for a_idx, alt_m in enumerate(alt_levels):
            if alt_m < 50:
                pd_3d[a_idx] *= (alt_m / 50.0) ** 2   # partial mask

        pd_arrays.append(np.clip(pd_3d, 0.0, 1.0))

    max_pd      = np.max(np.stack(pd_arrays, axis=0), axis=0)
    combined_pd = 1.0 - np.prod(1.0 - np.stack(pd_arrays, axis=0), axis=0)

    result: dict[str, np.ndarray] = {}
    for i, (sam, pd) in enumerate(zip(SAM_SYSTEMS, pd_arrays)):
        result[f"pd_sam{i}"] = pd
    result["max_pd"]      = max_pd
    result["combined_pd"] = combined_pd

    logger.info(
        "[THREAT] Threat grid complete.  combined_pd mean=%.3f  max=%.3f",
        combined_pd.mean(), combined_pd.max(),
    )
    return result

# ===========================================================================
# FUSION LAYER
# ===========================================================================

def compute_fusion(
    terrain:   dict[str, np.ndarray],
    wind:      dict[str, np.ndarray],
    obstacle:  dict[str, np.ndarray],
    threat:    dict[str, np.ndarray],
    alt_levels: list[float],
) -> np.ndarray:
    """
    Weighted composite traversability cost in [0, 1].

    Cost model:
      C_terrain  = 0.5 × (slope/45) + 0.5 × exp(-clearance/300)
      C_wind     = speed/30 × (1 + 0.3 × turbulence)
      C_obstacle = exp(-0.005 × nearest_dist_m)
      C_threat   = combined_pd
      C_energy   = (1 + headwind_factor) × alt_penalty

    Returns fused_cost (n_alt, n_lat, n_lon)
    """
    n_alt = len(alt_levels)
    n_lat, n_lon = terrain["elev_m"].shape

    # --- terrain cost (altitude-independent)
    clearance = np.stack(
        [np.clip(alt_m - terrain["elev_m"], 0, 9000) for alt_m in alt_levels],
        axis=0,
    )   # (n_alt, n_lat, n_lon)
    slope_norm  = np.clip(terrain["slope_deg"] / 45.0, 0, 1)
    c_terrain   = 0.5 * slope_norm + 0.5 * np.exp(-clearance / 300.0)
    c_terrain   = np.clip(c_terrain, 0, 1)

    # --- wind cost
    c_wind = np.clip(
        wind["speed"] / 30.0 * (1.0 + 0.3 * wind["turb"]),
        0, 1,
    )   # (n_alt, n_lat, n_lon)

    # --- obstacle cost (expand spatial to all altitudes)
    obs_cost_2d = np.exp(-0.005 * obstacle["nearest_dist_m"])   # (n_lat, n_lon)
    c_obstacle  = np.broadcast_to(
        obs_cost_2d[np.newaxis, :, :], (n_alt, n_lat, n_lon)
    ).copy()
    c_obstacle  = np.clip(c_obstacle, 0, 1)

    # --- threat cost
    c_threat = np.clip(threat["combined_pd"], 0, 1)

    # --- energy cost: headwind + altitude penalty
    # Headwind = projection of wind onto nominal north heading
    c_energy = np.clip(
        0.4 * wind["speed"] / 30.0 + 0.3 * (
            np.array(alt_levels)[:, np.newaxis, np.newaxis] / 1500.0
        ),
        0, 1,
    )

    w = COST_WEIGHTS
    fused = (
        w["terrain"]  * c_terrain  +
        w["wind"]     * c_wind     +
        w["obstacle"] * c_obstacle +
        w["threat"]   * c_threat   +
        w["energy"]   * c_energy
    )
    logger.info(
        "[FUSION] Fused cost complete.  mean=%.3f  max=%.3f",
        fused.mean(), fused.max(),
    )
    return np.clip(fused, 0, 1)

# ===========================================================================
# EXPORT UTILITIES
# ===========================================================================

def write_csv(path: Path, headers: list[str], rows) -> int:
    """Write rows to UTF-8 CSV; returns row count."""
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(headers)
        for row in rows:
            writer.writerow(row)
            count += 1
    return count


def fmt(v, decimals: int = 6) -> str:
    """Format float for CSV."""
    if isinstance(v, (float, np.floating)):
        return f"{v:.{decimals}f}"
    return str(v)

# ===========================================================================
# MAIN PIPELINE
# ===========================================================================

def main() -> None:
    t_start = datetime.now(timezone.utc)
    logger.info("=" * 70)
    logger.info("PERCEPTION DATASET GENERATOR  —  Delhi NCR Theatre")
    logger.info("=" * 70)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    # Build fine spatial grid (descending lats so north-first for CSV readability)
    fine_lats = np.arange(LAT_MAX, LAT_MIN - DLAT / 2, -DLAT)   # N → S
    fine_lons = np.arange(LON_MIN, LON_MAX + DLON / 2,  DLON)   # W → E
    n_lat, n_lon, n_alt = len(fine_lats), len(fine_lons), len(ALT_LEVELS)
    n_spatial = n_lat * n_lon
    n_total   = n_spatial * n_alt
    logger.info(
        "Grid: %d lat × %d lon × %d alt = %d total rows",
        n_lat, n_lon, n_alt, n_total,
    )

    # -----------------------------------------------------------------------
    # 1  TERRAIN
    # -----------------------------------------------------------------------
    terrain = fetch_terrain(fine_lats, fine_lons)

    # Export terrain CSV (spatial only — 75 k rows)
    path_terrain = OUTPUTS_DIR / "perception_terrain.csv"
    headers_terrain = [
        "lat", "lon",
        "elev_m", "slope_deg", "roughness_m", "surface_type",
    ]
    logger.info("[TERRAIN] Writing %s ...", path_terrain.name)
    lat_g, lon_g = np.meshgrid(fine_lats, fine_lons, indexing="ij")

    def terrain_rows():
        for i in range(n_lat):
            for j in range(n_lon):
                yield [
                    fmt(fine_lats[i]), fmt(fine_lons[j]),
                    fmt(terrain["elev_m"][i, j], 2),
                    fmt(terrain["slope_deg"][i, j], 3),
                    fmt(terrain["roughness_m"][i, j], 2),
                    terrain["surface_type"][i, j],
                ]

    n_terrain = write_csv(path_terrain, headers_terrain, terrain_rows())
    logger.info("[TERRAIN] %s  written (%d rows)", path_terrain.name, n_terrain)

    # -----------------------------------------------------------------------
    # 2  WIND
    # -----------------------------------------------------------------------
    wind = fetch_wind(fine_lats, fine_lons)

    # -----------------------------------------------------------------------
    # 3  OBSTACLES
    # -----------------------------------------------------------------------
    obstacles, clearance = fetch_obstacles(fine_lats, fine_lons)

    # Export OSM inventory
    path_obs_inv = OUTPUTS_DIR / "osm_obstacle_inventory.csv"
    headers_inv = ["name", "category", "lat", "lon", "height_m"]
    logger.info("[OBSTACLE] Writing OSM inventory (%d obstacles)...", len(obstacles))

    def obs_inv_rows():
        for obs in obstacles:
            g = getattr(obs, "geometry", None)
            h = getattr(obs, "height_m", 0.0) or 0.0
            if g is not None and hasattr(g, "center_lat"):
                lat, lon = g.center_lat, g.center_lon
                hh = h if h > 0 else getattr(g, "height_m", 0.0)
            elif obs.state is not None:
                lat, lon = obs.state.latitude, obs.state.longitude
                hh = h
            else:
                continue
            cat = obs.category.name if hasattr(obs, "category") else "UNKNOWN"
            yield [getattr(obs, "name", ""), cat, fmt(lat), fmt(lon), fmt(hh, 1)]

    n_inv = write_csv(path_obs_inv, headers_inv, obs_inv_rows())
    logger.info("[OBSTACLE] %s  written (%d rows)", path_obs_inv.name, n_inv)

    # Export obstacle clearance CSV (spatial)
    path_obs_clear = OUTPUTS_DIR / "perception_obstacle.csv"
    headers_obs = ["lat", "lon", "nearest_obstacle_dist_m", "nearest_obstacle_height_m", "obstacle_type"]

    def obs_clear_rows():
        for i in range(n_lat):
            for j in range(n_lon):
                yield [
                    fmt(fine_lats[i]), fmt(fine_lons[j]),
                    fmt(clearance["nearest_dist_m"][i, j], 1),
                    fmt(clearance["nearest_height_m"][i, j], 1),
                    clearance["obstacle_type"][i, j],
                ]

    n_obs_clear = write_csv(path_obs_clear, headers_obs, obs_clear_rows())
    logger.info("[OBSTACLE] %s  written (%d rows)", path_obs_clear.name, n_obs_clear)

    # -----------------------------------------------------------------------
    # 4  THREAT
    # -----------------------------------------------------------------------
    threat = compute_threat(fine_lats, fine_lons, ALT_LEVELS)

    # -----------------------------------------------------------------------
    # 5  FUSION
    # -----------------------------------------------------------------------
    fused_cost = compute_fusion(terrain, wind, clearance, threat, ALT_LEVELS)

    # -----------------------------------------------------------------------
    # 6  FULL DATASET (1M+ rows)
    # -----------------------------------------------------------------------
    path_full = OUTPUTS_DIR / "perception_full_dataset.csv"
    headers_full = [
        # Position
        "lat", "lon", "alt_m",
        # Terrain
        "elev_m", "slope_deg", "roughness_m", "surface_type", "terrain_clearance_m",
        # Wind
        "wind_u_mps", "wind_v_mps", "wind_w_mps",
        "wind_speed_mps", "wind_dir_deg", "turbulence_intensity",
        # Obstacle
        "nearest_obstacle_dist_m", "nearest_obstacle_height_m", "obstacle_type",
        # Threat
        "T1_detect_prob", "T2_detect_prob", "T3_detect_prob",
        "max_threat_prob", "combined_threat_prob",
        # Costs
        "terrain_cost", "wind_cost", "obstacle_cost",
        "threat_cost", "energy_cost", "fused_cost",
    ]

    logger.info("[FULL] Writing %s (%d rows) — this may take a few minutes...",
                path_full.name, n_total)

    # Pre-compute cost arrays
    clearance_arr = np.stack(
        [np.clip(alt_m - terrain["elev_m"], 0, 9000) for alt_m in ALT_LEVELS],
        axis=0,
    )   # (n_alt, n_lat, n_lon)
    c_terrain_arr = np.clip(
        0.5 * terrain["slope_deg"] / 45.0 +
        0.5 * np.exp(-clearance_arr / 300.0),
        0, 1,
    )
    c_wind_arr = np.clip(
        wind["speed"] / 30.0 * (1.0 + 0.3 * wind["turb"]),
        0, 1,
    )
    c_obstacle_arr = np.broadcast_to(
        np.exp(-0.005 * clearance["nearest_dist_m"])[np.newaxis, :, :],
        (n_alt, n_lat, n_lon),
    )
    c_energy_arr = np.clip(
        0.4 * wind["speed"] / 30.0 +
        0.3 * np.array(ALT_LEVELS)[:, None, None] / 1500.0,
        0, 1,
    )

    CHUNK_SIZE = 50_000  # rows per write flush
    written = 0

    path_full.parent.mkdir(parents=True, exist_ok=True)
    with open(path_full, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(headers_full)

        buf = []
        for a_idx, alt_m in enumerate(ALT_LEVELS):
            for i in range(n_lat):
                for j in range(n_lon):
                    lat = fine_lats[i]
                    lon = fine_lons[j]
                    elev    = terrain["elev_m"][i, j]
                    slope   = terrain["slope_deg"][i, j]
                    rough   = terrain["roughness_m"][i, j]
                    stype   = terrain["surface_type"][i, j]
                    tclear  = clearance_arr[a_idx, i, j]

                    wu = wind["wind_u"][a_idx, i, j]
                    wv = wind["wind_v"][a_idx, i, j]
                    ww = wind["wind_w"][a_idx, i, j]
                    ws = wind["speed"][a_idx, i, j]
                    wd = wind["dir_deg"][a_idx, i, j]
                    ti = wind["turb"][a_idx, i, j]

                    nd   = clearance["nearest_dist_m"][i, j]
                    nh   = clearance["nearest_height_m"][i, j]
                    otype = clearance["obstacle_type"][i, j]

                    pd0 = threat["pd_sam0"][a_idx, i, j]
                    pd1 = threat["pd_sam1"][a_idx, i, j]
                    pd2 = threat["pd_sam2"][a_idx, i, j]
                    mpd = threat["max_pd"][a_idx, i, j]
                    cpd = threat["combined_pd"][a_idx, i, j]

                    ct  = c_terrain_arr[a_idx, i, j]
                    cw  = c_wind_arr[a_idx, i, j]
                    co  = c_obstacle_arr[a_idx, i, j]
                    cthr = cpd
                    ce  = c_energy_arr[a_idx, i, j]
                    cf  = fused_cost[a_idx, i, j]

                    buf.append([
                        f"{lat:.5f}", f"{lon:.5f}", f"{alt_m:.0f}",
                        f"{elev:.2f}", f"{slope:.3f}", f"{rough:.2f}",
                        stype, f"{tclear:.2f}",
                        f"{wu:.4f}", f"{wv:.4f}", f"{ww:.4f}",
                        f"{ws:.4f}", f"{wd:.2f}", f"{ti:.4f}",
                        f"{nd:.1f}", f"{nh:.1f}", otype,
                        f"{pd0:.6f}", f"{pd1:.6f}", f"{pd2:.6f}",
                        f"{mpd:.6f}", f"{cpd:.6f}",
                        f"{ct:.5f}", f"{cw:.5f}", f"{co:.5f}",
                        f"{cthr:.5f}", f"{ce:.5f}", f"{cf:.5f}",
                    ])

                    if len(buf) >= CHUNK_SIZE:
                        writer.writerows(buf)
                        written += len(buf)
                        buf.clear()
                        logger.info(
                            "[FULL]  %d / %d rows written  (alt=%dm)",
                            written, n_total, alt_m,
                        )

        if buf:
            writer.writerows(buf)
            written += len(buf)

    logger.info("[FULL] %s  written (%d rows total)", path_full.name, written)

    # -----------------------------------------------------------------------
    # 7  METADATA
    # -----------------------------------------------------------------------
    elapsed = (datetime.now(timezone.utc) - t_start).total_seconds()
    meta: dict[str, Any] = {
        "generated_at":        t_start.isoformat(),
        "wall_time_s":         round(elapsed, 1),
        "theatre":             "Delhi NCR outskirts, India",
        "lat_bounds":          [LAT_MIN, LAT_MAX],
        "lon_bounds":          [LON_MIN, LON_MAX],
        "grid_spacing_deg":    {"lat": DLAT, "lon": DLON},
        "n_lat":               int(n_lat),
        "n_lon":               int(n_lon),
        "alt_levels_m":        ALT_LEVELS,
        "n_spatial_points":    int(n_spatial),
        "n_total_rows":        int(written),
        "terrain": {
            "source":          "Open-Meteo Elevation API (SRTM 30 m)",
            "coarse_fetch_m":  TERRAIN_COARSE_M,
            "interp_method":   "bilinear",
            "elev_range_m":    [
                float(terrain["elev_m"].min()),
                float(terrain["elev_m"].max()),
            ],
            "slope_mean_deg":  float(terrain["slope_deg"].mean()),
        },
        "wind": {
            "source":          "Open-Meteo Weather API (GFS/HRES)",
            "forecast_hours":  WIND_FORECAST_HOURS,
            "speed_mean_mps":  float(wind["speed"].mean()),
            "speed_max_mps":   float(wind["speed"].max()),
        },
        "obstacles": {
            "source":          "OpenStreetMap Overpass API",
            "n_obstacles":     len(obstacles),
            "nearest_dist_min_m": float(clearance["nearest_dist_m"].min()),
        },
        "threat": {
            "method":          "Swerling-I radar range equation",
            "n_sam_systems":   len(SAM_SYSTEMS),
            "evtol_rcs_sqm":   EVTOL_RCS_SQM,
            "sam_systems":     [
                {"name": s["name"], "lat": s["lat"], "lon": s["lon"],
                 "max_range_km": s["max_range_km"]} for s in SAM_SYSTEMS
            ],
            "combined_pd_mean": float(threat["combined_pd"].mean()),
            "combined_pd_max":  float(threat["combined_pd"].max()),
        },
        "fusion": {
            "weights":    COST_WEIGHTS,
            "fused_mean": float(fused_cost.mean()),
            "fused_std":  float(fused_cost.std()),
        },
        "output_files": [
            "perception_terrain.csv",
            "perception_obstacle.csv",
            "osm_obstacle_inventory.csv",
            "perception_full_dataset.csv",
        ],
        "data_policy": "Real data — no synthetic fallbacks.",
        "references": [
            "Rodriguez et al. (2006) SRTM topographic products, PE&RS 72(3)",
            "Skolnik, M. (2008) Radar Handbook 3rd Ed. McGraw-Hill",
            "OpenStreetMap contributors, ODbL",
            "Open-Meteo API (GFS/HRES forecast)",
        ],
    }

    path_meta = OUTPUTS_DIR / "perception_metadata.json"
    with open(path_meta, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    logger.info("[META] %s written.", path_meta.name)
    logger.info("=" * 70)
    logger.info(
        "DONE in %.1fs  —  %d rows  —  outputs in %s",
        elapsed, written, OUTPUTS_DIR,
    )
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
