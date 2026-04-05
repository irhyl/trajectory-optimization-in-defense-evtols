#!/usr/bin/env python3
"""
Exhaustive Visual Analysis Suite — Defense eVTOL Perception Dataset.

Generates publication-quality figures (300 DPI PNG + SVG) for all five
perception layers: terrain, wind, obstacles, threat, fusion.

Categories:
  A.  Spatial maps      — heatmaps, quiver plots, contours
  B.  Statistical       — distributions, CDFs, Q-Q plots, box plots
  C.  Multi-altitude    — altitude profiles, wind shear
  D.  Correlation       — cross-layer correlation matrix, scatter plots
  E.  Operational       — safe-corridor analysis, cost breakdown, Pareto
  F.  Obstacle          — density maps, height CDFs, type breakdown
  G.  Threat            — P_d contours, SAM coverage radii, range plots
  H.  Fusion            — cost component contribution, risk surface

All figures saved to:  visuals/perception/{category}_{name}.{png,svg,pdf}

Usage:
    python scripts/generate_perception_visuals.py

Author: Defense eVTOL Research Team
"""

from __future__ import annotations

import csv
import io
import json
import logging
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib.ticker import MultipleLocator
from scipy.stats import gaussian_kde, pearsonr
from scipy.ndimage import gaussian_filter

# Bootstrap
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("perception_visuals")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR    = REPO_ROOT / "outputs" / "perception_dataset"
VISUALS_DIR = REPO_ROOT / "visuals" / "perception"
VISUALS_DIR.mkdir(parents=True, exist_ok=True)

META_PATH   = DATA_DIR / "perception_metadata.json"
FULL_CSV    = DATA_DIR / "perception_full_dataset.csv"
TERRAIN_CSV = DATA_DIR / "perception_terrain.csv"
OBSTACLE_CSV= DATA_DIR / "perception_obstacle.csv"
OSM_INV_CSV = DATA_DIR / "osm_obstacle_inventory.csv"

# Theatre bounds
LAT_MIN, LAT_MAX = 28.40, 28.90
LON_MIN, LON_MAX = 76.90, 77.50
SAM_SYSTEMS = [
    {"name": "S-300V",  "lat": 28.60, "lon": 77.10, "range_km": 150.0, "color": "red"},
    {"name": "SA-11",   "lat": 28.70, "lon": 77.25, "range_km": 100.0, "color": "darkorange"},
    {"name": "SA-22",   "lat": 28.75, "lon": 77.35, "range_km": 120.0, "color": "crimson"},
]
ALT_LEVELS = [50, 100, 150, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1200, 1500]

# Figure style
plt.rcParams.update({
    "font.family":      "DejaVu Serif",
    "font.size":        10,
    "axes.titlesize":   12,
    "axes.labelsize":   10,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "legend.fontsize":  9,
    "figure.dpi":       150,
    "savefig.dpi":      300,
    "figure.facecolor": "white",
    "axes.grid":        True,
    "grid.alpha":       0.3,
})

# Custom colourmaps
THREAT_CMAP = LinearSegmentedColormap.from_list(
    "threat", ["#1a9641", "#fdae61", "#d73027"], N=256
)
TERRAIN_CMAP = "terrain"
WIND_CMAP    = "viridis"
OBSTACLE_CMAP = LinearSegmentedColormap.from_list(
    "obstacle", ["#ffffcc", "#fd8d3c", "#800026"], N=256
)

# ---------------------------------------------------------------------------
# Data loader
# ---------------------------------------------------------------------------

def load_dataset() -> dict:
    """Load full CSV into numpy arrays, returning dict of 3D arrays [alt, lat, lon]."""
    logger.info("Loading dataset from %s ...", FULL_CSV.name)
    rows = []
    with open(FULL_CSV, encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append(row)
    logger.info("  %d rows loaded.", len(rows))

    lats_u  = sorted({float(r["lat"])  for r in rows}, reverse=True)
    lons_u  = sorted({float(r["lon"])  for r in rows})
    alts_u  = sorted({int(r["alt_m"])  for r in rows})
    nl, nlo, na = len(lats_u), len(lons_u), len(alts_u)

    lat_idx  = {v: i for i, v in enumerate(lats_u)}
    lon_idx  = {v: i for i, v in enumerate(lons_u)}
    alt_idx  = {v: i for i, v in enumerate(alts_u)}

    fields = [
        "elev_m", "slope_deg", "roughness_m", "terrain_clearance_m",
        "wind_u_mps", "wind_v_mps", "wind_w_mps", "wind_speed_mps",
        "wind_dir_deg", "turbulence_intensity",
        "nearest_obstacle_dist_m", "nearest_obstacle_height_m",
        "T1_detect_prob", "T2_detect_prob", "T3_detect_prob",
        "max_threat_prob", "combined_threat_prob",
        "terrain_cost", "wind_cost", "obstacle_cost",
        "threat_cost", "energy_cost", "fused_cost",
    ]
    arrays = {f: np.full((na, nl, nlo), np.nan) for f in fields}

    for row in rows:
        la  = float(row["lat"])
        lo  = float(row["lon"])
        alt = int(row["alt_m"])
        ai  = alt_idx[alt]
        li  = lat_idx[la]
        loi = lon_idx[lo]
        for f in fields:
            arrays[f][ai, li, loi] = float(row[f])

    lats = np.array(lats_u)
    lons = np.array(lons_u)

    logger.info("  Arrays shaped: %s  (n_alt, n_lat, n_lon)", arrays["elev_m"].shape)
    return {
        "lats": lats, "lons": lons, "alts": np.array(alts_u),
        **arrays,
    }


def load_osm() -> dict:
    rows = []
    with open(OSM_INV_CSV, encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append(row)
    return rows


def save_fig(fig: plt.Figure, name: str) -> None:
    stem = VISUALS_DIR / name
    fig.savefig(str(stem) + ".png", bbox_inches="tight")
    fig.savefig(str(stem) + ".svg", bbox_inches="tight")
    fig.savefig(str(stem) + ".pdf", bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved: %s.{png,svg,pdf}", name)


# ===========================================================================
# A. SPATIAL MAPS
# ===========================================================================

def plot_A1_terrain_elevation(d: dict) -> None:
    """Terrain elevation heatmap with contours."""
    logger.info("[A1] Terrain elevation map")
    elev = d["elev_m"][0]   # altitude-independent
    lats, lons = d["lats"], d["lons"]

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.pcolormesh(lons, lats, elev, cmap="terrain", shading="auto")
    cs = ax.contour(lons, lats, elev, levels=10, colors="white", linewidths=0.5, alpha=0.7)
    ax.clabel(cs, fmt="%d m", fontsize=7, inline=True)
    cbar = fig.colorbar(im, ax=ax, label="Elevation (m MSL)")
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.set_title("Figure A1 — SRTM Terrain Elevation: Delhi NCR Theatre\n"
                 "(Open-Meteo / NASA SRTM 30 m, bilinear interpolated to 222 m)")
    # Mission origin/goal
    ax.plot(77.0, 28.5, "g^", ms=10, label="Origin (28.5°N, 77.0°E)")
    ax.plot(77.4, 28.85,"rs", ms=10, label="Goal   (28.85°N, 77.4°E)")
    for sam in SAM_SYSTEMS:
        ax.plot(sam["lon"], sam["lat"], "kX", ms=8)
        ax.annotate(sam["name"], (sam["lon"], sam["lat"]),
                    textcoords="offset points", xytext=(5, 5), fontsize=7,
                    color="black", fontweight="bold")
    ax.legend(loc="upper left", fontsize=8)
    save_fig(fig, "A1_terrain_elevation")


def plot_A2_slope_roughness(d: dict) -> None:
    """Slope and roughness side-by-side."""
    logger.info("[A2] Slope + roughness map")
    slope = d["slope_deg"][0]
    rough = d["roughness_m"][0]
    lats, lons = d["lats"], d["lons"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    im0 = axes[0].pcolormesh(lons, lats, slope, cmap="YlOrRd", shading="auto",
                              vmin=0, vmax=slope.max())
    fig.colorbar(im0, ax=axes[0], label="Slope (°)")
    axes[0].set_title("Terrain Slope (°)")

    im1 = axes[1].pcolormesh(lons, lats, rough, cmap="copper", shading="auto")
    fig.colorbar(im1, ax=axes[1], label="Roughness (m RMS)")
    axes[1].set_title("Terrain Roughness (m RMS, 3×3 kernel)")

    for ax in axes:
        ax.set_xlabel("Longitude (°E)")
        ax.set_ylabel("Latitude (°N)")

    fig.suptitle("Figure A2 — Terrain Derivatives: Slope and Roughness\n"
                 "Delhi NCR Gangetic Plain — characteristically flat (<2.5° slope)", y=1.02)
    save_fig(fig, "A2_slope_roughness")


def plot_A3_wind_quiver(d: dict) -> None:
    """Wind vector field quiver at 200 m and 1000 m."""
    logger.info("[A3] Wind quiver maps")
    alts = list(d["alts"])
    lats, lons = d["lats"], d["lons"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, alt_m in zip(axes, [200, 1000]):
        ai = list(alts).index(alt_m)
        U = d["wind_u_mps"][ai]
        V = d["wind_v_mps"][ai]
        spd = d["wind_speed_mps"][ai]

        # Subsample for quiver
        step = 8
        im = ax.pcolormesh(lons, lats, spd, cmap=WIND_CMAP, shading="auto", alpha=0.85)
        ax.quiver(
            lons[::step], lats[::step],
            U[::step, ::step], V[::step, ::step],
            scale=80, width=0.003, headwidth=3, color="white", alpha=0.8,
        )
        fig.colorbar(im, ax=ax, label="Wind speed (m/s)")
        ax.set_title(f"GFS Wind Field at {alt_m} m MSL")
        ax.set_xlabel("Longitude (°E)")
        ax.set_ylabel("Latitude (°N)")

    fig.suptitle("Figure A3 — Real GFS/HRES Wind Vector Fields (Open-Meteo, t=0)\n"
                 "Colour: speed magnitude  /  Arrows: direction")
    save_fig(fig, "A3_wind_quiver")


def plot_A4_obstacle_density(d: dict, osm_rows: list) -> None:
    """Obstacle nearest-distance + KDE density of obstacle locations."""
    logger.info("[A4] Obstacle density map")
    nd = d["nearest_obstacle_dist_m"][0]
    lats, lons = d["lats"], d["lons"]

    # Extract obstacle positions
    olats = np.array([float(r["lat"]) for r in osm_rows if r["lat"] and r["lon"]])
    olons = np.array([float(r["lon"]) for r in osm_rows if r["lat"] and r["lon"]])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    im0 = axes[0].pcolormesh(lons, lats, np.log1p(nd), cmap="RdYlGn_r", shading="auto")
    fig.colorbar(im0, ax=axes[0], label="log(1 + nearest dist) (m)")
    axes[0].set_title("Nearest Obstacle Distance (log scale)")

    axes[1].scatter(olons, olats, s=0.3, c="darkred", alpha=0.3, rasterized=True)
    axes[1].set_xlim(LON_MIN, LON_MAX)
    axes[1].set_ylim(LAT_MIN, LAT_MAX)
    axes[1].set_title(f"OSM Obstacle Positions (N={len(olats):,})")

    for ax in axes:
        ax.set_xlabel("Longitude (°E)")
        ax.set_ylabel("Latitude (°N)")

    fig.suptitle("Figure A4 — Obstacle Spatial Distribution\n"
                 "Left: clearance field  /  Right: raw OSM obstacle positions")
    save_fig(fig, "A4_obstacle_density")


def plot_A5_threat_pd_contours(d: dict) -> None:
    """SAM detection probability contour map at 200 m."""
    logger.info("[A5] Threat P_d contours")
    alts = list(d["alts"])
    ai = list(alts).index(200)
    lats, lons = d["lats"], d["lons"]

    # Since combined_pd ≈ 1 everywhere, show individual SAM Pd (which shows gradient)
    T1 = d["T1_detect_prob"][ai]
    T2 = d["T2_detect_prob"][ai]
    T3 = d["T3_detect_prob"][ai]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, Tmap, sam in zip(axes, [T1, T2, T3], SAM_SYSTEMS):
        im = ax.pcolormesh(lons, lats, Tmap, cmap=THREAT_CMAP,
                           shading="auto", vmin=0.998, vmax=1.0)
        cs = ax.contour(lons, lats, Tmap,
                        levels=[0.9990, 0.9993, 0.9995, 0.9998, 0.9999],
                        colors="white", linewidths=0.8)
        ax.clabel(cs, fmt="%.4f", fontsize=7)
        fig.colorbar(im, ax=ax, label="P_detection")
        ax.plot(sam["lon"], sam["lat"], "kX", ms=12)
        ax.set_title(f"{sam['name']} (R_max={sam['range_km']:.0f} km)")
        ax.set_xlabel("Longitude (°E)")
        ax.set_ylabel("Latitude (°N)")

    fig.suptitle("Figure A5 — Individual SAM Detection Probability P_d at 200 m MSL\n"
                 "Swerling Case I: P_d = P_fa^(1/(1+SNR)),  SNR ∝ R^{-4}  [Skolnik 2008]",
                 y=1.02)
    save_fig(fig, "A5_threat_pd_contours")


def plot_A6_fused_cost_map(d: dict) -> None:
    """Fused traversability cost maps at 3 altitudes."""
    logger.info("[A6] Fused cost maps")
    alts = list(d["alts"])
    lats, lons = d["lats"], d["lons"]
    plot_alts = [50, 200, 500]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, alt_m in zip(axes, plot_alts):
        ai = list(alts).index(alt_m)
        C = d["fused_cost"][ai]
        im = ax.pcolormesh(lons, lats, C, cmap="RdYlGn_r",
                           shading="auto", vmin=C.min(), vmax=C.max())
        fig.colorbar(im, ax=ax, label="Fused cost [0–1]")
        ax.set_title(f"Altitude = {alt_m} m")
        ax.set_xlabel("Longitude (°E)")
        ax.set_ylabel("Latitude (°N)")
        # Plot SAM sites
        for sam in SAM_SYSTEMS:
            ax.plot(sam["lon"], sam["lat"], "kX", ms=8)
        ax.plot(77.0, 28.5, "g^", ms=8)
        ax.plot(77.4, 28.85, "rs", ms=8)

    fig.suptitle("Figure A6 — Fused Traversability Cost C_total(x, alt)\n"
                 "C = 0.15·C_terrain + 0.10·C_wind + 0.20·C_obstacle + 0.40·C_threat + 0.15·C_energy",
                 y=1.02)
    save_fig(fig, "A6_fused_cost_map")


# ===========================================================================
# B. STATISTICAL DISTRIBUTIONS
# ===========================================================================

def plot_B1_elevation_distribution(d: dict) -> None:
    logger.info("[B1] Elevation distribution")
    elev_flat = d["elev_m"][0].ravel()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].hist(elev_flat, bins=80, color="#2171b5", edgecolor="none", density=True)
    axes[0].set_xlabel("Elevation (m MSL)")
    axes[0].set_ylabel("Probability density")
    axes[0].set_title("Elevation PDF")

    # Cumulative
    sorted_e = np.sort(elev_flat)
    cdf = np.arange(1, len(sorted_e)+1) / len(sorted_e)
    axes[1].plot(sorted_e, cdf, color="#2171b5", lw=2)
    axes[1].set_xlabel("Elevation (m MSL)")
    axes[1].set_ylabel("CDF")
    axes[1].set_title("Elevation CDF")

    axes[2].hist(d["slope_deg"][0].ravel(), bins=80, color="#d95f0e", edgecolor="none",
                 density=True)
    axes[2].set_xlabel("Slope (°)")
    axes[2].set_ylabel("Probability density")
    axes[2].set_title("Slope PDF (Delhi NCR: Gangetic plain)")

    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.suptitle("Figure B1 — Terrain Statistics: Elevation & Slope Distributions")
    save_fig(fig, "B1_terrain_distributions")


def plot_B2_wind_distributions(d: dict) -> None:
    logger.info("[B2] Wind statistical distributions")
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # Wind speed at all altitudes combined
    spd_all = d["wind_speed_mps"].ravel()
    axes[0, 0].hist(spd_all, bins=80, color="#238b45", edgecolor="none", density=True)
    axes[0, 0].set_xlabel("Wind speed (m/s)")
    axes[0, 0].set_title("Wind Speed PDF (all altitudes)")

    # Wind direction (polar histogram / rose equivalent using 2D bar)
    dir_all = d["wind_dir_deg"].ravel()
    bins_dir = np.linspace(0, 360, 37)
    hist_dir, _ = np.histogram(dir_all, bins=bins_dir)
    bin_centers = (bins_dir[:-1] + bins_dir[1:]) / 2
    axes[0, 1].bar(bin_centers, hist_dir, width=10, color="#2ca25f", edgecolor="none")
    axes[0, 1].set_xlabel("Wind direction (FROM, °)")
    axes[0, 1].set_title("Wind Direction Histogram")

    # Wind speed vs altitude box plots
    alts = d["alts"]
    bplot_data = [d["wind_speed_mps"][ai].ravel() for ai in range(len(alts))]
    axes[0, 2].boxplot(bplot_data, labels=[str(a) for a in alts],
                       notch=False, medianprops={"color": "red"})
    axes[0, 2].set_xlabel("Altitude (m)")
    axes[0, 2].set_ylabel("Wind speed (m/s)")
    axes[0, 2].set_title("Wind Speed vs Altitude (box plots)")
    axes[0, 2].tick_params(axis="x", rotation=45)

    # Turbulence
    turb_all = d["turbulence_intensity"].ravel()
    axes[1, 0].hist(turb_all, bins=60, color="#fd8d3c", edgecolor="none", density=True)
    axes[1, 0].set_xlabel("Turbulence intensity (dimensionless)")
    axes[1, 0].set_title("Turbulence Intensity PDF")

    # Wind U vs V scatter
    ai_200 = list(alts).index(200)
    U = d["wind_u_mps"][ai_200].ravel()
    V = d["wind_v_mps"][ai_200].ravel()
    axes[1, 1].scatter(U, V, s=0.5, alpha=0.2, c="#756bb1", rasterized=True)
    axes[1, 1].set_xlabel("U component (m/s)  [eastward]")
    axes[1, 1].set_ylabel("V component (m/s)  [northward]")
    axes[1, 1].set_title("U–V Wind Components at 200 m (Hodograph)")
    axes[1, 1].axhline(0, color="k", lw=0.5)
    axes[1, 1].axvline(0, color="k", lw=0.5)

    # Altitude vs mean wind
    mean_spd = [d["wind_speed_mps"][ai].mean() for ai in range(len(alts))]
    axes[1, 2].plot(mean_spd, alts, "o-", color="#d7301f", lw=2, ms=5)
    axes[1, 2].set_xlabel("Mean wind speed (m/s)")
    axes[1, 2].set_ylabel("Altitude (m MSL)")
    axes[1, 2].set_title("Wind Speed Profile (mean over theatre)")

    fig.suptitle("Figure B2 — Wind Statistical Analysis (GFS/HRES Forecast)", y=1.02)
    plt.tight_layout()
    save_fig(fig, "B2_wind_distributions")


def plot_B3_obstacle_distributions(d: dict, osm_rows: list) -> None:
    logger.info("[B3] Obstacle distributions")
    import collections

    heights = [float(r["height_m"]) for r in osm_rows if r["height_m"]]
    categories = collections.Counter(r["category"] for r in osm_rows)
    nd_flat = d["nearest_obstacle_dist_m"][0].ravel()

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Height distribution
    h_arr = np.array(heights)
    axes[0].hist(h_arr[h_arr > 0], bins=60, color="#800026", edgecolor="none", density=True)
    axes[0].set_xlabel("Obstacle height (m)")
    axes[0].set_title("Obstacle Height PDF")
    axes[0].set_yscale("log")

    # Category pie
    labels = [f"{k}\n({v})" for k, v in categories.most_common(8)]
    vals   = [v for _, v in categories.most_common(8)]
    axes[1].pie(vals, labels=labels, autopct="%1.1f%%",
                colors=plt.cm.Set2.colors[:len(vals)])
    axes[1].set_title("OSM Obstacle Category Breakdown")

    # Nearest obstacle distance CDF
    sorted_nd = np.sort(nd_flat)
    cdf = np.arange(1, len(sorted_nd)+1) / len(sorted_nd)
    axes[2].plot(sorted_nd, cdf, color="#800026", lw=2)
    axes[2].axvline(100, color="orange", ls="--", label="100 m (danger zone)")
    axes[2].axvline(200, color="green",  ls="--", label="200 m (safe)")
    axes[2].set_xlabel("Nearest obstacle distance (m)")
    axes[2].set_ylabel("CDF")
    axes[2].set_title("Nearest Obstacle Distance CDF")
    axes[2].legend(fontsize=8)

    fig.suptitle("Figure B3 — Obstacle Statistical Analysis (OpenStreetMap)")
    save_fig(fig, "B3_obstacle_distributions")


def plot_B4_cost_distributions(d: dict) -> None:
    logger.info("[B4] Cost component distributions")
    ai_200 = list(d["alts"]).index(200)

    costs = {
        "Terrain": d["terrain_cost"][ai_200].ravel(),
        "Wind":    d["wind_cost"][ai_200].ravel(),
        "Obstacle":d["obstacle_cost"][ai_200].ravel(),
        "Threat":  d["threat_cost"][ai_200].ravel(),
        "Energy":  d["energy_cost"][ai_200].ravel(),
        "Fused":   d["fused_cost"][ai_200].ravel(),
    }
    colors = ["#2b8cbe", "#78c679", "#fc8d59", "#d7191c", "#a6761d", "#6a3d9a"]

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, (name, vals), col in zip(axes.flat, costs.items(), colors):
        ax.hist(vals, bins=60, color=col, edgecolor="none", density=True)
        ax.axvline(vals.mean(), color="black", ls="--", lw=1.5, label=f"mean={vals.mean():.3f}")
        ax.set_xlabel("Cost value [0–1]")
        ax.set_ylabel("Density")
        ax.set_title(f"{name} cost (alt=200 m)")
        ax.legend(fontsize=8)

    fig.suptitle("Figure B4 — Cost Component Distributions at 200 m MSL\n"
                 "C ∈ [0,1]  /  Higher = higher traversal penalty")
    save_fig(fig, "B4_cost_distributions")


# ===========================================================================
# C. MULTI-ALTITUDE PROFILES
# ===========================================================================

def plot_C1_wind_altitude_profile(d: dict) -> None:
    logger.info("[C1] Wind altitude profiles")
    alts = d["alts"]
    # Along-track profile: fixed lon=77.2 (approx middle longitude)
    lons = d["lons"]
    lon_idx = int(len(lons) * 0.5)

    U_profile = d["wind_u_mps"][:, :, lon_idx]  # [alt, lat]
    V_profile = d["wind_v_mps"][:, :, lon_idx]
    spd_profile = d["wind_speed_mps"][:, :, lon_idx]
    turb_profile = d["turbulence_intensity"][:, :, lon_idx]

    lats = d["lats"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    im0 = axes[0].pcolormesh(lats, alts, spd_profile, cmap=WIND_CMAP, shading="auto")
    fig.colorbar(im0, ax=axes[0], label="Wind speed (m/s)")
    axes[0].set_xlabel("Latitude (°N)")
    axes[0].set_ylabel("Altitude (m MSL)")
    axes[0].set_title("Wind Speed Cross-Section (lon ≈ 77.2°E)")

    im1 = axes[1].pcolormesh(lats, alts, turb_profile, cmap="hot_r", shading="auto",
                              vmin=0, vmax=1)
    fig.colorbar(im1, ax=axes[1], label="Turbulence intensity")
    axes[1].set_xlabel("Latitude (°N)")
    axes[1].set_ylabel("Altitude (m MSL)")
    axes[1].set_title("Turbulence Cross-Section (lon ≈ 77.2°E)")

    fig.suptitle("Figure C1 — Altitude–Latitude Cross-Sections\n"
                 "ISA pressure-altitude mapping: h = 44330·(1−(P/P₀)^0.1903)")
    save_fig(fig, "C1_wind_altitude_profile")


def plot_C2_terrain_clearance_profile(d: dict) -> None:
    logger.info("[C2] Terrain clearance vs altitude")
    alts = d["alts"]
    tc = d["terrain_clearance_m"]

    mean_tc = [tc[ai].mean() for ai in range(len(alts))]
    min_tc  = [tc[ai].min()  for ai in range(len(alts))]
    p5_tc   = [np.percentile(tc[ai].ravel(), 5) for ai in range(len(alts))]
    p95_tc  = [np.percentile(tc[ai].ravel(), 95) for ai in range(len(alts))]

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot(mean_tc, alts, "b-o", lw=2, ms=6, label="Mean clearance")
    ax.plot(min_tc,  alts, "r-s", lw=1.5, ms=5, label="Min clearance")
    ax.fill_betweenx(alts, p5_tc, p95_tc, alpha=0.2, color="blue", label="5th–95th percentile")
    ax.axvline(50, color="orange", ls="--", lw=1.5, label="50 m (min safe AGL)")
    ax.set_xlabel("Terrain clearance (m)")
    ax.set_ylabel("Altitude MSL (m)")
    ax.set_title("Figure C2 — Terrain Clearance vs Altitude\n"
                 "clearance = alt_MSL − elevation_SRTM")
    ax.legend()
    save_fig(fig, "C2_terrain_clearance_profile")


def plot_C3_fused_cost_altitude(d: dict) -> None:
    logger.info("[C3] Fused cost vs altitude")
    alts = d["alts"]

    mean_c = [d["fused_cost"][ai].mean()          for ai in range(len(alts))]
    tc_mean= [d["terrain_cost"][ai].mean()        for ai in range(len(alts))]
    wc_mean= [d["wind_cost"][ai].mean()           for ai in range(len(alts))]
    oc_mean= [d["obstacle_cost"][ai].mean()       for ai in range(len(alts))]
    thr_mean=[d["threat_cost"][ai].mean()         for ai in range(len(alts))]
    ec_mean= [d["energy_cost"][ai].mean()         for ai in range(len(alts))]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(mean_c,   alts, "k-o",  lw=3, ms=7, label="Fused (total)")
    ax.plot(tc_mean,  alts, "b--^", lw=1.5, ms=5, label="Terrain (×0.15)")
    ax.plot(wc_mean,  alts, "g--s", lw=1.5, ms=5, label="Wind (×0.10)")
    ax.plot(oc_mean,  alts, "m--D", lw=1.5, ms=5, label="Obstacle (×0.20)")
    ax.plot(thr_mean, alts, "r--v", lw=1.5, ms=5, label="Threat (×0.40)")
    ax.plot(ec_mean,  alts, "y--p", lw=1.5, ms=5, label="Energy (×0.15)")
    ax.set_xlabel("Mean cost value [0–1]")
    ax.set_ylabel("Altitude (m MSL)")
    ax.set_title("Figure C3 — Cost Component Profiles vs Altitude\n"
                 "Mean over entire theatre spatial domain")
    ax.legend(loc="right")
    save_fig(fig, "C3_fused_cost_altitude")


# ===========================================================================
# D. CORRELATION ANALYSIS
# ===========================================================================

def plot_D1_correlation_matrix(d: dict) -> None:
    logger.info("[D1] Cross-layer correlation matrix")
    ai = list(d["alts"]).index(200)

    field_names = [
        "elev_m", "slope_deg", "roughness_m",
        "wind_speed_mps", "turbulence_intensity",
        "nearest_obstacle_dist_m",
        "T1_detect_prob", "combined_threat_prob",
        "terrain_cost", "wind_cost", "obstacle_cost",
        "threat_cost", "energy_cost", "fused_cost",
    ]
    label_map = {
        "elev_m": "Elevation",
        "slope_deg": "Slope",
        "roughness_m": "Roughness",
        "wind_speed_mps": "Wind speed",
        "turbulence_intensity": "Turbulence",
        "nearest_obstacle_dist_m": "Obs. dist.",
        "T1_detect_prob": "T1 P_d",
        "combined_threat_prob": "Combined P_d",
        "terrain_cost": "C_terrain",
        "wind_cost": "C_wind",
        "obstacle_cost": "C_obstacle",
        "threat_cost": "C_threat",
        "energy_cost": "C_energy",
        "fused_cost": "C_fused",
    }
    X = np.column_stack([d[f][ai].ravel() for f in field_names])
    # Subsample for speed
    rng = np.random.default_rng(42)
    idx = rng.choice(X.shape[0], size=min(20000, X.shape[0]), replace=False)
    Xs = X[idx]
    C = np.corrcoef(Xs.T)
    labels = [label_map[f] for f in field_names]

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    fig.colorbar(im, ax=ax, label="Pearson r")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{C[i,j]:.2f}", ha="center", va="center",
                    fontsize=6.5, color="black" if abs(C[i,j]) < 0.7 else "white")
    ax.set_title("Figure D1 — Cross-Layer Pearson Correlation Matrix\n"
                 "(altitude = 200 m, n = 20,000 random samples)", pad=15)
    save_fig(fig, "D1_correlation_matrix")


def plot_D2_scatter_pairs(d: dict) -> None:
    logger.info("[D2] Key scatter pairs")
    ai = list(d["alts"]).index(200)
    rng = np.random.default_rng(42)
    n = 5000
    nd  = d["nearest_obstacle_dist_m"][ai].ravel()
    co  = d["obstacle_cost"][ai].ravel()
    ws  = d["wind_speed_mps"][ai].ravel()
    cw  = d["wind_cost"][ai].ravel()
    elev= d["elev_m"][ai].ravel()
    ct  = d["terrain_cost"][ai].ravel()
    cf  = d["fused_cost"][ai].ravel()

    idx = rng.choice(len(nd), size=n, replace=False)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    pairs = [
        (nd[idx], co[idx], "Nearest obs. dist. (m)", "C_obstacle", "#d73027"),
        (ws[idx], cw[idx], "Wind speed (m/s)", "C_wind", "#1d91c0"),
        (elev[idx], cf[idx], "Elevation (m)", "C_fused", "#41ab5d"),
    ]
    for ax, (x, y, xl, yl, col) in zip(axes, pairs):
        ax.scatter(x, y, s=3, alpha=0.4, c=col, rasterized=True)
        # Trend line
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        xs = np.linspace(x.min(), x.max(), 100)
        ax.plot(xs, p(xs), "k--", lw=1.5)
        r, _ = pearsonr(x, y)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_title(f"r = {r:.3f}")
    fig.suptitle("Figure D2 — Key Cross-Layer Scatter Plots (n=5,000 samples, alt=200 m)")
    save_fig(fig, "D2_scatter_pairs")


# ===========================================================================
# E. OPERATIONAL ANALYSIS
# ===========================================================================

def plot_E1_cost_component_breakdown(d: dict) -> None:
    logger.info("[E1] Cost component breakdown bar chart")
    alts = d["alts"]
    components = {
        "Terrain (w=0.15)": [d["terrain_cost"][ai].mean() * 0.15 for ai in range(len(alts))],
        "Wind (w=0.10)":    [d["wind_cost"][ai].mean()    * 0.10 for ai in range(len(alts))],
        "Obstacle (w=0.20)":[d["obstacle_cost"][ai].mean()* 0.20 for ai in range(len(alts))],
        "Threat (w=0.40)":  [d["threat_cost"][ai].mean()  * 0.40 for ai in range(len(alts))],
        "Energy (w=0.15)":  [d["energy_cost"][ai].mean()  * 0.15 for ai in range(len(alts))],
    }
    colors = ["#2b8cbe", "#78c679", "#fc8d59", "#d7191c", "#a6761d"]

    fig, ax = plt.subplots(figsize=(12, 7))
    bottom = np.zeros(len(alts))
    for (label, vals), col in zip(components.items(), colors):
        vals_arr = np.array(vals)
        ax.barh(range(len(alts)), vals_arr, left=bottom, label=label, color=col, height=0.75)
        bottom += vals_arr

    ax.set_yticks(range(len(alts)))
    ax.set_yticklabels([str(a) for a in alts])
    ax.set_xlabel("Weighted mean cost contribution to C_fused")
    ax.set_ylabel("Altitude (m MSL)")
    ax.set_title("Figure E1 — Cost Component Breakdown by Altitude\n"
                 "C_total = Σᵢ wᵢ · Cᵢ  (weighted contributions shown)")
    ax.legend(loc="lower right")
    save_fig(fig, "E1_cost_component_breakdown")


def plot_E2_safe_corridor_analysis(d: dict) -> None:
    logger.info("[E2] Safe corridor analysis")
    alts = d["alts"]
    lats, lons = d["lats"], d["lons"]

    # "Safe" = fused_cost < 0.50 AND terrain_clearance > 50m
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    alts_list = list(alts)
    for ax, alt_m in zip(axes, [50, 200, 500]):
        ai = alts_list.index(alt_m)
        C = d["fused_cost"][ai]
        TC = d["terrain_clearance_m"][ai]
        safe = (C < 0.50) & (TC > 50.0)
        ax.pcolormesh(lons, lats, C.astype(float), cmap="RdYlGn_r", shading="auto",
                      alpha=0.7, vmin=0.44, vmax=0.74)
        # Overlay safe mask
        ax.pcolormesh(lons, lats,
                      np.where(safe, 1.0, np.nan),
                      cmap=LinearSegmentedColormap.from_list("g", ["#1a9641","#1a9641"]),
                      shading="auto", alpha=0.5)
        pct = safe.mean() * 100
        ax.set_title(f"Alt={alt_m}m  Safe: {pct:.1f}%")
        ax.set_xlabel("Longitude (°E)")
        ax.set_ylabel("Latitude (°N)")
        for sam in SAM_SYSTEMS:
            ax.plot(sam["lon"], sam["lat"], "kX", ms=8)
        ax.plot(77.0, 28.5, "g^", ms=8)
        ax.plot(77.4, 28.85, "rs", ms=8)

    fig.suptitle("Figure E2 — Safe Corridor Analysis (C_fused < 0.50, terrain clearance > 50 m)\n"
                 "Green overlay = safe  /  ×= SAM site  /  triangle/square = origin/goal",
                 y=1.02)
    save_fig(fig, "E2_safe_corridor")


def plot_E3_mission_cross_section(d: dict) -> None:
    logger.info("[E3] Mission trajectory cross-section")
    # Sample a straight-line mission path: (28.5, 77.0) → (28.85, 77.4)
    n_pts = 100
    mission_lats = np.linspace(28.5, 28.85, n_pts)
    mission_lons = np.linspace(77.0, 77.4, n_pts)

    alts = d["alts"]
    lats, lons = d["lats"], d["lons"]

    # Bilinear lookup for each point at each altitude
    def lookup(arr, lat, lon):
        i = np.argmin(np.abs(lats - lat))
        j = np.argmin(np.abs(lons - lon))
        return arr[i, j]

    dist_km = np.linspace(0, 60, n_pts)
    costs_200 = [lookup(d["fused_cost"][list(alts).index(200)], la, lo)
                 for la, lo in zip(mission_lats, mission_lons)]
    threat_200 = [lookup(d["T1_detect_prob"][list(alts).index(200)], la, lo)
                  for la, lo in zip(mission_lats, mission_lons)]
    obs_dist   = [lookup(d["nearest_obstacle_dist_m"][list(alts).index(200)], la, lo)
                  for la, lo in zip(mission_lats, mission_lons)]
    elev_path  = [lookup(d["elev_m"][0], la, lo) for la, lo in zip(mission_lats, mission_lons)]

    fig, axes = plt.subplots(4, 1, figsize=(12, 14), sharex=True)
    axes[0].plot(dist_km, elev_path, "saddlebrown", lw=2)
    axes[0].fill_between(dist_km, 0, elev_path, alpha=0.3, color="saddlebrown")
    axes[0].set_ylabel("Elevation (m MSL)")
    axes[0].set_title("SRTM Terrain Elevation")

    axes[1].plot(dist_km, obs_dist, "darkred", lw=2)
    axes[1].axhline(100, color="orange", ls="--", label="100 m danger")
    axes[1].axhline(200, color="green",  ls="--", label="200 m safe")
    axes[1].set_ylabel("Nearest obstacle (m)")
    axes[1].set_title("Obstacle Clearance")
    axes[1].legend(fontsize=8)

    axes[2].plot(dist_km, threat_200, "crimson", lw=2)
    axes[2].set_ylim(0.998, 1.001)
    axes[2].set_ylabel("P_d (S-300V)")
    axes[2].set_title("SAM Detection Probability")

    axes[3].plot(dist_km, costs_200, "k", lw=2)
    axes[3].fill_between(dist_km, 0, costs_200, alpha=0.2, color="k")
    axes[3].set_ylabel("Fused cost")
    axes[3].set_xlabel("Distance along track (km)")
    axes[3].set_title("Fused Traversability Cost")

    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.suptitle("Figure E3 — Mission Cross-Section: Origin → Goal (straight line)\n"
                 "(28.5°N,77.0°E) → (28.85°N,77.4°E)  at 200 m MSL", y=1.01)
    plt.tight_layout()
    save_fig(fig, "E3_mission_cross_section")


# ===========================================================================
# F. THREAT ANALYSIS
# ===========================================================================

def plot_F1_sam_range_rings(d: dict) -> None:
    logger.info("[F1] SAM range rings and coverage")
    lats, lons = d["lats"], d["lons"]
    ai = list(d["alts"]).index(200)
    comb = d["combined_threat_prob"][ai]

    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.pcolormesh(lons, lats, comb, cmap=THREAT_CMAP,
                       shading="auto", vmin=0.998, vmax=1.0)
    fig.colorbar(im, ax=ax, label="P_d (combined)")

    import math
    R = 6371000.0
    for sam in SAM_SYSTEMS:
        # Draw range ring
        angles = np.linspace(0, 2 * math.pi, 360)
        ring_lats = sam["lat"] + (sam["range_km"] * 1000 / R) * np.degrees(1) * np.cos(angles)
        ring_lons = sam["lon"] + (sam["range_km"] * 1000 / (R * math.cos(math.radians(sam["lat"])))) * np.degrees(1) * np.sin(angles)
        ax.plot(ring_lons, ring_lats, "--", lw=1.5, color=sam["color"],
                label=f"{sam['name']} R={sam['range_km']:.0f} km")
        ax.plot(sam["lon"], sam["lat"], "X", ms=14, color=sam["color"], mew=2)

    # Theatre boundary
    rect = plt.Rectangle((LON_MIN, LAT_MIN), LON_MAX - LON_MIN, LAT_MAX - LAT_MIN,
                          fill=False, edgecolor="white", lw=2, ls="--")
    ax.add_patch(rect)
    ax.plot(77.0, 28.5, "g^", ms=12, label="Origin")
    ax.plot(77.4, 28.85, "ws", ms=12, label="Goal")
    ax.set_xlim(76.2, 78.1)
    ax.set_ylim(27.9, 29.4)
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title("Figure F1 — SAM Coverage Map with Engagement Range Rings\n"
                 "All three systems fully envelop the 55 × 55 km theatre (P_d > 0.998 everywhere)")
    save_fig(fig, "F1_sam_range_rings")


def plot_F2_pd_range_curve(d: dict) -> None:
    logger.info("[F2] P_d vs range curve")
    import math
    P_FA = 1e-6
    SNR_AT_RMAX = math.log(P_FA) / math.log(0.9) - 1.0

    ranges_km = np.linspace(1, 200, 500)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for sam in SAM_SYSTEMS:
        R_max = sam["range_km"] * 1000
        R = ranges_km * 1000
        snr = SNR_AT_RMAX * (R_max / np.maximum(R, 500)) ** 4
        pd = P_FA ** (1.0 / (1.0 + snr))
        pd = np.where(R > R_max, 0.0, pd)
        axes[0].plot(ranges_km, pd, label=sam["name"], color=sam["color"], lw=2)
        axes[1].semilogy(ranges_km, np.maximum(1 - pd, 1e-10),
                         label=sam["name"], color=sam["color"], lw=2)

    # Mark theatre extent
    theatre_max_range = 55.0   # km approx half-diagonal
    for ax in axes:
        ax.axvline(theatre_max_range, color="blue", ls=":", lw=1.5,
                   label=f"Theatre half-diag ({theatre_max_range} km)")

    axes[0].set_xlabel("Slant range (km)")
    axes[0].set_ylabel("P_detection (Swerling I)")
    axes[0].set_title("Detection Probability vs Range")
    axes[0].legend()

    axes[1].set_xlabel("Slant range (km)")
    axes[1].set_ylabel("P_survival = 1 − P_d (log scale)")
    axes[1].set_title("Survival Probability vs Range")
    axes[1].legend()

    fig.suptitle("Figure F2 — Swerling Case I P_d vs Slant Range\n"
                 "P_d = P_fa^(1/(1+SNR)),  SNR = SNR_max·(R_max/R)⁴  [Radar Range Equation]")
    save_fig(fig, "F2_pd_range_curve")


# ===========================================================================
# G. WIND ROSE
# ===========================================================================

def plot_G1_wind_rose(d: dict) -> None:
    logger.info("[G1] Wind rose (polar)")
    alts = d["alts"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5),
                             subplot_kw={"projection": "polar"})

    for ax, alt_m in zip(axes, [50, 200, 1000]):
        ai = list(alts).index(alt_m)
        dirs  = d["wind_dir_deg"][ai].ravel()
        speeds= d["wind_speed_mps"][ai].ravel()

        n_dir = 16
        bins  = np.linspace(0, 360, n_dir + 1)
        theta = np.radians((bins[:-1] + bins[1:]) / 2)
        # Count in each dir bin
        hist, _ = np.histogram(dirs, bins=bins)
        width = 2 * np.pi / n_dir
        bars = ax.bar(theta, hist / hist.max(), width=width, bottom=0,
                      color=plt.cm.plasma(hist / hist.max()), alpha=0.8, edgecolor="none")
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_title(f"Alt = {alt_m} m\nMean {speeds.mean():.1f} m/s", pad=15)
        ax.set_xticks(np.radians([0, 45, 90, 135, 180, 225, 270, 315]))
        ax.set_xticklabels(["N", "NE", "E", "SE", "S", "SW", "W", "NW"], fontsize=8)

    fig.suptitle("Figure G1 — Wind Rose: Frequency of Wind FROM Direction\n"
                 "Delhi NCR Theatre — Real GFS/HRES Forecast (Open-Meteo)")
    save_fig(fig, "G1_wind_rose")


# ===========================================================================
# H. SUMMARY DASHBOARD
# ===========================================================================

def plot_H1_summary_dashboard(d: dict) -> None:
    logger.info("[H1] Summary dashboard")
    alts = d["alts"]
    ai = list(alts).index(200)
    lats, lons = d["lats"], d["lons"]

    fig = plt.figure(figsize=(20, 14))
    gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.4)

    # (0,0) Elevation
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.pcolormesh(lons, lats, d["elev_m"][0], cmap="terrain", shading="auto")
    ax0.set_title("Elevation (m)")
    ax0.set_aspect("equal")

    # (0,1) Wind speed
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.pcolormesh(lons, lats, d["wind_speed_mps"][ai], cmap=WIND_CMAP, shading="auto")
    ax1.set_title("Wind speed (m/s, 200m)")
    ax1.set_aspect("equal")

    # (0,2) Obstacle clearance
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.pcolormesh(lons, lats, np.log1p(d["nearest_obstacle_dist_m"][0]),
                   cmap="RdYlGn", shading="auto")
    ax2.set_title("log(Obs. dist.) [m]")
    ax2.set_aspect("equal")

    # (0,3) Fused cost
    ax3 = fig.add_subplot(gs[0, 3])
    ax3.pcolormesh(lons, lats, d["fused_cost"][ai], cmap="RdYlGn_r", shading="auto")
    ax3.set_title("Fused cost (200 m)")
    ax3.set_aspect("equal")

    # (1,0-1) Cost component stacked bar
    ax4 = fig.add_subplot(gs[1, :2])
    comps = ["terrain_cost", "wind_cost", "obstacle_cost", "threat_cost", "energy_cost"]
    weights = [0.15, 0.10, 0.20, 0.40, 0.15]
    means = [d[c][ai].mean() * w for c, w in zip(comps, weights)]
    cols  = ["#2b8cbe","#78c679","#fc8d59","#d7191c","#a6761d"]
    labels = ["Terrain", "Wind", "Obstacle", "Threat", "Energy"]
    ax4.bar(labels, means, color=cols)
    ax4.set_ylabel("Weighted mean cost")
    ax4.set_title("Cost Component Contributions at 200 m MSL")

    # (1,2-3) Wind altitude profile
    ax5 = fig.add_subplot(gs[1, 2:])
    mean_spd = [d["wind_speed_mps"][a_].mean() for a_ in range(len(alts))]
    ax5.barh(range(len(alts)), mean_spd, color=plt.cm.viridis(np.array(mean_spd)/max(mean_spd)))
    ax5.set_yticks(range(len(alts)))
    ax5.set_yticklabels([str(a) for a in alts], fontsize=8)
    ax5.set_xlabel("Mean wind speed (m/s)")
    ax5.set_ylabel("Altitude (m)")
    ax5.set_title("Wind Speed Profile")

    # (2,0-1) Elevation hist
    ax6 = fig.add_subplot(gs[2, :2])
    ax6.hist(d["elev_m"][0].ravel(), bins=60, color="#2171b5", edgecolor="none", density=True)
    ax6.set_xlabel("Elevation (m)")
    ax6.set_title("Elevation Distribution")

    # (2,2-3) Fused cost hist
    ax7 = fig.add_subplot(gs[2, 2:])
    ax7.hist(d["fused_cost"][ai].ravel(), bins=60, color="#8856a7", edgecolor="none",
             density=True)
    ax7.set_xlabel("Fused cost [0–1]")
    ax7.set_title("Fused Cost Distribution (200 m)")

    fig.suptitle("Figure H1 — Perception Dataset Summary Dashboard\n"
                 "Delhi NCR Theatre  |  1,057,714 rows  |  5 perception layers",
                 fontsize=14, fontweight="bold")
    save_fig(fig, "H1_summary_dashboard")


# ===========================================================================
# MAIN
# ===========================================================================

def main() -> None:
    logger.info("=" * 65)
    logger.info("PERCEPTION VISUAL ANALYSIS SUITE")
    logger.info("Output directory: %s", VISUALS_DIR)
    logger.info("=" * 65)

    # Load data
    d = load_dataset()
    osm_rows = load_osm()

    # --- A: Spatial maps
    plot_A1_terrain_elevation(d)
    plot_A2_slope_roughness(d)
    plot_A3_wind_quiver(d)
    plot_A4_obstacle_density(d, osm_rows)
    plot_A5_threat_pd_contours(d)
    plot_A6_fused_cost_map(d)

    # --- B: Distributions
    plot_B1_elevation_distribution(d)
    plot_B2_wind_distributions(d)
    plot_B3_obstacle_distributions(d, osm_rows)
    plot_B4_cost_distributions(d)

    # --- C: Altitude profiles
    plot_C1_wind_altitude_profile(d)
    plot_C2_terrain_clearance_profile(d)
    plot_C3_fused_cost_altitude(d)

    # --- D: Correlation
    plot_D1_correlation_matrix(d)
    plot_D2_scatter_pairs(d)

    # --- E: Operational
    plot_E1_cost_component_breakdown(d)
    plot_E2_safe_corridor_analysis(d)
    plot_E3_mission_cross_section(d)

    # --- F: Threat
    plot_F1_sam_range_rings(d)
    plot_F2_pd_range_curve(d)

    # --- G: Wind rose
    plot_G1_wind_rose(d)

    # --- H: Dashboard
    plot_H1_summary_dashboard(d)

    # Final count
    pngs = list(VISUALS_DIR.glob("*.png"))
    logger.info("=" * 65)
    logger.info("DONE — %d figures saved to %s", len(pngs), VISUALS_DIR)
    logger.info("=" * 65)


if __name__ == "__main__":
    main()
