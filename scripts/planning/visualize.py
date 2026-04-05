#!/usr/bin/env python3
"""
Exhaustive Visual Analysis Suite — Defense eVTOL Planning Dataset
=================================================================

Generates publication-quality figures (300 DPI PNG + SVG + PDF) for the
planning trajectory dataset (test_final.parquet, 2000 rows, 25 columns).

Categories
----------
  A.  Spatial Maps          — start/goal density, trajectory geo scatter
  B.  Path Metric Stats     — path length, n_waypoints, planning time
  C.  Cost Field Analysis   — terrain/wind/obstacle/threat/fused distributions, CDF
  D.  Energy Analysis       — energy histograms, vs path length, altitude gain
  E.  Threat Analysis       — threat cost, max combined threat, feasibility
  F.  Risk & Feasibility    — pie chart, threshold sensitivity, ROC-style curves
  G.  Algorithm Performance — planning time scaling, RRT* cost, waypoint efficiency
  H.  Correlation Analysis  — full heatmap, cross-metric scatter matrix
  I.  Comparative by Label  — violin plots, box plots (all cost columns)
  J.  Multi-variate         — parallel coordinates, PCA biplot, pairplot
  K.  Operational Analysis  — Pareto front energy vs threat, cost breakdown stacked

All figures saved to:
  visuals/planning/{A1_...}.{png,svg,pdf}
  outputs/visuals/planning/{A1_...}.{png,svg,pdf}

Usage:
    python scripts/visualize_planning_dataset.py

Author: Defense eVTOL Research Team
"""

from __future__ import annotations

import io
import logging
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.ndimage import gaussian_filter
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("planning_visuals")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_PATH   = REPO_ROOT / "outputs" / "planning_dataset" / "test_final.parquet"
OUT_DIRS = [
    REPO_ROOT / "visuals" / "planning",
    REPO_ROOT / "outputs" / "visuals" / "planning",
]
for d in OUT_DIRS:
    d.mkdir(parents=True, exist_ok=True)

FORMATS = ["png", "svg", "pdf"]
DPI     = 300

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
PALETTE     = {"low": "#2196F3", "high": "#F44336"}
RISK_COLORS = ["#2196F3", "#F44336"]
COST_COLORS = {
    "terrain_cost_mean":  "#795548",
    "wind_cost_mean":     "#03A9F4",
    "obstacle_cost_mean": "#FF9800",
    "threat_cost":        "#F44336",
    "fused_cost_mean":    "#9C27B0",
}
STYLE = "seaborn-v0_8-whitegrid"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def savefig(name: str) -> None:
    for fmt in FORMATS:
        for out_dir in OUT_DIRS:
            out_dir.mkdir(parents=True, exist_ok=True)
            fig_path = out_dir / f"{name}.{fmt}"
            plt.savefig(fig_path, dpi=DPI, bbox_inches="tight")
    logger.info("  Saved  %s  [png+svg+pdf]", name)
    plt.close("all")


def hist_kde(ax: plt.Axes, data: pd.Series, color: str, label: str = "",
             bins: int = 40, xlabel: str = "", title: str = "") -> None:
    vals = data.dropna().values
    data_range = vals.max() - vals.min() if len(vals) > 1 else 0.0
    if data_range < 1e-6 * max(abs(float(np.mean(vals))), 1e-12):
        ax.text(0.5, 0.5, f"No variance\n(all ~{vals.mean():.4g})",
                ha="center", va="center", transform=ax.transAxes, fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")
        return
    ax.hist(vals, bins=bins, density=True, alpha=0.5, color=color,
            edgecolor="white", linewidth=0.4, label="Histogram")
    kde = gaussian_kde(vals, bw_method="scott")
    xs = np.linspace(vals.min(), vals.max(), 500)
    ax.plot(xs, kde(xs), color=color, linewidth=2.0, label="KDE")
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    if label:
        ax.legend(fontsize=9)


def cdf_plot(ax: plt.Axes, data: pd.Series, color: str,
             xlabel: str = "", title: str = "") -> None:
    vals = np.sort(data.dropna().values)
    cdf  = np.arange(1, len(vals) + 1) / len(vals)
    ax.plot(vals, cdf, color=color, linewidth=2.0)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel("CDF", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1)
    ax.axhline(0.5, color="grey", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.axhline(0.9, color="grey", linewidth=0.8, linestyle=":", alpha=0.6)


def _risk_label(r: int) -> str:
    return "Low Risk" if r == 0 else "High Risk"


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load() -> pd.DataFrame:
    logger.info("Loading %s", DATA_PATH)
    df = pd.read_parquet(DATA_PATH)
    logger.info("  Shape: %s  |  Columns: %s", df.shape, df.columns.tolist())

    # Derived columns used across categories
    df["path_length_km"]      = df["path_length_m"] / 1000.0
    df["energy_per_km"]       = df["energy_cost_wh"] / (df["path_length_km"].clip(lower=0.01))
    df["straight_dist_km"]    = haversine_km(
        df["start_lat"], df["start_lon"], df["goal_lat"], df["goal_lon"])
    df["detour_ratio"]        = df["path_length_km"] / (df["straight_dist_km"].clip(lower=0.01))
    df["alt_gain_m"]          = (df["goal_alt_m"] - df["start_alt_m"]).clip(lower=0)
    df["alt_drop_m"]          = (df["start_alt_m"] - df["goal_alt_m"]).clip(lower=0)
    df["mean_alt_m"]          = (df["start_alt_m"] + df["goal_alt_m"]) / 2.0
    df["risk_str"]            = df["risk_label"].map({0: "Low", 1: "High"})
    return df


def haversine_km(lat1, lon1, lat2, lon2) -> pd.Series:
    R = 6371.0
    la1, lo1 = np.radians(lat1), np.radians(lon1)
    la2, lo2 = np.radians(lat2), np.radians(lon2)
    dlat = la2 - la1
    dlon = lo2 - lo1
    a = np.sin(dlat / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin(dlon / 2) ** 2
    return pd.Series(R * 2 * np.arcsin(np.clip(np.sqrt(a), 0, 1)), index=lat1.index)


# ===========================================================================
# CATEGORY A — Spatial Maps
# ===========================================================================

def cat_A(df: pd.DataFrame) -> None:
    logger.info("Category A: Spatial Maps")
    plt.style.use(STYLE)

    # A1 — Start/Goal density heatmap
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for ax, col_lat, col_lon, title in [
        (axes[0], "start_lat", "start_lon", "A1a. Start-Point Density"),
        (axes[1], "goal_lat",  "goal_lon",  "A1b. Goal-Point Density"),
    ]:
        lats = df[col_lat].values
        lons = df[col_lon].values
        h, ye, xe = np.histogram2d(lats, lons, bins=40)
        h = gaussian_filter(h, sigma=1.0)
        im = ax.pcolormesh(xe, ye, h, cmap="YlOrRd", shading="auto")
        plt.colorbar(im, ax=ax, label="Count")
        ax.scatter(lons[df["risk_label"] == 0], lats[df["risk_label"] == 0],
                   s=3, alpha=0.3, color=PALETTE["low"], label="Low Risk")
        ax.scatter(lons[df["risk_label"] == 1], lats[df["risk_label"] == 1],
                   s=3, alpha=0.5, color=PALETTE["high"], label="High Risk")
        ax.set_xlabel("Longitude", fontsize=10)
        ax.set_ylabel("Latitude", fontsize=10)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.legend(fontsize=8, markerscale=3)
    fig.suptitle("Defense eVTOL — Spatial Distribution of Mission Endpoints",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    savefig("A1_spatial_density")

    # A2 — Trajectory vector map (start→goal arrows, coloured by risk)
    fig, ax = plt.subplots(figsize=(12, 9))
    sample = df.sample(min(400, len(df)), random_state=42)
    for _, row in sample.iterrows():
        color = PALETTE["high"] if row["risk_label"] == 1 else PALETTE["low"]
        ax.annotate("",
                    xy=(row["goal_lon"], row["goal_lat"]),
                    xytext=(row["start_lon"], row["start_lat"]),
                    arrowprops=dict(arrowstyle="->", color=color, alpha=0.4, lw=0.8))
    legend_els = [Line2D([0], [0], color=PALETTE["low"],  lw=1.5, label="Low Risk"),
                  Line2D([0], [0], color=PALETTE["high"], lw=1.5, label="High Risk")]
    ax.legend(handles=legend_els, fontsize=10)
    ax.set_xlabel("Longitude", fontsize=11)
    ax.set_ylabel("Latitude", fontsize=11)
    ax.set_title("A2. Trajectory Vector Map (400 sampled routes)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    savefig("A2_trajectory_vectors")

    # A3 — Detour ratio geo scatter
    fig, ax = plt.subplots(figsize=(10, 8))
    sc = ax.scatter(df["start_lon"], df["start_lat"],
                    c=df["detour_ratio"].clip(upper=3.0), cmap="plasma",
                    s=8, alpha=0.6)
    plt.colorbar(sc, ax=ax, label="Detour Ratio (path / straight-line)")
    ax.set_xlabel("Start Longitude", fontsize=11)
    ax.set_ylabel("Start Latitude", fontsize=11)
    ax.set_title("A3. Detour Ratio by Start Location", fontsize=12, fontweight="bold")
    plt.tight_layout()
    savefig("A3_detour_ratio_map")


# ===========================================================================
# CATEGORY B — Path Metric Statistics
# ===========================================================================

def cat_B(df: pd.DataFrame) -> None:
    logger.info("Category B: Path Metric Statistics")
    plt.style.use(STYLE)

    # B1 — Path length + detour ratio distributions
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    pairs = [
        (axes[0, 0], df["path_length_km"],    "#1565C0", "Path Length (km)",          "B1a. Path Length Distribution"),
        (axes[0, 1], df["straight_dist_km"],  "#0288D1", "Straight-line Dist. (km)",  "B1b. Straight-line Distance"),
        (axes[1, 0], df["detour_ratio"],       "#6A1B9A", "Detour Ratio",              "B1c. Detour Ratio Distribution"),
        (axes[1, 1], df["n_waypoints"].astype(float), "#2E7D32", "# Waypoints",       "B1d. Waypoint Count"),
    ]
    for ax, col, color, xlabel, title in pairs:
        hist_kde(ax, col, color, xlabel=xlabel, title=title)
    fig.suptitle("Path Geometry Statistics", fontsize=13, fontweight="bold")
    plt.tight_layout()
    savefig("B1_path_geometry")

    # B2 — Path length by risk label (side-by-side violin)
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    for ax, col, ylabel in [
        (axes[0], "path_length_km", "Path Length (km)"),
        (axes[1], "n_waypoints",    "# Waypoints"),
        (axes[2], "detour_ratio",   "Detour Ratio"),
    ]:
        sns.violinplot(data=df, x="risk_str", y=col, palette=RISK_COLORS,
                       order=["Low", "High"], ax=ax, cut=0)
        ax.set_xlabel("Risk Label", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(f"B2. {ylabel} by Risk", fontsize=11, fontweight="bold")
    plt.tight_layout()
    savefig("B2_path_metrics_by_risk")

    # B3 — CDF panel for path metrics
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    cdf_plot(axes[0], df["path_length_km"],  "#1565C0", "km", "B3a. CDF Path Length")
    cdf_plot(axes[1], df["detour_ratio"],    "#6A1B9A", "",   "B3b. CDF Detour Ratio")
    cdf_plot(axes[2], df["n_waypoints"].astype(float), "#2E7D32", "", "B3c. CDF Waypoints")
    plt.tight_layout()
    savefig("B3_path_metrics_cdf")


# ===========================================================================
# CATEGORY C — Cost Field Analysis
# ===========================================================================

def cat_C(df: pd.DataFrame) -> None:
    logger.info("Category C: Cost Field Analysis")
    plt.style.use(STYLE)

    cost_cols = list(COST_COLORS.keys())

    # C1 — 5-panel histogram of all cost fields
    fig, axes = plt.subplots(1, 5, figsize=(22, 5))
    for ax, col in zip(axes, cost_cols):
        hist_kde(ax, df[col], COST_COLORS[col],
                 xlabel=col.replace("_", " ").replace("mean", "").strip(),
                 title=f"C1. {col.split('_')[0].title()}")
    fig.suptitle("Cost Field Distributions", fontsize=13, fontweight="bold")
    plt.tight_layout()
    savefig("C1_cost_distributions")

    # C2 — CDF for all cost fields overlaid
    fig, ax = plt.subplots(figsize=(10, 7))
    for col in cost_cols:
        vals = np.sort(df[col].dropna().values)
        cdf  = np.arange(1, len(vals) + 1) / len(vals)
        ax.plot(vals, cdf, color=COST_COLORS[col], linewidth=2.0,
                label=col.replace("_mean", "").replace("_", " ").title())
    ax.set_xlabel("Cost Value", fontsize=11)
    ax.set_ylabel("CDF", fontsize=11)
    ax.set_title("C2. Cumulative Distribution — All Cost Fields", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.axhline(0.5, color="grey", linewidth=0.7, linestyle="--", alpha=0.5)
    ax.axhline(0.9, color="grey", linewidth=0.7, linestyle=":", alpha=0.5)
    plt.tight_layout()
    savefig("C2_cost_cdf_overlay")

    # C3 — Box plots of cost fields by risk label
    fig, axes = plt.subplots(1, 5, figsize=(22, 6))
    for ax, col in zip(axes, cost_cols):
        sns.boxplot(data=df, x="risk_str", y=col, palette=RISK_COLORS,
                    order=["Low", "High"], ax=ax, width=0.5)
        ax.set_xlabel("Risk", fontsize=9)
        ax.set_ylabel(col.replace("_mean", ""), fontsize=9)
        ax.set_title(col.split("_")[0].title(), fontsize=10, fontweight="bold")
    fig.suptitle("C3. Cost Fields by Risk Label", fontsize=13, fontweight="bold")
    plt.tight_layout()
    savefig("C3_cost_boxplots_by_risk")

    # C4 — Radar chart: mean costs per risk class
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), subplot_kw=dict(polar=True))
    labels = [c.replace("_cost_mean", "").replace("_cost", "").replace("_mean", "").title()
              for c in cost_cols]
    n = len(cost_cols)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]
    for ax, risk_val, color, title in [
        (axes[0], 0, PALETTE["low"],  "Low Risk Trajectories"),
        (axes[1], 1, PALETTE["high"], "High Risk Trajectories"),
    ]:
        vals_r = [df.loc[df["risk_label"] == risk_val, c].mean() for c in cost_cols]
        vals_r += vals_r[:1]
        ax.plot(angles, vals_r, color=color, linewidth=2)
        ax.fill(angles, vals_r, color=color, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_title(f"C4. {title}", fontsize=11, fontweight="bold", pad=15)
    plt.tight_layout()
    savefig("C4_cost_radar")


# ===========================================================================
# CATEGORY D — Energy Analysis
# ===========================================================================

def cat_D(df: pd.DataFrame) -> None:
    logger.info("Category D: Energy Analysis")
    plt.style.use(STYLE)

    # D1 — Energy histogram + KDE by risk
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for risk_val, color, label in [(0, PALETTE["low"], "Low Risk"),
                                    (1, PALETTE["high"], "High Risk")]:
        sub = df.loc[df["risk_label"] == risk_val, "energy_cost_wh"]
        hist_kde(axes[0], sub, color, label=label,
                 xlabel="Energy (Wh)", title="D1a. Energy Distribution by Risk")
    axes[0].legend()
    cdf_plot(axes[1], df["energy_cost_wh"], "#9C27B0",
             xlabel="Energy (Wh)", title="D1b. Energy CDF")
    plt.tight_layout()
    savefig("D1_energy_distributions")

    # D2 — Energy vs path length scatter
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for risk_val, color, label in [(0, PALETTE["low"], "Low Risk"),
                                    (1, PALETTE["high"], "High Risk")]:
        mask = df["risk_label"] == risk_val
        axes[0].scatter(df.loc[mask, "path_length_km"],
                        df.loc[mask, "energy_cost_wh"],
                        c=color, s=6, alpha=0.5, label=label)
    axes[0].set_xlabel("Path Length (km)", fontsize=10)
    axes[0].set_ylabel("Energy (Wh)", fontsize=10)
    axes[0].set_title("D2a. Energy vs Path Length", fontsize=11, fontweight="bold")
    axes[0].legend()
    # Energy per km by risk
    sns.violinplot(data=df, x="risk_str", y="energy_per_km",
                   palette=RISK_COLORS, order=["Low", "High"], ax=axes[1], cut=0)
    axes[1].set_xlabel("Risk Label", fontsize=10)
    axes[1].set_ylabel("Energy per km (Wh/km)", fontsize=10)
    axes[1].set_title("D2b. Energy Efficiency by Risk", fontsize=11, fontweight="bold")
    plt.tight_layout()
    savefig("D2_energy_vs_path")

    # D3 — Energy vs altitude gain, time cost
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sc = axes[0].scatter(df["alt_gain_m"], df["energy_cost_wh"],
                         c=df["risk_label"], cmap="coolwarm", s=8, alpha=0.6)
    plt.colorbar(sc, ax=axes[0], label="Risk (0=Low, 1=High)")
    axes[0].set_xlabel("Altitude Gain (m)", fontsize=10)
    axes[0].set_ylabel("Energy (Wh)", fontsize=10)
    axes[0].set_title("D3a. Energy vs Altitude Gain", fontsize=11, fontweight="bold")

    sc2 = axes[1].scatter(df["time_cost_s"], df["energy_cost_wh"],
                          c=df["risk_label"], cmap="coolwarm", s=8, alpha=0.6)
    plt.colorbar(sc2, ax=axes[1], label="Risk")
    axes[1].set_xlabel("Time Cost (s)", fontsize=10)
    axes[1].set_ylabel("Energy (Wh)", fontsize=10)
    axes[1].set_title("D3b. Energy vs Time Cost", fontsize=11, fontweight="bold")
    plt.tight_layout()
    savefig("D3_energy_vs_altitude_time")


# ===========================================================================
# CATEGORY E — Threat Analysis
# ===========================================================================

def cat_E(df: pd.DataFrame) -> None:
    logger.info("Category E: Threat Analysis")
    plt.style.use(STYLE)

    # E1 — Threat cost + max combined threat distributions
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    hist_kde(axes[0], df["threat_cost"], "#F44336",
             xlabel="Threat Cost", title="E1a. Threat Cost Distribution")
    hist_kde(axes[1], df["max_combined_threat"], "#B71C1C",
             xlabel="Max Combined Threat", title="E1b. Max Combined Threat")
    # threat cost split by risk
    for risk_val, color, label in [(0, PALETTE["low"], "Low Risk"),
                                    (1, PALETTE["high"], "High Risk")]:
        sub = df.loc[df["risk_label"] == risk_val, "threat_cost"]
        vals = sub.dropna().values
        if len(vals) > 1:
            kde = gaussian_kde(vals, bw_method="scott")
            xs  = np.linspace(vals.min(), vals.max(), 400)
            axes[2].plot(xs, kde(xs), color=color, linewidth=2.0, label=label)
    axes[2].set_xlabel("Threat Cost", fontsize=10)
    axes[2].set_ylabel("Density", fontsize=10)
    axes[2].set_title("E1c. Threat KDE by Risk Label", fontsize=11, fontweight="bold")
    axes[2].legend()
    plt.tight_layout()
    savefig("E1_threat_distributions")

    # E2 — Threat vs feasibility scatter + threat CDF by risk
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sc = axes[0].scatter(df["threat_cost"], df["max_combined_threat"],
                         c=df["risk_label"], cmap="coolwarm", s=8, alpha=0.6)
    plt.colorbar(sc, ax=axes[0], label="Risk (0=Low, 1=High)")
    axes[0].set_xlabel("Threat Cost", fontsize=10)
    axes[0].set_ylabel("Max Combined Threat", fontsize=10)
    axes[0].set_title("E2a. Threat Cost vs Max Combined", fontsize=11, fontweight="bold")

    for risk_val, color, label in [(0, PALETTE["low"], "Low Risk"),
                                    (1, PALETTE["high"], "High Risk")]:
        sub = np.sort(df.loc[df["risk_label"] == risk_val, "threat_cost"].dropna().values)
        cdf = np.arange(1, len(sub) + 1) / len(sub)
        axes[1].plot(sub, cdf, color=color, linewidth=2.0, label=label)
    axes[1].set_xlabel("Threat Cost", fontsize=10)
    axes[1].set_ylabel("CDF", fontsize=10)
    axes[1].set_title("E2b. Threat CDF by Risk Label", fontsize=11, fontweight="bold")
    axes[1].legend()
    plt.tight_layout()
    savefig("E2_threat_vs_risk")

    # E3 — Threat threshold sensitivity (what fraction exceed threshold)
    fig, ax = plt.subplots(figsize=(10, 6))
    thresholds = np.linspace(0, df["threat_cost"].max(), 100)
    for risk_val, color, label in [(0, PALETTE["low"], "Low Risk"),
                                    (1, PALETTE["high"], "High Risk")]:
        sub = df.loc[df["risk_label"] == risk_val, "threat_cost"].dropna().values
        frac = [(sub > t).mean() for t in thresholds]
        ax.plot(thresholds, frac, color=color, linewidth=2.2, label=label)
    ax.set_xlabel("Threat Cost Threshold", fontsize=11)
    ax.set_ylabel("Fraction Exceeding Threshold", fontsize=11)
    ax.set_title("E3. Threat Threshold Sensitivity", fontsize=12, fontweight="bold")
    ax.legend()
    ax.axhline(0.1, color="grey", linewidth=0.8, linestyle="--", alpha=0.5)
    plt.tight_layout()
    savefig("E3_threat_threshold_sensitivity")


# ===========================================================================
# CATEGORY F — Risk & Feasibility
# ===========================================================================

def cat_F(df: pd.DataFrame) -> None:
    logger.info("Category F: Risk & Feasibility")
    plt.style.use(STYLE)

    # F1 — Pie charts: risk label, feasibility, altitude_clearance_ok, speed_ok
    fig, axes = plt.subplots(1, 4, figsize=(20, 6))
    for ax, col, title in [
        (axes[0], "risk_label",           "F1a. Risk Label"),
        (axes[1], "feasible",             "F1b. Feasible"),
        (axes[2], "altitude_clearance_ok","F1c. Alt. Clearance OK"),
        (axes[3], "speed_ok",             "F1d. Speed OK"),
    ]:
        vc = df[col].value_counts().sort_index()
        colors_pie = [PALETTE["low"], PALETTE["high"]] if len(vc) == 2 else None
        wedges, texts, autotexts = ax.pie(
            vc.values, labels=[str(k) for k in vc.index],
            autopct="%1.1f%%", startangle=90,
            colors=colors_pie,
        )
        for t in autotexts:
            t.set_fontsize(9)
        ax.set_title(title, fontsize=11, fontweight="bold")
    plt.tight_layout()
    savefig("F1_feasibility_pie_charts")

    # F2 — Risk label distribution + feasibility breakdown
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    risk_counts = df["risk_label"].value_counts().sort_index()
    axes[0].bar(["Low Risk (0)", "High Risk (1)"], risk_counts.values,
                color=RISK_COLORS, edgecolor="white", linewidth=0.5)
    for i, v in enumerate(risk_counts.values):
        axes[0].text(i, v + 5, str(v), ha="center", fontsize=10)
    axes[0].set_ylabel("Count", fontsize=10)
    axes[0].set_title("F2a. Risk Label Count", fontsize=11, fontweight="bold")

    feasibility_by_risk = df.groupby("risk_label")["feasible"].mean()
    axes[1].bar(["Low Risk (0)", "High Risk (1)"], feasibility_by_risk.values,
                color=RISK_COLORS, edgecolor="white")
    axes[1].set_ylabel("Fraction Feasible", fontsize=10)
    axes[1].set_ylim(0, 1.1)
    axes[1].set_title("F2b. Feasibility Rate by Risk", fontsize=11, fontweight="bold")
    for i, v in enumerate(feasibility_by_risk.values):
        axes[1].text(i, v + 0.01, f"{v:.2%}", ha="center", fontsize=10)
    plt.tight_layout()
    savefig("F2_risk_feasibility_bar")

    # F3 — Fused cost threshold sensitivity (Precision-Recall style)
    fig, ax = plt.subplots(figsize=(10, 6))
    thresholds = np.linspace(df["fused_cost_mean"].min(), df["fused_cost_mean"].max(), 200)
    precision_list, recall_list = [], []
    y_true = df["risk_label"].values
    scores = df["fused_cost_mean"].values
    for t in thresholds:
        pred = (scores >= t).astype(int)
        tp = ((pred == 1) & (y_true == 1)).sum()
        fp = ((pred == 1) & (y_true == 0)).sum()
        fn = ((pred == 0) & (y_true == 1)).sum()
        precision_list.append(tp / (tp + fp + 1e-9))
        recall_list.append(tp / (tp + fn + 1e-9))
    ax.plot(recall_list, precision_list, color="#9C27B0", linewidth=2.2)
    ax.set_xlabel("Recall (Sensitivity)", fontsize=11)
    ax.set_ylabel("Precision (PPV)", fontsize=11)
    ax.set_title("F3. Precision-Recall Curve (Fused Cost as Risk Score)", fontsize=12, fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    savefig("F3_precision_recall")


# ===========================================================================
# CATEGORY G — Algorithm Performance
# ===========================================================================

def cat_G(df: pd.DataFrame) -> None:
    logger.info("Category G: Algorithm Performance")
    plt.style.use(STYLE)

    # G1 — Planning time distribution
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    hist_kde(axes[0], df["planning_time_s"], "#00695C",
             xlabel="Planning Time (s)", title="G1a. Planning Time Distribution")
    cdf_plot(axes[1], df["planning_time_s"], "#00695C",
             xlabel="Planning Time (s)", title="G1b. Planning Time CDF")
    sc = axes[2].scatter(df["path_length_km"], df["planning_time_s"],
                         c=df["risk_label"], cmap="coolwarm", s=8, alpha=0.6)
    plt.colorbar(sc, ax=axes[2], label="Risk")
    axes[2].set_xlabel("Path Length (km)", fontsize=10)
    axes[2].set_ylabel("Planning Time (s)", fontsize=10)
    axes[2].set_title("G1c. Planning Time vs Path Length", fontsize=11, fontweight="bold")
    plt.tight_layout()
    savefig("G1_planning_time")

    # G2 — RRT* cost analysis
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    hist_kde(axes[0], df["rrt_cost"], "#1B5E20",
             xlabel="RRT* Cost", title="G2a. RRT* Cost Distribution")
    axes[1].scatter(df["path_length_km"], df["rrt_cost"],
                    c=df["risk_label"], cmap="coolwarm", s=8, alpha=0.6)
    axes[1].set_xlabel("Path Length (km)", fontsize=10)
    axes[1].set_ylabel("RRT* Cost", fontsize=10)
    axes[1].set_title("G2b. RRT* Cost vs Path Length", fontsize=11, fontweight="bold")
    axes[2].scatter(df["rrt_cost"], df["fused_cost_mean"],
                    c=df["risk_label"], cmap="coolwarm", s=8, alpha=0.6)
    axes[2].set_xlabel("RRT* Cost", fontsize=10)
    axes[2].set_ylabel("Fused Cost Mean", fontsize=10)
    axes[2].set_title("G2c. RRT* Cost vs Fused Cost", fontsize=11, fontweight="bold")
    plt.tight_layout()
    savefig("G2_rrt_cost_analysis")

    # G3 — Waypoints efficiency
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].scatter(df["n_waypoints"], df["path_length_km"],
                    c=df["risk_label"], cmap="coolwarm", s=8, alpha=0.6)
    axes[0].set_xlabel("# Waypoints", fontsize=10)
    axes[0].set_ylabel("Path Length (km)", fontsize=10)
    axes[0].set_title("G3a. Waypoints vs Path Length", fontsize=11, fontweight="bold")
    axes[1].scatter(df["n_waypoints"], df["planning_time_s"],
                    c=df["risk_label"], cmap="coolwarm", s=8, alpha=0.6)
    axes[1].set_xlabel("# Waypoints", fontsize=10)
    axes[1].set_ylabel("Planning Time (s)", fontsize=10)
    axes[1].set_title("G3b. Waypoints vs Planning Time", fontsize=11, fontweight="bold")
    plt.tight_layout()
    savefig("G3_waypoint_efficiency")


# ===========================================================================
# CATEGORY H — Correlation Analysis
# ===========================================================================

def cat_H(df: pd.DataFrame) -> None:
    logger.info("Category H: Correlation Analysis")
    plt.style.use(STYLE)

    num_cols = [
        "path_length_km", "straight_dist_km", "detour_ratio", "n_waypoints",
        "planning_time_s", "rrt_cost", "time_cost_s", "energy_cost_wh",
        "energy_per_km", "threat_cost", "terrain_cost_mean", "wind_cost_mean",
        "obstacle_cost_mean", "fused_cost_mean", "max_combined_threat",
        "alt_gain_m", "risk_label",
    ]
    available = [c for c in num_cols if c in df.columns]
    corr = df[available].corr()

    # H1 — Full correlation heatmap
    fig, ax = plt.subplots(figsize=(14, 12))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, cmap="RdBu_r", vmin=-1, vmax=1,
                annot=True, fmt=".2f", annot_kws={"size": 7},
                square=True, linewidths=0.4, ax=ax)
    ax.set_title("H1. Full Pearson Correlation Matrix — Planning Dataset",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    savefig("H1_correlation_matrix")

    # H2 — Cost cross-scatter (terrain vs wind vs threat vs fused)
    cost_pairs = [
        ("terrain_cost_mean", "wind_cost_mean"),
        ("terrain_cost_mean", "threat_cost"),
        ("wind_cost_mean",    "obstacle_cost_mean"),
        ("threat_cost",       "fused_cost_mean"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    for ax, (cx, cy) in zip(axes.flat, cost_pairs):
        sc = ax.scatter(df[cx], df[cy], c=df["risk_label"],
                        cmap="coolwarm", s=6, alpha=0.5)
        plt.colorbar(sc, ax=ax, label="Risk")
        ax.set_xlabel(cx.replace("_mean", "").replace("_", " ").title(), fontsize=9)
        ax.set_ylabel(cy.replace("_mean", "").replace("_", " ").title(), fontsize=9)
        ax.set_title(f"H2. {cx.split('_')[0].title()} vs {cy.split('_')[0].title()}",
                     fontsize=10, fontweight="bold")
    plt.tight_layout()
    savefig("H2_cost_cross_scatter")


# ===========================================================================
# CATEGORY I — Comparative by Label (Violin / Box)
# ===========================================================================

def cat_I(df: pd.DataFrame) -> None:
    logger.info("Category I: Comparative by Risk Label")
    plt.style.use(STYLE)

    cost_cols = list(COST_COLORS.keys())

    # I1 — Violin grid: all 5 costs + energy + path_length
    cols_v = cost_cols + ["energy_cost_wh", "path_length_km", "planning_time_s"]
    fig, axes = plt.subplots(2, 4, figsize=(22, 11))
    for ax, col in zip(axes.flat, cols_v):
        sns.violinplot(data=df, x="risk_str", y=col, palette=RISK_COLORS,
                       order=["Low", "High"], ax=ax, cut=0, inner="quartile")
        ax.set_xlabel("Risk", fontsize=9)
        ax.set_ylabel(col.replace("_mean", "").replace("_", " "), fontsize=8)
        ax.set_title(col.split("_")[0].title(), fontsize=10, fontweight="bold")
    # hide spare subplot
    if len(cols_v) < len(axes.flat):
        for ax in axes.flat[len(cols_v):]:
            ax.set_visible(False)
    fig.suptitle("I1. Violin Plots — All Metrics by Risk Label",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    savefig("I1_violin_grid")

    # I2 — Strip + box overlay for costs
    fig, axes = plt.subplots(1, 5, figsize=(22, 6))
    for ax, col in zip(axes, cost_cols):
        sns.boxplot(data=df, x="risk_str", y=col, palette=RISK_COLORS,
                    order=["Low", "High"], ax=ax, width=0.45, fliersize=0)
        sns.stripplot(data=df.sample(min(300, len(df)), random_state=7),
                      x="risk_str", y=col, order=["Low", "High"],
                      color="black", size=2, alpha=0.3, ax=ax)
        ax.set_xlabel("Risk", fontsize=9)
        ax.set_ylabel(col.replace("_mean", ""), fontsize=8)
        ax.set_title(col.split("_")[0].title(), fontsize=10, fontweight="bold")
    fig.suptitle("I2. Box + Strip Plot — Cost Fields by Risk Label",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    savefig("I2_box_strip_costs")


# ===========================================================================
# CATEGORY J — Multi-variate (Parallel Coords, PCA, Pairplot)
# ===========================================================================

def cat_J(df: pd.DataFrame) -> None:
    logger.info("Category J: Multi-variate Analysis")
    plt.style.use(STYLE)

    num_cols = [
        "path_length_km", "n_waypoints", "planning_time_s",
        "energy_cost_wh", "threat_cost", "terrain_cost_mean",
        "wind_cost_mean", "fused_cost_mean", "detour_ratio",
    ]
    available = [c for c in num_cols if c in df.columns]
    df_n = df[available + ["risk_str"]].dropna()

    # J1 — Parallel Coordinates (normalised)
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    arr = scaler.fit_transform(df_n[available])
    df_norm = pd.DataFrame(arr, columns=available)
    df_norm["risk_str"] = df_n["risk_str"].values

    sample_j = df_norm.sample(min(300, len(df_norm)), random_state=42)
    fig, ax = plt.subplots(figsize=(18, 7))
    for risk_val, color, label in [("Low", PALETTE["low"], "Low Risk"),
                                    ("High", PALETTE["high"], "High Risk")]:
        sub = sample_j.loc[sample_j["risk_str"] == risk_val, available].values
        for row in sub:
            ax.plot(range(len(available)), row, color=color, alpha=0.1, linewidth=0.6)
    ax.set_xticks(range(len(available)))
    ax.set_xticklabels([c.replace("_mean", "").replace("_", "\n") for c in available],
                       fontsize=8)
    ax.set_ylabel("Normalised Value (0–1)", fontsize=10)
    ax.set_title("J1. Parallel Coordinates (300 sampled, normalised)",
                 fontsize=12, fontweight="bold")
    legend_els = [Line2D([0], [0], color=PALETTE["low"],  lw=1.5, label="Low Risk"),
                  Line2D([0], [0], color=PALETTE["high"], lw=1.5, label="High Risk")]
    ax.legend(handles=legend_els, fontsize=10)
    plt.tight_layout()
    savefig("J1_parallel_coords")

    # J2 — PCA biplot
    X = df[available].dropna()
    sc = StandardScaler()
    X_s = sc.fit_transform(X)
    pca = PCA(n_components=2, random_state=42)
    comps = pca.fit_transform(X_s)
    labels_pca = df.loc[X.index, "risk_label"].values

    fig, ax = plt.subplots(figsize=(10, 8))
    for rv, color, lbl in [(0, PALETTE["low"], "Low Risk"),
                            (1, PALETTE["high"], "High Risk")]:
        mask = labels_pca == rv
        ax.scatter(comps[mask, 0], comps[mask, 1],
                   c=color, s=8, alpha=0.5, label=lbl)
    # Loadings
    scale = np.abs(comps).max() * 0.4
    for i, feat in enumerate(available):
        ax.annotate("", xy=(pca.components_[0, i] * scale,
                             pca.components_[1, i] * scale),
                    xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color="#333", lw=1.0))
        ax.text(pca.components_[0, i] * scale * 1.1,
                pca.components_[1, i] * scale * 1.1,
                feat.replace("_mean", "").replace("_", " "), fontsize=7, color="#333")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var.)", fontsize=11)
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var.)", fontsize=11)
    ax.set_title("J2. PCA Biplot — Planning Feature Space", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    plt.tight_layout()
    savefig("J2_pca_biplot")

    # J3 — Seaborn pairplot (subset of most informative features)
    logger.info("  Generating pairplot (may take a moment)...")
    pp_cols = ["path_length_km", "energy_cost_wh", "threat_cost",
               "fused_cost_mean", "planning_time_s", "risk_str"]
    pp_cols = [c for c in pp_cols if c in df.columns]
    pp_df = df[pp_cols].dropna().sample(min(500, len(df)), random_state=42)
    g = sns.pairplot(pp_df, hue="risk_str", palette={"Low": PALETTE["low"], "High": PALETTE["high"]},
                     plot_kws={"s": 8, "alpha": 0.5}, diag_kind="kde")
    g.fig.suptitle("J3. Pairplot — Key Planning Metrics by Risk", y=1.01,
                   fontsize=13, fontweight="bold")
    savefig("J3_pairplot")


# ===========================================================================
# CATEGORY K — Operational Analysis
# ===========================================================================

def cat_K(df: pd.DataFrame) -> None:
    logger.info("Category K: Operational Analysis")
    plt.style.use(STYLE)

    # K1 — Pareto front: energy vs threat (highlight Pareto-optimal pts)
    fig, ax = plt.subplots(figsize=(11, 8))
    low  = df[df["risk_label"] == 0]
    high = df[df["risk_label"] == 1]
    ax.scatter(high["threat_cost"], high["energy_cost_wh"],
               c=PALETTE["high"], s=8, alpha=0.4, label="High Risk")
    ax.scatter(low["threat_cost"],  low["energy_cost_wh"],
               c=PALETTE["low"],  s=8, alpha=0.4, label="Low Risk")

    # Approximate Pareto front
    pts = df[["threat_cost", "energy_cost_wh"]].dropna().values
    is_pareto = np.ones(len(pts), dtype=bool)
    for i, c in enumerate(pts):
        if is_pareto[i]:
            is_pareto[is_pareto] = ~(
                np.all(pts[is_pareto] <= c, axis=1) &
                np.any(pts[is_pareto] < c, axis=1)
            )
            is_pareto[i] = True
    pareto_pts = pts[is_pareto]
    idx_sort = np.argsort(pareto_pts[:, 0])
    pareto_pts = pareto_pts[idx_sort]
    ax.plot(pareto_pts[:, 0], pareto_pts[:, 1], "k-", linewidth=1.5, label="Pareto Front")
    ax.scatter(pareto_pts[:, 0], pareto_pts[:, 1], c="black", s=15, zorder=5)

    ax.set_xlabel("Threat Cost", fontsize=11)
    ax.set_ylabel("Energy Cost (Wh)", fontsize=11)
    ax.set_title("K1. Pareto Front — Energy vs Threat Trade-off",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    plt.tight_layout()
    savefig("K1_pareto_front")

    # K2 — Stacked cost breakdown (mean per risk class)
    cost_cols = list(COST_COLORS.keys())
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    for ax, risk_val, risk_lbl in [(axes[0], 0, "Low Risk"), (axes[1], 1, "High Risk")]:
        sub = df[df["risk_label"] == risk_val]
        means = [sub[c].mean() for c in cost_cols]
        total = sum(means) or 1.0
        fracs = [m / total for m in means]
        labels = [c.replace("_cost_mean", "").replace("_cost", "").replace("_mean", "")
                    .replace("_", " ").title() for c in cost_cols]
        colors = list(COST_COLORS.values())
        wedges, texts, autotexts = ax.pie(fracs, labels=labels, autopct="%1.1f%%",
                                           colors=colors, startangle=90)
        for t in autotexts:
            t.set_fontsize(8)
        ax.set_title(f"K2. Cost Breakdown — {risk_lbl}", fontsize=11, fontweight="bold")
    plt.tight_layout()
    savefig("K2_cost_breakdown_pie")

    # K3 — Summary stat table figure
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.axis("off")
    stat_cols = ["path_length_km", "energy_cost_wh", "threat_cost",
                 "fused_cost_mean", "planning_time_s", "detour_ratio"]
    stat_cols = [c for c in stat_cols if c in df.columns]
    stats = df[stat_cols].describe().loc[["count", "mean", "std", "min", "50%", "max"]]
    stats.index = ["N", "Mean", "Std", "Min", "Median", "Max"]
    col_labels = [c.replace("_mean", "").replace("_", " ") for c in stat_cols]
    cell_text = [[f"{v:.3g}" for v in row] for row in stats.values]
    table = ax.table(cellText=cell_text, rowLabels=stats.index,
                     colLabels=col_labels, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.3, 1.6)
    ax.set_title("K3. Summary Statistics — Key Planning Metrics",
                 fontsize=13, fontweight="bold", pad=20)
    plt.tight_layout()
    savefig("K3_summary_stats_table")


# ===========================================================================
# MAIN
# ===========================================================================

def main() -> None:
    logger.info("=" * 65)
    logger.info("  Defense eVTOL Planning Dataset — Exhaustive Visual Analysis")
    logger.info("=" * 65)

    df = load()

    # Dataset summary
    logger.info("\nDataset summary:")
    logger.info("  Total trajectories : %d", len(df))
    logger.info("  Low risk  (0)      : %d", (df["risk_label"] == 0).sum())
    logger.info("  High risk (1)      : %d", (df["risk_label"] == 1).sum())
    logger.info("  Mean path length   : %.1f km", df["path_length_km"].mean())
    logger.info("  Mean energy        : %.2f Wh", df["energy_cost_wh"].mean())
    logger.info("  Mean fused cost    : %.4f", df["fused_cost_mean"].mean())
    logger.info("  Output directories :")
    for d in OUT_DIRS:
        logger.info("    %s", d)

    cat_A(df)
    cat_B(df)
    cat_C(df)
    cat_D(df)
    cat_E(df)
    cat_F(df)
    cat_G(df)
    cat_H(df)
    cat_I(df)
    cat_J(df)
    cat_K(df)

    # Summary
    all_pngs = sorted((OUT_DIRS[0]).glob("*.png"))
    logger.info("\n" + "=" * 65)
    logger.info("All done.  %d figure(s) saved (PNG + SVG + PDF) to:", len(all_pngs))
    for d in OUT_DIRS:
        logger.info("  %s", d)
    for p in all_pngs:
        logger.info("    %-55s  %6.1f KB", p.name, p.stat().st_size / 1024)
    logger.info("=" * 65)


if __name__ == "__main__":
    main()
