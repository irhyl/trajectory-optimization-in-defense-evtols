"""
Vehicle Layer Dataset Visualizer

Generates publication-quality figures from outputs/vehicle/vehicle_dataset.parquet.
All figures: 300 DPI, PNG + SVG + PDF
Output directories:
  visuals/vehicle/
  outputs/visuals/vehicle/

Categories:
  A  Mission profile overview (phase fractions, durations, speed)
  B  Rotor / BEMT performance (collective, thrust, FM, tip speed)
  C  Cruise aerodynamics (CL, CD, L/D, drag polar, Re, Mach)
  D  Energy & battery (SOC, energy consumed, voltage, C-rate)
  E  Thermal management (motor/battery temperatures, margins)
  F  Acoustic signatures (SPL breakdown, detection range, BPF)
  G  Infrared signatures (MWIR/LWIR, contrast, skin temp)
  H  RCS (X-band / Ku-band, nose-on vs broadside, blade flash)
  I  Cross-domain correlations (heatmap, scatter)
  J  Comparative by risk label (violin / box)
  K  Operational summary (Pareto, radar chart, mission envelope)
"""

from __future__ import annotations

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.cm as cm
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).resolve().parent.parent
SRC     = ROOT / "outputs" / "vehicle" / "vehicle_dataset.parquet"
OUT_DIRS = [
    ROOT / "visuals"  / "vehicle",
]
for d in OUT_DIRS:
    d.mkdir(parents=True, exist_ok=True)

DPI = 300
FMTS = ["png", "pdf"]

plt.style.use("seaborn-v0_8-whitegrid")
PALETTE = ["#2196F3", "#F44336", "#4CAF50", "#FF9800", "#9C27B0", "#00BCD4"]

# ── Load ───────────────────────────────────────────────────────────────────────
print(f"Loading {SRC}")
df = pd.read_parquet(SRC)
print(f"  {len(df)} rows × {len(df.columns)} columns")

# Derived columns
df["path_length_km"]      = df["path_length_m"] / 1000.0
df["energy_consumed_kWh"] = df["energy_consumed_wh"] / 1000.0
df["soc_delta"]           = df["soc_initial"] - df["soc_final"]
df["mission_time_min"]    = df["mission_time_s"] / 60.0
df["risk_str"]            = df["risk_label"].map({0: "Low Risk", 1: "High Risk"})
df["power_hover_kW"]      = df["power_hover_elec_W"] / 1000.0
df["power_cruise_kW"]     = df["power_cruise_elec_W"] / 1000.0
df["motor_temp_rise_C"]   = df["motor_peak_temp_C"] - 20.0

RISK_COLORS = {"Low Risk": PALETTE[0], "High Risk": PALETTE[1]}
RISK_LABELS = ["Low Risk", "High Risk"]


# ── Utility ────────────────────────────────────────────────────────────────────

def save_fig(fig, name: str):
    for d in OUT_DIRS:
        for fmt in FMTS:
            fig.savefig(d / f"{name}.{fmt}", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {name}")


def hist_kde(ax, data, label="", color=PALETTE[0], bins=35, xlabel="", title=""):
    vals = data.dropna().values
    if len(vals) < 2:
        return
    rng = vals.max() - vals.min()
    if rng < 1e-6 * max(abs(vals.mean()), 1e-12):
        ax.text(0.5, 0.5, f"No variance\n(all ≈{vals.mean():.4g})",
                ha="center", va="center", transform=ax.transAxes, fontsize=9)
        ax.set_title(title); ax.set_xlabel(xlabel); return
    from scipy.stats import gaussian_kde
    ax.hist(vals, bins=bins, density=True, alpha=0.55, color=color, edgecolor="white", lw=0.4)
    xs = np.linspace(vals.min(), vals.max(), 300)
    kde = gaussian_kde(vals, bw_method=0.25)
    ax.plot(xs, kde(xs), color=color, lw=2)
    ax.axvline(np.mean(vals), color="k", lw=1.2, ls="--", alpha=0.7, label=f"μ={np.mean(vals):.3g}")
    ax.legend(fontsize=8)
    ax.set_title(title, fontsize=11); ax.set_xlabel(xlabel, fontsize=9)


def scatter_colored(ax, x, y, c, cmap="viridis", xlabel="", ylabel="", title="", s=12, alpha=0.6):
    sc = ax.scatter(x, y, c=c, cmap=cmap, s=s, alpha=alpha, edgecolors="none")
    ax.set_xlabel(xlabel, fontsize=9); ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=11)
    return sc


# ── A: Mission profile overview ───────────────────────────────────────────────
print("A: Mission profile overview")

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle("A — Mission Profile Overview", fontsize=14, fontweight="bold")

hist_kde(axes[0,0], df["path_length_km"], color=PALETTE[0],
         xlabel="Path length [km]", title="A1 Path Length Distribution")
hist_kde(axes[0,1], df["mission_time_min"], color=PALETTE[1],
         xlabel="Mission time [min]", title="A2 Mission Duration")
hist_kde(axes[0,2], df["cruise_speed_ms"], color=PALETTE[2],
         xlabel="Cruise speed [m/s]", title="A3 Cruise Speed")

# Phase breakdown stacked bar (binned by path length)
bins_pl = pd.cut(df["path_length_km"], bins=6)
phase_means = df.groupby(bins_pl, observed=True)[["hover_fraction","transition_time_s","cruise_fraction"]].mean()
bin_labels = [str(b) for b in phase_means.index]
x = np.arange(len(phase_means))
axes[1,0].bar(x, phase_means["hover_fraction"],     label="Hover",      color=PALETTE[1], alpha=0.85)
axes[1,0].bar(x, phase_means["cruise_fraction"],    bottom=phase_means["hover_fraction"],
              label="Cruise", color=PALETTE[0], alpha=0.85)
axes[1,0].bar(x, 1 - phase_means["hover_fraction"] - phase_means["cruise_fraction"],
              bottom=phase_means["hover_fraction"] + phase_means["cruise_fraction"],
              label="Transition", color=PALETTE[3], alpha=0.85)
axes[1,0].set_xticks(x); axes[1,0].set_xticklabels(bin_labels, rotation=25, fontsize=7)
axes[1,0].set_ylabel("Time fraction"); axes[1,0].legend(fontsize=8)
axes[1,0].set_title("A4 Flight Phase vs Path Length", fontsize=11)

hist_kde(axes[1,1], df["hover_fraction"], color=PALETTE[3],
         xlabel="Hover fraction", title="A5 Hover Time Fraction")
hist_kde(axes[1,2], df["n_waypoints"].astype(float), color=PALETTE[4],
         xlabel="Number of waypoints", title="A6 Waypoint Count")

plt.tight_layout()
save_fig(fig, "A01_mission_profile_overview")

# ── B: Rotor / BEMT performance ───────────────────────────────────────────────
print("B: Rotor / BEMT performance")

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle("B — Rotor & BEMT Performance", fontsize=14, fontweight="bold")

hist_kde(axes[0,0], df["collective_hover_deg"], color=PALETTE[0],
         xlabel="Collective [deg]", title="B1 Hover Collective Pitch")
hist_kde(axes[0,1], df["thrust_per_rotor_N"] / 1000, color=PALETTE[1],
         xlabel="Thrust per rotor [kN]", title="B2 Rotor Thrust")
hist_kde(axes[0,2], df["v_induced_hover_ms"], color=PALETTE[2],
         xlabel="Induced velocity [m/s]", title="B3 Hover Induced Velocity")

hist_kde(axes[1,0], df["power_hover_kW"], color=PALETTE[3],
         xlabel="Power [kW]", title="B4 Hover Electrical Power")
hist_kde(axes[1,1], df["CT_hover"], color=PALETTE[4],
         xlabel="C_T", title="B5 Thrust Coefficient (Hover)")

# Power vs path length colored by hover fraction
sc = scatter_colored(axes[1,2],
    df["path_length_km"], df["power_hover_kW"],
    c=df["hover_fraction"], cmap="plasma",
    xlabel="Path length [km]", ylabel="Hover power [kW]",
    title="B6 Hover Power vs Path Length")
plt.colorbar(sc, ax=axes[1,2], label="Hover fraction")

plt.tight_layout()
save_fig(fig, "B01_rotor_bemt_performance")

# ── C: Cruise aerodynamics ────────────────────────────────────────────────────
print("C: Cruise aerodynamics")

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle("C — Cruise Aerodynamics", fontsize=14, fontweight="bold")

hist_kde(axes[0,0], df["CL_cruise"], color=PALETTE[0],
         xlabel="$C_L$", title="C1 Cruise Lift Coefficient")
hist_kde(axes[0,1], df["CD_cruise"] * 1000, color=PALETTE[1],
         xlabel="$C_D$ [counts]", title="C2 Total Drag Coefficient")
hist_kde(axes[0,2], df["LD_ratio"], color=PALETTE[2],
         xlabel="L/D", title="C3 Lift-to-Drag Ratio")

# Drag polar
ax = axes[1,0]
sc = ax.scatter(df["CD_cruise"], df["CL_cruise"], c=df["cruise_speed_ms"],
                cmap="coolwarm", s=8, alpha=0.5, edgecolors="none")
ax.set_xlabel("$C_D$"); ax.set_ylabel("$C_L$")
ax.set_title("C4 Drag Polar (color = cruise speed)", fontsize=11)
plt.colorbar(sc, ax=ax, label="V [m/s]")

hist_kde(axes[1,1], df["Re_cruise"] / 1e6, color=PALETTE[3],
         xlabel="Re [×10⁶]", title="C5 Reynolds Number (cruise)")
hist_kde(axes[1,2], df["Mach_cruise"], color=PALETTE[4],
         xlabel="Mach", title="C6 Cruise Mach Number")

plt.tight_layout()
save_fig(fig, "C01_cruise_aerodynamics")

# C7: drag breakdown pie (mean)
fig, ax = plt.subplots(figsize=(6, 6))
cd_wing_mean  = df["CD_wing"].mean()
cd_fuse_mean  = df["CD_fuselage"].mean()
labels = ["Wing (induced+profile)", "Fuselage"]
vals   = [cd_wing_mean, cd_fuse_mean]
ax.pie(vals, labels=labels, autopct="%.1f%%",
       colors=[PALETTE[0], PALETTE[1]], startangle=90,
       wedgeprops={"edgecolor": "white", "linewidth": 1.2})
ax.set_title("C7 Mean Cruise Drag Breakdown", fontsize=12, fontweight="bold")
save_fig(fig, "C07_drag_breakdown_pie")

# ── D: Energy & battery ───────────────────────────────────────────────────────
print("D: Energy & battery")

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle("D — Energy & Battery", fontsize=14, fontweight="bold")

hist_kde(axes[0,0], df["energy_consumed_kWh"], color=PALETTE[0],
         xlabel="Energy consumed [kWh]", title="D1 Total Energy Consumed")
hist_kde(axes[0,1], df["soc_final"], color=PALETTE[1],
         xlabel="Final SOC", title="D2 Battery Final SOC")
hist_kde(axes[0,2], df["soc_delta"], color=PALETTE[2],
         xlabel="ΔSOC", title="D3 SOC Depletion per Mission")

ax = axes[1,0]
sc = scatter_colored(ax,
    df["path_length_km"], df["energy_consumed_kWh"],
    c=df["cruise_speed_ms"], cmap="viridis",
    xlabel="Path length [km]", ylabel="Energy [kWh]",
    title="D4 Energy vs Distance (color = speed)")
plt.colorbar(sc, ax=ax, label="Speed [m/s]")

hist_kde(axes[1,1], df["battery_peak_crate"], color=PALETTE[3],
         xlabel="Peak C-rate", title="D5 Battery Peak C-Rate")
hist_kde(axes[1,2], df["battery_V_min"], color=PALETTE[4],
         xlabel="Min terminal voltage [V]", title="D6 Minimum Bus Voltage")

plt.tight_layout()
save_fig(fig, "D01_energy_battery")

# D7: Energy vs planned estimate
fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(df["energy_cost_wh_plan"] / 1000, df["energy_consumed_kWh"],
           s=10, alpha=0.5, color=PALETTE[0], edgecolors="none")
lim_min = min(df["energy_cost_wh_plan"].min() / 1000, df["energy_consumed_kWh"].min()) * 0.95
lim_max = max(df["energy_cost_wh_plan"].max() / 1000, df["energy_consumed_kWh"].max()) * 1.05
ax.plot([lim_min, lim_max], [lim_min, lim_max], "k--", lw=1.2, alpha=0.6, label="1:1")
ax.set_xlabel("Planned energy [kWh]"); ax.set_ylabel("Simulated energy [kWh]")
ax.set_title("D7 Planned vs Simulated Energy Consumption", fontsize=12, fontweight="bold")
ax.legend()
save_fig(fig, "D07_planned_vs_simulated_energy")

# D8: SOC profile (sample 5 missions)
fig, ax = plt.subplots(figsize=(9, 5))
sample_speeds = df["cruise_speed_ms"].quantile([0.1, 0.3, 0.5, 0.7, 0.9]).values
colors_d8 = cm.viridis(np.linspace(0.1, 0.9, 5))
for spd, col in zip(sample_speeds, colors_d8):
    row = df.iloc[(df["cruise_speed_ms"] - spd).abs().argmin()]
    t_hover = row["hover_time_s"] / 2
    t_trans = row["transition_time_s"] / 2
    t_cruise = row["cruise_time_s"]
    t_total  = row["mission_time_s"]
    times = np.array([0, t_hover, t_hover + t_trans, t_hover + t_trans + t_cruise,
                      t_hover + t_trans + t_cruise + t_trans, t_total])
    socs  = np.linspace(row["soc_initial"], row["soc_final"], len(times))
    ax.plot(times / 60, socs, color=col, lw=2, alpha=0.85,
            label=f"V={row['cruise_speed_ms']:.0f} m/s")
ax.axhline(0.10, color="red", lw=1, ls="--", alpha=0.7, label="SOC_min=10%")
ax.set_xlabel("Mission time [min]"); ax.set_ylabel("State of Charge")
ax.set_title("D8 Representative SOC Profiles", fontsize=12, fontweight="bold")
ax.legend(fontsize=8)
save_fig(fig, "D08_soc_profiles")

# ── E: Thermal management ─────────────────────────────────────────────────────
print("E: Thermal management")

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle("E — Thermal Management", fontsize=14, fontweight="bold")

hist_kde(axes[0,0], df["motor_peak_temp_C"], color=PALETTE[0],
         xlabel="Motor peak temp [°C]", title="E1 Motor Peak Temperature")
hist_kde(axes[0,1], df["battery_peak_temp_C"], color=PALETTE[1],
         xlabel="Battery peak temp [°C]", title="E2 Battery Peak Temperature")
hist_kde(axes[0,2], df["thermal_margin_motor_C"], color=PALETTE[2],
         xlabel="Thermal margin [°C]", title="E3 Motor Thermal Margin")

# Motor vs battery temp correlation
ax = axes[1,0]
ax.scatter(df["motor_peak_temp_C"], df["battery_peak_temp_C"],
           c=df["hover_fraction"], cmap="plasma", s=10, alpha=0.6)
ax.set_xlabel("Motor peak temp [°C]"); ax.set_ylabel("Battery peak temp [°C]")
ax.set_title("E4 Motor vs Battery Temperature", fontsize=11)

# Temperature rise vs power
ax = axes[1,1]
ax.scatter(df["power_hover_kW"], df["motor_temp_rise_C"],
           c=df["mission_time_min"], cmap="viridis", s=10, alpha=0.6)
ax.set_xlabel("Hover power [kW]"); ax.set_ylabel("Motor temp rise [°C]")
ax.set_title("E5 Motor Rise vs Hover Power", fontsize=11)

hist_kde(axes[1,2], df["motor_temp_final_C"], color=PALETTE[4],
         xlabel="Motor final temp [°C]", title="E6 Motor Temperature at Landing")

plt.tight_layout()
save_fig(fig, "E01_thermal_management")

# ── F: Acoustic signatures ────────────────────────────────────────────────────
print("F: Acoustic signatures")

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle("F — Acoustic Signatures", fontsize=14, fontweight="bold")

hist_kde(axes[0,0], df["spl_hover_total_dB"], color=PALETTE[0],
         xlabel="SPL [dB]", title="F1 Hover SPL (total, at 30 m)")
hist_kde(axes[0,1], df["spl_cruise_total_dB"], color=PALETTE[1],
         xlabel="SPL [dB]", title="F2 Cruise SPL (total, at 30 m)")
hist_kde(axes[0,2], df["detection_range_hover_km"], color=PALETTE[2],
         xlabel="Detection range [km]", title="F3 Acoustic Detection Range (hover)")

# SPL breakdown bar chart
ax = axes[1,0]
components = ["spl_hover_loading_dB", "spl_hover_bvi_dB"]
comp_labels = ["Loading", "BVI"]
comp_means  = [df[c].mean() for c in components]
bars = ax.bar(comp_labels, comp_means, color=PALETTE[:len(comp_labels)], edgecolor="white")
ax.bar_label(bars, fmt="%.1f dB", fontsize=8, padding=2)
ax.set_ylabel("SPL [dB]"); ax.set_title("F4 Mean Hover SPL Component Breakdown", fontsize=11)

hist_kde(axes[1,1], df["bpf_hover_Hz"], color=PALETTE[3],
         xlabel="BPF [Hz]", title="F5 Blade Passage Frequency (hover)")

# Detection range vs path length
sc = scatter_colored(axes[1,2],
    df["path_length_km"], df["detection_range_cruise_km"],
    c=df["cruise_speed_ms"], cmap="coolwarm",
    xlabel="Path length [km]", ylabel="Detection range [km]",
    title="F6 Acoustic Detection Range (cruise)")
plt.colorbar(sc, ax=axes[1,2], label="Speed [m/s]")

plt.tight_layout()
save_fig(fig, "F01_acoustic_signatures")

# F7: Hover vs cruise SPL comparison (violin by risk)
fig, ax = plt.subplots(figsize=(8, 5))
positions = [1, 2, 3, 4]
data_vio = [
    df.loc[df["risk_label"]==0, "spl_hover_total_dB"].dropna().values,
    df.loc[df["risk_label"]==1, "spl_hover_total_dB"].dropna().values,
    df.loc[df["risk_label"]==0, "spl_cruise_total_dB"].dropna().values,
    df.loc[df["risk_label"]==1, "spl_cruise_total_dB"].dropna().values,
]
parts = ax.violinplot(data_vio, positions=positions, showmedians=True, showextrema=True)
for i, pc in enumerate(parts["bodies"]):
    pc.set_facecolor(PALETTE[0] if i % 2 == 0 else PALETTE[1])
    pc.set_alpha(0.75)
ax.set_xticks(positions)
ax.set_xticklabels(["Hover\nLow Risk", "Hover\nHigh Risk",
                     "Cruise\nLow Risk", "Cruise\nHigh Risk"])
ax.set_ylabel("SPL [dB]")
ax.set_title("F7 SPL Distributions: Hover vs Cruise, by Risk Label", fontsize=12, fontweight="bold")
legend_handles = [Patch(facecolor=PALETTE[0], label="Low Risk"),
                  Patch(facecolor=PALETTE[1], label="High Risk")]
ax.legend(handles=legend_handles, fontsize=9)
save_fig(fig, "F07_spl_violin_by_risk")

# ── G: Infrared signatures ────────────────────────────────────────────────────
print("G: Infrared signatures")

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle("G — Infrared Signatures", fontsize=14, fontweight="bold")

hist_kde(axes[0,0], df["ir_mwir_hover_W_sr"], color=PALETTE[0],
         xlabel="I_MWIR [W/sr]", title="G1 MWIR Intensity (hover)")
hist_kde(axes[0,1], df["ir_lwir_hover_W_sr"], color=PALETTE[1],
         xlabel="I_LWIR [W/sr]", title="G2 LWIR Intensity (hover)")
hist_kde(axes[0,2], df["ir_contrast_hover"], color=PALETTE[2],
         xlabel="Contrast ratio", title="G3 IR Contrast vs Background (hover)")

hist_kde(axes[1,0], df["ir_mwir_cruise_W_sr"], color=PALETTE[3],
         xlabel="I_MWIR [W/sr]", title="G4 MWIR Intensity (cruise)")

# MWIR vs motor temperature
ax = axes[1,1]
sc = scatter_colored(ax,
    df["motor_peak_temp_C"], df["ir_mwir_hover_W_sr"],
    c=df["hover_fraction"], cmap="hot",
    xlabel="Motor peak temp [°C]", ylabel="I_MWIR [W/sr]",
    title="G5 MWIR vs Motor Temperature")
plt.colorbar(sc, ax=ax, label="Hover fraction")

# Hover vs cruise IR
ax = axes[1,2]
ax.scatter(df["ir_mwir_hover_W_sr"], df["ir_mwir_cruise_W_sr"],
           c=df["cruise_speed_ms"], cmap="plasma", s=10, alpha=0.5)
diag_max = max(df["ir_mwir_hover_W_sr"].max(), df["ir_mwir_cruise_W_sr"].max())
ax.plot([0, diag_max], [0, diag_max], "k--", lw=1.2, alpha=0.6)
ax.set_xlabel("MWIR hover [W/sr]"); ax.set_ylabel("MWIR cruise [W/sr]")
ax.set_title("G6 MWIR: Hover vs Cruise", fontsize=11)

plt.tight_layout()
save_fig(fig, "G01_infrared_signatures")

# ── H: RCS ───────────────────────────────────────────────────────────────────
print("H: RCS")

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle("H — Radar Cross Section", fontsize=14, fontweight="bold")

hist_kde(axes[0,0], df["rcs_cruise_x_dBsm"], color=PALETTE[0],
         xlabel="RCS [dBsm]", title="H1 X-band RCS (cruise, broadside)")
hist_kde(axes[0,1], df["rcs_noseon_x_dBsm"], color=PALETTE[1],
         xlabel="RCS [dBsm]", title="H2 X-band RCS (nose-on)")
hist_kde(axes[0,2], df["rcs_hover_x_dBsm"], color=PALETTE[2],
         xlabel="RCS [dBsm]", title="H3 X-band RCS (hover, broadside)")

# Broadside vs nose-on
ax = axes[1,0]
ax.scatter(df["rcs_cruise_x_dBsm"], df["rcs_noseon_x_dBsm"],
           c=df["cruise_speed_ms"], cmap="viridis", s=10, alpha=0.5)
ax.plot([-30, 5], [-30, 5], "k--", lw=1.2, alpha=0.5)
ax.set_xlabel("Broadside X-band [dBsm]"); ax.set_ylabel("Nose-on X-band [dBsm]")
ax.set_title("H4 Broadside vs Nose-On RCS", fontsize=11)

# RCS by band (bar)
ax = axes[1,1]
rcs_x_mean  = df["rcs_cruise_x_m2"].mean()
rcs_ku_mean = df["rcs_cruise_ku_m2"].mean()
rcs_labels  = ["X-band\n(10 GHz)", "Ku-band\n(15 GHz)"]
rcs_vals    = [rcs_x_mean, rcs_ku_mean]
bars = ax.bar(rcs_labels, rcs_vals, color=[PALETTE[0], PALETTE[2]], edgecolor="white")
ax.bar_label(bars, fmt="%.3f m²", fontsize=9, padding=2)
ax.set_ylabel("Mean RCS [m²]")
ax.set_title("H5 Mean RCS by Radar Band (cruise)", fontsize=11)

# Hover vs cruise RCS
ax = axes[1,2]
ax.scatter(df["rcs_hover_x_m2"], df["rcs_cruise_x_m2"],
           c=df["hover_fraction"], cmap="plasma", s=10, alpha=0.6)
diag_val = max(df["rcs_hover_x_m2"].max(), df["rcs_cruise_x_m2"].max())
ax.plot([0, diag_val], [0, diag_val], "k--", lw=1.2, alpha=0.5)
ax.set_xlabel("Hover RCS [m²]"); ax.set_ylabel("Cruise RCS [m²]")
ax.set_title("H6 RCS: Hover vs Cruise", fontsize=11)

plt.tight_layout()
save_fig(fig, "H01_rcs_signatures")

# H7: RCS polar pattern (theoretical, computed once)
fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"polar": True})
azimuths = np.linspace(0, 2 * np.pi, 361)
# Simple analytic model for display
az_deg = np.degrees(azimuths)
ram_f  = 10 ** (-10.0 / 10.0)
rcs_cruise_pat = (0.1 * (0.3 + 0.7 * np.abs(np.sin(azimuths))) * ram_f +
                  0.5 * (1.0 - 0.9 * np.abs(np.cos(azimuths))) * ram_f)
rcs_hover_pat  = rcs_cruise_pat * 1.3   # more exposed in hover (nacelles vertical)
rcs_nose_pat   = rcs_cruise_pat * 0.3 * (0.3 + 0.7 * np.abs(np.cos(azimuths - 0)))  # nose shaping

rcs_db_cruise = 10 * np.log10(np.clip(rcs_cruise_pat, 1e-5, None))
rcs_db_hover  = 10 * np.log10(np.clip(rcs_hover_pat, 1e-5, None))

ax.plot(azimuths, rcs_db_cruise - rcs_db_cruise.min(),
        color=PALETTE[0], lw=2, label="Cruise (θ_nac=0°)")
ax.plot(azimuths, rcs_db_hover - rcs_db_hover.min(),
        color=PALETTE[1], lw=2, label="Hover (θ_nac=90°)", ls="--")
ax.set_title("H7 RCS Azimuth Pattern (relative, X-band)\n0° = nose-on",
             fontsize=11, fontweight="bold", pad=20)
ax.legend(loc="lower left", fontsize=9)
save_fig(fig, "H07_rcs_polar_pattern")

# ── I: Cross-domain correlations ──────────────────────────────────────────────
print("I: Cross-domain correlations")

corr_cols = [
    "path_length_km", "cruise_speed_ms", "hover_fraction",
    "power_hover_kW", "power_cruise_kW",
    "soc_final", "energy_consumed_kWh",
    "motor_peak_temp_C", "battery_peak_temp_C",
    "spl_hover_total_dB", "spl_cruise_total_dB",
    "detection_range_hover_km",
    "ir_mwir_hover_W_sr", "ir_contrast_hover",
    "rcs_cruise_x_dBsm", "rcs_noseon_x_dBsm",
    "CL_cruise", "LD_ratio", "figure_of_merit",
    "max_combined_threat",
]
corr_cols = [c for c in corr_cols if c in df.columns]
corr_mat = df[corr_cols].corr()

fig, ax = plt.subplots(figsize=(14, 12))
im = ax.imshow(corr_mat, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
ax.set_xticks(range(len(corr_cols))); ax.set_xticklabels(corr_cols, rotation=45, ha="right", fontsize=7)
ax.set_yticks(range(len(corr_cols))); ax.set_yticklabels(corr_cols, fontsize=7)
for i in range(len(corr_cols)):
    for j in range(len(corr_cols)):
        ax.text(j, i, f"{corr_mat.iloc[i, j]:.1f}",
                ha="center", va="center", fontsize=4.5, color="black")
plt.colorbar(im, ax=ax, label="Pearson r")
ax.set_title("I1 Vehicle Cross-Domain Correlation Matrix", fontsize=13, fontweight="bold")
plt.tight_layout()
save_fig(fig, "I01_correlation_matrix")

# I2: Selected scatter pairs
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle("I2 — Key Cross-Domain Scatter Plots", fontsize=14, fontweight="bold")

pairs = [
    ("path_length_km",    "energy_consumed_kWh",  "cruise_speed_ms",    "coolwarm",  "Path [km]", "Energy [kWh]", "Energy vs Distance"),
    ("hover_fraction",    "spl_hover_total_dB",   "power_hover_kW",     "viridis",   "Hover frac", "Hover SPL [dB]", "SPL vs Hover Fraction"),
    ("motor_peak_temp_C", "ir_mwir_hover_W_sr",   "hover_fraction",     "plasma",    "Motor T [°C]", "MWIR [W/sr]", "IR vs Motor Temp"),
    ("rcs_cruise_x_dBsm","detection_range_hover_km","max_combined_threat","RdYlGn","RCS [dBsm]","Detect range [km]","Acoustic vs RCS"),
    ("LD_ratio",          "soc_final",             "mission_time_min",   "viridis",   "L/D", "Final SOC", "Efficiency vs Endurance"),
    ("spl_hover_total_dB","rcs_hover_x_dBsm",      "hover_fraction",     "plasma",    "Hover SPL [dB]", "RCS hover [dBsm]", "Acoustic vs Radar (hover)"),
]
for ax, (xc, yc, cc, cmap, xl, yl, tl) in zip(axes.flat, pairs):
    if all(c in df.columns for c in [xc, yc, cc]):
        sc = ax.scatter(df[xc], df[yc], c=df[cc], cmap=cmap, s=10, alpha=0.6)
        plt.colorbar(sc, ax=ax, label=cc.replace("_", " "))
    ax.set_xlabel(xl, fontsize=8); ax.set_ylabel(yl, fontsize=8)
    ax.set_title(tl, fontsize=10)

plt.tight_layout()
save_fig(fig, "I02_cross_domain_scatter")

# ── J: Comparative by risk label ─────────────────────────────────────────────
print("J: Comparative by risk label")

fig, axes = plt.subplots(2, 4, figsize=(18, 8))
fig.suptitle("J — Vehicle Performance by Risk Label", fontsize=14, fontweight="bold")

box_cols = [
    ("energy_consumed_kWh",   "Energy [kWh]",  "J1 Energy Consumed"),
    ("soc_final",             "Final SOC",      "J2 Final SOC"),
    ("spl_hover_total_dB",    "SPL [dB]",       "J3 Hover SPL"),
    ("detection_range_hover_km", "Det. range [km]", "J4 Acoustic Detection"),
    ("rcs_cruise_x_dBsm",    "RCS [dBsm]",     "J5 Cruise RCS"),
    ("motor_peak_temp_C",     "T [°C]",         "J6 Motor Peak Temp"),
    ("ir_mwir_hover_W_sr",   "MWIR [W/sr]",    "J7 MWIR (hover)"),
    ("LD_ratio",              "L/D",             "J8 L/D Ratio"),
]
for ax, (col, ylabel, title) in zip(axes.flat, box_cols):
    if col not in df.columns:
        continue
    data_box = [df.loc[df["risk_label"]==0, col].dropna().values,
                df.loc[df["risk_label"]==1, col].dropna().values]
    bp = ax.boxplot(data_box, patch_artist=True,
                    medianprops={"color": "black", "lw": 2},
                    whiskerprops={"lw": 1.2},
                    capprops={"lw": 1.2})
    for patch, color in zip(bp["boxes"], [PALETTE[0], PALETTE[1]]):
        patch.set_facecolor(color); patch.set_alpha(0.75)
    ax.set_xticks([1, 2]); ax.set_xticklabels(["Low Risk", "High Risk"], fontsize=9)
    ax.set_ylabel(ylabel, fontsize=8); ax.set_title(title, fontsize=10)

plt.tight_layout()
save_fig(fig, "J01_comparative_by_risk")

# ── K: Operational summary ────────────────────────────────────────────────────
print("K: Operational summary")

# K1: Multi-signature Pareto (SPL vs RCS vs IR)
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")
sc = ax.scatter(
    df["spl_hover_total_dB"],
    df["rcs_cruise_x_dBsm"],
    df["ir_mwir_hover_W_sr"],
    c=df["energy_consumed_kWh"], cmap="plasma", s=12, alpha=0.6
)
ax.set_xlabel("Hover SPL [dB]", fontsize=9)
ax.set_ylabel("Cruise RCS [dBsm]", fontsize=9)
ax.set_zlabel("MWIR [W/sr]", fontsize=9)
ax.set_title("K1 Multi-Signature Pareto Space\n(color = energy consumed [kWh])",
             fontsize=12, fontweight="bold")
plt.colorbar(sc, ax=ax, pad=0.1, label="Energy [kWh]")
save_fig(fig, "K01_multi_signature_pareto_3d")

# K2: Radar chart — mean vehicle performance profile
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})
categories = [
    "Endurance\n(SOC final)",
    "Aero\nefficiency\n(L/D norm)",
    "Acoustic\nstealth\n(low SPL)",
    "IR stealth\n(low MWIR)",
    "Radar\nstealth\n(low RCS)",
    "Thermal\nmargin",
    "Energy\nefficiency",
]
n_cat = len(categories)
angles = np.linspace(0, 2 * np.pi, n_cat, endpoint=False).tolist()
angles += angles[:1]

def norm01(x, lo, hi):
    return float(np.clip((x - lo) / max(hi - lo, 1e-9), 0, 1))

risk0 = df[df["risk_label"] == 0]
risk1 = df[df["risk_label"] == 1]

for group, color, label in [(risk0, PALETTE[0], "Low Risk"), (risk1, PALETTE[1], "High Risk")]:
    vals = [
        group["soc_final"].mean(),                               # high is good
        norm01(group["LD_ratio"].mean(), 0, 30),
        1 - norm01(group["spl_hover_total_dB"].mean(), 60, 110),  # low SPL = good stealth
        1 - norm01(group["ir_mwir_hover_W_sr"].mean(), 0, 20),
        1 - norm01(group["rcs_cruise_x_dBsm"].mean(), -20, 10),
        norm01(group["thermal_margin_motor_C"].mean(), 0, 130),
        1 - norm01(group["energy_consumed_kWh"].mean(), 0, 20),
    ]
    vals += vals[:1]
    ax.plot(angles, vals, color=color, lw=2, label=label)
    ax.fill(angles, vals, color=color, alpha=0.18)

ax.set_thetagrids(np.degrees(angles[:-1]), categories, fontsize=9)
ax.set_ylim(0, 1)
ax.set_title("K2 Vehicle Performance Radar Chart\n(normalised 0→1, higher = better)",
             fontsize=12, fontweight="bold", pad=30)
ax.legend(loc="lower left", bbox_to_anchor=(0.0, -0.15), fontsize=10)
save_fig(fig, "K02_performance_radar_chart")

# K3: Mission feasibility / energy matrix
fig, ax = plt.subplots(figsize=(9, 6))
scatter_data = ax.scatter(
    df["path_length_km"],
    df["energy_consumed_kWh"],
    c=df["max_combined_threat"],
    cmap="RdYlGn_r",
    s=df["soc_final"] * 60,
    alpha=0.65,
    edgecolors="none",
)
plt.colorbar(scatter_data, ax=ax, label="Max combined threat")
ax.axhline(CAP_WH_PACK := 40.0, color="red", ls="--", lw=1.5,
           label=f"Pack capacity ({CAP_WH_PACK} kWh)")
ax.set_xlabel("Path length [km]"); ax.set_ylabel("Energy consumed [kWh]")
ax.set_title("K3 Mission Energy Envelope\n(size = final SOC, color = threat level)",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=9)
save_fig(fig, "K03_mission_energy_envelope")

# K4: Signature tradeoff — hovering vs cruise
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("K4 — Signature Budget: Hover vs Cruise", fontsize=13, fontweight="bold")

sig_hover  = [df["spl_hover_total_dB"].mean(),
              df["ir_mwir_hover_W_sr"].mean() * 3,   # scaled for viz
              df["rcs_hover_x_m2"].mean() * 100]
sig_cruise = [df["spl_cruise_total_dB"].mean(),
              df["ir_mwir_cruise_W_sr"].mean() * 3,
              df["rcs_cruise_x_m2"].mean() * 100]
sig_labels = ["Acoustic SPL [dB]", "MWIR ×3 [W/sr]", "RCS ×100 [m²]"]
x = np.arange(len(sig_labels))
w = 0.35
axes[0].bar(x - w/2, sig_hover,  w, label="Hover",  color=PALETTE[1], alpha=0.85)
axes[0].bar(x + w/2, sig_cruise, w, label="Cruise", color=PALETTE[0], alpha=0.85)
axes[0].set_xticks(x); axes[0].set_xticklabels(sig_labels, fontsize=9)
axes[0].legend(); axes[0].set_ylabel("Signature level (varied units)")
axes[0].set_title("Absolute Signature Budget")

# Relative ratio hover/cruise
ratio = np.array(sig_hover) / np.clip(sig_cruise, 1e-9, None)
axes[1].bar(sig_labels, ratio - 1.0, color=[PALETTE[1] if r > 1 else PALETTE[0] for r in ratio], alpha=0.85)
axes[1].axhline(0, color="black", lw=1.2)
axes[1].set_ylabel("(Hover / Cruise) − 1   (positive = hover worse)")
axes[1].set_title("Hover vs Cruise Signature Ratio")

plt.tight_layout()
save_fig(fig, "K04_signature_budget_comparison")

print(f"\nAll figures written to visuals/vehicle/ and outputs/visuals/vehicle/")
