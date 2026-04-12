"""
Control Layer Dataset Visualizer

Generates publication-quality figures from outputs/control/control_dataset.parquet.
All figures: 300 DPI, PNG + PDF
Output directory: visuals/control/

Figure categories:
  A  Tracking performance (position, velocity, altitude error distributions)
  B  Control effort (thrust commands, moments, saturation)
  C  Mode transitions (phase fractions, settling times)
  D  Motor allocation (per-motor thrust, balance, PWM)
  E  Risk label comparison (tracking errors by risk label, box plots)
  F  ITAE quality metrics
  G  Mission success / abort analysis
  H  Cross-correlations (energy vs tracking, speed vs settling)
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch
from pathlib import Path
try:
    from scipy.stats import gaussian_kde
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).resolve().parent.parent.parent
SRC     = ROOT / "outputs" / "control" / "control_dataset.parquet"
OUT_DIR = ROOT / "visuals" / "control"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DPI  = 300
FMTS = ["png", "pdf"]   # NO SVG

plt.style.use("seaborn-v0_8-whitegrid")
PALETTE = ["#2196F3", "#F44336", "#4CAF50", "#FF9800", "#9C27B0", "#00BCD4",
           "#795548", "#607D8B"]
RISK_COLORS = {0: PALETTE[0], 1: PALETTE[1]}
RISK_LABELS = {0: "Low Risk", 1: "High Risk"}

# ── Load ───────────────────────────────────────────────────────────────────────
print(f"Loading {SRC}")
df = pd.read_parquet(SRC)
print(f"  {len(df)} rows × {len(df.columns)} columns")

# Derived
df["risk_str"]        = df["risk_label"].map(RISK_LABELS)
df["pos_error_km"]    = df["pos_error_mean_m"] / 1000.0
df["mission_time_min"]= df["mission_time_s"] / 60.0
df["thrust_frac"]     = df["thrust_cmd_mean_N"] / 4800.0
df["abort_str"]       = df["mission_abort"].map({0: "Success", 1: "Abort"})


# ── Helpers ────────────────────────────────────────────────────────────────────

def save_fig(fig, name: str):
    for fmt in FMTS:
        fig.savefig(OUT_DIR / f"{name}.{fmt}", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {name}")


def hist_kde(ax, data, color=PALETTE[0], bins=35, xlabel="", title="", label=""):
    vals = data.dropna().values.astype(float)
    if len(vals) < 2:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title); ax.set_xlabel(xlabel); return
    rng = vals.max() - vals.min()
    if rng < 1e-9:
        ax.axvline(vals.mean(), color=color, lw=2, label=f"all={vals.mean():.4g}")
        ax.set_title(title); ax.set_xlabel(xlabel); ax.legend(fontsize=8); return
    ax.hist(vals, bins=bins, density=True, alpha=0.5, color=color, edgecolor="white", lw=0.3)
    if HAS_SCIPY and len(vals) >= 5:
        xs = np.linspace(vals.min(), vals.max(), 300)
        kde = gaussian_kde(vals, bw_method=0.25)
        ax.plot(xs, kde(xs), color=color, lw=2, label=label)
    ax.axvline(np.mean(vals), color="k", lw=1.2, ls="--", alpha=0.7,
               label=f"μ={np.mean(vals):.3g}")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.legend(fontsize=8)


def boxplot_by_risk(ax, col, ylabel="", title=""):
    groups = [df.loc[df["risk_label"] == k, col].dropna().values for k in [0, 1]]
    bp = ax.boxplot(groups, patch_artist=True, widths=0.5,
                    medianprops=dict(color="black", lw=2))
    colors = [PALETTE[0], PALETTE[1]]
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c); patch.set_alpha(0.7)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Low Risk", "High Risk"], fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=11)


# ═══════════════════════════════════════════════════════════════════════════════
# A: Tracking performance
# ═══════════════════════════════════════════════════════════════════════════════
print("A: Tracking performance")
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle("A — Tracking Performance Distributions", fontsize=14, fontweight="bold")

hist_kde(axes[0,0], df["pos_error_mean_m"],  color=PALETTE[0],
         xlabel="Position error mean [m]",     title="A1 Mean Position Error")
hist_kde(axes[0,1], df["pos_error_rms_m"],   color=PALETTE[1],
         xlabel="Position error RMS [m]",      title="A2 RMS Position Error")
hist_kde(axes[0,2], df["pos_error_max_m"],   color=PALETTE[2],
         xlabel="Position error max [m]",      title="A3 Max Position Error")

hist_kde(axes[1,0], df["vel_error_mean_ms"], color=PALETTE[3],
         xlabel="Velocity error mean [m/s]",   title="A4 Mean Velocity Error")
hist_kde(axes[1,1], df["alt_error_mean_m"],  color=PALETTE[4],
         xlabel="Altitude error mean [m]",     title="A5 Mean Altitude Error")
hist_kde(axes[1,2], df["att_error_mean_rad"], color=PALETTE[5],
         xlabel="Attitude error mean [rad]",   title="A6 Mean Attitude Error")

plt.tight_layout()
save_fig(fig, "A01_tracking_performance_distributions")

# A2: Tracking errors over mission time scatter
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("A2 — Tracking Error vs Mission Duration", fontsize=13, fontweight="bold")
colors_risk = df["risk_label"].map(RISK_COLORS)

for ax, (col, lbl) in zip(axes, [
    ("pos_error_rms_m",   "Position RMS error [m]"),
    ("alt_error_rms_m",   "Altitude RMS error [m]"),
    ("att_error_rms_rad", "Attitude RMS error [rad]"),
]):
    sc = ax.scatter(df["mission_time_min"], df[col],
                    c=df["risk_label"], cmap="coolwarm", s=8, alpha=0.5)
    ax.set_xlabel("Mission time [min]", fontsize=9)
    ax.set_ylabel(lbl, fontsize=9)
    ax.set_title(col.replace("_", " ").title(), fontsize=10)

legend_elements = [Patch(facecolor=PALETTE[0], label="Low Risk"),
                   Patch(facecolor=PALETTE[1], label="High Risk")]
axes[-1].legend(handles=legend_elements, fontsize=8)
plt.tight_layout()
save_fig(fig, "A02_tracking_vs_mission_time")


# ═══════════════════════════════════════════════════════════════════════════════
# B: Control effort
# ═══════════════════════════════════════════════════════════════════════════════
print("B: Control effort")
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle("B — Control Effort Distributions", fontsize=14, fontweight="bold")

hist_kde(axes[0,0], df["thrust_cmd_mean_N"], color=PALETTE[0],
         xlabel="Thrust command mean [N]",     title="B1 Mean Thrust Command")
hist_kde(axes[0,1], df["thrust_cmd_max_N"],  color=PALETTE[1],
         xlabel="Thrust command max [N]",      title="B2 Max Thrust Command")
hist_kde(axes[0,2], df["thrust_cmd_std_N"],  color=PALETTE[2],
         xlabel="Thrust std [N]",             title="B3 Thrust Variability")

hist_kde(axes[1,0], df["moment_x_std_Nm"],  color=PALETTE[3],
         xlabel="Roll moment std [N·m]",      title="B4 Roll Moment Std")
hist_kde(axes[1,1], df["moment_y_std_Nm"],  color=PALETTE[4],
         xlabel="Pitch moment std [N·m]",     title="B5 Pitch Moment Std")
hist_kde(axes[1,2], df["moment_z_std_Nm"],  color=PALETTE[5],
         xlabel="Yaw moment std [N·m]",       title="B6 Yaw Moment Std")

plt.tight_layout()
save_fig(fig, "B01_control_effort_distributions")

# B2: Saturation analysis
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("B2 — Saturation & Attitude Command Analysis", fontsize=13, fontweight="bold")

hist_kde(axes[0], df["n_saturations"].astype(float), color=PALETTE[0],
         xlabel="Saturation count", title="B7 Control Saturations")
hist_kde(axes[1], df["roll_cmd_max_rad"] * 180/np.pi, color=PALETTE[1],
         xlabel="Max roll command [deg]", title="B8 Max Roll Command")
hist_kde(axes[2], df["pitch_cmd_max_rad"] * 180/np.pi, color=PALETTE[2],
         xlabel="Max pitch command [deg]", title="B9 Max Pitch Command")

plt.tight_layout()
save_fig(fig, "B02_saturation_attitude_commands")


# ═══════════════════════════════════════════════════════════════════════════════
# C: Mode transitions
# ═══════════════════════════════════════════════════════════════════════════════
print("C: Mode transitions")
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle("C — Mode Transitions & Phase Analysis", fontsize=14, fontweight="bold")

# Phase fraction stacked bars (binned by mission time)
bins_t = pd.cut(df["mission_time_min"], bins=6)
phase_means = df.groupby(bins_t, observed=True)[
    ["hover_frac_ctrl", "transition_frac_ctrl", "cruise_frac_ctrl"]
].mean()
bin_lbl = [str(b) for b in phase_means.index]
x = np.arange(len(phase_means))
axes[0,0].bar(x, phase_means["hover_frac_ctrl"],       label="Hover",      color=PALETTE[1], alpha=0.85)
axes[0,0].bar(x, phase_means["cruise_frac_ctrl"],
              bottom=phase_means["hover_frac_ctrl"],   label="Cruise",     color=PALETTE[0], alpha=0.85)
axes[0,0].bar(x, phase_means["transition_frac_ctrl"],
              bottom=phase_means["hover_frac_ctrl"] + phase_means["cruise_frac_ctrl"],
              label="Transition", color=PALETTE[3], alpha=0.85)
axes[0,0].set_xticks(x); axes[0,0].set_xticklabels(bin_lbl, rotation=30, fontsize=7)
axes[0,0].set_ylabel("Time fraction"); axes[0,0].legend(fontsize=8)
axes[0,0].set_title("C1 Phase Fractions vs Mission Duration", fontsize=10)

hist_kde(axes[0,1], df["hover_frac_ctrl"],      color=PALETTE[1],
         xlabel="Hover fraction",               title="C2 Hover Fraction Distribution")
hist_kde(axes[0,2], df["cruise_frac_ctrl"],     color=PALETTE[0],
         xlabel="Cruise fraction",              title="C3 Cruise Fraction Distribution")

hist_kde(axes[1,0], df["alt_settling_mean_s"], color=PALETTE[2],
         xlabel="Mean settling time [s]",       title="C4 Mean Settling Time")
hist_kde(axes[1,1], df["alt_settling_max_s"],  color=PALETTE[4],
         xlabel="Max settling time [s]",        title="C5 Max Settling Time")
hist_kde(axes[1,2], df["n_mode_transitions"].astype(float), color=PALETTE[5],
         xlabel="Mode transitions count",       title="C6 Mode Transition Count")

plt.tight_layout()
save_fig(fig, "C01_mode_transitions_phases")


# ═══════════════════════════════════════════════════════════════════════════════
# D: Motor allocation
# ═══════════════════════════════════════════════════════════════════════════════
print("D: Motor allocation")
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle("D — Motor Allocation & PWM Utilization", fontsize=14, fontweight="bold")

# Per-motor mean thrust
motor_cols = ["motor_T_m0_mean_N", "motor_T_m1_mean_N",
              "motor_T_m2_mean_N", "motor_T_m3_mean_N"]
motor_means = [df[c].mean() for c in motor_cols]
motor_stds  = [df[c].std() for c in motor_cols]
axes[0,0].bar(["FL","FR","RL","RR"], motor_means, yerr=motor_stds,
              color=PALETTE[:4], alpha=0.8, capsize=5)
axes[0,0].set_ylabel("Mean thrust [N]")
axes[0,0].set_title("D1 Per-Motor Mean Thrust", fontsize=11)

hist_kde(axes[0,1], df["motor_T_balance_N"], color=PALETTE[4],
         xlabel="Thrust balance std [N]",    title="D2 Motor Balance (Std)")
hist_kde(axes[0,2], df["motor_T_mean_N"],    color=PALETTE[0],
         xlabel="Mean total motor thrust [N]", title="D3 Total Motor Thrust")

hist_kde(axes[1,0], df["pwm_mean_us"],         color=PALETTE[1],
         xlabel="PWM mean [µs]",              title="D4 Mean PWM Signal")
hist_kde(axes[1,1], df["pwm_utilisation_pct"], color=PALETTE[2],
         xlabel="PWM utilisation [%]",        title="D5 PWM Utilization")

# Scatter: motor balance vs position error
axes[1,2].scatter(df["motor_T_balance_N"], df["pos_error_rms_m"],
                  c=df["risk_label"], cmap="coolwarm", s=8, alpha=0.5)
axes[1,2].set_xlabel("Motor balance std [N]", fontsize=9)
axes[1,2].set_ylabel("Position RMS error [m]", fontsize=9)
axes[1,2].set_title("D6 Balance vs Tracking", fontsize=11)

plt.tight_layout()
save_fig(fig, "D01_motor_allocation_pwm")

# D2: Nacelle analysis
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("D2 — Nacelle Transition Analysis", fontsize=13, fontweight="bold")
hist_kde(axes[0], df["nacelle_mean_deg"],       color=PALETTE[0],
         xlabel="Mean nacelle angle [deg]",     title="D7 Mean Nacelle Angle")
hist_kde(axes[1], df["nacelle_transitions_n"].astype(float), color=PALETTE[1],
         xlabel="Nacelle transitions count",    title="D8 Nacelle Transition Count")
hist_kde(axes[2], df["nacelle_final_deg"],      color=PALETTE[2],
         xlabel="Final nacelle angle [deg]",    title="D9 Final Nacelle Angle")
plt.tight_layout()
save_fig(fig, "D02_nacelle_analysis")


# ═══════════════════════════════════════════════════════════════════════════════
# E: Risk label comparison (box plots)
# ═══════════════════════════════════════════════════════════════════════════════
print("E: Risk label comparison")
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle("E — Control Performance by Risk Label", fontsize=14, fontweight="bold")

boxplot_by_risk(axes[0,0], "pos_error_rms_m",   "Position RMS error [m]",  "E1 Pos Error by Risk")
boxplot_by_risk(axes[0,1], "alt_error_rms_m",   "Altitude RMS error [m]",  "E2 Alt Error by Risk")
boxplot_by_risk(axes[0,2], "att_error_rms_rad",  "Attitude RMS error [rad]","E3 Att Error by Risk")
boxplot_by_risk(axes[1,0], "n_saturations",      "Saturation count",        "E4 Saturations by Risk")
boxplot_by_risk(axes[1,1], "alt_settling_mean_s","Settling time [s]",      "E5 Settling by Risk")
boxplot_by_risk(axes[1,2], "itae_pos",           "ITAE position",           "E6 ITAE Pos by Risk")

plt.tight_layout()
save_fig(fig, "E01_performance_by_risk")

# E2: Violin plots
fig, axes = plt.subplots(1, 3, figsize=(15, 6))
fig.suptitle("E2 — Violin Plots by Risk Label", fontsize=13, fontweight="bold")

for ax, (col, lbl) in zip(axes, [
    ("pos_error_mean_m",   "Pos error mean [m]"),
    ("vel_error_mean_ms",  "Vel error mean [m/s]"),
    ("thrust_cmd_mean_N",  "Thrust cmd mean [N]"),
]):
    groups = [df.loc[df["risk_label"] == k, col].dropna().values for k in [0, 1]]
    parts = ax.violinplot(groups, positions=[1, 2], showmedians=True, showmeans=True)
    for i, body in enumerate(parts["bodies"]):
        body.set_facecolor([PALETTE[0], PALETTE[1]][i])
        body.set_alpha(0.6)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Low Risk", "High Risk"], fontsize=9)
    ax.set_ylabel(lbl, fontsize=9)
    ax.set_title(col.replace("_", " ").title(), fontsize=10)

plt.tight_layout()
save_fig(fig, "E02_violin_by_risk")


# ═══════════════════════════════════════════════════════════════════════════════
# F: ITAE quality metrics
# ═══════════════════════════════════════════════════════════════════════════════
print("F: ITAE metrics")
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle("F — ITAE Quality Metrics", fontsize=14, fontweight="bold")

hist_kde(axes[0,0], df["itae_pos"], color=PALETTE[0],
         xlabel="ITAE position", title="F1 ITAE Position Error")
hist_kde(axes[0,1], df["itae_alt"], color=PALETTE[1],
         xlabel="ITAE altitude", title="F2 ITAE Altitude Error")
hist_kde(axes[0,2], df["itae_att"], color=PALETTE[2],
         xlabel="ITAE attitude", title="F3 ITAE Attitude Error")

# ITAE vs mission time
axes[1,0].scatter(df["mission_time_min"], df["itae_pos"],
                  c=df["risk_label"], cmap="coolwarm", s=8, alpha=0.5)
axes[1,0].set_xlabel("Mission time [min]", fontsize=9)
axes[1,0].set_ylabel("ITAE position", fontsize=9)
axes[1,0].set_title("F4 ITAE Pos vs Mission Time", fontsize=11)

axes[1,1].scatter(df["itae_pos"], df["itae_alt"],
                  c=df["risk_label"], cmap="coolwarm", s=8, alpha=0.5)
axes[1,1].set_xlabel("ITAE position", fontsize=9)
axes[1,1].set_ylabel("ITAE altitude", fontsize=9)
axes[1,1].set_title("F5 ITAE Pos vs Alt", fontsize=11)

# ITAE by risk box
boxplot_by_risk(axes[1,2], "itae_alt", "ITAE altitude", "F6 ITAE Alt by Risk")

plt.tight_layout()
save_fig(fig, "F01_itae_metrics")


# ═══════════════════════════════════════════════════════════════════════════════
# G: Mission success / abort analysis
# ═══════════════════════════════════════════════════════════════════════════════
print("G: Mission success/abort")
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle("G — Mission Success & Abort Analysis", fontsize=14, fontweight="bold")

# G1: Abort rate pie
abort_counts = df["mission_abort"].value_counts()
success_n = abort_counts.get(0, 0)
abort_n   = abort_counts.get(1, 0)
axes[0,0].pie([success_n, abort_n],
              labels=["Success", "Abort"],
              colors=[PALETTE[0], PALETTE[1]],
              autopct="%1.1f%%", startangle=90)
axes[0,0].set_title("G1 Mission Outcome", fontsize=11)

# G2: Abort by risk
abort_by_risk = df.groupby("risk_str")["mission_abort"].mean() * 100
axes[0,1].bar(abort_by_risk.index, abort_by_risk.values,
              color=[PALETTE[0], PALETTE[1]], alpha=0.8)
axes[0,1].set_ylabel("Abort rate [%]")
axes[0,1].set_title("G2 Abort Rate by Risk Label", fontsize=11)

# G3: Waypoints reached distribution
hist_kde(axes[0,2], df["n_wp_reached"].astype(float), color=PALETTE[2],
         xlabel="Waypoints reached", title="G3 Waypoints Reached")

# G4: Stall events
hist_kde(axes[1,0], df["n_stall_events"].astype(float), color=PALETTE[3],
         xlabel="Stall event count", title="G4 Stall Events")

# G5: Tracking error for success vs abort
for outcome, color, lbl in [(0, PALETTE[0], "Success"), (1, PALETTE[1], "Abort")]:
    subset = df.loc[df["mission_abort"] == outcome, "pos_error_rms_m"].dropna().values
    if len(subset) > 1:
        axes[1,1].hist(subset, bins=30, alpha=0.5, density=True, color=color, label=lbl)
axes[1,1].set_xlabel("Position RMS error [m]", fontsize=9)
axes[1,1].set_title("G5 Pos Error: Success vs Abort", fontsize=11)
axes[1,1].legend(fontsize=8)

# G6: Mission success rate vs cruise speed
bins_sp = pd.cut(df["cruise_speed_ref_ms"], bins=6)
success_by_speed = (1 - df.groupby(bins_sp, observed=True)["mission_abort"].mean()) * 100
x_lbl = [str(b) for b in success_by_speed.index]
x = np.arange(len(success_by_speed))
axes[1,2].bar(x, success_by_speed.values, color=PALETTE[0], alpha=0.8)
axes[1,2].set_xticks(x); axes[1,2].set_xticklabels(x_lbl, rotation=30, fontsize=7)
axes[1,2].set_ylabel("Success rate [%]")
axes[1,2].set_title("G6 Success Rate vs Cruise Speed", fontsize=11)
axes[1,2].set_ylim(0, 105)

plt.tight_layout()
save_fig(fig, "G01_mission_success_abort")


# ═══════════════════════════════════════════════════════════════════════════════
# H: Cross-correlations
# ═══════════════════════════════════════════════════════════════════════════════
print("H: Cross-correlations")
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle("H — Cross-Correlations: Energy, Tracking & Speed", fontsize=14, fontweight="bold")

# H1: Energy vs position error
axes[0,0].scatter(df["energy_consumed_wh"] / 1000, df["pos_error_rms_m"],
                  c=df["risk_label"], cmap="coolwarm", s=8, alpha=0.5)
axes[0,0].set_xlabel("Energy consumed [kWh]", fontsize=9)
axes[0,0].set_ylabel("Position RMS error [m]", fontsize=9)
axes[0,0].set_title("H1 Energy vs Position Error", fontsize=11)

# H2: Cruise speed vs settling time
axes[0,1].scatter(df["cruise_speed_ref_ms"], df["alt_settling_mean_s"],
                  c=df["risk_label"], cmap="coolwarm", s=8, alpha=0.5)
axes[0,1].set_xlabel("Cruise speed [m/s]", fontsize=9)
axes[0,1].set_ylabel("Mean settling time [s]", fontsize=9)
axes[0,1].set_title("H2 Speed vs Settling Time", fontsize=11)

# H3: Path length vs ITAE
axes[0,2].scatter(df["path_length_m"] / 1000, df["itae_pos"],
                  c=df["risk_label"], cmap="coolwarm", s=8, alpha=0.5)
axes[0,2].set_xlabel("Path length [km]", fontsize=9)
axes[0,2].set_ylabel("ITAE position", fontsize=9)
axes[0,2].set_title("H3 Path Length vs ITAE Pos", fontsize=11)

# H4: Thrust fraction vs altitude error
axes[1,0].scatter(df["thrust_frac"], df["alt_error_rms_m"],
                  c=df["risk_label"], cmap="coolwarm", s=8, alpha=0.5)
axes[1,0].set_xlabel("Thrust fraction", fontsize=9)
axes[1,0].set_ylabel("Altitude RMS error [m]", fontsize=9)
axes[1,0].set_title("H4 Thrust Load vs Alt Error", fontsize=11)

# H5: Saturations vs position error
axes[1,1].scatter(df["n_saturations"], df["pos_error_rms_m"],
                  c=df["risk_label"], cmap="coolwarm", s=8, alpha=0.5)
axes[1,1].set_xlabel("Saturation count", fontsize=9)
axes[1,1].set_ylabel("Position RMS error [m]", fontsize=9)
axes[1,1].set_title("H5 Saturations vs Tracking", fontsize=11)

# H6: Correlation heatmap (select numeric columns)
corr_cols = ["pos_error_rms_m", "alt_error_rms_m", "att_error_rms_rad",
             "thrust_cmd_mean_N", "n_saturations", "alt_settling_mean_s",
             "itae_pos", "itae_alt", "pwm_utilisation_pct", "energy_consumed_wh"]
corr_cols = [c for c in corr_cols if c in df.columns]
corr_matrix = df[corr_cols].corr()
im = axes[1,2].imshow(corr_matrix.values, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
axes[1,2].set_xticks(range(len(corr_cols)))
axes[1,2].set_yticks(range(len(corr_cols)))
short_labels = [c.split("_")[0] + "_" + c.split("_")[-1] for c in corr_cols]
axes[1,2].set_xticklabels(short_labels, rotation=45, ha="right", fontsize=7)
axes[1,2].set_yticklabels(short_labels, fontsize=7)
plt.colorbar(im, ax=axes[1,2], shrink=0.8)
axes[1,2].set_title("H6 Correlation Heatmap", fontsize=11)

plt.tight_layout()
save_fig(fig, "H01_cross_correlations")

# H2: Additional cross-correlation heatmap (larger)
fig, ax = plt.subplots(figsize=(10, 8))
fig.suptitle("H2 — Extended Correlation Heatmap", fontsize=13, fontweight="bold")
extended_cols = [
    "pos_error_mean_m", "pos_error_rms_m", "vel_error_mean_ms", "alt_error_mean_m",
    "att_error_mean_rad", "thrust_cmd_mean_N", "thrust_cmd_std_N",
    "moment_x_std_Nm", "moment_y_std_Nm", "n_saturations",
    "alt_settling_mean_s", "itae_pos", "itae_alt", "itae_att",
    "pwm_utilisation_pct", "motor_T_balance_N", "mission_time_s", "cruise_speed_ref_ms"
]
extended_cols = [c for c in extended_cols if c in df.columns]
corr_ext = df[extended_cols].corr()
im2 = ax.imshow(corr_ext.values, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
ax.set_xticks(range(len(extended_cols)))
ax.set_yticks(range(len(extended_cols)))
short = [c[:14] for c in extended_cols]
ax.set_xticklabels(short, rotation=45, ha="right", fontsize=7)
ax.set_yticklabels(short, fontsize=7)
plt.colorbar(im2, ax=ax, shrink=0.85)
plt.tight_layout()
save_fig(fig, "H02_extended_correlation_heatmap")


# ═══════════════════════════════════════════════════════════════════════════════
# Summary figure: Control KPI overview
# ═══════════════════════════════════════════════════════════════════════════════
print("Summary: Control KPI overview")
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
fig.suptitle("Control Layer — Key Performance Indicators Summary", fontsize=14, fontweight="bold")

kpis = [
    ("pos_error_rms_m",     "Position RMS error [m]",   PALETTE[0]),
    ("alt_error_rms_m",     "Altitude RMS error [m]",   PALETTE[1]),
    ("att_error_rms_rad",   "Attitude RMS error [rad]", PALETTE[2]),
    ("vel_error_rms_ms",    "Velocity RMS error [m/s]", PALETTE[3]),
    ("thrust_cmd_mean_N",   "Mean thrust cmd [N]",       PALETTE[4]),
    ("pwm_utilisation_pct", "PWM utilisation [%]",       PALETTE[5]),
    ("alt_settling_mean_s","Mean settling time [s]",    PALETTE[6]),
    ("itae_pos",            "ITAE position",             PALETTE[7]),
]

for ax, (col, lbl, color) in zip(axes.flat, kpis):
    hist_kde(ax, df[col], color=color, xlabel=lbl, title=lbl, bins=30)

plt.tight_layout()
save_fig(fig, "Z01_control_kpi_summary")

# ═══════════════════════════════════════════════════════════════════════════════
# PID integral state distributions
# ═══════════════════════════════════════════════════════════════════════════════
print("PID integral states")
fig, axes = plt.subplots(1, 4, figsize=(16, 5))
fig.suptitle("PID Integral States at Mission End", fontsize=13, fontweight="bold")

pid_cols = [
    ("pid_int_vel_final",  "Velocity integral final",  PALETTE[0]),
    ("pid_int_alt_final",  "Altitude integral final",  PALETTE[1]),
    ("pid_int_att_final",  "Attitude integral final",  PALETTE[2]),
    ("pid_int_rate_final", "Rate integral final",      PALETTE[3]),
]
for ax, (col, lbl, color) in zip(axes, pid_cols):
    hist_kde(ax, df[col], color=color, xlabel=lbl, title=lbl, bins=30)

plt.tight_layout()
save_fig(fig, "Z02_pid_integral_states")


# ═══════════════════════════════════════════════════════════════════════════════
# Final report
# ═══════════════════════════════════════════════════════════════════════════════
saved_files = sorted(OUT_DIR.glob("*.png"))
print(f"\n-- Visualization Complete --------------------------------------------------")
print(f"  Output directory: {OUT_DIR}")
print(f"  Files saved: {len(saved_files)} PNG + {len(saved_files)} PDF = {len(saved_files)*2} total")
for f in saved_files:
    print(f"  {f.name}")
