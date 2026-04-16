"""
Single-Mission Validation
=========================
Runs one representative mission through the control simulator with full
time-series logging and saves trajectory validation plots to outputs/control/.
"""
from __future__ import annotations
import math, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts" / "control"))
from dataset import (
    simulate_mission, build_phase_schedule,
    DT, MASS_KG, WEIGHT_N, MAX_THRUST_N, MIN_THRUST_N, T_PER,
    L_ARM, IXX, IYY, IZZ, G,
    KP_ALT, KI_ALT, KD_ALT, KP_CR, KI_CR, KD_CR,
    KP_VEL, KI_VEL, KD_VEL, KP_ATT, KI_ATT, KD_ATT,
    KP_RT,  KI_RT,  KD_RT,  KP_POS, KI_POS, KD_POS,
    CR_MAX, VEL_CMD_MAX, ATT_MAX, RATE_MAX,
    B_PINV, PWM_MIN, PWM_RANGE,
    PIDState, nacelle_angle_for_phase,
)

OUT = ROOT / "outputs" / "control"
OUT.mkdir(parents=True, exist_ok=True)

# ── Pick a representative mission from the vehicle dataset ─────────────────────
vdf = pd.read_parquet(ROOT / "outputs" / "vehicle" / "vehicle_dataset.parquet")
# Choose median cruise speed / altitude mission
row = vdf.iloc[(vdf["cruise_speed_ms"] - vdf["cruise_speed_ms"].median()).abs().argsort().iloc[0]]
print(f"Mission: mission_time={row['mission_time_s']:.1f}s  "
      f"cruise_speed={row['cruise_speed_ms']:.1f}m/s  "
      f"cruise_alt={row['cruise_altitude_m']:.0f}m")

mission_time = float(row["mission_time_s"])
cruise_speed = float(row["cruise_speed_ms"])
cruise_alt   = float(row["cruise_altitude_m"])
hover_s      = float(row.get("hover_time_s", 60.0))

schedule = build_phase_schedule(mission_time, hover_s, cruise_speed, cruise_alt)
print("Phase schedule:")
for ph, dur, spd, alt in schedule:
    print(f"  {ph:8s}  {dur:6.1f}s  ref_spd={spd:.1f}m/s  ref_alt={alt:.0f}m")

# ── Full time-series simulation (mirrors dataset.py but stores every step) ─────
n_steps = int(mission_time / DT) + 1
ts   = np.zeros(n_steps)
pxs  = np.zeros(n_steps); pys = np.zeros(n_steps); pzs = np.zeros(n_steps)
vxs  = np.zeros(n_steps); vys = np.zeros(n_steps); vzs = np.zeros(n_steps)
rolls= np.zeros(n_steps); pitchs=np.zeros(n_steps); yaws=np.zeros(n_steps)
alt_refs = np.zeros(n_steps)
vx_refs  = np.zeros(n_steps)
px_refs  = np.zeros(n_steps)
thrusts  = np.zeros(n_steps)
phase_ids= np.zeros(n_steps, dtype=int)
PHASE_MAP = {p: i for i, p in enumerate(["TAKEOFF","HOVER","TRANS1","CRUISE","TRANS2","HOVER2","LAND"])}

# State init
px=py=pz=vx=vy=vz=roll=pitch=yaw=p_rate=q_rate=r_rate=0.0
pid_alt_outer=PIDState(); pid_alt_inner=PIDState()
pid_vx=PIDState(); pid_vy=PIDState()
pid_roll=PIDState(); pid_pitch=PIDState(); pid_yaw=PIDState()
pid_p=PIDState(); pid_q=PIDState(); pid_r=PIDState()
pid_px=PIDState(); pid_py=PIDState()

phase_idx=0; phase_time_elapsed=0.0
current_phase, phase_dur, ref_spd, ref_alt_phase = schedule[0]
prev_phase=current_phase
nacelle_deg=90.0; nacelle_target=90.0; nacelle_tilt_rate=10.0
px_phase_ref=0.0; px_phase_start=0.0
alt_ref=0.0

for step in range(n_steps):
    t = step * DT
    phase_time_elapsed += DT
    if phase_time_elapsed >= phase_dur and phase_idx < len(schedule) - 1:
        phase_idx += 1
        current_phase, phase_dur, ref_spd, ref_alt_phase = schedule[phase_idx]
        phase_time_elapsed = 0.0
        nacelle_target = nacelle_angle_for_phase(current_phase)
        px_phase_start = px; px_phase_ref = 0.0

    diff = nacelle_target - nacelle_deg
    if abs(diff) > 0.1:
        nacelle_deg += math.copysign(min(nacelle_tilt_rate * DT, abs(diff)), diff)
    prev_phase = current_phase

    # Reference generation
    if current_phase == "TAKEOFF":
        alt_ref = cruise_alt * (phase_time_elapsed / phase_dur)
        vx_ref = vy_ref = 0.0
    elif current_phase == "LAND":
        alt_ref = cruise_alt * max(0.0, 1.0 - phase_time_elapsed / phase_dur)
        vx_ref = vy_ref = 0.0
    else:
        alt_ref = ref_alt_phase
        ramp = min(1.0, phase_time_elapsed / max(3.0, phase_dur * 0.15))
        vx_ref = ref_spd * ramp; vy_ref = 0.0

    if current_phase == "CRUISE":
        px_phase_ref += vx_ref * DT
        pos_err_x = px_phase_ref - (px - px_phase_start)
    else:
        pos_err_x = 0.0
    pos_err_y = -py

    vx_cmd = vx_ref + pid_px.update(pos_err_x, KP_POS, KI_POS, KD_POS, DT, 10.0)
    vy_cmd = pid_py.update(pos_err_y, KP_POS, KI_POS, KD_POS, DT, 10.0)
    vx_cmd = max(-VEL_CMD_MAX, min(VEL_CMD_MAX, vx_cmd))
    vy_cmd = max(-VEL_CMD_MAX, min(VEL_CMD_MAX, vy_cmd))

    alt_err = alt_ref - pz
    cr_cmd = pid_alt_outer.update(alt_err, KP_ALT, KI_ALT, KD_ALT, DT, 20.0)
    cr_cmd = max(-CR_MAX, min(CR_MAX, cr_cmd))
    cr_err = cr_cmd - vz
    thrust_collective = pid_alt_inner.update(cr_err, KP_CR, KI_CR, KD_CR, DT, 800.0)
    thrust_collective += WEIGHT_N
    thrust_collective = max(MIN_THRUST_N, min(MAX_THRUST_N, thrust_collective))

    vx_err = vx_cmd - vx; vy_err = vy_cmd - vy
    accel_x_cmd = pid_vx.update(vx_err, KP_VEL, KI_VEL, KD_VEL, DT, 5.0) * G
    accel_y_cmd = pid_vy.update(vy_err, KP_VEL, KI_VEL, KD_VEL, DT, 5.0) * G
    pitch_ref = math.atan2(-accel_x_cmd, G)
    roll_ref  = math.atan2( accel_y_cmd, G)
    pitch_ref = max(-ATT_MAX, min(ATT_MAX, pitch_ref))
    roll_ref  = max(-ATT_MAX, min(ATT_MAX, roll_ref))

    roll_err = roll_ref - roll; pitch_err = pitch_ref - pitch; yaw_err = -yaw
    p_cmd = max(-RATE_MAX, min(RATE_MAX, pid_roll.update(roll_err,   KP_ATT, KI_ATT, KD_ATT, DT, 2.0)))
    q_cmd = max(-RATE_MAX, min(RATE_MAX, pid_pitch.update(pitch_err, KP_ATT, KI_ATT, KD_ATT, DT, 2.0)))
    r_cmd = max(-RATE_MAX, min(RATE_MAX, pid_yaw.update(yaw_err,     KP_ATT, KI_ATT, KD_ATT, DT, 2.0)))

    p_err = p_cmd - p_rate; q_err = q_cmd - q_rate; r_err = r_cmd - r_rate
    Mx = pid_p.update(p_err, KP_RT, KI_RT, KD_RT, DT, 50.0)
    My = pid_q.update(q_err, KP_RT, KI_RT, KD_RT, DT, 50.0)
    Mz = pid_r.update(r_err, KP_RT, KI_RT, KD_RT, DT, 50.0)

    T0 = B_PINV[0,0]*thrust_collective + B_PINV[0,1]*Mx + B_PINV[0,2]*My + B_PINV[0,3]*Mz
    T1 = B_PINV[1,0]*thrust_collective + B_PINV[1,1]*Mx + B_PINV[1,2]*My + B_PINV[1,3]*Mz
    T2 = B_PINV[2,0]*thrust_collective + B_PINV[2,1]*Mx + B_PINV[2,2]*My + B_PINV[2,3]*Mz
    T3 = B_PINV[3,0]*thrust_collective + B_PINV[3,1]*Mx + B_PINV[3,2]*My + B_PINV[3,3]*Mz
    C0 = max(MIN_THRUST_N, min(T_PER, T0)); C1 = max(MIN_THRUST_N, min(T_PER, T1))
    C2 = max(MIN_THRUST_N, min(T_PER, T2)); C3 = max(MIN_THRUST_N, min(T_PER, T3))
    Fz_actual = C0+C1+C2+C3

    cos_r=math.cos(roll); sin_r=math.sin(roll)
    cos_p=math.cos(pitch); sin_p=math.sin(pitch)
    Fz_world = cos_r*cos_p*Fz_actual
    Fx_world = -sin_p*Fz_actual - 2.5*vx
    Fy_world =  sin_r*cos_p*Fz_actual - 2.5*vy
    vx += (Fx_world/MASS_KG)*DT; vy += (Fy_world/MASS_KG)*DT
    vz += ((Fz_world-WEIGHT_N)/MASS_KG)*DT
    px += vx*DT; py += vy*DT; pz += vz*DT
    if pz < 0.0: pz = 0.0
    p_rate += ((Mx - (IZZ-IYY)*q_rate*r_rate)/IXX)*DT
    q_rate += ((My - (IXX-IZZ)*p_rate*r_rate)/IYY)*DT
    r_rate += ((Mz - (IYY-IXX)*p_rate*q_rate)/IZZ)*DT
    tan_p = math.tan(pitch); inv_cp = 1.0/max(cos_p,1e-4)
    roll  += (p_rate + sin_r*tan_p*q_rate + cos_r*tan_p*r_rate)*DT
    pitch += (cos_r*q_rate - sin_r*r_rate)*DT
    yaw   += (sin_r*inv_cp*q_rate + cos_r*inv_cp*r_rate)*DT
    roll  = max(-1.5708, min(1.5708, roll))
    pitch = max(-1.5708, min(1.5708, pitch))

    ts[step]=t; pxs[step]=px; pys[step]=py; pzs[step]=pz
    vxs[step]=vx; vys[step]=vy; vzs[step]=vz
    rolls[step]=roll; pitchs[step]=pitch; yaws[step]=yaw
    alt_refs[step]=alt_ref; vx_refs[step]=vx_ref
    px_refs[step]=px_phase_start+px_phase_ref
    thrusts[step]=thrust_collective
    phase_ids[step]=PHASE_MAP.get(current_phase, 0)

# ── Plot ───────────────────────────────────────────────────────────────────────
PHASE_NAMES = ["TAKEOFF","HOVER","TRANS1","CRUISE","TRANS2","HOVER2","LAND"]
PHASE_COLORS = ["#4e9af1","#2ecc71","#f39c12","#e74c3c","#f39c12","#2ecc71","#9b59b6"]

fig = plt.figure(figsize=(14, 10))
fig.suptitle(
    f"Single-Mission Validation  |  cruise_speed={cruise_speed:.0f} m/s  "
    f"cruise_alt={cruise_alt:.0f} m  mission_time={mission_time:.0f} s",
    fontsize=11, fontweight="bold"
)
gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.30)

def shade_phases(ax):
    boundaries = [0.0]
    labels = []
    for ph, dur, _, _ in schedule:
        boundaries.append(boundaries[-1] + dur)
        labels.append(ph)
    for i, (t0, t1) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        ax.axvspan(t0, t1, alpha=0.08, color=PHASE_COLORS[i % len(PHASE_COLORS)])
    return boundaries, labels

# 1. Altitude tracking
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(ts, alt_refs, "k--", lw=1, label="ref")
ax1.plot(ts, pzs, color="#e74c3c", lw=1.2, label="actual")
shade_phases(ax1)
ax1.set_xlabel("Time (s)"); ax1.set_ylabel("Altitude (m)")
ax1.set_title("Altitude Tracking"); ax1.legend(fontsize=8)
alt_err_arr = np.abs(alt_refs - pzs)
ax1.text(0.98, 0.05, f"mean err={alt_err_arr.mean():.1f}m", transform=ax1.transAxes,
         ha="right", fontsize=8, color="#e74c3c")

# 2. Velocity tracking
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(ts, vx_refs, "k--", lw=1, label="ref vx")
ax2.plot(ts, vxs, color="#3498db", lw=1.2, label="actual vx")
shade_phases(ax2)
ax2.set_xlabel("Time (s)"); ax2.set_ylabel("Velocity (m/s)")
ax2.set_title("Cruise Velocity Tracking"); ax2.legend(fontsize=8)
cruise_mask = phase_ids == PHASE_MAP["CRUISE"]
if cruise_mask.any():
    ve = np.abs(vx_refs[cruise_mask] - vxs[cruise_mask])
    ax2.text(0.98, 0.05, f"cruise mean err={ve.mean():.2f}m/s", transform=ax2.transAxes,
             ha="right", fontsize=8, color="#3498db")

# 3. X-position tracking
ax3 = fig.add_subplot(gs[1, 0])
ax3.plot(ts, px_refs/1000, "k--", lw=1, label="ref x")
ax3.plot(ts, pxs/1000, color="#27ae60", lw=1.2, label="actual x")
shade_phases(ax3)
ax3.set_xlabel("Time (s)"); ax3.set_ylabel("X Position (km)")
ax3.set_title("X-Position Tracking"); ax3.legend(fontsize=8)
pos_err = np.abs(px_refs - pxs)
if cruise_mask.any():
    ax3.text(0.98, 0.05, f"cruise mean err={pos_err[cruise_mask].mean():.1f}m",
             transform=ax3.transAxes, ha="right", fontsize=8, color="#27ae60")

# 4. Attitude (pitch)
ax4 = fig.add_subplot(gs[1, 1])
ax4.plot(ts, np.degrees(pitchs), color="#8e44ad", lw=1.2, label="pitch")
ax4.plot(ts, np.degrees(rolls),  color="#ae6525", lw=1.0, label="roll", alpha=0.7)
shade_phases(ax4)
ax4.axhline(0, color="k", lw=0.5, ls="--")
ax4.set_xlabel("Time (s)"); ax4.set_ylabel("Angle (deg)")
ax4.set_title("Attitude (Pitch / Roll)"); ax4.legend(fontsize=8)
ax4.text(0.98, 0.05, f"pitch @ cruise≈{np.degrees(pitchs[cruise_mask]).mean():.1f}°",
         transform=ax4.transAxes, ha="right", fontsize=8, color="#8e44ad")

# 5. Collective thrust
ax5 = fig.add_subplot(gs[2, 0])
ax5.plot(ts, thrusts, color="#c0392b", lw=1.0)
ax5.axhline(WEIGHT_N, color="k", lw=0.8, ls="--", label=f"weight={WEIGHT_N:.0f}N")
shade_phases(ax5)
ax5.set_xlabel("Time (s)"); ax5.set_ylabel("Thrust (N)")
ax5.set_title("Collective Thrust Command"); ax5.legend(fontsize=8)

# 6. Phase legend + stats table
ax6 = fig.add_subplot(gs[2, 1])
ax6.axis("off")
stats = [
    ["Metric", "Value"],
    ["pos_error mean", f"{pos_err[cruise_mask].mean():.1f} m" if cruise_mask.any() else "N/A"],
    ["alt_error mean", f"{alt_err_arr.mean():.1f} m"],
    ["cruise vx mean", f"{vxs[cruise_mask].mean():.1f} m/s" if cruise_mask.any() else "N/A"],
    ["cruise vx ref",  f"{vx_refs[cruise_mask].mean():.1f} m/s" if cruise_mask.any() else "N/A"],
    ["pitch @ cruise", f"{np.degrees(pitchs[cruise_mask]).mean():.1f}°" if cruise_mask.any() else "N/A"],
    ["thrust mean",    f"{thrusts.mean():.0f} N"],
    ["weight",         f"{WEIGHT_N:.0f} N"],
]
tbl = ax6.table(cellText=stats[1:], colLabels=stats[0],
                loc="center", cellLoc="left")
tbl.auto_set_font_size(False); tbl.set_fontsize(8.5)
tbl.scale(1.0, 1.4)
ax6.set_title("Summary Stats (Cruise Phase)", fontsize=9)

out_path = OUT / "validation_single_mission.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Plot saved: {out_path}")
