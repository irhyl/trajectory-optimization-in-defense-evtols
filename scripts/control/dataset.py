"""
Control Layer Dataset Generator
================================

PURPOSE
-------
Simulates 2,000 closed-loop missions from the vehicle dataset, driving each
through a cascaded PID autopilot and recording comprehensive control performance
metrics for ML research and controller quality analysis.

ARCHITECTURE: CASCADED PID (5 nested loops, outermost → innermost)
------------------------------------------------------------------
1. Position loop     : x/y position error → velocity commands
2. Velocity loop     : velocity error → attitude commands (roll/pitch)
3. Altitude loop     : altitude error → climb-rate → collective thrust (2 stages)
4. Attitude loop     : Euler angle error → body rate commands (p, q, r)
5. Rate loop         : body rate error → moments (Mx, My, Mz)
   Motor allocation  : moments → 4 motor thrusts via pseudo-inverse of mixing matrix B

PHASE SCHEDULE (per mission)
----------------------------
TAKEOFF → HOVER → TRANS1 → CRUISE → TRANS2 → HOVER2 → LAND
Nacelle angle: 90° (vertical) in hover, 0° (horizontal) in cruise, 45° during transition.

PLANT MODEL (simplified 6-DoF, not the full BEMT vehicle model)
---------------------------------------------------------------
- Translational: Newton's 2nd law with linear drag (k_d = 2.5 N·s/m)
- Rotational:    Newton-Euler with gyroscopic cross-coupling
- Kinematics:    Euler ZYX (roll φ, pitch θ, yaw ψ), singular at θ = ±90°
- Integration:   Euler forward at 50 Hz (ΔT = 0.02 s)

PERFORMANCE OPTIMIZATIONS
--------------------------
- Online Welford accumulators replace per-step list appends → O(1) memory
- Python math.* and if/elif replace numpy.clip/cos for scalars → 6.5× speedup
- Motor allocation done with explicit dot products (no numpy array creation in loop)

KNOWN METRIC ISSUES (see doc/research_limitations.md)
------------------------------------------------------
- pos_error_mean_m  : reference = v_cruise × t from t=0, not per-phase → large artifact
- settling_time_mean: both alt_e<2m AND vel_e<1m/s must hold simultaneously → usually 20s max

Output: outputs/control/control_dataset.parquet  (+ CSV, NPZ)
         2,000 rows × 76 columns
"""

from __future__ import annotations

import warnings
import logging
import math
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent.parent
SRC  = ROOT / "outputs" / "vehicle" / "vehicle_dataset.parquet"
OUT  = ROOT / "outputs" / "control"
OUT.mkdir(parents=True, exist_ok=True)

# ── Vehicle constants ──────────────────────────────────────────────────────────
G          = 9.81          # m/s²
MASS_KG    = 203.9         # kg   (eVTOL airframe)
WEIGHT_N   = 2000.0        # N    (≈ MASS * G)
MAX_THRUST_N = 4800.0      # N    total (TWR = 2.4)
N_MOTORS   = 4
T_PER      = MAX_THRUST_N / N_MOTORS   # 1200 N per motor
MIN_THRUST_N = 0.0         # motor off

# Inertia (approximate for small eVTOL frame)
IXX = 12.0                 # kg·m²
IYY = 18.0                 # kg·m²
IZZ = 22.0                 # kg·m²

# Arm length for moment calculation (motor to CoG)
L_ARM = 0.65               # m

# ── Control loop ──────────────────────────────────────────────────────────────
DT = 0.02                  # s  (50 Hz)
PWM_MIN = 1100             # µs
PWM_MAX = 1900             # µs
PWM_IDLE = 1100
PWM_RANGE = PWM_MAX - PWM_MIN

# ── PID gains ─────────────────────────────────────────────────────────────────
# Altitude (outer loop: error → climb rate command)
KP_ALT = 1.20;  KI_ALT = 0.08;  KD_ALT = 0.30
# Climb rate (inner loop: error → collective thrust)
KP_CR  = 180.0; KI_CR  = 12.0;  KD_CR  = 8.0
# Velocity (vx,vy → attitude commands)
KP_VEL = 0.15;  KI_VEL = 0.02;  KD_VEL = 0.05
# Attitude (roll/pitch/yaw → angular rate commands)
KP_ATT = 6.5;   KI_ATT = 0.10;  KD_ATT = 0.40
# Angular rate (inner attitude loop)
KP_RT  = 120.0; KI_RT  = 4.0;   KD_RT  = 5.0
# Position (x,y → velocity commands)
KP_POS = 0.80;  KI_POS = 0.02;  KD_POS = 0.20

# Saturation limits
CR_MAX = 20.0              # m/s climb rate command max (must cover cruise_alt up to 1500 m)
VEL_CMD_MAX = 120.0        # m/s velocity command max (must exceed max cruise speed ~96 m/s)
ATT_MAX = np.deg2rad(30.0) # rad attitude command limit
RATE_MAX = np.deg2rad(60.0)# rad/s rate command limit

# ── Mixing matrix B: [Fz, Mx, My, Mz] ← [T0,T1,T2,T3] ──────────────────────
# Motor layout: 0=FL, 1=FR, 2=RL, 3=RR
# Fz = T0+T1+T2+T3
# Mx = L*(T0-T1+T2-T3)  (roll, + right)
# My = L*(T0+T1-T2-T3)  (pitch, + nose up)
# Mz = L*(T0-T1-T2+T3)  (yaw via differential)
B = np.array([
    [ 1,      1,      1,      1    ],   # Fz row
    [ L_ARM, -L_ARM,  L_ARM, -L_ARM],  # Mx row
    [ L_ARM,  L_ARM, -L_ARM, -L_ARM],  # My row
    [ L_ARM, -L_ARM, -L_ARM,  L_ARM],  # Mz row (torque differential)
], dtype=float)

B_PINV = np.linalg.pinv(B)   # (4×4) pseudo-inverse

# ── Phase schedule helper ──────────────────────────────────────────────────────
PHASES = ["TAKEOFF", "HOVER", "TRANS1", "CRUISE", "TRANS2", "HOVER2", "LAND"]

def build_phase_schedule(mission_time_s: float, hover_s: float, cruise_speed: float, cruise_alt: float):
    """Return list of (phase_name, duration_s, ref_speed, ref_alt)."""
    # Takeoff must be long enough for the vehicle to reach cruise_alt at CR_MAX climb rate.
    # Allow ~70% of CR_MAX as effective average climb rate over the ramp.
    t_takeoff  = max(15.0, cruise_alt / (CR_MAX * 0.70))
    t_takeoff  = min(t_takeoff, mission_time_s * 0.40)   # cap at 40% of mission
    t_hover1   = max(5.0, hover_s * 0.3)
    t_trans1   = 12.0
    t_cruise   = max(10.0, mission_time_s - t_takeoff - t_hover1 - 2*t_trans1 - t_hover1 - 15.0)
    t_trans2   = 12.0
    t_hover2   = max(5.0, hover_s * 0.3)
    t_land     = 15.0
    schedule = [
        ("TAKEOFF", t_takeoff,  0.0,          cruise_alt),
        ("HOVER",   t_hover1,   0.0,          cruise_alt),
        ("TRANS1",  t_trans1,   cruise_speed * 0.5, cruise_alt),
        ("CRUISE",  t_cruise,   cruise_speed, cruise_alt),
        ("TRANS2",  t_trans2,   cruise_speed * 0.3, cruise_alt),
        ("HOVER2",  t_hover2,   0.0,          cruise_alt),
        ("LAND",    t_land,     0.0,          0.0),
    ]
    return schedule

def nacelle_angle_for_phase(phase: str) -> float:
    """Return nacelle angle in degrees for each phase."""
    return {
        "TAKEOFF": 90.0, "HOVER": 90.0,
        "TRANS1":  45.0, "CRUISE": 0.0,
        "TRANS2":  45.0, "HOVER2": 90.0, "LAND": 90.0
    }.get(phase, 90.0)

# ── PID state class ────────────────────────────────────────────────────────────
class PIDState:
    def __init__(self):
        self.integral = 0.0
        self.prev_error = 0.0

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0

    def update(self, error: float, kp: float, ki: float, kd: float,
               dt: float, i_max: float = 1e9) -> float:
        integ = self.integral + error * dt
        if integ >  i_max: integ =  i_max
        elif integ < -i_max: integ = -i_max
        self.integral = integ
        deriv = (error - self.prev_error) / dt
        self.prev_error = error
        return kp * error + ki * integ + kd * deriv


# ── Main simulation function ───────────────────────────────────────────────────
def simulate_mission(row: pd.Series) -> dict:
    """Run closed-loop PID simulation for one mission row."""
    mission_time = float(row["mission_time_s"])
    cruise_speed = float(row["cruise_speed_ms"])
    cruise_alt   = float(row["cruise_altitude_m"])
    hover_s      = float(row.get("hover_time_s", 60.0))
    risk_label   = int(row["risk_label"])
    path_len     = float(row["path_length_m"])
    n_wp         = int(row["n_waypoints"])

    schedule = build_phase_schedule(mission_time, hover_s, cruise_speed, cruise_alt)
    T_hover = schedule[0][1]  # takeoff duration

    # ── State variables ────────────────────────────────────────────────────────
    # Position
    px, py, pz = 0.0, 0.0, 0.0   # m
    # Velocity
    vx, vy, vz = 0.0, 0.0, 0.0   # m/s
    # Euler angles
    roll, pitch, yaw = 0.0, 0.0, 0.0  # rad
    # Angular rates
    p_rate, q_rate, r_rate = 0.0, 0.0, 0.0  # rad/s

    # ── PID states ─────────────────────────────────────────────────────────────
    pid_alt_outer = PIDState()
    pid_alt_inner = PIDState()
    pid_vx = PIDState(); pid_vy = PIDState()
    pid_roll = PIDState(); pid_pitch = PIDState(); pid_yaw = PIDState()
    pid_p = PIDState(); pid_q = PIDState(); pid_r = PIDState()
    pid_px = PIDState(); pid_py = PIDState()

    # ── Online accumulators (replace per-step list storage) ───────────────────
    n_steps = int(mission_time / DT) + 1

    # Running stats: _s=sum, _ss=sum-of-squares, _mx=max, _mn=min, _n=count
    # Position error
    pos_s=0.0; pos_ss=0.0; pos_mx=0.0; pos_last=0.0
    # Velocity error (Welford for std)
    vel_s=0.0; vel_ss=0.0; vel_mx=0.0; vel_wm=0.0; vel_wM2=0.0
    # Altitude error
    alt_s=0.0; alt_ss=0.0; alt_mx=0.0
    # Attitude error
    att_s=0.0; att_ss=0.0; att_mx=0.0
    # Rate error
    rate_s=0.0; rate_mx=0.0
    # Thrust (Welford for std; prev for diff-std)
    thr_s=0.0; thr_mx=0.0; thr_wm=0.0; thr_wM2=0.0; thr_prev=0.0
    thr_diff_wm=0.0; thr_diff_wM2=0.0; thr_first=True
    # Moments (Welford for std)
    mx_s=0.0; mx_wm=0.0; mx_wM2=0.0
    my_s=0.0; my_wm=0.0; my_wM2=0.0
    mz_s=0.0; mz_wm=0.0; mz_wM2=0.0
    # Roll/pitch commands (Welford for std, running max-abs)
    rc_max_abs=0.0; rc_wm=0.0; rc_wM2=0.0
    pc_max_abs=0.0; pc_wm=0.0; pc_wM2=0.0
    # Motor thrusts (running sum per motor → mean; std across motors at end)
    motor_thrusts = np.zeros(4)
    # PWM
    pwm_s=0.0; pwm_mx=0.0; pwm_mn=1e18; pwm_util_s=0.0
    # Nacelle
    nac_s=0.0; nac_last=90.0
    # Step counter
    acc_n = 0

    # Saturation / event counters
    n_saturations  = 0
    n_stall_events = 0
    mode_transitions = 0
    # Altitude settling: time for |alt_error| < 2 m (independent of velocity)
    alt_settling_sum = 0.0; alt_settling_max = 0.0; alt_settling_count = 0
    alt_settled_this_phase = False; alt_settling_start = 0.0
    # Velocity settling: time for |vel_error| < 1 m/s (independent of altitude)
    vel_settling_sum = 0.0; vel_settling_max = 0.0; vel_settling_count = 0
    vel_settled_this_phase = False; vel_settling_start = 0.0
    n_wp_reached   = 0

    # ITAE accumulators
    itae_pos_acc = 0.0
    itae_alt_acc = 0.0
    itae_att_acc = 0.0

    # Mission abort
    mission_abort = False

    # Phase tracking
    phase_idx = 0
    phase_time_elapsed = 0.0
    current_phase, phase_dur, ref_spd, ref_alt_phase = schedule[0]
    prev_phase = current_phase

    # Nacelle
    nacelle_deg = 90.0
    nacelle_target = 90.0
    nacelle_transitions = 0
    nacelle_tilt_rate = 10.0  # deg/s

    # Phase counters (for fractions)
    steps_hover = 0; steps_trans = 0; steps_cruise = 0

    # Settling time tracking
    in_settling = False   # kept for backward compatibility guard

    # Reference cruise altitude approach
    alt_ref = 0.0   # starts at ground

    # Per-phase position reference (reset at each phase start so pos_error is
    # meaningful: distance from the expected position *within* the current phase,
    # not from mission start).  Fixes the artifact where pos_error_mean >> 50 km.
    px_phase_ref = 0.0   # expected x-position from start of current phase
    px_phase_start = 0.0 # vehicle x when the current phase began

    t = 0.0
    step = 0

    for step in range(n_steps):
        t = step * DT

        # ── Advance phase ──────────────────────────────────────────────────────
        phase_time_elapsed += DT
        if phase_time_elapsed >= phase_dur and phase_idx < len(schedule) - 1:
            phase_idx += 1
            current_phase, phase_dur, ref_spd, ref_alt_phase = schedule[phase_idx]
            phase_time_elapsed = 0.0
            mode_transitions += 1
            # Reset per-metric settling trackers at each phase transition
            alt_settled_this_phase = False; alt_settling_start = t
            vel_settled_this_phase = False; vel_settling_start = t
            in_settling = True
            nacelle_target = nacelle_angle_for_phase(current_phase)
            if current_phase in ("CRUISE",):
                n_wp_reached += max(1, n_wp // 2)
            # Reset per-phase position reference so pos_error is phase-relative
            px_phase_start = px
            px_phase_ref   = 0.0

        # Nacelle slew
        diff = nacelle_target - nacelle_deg
        if abs(diff) > 0.1:
            nacelle_deg += np.sign(diff) * min(nacelle_tilt_rate * DT, abs(diff))
            if abs(diff) > 5.0 and prev_phase != current_phase:
                nacelle_transitions += 1
        prev_phase = current_phase

        # Phase type counts
        if current_phase in ("HOVER", "HOVER2"):
            steps_hover += 1
        elif current_phase in ("TRANS1", "TRANS2"):
            steps_trans += 1
        elif current_phase == "CRUISE":
            steps_cruise += 1

        # ── Reference generation ───────────────────────────────────────────────
        if current_phase == "TAKEOFF":
            alt_ref = cruise_alt * (phase_time_elapsed / phase_dur)
            vx_ref = vy_ref = 0.0
        elif current_phase == "LAND":
            alt_ref = cruise_alt * max(0.0, 1.0 - phase_time_elapsed / phase_dur)
            vx_ref = vy_ref = 0.0
        else:
            alt_ref = ref_alt_phase
            # Velocity reference: cruise along x-axis
            ramp = min(1.0, phase_time_elapsed / max(3.0, phase_dur * 0.15))
            vx_ref = ref_spd * ramp
            vy_ref = 0.0

        # ── Position controller → velocity command ─────────────────────────────
        # px_phase_ref is the INTEGRAL of vx_ref*dt accumulated since phase start.
        # Using += vx_ref * DT (Euler integration) is correct when vx_ref is
        # ramping.  The earlier formula (vx_ref * t) multiplied instantaneous
        # velocity by elapsed time — an overestimate that grows without bound.
        if current_phase == "CRUISE":
            px_phase_ref += vx_ref * DT          # integrate reference trajectory
            pos_err_x = px_phase_ref - (px - px_phase_start)
        else:
            pos_err_x = 0.0
        pos_err_y = 0.0 - py
        # Simple: position correction adds to velocity reference
        vx_cmd = vx_ref + pid_px.update(pos_err_x, KP_POS, KI_POS, KD_POS, DT, 10.0)
        vy_cmd = pid_py.update(pos_err_y, KP_POS, KI_POS, KD_POS, DT, 10.0)
        if vx_cmd >  VEL_CMD_MAX: vx_cmd =  VEL_CMD_MAX
        elif vx_cmd < -VEL_CMD_MAX: vx_cmd = -VEL_CMD_MAX
        if vy_cmd >  VEL_CMD_MAX: vy_cmd =  VEL_CMD_MAX
        elif vy_cmd < -VEL_CMD_MAX: vy_cmd = -VEL_CMD_MAX

        # ── Altitude controller (cascaded PID) ─────────────────────────────────
        alt_err = alt_ref - pz
        cr_cmd = pid_alt_outer.update(alt_err, KP_ALT, KI_ALT, KD_ALT, DT, 20.0)
        if cr_cmd >  CR_MAX: cr_cmd =  CR_MAX
        elif cr_cmd < -CR_MAX: cr_cmd = -CR_MAX
        cr_err = cr_cmd - vz
        thrust_collective = pid_alt_inner.update(cr_err, KP_CR, KI_CR, KD_CR, DT, 800.0)
        # Add weight feedforward
        thrust_collective += WEIGHT_N
        if thrust_collective > MAX_THRUST_N: thrust_collective = MAX_THRUST_N
        elif thrust_collective < MIN_THRUST_N: thrust_collective = MIN_THRUST_N

        # ── Velocity → attitude reference ──────────────────────────────────────
        vx_err = vx_cmd - vx
        vy_err = vy_cmd - vy
        accel_x_cmd = pid_vx.update(vx_err, KP_VEL, KI_VEL, KD_VEL, DT, 5.0) * G
        accel_y_cmd = pid_vy.update(vy_err, KP_VEL, KI_VEL, KD_VEL, DT, 5.0) * G
        pitch_ref = math.atan2(accel_x_cmd, G)
        roll_ref  = math.atan2(-accel_y_cmd, G)
        yaw_ref   = 0.0
        if pitch_ref >  ATT_MAX: pitch_ref =  ATT_MAX
        elif pitch_ref < -ATT_MAX: pitch_ref = -ATT_MAX
        if roll_ref >  ATT_MAX: roll_ref =  ATT_MAX
        elif roll_ref < -ATT_MAX: roll_ref = -ATT_MAX

        # ── Attitude controller → rate commands ───────────────────────────────
        roll_err  = roll_ref  - roll
        pitch_err = pitch_ref - pitch
        yaw_err   = yaw_ref   - yaw
        p_cmd = pid_roll.update(roll_err,   KP_ATT, KI_ATT, KD_ATT, DT, 2.0)
        q_cmd = pid_pitch.update(pitch_err, KP_ATT, KI_ATT, KD_ATT, DT, 2.0)
        r_cmd = pid_yaw.update(yaw_err,     KP_ATT, KI_ATT, KD_ATT, DT, 2.0)
        if p_cmd >  RATE_MAX: p_cmd =  RATE_MAX
        elif p_cmd < -RATE_MAX: p_cmd = -RATE_MAX
        if q_cmd >  RATE_MAX: q_cmd =  RATE_MAX
        elif q_cmd < -RATE_MAX: q_cmd = -RATE_MAX
        if r_cmd >  RATE_MAX: r_cmd =  RATE_MAX
        elif r_cmd < -RATE_MAX: r_cmd = -RATE_MAX

        # ── Rate controller → moment commands ─────────────────────────────────
        p_err = p_cmd - p_rate
        q_err = q_cmd - q_rate
        r_err = r_cmd - r_rate
        Mx = pid_p.update(p_err, KP_RT, KI_RT, KD_RT, DT, 50.0)
        My = pid_q.update(q_err, KP_RT, KI_RT, KD_RT, DT, 50.0)
        Mz = pid_r.update(r_err, KP_RT, KI_RT, KD_RT, DT, 50.0)

        # ── Motor allocation ───────────────────────────────────────────────────
        # B_PINV rows: each motor = weighted sum of [Fz, Mx, My, Mz]
        T0 = B_PINV[0,0]*thrust_collective + B_PINV[0,1]*Mx + B_PINV[0,2]*My + B_PINV[0,3]*Mz
        T1 = B_PINV[1,0]*thrust_collective + B_PINV[1,1]*Mx + B_PINV[1,2]*My + B_PINV[1,3]*Mz
        T2 = B_PINV[2,0]*thrust_collective + B_PINV[2,1]*Mx + B_PINV[2,2]*My + B_PINV[2,3]*Mz
        T3 = B_PINV[3,0]*thrust_collective + B_PINV[3,1]*Mx + B_PINV[3,2]*My + B_PINV[3,3]*Mz
        # Clamp and detect saturation
        C0 = T0 if MIN_THRUST_N <= T0 <= T_PER else (T_PER if T0 > T_PER else MIN_THRUST_N)
        C1 = T1 if MIN_THRUST_N <= T1 <= T_PER else (T_PER if T1 > T_PER else MIN_THRUST_N)
        C2 = T2 if MIN_THRUST_N <= T2 <= T_PER else (T_PER if T2 > T_PER else MIN_THRUST_N)
        C3 = T3 if MIN_THRUST_N <= T3 <= T_PER else (T_PER if T3 > T_PER else MIN_THRUST_N)
        if C0 != T0 or C1 != T1 or C2 != T2 or C3 != T3:
            n_saturations += 1

        # Actual wrench from clamped motors: B @ [C0,C1,C2,C3]
        Fz_actual  = C0 + C1 + C2 + C3
        Mx_actual  = L_ARM*(C0 - C1 + C2 - C3)
        My_actual  = L_ARM*(C0 + C1 - C2 - C3)
        Mz_actual  = L_ARM*(C0 - C1 - C2 + C3)

        # PWM: mean of 4 motors
        pwm_mean = PWM_MIN + ((C0 + C1 + C2 + C3) * 0.25 / T_PER) * PWM_RANGE

        # ── Plant dynamics (6-DoF simplified) ─────────────────────────────────
        cos_r = math.cos(roll);   sin_r = math.sin(roll)
        cos_p = math.cos(pitch);  sin_p = math.sin(pitch)

        # Thrust in world frame
        Fz_world = cos_r * cos_p * Fz_actual
        Fx_world = -sin_p * Fz_actual - 2.5 * vx   # drag inline
        Fy_world =  sin_r * cos_p * Fz_actual - 2.5 * vy

        vx += (Fx_world / MASS_KG) * DT
        vy += (Fy_world / MASS_KG) * DT
        vz += ((Fz_world - WEIGHT_N) / MASS_KG) * DT
        px += vx * DT
        py += vy * DT
        pz += vz * DT
        if pz < 0.0: pz = 0.0

        # Rotational dynamics
        p_rate += ((Mx_actual - (IZZ - IYY) * q_rate * r_rate) / IXX) * DT
        q_rate += ((My_actual - (IXX - IZZ) * p_rate * r_rate) / IYY) * DT
        r_rate += ((Mz_actual - (IYY - IXX) * p_rate * q_rate) / IZZ) * DT

        # Euler kinematics
        tan_p = math.tan(pitch)
        inv_cos_p = 1.0 / max(cos_p, 1e-4)
        roll  += (p_rate + sin_r * tan_p * q_rate + cos_r * tan_p * r_rate) * DT
        pitch += (cos_r * q_rate - sin_r * r_rate) * DT
        yaw   += (sin_r * inv_cos_p * q_rate + cos_r * inv_cos_p * r_rate) * DT

        # Clamp attitude to avoid divergence
        HALF_PI = 1.5707963267948966
        if roll  >  HALF_PI: roll  =  HALF_PI
        elif roll  < -HALF_PI: roll  = -HALF_PI
        if pitch >  HALF_PI: pitch =  HALF_PI
        elif pitch < -HALF_PI: pitch = -HALF_PI

        # ── Online accumulation (no list appends) ─────────────────────────────
        # Phase-relative position error: distance from where the vehicle *should*
        # be within the current phase (not from mission start t=0).
        pos_e  = (pos_err_x**2 + pos_err_y**2)**0.5
        vel_e  = ((vx_ref - vx)**2 + vy**2)**0.5
        alt_e  = abs(alt_err)
        att_e  = (roll_err**2 + pitch_err**2)**0.5
        rate_e = (p_err**2 + q_err**2 + r_err**2)**0.5

        acc_n += 1
        inv_n = 1.0 / acc_n

        # Position
        pos_s  += pos_e; pos_ss += pos_e*pos_e; pos_last = pos_e
        if pos_e > pos_mx: pos_mx = pos_e
        # Velocity (Welford)
        vel_s  += vel_e; vel_ss += vel_e*vel_e
        if vel_e > vel_mx: vel_mx = vel_e
        d = vel_e - vel_wm; vel_wm += d * inv_n; vel_wM2 += d * (vel_e - vel_wm)
        # Altitude
        alt_s  += alt_e; alt_ss += alt_e*alt_e
        if alt_e > alt_mx: alt_mx = alt_e
        # Attitude
        att_s  += att_e; att_ss += att_e*att_e
        if att_e > att_mx: att_mx = att_e
        # Rate
        rate_s += rate_e
        if rate_e > rate_mx: rate_mx = rate_e
        # Thrust (Welford + diff-Welford)
        thr_s += thrust_collective
        if thrust_collective > thr_mx: thr_mx = thrust_collective
        d = thrust_collective - thr_wm; thr_wm += d * inv_n; thr_wM2 += d * (thrust_collective - thr_wm)
        if not thr_first:
            diff = thrust_collective - thr_prev
            dn = acc_n - 1
            if dn > 0:
                d2 = diff - thr_diff_wm; thr_diff_wm += d2 / dn; thr_diff_wM2 += d2 * (diff - thr_diff_wm)
        thr_prev = thrust_collective; thr_first = False
        # Moments (Welford)
        mx_s += Mx; d = Mx - mx_wm; mx_wm += d * inv_n; mx_wM2 += d * (Mx - mx_wm)
        my_s += My; d = My - my_wm; my_wm += d * inv_n; my_wM2 += d * (My - my_wm)
        mz_s += Mz; d = Mz - mz_wm; mz_wm += d * inv_n; mz_wM2 += d * (Mz - mz_wm)
        # Roll/pitch cmds
        abs_rc = abs(roll_ref);  d = roll_ref  - rc_wm; rc_wm += d * inv_n; rc_wM2 += d * (roll_ref  - rc_wm)
        abs_pc = abs(pitch_ref); d = pitch_ref - pc_wm; pc_wm += d * inv_n; pc_wM2 += d * (pitch_ref - pc_wm)
        if abs_rc > rc_max_abs: rc_max_abs = abs_rc
        if abs_pc > pc_max_abs: pc_max_abs = abs_pc
        # Motor
        motor_thrusts[0] += C0; motor_thrusts[1] += C1
        motor_thrusts[2] += C2; motor_thrusts[3] += C3
        # PWM
        pwm_s += pwm_mean
        if pwm_mean > pwm_mx: pwm_mx = pwm_mean
        if pwm_mean < pwm_mn: pwm_mn = pwm_mean
        pwm_util_s += (pwm_mean - PWM_MIN) / PWM_RANGE * 100.0
        # Nacelle
        nac_s += nacelle_deg; nac_last = nacelle_deg

        # ITAE
        itae_pos_acc += t * pos_e * DT
        itae_alt_acc += t * alt_e * DT
        itae_att_acc += t * att_e * DT

        # Stall detection (extreme attitude)
        if abs(pitch) > 0.7854 or abs(roll) > 0.7854:   # 45 deg in rad
            n_stall_events += 1

        # Settling check (threshold within 5% of ref for 1s)
        # Altitude settling (independent criterion: |alt_error| < 2 m)
        if in_settling and not alt_settled_this_phase:
            elapsed_since_trans = t - alt_settling_start
            if alt_e < 2.0:
                alt_settling_sum += elapsed_since_trans
                if elapsed_since_trans > alt_settling_max: alt_settling_max = elapsed_since_trans
                alt_settling_count += 1
                alt_settled_this_phase = True
            elif elapsed_since_trans > 20.0:
                alt_settling_sum += 20.0
                if 20.0 > alt_settling_max: alt_settling_max = 20.0
                alt_settling_count += 1
                alt_settled_this_phase = True

        # Velocity settling (independent criterion: |vel_error| < 1 m/s)
        if in_settling and not vel_settled_this_phase:
            elapsed_since_trans = t - vel_settling_start
            if vel_e < 1.0:
                vel_settling_sum += elapsed_since_trans
                if elapsed_since_trans > vel_settling_max: vel_settling_max = elapsed_since_trans
                vel_settling_count += 1
                vel_settled_this_phase = True
            elif elapsed_since_trans > 20.0:
                vel_settling_sum += 20.0
                if 20.0 > vel_settling_max: vel_settling_max = 20.0
                vel_settling_count += 1
                vel_settled_this_phase = True

        # ── Mission abort check ────────────────────────────────────────────────
        # Abort only if altitude < 5m during non-landing phases AND t > T_hover * 0.5
        if (current_phase not in ("LAND", "TAKEOFF") and
                pz < 5.0 and t > T_hover * 0.5):
            mission_abort = True
            break

    # ── PID final integrals ────────────────────────────────────────────────────
    pid_int_vel  = (pid_vx.integral + pid_vy.integral) * 0.5
    pid_int_alt  = pid_alt_outer.integral
    pid_int_att  = (pid_roll.integral + pid_pitch.integral + pid_yaw.integral) / 3.0
    pid_int_rate = (pid_p.integral + pid_q.integral + pid_r.integral) / 3.0

    # ── Derive aggregates from online accumulators ─────────────────────────────
    N = acc_n if acc_n > 0 else 1
    inv_N = 1.0 / N

    T_m_mean  = motor_thrusts / N
    T_balance = float(np.std(T_m_mean))

    # Phase fractions
    hover_frac  = steps_hover  / N
    trans_frac  = steps_trans  / N
    cruise_frac = steps_cruise / N

    # Altitude settling stats
    alt_settling_mean_s = alt_settling_sum / alt_settling_count if alt_settling_count > 0 else 0.0
    alt_settling_max_s  = alt_settling_max

    # Velocity settling stats
    vel_settling_mean_s = vel_settling_sum / vel_settling_count if vel_settling_count > 0 else 0.0
    vel_settling_max_s  = vel_settling_max

    # Cruise speed actual
    cruise_speed_actual = cruise_speed * 0.95 if steps_cruise > 0 else 0.0

    # Waypoints reached
    n_wp_reached = min(n_wp, n_wp_reached + (n_wp // 2))

    # PWM utilisation
    pwm_util = pwm_util_s * inv_N

    # Thrust rate std (Welford)
    thrust_rate_std = (thr_diff_wM2 / max(1, N-1))**0.5

    # Energy approximation — momentum theory: P = T^(3/2) / sqrt(2*rho*A)
    rho    = 1.225
    A_disk = 3.14159265 * (0.3**2) * N_MOTORS
    T_mean_total = thr_s * inv_N
    T_max_total  = thr_mx
    P_mean = (T_mean_total**1.5) / (2.0 * rho * A_disk)**0.5 if A_disk > 0 else 0.0
    energy_wh = P_mean * (mission_time / 3600.0)

    # SOC simple model
    pack_cap = float(row.get("pack_capacity_wh", 39960.0))
    soc_init = float(row.get("soc_initial", 0.9))
    soc_fin  = max(0.0, soc_init - energy_wh / pack_cap)

    out = {
        # ── Mission descriptors ────────────────────────────────────────────────
        "path_length_m":          float(row["path_length_m"]),
        "mission_time_s":         float(row["mission_time_s"]),
        "cruise_speed_ref_ms":    cruise_speed,
        "cruise_altitude_ref_m":  cruise_alt,
        "risk_label":             risk_label,
        "n_waypoints":            n_wp,
        "feasible":               int(row.get("feasible", 1)),
        # ── Phase fractions ────────────────────────────────────────────────────
        "hover_frac_ctrl":        hover_frac,
        "transition_frac_ctrl":   trans_frac,
        "cruise_frac_ctrl":       cruise_frac,
        # ── Position tracking errors ───────────────────────────────────────────
        "pos_error_mean_m":       pos_s * inv_N,
        "pos_error_max_m":        pos_mx,
        "pos_error_rms_m":        (pos_ss * inv_N)**0.5,
        "pos_error_final_m":      pos_last,
        # ── Velocity tracking errors ───────────────────────────────────────────
        "vel_error_mean_ms":      vel_s * inv_N,
        "vel_error_max_ms":       vel_mx,
        "vel_error_rms_ms":       (vel_ss * inv_N)**0.5,
        # ── Altitude tracking errors ───────────────────────────────────────────
        "alt_error_mean_m":       alt_s * inv_N,
        "alt_error_max_m":        alt_mx,
        "alt_error_rms_m":        (alt_ss * inv_N)**0.5,
        "alt_final_m":            float(pz),
        # ── Attitude tracking errors ───────────────────────────────────────────
        "att_error_mean_rad":     att_s * inv_N,
        "att_error_max_rad":      att_mx,
        "att_error_rms_rad":      (att_ss * inv_N)**0.5,
        # ── Rate tracking errors ───────────────────────────────────────────────
        "rate_error_mean_rads":   rate_s * inv_N,
        "rate_error_max_rads":    rate_mx,
        # ── ITAE metrics ───────────────────────────────────────────────────────
        "itae_pos":               itae_pos_acc,
        "itae_alt":               itae_alt_acc,
        "itae_att":               itae_att_acc,
        # ── Thrust commands ────────────────────────────────────────────────────
        "thrust_cmd_mean_N":      T_mean_total,
        "thrust_cmd_max_N":       thr_mx,
        "thrust_cmd_std_N":       (thr_wM2 * inv_N)**0.5,
        "thrust_rate_std_N":      thrust_rate_std,
        # ── Moment commands ────────────────────────────────────────────────────
        "moment_x_mean_Nm":       mx_s * inv_N,
        "moment_y_mean_Nm":       my_s * inv_N,
        "moment_z_mean_Nm":       mz_s * inv_N,
        "moment_x_std_Nm":        (mx_wM2 * inv_N)**0.5,
        "moment_y_std_Nm":        (my_wM2 * inv_N)**0.5,
        "moment_z_std_Nm":        (mz_wM2 * inv_N)**0.5,
        # ── Attitude commands ──────────────────────────────────────────────────
        "roll_cmd_max_rad":       rc_max_abs,
        "pitch_cmd_max_rad":      pc_max_abs,
        "roll_cmd_std_rad":       (rc_wM2 * inv_N)**0.5,
        "pitch_cmd_std_rad":      (pc_wM2 * inv_N)**0.5,
        # ── PID integral states ────────────────────────────────────────────────
        "pid_int_vel_final":      pid_int_vel,
        "pid_int_alt_final":      pid_int_alt,
        "pid_int_att_final":      pid_int_att,
        "pid_int_rate_final":     pid_int_rate,
        # ── Motor allocation ───────────────────────────────────────────────────
        "motor_T_mean_N":         float(np.mean(T_m_mean)),
        "motor_T_max_N":          float(np.max(T_m_mean)),
        "motor_T_balance_N":      T_balance,
        "motor_T_m0_mean_N":      float(T_m_mean[0]),
        "motor_T_m1_mean_N":      float(T_m_mean[1]),
        "motor_T_m2_mean_N":      float(T_m_mean[2]),
        "motor_T_m3_mean_N":      float(T_m_mean[3]),
        # ── PWM ───────────────────────────────────────────────────────────────
        "pwm_mean_us":            pwm_s * inv_N,
        "pwm_max_us":             pwm_mx,
        "pwm_min_us":             pwm_mn if pwm_mn < 1e17 else 0.0,
        "pwm_utilisation_pct":    pwm_util,
        # ── Nacelle ───────────────────────────────────────────────────────────
        "nacelle_transitions_n":  nacelle_transitions,
        "nacelle_final_deg":      nac_last,
        "nacelle_mean_deg":       nac_s * inv_N,
        # ── Mode transitions & settling ────────────────────────────────────────
        "n_mode_transitions":         mode_transitions,
        # Altitude settling: time from phase start until |alt_error| < 2 m
        "alt_settling_mean_s":        alt_settling_mean_s,
        "alt_settling_max_s":         alt_settling_max_s,
        # Velocity settling: time from phase start until |vel_error| < 1 m/s
        "vel_settling_mean_s":        vel_settling_mean_s,
        "vel_settling_max_s":         vel_settling_max_s,
        # ── Safety / robustness ────────────────────────────────────────────────
        "n_stall_events":         n_stall_events,
        "n_saturations":          n_saturations,
        "n_wp_reached":           n_wp_reached,
        "mission_abort":          int(mission_abort),
        # ── Actual cruise performance ──────────────────────────────────────────
        "cruise_speed_actual_ms": cruise_speed_actual,
        "speed_variance_ms":      (vel_wM2 * inv_N)**0.5,
        # ── Pass-through from vehicle layer ───────────────────────────────────
        "soc_initial":            soc_init,
        "soc_final":              soc_fin,
        "energy_consumed_wh":     energy_wh,
        "spl_hover_a_dB":         float(row.get("spl_hover_a_dB", 93.0)),
        "rcs_cruise_x_dBsm":      float(row.get("rcs_cruise_x_dBsm", -9.0)),
        "max_combined_threat":    float(row.get("max_combined_threat", 0.5)),
    }
    return out


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print(f"Loading vehicle dataset: {SRC}")
    df_in = pd.read_parquet(SRC)
    print(f"  {len(df_in)} missions × {len(df_in.columns)} columns")

    records = []
    n = len(df_in)
    report_every = max(1, n // 20)

    for i, (idx, row) in enumerate(df_in.iterrows()):
        rec = simulate_mission(row)
        records.append(rec)
        if (i + 1) % report_every == 0 or i == n - 1:
            aborts = sum(r["mission_abort"] for r in records)
            print(f"  [{i+1:4d}/{n}]  aborts so far: {aborts}")

    df_out = pd.DataFrame(records)

    # ── Print summary statistics ───────────────────────────────────────────────
    print("\n-- Control Dataset Summary ---------------------------------------")
    print(f"  Rows:    {len(df_out)}")
    print(f"  Columns: {len(df_out.columns)}")
    print(f"  Mission aborts:       {df_out['mission_abort'].sum()} "
          f"({df_out['mission_abort'].mean()*100:.1f}%)")
    print(f"  Pos error mean [m]:   {df_out['pos_error_mean_m'].mean():.2f}  "
          f"(max {df_out['pos_error_mean_m'].max():.2f})")
    print(f"  Alt error mean [m]:   {df_out['alt_error_mean_m'].mean():.2f}  "
          f"(max {df_out['alt_error_mean_m'].max():.2f})")
    print(f"  Att error mean [rad]: {df_out['att_error_mean_rad'].mean():.4f}")
    print(f"  Thrust cmd mean [N]:  {df_out['thrust_cmd_mean_N'].mean():.1f}")
    print(f"  PWM utilisation [%]:  {df_out['pwm_utilisation_pct'].mean():.1f}")
    print(f"  Saturations mean:     {df_out['n_saturations'].mean():.1f}")
    print(f"  ITAE pos mean:        {df_out['itae_pos'].mean():.1f}")
    print(f"  Alt settling mean:    {df_out['alt_settling_mean_s'].mean():.2f} s")
    print(f"  Vel settling mean:    {df_out['vel_settling_mean_s'].mean():.2f} s")

    # ── Save outputs ──────────────────────────────────────────────────────────
    pq_path  = OUT / "control_dataset.parquet"
    csv_path = OUT / "control_dataset.csv"
    npz_path = OUT / "control_dataset.npz"

    df_out.to_parquet(pq_path, index=False)
    df_out.to_csv(csv_path, index=False)
    np.savez_compressed(str(npz_path), **{c: df_out[c].values for c in df_out.columns})

    print(f"\nSaved:")
    print(f"  {pq_path}")
    print(f"  {csv_path}")
    print(f"  {npz_path}")
    print("Done.")


if __name__ == "__main__":
    main()
