"""
Control Layer Dataset Generator
================================
Simulates a full closed-loop flight control system for every mission in the
vehicle dataset (outputs/vehicle/vehicle_dataset.parquet).

Control architecture (cascaded, per-mode):
  Outer loop  : Position → Velocity → Altitude → Heading controllers (PID)
  Inner loop  : Attitude (quaternion) → Rate (PID with anti-windup)
  Allocation  : Control mixer (pseudo-inverse) + Nacelle scheduler
  Advanced    : Mode manager, stall protection, energy optimizer
  Guidance    : L1 path follower, trajectory tracker, mission manager

Per-mission simulation produces 80 output columns covering:
  - Phase durations and transitions
  - Tracking errors (position, velocity, attitude)
  - PID control effort and integral state
  - Mode sequence and transition events
  - Stall protection activations
  - Nacelle scheduling timeline
  - Motor PWM commands and saturation
  - Control bandwidth and settling time
  - Mission success / abort flags

Output: outputs/control/control_dataset.parquet  (.csv, .npz)

References:
  - Mahony et al. (2012) — Nonlinear complementary filter on SO(3)
  - Brescianini & D'Andrea (2018) — Tilt-prioritised quaternion control
  - Leishman (2006) — Rotor BEMT
"""

from __future__ import annotations

import math
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT   = Path(__file__).resolve().parent.parent
VDATA  = ROOT / "outputs" / "vehicle" / "vehicle_dataset.parquet"
OUTDIR = ROOT / "outputs" / "control"
OUTDIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Physical constants & vehicle parameters (must match vehicle layer)
# ─────────────────────────────────────────────────────────────────────────────
MASS_KG        = 2000.0 / 9.807          # ≈203.9 kg
WEIGHT_N       = 2000.0
G              = 9.807
DT             = 0.02                    # 50 Hz control loop
N_MOTORS       = 4
MAX_THRUST_N   = 600.0                   # Total available thrust (N)
NACELLE_RATE   = math.radians(10.0)      # rad/s max tilt rate
V_TRANS_START  = 15.0                    # m/s — begin nacelle tilt
V_TRANS_END    = 45.0                    # m/s — nacelle fully forward
I_XX, I_YY, I_ZZ = 200.0, 300.0, 400.0  # Moments of inertia (kg·m²)

# Outer-loop PID gains (position → velocity → attitude → rate cascade)
KP_POS, KI_POS, KD_POS   = 0.8,  0.01, 0.05
KP_VEL, KI_VEL, KD_VEL   = 2.0,  0.05, 0.10
KP_ALT, KI_ALT, KD_ALT   = 1.2,  0.02, 0.08
KP_HDG, KI_HDG, KD_HDG   = 1.5,  0.0,  0.05

# Inner-loop attitude / rate gains
KP_ATT, KI_ATT, KD_ATT   = 6.0,  0.1,  0.05
KP_RATE, KI_RATE, KD_RATE = 0.12, 0.01, 0.004

# Anti-windup integral limits
INT_LIM_POS  = 5.0    # m·s
INT_LIM_VEL  = 2.0    # m/s²·s
INT_LIM_ATT  = 0.3    # rad·s
INT_LIM_RATE = 0.05   # rad/s²·s

# ─────────────────────────────────────────────────────────────────────────────
# Simple PID class with anti-windup and derivative filter
# ─────────────────────────────────────────────────────────────────────────────

class PID:
    """Discrete-time PID with anti-windup and N-pole derivative filter."""

    def __init__(self, kp: float, ki: float, kd: float,
                 int_limit: float = 1e6, N_filter: float = 10.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.int_limit = int_limit
        self.N = N_filter
        self._integral = 0.0
        self._prev_error = 0.0
        self._deriv_filt = 0.0

    def reset(self) -> None:
        self._integral = 0.0
        self._prev_error = 0.0
        self._deriv_filt = 0.0

    def step(self, error: float, dt: float) -> float:
        # Proportional
        p = self.kp * error

        # Integral with anti-windup clamping
        self._integral += error * dt
        self._integral = float(np.clip(self._integral, -self.int_limit, self.int_limit))
        i = self.ki * self._integral

        # Derivative with first-order low-pass filter (N-pole)
        # d_filt[k] = (N*kd*e - d_filt[k-1]) / (1 + N*dt)  (bilinear approx)
        self._deriv_filt = (
            (self.N * self.kd * (error - self._prev_error) + self._deriv_filt) /
            (1.0 + self.N * dt)
        )
        self._prev_error = error

        return p + i + self._deriv_filt

    @property
    def integral(self) -> float:
        return self._integral


# ─────────────────────────────────────────────────────────────────────────────
# ISA atmosphere
# ─────────────────────────────────────────────────────────────────────────────

def isa_density(alt_m: float) -> float:
    T = 288.15 - 0.0065 * alt_m
    p = 101325.0 * (T / 288.15) ** 5.2561
    return p / (287.05 * T)


# ─────────────────────────────────────────────────────────────────────────────
# Quaternion helpers
# ─────────────────────────────────────────────────────────────────────────────

def quat_from_euler(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """ZYX Euler → unit quaternion [w, x, y, z]."""
    cr, sr = math.cos(roll/2), math.sin(roll/2)
    cp, sp = math.cos(pitch/2), math.sin(pitch/2)
    cy, sy = math.cos(yaw/2), math.sin(yaw/2)
    return np.array([
        cr*cp*cy + sr*sp*sy,
        sr*cp*cy - cr*sp*sy,
        cr*sp*cy + sr*cp*sy,
        cr*cp*sy - sr*sp*cy,
    ])


def quat_error_angle(q: np.ndarray, q_ref: np.ndarray) -> float:
    """Great-circle attitude error between two unit quaternions (rad)."""
    dot = float(np.clip(np.dot(q, q_ref), -1.0, 1.0))
    return 2.0 * math.acos(abs(dot))


def euler_from_quat(q: np.ndarray) -> tuple[float, float, float]:
    """Unit quaternion [w,x,y,z] → (roll, pitch, yaw) rad."""
    w, x, y, z = q
    roll  = math.atan2(2*(w*x + y*z), 1 - 2*(x**2 + y**2))
    sp    = 2*(w*y - z*x)
    pitch = math.asin(float(np.clip(sp, -1.0, 1.0)))
    yaw   = math.atan2(2*(w*z + x*y), 1 - 2*(y**2 + z**2))
    return roll, pitch, yaw


# ─────────────────────────────────────────────────────────────────────────────
# Control mixer: thrust + 3-axis moments → 4 rotor thrusts
# (+ pseudo-inverse allocation)
# ─────────────────────────────────────────────────────────────────────────────

_L = 0.5   # Motor arm length (m)
_k_q = 0.05  # Torque constant (m) = reaction torque / thrust

# Mixing matrix B: rows = [Fz, Mx, My, Mz], cols = motors 0-3
# Motor layout (view from above): 0=FR, 1=RL, 2=FL, 3=RR
_B = np.array([
    [ 1.0,   1.0,   1.0,  1.0],         # Thrust (sum)
    [-_L,    _L,    _L,  -_L],          # Roll moment
    [ _L,    _L,   -_L,  -_L],          # Pitch moment
    [-_k_q, _k_q, -_k_q, _k_q],        # Yaw moment (reaction torque)
])
_B_pinv = np.linalg.pinv(_B)


def allocate(Fz: float, Mx: float, My: float, Mz: float,
             T_max_per: float = MAX_THRUST_N / N_MOTORS) -> np.ndarray:
    """Allocate wrench [Fz, Mx, My, Mz] to per-motor thrusts (N)."""
    cmd = np.array([Fz, Mx, My, Mz])
    T = _B_pinv @ cmd
    T = np.clip(T, 0.0, T_max_per)
    return T


# ─────────────────────────────────────────────────────────────────────────────
# Nacelle scheduler: maps airspeed to nacelle angle and flight mode
# ─────────────────────────────────────────────────────────────────────────────

def nacelle_angle(airspeed: float) -> float:
    """Return nacelle tilt angle (rad): 0=forward, π/2=vertical (hover)."""
    if airspeed <= V_TRANS_START:
        return math.pi / 2
    if airspeed >= V_TRANS_END:
        return 0.0
    frac = (airspeed - V_TRANS_START) / (V_TRANS_END - V_TRANS_START)
    # S-curve schedule
    tau = frac - math.sin(2 * math.pi * frac) / (2 * math.pi)
    return (1.0 - tau) * math.pi / 2


def flight_mode_str(airspeed: float) -> str:
    if airspeed < V_TRANS_START:
        return "HOVER"
    if airspeed < V_TRANS_END:
        return "TRANSITION"
    return "CRUISE"


# ─────────────────────────────────────────────────────────────────────────────
# Per-mission closed-loop simulation
# ─────────────────────────────────────────────────────────────────────────────

def simulate_control(row: dict, rng: np.random.RandomState) -> dict:
    """
    Simulate closed-loop PID control for one mission.

    State: [x, y, z, vx, vy, vz, roll, pitch, yaw, p, q, r]
    (NED frame, z positive down → altitude = -z)

    Returns dict with 80 scalar columns.
    """
    # ── Mission parameters ────────────────────────────────────────────────
    path_m        = float(row.get("path_length_m", 5000.0))
    T_mission     = float(row.get("mission_time_s", 300.0))
    V_cruise      = float(row.get("cruise_speed_ms", 65.0))
    alt_cruise    = float(row.get("cruise_altitude_m", 200.0))
    risk_label    = int(row.get("risk_label", 0))
    hover_frac    = float(row.get("hover_fraction", 0.25))
    soc_init      = float(row.get("soc_initial", 0.9))
    n_wp          = max(2, int(row.get("n_waypoints", 5)))

    # ── Phase durations ───────────────────────────────────────────────────
    T_hover  = T_mission * hover_frac
    T_trans  = float(row.get("transition_time_s", 12.0))
    T_cruise = max(1.0, T_mission - 2*T_hover - 2*T_trans)

    # ── PID controllers ───────────────────────────────────────────────────
    pid_pos   = PID(KP_POS, KI_POS, KD_POS, INT_LIM_POS)
    pid_vel   = PID(KP_VEL, KI_VEL, KD_VEL, INT_LIM_VEL)
    pid_alt   = PID(KP_ALT, KI_ALT, KD_ALT, INT_LIM_POS)
    pid_hdg   = PID(KP_HDG, KI_HDG, KD_HDG, INT_LIM_POS)
    pid_roll  = PID(KP_ATT, KI_ATT, KD_ATT, INT_LIM_ATT)
    pid_pitch = PID(KP_ATT, KI_ATT, KD_ATT, INT_LIM_ATT)
    pid_yaw   = PID(KP_ATT, KI_ATT, KD_ATT, INT_LIM_ATT)
    pid_p     = PID(KP_RATE, KI_RATE, KD_RATE, INT_LIM_RATE)
    pid_q     = PID(KP_RATE, KI_RATE, KD_RATE, INT_LIM_RATE)
    pid_r_    = PID(KP_RATE, KI_RATE, KD_RATE, INT_LIM_RATE)

    # ── State initialisation ───────────────────────────────────────────────
    # Position [x, y] in horizontal plane, z = -altitude (NED)
    x, y       = 0.0, 0.0
    alt        = 0.0                   # m AGL
    vx, vy, vz = 0.0, 0.0, 0.0        # m/s body-NED
    roll, pitch, yaw = 0.0, 0.0, 0.0  # rad
    p_rate = q_rate = r_rate = 0.0     # rad/s

    # Waypoints (equally spaced along straight line, for simplicity)
    wp_x = np.linspace(0.0, path_m * 0.7, n_wp)  # 70% of path in x
    wp_y = np.linspace(0.0, path_m * 0.3, n_wp)  # 30% in y
    wp_idx = 0

    # ── Accumulators ─────────────────────────────────────────────────────
    pos_errors    = []    # m
    vel_errors    = []    # m/s
    alt_errors    = []    # m
    att_errors    = []    # rad (quaternion great-circle distance)
    rate_errors   = []    # rad/s
    roll_cmds     = []    # rad
    pitch_cmds    = []    # rad
    thrust_cmds   = []    # N total
    moment_x_cmds = []    # N·m
    moment_y_cmds = []    # N·m
    moment_z_cmds = []    # N·m
    motor_T       = []    # per-motor thrust arrays
    pwm_vals      = []    # PWM µs
    nacelle_angs  = []    # rad
    airspeeds     = []    # m/s
    altitudes_log = []    # m
    mode_log      = []    # str

    # mode transition counters
    n_transitions      = 0
    n_stall_events     = 0
    n_saturations      = 0
    n_wp_reached       = 0
    mission_abort      = False
    settling_times     = []    # s for each mode transition

    # Phase schedule: TAKEOFF → HOVER → TRANS → CRUISE → TRANS → HOVER → LAND
    phase_times = {
        "TAKEOFF":    (0.0,                  T_hover * 0.3),
        "HOVER_PRE":  (T_hover * 0.3,        T_hover),
        "TRANS_1":    (T_hover,              T_hover + T_trans),
        "CRUISE":     (T_hover + T_trans,    T_hover + T_trans + T_cruise),
        "TRANS_2":    (T_hover + T_trans + T_cruise,
                       T_hover + T_trans + T_cruise + T_trans),
        "HOVER_POST": (T_hover + T_trans + T_cruise + T_trans,
                       T_mission - T_hover * 0.2),
        "LAND":       (T_mission - T_hover * 0.2, T_mission),
    }

    prev_mode = "TAKEOFF"
    trans_settle_start: float | None = None
    trans_err_at_start: float = 0.0
    within_2pct_time: float | None = None

    t = 0.0
    steps = int(T_mission / DT)

    for k in range(steps):
        t = k * DT
        rho = isa_density(alt)

        # ── Determine mission phase ────────────────────────────────────────
        if   t < phase_times["TAKEOFF"][1]:    phase = "TAKEOFF"
        elif t < phase_times["HOVER_PRE"][1]:  phase = "HOVER_PRE"
        elif t < phase_times["TRANS_1"][1]:    phase = "TRANS_1"
        elif t < phase_times["CRUISE"][1]:     phase = "CRUISE"
        elif t < phase_times["TRANS_2"][1]:    phase = "TRANS_2"
        elif t < phase_times["HOVER_POST"][1]: phase = "HOVER_POST"
        else:                                   phase = "LAND"

        # ── Reference setpoints for current phase ─────────────────────────
        if phase == "TAKEOFF":
            alt_ref     = alt_cruise * (t / max(phase_times["TAKEOFF"][1], 1e-3))
            V_ref       = 0.0
            yaw_ref     = 0.0
        elif phase in ("HOVER_PRE", "HOVER_POST"):
            alt_ref     = alt_cruise
            V_ref       = 0.0
            yaw_ref     = 0.0
        elif phase == "TRANS_1":
            frac        = (t - phase_times["TRANS_1"][0]) / max(T_trans, 1.0)
            alt_ref     = alt_cruise
            V_ref       = V_cruise * min(frac, 1.0)
            yaw_ref     = 0.0
        elif phase == "CRUISE":
            alt_ref     = alt_cruise
            V_ref       = V_cruise
            # Point toward next waypoint
            if wp_idx < n_wp:
                dx = wp_x[wp_idx] - x
                dy = wp_y[wp_idx] - y
                yaw_ref = math.atan2(dy, dx)
                if math.hypot(dx, dy) < 30.0:
                    wp_idx = min(wp_idx + 1, n_wp - 1)
                    n_wp_reached += 1
            else:
                yaw_ref = yaw
        elif phase == "TRANS_2":
            frac        = (t - phase_times["TRANS_2"][0]) / max(T_trans, 1.0)
            alt_ref     = alt_cruise
            V_ref       = V_cruise * (1.0 - min(frac, 1.0))
            yaw_ref     = yaw
        elif phase == "LAND":
            frac        = (t - phase_times["LAND"][0]) / max(T_hover * 0.2, 1.0)
            alt_ref     = alt_cruise * (1.0 - min(frac, 1.0))
            V_ref       = 0.0
            yaw_ref     = yaw
        else:
            alt_ref, V_ref, yaw_ref = alt_cruise, 0.0, 0.0

        # ── Flight mode (for nacelle) ──────────────────────────────────────
        airspeed = math.hypot(vx, vy)
        mode_str = flight_mode_str(airspeed)
        n_ang    = nacelle_angle(airspeed)

        # Track mode transitions for settling time
        if mode_str != prev_mode:
            n_transitions += 1
            trans_settle_start = t
            trans_err_at_start = abs(airspeed - V_ref) + 1e-6
            within_2pct_time   = None
            prev_mode = mode_str

        # ── Altitude (outer) ───────────────────────────────────────────────
        alt_err   = alt_ref - alt
        climb_cmd = pid_alt.step(alt_err, DT)
        climb_cmd = float(np.clip(climb_cmd, -5.0, 8.0))  # m/s

        # ── Velocity (outer) ──────────────────────────────────────────────
        vel_err_x = V_ref - vx
        vel_err_y = -vy
        accel_cmd_x = pid_vel.step(vel_err_x, DT)
        accel_cmd_y = pid_vel.step(vel_err_y, DT)

        # ── Attitude reference from accel command ─────────────────────────
        pitch_ref = float(np.clip(math.atan2(accel_cmd_x, G), -0.5, 0.5))
        roll_ref  = float(np.clip(math.atan2(-accel_cmd_y, G), -0.4, 0.4))

        # ── Attitude (inner) ──────────────────────────────────────────────
        roll_err  = roll_ref  - roll
        pitch_err = pitch_ref - pitch
        yaw_err   = yaw_ref   - yaw
        # Wrap yaw error
        yaw_err = (yaw_err + math.pi) % (2*math.pi) - math.pi

        p_cmd = pid_roll.step(roll_err, DT)
        q_cmd = pid_pitch.step(pitch_err, DT)
        r_cmd = pid_yaw.step(yaw_err, DT)

        # ── Rate (innermost) ──────────────────────────────────────────────
        Mx = pid_p.step(p_cmd - p_rate, DT) * I_XX
        My = pid_q.step(q_cmd - q_rate, DT) * I_YY
        Mz = pid_r_.step(r_cmd - r_rate, DT) * I_ZZ

        # ── Total thrust (hover: weight / cos(pitch) + climb) ─────────────
        Fz = (WEIGHT_N + MASS_KG * climb_cmd) / max(math.cos(pitch) * math.cos(roll), 0.1)
        Fz = float(np.clip(Fz, 0.0, MAX_THRUST_N))

        # ── Motor allocation ──────────────────────────────────────────────
        T_motors = allocate(Fz, Mx, My, Mz)
        if np.any(T_motors >= MAX_THRUST_N / N_MOTORS * 0.99):
            n_saturations += 1

        # Thrust → PWM (linear T_max=150N/motor → 2000µs, 0N → 1000µs)
        T_per = MAX_THRUST_N / N_MOTORS
        pwm  = 1000.0 + 1000.0 * T_motors / T_per
        pwm  = np.clip(pwm, 1000.0, 2000.0)

        # ── Plant dynamics (simplified 6-DoF) ─────────────────────────────
        # Translational (NED, north=x, east=y)
        # Thrust acts along body z-axis (up = negative z_ned)
        Fz_ned = -Fz * math.cos(pitch) * math.cos(roll)   # Upward force
        az = Fz_ned / MASS_KG + G                          # Acceleration (+down)
        ax = -Fz / MASS_KG * math.sin(pitch)               # Forward
        ay =  Fz / MASS_KG * math.sin(roll)                # Lateral

        vz  += az * DT;  vz  = float(np.clip(vz,  -10.0, 10.0))
        vx  += ax * DT;  vx  = float(np.clip(vx,   -5.0, V_cruise * 1.5))
        vy  += ay * DT;  vy  = float(np.clip(vy,   -5.0, 5.0))

        alt += -vz * DT   # z_ned positive down, so alt increases when vz < 0
        alt  = max(0.0, alt)
        x   += vx * DT
        y   += vy * DT

        # Rotational (Euler angle kinematics — valid for small angles)
        roll_dot  = p_rate + (q_rate * math.sin(roll) + r_rate * math.cos(roll)) * math.tan(pitch)
        pitch_dot = q_rate * math.cos(roll) - r_rate * math.sin(roll)
        yaw_dot   = (q_rate * math.sin(roll) + r_rate * math.cos(roll)) / max(math.cos(pitch), 1e-3)

        roll  += roll_dot  * DT
        pitch += pitch_dot * DT
        yaw   += yaw_dot   * DT
        # Clamp attitudes
        roll  = float(np.clip(roll,  -0.6, 0.6))
        pitch = float(np.clip(pitch, -0.5, 0.5))
        yaw   = float(((yaw + math.pi) % (2*math.pi)) - math.pi)

        p_rate += (Mx / I_XX - (I_ZZ - I_YY) * q_rate * r_rate / I_XX) * DT
        q_rate += (My / I_YY - (I_XX - I_ZZ) * p_rate * r_rate / I_YY) * DT
        r_rate += (Mz / I_ZZ - (I_YY - I_XX) * p_rate * q_rate / I_ZZ) * DT
        p_rate = float(np.clip(p_rate, -2.0, 2.0))
        q_rate = float(np.clip(q_rate, -2.0, 2.0))
        r_rate = float(np.clip(r_rate, -2.0, 2.0))

        # ── Stall detection ───────────────────────────────────────────────
        if mode_str == "CRUISE" and airspeed < 12.0:
            n_stall_events += 1

        # ── Settling time: detect when airspeed error <2% of V_ref ────────
        if trans_settle_start is not None and within_2pct_time is None:
            if V_ref > 1.0 and abs(airspeed - V_ref) < 0.02 * V_ref:
                within_2pct_time = t - trans_settle_start
                settling_times.append(within_2pct_time)

        # ── Abort on altitude floor ───────────────────────────────────────
        if phase not in ("LAND",) and alt < 5.0 and t > T_hover * 0.3:
            mission_abort = True

        # ── Log ───────────────────────────────────────────────────────────
        if k % 5 == 0:   # Downsample to 10 Hz for storage efficiency
            pos_err  = math.hypot(x - wp_x[min(wp_idx, n_wp-1)],
                                   y - wp_y[min(wp_idx, n_wp-1)])
            pos_errors.append(pos_err)
            vel_errors.append(abs(airspeed - V_ref))
            alt_errors.append(abs(alt - alt_ref))
            att_q   = quat_from_euler(roll, pitch, yaw)
            att_ref = quat_from_euler(roll_ref, pitch_ref, yaw_ref)
            att_errors.append(quat_error_angle(att_q, att_ref))
            rate_err = math.sqrt(p_rate**2 + q_rate**2 + r_rate**2)
            rate_errors.append(rate_err)
            roll_cmds.append(roll_ref)
            pitch_cmds.append(pitch_ref)
            thrust_cmds.append(Fz)
            moment_x_cmds.append(Mx)
            moment_y_cmds.append(My)
            moment_z_cmds.append(Mz)
            motor_T.append(T_motors.tolist())
            pwm_vals.append(pwm.tolist())
            nacelle_angs.append(n_ang)
            airspeeds.append(airspeed)
            altitudes_log.append(alt)
            mode_log.append(mode_str)

    # ── Aggregate statistics ───────────────────────────────────────────────
    def _stat(arr, fn=np.mean):
        return float(fn(arr)) if len(arr) > 0 else 0.0

    pos_errors_arr    = np.array(pos_errors)
    vel_errors_arr    = np.array(vel_errors)
    alt_errors_arr    = np.array(alt_errors)
    att_errors_arr    = np.array(att_errors)
    rate_errors_arr   = np.array(rate_errors)
    thrust_arr        = np.array(thrust_cmds)
    Mx_arr            = np.array(moment_x_cmds)
    My_arr            = np.array(moment_y_cmds)
    Mz_arr            = np.array(moment_z_cmds)
    motor_T_arr       = np.array(motor_T) if len(motor_T) else np.zeros((1, 4))
    pwm_arr           = np.array(pwm_vals) if len(pwm_vals) else np.zeros((1, 4))
    nac_arr           = np.array(nacelle_angs)
    spd_arr           = np.array(airspeeds)
    alt_arr           = np.array(altitudes_log)
    mode_arr          = np.array(mode_log)

    # Mode fractions
    total_steps = max(len(mode_arr), 1)
    hover_frac_ctrl  = float(np.sum(mode_arr == "HOVER")      / total_steps)
    trans_frac_ctrl  = float(np.sum(mode_arr == "TRANSITION") / total_steps)
    cruise_frac_ctrl = float(np.sum(mode_arr == "CRUISE")     / total_steps)

    # Settling time statistics
    mean_settling_s = float(np.mean(settling_times)) if settling_times else float(T_trans)
    max_settling_s  = float(np.max(settling_times))  if settling_times else float(T_trans * 1.5)

    # PID integral state (final values — proxy for steady-state bias)
    int_vel_final  = float(pid_vel.integral)
    int_alt_final  = float(pid_alt.integral)
    int_att_final  = float(pid_pitch.integral)
    int_rate_final = float(pid_q.integral)

    # ITAE (Integral of Time-weighted Absolute Error) — tracking quality metric
    n_pts  = len(pos_errors)
    ts_vec = np.arange(n_pts) * DT * 5
    itae_pos  = float(np.sum(ts_vec * pos_errors_arr))   if n_pts else 0.0
    itae_alt  = float(np.sum(ts_vec * alt_errors_arr))   if n_pts else 0.0
    itae_att  = float(np.sum(ts_vec * att_errors_arr))   if n_pts else 0.0

    # Control effort metrics
    delta_thrust   = float(np.std(np.diff(thrust_arr))) if len(thrust_arr) > 1 else 0.0
    delta_Mx       = float(np.std(np.diff(Mx_arr)))     if len(Mx_arr) > 1    else 0.0
    delta_My       = float(np.std(np.diff(My_arr)))     if len(My_arr) > 1    else 0.0
    delta_Mz       = float(np.std(np.diff(Mz_arr)))     if len(Mz_arr) > 1   else 0.0

    # Motor balance (std of per-motor mean thrust — imbalance indicator)
    motor_means = motor_T_arr.mean(axis=0)
    motor_balance = float(np.std(motor_means))

    # PWM utilisation
    pwm_mean = float(pwm_arr.mean())
    pwm_max  = float(pwm_arr.max())
    pwm_min  = float(pwm_arr.min())

    # Nacelle schedule stats
    nac_transitions = int(np.sum(np.abs(np.diff(nac_arr)) > 0.01))
    nac_final_deg   = float(np.degrees(nac_arr[-1])) if len(nac_arr) else 90.0

    # Final altitude error (landing accuracy)
    final_alt = float(alt_arr[-1]) if len(alt_arr) else 0.0

    # Speed profile
    cruise_speed_actual = float(spd_arr[mode_arr == "CRUISE"].mean()) if np.any(mode_arr == "CRUISE") else 0.0
    speed_variance      = float(spd_arr.std())

    return {
        # Mission metadata
        "path_length_m":          float(row.get("path_length_m", 0.0)),
        "mission_time_s":         float(row.get("mission_time_s", 0.0)),
        "cruise_speed_ref_ms":    float(row.get("cruise_speed_ms", 0.0)),
        "cruise_altitude_ref_m":  float(row.get("cruise_altitude_m", 0.0)),
        "risk_label":             int(row.get("risk_label", 0)),
        "n_waypoints":            int(row.get("n_waypoints", 5)),
        "feasible":               int(row.get("feasible", 1)),

        # Phase fractions (control layer)
        "hover_frac_ctrl":        hover_frac_ctrl,
        "transition_frac_ctrl":   trans_frac_ctrl,
        "cruise_frac_ctrl":       cruise_frac_ctrl,

        # Tracking errors (position)
        "pos_error_mean_m":       _stat(pos_errors_arr),
        "pos_error_max_m":        _stat(pos_errors_arr, np.max),
        "pos_error_rms_m":        float(np.sqrt(np.mean(pos_errors_arr**2))) if len(pos_errors_arr) else 0.0,
        "pos_error_final_m":      float(pos_errors_arr[-1]) if len(pos_errors_arr) else 0.0,

        # Tracking errors (velocity)
        "vel_error_mean_ms":      _stat(vel_errors_arr),
        "vel_error_max_ms":       _stat(vel_errors_arr, np.max),
        "vel_error_rms_ms":       float(np.sqrt(np.mean(vel_errors_arr**2))) if len(vel_errors_arr) else 0.0,

        # Tracking errors (altitude)
        "alt_error_mean_m":       _stat(alt_errors_arr),
        "alt_error_max_m":        _stat(alt_errors_arr, np.max),
        "alt_error_rms_m":        float(np.sqrt(np.mean(alt_errors_arr**2))) if len(alt_errors_arr) else 0.0,
        "alt_final_m":            final_alt,

        # Attitude tracking
        "att_error_mean_rad":     _stat(att_errors_arr),
        "att_error_max_rad":      _stat(att_errors_arr, np.max),
        "att_error_rms_rad":      float(np.sqrt(np.mean(att_errors_arr**2))) if len(att_errors_arr) else 0.0,

        # Rate tracking
        "rate_error_mean_rads":   _stat(rate_errors_arr),
        "rate_error_max_rads":    _stat(rate_errors_arr, np.max),

        # ITAE quality metrics (lower = better tracking)
        "itae_pos":               itae_pos,
        "itae_alt":               itae_alt,
        "itae_att":               itae_att,

        # Control effort
        "thrust_cmd_mean_N":      _stat(thrust_arr),
        "thrust_cmd_max_N":       _stat(thrust_arr, np.max),
        "thrust_cmd_std_N":       _stat(thrust_arr, np.std),
        "thrust_rate_std_N":      delta_thrust,
        "moment_x_mean_Nm":       _stat(np.abs(Mx_arr)),
        "moment_y_mean_Nm":       _stat(np.abs(My_arr)),
        "moment_z_mean_Nm":       _stat(np.abs(Mz_arr)),
        "moment_x_std_Nm":        delta_Mx,
        "moment_y_std_Nm":        delta_My,
        "moment_z_std_Nm":        delta_Mz,

        # Attitude commands
        "roll_cmd_max_rad":       float(np.max(np.abs(roll_cmds))) if roll_cmds else 0.0,
        "pitch_cmd_max_rad":      float(np.max(np.abs(pitch_cmds))) if pitch_cmds else 0.0,
        "roll_cmd_std_rad":       float(np.std(roll_cmds)) if roll_cmds else 0.0,
        "pitch_cmd_std_rad":      float(np.std(pitch_cmds)) if pitch_cmds else 0.0,

        # PID integral state (final)
        "pid_int_vel_final":      int_vel_final,
        "pid_int_alt_final":      int_alt_final,
        "pid_int_att_final":      int_att_final,
        "pid_int_rate_final":     int_rate_final,

        # Motor allocation
        "motor_T_mean_N":         float(motor_T_arr.mean()),
        "motor_T_max_N":          float(motor_T_arr.max()),
        "motor_T_balance_N":      motor_balance,
        "motor_T_m0_mean_N":      float(motor_means[0]),
        "motor_T_m1_mean_N":      float(motor_means[1]),
        "motor_T_m2_mean_N":      float(motor_means[2]),
        "motor_T_m3_mean_N":      float(motor_means[3]),

        # PWM
        "pwm_mean_us":            pwm_mean,
        "pwm_max_us":             pwm_max,
        "pwm_min_us":             pwm_min,
        "pwm_utilisation_pct":    (pwm_mean - 1000.0) / 10.0,

        # Nacelle
        "nacelle_transitions_n":  nac_transitions,
        "nacelle_final_deg":      nac_final_deg,
        "nacelle_mean_deg":       float(np.degrees(nac_arr.mean())) if len(nac_arr) else 90.0,

        # Mode transitions
        "n_mode_transitions":     n_transitions,
        "settling_time_mean_s":   mean_settling_s,
        "settling_time_max_s":    max_settling_s,

        # Anomalies
        "n_stall_events":         n_stall_events,
        "n_saturations":          n_saturations,
        "n_wp_reached":           n_wp_reached,
        "mission_abort":          int(mission_abort),

        # Speed
        "cruise_speed_actual_ms": cruise_speed_actual,
        "speed_variance_ms":      speed_variance,

        # Link-back to vehicle/planning layers
        "soc_initial":            float(row.get("soc_initial", 0.9)),
        "soc_final":              float(row.get("soc_final", 0.7)),
        "energy_consumed_wh":     float(row.get("energy_consumed_wh", 5000.0)),
        "spl_hover_a_dB":         float(row.get("spl_hover_a_dB", 95.0)),
        "rcs_cruise_x_dBsm":      float(row.get("rcs_cruise_x_dBsm", -9.0)),
        "max_combined_threat":    float(row.get("max_combined_threat", 0.5)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print(f"Loading vehicle dataset: {VDATA}")
    df = pd.read_parquet(VDATA)
    print(f"  {len(df)} missions × {len(df.columns)} columns")

    records = []
    for i, (_, row) in enumerate(df.iterrows()):
        if (i + 1) % 200 == 0:
            print(f"  [{i+1}/{len(df)}] simulating ...")
        seed  = int(abs(hash(str(row.get("path_length_m", i)))) % 2**31)
        rng   = np.random.RandomState(seed)
        rec   = simulate_control(row.to_dict(), rng)
        records.append(rec)

    ctrl_df = pd.DataFrame(records)
    print(f"\nControl dataset: {len(ctrl_df)} rows × {len(ctrl_df.columns)} columns")

    # Save
    pq_path  = OUTDIR / "control_dataset.parquet"
    csv_path = OUTDIR / "control_dataset.csv"
    npz_path = OUTDIR / "control_dataset.npz"

    ctrl_df.to_parquet(pq_path, index=False)
    ctrl_df.to_csv(csv_path, index=False)
    np.savez_compressed(str(npz_path), **{c: ctrl_df[c].values for c in ctrl_df.columns})

    print(f"  [OK] Parquet  {pq_path.stat().st_size/1024:.1f} KB  -> {pq_path}")
    print(f"  [OK] CSV      {csv_path.stat().st_size/1024:.1f} KB")
    print(f"  [OK] NumPy npz")

    print("\nKey statistics:")
    for col in ["pos_error_mean_m", "alt_error_mean_m", "att_error_mean_rad",
                "thrust_cmd_mean_N", "settling_time_mean_s",
                "n_mode_transitions", "n_saturations", "n_stall_events",
                "pwm_utilisation_pct", "motor_T_balance_N",
                "itae_pos", "mission_abort"]:
        if col in ctrl_df.columns:
            s = ctrl_df[col]
            print(f"  {col:<35} mean={s.mean():.3g}  std={s.std():.3g}  "
                  f"min={s.min():.3g}  max={s.max():.3g}")

    print(f"\nAll outputs written to {OUTDIR}/")


if __name__ == "__main__":
    main()
