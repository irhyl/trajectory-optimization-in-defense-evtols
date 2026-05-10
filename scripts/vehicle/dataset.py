"""
Vehicle Layer Dataset Generator

For each mission in the planning dataset (test_final.parquet), simulate the
vehicle execution using physics-based models and record extensive outputs.

Physics implemented:
- Blade Element Momentum Theory (BEMT, actuator-disk + profile-drag)
- Parabolic drag polar + Prandtl lifting-line cruise aerodynamics
- 2-RC equivalent-circuit battery with Arrhenius resistance, coulomb counting
- Lumped thermal model (battery, motor, coolant) with liquid cooling
- SPL acoustic model: thickness + loading + BVI + broadband
- Stefan-Boltzmann IR radiance in SWIR/MWIR/LWIR bands
- Flat-plate RCS model (X-band and Ku-band) with blade-flash modulation

Output: outputs/vehicle/vehicle_dataset.parquet  (+ CSV, NPZ)

Usage:
    python scripts/generate_vehicle_dataset.py
"""

from __future__ import annotations

import sys
import json
import logging
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent.parent
# Use the new planning dataset (with distance-decay threat gradient) if available,
# falling back to the original test_final.parquet
_new_plan = ROOT / "outputs" / "planning_dataset" / "planning_dataset.parquet"
_old_plan = ROOT / "outputs" / "planning_dataset" / "test_final.parquet"
SRC  = _new_plan if _new_plan.exists() else _old_plan
OUT  = ROOT / "outputs" / "vehicle"
OUT.mkdir(parents=True, exist_ok=True)

# ── Physical constants ─────────────────────────────────────────────────────────
G           = 9.81          # m/s²
RHO_SL      = 1.225         # kg/m³  sea-level density
T_SL        = 288.15        # K      sea-level temperature
LAPSE       = 0.0065        # K/m    ISA lapse rate
R_AIR       = 287.05        # J/kg·K
STEFAN_BOLTZMANN = 5.67e-8  # W/m²K⁴
SPEED_SOUND = 340.3         # m/s    at SL

# ── Vehicle configuration (TiltrotorConfig defaults) ──────────────────────────
MTOW        = 1350.0        # kg  (800 + 400 + 150)
WEIGHT      = MTOW * G      # N

# Rotors (2 proprotors)
N_ROTORS    = 2
N_BLADES    = 4
R_ROTOR     = 2.0           # m
A_ROTOR     = np.pi * R_ROTOR**2   # 12.57 m²
SIGMA       = 0.1636        # solidity = N_b·c / (pi·R)   (c_mean=0.128m)
CL_ALPHA_R  = 5.73          # /rad  NACA 0012 lift-curve slope
CD0_ROTOR   = 0.01          # profile drag
CL_MAX_R    = 1.4

OMEGA_HOVER = 85.0          # rad/s   (~810 RPM rotor)  V_tip≈170 m/s
OMEGA_CRUISE = 70.0         # rad/s   (~670 RPM rotor)  reduced in cruise

GEAR_RATIO  = 2.0           # motor:rotor
ETA_MECH    = 0.98          # gearbox efficiency

# Wing (NACA 23015, span=10 m)
S_WING      = 11.5          # m²  (mean chord 1.15 m × 10 m)
AR_WING     = 10.0**2 / S_WING   # aspect ratio ≈ 8.7
CD0_WING    = 0.008
K_WING      = 0.04          # induced drag factor  (= 1/(π·e·AR))
CL0_WING    = 0.3
CL_ALPHA_W  = 5.5           # /rad

# Fuselage
CD_FUSE_REF = 0.3           # referenced to frontal area 2.5 m²
S_FUSE_FRONT = 2.5          # m²

# Battery (108S × 20P, NMC811 5 Ah cells)
CELLS_S     = 108
CELLS_P     = 20
V_NOM_CELL  = 3.7           # V
V_MAX_CELL  = 4.2           # V
V_MIN_CELL  = 3.0           # V
CAP_AH_CELL = 5.0           # Ah
R0_CELL     = 0.020         # Ω
R1_CELL     = 0.015         # Ω  RC1
C1_CELL     = 1000.0        # F
R2_CELL     = 0.010         # Ω  RC2
C2_CELL     = 10000.0       # F

V_NOM_PACK  = CELLS_S * V_NOM_CELL          # 399.6 V
V_MAX_PACK  = CELLS_S * V_MAX_CELL          # 453.6 V
CAP_AH_PACK = CELLS_P * CAP_AH_CELL         # 100 Ah
CAP_WH_PACK = CAP_AH_PACK * V_NOM_PACK      # 39960 Wh ≈ 40 kWh
R0_PACK     = R0_CELL * CELLS_S / CELLS_P   # 0.108 Ω
R1_PACK     = R1_CELL * CELLS_S / CELLS_P
R2_PACK     = R2_CELL * CELLS_S / CELLS_P

CELL_MASS   = 0.07          # kg per cell
PACK_CELLS  = CELLS_S * CELLS_P             # 2160 cells
PACK_MASS   = PACK_CELLS * CELL_MASS * 1.3  # kg  (+30% overhead)
PACK_CP     = 1000.0        # J/kg·K  specific heat
R_THERMAL   = 2.0           # K/W  pack to ambient

SOC_INIT    = 0.90          # initial state of charge
SOC_MIN     = 0.10
CYCLE_DEG_PER_FEC = 0.0001  # SOH loss per full equivalent cycle
CALENDAR_DEG_PER_DAY = 1e-5  # SOH loss per day at 25°C

# Motor
MOTOR_KT    = 0.955         # N·m/A
MOTOR_RS    = 0.02          # Ω  phase resistance
P_MOTOR_MAX = 150e3         # W  rated
ETA_MOTOR_PEAK = 0.95
MOTOR_C_TH  = 5000.0        # J/K   thermal mass
R_TH_MOTOR  = 0.1           # K/W   to ambient
T_MOTOR_MAX = 150.0         # °C

# Thermal system
BATT_MASS_TH = 200.0        # kg  battery thermal mass for ThermalModel
MOTOR_MASS_TH = 15.0        # kg
COOL_VOLUME = 10.0          # L
RHO_COOL    = 1050.0        # kg/m³  water-glycol
CP_COOL     = 3800.0        # J/kg·K
PUMP_MAX    = 0.5           # kg/s  max coolant flow

# Acoustic config
BPF_BLADES  = N_BLADES
REF_DIST    = 30.0          # m   reference distance for SPL
DETECT_THRESH = 55.0        # dB(A)  detection threshold

# Signatures
RCS_BODY_REF  = 0.1         # m²  X-band broadside (stealthy design)
RCS_WING_REF  = 0.5
RCS_NAC_REF   = 0.2
RCS_BLADE_REF = 0.01        # m² per blade
RAM_DB        = 10.0        # dB  RAM reduction
EMIS_MOTOR  = 0.3
EMIS_SKIN   = 0.8
SIGMA_SB    = 5.67e-8

# ── Helper functions ──────────────────────────────────────────────────────────

def isa_density(alt_m: float) -> float:
    """ISA air density at altitude."""
    T = T_SL - LAPSE * alt_m
    rho = RHO_SL * (T / T_SL) ** (G / (LAPSE * R_AIR) - 1)
    return float(np.clip(rho, 0.05, RHO_SL))


def bemt_hover(omega: float, collective_rad: float, rho: float) -> dict:
    """
    Single-pass BEMT hover computation with Prandtl tip loss.
    Returns thrust [N], torque [N·m], power [W], FM, v_induced [m/s].
    """
    V_tip = omega * R_ROTOR
    r_stations = np.linspace(0.1, 1.0, 25)
    dr = r_stations[1] - r_stations[0]

    # Linear taper c(r): root=0.2m, tip=0.1m
    c = 0.2 - 0.1 * r_stations

    # Linear twist: -12 deg root-to-tip (matches RotorConfig.twist = -12)
    twist_rad = np.radians(-12.0) * r_stations

    # Initial induced velocity estimate (momentum theory)
    CT_est = 0.5 * SIGMA * CL_ALPHA_R * (collective_rad / 3.0)
    T_est  = CT_est * rho * A_ROTOR * V_tip**2
    v_i    = float(np.sqrt(max(T_est / (2.0 * rho * A_ROTOR), 0.1)))

    # 2 iterations for induced velocity convergence
    for _ in range(4):
        thrust = torque = 0.0
        for i, r in enumerate(r_stations):
            R_loc  = r * R_ROTOR
            c_loc  = c[i]
            theta  = collective_rad + twist_rad[i]
            U_T    = omega * R_loc
            U_P    = v_i
            if U_T < 0.1:
                continue
            phi   = np.arctan2(U_P, U_T)
            alpha = theta - phi
            cl    = np.clip(CL_ALPHA_R * alpha, -CL_MAX_R, CL_MAX_R)
            cd    = CD0_ROTOR + 0.01 * alpha**2
            U2    = U_T**2 + U_P**2
            dL    = 0.5 * rho * U2 * c_loc * cl
            dD    = 0.5 * rho * U2 * c_loc * cd
            # Prandtl tip loss
            if r > 0.95:
                f = N_BLADES / 2.0 * (1.0 - r) / (r * max(phi, 0.01))
                B = 2.0 / np.pi * np.arccos(np.exp(-abs(f)))
            else:
                B = 1.0
            dT = (dL * np.cos(phi) - dD * np.sin(phi)) * dr * R_ROTOR
            dQ = (dL * np.sin(phi) + dD * np.cos(phi)) * R_loc * dr * R_ROTOR
            thrust += dT * B * N_BLADES
            torque += dQ * B * N_BLADES
        thrust = max(thrust, 0.0)
        v_i = float(np.sqrt(max(thrust / (2.0 * rho * A_ROTOR), 0.01)))

    power   = torque * omega
    P_ideal = thrust**1.5 / np.sqrt(2.0 * rho * A_ROTOR) if thrust > 0 else 0.0
    fm      = P_ideal / power if power > 0 else 0.0
    return {"thrust": thrust, "torque": torque, "power": power,
            "FM": float(np.clip(fm, 0, 1)), "v_induced": v_i,
            "CT": thrust / (rho * A_ROTOR * V_tip**2) if V_tip > 0 else 0.0}


def find_hover_collective(target_thrust_per_rotor: float, omega: float, rho: float) -> float:
    """Binary search for collective that gives target thrust per rotor."""
    lo, hi = np.radians(2.0), np.radians(20.0)
    for _ in range(20):
        mid = (lo + hi) / 2.0
        res = bemt_hover(omega, mid, rho)
        if res["thrust"] < target_thrust_per_rotor:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def cruise_aero(airspeed: float, rho: float, altitude: float = 200.0) -> dict:
    """
    Cruise aerodynamics using parabolic drag polar (Prandtl lifting line).
    Returns CL, CD, L/D, thrust_required [N], power_required [W].
    """
    q = 0.5 * rho * airspeed**2
    if q < 1e-3:
        return {"CL": 0, "CD": 0, "LD": 0, "T_req": 0, "P_req": 0}

    # Wing
    CL = WEIGHT / (q * S_WING)
    CL = float(np.clip(CL, 0.0, 1.4))
    e_oswald = max(0.5, 1.78 * (1.0 - 0.045 * AR_WING**0.68) - 0.64)
    k_eff = 1.0 / (np.pi * e_oswald * AR_WING)
    CD_wing = CD0_WING + k_eff * CL**2

    # Fuselage contribution (referenced to wing area)
    q_ref = 0.5 * rho * airspeed**2
    D_fuse = CD_FUSE_REF * S_FUSE_FRONT * q_ref   # fuselage drag [N]
    CD_fuse_equiv = D_fuse / (q * S_WING)

    CD = CD_wing + CD_fuse_equiv

    # Reynolds and Mach
    nu = 1.5e-5   # m²/s  kinematic viscosity
    mac = S_WING / 10.0  # mean aerodynamic chord ≈ 1.15 m
    Re = airspeed * mac / nu
    Mach = airspeed / SPEED_SOUND

    T_req = CD * q * S_WING        # total drag = thrust required
    P_req = T_req * airspeed       # shaft power required

    return {
        "CL": CL, "CD": CD, "LD": CL / CD if CD > 0 else 0,
        "T_req": T_req, "P_req": P_req,
        "Re": Re, "Mach": Mach,
        "q": q, "CD_wing": CD_wing, "CD_fuse": CD_fuse_equiv,
        "e_oswald": e_oswald,
    }


def battery_step(soc: float, soh: float, v_rc1: float, v_rc2: float,
                 T_batt: float, power_demand: float, dt: float,
                 T_ambient: float = 25.0) -> dict:
    """
    2-RC equivalent circuit battery step.
    Returns updated soc, soh, v_rc1, v_rc2, T_batt, V_terminal, I, P_loss.
    """
    # Arrhenius resistance factor
    T_K   = T_batt + 273.15
    T_ref = 298.15
    Ea    = 20000.0
    Rgas  = 8.314
    r_fac = float(np.clip(np.exp(Ea / Rgas * (1.0 / T_K - 1.0 / T_ref)), 0.5, 5.0))

    # OCV interpolation (NMC)
    soc_pts = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ocv_cell = np.array([3.00, 3.40, 3.55, 3.62, 3.68, 3.73, 3.80, 3.88, 3.98, 4.10, 4.20])
    V_oc = float(np.interp(np.clip(soc, 0.0, 1.0), soc_pts, ocv_cell)) * CELLS_S

    # Pack resistances
    R0 = R0_PACK * r_fac
    R1 = R1_PACK * r_fac
    R2 = R2_PACK * r_fac
    C1 = C1_CELL / CELLS_S * CELLS_P
    C2 = C2_CELL / CELLS_S * CELLS_P

    # Solve current from quadratic: R0·I² - V_oc·I + P = 0
    disc = V_oc**2 - 4.0 * R0 * power_demand
    if disc < 0:
        I = V_oc / (2.0 * R0)
    else:
        I = (V_oc - np.sqrt(disc)) / (2.0 * R0)
    I_max = 5.0 * CAP_AH_PACK  # 5C
    I = float(np.clip(I, 0.0, I_max))

    # Update RC voltages
    tau1 = R1 * C1
    tau2 = R2 * C2
    v_rc1_new = v_rc1 * np.exp(-dt / tau1) + R1 * I * (1.0 - np.exp(-dt / tau1))
    v_rc2_new = v_rc2 * np.exp(-dt / tau2) + R2 * I * (1.0 - np.exp(-dt / tau2))

    # Terminal voltage
    V_t = V_oc - I * R0 - v_rc1_new - v_rc2_new
    V_t = float(np.clip(V_t, CELLS_S * V_MIN_CELL, CELLS_S * V_MAX_CELL))

    # SOC update (coulomb counting)
    eta_c = 0.995
    delta_ah = I * dt / 3600.0
    delta_soc = delta_ah * eta_c / (CAP_AH_PACK * soh)
    soc_new = float(np.clip(soc - delta_soc, SOC_MIN, 1.0))

    # Ohmic heat generation
    Q_ohmic     = I**2 * R0
    dVdT        = -0.0004 * CELLS_S
    Q_reversible = I * T_K * dVdT
    Q_gen       = Q_ohmic + Q_reversible

    # Thermal dynamics
    Q_cool = (T_batt - T_ambient) / R_THERMAL
    dT_batt = (Q_gen - Q_cool) / (PACK_MASS * PACK_CP) * dt
    T_batt_new = T_batt + dT_batt

    # SOH aging
    fec  = delta_ah / (2.0 * CAP_AH_PACK)
    c_rate = I / CAP_AH_PACK
    rate_stress = 1.0 + 0.5 * max(0, c_rate - 1.0)
    cal_deg = CALENDAR_DEG_PER_DAY * (dt / 86400.0) * np.exp((T_batt - 25.0) / 10.0)
    soh_new = float(max(0.0, soh - fec * CYCLE_DEG_PER_FEC * rate_stress - cal_deg))

    return {
        "soc": soc_new, "soh": soh_new,
        "v_rc1": float(v_rc1_new), "v_rc2": float(v_rc2_new),
        "T_batt": float(T_batt_new),
        "V_terminal": V_t,
        "I": I,
        "P_loss": float(Q_ohmic),
        "c_rate": c_rate,
        "energy_wh": V_t * I * dt / 3600.0,
    }


def motor_efficiency(torque_Nm: float, omega_rads: float, T_winding: float = 50.0) -> dict:
    """Lumped PMSM efficiency from analytical model."""
    if omega_rads < 0.1 or torque_Nm < 0.01:
        return {"eta": 0.0, "P_mech": 0.0, "P_elec": 0.0, "P_loss": 0.0}

    P_mech = torque_Nm * omega_rads
    i_q = torque_Nm / MOTOR_KT
    Rs_eff = MOTOR_RS * (1.0 + 0.004 * (T_winding - 25.0))
    P_cu   = 1.5 * Rs_eff * i_q**2

    speed_ratio = omega_rads / (4000.0 * 2.0 * np.pi / 60.0)  # vs max_rpm
    P_core = 0.001 * P_MOTOR_MAX * (speed_ratio**1.5)
    P_loss = P_cu + P_core
    P_elec = P_mech + P_loss
    eta = P_mech / P_elec if P_elec > 0 else 0.0
    return {"eta": float(np.clip(eta, 0, 1)), "P_mech": P_mech,
            "P_elec": P_elec, "P_loss_cu": P_cu, "P_loss_core": P_core,
            "P_loss": P_loss, "i_q": i_q}


def acoustic_spl(
    rotor_rpm: float, rotor_thrust: float, rotor_power: float,
    tip_mach: float, nacelle_deg: float, airspeed: float,
    descent_rate: float = 0.0,
) -> dict:
    """Compute SPL components at reference distance REF_DIST."""
    A_disk      = N_ROTORS * A_ROTOR
    thrust_per  = rotor_thrust / N_ROTORS if N_ROTORS > 0 else 0
    disk_load   = thrust_per / A_ROTOR

    omega       = rotor_rpm * 2.0 * np.pi / 60.0
    V_tip       = omega * R_ROTOR
    bpf         = N_BLADES * rotor_rpm / 60.0

    # Thickness noise (monopole): ~ M_tip^6
    spl_thick = max(0, 60.0 + 60.0 * np.log10(tip_mach + 0.01))

    # Loading noise (dipole)
    T_ref, V_ref = 1000.0, 150.0
    spl_load = max(0, 85.0
        + 10.0 * np.log10((thrust_per / T_ref)**2 + 0.01)
        + 10.0 * np.log10((V_tip / V_ref)**2 + 0.01))

    # BVI noise
    if airspeed > 1.0:
        desc_angle = np.degrees(np.arctan2(descent_rate, airspeed))
    else:
        desc_angle = 0.0
    bvi_fac = np.exp(-0.1 * (desc_angle - 10.0)**2)
    trans_fac = 1.0 + 0.5 * np.sin(np.radians(nacelle_deg))
    spl_bvi_base = 75.0 + 10.0 * np.log10(disk_load / 200.0 + 0.01)
    spl_bvi = spl_bvi_base * bvi_fac * trans_fac

    # Broadband
    P_ref = 100000.0
    spl_bb = max(0, 65.0 + 5.0 * np.log10(rotor_power / P_ref + 0.01))

    # Motor EM
    spl_motor_em = 55.0 + 5.0 * np.log10(rotor_power / P_ref + 0.01)
    spl_inv      = 50.0 + 3.0 * np.log10(rotor_power / P_ref + 0.01)
    # Airframe
    spl_af = 50.0 + 60.0 * np.log10(airspeed / 100.0) if airspeed > 10.0 else 0.0

    def sum_spl(levels):
        valid = [l for l in levels if l > 0]
        if not valid:
            return 0.0
        return 10.0 * np.log10(sum(10.0**(l / 10.0) for l in valid))

    spl_rotor = sum_spl([spl_thick, spl_load, spl_bvi, spl_bb])
    spl_motor = sum_spl([spl_motor_em, spl_inv])
    spl_total = sum_spl([spl_rotor, spl_motor, spl_af])
    spl_a     = spl_total - 5.0   # A-weighting correction

    # Detection range (SPL(r) = SPL_ref - 20·log(r/r_ref) - α·r)
    alpha_abs = 1.0  # dB/km at ~500 Hz
    det_range = 10.0  # default max
    for r in np.linspace(100, 10000, 100):
        spl_r = spl_a - 20.0 * np.log10(r / REF_DIST) - alpha_abs * r / 1000.0
        if spl_r <= DETECT_THRESH:
            det_range = r / 1000.0
            break

    return {
        "bpf_Hz": bpf,
        "spl_thickness": spl_thick, "spl_loading": spl_load,
        "spl_bvi": spl_bvi, "spl_broadband": spl_bb,
        "spl_rotor": spl_rotor, "spl_motor": spl_motor, "spl_total": spl_total,
        "spl_a_weighted": spl_a, "detection_range_km": det_range,
        "tip_mach": tip_mach,
    }


def ir_signature(T_motor: float, T_batt: float, T_skin: float,
                 T_exhaust: float, T_ambient: float = 25.0) -> dict:
    """Stefan-Boltzmann IR radiance per component in MWIR / LWIR."""
    def emit(T_C, eps, area):
        T_K = T_C + 273.15
        power = eps * SIGMA_SB * area * T_K**4
        return power / np.pi   # W/sr  (Lambertian)

    ir_motor  = emit(T_motor,   EMIS_MOTOR, 0.2)
    ir_skin   = emit(T_skin,    EMIS_SKIN,  10.0)
    ir_exhaust = emit(T_exhaust, 0.9,       0.1)
    ir_batt   = emit(T_batt,    0.9,        0.0)  # internal, hidden

    T_back    = T_ambient + 273.15
    bg_emit   = SIGMA_SB * T_back**4 * 10.0 / np.pi
    net       = max(0, ir_motor + ir_skin + ir_exhaust - bg_emit * 0.8)

    swir  = 0.01 * net
    mwir  = 0.15 * net
    lwir  = 0.84 * net

    T_ref = 298.15
    T_avg = 0.5 * (T_motor + T_skin) + 273.15
    contrast = (T_avg / T_ref)**4

    return {
        "ir_motor_W_sr": ir_motor, "ir_skin_W_sr": ir_skin,
        "ir_swir_W_sr": swir, "ir_mwir_W_sr": mwir, "ir_lwir_W_sr": lwir,
        "ir_contrast": contrast,
    }


def rcs_compute(azimuth_deg: float, elevation_deg: float,
                rotor_rpm: float, nacelle_deg: float, time: float = 0.0) -> dict:
    """Aspect-dependent RCS at X-band and Ku-band with blade-flash."""
    ram = 10.0**(-RAM_DB / 10.0)
    az  = np.radians(azimuth_deg)
    el  = np.radians(elevation_deg)
    nac = np.radians(nacelle_deg)

    # Body
    horiz = 0.3 + 0.7 * abs(np.sin(az))
    vert  = (1.0 + 0.3 * abs(np.sin(el))) if elevation_deg < 0 else (0.8 + 0.2 * abs(np.sin(el)))
    shaping = 10.0**(-15.0 / 10.0) if abs(azimuth_deg) < 30 or abs(azimuth_deg - 180) < 30 else 1.0
    rcs_body = RCS_BODY_REF * horiz * vert * shaping * ram

    # Wing
    rcs_wing = RCS_WING_REF * (1.0 - 0.9 * abs(np.cos(az))) * (1.0 + 2.0 * abs(np.sin(el))) * ram

    # Nacelle
    side_f = abs(np.sin(az))
    front_f = abs(np.cos(az)) * np.cos(nac)
    vert_f  = abs(np.sin(el)) * np.sin(nac)
    rcs_nac = 2.0 * RCS_NAC_REF * float(np.clip(side_f + front_f + vert_f, 0.1, 3.0)) * ram

    # Rotor blade flash
    if rotor_rpm > 0:
        omega_r = rotor_rpm * 2.0 * np.pi / 60.0
        phase   = omega_r * time
        blade_phase = (phase % (2.0 * np.pi / N_BLADES)) / (2.0 * np.pi / N_BLADES) * 2.0 * np.pi
        flash   = abs(np.sin(blade_phase))
        disk_vis = abs(np.sin(nac)) * 0.5 + abs(np.cos(nac)) * abs(np.cos(az)) * 0.5
        rcs_rotor = N_ROTORS * N_BLADES * RCS_BLADE_REF * flash * disk_vis * ram * 2.0
    else:
        rcs_rotor = 0.0

    # Cavity
    front_vis = abs(np.cos(az))
    rcs_cav = RCS_NAC_REF * front_vis * (1.5 if abs(azimuth_deg) < 90 or abs(azimuth_deg) > 270 else 1.0) * ram * 3.0

    rcs_x = rcs_body + rcs_wing + rcs_nac + rcs_rotor + rcs_cav

    # Ku-band scaling (λ_Ku ≈ 0.020 m, λ_X ≈ 0.030 m)
    ku_scale = (0.030 / 0.020)**0.5
    rcs_ku = rcs_x * ku_scale

    def to_dbsm(sigma): return 10.0 * np.log10(sigma) if sigma > 0 else -100.0

    return {
        "rcs_x_m2": rcs_x, "rcs_x_dBsm": to_dbsm(rcs_x),
        "rcs_ku_m2": rcs_ku, "rcs_ku_dBsm": to_dbsm(rcs_ku),
        "rcs_body": rcs_body, "rcs_wing": rcs_wing,
        "rcs_nacelle": rcs_nac, "rcs_rotor": rcs_rotor,
    }


# ── Mission simulation ─────────────────────────────────────────────────────────

def simulate_mission(row: pd.Series, dt: float = 5.0) -> dict:
    """
    Simulate one mission from the planning dataset.

    Mission profile:
        hover_takeoff (30 s) → climb (ramp speed/alt) →
        transition_to_cruise (30 s) → cruise (bulk) →
        transition_to_hover (30 s) → hover_landing (30 s)

    Args:
        row: One row from planning dataset.
        dt: Simulation time step [s].

    Returns:
        Dictionary of vehicle-layer outputs for this mission.
    """
    # ── Extract mission parameters ─────────────────────────────────────────
    path_m        = float(row["path_length_m"])
    time_s_plan   = float(row["time_cost_s"])
    energy_wh_plan = float(row["energy_cost_wh"])
    alt_start     = float(row.get("start_alt_m", 100.0))
    alt_goal      = float(row.get("goal_alt_m", 150.0))
    n_wpts        = int(row["n_waypoints"])
    feasible      = int(row["feasible"])

    alt_cruise    = max(alt_start, alt_goal, 200.0)  # cruise altitude
    rho           = isa_density(alt_cruise)

    # ── Physics-based cruise speed assignment ─────────────────────────────
    # Range-optimal speed for parabolic drag polar: V_opt = sqrt(2W/(ρ·S·√(CD0/k)))
    k_ind         = 0.04   # induced drag factor
    V_opt         = np.sqrt(2.0 * WEIGHT / (rho * S_WING * np.sqrt(CD0_WING / k_ind)))
    V_opt         = float(np.clip(V_opt, 50.0, 90.0))

    # Path-length adjustment: longer missions fly closer to optimal/max-range speed
    path_km       = path_m / 1000.0
    V_path_adj    = float(np.clip(4.0 * (path_km - 10.0) / 10.0, -10.0, 15.0))

    # Risk-level adjustment: high-risk missions sprint to minimise exposure
    risk_adj      = 10.0 if int(row.get("risk_label", 0)) == 1 else 0.0

    # Deterministic noise seeded from row index so regeneration is reproducible
    rng_seed      = int(abs(hash(str(path_m) + str(alt_start))) % 2**31)
    rng           = np.random.RandomState(rng_seed)
    V_noise       = float(rng.normal(0.0, 5.0))

    V_cruise      = float(np.clip(V_opt + V_path_adj + risk_adj + V_noise, 40.0, 110.0))

    # ── Phase durations ────────────────────────────────────────────────────
    t_hover_to    = 30.0     # s  takeoff hover
    t_trans_to    = 30.0     # s  transition to cruise
    t_trans_land  = 30.0     # s  transition back
    t_hover_land  = 30.0     # s  landing hover

    # Cruise time: from path length minus hover/transition segments
    cruise_dist   = max(path_m - V_cruise * (t_trans_to + t_trans_land), 0.0)
    t_cruise      = cruise_dist / max(V_cruise, 1.0)
    t_total       = t_hover_to + t_trans_to + t_cruise + t_trans_land + t_hover_land

    # ── Hover performance (BEMT) ───────────────────────────────────────────
    T_per_rotor   = WEIGHT / N_ROTORS
    omega_h       = OMEGA_HOVER
    col_hover     = find_hover_collective(T_per_rotor, omega_h, rho)
    hover_res     = bemt_hover(omega_h, col_hover, rho)

    T_hover       = hover_res["thrust"] * N_ROTORS   # total [N]
    P_hover_shaft = hover_res["power"] * N_ROTORS    # shaft [W]
    P_hover_elec  = P_hover_shaft / (ETA_MOTOR_PEAK * ETA_MECH)
    v_induced_h   = hover_res["v_induced"]

    # Analytical figure of merit (actuator disk + Glauert profile drag integral)
    # FM = P_ideal / P_total  where P_ideal = T^1.5/sqrt(2ρA),
    # P_profile = (σ·Cd0/8)·ρ·A·V_tip³   (Prouty 2002, Ch.3)
    P_ideal_r  = T_per_rotor**1.5 / np.sqrt(2.0 * rho * A_ROTOR)
    P_prof_r   = (SIGMA * CD0_ROTOR / 8.0) * rho * A_ROTOR * (omega_h * R_ROTOR)**3
    FM         = float(np.clip(P_ideal_r / (P_ideal_r + P_prof_r), 0.0, 1.0))
    tip_speed_h   = omega_h * R_ROTOR                # m/s
    tip_mach_h    = tip_speed_h / SPEED_SOUND

    # ── Cruise performance ─────────────────────────────────────────────────
    cruise_res = cruise_aero(V_cruise, rho, alt_cruise)
    P_cruise_shaft = cruise_res["P_req"] / ETA_MOTOR_PEAK  # shaft power
    P_cruise_elec  = P_cruise_shaft / ETA_MECH
    omega_c        = OMEGA_CRUISE
    tip_speed_c    = omega_c * R_ROTOR
    tip_mach_c     = tip_speed_c / SPEED_SOUND

    # Torque per rotor in cruise
    torque_cruise = P_cruise_shaft / (N_ROTORS * omega_c) if omega_c > 0 else 0

    # ── Transition: average of hover and cruise power ──────────────────────
    P_trans_elec   = 0.6 * P_hover_elec + 0.4 * P_cruise_elec
    nacelle_avg    = 45.0   # degrees during transition

    # ── Motor performance ──────────────────────────────────────────────────
    T_wind_init  = 50.0
    # Motor torque = shaft power at rotor / (gear ratio × rotor speed)
    torque_motor_hover  = P_hover_shaft / N_ROTORS / max(omega_h * GEAR_RATIO, 0.01)
    torque_motor_cruise = P_cruise_shaft / N_ROTORS / max(omega_c * GEAR_RATIO, 0.01)
    motor_hover  = motor_efficiency(torque_motor_hover,  omega_h * GEAR_RATIO, T_wind_init)
    motor_cruise = motor_efficiency(torque_motor_cruise, omega_c * GEAR_RATIO, T_wind_init)

    # ── Battery simulation ─────────────────────────────────────────────────
    soc       = SOC_INIT
    soh       = 1.0
    v_rc1     = 0.0
    v_rc2     = 0.0
    T_batt    = 25.0
    T_motor   = 50.0
    T_ambient = 20.0

    # Simplified phase loop (large dt for speed)
    phases = [
        ("hover",       t_hover_to,   P_hover_elec,  omega_h,  90.0),
        ("transition",  t_trans_to,   P_trans_elec,  (omega_h+omega_c)/2, nacelle_avg),
        ("cruise",      t_cruise,     P_cruise_elec, omega_c,  0.0),
        ("transition2", t_trans_land, P_trans_elec,  (omega_h+omega_c)/2, nacelle_avg),
        ("hover2",      t_hover_land, P_hover_elec,  omega_h,  90.0),
    ]

    records = {
        "soc_list": [], "T_batt_list": [], "T_motor_list": [],
        "power_list": [], "V_t_list": [], "c_rate_list": [],
    }
    soc_peak_power = SOC_INIT
    peak_c_rate    = 0.0
    peak_T_batt    = T_batt
    peak_T_motor   = T_motor
    I_peak         = 0.0
    energy_consumed = 0.0

    for phase_name, t_phase, P_elec, omega_ph, nac_deg in phases:
        n_steps = max(1, int(t_phase / dt))
        dt_eff  = t_phase / n_steps
        for _ in range(n_steps):
            # Battery step
            bs = battery_step(soc, soh, v_rc1, v_rc2, T_batt,
                               P_elec, dt_eff, T_ambient)
            soc    = bs["soc"]
            soh    = bs["soh"]
            v_rc1  = bs["v_rc1"]
            v_rc2  = bs["v_rc2"]
            T_batt = bs["T_batt"]
            energy_consumed += bs["energy_wh"]

            # Motor thermal (simplified 1st-order)
            P_phase_shaft = P_hover_shaft if "hover" in phase_name else \
                            (P_cruise_shaft if "cruise" in phase_name else P_trans_elec * ETA_MOTOR_PEAK)
            torque_ph = P_phase_shaft / N_ROTORS / max(omega_ph * GEAR_RATIO, 0.01)
            m_res     = motor_efficiency(torque_ph, omega_ph * GEAR_RATIO, T_motor)
            Q_motor   = m_res["P_loss"] * N_ROTORS / 2.0  # per motor
            T_motor_ss = T_ambient + Q_motor / 10.0
            T_motor    = T_motor + (T_motor_ss - T_motor) * (dt_eff / (MOTOR_C_TH / 10.0))
            T_motor    = float(np.clip(T_motor, T_ambient, T_MOTOR_MAX))

            records["soc_list"].append(soc)
            records["T_batt_list"].append(T_batt)
            records["T_motor_list"].append(T_motor)
            records["power_list"].append(P_elec)
            records["V_t_list"].append(bs["V_terminal"])
            records["c_rate_list"].append(bs["c_rate"])

            peak_c_rate  = max(peak_c_rate, bs["c_rate"])
            peak_T_batt  = max(peak_T_batt, T_batt)
            peak_T_motor = max(peak_T_motor, T_motor)
            I_peak       = max(I_peak, bs["I"])

    soc_final = soc
    soh_final = soh

    # ── Acoustic ───────────────────────────────────────────────────────────
    rotor_rpm_h = omega_h * 60.0 / (2.0 * np.pi)
    rotor_rpm_c = omega_c * 60.0 / (2.0 * np.pi)

    ac_hover  = acoustic_spl(rotor_rpm_h, T_hover, P_hover_shaft,
                              tip_mach_h, 90.0, 0.0, 0.0)
    ac_cruise = acoustic_spl(rotor_rpm_c, WEIGHT, P_cruise_shaft,
                              tip_mach_c, 0.0, V_cruise, 0.0)

    # ── IR signature ──────────────────────────────────────────────────────
    ir_hover  = ir_signature(peak_T_motor, peak_T_batt, T_ambient + 5, T_ambient + 15)
    ir_cruise = ir_signature(T_motor, T_batt, T_ambient + 2 + 0.5 * V_cruise / 10.0,
                              T_ambient + 10, T_ambient)

    # ── RCS ───────────────────────────────────────────────────────────────
    # Threat azimuth drawn from mission geometry (angle from start to goal)
    import math
    dlat  = float(row.get("goal_lat", 28.85)) - float(row.get("start_lat", 28.78))
    dlon  = float(row.get("goal_lon", 77.05)) - float(row.get("start_lon", 77.08))
    az_threat = (math.degrees(math.atan2(dlon, dlat)) + 360) % 360  # 0–360 deg
    el_threat = rng.uniform(-15.0, 15.0)   # elevation jitter from ground threat

    rcs_hover  = rcs_compute(az_threat, el_threat, rotor_rpm_h, 90.0)
    rcs_cruise = rcs_compute(az_threat, el_threat, rotor_rpm_c, 0.0)
    rcs_nose   = rcs_compute(0.0, 0.0, rotor_rpm_c, 0.0)   # nose-on (stealthiest)

    # ── Control: nacelle transition timing ────────────────────────────────
    tilt_rate_dps = 10.0   # °/s  (NacelleConfig default)
    nacelle_trans_s = 90.0 / tilt_rate_dps   # 9 s for full 90° tilt

    # ── Propulsive efficiency (cruise) ────────────────────────────────────
    eta_prop = P_cruise_shaft / P_cruise_elec if P_cruise_elec > 0 else 0.0

    # ── Build output record ────────────────────────────────────────────────
    return {
        # Mission identifiers
        "path_length_m":        path_m,
        "mission_time_s":       t_total,
        "mission_time_plan_s":  time_s_plan,
        "cruise_speed_ms":      V_cruise,
        "cruise_altitude_m":    alt_cruise,
        "n_waypoints":          n_wpts,
        "feasible":             feasible,

        # Phase breakdown
        "hover_time_s":        t_hover_to + t_hover_land,
        "transition_time_s":   t_trans_to + t_trans_land,
        "cruise_time_s":       t_cruise,
        "hover_fraction":      (t_hover_to + t_hover_land) / max(t_total, 1),
        "cruise_fraction":     t_cruise / max(t_total, 1),

        # Rotor / BEMT (hover)
        "collective_hover_deg": float(np.degrees(col_hover)),
        "rotor_rpm_hover":     float(rotor_rpm_h),
        "thrust_per_rotor_N":  hover_res["thrust"],
        "total_thrust_hover_N": T_hover,
        "v_induced_hover_ms":  v_induced_h,
        "CT_hover":            hover_res["CT"],
        "figure_of_merit":     FM,
        "tip_speed_hover_ms":  tip_speed_h,
        "tip_mach_hover":      tip_mach_h,
        "power_hover_shaft_W": P_hover_shaft,
        "power_hover_elec_W":  P_hover_elec,

        # Cruise aerodynamics
        "cruise_speed_ms":     V_cruise,
        "CL_cruise":           cruise_res["CL"],
        "CD_cruise":           cruise_res["CD"],
        "LD_ratio":            cruise_res["LD"],
        "q_cruise_Pa":         cruise_res["q"],
        "Re_cruise":           cruise_res.get("Re", 0),
        "Mach_cruise":         cruise_res.get("Mach", 0),
        "CD_wing":             cruise_res.get("CD_wing", 0),
        "CD_fuselage":         cruise_res.get("CD_fuse", 0),
        "e_oswald":            cruise_res.get("e_oswald", 0),
        "thrust_cruise_N":     cruise_res["T_req"],
        "power_cruise_shaft_W": P_cruise_shaft,
        "power_cruise_elec_W": P_cruise_elec,
        "propulsive_efficiency": eta_prop,
        "tip_mach_cruise":     tip_mach_c,

        # Motor
        "motor_eta_hover":     motor_hover["eta"],
        "motor_eta_cruise":    motor_cruise["eta"],
        "motor_peak_torque_Nm": max(torque_motor_hover, torque_motor_cruise),
        "motor_peak_current_A": max(motor_hover.get("i_q", 0), motor_cruise.get("i_q", 0)),
        "motor_peak_temp_C":   peak_T_motor,
        "motor_temp_final_C":  T_motor,

        # Battery
        "soc_initial":         SOC_INIT,
        "soc_final":           soc_final,
        "soc_minimum":         min(records["soc_list"]) if records["soc_list"] else soc_final,
        "soh_final":           soh_final,
        "energy_consumed_wh":  energy_consumed,
        "energy_consumed_plan_wh": energy_wh_plan,
        "energy_remaining_wh": soc_final * CAP_WH_PACK,
        "battery_peak_temp_C": peak_T_batt,
        "battery_temp_final_C": T_batt,
        "battery_peak_current_A": I_peak,
        "battery_peak_crate":  peak_c_rate,
        "battery_V_min":       min(records["V_t_list"]) if records["V_t_list"] else 0.0,
        "battery_V_final":     records["V_t_list"][-1] if records["V_t_list"] else 0.0,
        "pack_capacity_wh":    CAP_WH_PACK,
        "range_km_remaining":  soc_final * CAP_WH_PACK / max(P_cruise_elec / 1000.0, 0.1) * V_cruise / 3600.0,

        # Thermal
        "motor_temp_rise_C":   peak_T_motor - T_ambient,
        "battery_temp_rise_C": peak_T_batt - 25.0,
        "thermal_margin_motor_C": T_MOTOR_MAX - peak_T_motor,

        # Nacelle
        "nacelle_tilt_time_s":   nacelle_trans_s,
        "nacelle_transition_s":  t_trans_to + t_trans_land,

        # Acoustic (hover)
        "spl_hover_total_dB":   ac_hover["spl_total"],
        "spl_hover_a_dB":       ac_hover["spl_a_weighted"],
        "spl_hover_rotor_dB":   ac_hover["spl_rotor"],
        "spl_hover_loading_dB": ac_hover["spl_loading"],
        "spl_hover_bvi_dB":     ac_hover["spl_bvi"],
        "bpf_hover_Hz":         ac_hover["bpf_Hz"],
        "detection_range_hover_km": ac_hover["detection_range_km"],

        # Acoustic (cruise)
        "spl_cruise_total_dB":  ac_cruise["spl_total"],
        "spl_cruise_a_dB":      ac_cruise["spl_a_weighted"],
        "spl_cruise_rotor_dB":  ac_cruise["spl_rotor"],
        "bpf_cruise_Hz":        ac_cruise["bpf_Hz"],
        "detection_range_cruise_km": ac_cruise["detection_range_km"],

        # IR (hover)
        "ir_mwir_hover_W_sr":   ir_hover["ir_mwir_W_sr"],
        "ir_lwir_hover_W_sr":   ir_hover["ir_lwir_W_sr"],
        "ir_contrast_hover":    ir_hover["ir_contrast"],
        "ir_motor_hover_W_sr":  ir_hover["ir_motor_W_sr"],

        # IR (cruise)
        "ir_mwir_cruise_W_sr":  ir_cruise["ir_mwir_W_sr"],
        "ir_lwir_cruise_W_sr":  ir_cruise["ir_lwir_W_sr"],
        "ir_contrast_cruise":   ir_cruise["ir_contrast"],

        # RCS (hover, broadside)
        "rcs_hover_x_m2":       rcs_hover["rcs_x_m2"],
        "rcs_hover_x_dBsm":     rcs_hover["rcs_x_dBsm"],
        "rcs_hover_ku_m2":      rcs_hover["rcs_ku_m2"],

        # RCS (cruise, broadside)
        "rcs_cruise_x_m2":      rcs_cruise["rcs_x_m2"],
        "rcs_cruise_x_dBsm":    rcs_cruise["rcs_x_dBsm"],
        "rcs_cruise_ku_m2":     rcs_cruise["rcs_ku_m2"],
        "rcs_rotor_x_m2":       rcs_cruise["rcs_rotor"],

        # RCS (nose-on)
        "rcs_noseon_x_m2":      rcs_nose["rcs_x_m2"],
        "rcs_noseon_x_dBsm":    rcs_nose["rcs_x_dBsm"],

        # Mission feasibility / risk passthrough
        "risk_label":           int(row["risk_label"]),
        "max_combined_threat":  float(row["max_combined_threat"]),
        "energy_cost_wh_plan":  float(row["energy_cost_wh"]),
        "threat_cost":          float(row["threat_cost"]),
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print(f"Loading planning dataset: {SRC}")
    df_plan = pd.read_parquet(SRC)
    n = len(df_plan)
    print(f"  {n} missions to simulate")

    results = []
    for i, (_, row) in enumerate(df_plan.iterrows()):
        if i % 200 == 0:
            print(f"  [{i+1}/{n}] simulating ...")
        try:
            rec = simulate_mission(row)
            results.append(rec)
        except Exception as exc:
            print(f"  [WARN] row {i} failed: {exc}")
            results.append({"feasible": 0, "path_length_m": row.get("path_length_m", 0)})

    df_out = pd.DataFrame(results)
    print(f"\nVehicle dataset: {len(df_out)} rows × {len(df_out.columns)} columns")
    print(f"Columns: {list(df_out.columns)}")

    # ── Save ─────────────────────────────────────────────────────────────────
    p = OUT / "vehicle_dataset.parquet"
    df_out.to_parquet(p, index=False)
    print(f"  [OK] Parquet    {p.stat().st_size/1024:.1f} KB  -> {p}")

    p = OUT / "vehicle_dataset.csv"
    df_out.to_csv(p, index=False)
    print(f"  [OK] CSV        {p.stat().st_size/1024:.1f} KB")

    # NumPy (numeric only)
    num_cols = df_out.select_dtypes(include="number").columns.tolist()
    p = OUT / "vehicle_dataset.npz"
    np.savez_compressed(p, **{c: df_out[c].values for c in num_cols})
    print(f"  [OK] NumPy npz  {p.stat().st_size/1024:.1f} KB")

    # Summary statistics
    print("\nKey statistics:")
    cols_summary = [
        "hover_fraction", "cruise_fraction",
        "power_hover_elec_W", "power_cruise_elec_W",
        "soc_final", "energy_consumed_wh",
        "battery_peak_temp_C", "motor_peak_temp_C",
        "figure_of_merit", "LD_ratio",
        "spl_hover_total_dB", "spl_cruise_total_dB",
        "detection_range_hover_km",
        "rcs_cruise_x_dBsm", "rcs_noseon_x_dBsm",
        "ir_mwir_hover_W_sr",
    ]
    for col in cols_summary:
        if col in df_out.columns:
            vals = df_out[col].dropna()
            if len(vals):
                print(f"  {col:<40s}  mean={vals.mean():.3g}  std={vals.std():.3g}  "
                      f"min={vals.min():.3g}  max={vals.max():.3g}")

    print(f"\nAll outputs written to {OUT}/")


if __name__ == "__main__":
    main()
