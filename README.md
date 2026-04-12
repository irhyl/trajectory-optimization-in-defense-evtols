# Trajectory Optimization in Defense eVTOLs

A full-stack autonomous mission planning and control framework for defense electric vertical take-off and landing (eVTOL) aircraft operating in contested airspace. The system integrates real geospatial perception, multi-objective trajectory optimization, cascaded flight control, and six-degree-of-freedom vehicle dynamics into a layered architecture generating publication-quality datasets for NeurIPS Datasets & Benchmarks submission.

---

## Table of Contents

1. [Research Motivation](#1-research-motivation)
2. [Architecture Overview](#2-architecture-overview)
3. [Layer Descriptions](#3-layer-descriptions)
4. [Repository Structure](#4-repository-structure)
5. [Installation](#5-installation)
6. [Running the Pipeline](#6-running-the-pipeline)
7. [Dataset Descriptions](#7-dataset-descriptions)
8. [Visualizations](#8-visualizations)
9. [Key Results](#9-key-results)
10. [Known Limitations](#10-known-limitations)
11. [NeurIPS Submission Context](#11-neurips-submission-context)
12. [References](#12-references)
13. [Citation](#13-citation)

---

## 1. Research Motivation

Autonomous eVTOL aircraft are emerging as critical assets in defense logistics, ISR (intelligence, surveillance, and reconnaissance), and medical evacuation in contested environments. Unlike civilian urban air mobility, defense missions require simultaneous optimization of:

- **Energy efficiency** — battery-constrained range with no ground refueling
- **Mission time** — time-critical deliveries under dynamic threat windows
- **Threat exposure minimization** — radar cross-section, acoustic, and infrared signature management against ground-based air defenses (GBAD)
- **Terrain following** — low-altitude NOE (nap-of-earth) flight to exploit terrain masking

Existing trajectory planners address at most two of these objectives. This work presents the first open dataset and framework that jointly models all four within a real geospatial environment, with end-to-end simulation from mission planning through closed-loop flight control.

---

## 2. Architecture Overview

The system implements a four-layer autonomous mission stack:

```
┌─────────────────────────────────────────────────────────────────┐
│                     MISSION INPUT                               │
│              (start/goal lat-lon, mission profile)              │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 1: PERCEPTION                                            │
│  ─────────────────────                                          │
│  • Terrain: NASA SRTM30 elevation, slope, roughness             │
│  • Wind: Open-Meteo GFS forecast (u, v, w components)          │
│  • Obstacles: OpenStreetMap Overpass API + building geometry    │
│  • Threat: Analytical SAM detection probability (Mahafza 2005) │
│  • Output: 4D cost field C(φ, λ, h, t) — 1,057,714 grid cells │
└─────────────────────────┬───────────────────────────────────────┘
                          │  cost field
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 2: PLANNING                                              │
│  ─────────────────────                                          │
│  • Sampling: RRT* (Karaman & Frazzoli 2011) — collision-free   │
│  • Optimization: NSGA-III (Deb & Jain 2014) — Pareto-optimal  │
│  • Objectives: energy [Wh], time [s], threat exposure [0-1]    │
│  • Constraints: terrain clearance ≥ 30 m, speed ∈ [20, 100]   │
│  • Output: 2,000 Pareto-optimal trajectory records             │
└─────────────────────────┬───────────────────────────────────────┘
                          │  waypoints + speed profile
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 3: CONTROL   (50 Hz cascaded PID)                        │
│  ─────────────────────                                          │
│  • Outer loop: position/altitude/heading → attitude commands    │
│  • Inner loop: attitude → body angular rates                    │
│  • Innermost: rates → moments via rate PID                      │
│  • Allocation: pseudo-inverse mixing matrix B^† → 4 motors     │
│  • Phase scheduler: TAKEOFF→HOVER→TRANS→CRUISE→LAND            │
│  • Output: 2,000 closed-loop simulation records, 76 metrics     │
└─────────────────────────┬───────────────────────────────────────┘
                          │  control inputs
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 4: VEHICLE DYNAMICS  (6-DoF)                             │
│  ─────────────────────                                          │
│  • Rigid body: Newton-Euler equations with gyroscopic terms     │
│  • Propulsion: BEMT rotor model, tilted-nacelle transition      │
│  • Energy: electrochemical battery model with Peukert effect    │
│  • Signatures: acoustic (SPL), infrared (LWIR), RCS (Mie/PO)   │
│  • Output: 2,000 vehicle state records, 95 physics features     │
└─────────────────────────────────────────────────────────────────┘
```

Data flows strictly top-down. Each layer's outputs are the next layer's inputs.

---

## 3. Layer Descriptions

### 3.1 Perception Layer

Builds a 4-dimensional cost field over the operating area (Delhi-NCR region, India: 28.7°–29.0°N, 76.9°–77.4°E) at three altitude bands (50 m, 150 m, 300 m AGL).

| Sub-layer | Source | Resolution | Key Outputs |
|-----------|--------|------------|-------------|
| Terrain | NASA SRTM30 via Open-Elevation API | ~90 m | `elev_m`, `slope_deg`, `roughness_m` |
| Wind | Open-Meteo GFS (hourly) | Point forecast | `wind_u/v/w_mps`, `turbulence_intensity` |
| Obstacle | OpenStreetMap Overpass API | Object-level | `nearest_obstacle_dist_m`, `obstacle_type` |
| Threat | Analytical SAM model (3 emitter types) | Continuous | `T1/T2/T3_detect_prob`, `combined_threat_prob` |
| Fusion | Weighted harmonic composite | — | `fused_cost ∈ [0, 1]` |

**Note on threat saturation:** The operating area in the modeled scenario is assumed to be covered by overlapping SAM networks (Type-1: long-range surveillance, Type-2: medium-range fire control, Type-3: short-range MANPADS). The analytical Mahafza detection probability model yields `combined_threat_prob ≈ 1.0` everywhere at operational altitudes — not a modeling error, but a reflection of the contested airspace assumption. The planner therefore optimizes *exposure time* and *aspect angle* rather than seeking threat-free corridors.

### 3.2 Planning Layer

Two-phase trajectory generation:

**Phase 1 — RRT* (feasibility):** Karaman & Frazzoli (2011) asymptotically optimal sampling-based planner operating in the perception cost field. Guarantees obstacle avoidance and terrain clearance.

**Phase 2 — NSGA-III (Pareto optimization):** Deb & Jain (2014) many-objective genetic algorithm with three objectives:
- `f₁ = energy_cost_wh` — momentum theory hover power + parasitic drag in cruise
- `f₂ = time_cost_s` — path-length / cruise-speed + hover durations  
- `f₃ = threat_cost` — integrated detection probability along trajectory

Knee-point selection from Pareto front: minimize L₂ distance to utopia point after normalization.

### 3.3 Control Layer

Cascaded PID architecture at 50 Hz:

```
Position cmd → [Position PID] → Velocity cmd
Velocity cmd → [Velocity PID] → Attitude cmd (roll/pitch/yaw)
Attitude cmd → [Attitude PID] → Body rate cmd (p, q, r)
Rate cmd     → [Rate PID]     → Moments (Mx, My, Mz)
Moments      → [Mixer B^†]    → Motor thrusts T₀...T₃
```

6-DoF simplified plant with Euler kinematics, gyroscopic cross-coupling in rotational dynamics, and linear drag model.

Phase schedule (per mission): TAKEOFF → HOVER → TRANS1 → CRUISE → TRANS2 → HOVER2 → LAND

### 3.4 Vehicle Layer

Newton-Euler 6-DoF dynamics with:
- **Propulsion:** Blade Element Momentum Theory (BEMT) rotor model; tilted nacelle tiltrotor transition model
- **Aerodynamics:** Wing lift/drag (finite span correction), fuselage parasite drag, sideslip yaw stability
- **Energy:** Electrochemical battery with Peukert capacity de-rating, internal resistance, thermal coupling
- **Acoustic signature:** A-weighted SPL from rotor thrust noise model
- **Infrared signature:** LWIR radiant intensity from motor/rotor thermal model
- **RCS:** Physical optics cross-section estimation for key aspect angles

---

## 4. Repository Structure

```
trajectory-optimization-in-defense-evtols/
├── src/evtol/                    # Core library (137 Python modules)
│   ├── perception/               # Terrain, wind, obstacle, threat, fusion
│   ├── planning/                 # RRT*, NSGA-III, trajectory, constraints
│   ├── control/                  # Cascaded PID, modes, motor allocation
│   └── vehicle/                  # Dynamics, propulsion, energy, signatures
│
├── scripts/                      # Pipeline runners (generate datasets + visuals)
│   ├── perception/dataset.py     # Perception dataset generator (1,057,714 rows)
│   ├── perception/visuals.py     # Perception visualizations (21 figures)
│   ├── planning/dataset.py       # Planning dataset generator (2,000 records)
│   ├── planning/visualize.py     # Planning visualizations (38 figures)
│   ├── vehicle/dataset.py        # Vehicle dynamics simulator (2,000 records)
│   ├── vehicle/visualize.py      # Vehicle visualizations (20 figures + SVG)
│   ├── control/dataset.py        # Control simulation (2,000 closed-loop records)
│   └── control/visualize.py      # Control visualizations (15 figures)
│
├── outputs/                      # Generated datasets
│   ├── perception_dataset/exports/   # 1,057,714-row perception grid
│   ├── planning_dataset/             # 2,000-row trajectory plans
│   ├── vehicle/                      # 2,000-row vehicle states (95 features)
│   └── control/                      # 2,000-row control records (76 features)
│
├── visuals/                      # Generated figures (210 files: PNG + PDF)
│   ├── perception/               # 21 figures (maps, distributions, profiles)
│   ├── planning/                 # 38 figures (spatial, cost, Pareto, PCA)
│   ├── vehicle/                  # 20 figures + SVG (dynamics, energy, signatures)
│   └── control/                  # 15 figures (tracking, moments, motor alloc)
│
├── doc/                          # Technical documentation
│   ├── introduction.md           # Research context and motivation
│   ├── architecture.md           # System design and data flow
│   ├── perception_layer.md       # Perception math, APIs, physics
│   ├── planning_layer.md         # RRT*, NSGA-III, constraints derivation
│   ├── vehicle_layer.md          # 6-DoF dynamics, propulsion, signatures
│   ├── control_layer.md          # Cascaded PID, mixing matrix, phase scheduler
│   ├── data_guide.md             # Dataset schemas, column definitions
│   └── research_limitations.md  # Known issues and NeurIPS submission notes
│
├── pyproject.toml                # Package configuration
└── README.md                     # This file
```

---

## 5. Installation

**Requirements:** Python 3.11+, pip

```bash
git clone https://github.com/your-org/trajectory-optimization-in-defense-evtols
cd trajectory-optimization-in-defense-evtols

# Create virtual environment
python -m venv .venv
source .venv/bin/activate          # Linux/macOS
.venv\Scripts\activate             # Windows

# Install package and dependencies
pip install -e .
```

**Key dependencies** (from `pyproject.toml`):
- `numpy`, `scipy`, `pandas` — numerics and data
- `matplotlib`, `seaborn` — visualization
- `pyarrow` — Parquet I/O
- `requests` — external API calls (elevation, wind, OSM)

**External data sources** (fetched at runtime, cached locally):
- NASA SRTM terrain: Open-Elevation API
- Wind forecasts: Open-Meteo GFS
- Obstacle data: OpenStreetMap Overpass API

---

## 6. Running the Pipeline

Each layer must be run in order — outputs of layer N feed layer N+1.

### Step 1: Perception Dataset

```bash
python scripts/perception/dataset.py
# Output: outputs/perception_dataset/exports/perception_full_dataset.parquet
# Size: ~44 MB, 1,057,714 rows × 28 columns

python scripts/perception/visuals.py
# Output: visuals/perception/ (21 PNG + PDF figures)
```

### Step 2: Planning Dataset

```bash
python scripts/planning/dataset.py
# Output: outputs/planning_dataset/test_final.parquet
# Size: ~270 KB, 2,000 rows × 25 columns

python scripts/planning/visualize.py
# Output: visuals/planning/ (38 PNG + PDF figures)
```

### Step 3: Vehicle Dataset

```bash
python scripts/vehicle/dataset.py
# Output: outputs/vehicle/vehicle_dataset.parquet
# Size: ~911 KB, 2,000 rows × 95 columns

python scripts/vehicle/visualize.py
# Output: visuals/vehicle/ (20 PNG + PDF + SVG figures)
```

### Step 4: Control Dataset

```bash
python scripts/control/dataset.py
# Output: outputs/control/control_dataset.parquet
# Size: ~990 KB, 2,000 rows × 76 columns
# Runtime: ~12 minutes (50 Hz simulation, 2000 missions)

python scripts/control/visualize.py
# Output: visuals/control/ (15 PNG + PDF figures)
```

---

## 7. Dataset Descriptions

### 7.1 Perception Dataset

**File:** `outputs/perception_dataset/exports/perception_full_dataset.parquet`
**Size:** 1,057,714 rows × 28 columns (44 MB Parquet / 213 MB CSV)

Geographic coverage: 28.7°–29.0°N, 76.9°–77.4°E (Delhi-NCR region)
Altitude bands: 50 m, 150 m, 300 m AGL
Grid resolution: ~200 m horizontal

| Column | Unit | Description |
|--------|------|-------------|
| `lat`, `lon` | °(WGS-84) | Grid cell center coordinates |
| `alt_m` | m AGL | Altitude band |
| `elev_m` | m MSL | Terrain elevation (SRTM30) |
| `slope_deg` | ° | Terrain slope |
| `roughness_m` | m | Terrain roughness index |
| `wind_u/v/w_mps` | m/s | Wind velocity components (NED) |
| `wind_speed_mps` | m/s | Wind speed magnitude |
| `turbulence_intensity` | — | Normalized turbulence [0,1] |
| `nearest_obstacle_dist_m` | m | Range to nearest OSM obstacle |
| `T1/T2/T3_detect_prob` | — | SAM detection probability by type |
| `combined_threat_prob` | — | Combined threat [0,1] |
| `terrain/wind/obstacle/threat/energy/fused_cost` | — | Cost components [0,1] |

### 7.2 Planning Dataset

**File:** `outputs/planning_dataset/test_final.parquet`
**Size:** 2,000 rows × 25 columns (270 KB)

Each row represents one Pareto-optimal mission trajectory from start to goal.

| Column | Unit | Description |
|--------|------|-------------|
| `start/goal_lat/lon/alt_m` | °, m | Mission endpoints |
| `waypoint_lats/lons/alts` | `\|`-separated | Intermediate waypoints |
| `n_waypoints` | — | Total waypoint count |
| `path_length_m` | m | Total arc length |
| `planning_time_s` | s | RRT* wall-clock budget |
| `time_cost_s` | s | Estimated flight duration |
| `energy_cost_wh` | Wh | Estimated energy consumption |
| `threat_cost` | — | Integrated threat exposure [0,1] |
| `fused_cost_mean` | — | Mean perception cost along path |
| `feasible` | {0,1} | Constraint satisfaction flag |
| `risk_label` | {0,1} | Low (0) / high (1) risk classification |

### 7.3 Vehicle Dataset

**File:** `outputs/vehicle/vehicle_dataset.parquet`
**Size:** 2,000 rows × 95 columns (911 KB)

Physics-based simulation of each planned trajectory through 6-DoF dynamics.

Key feature groups:

| Group | Examples |
|-------|---------|
| Mission profile | `cruise_speed_ms`, `cruise_altitude_m`, `mission_time_s` |
| Rotor/propulsion | `T_hover_N`, `P_hover_kW`, `rpm_cruise`, BEMT efficiency |
| Aerodynamics | `CL_cruise`, `CD_cruise`, `L/D_ratio`, `drag_parasitic_N` |
| Energy/battery | `energy_total_wh`, `soc_initial/final`, `pack_capacity_wh` |
| Thermal | `motor_temp_rise_K`, `winding_temp_C` |
| Acoustic | `spl_hover_a_dB`, `spl_cruise_a_dB` |
| Infrared | `ir_radiance_hover`, `ir_radiance_cruise` |
| RCS | `rcs_cruise_x_dBsm`, `rcs_cruise_z_dBsm` |
| Labels | `risk_label`, `feasible` |

### 7.4 Control Dataset

**File:** `outputs/control/control_dataset.parquet`
**Size:** 2,000 rows × 76 columns (990 KB)

Closed-loop 50 Hz simulation results for each mission.

| Group | Examples | Typical Values |
|-------|---------|---------------|
| Altitude tracking | `alt_error_mean_m`, `alt_error_rms_m` | 10.4 ± 5.5 m |
| Attitude tracking | `att_error_mean_rad`, `att_error_rms_rad` | 0.0013 ± 0.0004 rad |
| Thrust | `thrust_cmd_mean_N`, `thrust_cmd_max_N` | 2613 ± 20 N |
| Motor allocation | `motor_T_m0–3_mean_N`, `motor_T_balance_N` | ~653 N/motor |
| PWM | `pwm_mean_us`, `pwm_utilisation_pct` | 54.4% |
| ITAE | `itae_pos`, `itae_alt`, `itae_att` | Quality integrals |
| Mission | `mission_abort`, `n_saturations`, `n_stall_events` | 0 aborts |
| Energy | `energy_consumed_wh`, `soc_final` | 5903 Wh, SOC 0.75 |

---

## 8. Visualizations

All figures are at 300 DPI, available as PNG and PDF in `visuals/`.

| Module | Figures | Key Content |
|--------|---------|-------------|
| Perception (21) | A1–A6, B1–B4, C1–C3, D1–D2, E1–E3, F1–F2, G1, H1 | Terrain maps, wind quiver, threat contours, cost distributions, altitude profiles, SAM range rings, wind rose |
| Planning (38+) | A1–A3, B1–B3, C1–C4, D1–D3, E1–E3, F1–F3, G1–G3, H1–H2, I1–I2, J1–J3, K1–K3 | Spatial density, Pareto fronts, cost CDFs, PCA biplot, parallel coordinates |
| Vehicle (20+) | A01, B01, C01/C07, D01/D07/D08, E01, F01/F07, G01, H01/H07, I01–I02, J01, K01–K04 | Mission profiles, BEMT curves, energy envelopes, SOC profiles, signature budgets, RCS polar patterns, multi-signature Pareto 3D |
| Control (15) | A01–A02, B01–B02, C01, D01–D02, E01–E02, F01, G01, H01–H02, Z01–Z02 | Tracking distributions, saturation analysis, phase fractions, motor balance, ITAE, correlation heatmaps |

---

## 9. Key Results

### Dataset Scale

| Dataset | Rows | Features | Size |
|---------|------|----------|------|
| Perception grid | 1,057,714 | 28 | 44 MB |
| Planning trajectories | 2,000 | 25 | 270 KB |
| Vehicle simulations | 2,000 | 95 | 911 KB |
| Control simulations | 2,000 | 76 | 990 KB |

### Control Performance Summary

| Metric | Mean | Std | Physical Interpretation |
|--------|------|-----|------------------------|
| Altitude error | 10.4 m | 5.5 m | Acceptable for NOE flight |
| Attitude error | 0.0013 rad | 0.0004 rad | Excellent (0.07°) |
| Thrust mean | 2,613 N | 20 N | 30% above hover weight (2,000 N) for margin |
| Mission abort rate | 0.0% | — | Controller stable across all 2,000 missions |
| Motor saturations | 0.5 | — | Near-zero, excellent allocation headroom |
| SOC final | 0.752 | — | 24.8% battery consumed on average |

### Vehicle Physics Summary

| Metric | Range | Notes |
|--------|-------|-------|
| Cruise speed | 48.5–94.4 m/s | Matches NSGA-III feasibility bounds |
| Cruise altitude | 200 m (fixed) | Single-altitude scenario |
| Acoustic (hover) | 92.84 dB(A) | Typical tiltrotor hover SPL |
| RCS (cruise, X-band) | −8.84 ± 0.46 dBsm | Small rotorcraft signature |

---

## 10. Known Limitations

### 10.1 Threat Saturation

`combined_threat_prob = 1.000` for all grid cells. The modeled scenario places three SAM systems whose coverage areas fully overlap the operating region. This is intentional (contested airspace) but means the threat cost provides no spatial gradient for the planner. **Future work:** Scenario with SAM coverage gaps to enable threat-aware routing with meaningful differentiation.

### 10.2 Position Error Metric

`pos_error_mean_m ≈ 51,000 m` appears alarming but is an artifact of the reference signal definition: during CRUISE phase, the reference position accumulates as `x_ref = v_cruise × t` from mission start (t=0), not from cruise-phase start. The altitude controller (10.4 m error) and attitude controller (0.001 rad) demonstrate the controller is functional; the position metric should be interpreted as waypoint-relative error only during cruise.

### 10.3 Settling Time Ceiling

`settling_time_mean = 20.0 s` for most missions. The settling criterion (alt_error < 2.0 m AND vel_error < 1.0 m/s simultaneously) is strict. The individual altitude settling is demonstrably good (10.4 m mean). The metric ceiling reflects that simultaneous satisfaction of both criteria within 20 s is rare given the sequential cascaded response.

### 10.4 Acoustic Variance

`spl_hover_a_dB = 92.84 ± 0.00 dB` — zero variance. The acoustic model uses rotor thrust as the primary variable; since all missions hover at similar thrust (~2000 N), SPL variance is sub-0.01 dB. A more detailed model incorporating blade passage frequency, RPM variation, and terrain shielding would show meaningful variation.

### 10.5 Simulation Only

All results are simulation-based. No hardware-in-the-loop (HITL) or physical flight test data are included. Sim-to-real transfer gap is unquantified.

---

## 11. NeurIPS Submission Context

This work targets the **NeurIPS Datasets and Benchmarks track**.

### Novel Contributions

1. **First open dataset** combining real geospatial perception (terrain + wind + obstacles + SAM threat) with physics-based trajectory planning, 6-DoF vehicle dynamics, and closed-loop control — all in a defense eVTOL context.
2. **Multi-signature benchmark:** Simultaneous acoustic, infrared, and RCS modeling under trajectory optimization — no prior open dataset includes all three.
3. **End-to-end layered architecture:** Enables cross-layer ML tasks not possible with single-domain datasets.

### Suggested ML Tasks

The dataset supports multiple supervised and reinforcement learning tasks:

| Task | Input Features | Label | Type |
|------|---------------|-------|------|
| Risk classification | Planning + vehicle features | `risk_label` | Binary classification |
| Abort prediction | Control + vehicle features | `mission_abort` | Binary classification |
| Energy prediction | Planning features | `energy_cost_wh` | Regression |
| Threat exposure prediction | Perception features | `threat_cost` | Regression |
| Optimal waypoint selection | Perception grid | `fused_cost` | Ranking |
| SOC forecasting | Vehicle features | `soc_final` | Regression |

### What Would Strengthen the Submission

- [ ] Baseline ML results (logistic regression, gradient boosting, MLP) on each task
- [ ] Formal train/validation/test split (80/10/10) provided as index files
- [ ] Statistical significance tests for risk-label performance differences
- [ ] Ablation: single-objective vs three-objective Pareto planning
- [ ] Comparison against existing eVTOL trajectory datasets (if any)
- [ ] Hardware-in-the-loop validation on a small-scale testbed

---

## 12. References

1. Karaman, S., & Frazzoli, E. (2011). Sampling-based algorithms for optimal motion planning. *International Journal of Robotics Research*, 30(7), 846–894.
2. Deb, K., & Jain, H. (2014). An evolutionary many-objective optimization algorithm using reference-point-based nondominated sorting approach, Part I. *IEEE Transactions on Evolutionary Computation*, 18(4), 577–601.
3. Mahafza, B. R. (2005). *Radar Systems Analysis and Design Using MATLAB* (2nd ed.). Chapman & Hall/CRC.
4. Johnson, W. (2013). *Rotorcraft Aeromechanics*. Cambridge University Press.
5. Wertz, J. R. (ed.) (1978). *Spacecraft Attitude Determination and Control*. Reidel.
6. Plett, G. L. (2015). *Battery Management Systems, Volume I: Battery Modeling*. Artech House.
7. Bristeau, P.-J., et al. (2009). The role of propeller aerodynamics in the model of a quadrotor UAV. *IFAC Proceedings*, 42(14).
8. Rao, A. V. (2009). A survey of numerical methods for optimal control. *Advances in the Astronautical Sciences*, 135(1), 497–528.

---

*Operating area: Delhi-NCR, India. All geospatial data from publicly available APIs.*
