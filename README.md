# Trajectory Optimization in Defense eVTOLs

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.11+"/>
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge" alt="License MIT"/>
</p>

A full-stack autonomous mission planning and control framework for defense electric vertical take-off and landing (eVTOL) aircraft operating in contested airspace. The system integrates real geospatial perception, multi-objective trajectory optimization, cascaded flight control, and six-degree-of-freedom vehicle dynamics into a four-layer architecture generating publication-quality datasets across six Indian geographic theatres.

---

## Table of Contents

1. [Research Motivation](#1-research-motivation)
2. [Architecture Overview](#2-architecture-overview)
3. [Layer Descriptions](#3-layer-descriptions)
4. [Multi-Region Dataset](#4-multi-region-dataset)
5. [Repository Structure](#5-repository-structure)
6. [Installation](#6-installation)
7. [Running the Pipeline](#7-running-the-pipeline)
8. [Dataset Schema](#8-dataset-schema)
9. [ML Baseline Results](#9-ml-baseline-results)
10. [Key Results](#10-key-results)
11. [Known Limitations](#11-known-limitations)
12. [References](#13-references)
13. [Citation](#14-citation)

---

## 1. Research Motivation

Autonomous eVTOL aircraft are emerging as critical assets in defense logistics, ISR (intelligence, surveillance, and reconnaissance), and medical evacuation in contested environments. Unlike civilian urban air mobility, defense missions require simultaneous optimization of:

- **Energy efficiency** — battery-constrained range with no ground refueling
- **Mission time** — time-critical deliveries under dynamic threat windows
- **Threat exposure minimization** — radar cross-section, acoustic, and infrared signature management against ground-based air defenses (GBAD)
- **Terrain following** — low-altitude NOE (nap-of-earth) flight to exploit terrain masking

Existing trajectory planners address at most two of these objectives. This work presents the first open dataset and framework that jointly models all four within real geospatial environments spanning diverse Indian geographic theatres, with end-to-end simulation from mission planning through closed-loop flight control.

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
│  • Wind: Open-Meteo GFS forecast (u, v, w components)           │
│  • Obstacles: OpenStreetMap Overpass API + building geometry    │
│  • Threat: Analytical SAM detection probability (Mahafza 2005)  │
│  • Output: 4D cost field C(φ, λ, h, t) — per-region grid        │
└─────────────────────────┬───────────────────────────────────────┘
                          │  cost field
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 2: PLANNING                                              │
│  ─────────────────────                                          │
│  • Sampling: RRT* (Karaman & Frazzoli 2011) — collision-free    │
│  • Optimization: NSGA-III (Deb & Jain 2014) — Pareto-optimal    │
│  • Objectives: energy [Wh], time [s], threat exposure [0-1]     │
│  • Constraints: terrain clearance >= 30 m, speed in [20, 100]   │
│  • Output: 2,000–10,000 Pareto-optimal trajectory records       │
└─────────────────────────┬───────────────────────────────────────┘
                          │  waypoints + speed profile
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 3: CONTROL   (50 Hz cascaded PID)                        │
│  ─────────────────────                                          │
│  • Outer loop: position/altitude/heading → attitude commands    │
│  • Inner loop: attitude → body angular rates                    │
│  • Innermost: rates → moments via rate PID                      │
│  • Allocation: pseudo-inverse mixing matrix B^† → 4 motors      │
│  • Phase scheduler: TAKEOFF→HOVER→TRANS→CRUISE→LAND             │
│  • Output: closed-loop simulation records, 76 metrics           │
└─────────────────────────┬───────────────────────────────────────┘
                          │  control inputs
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 4: VEHICLE DYNAMICS  (6-DoF)                             │
│  ─────────────────────                                          │
│  • Rigid body: Newton-Euler equations with gyroscopic terms     │
│  • Propulsion: BEMT rotor model, tilted-nacelle transition      │
│  • Energy: electrochemical battery model with Peukert effect    │
│  • Signatures: acoustic (SPL), infrared (LWIR), RCS (Mie/PO)    │
│  • Output: vehicle state records, 95 physics features           │
└─────────────────────────────────────────────────────────────────┘
```

Data flows strictly top-down. Each layer's outputs are the next layer's inputs.

---

## 3. Layer Descriptions

### 3.1 Perception Layer

Builds a 4-dimensional cost field over each operating area at multiple altitude bands (50 m, 150 m, 300 m AGL). All data sourced from real public APIs — no synthetic fallbacks.

| Sub-layer | Source | Resolution | Key Outputs |
|-----------|--------|------------|-------------|
| Terrain | NASA SRTM30 via Open-Meteo Elevation API | ~90 m | `elev_m`, `slope_deg`, `roughness_m` |
| Wind | Open-Meteo GFS (hourly forecast) | Point forecast | `wind_u/v/w_mps`, `turbulence_intensity` |
| Obstacle | OpenStreetMap Overpass API | Object-level | `nearest_obstacle_dist_m`, `obstacle_type` |
| Threat | Analytical SAM model (3 emitter types) | Continuous | `T1/T2/T3_detect_prob`, `combined_threat_prob` |
| Fusion | Weighted harmonic composite | — | `fused_cost` in [0, 1] |

The threat model implements the Mahafza (2005) radar range equation for three emitter classes: Type-1 long-range surveillance radar (L-band, 150 km), Type-2 medium-range fire control radar (X-band, 40 km), and Type-3 short-range MANPADS (infrared seeker, 6 km). Combined detection probability uses Swerling Case I target statistics with the Marcum Q-function for fluctuating targets.

### 3.2 Planning Layer

Two-phase trajectory generation:

**Phase 1 — RRT* (feasibility):** Karaman & Frazzoli (2011) asymptotically optimal sampling-based planner. The rewiring radius follows the theoretical bound gamma = (2(1+1/d) * vol(X_free)/vol(unit_ball))^(1/d) where d=3 (spatial dimension), guaranteeing asymptotic optimality. Operates in the perception cost field with terrain clearance and speed constraints enforced at every node.

**Phase 2 — NSGA-III (Pareto optimization):** Deb & Jain (2014) reference-point-based many-objective EA with three objectives:
- `f1 = energy_cost_wh` — momentum theory hover power + parasitic drag in cruise
- `f2 = time_cost_s` — path-length / cruise-speed + hover durations
- `f3 = threat_cost` — distance-decay integrated detection probability (effective SAM ranges 20–30 km) producing mean=0.79, std=0.12 spatial gradient

Knee-point selection from Pareto front: minimize L2 distance to utopia point after normalization.

### 3.3 Control Layer

Cascaded PID architecture at 50 Hz:

```
Position cmd → [Position PID]  → Velocity cmd
Velocity cmd → [Velocity PID]  → Attitude cmd (roll/pitch/yaw)
Attitude cmd → [Attitude PID]  → Body rate cmd (p, q, r)
Rate cmd     → [Rate PID]      → Moments (Mx, My, Mz)
Moments      → [Mixer B†]      → Motor thrusts T0...T3
```

6-DoF simplified plant with Euler ZYX kinematics, gyroscopic cross-coupling (omega x I*omega) in rotational dynamics, linear drag model, and phase-aware reference generation. Phase schedule per mission: TAKEOFF → HOVER → TRANS1 → CRUISE → TRANS2 → HOVER2 → LAND.

### 3.4 Vehicle Layer

Newton-Euler 6-DoF dynamics with:

- **Propulsion:** Blade Element Momentum Theory rotor model; tilted-nacelle tiltrotor transition model
- **Aerodynamics:** Wing lift/drag (finite-span Oswald efficiency), fuselage parasite drag, sideslip yaw stability
- **Energy:** Electrochemical battery with Peukert capacity de-rating (k=1.05), internal resistance, thermal coupling
- **Acoustic signature:** A-weighted SPL from rotor thrust noise model
- **Infrared signature:** LWIR radiant intensity from motor/rotor thermal model
- **RCS:** Physical optics cross-section estimation for key aspect angles

State vector: 23 elements — position NED [3], velocity body [3], attitude quaternion [w,x,y,z] [4], angular velocity [3], nacelle angles [2], nacelle rates [2], rotor speeds [2], collective pitch [2], battery SOC [1], battery temperature [1].

---

## 4. Multi-Region Dataset

The dataset covers six Indian geographic theatres representing diverse terrain, climate, and threat scenarios:

| Region | Area | Terrain Type | Missions | Positive Rate (T1) |
|--------|------|--------------|----------|--------------------|
| Delhi (NCR) | 28.7–29.0°N, 76.9–77.4°E | Urban/semi-arid plain | 10,000 | ~26% |
| Mumbai | 18.85–19.35°N, 72.75–73.35°E | Coastal urban | 2,000 | ~12% |
| Bangalore | 12.8–13.1°N, 77.5–77.8°E | Elevated plateau | 2,000 | ~69% |
| Arunachal Pradesh | 27.1–27.6°N, 93.5–94.0°E | High-altitude mountainous | 2,000 | ~0.1% |
| Odisha | 20.1–20.5°N, 85.6–86.0°E | Coastal flat | 2,000 | ~97% |
| Ladakh | 34.0–34.4°N, 77.5–78.0°E | High-altitude arid | 2,000 | ~0.65% |

All datasets generated from real API sources. Odisha note: `obstacle_cost` column is maximum (1.0) for all rows due to an Overpass API gateway timeout during collection; all other columns are valid.

---

## 5. Repository Structure

```
trajectory-optimization-in-defense-evtols/
│
├── src/evtol/                        # Core library (137 Python modules)
│   ├── core/                         # Canonical cross-layer types
│   │   └── state.py                  # VehicleState, FlightPhase enum
│   ├── perception/                   # Terrain, wind, obstacle, threat, fusion
│   │   ├── sensor_fusion.py          # 6-state Kalman tracker (TrackedThreat)
│   │   └── fusion_orchestrator.py    # 20 Hz threat-map fusion cycle
│   ├── planning/                     # RRT*, NSGA-III, trajectory, constraints
│   │   ├── rrt_star.py               # Canonical threat-aware RRT*
│   │   └── optimization/nsga3.py     # Canonical NSGA-III (457 lines)
│   ├── control/                      # Cascaded PID, modes, motor allocation
│   │   ├── cascaded_control.py       # Main cascade controller
│   │   └── sitl_simulator.py         # SITL bridge (SITLState telemetry)
│   └── vehicle/                      # Dynamics, propulsion, energy, signatures
│       ├── vehicle_model.py          # TiltrotorVehicle — canonical 6-DoF class
│       └── dynamics/state.py         # VehicleState (23-element NED state)
│
├── scripts/                          # Pipeline runners and utilities
│   ├── perception/                   # dataset.py, visuals.py, region configs
│   ├── planning/                     # dataset.py, visualize.py, run_mumbai.py
│   ├── vehicle/                      # dataset.py, visualize.py
│   ├── control/                      # dataset.py, visualize.py, validate_single_mission.py
│   ├── ml/                           # baseline_experiments.py, run_all_regions.py
│   ├── run_region.py                 # Single-region full-pipeline runner
│   ├── export_all_formats.py         # Export all datasets to all tabular/image formats
│   └── region_configs.py             # Bounding boxes, threat configs, per-region settings
│
├── datasets/                         # All generated datasets (tracked in git)
│   ├── delhi/                        # 10,000 missions — primary benchmark region
│   │   ├── planning_dataset/         # planning_dataset_10k.parquet (.csv .h5 .feather .pkl .json)
│   │   ├── vehicle/                  # vehicle_dataset.parquet (all formats)
│   │   ├── control/                  # control_dataset.parquet (all formats)
│   │   ├── perception_dataset/       # exports/ (parquet per sub-layer + metadata)
│   │   ├── splits/                   # train/val/test index arrays (.npy)
│   │   └── ml/                       # baseline_results.csv, baseline_results.json
│   ├── mumbai/                       # 2,000 missions (same structure)
│   ├── bangalore/                    # 2,000 missions
│   ├── arunachal/                    # 2,000 missions
│   ├── odisha/                       # 2,000 missions
│   ├── ladakh/                       # 2,000 missions
│   ├── ml_all_regions_summary.csv    # Cross-region ML results
│   └── ml_plots/                     # Cross-region comparison charts (PNG PDF SVG EPS)
│
├── doc/                              # Technical documentation
│   ├── introduction.md               # Research context and motivation
│   ├── architecture.md               # System design and data flow
│   ├── perception_layer.md           # Perception math, APIs, physics
│   ├── planning_layer.md             # RRT*, NSGA-III, constraints derivation
│   ├── vehicle_layer.md              # 6-DoF dynamics, propulsion, signatures
│   ├── control_layer.md              # Cascaded PID, mixing matrix, phase scheduler
│   ├── data_guide.md                 # Dataset schemas, column definitions, load code
│   ├── datasheet.md                  # Gebru et al. 2021 datasheet
│   └── research_limitations.md       # Known issues and NeurIPS submission notes
│
├── notebooks/tutorial.ipynb          # Quickstart notebook
├── DATASET_CARD.md                   # HuggingFace dataset card
├── evtol_dataset.py                  # HF load_dataset() loader
├── papers_with_code.json             # Papers With Code metadata
├── pyproject.toml                    # Package configuration
└── README.md                         # This file
```

---

## 6. Installation

**Requirements:** Python 3.11+

```bash
git clone https://github.com/irhyl/trajectory-optimization-in-defense-evtols
cd trajectory-optimization-in-defense-evtols

python -m venv .venv
source .venv/bin/activate        # Linux / macOS
.venv\Scripts\activate           # Windows

pip install -e .
```

**Key dependencies:**

| Package | Purpose |
|---------|---------|
| `numpy`, `scipy` | Numerics, linear algebra, optimization |
| `pandas`, `pyarrow` | DataFrames and Parquet I/O |
| `scikit-learn` | ML baselines, preprocessing, cross-validation |
| `matplotlib`, `seaborn` | Visualization |
| `requests` | External API calls (elevation, wind, OSM) |
| `tables` | HDF5 export (PyTables) |

**External data sources (fetched at runtime, cached locally):**
- Terrain elevation: Open-Meteo Elevation API (NASA SRTM30)
- Wind forecasts: Open-Meteo GFS atmospheric model
- Obstacle geometry: OpenStreetMap Overpass API

---

## 7. Running the Pipeline

### Single region (recommended starting point)

```bash
python scripts/run_region.py --region delhi
# Runs all 4 layers for Delhi and saves to datasets/delhi/
```

### Layer-by-layer (full control)

```bash
# Step 1 — Perception
python scripts/perception/dataset.py

# Step 2 — Planning
python scripts/planning/dataset.py

# Step 3 — Vehicle simulation
python scripts/vehicle/dataset.py

# Step 4 — Control simulation
python scripts/control/dataset.py
```

### ML baselines

```bash
# Delhi (primary benchmark)
python scripts/ml/baseline_experiments.py

# All 5 non-Delhi regions
python scripts/ml/run_all_regions.py
```

### Format and visual export

```bash
# Export all datasets to all tabular formats + all images to PDF/SVG/EPS
python scripts/export_all_formats.py
```

---

## 8. Dataset Schema

### Quick load

```python
import pandas as pd

# Delhi — primary benchmark
plan = pd.read_parquet("datasets/delhi/planning_dataset/planning_dataset_10k.parquet")
veh  = pd.read_parquet("datasets/delhi/vehicle/vehicle_dataset.parquet")
ctrl = pd.read_parquet("datasets/delhi/control/control_dataset.parquet")

# Any other region (2,000 rows each)
plan = pd.read_parquet("datasets/mumbai/planning_dataset/planning_mumbai.parquet")
```

### Available formats per dataset

Every dataset file is available in: `parquet`, `csv`, `h5` (HDF5/PyTables), `feather`, `pkl` (pickle), `npz` (numeric arrays), `json` (planning/control only — vehicle datasets exceed the JSON size limit).

### Planning dataset (25 columns)

| Column | Type | Description |
|--------|------|-------------|
| `start/goal_lat/lon` | float | Mission endpoints (WGS-84) |
| `n_waypoints` | int | Total waypoint count |
| `path_length_m` | float | Total arc length [m] |
| `time_cost_s` | float | Estimated flight duration [s] |
| `energy_cost_wh` | float | Estimated energy [Wh] |
| `threat_cost` | float | Integrated threat exposure [0,1] |
| `terrain_cost_mean` | float | Mean terrain cost along path |
| `wind_cost_mean` | float | Mean wind cost along path |
| `obstacle_cost_mean` | float | Mean obstacle cost along path |
| `fused_cost_mean` | float | Mean composite cost along path |
| `risk_label` | int | 0=low-risk, 1=high-risk (threshold: fused_cost >= 0.55) |
| `feasible` | int | Constraint satisfaction: terrain clearance + speed bounds |

### Vehicle dataset (95 columns, key groups)

| Group | Columns | Description |
|-------|---------|-------------|
| Mission | `cruise_speed_ms`, `cruise_altitude_m`, `mission_time_s`, `hover_time_s` | Mission parameters |
| Rotor | `T_hover_N`, `P_hover_kW`, `rpm_cruise`, `figure_of_merit` | BEMT propulsion |
| Aerodynamics | `CL_cruise`, `CD_cruise`, `LD_ratio`, `drag_parasitic_N` | Wing/fuselage |
| Energy | `energy_total_wh`, `soc_initial`, `soc_final`, `pack_capacity_wh` | Battery |
| Thermal | `motor_temp_rise_K`, `winding_temp_C`, `battery_temp_C` | Temperature |
| Acoustic | `spl_hover_a_dB`, `spl_cruise_a_dB` | Noise signature |
| Infrared | `ir_radiance_hover_wsr`, `ir_radiance_cruise_wsr` | IR signature |
| RCS | `rcs_cruise_x_dBsm`, `rcs_cruise_z_dBsm` | Radar signature |

### Control dataset (76 columns, key groups)

| Group | Columns | Description |
|-------|---------|-------------|
| Tracking | `alt_error_mean_m`, `att_error_mean_rad`, `pos_error_mean_m` | Controller tracking |
| Thrust | `thrust_cmd_mean_N`, `thrust_cmd_max_N` | Collective thrust |
| Motors | `motor_T_m0–3_mean_N`, `motor_T_balance_N` | Per-motor allocation |
| Quality | `itae_pos`, `itae_alt`, `itae_att` | ITAE integrals |
| Flags | `mission_abort`, `n_saturations` | Control health |
| Energy | `energy_consumed_wh`, `soc_final` | Closed-loop energy |

Full schema documentation: [doc/data_guide.md](doc/data_guide.md)

---

## 9. ML Baseline Results

All tasks use 5-fold cross-validation. Models: Logistic Regression / Ridge, Gradient Boosting, MLP.

### Task 1: Risk Classification (AUC-ROC, GradientBoosting)

| Region | AUC | F1 | Note |
|--------|-----|----|------|
| Delhi | 0.9996 | 0.9952 | 10K missions |
| Mumbai | 0.9942 | 0.9000 | 12% positive rate |
| Bangalore | 0.9963 | 0.9810 | 69% positive rate |
| Arunachal Pradesh | 0.1997 | 0.0000 | 0.1% positive rate — near-degenerate |
| Odisha | 0.9950 | 0.9974 | 97% positive rate |
| Ladakh | 0.8064 | 0.5556 | 0.65% positive rate |

### Task 2: Energy Regression (R², GradientBoosting)

| Region | R² | MAE (Wh) |
|--------|----|----------|
| Delhi | 0.9972 | 29.4 |
| Mumbai | 0.9964 | 32.2 |
| Bangalore | 0.9968 | 31.1 |
| Arunachal Pradesh | 0.9986 | 28.9 |
| Odisha | 0.9986 | 25.7 |
| Ladakh | 0.9959 | 30.5 |

Energy regression is stable across all regions (R² > 0.995 everywhere) — energy is primarily determined by path length and speed, which are geography-independent physical laws. Risk classification variability across regions is a genuine finding, reflecting how fused cost thresholding interacts with region-specific terrain and obstacle density.

Cross-region plots: [datasets/ml_plots/](datasets/ml_plots/)

---

## 10. Key Results

### Dataset Scale

| Dataset | Regions | Total Rows | Features | Formats Available |
|---------|---------|-----------|----------|------------------|
| Planning | 6 | 22,000 | 25 | parquet, csv, h5, feather, pkl, npz, json |
| Vehicle | 6 | 22,000 | 95 | parquet, csv, h5, feather, pkl, npz |
| Control | 6 | 22,000 | 76 | parquet, csv, h5, feather, pkl, npz, json |
| Perception (Delhi) | 1 | 1,057,714 | 28 | parquet, csv, h5, feather, pkl, npz |

### Control Performance (Delhi 10K)

| Metric | Mean | Std | Interpretation |
|--------|------|-----|----------------|
| Altitude error | 22.3 m | — | Post-fix value (was 248 m before phase-reset fix) |
| Attitude error | 0.012 rad | 0.004 rad | Excellent (< 0.7°) |
| Thrust mean | ~2,613 N | 20 N | 30% above hover weight (2,000 N) for margin |
| Mission abort rate | 0.0% | — | Stable across all missions |
| Motor saturations | ~46 | — | Per mission, well within design envelope |
| Velocity settling | 2.7 s | — | Post-fix value (was 16.4 s) |

### Vehicle Physics (Delhi 10K)

| Metric | Range | Notes |
|--------|-------|-------|
| Cruise speed | 48.5–94.4 m/s | Physics-based BEMT model |
| Figure of merit | ~0.866 | Momentum theory, good rotorcraft value |
| Hover power | ~83 kW | Within expected range for 200 kg class |
| Acoustic (hover) | 92.84 dB(A) | Typical tiltrotor hover SPL |
| RCS (cruise, X-band) | -8.84 +/- 0.46 dBsm | Small rotorcraft signature |

---

## 11. Known Limitations

### 11.1 Threat Saturation (Disclosed)

`combined_threat_prob = 1.000` for all Delhi perception grid cells. The modeled scenario places SAM systems whose coverage fully overlaps the operating region — intentional (contested airspace assumption). The planner uses a distance-decay surrogate (`threat_cost`, mean=0.79, std=0.12) for spatial differentiation. The saturation is retained in the dataset for transparency.

### 11.2 Arunachal / Ladakh Risk Label Imbalance

`risk_label` positive rate is 0.1% (Arunachal) and 0.65% (Ladakh) — too few positive samples for meaningful classification. The fused_cost threshold (0.55) produces near-zero positive rates in high-altitude terrain with minimal obstacles. Future work: region-adaptive thresholding.

### 11.3 Single-Rate Control Plant

The 50 Hz simplified plant cannot model GPS/IMU loop-rate separation (outer loops at 5–10 Hz, inner at 200+ Hz in real autopilots). Sensor noise was removed to preserve physically correct metrics, documented as a known gap requiring HITL validation.

### 11.4 Odisha Obstacle Data

`obstacle_cost` column is maximum (1.0) for all Odisha rows due to an Overpass API gateway timeout. All other Odisha columns are valid and Overpass failure is documented.

### 11.5 Simulation Only

All results are simulation-based. No hardware-in-the-loop or physical flight test data are included.

---

## 12. Context

### Contributions

1. **First open dataset** combining real geospatial perception (terrain + wind + obstacles + SAM threat) with physics-based trajectory planning, 6-DoF vehicle dynamics, and closed-loop control — all in a defense eVTOL context.
2. **Multi-signature benchmark:** Simultaneous acoustic, infrared, and RCS modeling under trajectory optimization — no prior open dataset includes all three.
3. **Geographic diversity:** Six Indian theatres (urban, coastal, plateau, high-altitude mountainous, arid) enabling cross-region generalization studies.
4. **End-to-end layered architecture:** Enables cross-layer ML tasks not possible with single-domain datasets.

### ML Tasks Defined

| Task | Input | Target | Type | Best Result |
|------|-------|--------|------|-------------|
| T1 | Planning features | `risk_label` | Binary classification | AUC 0.9998 (Odisha, LR) |
| T2 | Planning features | `energy_consumed_wh` | Regression | R² 0.9986 (Arunachal + Odisha, GBM) |
| T3 | Control features | `mission_abort` | Binary classification | Trivial (0% abort rate) |
| T4 | Mission params | `alt_error_mean_m` | Regression | — |
| T5 | Perception features | `threat_cost` | Regression | — |
| T6 | Vehicle features | `soc_final` | Regression | — |

---

## 13. References

1. Karaman, S., & Frazzoli, E. (2011). Sampling-based algorithms for optimal motion planning. *International Journal of Robotics Research*, 30(7), 846–894.
2. Deb, K., & Jain, H. (2014). An evolutionary many-objective optimization algorithm using reference-point-based nondominated sorting approach, Part I. *IEEE Transactions on Evolutionary Computation*, 18(4), 577–601.
3. Mahafza, B. R. (2005). *Radar Systems Analysis and Design Using MATLAB* (2nd ed.). Chapman & Hall/CRC.
4. Johnson, W. (2013). *Rotorcraft Aeromechanics*. Cambridge University Press.
5. Diebel, J. (2006). Representing attitude: Euler angles, unit quaternions, and rotation vectors. *Stanford University Technical Report*.
6. Plett, G. L. (2015). *Battery Management Systems, Volume I: Battery Modeling*. Artech House.
7. Bar-Shalom, Y., Li, X. R., & Kirubarajan, T. (2001). *Estimation with Applications to Tracking and Navigation*. Wiley.
8. Gebru, T. et al. (2021). Datasheets for datasets. *Communications of the ACM*, 64(12), 86–92.
9. Bristeau, P.-J. et al. (2009). The role of propeller aerodynamics in the model of a quadrotor UAV. *IFAC Proceedings*, 42(14).

---

<p align="center">
  <img src="https://img.shields.io/badge/Data%20Source-Open--Meteo%20%7C%20OSM%20%7C%20NASA%20SRTM-blue?style=flat-square" alt="Data Sources"/>
</p>

---

*Made with ❤️ by aditi ramakrishnan*
