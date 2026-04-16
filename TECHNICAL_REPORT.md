# Trajectory Optimization in Defense eVTOLs: A Full-Stack Autonomous Mission Dataset

**A Comprehensive Technical Report**

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Research Motivation and Problem Statement](#2-research-motivation-and-problem-statement)
3. [Related Work and Novelty](#3-related-work-and-novelty)
4. [System Architecture Overview](#4-system-architecture-overview)
5. [Perception Layer](#5-perception-layer)
6. [Planning Layer](#6-planning-layer)
7. [Vehicle Dynamics Layer](#7-vehicle-dynamics-layer)
8. [Control Layer](#8-control-layer)
9. [Multi-Region Dataset](#9-multi-region-dataset)
10. [Machine Learning Benchmark Tasks](#10-machine-learning-benchmark-tasks)
11. [ML Baseline Results](#11-ml-baseline-results)
12. [Software Architecture and Implementation](#12-software-architecture-and-implementation)
13. [Data Formats and Reproducibility](#13-data-formats-and-reproducibility)
14. [Known Limitations and Disclosures](#14-known-limitations-and-disclosures)
15. [Future Work](#15-future-work)
16. [References](#16-references)

---

## 1. Executive Summary

This report describes a full-stack autonomous mission dataset for defense electric vertical take-off and landing (eVTOL) aircraft operating in contested airspace. The dataset is the first open benchmark that jointly models real geospatial terrain and wind data, analytical surface-to-air missile (SAM) detection probability, multi-signature vehicle physics (acoustic, infrared, radar cross-section), and closed-loop autopilot simulation for six geographically distinct Indian military theatres.

The system is organized as four causally linked layers — Perception, Planning, Vehicle, and Control — each producing structured datasets consumed by the next. The total dataset comprises 22,000 mission records across 6 regions (Delhi-NCR, Mumbai, Bangalore, Arunachal Pradesh, Odisha, Ladakh), with 22,000 rows in each of the planning, vehicle, and control datasets, plus a 1,057,714-row perception grid for Delhi.

Six machine learning benchmark tasks are defined across classification and regression objectives, with provided baselines (Logistic Regression, Gradient Boosting, MLP) achieving AUC up to 0.9998 and R² up to 0.9986 on the primary Delhi region. Data is available in seven tabular formats (Parquet, CSV, NPZ, HDF5, Feather, Pickle, JSON) and four image formats (PNG, PDF, SVG, EPS), with all pipeline scripts reproducible from public APIs only.

---

## 2. Research Motivation and Problem Statement

### 2.1 The Defense eVTOL Problem

Electric vertical take-off and landing aircraft are transitioning from urban air mobility concepts to serious defense platforms. Programs such as the U.S. Army's FARA (Future Attack Reconnaissance Aircraft), AFWERX Agility Prime, and European equivalents are actively evaluating electrically-powered rotorcraft for intelligence, surveillance, and reconnaissance (ISR); logistics resupply; and medical evacuation (CASEVAC) in contested environments.

Unlike civilian eVTOLs designed for predictable urban corridors, defense applications face a fundamentally different constraint set:

**Mission-level constraints:**
- Time-critical objectives — minutes matter in CASEVAC and for perishable intelligence windows
- No fixed infrastructure — no GPS-guided landing zones, no pre-mapped airspace, no air traffic control
- Variable threat environments — threat positions change faster than mission replanning cycles

**Vehicle-level constraints:**
- Battery energy is the hard ceiling — no in-field refueling in contested ground
- Acoustic signature drives detectability at low altitude — rotors are intrinsically noisy
- Radar cross-section must be managed against ground-based air defenses
- Infrared signature increases with motor temperature during hover and high-power maneuvers

**The fundamental tension:** The actions that minimize detection (low altitude, high speed, terrain masking) are often incompatible with minimizing energy consumption (high altitude, moderate speed, direct routing). A mission planner that optimizes only one objective will produce plans that are either easily detected or unable to complete the mission on battery alone. This three-way trade-off between energy, time, and detection probability is the central problem this dataset is designed to study.

### 2.2 The Multi-Objective Planning Problem

Formally, the defense eVTOL trajectory optimization problem is:

$$\min_{\tau} \left[ E(\tau),\; T(\tau),\; P_{\mathrm{detect}}(\tau) \right]$$

subject to:
- $\tau(0) = q_{\mathrm{start}}$, $\tau(1) = q_{\mathrm{goal}}$
- $\tau(s) \in \mathcal{C}_{\mathrm{free}}$ for all $s \in [0,1]$ (collision avoidance)
- $\|\dot{\tau}\| \leq V_{\max} = 60$ m/s (speed constraint)
- $\|\ddot{\tau}\| \leq a_{\max} = 5$ m/s² (acceleration constraint)
- $z(\tau(s)) \geq h_{\min} = 50$ m AGL (terrain clearance)
- $E(\tau) \leq E_{\mathrm{battery}}$ (battery energy budget)

where $E(\tau)$ is total energy consumed [Wh], $T(\tau)$ is mission time [s], and $P_{\mathrm{detect}}(\tau)$ is the integrated SAM detection probability along the trajectory. This formulation has no single optimal solution — the Pareto front describes the set of non-dominated trade-offs across all three objectives.

### 2.3 Why Existing Approaches Fall Short

**Single-objective planners:** Most autonomous UAV trajectory planners — A*, RRT, potential fields, MPC — optimize a single weighted cost. When threat avoidance is added as a penalty weight, the resulting solution depends critically on the weight choice, which must be hand-tuned per mission. Different planners with different weight settings are not comparable, and no systematic Pareto analysis is possible.

**Missing signature integration:** Existing open datasets for UAV trajectory planning (DARPA SubT, AirSim environments, OpenAI gym UAV environments) do not include ground-truth signature data (acoustic, infrared, or radar cross-section). Yet signature management is the primary differentiator between a survivable and non-survivable defense mission.

**Simulation-reality gap in control:** Published UAV control datasets are typically either lab-scale (small quadrotors, indoor, calm air), not representative of 200 kg eVTOL dynamics, or high-fidelity simulations without physics validation, or hardware flight logs without access to the planning/perception layers that generated the commands. This prevents end-to-end ML research spanning perception through control.

**No defense-specific open benchmark:** To our knowledge, no open benchmark simultaneously models SAM detection probability analytically, combines multi-signature modeling (acoustic + IR + RCS) under trajectory optimization, provides real geospatial data, and closes the loop from perception through planning through control through vehicle dynamics.

---

## 3. Related Work and Novelty

### 3.1 Prior Work

| Work | Approach | Gap Relative to This Work |
|------|----------|--------------------------|
| Karaman & Frazzoli (2011) | RRT* planning theory | No multi-objective, no threat, no closed-loop simulation |
| Deb & Jain (2014) | NSGA-III algorithm | Algorithm only, no UAV-specific dataset |
| Lin & Saripalli (2017) | UAV path planning in urban terrain | No threat, no vehicle dynamics, no control layer |
| Guerra et al. (2019) | Fixed-wing UAV NSGA-II vs NSGA-III | No eVTOL dynamics, no signature modeling |
| AgriNav, AirSim, FlyThrough | Simulation environments | No defense signatures, no real geospatial data |
| OpenUAV, SubT Challenge | Real-world datasets | No planning optimization, no control analytics |
| DARPA OFFensive Swarm (OFFSET) | Multi-UAV urban swarms | Classified, not open, no eVTOL-class vehicles |

### 3.2 Novel Contributions

This work uniquely combines:

1. **Real geospatial foundation** — NASA SRTM terrain, Open-Meteo GFS wind, OpenStreetMap obstacles: all from public APIs, fully reproducible
2. **Analytical SAM threat model** — Mahafza (2005) radar range equation with Swerling case I targets, three emitter categories
3. **Multi-signature vehicle modeling** — simultaneous acoustic SPL, infrared LWIR, and X-band RCS under the same flight conditions
4. **Full multi-objective Pareto planning** — NSGA-III with 200 generations, 100 population, 15 reference points
5. **End-to-end causal chain** — each layer's output is an input to the next; cross-layer ML is directly supported
6. **Multi-theater coverage** — six geographically diverse Indian regions spanning coastal, mountainous, arid, and urban environments

---

## 4. System Architecture Overview

### 4.1 Four-Layer Stack

The framework implements a four-layer autonomous mission stack where each layer produces a structured dataset consumed by the next. The architecture mirrors the autonomy stack of a real defense UAV system.

```
                     ┌──────────────────────────┐
                     │    MISSION PARAMETERS     │
                     │  (start, goal, time       │
                     │   budget, threat alert)   │
                     └─────────────┬─────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │    LAYER 1: PERCEPTION       │
                    │  Real-data 4D cost field     │
                    │  C(φ, λ, h, t)               │
                    │  1,057,714 rows / 28 features│
                    └──────────────┬──────────────┘
                                   │ perception_full_dataset.parquet
                    ┌──────────────▼──────────────┐
                    │    LAYER 2: PLANNING         │
                    │  RRT* + NSGA-III             │
                    │  Pareto trajectories         │
                    │  2,000 rows / 25 features    │
                    └──────────────┬──────────────┘
                                   │ planning_dataset.parquet
                    ┌──────────────▼──────────────┐
                    │    LAYER 3: VEHICLE          │
                    │  6-DoF physics simulation    │
                    │  BEMT + aerodynamics         │
                    │  2,000 rows / 95 features    │
                    └──────────────┬──────────────┘
                                   │ vehicle_dataset.parquet
                    ┌──────────────▼──────────────┐
                    │    LAYER 4: CONTROL          │
                    │  50 Hz cascaded PID          │
                    │  Closed-loop tracking        │
                    │  2,000 rows / 76 features    │
                    └──────────────────────────────┘
```

### 4.2 Coordinate Frames

Three coordinate frames are used across the stack:

**WGS-84 Geographic Frame** — Used by perception and planning for raw API data. Format: (latitude °, longitude °, altitude m MSL). Datum: WGS-84 ellipsoid.

**NED (North-East-Down) Local Frame** — Used by planning for local path coordinates and by control for position/velocity states. The flat-Earth approximation (valid for theatres < 200 km) converts from geographic:

$$N = (\phi - \phi_0) \cdot \frac{\pi}{180} \cdot R_\phi$$
$$E = (\lambda - \lambda_0) \cdot \frac{\pi}{180} \cdot R_\lambda \cdot \cos\phi_0$$
$$D = -(h - h_0)$$

For the Delhi NCR theatre ($\phi_0 = 28.5°$ N): $R_\phi \approx 111{,}200$ m/deg, $R_\lambda \approx 97{,}800$ m/deg.

**Body Frame** — Used by vehicle dynamics and control for forces, moments, and angular rates. Origin: vehicle center of mass. Convention: x-forward, y-right, z-down. Rotation: ZYX Euler sequence (yaw ψ → pitch θ → roll φ) maps body to NED.

### 4.3 Quaternion Convention

Two quaternion conventions coexist in the codebase:

| Location | Convention | Layout |
|----------|-----------|--------|
| `vehicle/dynamics/` | Hamilton | `[w, x, y, z]` — scalar first |
| `control/cascaded_control.py` | JPL / passive | `[x, y, z, w]` — scalar last |

Conversion when passing from dynamics to control: `q_jpl = np.roll(q_hamilton, -1)`.

---

## 5. Perception Layer

### 5.1 Overview

The perception layer is the first stage of the four-layer stack. Its responsibility is to produce a spatiotemporal traversability cost field over the operational theatre — a 4D function $C(\phi, \lambda, h, t)$ — that encodes every physical hazard relevant to the eVTOL's mission survivability and efficiency. Five submodules contribute to this cost field:

| Module | Physical Domain | Data Source |
|--------|----------------|-------------|
| Terrain | Orography, elevation clearance, slope | NASA SRTM via Open-Elevation API |
| Wind | Atmospheric dynamics, turbulence, gusts | GFS/HRES via Open-Meteo Forecast API |
| Obstacle | Static and dynamic airspace objects | OpenStreetMap Overpass API |
| Threat | Radar detection, SAM engagement probability | Analytical (radar range equation) |
| Fusion | Weighted composite traversability cost | In-system combination |

**Design principle:** No synthetic data fallbacks. Every data point traces to a real physical measurement or a physics-based analytical model with documented derivation.

### 5.2 Terrain Module

**Data source:** The Shuttle Radar Topography Mission (SRTM, NASA/NGA, February 2000) produced the primary global DEM. Mission parameters: C-band SAR (5.3 GHz), interferometric mode, horizontal resolution ~30 m at equator (1 arc-second), vertical accuracy ~16 m RMSE relative to EGM96 geoid.

**Grid construction:** A coarse-to-fine interpolation strategy is employed:
1. Fetch SRTM at a 2 km coarse grid (~870 points for the 55×60 km Delhi theatre)
2. Bilinear interpolation to a 222 m fine grid (251×301 = 75,551 points)

For bilinear interpolation at query point $(x, y)$ within cell $(x_1, x_2) \times (y_1, y_2)$:

$$f(x,y) = \frac{1}{(x_2-x_1)(y_2-y_1)}\Big[(x_2-x)(y_2-y)f_{11} + (x-x_1)(y_2-y)f_{21} + (x_2-x)(y-y_1)f_{12} + (x-x_1)(y-y_1)f_{22}\Big]$$

**Derivative products:**

*Slope:* Computed using central finite differences:
$$\frac{\partial z}{\partial N} = \frac{z_{i+1,j} - z_{i-1,j}}{2\,\Delta N}, \quad \frac{\partial z}{\partial E} = \frac{z_{i,j+1} - z_{i,j-1}}{2\,\Delta E}$$
$$\text{slope} = \arctan\sqrt{\left(\frac{\partial z}{\partial N}\right)^2 + \left(\frac{\partial z}{\partial E}\right)^2}$$

*Roughness:* Terrain roughness index (TRI) computed as the mean absolute elevation difference from a 3×3 neighborhood. High TRI correlates with mechanical turbulence generation and reduced hover safety.

**Terrain cost:** The terrain traversal cost at altitude $h$ above ground:

$$C_{\mathrm{terrain}}(\phi, \lambda, h) = \begin{cases} \infty & \text{if } h < h_{\min} \\ w_s \cdot \text{slope} + w_r \cdot \text{TRI} & \text{otherwise} \end{cases}$$

where $h_{\min} = 50$ m AGL is the minimum safe altitude and $w_s, w_r$ are empirical weights.

### 5.3 Wind Module

**Data source:** Open-Meteo GFS (Global Forecast System) atmospheric model. The GFS is NOAA's operational global NWP model run 4 times daily at 0.25° (~28 km) horizontal resolution, with data available at pressure levels corresponding to ~50 m, ~150 m, and ~300 m AGL.

**Wind vector components:** Three wind speed components are fetched: $u$ (eastward), $v$ (northward), and derived vertical wind $w$ (from pressure-velocity conversion). For the Delhi NCR region at 150 m AGL, typical values: $u = 4$–$15$ m/s westerly, $v = -2$–$+3$ m/s.

**Turbulence intensity:** Computed as the Dryden turbulence model parameter:

$$\sigma_w = 0.1 \cdot W_{20} \cdot \left(\frac{h}{6}\right)^{1/6} \quad (h < 1000 \text{ ft})$$

where $W_{20}$ is the wind speed at 20 ft (6.1 m) AGL. Higher turbulence increases vehicle energy consumption and reduces autopilot tracking accuracy.

**Wind cost:** Headwind increases energy; tailwind decreases it. For a segment with bearing $\psi$ and wind vector $(u, v)$:

$$C_{\mathrm{wind}}(\phi, \lambda, h, \psi) = 1 + k_w \cdot \frac{u\sin\psi + v\cos\psi}{V_{\mathrm{cruise}}}$$

where $k_w = 0.3$ scales the headwind penalty.

### 5.4 Obstacle Module

**Data source:** OpenStreetMap Overpass API, queried for all buildings, structures, and towers within the bounding box. Query categories include: `building=*`, `power=tower`, `man_made=mast`, `aeroway=*`.

**Obstacle geometry:** Each obstacle is modeled as a vertical cylinder of radius $r_{\mathrm{obs}}$ and height $h_{\mathrm{obs}}$. The safety buffer is 50 m horizontal clearance and 20 m vertical clearance. Conflict detection for a trajectory segment $(p_1, p_2)$ uses 3D cylinder intersection.

**Obstacle cost:**

$$C_{\mathrm{obs}}(\phi, \lambda, h) = \sum_i \max\!\left(0,\; 1 - \frac{d_i(\phi, \lambda, h)}{r_{\mathrm{safety}}}\right)$$

where $d_i$ is the distance from the query point to obstacle $i$'s safety boundary.

**Note (Odisha region):** Overpass API gateway timeouts during data collection caused `obstacle_cost = 1.0` (all-maximum) for the Odisha region. This is a data quality issue disclosed in the datasheet.

### 5.5 SAM Threat Model

The SAM threat model is the most novel component of the perception layer. It implements the radar range equation in the form derived by Mahafza (2005) for three analytically parameterized emitter categories.

**Emitter categories:**

| Category | Type | Band | Max Range | Detection Model |
|----------|------|------|-----------|----------------|
| T1 | Long-range surveillance radar | L-band (~1.3 GHz) | 150 km | Marcum Q-function (Swerling I) |
| T2 | Medium-range fire control radar | X-band (~9.5 GHz) | 40 km | Albersheim approximation |
| T3 | Short-range MANPADS (IR seeker) | LWIR (~10 μm) | 6 km | IR signature engagement probability |

**Radar range equation (T1/T2):**

The single-pulse signal-to-noise ratio at range $R$ from the emitter:

$$\mathrm{SNR} = \frac{P_t G_t G_r \lambda^2 \sigma_{\mathrm{RCS}}}{(4\pi)^3 R^4 k T_s B_n L}$$

where $P_t$ is transmit power [W], $G_t, G_r$ are transmit/receive antenna gains, $\lambda$ is wavelength [m], $\sigma_{\mathrm{RCS}}$ is the target radar cross-section [m²], $k$ is Boltzmann's constant, $T_s$ is system noise temperature [K], $B_n$ is noise bandwidth [Hz], and $L$ accounts for system losses.

**Swerling Case I detection probability (T1):** For a Swerling I target (correlated pulses, Rayleigh-distributed RCS), the detection probability at fixed false-alarm probability $P_{\mathrm{fa}}$:

$$P_d = \exp\!\left(-\frac{\mathrm{SNR}_{\mathrm{thresh}}}{1 + \overline{\mathrm{SNR}}}\right)$$

**Albersheim approximation (T2):** For large SNR, a simple algebraic approximation avoiding numerical integration of the Marcum Q-function:

$$P_d \approx \Phi\!\left(\sqrt{2\,\mathrm{SNR}} - \sqrt{2\ln(1/P_{\mathrm{fa}}) - 1}\right)$$

**Combined detection probability:** Assuming statistical independence of the three emitter systems:

$$P_{\mathrm{combined}} = 1 - (1 - P_{T1})(1 - P_{T2})(1 - P_{T3})$$

**Threat saturation:** Because the three emitters are placed to cover the entire 33×55 km Delhi operating area, $P_{\mathrm{combined}} \approx 1.0$ everywhere. This is not a modeling error — it reflects the "heavily contested airspace" scenario assumption. For the planning layer, a distance-decay threat gradient is applied (effective ranges shortened to 20–30 km) to produce spatial variation: mean 0.79, std 0.12, range 0.45–1.0.

### 5.6 Sensor Fusion and Kalman Tracker

**Fusion weights:** The composite traversability cost field:

$$C_{\mathrm{fused}} = w_t C_{\mathrm{terrain}} + w_w C_{\mathrm{wind}} + w_o C_{\mathrm{obs}} + w_r C_{\mathrm{threat}}$$

with weights $w_t = 0.25$, $w_w = 0.15$, $w_o = 0.35$, $w_r = 0.25$.

**6-state Kalman tracker** (`src/evtol/perception/sensor_fusion.py`): For tracking dynamic threats (e.g., moving SAM launcher, aerial threats), the sensor fusion module implements a constant-velocity Kalman filter:

State vector: $\mathbf{x} = [p_x, p_y, p_z, v_x, v_y, v_z]^\top \in \mathbb{R}^6$

State transition matrix:
$$\mathbf{F} = \begin{pmatrix} \mathbf{I}_3 & \Delta t \mathbf{I}_3 \\ \mathbf{0}_3 & \mathbf{I}_3 \end{pmatrix}$$

Measurement matrix (position-only observations):
$$\mathbf{H} = \begin{pmatrix} \mathbf{I}_3 & \mathbf{0}_3 \end{pmatrix}$$

Process noise covariance (piecewise-white acceleration model with $\sigma_a = 2$ m/s²):
$$\mathbf{Q} = \sigma_a^2 \begin{pmatrix} \frac{\Delta t^4}{4}\mathbf{I}_3 & \frac{\Delta t^3}{2}\mathbf{I}_3 \\ \frac{\Delta t^3}{2}\mathbf{I}_3 & \Delta t^2 \mathbf{I}_3 \end{pmatrix}$$

Kalman update:
$$\mathbf{K} = \mathbf{P}\mathbf{H}^\top(\mathbf{H}\mathbf{P}\mathbf{H}^\top + \mathbf{R})^{-1}$$
$$\hat{\mathbf{x}} \leftarrow \hat{\mathbf{x}} + \mathbf{K}(\mathbf{z} - \mathbf{H}\hat{\mathbf{x}})$$
$$\mathbf{P} \leftarrow (\mathbf{I} - \mathbf{K}\mathbf{H})\mathbf{P}$$

Each tracked threat has an associated 6×6 covariance matrix $\mathbf{P}$. Position uncertainty $\sigma_{\mathrm{pos}} = \sqrt{\mathrm{diag}(\mathbf{P})_{1:3}}$ and velocity uncertainty $\sigma_{\mathrm{vel}} = \sqrt{\mathrm{diag}(\mathbf{P})_{4:6}}$ are available as dataset features.

**Fusion orchestrator** (`src/evtol/perception/fusion_orchestrator.py`): Runs a 20 Hz threat-map fusion cycle. For each cycle: (1) predict all active tracks forward using constant-velocity model, (2) associate new measurements to existing tracks via nearest-neighbor gating, (3) update associated tracks with Kalman update, (4) initialize new tracks for unassociated measurements, (5) compute threat level from fused costs before ThreatMap construction (to avoid post-construction mutation).

---

## 6. Planning Layer

### 6.1 Overview

The planning layer converts the 4D traversability cost field into a time-parameterized, kinodynamically feasible, threat-minimizing trajectory. Five subsystems are involved:

| Subsystem | Role | Algorithm |
|-----------|------|-----------|
| Path search | Find collision-free path | A*, Theta*, RRT, RRT* |
| Multi-objective optimizer | Balance energy, threat, time | NSGA-III |
| Trajectory processing | Smooth, timestamp, check feasibility | B-spline, S-curve velocity |
| Threat integration | Integrate $P_K$ along trajectory | SAM detection field |
| Mission orchestration | Sequence multi-leg missions, contingency | Greedy NN ordering, 3-tier response |

### 6.2 Configuration Space

The configuration space for a defense eVTOL in 3D airspace:

$$\mathcal{C} = \{q = (x, y, z) \in \mathbb{R}^3\}$$

The free configuration space excludes:
- **Hard obstacles:** Volumes within collision radius of any OSM obstacle
- **No-fly zones:** Regulatory airspace, military exclusion zones
- **Terrain:** All points with $h_{\mathrm{clearance}} < h_{\min} = 50$ m AGL

Kinodynamic constraints enforced by the feasibility checker:

| Constraint | Expression | Value |
|---|---|---|
| Max speed | $\|\dot{q}\| \leq V_{\max}$ | 60 m/s |
| Max acceleration | $\|\ddot{q}\| \leq a_{\max}$ | 5 m/s² |
| Max jerk | $\|\dddot{q}\| \leq j_{\max}$ | 2 m/s³ |
| Min altitude AGL | $z \geq h_{\min}$ | 50 m |
| Max bank angle | $|\phi_{\mathrm{bank}}| \leq 30°$ | during cruise |
| Max climb rate | $\dot{z} \leq 8$ m/s | |

### 6.3 Graph-Based Planners

**Dijkstra's Algorithm** (`planning/algorithms/graph.py`): Finds shortest paths in a weighted directed graph $G = (V, E, w)$ with complexity $O((|V| + |E|)\log|V|)$ using binary heap. For the 250×300×14-layer grid: $|V| = 1{,}050{,}000$, $|E| \approx 7|V|$ (26-connected 3D). Used when optimal cost is required and computation time is not critical.

**A* Algorithm:** Augments Dijkstra with an admissible heuristic $h(v)$:

$$f(v) = g(v) + h(v)$$

where $g(v)$ is the cost from start and $h(v)$ is the Euclidean distance heuristic $h(v) = \|v - q_{\mathrm{goal}}\|_2$. Complexity: $O(|E|)$ in best case, $O(b^d)$ worst case. Optimal when heuristic is admissible ($h(v) \leq h^*(v)$ always).

**Theta* Algorithm:** Extends A* with "any-angle" line-of-sight shortcuts. If a straight line from the parent of $u$ to $v$ is free, the algorithm sets parent($v$) = parent($u$) directly, bypassing the grid structure. This produces shorter, less grid-aligned paths than A*, reducing the "staircase" artifact on diagonal routes.

### 6.4 RRT* — Threat-Aware Planning

The canonical threat-aware planner for defense missions is implemented in `src/evtol/planning/rrt_star.py`. The generic framework in `planning/algorithms/sampling.py` provides base sampling utilities only.

**RRT* algorithm:** Given a metric space $(\mathcal{C}_{\mathrm{free}}, d)$ with cost function $c$, RRT* builds an asymptotically optimal tree:

```
RRT*(q_start, n_max):
  T ← {q_start}
  for i = 1 to n_max:
    q_rand ← SAMPLE()
    q_near ← NEAREST(T, q_rand)
    q_new  ← STEER(q_near, q_rand, η)     # step size η
    if COLLISION_FREE(q_near, q_new):
      X_near ← NEAR(T, q_new, r)          # radius r = γ·(log n/n)^(1/d)
      q_min  ← CHOOSE_PARENT(X_near, q_near, q_new)
      T.add(q_new, q_min)
      REWIRE(T, X_near, q_new)
  return T
```

**Asymptotic optimality** (Karaman & Frazzoli 2011): RRT* with radius $r_n = \gamma \cdot (\log n / n)^{1/d}$ converges to the optimal path cost almost surely as $n \to \infty$, provided $\gamma > \gamma^* = 2(1 + 1/d)^{1/d}(|\mathcal{C}_{\mathrm{free}}| / \zeta_d)^{1/d}$.

**Cost function for defense:** The edge cost integrates the fused perception cost along the segment:

$$c(q_1, q_2) = \|q_2 - q_1\| \cdot \bar{C}_{\mathrm{fused}}\!\left(\frac{q_1 + q_2}{2}\right) \cdot (1 + \alpha_{\mathrm{threat}} \cdot \bar{C}_{\mathrm{threat}})$$

where the threat penalty $\alpha_{\mathrm{threat}} = 2.0$ doubles the effective cost in high-threat regions.

### 6.5 NSGA-III Multi-Objective Optimization

**Algorithm overview** (Deb & Jain 2014): NSGA-III extends NSGA-II with structured reference points on the normalized hyperplane, ensuring uniform diversity across the Pareto front.

**Optimization problem:**

$$\min_{\mathbf{x}} \mathbf{F}(\mathbf{x}) = [E(\mathbf{x}),\; T(\mathbf{x}),\; P_{\mathrm{detect}}(\mathbf{x})]$$

where $\mathbf{x}$ is a chromosome encoding waypoint positions and altitudes along the RRT*-planned path.

**Configuration:**

| Parameter | Value |
|-----------|-------|
| Population size | 100 individuals |
| Generations | 200 |
| Reference points | 15 (Das & Dennis structured simplex) |
| Crossover | SBX (simulated binary crossover), $\eta_c = 15$ |
| Mutation | Polynomial mutation, $\eta_m = 20$ |
| Chromosome | Waypoint positions + altitudes (continuous) |
| Constraint handling | Penalty for terrain violation, speed bounds |

**Non-dominated sorting:** For a population of size $N$, solutions are partitioned into fronts $F_1, F_2, \ldots$ where $F_1$ contains all non-dominated solutions. A solution $\mathbf{x}_i$ dominates $\mathbf{x}_j$ if $F_k(\mathbf{x}_i) \leq F_k(\mathbf{x}_j)$ for all $k$ and strictly less for at least one $k$.

**Reference point association:** Each solution on front $F_{l+1}$ (the partial front) is associated with the nearest reference point in the normalized objective space. Niching count $\rho_j$ counts how many members of $F_1 \cup \ldots \cup F_l$ are associated with reference point $j$. Solutions associated with the least-crowded reference points are preferentially selected.

**Knee-point selection** (for the recommended trajectory):
1. Normalize each objective to $[0, 1]$ using min/max of the Pareto front
2. Compute utopia point (minimum of each normalized objective)
3. Select the solution minimizing L₂ distance to the utopia point

This selection produces a balanced trade-off — not extreme on any single objective.

### 6.6 Trajectory Processing Pipeline

After NSGA-III produces the Pareto front, the recommended trajectory is processed through the following pipeline:

1. **B-spline smoothing** — fit a 5th-order B-spline to the RRT* waypoints, ensuring C² continuity (continuous position, velocity, and acceleration)
2. **Velocity profiling** — S-curve velocity schedule satisfying max speed, acceleration, and jerk constraints
3. **Feasibility checking** — verify all kinodynamic constraints after smoothing
4. **Timestamping** — compute mission time, phase transitions, and waypoint arrival times
5. **Energy evaluation** — compute total energy from the momentum-theory power model along the time-parameterized path

**Energy model (planning layer):**

For each flight segment, the required power is estimated from the induced velocity using actuator disk theory:

$$v_i = \sqrt{\frac{T}{2\rho A_{\mathrm{disk}}}}$$

$$P_{\mathrm{hover}} = T \cdot v_i = \frac{T^{3/2}}{\sqrt{2\rho A_{\mathrm{disk}}}}$$

In cruise, wing lift partially offloads the rotor, reducing induced power. The energy model in the planning layer is a conservative estimate — the vehicle layer provides the high-fidelity energy computation.

---

## 7. Vehicle Dynamics Layer

### 7.1 Vehicle Configuration

The simulated vehicle is a quad-tiltrotor eVTOL with four tilting nacelles, designed for the 450-lb gross weight class.

**Physical parameters:**

| Symbol | Value | Units | Description |
|--------|-------|-------|-------------|
| $W$ | 2,000 | N | Maximum take-off weight |
| $m$ | 203.9 | kg | Vehicle mass |
| $R$ | 1.0 | m | Rotor radius |
| $N_b$ | 4 | — | Blades per rotor |
| $N_r$ | 4 | — | Number of rotors |
| $\sigma$ | 0.05 | — | Rotor solidity $= N_b c / (\pi R)$ |
| $S$ | 4.0 | m² | Wing reference area |
| $AR$ | 8.0 | — | Wing aspect ratio |
| $b$ | 5.66 | m | Wing span $= \sqrt{AR \cdot S}$ |
| $C_{D,0}$ | 0.02 | — | Wing zero-lift drag coefficient |
| $E_{\mathrm{batt}}$ | 50,000 | Wh | Pack energy capacity |
| $V_{\mathrm{pack}}$ | 400 | V | Nominal pack voltage |
| $P_{\mathrm{motor,max}}$ | 50,000 | W | Peak motor shaft power |

**International Standard Atmosphere (ISA):**

$$T(h) = T_0 - L \cdot h, \quad p(h) = p_0\!\left(\frac{T(h)}{T_0}\right)^{g_0/(LR_{\mathrm{air}})}, \quad \rho(h) = \frac{p(h)}{R_{\mathrm{air}} \cdot T(h)}$$

where $T_0 = 288.15$ K, $p_0 = 101{,}325$ Pa, $L = 0.0065$ K/m, $g_0 = 9.807$ m/s². Sea-level density $\rho_0 = 1.225$ kg/m³.

### 7.2 Six-Degree-of-Freedom Equations of Motion

**State vector** (13-dimensional):

$$\mathbf{x} = \begin{pmatrix} \mathbf{r} \\ \mathbf{v} \\ \mathbf{q} \\ \boldsymbol{\omega} \end{pmatrix}, \quad \mathbf{r} \in \mathbb{R}^3 \text{ (NED position)}, \quad \mathbf{v} \in \mathbb{R}^3 \text{ (body velocity)}, \quad \mathbf{q} \in \mathbb{R}^4 \text{ (quaternion)}, \quad \boldsymbol{\omega} \in \mathbb{R}^3 \text{ (body rates)}$$

**Translational equations (Newton's second law in body frame):**

$$m \dot{\mathbf{v}} = \mathbf{F}_{\mathrm{aero}} + \mathbf{F}_{\mathrm{prop}} + \mathbf{F}_{\mathrm{grav}} - m \boldsymbol{\omega} \times \mathbf{v}$$

The gravity vector in body frame: $\mathbf{F}_{\mathrm{grav}} = m \mathbf{C}_{nb} [0, 0, g_0]^\top$ where $\mathbf{C}_{nb}$ is the Direction Cosine Matrix derived from the quaternion.

**Rotational equations (Euler's equations):**

$$\mathbf{I} \dot{\boldsymbol{\omega}} = \boldsymbol{\tau}_{\mathrm{aero}} + \boldsymbol{\tau}_{\mathrm{prop}} - \boldsymbol{\omega} \times (\mathbf{I} \boldsymbol{\omega})$$

Diagonal inertia tensor: $I_{xx} = 200$ kg·m², $I_{yy} = 300$ kg·m², $I_{zz} = 400$ kg·m².

**Quaternion kinematics** (avoids gimbal lock):

$$\dot{\mathbf{q}} = \frac{1}{2} \boldsymbol{\Omega}(\boldsymbol{\omega}) \mathbf{q}, \quad \boldsymbol{\Omega}(\boldsymbol{\omega}) = \begin{pmatrix} 0 & -p & -q & -r \\ p & 0 & r & -q \\ q & -r & 0 & p \\ r & q & -p & 0 \end{pmatrix}$$

**Numerical integration:** 4th-order Runge-Kutta (RK4) with quaternion renormalization after each step:

$$\mathbf{x}_{n+1} = \mathbf{x}_n + \frac{\Delta t}{6}(\mathbf{k}_1 + 2\mathbf{k}_2 + 2\mathbf{k}_3 + \mathbf{k}_4), \quad \mathbf{q} \leftarrow \mathbf{q}/\|\mathbf{q}\|$$

### 7.3 Blade Element Momentum Theory (BEMT)

BEMT combines two complementary aerodynamic theories: Blade Element Theory (BET), which computes section forces using 2D airfoil aerodynamics, and Momentum Theory (MT), which relates rotor induced velocity to thrust via conservation of momentum.

**Blade element forces:** At radial station $r$, the local angle of attack $\alpha$ and the section lift and drag are:

$$L(r) = \frac{1}{2}\rho V_r^2 c \cdot C_l(\alpha), \quad D(r) = \frac{1}{2}\rho V_r^2 c \cdot C_d(\alpha)$$

where $V_r = \sqrt{(\Omega r)^2 + v_i^2}$ is the resultant velocity, and the lift curve slope $C_{l,\alpha} = 2\pi$ (thin-airfoil theory).

**Momentum theory (actuator disk):**

$$T = 2\rho A_{\mathrm{disk}} v_i (v_{\infty} + v_i)$$

For hover ($v_{\infty} = 0$): $T = 2\rho A_{\mathrm{disk}} v_i^2$, giving $v_i = \sqrt{T/(2\rho A_{\mathrm{disk}})}$.

**Figure of Merit (FM):** Ratio of ideal induced power to actual shaft power:

$$\mathrm{FM} = \frac{P_{\mathrm{ideal}}}{P_{\mathrm{actual}}} = \frac{T^{3/2}/\sqrt{2\rho A_{\mathrm{disk}}}}{P_{\mathrm{shaft}}}$$

Typical values: FM = 0.65–0.80 for well-designed rotors.

**BEMT iteration:** Since $v_i$ appears on both sides (through inflow ratio $\lambda = v_i/(\Omega R)$), the BEMT equations are solved iteratively:
1. Assume initial $v_i$ from momentum theory
2. Compute section forces at each blade station
3. Integrate total thrust $T$
4. Update $v_i$ from momentum equation
5. Repeat until convergence ($|\Delta v_i| < 10^{-6}$ m/s)

### 7.4 Nacelle Tilt Transition Mechanics

The tiltrotor transitions between helicopter mode (nacelles vertical, $\delta_n = 90°$) and airplane mode (nacelles horizontal, $\delta_n = 0°$). The nacelle tilt angle $\delta_n(t)$ is commanded by the phase scheduler, with the transition following a half-cosine profile to avoid abrupt load changes:

$$\delta_n(t) = \frac{\delta_{\mathrm{start}} + \delta_{\mathrm{end}}}{2} + \frac{\delta_{\mathrm{start}} - \delta_{\mathrm{end}}}{2}\cos\!\left(\pi\frac{t - t_{\mathrm{start}}}{t_{\mathrm{transition}}}\right)$$

The aerodynamic forces during transition blend between the rotor-dominated hover model and the wing-dominated cruise model:

$$\mathbf{F}_{\mathrm{total}} = \sin(\delta_n)\, \mathbf{F}_{\mathrm{rotor}} + \cos(\delta_n)\, \mathbf{F}_{\mathrm{wing}}$$

Transition time: $t_{\mathrm{transition}} = 12$ s (half-cosine, nacelle 90° → 0°).

### 7.5 Battery and Energy Model

**Vehicle layer — 2-RC Equivalent Circuit Model (ECM):**

The battery is modeled as an ideal EMF source in series with an internal resistance $R_0$ and two RC branches $(R_1, C_1)$ and $(R_2, C_2)$ capturing polarization dynamics:

$$V_{\mathrm{terminal}} = V_{\mathrm{OCV}}(\mathrm{SOC}) - I R_0 - V_1 - V_2$$
$$\dot{V}_1 = -V_1/(R_1 C_1) + I/C_1, \quad \dot{V}_2 = -V_2/(R_2 C_2) + I/C_2$$

State of charge via Coulomb counting with Peukert correction:

$$\mathrm{SOC}(t) = \mathrm{SOC}(0) - \frac{1}{Q_{\mathrm{nom}}} \int_0^t I(t') \, dt', \quad Q_{\mathrm{eff}} = Q_{\mathrm{nom}} \cdot (I_{\mathrm{nom}}/I)^{k-1}, \; k = 1.05$$

**Control layer — momentum theory estimate:**

$$P_{\mathrm{mean}} = \frac{T_{\mathrm{mean}}^{3/2}}{\sqrt{2\rho A_{\mathrm{disk}}}}, \quad E = P_{\mathrm{mean}} \cdot T_{\mathrm{mission}} / 3600 \text{ [Wh]}$$

This is a conservative estimate (mean thrust, ignores wing lift in cruise). The vehicle layer energy is more accurate.

### 7.6 Acoustic Signature Model

Rotor noise is dominated by rotational noise (harmonic tones at blade passing frequency) and broadband noise (random turbulence interaction). The A-weighted sound pressure level at hover:

$$\mathrm{SPL}(r, h) = \mathrm{SPL}_0 + 10\log_{10}\!\left(\frac{T}{T_{\mathrm{ref}}}\right)^{k_T} - 20\log_{10}(r/r_{\mathrm{ref}}) - \Delta L_{\mathrm{atm}}(h)$$

where $k_T \approx 2.5$ (empirical thrust-noise exponent), $r$ is ground range to observer, and $\Delta L_{\mathrm{atm}}$ accounts for atmospheric absorption at altitude.

**Result:** At fixed vehicle mass (203.9 kg, hover thrust 2,000 N), $\mathrm{SPL}_{\mathrm{hover}} = 92.84$ dB(A) for all missions. This zero-variance behavior is a disclosed limitation (see Section 14).

### 7.7 Infrared Signature Model

The IR signature is driven by motor thermal emission. Motor temperature $T_m$ evolves as:

$$m_m c_m \dot{T}_m = P_{\mathrm{copper}}(t) - \frac{T_m - T_{\mathrm{ambient}}}{R_{\mathrm{th}}}$$

where $P_{\mathrm{copper}} = I^2 R_{\mathrm{winding}}$ is copper loss and $R_{\mathrm{th}}$ is the motor thermal resistance. The LWIR radiance (Stefan-Boltzmann):

$$M_{\mathrm{IR}} = \varepsilon \sigma T_m^4$$

where $\varepsilon \approx 0.9$ (painted aluminum nacelle), $\sigma = 5.67 \times 10^{-8}$ W·m⁻²·K⁻⁴.

The IR detection range is computed from a two-color pyrometer sensitivity model calibrated to MANPADS seeker characteristics.

### 7.8 Radar Cross-Section Model

The RCS at X-band (9.5 GHz) is estimated using physical optics for two aspects:

**Frontal RCS** (dominant in cruise, adversary radar ahead):

$$\sigma_{\mathrm{frontal}} = \frac{4\pi A_{\mathrm{frontal}}^2}{\lambda^2}$$

where $A_{\mathrm{frontal}}$ is the frontal projected area (fuselage + wing leading edge), $\lambda = c/f = 0.0316$ m.

**Top-down RCS** (dominant during hover, threat radar above or at angle):

$$\sigma_{\mathrm{top}} = \frac{4\pi A_{\mathrm{top}}^2}{\lambda^2}$$

where $A_{\mathrm{top}}$ includes the wing planform and rotor disk area.

Typical values: $\sigma_{\mathrm{frontal}} \approx -10$ to $+5$ dBsm; $\sigma_{\mathrm{top}} \approx 0$ to $+10$ dBsm.

---

## 8. Control Layer

### 8.1 Overview

The control layer converts planned waypoint sequences into motor thrust commands that physically drive the vehicle. It implements a cascaded PID architecture running at 50 Hz (Δt = 0.02 s), covering the full mission profile from takeoff through hover, transition, cruise, and landing.

### 8.2 Cascaded Loop Architecture

Five nested loops from outermost (slow) to innermost (fast):

```
[Position Loop]   x_cmd, y_cmd  →  v_x_cmd, v_y_cmd       (Kp_pos = 0.80)
[Velocity Loop]   v_cmd         →  pitch_cmd, roll_cmd      (Kp_vel = 0.15)
[Altitude Loop]   z_cmd         →  climb_rate_cmd → F_coll  (Kp_alt = 1.20)
[Attitude Loop]   φ_cmd, θ_cmd  →  p_cmd, q_cmd, r_cmd     (Kp_att = 6.50)
[Rate Loop]       p/q/r_cmd     →  Mx, My, Mz               (Kp_rt  = 120.0)
```

**Gain table:**

| Loop | $K_p$ | $K_i$ | $K_d$ | Anti-windup |
|------|-------|-------|-------|-------------|
| Position (x, y) | 0.80 | 0.02 | 0.20 | ±10 m·s |
| Velocity (Vx, Vy) | 0.15 | 0.02 | 0.05 | ±5 m·s²/g |
| Altitude (outer) | 1.20 | 0.08 | 0.30 | ±20 m/s |
| Altitude (inner) | 180.0 | 12.0 | 8.0 | ±800 N·s |
| Attitude (φ, θ, ψ) | 6.50 | 0.10 | 0.40 | ±2 rad/s |
| Rate (p, q, r) | 120.0 | 4.0 | 5.0 | ±50 N·m·s |

### 8.3 PID Mathematics

**Discrete-time PID with clamped integration:**

$$u(k) = K_p \cdot e(k) + K_i \cdot \sigma(k) + K_d \cdot \dot{e}(k)$$

$$\sigma(k) = \mathrm{clip}\!\left(\sigma(k-1) + e(k)\Delta t,\; -\sigma_{\max},\; +\sigma_{\max}\right)$$

$$\dot{e}(k) = \frac{e(k) - e(k-1)}{\Delta t}$$

**Altitude loop (cascaded):**

Outer loop (altitude → climb rate):
$$\dot{z}_{\mathrm{cmd}}(k) = K_{p,\mathrm{alt}} \cdot e_z(k) + K_{i,\mathrm{alt}} \cdot \sigma_z(k) + K_{d,\mathrm{alt}} \cdot \dot{e}_z(k)$$

Inner loop (climb rate → collective thrust with weight feedforward):
$$F_{\mathrm{coll}}(k) = \underbrace{W}_{\text{feedforward}} + K_{p,\mathrm{cr}} \cdot e_{\dot{z}}(k) + K_{i,\mathrm{cr}} \cdot \sigma_{\dot{z}}(k) + K_{d,\mathrm{cr}} \cdot \dot{e}_{\dot{z}}(k)$$

The weight feedforward $W = mg = 2{,}000$ N cancels steady-state gravity, eliminating the need for a large integral term to hold hover altitude.

**Velocity-to-attitude conversion:**

$$\theta_{\mathrm{cmd}} = \arctan\!\left(\frac{-a_{x,\mathrm{cmd}}}{g}\right), \quad \phi_{\mathrm{cmd}} = \arctan\!\left(\frac{a_{y,\mathrm{cmd}}}{g}\right)$$

where $a_{x,\mathrm{cmd}} = K_{p,v} \cdot e_{v_x} + K_{i,v} \cdot \sigma_{v_x} + K_{d,v} \cdot \dot{e}_{v_x}$. The sign convention (negative pitch for forward acceleration) was a critical bug fix (see Section 14).

### 8.4 Phase Scheduler

Each mission follows a deterministic phase schedule:

| Phase | Duration | Nacelle | Velocity Ref | Altitude Ref |
|-------|----------|---------|--------------|--------------|
| TAKEOFF | min(15 s, 5% mission time) | 90° (vertical) | 0 m/s | Ramp 0 → cruise_alt |
| HOVER | 30% of planned hover time | 90° | 0 m/s | cruise_alt |
| TRANS1 | 12 s | 45° | 50% cruise speed | cruise_alt |
| CRUISE | Remaining | 0° (horizontal) | cruise speed | cruise_alt |
| TRANS2 | 12 s | 45° | 30% cruise speed | cruise_alt |
| HOVER2 | 30% of planned hover time | 90° | 0 m/s | cruise_alt |
| LAND | 15 s | 90° | 0 m/s | Ramp cruise_alt → 0 |

Takeoff altitude is ramped at a rate the vehicle can physically achieve. The altitude-dependent takeoff duration ensures the climb rate reference never exceeds $C_{R,\mathrm{max}} = 20$ m/s (revised from original 8 m/s to support high-altitude regions).

### 8.5 Motor Allocation (Mixing Matrix)

The four motor thrusts $T_1, T_2, T_3, T_4$ (and corresponding torques) are determined from collective thrust $F_{\mathrm{coll}}$, roll moment $M_x$, pitch moment $M_y$, and yaw moment $M_z$:

$$\begin{pmatrix} T_1 \\ T_2 \\ T_3 \\ T_4 \end{pmatrix} = \mathbf{M}^{-1} \begin{pmatrix} F_{\mathrm{coll}} \\ M_x \\ M_y \\ M_z \end{pmatrix}$$

where $\mathbf{M}$ is the 4×4 mixing matrix determined by rotor geometry (arm length $l = 1$ m, rotor torque constant $k_\tau = 0.05$ m). The matrix is invertible for the symmetric quad configuration.

---

## 9. Multi-Region Dataset

### 9.1 Geographic Coverage

The dataset covers six Indian geographic regions representing diverse operational theatres:

| Region | Center | Terrain Type | Elevation | Key Features |
|--------|--------|-------------|-----------|--------------|
| Delhi NCR | 28.85°N, 77.15°E | Mixed urban/agricultural | 190–305 m MSL | Primary benchmark; Aravalli foothills |
| Mumbai | 19.08°N, 72.88°E | Coastal urban | 0–50 m MSL | Sea breeze, dense obstacle field |
| Bangalore | 12.97°N, 77.59°E | Elevated plateau | 850–950 m MSL | Elevated ISA density reduction |
| Arunachal Pradesh | 27.10°N, 93.62°E | High-altitude mountainous | 500–4000 m MSL | Extreme terrain relief, sparse obstacles |
| Odisha | 20.29°N, 85.82°E | Coastal flat | 0–100 m MSL | Low terrain variability |
| Ladakh | 34.17°N, 77.58°E | High-altitude arid | 3400–5500 m MSL | Very low air density, extreme cold |

### 9.2 Dataset Composition

**Per-region record counts:**

| Region | Planning | Vehicle | Control | Notes |
|--------|----------|---------|---------|-------|
| Delhi NCR | 10,000 | 10,000 | 10,000 | Primary benchmark |
| Mumbai | 2,000 | 2,000 | 2,000 | |
| Bangalore | 2,000 | 2,000 | 2,000 | |
| Arunachal Pradesh | 2,000 | 2,000 | 2,000 | |
| Odisha | 2,000 | 2,000 | 2,000 | obstacle_cost = 1.0 (API timeout) |
| Ladakh | 2,000 | 2,000 | 2,000 | Very low air density |
| **Total** | **22,000** | **22,000** | **22,000** | |

**Directory structure:**

```
datasets/
├── delhi/
│   ├── planning/               # planning_dataset.parquet + splits/
│   ├── vehicle/                # vehicle_dataset.parquet
│   ├── control/                # control_dataset.parquet
│   └── perception/             # perception_full_dataset.parquet (~1M rows)
├── mumbai/
│   ├── planning_dataset/       # planning_mumbai.parquet
│   ├── vehicle/
│   └── control/
├── bangalore/   ...
├── arunachal/   ...
├── odisha/      ...
└── ladakh/      ...
```

### 9.3 Data Splits

Stratified 80/10/10 train/validation/test splits are pre-computed for each region, stratified by `risk_label` to ensure balanced class distribution across splits. Seed: 42. Index arrays are stored as `.npy` files in `<region>/splits/`.

### 9.4 Feature Schemas

**Planning layer** (25 features per mission):

Key features include: `start_lat`, `start_lon`, `goal_lat`, `goal_lon` (mission geometry), `path_length_m` (Euclidean and RRT* path length), `cruise_speed_ms`, `cruise_altitude_m`, `mission_time_s`, `hover_time_s` (mission profile), `energy_planned_wh` (planning layer energy estimate), `threat_cost` (integrated SAM exposure), `risk_label` (binary, fused_cost ≥ 0.55), `n_waypoints` (RRT* path complexity).

**Vehicle layer** (95 features per mission):

Includes all planning features plus: `energy_consumed_wh` (high-fidelity energy from BEMT), `soc_initial`, `soc_final`, `max_soc_drop`, `pack_capacity_wh` (energy system), `spl_hover_a_dB`, `spl_cruise_a_dB` (acoustic), `ir_hover_W_sr`, `ir_cruise_W_sr` (infrared), `rcs_hover_dBsm`, `rcs_cruise_x_dBsm`, `rcs_cruise_z_dBsm` (radar cross-section), `rotor_efficiency_hover`, `rotor_efficiency_cruise` (propulsion), `max_combined_threat` (re-computed from planning).

**Control layer** (76 features per mission):

Includes mission parameters from vehicle layer plus closed-loop metrics: `pos_error_mean_m`, `alt_error_mean_m`, `alt_error_rms_m`, `vel_settling_mean_s`, `alt_settling_mean_s` (tracking performance), `n_saturations` (number of actuator saturation events), `energy_control_wh` (momentum-theory energy estimate), `soc_control_final`, `mission_abort` (binary abort flag, 0 for all current missions).

---

## 10. Machine Learning Benchmark Tasks

### 10.1 Task Definitions

Six ML tasks are formally defined, spanning classification and regression:

**T1 — Risk Classification (Primary Task):**
- Target: `risk_label ∈ {0, 1}` (binary, fused_cost ≥ 0.55 → high risk)
- Input features: 25 planning features
- Metric: AUC-ROC, F1 score, accuracy
- Challenge: Identify which missions require additional planning attention before flight

**T2 — Energy Regression:**
- Target: `energy_consumed_wh` (continuous, ~100–3,000 Wh range)
- Input features: Planning geometry + wind features (no vehicle-layer features)
- Metric: R², MAE, RMSE
- Challenge: Accurate pre-flight energy forecasting from mission plan alone

**T3 — Abort Classification:**
- Target: `mission_abort ∈ {0, 1}`
- Status: Trivially degenerate — `mission_abort = 0` for all current records. T3 is excluded from current baselines.
- Note: Requires missions that push the vehicle beyond battery or thermal limits; reserved for future dataset extension

**T4 — Altitude Tracking Regression:**
- Target: `alt_error_mean_m` (continuous)
- Input features: Vehicle parameters + planning features
- Metric: R², RMSE

**T5 — Threat Exposure Regression:**
- Target: `threat_cost` (continuous, 0.45–1.0)
- Input features: Mission geometry + terrain features
- Metric: R², MAE

**T6 — Final SOC Regression:**
- Target: `soc_final` (continuous, battery state of charge at mission end)
- Input features: Planning + vehicle features
- Metric: R², MAE

### 10.2 Experimental Protocol

All ML baselines use:
- **5-fold stratified cross-validation** (stratified on `risk_label` for classification tasks, on binned target for regression)
- **Preprocessing:** StandardScaler (zero mean, unit variance) on continuous features; label encoding for any categorical features
- **Three model families:**
  1. Logistic Regression / Ridge Regression (linear baseline)
  2. Gradient Boosting Machine (GBM) — 100 estimators, max_depth=4
  3. Multi-Layer Perceptron (MLP) — hidden layers (128, 64), ReLU, Adam optimizer, 200 epochs
- **Evaluation:** Mean ± std across 5 folds; best model selected per metric

---

## 11. ML Baseline Results

### 11.1 Delhi NCR — Primary Benchmark

**T1 — Risk Classification:**

| Model | AUC-ROC | F1 Score | Accuracy |
|-------|---------|----------|----------|
| Logistic Regression | 0.9995 | 0.9987 | 0.9998 |
| Gradient Boosting | 0.9997 | 0.9991 | 0.9998 |
| MLP | 0.9998 | 0.9993 | 0.9999 |

**T2 — Energy Regression:**

| Model | R² | MAE (Wh) | RMSE (Wh) |
|-------|----|----------|-----------|
| Ridge Regression | 0.9981 | 12.4 | 18.7 |
| Gradient Boosting | 0.9986 | 9.8 | 15.2 |
| MLP | 0.9984 | 10.6 | 16.3 |

The extremely high performance (AUC 0.9998, R² 0.9986) on Delhi reflects that the dataset is generated from a deterministic physics-based pipeline — the features fully determine the targets within the dataset. This is a characteristic of simulation-generated datasets and is disclosed in the datasheet.

### 11.2 Multi-Region Results Summary

**T1 AUC-ROC across regions (best model):**

| Region | T1 AUC | Notes |
|--------|--------|-------|
| Delhi NCR | 0.9998 | Near-perfect |
| Mumbai | 0.9942 | Coastal urban diversity |
| Bangalore | 0.9908 | Elevated plateau |
| Arunachal Pradesh | ~0.50–0.85 | Near-degenerate: near-zero positive rate in high-altitude terrain |
| Odisha | 0.9855 | obstacle_cost = 1.0 for all records |
| Ladakh | ~0.50–0.80 | Near-degenerate: high-altitude arid, very low risk_label positive rate |

**T2 R² across regions (best model):**

| Region | T2 R² | Notes |
|--------|-------|-------|
| Delhi NCR | 0.9986 | |
| Mumbai | 0.9921 | |
| Bangalore | 0.9878 | |
| Arunachal Pradesh | 0.9834 | Lower wind variance at high altitude |
| Odisha | 0.9896 | |
| Ladakh | 0.9801 | Very low air density affects energy scaling |

**Finding on Arunachal Pradesh and Ladakh T1:** The `risk_label` threshold (fused_cost ≥ 0.55) produces near-zero positive rates in high-altitude mountainous and arid terrain, where obstacle density is very low and the cost field is dominated by wind. This is a genuine geography-dependent calibration issue — the threshold was tuned for Delhi's mixed urban/agricultural terrain. Threshold calibration per region is recommended for future work.

**T3 (Abort Classification):** Skipped for all non-Delhi regions. `mission_abort_rate = 0.00%` everywhere; falls below the 2% minimum class frequency threshold for meaningful classification.

---

## 12. Software Architecture and Implementation

### 12.1 Repository Structure

```
trajectory-optimization-in-defense-evtols/
├── src/evtol/                # Core Python library
│   ├── core/                 # Canonical cross-layer types
│   │   ├── state.py          # VehicleState (re-exported), FlightPhase enum
│   │   └── environment.py    # WindField, ThreatField interfaces
│   ├── perception/           # Perception pipeline
│   ├── planning/             # Planning algorithms
│   ├── control/              # Autopilot and SITL
│   └── vehicle/              # Physics models
├── scripts/                  # Dataset generation pipeline
│   ├── perception/           # dataset.py, visuals.py
│   ├── planning/             # dataset.py, visualize.py
│   ├── vehicle/              # dataset.py, visualize.py
│   ├── control/              # dataset.py, visualize.py
│   ├── ml/                   # baseline_experiments.py, run_all_regions.py
│   └── export_all_formats.py # Multi-format export
├── datasets/                 # All output datasets
│   ├── delhi/
│   ├── mumbai/
│   ├── bangalore/
│   ├── arunachal/
│   ├── odisha/
│   └── ladakh/
├── doc/                      # Technical documentation
│   ├── architecture.md
│   ├── perception_layer.md
│   ├── planning_layer.md
│   ├── vehicle_layer.md
│   ├── control_layer.md
│   ├── datasheet.md
│   ├── data_guide.md
│   └── research_limitations.md
└── notebooks/                # Tutorial Jupyter notebook
```

### 12.2 Type Ownership and Import Rules

Enforced architecture invariants:

| Type | Canonical Location | Import Path |
|------|--------------------|------------|
| `VehicleState` | `vehicle/dynamics/state.py` | `from evtol.core.state import VehicleState` |
| `FlightPhase` | `evtol/core/state.py` | `from evtol.core.state import FlightPhase` |
| `SITLState` | `control/sitl_simulator.py` | Hardware telemetry (ENU/Euler) — not `VehicleState` |
| `ModeInputState` | `control/advanced_modes/advanced_modes.py` | Mode-selector inputs |
| `TiltrotorVehicle` | `vehicle/vehicle_model.py` | Canonical 6-DoF simulation class |
| Threat-aware RRT* | `planning/rrt_star.py` | Defense mission use case |
| Generic sampling | `planning/algorithms/sampling.py` | Framework utilities only |
| NSGA-III (canonical) | `planning/optimization/nsga3.py` | Full implementation |
| NSGA-III (re-export) | `planning/nsga3_optimizer.py` | Thin re-export for compatibility |

### 12.3 Circular Import Resolution

The `evtol.core.state` module re-exports `VehicleState` from `vehicle/dynamics/state.py` without triggering `vehicle/__init__.py` (which would cause a circular import). This is achieved via `importlib.util.spec_from_file_location`:

```python
# src/evtol/core/state.py (simplified)
import importlib.util, pathlib, sys

_state_file = pathlib.Path(__file__).parent.parent / "vehicle" / "dynamics" / "state.py"
_spec = importlib.util.spec_from_file_location("evtol.vehicle.dynamics.state", _state_file)
_mod = importlib.util.module_from_spec(_spec)
if "evtol.vehicle.dynamics.state" not in sys.modules:
    sys.modules["evtol.vehicle.dynamics.state"] = _mod
    _spec.loader.exec_module(_mod)
VehicleState = sys.modules["evtol.vehicle.dynamics.state"].VehicleState
```

This pattern bypasses the package hierarchy by loading the file directly by path, breaking the cycle: `core.state → vehicle.__init__ → vehicle_model → core.state`.

### 12.4 Data Pipeline

The pipeline is designed to be fully reproducible from public APIs:

1. **`scripts/perception/dataset.py`** — Calls Open-Elevation (SRTM), Open-Meteo (GFS), and OSM Overpass APIs. Runtime ~30 min (API-bound). Output: perception_full_dataset.parquet (~1M rows, 28 features).

2. **`scripts/planning/dataset.py`** — Loads perception cost field, runs RRT* + NSGA-III for each mission. Runtime ~25 hours for 10,000 records (parallelized). Output: planning_dataset.parquet.

3. **`scripts/vehicle/dataset.py`** — Loads planning dataset, runs 6-DoF simulation (BEMT + aerodynamics + battery) for each mission. Runtime ~10 min. Output: vehicle_dataset.parquet.

4. **`scripts/control/dataset.py`** — Loads vehicle dataset, runs closed-loop 50 Hz PID simulation. Runtime ~12 min. Output: control_dataset.parquet.

5. **`scripts/export_all_formats.py`** — Exports all parquets to csv/npz/h5/feather/pkl/json and all PNGs to pdf/svg/eps.

### 12.5 Dependencies

| Package | Version | Role |
|---------|---------|------|
| Python | 3.11+ | Core language |
| pandas | 2.x | Tabular data handling |
| numpy | 1.26+ | Numerical arrays |
| scikit-learn | 1.4+ | ML baselines |
| matplotlib | 3.8+ | Visualization |
| pyarrow | 14+ | Parquet I/O |
| scipy | 1.12+ | Scientific algorithms |
| requests | 2.31+ | API calls |
| tables (PyTables) | 3.9+ | HDF5 export |

---

## 13. Data Formats and Reproducibility

### 13.1 Available Formats

Every dataset is available in seven tabular formats and every visualization in four image formats:

**Tabular formats:**

| Format | Extension | Use Case |
|--------|-----------|----------|
| Apache Parquet | `.parquet` | Primary format — columnar, compressed, typed |
| CSV | `.csv` | Universal compatibility |
| NumPy compressed | `.npz` | Direct numpy array loading, numerical columns only |
| HDF5 (PyTables) | `.h5` | Large-scale ML frameworks, blosc compression |
| Apache Feather | `.feather` | Fast I/O for pandas/R interoperability |
| Python Pickle | `.pkl` | Python-native serialization (full DataFrame) |
| JSON | `.json` | Human-readable (skipped if > 500,000 cells) |

**Image formats:**

| Format | Extension | Use Case |
|--------|-----------|----------|
| PNG | `.png` | Primary raster format |
| PDF | `.pdf` | Publication-quality vector |
| SVG | `.svg` | Web-scalable vector |
| EPS | `.eps` | LaTeX figure inclusion |

### 13.2 Loading Data

**Python (pandas):**
```python
import pandas as pd

# Load primary Delhi planning dataset
df = pd.read_parquet("datasets/delhi/planning/planning_dataset.parquet")

# Or load CSV version
df = pd.read_csv("datasets/delhi/planning/planning_dataset.csv")
```

**NumPy:**
```python
import numpy as np
data = np.load("datasets/delhi/planning/planning_dataset.npz")
# Access columns: data['path_length_m'], data['energy_planned_wh'], ...
```

**HDF5:**
```python
import pandas as pd
df = pd.read_hdf("datasets/delhi/planning/planning_dataset.h5", key="data")
```

### 13.3 Reproducibility

The full pipeline is reproducible from public APIs with no proprietary data requirements:
- All API endpoints are documented in `doc/data_guide.md`
- API rate limiting parameters and retry logic are documented in `scripts/perception/dataset.py`
- Random seeds for NSGA-III and train/val/test splits: seed=42
- All intermediate outputs are versioned in the `datasets/` directory

---

## 14. Known Limitations and Disclosures

### 14.1 Threat Saturation

`combined_threat_prob = 1.000 ± 0.000` for all perception grid cells (by design — the scenario models "heavily contested airspace"). The planning layer substitutes a distance-decay threat gradient (mean=0.79, std=0.12) to provide spatial variation for the optimizer. The physics-based `combined_threat_prob` is retained for disclosure purposes but not used in planning cost computation.

**Impact:** The three-objective Pareto front reduces to effectively a 2D trade-off (energy vs. time) in the raw threat field. The spatial threat gradient used in planning has genuine variation but is not physically derived from the Mahafza radar model.

### 14.2 Position Error Metric (Historical)

Prior to v1.3, `pos_error_mean_m ≈ 43,647 m` due to a reference signal accumulation bug and incorrect pitch/roll sign convention. Three bugs were corrected:
1. Per-phase position reference reset (`px_phase_ref` reset at every phase transition)
2. Pitch/roll sign error in `scripts/control/dataset.py` (`atan2(-accel_x_cmd, G)` not `atan2(accel_x_cmd, G)`)
3. Velocity command ceiling too low (`VEL_CMD_MAX` 25 → 120 m/s)

**After fixes:** `pos_error_mean = 12.3 m`, `alt_error_mean = 22.3 m`, `vel_settling_mean = 2.7 s`.

### 14.3 Zero Acoustic Variance

`spl_hover_a_dB = 92.84 ± 0.00 dB(A)` for all missions. Fixed vehicle mass → fixed hover thrust → fixed acoustic signature. This feature has zero discriminative power for ML tasks. **Recommended fix:** Vary vehicle mass or model RPM-dependent noise with efficiency optimization.

### 14.4 Near-Degenerate T1 in High-Altitude Regions

Arunachal Pradesh and Ladakh have near-zero `risk_label` positive rates (< 5%) due to very low obstacle density and a `risk_label` threshold (fused_cost ≥ 0.55) calibrated for the Delhi urban environment. Classification baselines yield AUC near 0.5 for these regions using the current threshold.

### 14.5 T3 Mission Abort Triviality

`mission_abort = 0` for all 22,000 missions. The abort logic (triggered by battery depletion or thermal overrun) is never activated because the planning layer constrains all missions to the vehicle's energy budget. T3 is excluded from current baselines.

### 14.6 Simplified Control Plant

The control layer uses a simplified 6-DoF plant (linear drag, no wing lift in cruise, no rotor dynamics, no sensor noise). Specific approximations:
- Drag force: $F_{\mathrm{drag}} = -k_v \mathbf{v}$ (linear in body velocity, no quadratic term)
- No aerodynamic coupling between rotor and wing in transition flight
- Sensor noise: zero-mean Gaussian noise added to position and velocity states only

This does not affect the vehicle layer (which uses full BEMT + detailed aerodynamics), but the control tracking metrics reflect the simplified plant's dynamics.

### 14.7 Odisha Obstacle Data

`obstacle_cost = 1.0` (maximum) for all Odisha records due to Overpass API gateway timeouts during data collection. The obstacle cost column is not usable for Odisha-specific ML tasks.

### 14.8 Dual-Use Assessment

The SAM threat model uses parameters from Mahafza (2005), a publicly available textbook. No classified system parameters are modeled. The geographic region (Delhi-NCR) is public domain terrain data. All obstacle data is from OpenStreetMap's public dataset. We judge the research benefit of an open defense autonomous systems benchmark to outweigh dual-use concerns, consistent with dual-use reasoning frameworks established in prior adversarial ML and cybersecurity benchmarking literature.

---

## 15. Future Work

### 15.1 Partial SAM Coverage Scenario

Add an alternate threat scenario with 2–3 emitters with non-overlapping detection ranges, creating genuine threat-free corridors. This would make the three-objective Pareto front meaningful and enable threat-avoidance route planning research.

### 15.2 Variable Vehicle Mass

Introduce mission-to-mission variation in payload mass (0–50 kg) to generate acoustic and performance variability. This would create genuine discriminative power for the acoustic signature features.

### 15.3 Dynamic Threat Scenarios

Model dynamic threats (mobile SAM launcher, airborne threats) using the Kalman tracker implemented in `sensor_fusion.py`. Generate planning trajectories that react to threat movement, creating time-varying cost fields.

### 15.4 Hardware-in-the-Loop Integration

Connect the SITL (`control/sitl_simulator.py`) to a real flight controller (PX4 or ArduPilot) over MAVLink for hardware-in-the-loop validation of control performance metrics.

### 15.5 Region-Specific Risk Thresholds

Calibrate the `risk_label` threshold per region based on the local cost field distribution, instead of a global 0.55 threshold. This would produce balanced class distributions for all six regions.

### 15.6 Neural Trajectory Optimizer

Train a neural network to approximate the NSGA-III Pareto front directly, enabling sub-second trajectory optimization for real-time replanning. The 22,000-record dataset is a suitable training set for a small neural approximator.

### 15.7 Multi-Vehicle Scenarios

Extend the dataset to cooperative multi-vehicle missions (formation flight, relay resupply chains), leveraging the modular four-layer architecture.

### 15.8 Real Hardware Validation

Validate the vehicle physics model (BEMT power, transition aerodynamics, battery ECM) against flight data from a real 200-class tiltrotor UAV. The vehicle layer's 95-feature dataset provides comprehensive ground truth for physics model validation.

---

## 16. References

1. **Karaman, S., & Frazzoli, E. (2011).** Sampling-based algorithms for optimal motion planning. *International Journal of Robotics Research*, 30(7), 846–894.

2. **Deb, K., & Jain, H. (2014).** An evolutionary many-objective optimization algorithm using reference-point-based nondominated sorting approach. *IEEE Transactions on Evolutionary Computation*, 18(4), 577–601.

3. **Mahafza, B. R. (2005).** *Radar Systems Analysis and Design Using MATLAB* (2nd ed.). CRC Press. [Primary source for SAM detection probability model.]

4. **Johnson, W. (2013).** *Rotorcraft Aeromechanics*. Cambridge University Press. [BEMT theory, rotor noise model.]

5. **Rodriguez, E., Morris, C. S., & Belz, J. E. (2006).** A global assessment of the SRTM performance. *Photogrammetric Engineering & Remote Sensing*, 72(3), 249–260.

6. **Lin, Y., & Saripalli, S. (2017).** Sampling-based path planning for UAV collision avoidance. *IEEE Transactions on Intelligent Transportation Systems*, 14(2), 916–928.

7. **Guerra, W., et al. (2019).** Fast trajectory optimization for agile quadrotor maneuvers with a cable-suspended payload. *Robotics: Science and Systems (RSS)*.

8. **Gebru, T., et al. (2021).** Datasheets for datasets. *Communications of the ACM*, 64(12), 86–92. [Framework followed for the dataset datasheet.]

9. **Das, I., & Dennis, J. E. (1998).** Normal-boundary intersection: A new method for generating the Pareto surface in nonlinear multicriteria optimization problems. *SIAM Journal on Optimization*, 8(3), 631–657. [Reference point generation for NSGA-III.]

10. **Open-Meteo (2023).** *Open-Meteo Weather API Documentation.* Open-Meteo.com. [GFS wind forecast and SRTM elevation API.]

11. **OpenStreetMap Contributors (2023).** *Overpass API*. overpass-api.de. [Obstacle data source.]

12. **Farrell, J. A., & Barth, M. (1999).** *The Global Positioning System and Inertial Navigation*. McGraw-Hill. [WGS-84 coordinate frame definitions.]

13. **Selig, M. S. (2010).** *UIUC Airfoil Database.* University of Illinois at Urbana-Champaign. [Airfoil lift/drag coefficients for BEMT.]

14. **Stevens, B. L., Lewis, F. L., & Johnson, E. N. (2015).** *Aircraft Control and Simulation* (3rd ed.). Wiley. [6-DoF equations of motion, PID autopilot design.]

15. **Pedregosa, F., et al. (2011).** Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830. [ML baseline implementation.]

---

*Made with ❤️ by Aditi Ramakrishnan*
