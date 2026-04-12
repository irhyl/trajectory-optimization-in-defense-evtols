# System Architecture

## 1. Overview

The framework implements a **four-layer autonomous mission stack** where each layer produces a structured dataset consumed by the next. The architecture mirrors the autonomy stack of a real defense UAV system, making the dataset directly applicable to training and evaluating real autopilot components.

```
                        ┌──────────────────────┐
                        │   MISSION PARAMETERS  │
                        │  (start, goal, time   │
                        │   budget, threat alert)│
                        └──────────┬───────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │      LAYER 1: PERCEPTION     │
                    │                              │
                    │  Inputs:  GPS coords, time   │
                    │  Process: API calls + fusion  │
                    │  Output:  4D cost field C(.) │
                    │           1,057,714 rows     │
                    │           28 features        │
                    └──────────────┬──────────────┘
                                   │ perception_full_dataset.parquet
                    ┌──────────────▼──────────────┐
                    │      LAYER 2: PLANNING       │
                    │                              │
                    │  Inputs:  cost field, goals  │
                    │  Process: RRT* + NSGA-III    │
                    │  Output:  Pareto trajectories │
                    │           2,000 records      │
                    │           25 features        │
                    └──────────────┬──────────────┘
                                   │ test_final.parquet
                    ┌──────────────▼──────────────┐
                    │      LAYER 3: VEHICLE        │
                    │                              │
                    │  Inputs:  trajectory plan    │
                    │  Process: 6-DoF simulation   │
                    │  Output:  physics features   │
                    │           2,000 records      │
                    │           95 features        │
                    └──────────────┬──────────────┘
                                   │ vehicle_dataset.parquet
                    ┌──────────────▼──────────────┐
                    │      LAYER 4: CONTROL        │
                    │                              │
                    │  Inputs:  vehicle parameters │
                    │  Process: 50 Hz PID sim      │
                    │  Output:  control metrics    │
                    │           2,000 records      │
                    │           76 features        │
                    └─────────────────────────────┘
```

---

## 2. Data Flow

### 2.1 Perception → Planning

The planning layer queries the perception dataset via spatial interpolation. For each candidate waypoint (lat, lon, alt), it interpolates the fused cost field to get a scalar traversal cost. This cost is used:
- As the edge weight in RRT* (costmap-based planning)
- As the threat component in the NSGA-III energy/time/threat objectives

The bridge is implemented in `src/evtol/planning/perception_integration.py`.

### 2.2 Planning → Vehicle

Each planning trajectory record contains the mission profile:
- `cruise_speed_ms` — used for propulsion regime selection
- `cruise_altitude_m` — used for aerodynamic model interpolation
- `mission_time_s` — used to set simulation duration
- `hover_time_s` — used to compute phase schedule durations
- `n_waypoints` — used for phase transition counting

The vehicle layer simulates each trajectory through physics models without re-running the planner.

### 2.3 Vehicle → Control

The control layer receives per-mission parameters from the vehicle dataset:
- `mission_time_s`, `cruise_speed_ms`, `cruise_altitude_m` → phase schedule
- `hover_time_s` → HOVER and HOVER2 durations
- `pack_capacity_wh`, `soc_initial` → energy/SOC computation
- `spl_hover_a_dB`, `rcs_cruise_x_dBsm`, `max_combined_threat` → passed through

The control simulation then closes the loop with a simplified 6-DoF plant, recording tracking performance metrics that are NOT in the vehicle dataset (which uses a higher-fidelity offline model).

---

## 3. Module Structure

### 3.1 src/evtol/ — Core Library

```
src/evtol/
├── perception/          # Real-data perception and cost field
│   ├── terrain/         # SRTM elevation, slope, roughness
│   ├── wind/            # GFS wind forecast, turbulence
│   ├── obstacle/        # OSM geometry, conflict detection
│   ├── threat/          # SAM detection probability, engagement model
│   └── fusion/          # Weighted cost aggregation
│
├── planning/            # Multi-objective trajectory planning
│   ├── algorithms/      # RRT* (sampling.py), graph search (graph.py)
│   ├── optimization/    # NSGA-III (nsga3.py), Pareto (pareto.py)
│   ├── core/            # Trajectory, cost, constraints, energy evaluator
│   ├── trajectory/      # Smooth trajectory generation, replanning
│   ├── mission/         # Mission planner, contingency
│   └── robust/          # Chance constraints, uncertainty propagation
│
├── control/             # Cascaded PID autopilot
│   ├── outer_loop/      # Position, velocity, altitude, heading
│   ├── inner_loop/      # Attitude, rate
│   ├── modes/           # Hover, cruise, transition, mode manager
│   ├── allocation/      # Mixing matrix, nacelle scheduler
│   └── guidance/        # Mission manager, path follower, tracker
│
└── vehicle/             # Vehicle physics models
    ├── dynamics/        # Newton-Euler 6-DoF, state vector, integrator
    ├── propulsion/      # BEMT rotor, motor, nacelle, power system
    ├── aerodynamics/    # Wing, fuselage, transition aerodynamics
    ├── energy/          # Battery model, thermal, power management
    └── signatures/      # Acoustic, infrared, RCS models
```

### 3.2 scripts/ — Dataset Pipeline

Each layer has a dedicated dataset generator and visualizer:

| Script | Input | Output | Runtime |
|--------|-------|--------|---------|
| `perception/dataset.py` | API calls | 1,057,714-row Parquet | ~30 min (API-bound) |
| `perception/visuals.py` | perception Parquet | 21 figures | ~5 min |
| `planning/dataset.py` | perception Parquet | 2,000-row Parquet | ~15 min |
| `planning/visualize.py` | planning Parquet | 38 figures | ~3 min |
| `vehicle/dataset.py` | planning Parquet | 2,000-row Parquet | ~10 min |
| `vehicle/visualize.py` | vehicle Parquet | 20 figures + SVG | ~5 min |
| `control/dataset.py` | vehicle Parquet | 2,000-row Parquet | ~12 min |
| `control/visualize.py` | control Parquet | 15 figures | ~2 min |

---

## 4. Coordinate Frames

Three coordinate frames are used across the stack:

### 4.1 Geographic (WGS-84)
- **Used by:** Perception layer, planning layer (waypoints)
- **Format:** (latitude °, longitude °, altitude m MSL)
- **Datum:** WGS-84 ellipsoid

### 4.2 NED (North-East-Down)
- **Used by:** Planning (local path), control (position/velocity states)
- **Conversion from geographic (flat-earth approximation):**

  ```
  N = (lat - lat₀) × (π/180) × R_earth
  E = (lon - lon₀) × (π/180) × R_earth × cos(lat₀)
  D = -(alt - alt₀)
  ```

  Valid for distances < 100 km (≪ Earth radius).

- **Note:** Altitude in NED is negative-up (D = -h). The control layer uses positive-up altitude (h = -D) to match common PID convention.

### 4.3 Body Frame
- **Used by:** Vehicle dynamics, control (moments, rates)
- **Origin:** Vehicle center of mass
- **Convention:** x-forward, y-right, z-down (NED body)
- **Rotation:** ZYX Euler sequence (yaw ψ → pitch θ → roll φ) maps body to NED

### 4.4 Wind Frame
- **Used by:** Aerodynamics module (wing lift/drag)
- **Definition:** x-axis aligned with airspeed vector
- **Conversion:** body → wind via angle of attack α and sideslip angle β

---

## 5. Quaternion Convention

Two quaternion conventions coexist in the codebase:

| Location | Convention | Layout |
|----------|-----------|--------|
| `vehicle/dynamics/` | Hamilton | [w, x, y, z] — scalar first |
| `control/cascaded_control.py` | JPL / passive | [x, y, z, w] — scalar last |

**Conversion:** When passing quaternions from the dynamics module to the cascaded controller:
```python
q_hamilton = [w, x, y, z]
q_jpl      = np.roll(q_hamilton, -1)  # → [x, y, z, w]
```

This is documented in `control/cascaded_control.py` with a conversion note. The dataset scripts use Euler angles throughout, so this does not affect the dataset outputs.

---

## 6. Threat Model Architecture

The SAM threat model is the most novel component of the perception layer:

```
Three emitter types (parameterized):
  T1: Long-range surveillance radar
      Frequency: L-band (~1.3 GHz)
      Range: 150 km
      Pd model: Marcum Q-function (Swerling case I target)

  T2: Medium-range fire control radar
      Frequency: X-band (~9.5 GHz)
      Range: 40 km
      Pd model: Albersheim approximation

  T3: Short-range MANPADS (infrared seeker)
      Range: 6 km
      Pd model: IR signature-based engagement probability

Combined detection probability (at least one system detects):
  P_combined = 1 - (1 - P_T1)(1 - P_T2)(1 - P_T3)
```

Because the three emitters have overlapping coverage of the entire operating area, `P_combined ≈ 1.0` everywhere. This is not a modeling error — it reflects the contested airspace assumption. The planner therefore minimizes *total threat exposure time* (integral of P_detect × dt along the trajectory) rather than finding paths through low-threat zones.

---

## 7. NSGA-III Population and Reference Points

The multi-objective optimizer uses the following configuration:

| Parameter | Value |
|-----------|-------|
| Population size | 100 individuals |
| Generations | 200 |
| Reference points | 15 (Das & Dennis structured simplex) |
| Crossover type | SBX (simulated binary crossover), η=15 |
| Mutation type | Polynomial mutation, η=20 |
| Chromosome | Waypoint positions + altitudes (continuous) |
| Constraint handling | Penalty for terrain violation, speed bounds |

The reference point structure from Das & Dennis (1998) distributes candidate solutions uniformly across the 3-objective Pareto surface. With 15 reference points in 3D objective space, the final population covers the entire Pareto front.

**Knee-point selection** (for the "recommended" trajectory):
1. Normalize each objective to [0,1] using min/max of the Pareto front
2. Compute utopia point (minimum of each normalized objective)
3. Select the solution minimizing L₂ distance to utopia

---

## 8. Battery and Energy Model

Battery state-of-charge (SOC) is tracked using a simple Coulomb counting model in the vehicle layer and a momentum-theory power estimate in the control layer.

### Vehicle Layer (Coulomb Counting)

$$\text{SOC}(t) = \text{SOC}(0) - \frac{1}{Q_{\text{nom}}} \int_0^t I(t') \, dt'$$

where $I(t)$ is discharge current derived from instantaneous power $P(t) = F_z \cdot v_i$ (momentum theory) divided by bus voltage $V_{\text{bus}} = 400$ V. Peukert effect is applied for high-rate discharge: $Q_{\text{eff}} = Q_{\text{nom}} \cdot (I_{\text{nom}} / I)^{k-1}$ with $k = 1.05$.

### Control Layer (Simplified)

$$P_{\text{mean}} = T_{\text{mean}}^{3/2} / \sqrt{2 \rho A_{\text{disk}}}$$
$$E = P_{\text{mean}} \cdot T_{\text{mission}} / 3600 \quad [\text{Wh}]$$

This is a conservative estimate (uses mean thrust, ignores wing lift in cruise). The vehicle layer energy is more accurate.

---

## 9. Design Decisions

### Why separate Vehicle and Control layers?

The vehicle layer uses a high-fidelity offline physics model (BEMT rotors, detailed aerodynamics, electrochemical battery). The control layer uses a simplified 50 Hz plant to close the feedback loop efficiently across 2,000 missions. Running the full BEMT model at 50 Hz for 2,000 missions would require ~48 hours; the simplified plant runs in 12 minutes.

The two layers are complementary: the vehicle layer provides high-fidelity physics features; the control layer provides realistic closed-loop tracking metrics.

### Why fixed cruise altitude?

Setting cruise altitude to 200 m AGL for all missions allows the aerodynamic model to be evaluated at a single operating point (avoiding altitude-dependent drag/lift coefficient changes). A multi-altitude scenario is left for future work.

### Why 2,000 records per layer?

The planning dataset bottleneck is NSGA-III runtime (~45 s per trajectory at 200 generations). At 2,000 records, the planning stage runs in ~25 hours (parallelized). Larger datasets are possible with fewer NSGA-III generations or more parallel workers.

### Why store in Parquet?

Parquet provides:
- ~3–5× compression vs CSV with no information loss
- Column-pruning (only read columns needed for a given ML task)
- Built-in schema / data types (avoids CSV parsing ambiguity)
- Compatible with pandas, PyArrow, Polars, DuckDB, Spark
