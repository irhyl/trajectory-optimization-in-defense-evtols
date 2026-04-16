# Dataset Guide — Schemas, Column Definitions, and Usage

This document provides complete schema definitions for all four datasets, guidance on how to load them, and recommended ML task configurations.

---

## 1. Quick Load Reference

```python
import pandas as pd

# --- Delhi (primary, 10,000-mission benchmark region) ---
plan = pd.read_parquet("outputs/delhi/planning_dataset/planning_dataset_10k.parquet")
veh  = pd.read_parquet("outputs/delhi/vehicle/vehicle_dataset.parquet")
ctrl = pd.read_parquet("outputs/delhi/control/control_dataset.parquet")

# --- Other regions (2,000 missions each) ---
for region in ["mumbai", "bangalore", "arunachal", "odisha", "ladakh"]:
    plan_r = pd.read_parquet(f"outputs/{region}/planning/planning_dataset.parquet")
    veh_r  = pd.read_parquet(f"outputs/{region}/vehicle/vehicle_dataset.parquet")
    ctrl_r = pd.read_parquet(f"outputs/{region}/control/control_dataset.parquet")

# --- 80/10/10 stratified splits (by risk_label) per region ---
train = pd.read_parquet("outputs/delhi/splits/train.parquet")
val   = pd.read_parquet("outputs/delhi/splits/val.parquet")
test  = pd.read_parquet("outputs/delhi/splits/test.parquet")
```

All datasets are stored as Apache Parquet (primary format). The `outputs/` directory is
organized as:

```
outputs/
├── delhi/
│   ├── planning_dataset/   planning_dataset_10k.parquet  (10K rows × 25 cols)
│   ├── vehicle/            vehicle_dataset.parquet        (10K rows × 95 cols)
│   ├── control/            control_dataset.parquet        (10K rows × 76 cols)
│   ├── perception/         perception data for NCR region
│   ├── splits/             train.parquet, val.parquet, test.parquet
│   └── ml/                 baseline model results
├── mumbai/        (same structure, 2K rows each)
├── bangalore/     (same structure, 2K rows each)
├── arunachal/     (same structure, 2K rows each)
├── odisha/        (same structure, 2K rows each; obstacle_cost=1.0 due to API timeout)
├── ladakh/        (same structure, 2K rows each)
└── planning_dataset/  planning_dataset_10k.parquet  (legacy fallback copy)
```

---

## 2. Perception Dataset Schema

**File:** `outputs/perception_dataset/exports/perception_full_dataset.parquet`
**Dimensions:** 1,057,714 rows × 28 columns
**Coverage:** 28.7°–29.0°N, 76.9°–77.4°E, altitudes {50, 150, 300} m AGL

### Grid Structure

The grid has shape approximately (150 lat × 250 lon × 3 alt) = 112,500 cells per layer, for 337,500 unique (lat, lon, alt) combinations, replicated across the dataset with slight variations from wind forecast timing.

```python
# Reconstruct 3D grid
lat_vals = perc['lat'].unique()   # ~150 values
lon_vals = perc['lon'].unique()   # ~250 values
alt_vals = perc['alt_m'].unique() # [50, 150, 300]
```

### Column Definitions

#### Spatial Coordinates
| Column | Type | Unit | Description |
|--------|------|------|-------------|
| `lat` | float64 | ° | WGS-84 latitude |
| `lon` | float64 | ° | WGS-84 longitude |
| `alt_m` | float64 | m AGL | Altitude above ground level |

#### Terrain Features
| Column | Type | Unit | Description | Source |
|--------|------|------|-------------|--------|
| `elev_m` | float64 | m MSL | Terrain elevation | NASA SRTM30 via Open-Elevation |
| `slope_deg` | float64 | ° | Terrain slope angle | Computed from SRTM gradient |
| `roughness_m` | float64 | m | Terrain roughness index (std of elevation in 3×3 window) | SRTM |
| `surface_type` | int8 | — | Surface category: 0=flat, 1=hilly, 2=urban, 3=water | OSM + DEM |
| `terrain_clearance_m` | float64 | m | `alt_m - elev_m` (distance to ground) | Computed |

#### Wind Features
| Column | Type | Unit | Description | Source |
|--------|------|------|-------------|--------|
| `wind_u_mps` | float64 | m/s | Eastward wind component (positive = from west) | Open-Meteo GFS |
| `wind_v_mps` | float64 | m/s | Northward wind component (positive = from south) | Open-Meteo GFS |
| `wind_w_mps` | float64 | m/s | Vertical wind component (positive = upward) | Open-Meteo GFS |
| `wind_speed_mps` | float64 | m/s | Wind speed magnitude $\sqrt{u^2+v^2+w^2}$ | Computed |
| `wind_dir_deg` | float64 | ° | Wind direction (meteorological convention) | Computed |
| `turbulence_intensity` | float64 | — | Normalized turbulence [0,1]; $\sigma_v / V_{\text{mean}}$ | Computed |

#### Obstacle Features
| Column | Type | Unit | Description | Source |
|--------|------|------|-------------|--------|
| `nearest_obstacle_dist_m` | float64 | m | Horizontal range to nearest OSM obstacle | OSM Overpass |
| `nearest_obstacle_height_m` | float64 | m | Height of nearest obstacle | OSM Overpass |
| `obstacle_type` | int8 | — | Category: 0=none, 1=building, 2=tower, 3=tree, 4=powerline | OSM |

#### Threat Features
| Column | Type | Unit | Description |
|--------|------|------|-------------|
| `T1_detect_prob` | float64 | — | Detection probability: long-range surveillance radar [0,1] |
| `T2_detect_prob` | float64 | — | Detection probability: medium-range fire control [0,1] |
| `T3_detect_prob` | float64 | — | Detection probability: MANPADS IR seeker [0,1] |
| `max_threat_prob` | float64 | — | `max(T1, T2, T3)` |
| `combined_threat_prob` | float64 | — | `1 - (1-T1)(1-T2)(1-T3)` — prob of detection by ANY system |

#### Cost Features
| Column | Type | Range | Description |
|--------|------|-------|-------------|
| `terrain_cost` | float64 | [0,1] | Penalty for terrain proximity and slope |
| `wind_cost` | float64 | [0,1] | Penalty for headwind and turbulence |
| `obstacle_cost` | float64 | [0,1] | Penalty for obstacle proximity |
| `threat_cost` | float64 | [0,1] | Normalized `combined_threat_prob` |
| `energy_cost` | float64 | [0,1] | Estimated power required at this point (normalized) |
| `fused_cost` | float64 | [0,1] | Weighted composite: `w_t·terrain + w_w·wind + w_o·obstacle + w_th·threat + w_e·energy` |

**Fusion weights (default):** terrain=0.20, wind=0.15, obstacle=0.25, threat=0.25, energy=0.15

---

## 3. Planning Dataset Schema

**File:** `outputs/planning_dataset/test_final.parquet`
**Dimensions:** 2,000 rows × 25 columns

Each row represents one Pareto-optimal trajectory (knee-point selected from NSGA-III front).

### Column Definitions

#### Start/Goal Coordinates
| Column | Type | Unit | Description |
|--------|------|------|-------------|
| `start_lat`, `start_lon`, `start_alt_m` | float64 | °, °, m | Mission start position |
| `goal_lat`, `goal_lon`, `goal_alt_m` | float64 | °, °, m | Mission goal position |

#### Waypoints
| Column | Type | Description |
|--------|------|-------------|
| `waypoint_lats` | str | `\|`-separated latitude values of intermediate waypoints |
| `waypoint_lons` | str | `\|`-separated longitude values |
| `waypoint_alts` | str | `\|`-separated altitude values (m AGL) |
| `n_waypoints` | int32 | Total waypoint count including start and goal |

```python
# Parse waypoints for a single row
import numpy as np
row = plan.iloc[0]
lats = np.array([float(x) for x in row['waypoint_lats'].split('|')])
lons = np.array([float(x) for x in row['waypoint_lons'].split('|')])
alts = np.array([float(x) for x in row['waypoint_alts'].split('|')])
```

#### Path Geometry
| Column | Type | Unit | Description |
|--------|------|------|-------------|
| `path_length_m` | float64 | m | Total arc length (sum of segment distances) |
| `planning_time_s` | float64 | s | RRT* wall-clock budget consumed |

#### Objective Values (NSGA-III)
| Column | Type | Unit | Description |
|--------|------|------|-------------|
| `time_cost_s` | float64 | s | Estimated flight time along trajectory |
| `energy_cost_wh` | float64 | Wh | Estimated energy consumption |
| `threat_cost` | float64 | — | Integrated detection probability [0,1] |

#### Cost Field Averages Along Trajectory
| Column | Type | Unit | Description |
|--------|------|------|-------------|
| `terrain_cost_mean` | float64 | — | Mean terrain cost of traversed cells |
| `wind_cost_mean` | float64 | — | Mean wind cost |
| `obstacle_cost_mean` | float64 | — | Mean obstacle cost |
| `fused_cost_mean` | float64 | — | Mean fused traversal cost |

#### Labels
| Column | Type | Range | Description |
|--------|------|-------|-------------|
| `feasible` | int8 | {0,1} | 1 if all constraints satisfied (terrain clearance, speed bounds) |
| `altitude_clearance_ok` | int8 | {0,1} | 1 if terrain clearance ≥ 30 m at all waypoints |
| `speed_ok` | int8 | {0,1} | 1 if planned speed ∈ [20, 100] m/s |
| `risk_label` | int8 | {0,1} | 0=low risk, 1=high risk (combined threat + path geometry) |

---

## 4. Vehicle Dataset Schema

**File:** `outputs/vehicle/vehicle_dataset.parquet`
**Dimensions:** 2,000 rows × 95 columns

Physics-based simulation of each trajectory through the 6-DoF vehicle model.

### Column Group Summary

| Group | Columns | Key Metrics |
|-------|---------|-------------|
| Mission profile | 10 | `mission_time_s`, `cruise_speed_ms`, `n_waypoints` |
| Rotor/propulsion (BEMT) | 12 | `T_hover_N`, `P_hover_kW`, `FM_hover`, `rpm_cruise` |
| Aerodynamics (cruise) | 10 | `CL_cruise`, `CD_cruise`, `LD_ratio`, `drag_parasitic_N` |
| Energy & battery | 15 | `energy_total_wh`, `soc_initial`, `soc_final`, `pack_capacity_wh` |
| Thermal | 8 | `motor_temp_rise_K`, `winding_temp_C`, `thermal_derating_frac` |
| Acoustic signature | 6 | `spl_hover_a_dB`, `spl_cruise_a_dB`, `spl_peak_dB` |
| Infrared signature | 8 | `ir_radiance_hover_Wsr`, `ir_radiance_cruise_Wsr` |
| RCS signature | 10 | `rcs_cruise_x_dBsm`, `rcs_top_x_dBsm`, `rcs_cruise_z_dBsm` |
| Phase timing | 7 | `hover_time_s`, `cruise_time_s`, `transition_time_s` |
| Labels | 5 | `risk_label`, `feasible`, `soc_ok`, `thermal_ok`, `noise_ok` |

### Selected Key Columns

```python
# Most useful columns for ML tasks
key_cols = [
    # Mission profile
    'path_length_m', 'mission_time_s', 'cruise_speed_ms', 
    'cruise_altitude_m', 'n_waypoints',
    # Energy  
    'energy_total_wh', 'soc_initial', 'soc_final', 'pack_capacity_wh',
    # Propulsion
    'T_hover_N', 'P_hover_kW', 'FM_hover',
    # Cruise aerodynamics
    'LD_ratio', 'CD_cruise',
    # Signatures
    'spl_hover_a_dB', 'rcs_cruise_x_dBsm', 'max_combined_threat',
    # Labels
    'risk_label', 'feasible'
]
```

---

## 5. Control Dataset Schema

**File:** `outputs/control/control_dataset.parquet`
**Dimensions:** 2,000 rows × 76 columns

See [control_layer.md §8](control_layer.md#8-dataset-schema) for the complete column-by-column reference.

### Quick Column Categories

```python
# Tracking quality (recommended for ML)
tracking_cols = [
    'alt_error_mean_m', 'alt_error_rms_m',       # altitude tracking
    'att_error_mean_rad', 'att_error_rms_rad',     # attitude tracking
    'vel_error_mean_ms', 'vel_error_rms_ms',       # velocity tracking
    'itae_alt', 'itae_att',                        # ITAE quality metrics
]

# Control effort
effort_cols = [
    'thrust_cmd_mean_N', 'thrust_cmd_std_N',
    'moment_x_std_Nm', 'moment_y_std_Nm', 'moment_z_std_Nm',
    'n_saturations', 'pwm_utilisation_pct',
]

# Mission outcome
outcome_cols = [
    'mission_abort', 'n_stall_events', 'n_wp_reached',
    'soc_final', 'energy_consumed_wh',
    'settling_time_mean_s',   # ⚠ usually 20s max (see limitations)
]

# ⚠ DO NOT USE for tracking quality:
# 'pos_error_mean_m' — artifact of cumulative reference (see limitations)
```

---

## 6. Recommended ML Task Configurations

### Task 1: Risk Classification (Binary)

```python
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

# Join planning + vehicle features
df = pd.merge(plan, veh[['risk_label', 'energy_total_wh', 'soc_final', 
                          'spl_hover_a_dB', 'rcs_cruise_x_dBsm']], 
              left_index=True, right_index=True)

X = df[['path_length_m', 'energy_cost_wh', 'threat_cost', 
        'fused_cost_mean', 'n_waypoints', 'time_cost_s']].values
y = df['risk_label_x'].values  # from planning dataset

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

clf = GradientBoostingClassifier(n_estimators=100, random_state=42)
cv_scores = cross_val_score(clf, X_scaled, y, cv=StratifiedKFold(5), scoring='roc_auc')
print(f"AUC-ROC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
```

### Task 2: Energy Forecasting (Regression)

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

X = plan[['path_length_m', 'time_cost_s', 'n_waypoints', 
          'wind_cost_mean', 'terrain_cost_mean']].values
y = ctrl['energy_consumed_wh'].values  # control-layer energy

# 80/20 split (stratified by risk label)
from sklearn.model_selection import train_test_split
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

reg = RandomForestRegressor(n_estimators=200, random_state=42)
reg.fit(X_tr, y_tr)
print(f"R² = {r2_score(y_te, reg.predict(X_te)):.3f}")
```

### Task 3: Abort Prediction

```python
# Predict mission abort before flight using planning + vehicle features
X = pd.concat([
    plan[['path_length_m', 'energy_cost_wh', 'time_cost_s', 'fused_cost_mean']],
    veh[['cruise_speed_ms', 'soc_initial', 'T_hover_N', 'P_hover_kW']]
], axis=1).values
y = ctrl['mission_abort'].values   # 0 for all 2000 in current dataset

# NOTE: All missions succeed, so this task needs a harder scenario or
# synthetic abort examples to be meaningful.
```

### Task 4: Perception Cost Prediction (Spatial ML)

```python
# Predict fused_cost from raw terrain/wind/obstacle features
X = perc[['elev_m', 'slope_deg', 'roughness_m', 
           'wind_speed_mps', 'turbulence_intensity',
           'nearest_obstacle_dist_m', 'alt_m']].values
y = perc['fused_cost'].values

# Train/test split by geographic region (not random) to test generalization
lat_mid = perc['lat'].median()
train_mask = perc['lat'] < lat_mid
# ...
```

---

## 7. Dataset Statistics Summary

### Perception Dataset

| Feature | Mean | Std | Min | Max |
|---------|------|-----|-----|-----|
| `elev_m` | 218.0 | 16.4 | 190.4 | 305.0 |
| `wind_speed_mps` | 6.21 | 2.41 | 1.80 | 15.10 |
| `turbulence_intensity` | 0.18 | 0.06 | 0.05 | 0.45 |
| `nearest_obstacle_dist_m` | 412 | 280 | 2.1 | 2,840 |
| `combined_threat_prob` | 1.000 | 0.000 | 1.000 | 1.000 |
| `fused_cost` | 0.526 | 0.049 | 0.447 | 0.735 |

### Planning Dataset

| Feature | Mean | Std | Min | Max |
|---------|------|-----|-----|-----|
| `path_length_m` | 12,847 | 6,892 | 222 | 39,074 |
| `n_waypoints` | 5.2 | 1.8 | 2 | 12 |
| `time_cost_s` | 264 | 83 | 120 | 612 |
| `energy_cost_wh` | 1,711 | 739 | 267 | 4,689 |
| `fused_cost_mean` | 0.524 | 0.038 | 0.449 | 0.681 |
| Risk label (0/1) | — | — | 1,479 | 521 |

### Vehicle Dataset

| Feature | Mean | Std | Min | Max |
|---------|------|-----|-----|-----|
| `cruise_speed_ms` | 71.3 | 10.8 | 48.5 | 94.4 |
| `cruise_altitude_m` | 200 | 0 | 200 | 200 |
| `soc_final` | ~0.82 | ~0.05 | ~0.65 | ~0.97 |
| `spl_hover_a_dB` | 92.84 | 0.00 | 92.84 | 92.84 |
| `rcs_cruise_x_dBsm` | −8.84 | 0.46 | −10.2 | −7.5 |

### Control Dataset

| Feature | Mean | Std | Min | Max |
|---------|------|-----|-----|-----|
| `alt_error_mean_m` | 10.37 | 5.45 | 0.12 | 24.54 |
| `att_error_mean_rad` | 0.0013 | 0.0004 | 0.0004 | 0.003 |
| `thrust_cmd_mean_N` | 2,613 | 20 | 2,539 | 2,702 |
| `energy_consumed_wh` | 5,903 | 1,906 | 2,211 | 14,870 |
| `soc_final` | 0.752 | 0.048 | 0.628 | 0.945 |
| `mission_abort` | 0.000 | — | 0 | 0 |

---

## 8. Data Provenance

| Dataset | Generated By | Runtime | Cache |
|---------|-------------|---------|-------|
| Perception | `scripts/perception/dataset.py` | ~30 min (API calls) | `outputs/perception_dataset/cache/` |
| Planning | `scripts/planning/dataset.py` | ~15 min | — |
| Vehicle | `scripts/vehicle/dataset.py` | ~10 min | — |
| Control | `scripts/control/dataset.py` | ~12 min | — |

**Reproducibility:** All scripts are deterministic given the same API responses. The perception layer caches API responses in `outputs/perception_dataset/cache/` to allow re-runs without re-querying external services. Seeds for NSGA-III random number generators are fixed (see `src/evtol/planning/optimization/nsga3.py`).

---

## 9. License

All datasets generated by this framework are released under **CC BY 4.0** (Creative Commons Attribution 4.0 International).

Input data licenses:
- NASA SRTM terrain: Public domain (U.S. Government)
- Open-Meteo wind: CC BY 4.0
- OpenStreetMap obstacles: Open Database License (ODbL)

Attribution requirement: If you use this dataset, cite the project as shown in [README.md §13](../README.md#13-citation).
