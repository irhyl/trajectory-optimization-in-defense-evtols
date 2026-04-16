# Trajectory Optimization in Defense eVTOLs

**First open multi-objective dataset for defense eVTOL trajectory planning in contested airspace.**

Combines real geospatial perception data (NASA SRTM terrain, Open-Meteo wind, OpenStreetMap obstacles), analytical SAM threat modeling (Mahafza 2005 radar range equation), RRT*-planned trajectories, 6-DoF vehicle physics simulation, and closed-loop cascaded PID autopilot data across 10,000 missions.

## Dataset Description

### Four Linked Layers

| Layer | Rows | Features | File |
|-------|------|----------|------|
| Perception | 1,057,714 | 28 | `perception_dataset/perception_dataset.parquet` |
| Planning | 10,000 | 25 | `planning_dataset/planning_dataset_10k.parquet` |
| Vehicle | 10,000 | 95 | `vehicle/vehicle_dataset.parquet` |
| Control | 10,000 | 78 | `control/control_dataset.parquet` |

Layers are row-aligned by mission index: row `i` in Planning, Vehicle, and Control all describe the same mission.

### Operating Areas (6 Regions)

| Region | Bounds | Terrain | Threat Scenario |
|--------|--------|---------|-----------------|
| Delhi NCR | 28.40–28.90°N, 76.90–77.50°E | Indo-Gangetic plain, 190–306 m | S-300V / SA-11 / SA-22 (continental) |
| Mumbai MMR | 18.85–19.35°N, 72.75–73.35°E | Coastal + W. Ghats, -3–776 m | S-125 / Barak-8 / Akash (naval) |
| Bengaluru | 12.80–13.30°N, 77.40–77.95°E | Deccan plateau, 706–953 m | Akash Mk2 / QRSAM / SA-6 |
| Arunachal Pradesh | 27.50–27.95°N, 93.80–94.40°E | E. Himalayas, 500–4000 m | HQ-9B / HQ-16 / HQ-17A (LAC) |
| Odisha | 19.60–20.10°N, 85.60–86.20°E | Bay of Bengal coast, 0–100 m | Barak-8 / SA-3 / Akash (coastal) |
| Ladakh | 34.00–34.50°N, 77.30–77.90°E | High-altitude desert, 3400–5200 m | HQ-9B / SA-15 / ZU-23 (Karakoram) |

All regions: 14 altitude bands (50–1500 m AGL), 0.002° grid (~222 m spacing).

### ML Tasks

| Task | Target | Type | Best Baseline |
|------|--------|------|---------------|
| T1 | `risk_label` | Binary classification | AUC 0.9996 (MLP) |
| T2 | `energy_consumed_wh` | Regression | R² 0.9972 (GBM) |
| T3 | `mission_abort` | Binary classification | Trivial (all 0) |
| T4 | `alt_error_mean_m` | Regression | — |
| T5 | `threat_cost` | Regression | — |
| T6 | `soc_final` | Regression | — |

## Quickstart

```python
from datasets import load_dataset

# Load the planning layer
ds = load_dataset("ramakrishnan2026/evtol-trajectory", "planning")
df = ds["train"].to_pandas()

# Or load all four layers
for split in ["perception", "planning", "vehicle", "control"]:
    ds = load_dataset("ramakrishnan2026/evtol-trajectory", split)
```

Alternatively, load directly with pandas:

```python
import pandas as pd

plan = pd.read_parquet("outputs/planning_dataset/planning_dataset_10k.parquet")
ctrl = pd.read_parquet("outputs/control/control_dataset.parquet")

# T1: Risk classification
X = plan[["path_length_m", "n_waypoints", "threat_cost",
          "terrain_cost_mean", "wind_cost_mean", "obstacle_cost_mean"]]
y = plan["risk_label"]

# T2: Energy regression (planning-layer features → control-layer energy)
merged = pd.concat([plan, ctrl.drop(columns=plan.columns, errors="ignore")], axis=1)
X_e = merged[["path_length_m", "n_waypoints", "time_cost_s",
              "cruise_speed_ref_ms", "cruise_altitude_ref_m", "risk_label"]]
y_e = merged["energy_consumed_wh"]
```

## Dataset Structure

### Perception Layer (1,057,714 rows)

Grid cells at 0.002° × 0.002° resolution across the operating area.

| Column | Description |
|--------|-------------|
| `lat`, `lon`, `alt_m` | Grid cell coordinates |
| `elevation_m` | NASA SRTM terrain elevation |
| `slope_deg` | Terrain slope from finite differences |
| `wind_speed_ms` | Wind speed magnitude (Open-Meteo GFS) |
| `wind_u_ms`, `wind_v_ms`, `wind_w_ms` | Wind vector components |
| `obstacle_dist_m` | Distance to nearest OSM obstacle |
| `combined_threat_prob` | SAM detection probability (Marcum Q-function) |
| `fused_cost` | Weighted combination of all cost components |
| `risk_label` | Binary: fused_cost ≥ 0.55 |

### Planning Layer (10,000 rows)

RRT*-planned trajectories from random start/goal pairs.

| Column | Description |
|--------|-------------|
| `path_length_m` | Total trajectory arc length |
| `n_waypoints` | Number of waypoints |
| `time_cost_s` | Estimated flight time |
| `energy_cost_wh` | NSGA-III estimated energy |
| `threat_cost` | Integrated threat exposure [0–1] |
| `risk_label` | Binary risk classification |

### Vehicle Layer (10,000 rows)

6-DoF physics simulation (BEMT rotor, aerodynamics, battery).

| Column | Description |
|--------|-------------|
| `cruise_speed_ms` | Cruise airspeed |
| `cruise_altitude_m` | Cruise altitude AGL |
| `energy_consumed_wh` | Total energy consumed |
| `soc_final` | Final battery state of charge |
| `spl_hover_total_dB` | Hover acoustic signature |
| `rcs_cruise_x_dBsm` | Radar cross-section in cruise |

### Control Layer (10,000 rows)

Closed-loop cascaded PID simulation at 50 Hz.

| Column | Description |
|--------|-------------|
| `pos_error_mean_m` | Mean cruise position tracking error |
| `alt_error_mean_m` | Mean altitude tracking error |
| `att_error_mean_rad` | Mean attitude error |
| `energy_consumed_wh` | Simulated energy consumption |
| `alt_settling_mean_s` | Altitude settling time |
| `vel_settling_mean_s` | Velocity settling time |
| `n_saturations` | Actuator saturation count |

## Data Splits

Stratified 80/10/10 train/val/test splits by `risk_label` (seed=42) are provided as NumPy index arrays in `outputs/splits/`.

```python
import numpy as np
import pandas as pd

plan = pd.read_parquet("outputs/planning_dataset/planning_dataset_10k.parquet")
train_idx = np.load("outputs/splits/planning_train_idx.npy")
val_idx   = np.load("outputs/splits/planning_val_idx.npy")
test_idx  = np.load("outputs/splits/planning_test_idx.npy")

train, val, test = plan.iloc[train_idx], plan.iloc[val_idx], plan.iloc[test_idx]
```

## Known Limitations

1. **Single geography**: Delhi NCR only. Mumbai theatre runner scripts provided (`scripts/perception/run_mumbai.py`, `scripts/planning/run_mumbai.py`) but not yet executed.
2. **Threat saturation**: Physics-based `combined_threat_prob = 1.0` everywhere (fully contested scenario). Planning layer uses a distance-decay gradient for spatial variation.
3. **Zero acoustic variance**: All missions use the same vehicle mass → identical hover SPL.
4. **No sensor noise in control sim**: Single-rate 50 Hz plant cannot model GPS/IMU loop separation without derivative kick artifacts.

See `doc/research_limitations.md` for full technical assessment.

## Baseline Results (5-Fold Cross-Validation)

| Task | Model | Metric | Score |
|------|-------|--------|-------|
| T1: Risk Classification | MLP | AUC-ROC | 0.9996 |
| T1: Risk Classification | Logistic Regression | AUC-ROC | 0.9995 |
| T1: Risk Classification | Gradient Boosting | AUC-ROC | 0.9929 |
| T2: Energy Regression | Gradient Boosting | R² | 0.9972 |
| T2: Energy Regression | Ridge | R² | 0.9896 |
| T2: Energy Regression | MLP | R² | 0.9900 |

Full results in `outputs/ml/baseline_results.csv`.

## License

- **Code**: MIT License
- **Generated dataset** (`outputs/`): CC-BY 4.0
- **NASA SRTM input**: Public domain
- **Open-Meteo input**: CC-BY 4.0
- **OpenStreetMap input**: Open Database License (ODbL)
