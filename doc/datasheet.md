# Datasheet for Dataset: Trajectory Optimization in Defense eVTOLs

Following the "Datasheets for Datasets" framework (Gebru et al., 2021).
Gebru, T. et al. (2021). Datasheets for datasets. Communications of the ACM, 64(12), 86-92.

---

## 1. Motivation

**1.1 For what purpose was the dataset created?**
To enable ML research on multi-objective trajectory optimization for defense eVTOL aircraft operating in contested airspace. No comparable open dataset combining real geospatial perception data, analytical SAM threat modeling, and end-to-end closed-loop autopilot simulation existed for this domain.

**1.2 Who funded creation?**
Created using only free public APIs and open-source software tools.

**1.3 Other comments?**
The dataset is fully reproducible from the provided pipeline scripts, which access only public APIs (NASA SRTM, Open-Meteo, OpenStreetMap Overpass).

---

## 2. Composition

**2.1 What do the instances represent?**

Four linked layers, collected across six Indian geographic regions:

- **Perception**: Grid cells at terrain-resolved altitude bands, each with terrain, wind, obstacle, and SAM threat features derived from real APIs.
- **Planning**: Mission trajectories with RRT*-planned waypoints optimized via NSGA-III across energy/time/threat objectives.
- **Vehicle**: Per-mission 6-DoF physics simulation records (BEMT rotor, aerodynamics, battery).
- **Control**: Per-mission closed-loop PID simulation records at 50 Hz.

**2.2 How many instances?**

| Region | Planning rows | Vehicle rows | Control rows | Notes |
|--------|--------------|-------------|-------------|-------|
| Delhi (NCR) | 10,000 | 10,000 | 10,000 | Primary benchmark region |
| Mumbai | 2,000 | 2,000 | 2,000 | Coastal urban |
| Bangalore | 2,000 | 2,000 | 2,000 | Elevated plateau |
| Arunachal Pradesh | 2,000 | 2,000 | 2,000 | High-altitude mountainous |
| Odisha | 2,000 | 2,000 | 2,000 | Coastal flat |
| Ladakh | 2,000 | 2,000 | 2,000 | High-altitude arid |
| **Total** | **22,000** | **22,000** | **22,000** | |

Each region has per-region perception data (terrain/wind/obstacles from real APIs).
Delhi planning dataset: 25 features. Vehicle: 95 features. Control: 76 features.

**2.3 Complete coverage or sample?**
Perception layer is complete coverage at the defined grid resolution (0.002 deg resolution per region). Planning/vehicle/control are a sampled set of missions from the space of all possible start/goal pairs within each region's bounding box.

**2.4 What data does each instance consist of?**
Rows in Parquet files, organized under `outputs/<region>/`. Full schema documentation in `doc/data_guide.md`.

**2.5 Labels and targets?**
Six ML tasks defined (see `doc/introduction.md`):
- T1: `risk_label` — binary classification (0=low-risk, 1=high-risk)
- T2: `energy_consumed_wh` — regression (closed-loop energy in Wh)
- T3: `mission_abort` — binary classification (trivial: all 0 in current data)
- T4: `alt_error_mean_m` — regression (altitude tracking error in m)
- T5: `threat_cost` — regression (trajectory threat exposure probability)
- T6: `soc_final` — regression (final battery state of charge)

**2.6 Missing data?**
Odisha region: `obstacle_cost` column is all-maximum (1.0) due to Overpass API gateway timeouts during data collection. All other fields are present. This is documented in `outputs/odisha/` metadata.

**2.7 Relationships between instances?**
Within each region, the four layers align by row index (row i corresponds to the same mission across planning/vehicle/control layers). Shared columns (`start_lat/lon`, `goal_lat/lon`, `path_length_m`) serve as join keys. Cross-region comparisons use the `region` column added to each dataset.

**2.8 Recommended data splits?**
Yes. Stratified 80/10/10 train/val/test splits by `risk_label` are in `outputs/splits/` as numpy index arrays. Seed: 42. See `scripts/ml/baseline_experiments.py`.

**2.9 Errors, noise, or redundancies?**
Documented in detail in `doc/research_limitations.md`:
- **Threat saturation**: `combined_threat_prob = 1.0` everywhere due to SAM emitter placement. Planning layer uses a distance-decay gradient instead (std=0.12, range 0.45-1.0).
- **Zero acoustic variance**: All hover SPL values identical (fixed vehicle mass). Zero discriminative power.
- **Simplified control plant**: Linear drag only, no wing lift in cruise, no rotor dynamics.
- **No mission aborts**: `mission_abort=0` for all 2,000 records. T3 is trivial on this dataset.

**2.10 Self-contained?**
Yes. Files in `outputs/` have no runtime external dependencies.

**2.11 Confidential data?**
No. All input sources are public domain or open license. See `doc/research_limitations.md Section 5.1` for full dual-use assessment.

**2.12 Offensive content?**
No. Numerical simulation data only.

---

## 3. Collection Process

**3.1 How was data acquired?**

| Source | Data | Mechanism |
|--------|------|-----------|
| NASA SRTM v3 | Terrain (elevation, slope) | Open-Elevation REST API |
| Open-Meteo GFS | Wind (u, v, w components) | REST API, t=0 snapshot |
| OpenStreetMap | Obstacles | Overpass API polygon queries |
| Analytical model | SAM threat | Mahafza (2005) radar range equation |
| RRT* algorithm | Planning trajectories | Computed from perception cost field |
| 6-DoF simulation | Vehicle physics | Computed from planning trajectories |
| PID autopilot | Control metrics | Computed from vehicle parameters at 50 Hz |

**3.2 Preprocessing mechanisms?**
- Terrain: bilinear interpolation from 2 km coarse grid to 222 m fine grid
- Wind: inverse distance weighting (IDW) from 5 anchor points (center + 4 corners)
- Obstacles: Overpass API polygon queries, filtered by building/road OSM tags
- Threat: vectorized Albersheim approximation / Marcum Q-function per grid cell

**3.3 Sampling strategy?**
Random start/goal pairs sampled uniformly from the perception grid, filtered for 0.5-20 km separation distance. No geographic stratification.

**3.4 Who was involved?**
Fully automated via `scripts/`. No human annotators.

**3.5 Timeframe?**
Single API session. Wind is a 6-hour forecast snapshot. Static dataset -- no temporal variation.

**3.6 Ethical review?**
Threat model uses only publicly available textbook parameters. See `doc/research_limitations.md Section 5` for full dual-use assessment.

**3.7 Data from individuals?**
Not applicable. No personal data of any kind.

---

## 4. Preprocessing / Cleaning / Labeling

**4.1 Was preprocessing done?**
- Terrain slope from finite differences of SRTM elevation
- Wind speed from vector magnitude: sqrt(u^2 + v^2 + w^2)
- Obstacle cost from proximity to nearest OSM feature (Gaussian decay)
- All five cost components normalized to [0, 1] before weighted fusion
- `risk_label` = int(fused_cost_mean >= 0.55)
- Distance-decay threat gradient computed in planning layer to replace saturated physics-based values

**4.2 Was raw data saved?**
Yes. Perception CSV contains both raw API values and derived cost fields. Cache directory stores intermediate computed arrays.

**4.3 Is preprocessing software available?**
Yes. All code in `scripts/` and `src/evtol/` under MIT License. Fully reproducible from scratch with public API access.

---

## 5. Uses

**5.1 Prior uses?**
Baseline ML experiments on T1 (risk classification) and T2 (energy regression) in `outputs/ml/baseline_results.csv` using 5-fold cross-validation with LR, GBM, MLP.

**5.2 Repository for related papers?**
Not yet. Initial release.

**5.3 Other potential tasks?**
- Safe reinforcement learning for UAV navigation in contested airspace
- Surrogate modeling of high-fidelity flight simulators
- Multi-task learning across layered autonomy objectives
- Anomaly detection in PID autopilot performance
- Uncertainty quantification for trajectory planning under threat

**5.4 Composition issues affecting future uses?**
- Operating area is fixed (Delhi NCR, India). Geographic generalization untested.
- Threat model assumes fully contested scenario; partial coverage needs different SAM configuration.
- Single vehicle platform and mass; no fleet-level variation.
- 2,000 records sufficient for classical ML, insufficient for large neural networks or offline RL.

**5.5 Tasks the dataset should NOT be used for?**
- Real-world operational mission planning (simplified physics models not suitable).
- Training systems intended to improve adversarial threat detection capabilities.

---

## 6. Distribution

**6.1 Distribution to third parties?**
Yes. Open-source release. Hugging Face Datasets hosting planned as future step.

**6.2 When?**
Upon NeurIPS Datasets and Benchmarks acceptance.

**6.3 License?**
- Code: MIT License
- Generated dataset (`outputs/`): CC-BY 4.0
- NASA SRTM input: public domain
- Open-Meteo input: CC-BY 4.0
- OpenStreetMap input: Open Database License (ODbL)

**6.4 Third-party IP restrictions?**
None. All external data sources permit open research use.

**6.5 Export controls?**
None. Threat model uses only publicly available textbook parameters (Mahafza 2005). No classified information included.

---

## 7. Maintenance

**7.1 Who maintains the dataset?**
Aditi Ramakrishnan, via GitHub issues.

**7.2 Erratum?**
Not at initial release.

**7.3 Will the dataset be updated?**
Planned updates:
- Second geographic region (generalization benchmark)
- 10,000-50,000 planning records (current: 2,000)
- HITL simulation validation

**7.4 Will older versions remain available?**
Yes, via GitHub version tags.

**7.5 Mechanism for contributions?**
MIT License, pull requests welcome. Adding a new geographic region requires only updating `LAT_MIN/MAX`, `LON_MIN/MAX` in `scripts/perception/dataset.py` and re-running the pipeline.

---

## Citation

```bibtex
@misc{ramakrishnan2026evtol,
  author = {Ramakrishnan, Aditi},
  title  = {Trajectory Optimization in Defense eVTOLs:
             An Open Multi-Objective Dataset and Benchmark},
  year   = {2026},
  note   = {NeurIPS Datasets and Benchmarks submission}
}
```

*Follows the Gebru et al. (2021) datasheet template. Version 1.0, 2026-04-12.*
