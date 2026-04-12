# Research Limitations and NeurIPS Submission Notes

This document provides an honest technical assessment of the current state of the research for a NeurIPS Datasets & Benchmarks submission. It covers known data quality issues, modeling approximations, and what would be needed to strengthen the submission.

---

## 1. Data Quality Issues

### 1.1 Threat Saturation (Critical to Communicate)

**Observation:** `combined_threat_prob = 1.000 ± 0.000` for all 1,057,714 perception grid cells.

**Root cause:** The three SAM emitter positions are placed such that their coverage areas (based on maximum detection range) overlap the entire 33×55 km operating area at all three altitude bands. At any point in the area, at least one emitter can detect the vehicle with probability approaching 1.0.

**Is this a bug?** No. The Mahafza model correctly computes detection probability based on range, RCS, and SNR threshold. The emitter placement reflects the scenario assumption: "heavily contested airspace with no safe corridors."

**Consequence for planning:** The `threat_cost` column in the planning dataset is effectively constant (≈1.0 ± 0.0001). This means the NSGA-III threat objective provides no spatial gradient — the planner cannot differentiate between trajectories on the basis of threat exposure. The Pareto front therefore effectively reduces to a 2D trade-off (energy vs. time).

**What this means for the paper:** This must be disclosed prominently. The dataset demonstrates the *framework* for multi-objective defense trajectory planning, but the threat diversity is limited in this scenario.

**Recommended fix:** Add an alternate scenario with partial SAM coverage (2–3 emitters with non-overlapping detection ranges), creating genuine threat-free corridors. This would make the three-objective Pareto front meaningful.

**Status (2026-04-12): ADDRESSED.** A distance-decay threat gradient is now computed in `scripts/planning/dataset.py::_compute_threat_gradient()` using shortened effective ranges (20–30 km) per emitter. This produces `threat_cost` with mean=0.79, std=0.12, range 0.45–1.0 — genuine spatial variation usable by the planner. The physics-based `combined_threat_prob` (= 1.0 everywhere) is retained in the perception CSV for disclosure purposes but is no longer used in planning cost computation.

---

### 1.2 Position Error Metric (Not a Bug, But Misleading)

**Observation:** `pos_error_mean_m ≈ 51,000 m` (51 km average).

**Root cause:** The reference position during cruise is computed as:
```python
pos_ref_x = v_cruise × t   (from t=0, not from cruise-phase start)
```

Since `t` accumulates from mission start, and TAKEOFF + HOVER phases take ~30–60 s before cruise begins, the reference has already "traveled" 30–60 s × cruise_speed = 1,500–5,600 m before the vehicle reaches cruise speed. This gap compounds over the cruise duration.

**Is this a bug?** It is a design error in the reference signal. The controller's altitude (10.4 m) and attitude (0.001 rad) tracking are excellent — the vehicle is flying correctly. The position metric is wrong, not the flight.

**Consequence for ML:** Any model trained to predict `pos_error_mean_m` will be learning the relationship between cruise speed × mission time and a phantom position error, not actual tracking quality.

**Recommended fix (already noted in §10.2 of control_layer.md):** Reset the position reference at the start of each phase. The fix is a one-line change in `scripts/control/dataset.py`.

**Status (2026-04-12): FIXED.** Per-phase position reference reset implemented in `scripts/control/dataset.py`. Variables `px_phase_start` and `px_phase_ref` are reset at every phase transition. Position error is now computed as `vx_ref * phase_time_elapsed - (px - px_phase_start)`, eliminating the cumulative artifact.

---

### 1.3 Settling Time Ceiling

**Observation:** `settling_time_mean_s = 20.0 s` for most missions.

**Root cause:** The settling criterion requires BOTH `alt_error < 2.0 m` AND `vel_error < 1.0 m/s` simultaneously. During TAKEOFF → HOVER transition, altitude settles quickly but horizontal velocity overshoots. During HOVER → CRUISE transition, both errors are large simultaneously for an extended period.

**Is this a bug?** The criterion is overly conservative. Individual altitude settling is demonstrably good.

**Impact:** The metric is not useful for comparing controller variants or for ML prediction tasks.

**Recommended fix:** Separate altitude settling time from velocity settling time, and report each independently.

**Status (2026-04-12): FIXED.** Joint criterion replaced with independent trackers. Output columns are now `alt_settling_mean_s`, `alt_settling_max_s`, `vel_settling_mean_s`, `vel_settling_max_s` in the control dataset.

---

### 1.4 Zero Acoustic Variance

**Observation:** `spl_hover_a_dB = 92.84 ± 0.00 dB(A)`.

**Root cause:** The acoustic model in the vehicle layer uses hover thrust as the primary input. Since all missions have the same vehicle mass (203.9 kg → 2000 N hover thrust), all hover SPL values are identical.

**Is this meaningful?** For a single-vehicle scenario, yes — all vehicles have the same acoustic signature in hover. For a fleet comparison or ML task, this feature has zero discriminative power.

**Recommended fix:** Vary vehicle mass across missions, or incorporate RPM-dependent noise models where RPM varies with efficiency optimization.

---

### 1.5 Planning Dataset Size Mismatch

**Observation:** The `scripts/planning/dataset.py` docstring describes generating "1M–10M planning trajectory records," but the actual output is 2,000 records.

**Root cause:** Full NSGA-III optimization (200 generations × 100 population × 2,000 missions) is the bottleneck. The 1M figure refers to the raw sampling budget within RRT*.

**Impact:** 2,000 records is sufficient for feature analysis and small ML baselines, but insufficient for training large neural networks. The ML baselines suggested for NeurIPS should use appropriate model complexity for 2,000 samples (logistic regression, gradient boosting, small MLP with cross-validation).

---

## 2. Modeling Approximations

### 2.1 Control Layer — Simplified Plant

The control dataset uses a simplified 6-DoF plant, not the full BEMT + detailed aerodynamics model from the vehicle layer. Specific approximations:

| Simplification | Impact | Full model behavior |
|----------------|--------|---------------------|
| Linear drag only ($k_d v$) | Cruise drag underestimated at high speed | Quadratic drag + induced drag |
| No wing lift in cruise | Thrust overestimated in cruise | Wing provides ~50% of lift at 70 m/s |
| Euler forward integration | Attitude error ~0.01 rad over 600 s mission | RK4 eliminates this |
| No sensor noise | Controller performance overoptimistic | Realistic AHRS + GPS noise |
| No rotor dynamics | Instant thrust response | ~0.1–0.5 s motor + rotor time constant |

### 2.2 Planning Layer — Flat-Earth Approximation

The NED coordinate conversion uses flat-earth approximation:
$$N = \Delta\phi \cdot R, \quad E = \Delta\lambda \cdot R \cdot \cos\phi_0$$

For the 33 km × 55 km operating area, the earth curvature error is < 0.1%, which is negligible compared to other model errors.

### 2.3 Vehicle Layer — No Blade Stall Model

The BEMT rotor model does not include blade stall at high collective pitch or high advance ratio. This means rotor efficiency is overestimated at the edges of the operating envelope (very high speed or very high altitude). For the modeled scenario (200 m AGL, 48–94 m/s cruise), blade stall is unlikely but not impossible at peak speed.

### 2.4 Euler Kinematics Singularity

The Euler ZYX kinematics have a singularity at $\theta = \pm 90°$ (gimbal lock). This is handled by clamping $\cos\theta \geq 10^{-4}$. For a tiltrotor in normal flight ($|\theta| < 30°$), this never triggers. For aggressive maneuvers, a quaternion integrator should be used.

---

## 3. NeurIPS Datasets & Benchmarks Track Assessment

### 3.1 What the Track Requires

The NeurIPS D&B track expects:
1. **A novel dataset** with significant scale, quality, or novelty
2. **Clear ML tasks** supported by the dataset
3. **Baseline results** from at least 2–3 ML methods
4. **Proper evaluation methodology** (train/test splits, significance tests)
5. **Comparison to existing datasets** if applicable
6. **Discussion of limitations and ethical considerations**

### 3.2 Current Status

*Last updated: 2026-04-12*

| Requirement | Status | Notes |
|-------------|--------|-------|
| Novel dataset | ✅ Strong | First open defense eVTOL dataset with real geospatial + multi-signature |
| Scale | ⚠️ Partial | 1M perception rows is good; 2K planning/vehicle/control rows is small |
| ML tasks defined | ✅ Done | 6 tasks defined in introduction.md |
| Baseline results | ✅ Done | LR, GBM, MLP on T1 (risk classification) and T2 (energy regression); results in outputs/ml/baseline_results.csv |
| Train/test splits | ✅ Done | 80/10/10 stratified splits in outputs/splits/ (seed=42) |
| Comparison to existing | ⚠️ Partial | Qualitative comparison only |
| Ethical discussion | ✅ Done | In introduction.md and doc/research_limitations.md §5 |
| Limitations | ✅ Done | This document |
| Datasheet | ✅ Done | Gebru et al. 2021 format in doc/datasheet.md |
| Tutorial notebook | ✅ Done | notebooks/tutorial.ipynb |
| Threat gradient fix | ✅ Done | Distance-decay model in scripts/planning/dataset.py |
| Position error fix | ✅ Done | Per-phase reset in scripts/control/dataset.py |
| Settling time fix | ✅ Done | Separate alt/vel metrics in scripts/control/dataset.py |

### 3.3 Priority Actions for Submission Readiness

**Must do (blockers) — all completed as of 2026-04-12:**

1. ✅ **Baseline ML experiments** — LR, GBM, MLP on T1 (risk classification, AUC up to 0.9998) and T2 (energy regression, R² up to 0.9997). Results in `outputs/ml/baseline_results.csv`. Script: `scripts/ml/baseline_experiments.py`.

2. ✅ **Train/val/test split files** — Stratified 80/10/10 by `risk_label`, seed=42. Saved in `outputs/splits/` as numpy index arrays.

3. ✅ **Fix the position error reference signal** — Per-phase reset implemented in `scripts/control/dataset.py`. Position error now measures actual tracking quality within each phase.

4. ✅ **Fix the threat scenario** — Distance-decay gradient (effective ranges 20–30 km) added in `scripts/planning/dataset.py::_compute_threat_gradient()`. Produces std=0.12 across the operating area.

**Should do (strengthen submission):**

1. ⬜ Generate 10,000–50,000 planning records (current: 2,000; bottleneck is RRT* runtime)
2. ⬜ Add a second geographic region (generalization benchmark)
3. ⬜ Run HITL simulation on a flight simulator (e.g., X-Plane + ArduPilot)
4. ✅ **Produce a datasheet** (Gebru et al. 2021 format) — done, see `doc/datasheet.md`.

**Could do (polish):**

1. ⬜ Host dataset on Hugging Face Datasets
2. ⬜ Provide a `load_dataset()` compatible wrapper
3. ✅ **Create a Jupyter notebook tutorial** — done, see `notebooks/tutorial.ipynb`.
4. ⬜ Submit dataset to Papers With Code

### 3.4 Positioning for NeurIPS

**Strongest framing:** "We present the first open benchmark for multi-objective trajectory optimization in contested airspace, combining real geospatial perception data, analytical SAM threat modeling, and end-to-end closed-loop simulation. We define six ML tasks and provide baseline results, enabling direct comparison for future methods."

**Weaker framings to avoid:**
- Claiming the NSGA-III or RRT* implementations are novel (they are not)
- Claiming state-of-the-art control performance (the controller is a standard cascaded PID)
- Claiming the dataset can train RL agents directly (2,000 records is too small for offline RL)
---

## 4. Code Quality Issues

### 4.1 Duplicate Implementations

| Component | Canonical | Duplicate | Recommendation |
|-----------|-----------|-----------|----------------|
| NSGA-III | `planning/optimization/nsga3.py` (978 lines) | `planning/nsga3_optimizer.py` (~300 lines) | Keep canonical, deprecate duplicate |
| RRT* | `planning/algorithms/sampling.py` | `planning/rrt_star.py` | Consolidate into algorithms/ |

### 4.2 Incompatible State Definitions

The `VehicleState` dataclass is defined independently in 5 files:
- `core/state.py`
- `vehicle/vehicle_model.py`
- `control/sitl_simulator.py`
- `control/advanced_modes/advanced_modes.py`
- `hardware/ardupilot_bridge.py`

These have different field sets and units. An adapter layer is needed for integration. The dataset scripts avoid this issue by using flat dictionaries, but any code that imports `VehicleState` from multiple modules will encounter type errors.

### 4.3 Motor/Nacelle Config Mismatches

`vehicle/propulsion/motor_model.py` references `config.torque_constant`, `config.pole_pairs` which do not exist in `MotorConfig`. These are non-blocking for dataset generation (the scripts bypass the config system) but would cause `AttributeError` if the full vehicle stack were used end-to-end.

---

## 5. Ethical and Security Considerations

### 5.1 Dual-Use Assessment

The primary dual-use concern is the SAM threat model. Our assessment:

**What is disclosed:**
- Detection probability formula type (Marcum Q-function / Albersheim approximation) — published in textbooks
- Generic threat category parameterization (long-range, medium-range, MANPADS) — publicly documented
- Geographic operating area (Delhi-NCR) — public satellite imagery and OSM data

**What is NOT disclosed:**
- Classified system parameters for any specific SAM system
- Actual deployment locations of any real air defense system
- Engagement rules of engagement or command/control specifics

**Conclusion:** The threat model is equivalent in sensitivity to what is published in open academic literature on radar systems design. It does not provide meaningful additional capability to an adversary beyond what is already publicly available.

### 5.2 Geographic Data

All geographic data is from publicly available sources:
- NASA SRTM: public domain
- Open-Meteo: CC-BY 4.0
- OpenStreetMap: ODbL (Open Database License)

No restricted-access or classified geographic data is used.

### 5.3 Reproduction Constraints

The dataset can be fully reproduced by running the pipeline scripts. The only external dependencies are:
- Open-Elevation API (public, rate-limited to 10 req/s)
- Open-Meteo API (public, no authentication required)
- OpenStreetMap Overpass API (public, subject to fair use)

No proprietary software or data sources are required.

---

## 6. Citation for This Limitations Document

When referencing the limitations analysis:

> Ramakrishnan, A. (2026). *Research Limitations and NeurIPS Submission Notes: Trajectory Optimization in Defense eVTOLs*. Technical document. Available at: doc/research_limitations.md in the project repository.
