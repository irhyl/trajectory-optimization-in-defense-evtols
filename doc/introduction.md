# Introduction — Research Context and Motivation

## 1. The Defense eVTOL Problem

Electric vertical take-off and landing (eVTOL) aircraft are transitioning from urban air mobility concepts to serious defense platforms. Programs such as the U.S. Army's FARA (Future Attack Reconnaissance Aircraft), AFWERX Agility Prime, and European equivalents are actively evaluating electrically-powered rotorcraft for ISR, logistics resupply, and medical evacuation in contested environments.

Unlike civilian eVTOLs designed for predictable urban corridors, defense applications face a fundamentally different constraint set:

**Mission-level constraints:**
- Time-critical objectives (minutes matter in CASEVAC, perishable intelligence windows)
- No fixed infrastructure (no GPS-guided landing zones, no pre-mapped airspace, no ATC)
- Variable threat environments (threat positions change faster than mission replanning cycles)

**Vehicle-level constraints:**
- Battery energy is the hard ceiling — no in-field refueling in contested ground
- Acoustic signature drives detectability at low altitude (rotors are loud)
- Radar cross-section must be managed against ground-based air defenses
- Infrared signature increases with motor temperature during hover

**Fundamental tension:** The actions that minimize detection (low altitude, high speed, terrain masking) are often incompatible with minimizing energy consumption (high altitude, moderate speed, direct routing). A mission planner that optimizes only one objective will produce plans that are either easily detected or unable to complete the mission on battery alone.

## 2. Why Existing Approaches Fall Short

### 2.1 Single-Objective Planners

Most autonomous UAV trajectory planners in the literature — whether A*, RRT, potential fields, or model-predictive control — optimize a single weighted cost. When threat avoidance is added as a penalty weight, the resulting solution depends critically on the weight choice, which must be hand-tuned per mission. Different planners with different weight settings are not directly comparable, and no systematic analysis of the Pareto trade-off is possible.

### 2.2 Missing Signature Integration

Existing open datasets for UAV trajectory planning (e.g., DARPA SubT, AirSim environments, OpenAI gym UAV environments) do not include any ground-truth signature data (acoustic, infrared, or radar cross-section). Yet signature management is the primary differentiator between a survivable and non-survivable defense mission. A planner trained on these datasets cannot learn the trade-off between altitude (for terrain masking against radar) and speed (for acoustic signature reduction).

### 2.3 Simulation-Reality Gap in Control

Published UAV control datasets are typically either:
- **Lab-scale** (small quadrotors, indoor, calm air), not representative of 200 kg eVTOL dynamics
- **High-fidelity simulation** without physics validation against real hardware
- **Hardware flight logs** without access to the planning/perception layers that generated the commands

This prevents end-to-end ML research that spans perception → planning → control.

### 2.4 No Defense-Specific Open Benchmark

To our knowledge, no open benchmark exists that:
1. Models SAM (surface-to-air missile) detection probability analytically
2. Combines multi-signature modeling (acoustic + IR + RCS) under trajectory optimization
3. Provides real geospatial data for a realistic operating theater
4. Closes the loop from perception through planning through control through vehicle dynamics

## 3. Our Contribution

This work introduces a **full-stack autonomous mission dataset** for defense eVTOL trajectory optimization with the following novel characteristics:

### 3.1 Real Geospatial Foundation

All perception data is derived from real physical sources with no synthetic fallbacks:
- **Terrain:** NASA SRTM30 elevation, 90 m resolution, accessed via Open-Elevation API
- **Wind:** Open-Meteo GFS atmospheric forecast model, real forecast data for the region
- **Obstacles:** OpenStreetMap Overpass API, actual building and infrastructure geometry
- **Threat model:** Analytical SAM detection probability derived from published radar theory (Mahafza 2005), parameterized for three real-world threat categories (GBAD long-range surveillance, medium-range fire control, short-range MANPADS)

### 3.2 Multi-Objective Pareto Planning

For the first time in an open dataset context, trajectory planning is formulated as a genuine three-objective optimization:

$$\min_\tau \left[ E(\tau),\; T(\tau),\; P_{\text{detect}}(\tau) \right]$$

where $E$ is energy consumption [Wh], $T$ is mission time [s], and $P_{\text{detect}}$ is integrated SAM detection probability [0,1]. The NSGA-III algorithm produces the complete Pareto front, making the trade-off explicit and quantifiable.

### 3.3 End-to-End Layered Architecture

The four-layer stack generates datasets that are causally linked:
- Planning trajectories were generated using the perception cost field
- Vehicle simulations used planning trajectories as inputs
- Control simulations used vehicle mission parameters as inputs

This causal chain enables cross-layer ML research: e.g., predicting control performance from planning features, or predicting battery consumption from perception complexity.

### 3.4 Multi-Signature Vehicle Modeling

The vehicle layer simultaneously models three detection signatures under actual mission dynamics:
- **Acoustic (SPL):** Rotor noise A-weighted, altitude-dependent propagation
- **Infrared (LWIR):** Motor and nacelle thermal emission under dynamic power demand
- **Radar (RCS):** Physical optics estimate at X-band for frontal and top-down aspects

No prior open UAV dataset includes all three under the same flight conditions.

## 4. Operating Scenario

The modeled scenario represents a contested urban-periphery environment:

**Geographic area:** Delhi-NCR region, India  
**Coordinates:** 28.7°–29.0°N, 76.9°–77.4°E (approximately 33 km × 55 km)  
**Altitude bands:** 50 m, 150 m, 300 m AGL  
**Terrain:** Mixed agricultural, peri-urban, urban; elevation 190–305 m MSL (Aravalli foothills)  
**Wind:** Prevailing westerly to north-westerly at 4–15 m/s (seasonal GFS forecast)

**Threat scenario:** Three overlapping SAM networks covering the entire operating area. This represents a heavily defended airspace — a "worst case" for a covert penetration mission. The planner cannot find a threat-free path; it can only minimize total exposure time and optimize aspect angles.

**Vehicle:** Tiltrotor quad-eVTOL  
- MTOW: ~204 kg (450 lb class)
- Thrust-to-weight ratio: 2.4 (max 4,800 N / 2,000 N weight)
- Battery: 44 kWh, 400V Li-NMC chemistry
- Cruise speed: 48–94 m/s (rotor efficiency optimized)
- Cruise altitude: 200 m AGL (fixed for this scenario)
- Endurance: ~45 minutes at 70 m/s cruise

## 5. Research Questions Addressed

The dataset is structured to support the following ML research questions:

**RQ1 — Risk Prediction:** Can mission risk be predicted from pre-flight planning features?  
*Task:* Binary classification, `risk_label ∈ {0,1}`, using planning + vehicle features.

**RQ2 — Energy Forecasting:** Can mission energy consumption be accurately predicted before flight?  
*Task:* Regression on `energy_consumed_wh`, using planning geometry and weather features.

**RQ3 — Control Stability Prediction:** Which mission parameters lead to control saturation or degraded tracking?  
*Task:* Classification on `n_saturations > 0`, regression on `alt_error_rms_m`.

**RQ4 — Optimal Trade-off Learning:** Can a model learn the Pareto-optimal trade-off between energy, time, and threat exposure from historical data?  
*Task:* Multi-output regression or Pareto-front approximation from the planning dataset.

**RQ5 — Perception-to-Planning Transfer:** Does a model trained on the perception cost field generalize to novel geographic areas?  
*Task:* Domain adaptation / transfer learning on the 4D cost field.

## 6. Broader Impact

**Positive impact:**
- Enables academic research on defense-relevant AI that would otherwise require classified environments or expensive hardware
- Provides a standardized benchmark for comparing multi-objective trajectory optimizers
- The real geospatial foundation makes results interpretable in real-world terms

**Potential risks:**
- Detailed SAM threat modeling: the Mahafza (2005) detection probability model used is available in published textbooks and does not expose classified system parameters. The parameters used are publicly documented for generic threat categories.
- The geographic region (Delhi-NCR) is publicly accessible terrain data — no sensitive infrastructure locations are modeled.
- All obstacle data is from OpenStreetMap's public dataset.

We judge the research benefit of an open benchmark for defense autonomous systems to outweigh these concerns, consistent with the dual-use reasoning frameworks established by prior work in adversarial machine learning and cybersecurity benchmarking.

## 7. Relationship to Prior Work

| Work | Approach | Gap vs. this work |
|------|----------|-------------------|
| Karaman & Frazzoli (2011) | RRT* planning | No multi-objective, no threat, no closed-loop simulation |
| Deb & Jain (2014) | NSGA-III | Algorithm only, no UAV-specific dataset |
| Lin & Saripalli (2017) | UAV path planning in urban terrain | No threat, no vehicle dynamics, no control layer |
| Guerra et al. (2019) | Fixed-wing UAV NSGA-II vs NSGA-III | No eVTOL dynamics, no signature modeling |
| AgriNav, AirSim, FlyThrough | Simulation environments | No defense signatures, no real geospatial data |
| OpenUAV, SubT Challenge | Real-world datasets | No planning optimization, no control analytics |

This work uniquely combines: real terrain + wind data, analytical SAM threat, multi-signature vehicle modeling, multi-objective planning, and closed-loop control simulation in a single causally-linked dataset.

## 8. Paper Structure (Suggested NeurIPS D&B Format)

1. Introduction (this document)
2. Related Work
3. Dataset Construction
   - 3.1 Geographic Area and Data Sources
   - 3.2 Perception Layer
   - 3.3 Planning Layer
   - 3.4 Vehicle Physics Layer
   - 3.5 Control Layer
4. Dataset Analysis
   - 4.1 Perception Statistics
   - 4.2 Planning Pareto Front Analysis
   - 4.3 Vehicle Physics Validation
   - 4.4 Control Performance Summary
5. Benchmark Tasks and Baselines
6. Limitations
7. Conclusion

## 9. References

1. Karaman, S., & Frazzoli, E. (2011). Sampling-based algorithms for optimal motion planning. *IJRR*, 30(7), 846–894.
2. Deb, K., & Jain, H. (2014). An evolutionary many-objective optimization algorithm using reference-point-based nondominated sorting. *IEEE TEC*, 18(4), 577–601.
3. Mahafza, B. R. (2005). *Radar Systems Analysis and Design Using MATLAB* (2nd ed.). CRC Press.
4. Johnson, W. (2013). *Rotorcraft Aeromechanics*. Cambridge University Press.
5. Guerra, W., et al. (2019). Fast trajectory optimization for agile quadrotor maneuvers with a cable-suspended payload. *RSS*.
6. Lin, Y., & Saripalli, S. (2017). Sampling-based path planning for UAV collision avoidance. *IEEE TASE*, 14(2), 916–928.
7. Open-Meteo (2023). *Open-Meteo Weather API Documentation*. https://open-meteo.com/
8. OpenStreetMap Contributors (2023). *Overpass API*. https://overpass-api.de/
