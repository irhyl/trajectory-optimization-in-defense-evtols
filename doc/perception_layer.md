# Perception Layer: Theory, Mathematics, Physics & Implementation

**Defense eVTOL Trajectory Optimization System**  
Version 2.0  |  Research Grade  |  Author: Defense eVTOL Research Team

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Coordinate Frames & Geodetic Mathematics](#2-coordinate-frames--geodetic-mathematics)
3. [Terrain Perception Layer](#3-terrain-perception-layer)
4. [Wind & Atmospheric Layer](#4-wind--atmospheric-layer)
5. [Obstacle Perception Layer](#5-obstacle-perception-layer)
6. [Threat Assessment Layer](#6-threat-assessment-layer)
7. [Sensor Fusion & Cost Aggregation](#7-sensor-fusion--cost-aggregation)
8. [Data Provenance & APIs](#8-data-provenance--apis)
9. [Dataset Schema & Planning Integration](#9-dataset-schema--planning-integration)
10. [Implementation Architecture](#10-implementation-architecture)
11. [Limitations & Future Work](#11-limitations--future-work)
12. [References](#12-references)

---

## 1. System Overview

The perception layer is the first stage of a four-layer autonomous mission stack:

```
Perception  →  Planning (RRT* + NSGA-III)  →  Control  →  Vehicle Dynamics
```

Its responsibility is to produce a spatiotemporal **traversability cost field** over the operational theatre — a 4D function $C(\phi, \lambda, h, t)$ where $\phi$ is latitude, $\lambda$ longitude, $h$ altitude, and $t$ time — that encodes every physical hazard relevant to the eVTOL's mission survivability and efficiency.

The five submodules that contribute to this cost field are:

| Module | Physical domain | Primary data source |
|--------|----------------|-------------------|
| Terrain | Orography, elevation clearance, slope | NASA SRTM via Open-Meteo Elevation API |
| Wind | Atmospheric dynamics, turbulence, gusts | GFS/HRES via Open-Meteo Forecast API |
| Obstacle | Static & dynamic airspace objects | OpenStreetMap Overpass API, OpenSky Network |
| Threat | Radar detection, SAM engagement probability | Analytical (radar range equation) |
| Fusion | Weighted composite traversability cost | In-system combination |

**Design principle**: No synthetic data fallbacks. Every data point in the dataset traces to a real physical measurement or a physics-based analytical model with documented derivation.

---

## 2. Coordinate Frames & Geodetic Mathematics

### 2.1 WGS-84 Geodetic Frame

All raw data is stored in the World Geodetic System 1984 (WGS-84) frame:

- Latitude $\phi$ (degrees North, −90° to +90°)
- Longitude $\lambda$ (degrees East, −180° to +180°)
- Height $h$ above the WGS-84 ellipsoid (metres)

WGS-84 defines Earth as an oblate spheroid with:
$$a = 6{,}378{,}137.0\text{ m},\quad b = 6{,}356{,}752.3\text{ m},\quad f = (a-b)/a \approx 1/298.257$$

### 2.2 Earth-Centred Earth-Fixed (ECEF)

ECEF Cartesian coordinates $(X, Y, Z)$ relate to WGS-84 via:

$$\begin{pmatrix} X \\ Y \\ Z \end{pmatrix} = \begin{pmatrix} (N(\phi) + h)\cos\phi\cos\lambda \\ (N(\phi) + h)\cos\phi\sin\lambda \\ \left(N(\phi)(1-e^2) + h\right)\sin\phi \end{pmatrix}$$

where the radius of curvature in the prime vertical is:
$$N(\phi) = \frac{a}{\sqrt{1 - e^2\sin^2\phi}},\quad e^2 = 1 - \frac{b^2}{a^2} \approx 0.00669438$$

### 2.3 NED (North-East-Down) Local Frame

The planning layer works in NED metres relative to a reference point $(\phi_0, \lambda_0, h_0)$. The flat-Earth approximation (valid for theatres $\lesssim$200 km) gives:

$$N = (\phi - \phi_0) \cdot \frac{\pi}{180} \cdot R_\phi$$
$$E = (\lambda - \lambda_0) \cdot \frac{\pi}{180} \cdot R_\lambda \cdot \cos\phi_0$$
$$D = -(h - h_0)$$

where:
$$R_\phi = 6{,}356{,}752.3 + (6{,}378{,}137.0 - 6{,}356{,}752.3)\cos^2\phi_0 \approx 111{,}320 \text{ m/deg (mean)}$$
$$R_\lambda = R_\phi \cos\phi_0$$

For the Delhi NCR theatre ($\phi_0 = 28.5°$N):
- $R_\phi \approx 111{,}200$ m/deg
- $R_\lambda \approx 97{,}800$ m/deg

### 2.4 Haversine Formula

The Haversine formula gives the great-circle distance between two geodetic points:

$$d = 2R\arcsin\sqrt{\sin^2\frac{\Delta\phi}{2} + \cos\phi_1\cos\phi_2\sin^2\frac{\Delta\lambda}{2}}$$

where $R = 6{,}371{,}000$ m (mean Earth radius) and angles are in radians. This formula is numerically stable for both small and large distances.

---

## 3. Terrain Perception Layer

### 3.1 Physical Significance

For low-altitude eVTOL operations (50–1500 m AGL), terrain poses three distinct hazards:

1. **Collision**: If altitude MSL < elevation MSL + minimum separation.
2. **Radar masking**: Terrain ridges block radar line-of-sight, providing electromagnetic concealment.
3. **Turbulence generation**: Orographic flow over terrain produces mechanical turbulence.

### 3.2 Data Source: SRTM

The Shuttle Radar Topography Mission (SRTM, NASA/NGA, February 2000) produced the primary global DEM used in this system.

**Mission parameters**:
- C-band SAR (5.3 GHz) in interferometric mode
- Single-pass across-track interferometry
- Coverage: 56°S to 60°N
- Horizontal resolution: 1 arc-second (≈30 m at equator) globally released 2014
- Vertical accuracy: ~16 m RMSE (1σ) relative to EGM96 geoid [Rodriguez et al. 2006]
- Vertical datum: EGM96 (Earth Gravitational Model 1996)

**Elevation measurement**:
$$h_{SRTM} = h_{ellipsoid} - N_{geoid}$$
where $N_{geoid}$ is the geoid undulation from EGM96. For Delhi NCR, $N_{geoid} \approx -46$ m.

### 3.3 Grid Construction

Data is fetched via the Open-Meteo Elevation API which wraps SRTM at the following endpoint:

```
GET https://api.open-meteo.com/v1/elevation?latitude={lat}&longitude={lon}
```

**Coarse-to-fine interpolation strategy**:

1. Fetch SRTM at 2 km coarse grid (≈870 points for the 55×60 km theatre)
2. Bilinear interpolation to 222 m fine grid (251×301 = 75,551 points)

For bilinear interpolation at query point $(x, y)$ within a cell $(x_1, x_2) \times (y_1, y_2)$:

$$f(x,y) = \frac{1}{(x_2-x_1)(y_2-y_1)}\Big[(x_2-x)(y_2-y)f_{11} + (x-x_1)(y_2-y)f_{21}$$
$$+ (x_2-x)(y-y_1)f_{12} + (x-x_1)(y-y_1)f_{22}\Big]$$

**API rate limiting**: Open-Meteo allows 100 points per request, 600 req/min (free tier). Batching with `rate_limit_delay_s = 1.5 s` between batches.

### 3.4 Derivative Products

#### Slope

Slope is computed from the fine-grid elevation array using central finite differences:

$$\frac{\partial z}{\partial N} = \frac{z_{i+1,j} - z_{i-1,j}}{2\,\Delta N},\quad \frac{\partial z}{\partial E} = \frac{z_{i,j+1} - z_{i,j-1}}{2\,\Delta E}$$

$$\alpha_{slope} = \arctan\sqrt{\left(\frac{\partial z}{\partial N}\right)^2 + \left(\frac{\partial z}{\partial E}\right)^2}$$

where $\Delta N = \Delta\phi \cdot 111{,}320$ m/deg and $\Delta E = \Delta\lambda \cdot 111{,}320 \cdot \cos\phi_0$ m/deg.

For Delhi NCR: mean slope = **0.16°**, max = **2.4°** (flat Gangetic plain, as expected).

#### Roughness

Terrain roughness is estimated as the standard deviation of elevation in a 3×3 neighbourhood:

$$\sigma_{rough}(i,j) = \text{std}(\{z_{k,l}: |k-i| \leq 1, |l-j| \leq 1\})$$

This is a proxy for the Vector Ruggedness Measure (VRM) and indicates landing zone suitability and turbulence generation potential.

#### Terrain Clearance

$$h_{clearance}(lat,lon,alt_{MSL}) = alt_{MSL} - z_{SRTM}(lat,lon)$$

A minimum $h_{clearance} = 50$ m is required for safe flight.

### 3.5 Terrain Cost Function

$$C_{terrain}(x) = 0.5 \cdot \frac{\alpha_{slope}}{45°} + 0.5 \cdot \exp\!\left(-\frac{h_{clearance}}{300}\right)$$

Properties:
- $C_{terrain} \to 1$ when slope → 45° (RRT* avoidance threshold) or clearance → 0
- $C_{terrain} \to 0$ when slope → 0° and clearance ≫ 300 m
- $C \in [0, 1]$ always

---

## 4. Wind & Atmospheric Layer

### 4.1 Physical Significance

Wind affects the eVTOL through four mechanisms:

1. **Energy**: Headwind increases required thrust and energy consumption
2. **Turbulence**: Gust loads stress airframe; control bandwidth required
3. **Crosswind**: Limits operational envelope, particularly in hover-to-cruise transition
4. **Wind shear**: Sharp velocity gradients require rapid control response

### 4.2 Atmospheric Structure: International Standard Atmosphere

The ISA defines the standard pressure–altitude relationship:

$$P(h) = P_0 \left(1 - \frac{L \cdot h}{T_0}\right)^{g/(RL)}$$

For $h < 11{,}000$ m (troposphere):
- $P_0 = 101{,}325$ Pa, $T_0 = 288.15$ K
- Lapse rate $L = 0.0065$ K/m
- $g = 9.80665$ m/s², $R = 287.05$ J/(kg·K)

Simplified pressure-altitude inversion:
$$h(P) = 44{,}330 \cdot \left[1 - \left(\frac{P}{P_0}\right)^{0.1903}\right]$$

Standard pressure levels and their ISA altitudes used in this system:

| Pressure (hPa) | ISA altitude (m) |
|---|---|
| 1000 | 111 |
| 925 | 762 |
| 850 | 1458 |
| 700 | 3012 |
| 500 | 5574 |

### 4.3 Data Source: Open-Meteo GFS/HRES

Open-Meteo provides free access to operational Numerical Weather Prediction (NWP) model output:

- **NOAA GFS** (Global Forecast System): 0.25° resolution, 16-day forecast, 4× daily updates
- **ECMWF HRES** (High RESolution): 0.1° resolution for 10-day forecast
- **DWD ICON**: regional ensemble model

Wind data is provided at pressure levels in the following variables:
- `wind_speed_{P}hPa` (m/s): magnitude $|W| = \sqrt{U^2 + V^2}$
- `wind_direction_{P}hPa` (°): meteorological FROM direction

**Meteorological convention** (FROM direction $\theta$, clockwise from North):
$$U = -|W|\sin\theta \quad \text{(eastward, positive East)}$$
$$V = -|W|\cos\theta \quad \text{(northward, positive North)}$$

**NED convention** (this system):
$$\text{wind\_NED} = [V_{met},\; U_{met},\; -W_{vertical}]$$
i.e., $[northward, eastward, -upward]$.

### 4.4 Spatial Interpolation

Five anchor points are used (centre + 4 corners of theatre). For each anchor, a separate API request is made. The fine spatial grid is filled via Inverse Distance Weighting (IDW):

$$\hat{f}(x) = \frac{\sum_{i=1}^{N} w_i(x) \cdot f_i}{\sum_{i=1}^{N} w_i(x)}, \quad w_i(x) = \frac{1}{d_i(x)^2}$$

where $d_i(x)$ is the distance from grid point $x$ to anchor $i$.

### 4.5 Turbulence Estimation

Turbulence intensity (TI) is estimated from vertical wind shear between adjacent pressure levels:

$$TI(h) = \frac{|\Delta|W|/\Delta h| \cdot h}{\overline{|W|}}$$

where $\Delta|W|/\Delta h$ is the wind speed shear and $\overline{|W|}$ is the mean speed at level $h$. This approximates the mechanical turbulence production term in the TKE (Turbulent Kinetic Energy) equation:

$$\frac{\partial k}{\partial t} = -\overline{u_i'u_j'}\frac{\partial \overline{u_i}}{\partial x_j} + \frac{g}{\theta_0}\overline{w'\theta'} - \epsilon$$

where the first term is shear production, second is buoyant production, third is dissipation.

### 4.6 Wind Cost Function

$$C_{wind}(x) = \frac{|W|}{W_{max}} \cdot \left(1 + 0.3 \cdot TI\right)$$

where $W_{max} = 30$ m/s (eVTOL operational limit). The turbulence factor amplifies cost in regions of high shear by up to 30%.

The headwind component (for energy calculation, not the cost function):
$$W_{head} = \mathbf{W} \cdot \hat{v}_{track}$$
$$\Delta P_{headwind} = \frac{1}{2}\rho A_{disk} \left[(V_{as} + W_{head})^2 - V_{as}^2\right]$$

---

## 5. Obstacle Perception Layer

### 5.1 Physical Significance

Low-altitude airspace is dense with vertical obstacles:

- **Buildings**: Urban high-rises, commercial towers (10–300 m)
- **Power infrastructure**: Pylons (30–80 m) and conductors (invisible to camera)
- **Telecom towers**: Microwave masts (20–120 m), often with guy wires extending 100–200 m horizontally
- **Religious structures**: Minarets, gopurams, steeples (15–80 m)
- **Chimneys**: Industrial stacks (50–300 m) with thermal plumes

The Delhi NCR theatre contains **18,690 real obstacles** from the OpenStreetMap dataset.

### 5.2 Data Source: OpenStreetMap Overpass API

OpenStreetMap (OSM) is a collaborative geographic database (ODbL licence). The Overpass API allows structured queries:

```
GET https://overpass-api.de/api/interpreter?data=[out:json];...
```

The Overpass QL query for our theatre:
```
[out:json][timeout:120];
(
  way["building"](28.40,76.90,28.90,77.50);
  node["man_made"~"tower|mast"](28.40,76.90,28.90,77.50);
  way["power"="tower"](28.40,76.90,28.90,77.50);
  way["power"="line"](28.40,76.90,28.90,77.50);
  node["man_made"="chimney"](28.40,76.90,28.90,77.50);
  node["building"~"temple|mosque|church"](28.40,76.90,28.90,77.50);
);
out center geom;
```

### 5.3 Height Estimation

OSM height data comes from the `height` or `building:levels` tags. Where absent:

$$h_{est} = \begin{cases}
n_{levels} \times 3.0 \text{ m} & \text{if } n_{levels} \text{ tagged}\\
10.0 \text{ m} & \text{default building}\\
30.0 \text{ m} & \text{default tower}\\
40.0 \text{ m} & \text{default power pylon}
\end{cases}$$

**Delhi NCR obstacle statistics** from this dataset:
- Buildings: 7,936 (mean height ~10 m, max 300 m for identified high-rises)
- Power pylons: 8,771 (highest concentration — national grid infrastructure)
- Towers: 591
- Chimneys: 324
- Religious structures: 131

### 5.4 KDTree Nearest-Obstacle Query

For the 75,551 fine-grid spatial points, nearest-obstacle distance is computed via a 2D KD-Tree (SciPy `cKDTree`) over the planar projection:

$$x_{cart} = \phi_{obs} \times 111{,}320 \text{ m/deg}$$
$$y_{cart} = \lambda_{obs} \times 111{,}320 \times \cos\phi_0 \text{ m/deg}$$

KDTree query complexity: $O(n \log n)$ construction, $O(\log N)$ per query. For $N = 17{,}753$ obstacles and $M = 75{,}551$ queries: total ≈ 0.1 s.

### 5.5 Clearance & Safety Geometry

Three separation zones are defined:

| Zone | Radius | Semantic |
|------|--------|---------|
| Hard collision | $r_{obs} + 10$ m | Structural contact |
| Danger zone | $r_{obs} + 50$ m | Uncontrolled flight |
| Safe zone | $r_{obs} + 200$ m | Normal operations |

Where $r_{obs}$ is the obstacle's bounding cylinder radius.

### 5.6 Conflict Detection: CPA/TTC Analysis

For dynamic obstacles (tracked aircraft from OpenSky), conflict detection uses:

**Closest Point of Approach (CPA)**:
$$\tau_{CPA} = -\frac{\Delta\mathbf{r} \cdot \Delta\mathbf{v}}{|\Delta\mathbf{v}|^2}$$

**Separation at CPA**:
$$d_{CPA} = |\Delta\mathbf{r}(\tau_{CPA})| = |\Delta\mathbf{r} + \tau_{CPA}\Delta\mathbf{v}|$$

**Time to Collision (TTC)** for spherical safety volumes of radii $r_1, r_2$:
$$\tau_{TTC} = \frac{-(\Delta\mathbf{r}\cdot\Delta\mathbf{v}) - \sqrt{(\Delta\mathbf{r}\cdot\Delta\mathbf{v})^2 - |\Delta\mathbf{v}|^2(|\Delta\mathbf{r}|^2 - (r_1+r_2)^2)}}{|\Delta\mathbf{v}|^2}$$

if the discriminant is positive (collision course).

### 5.7 Obstacle Cost Function

$$C_{obstacle}(x) = \exp(-\lambda \cdot d_{nearest}(x))$$

where $\lambda = 0.005$ m⁻¹ (decay rate, tuned so $C_{obs} = 0.37$ at $d = 200$ m). This exponential form:
- Approaches 1 at $d \to 0$ (collision)
- Approaches 0 at $d \to \infty$ (clear airspace)
- Is differentiable everywhere (gradient-based planning compatible)

---

## 6. Threat Assessment Layer

### 6.1 Physics of Radar Detection

#### 6.1.1 Radar Range Equation

The fundamental radar range equation (Friis, 1946; Skolnik, 2008) relates received power to range:

$$P_r = \frac{P_t \cdot G_t \cdot G_r \cdot \lambda^2 \cdot \sigma}{(4\pi)^3 \cdot R^4}$$

where:
- $P_t$ = transmitted power (W)
- $G_t = G_r = G$ = antenna gain (monostatic radar: transmit = receive)
- $\lambda = c/f$ = wavelength (m)
- $\sigma$ = target Radar Cross Section (m²)
- $R$ = slant range (m)

**Signal-to-Noise Ratio**:
$$\text{SNR} = \frac{P_r}{P_n} = \frac{P_t G^2 \lambda^2 \sigma}{(4\pi)^3 R^4 \cdot kTBF}$$

where $k = 1.38\times10^{-23}$ J/K (Boltzmann), $T = 290$ K (system temperature), $B$ = bandwidth (Hz), $F$ = noise figure.

**Maximum detection range** $R_{max}$ (at SNR = SNR$_{min}$, $\sigma = 1$ m²):
$$R_{max} = \left[\frac{P_t G^2 \lambda^2}{(4\pi)^3 \cdot \text{SNR}_{min} \cdot kTBF}\right]^{1/4}$$

For a target with RCS $\sigma \neq 1$ m²:
$$R_{max}(\sigma) = R_{max,1} \cdot \sigma^{1/4}$$

The eVTOL is modelled with $\sigma_{eVTOL} = 0.5$ m² (low-observable configuration).

#### 6.1.2 Swerling Target Fluctuation Models

Real targets fluctuate due to changing aspect angles. Swerling (1954, 1960) proposed five statistical models:

| Swerling Case | RCS distribution | Fluctuation rate |
|---|---|---|
| 0 (non-fluctuating) | Constant | — |
| I | Rayleigh ($\chi^2_2$) | Slow (scan-to-scan) |
| II | Rayleigh ($\chi^2_2$) | Fast (pulse-to-pulse) |
| III | Chi-squared ($\chi^2_4$) | Slow |
| IV | Chi-squared ($\chi^2_4$) | Fast |

For **Swerling Case I** (many equal-amplitude scatterers, scan-to-scan fluctuation — appropriate for an eVTOL with rotating blades):

$$P_d = P_{fa}^{1/(1+\text{SNR})}$$

where $P_{fa}$ is the false-alarm probability. This is the Marcum Q-function approximation for the Swerling-I case.

The SNR relationship to range:
$$\text{SNR}(R) = \text{SNR}_{max} \cdot \left(\frac{R_{max}}{R}\right)^4$$

Substituting:
$$\boxed{P_d(R) = P_{fa}^{\left[1 + \text{SNR}_{max}\cdot(R_{max}/R)^4\right]^{-1}}}$$

This is the formula implemented in `threat/detection_model.py`.

#### 6.1.3 Parametrisation

$\text{SNR}_{max}$ is set so that $P_d = 0.9$ at $R = R_{max}$:

$$0.9 = P_{fa}^{1/(1+\text{SNR}_{max})}$$
$$\text{SNR}_{max} = \frac{\ln P_{fa}}{\ln 0.9} - 1 \approx 13.15 \quad \text{(for } P_{fa} = 10^{-6}\text{)}$$

### 6.2 SAM System Threat Model

Three threat systems are modelled in the Delhi NCR theatre:

| System | Type | Freq. | $R_{max}$ | Priority |
|--------|------|-------|-----------|---------|
| S-300V | Long-range SAM | S-band (3 GHz) | 150 km | 9 |
| SA-11 Buk | Medium-range SAM | C-band (6 GHz) | 100 km | 7 |
| SA-22 Pantsir | SHORAD/SPAAG | X-band (9.4 GHz) | 120 km | 8 |

**Altitude masking**: For $h < 50$ m AGL, terrain masking reduces detection probability:
$$P_d^{masked}(h) = P_d^{free-space} \cdot \left(\frac{h}{50}\right)^2 \quad h < 50\text{ m}$$

This models the effect of terrain diffraction and radar horizon limits on near-ground targets.

### 6.3 Engagement Probability

The kill chain probability for a single engagement is:

$$P_{kill} = P_d \cdot P_e \cdot P_k$$

where:
- $P_d$ = radar detection probability (computed above)
- $P_e$ = engagement probability (weapon system availability, reaction time)
- $P_k$ = kill probability (warhead lethality vs platform vulnerability)

For the threat cost, we use $P_d$ as the dominant and most physically derivable factor.

### 6.4 Combined (Aggregated) Threat

For $N$ independent SAM systems, the probability of detection by at least one system:

$$P_{combined} = 1 - \prod_{i=1}^{N}(1 - P_{d,i})$$

This is the **probabilistic OR aggregation** — used in `threat/threat_aggregator.py`.

### 6.5 Threat Cost Function

$$C_{threat}(x) = P_{combined}(x) \in [0, 1]$$

**Current theatre analysis**: The entire 55×60 km theatre lies within the engagement envelopes of all three SAM systems. The maximum range from theatre edge to SAM is:
- S-300V to NW corner: 29.6 km ≪ 150 km → $P_d > 0.999$
- SA-11 to SW corner: ~38 km ≪ 100 km → $P_d > 0.999$

This means $C_{threat} \approx 1$ everywhere — the planning implication is that threat avoidance must use the terrain masking effect (NOE — Nap-of-Earth flight) rather than geographical separation.

---

## 7. Sensor Fusion & Cost Aggregation

### 7.1 Weighted Linear Fusion

The total traversability cost is a weighted linear combination:

$$C_{total}(\mathbf{x}) = \sum_{i} w_i \cdot C_i(\mathbf{x})$$

with constraints $w_i \geq 0$, $\sum_i w_i = 1$. This is a **multi-attribute utility theory** formulation where each cost component is normalised to $[0,1]$.

**Weights used** (reflecting mission priority):

| Component | Weight | Rationale |
|-----------|--------|-----------|
| $C_{terrain}$ | 0.15 | Relatively low — theatre is flat |
| $C_{wind}$ | 0.10 | Moderate — energy impact |
| $C_{obstacle}$ | 0.20 | Urban area — significant hazard |
| $C_{threat}$ | 0.40 | Dominant — primary mission constraint |
| $C_{energy}$ | 0.15 | Secondary operational constraint |

### 7.2 Conditional Value at Risk (CVaR)

For risk-averse planning, the weighted sum can be replaced by:

$$\text{CVaR}_\alpha(C) = E[C \mid C \geq \text{VaR}_\alpha(C)]$$

where $\text{VaR}_\alpha$ is the $\alpha$-quantile of the cost distribution. For $\alpha = 0.95$, this captures the expected cost in the worst 5% of cells along a path, producing more conservative route choices.

### 7.3 Survival Probability

For path-level planning, the route survivability is:

$$P_{survival}(\pi) = \prod_{j=1}^{N_{wp}} (1 - P_{kill}(\mathbf{x}_j))$$

For a trajectory through $N_{wp}$ waypoints. The planning objective is to maximise $P_{survival}$.

### 7.4 Gradient for Optimisation

The gradient of $C_{total}$ w.r.t. position is required by gradient-based planners:

$$\nabla C_{total} = \sum_i w_i \nabla C_i$$

Analytical gradients are available for the terrain and obstacle components. Wind and threat components use central finite differences (2nd-order accurate):

$$\frac{\partial C}{\partial x_k} \approx \frac{C(x + \delta e_k) - C(x - \delta e_k)}{2\delta}$$

with $\delta = 10^{-5}$ degrees.

### 7.5 Environment Model Architecture

The `fusion/environment_model.py` implements `EnvironmentModel` which:

1. Accepts queries in $(lat, lon, alt)$ form
2. Routes to each sub-model
3. Applies fusion method (WEIGHTED_SUM, PROBABILISTIC, MAX_COST, etc.)
4. Returns `CostResult` with cost, gradient, Hessian, and uncertainty

---

## 8. Data Provenance & APIs

### 8.1 Open-Meteo Elevation API

```
Endpoint: https://api.open-meteo.com/v1/elevation
Parameters: latitude, longitude (comma-separated batches)
Source: NASA SRTM (30 m, merged with void-filled products)
Rate limit: ~600 req/min (free tier, no API key required)
Batch size: 100 points per request (implemented with 1.5 s delay)
Cache: JSON + NPZ, TTL=∞ (terrain is static)
```

### 8.2 Open-Meteo Weather API

```
Endpoint: https://api.open-meteo.com/v1/forecast
Parameters: latitude, longitude, hourly=[wind_speed_{P}hPa, wind_direction_{P}hPa],
            forecast_hours, timezone=UTC
Sources: GFS (NOAA), ECMWF HRES, DWD ICON, Météo-France AROME
Rate limit: 600 req/min, 10,000 req/day (free tier, no key)
Cache: JSON, TTL=6h (forecast updates 4× daily)
```

### 8.3 OpenStreetMap Overpass API

```
Endpoint: https://overpass-api.de/api/interpreter
Query language: Overpass QL
Rate: 6 requests/min (polite use of public infrastructure)
Cache: JSON, TTL=86400 s (24 h)
License: OpenStreetMap data © ODbL contributors
```

### 8.4 OpenSky Network (Dynamic Obstacles)

```
Endpoint: https://opensky-network.org/api/states/all
Parameters: lamin, lamax, lomin, lomax (bounding box)
Rate limit: 100 API calls/day anonymous, 400/day registered
Data: ADS-B transponder broadcasts (ICAO 24-bit address, lat, lon, alt, velocity)
```

---

## 9. Dataset Schema & Planning Integration

### 9.1 Full Dataset Schema

The `perception_full_dataset.csv` has 1,057,714 rows × 28 columns:

| Column | Unit | Description |
|--------|------|-------------|
| `lat` | ° | WGS-84 latitude |
| `lon` | ° | WGS-84 longitude |
| `alt_m` | m MSL | Query altitude |
| `elev_m` | m MSL | SRTM surface elevation |
| `slope_deg` | ° | Terrain slope (gradient magnitude) |
| `roughness_m` | m | RMS elevation variation (3×3 kernel) |
| `surface_type` | str | flat/rolling/hilly/mountainous |
| `terrain_clearance_m` | m | alt_m − elev_m |
| `wind_u_mps` | m/s | Eastward wind (GFS/HRES) |
| `wind_v_mps` | m/s | Northward wind |
| `wind_w_mps` | m/s | Vertical wind (estimated) |
| `wind_speed_mps` | m/s | √(u²+v²) |
| `wind_dir_deg` | ° | Meteorological FROM direction |
| `turbulence_intensity` | — | Normalised shear-based TI |
| `nearest_obstacle_dist_m` | m | KDTree nearest OSM obstacle |
| `nearest_obstacle_height_m` | m | Height of nearest obstacle |
| `obstacle_type` | str | OSM category (BUILDING, POWER_PYLON, …) |
| `T1_detect_prob` | — | Swerling-I P_d for S-300V |
| `T2_detect_prob` | — | Swerling-I P_d for SA-11 |
| `T3_detect_prob` | — | Swerling-I P_d for SA-22 |
| `max_threat_prob` | — | max(T1,T2,T3) |
| `combined_threat_prob` | — | 1−Π(1−Tᵢ) |
| `terrain_cost` | — | C_terrain ∈ [0,1] |
| `wind_cost` | — | C_wind ∈ [0,1] |
| `obstacle_cost` | — | C_obstacle ∈ [0,1] |
| `threat_cost` | — | C_threat = combined_pd |
| `energy_cost` | — | C_energy ∈ [0,1] |
| `fused_cost` | — | C_total = Σᵢ wᵢCᵢ |

### 9.2 Planning Layer Integration

**RRT* edge cost** — replace Euclidean distance with:
$$c(x_1 \to x_2) = \|x_1 - x_2\|_2 \cdot \bar{C}_{fused}(x_1, x_2)$$

where $\bar{C}_{fused}$ is the mean fused cost along the edge (interpolated from the dataset).

**NSGA-III threat evaluator** — for a trajectory $\pi = \{x_j\}$:
$$\text{threat\_margin} = \min_j \text{dist}(x_j, \text{nearest SAM})$$

**Wind injection into vehicle simulation**:
```python
wind_ned = interpolate_from_dataset(lat, lon, alt,
    fields=["wind_u_mps", "wind_v_mps", "wind_w_mps"])
# NED convention: [northward, eastward, -upward]
```

---

## 10. Implementation Architecture

### 10.1 Package Structure

```
src/evtol/perception/
├── __init__.py                    # Public API, all exports
├── perception_api.py              # UnifiedPerceptionAPI — single query interface
├── sensor_fusion.py               # SensorFuser (6-state Kalman), PathThreatAnalyzer
│                                  #   TrackedThreat: state=[px,py,pz,vx,vy,vz],
│                                  #   P covariance 6×6, piecewise-white acceleration Q
├── fusion_orchestrator.py         # SensorFusionOrchestrator, ThreatMap
│                                  #   ThreatMap threat_level computed pre-construction
│                                  #   (was post-mutation bug — fixed 2026-04-15)
├── terrain/
│   ├── data_provider.py           # TerrainDataProvider (Open-Meteo SRTM)
│   ├── field_model.py             # TerrainFieldModel (interpolated grid)
│   ├── interpolator.py            # BilinearInterpolator
│   ├── output_manager.py          # TerrainOutputManager (CSV/NPZ/JSON)
│   └── visualization.py          # Contour, profile, 3D plotters
├── wind/
│   ├── data_provider.py           # WindDataProvider → OpenMeteoProvider
│   ├── field_model.py             # WindFieldModel (4D forecast grid)
│   ├── interpolator.py            # WindInterpolator (trilinear)
│   ├── output_manager.py          # WindOutputManager
│   └── visualization.py          # Quiver, rose, shear plotters
├── obstacle/
│   ├── data_provider.py           # OSMDataProvider, OpenSkyDataProvider
│   ├── obstacle_types.py          # 20+ obstacle dataclasses
│   ├── tracker.py                 # Kalman-filter track manager
│   ├── conflict_detector.py       # CPA/TTC conflict detection
│   └── geometry.py                # BoundingCylinder, KDTree, clearance
├── threat/
│   ├── detection_model.py         # RadarDetectionModel (Swerling cases)
│   ├── engagement_model.py        # EngagementModel (kill chain)
│   ├── threat_aggregator.py       # ThreatAggregator (probabilistic OR)
│   ├── threat_field.py            # ThreatField (spatial cost map)
│   └── threat_types.py            # RadarSpecification, ThreatZone etc.
└── fusion/
    ├── environment_model.py       # EnvironmentModel (full fusion)
    └── cost_functions.py          # CostFunction base + derivatives
```

### 10.2 Key Classes

#### `TerrainDataProvider`
- `.get_elevation_batch(lats, lons)` — batched SRTM query with cache
- `.get_elevation_grid(bounds, resolution_m)` — returns `ElevationGrid`

#### `WindDataProvider → OpenMeteoProvider`
- `.fetch_forecast(bounds, altitude_bands, forecast_hours)` — returns `WindForecast`
  - Shape: `[n_times, n_levels, n_lat, n_lon]`
  - `.get_level_index(alt_m)` — maps altitude to pressure level index

#### `OSMDataProvider`
- `.fetch_obstacles(bounds, categories)` — Overpass API query
- Returns list of typed `Obstacle` subclasses with `BoundingCylinder` geometry

#### `RadarDetectionModel`
- `.calculate_detection_probability(radar, range_m, alt_m, rcs, ...)` — full Swerling model
  - Computes SNR from radar range equation
  - Applies atmospheric loss
  - Computes P_d for selected Swerling case

#### `ThreatAggregator`
- `.get_aggregated_risk(lat, lon, alt)` — queries all active threats
- Aggregation methods: PROBABILISTIC_OR, MAXIMUM, WEIGHTED_SUM, BAYESIAN, DEMPSTER_SHAFER

#### `EnvironmentModel` (fusion)
- `.query(lat, lon, alt)` — returns `CostResult` with all components

### 10.3 Thread Safety

`UnifiedPerceptionAPI` uses `threading.RLock` for cache access. Multiple planning threads can query concurrently.

### 10.4 Caching Strategy

| Layer | Cache format | TTL | Key |
|-------|-------------|-----|-----|
| Terrain | NPZ (numpy) | infinite | hash(bounds, resolution) |
| Wind | JSON | 6 hours | hash(params) |
| OSM obstacles | JSON | 24 hours | hash(bounds, categories) |
| UnifiedAPI | in-memory dict | 1 second | "layer_lat_lon_alt" |

---

## 11. Limitations & Future Work

### 11.1 Current Limitations

**L1 — Temporal stationarity**: The dataset captures a single forecast snapshot. Real missions require temporal ensemble data (multiple forecast times) to model diurnal wind patterns and weather fronts.

**L2 — Threat saturation**: The Delhi NCR theatre lies entirely within all three SAM engagement envelopes. Combined $P_d \approx 1$ everywhere provides no spatial gradient for threat-avoidance planning. Resolution: expand theatre to include areas beyond SAM range, OR reduce SAM ranges to reflect specific variant characteristics.

**L3 — Terrain homogeneity**: Delhi NCR is geologically flat (Gangetic alluvial plain). All 1.057M rows have `surface_type = 'flat'` — correct physics, but zero surface-type classifier diversity. Resolution: add Himalayan foothills (Dehradun area) or Western Ghats theatre.

**L4 — Dynamic obstacles**: The current dataset is static (no OpenSky dynamic aircraft). Real-time operations require live ADS-B ingestion at 1 Hz.

**L5 — Weather phenomena**: Rain, fog, and icing are not modelled. These significantly impact radar performance (attenuation: ~0.01 dB/km at 3 GHz in heavy rain) and eVTOL rotor efficiency (icing: ~20% lift reduction).

### 11.2 Recommended Extensions

1. **Multi-theatre dataset**: Add 2–3 diverse theatres (mountainous, coastal, desert)
2. **Temporal ensemble**: 24 hours × 4 forecast runs = 4× temporal diversity
3. **SAM variant calibration**: Use SIPRI Arms Transfer Database for country-specific range parameters
4. **RF propagation model**: Replace simple $R^{-4}$ decay with terrain-masked two-ray ground reflection model
5. **RCS model improvement**: Aspect-dependent RCS(θ, φ) for eVTOL across frequency bands

---

## 12. References

### Terrain & Geodesy

[T1] Rodriguez, E., Morris, C. S., Belz, J. E. (2006). "A Global Assessment of the SRTM Performance." *Photogrammetric Engineering & Remote Sensing*, 72(3), 249–260.

[T2] Farr, T. G., et al. (2007). "The Shuttle Radar Topography Mission." *Reviews of Geophysics*, 45(2), RG2004.

[T3] WGS84 Earth Gravitational Model. National Geospatial-Intelligence Agency (NGA) Technical Report TR8350.2, 3rd Ed., 2004.

### Atmospheric Science & Wind

[W1] Hersbach, H., et al. (2020). "The ERA5 global reanalysis." *Quarterly Journal of the Royal Meteorological Society*, 146(730), 1999–2049.

[W2] WMO-No. 8 (2018). *Guide to Meteorological Instruments and Methods of Observation*. World Meteorological Organization.

[W3] Stull, R. B. (1988). *An Introduction to Boundary Layer Meteorology*. Springer, Dordrecht.

[W4] Open-Meteo (2023). *Open-Meteo API Documentation*. https://open-meteo.com/en/docs

### Obstacle & Airspace

[O1] OpenStreetMap contributors (2024). *Planet dump*. Retrieved from https://planet.openstreetmap.org. Licensed under ODbL.

[O2] Schäfer, M., Strohmeier, M., Lenders, V., Martinovic, I., Wilhelm, M. (2014). "Bringing Up OpenSky: A Large-scale ADS-B Sensor Network for Research." *IPSN 2014*.

### Radar & Threat

[R1] Skolnik, M. I. (2008). *Radar Handbook*, 3rd Ed. McGraw-Hill.

[R2] Mahafza, B. R. (2005). *Radar Systems Analysis and Design Using MATLAB*, 2nd Ed. Chapman & Hall/CRC.

[R3] Swerling, P. (1954). "Probability of Detection for Fluctuating Targets." *RAND Research Memorandum* RM-1217. (Repr. *IEEE Trans. IT*, 6(2):269–308, 1960.)

[R4] Richards, M. A., Scheer, J. A., Holm, W. A. (2010). *Principles of Modern Radar: Basic Principles*. SciTech Publishing.

[R5] Kopp, C. (2009). "SAM System Lethality Analysis." *Air Power Australia Technical Report* APA-TR-2009-0201.

### Motion Planning

[P1] Karaman, S., Frazzoli, E. (2011). "Sampling-based Algorithms for Optimal Motion Planning." *International Journal of Robotics Research*, 30(7), 846–894.

[P2] Deb, K., Jain, H. (2014). "An Evolutionary Many-Objective Optimization Algorithm Using Reference-Point-Based Nondominated Sorting Approach." *IEEE Trans. Evolutionary Computation*, 18(4), 577–601.

[P3] LaValle, S. M. (2006). *Planning Algorithms*. Cambridge University Press.

### Risk & Uncertainty

[U1] Rockafellar, R. T., Uryasev, S. (2000). "Optimization of Conditional Value-at-Risk." *Journal of Risk*, 2(3), 21–41.

[U2] Shafer, G. (1976). *A Mathematical Theory of Evidence*. Princeton University Press.

---
