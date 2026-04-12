# Control Layer — Technical Reference

**Module:** `src/evtol/control/`
**Dataset script:** `scripts/control/dataset.py`
**Output:** `outputs/control/control_dataset.parquet` (2,000 rows × 76 columns)
**Visualizations:** `visuals/control/` (15 figures)

---

## Table of Contents

1. [Overview](#1-overview)
2. [Control Architecture](#2-control-architecture)
3. [Cascade Loop Mathematics](#3-cascade-loop-mathematics)
4. [Phase Scheduler](#4-phase-scheduler)
5. [Motor Allocation](#5-motor-allocation)
6. [Plant Model (Simulation)](#6-plant-model-simulation)
7. [Dataset Generation](#7-dataset-generation)
8. [Dataset Schema](#8-dataset-schema)
9. [Performance Analysis](#9-performance-analysis)
10. [Known Limitations](#10-known-limitations)
11. [Module Reference](#11-module-reference)

---

## 1. Overview

The control layer converts planned waypoint sequences (from the planning layer) into motor thrust commands that physically drive the eVTOL vehicle. It implements a **cascaded PID architecture** running at **50 Hz** (Δt = 0.02 s), covering the full mission profile from takeoff through hover, transition, cruise, and landing.

The layer is evaluated by simulating 2,000 missions from the vehicle dataset, recording comprehensive control performance metrics for each. All simulations are closed-loop — the controller receives feedback from a simplified 6-DoF plant model and adapts in real time.

---

## 2. Control Architecture

### 2.1 Loop Topology

The cascade is organized as five nested loops from outermost (slow) to innermost (fast):

```
[Position Loop]   x_cmd, y_cmd  →  v_x_cmd, v_y_cmd       (Kp_pos = 0.80)
[Velocity Loop]   v_cmd         →  pitch_cmd, roll_cmd      (Kp_vel = 0.15)
[Altitude Loop]   z_cmd         →  climb_rate_cmd → F_coll  (Kp_alt = 1.20)
[Attitude Loop]   φ_cmd, θ_cmd  →  p_cmd, q_cmd, r_cmd     (Kp_att = 6.50)
[Rate Loop]       p/q/r_cmd     →  Mx, My, Mz               (Kp_rt  = 120.0)
```

This structure is standard in multirotor autopilot design (APM/PX4 heritage) and ensures that the inner loops settle faster than the outer loops drive them.

### 2.2 Gain Table

| Loop | Variable | Kp | Ki | Kd | Anti-windup limit |
|------|----------|----|----|----|--------------------|
| Position | x, y position | 0.80 | 0.02 | 0.20 | ±10 m·s |
| Velocity | Vx, Vy | 0.15 | 0.02 | 0.05 | ±5 m·s² / g |
| Altitude (outer) | altitude → climb rate | 1.20 | 0.08 | 0.30 | ±20 m/s |
| Altitude (inner) | climb rate → thrust | 180.0 | 12.0 | 8.0 | ±800 N·s |
| Attitude | φ, θ, ψ | 6.50 | 0.10 | 0.40 | ±2 rad/s |
| Rate | p, q, r | 120.0 | 4.0 | 5.0 | ±50 N·m·s |

### 2.3 Saturation Limits

| Signal | Minimum | Maximum | Units |
|--------|---------|---------|-------|
| Velocity command | −25 | +25 | m/s |
| Climb rate command | −8 | +8 | m/s |
| Collective thrust | 0 | 4,800 | N |
| Roll/pitch command | −30° | +30° | deg |
| Yaw rate command | −60°/s | +60°/s | deg/s |
| Body rates (p, q, r) | −60°/s | +60°/s | deg/s |

---

## 3. Cascade Loop Mathematics

### 3.1 PID Update Law

Each controller uses the standard discrete-time PID with clamped integration:

$$u(k) = K_p \cdot e(k) + K_i \cdot \sigma(k) + K_d \cdot \dot{e}(k)$$

where:

$$\sigma(k) = \text{clip}\!\left(\sigma(k-1) + e(k)\Delta t,\; -\sigma_{\max},\; +\sigma_{\max}\right)$$
$$\dot{e}(k) = \frac{e(k) - e(k-1)}{\Delta t}$$

The derivative is a simple backward difference, not a filtered derivative. For the rate loop at 50 Hz with well-damped plant dynamics, this is acceptable without additional low-pass filtering.

### 3.2 Altitude Loop (Cascaded)

**Outer loop** — altitude to climb-rate command:

$$\dot{z}_{\text{cmd}}(k) = K_{p,\text{alt}} \cdot e_z(k) + K_{i,\text{alt}} \cdot \sigma_z(k) + K_{d,\text{alt}} \cdot \dot{e}_z(k)$$

where $e_z = z_{\text{ref}} - z$ is the altitude error (positive = below reference).

**Inner loop** — climb-rate error to collective thrust:

$$F_{\text{coll}}(k) = W + K_{p,\text{cr}} \cdot e_{\dot{z}}(k) + K_{i,\text{cr}} \cdot \sigma_{\dot{z}}(k) + K_{d,\text{cr}} \cdot \dot{e}_{\dot{z}}(k)$$

The weight feedforward $W = mg = 2000\text{ N}$ cancels steady-state gravity, so the PID only has to correct deviations from hover thrust. This is a standard feedforward technique that eliminates the need for large integral terms.

### 3.3 Velocity-to-Attitude Conversion

Horizontal velocity errors are converted to attitude commands via the aerodynamic acceleration relationship. For a multirotor, the horizontal acceleration is produced by tilting the thrust vector:

$$\theta_{\text{cmd}} = \arctan\!\left(\frac{a_{x,\text{cmd}}}{g}\right), \qquad a_{x,\text{cmd}} = K_{p,v} \cdot e_{v_x} + \ldots$$

$$\phi_{\text{cmd}} = \arctan\!\left(\frac{-a_{y,\text{cmd}}}{g}\right)$$

This is the exact small-angle inversion of the thrust tilt equations, valid for $|\phi|, |\theta| \lesssim 30°$.

### 3.4 Attitude Loop

Attitude errors in Euler angles drive body-rate commands:

$$p_{\text{cmd}} = K_{p,\text{att}} \cdot (\phi_{\text{ref}} - \phi) + K_{i} \cdot \sigma_\phi + K_d \cdot \dot{e}_\phi$$

Yaw rate is commanded from the heading controller independently:

$$r_{\text{cmd}} = K_{p,\psi} \cdot \Delta\psi$$

where $\Delta\psi$ is the wrapped heading error (range $(-\pi, \pi]$).

### 3.5 Rate Loop

Body rate errors drive moment commands:

$$M_x = K_{p,\text{rt}} \cdot (p_{\text{cmd}} - p) + K_i \cdot \sigma_p + K_d \cdot \dot{e}_p$$

and similarly for $M_y$, $M_z$. The rate loop operates at the same 50 Hz as the outer loops in this simplified implementation. In practice (PX4, ArduPilot), the rate loop runs at 400–1000 Hz on a dedicated core.

---

## 4. Phase Scheduler

### 4.1 Phase Sequence

Each mission follows a deterministic phase schedule computed from mission parameters:

| Phase | Duration | Nacelle | Velocity Ref | Altitude Ref |
|-------|----------|---------|--------------|--------------|
| TAKEOFF | min(15 s, 5% of mission time) | 90° (vertical) | 0 m/s | Ramp 0→cruise_alt |
| HOVER | 30% of planned hover time | 90° | 0 m/s | cruise_alt |
| TRANS1 | 12 s | 45° | 50% cruise speed | cruise_alt |
| CRUISE | Remaining | 0° (horizontal) | cruise speed | cruise_alt |
| TRANS2 | 12 s | 45° | 30% cruise speed | cruise_alt |
| HOVER2 | 30% of planned hover time | 90° | 0 m/s | cruise_alt |
| LAND | 15 s | 90° | 0 m/s | Ramp cruise_alt→0 |

### 4.2 Nacelle Slew

The nacelle angle transitions between phases at a limited slew rate:

$$\dot{\alpha}_{\text{nacelle}} \leq 10 \text{ deg/s}$$

This rate limit is applied at every control step. A nacelle transition event is counted when the slew is active and the phase has changed, providing the `nacelle_transitions_n` metric.

### 4.3 Reference Generation

During TAKEOFF and LAND, the altitude reference is linearly ramped:

$$z_{\text{ref,TO}}(t) = z_{\text{cruise}} \cdot \frac{t_{\text{phase}}}{T_{\text{TO}}}, \qquad z_{\text{ref,LAND}}(t) = z_{\text{cruise}} \cdot \max\!\left(0, 1 - \frac{t_{\text{phase}}}{T_{\text{LAND}}}\right)$$

During CRUISE, velocity is ramped up over 15% of the phase duration to avoid step commands:

$$v_{x,\text{ref}}(t) = v_{\text{cruise}} \cdot \min\!\left(1, \frac{t_{\text{phase}}}{0.15 \cdot T_{\text{cruise}}}\right)$$

---

## 5. Motor Allocation

### 5.1 Mixing Matrix

The four-motor tiltrotor quad uses a standard quadrotor mixing geometry in hover (nacelles vertical). The relationship between motor thrusts and the vehicle wrench (total force and moments) is:

$$\begin{bmatrix} F_z \\ M_x \\ M_y \\ M_z \end{bmatrix} = \mathbf{B} \begin{bmatrix} T_0 \\ T_1 \\ T_2 \\ T_3 \end{bmatrix}$$

where motor layout is: 0=FL (front-left), 1=FR, 2=RL, 3=RR, and:

$$\mathbf{B} = \begin{bmatrix}
1 & 1 & 1 & 1 \\
L & -L & L & -L \\
L & L & -L & -L \\
L & -L & -L & L
\end{bmatrix}, \quad L = 0.65 \text{ m (arm length)}$$

Row 1 ($F_z$): all motors contribute equally to total lift.
Row 2 ($M_x$ / roll): FL+RL vs FR+RR differential.
Row 3 ($M_y$ / pitch): FL+FR vs RL+RR differential.
Row 4 ($M_z$ / yaw): diagonal differential (torque-based in hover; pure thrust differential here, no reaction torque modeling).

### 5.2 Control Allocation

The commanded wrench $\mathbf{w}_{\text{cmd}} = [F_z^{\text{cmd}}, M_x, M_y, M_z]^\top$ is inverted via the pseudo-inverse:

$$\mathbf{T} = \mathbf{B}^{\dagger} \mathbf{w}_{\text{cmd}}, \qquad \mathbf{B}^{\dagger} = \mathbf{B}^\top (\mathbf{B}\mathbf{B}^\top)^{-1}$$

Since $\mathbf{B}$ is square (4×4) and full-rank, $\mathbf{B}^{\dagger} = \mathbf{B}^{-1}$ exactly. The pseudo-inverse formalism is retained for generality (e.g., over-actuated hexarotor extension).

Individual motor thrusts are clamped to $[0, T_{\max}/4]$ where $T_{\max} = 4800$ N (thrust-to-weight ratio 2.4 at max power). Saturation events are counted per step.

### 5.3 Actual Wrench Recovery

After clamping, the actual wrench delivered to the vehicle is:

$$\mathbf{w}_{\text{actual}} = \mathbf{B} \cdot \text{clamp}(\mathbf{T}, 0, T_{\text{per motor}})$$

This properly accounts for saturation — the controller may command a moment it cannot achieve at thrust limits.

### 5.4 PWM Conversion

Motor thrusts are converted to PWM signals (µs) via linear mapping:

$$\text{PWM}_i = \text{PWM}_{\min} + \frac{T_i}{T_{\text{per motor}}} \cdot \Delta\text{PWM}$$

where $\text{PWM}_{\min} = 1100\text{ µs}$, $\text{PWM}_{\max} = 1900\text{ µs}$, $\Delta\text{PWM} = 800\text{ µs}$.

---

## 6. Plant Model (Simulation)

The control dataset uses a simplified 6-DoF plant for evaluation. This is **not** the full vehicle dynamics model from Layer 4 — it is a lightweight simulation sufficient to close the feedback loop and generate realistic control metrics.

### 6.1 Vehicle Parameters

| Parameter | Symbol | Value | Units |
|-----------|--------|-------|-------|
| Mass | $m$ | 203.9 | kg |
| Weight | $W = mg$ | 2000 | N |
| Max thrust | $T_{\max}$ | 4800 | N |
| Arm length | $L$ | 0.65 | m |
| Roll inertia | $I_{xx}$ | 12.0 | kg·m² |
| Pitch inertia | $I_{yy}$ | 18.0 | kg·m² |
| Yaw inertia | $I_{zz}$ | 22.0 | kg·m² |
| Linear drag | $k_d$ | 2.5 | N·s/m |

### 6.2 Translational Dynamics

The rotation from body frame to NED world frame uses ZYX Euler angles. The thrust force in world frame is:

$$F_{z,\text{world}} = T_{\text{coll}} \cos\phi \cos\theta$$
$$F_{x,\text{world}} = -T_{\text{coll}} \sin\theta - k_d v_x$$
$$F_y{,\text{world}} = T_{\text{coll}} \sin\phi \cos\theta - k_d v_y$$

This is the standard "tilt-then-thrust" decomposition valid for small-to-moderate bank angles ($|\phi|, |\theta| \lesssim 45°$). The linear drag model $-k_d v$ approximates form drag; a more accurate aerodynamic model is used in Layer 4.

Translational integration (Euler forward):

$$\mathbf{v}(k+1) = \mathbf{v}(k) + \frac{1}{m}\!\left(\mathbf{F}_{\text{world}} - W\hat{z}\right)\Delta t$$
$$\mathbf{p}(k+1) = \mathbf{p}(k) + \mathbf{v}(k)\Delta t$$

Ground constraint: $p_z(k) = \max(0, p_z(k))$ (vehicle cannot go below ground).

### 6.3 Rotational Dynamics

Newton-Euler equations with gyroscopic cross-coupling:

$$\dot{p} = \frac{M_x - (I_{zz} - I_{yy}) q r}{I_{xx}}$$
$$\dot{q} = \frac{M_y - (I_{xx} - I_{zz}) p r}{I_{yy}}$$
$$\dot{r} = \frac{M_z - (I_{yy} - I_{xx}) p q}{I_{zz}}$$

Note: the gyroscopic terms $(I_{zz} - I_{yy})qr$ couple the three rotation axes. For a symmetric quadrotor ($I_{xx} \approx I_{yy}$), these terms are small but non-zero and improve simulation fidelity.

### 6.4 Euler Kinematics

The relationship between body rates and Euler angle rates (ZYX convention):

$$\dot{\phi} = p + \tan\theta (q\sin\phi + r\cos\phi)$$
$$\dot{\theta} = q\cos\phi - r\sin\phi$$
$$\dot{\psi} = \frac{q\sin\phi + r\cos\phi}{\cos\theta}$$

The $\cos\theta$ denominator is clamped at $10^{-4}$ to avoid singularity at $\theta = \pm 90°$ (gimbal lock). Euler angle integration is via Euler forward. For the 0.02 s time step and the attitude rates encountered in hover/cruise ($|p|, |q| \lesssim 0.5$ rad/s), Euler forward is sufficiently accurate; a Runge-Kutta integrator would be preferred for large attitude maneuvers.

---

## 7. Dataset Generation

### 7.1 Simulation Loop

```python
for each mission in vehicle_dataset:
    1. Build phase schedule (7 phases, durations from mission parameters)
    2. Initialize state: p = v = 0, φ = θ = ψ = 0, ω = 0
    3. Initialize 12 PID states (one per control channel)
    4. Initialize online Welford accumulators (no list storage)
    5. for step = 0 to N_steps (= mission_time / 0.02):
        a. Advance phase, slew nacelle
        b. Generate references (alt_ref, vx_ref, vy_ref)
        c. Cascade: position → velocity → altitude → attitude → rate → moments
        d. Allocate motors via B^†, clamp to [0, T_per]
        e. Integrate plant (translational + rotational + kinematics)
        f. Accumulate Welford statistics (no Python list appends)
        g. Check abort: pz < 5 m during non-land phase
    6. Compute final aggregates from accumulators
    7. Append record to output
```

### 7.2 Performance Optimization

The simulation is optimized for speed using two techniques:

**Online Welford accumulators:** Instead of storing per-step arrays and computing statistics post-hoc, all means, RMS values, and standard deviations are computed using Welford's online algorithm:

$$\mu_n = \mu_{n-1} + \frac{x_n - \mu_{n-1}}{n}, \qquad M_{2,n} = M_{2,n-1} + (x_n - \mu_{n-1})(x_n - \mu_n)$$
$$\sigma^2 = M_{2,N} / N$$

This eliminates O(N) memory allocation per mission and reduces cache pressure substantially.

**Scalar math:** All scalar clip/trig operations use Python's built-in `math` module and `if/elif` branches rather than `numpy.clip` and `numpy.cos`. This eliminates numpy dispatch overhead (which adds ~2 µs per call for scalar inputs), yielding a **6.5× end-to-end speedup** (2.4 s/mission → 0.37 s/mission).

### 7.3 ITAE Metrics

Integral of Time-weighted Absolute Error (ITAE) — a classical control quality criterion that penalizes errors that persist for long times:

$$\text{ITAE}_{\text{pos}} = \int_0^T t \cdot |e_{\text{pos}}(t)| \, dt \approx \sum_k k \cdot \Delta t \cdot |e_k| \cdot \Delta t$$

Lower ITAE indicates faster settling and lower sustained error. The time-weighting discounts early transient errors (which are unavoidable at phase starts) and emphasizes steady-state performance.

### 7.4 Energy Model

Power is estimated via momentum theory:

$$P = T^{3/2} / \sqrt{2 \rho A_{\text{disk}}}$$

where $A_{\text{disk}} = 4 \times \pi (0.3)^2 = 1.131 \text{ m}^2$ (four 0.3 m radius rotors) and $\rho = 1.225 \text{ kg/m}^3$ (ISA sea level). This is the classic actuator disk result for ideal hover power. Energy is then:

$$E = P \cdot T_{\text{mission}} / 3600 \quad [\text{Wh}]$$

Note: this uses mean thrust over the mission, not instantaneous thrust. In cruise, actual power is lower (wing lift assists), so this is a conservative (upper-bound) estimate for non-hovering flight.

---

## 8. Dataset Schema

### 8.1 Complete Column List (76 columns)

**Mission Descriptors**

| Column | Type | Units | Description |
|--------|------|-------|-------------|
| `path_length_m` | float | m | Total path arc length |
| `mission_time_s` | float | s | Total mission duration |
| `cruise_speed_ref_ms` | float | m/s | Commanded cruise speed |
| `cruise_altitude_ref_m` | float | m | Commanded cruise altitude |
| `risk_label` | int | {0,1} | Low (0) / high (1) risk |
| `n_waypoints` | int | — | Waypoint count |
| `feasible` | int | {0,1} | Planning feasibility flag |

**Phase Fractions**

| Column | Type | Description |
|--------|------|-------------|
| `hover_frac_ctrl` | float | Fraction of steps in hover phases |
| `transition_frac_ctrl` | float | Fraction in transition phases |
| `cruise_frac_ctrl` | float | Fraction in cruise phase |

**Position Tracking**

| Column | Units | Description |
|--------|-------|-------------|
| `pos_error_mean_m` | m | Mean position error (⚠ see §10.1) |
| `pos_error_max_m` | m | Maximum position error |
| `pos_error_rms_m` | m | RMS position error |
| `pos_error_final_m` | m | Position error at mission end |

**Velocity Tracking**

| Column | Units | Description |
|--------|-------|-------------|
| `vel_error_mean_ms` | m/s | Mean speed tracking error |
| `vel_error_max_ms` | m/s | Peak speed tracking error |
| `vel_error_rms_ms` | m/s | RMS speed tracking error |

**Altitude Tracking**

| Column | Units | Description |
|--------|-------|-------------|
| `alt_error_mean_m` | m | Mean altitude error |
| `alt_error_max_m` | m | Peak altitude error |
| `alt_error_rms_m` | m | RMS altitude error |
| `alt_final_m` | m | Final altitude (should ≈ 0 after LAND) |

**Attitude Tracking**

| Column | Units | Description |
|--------|-------|-------------|
| `att_error_mean_rad` | rad | Mean attitude error $\sqrt{\phi_e^2 + \theta_e^2}$ |
| `att_error_max_rad` | rad | Peak attitude error |
| `att_error_rms_rad` | rad | RMS attitude error |

**Rate Tracking**

| Column | Units | Description |
|--------|-------|-------------|
| `rate_error_mean_rads` | rad/s | Mean body rate error $\|\mathbf{e}_\omega\|$ |
| `rate_error_max_rads` | rad/s | Peak rate error |

**ITAE Metrics**

| Column | Description |
|--------|-------------|
| `itae_pos` | ITAE of position error (lower = better) |
| `itae_alt` | ITAE of altitude error |
| `itae_att` | ITAE of attitude error |

**Thrust Commands**

| Column | Units | Description |
|--------|-------|-------------|
| `thrust_cmd_mean_N` | N | Mean collective thrust |
| `thrust_cmd_max_N` | N | Peak thrust command |
| `thrust_cmd_std_N` | N | Thrust variability (std) |
| `thrust_rate_std_N` | N | Step-to-step thrust rate variability |

**Moment Commands**

| Column | Units | Description |
|--------|-------|-------------|
| `moment_x/y/z_mean_Nm` | N·m | Mean roll/pitch/yaw moments |
| `moment_x/y/z_std_Nm` | N·m | Moment variability |

**Attitude Commands**

| Column | Units | Description |
|--------|-------|-------------|
| `roll_cmd_max_rad` | rad | Maximum commanded roll |
| `pitch_cmd_max_rad` | rad | Maximum commanded pitch |
| `roll/pitch_cmd_std_rad` | rad | Roll/pitch command variability |

**PID Integral States**

| Column | Description |
|--------|-------------|
| `pid_int_vel/alt/att/rate_final` | PID integral value at mission end; large values indicate steady-state error or slow disturbance rejection |

**Motor Allocation**

| Column | Units | Description |
|--------|-------|-------------|
| `motor_T_mean_N` | N | Mean thrust per motor (average) |
| `motor_T_max_N` | N | Maximum per-motor thrust |
| `motor_T_balance_N` | N | Std across 4 motor means (balance metric) |
| `motor_T_m0–3_mean_N` | N | Per-motor mean (FL, FR, RL, RR) |

**PWM Signals**

| Column | Units | Description |
|--------|-------|-------------|
| `pwm_mean/max/min_us` | µs | PWM statistics across mission |
| `pwm_utilisation_pct` | % | (PWM − 1100) / 800 × 100 mean |

**Nacelle**

| Column | Description |
|--------|-------------|
| `nacelle_transitions_n` | Number of nacelle slew events |
| `nacelle_final_deg` | Final nacelle angle (should = 90° after landing) |
| `nacelle_mean_deg` | Mission-average nacelle angle |

**Mode Transitions & Settling**

| Column | Description |
|--------|-------------|
| `n_mode_transitions` | Number of phase advances |
| `settling_time_mean_s` | Mean time to settle after phase change (threshold: alt_e < 2 m AND vel_e < 1 m/s) |
| `settling_time_max_s` | Worst-case settling time (capped at 20 s) |

**Safety & Robustness**

| Column | Description |
|--------|-------------|
| `n_stall_events` | Steps where $|\phi| > 45°$ or $|\theta| > 45°$ |
| `n_saturations` | Steps where any motor thrust was clamped |
| `n_wp_reached` | Estimated waypoints reached |
| `mission_abort` | 1 if vehicle crashed (altitude < 5 m outside landing) |

**Performance & Energy**

| Column | Units | Description |
|--------|-------|-------------|
| `cruise_speed_actual_ms` | m/s | Estimated achieved cruise speed (95% of ref) |
| `speed_variance_ms` | m/s | Velocity tracking variability |
| `soc_initial/final` | — | Battery state of charge [0,1] |
| `energy_consumed_wh` | Wh | Mission energy (momentum theory) |

**Signatures (from vehicle layer)**

| Column | Units | Description |
|--------|-------|-------------|
| `spl_hover_a_dB` | dB(A) | A-weighted hover SPL |
| `rcs_cruise_x_dBsm` | dBsm | X-band RCS in cruise, frontal aspect |
| `max_combined_threat` | — | Maximum threat probability encountered |

---

## 9. Performance Analysis

### 9.1 Summary Statistics (2,000-mission fleet)

| Metric | Mean | Std | Min | Max |
|--------|------|-----|-----|-----|
| Altitude error (mean) | 10.37 m | 5.45 m | 0.12 m | 24.54 m |
| Attitude error (mean) | 0.0013 rad | 0.0004 rad | 0.0004 rad | 0.003 rad |
| Thrust mean | 2,613 N | 20 N | 2,539 N | 2,702 N |
| Energy consumed | 5,903 Wh | 1,906 Wh | 2,211 Wh | 14,870 Wh |
| SOC final | 0.752 | 0.048 | 0.628 | 0.945 |
| Mission abort rate | 0.0% | — | — | — |
| Motor saturations | 0.51 | 1.2 | 0 | 15 |

### 9.2 Physical Interpretation

**Altitude error (10.4 m):** A PID altitude controller with realistic turbulence and wind would typically achieve 1–3 m. The higher value here is partly because the climbing/descending phases contribute large transient errors that are not excluded from the mean. During steady cruise, the error is lower.

**Attitude error (0.0013 rad = 0.07°):** Excellent. This corresponds to sub-0.1° attitude tracking, comparable to AHRS-equipped autopilots. The inner rate loop effectively suppresses attitude disturbances.

**Thrust (2,613 N vs 2,000 N hover weight):** The 30% excess (613 N) is primarily the altitude controller integral term compensating for altitude error during climbing phases. In steady hover, thrust converges to 2,000 N.

**Zero aborts:** The controller maintains altitude above the 5 m abort threshold throughout all 2,000 missions, confirming adequate stability margins.

---

## 10. Known Limitations

### 10.1 Position Error Reference Signal

The position error metric uses a reference that accumulates over the entire mission:

```
pos_error = |(v_ref × t − px)|
```

This means the position reference is the distance a vehicle traveling at cruise speed since t=0 would have covered — not the current waypoint position. The resulting values (mean ~51 km) are meaningless as a tracking metric and should **not** be interpreted as waypoint tracking error. Use `alt_error_mean_m` and `att_error_mean_rad` for controller quality assessment.

**Recommendation for future work:** Reset the position reference at each phase transition and use waypoint-relative error within each phase.

### 10.2 Settling Time Ceiling

The settling criterion requires `alt_e < 2.0 m AND vel_e < 1.0 m/s` simultaneously. During phase transitions (especially TAKEOFF → HOVER and TRANS → CRUISE), transient velocity errors can persist for >20 s even when altitude is tracking well. The 20 s ceiling is reached for most missions, making `settling_time_mean_s` uninformative for controller comparison.

**Recommendation:** Use separate settling criteria per metric (altitude settling independently of velocity settling).

### 10.3 No Aerodynamic Cruise Model

During cruise, the plant model applies rotor thrust even though a tiltrotor would have wing lift assisting. The result is that cruise thrust is overestimated, leading to overestimated energy consumption during cruise phases.

### 10.4 Euler Forward Integration

The 0.02 s Euler forward integration accumulates integration error over long missions. For a 600 s mission at 50 Hz (30,000 steps), attitude state errors can accumulate to ~0.01 rad. A 4th-order Runge-Kutta or Heun's method would eliminate this.

### 10.5 No Sensor Noise or Communication Delays

The simulation uses perfect state feedback (no noise, no latency). Real autopilot systems experience AHRS noise (~0.5° attitude), GPS noise (~2 m position), and communication delays (10–50 ms). Adding these would reduce controller performance and provide more realistic training data for robust control research.

---

## 11. Module Reference

### src/evtol/control/

| File | Purpose |
|------|---------|
| `controller_base.py` | Base classes, dataclasses for commands (AttitudeCommand, MomentCommand) |
| `cascaded_control.py` | Full cascaded PID with Lyapunov-based inner loop, gyroscopic feedforward |
| `flight_controller.py` | High-level flight controller integrating all modes |
| `sitl_simulator.py` | Software-in-the-loop simulation interface |
| `motor_controller.py` | Motor speed controller with ESC interface |
| `motor_interface.py` | Hardware abstraction for motor commands |

### src/evtol/control/outer_loop/

| File | Purpose |
|------|---------|
| `altitude_controller.py` | Cascaded altitude-to-climb-rate-to-thrust PID |
| `heading_controller.py` | Yaw rate command from heading error |
| `position_controller.py` | Position error to velocity command |
| `velocity_controller.py` | Velocity error to attitude command (atan2 conversion) |

### src/evtol/control/inner_loop/

| File | Purpose |
|------|---------|
| `attitude_controller.py` | Euler angle error to body rate command |
| `rate_controller.py` | Body rate error to moment command |

### src/evtol/control/modes/

| File | Purpose |
|------|---------|
| `hover_mode.py` | Multicopter-style hover: altitude + attitude |
| `cruise_mode.py` | Fixed-wing-style cruise: pitch for speed, roll for track |
| `transition_mode.py` | Blended hover/cruise with nacelle interpolation |
| `flight_mode.py` | Mode manager: selects active mode from vehicle state |

### src/evtol/control/allocation/

| File | Purpose |
|------|---------|
| `control_mixer.py` | Mixing matrix B and pseudo-inverse B†; thrust allocation |
| `nacelle_scheduler.py` | Nacelle angle commands by flight phase |
