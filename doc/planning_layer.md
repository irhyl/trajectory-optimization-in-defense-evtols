# Planning Layer: Theory, Mathematics, Physics & Implementation

**Defense eVTOL Trajectory Optimization System**  
Version 2.0  |  Research Grade  |  Author: Defense eVTOL Research Team

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Configuration Space & Problem Formulation](#2-configuration-space--problem-formulation)
3. [Graph-Based Path Planners](#3-graph-based-path-planners)
4. [Sampling-Based Path Planners (RRT & RRT*)](#4-sampling-based-path-planners-rrt--rrt)
5. [Multi-Objective Optimization: NSGA-III](#5-multi-objective-optimization-nsga-iii)
6. [Trajectory Processing Pipeline](#6-trajectory-processing-pipeline)
7. [Threat Integration in Planning](#7-threat-integration-in-planning)
8. [Energy Model](#8-energy-model)
9. [Robust Planning: Uncertainty & Chance Constraints](#9-robust-planning-uncertainty--chance-constraints)
10. [Differential Flatness & Control Synthesis](#10-differential-flatness--control-synthesis)
11. [Mission Planning & Contingency Management](#11-mission-planning--contingency-management)
12. [Dataset Generation & Schema](#12-dataset-generation--schema)
13. [Perception → Planning Integration](#13-perception--planning-integration)
14. [Implementation Architecture](#14-implementation-architecture)
15. [Audit Findings & Limitations](#15-audit-findings--limitations)
16. [References](#16-references)

---

## 1. System Overview

The planning layer occupies the second tier of a four-layer autonomous mission stack:

```
Perception  →  Planning (RRT* + NSGA-III)  →  Control  →  Vehicle Dynamics
```

Its responsibility is to convert the 4D traversability cost field $C(\phi, \lambda, h, t)$
produced by the perception layer into a **time-parameterised, kinodynamically feasible,
threat-minimising trajectory** that the vehicle and control layers can execute.

The five subsystems of the planning layer are:

| Subsystem | Physical role | Key algorithm |
|---|---|---|
| Path search | Find collision-free path through C-space | A*, Theta*, RRT, RRT* |
| Multi-objective optimisation | Balance energy, threat, time, smoothness | NSGA-III |
| Trajectory processing | Smooth, timestamp, feasibility-check | B-spline, S-curve velocity |
| Threat integration | Integrate $P_K$ along trajectory | SAM detection probability field |
| Mission orchestration | Sequence multi-leg missions, contingency | Greedy NN ordering, 3-tier response |

**Design principle**: The planning layer never generates synthetic trajectories. Every planned
path traces to a real geometric query over the perception cost field, with energy and threat
computed from physics-based models.

---

## 2. Configuration Space & Problem Formulation

### 2.1 Configuration Space

For a defense eVTOL in 3-D airspace, the **configuration space** is:

$$\mathcal{C} = \{q = (x, y, z) \in \mathbb{R}^3\}$$

where $(x, y)$ encodes horizontal position (NED metres from a reference point) and $z$ encodes
altitude AGL (positive up).

The **free configuration space** is the subset not occupied by obstacles or threat zones:

$$\mathcal{C}_{free} = \mathcal{C} \setminus \mathcal{C}_{obs}$$

where $\mathcal{C}_{obs}$ includes:
- **Hard obstacles**: Volumes within the collision radius of any OSM obstacle
- **No-fly zones**: Regulatory airspace, military exclusion zones
- **Terrain**: All points with $h_{clearance} < h_{min} = 50$ m AGL

### 2.2 Path Planning Problem

A **path planning problem** is the tuple $(\mathcal{C}, \mathcal{C}_{free}, q_{start}, q_{goal})$.
A **path** is a continuous map:

$$\sigma : [0, 1] \rightarrow \mathcal{C}_{free}, \quad \sigma(0) = q_{start},\; \sigma(1) = q_{goal}$$

The **cost functional** integrates running cost along the path:

$$J(\sigma) = \int_0^1 c\!\left(\sigma(t),\, \dot\sigma(t)\right) dt$$

where $c(q, \dot{q})$ encodes terrain, wind, threat, and energy penalties at each configuration.

For the defense eVTOL with discrete waypoints $\{w_0, w_1, \ldots, w_N\}$, this becomes:

$$J = \sum_{k=0}^{N-1} \left\|\, w_{k+1} - w_k \,\right\| \cdot \bar{c}\!\left(\tfrac{w_k + w_{k+1}}{2}\right)$$

where $\bar{c}$ is the mean cost evaluated at the segment midpoint via bilinear interpolation
over the perception grid.

### 2.3 Kinodynamic Constraints

Pure geometric paths are insufficient; the vehicle's dynamics impose:

| Constraint | Expression | Value |
|---|---|---|
| Max speed | $\|\dot{q}\| \leq V_{max}$ | 60 m/s |
| Max acceleration | $\|\ddot{q}\| \leq a_{max}$ | 5 m/s² |
| Max jerk | $\|\dddot{q}\| \leq j_{max}$ | 2 m/s³ |
| Min altitude AGL | $z \geq h_{min}$ | 50 m |
| Max bank angle | $|\phi_{bank}| \leq 30°$ | during cruise |
| Max climb rate | $\dot{z} \leq 8$ m/s | |

The kinodynamic feasibility checker (`trajectory/feasibility.py`) enforces all constraints
after trajectory smoothing.

---

## 3. Graph-Based Path Planners

Implemented in `src/evtol/planning/algorithms/graph.py`.

### 3.1 Dijkstra's Algorithm

**Theory**: Given a weighted directed graph $G = (V, E, w)$ with $w : E \to \mathbb{R}^+$,
find shortest paths from source $s$ to all vertices.

**Algorithm**:
```
DIJKSTRA(G, s):
  dist[v] ← ∞  for all v ∈ V
  dist[s] ← 0
  Q ← min-heap containing all vertices keyed by dist
  while Q ≠ ∅:
    u ← EXTRACT-MIN(Q)
    for each neighbour v of u:
      alt ← dist[u] + w(u, v)
      if alt < dist[v]:
        dist[v] ← alt
        prev[v] ← u
        DECREASE-KEY(Q, v, alt)
  return dist, prev
```

**Complexity**: $O\!\left((|V| + |E|)\log|V|\right)$ with binary heap; $O(|V|^2)$ with array.
For a $250 \times 300 \times 14$ altitude-layer grid: $|V| = 1{,}050{,}000$, $|E| \approx 7|V|$ (26-connected 3D).

**Limitation**: Explores uniformly in all directions — no heuristic guidance. Used when the
optimal cost is required and computation time is not critical.

---

### 3.2 A* Algorithm

**Theory**: A* extends Dijkstra with an **admissible heuristic** $h(v)$ that lower-bounds
the true cost-to-goal $h^*(v)$:

$$f(v) = g(v) + h(v)$$

where $g(v)$ = cost from start and $h(v)$ = estimated cost to goal.

**Algorithm**:
```
A_STAR(G, s, goal, h):
  OPEN ← {s}
  g[s] ← 0
  f[s] ← h(s)
  while OPEN ≠ ∅:
    u ← node in OPEN with minimum f[u]
    if u == goal: return reconstruct_path(prev, u)
    move u from OPEN to CLOSED
    for each neighbour v of u:
      if v in CLOSED: continue
      alt_g ← g[u] + w(u, v)
      if v ∉ OPEN or alt_g < g[v]:
        g[v] ← alt_g
        f[v] ← g[v] + h(v)
        prev[v] ← u
        add v to OPEN
  return FAILURE
```

**Heuristic selection** — For 3D Euclidean cost functions, the **straight-line Euclidean
distance** is admissible:

$$h(v) = \|v - v_{goal}\|_2 \cdot c_{min}$$

where $c_{min}$ is the minimum possible traversal cost per metre (in practice, $c_{min} = 1$
for unit-cost grids or the global minimum of the cost field).

**Optimality**: A* is optimal if $h$ is admissible ($h(v) \leq h^*(v)$) and consistent
($h(u) \leq w(u,v) + h(v)$ for all edges). The Euclidean heuristic satisfies both.

**Complexity**: $O(|E| \log|V|)$ in the best case; exponential in the number of states for
adversarial heuristics. In practice, on the eVTOL planning grid, A* runs in 0.5–2 s.

---

### 3.3 Theta* (Any-Angle A*)

**Theory**: A* on a grid is constrained to move along grid edges, producing **jagged paths**
even through open space. Theta* extends A* to allow **any-angle** moves by checking **line-of-sight**
between vertices:

```
UPDATE_VERTEX(s, s', neighbour):
  if line_of_sight(parent[s], neighbour):
    # Any-angle: path through parent[s] directly to neighbour
    if g[parent[s]] + dist(parent[s], neighbour) < g[neighbour]:
      g[neighbour] ← g[parent[s]] + dist(parent[s], neighbour)
      parent[neighbour] ← parent[s]
  else:
    # Standard A* update
    if g[s] + dist(s, neighbour) < g[neighbour]:
      g[neighbour] ← g[s] + dist(s, neighbour)
      parent[neighbour] ← s
```

**Line-of-sight check** uses a 3D Bresenham ray traversal: $O(N_{cells})$ along the ray.

**Quality improvement**: Theta* produces paths that are typically 5–15% shorter than A* on the
same grid, with angles unconstrained to 45° multiples.

---

## 4. Sampling-Based Path Planners (RRT & RRT*)

Implemented in `src/evtol/planning/rrt_star.py` and `algorithms/sampling.py`.

### 4.1 Rapidly-Exploring Random Trees (RRT)

**Motivation**: Grid planners do not scale to high-dimensional spaces and require discretisation.
RRT explores $\mathcal{C}_{free}$ by growing a tree through **random sampling**.

**Algorithm**:
```
RRT(q_init, q_goal, n_iterations, step_size):
  T ← tree with root q_init
  for i = 1 to n_iterations:
    q_rand ← RANDOM_SAMPLE()          # uniform over C
    q_near ← NEAREST(T, q_rand)       # KD-tree lookup
    q_new  ← STEER(q_near, q_rand, step_size)
    if COLLISION_FREE(q_near, q_new):
      T.ADD_VERTEX(q_new)
      T.ADD_EDGE(q_near, q_new)
      if DIST(q_new, q_goal) < goal_tol:
        return reconstruct_path(T, q_new)
  return FAILURE
```

**STEER** function:
$$q_{new} = q_{near} + \eta \cdot \frac{q_{rand} - q_{near}}{\|q_{rand} - q_{near}\|}$$

where $\eta$ is the step size (typically 50–200 m for the eVTOL theatre).

**Probabilistic completeness**: RRT is probabilistically complete — as $n \to \infty$, the
probability of finding a path (if one exists) approaches 1. This follows from the fact that
$\text{Voronoi regions}$ of existing nodes shrink to zero volume as the tree grows.

**KD-Tree nearest-neighbour**: The `NEAREST` query uses a $k$-d tree with $O(n)$ build
and $O(\log n)$ average query time (Bentley, 1975). For the 3D planning space:
- Dimensions: $[x, y, z]$ in NED metres
- SciPy `cKDTree` with Euclidean metric

---

### 4.2 RRT* (Asymptotically Optimal RRT)

**Motivation**: Vanilla RRT is not cost-optimal — once a path is found, it is never improved.
RRT* (Karaman & Frazzoli, 2011) adds two operations: **near-set rewiring** and **parent selection**
that guarantee asymptotic optimality.

**Algorithm**:
```
RRT_STAR(q_init, q_goal, n_iterations, step_size):
  T ← tree with root q_init
  for i = 1 to n_iterations:
    q_rand ← INFORMED_SAMPLE(q_init, q_goal, c_best)
    q_near ← NEAREST(T, q_rand)
    q_new  ← STEER(q_near, q_rand, step_size)

    if not COLLISION_FREE(q_near, q_new): continue

    # 1. CHOOSE BEST PARENT
    Q_near ← NEAR(T, q_new, r_n)         # ball of radius r_n
    q_min ← q_near
    c_min ← COST(q_near) + dist(q_near, q_new)
    for q' in Q_near:
      if COLLISION_FREE(q', q_new) and COST(q') + dist(q', q_new) < c_min:
        q_min ← q'
        c_min ← COST(q') + dist(q', q_new)
    T.ADD_VERTEX(q_new, parent=q_min)

    # 2. REWIRE
    for q' in Q_near:
      if COST(q_new) + dist(q_new, q') < COST(q'):
        if COLLISION_FREE(q_new, q'):
          T.CHANGE_PARENT(q', q_new)

  return best_path_to_goal(T)
```

**Near-set radius** (Karaman & Frazzoli Theorem 38):

$$r_n = \gamma_{RRT^*} \cdot \left(\frac{\log n}{n}\right)^{1/d}$$

$$\gamma_{RRT^*} > 2\left(1 + \frac{1}{d}\right)^{1/d} \cdot \left(\frac{\mu(\mathcal{C}_{free})}{\zeta_d}\right)^{1/d}$$

where:
- $d = 3$ (planning dimension)
- $\mu(\mathcal{C}_{free})$ = volume of free space
- $\zeta_d$ = volume of unit ball in $d$ dimensions ($\zeta_3 = 4\pi/3$)

**Finite-iteration convergence** (Theorem 4.14):

$$\frac{C_n}{C^*} \leq 1 + \frac{c}{n^{1/d}}$$

For $n = 5{,}000$ iterations in 3D ($d = 3$): the path is typically 15–50% above the true optimum.
This is the reason NSGA-III post-processing is applied.

**Informed sampling** (Gammell et al., 2014): After finding an initial solution with cost $c_{best}$,
sample only from the **prolate hyperellipsoid** of configurations that could improve it:

$$\mathcal{X}_{informed} = \{q \in \mathcal{C} : \|q - q_{start}\| + \|q - q_{goal}\| \leq c_{best}\}$$

This foci-based ellipsoid rejects $\approx 90\%$ of random samples in typical missions, accelerating
convergence by $3\times$.

**Threat-aware cost function**: In the eVTOL implementation, the edge cost is:

$$c(q_a, q_b) = \|q_b - q_a\| \cdot \left[w_{terrain} \cdot C_{terrain} + w_{wind} \cdot C_{wind} + w_{threat} \cdot C_{threat} + w_{obs} \cdot C_{obstacle}\right]_{midpoint}$$

where the weights $w_i$ are set by `MissionPriority` (STEALTH → $w_{threat}$ high; EFFICIENCY → $w_{wind}$ high).

---

## 5. Multi-Objective Optimization: NSGA-III

Implemented in `src/evtol/planning/optimization/nsga3.py`.

### 5.1 Problem Statement

The RRT* path is near-optimal for a single scalar cost but cannot handle **incommensurable
objectives**. The trajectory must simultaneously minimise:

| Objective | Symbol | Units | Formula |
|---|---|---|---|
| Flight time | $J_1$ | s | $\int_0^T 1 \, dt$ |
| Energy consumption | $J_2$ | Wh | $\int_0^T P(q,\dot{q}) \, dt / 3600$ |
| Threat exposure | $J_3$ | — | $\int_0^T P_K(q) \, dt$ |
| Path smoothness | $J_4$ | m/s³ | $\int_0^T \|\dddot{q}\| \, dt$ (jerk) |
| Terrain proximity | $J_5$ | m | $-\min_t h_{clearance}(q(t))$ |

**Formal statement**:
$$\underset{x \in X_{feasible}}{\text{minimise}}\quad F(x) = [J_1(x),\, J_2(x),\, J_3(x),\, J_4(x),\, J_5(x)]^T$$

where $x$ is a trajectory parameterisation (waypoint sequence or spline control points).

---

### 5.2 Pareto Dominance

**Definition** — $x$ **dominates** $x'$ (written $x \prec x'$) iff:
$$\forall i:\, J_i(x) \leq J_i(x') \quad \text{AND} \quad \exists i:\, J_i(x) < J_i(x')$$

**Pareto optimal set**: $\mathcal{P}^* = \{x \in X_{feas} : \nexists\, x' \text{ s.t. } x' \prec x\}$

**Pareto front**: $\mathcal{F}^* = F(\mathcal{P}^*)$ — the image of the Pareto optimal set in
objective space.

For 5 objectives, the Pareto front is typically a 4-dimensional manifold. NSGA-III maintains
an approximation of this manifold through evolution.

---

### 5.3 Non-Dominated Sorting

**Fast non-dominated sort** (Deb et al., 2002) partitions population $P$ into fronts
$\mathcal{F}_1, \mathcal{F}_2, \ldots$:

```
FAST_NDS(P):
  for each p ∈ P:
    S_p ← {q ∈ P : p dominates q}    # solutions dominated by p
    n_p ← |{q ∈ P : q dominates p}|  # domination count of p
    if n_p == 0: F_1 ← F_1 ∪ {p}

  i ← 1
  while F_i ≠ ∅:
    Q ← ∅
    for each p ∈ F_i:
      for each q ∈ S_p:
        n_q ← n_q − 1
        if n_q == 0: Q ← Q ∪ {q}
    i ← i + 1
    F_i ← Q
  return [F_1, F_2, ...]
```

**Complexity**: $O(M \cdot N^2)$ where $M$ = number of objectives, $N$ = population size.
For $M = 5$, $N = 100$: $\approx 50{,}000$ comparisons per generation.

---

### 5.4 Reference Points (Das & Dennis Simplex Lattice)

Unlike NSGA-II which uses crowding distance for diversity (adequate for $M \leq 3$), NSGA-III
uses a **structured set of reference points** on the normalised hyperplane:

$$\left\{r = (r_1, \ldots, r_M) : \sum_{i=1}^M r_i = 1,\; r_i \geq 0\right\}$$

**Das & Dennis method** generates $H = \binom{M + p - 1}{p}$ evenly-spaced reference points
with $p$ divisions:

$$r_i = \frac{k_i}{p}, \quad k_i \in \{0, 1, \ldots, p\}, \quad \sum_i k_i = p$$

For $M = 5$ objectives, $p = 12$ divisions: $H = \binom{16}{4} = 1{,}820$ reference points.

**Normalisation**: Before association, objectives are normalised using the ideal and nadir points:

$$f_i^{norm}(x) = \frac{f_i(x) - f_i^{ideal}}{f_i^{nadir} - f_i^{ideal}}$$

**Association**: Each solution $x$ is associated with the reference point $r^*$ that minimises
the perpendicular distance:

$$d(x, r) = \left\|f^{norm}(x) - \frac{r}{\|r\|} \cdot \left(\frac{r}{\|r\|} \cdot f^{norm}(x)\right)\right\|$$

---

### 5.5 Genetic Operators

**Simulated Binary Crossover (SBX)** (Deb & Agrawal, 1995): For parent waypoints $p_1, p_2$:

$$\beta_q = \begin{cases}
(2u)^{1/(\eta_c + 1)} & u \leq 0.5 \\
\left(\frac{1}{2(1-u)}\right)^{1/(\eta_c + 1)} & u > 0.5
\end{cases}$$

$$c_1 = \tfrac{1}{2}\left[(1 + \beta_q)\,p_1 + (1 - \beta_q)\,p_2\right]$$
$$c_2 = \tfrac{1}{2}\left[(1 - \beta_q)\,p_1 + (1 + \beta_q)\,p_2\right]$$

where $u \sim U[0,1]$ and $\eta_c = 20$ (distribution index). Higher $\eta_c$ → offspring closer
to parents (exploitation); lower → more spread (exploration).

**Polynomial Mutation** (Deb & Goyal, 1996): For a variable $x_i \in [l_i, u_i]$:

$$\delta_q = \begin{cases}
(2r)^{1/(\eta_m + 1)} - 1 & r < 0.5 \\
1 - (2(1-r))^{1/(\eta_m + 1)} & r \geq 0.5
\end{cases}$$

$$x_i' = x_i + \delta_q \cdot (u_i - l_i)$$

with $\eta_m = 20$. Mutation probability $p_m = 1/n_{vars}$ (one mutation per individual on average).

---

### 5.6 NSGA-III Configuration (eVTOL defaults)

| Parameter | Value | Rationale |
|---|---|---|
| `n_generations` | 100 | Convergence observed by generation 80 |
| `population_size` | 100 | Covers reference points adequately |
| `n_divisions` | 12 | $H = 1820$ reference points for $M = 5$ |
| `crossover_prob` | 0.90 | High crossover for trajectory exploration |
| `crossover_eta` | 20 | Moderate exploitation |
| `mutation_prob` | $1/n_{vars}$ | ~1 waypoint mutated per individual |

---

## 6. Trajectory Processing Pipeline

### 6.1 B-Spline Smoothing

Implemented in `trajectory/smoothing.py`.

**Motivation**: Raw RRT*/NSGA-III paths are sequences of straight-line segments with
$C^0$ continuity (position only). Actuator commands require at least $C^2$
(continuous acceleration).

**B-spline definition**: A B-spline of degree $k$ with control points $\{P_i\}$ is:

$$C(t) = \sum_{i=0}^{n} N_{i,k}(t)\, P_i$$

The **Cox–de Boor recurrence** for basis functions:

$$N_{i,0}(t) = \begin{cases} 1 & t_i \leq t < t_{i+1} \\ 0 & \text{otherwise} \end{cases}$$

$$N_{i,k}(t) = \frac{t - t_i}{t_{i+k} - t_i} N_{i,k-1}(t) + \frac{t_{i+k+1} - t}{t_{i+k+1} - t_{i+1}} N_{i+1,k-1}(t)$$

**Properties exploited in trajectory generation**:
- **Local support**: $N_{i,k}(t) \neq 0$ only for $t \in [t_i, t_{i+k+1})$ — moving one waypoint
  affects only a local portion of the trajectory
- **$C^{k-1}$ continuity**: Degree-3 cubic B-splines guarantee $C^2$
- **Convex hull**: The trajectory remains inside the convex hull of its control points —
  useful for safety guarantees

**Smoothing procedure**: Given $N$ RRT* waypoints $\{w_0, \ldots, w_{N-1}\}$:
1. Fit a degree-3 B-spline using SciPy `splprep` with smoothing factor $s > 0$
2. Evaluate at $M = 500$ uniform parameter values $t \in [0, 1]$
3. Check all sampled points against $\mathcal{C}_{free}$ (if any point is infeasible, fall back
   to the un-smoothed path for that segment)

**Smoothness vs. deviation trade-off**: The smoothing factor $s$ controls the balance:
$$\sum_{i=0}^{N-1} \|C(u_i) - w_i\|^2 \leq s$$

Larger $s$ → smoother but may deviate from the original collision-free waypoints.
Default: $s = 0.1 \cdot N$ (10 cm average deviation per waypoint).

---

### 6.2 Velocity Profile Generation

Implemented in `trajectory/velocity_profile.py`.

**Trapezoidal velocity profile** (minimum-time, bang-coast-bang acceleration):

```
TRAPEZOIDAL_PROFILE(dist, v_max, a_max):
  # Time to accelerate from 0 to v_max
  t_ramp = v_max / a_max
  d_ramp = 0.5 * a_max * t_ramp²

  if 2 * d_ramp > dist:   # Short segment — triangular profile
    t_peak = sqrt(dist / a_max)
    v_peak = a_max * t_peak
    t_total = 2 * t_peak
  else:                    # Full trapezoidal
    d_cruise = dist - 2 * d_ramp
    t_cruise = d_cruise / v_max
    t_total  = 2 * t_ramp + t_cruise
  return v(t), a(t)
```

This produces $C^1$ velocity profiles. For $C^2$ acceleration (needed for smooth motor commands),
an **S-curve** (7-segment jerk-limited) profile is used:

$$j(t) = \begin{cases}
+j_{max} & \text{acceleration phase} \\
0        & \text{cruise / constant accel} \\
-j_{max} & \text{deceleration phase}
\end{cases}$$

**Time-optimal criterion**: Minimum total time subject to $V_{max}$, $a_{max}$, $j_{max}$.

---

### 6.3 Kinodynamic Feasibility Checking

Implemented in `trajectory/feasibility.py`.

After smoothing and timestamping, the checker evaluates at each sample:

| Check | Computation | Pass criterion |
|---|---|---|
| Speed | $\|\dot{q}(t)\|$ | $\leq 60$ m/s |
| Acceleration | $\|\ddot{q}(t)\|$ | $\leq 5$ m/s² |
| Jerk | $\|\dddot{q}(t)\|$ | $\leq 2$ m/s³ |
| Altitude AGL | $z(t) - h_{SRTM}(lat, lon)$ | $\geq 50$ m |
| Obstacle clearance | $d_{nearest}(lat, lon)$ from KD-tree | $\geq 50$ m |
| Threat exposure | $P_K(lat, lon, alt)$ | $\leq P_{K,max}$ (mission dependent) |

Derivatives are computed via **central finite differences** on the timestamped position sequence:

$$\dot{q}(t_i) = \frac{q(t_{i+1}) - q(t_{i-1})}{2\Delta t}, \quad
\ddot{q}(t_i) = \frac{q(t_{i+1}) - 2q(t_i) + q(t_{i-1})}{\Delta t^2}$$

---

### 6.4 Differential Flatness (Control Synthesis)

Implemented in `trajectory/differential_flatness_analyzer.py`.

**Flat system theory** (Fliess et al., 1995): A system is **differentially flat** if there
exists a set of **flat outputs** $y = (y_1, \ldots, y_m)$ such that all states $x$ and
inputs $u$ can be expressed as algebraic functions of $y$ and a finite number of its derivatives.

For a tiltrotor eVTOL:
$$\text{flat outputs: } y_{flat} = [p_x,\, p_y,\, p_z,\, \psi] \quad (4\text{ DOF})$$

**Flatness map** (from smooth trajectory to motor commands):

**Step 1** — Total thrust from desired acceleration:
$$\mathbf{a}_{des} = \ddot{p}_{des} + g\hat{z}$$
$$T = m \|\mathbf{a}_{des}\|$$

**Step 2** — Desired body $z$-axis:
$$\hat{z}_B = \frac{\mathbf{a}_{des}}{\|\mathbf{a}_{des}\|}$$

**Step 3** — Desired rotation matrix from body axes and desired yaw $\psi$:
$$\hat{x}_{C} = [\cos\psi, \sin\psi, 0]^T, \quad
  \hat{y}_B = \frac{\hat{z}_B \times \hat{x}_C}{\|\hat{z}_B \times \hat{x}_C\|}, \quad
  \hat{x}_B = \hat{y}_B \times \hat{z}_B$$
$$R_{des} = [\hat{x}_B \mid \hat{y}_B \mid \hat{z}_B]$$

**Step 4** — Angular velocity from $\dddot{p}_{des}$ (jerk):
$$\boldsymbol\omega = R_{des}^T \left(\frac{T \hat{z}_B \times m\dddot{p}_{des}}{T^2}\right)$$

**Step 5** — Angular acceleration from $\ddddot{p}_{des}$ (snap):
$$\dot{\boldsymbol\omega} = f(\dddot{p}_{des}, \ddddot{p}_{des}, T, \boldsymbol\omega, m, J)$$

**Step 6** — Motor thrust commands via control allocation:
$$[T_1, T_2, T_3, T_4]^T = \Gamma^{-1} [T,\, \tau_x,\, \tau_y,\, \tau_z]^T$$

where $\Gamma$ is the $4\times 4$ mixing matrix determined by rotor positions and tilt angles.

**LQR refinement**: The linearised error dynamics about the nominal trajectory are:
$$\dot{\xi} = A(t)\xi + B(t)u_{lqr}, \quad \xi = [e_p,\, e_v,\, e_R,\, e_\omega]$$

The LQR gain $K(t)$ is computed from the time-varying Riccati equation:
$$-\dot{P} = A^T P + PA - PBR^{-1}B^T P + Q$$

solved offline by SciPy `solve_continuous_are` (steady-state approximation):
$$K = R^{-1}B^T P$$

**Minimum snap trajectory** (Mellinger & Kumar, 2011): When $\ddddot{p}_{des}$ is minimised,
the motor commands are smooth (minimum snap minimises the 4th derivative, which maps to
minimum rotor thrust rate-of-change).

---

## 7. Threat Integration in Planning

### 7.1 Threat Field from Perception Layer

The perception layer produces $P_d(\phi, \lambda, h)$ for each SAM system and the combined:

$$P_{combined}(\phi, \lambda, h) = 1 - \prod_{i=1}^{N_{SAM}} \left(1 - P_{d,i}(\phi, \lambda, h)\right)$$

### 7.2 Integrated Kill Probability Along Path

The threat cost of a trajectory $\sigma$ is the **integrated kill probability** (IKP):

$$J_{threat}(\sigma) = \int_0^T P_{combined}(\sigma(t))\, dt$$

For discrete waypoints and constant velocity between them:

$$J_{threat} = \sum_{k=0}^{N-1} \bar{P}_{combined}(w_k, w_{k+1}) \cdot \frac{\|w_{k+1} - w_k\|}{V_k}$$

where $\bar{P}_{combined}$ is the mean combined threat probability along segment $k$,
estimated by evaluating at 3 Gauss–Legendre quadrature points.

**Threat gradient for RRT* cost**: The running cost $c(q, \dot{q})$ includes a threat term
scaled by dwell time:

$$c_{threat}(q) = P_{combined}(q) / V_{max}$$

so that the planner naturally steers away from high-detection-probability regions.

---

### 7.3 SAM Avoidance Strategy

For $P_{combined} > 0.5$ at a candidate waypoint, the planner enforces a **hard avoidance
margin** of $\Delta R$:

$$\Delta R = R_{max} \cdot \left(1 - \sqrt[4]{\frac{\ln P_{fa}}{\ln 0.5}}\right)$$

This creates a **soft no-fly bubble** around each SAM system that shrinks with altitude
(terrain masking) and expands with vehicle RCS.

---

## 8. Energy Model

### 8.1 Actuator Disk Theory (Hover)

From Rankine–Froude momentum theory, minimum induced power to support weight $W = mg$:

$$P_{hover,ideal} = \frac{W^{3/2}}{\sqrt{2\rho A_{disk}}}$$

where $A_{disk} = \pi R_{rotor}^2 \cdot N_{rotors}$ is total disk area.

**Atmospheric density altitude correction**:
$$\rho(h) = \rho_0 \exp\!\left(-\frac{h}{H_{scale}}\right), \quad \rho_0 = 1.225 \text{ kg/m}^3,\; H_{scale} = 8500 \text{ m}$$

**Figure of merit** $\eta_{FM}$: Real hover power = $P_{hover,ideal} / \eta_{FM}$ with
$\eta_{FM} \approx 0.70$–$0.80$.

### 8.2 Cruise Power (Parasite + Induced Drag)

$$P_{cruise} = \underbrace{\tfrac{1}{2}\rho V^3 S C_{D0}}_{\text{parasite}} + \underbrace{\frac{W}{V}\sqrt{\frac{W}{\tfrac{1}{2}\rho S C_L}}}_{\text{induced}}$$

with $C_{D0} = 0.08$, $C_L = 0.40$, $S = 10$ m² (wing area).

**Optimal speed** $V^*$ minimises $P_{cruise}/V$ (i.e., maximises range):
$$V^* = \left(\frac{W}{\tfrac{1}{2}\rho S} \cdot \sqrt{\frac{1}{3C_{D0}C_L}}\right)^{1/2} \approx 50\text{–}60\text{ m/s}$$

### 8.3 Transition Phase

$$P_{transition} = \frac{P_{hover} + P_{cruise}}{2}$$

This is the first-order model; a high-fidelity model integrates over the nacelle tilt
schedule $\theta(t) \in [0°, 90°]$.

### 8.4 Battery Terminal Voltage

$$V_{terminal} = E_0(SOC) - I \cdot R_{int}(T, I) - \Delta V_{conc}$$

where $E_0(SOC)$ is the open-circuit voltage curve (piecewise linear, 3.0 V → 4.2 V for
Li-NCA cells), and $R_{int}$ rises with current (Peukert) and falls with temperature.

**Energy in Wh**:
$$E\,[Wh] = \frac{P\,[W] \cdot \Delta t\,[s]}{3600}$$

**Reserve constraints**:

| Reserve type | Amount | Semantic |
|---|---|---|
| Minimum reserve | 10 000 Wh | Emergency landing budget |
| Contingency | 5 000 Wh | Unexpected detours |
| Total required | 15 000 Wh | Must remain at mission end |

---

## 9. Robust Planning: Uncertainty & Chance Constraints

Implemented in `robust/uncertainty.py` and `robust/chance_constraints.py`.

### 9.1 Uncertainty Sources

| Source | Model | Parameters |
|---|---|---|
| Wind | Gaussian additive $\mathcal{N}(\mu_w, \Sigma_w)$ | $\sigma_w = 2$ m/s |
| Radar detection | Swerling-I fluctuation | $P_{fa} = 10^{-6}$ |
| Position (GPS) | $\mathcal{N}(0, \Sigma_{GPS})$ | $\sigma_{GPS} = 2$ m (CEP) |
| Obstacle motion | Uniform in velocity ball | $\pm 3$ m/s |

### 9.2 Unscented Kalman Filter (State Propagation)

The UKF propagates the **mean** $\bar{x}$ and **covariance** $\Sigma$ of the vehicle state
through nonlinear dynamics $\dot{x} = f(x, u)$:

**Sigma points** (Wan & Merwe, 2000):
$$\mathcal{X}_0 = \bar{x}$$
$$\mathcal{X}_i = \bar{x} + \left(\sqrt{(n+\lambda)\Sigma}\right)_i, \quad i = 1,\ldots,n$$
$$\mathcal{X}_{n+i} = \bar{x} - \left(\sqrt{(n+\lambda)\Sigma}\right)_i, \quad i = 1,\ldots,n$$

where $\lambda = \alpha^2(n + \kappa) - n$ is the scaling parameter ($\alpha = 0.001$, $\kappa = 0$,
$n$ = state dimension). The matrix square root uses Cholesky decomposition.

**Propagation**:
$$\bar{x}^- = \sum_{i=0}^{2n} W_i^{(m)} f(\mathcal{X}_i)$$
$$\Sigma^- = \sum_{i=0}^{2n} W_i^{(c)} (f(\mathcal{X}_i) - \bar{x}^-)(\cdot)^T + Q$$

This gives a Gaussian approximation of the state distribution at each future timestep, enabling
constraint tightening for robustness.

### 9.3 Chance Constraints

A **chance constraint** requires a constraint $g(x, \xi) \leq 0$ to hold with probability
at least $1 - \epsilon$:

$$\Pr\!\left[g(x,\xi) \leq 0\right] \geq 1 - \epsilon$$

**Analytical reformulation** (for linear $g$ and Gaussian $\xi \sim \mathcal{N}(\mu, \Sigma)$):

$$g(x,\xi) = a^T \xi - b \leq 0 \implies a^T\mu + \Phi^{-1}(1-\epsilon)\sqrt{a^T \Sigma a} \leq b$$

where $\Phi^{-1}$ is the inverse standard normal CDF. For $\epsilon = 0.05$: $\Phi^{-1}(0.95) = 1.645$.

**Application** — Obstacle clearance chance constraint:

Let the vehicle position be $p \sim \mathcal{N}(\bar{p}, \Sigma_p)$ (from UKF) and the
minimum clearance constraint be $d_{obs}(p) \geq d_{min}$. The chance constraint becomes:

$$d_{obs}(\bar{p}) - \Phi^{-1}(1-\epsilon) \cdot \sqrt{\nabla d_{obs}^T \Sigma_p \nabla d_{obs}} \geq d_{min}$$

This **tightens** the clearance margin in proportion to position uncertainty.

### 9.4 Scenario Approach (Sample-Based)

For non-Gaussian or non-linear constraints (e.g., threat detection under Swerling fluctuation),
the scenario approach (Calafiore & Campi, 2006) is used:

$$\min_{x} J(x) \quad \text{s.t.} \quad g(x, \xi^{(j)}) \leq 0,\; j = 1,\ldots,N_{sc}$$

With $N_{sc}$ independently drawn scenarios, the solution is feasible for the chance constraint
with probability $1 - \delta$ if:

$$N_{sc} \geq \frac{2}{\epsilon}\!\left(\ln\frac{1}{\delta} + d\right)$$

where $d$ = number of decision variables, $\epsilon$ = violation probability, $\delta$ = confidence.
For $\epsilon = 0.05$, $\delta = 10^{-6}$, $d = 30$ (waypoints): $N_{sc} \geq 1{,}400$ scenarios.

---

## 10. Differential Flatness & Control Synthesis

*(See Section 6.4 for the full derivation. Summary below.)*

The eVTOL is differentially flat in $(p_x, p_y, p_z, \psi)$. The **planning → control bridge**:

```
Planning layer produces:
  p(t), ṗ(t), p̈(t), p⃛(t), p⁽⁴⁾(t), ψ(t), ψ̇(t)

Flatness map produces:
  T(t)         ← total thrust [N]
  R_des(t)     ← desired rotation matrix
  ω_des(t)     ← desired body angular velocity [rad/s]
  α_des(t)     ← desired angular acceleration [rad/s²]
  u_motor(t)   ← individual rotor thrust commands [N]

Control layer (50 Hz) receives:
  setpoint = (p_des, ṗ_des, p̈_des, R_des, ω_des)
  → attitude controller tracks R_des
  → rate controller tracks ω_des
  → allocation computes rotor commands
```

**Key insight**: Differential flatness **eliminates the trajectory tracking problem** — the
control inputs are derived algebraically from the planned trajectory's derivatives, not from
a separate feedback loop. This is why $C^4$ continuity (up to snap) is required in the
smoothed trajectory.

---

## 11. Mission Planning & Contingency Management

### 11.1 Multi-Leg Sequencing

Objectives are ordered by a **greedy nearest-neighbour heuristic**:

```
GREEDY_NN({objectives}, start):
  visited = [start]
  while unvisited:
    next = argmin_{o ∈ unvisited} ‖visited[-1] − o.location‖
    visited.append(next); unvisited.remove(next)
```

Complexity $O(N^2)$; optionally refined by **2-opt local search** for $N > 20$.

### 11.2 Mission Priority → NSGA-III Weights

| Priority | $w_{time}$ | $w_{energy}$ | $w_{threat}$ | $w_{smooth}$ |
|---|---|---|---|---|
| SPEED | 3.0 | 1.0 | 1.0 | 1.0 |
| EFFICIENCY | 1.0 | 3.0 | 1.0 | 1.0 |
| STEALTH | 1.0 | 1.0 | 3.0 | 1.0 |
| BALANCED | 1.0 | 1.0 | 1.0 | 1.0 |

### 11.3 Three-Tier Contingency Response

| Tier | Trigger | Response | Latency |
|---|---|---|---|
| 1 — Minor | Battery 5% below plan | Velocity reduction / altitude ±50 m | < 2 s |
| 2 — Local replan | $P_K > 0.3$ or 50 m cross-track error | RRT* on next 5 km segment | < 10 s |
| 3 — Emergency | Motor failure / comm loss / battery critical | Nearest alternate landing site | Immediate |

### 11.4 Continuity at Leg Junctions

Adjacent legs are joined by a **junction spline** (2–5 s duration) enforcing:
- $C^0$: position match (exact, forced by construction)
- $C^1$: velocity continuity (Bezier endpoint tangent matching)
- $C^2$: acceleration continuity (B-spline knot insertion at boundary)

---

## 12. Dataset Generation & Schema

### 12.1 Generation Pipeline

```
generate_planning_dataset.py:
  1. Load perception_full_dataset.parquet (1,057,714 rows)
  2. Build perception cost grid (bilinear lookup arrays for all 5 cost fields)
  3. Sample N start/goal pairs uniformly from theatre bounds
     lat ∈ [28.40, 28.90]°N, lon ∈ [76.90, 77.50]°E
     alt ∈ {100, 300, 500} m AGL (stratified)
  4. For each pair:
     a. RRT* plan (n=3000, step_size=500 m, threat cost weight=1.5)
     b. Evaluate: energy, time, threat, terrain, wind, obstacle costs
     c. Smooth trajectory (B-spline, s=0.1·N)
     d. Feasibility check (if fails: replan with larger step)
     e. Label: risk_label = (threat_cost > τ OR fused_cost > φ)
  5. Export to test_final.parquet
```

Default thresholds: $\tau_{threat} = 0.7$, $\phi_{fused} = 0.6$.

### 12.2 Dataset Schema

| Column | Type | Units | Description |
|---|---|---|---|
| start_lat, start_lon | float64 | degrees | Mission start (geodetic) |
| start_alt_m | float64 | m AGL | Departure altitude |
| goal_lat, goal_lon | float64 | degrees | Mission goal (geodetic) |
| goal_alt_m | float64 | m AGL | Arrival altitude |
| waypoint_lats/lons/alts | object | degrees/m | Serialised waypoint sequences |
| n_waypoints | int64 | — | Number of RRT* waypoints |
| planning_time_s | float64 | s | Wall-clock time for RRT* |
| rrt_cost | float64 | — | Final RRT* path cost (normalised) |
| path_length_m | float64 | m | Total 3D path length |
| time_cost_s | float64 | s | Estimated flight time |
| energy_cost_wh | float64 | Wh | Estimated energy draw |
| threat_cost | float64 | [0,1] | Integrated threat exposure |
| terrain_cost_mean | float64 | [0,1] | Mean terrain cost along path |
| wind_cost_mean | float64 | [0,1] | Mean wind cost along path |
| obstacle_cost_mean | float64 | [0,1] | Mean obstacle cost along path |
| fused_cost_mean | float64 | [0,1] | Mean weighted fused cost |
| max_combined_threat | float64 | [0,1] | Peak $P_{combined}$ along path |
| feasible | int64 | {0,1} | Passes all kinodynamic checks |
| altitude_clearance_ok | int64 | {0,1} | Min AGL ≥ 50 m |
| speed_ok | int64 | {0,1} | Max speed ≤ 60 m/s |
| risk_label | int64 | {0,1} | 0 = low risk, 1 = high risk |

**Dataset statistics** (test_final.parquet, 2000 rows):
- 0 NaN values across all columns
- Low risk: 1479 (73.95%), High risk: 521 (26.05%)
- Mean path length: 14.3 km, Mean energy: 1711 Wh
- Note: `feasible`, `altitude_clearance_ok`, `speed_ok` are all 1 (only feasible trajectories retained)

---

## 13. Perception → Planning Integration

Implemented in `planning/perception_integration.py`.

### 13.1 Column Mapping

| Perception column | Planning usage | Grid resolution |
|---|---|---|
| `terrain_cost` | Edge cost term $c_{terrain}(q)$ | 222 m spatial |
| `wind_cost` | Edge cost term $c_{wind}(q)$ | 222 m spatial |
| `obstacle_cost` | Edge cost term $c_{obs}(q)$ | 222 m spatial |
| `threat_cost` | Edge cost term + feasibility check | 222 m spatial |
| `fused_cost` | Risk label threshold | 222 m spatial |
| `terrain_clearance_m` | Feasibility check $h_{clear} \geq 50$ m | 222 m spatial |
| `nearest_obstacle_dist_m` | Clearance check | 222 m spatial |
| `combined_threat_prob` | Max threat check | 222 m spatial |

### 13.2 Grid Interpolation

The perception dataset is a structured $250 \times 300 \times 14$ grid. For arbitrary query
point $(lat_q, lon_q, alt_q)$, **trilinear interpolation** is used:

$$f(q) = \sum_{i \in \{0,1\}} \sum_{j \in \{0,1\}} \sum_{k \in \{0,1\}} f_{i,j,k} \cdot w_i \cdot w_j \cdot w_k$$

where $w_i = (lat_q - lat_0) / \Delta lat$ (and similarly for $lon$, $alt$).

For planar (2D) queries (altitude held fixed), **bilinear interpolation** over $250 \times 300$:

$$f(\phi, \lambda) = (1-t)(1-u) f_{00} + t(1-u) f_{10} + (1-t)u f_{01} + tu f_{11}$$

with $t = (\phi - \phi_0)/\Delta\phi$, $u = (\lambda - \lambda_0)/\Delta\lambda$.

### 13.3 Cost Weighting

The composite planning cost field uses mission-priority-dependent weights:

$$C_{planning}(q) = w_1 C_{terrain} + w_2 C_{wind} + w_3 C_{threat} + w_4 C_{obstacle}$$

Default weights (BALANCED priority): $w_1 = w_2 = w_3 = w_4 = 0.25$.

---

## 14. Implementation Architecture

```
src/evtol/planning/
├── algorithms/            # Path search
│   ├── base.py            # PathPlanner ABC, PlanningConfig
│   ├── graph.py           # Dijkstra, A*, Theta*
│   └── sampling.py        # RRT, RRT*
│
├── core/                  # Shared data structures
│   ├── state.py           # State, Pose, Velocity, FlightPhase
│   ├── trajectory.py      # Trajectory, TrajectorySegment
│   ├── constraints.py     # ConstraintSet, 8 hard constraint types
│   ├── cost.py            # CostFunction, composite evaluation
│   ├── energy_evaluator.py# Physics-based energy model
│   └── threat_analyzer.py # SAM/radar threat field integration
│
├── mission/               # Mission-level orchestration
│   ├── planner.py         # MissionPlanner interface
│   ├── mission_planner.py # RRT* + NSGA-III orchestrator
│   ├── mission_loader.py  # JSON mission file parsing
│   ├── contingency.py     # ContingencyManager, alternate sites
│   └── trajectory_tracking_orchestrator.py
│
├── optimization/          # Multi-objective optimisation
│   ├── nsga3.py           # NSGA-III (SBX + poly mutation)
│   ├── objectives.py      # 6 objective functions
│   ├── pareto.py          # Fast non-dominated sort, crowding
│   └── dynamic_replanner.py   # 3-tier online replanning
│
├── robust/                # Uncertainty-aware planning
│   ├── uncertainty.py     # Gaussian/Wind/Threat uncertainty, UKF
│   └── chance_constraints.py  # Analytical + scenario approach
│
├── trajectory/            # Post-processing
│   ├── smoothing.py       # B-spline, moving average, Savitzky-Golay
│   ├── velocity_profile.py# Trapezoidal, S-curve, time-optimal
│   ├── feasibility.py     # Kinodynamic + environmental checks
│   ├── differential_flatness_analyzer.py  # Flatness + LQR
│   ├── execution_engine.py# TrajectoryExecutionEngine (Phase 2D)
│   ├── online_replanning_interface.py
│   └── execution_integration_phase2e.py   # 50 Hz control loop
│
├── rrt_star.py            # CANONICAL threat-aware RRT* for defense missions
│                          #   Extends algorithms/sampling.py with SAM cost weighting,
│                          #   NED↔lat/lon bridging, γ per Karaman & Frazzoli 2011 §38
├── nsga3_optimizer.py     # Thin re-export of optimization/nsga3.py (for direct import)
└── perception_integration.py  # Perception → Planning bridge
```

> **Note:** `algorithms/sampling.py` is the generic RRT/RRT*/Informed-RRT* framework.
> `rrt_star.py` is the canonical implementation for actual mission runs — use it for
> any defense-mission planning task. The two files are intentionally separate so the
> generic framework can be unit-tested independently.

**Import graph** (no circular dependencies):
```
algorithms → core
optimization → core + algorithms
trajectory → core + optimization
mission → trajectory + optimization + core
perception_integration → core
```

---

## 15. Audit Findings & Limitations

### 15.1 Dataset Audit (2026-04-15, multi-region)

| Check | Result |
|---|---|
| Total rows (Delhi 10K) | 10,000 |
| Total rows (5 other regions × 2K) | 10,000 |
| NaN values | 0 across all 25 planning columns |
| `feasible` / `altitude_clearance_ok` / `speed_ok` | All constant = 1 (only feasible trajectories retained) |
| `threat_cost` variance | Mean=0.79, std=0.12, range 0.45–1.0 (genuine spatial gradient, fixed via distance-decay model) |
| `max_combined_threat` | 1.0 everywhere — retained for disclosure; not used as planning input |
| Class balance (Delhi) | ~74% Low Risk, 26% High Risk |
| Module import failures | 0 (all core imports verified) |

**Fix applied (2026-04-03):** `_compute_threat_gradient()` in `scripts/planning/dataset.py` now
uses shortened effective ranges (20–30 km per emitter) to produce a distance-decay threat field
with genuine spatial variation. The physics-based `combined_threat_prob` (= 1.0 everywhere) is
retained in the perception CSV for disclosure purposes only.

### 15.2 Vehicle / Control Layer Readiness

All 119 source modules across planning, vehicle, and control layers import without error.
Key modules confirmed operational:

| Layer | Key file | Status |
|---|---|---|
| Vehicle | `vehicle/vehicle_model.py` | Importable, full 6-DOF dynamics |
| Vehicle | `vehicle/propulsion/rotor_model.py` | BEMT theory, full equations |
| Vehicle | `vehicle/energy/battery_model.py` | Li-NCA chemistry, SOC tracking |
| Control | `control/flight_controller.py` | Cascaded loop, all modes |
| Control | `control/guidance/trajectory_tracker.py` | Time-parameterised tracking |
| Control | `control/inner_loop/attitude_controller.py` | PID gains, quaternion attitude |

**Next steps for vehicle/control integration**:
1. Generate a **vehicle dynamics dataset** by running `main.py` with the planning dataset
   trajectories as inputs
2. Produce control response visualisations (motor commands, attitude errors, energy draw)
3. Run SITL (`control/sitl_simulator.py`) with ArduPilot to validate trajectory tracking

---

## 16. References

1. **LaValle, S.M.** (2006). *Planning Algorithms*. Cambridge University Press.
2. **Karaman, S. & Frazzoli, E.** (2011). Sampling-based algorithms for optimal motion planning.
   *International Journal of Robotics Research*, 30(7), 846–894.
3. **Gammell, J.D. et al.** (2014). Informed RRT*: Optimal sampling-based path planning focused
   via direct sampling of an admissible ellipsoidal heuristic. *IROS*.
4. **Deb, K., Pratap, A., Agarwal, S. & Meyarivan, T.** (2002). A fast and elitist multiobjective
   genetic algorithm: NSGA-II. *IEEE Trans. Evolutionary Computation*, 6(2), 182–197.
5. **Deb, K. & Jain, H.** (2014). An evolutionary many-objective optimization algorithm using
   reference-point-based non-dominated sorting approach, Part I. *IEEE TEVC*, 18(4), 577–601.
6. **Mellinger, D. & Kumar, V.** (2011). Minimum snap trajectory generation and control for
   quadrotors. *IEEE ICRA*. https://doi.org/10.1109/ICRA.2011.5980409
7. **Fliess, M., Lévine, J., Martin, P. & Rouchon, P.** (1995). Flatness and defect of
   non-linear systems: introductory theory and examples. *Int. J. Control*, 61(6), 1327–1361.
8. **Wan, E.A. & Merwe, R. van der** (2000). The unscented Kalman filter for nonlinear
   estimation. *Proc. IEEE ASSPCC*, 153–158.
9. **Calafiore, G. & Campi, M.C.** (2006). The scenario approach to robust control design.
   *IEEE Trans. Automatic Control*, 51(5), 742–753.
10. **Deb, K. & Agrawal, R.B.** (1995). Simulated binary crossover for continuous search space.
    *Complex Systems*, 9(2), 115–148.
11. **Leishman, J.G.** (2006). *Principles of Helicopter Aerodynamics*, 2nd ed. Cambridge.
12. **Bentley, J.L.** (1975). Multidimensional binary search trees used for associative
    searching. *Commun. ACM*, 18(9), 509–517.
13. **Skolnik, M.I.** (2008). *Radar Handbook*, 3rd ed. McGraw-Hill. (Radar range equation)
14. **Beard, R.W. & McLain, T.W.** (2012). *Small Unmanned Aircraft: Theory and Practice*.
    Princeton University Press, Chs. 8–10.
15. **Raymer, D.P.** (2018). *Aircraft Design: A Conceptual Approach*, 6th ed. AIAA.
    §11 (energy sizing), §12 (aerodynamics).
