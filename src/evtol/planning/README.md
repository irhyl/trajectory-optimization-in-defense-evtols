
# Trajectory Planning Layer

Comprehensive module for eVTOL route planning, optimization, and mission planning.

## Overview

The planning layer provides a complete framework for generating optimal trajectories for electric vertical takeoff and landing (eVTOL) aircraft. It integrates multi-objective optimization, energy management, risk assessment, and mission orchestration.

## Architecture

### Core Components

#### 1. **Base Classes** (`base.py`)

Abstract interfaces defining the planning layer API:

- **`Waypoint`**: 3D geographic waypoint with lat/lon/altitude
  - Distance calculation using Haversine formula
  - Hash-based equality for grid algorithms
  
- **`RoutePlan`**: Complete route specification with metrics
  - Waypoint sequence
  - Distance, energy, risk, and flight time estimates
  - Extensible metadata
  
- **`RoutePlanner`**: Abstract route planning interface
  - `plan(start, goal, time_iso, constraints)` → RoutePlan
  
- **`EnergyOptimizer`**: Battery and power management
  - `estimate_route_energy(route, time_iso)` → kwh
  - `optimize_route_for_energy(route, max_energy_kwh, time_iso)` → List[Waypoint]
  - `get_range_for_energy(start, energy_kwh, time_iso)` → (min_km, max_km)
  
- **`RiskManager`**: Threat assessment and contingency planning
  - `evaluate_route_risk(route, time_iso)` → [0-1] score
  - `evaluate_waypoint_risk(waypoint, time_iso)` → [0-1] score
  - `plan_contingency_route(current_route, retreat_waypoint, time_iso)` → RoutePlan
  
- **`Optimizer`**: Multi-objective optimization
  - `optimize(candidates, objectives, constraints)` → List[RoutePlan]
  
- **`MissionPlanner`**: Mission-level planning
  - `plan_mission(origin, destinations, time_iso, constraints)` → mission_dict
  - `generate_holding_pattern(reference_waypoint, duration_s, pattern_type)` → List[Waypoint]
  - `smooth_trajectory(waypoints, min_turn_radius_m, max_altitude_rate_m_s)` → List[Waypoint]

#### 2. **Routing Algorithms** (`routing/`)

- **`AStarPlanner`**: A* pathfinding with multi-objective costs
  - 8-connected grid search in lat/lon space
  - Admissible heuristic for optimality
  - Perception layer integration
  - Straight-line fallback
  - Path smoothing via moving average
  
- **`GraphRoutePlanner`**: Graph-based routing using NetworkX
  - Grid-based waypoint network
  - K-shortest path computation
  - Edge weighting from time/energy/risk

#### 3. **Energy Management** (`energy/`)

- **`EnergyOptimizer`**: Battery modeling and trajectory optimization
  - Battery capacity and reserve management
  - Energy cost per km estimation
  - Route truncation for energy constraints
  - Range estimation with safety margins
  - Integration with perception layer for wind/terrain effects

#### 4. **Risk Assessment** (`risk/`)

- **`RiskManager`**: Threat evaluation and contingency planning
  - Per-waypoint risk scoring
  - Route-wide risk averaging
  - Emergency diversion route generation
  - Integration with perception threat data

#### 5. **Mission Planning** (`mission/`)

- **`MissionPlanner`**: Multi-leg mission orchestration
  - Multi-destination mission planning
  - Holding pattern generation (circle, figure-eight)
  - Trajectory smoothing with flight envelope constraints
  - Time and distance schedule computation

#### 6. **Multi-Objective Optimization** (`optimization/`)

- **`ParetoFrontier`**: Pareto frontier computation
- **`Solution`**: Multi-objective solution representation
- **`DiverseRouteSelector`**: Select diverse Pareto-optimal routes
- **Scalarization methods**: Weighted sum, Tchebycheff

### Configuration

Planning behavior is controlled via `PlanningConfig`:

```yaml
# Routing parameters
routing:
  grid_resolution_deg: 0.01  # ~1 km at equator
  max_iterations: 1000
  goal_tolerance_km: 0.5
  smoothing_window: 5
  objective_weights:
    distance: 0.3
    energy: 0.3
    risk: 0.3
    time: 0.1

# Energy parameters
energy:
  battery_capacity_kwh: 120.0
  reserve_fraction: 0.15  # 15% battery reserve
  cruise_speed_mps: 35.0
  power_idle_kw: 10.0

# Risk parameters
risk:
  max_risk_score: 0.7

# Vehicle parameters
vehicle:
  min_turn_radius_m: 50.0
  max_altitude_rate_mps: 5.0
```

## Usage Examples

### Basic Route Planning

```python
from src.evtol.planning import AStarPlanner, Waypoint, PlanningConfig

config = PlanningConfig()
planner = AStarPlanner(config)

start = Waypoint(40.7128, -74.0060, 100)  # NYC
goal = Waypoint(34.0522, -118.2437, 100)   # LA

plan = planner.plan(start, goal, "2024-01-01T12:00:00Z")
print(f"Route: {len(plan.waypoints)} waypoints")
print(f"Distance: {plan.distance_km:.1f} km")
print(f"Energy: {plan.energy_kwh:.1f} kWh")
print(f"Risk: {plan.risk_score:.2f}")
```

### Energy-Constrained Planning

```python
from src.evtol.planning import EnergyOptimizerImpl

optimizer = EnergyOptimizerImpl(config)

# Get range for available energy
min_range, max_range = optimizer.get_range_for_energy(
    start_waypoint, 80.0, time_iso
)
print(f"Range with 80 kWh: {min_range:.1f}-{max_range:.1f} km")

# Optimize route within energy budget
optimized = optimizer.optimize_route_for_energy(
    route, max_energy_kwh=80.0, time_iso=time_iso
)
```

### Risk Assessment

```python
from src.evtol.planning import RiskManagerImpl

manager = RiskManagerImpl(config)

# Evaluate route risk
route_risk = manager.evaluate_route_risk(waypoints, time_iso)

# Plan emergency diversion
contingency = manager.plan_contingency_route(
    current_route, safe_waypoint, time_iso
)
```

### Mission Planning

```python
from src.evtol.planning import MissionPlannerImpl

planner = MissionPlannerImpl(config)

origin = Waypoint(40.7128, -74.0060, 100)
destinations = [
    Waypoint(40.8, -74.0, 150),
    Waypoint(40.9, -74.0, 200),
]

mission = planner.plan_mission(
    origin, destinations, "2024-01-01T12:00:00Z"
)

# Generate holding pattern
holding = planner.generate_holding_pattern(
    waypoint, duration_s=600, pattern_type="figure_eight"
)

# Smooth trajectory
smooth = planner.smooth_trajectory(waypoints, min_turn_radius_m=50)
```

## Integration Points

### Perception Layer

Planning layer queries the perception layer for:

- **Energy cost**: Cost per km considering wind, terrain, weather
- **Risk scores**: Threat levels from detection systems
- **Feasibility**: Altitude constraints, no-fly zones, obstacles

### Control Layer

Planning layer outputs serve as input to:

- **Trajectory tracking**: Waypoint sequences feed trajectory generators
- **Flight control**: Turns, climb rates, speed adjustments
- **Safety systems**: Risk scores trigger contingency procedures

## Testing

Comprehensive unit tests cover:

```bash
python -m unittest tests.unit.test_planning_layer -v
```

Tests include:

- Waypoint distance calculations and equality
- Route planning with A*
- Energy estimation and optimization
- Risk assessment
- Mission planning
- Holding patterns
- Trajectory smoothing

## Performance

- **Route planning**: O((V+E) log V) for V waypoints, E edges
- **Energy computation**: O(n) for n waypoints
- **Risk assessment**: O(n) for n waypoints
- **Smoothing**: O(n*w) for moving average with window w

## Future Enhancements

1. **Advanced path planning**
   - RRT* for high-dimensional spaces
   - Theta* for any-angle paths
   - Bidirectional search

2. **Optimization**
   - NSGA-III multi-objective optimization
   - Evolutionary algorithms
   - Dynamic programming

3. **Constraints**
   - Thermal updrafts for energy efficiency
   - Airspace restrictions
   - Weather-based rerouting

4. **Real-time planning**
   - Incremental replanning
   - Adaptive waypoint generation
   - In-flight corrections

## References

- Nash et al. (2007): Theta* - Any-Angle Path Planning on Grids
- Karaman & Frazzoli (2011): Sampling-based Algorithms for Optimal Motion Planning
- Pareto optimality in multi-objective optimization
- Haversine formula for geographic distances
