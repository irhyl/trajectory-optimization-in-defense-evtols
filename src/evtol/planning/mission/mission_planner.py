"""
Mission Planning Engine - Orchestrator

This module implements the mission planning engine that orchestrates:

1. **RRT* Path Planning**: Initial feasible path generation
   - Rapidly-exploring random trees
   - Informed tree rewiring for optimization
   - Constraint satisfaction (altitude, airspace)

2. **NSGA-III Multi-Objective Optimization**: Pareto frontier
   - Distance & energy efficiency objectives
   - Threat avoidance objective  
   - Timing window satisfaction
   - Smooth trajectory generation

3. **Mission Sequencing**: Multi-leg trajectories
   - Waypoint ordering
   - Phase transitions (hover→cruise)
   - Contingency planning

Workflow
========

1. Initialize mission with objectives and constraints
2. For each leg:
   a. Generate initial path with RRT*
   b. Evaluate path against energy and threat models
   c. Optimize with NSGA-III to find Pareto-optimal trajectories
   d. Select solution based on mission priority (speed vs. safety vs. efficiency)
3. Execute trajectory through vehicle controller

Decision Logic:
- Urban environment → prioritize energy efficiency (time + distance)
- High-threat environment → prioritize threat avoidance
- Time-critical → prioritize speed
- Blue force protection → maximize endurance reserve

Author: Defense eVTOL Research Team
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from enum import Enum, auto
from collections.abc import Callable
import logging

from ..core.trajectory import Trajectory
from ..core.constraints import ConstraintSet
from ..core.energy_evaluator import EnergyEvaluator, EnergyProfile
from ..core.threat_analyzer import ThreatAnalyzer, ThreatField
from ..rrt_star import RRTStarPlanner as RRTStar
from ..nsga3_optimizer import NSGA3Optimizer
from ..core.state import State, Pose
from ...vehicle.config import MassProperties, PropulsionConfig
from ...vehicle.energy.battery_model import BatteryPack as BatteryModel

logger = logging.getLogger(__name__)


class MissionPriority(Enum):
    """Mission execution priority."""
    COMPLETION = auto()        # Get there safely
    SPEED = auto()             # Minimize time
    EFFICIENCY = auto()         # Minimize energy
    STEALTH = auto()            # Minimize threat exposure
    ENDURANCE = auto()          # Maximize return reserves
    BALANCED = auto()           # Weighted combination


@dataclass
class MissionObjective:
    """Single mission objective/waypoint."""
    
    location: np.ndarray  # 3D position [m]
    objective_id: str = ""
    priority: int = 1  # 1=highest
    required: bool = True
    time_window: tuple[float, float] | None = None  # (earliest, latest) time
    dwell_time: float = 0.0  # Time to spend at objective [s]
    
    def __post_init__(self):
        self.location = np.asarray(self.location)


@dataclass
class MissionPlan:
    """Complete mission plan with trajectories for each leg."""
    
    objectives: list[MissionObjective]
    legs: list[MissionLeg] = field(default_factory=list)
    total_time: float = 0.0
    total_distance: float = 0.0
    total_energy: float = 0.0
    
    # Statistics
    max_threat: float = 0.0
    mean_threat: float = 0.0
    feasible: bool = False
    reason: str = ""


@dataclass
class MissionLeg:
    """Single leg of mission trajectory."""
    
    leg_id: int
    start_point: np.ndarray
    end_point: np.ndarray
    trajectory: Trajectory | None = None
    
    # Evaluations
    distance: float = 0.0
    time: float = 0.0
    energy: EnergyProfile | None = None
    threat_analysis: dict | None = None
    
    feasible: bool = False


class MissionPlanner:
    """
    Mission planning orchestrator.
    
    Coordinates RRT* and NSGA-III to generate optimal multi-leg missions
    under energy, threat, and timing constraints.
    """
    
    def __init__(
        self,
        mass_config: MassProperties,
        propulsion_config: PropulsionConfig,
        battery_model: BatteryModel,
        threat_field: ThreatField | None = None,
        priority: MissionPriority = MissionPriority.BALANCED,
    ):
        """
        Initialize mission planner.
        
        Args:
            mass_config: Vehicle mass properties
            propulsion_config: Motor/propeller config
            battery_model: Battery model
            threat_field: Optional threat environment
            priority: Mission execution priority
        """
        self.mass = mass_config
        self.propulsion = propulsion_config
        self.battery = battery_model
        self.priority = priority
        
        # Initialize evaluators
        self.energy_eval = EnergyEvaluator(
            mass_config,
            propulsion_config,
            battery_model,
        )
        
        self.threat_eval = None
        if threat_field is not None:
            self.threat_eval = ThreatAnalyzer(threat_field)
        
        # Planners (initialized on demand)
        self.rrt_planner = None
        self.nsga3_optimizer = None
        
        logger.info(f"MissionPlanner initialized: {priority.name} priority")
    
    def plan_mission(
        self,
        objectives: list[MissionObjective],
        start_state: State,
        max_altitude: float = 500.0,
        planning_time_per_leg: float = 30.0,
    ) -> MissionPlan:
        """
        Plan complete multi-leg mission.
        
        Args:
            objectives: Ordered list of mission objectives
            start_state: Initial vehicle state
            max_altitude: Maximum allowed altitude [m]
            planning_time_per_leg: Time budget for each leg [s]
            
        Returns:
            Mission plan with optimized trajectories
        """
        logger.info(f"Planning mission with {len(objectives)} objectives")
        
        # Validate inputs
        if not objectives:
            raise ValueError("Mission must have at least one objective")
        
        mission_plan = MissionPlan(objectives=objectives)
        legs = []
        
        # Plan each leg
        current_state = start_state
        total_time = 0.0
        
        for i, objective in enumerate(objectives):
            logger.info(f"Planning leg {i+1}/{len(objectives)}: {objective.objective_id}")
            
            # Plan leg from current position to objective
            leg = self._plan_leg(
                leg_id=i,
                start_state=current_state,
                end_position=objective.location,
                max_altitude=max_altitude,
                time_budget=planning_time_per_leg,
            )
            
            if leg is None or not leg.feasible:
                logger.warning(f"Leg {i} planning failed")
                mission_plan.feasible = False
                mission_plan.reason = f"Leg {i} infeasible"
                return mission_plan
            
            legs.append(leg)
            
            # Update state for next leg
            if leg.trajectory is not None:
                current_state = leg.trajectory.segments[-1].end
                total_time += leg.time
        
        # Finalize plan
        mission_plan.legs = legs
        mission_plan.total_time = total_time
        mission_plan.total_distance = sum(leg.distance for leg in legs)
        mission_plan.total_energy = sum(
            leg.energy.total_energy for leg in legs if leg.energy
        )
        mission_plan.feasible = True
        mission_plan.reason = "Plan success"
        
        return mission_plan
    
    def _plan_leg(
        self,
        leg_id: int,
        start_state: State,
        end_position: np.ndarray,
        max_altitude: float,
        time_budget: float,
    ) -> MissionLeg | None:
        """
        Plan single mission leg using RRT* + NSGA-III.
        
        Args:
            leg_id: Leg number
            start_state: Initial state
            end_position: Target position
            max_altitude: Altitude constraint
            time_budget: Planning time available
            
        Returns:
            Optimized leg trajectory, or None if planning failed
        """
        # Initialize RRT* planner if needed
        if self.rrt_planner is None:
            self._init_rrt_planner(max_altitude)
        
        # Step 1: Generate initial feasible path with RRT*
        logger.debug("Step 1: RRT* path generation")
        raw_path, path_cost = self.rrt_planner.plan(
            start=start_state.pose.position.tolist(),
            goal=end_position.tolist(),
        )

        if raw_path is None:
            logger.warning(f"Leg {leg_id}: RRT* failed to find path")
            return None
        initial_path = raw_path
        
        logger.debug(f"Initial path: {len(initial_path)} waypoints")
        
        # Step 2: Evaluate initial path
        initial_trajectory = self._waypoints_to_trajectory(initial_path, start_state)
        energy_profile = self.energy_eval.evaluate_trajectory(initial_trajectory)
        
        # Check initial feasibility
        if not energy_profile.mission_feasible:
            logger.warning(f"Leg {leg_id}: Initial path infeasible (energy)")
            return None
        
        # Step 3: Optimize with NSGA-III if threat evaluation enabled
        if self.threat_eval is not None:
            logger.debug("Step 3: NSGA-III multi-objective optimisation")

            if self.nsga3_optimizer is None:
                self._init_nsga3_optimizer()

            # Build seed population from RRT* path by perturbing waypoints
            from ..nsga3_optimizer import Trajectory as NsgaTrajectory
            seed_trajs = []
            base_wps = [np.asarray(wp) for wp in initial_path]
            base_vels = [50.0] * len(base_wps)
            for _ in range(20):
                wps = [wp + np.random.randn(3) * 50.0 for wp in base_wps]
                seed_trajs.append(NsgaTrajectory(waypoints=wps, velocities=list(base_vels)))

            pareto_solutions = self.nsga3_optimizer.optimize(
                initial_trajectories=seed_trajs,
                vehicle_evaluator=self._make_energy_evaluator_for_nsga(),
                threat_evaluator=self._make_threat_evaluator_for_nsga(),
                max_time=time_budget * 0.7,
            )

            # Convert best NSGA trajectory back to planning Trajectory
            best_nsga = self._select_best_nsga_solution(pareto_solutions)
            if best_nsga is not None:
                best_trajectory = self._waypoints_to_trajectory(
                    best_nsga.waypoints, start_state
                )
            else:
                best_trajectory = initial_trajectory
        else:
            best_trajectory = initial_trajectory
        
        # Step 4: Build leg result
        result = MissionLeg(
            leg_id=leg_id,
            start_point=start_state.pose.position.copy(),
            end_point=end_position.copy(),
            trajectory=best_trajectory,
        )
        
        # Compute metrics
        result.energy = self.energy_eval.evaluate_trajectory(best_trajectory)
        result.distance = self._compute_distance(best_trajectory)
        result.time = best_trajectory.duration if best_trajectory else 0.0
        
        if self.threat_eval is not None:
            result.threat_analysis = self.threat_eval.analyze_trajectory(
                best_trajectory
            )
        
        # Final feasibility check
        result.feasible = result.energy.mission_feasible
        
        return result
    
    def _init_rrt_planner(self, max_altitude: float) -> None:
        """Initialize RRT* planner with constraints."""
        # Create bounding box for planning
        search_radius = 10_000.0  # 10 km
        
        self.rrt_planner = RRTStar(
            bounds=np.array([
                [-search_radius, search_radius],
                [-search_radius, search_radius],
                [0, max_altitude],
            ]),
            step_size=100.0,  # 100 m steps
            goal_sample_ratio=0.1,  # 10% goal biasing
        )
    
    def _init_nsga3_optimizer(self) -> None:
        """Initialize NSGA-III optimizer."""
        from ..nsga3_optimizer import (
            NSGA3Optimizer, NSGA3Config,
            EnergyObjective, TimeObjective, ThreatMarginObjective,
        )
        objectives = [EnergyObjective(), TimeObjective(), ThreatMarginObjective()]
        config = NSGA3Config(population_size=50, generations=20, mutation_prob=0.2)
        self.nsga3_optimizer = NSGA3Optimizer(objectives=objectives, config=config)
    
    def _make_energy_objective(self) -> Callable:
        """Create energy minimization objective (planning Trajectory)."""
        def objective(trajectory: Trajectory) -> float:
            return self.energy_eval.energy_objective(trajectory)
        return objective

    def _make_threat_objective(self) -> Callable:
        """Create threat minimization objective (planning Trajectory)."""
        if self.threat_eval is None:
            return lambda t: 0.0

        def objective(trajectory: Trajectory) -> float:
            return self.threat_eval.threat_objective(trajectory)
        return objective

    def _make_energy_evaluator_for_nsga(self) -> Callable:
        """Vehicle evaluator for NSGA NsgaTrajectory: fills .energy and .time."""
        def evaluator(nsga_traj) -> None:
            total_dist = sum(
                np.linalg.norm(np.asarray(nsga_traj.waypoints[i + 1]) -
                               np.asarray(nsga_traj.waypoints[i]))
                for i in range(len(nsga_traj.waypoints) - 1)
            )
            avg_speed = max(np.mean(nsga_traj.velocities), 1.0)
            nsga_traj.time = total_dist / avg_speed
            # Simple power model: hover + cruise (Wh)
            nsga_traj.energy = (15000 + 50 * avg_speed ** 2) * nsga_traj.time / 3600 / 1000
        return evaluator

    def _make_threat_evaluator_for_nsga(self) -> Callable:
        """Threat evaluator for NSGA NsgaTrajectory: fills .threat_margin."""
        if self.threat_eval is None:
            return lambda t: None

        def evaluator(nsga_traj) -> None:
            min_margin = float('inf')
            for wp in nsga_traj.waypoints:
                wp_arr = np.asarray(wp)
                for threat in self.threat_eval.field.threats.values():
                    dist = np.linalg.norm(wp_arr - threat.position)
                    min_margin = min(min_margin, dist - threat.kill_radius)
            nsga_traj.threat_margin = max(0.0, min_margin)
        return evaluator

    def _select_best_nsga_solution(self, pareto_solutions: list):
        """Pick the knee-point NSGA solution based on mission priority."""
        if not pareto_solutions:
            return None
        if len(pareto_solutions) == 1:
            return pareto_solutions[0]

        best = pareto_solutions[0]
        best_score = float('inf')
        for sol in pareto_solutions:
            if self.priority == MissionPriority.SPEED:
                score = sol.time
            elif self.priority == MissionPriority.EFFICIENCY:
                score = sol.energy
            elif self.priority == MissionPriority.STEALTH:
                score = -sol.threat_margin
            else:
                score = 0.3 * sol.time + 0.4 * sol.energy + 0.3 * (-sol.threat_margin)
            if score < best_score:
                best_score = score
                best = sol
        return best
    
    def _waypoints_to_trajectory(
        self,
        waypoints: list,
        start_state: State,
    ) -> Trajectory:
        """
        Convert RRT* waypoint list to Trajectory object.

        Args:
            waypoints: List of np.ndarray [x, y, z] positions from RRT*
            start_state: Initial vehicle state (for energy/time initialisation)

        Returns:
            Trajectory object with monotonically increasing timestamps
        """
        from ..core.state import State, Pose, Velocity
        from ..core.trajectory import Trajectory

        states = []
        current_time = start_state.time
        current_energy = start_state.energy

        for i, wp in enumerate(waypoints):
            wp = np.asarray(wp, dtype=float)

            if i > 0:
                prev_wp = np.asarray(waypoints[i - 1], dtype=float)
                dist = np.linalg.norm(wp - prev_wp)
                horiz_dist = np.linalg.norm((wp - prev_wp)[:2])
                speed = 5.0 if horiz_dist < 10.0 else 50.0
                current_time += dist / max(speed, 0.1)
                current_energy = max(0.05, current_energy - dist * 1e-5)

                direction = wp - prev_wp
                d = np.linalg.norm(direction)
                vel_vec = (direction / d * speed) if d > 1e-6 else np.zeros(3)
            else:
                vel_vec = start_state.velocity.linear.copy()

            state = State(
                pose=Pose(position=wp.copy(), frame=start_state.pose.frame),
                velocity=Velocity(linear=vel_vec),
                energy=current_energy,
                time=current_time,
            )
            states.append(state)

        # Ensure at least 2 states
        if len(states) < 2:
            states.append(states[0])

        return Trajectory(states=states)
    
    def _select_best_solution(
        self,
        pareto_solutions: list[Trajectory],
    ) -> Trajectory:
        """
        Select single solution from Pareto front based on mission priority.
        
        Args:
            pareto_solutions: List of non-dominated trajectories
            
        Returns:
            Selected trajectory
        """
        if not pareto_solutions:
            return None
        
        if len(pareto_solutions) == 1:
            return pareto_solutions[0]
        
        # Score each solution based on priority
        best_trajectory = pareto_solutions[0]
        best_score = float('inf')
        
        for trajectory in pareto_solutions:
            energy = self.energy_eval.evaluate_trajectory(trajectory)
            distance = self._compute_distance(trajectory)
            time = trajectory.duration if trajectory else 0.0
            
            # Compute priority-weighted score
            if self.priority == MissionPriority.SPEED:
                score = time
            elif self.priority == MissionPriority.EFFICIENCY:
                score = energy.total_energy
            elif self.priority == MissionPriority.ENDURANCE:
                score = -energy.reserve_energy  # Maximize reserve
            elif self.priority == MissionPriority.STEALTH and self.threat_eval:
                threat_analysis = self.threat_eval.analyze_trajectory(trajectory)
                score = threat_analysis['cumulative_threat']
            else:  # BALANCED
                # Weighted combination
                score = (
                    time * 0.3 +
                    energy.total_energy * 0.4 +
                    distance * 0.3
                )
            
            if score < best_score:
                best_score = score
                best_trajectory = trajectory
        
        return best_trajectory
    
    def _compute_distance(self, trajectory: Trajectory) -> float:
        """Compute total horizontal distance."""
        total = 0.0
        for segment in trajectory.segments:
            d = np.linalg.norm(
                segment.end.pose.position[:2] -
                segment.start.pose.position[:2]
            )
            total += d
        return total
