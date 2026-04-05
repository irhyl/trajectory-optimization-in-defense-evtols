"""
Base Path Planner

This module defines the abstract base class for all path planners and
common data structures for planning results.

Design Principles
=================

1. **Separation of Concerns**: Planners are decoupled from cost functions,
   constraints, and environment models.

2. **Configurability**: All parameters are configurable via PlanningConfig.

3. **Observability**: Planning process exposes statistics and intermediate
   results for debugging and visualization.

4. **Extensibility**: New algorithms inherit from PathPlanner and implement
   the plan() method.

Author: Defense eVTOL Research Team
"""

from __future__ import annotations

import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any
from datetime import datetime
import time
import logging

from ..core.state import State, Pose, Velocity, CoordinateFrame
from ..core.trajectory import Trajectory
from ..core.cost import CostFunction, CompositeCost
from ..core.constraints import ConstraintSet, ConstraintViolation

logger = logging.getLogger(__name__)


@dataclass
class PlanningConfig:
    """
    Configuration for path planning algorithms.

    Attributes:
        resolution: Grid resolution for discretization (meters)
        timeout: Maximum planning time (seconds)
        max_iterations: Maximum iterations/node expansions
        goal_tolerance: Distance to goal for termination (meters)
        altitude_layers: Number of altitude layers in 3D grid
        heuristic_weight: Weight on heuristic (for A*, >1 = greedy)
        smoothing: Post-processing smoothing enabled
        verbose: Print debug information
    """
    # Discretization
    resolution: float = 500.0  # meters
    altitude_layers: int = 10
    altitude_min: float = 50.0
    altitude_max: float = 3000.0

    # Search parameters
    timeout: float = 60.0  # seconds
    max_iterations: int = 100000
    goal_tolerance: float = 1000.0  # meters

    # Algorithm tuning
    heuristic_weight: float = 1.0  # ε for ε-greedy A*

    # Post-processing
    smoothing: bool = True
    smoothing_iterations: int = 3

    # Output
    verbose: bool = False
    store_expanded_nodes: bool = False

    @property
    def altitude_step(self) -> float:
        """Altitude step between layers."""
        return (self.altitude_max - self.altitude_min) / max(1, self.altitude_layers - 1)


@dataclass
class PlanningResult:
    """
    Result from a path planning operation.

    Attributes:
        trajectory: Computed trajectory (None if planning failed)
        success: Whether planning succeeded
        cost: Total cost of trajectory
        objective_values: Individual objective values (for multi-objective)
        statistics: Planning statistics
        violations: Constraint violations (if any)
        metadata: Additional algorithm-specific data
    """
    trajectory: Trajectory | None = None
    success: bool = False
    cost: float = float('inf')
    objective_values: dict[str, float] = field(default_factory=dict)

    # Statistics
    statistics: dict[str, Any] = field(default_factory=dict)

    # Constraint info
    violations: list[ConstraintViolation] = field(default_factory=list)
    is_feasible: bool = True

    # Algorithm-specific
    metadata: dict[str, Any] = field(default_factory=dict)

    # Timing
    planning_time: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def __post_init__(self):
        if self.trajectory is not None and self.success:
            self.statistics.setdefault("trajectory_length", self.trajectory.distance)
            self.statistics.setdefault("trajectory_duration", self.trajectory.duration)

    @property
    def nodes_expanded(self) -> int:
        """Number of nodes expanded during search."""
        return self.statistics.get("nodes_expanded", 0)

    @property
    def iterations(self) -> int:
        """Number of iterations."""
        return self.statistics.get("iterations", 0)

    def summary(self) -> str:
        """Return summary string."""
        if self.success:
            return (
                f"Planning SUCCESS in {self.planning_time:.2f}s: "
                f"cost={self.cost:.2f}, distance={self.trajectory.distance/1000:.1f}km, "
                f"duration={self.trajectory.duration:.0f}s, nodes={self.nodes_expanded}"
            )
        else:
            return f"Planning FAILED after {self.planning_time:.2f}s"


class PathPlanner(ABC):
    """
    Abstract base class for path planning algorithms.

    Subclasses must implement:
    - plan(): Execute planning from start to goal
    - _initialize(): Setup algorithm-specific data structures

    The base class provides:
    - Configuration management
    - Cost and constraint evaluation
    - Result packaging
    - Timing and statistics

    Usage:
        >>> planner = AStarPlanner(config=PlanningConfig(resolution=250))
        >>> result = planner.plan(start_state, goal_state, cost_func, constraints)
        >>> if result.success:
        >>>     trajectory = result.trajectory
    """

    def __init__(
        self,
        config: PlanningConfig | None = None,
        cost_function: CostFunction | None = None,
        constraints: ConstraintSet | None = None,
    ):
        """
        Initialize planner.

        Args:
            config: Planning configuration
            cost_function: Cost function for optimization
            constraints: Constraint set for feasibility
        """
        self.config = config or PlanningConfig()
        self.cost_function = cost_function
        self.constraints = constraints

        # Statistics tracking
        self._stats: dict[str, Any] = {}
        self._expanded_nodes: list[Any] = []

        # Timing
        self._start_time: float = 0.0

    @abstractmethod
    def plan(
        self,
        start: State,
        goal: State,
        cost_function: CostFunction | None = None,
        constraints: ConstraintSet | None = None,
    ) -> PlanningResult:
        """
        Plan a trajectory from start to goal.

        Args:
            start: Starting state
            goal: Goal state
            cost_function: Override cost function (optional)
            constraints: Override constraints (optional)

        Returns:
            PlanningResult with trajectory and statistics
        """
        pass

    def _initialize(self, start: State, goal: State) -> None:
        """
        Initialize algorithm for new planning request.

        Override in subclasses for algorithm-specific setup.

        Args:
            start: Starting state
            goal: Goal state
        """
        self._stats = {
            "nodes_expanded": 0,
            "iterations": 0,
            "open_set_max_size": 0,
        }
        self._expanded_nodes = []
        self._start_time = time.time()

    def _check_timeout(self) -> bool:
        """Check if planning has exceeded timeout."""
        return (time.time() - self._start_time) > self.config.timeout

    def _elapsed_time(self) -> float:
        """Get elapsed planning time."""
        return time.time() - self._start_time

    def _distance(self, state1: State, state2: State) -> float:
        """Compute distance between two states."""
        return state1.distance_to(state2)

    def _heuristic(self, state: State, goal: State) -> float:
        """
        Compute heuristic estimate of cost-to-go.

        Default: Euclidean distance (admissible for path length).
        Override for problem-specific heuristics.

        Args:
            state: Current state
            goal: Goal state

        Returns:
            Heuristic cost estimate (must be admissible)
        """
        return self._distance(state, goal)

    def _is_goal(self, state: State, goal: State) -> bool:
        """Check if state is within goal tolerance."""
        return self._distance(state, goal) <= self.config.goal_tolerance

    def _evaluate_cost(
        self,
        trajectory: Trajectory,
        cost_function: CostFunction | None = None,
    ) -> float:
        """
        Evaluate trajectory cost.

        Args:
            trajectory: Trajectory to evaluate
            cost_function: Cost function (uses instance default if None)

        Returns:
            Cost value
        """
        cf = cost_function or self.cost_function
        if cf is None:
            # Default: path length
            return trajectory.distance
        return cf.evaluate(trajectory)

    def _check_constraints(
        self,
        trajectory: Trajectory,
        constraints: ConstraintSet | None = None,
    ) -> tuple[bool, list[ConstraintViolation]]:
        """
        Check trajectory against constraints.

        Args:
            trajectory: Trajectory to check
            constraints: Constraint set (uses instance default if None)

        Returns:
            (is_feasible, list of violations)
        """
        cs = constraints or self.constraints
        if cs is None:
            return True, []

        violations = cs.evaluate(trajectory)
        is_feasible = cs.is_feasible(trajectory)

        return is_feasible, violations

    def _create_result(
        self,
        trajectory: Trajectory | None,
        success: bool,
        cost_function: CostFunction | None = None,
        constraints: ConstraintSet | None = None,
    ) -> PlanningResult:
        """
        Create planning result with statistics.

        Args:
            trajectory: Computed trajectory (or None)
            success: Whether planning succeeded
            cost_function: Cost function used
            constraints: Constraints used

        Returns:
            PlanningResult
        """
        planning_time = self._elapsed_time()

        # Compute cost if trajectory exists
        cost = float('inf')
        objective_values = {}
        if trajectory is not None and success:
            cf = cost_function or self.cost_function
            if cf is not None:
                cost = cf.evaluate(trajectory)
                if isinstance(cf, CompositeCost):
                    objective_values = cf.evaluate_objectives(trajectory)

        # Check constraints
        is_feasible = True
        violations = []
        if trajectory is not None:
            is_feasible, violations = self._check_constraints(trajectory, constraints)

        result = PlanningResult(
            trajectory=trajectory,
            success=success,
            cost=cost,
            objective_values=objective_values,
            statistics=dict(self._stats),
            violations=violations,
            is_feasible=is_feasible,
            planning_time=planning_time,
            metadata={"algorithm": self.__class__.__name__},
        )

        if self.config.store_expanded_nodes:
            result.metadata["expanded_nodes"] = self._expanded_nodes

        return result

    def _smooth_trajectory(self, trajectory: Trajectory) -> Trajectory:
        """
        Apply post-processing smoothing to trajectory.

        Args:
            trajectory: Input trajectory

        Returns:
            Smoothed trajectory
        """
        if not self.config.smoothing:
            return trajectory

        return trajectory.smooth(window_size=5)

    @property
    def name(self) -> str:
        """Planner name."""
        return self.__class__.__name__


class GridPlanner(PathPlanner):
    """
    Base class for grid-based planners.

    Provides:
    - 3D grid discretization
    - Neighbor generation
    - State-to-grid conversion

    Subclasses: A*, Dijkstra, Theta*
    """

    def __init__(
        self,
        config: PlanningConfig | None = None,
        cost_function: CostFunction | None = None,
        constraints: ConstraintSet | None = None,
        bounds: tuple[float, float, float, float] | None = None,
    ):
        """
        Initialize grid planner.

        Args:
            config: Planning configuration
            cost_function: Cost function
            constraints: Constraint set
            bounds: (lat_min, lat_max, lon_min, lon_max) or None for auto
        """
        super().__init__(config, cost_function, constraints)
        self.bounds = bounds or (26.0, 30.0, 70.0, 78.0)

        # Grid parameters (computed on initialization)
        self._lat_cells = 0
        self._lon_cells = 0
        self._alt_cells = self.config.altitude_layers

    def _setup_grid(self, start: State, goal: State) -> None:
        """
        Set up 3D grid for planning.

        Works for both geodetic and NED coordinate frames:
        - Geodetic: bounds are (lat_min, lat_max, lon_min, lon_max) in degrees
        - NED: bounds are (x_min, x_max, y_min, y_max) in meters

        Args:
            start: Start state (for bounds and frame detection)
            goal: Goal state (for bounds)
        """
        # Compute grid dimensions
        x_min, x_max, y_min, y_max = self.bounds

        # Detect frame: if bounds are small numbers (< 200), assume geodetic
        # If bounds are large (meters), assume NED
        is_ned = abs(x_max - x_min) > 1000 or abs(y_max - y_min) > 1000

        if is_ned:
            # NED: bounds are already in meters, resolution is in meters
            self._x_res = self.config.resolution  # meters
            self._y_res = self.config.resolution  # meters
        else:
            # Geodetic: convert resolution from meters to degrees
            self._x_res = self.config.resolution / 111000  # degrees
            self._y_res = self.config.resolution / (111000 * np.cos(np.radians((x_min + x_max) / 2)))

        self._x_cells = max(1, int((x_max - x_min) / self._x_res) + 1)
        self._y_cells = max(1, int((y_max - y_min) / self._y_res) + 1)
        self._alt_cells = self.config.altitude_layers

        # Legacy names for compatibility
        self._lat_res = self._x_res
        self._lon_res = self._y_res
        self._lat_min = x_min
        self._lon_min = y_min
        self._lat_cells = self._x_cells
        self._lon_cells = self._y_cells

        if self.config.verbose:
            logger.info(
                f"Grid: {self._x_cells} x {self._y_cells} x {self._alt_cells} "
                f"= {self._x_cells * self._y_cells * self._alt_cells} cells "
                f"({'NED' if is_ned else 'Geodetic'} frame)"
            )

    def _state_to_grid(self, state: State) -> tuple[int, int, int]:
        """
        Convert state to grid indices.

        Args:
            state: State to convert

        Returns:
            (i_lat, i_lon, i_alt) grid indices
        """
        pos = state.pose.position

        i_lat = int((pos[0] - self._lat_min) / self._lat_res)
        i_lon = int((pos[1] - self._lon_min) / self._lon_res)
        i_alt = int((pos[2] - self.config.altitude_min) / self.config.altitude_step)

        # Clamp to grid bounds
        i_lat = max(0, min(self._lat_cells - 1, i_lat))
        i_lon = max(0, min(self._lon_cells - 1, i_lon))
        i_alt = max(0, min(self._alt_cells - 1, i_alt))

        return (i_lat, i_lon, i_alt)

    def _grid_to_state(
        self,
        i_lat: int,
        i_lon: int,
        i_alt: int,
        time: float = 0.0,
        energy: float = 1.0,
        frame: CoordinateFrame = None,
    ) -> State:
        """
        Convert grid indices to state.

        Args:
            i_lat: Latitude/X index
            i_lon: Longitude/Y index
            i_alt: Altitude index
            time: State time
            energy: State energy
            frame: Coordinate frame (auto-detect if None)

        Returns:
            State at grid cell center
        """
        x = self._lat_min + (i_lat + 0.5) * self._lat_res
        y = self._lon_min + (i_lon + 0.5) * self._lon_res
        alt = self.config.altitude_min + (i_alt + 0.5) * self.config.altitude_step

        # Auto-detect frame based on bounds magnitude
        if frame is None:
            x_range = abs(self.bounds[1] - self.bounds[0])
            frame = CoordinateFrame.NED if x_range > 1000 else CoordinateFrame.GEODETIC

        return State(
            pose=Pose(np.array([x, y, alt]), np.zeros(3), frame=frame),
            velocity=Velocity(np.zeros(3)),
            energy=energy,
            time=time,
        )

    def _get_neighbors(
        self,
        i_lat: int,
        i_lon: int,
        i_alt: int,
    ) -> list[tuple[int, int, int]]:
        """
        Get valid neighbor grid cells.

        Uses 26-connectivity (3D Moore neighborhood).

        Args:
            i_lat: Current latitude index
            i_lon: Current longitude index
            i_alt: Current altitude index

        Returns:
            List of neighbor (i_lat, i_lon, i_alt) tuples
        """
        neighbors = []

        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                for dk in [-1, 0, 1]:
                    if di == 0 and dj == 0 and dk == 0:
                        continue

                    ni = i_lat + di
                    nj = i_lon + dj
                    nk = i_alt + dk

                    # Check bounds
                    if 0 <= ni < self._lat_cells and 0 <= nj < self._lon_cells and 0 <= nk < self._alt_cells:
                        neighbors.append((ni, nj, nk))

        return neighbors

    def _edge_cost(
        self,
        from_idx: tuple[int, int, int],
        to_idx: tuple[int, int, int],
    ) -> float:
        """
        Compute edge cost between adjacent grid cells.

        Default: Euclidean distance. Override for custom costs.

        Args:
            from_idx: Source cell indices
            to_idx: Target cell indices

        Returns:
            Edge cost
        """
        from_state = self._grid_to_state(*from_idx)
        to_state = self._grid_to_state(*to_idx)

        return self._distance(from_state, to_state)
