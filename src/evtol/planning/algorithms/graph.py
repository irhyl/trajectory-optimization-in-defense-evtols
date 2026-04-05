"""
Graph-Based Path Planners

This module implements graph-based path planning algorithms:

- **A***: Optimal heuristic search
- **Theta***: Any-angle path planning (smoother paths)
- **Dijkstra**: Baseline shortest path (no heuristic)

All planners operate on a discretized 3D grid and support:
- Multi-objective cost functions
- Constraint checking
- Custom heuristics
- Terrain/threat field integration

Author: Defense eVTOL Research Team
"""

from __future__ import annotations

import numpy as np
import heapq
from dataclasses import dataclass, field
from typing import Any
import logging

from .base import GridPlanner, PlanningConfig, PlanningResult
from ..core.state import State
from ..core.trajectory import Trajectory
from ..core.cost import CostFunction

logger = logging.getLogger(__name__)


@dataclass(order=True)
class SearchNode:
    """
    Node in the search graph.

    Attributes:
        f: Total cost (g + h) for priority queue ordering
        g: Cost from start
        h: Heuristic cost to goal
        idx: Grid indices (i_lat, i_lon, i_alt)
        parent: Parent node index (for path reconstruction)
    """
    f: float
    g: float = field(compare=False)
    h: float = field(compare=False)
    idx: tuple[int, int, int] = field(compare=False)
    parent: tuple[int, int, int] | None = field(default=None, compare=False)


class AStarPlanner(GridPlanner):
    """
    A* Path Planner

    A* is an optimal graph search algorithm that uses heuristics to guide
    the search towards the goal. It maintains:

    - g(n): Cost from start to node n
    - h(n): Heuristic estimate from n to goal
    - f(n) = g(n) + w·h(n): Total estimated cost

    With w=1 (default), A* is optimal if h is admissible.
    With w>1, A* becomes greedy (faster but may not find optimal).

    Implementation Details:
    - Binary heap priority queue for O(log n) operations
    - Hash set for closed list (O(1) lookup)
    - 26-connectivity in 3D grid

    Example:
        >>> planner = AStarPlanner(
        ...     config=PlanningConfig(resolution=500, heuristic_weight=1.0),
        ...     cost_function=create_default_cost(),
        ... )
        >>> result = planner.plan(start, goal)
    """

    def __init__(
        self,
        config: PlanningConfig | None = None,
        cost_function: CostFunction | None = None,
        constraints: Any | None = None,
        bounds: tuple[float, float, float, float] | None = None,
        obstacle_field: Any | None = None,
        threat_field: Any | None = None,
        terrain_field: Any | None = None,
    ):
        """
        Initialize A* planner.

        Args:
            config: Planning configuration
            cost_function: Cost function for edge evaluation
            constraints: Constraint set
            bounds: Grid bounds (lat_min, lat_max, lon_min, lon_max)
            obstacle_field: Obstacle distance field (optional)
            threat_field: Threat probability field (optional)
            terrain_field: Terrain elevation field (optional)
        """
        super().__init__(config, cost_function, constraints, bounds)

        self.obstacle_field = obstacle_field
        self.threat_field = threat_field
        self.terrain_field = terrain_field

        # Search data structures
        self._open_set: list[SearchNode] = []
        self._closed_set: set[tuple[int, int, int]] = set()
        self._came_from: dict[tuple[int, int, int], tuple[int, int, int]] = {}
        self._g_score: dict[tuple[int, int, int], float] = {}

    def plan(
        self,
        start: State,
        goal: State,
        cost_function: CostFunction | None = None,
        constraints: Any | None = None,
    ) -> PlanningResult:
        """
        Plan path from start to goal using A*.

        Args:
            start: Starting state
            goal: Goal state
            cost_function: Override cost function
            constraints: Override constraints

        Returns:
            PlanningResult with trajectory and statistics
        """
        # Initialize
        self._initialize(start, goal)
        self._setup_grid(start, goal)

        cf = cost_function or self.cost_function
        cs = constraints or self.constraints

        # Convert to grid indices
        start_idx = self._state_to_grid(start)
        goal_idx = self._state_to_grid(goal)

        if self.config.verbose:
            logger.info(f"A* planning from {start_idx} to {goal_idx}")

        # Initialize data structures
        self._open_set = []
        self._closed_set = set()
        self._came_from = {}
        self._g_score = {start_idx: 0.0}

        # Start node
        h = self._heuristic_grid(start_idx, goal_idx)
        start_node = SearchNode(
            f=self.config.heuristic_weight * h,
            g=0.0,
            h=h,
            idx=start_idx,
            parent=None,
        )
        heapq.heappush(self._open_set, start_node)

        # Main search loop
        while self._open_set:
            # Check termination conditions
            if self._check_timeout():
                logger.warning("A* timeout")
                return self._create_result(None, False, cf, cs)

            if self._stats["iterations"] >= self.config.max_iterations:
                logger.warning("A* max iterations reached")
                return self._create_result(None, False, cf, cs)

            self._stats["iterations"] += 1
            self._stats["open_set_max_size"] = max(
                self._stats["open_set_max_size"], len(self._open_set)
            )

            # Pop best node
            current = heapq.heappop(self._open_set)

            # Skip if already expanded
            if current.idx in self._closed_set:
                continue

            self._closed_set.add(current.idx)
            self._stats["nodes_expanded"] += 1

            if self.config.store_expanded_nodes:
                self._expanded_nodes.append(current.idx)

            # Goal check
            if current.idx == goal_idx or self._is_goal_grid(current.idx, goal_idx):
                # Reconstruct path
                trajectory = self._reconstruct_path(current.idx, start, goal)

                if self.config.smoothing:
                    trajectory = self._smooth_trajectory(trajectory)

                if self.config.verbose:
                    logger.info(f"A* found path: {self._stats['nodes_expanded']} nodes")

                return self._create_result(trajectory, True, cf, cs)

            # Expand neighbors
            for neighbor_idx in self._get_neighbors(*current.idx):
                if neighbor_idx in self._closed_set:
                    continue

                # Check obstacle collision
                if not self._is_valid_cell(neighbor_idx):
                    continue

                # Compute edge cost
                edge_cost = self._compute_edge_cost(current.idx, neighbor_idx, cf)

                # Check if edge is traversable
                if not self._is_valid_edge(current.idx, neighbor_idx):
                    continue

                # Update g-score
                tentative_g = current.g + edge_cost

                if neighbor_idx not in self._g_score or tentative_g < self._g_score[neighbor_idx]:
                    self._came_from[neighbor_idx] = current.idx
                    self._g_score[neighbor_idx] = tentative_g

                    h = self._heuristic_grid(neighbor_idx, goal_idx)
                    f = tentative_g + self.config.heuristic_weight * h

                    neighbor_node = SearchNode(
                        f=f,
                        g=tentative_g,
                        h=h,
                        idx=neighbor_idx,
                        parent=current.idx,
                    )
                    heapq.heappush(self._open_set, neighbor_node)

        # No path found
        logger.warning("A* failed: no path found")
        return self._create_result(None, False, cf, cs)

    def _heuristic_grid(
        self,
        idx: tuple[int, int, int],
        goal_idx: tuple[int, int, int],
    ) -> float:
        """
        Compute heuristic from grid cell to goal.

        Uses Euclidean distance scaled by resolution.
        Works for both NED (meters) and geodetic (degrees) grids.

        Args:
            idx: Current cell
            goal_idx: Goal cell

        Returns:
            Heuristic cost estimate in meters
        """
        di = abs(idx[0] - goal_idx[0])
        dj = abs(idx[1] - goal_idx[1])
        dk = abs(idx[2] - goal_idx[2])

        # Check if using NED (bounds > 1000) or geodetic
        x_range = abs(self.bounds[1] - self.bounds[0])
        is_ned = x_range > 1000

        if is_ned:
            # NED: resolution is already in meters
            x_dist = di * self._lat_res  # meters
            y_dist = dj * self._lon_res  # meters
        else:
            # Geodetic: convert degrees to meters
            x_dist = di * self._lat_res * 111000
            y_dist = dj * self._lon_res * 111000 * np.cos(np.radians(self._lat_min + idx[0] * self._lat_res))

        alt_dist = dk * self.config.altitude_step

        return np.sqrt(x_dist**2 + y_dist**2 + alt_dist**2)

    def _is_goal_grid(
        self,
        idx: tuple[int, int, int],
        goal_idx: tuple[int, int, int],
    ) -> bool:
        """Check if cell is within goal tolerance."""
        state = self._grid_to_state(*idx)
        goal_state = self._grid_to_state(*goal_idx)
        return self._distance(state, goal_state) <= self.config.goal_tolerance

    def _is_valid_cell(self, idx: tuple[int, int, int]) -> bool:
        """
        Check if cell is valid (not in obstacle).

        Args:
            idx: Cell indices

        Returns:
            True if cell is free
        """
        state = self._grid_to_state(*idx)
        pos = state.pose.position

        # Terrain check
        if self.terrain_field is not None:
            terrain_elev = self.terrain_field.get_elevation(pos[0], pos[1])
            if pos[2] < terrain_elev + 30:  # Minimum clearance
                return False

        # Obstacle check
        if self.obstacle_field is not None:
            # Assume obstacle_field(pos) returns distance
            if self.obstacle_field(pos) < 50:  # Minimum distance
                return False

        return True

    def _is_valid_edge(
        self,
        from_idx: tuple[int, int, int],
        to_idx: tuple[int, int, int],
    ) -> bool:
        """
        Check if edge is traversable.

        Checks:
        - Terrain collision along edge
        - Constraint violations

        Args:
            from_idx: Source cell
            to_idx: Target cell

        Returns:
            True if edge is valid
        """
        # Sample edge for terrain collision
        from_state = self._grid_to_state(*from_idx)
        to_state = self._grid_to_state(*to_idx)

        if self.terrain_field is not None:
            # Check midpoint
            mid_pos = (from_state.pose.position + to_state.pose.position) / 2
            terrain_elev = self.terrain_field.get_elevation(mid_pos[0], mid_pos[1])
            if mid_pos[2] < terrain_elev + 30:
                return False

        return True

    def _compute_edge_cost(
        self,
        from_idx: tuple[int, int, int],
        to_idx: tuple[int, int, int],
        cost_function: CostFunction | None,
    ) -> float:
        """
        Compute edge cost including all objectives.

        Args:
            from_idx: Source cell
            to_idx: Target cell
            cost_function: Cost function

        Returns:
            Edge cost
        """
        from_state = self._grid_to_state(*from_idx)
        to_state = self._grid_to_state(*to_idx)

        # Base cost: Euclidean distance
        distance = self._distance(from_state, to_state)
        cost = distance

        # Threat cost
        if self.threat_field is not None:
            mid_pos = (from_state.pose.position + to_state.pose.position) / 2
            threat = self.threat_field(mid_pos[0], mid_pos[1], mid_pos[2])
            # Add threat penalty (scaled by distance)
            cost += threat * distance * 10  # Threat weight

        # Altitude penalty (prefer lower altitudes for terrain masking)
        avg_alt = (from_state.pose.position[2] + to_state.pose.position[2]) / 2
        alt_penalty = 0.001 * avg_alt  # Small altitude penalty
        cost += alt_penalty

        return cost

    def _reconstruct_path(
        self,
        goal_idx: tuple[int, int, int],
        start: State,
        goal: State,
    ) -> Trajectory:
        """
        Reconstruct path from search result.

        Args:
            goal_idx: Goal cell reached
            start: Original start state
            goal: Original goal state

        Returns:
            Trajectory from start to goal
        """
        # Backtrack through came_from
        path_indices = [goal_idx]
        current = goal_idx

        while current in self._came_from:
            current = self._came_from[current]
            path_indices.append(current)

        path_indices.reverse()

        # Convert to states
        states = []
        time = 0.0
        energy = start.energy

        for i, idx in enumerate(path_indices):
            state = self._grid_to_state(*idx, time=time, energy=energy)

            # Update time based on distance
            if i > 0:
                prev_state = states[-1]
                dist = self._distance(prev_state, state)
                time += dist / 40.0  # Assume 40 m/s average speed

                # Update energy (simple model)
                energy -= dist * 0.00001  # Energy per meter
                energy = max(0.1, energy)

            state.time = time
            state.energy = energy
            states.append(state)

        # Replace first and last with actual start/goal
        # Ensure proper time progression by copying calculated times/energies
        if states:
            # Keep start's position/orientation but ensure time=0
            start_copy = State(
                pose=start.pose,
                velocity=start.velocity,
                energy=start.energy,
                time=0.0,
            )
            states[0] = start_copy

            # Keep goal's position/orientation but use calculated time/energy
            goal_copy = State(
                pose=goal.pose,
                velocity=goal.velocity,
                energy=states[-1].energy,  # Use calculated energy
                time=states[-1].time,      # Use calculated time
            )
            states[-1] = goal_copy

        return Trajectory(states=states)


class ThetaStarPlanner(AStarPlanner):
    """
    Theta* Path Planner

    Theta* is an any-angle path planning variant of A* that produces
    smoother paths by checking line-of-sight to ancestors.

    Unlike A*, which constrains paths to grid edges, Theta* can create
    direct connections between non-adjacent cells if there's clear LOS.

    This produces shorter, smoother paths but is more computationally
    expensive due to LOS checks.

    Key difference from A*:
        When expanding a node, Theta* checks if the parent's parent
        has LOS to the current node. If so, it skips the intermediate
        parent, creating a straighter path.
    """

    def plan(
        self,
        start: State,
        goal: State,
        cost_function: CostFunction | None = None,
        constraints: Any | None = None,
    ) -> PlanningResult:
        """
        Plan path using Theta* (any-angle A*).

        Similar to A*, but with LOS checks for path smoothing.
        """
        # Initialize (same as A*)
        self._initialize(start, goal)
        self._setup_grid(start, goal)

        cf = cost_function or self.cost_function
        cs = constraints or self.constraints

        start_idx = self._state_to_grid(start)
        goal_idx = self._state_to_grid(goal)

        if self.config.verbose:
            logger.info(f"Theta* planning from {start_idx} to {goal_idx}")

        self._open_set = []
        self._closed_set = set()
        self._came_from = {}
        self._g_score = {start_idx: 0.0}

        h = self._heuristic_grid(start_idx, goal_idx)
        start_node = SearchNode(f=h, g=0.0, h=h, idx=start_idx, parent=None)
        heapq.heappush(self._open_set, start_node)

        while self._open_set:
            if self._check_timeout():
                return self._create_result(None, False, cf, cs)

            if self._stats["iterations"] >= self.config.max_iterations:
                return self._create_result(None, False, cf, cs)

            self._stats["iterations"] += 1

            current = heapq.heappop(self._open_set)

            if current.idx in self._closed_set:
                continue

            self._closed_set.add(current.idx)
            self._stats["nodes_expanded"] += 1

            if current.idx == goal_idx or self._is_goal_grid(current.idx, goal_idx):
                trajectory = self._reconstruct_path(current.idx, start, goal)
                if self.config.smoothing:
                    trajectory = self._smooth_trajectory(trajectory)
                return self._create_result(trajectory, True, cf, cs)

            parent_idx = self._came_from.get(current.idx)

            for neighbor_idx in self._get_neighbors(*current.idx):
                if neighbor_idx in self._closed_set:
                    continue

                if not self._is_valid_cell(neighbor_idx):
                    continue

                # Theta* key modification: check LOS to grandparent
                if parent_idx is not None and self._has_los(parent_idx, neighbor_idx):
                    # Path through grandparent is shorter
                    edge_cost = self._compute_edge_cost(parent_idx, neighbor_idx, cf)
                    tentative_g = self._g_score[parent_idx] + edge_cost
                    actual_parent = parent_idx
                else:
                    # Standard A* path through current
                    edge_cost = self._compute_edge_cost(current.idx, neighbor_idx, cf)
                    tentative_g = current.g + edge_cost
                    actual_parent = current.idx

                if neighbor_idx not in self._g_score or tentative_g < self._g_score[neighbor_idx]:
                    self._came_from[neighbor_idx] = actual_parent
                    self._g_score[neighbor_idx] = tentative_g

                    h = self._heuristic_grid(neighbor_idx, goal_idx)
                    f = tentative_g + self.config.heuristic_weight * h

                    heapq.heappush(self._open_set, SearchNode(
                        f=f, g=tentative_g, h=h, idx=neighbor_idx, parent=actual_parent
                    ))

        return self._create_result(None, False, cf, cs)

    def _has_los(
        self,
        from_idx: tuple[int, int, int],
        to_idx: tuple[int, int, int],
    ) -> bool:
        """
        Check line-of-sight between two grid cells.

        Uses Bresenham-like line stepping with terrain checks.

        Args:
            from_idx: Source cell
            to_idx: Target cell

        Returns:
            True if clear LOS exists
        """
        from_state = self._grid_to_state(*from_idx)
        to_state = self._grid_to_state(*to_idx)

        # Sample along line
        n_samples = max(
            abs(to_idx[0] - from_idx[0]),
            abs(to_idx[1] - from_idx[1]),
            abs(to_idx[2] - from_idx[2]),
        ) + 1

        for i in range(1, n_samples):
            alpha = i / n_samples
            mid_pos = from_state.pose.position * (1 - alpha) + to_state.pose.position * alpha

            # Check terrain
            if self.terrain_field is not None:
                terrain_elev = self.terrain_field.get_elevation(mid_pos[0], mid_pos[1])
                if mid_pos[2] < terrain_elev + 30:
                    return False

            # Check obstacles
            if self.obstacle_field is not None:
                if self.obstacle_field(mid_pos) < 30:
                    return False

        return True


class DijkstraPlanner(AStarPlanner):
    """
    Dijkstra's Algorithm (A* with h=0)

    Dijkstra's algorithm is A* without a heuristic. It guarantees
    the optimal path but explores more nodes than A*.

    Useful as a baseline and when heuristics are unreliable.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Override heuristic weight to zero
        self.config.heuristic_weight = 0.0

    def _heuristic_grid(
        self,
        idx: tuple[int, int, int],
        goal_idx: tuple[int, int, int],
    ) -> float:
        """Dijkstra: no heuristic."""
        return 0.0
