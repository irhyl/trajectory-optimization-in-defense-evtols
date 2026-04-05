"""
Sampling-Based Path Planners

This module implements sampling-based path planning algorithms:

- **RRT**: Rapidly-exploring Random Tree (fast, probabilistically complete)
- **RRT***: Asymptotically optimal RRT (converges to optimal as samples → ∞)
- **Informed RRT***: Focused sampling using ellipsoidal heuristic

Sampling-based planners are effective for:
- High-dimensional spaces
- Complex obstacle configurations
- When resolution-completeness isn't required

Trade-offs vs Graph-Based:
- (+) Don't require discretization
- (+) Handle continuous spaces naturally
- (+) Scale better to high dimensions
- (-) Solutions are suboptimal initially
- (-) Require post-processing for smoothness

Author: Defense eVTOL Research Team
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Any
import logging

from .base import PathPlanner, PlanningConfig, PlanningResult
from ..core.state import State, Pose, Velocity
from ..core.trajectory import Trajectory
from ..core.cost import CostFunction

logger = logging.getLogger(__name__)


@dataclass
class RRTNode:
    """
    Node in the RRT tree.

    Attributes:
        state: Vehicle state at this node
        parent: Index of parent node
        cost: Cost from root to this node
    """
    state: State
    parent: int | None = None
    cost: float = 0.0

    @property
    def position(self) -> np.ndarray:
        """Node position."""
        return self.state.pose.position


class RRTPlanner(PathPlanner):
    """
    Rapidly-exploring Random Tree (RRT) Planner

    RRT incrementally builds a tree by:
    1. Sample random point in state space
    2. Find nearest node in tree
    3. Extend towards sample (with step limit)
    4. Add new node if collision-free

    Properties:
    - Probabilistically complete (will find path if one exists)
    - Fast in practice for many problems
    - Solutions are not optimal (can be jagged)

    Configuration:
        - step_size: Maximum extension distance per iteration
        - goal_bias: Probability of sampling goal instead of random
        - max_iterations: Maximum tree nodes

    Example:
        >>> planner = RRTPlanner(
        ...     config=PlanningConfig(max_iterations=5000),
        ...     step_size=500.0,
        ...     goal_bias=0.1,
        ... )
        >>> result = planner.plan(start, goal)
    """

    def __init__(
        self,
        config: PlanningConfig | None = None,
        cost_function: CostFunction | None = None,
        constraints: Any | None = None,
        bounds: tuple[float, float, float, float] | None = None,
        step_size: float = 500.0,
        goal_bias: float = 0.1,
        terrain_field: Any | None = None,
        threat_field: Any | None = None,
    ):
        """
        Initialize RRT planner.

        Args:
            config: Planning configuration
            cost_function: Cost function
            constraints: Constraint set
            bounds: (lat_min, lat_max, lon_min, lon_max)
            step_size: Max extension per step (meters)
            goal_bias: Probability of sampling goal
            terrain_field: Terrain field for collision checking
            threat_field: Threat field for cost evaluation
        """
        super().__init__(config, cost_function, constraints)

        self.bounds = bounds or (26.0, 30.0, 70.0, 78.0)
        self.step_size = step_size
        self.goal_bias = goal_bias
        self.terrain_field = terrain_field
        self.threat_field = threat_field

        # Tree
        self._nodes: list[RRTNode] = []

        # Random generator
        self._rng = np.random.default_rng()

    def plan(
        self,
        start: State,
        goal: State,
        cost_function: CostFunction | None = None,
        constraints: Any | None = None,
    ) -> PlanningResult:
        """
        Plan path using RRT.

        Args:
            start: Starting state
            goal: Goal state
            cost_function: Override cost function
            constraints: Override constraints

        Returns:
            PlanningResult with trajectory
        """
        self._initialize(start, goal)

        cf = cost_function or self.cost_function
        cs = constraints or self.constraints

        # Initialize tree with start
        self._nodes = [RRTNode(state=start, parent=None, cost=0.0)]

        goal_reached = False
        goal_node_idx = None

        if self.config.verbose:
            logger.info(f"RRT planning with step_size={self.step_size}")

        for iteration in range(self.config.max_iterations):
            if self._check_timeout():
                logger.warning("RRT timeout")
                break

            self._stats["iterations"] = iteration + 1

            # Sample random point (with goal bias)
            if self._rng.random() < self.goal_bias:
                sample = goal.pose.position.copy()
            else:
                sample = self._sample_free()

            # Find nearest node
            nearest_idx = self._find_nearest(sample)
            nearest_node = self._nodes[nearest_idx]

            # Steer towards sample
            new_state = self._steer(nearest_node.state, sample)

            # Check if path is collision-free
            if not self._is_collision_free(nearest_node.state, new_state):
                continue

            # Add new node
            new_cost = nearest_node.cost + self._distance(nearest_node.state, new_state)
            new_node = RRTNode(state=new_state, parent=nearest_idx, cost=new_cost)
            self._nodes.append(new_node)
            self._stats["nodes_expanded"] = len(self._nodes)

            # Check if goal reached
            if self._is_goal(new_state, goal):
                goal_reached = True
                goal_node_idx = len(self._nodes) - 1

                if self.config.verbose:
                    logger.info(f"RRT found path at iteration {iteration}")
                break

        if goal_reached and goal_node_idx is not None:
            trajectory = self._extract_path(goal_node_idx, goal)

            if self.config.smoothing:
                trajectory = self._smooth_trajectory(trajectory)

            return self._create_result(trajectory, True, cf, cs)

        logger.warning("RRT failed: goal not reached")
        return self._create_result(None, False, cf, cs)

    def _sample_free(self) -> np.ndarray:
        """
        Sample random point in free space.

        Works for both geodetic (lat, lon, alt) and NED (x, y, z) coordinates.
        Bounds interpretation:
        - Geodetic: (lat_min, lat_max, lon_min, lon_max)
        - NED: (x_min, x_max, y_min, y_max) in meters

        Returns:
            Random position [x, y, alt]
        """
        x_min, x_max, y_min, y_max = self.bounds

        x = self._rng.uniform(x_min, x_max)
        y = self._rng.uniform(y_min, y_max)
        alt = self._rng.uniform(self.config.altitude_min, self.config.altitude_max)

        return np.array([x, y, alt])

    def _find_nearest(self, sample: np.ndarray) -> int:
        """
        Find nearest node to sample using Euclidean distance.

        For NED coordinates, distances are already in meters.
        For geodetic, this is an approximation.

        Args:
            sample: Sample position [x, y, alt]

        Returns:
            Index of nearest node
        """
        min_dist = float('inf')
        nearest_idx = 0

        for i, node in enumerate(self._nodes):
            diff = node.position - sample
            dist = np.linalg.norm(diff)  # Direct Euclidean for NED (meters)

            if dist < min_dist:
                min_dist = dist
                nearest_idx = i

        return nearest_idx

    def _steer(self, from_state: State, to_point: np.ndarray) -> State:
        """
        Steer from state towards point, limited by step size.

        For NED coordinates, step_size is directly in meters.

        Args:
            from_state: Starting state
            to_point: Target point [x, y, alt]

        Returns:
            New state at most step_size away (in meters)
        """
        direction = to_point - from_state.pose.position
        dist_meters = np.linalg.norm(direction)  # Direct Euclidean for NED

        if dist_meters <= self.step_size:
            new_pos = to_point
            actual_dist = dist_meters
        else:
            # Scale direction to step_size
            scale = self.step_size / max(dist_meters, 0.001)
            new_pos = from_state.pose.position + direction * scale
            actual_dist = self.step_size

        # Ensure minimum time increment to avoid zero-duration segments
        speed = 40.0  # Assume 40 m/s cruise speed
        time_increment = max(actual_dist / speed, 0.1)  # At least 0.1 second

        # Calculate energy with clamping to avoid floating-point underflow
        new_energy = max(0.0, min(1.0, from_state.energy - actual_dist * 0.00001))

        # Create new state (preserve original frame)
        return State(
            pose=Pose(new_pos, np.zeros(3), frame=from_state.pose.frame),
            velocity=Velocity(direction / max(dist_meters, 0.001) * speed),
            energy=new_energy,
            time=from_state.time + time_increment,
        )

    def _is_collision_free(self, from_state: State, to_state: State) -> bool:
        """
        Check if path between states is collision-free.

        Args:
            from_state: Start of path
            to_state: End of path

        Returns:
            True if path is clear
        """
        # Sample along path
        n_samples = max(5, int(self._distance(from_state, to_state) / 100))

        for i in range(n_samples):
            alpha = i / max(1, n_samples - 1)
            pos = from_state.pose.position * (1 - alpha) + to_state.pose.position * alpha

            # Terrain check
            if self.terrain_field is not None:
                terrain_elev = self.terrain_field.get_elevation(pos[0], pos[1])
                if pos[2] < terrain_elev + 30:
                    return False

            # Altitude bounds
            if pos[2] < self.config.altitude_min or pos[2] > self.config.altitude_max:
                return False

        return True

    def _extract_path(self, goal_idx: int, goal: State) -> Trajectory:
        """
        Extract path from tree.

        Args:
            goal_idx: Index of goal node
            goal: Original goal state

        Returns:
            Trajectory from start to goal
        """
        path = []
        idx = goal_idx

        while idx is not None:
            path.append(self._nodes[idx].state)
            idx = self._nodes[idx].parent

        path.reverse()

        # Replace last with actual goal, ensuring time is strictly increasing
        if path:
            last_time = path[-1].time
            # Compute time to goal based on distance
            dist_to_goal = path[-1].distance_to(goal) if len(path) > 0 else 0
            time_to_goal = max(dist_to_goal / 40.0, 0.1)  # At least 0.1 second

            # Create goal state with proper time
            goal_with_time = State(
                pose=goal.pose,
                velocity=goal.velocity,
                energy=goal.energy,
                time=last_time + time_to_goal,
            )
            path[-1] = goal_with_time

        # Ensure all times are strictly increasing (fix any duplicates)
        for i in range(1, len(path)):
            if path[i].time <= path[i-1].time:
                # Increment time slightly
                path[i] = State(
                    pose=path[i].pose,
                    velocity=path[i].velocity,
                    energy=path[i].energy,
                    time=path[i-1].time + 0.1,
                )

        return Trajectory(states=path)


class RRTStarPlanner(RRTPlanner):
    """
    RRT* (Optimal RRT) Planner

    RRT* extends RRT with two key modifications:

    1. **Near neighbors**: When adding a node, check all nodes within
       a radius (not just nearest) for a better parent.

    2. **Rewiring**: After adding a node, check if it provides a better
       path to nearby nodes and rewire if so.

    These modifications make RRT* asymptotically optimal: as the number
    of samples approaches infinity, the solution converges to optimal.

    Trade-off: More expensive per iteration than RRT, but produces
    much better paths.

    The neighbor radius shrinks as the tree grows:
        r_n = min(γ × (log(n)/n)^(1/d), step_size)

    where γ is a constant based on state space volume and d is dimension.
    """

    def __init__(
        self,
        config: PlanningConfig | None = None,
        cost_function: CostFunction | None = None,
        constraints: Any | None = None,
        bounds: tuple[float, float, float, float] | None = None,
        step_size: float = 500.0,
        goal_bias: float = 0.1,
        terrain_field: Any | None = None,
        threat_field: Any | None = None,
        gamma: float = 1.5,
    ):
        """
        Initialize RRT* planner.

        Args:
            gamma: Radius constant for near neighbor search
        """
        super().__init__(
            config, cost_function, constraints, bounds,
            step_size, goal_bias, terrain_field, threat_field
        )
        self.gamma = gamma

    def plan(
        self,
        start: State,
        goal: State,
        cost_function: CostFunction | None = None,
        constraints: Any | None = None,
    ) -> PlanningResult:
        """
        Plan path using RRT*.

        Args:
            start: Starting state
            goal: Goal state
            cost_function: Override cost function
            constraints: Override constraints

        Returns:
            PlanningResult with trajectory
        """
        self._initialize(start, goal)

        cf = cost_function or self.cost_function
        cs = constraints or self.constraints

        # Initialize tree
        self._nodes = [RRTNode(state=start, parent=None, cost=0.0)]

        goal_reached = False
        best_goal_idx = None
        best_goal_cost = float('inf')

        if self.config.verbose:
            logger.info(f"RRT* planning with step_size={self.step_size}")

        for iteration in range(self.config.max_iterations):
            if self._check_timeout():
                break

            self._stats["iterations"] = iteration + 1

            # Sample
            if self._rng.random() < self.goal_bias:
                sample = goal.pose.position.copy()
            else:
                sample = self._sample_free()

            # Find nearest
            nearest_idx = self._find_nearest(sample)
            nearest_node = self._nodes[nearest_idx]

            # Steer
            new_state = self._steer(nearest_node.state, sample)

            if not self._is_collision_free(nearest_node.state, new_state):
                continue

            # Find near neighbors
            radius = self._get_neighbor_radius()
            near_indices = self._find_near(new_state.pose.position, radius)

            # Choose best parent from near neighbors
            best_parent_idx = nearest_idx
            best_cost = nearest_node.cost + self._distance(nearest_node.state, new_state)

            for near_idx in near_indices:
                near_node = self._nodes[near_idx]
                if self._is_collision_free(near_node.state, new_state):
                    cost = near_node.cost + self._distance(near_node.state, new_state)
                    if cost < best_cost:
                        best_cost = cost
                        best_parent_idx = near_idx

            # Add new node
            new_node = RRTNode(state=new_state, parent=best_parent_idx, cost=best_cost)
            new_idx = len(self._nodes)
            self._nodes.append(new_node)

            # Rewire near neighbors
            for near_idx in near_indices:
                near_node = self._nodes[near_idx]
                if near_idx == best_parent_idx:
                    continue

                if self._is_collision_free(new_state, near_node.state):
                    new_cost = new_node.cost + self._distance(new_state, near_node.state)
                    if new_cost < near_node.cost:
                        self._nodes[near_idx].parent = new_idx
                        self._nodes[near_idx].cost = new_cost

            self._stats["nodes_expanded"] = len(self._nodes)

            # Check goal
            if self._is_goal(new_state, goal):
                goal_reached = True
                if best_cost < best_goal_cost:
                    best_goal_idx = new_idx
                    best_goal_cost = best_cost

                    if self.config.verbose:
                        logger.info(f"RRT* found path with cost {best_cost:.1f}")

        if goal_reached and best_goal_idx is not None:
            trajectory = self._extract_path(best_goal_idx, goal)

            if self.config.smoothing:
                trajectory = self._smooth_trajectory(trajectory)

            return self._create_result(trajectory, True, cf, cs)

        logger.warning("RRT* failed: goal not reached")
        return self._create_result(None, False, cf, cs)

    def _get_neighbor_radius(self) -> float:
        """
        Compute neighbor radius for current tree size.

        r_n = min(γ × (log(n)/n)^(1/d), step_size)
        """
        n = len(self._nodes)
        d = 3  # 3D space

        if n < 2:
            return self.step_size

        radius = self.gamma * (np.log(n) / n) ** (1.0 / d)

        # Scale by approximate domain size
        lat_range = self.bounds[1] - self.bounds[0]
        scale = lat_range * 111000  # Convert to meters
        radius *= scale

        return min(radius, self.step_size * 2)

    def _find_near(self, point: np.ndarray, radius: float) -> list[int]:
        """
        Find all nodes within radius of point.

        For NED coordinates, radius is directly in meters.

        Args:
            point: Query point [x, y, alt]
            radius: Search radius in meters

        Returns:
            List of node indices
        """
        near = []

        for i, node in enumerate(self._nodes):
            diff = node.position - point
            dist = np.linalg.norm(diff)  # Direct Euclidean for NED

            if dist <= radius:
                near.append(i)

        return near
