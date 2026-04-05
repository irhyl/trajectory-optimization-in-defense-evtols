"""
Cost Function Framework

This module defines the cost function framework for trajectory optimization.
Cost functions are modular, composable, and differentiable where possible.

Mathematical Framework
======================

The total cost is a weighted sum (or Pareto composition) of objectives:

    J(τ) = Σᵢ wᵢ · Jᵢ(τ)

where each Jᵢ is computed by integrating along the trajectory:

    Jᵢ(τ) = ∫₀ᵀ Lᵢ(x(t), u(t)) dt + Φᵢ(x(T))

- Lᵢ: Running cost (Lagrangian)
- Φᵢ: Terminal cost

Standard Objectives
-------------------

1. **Time**: J_time = T (minimize mission duration)

2. **Energy**: J_energy = ∫ P(v, climb) dt (battery consumption)

3. **Risk**: J_risk = ∫ P_threat(x) dt (integrated threat exposure)

4. **Path Length**: J_length = ∫ ||v|| dt (total distance)

5. **Smoothness**: J_smooth = ∫ ||u̇||² dt (control rate of change)

6. **Terminal**: Φ = ||x(T) - x_goal||² (goal deviation)

For multi-objective optimization (NSGA-III), we don't scalarize but instead
find the Pareto frontier of non-dominated solutions.

Author: Defense eVTOL Research Team
"""

from __future__ import annotations

import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from collections.abc import Callable
from enum import Enum
import logging

from .trajectory import Trajectory

logger = logging.getLogger(__name__)


class ObjectiveType(Enum):
    """Type of optimization objective."""
    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"
    TARGET = "target"  # Minimize deviation from target value


@dataclass
class CostComponent:
    """
    A single component of the cost function.

    Each component has:
    - Name for identification
    - Weight for scalarization
    - Type (minimize/maximize/target)
    - Optional bounds for normalization

    Attributes:
        name: Component identifier
        weight: Weight in scalarized cost
        objective_type: Optimization direction
        bounds: (min, max) for normalization
        target: Target value (for TARGET type)
    """
    name: str
    weight: float = 1.0
    objective_type: ObjectiveType = ObjectiveType.MINIMIZE
    bounds: tuple[float, float] | None = None
    target: float | None = None

    def normalize(self, value: float) -> float:
        """
        Normalize value to [0, 1] range.

        Args:
            value: Raw cost value

        Returns:
            Normalized value
        """
        if self.bounds is None:
            return value

        min_val, max_val = self.bounds
        if max_val <= min_val:
            return 0.0

        return (value - min_val) / (max_val - min_val)


class CostFunction(ABC):
    """
    Abstract base class for cost functions.

    Cost functions evaluate trajectories and return scalar costs.
    They should be:
    - Deterministic
    - Non-negative (for most cases)
    - Differentiable (when possible, for gradient-based optimization)

    Subclasses must implement:
    - evaluate(): Compute cost for a trajectory
    - gradient(): (optional) Compute cost gradient
    """

    @abstractmethod
    def evaluate(self, trajectory: Trajectory) -> float:
        """
        Evaluate cost for a trajectory.

        Args:
            trajectory: Input trajectory

        Returns:
            Cost value (scalar)
        """
        pass

    def gradient(self, trajectory: Trajectory) -> np.ndarray | None:
        """
        Compute gradient of cost with respect to trajectory parameters.

        Args:
            trajectory: Input trajectory

        Returns:
            Gradient array or None if not differentiable
        """
        return None

    @property
    def name(self) -> str:
        """Cost function name."""
        return self.__class__.__name__

    @property
    def is_differentiable(self) -> bool:
        """Whether gradient is available."""
        return False


class TimeCost(CostFunction):
    """
    Time cost: Minimize total trajectory duration.

    J_time = T = t_final - t_initial
    """

    def evaluate(self, trajectory: Trajectory) -> float:
        return trajectory.duration

    def gradient(self, trajectory: Trajectory) -> np.ndarray:
        """Gradient w.r.t. time at each state."""
        n = len(trajectory)
        grad = np.zeros(n)
        grad[-1] = 1.0   # dJ/dt_final = 1
        grad[0] = -1.0   # dJ/dt_initial = -1
        return grad

    @property
    def is_differentiable(self) -> bool:
        return True


class EnergyCost(CostFunction):
    """
    Energy cost: Minimize battery consumption.

    J_energy = E_initial - E_final = ∫ P(v, climb) dt

    We use a simplified power model:
    P = P_hover + P_cruise(v) + P_climb(climb_rate)
    """

    def __init__(
        self,
        power_hover: float = 15000.0,     # Hover power (W)
        power_cruise_coeff: float = 50.0,  # Cruise power coefficient (W/(m/s)²)
        power_climb_coeff: float = 2000.0, # Climb power coefficient (W/(m/s))
        battery_capacity: float = 50.0,    # Battery capacity (kWh)
    ):
        self.power_hover = power_hover
        self.power_cruise_coeff = power_cruise_coeff
        self.power_climb_coeff = power_climb_coeff
        self.battery_capacity = battery_capacity * 3600 * 1000  # Convert to Joules

    def evaluate(self, trajectory: Trajectory) -> float:
        """Compute energy consumption in fraction of battery."""
        total_energy = 0.0

        for segment in trajectory.segments:
            # Average speed over segment
            speed = segment.average_speed

            # Climb rate
            climb_rate = segment.altitude_change / segment.duration if segment.duration > 0 else 0

            # Power model: quadratic in speed, linear in climb
            power = (
                self.power_hover +
                self.power_cruise_coeff * speed ** 2 +
                self.power_climb_coeff * abs(climb_rate)
            )

            # Energy = Power × Time
            total_energy += power * segment.duration

        # Return as fraction of battery capacity
        return total_energy / self.battery_capacity


class PathLengthCost(CostFunction):
    """
    Path length cost: Minimize total distance traveled.

    J_length = Σᵢ ||xᵢ₊₁ - xᵢ||
    """

    def __init__(self, normalize_by: float = 1000.0):
        """
        Args:
            normalize_by: Distance normalization factor (meters)
        """
        self.normalize_by = normalize_by

    def evaluate(self, trajectory: Trajectory) -> float:
        return trajectory.distance / self.normalize_by


class ThreatExposureCost(CostFunction):
    """
    Threat exposure cost: Minimize integrated exposure to threats.

    J_threat = ∫ Σⱼ P_hit(x, threat_j) dt

    This requires a threat field that maps position → threat probability.
    """

    def __init__(
        self,
        threat_field: Callable[[np.ndarray], float] | None = None,
        threat_positions: np.ndarray | None = None,
        threat_ranges: np.ndarray | None = None,
        lethal_prob_max: float = 0.9,
    ):
        """
        Args:
            threat_field: Function (position) -> threat probability
            threat_positions: Nx3 array of threat positions (if no field provided)
            threat_ranges: N array of threat effective ranges
            lethal_prob_max: Maximum kill probability at threat center
        """
        self.threat_field = threat_field
        self.threat_positions = threat_positions if threat_positions is not None else np.array([])
        self.threat_ranges = threat_ranges if threat_ranges is not None else np.array([])
        self.lethal_prob_max = lethal_prob_max

    def _default_threat_field(self, position: np.ndarray) -> float:
        """
        Default threat field using threat positions and ranges.

        Uses Gaussian decay: P(x) = P_max × exp(-||x - x_threat||² / (2σ²))
        """
        if len(self.threat_positions) == 0:
            return 0.0

        total_prob = 0.0
        for i, threat_pos in enumerate(self.threat_positions):
            dist = np.linalg.norm(position - threat_pos)
            sigma = self.threat_ranges[i] / 2.0 if i < len(self.threat_ranges) else 5000.0

            # Gaussian threat model
            prob = self.lethal_prob_max * np.exp(-dist**2 / (2 * sigma**2))

            # Aggregate using union of independent events
            total_prob = 1.0 - (1.0 - total_prob) * (1.0 - prob)

        return total_prob

    def evaluate(self, trajectory: Trajectory) -> float:
        """Compute integrated threat exposure."""
        if self.threat_field is None:
            field = self._default_threat_field
        else:
            field = self.threat_field

        total_exposure = 0.0

        for segment in trajectory.segments:
            # Sample threat at segment midpoint
            mid_pos = (segment.start.pose.position + segment.end.pose.position) / 2
            prob = field(mid_pos)

            # Integrate: exposure × time
            total_exposure += prob * segment.duration

        return total_exposure


class SmoothnessCost(CostFunction):
    """
    Smoothness cost: Penalize jerky trajectories.

    J_smooth = Σᵢ ||aᵢ₊₁ - aᵢ||² (acceleration changes)

    Smooth trajectories are:
    - More comfortable
    - More energy efficient
    - Easier to track
    """

    def __init__(self, weight: float = 1.0):
        self.weight = weight

    def evaluate(self, trajectory: Trajectory) -> float:
        if len(trajectory) < 3:
            return 0.0

        # Compute accelerations
        positions = trajectory.get_positions()
        times = trajectory.get_times()

        # First derivatives (velocities)
        velocities = np.diff(positions, axis=0)
        dt = np.diff(times)
        dt = np.where(dt > 0, dt, 1.0)  # Avoid division by zero
        velocities = velocities / dt[:, np.newaxis]

        # Second derivatives (accelerations)
        accelerations = np.diff(velocities, axis=0)
        dt2 = (dt[:-1] + dt[1:]) / 2
        dt2 = np.where(dt2 > 0, dt2, 1.0)
        accelerations = accelerations / dt2[:, np.newaxis]

        # Sum of squared acceleration changes (jerk-like metric)
        jerk = np.diff(accelerations, axis=0)
        return self.weight * np.sum(jerk ** 2)


class TerminalCost(CostFunction):
    """
    Terminal cost: Penalize deviation from goal state.

    Φ(x(T)) = w_pos ||p(T) - p_goal||² + w_vel ||v(T) - v_goal||²
    """

    def __init__(
        self,
        goal_position: np.ndarray,
        goal_velocity: np.ndarray | None = None,
        position_weight: float = 1.0,
        velocity_weight: float = 0.1,
    ):
        self.goal_position = np.asarray(goal_position)
        self.goal_velocity = np.asarray(goal_velocity) if goal_velocity is not None else np.zeros(3)
        self.position_weight = position_weight
        self.velocity_weight = velocity_weight

    def evaluate(self, trajectory: Trajectory) -> float:
        final_state = trajectory.end_state

        # Position error
        pos_error = np.linalg.norm(final_state.pose.position - self.goal_position)

        # Velocity error
        vel_error = np.linalg.norm(final_state.velocity.linear - self.goal_velocity)

        return (
            self.position_weight * pos_error ** 2 +
            self.velocity_weight * vel_error ** 2
        )


class DetectionCost(CostFunction):
    """
    Detection cost: Minimize probability of radar detection.

    Uses RCS and radar equation:
    P_detect ∝ (RCS × P_tx) / R⁴

    This encourages:
    - Terrain masking
    - Low-RCS approach angles
    - Maximum range from radars
    """

    def __init__(
        self,
        radar_positions: np.ndarray,
        radar_powers: np.ndarray | None = None,
        rcs_model: Callable[[float], float] | None = None,  # aspect -> RCS
    ):
        """
        Args:
            radar_positions: Nx3 array of radar positions
            radar_powers: N array of transmitted powers (normalized)
            rcs_model: Function (aspect_angle) -> RCS in m²
        """
        self.radar_positions = np.asarray(radar_positions)
        self.radar_powers = np.asarray(radar_powers) if radar_powers is not None else np.ones(len(radar_positions))
        self.rcs_model = rcs_model or (lambda a: 1.0)  # Default: constant RCS

    def evaluate(self, trajectory: Trajectory) -> float:
        total_detection = 0.0

        for segment in trajectory.segments:
            pos = (segment.start.pose.position + segment.end.pose.position) / 2

            for i, radar_pos in enumerate(self.radar_positions):
                # Range to radar
                R = np.linalg.norm(pos - radar_pos)
                if R < 100:  # Avoid singularity
                    R = 100

                # Aspect angle (simplified)
                heading = segment.end.pose.position - segment.start.pose.position
                to_radar = radar_pos - pos
                if np.linalg.norm(heading) > 0 and np.linalg.norm(to_radar) > 0:
                    aspect = np.arccos(
                        np.clip(np.dot(heading, to_radar) /
                               (np.linalg.norm(heading) * np.linalg.norm(to_radar)), -1, 1)
                    )
                else:
                    aspect = 0

                # RCS
                rcs = self.rcs_model(aspect)

                # Detection metric (radar equation simplified)
                detection = (self.radar_powers[i] * rcs) / (R ** 4)
                total_detection += detection * segment.duration

        return total_detection


@dataclass
class CompositeCost:
    """
    Composite cost function combining multiple objectives.

    Supports:
    - Weighted sum scalarization
    - Individual objective evaluation (for Pareto)
    - Constraint handling

    Attributes:
        costs: List of (CostFunction, weight) pairs
        name: Composite cost name
    """
    costs: list[tuple[CostFunction, float]]
    name: str = "composite"

    def add_cost(self, cost: CostFunction, weight: float = 1.0) -> None:
        """Add a cost component."""
        self.costs.append((cost, weight))

    def evaluate(self, trajectory: Trajectory) -> float:
        """Evaluate weighted sum of all costs."""
        total = 0.0
        for cost, weight in self.costs:
            total += weight * cost.evaluate(trajectory)
        return total

    def evaluate_objectives(self, trajectory: Trajectory) -> dict[str, float]:
        """Evaluate each objective individually."""
        return {cost.name: cost.evaluate(trajectory) for cost, _ in self.costs}

    def evaluate_vector(self, trajectory: Trajectory) -> np.ndarray:
        """Return objectives as vector (for multi-objective optimization)."""
        return np.array([cost.evaluate(trajectory) for cost, _ in self.costs])

    @property
    def n_objectives(self) -> int:
        """Number of objectives."""
        return len(self.costs)

    @property
    def objective_names(self) -> list[str]:
        """Names of all objectives."""
        return [cost.name for cost, _ in self.costs]


def create_default_cost() -> CompositeCost:
    """
    Create default cost function for defense eVTOL planning.

    Default objectives:
    1. Time (weight=1.0)
    2. Energy (weight=0.5)
    3. Path length (weight=0.3)
    4. Smoothness (weight=0.1)

    Returns:
        CompositeCost with standard objectives
    """
    return CompositeCost(
        costs=[
            (TimeCost(), 1.0),
            (EnergyCost(), 0.5),
            (PathLengthCost(), 0.3),
            (SmoothnessCost(), 0.1),
        ],
        name="default_defense_evtol",
    )


def create_stealth_cost(
    threat_positions: np.ndarray,
    threat_ranges: np.ndarray,
    radar_positions: np.ndarray,
) -> CompositeCost:
    """
    Create cost function optimized for stealth operations.

    Prioritizes:
    - Threat avoidance
    - Radar detection minimization
    - Time (secondary)
    - Energy (tertiary)

    Args:
        threat_positions: Threat locations
        threat_ranges: Threat effective ranges
        radar_positions: Radar locations

    Returns:
        CompositeCost for stealth missions
    """
    return CompositeCost(
        costs=[
            (ThreatExposureCost(
                threat_positions=threat_positions,
                threat_ranges=threat_ranges,
            ), 2.0),
            (DetectionCost(radar_positions=radar_positions), 1.5),
            (TimeCost(), 0.5),
            (EnergyCost(), 0.3),
        ],
        name="stealth_mission",
    )
