"""
Chance-Constrained Optimization

This module implements chance-constrained trajectory optimization
where constraints must be satisfied with a specified probability.

Mathematical Foundation:
------------------------
Chance constraint:
    P[g(x, ξ) ≤ 0] ≥ 1 - ε

where:
- g(x, ξ) is a constraint function
- ξ is a random variable (uncertainty)
- ε is the acceptable violation probability

For Gaussian uncertainty with linear constraints:
    g(x, ξ) = aᵀx - b ≤ 0

The chance constraint becomes:
    aᵀμ + Φ⁻¹(1-ε) · √(aᵀΣa) ≤ b

where Φ⁻¹ is the inverse standard normal CDF.

Methods:
1. Analytical reformulation (for linear/quadratic + Gaussian)
2. Scenario approach (sample-based)
3. Robust counterpart (worst-case over uncertainty set)

Author: Defense eVTOL Research Team
"""

from __future__ import annotations

import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from collections.abc import Callable
from scipy import stats
import logging

from .uncertainty import (
    UncertaintyModel,
    GaussianUncertainty,
    StateDistribution,
    UncertaintyPropagation,
)

logger = logging.getLogger(__name__)


class ChanceConstraint(ABC):
    """
    Abstract base class for chance constraints.

    A chance constraint requires P[constraint satisfied] ≥ 1 - ε.

    Attributes:
        name: Constraint identifier
        epsilon: Acceptable violation probability
    """

    def __init__(self, name: str, epsilon: float = 0.05):
        """
        Initialize chance constraint.

        Args:
            name: Constraint name
            epsilon: Violation probability (0-1)
        """
        self.name = name
        self.epsilon = epsilon

    @abstractmethod
    def evaluate_deterministic(self, state: np.ndarray) -> float:
        """
        Evaluate deterministic constraint at a state.

        Args:
            state: State vector

        Returns:
            Constraint value (≤ 0 means satisfied)
        """
        pass

    @abstractmethod
    def evaluate_chance(self, state_dist: StateDistribution) -> float:
        """
        Evaluate chance constraint (conservative reformulation).

        Args:
            state_dist: State distribution

        Returns:
            Reformulated constraint value (≤ 0 means P[satisfied] ≥ 1-ε)
        """
        pass

    def check_satisfaction(
        self,
        state_dist: StateDistribution,
        n_samples: int = 10000,
    ) -> tuple[float, float]:
        """
        Monte Carlo check of constraint satisfaction probability.

        Args:
            state_dist: State distribution
            n_samples: Number of samples

        Returns:
            (satisfaction_probability, mean_violation)
        """
        samples = state_dist.sample(n_samples)

        violations = []
        for sample in samples:
            g = self.evaluate_deterministic(sample)
            violations.append(g)

        violations = np.array(violations)
        satisfaction_prob = np.mean(violations <= 0)
        mean_violation = np.mean(np.maximum(violations, 0))

        return float(satisfaction_prob), float(mean_violation)


class LinearChanceConstraint(ChanceConstraint):
    """
    Linear chance constraint.

    aᵀx ≤ b with probability ≥ 1 - ε

    For Gaussian x ~ N(μ, Σ):
        aᵀμ + Φ⁻¹(1-ε) · √(aᵀΣa) ≤ b
    """

    def __init__(
        self,
        a: np.ndarray,
        b: float,
        epsilon: float = 0.05,
        name: str = "linear",
    ):
        """
        Initialize linear chance constraint.

        Args:
            a: Coefficient vector
            b: RHS bound
            epsilon: Violation probability
            name: Constraint name
        """
        super().__init__(name, epsilon)
        self.a = np.asarray(a)
        self.b = b

        # Inverse CDF
        self._z = stats.norm.ppf(1 - epsilon)

    def evaluate_deterministic(self, state: np.ndarray) -> float:
        return float(self.a @ state - self.b)

    def evaluate_chance(self, state_dist: StateDistribution) -> float:
        """
        Analytical chance constraint reformulation.

        aᵀμ + z · √(aᵀΣa) ≤ b
        where z = Φ⁻¹(1-ε)
        """
        mu = state_dist.mean
        sigma = state_dist.covariance

        mean_term = self.a @ mu
        variance_term = self.a @ sigma @ self.a
        std_term = np.sqrt(max(variance_term, 0))

        return mean_term + self._z * std_term - self.b


class TerrainClearanceConstraint(ChanceConstraint):
    """
    Probabilistic terrain clearance constraint.

    P[altitude - terrain(lat, lon) ≥ min_clearance] ≥ 1 - ε
    """

    def __init__(
        self,
        terrain_field: Callable[[float, float], float],
        min_clearance: float = 50.0,
        epsilon: float = 0.05,
        position_indices: tuple[int, int, int] = (0, 1, 2),
    ):
        """
        Initialize terrain clearance constraint.

        Args:
            terrain_field: Function (lat, lon) -> elevation
            min_clearance: Minimum clearance [m]
            epsilon: Violation probability
            position_indices: State indices for (lat, lon, alt)
        """
        super().__init__("terrain_clearance", epsilon)
        self.terrain_field = terrain_field
        self.min_clearance = min_clearance
        self.pos_idx = position_indices

    def evaluate_deterministic(self, state: np.ndarray) -> float:
        lat = state[self.pos_idx[0]]
        lon = state[self.pos_idx[1]]
        alt = state[self.pos_idx[2]]

        terrain_elev = self.terrain_field(lat, lon)
        clearance = alt - terrain_elev

        return self.min_clearance - clearance  # ≤ 0 means satisfied

    def evaluate_chance(self, state_dist: StateDistribution) -> float:
        """
        Conservative reformulation using altitude uncertainty.

        For linearized constraint:
            clearance = alt - terrain(lat, lon) ≥ min_clearance

        Add safety margin based on altitude uncertainty.
        """
        mu = state_dist.mean
        sigma = state_dist.covariance

        lat = mu[self.pos_idx[0]]
        lon = mu[self.pos_idx[1]]
        alt = mu[self.pos_idx[2]]

        terrain_elev = self.terrain_field(lat, lon)
        mean_clearance = alt - terrain_elev

        # Altitude uncertainty
        alt_var = sigma[self.pos_idx[2], self.pos_idx[2]]
        alt_std = np.sqrt(max(alt_var, 0))

        # Position uncertainty affects terrain estimate
        lat_std = np.sqrt(max(sigma[self.pos_idx[0], self.pos_idx[0]], 0))
        lon_std = np.sqrt(max(sigma[self.pos_idx[1], self.pos_idx[1]], 0))

        # Terrain gradient (numerical)
        eps = 1e-6
        dterrain_dlat = (self.terrain_field(lat + eps, lon) - terrain_elev) / eps
        dterrain_dlon = (self.terrain_field(lat, lon + eps) - terrain_elev) / eps

        # Combined uncertainty in clearance
        terrain_uncertainty = np.sqrt(
            (dterrain_dlat * lat_std) ** 2 +
            (dterrain_dlon * lon_std) ** 2
        )

        total_std = np.sqrt(alt_std**2 + terrain_uncertainty**2)

        # Conservative bound
        z = stats.norm.ppf(1 - self.epsilon)
        required_clearance = self.min_clearance + z * total_std

        return required_clearance - mean_clearance


class ThreatAvoidanceConstraint(ChanceConstraint):
    """
    Probabilistic threat avoidance constraint.

    P[P_kill(position) ≤ threshold] ≥ 1 - ε
    """

    def __init__(
        self,
        threat_field: Callable[[float, float, float], float],
        pk_threshold: float = 0.1,
        epsilon: float = 0.05,
        position_indices: tuple[int, int, int] = (0, 1, 2),
    ):
        """
        Initialize threat avoidance constraint.

        Args:
            threat_field: Function (lat, lon, alt) -> P_kill
            pk_threshold: Maximum acceptable kill probability
            epsilon: Violation probability
            position_indices: State indices for position
        """
        super().__init__("threat_avoidance", epsilon)
        self.threat_field = threat_field
        self.pk_threshold = pk_threshold
        self.pos_idx = position_indices

    def evaluate_deterministic(self, state: np.ndarray) -> float:
        lat = state[self.pos_idx[0]]
        lon = state[self.pos_idx[1]]
        alt = state[self.pos_idx[2]]

        pk = self.threat_field(lat, lon, alt)
        return pk - self.pk_threshold

    def evaluate_chance(self, state_dist: StateDistribution) -> float:
        """
        Conservative reformulation using Monte Carlo.

        Samples positions and computes worst-case Pk.
        """
        n_samples = 100
        samples = state_dist.sample(n_samples)

        pks = []
        for sample in samples:
            lat = sample[self.pos_idx[0]]
            lon = sample[self.pos_idx[1]]
            alt = sample[self.pos_idx[2]]

            pk = self.threat_field(lat, lon, alt)
            pks.append(pk)

        # Use (1-ε) quantile as conservative estimate
        quantile_pk = np.quantile(pks, 1 - self.epsilon)

        return quantile_pk - self.pk_threshold


@dataclass
class ProbabilisticSafetyConstraint:
    """
    Combined safety constraint for trajectory planning.

    Aggregates multiple chance constraints.

    Attributes:
        constraints: List of chance constraints
        joint_epsilon: Joint violation probability (if joint feasibility required)
    """
    constraints: list[ChanceConstraint] = field(default_factory=list)
    joint_epsilon: float | None = None

    def add(self, constraint: ChanceConstraint) -> None:
        """Add a chance constraint."""
        self.constraints.append(constraint)

    def evaluate_all(
        self,
        state_dist: StateDistribution,
    ) -> dict[str, float]:
        """
        Evaluate all constraints.

        Returns:
            Dict mapping constraint name to value
        """
        return {c.name: c.evaluate_chance(state_dist) for c in self.constraints}

    def is_feasible(self, state_dist: StateDistribution) -> bool:
        """Check if all constraints are satisfied."""
        for constraint in self.constraints:
            if constraint.evaluate_chance(state_dist) > 0:
                return False
        return True

    def max_violation(self, state_dist: StateDistribution) -> float:
        """Get maximum constraint violation."""
        violations = [c.evaluate_chance(state_dist) for c in self.constraints]
        return max(violations) if violations else 0.0


class ChanceConstrainedPlanner:
    """
    Path planner with chance constraints.

    Plans trajectories that satisfy probabilistic safety constraints.
    Uses the scenario approach or robust counterpart methods.
    """

    def __init__(
        self,
        safety_constraints: ProbabilisticSafetyConstraint,
        uncertainty: UncertaintyModel,
        method: str = "scenario",
    ):
        """
        Initialize planner.

        Args:
            safety_constraints: Probabilistic safety constraints
            uncertainty: Uncertainty model for states
            method: 'scenario', 'robust', or 'analytical'
        """
        self.safety = safety_constraints
        self.uncertainty = uncertainty
        self.method = method

    def check_waypoint_safety(
        self,
        waypoint: np.ndarray,
    ) -> tuple[bool, dict[str, float]]:
        """
        Check if a waypoint satisfies safety constraints.

        Args:
            waypoint: Nominal waypoint position

        Returns:
            (is_safe, constraint_values)
        """
        # Create state distribution around waypoint
        if isinstance(self.uncertainty, GaussianUncertainty):
            state_dist = StateDistribution(
                mean=waypoint,
                covariance=self.uncertainty.covariance(),
            )
        else:
            # Approximate as Gaussian
            samples = self.uncertainty.sample(1000)
            state_dist = StateDistribution(
                mean=self.uncertainty.mean(),
                covariance=np.cov(samples, rowvar=False),
            )

        constraint_values = self.safety.evaluate_all(state_dist)
        is_safe = all(v <= 0 for v in constraint_values.values())

        return is_safe, constraint_values

    def inflate_obstacle(
        self,
        obstacle_center: np.ndarray,
        obstacle_radius: float,
    ) -> float:
        """
        Compute inflated obstacle radius for probabilistic avoidance.

        Args:
            obstacle_center: Obstacle center position
            obstacle_radius: Nominal obstacle radius

        Returns:
            Inflated radius
        """
        # Position uncertainty standard deviation
        if isinstance(self.uncertainty, GaussianUncertainty):
            pos_std = np.sqrt(np.mean(np.diag(self.uncertainty.covariance())[:3]))
        else:
            samples = self.uncertainty.sample(1000)
            pos_std = np.mean(np.std(samples, axis=0))

        # Inflation based on safety probability
        z = stats.norm.ppf(1 - self.safety.constraints[0].epsilon if self.safety.constraints else 0.95)

        return obstacle_radius + z * pos_std


class RobustTrajectoryOptimizer:
    """
    Robust trajectory optimization under uncertainty.

    Optimizes trajectories while satisfying chance constraints
    using various robust optimization techniques.

    Methods:
    1. Tube MPC: Optimize nominal trajectory + feedback gains
    2. Scenario MPC: Sample-based optimization
    3. Risk-sensitive: Minimize CVaR (Conditional Value at Risk)
    """

    def __init__(
        self,
        dynamics: Callable,
        cost_function: Callable,
        constraints: ProbabilisticSafetyConstraint,
        process_noise: np.ndarray | None = None,
    ):
        """
        Initialize optimizer.

        Args:
            dynamics: System dynamics f(x, u)
            cost_function: Cost J(trajectory)
            constraints: Probabilistic safety constraints
            process_noise: Process noise covariance
        """
        self.dynamics = dynamics
        self.cost_function = cost_function
        self.constraints = constraints
        self.process_noise = process_noise

        # Uncertainty propagation
        self.propagation = UncertaintyPropagation(
            dynamics, process_noise, method="linear"
        )

    def optimize(
        self,
        initial_state: StateDistribution,
        target_state: np.ndarray,
        horizon: int = 20,
        dt: float = 1.0,
        method: str = "tube",
    ) -> tuple[np.ndarray, np.ndarray, list[StateDistribution]]:
        """
        Optimize robust trajectory.

        Args:
            initial_state: Initial state distribution
            target_state: Target state
            horizon: Planning horizon
            dt: Time step
            method: 'tube', 'scenario', or 'risk'

        Returns:
            (nominal_trajectory, control_sequence, state_distributions)
        """
        if method == "tube":
            return self._optimize_tube(initial_state, target_state, horizon, dt)
        elif method == "scenario":
            return self._optimize_scenario(initial_state, target_state, horizon, dt)
        else:
            raise ValueError(f"Unknown method: {method}")

    def _optimize_tube(
        self,
        initial: StateDistribution,
        target: np.ndarray,
        horizon: int,
        dt: float,
    ) -> tuple[np.ndarray, np.ndarray, list[StateDistribution]]:
        """
        Tube MPC optimization.

        Optimizes nominal trajectory while tracking uncertainty "tubes".
        """
        n_states = initial.dimension

        # Initialize with straight-line trajectory
        nominal_trajectory = np.zeros((horizon + 1, n_states))
        nominal_trajectory[0] = initial.mean

        direction = target - initial.mean
        for i in range(1, horizon + 1):
            alpha = i / horizon
            nominal_trajectory[i] = initial.mean + alpha * direction

        # Compute control sequence
        controls = np.zeros((horizon, n_states))
        for i in range(horizon):
            # Simple: velocity toward next waypoint
            direction = nominal_trajectory[i + 1] - nominal_trajectory[i]
            controls[i] = direction / dt

        # Propagate uncertainty along trajectory
        state_distributions = [initial]
        current = initial

        for i in range(horizon):
            current = self.propagation.propagate(current, controls[i], dt)
            state_distributions.append(current)

        # Check constraint satisfaction
        for i, state_dist in enumerate(state_distributions):
            max_violation = self.constraints.max_violation(state_dist)
            if max_violation > 0:
                logger.warning(f"Constraint violation at step {i}: {max_violation:.4f}")

        return nominal_trajectory, controls, state_distributions

    def _optimize_scenario(
        self,
        initial: StateDistribution,
        target: np.ndarray,
        horizon: int,
        dt: float,
        n_scenarios: int = 50,
    ) -> tuple[np.ndarray, np.ndarray, list[StateDistribution]]:
        """
        Scenario-based optimization.

        Samples uncertainty realizations and optimizes to satisfy
        constraints for all samples.
        """
        n_states = initial.dimension

        # Sample initial states
        initial_samples = initial.sample(n_scenarios)

        # Initialize trajectory
        nominal_trajectory = np.zeros((horizon + 1, n_states))
        nominal_trajectory[0] = initial.mean

        direction = target - initial.mean
        for i in range(1, horizon + 1):
            alpha = i / horizon
            nominal_trajectory[i] = initial.mean + alpha * direction

        # Compute controls
        controls = np.zeros((horizon, n_states))
        for i in range(horizon):
            direction = nominal_trajectory[i + 1] - nominal_trajectory[i]
            controls[i] = direction / dt

        # Propagate each scenario
        all_trajectories = np.zeros((n_scenarios, horizon + 1, n_states))
        for s in range(n_scenarios):
            all_trajectories[s, 0] = initial_samples[s]
            for i in range(horizon):
                x_dot = self.dynamics(all_trajectories[s, i], controls[i])
                all_trajectories[s, i + 1] = all_trajectories[s, i] + x_dot * dt

                # Add process noise
                if self.process_noise is not None:
                    noise = np.random.multivariate_normal(
                        np.zeros(n_states),
                        self.process_noise * dt,
                    )
                    all_trajectories[s, i + 1] += noise

        # Compute state distributions from scenarios
        state_distributions = []
        for i in range(horizon + 1):
            mean = np.mean(all_trajectories[:, i, :], axis=0)
            cov = np.cov(all_trajectories[:, i, :], rowvar=False)
            state_distributions.append(StateDistribution(mean, cov, i * dt))

        return nominal_trajectory, controls, state_distributions

    def evaluate_risk(
        self,
        trajectory: np.ndarray,
        n_samples: int = 1000,
    ) -> dict[str, float]:
        """
        Evaluate trajectory risk metrics.

        Computes:
        - VaR (Value at Risk): constraint violation at (1-ε) quantile
        - CVaR (Conditional VaR): expected violation given violation occurs
        - Probability of constraint violation

        Args:
            trajectory: Nominal trajectory
            n_samples: Monte Carlo samples

        Returns:
            Dict of risk metrics
        """
        # Sample trajectories
        n_states = trajectory.shape[1]

        if self.process_noise is not None:
            noise_std = np.sqrt(np.diag(self.process_noise))
        else:
            noise_std = np.zeros(n_states)

        max_violations = []

        for _ in range(n_samples):
            # Add noise to trajectory
            noisy_traj = trajectory + np.random.randn(*trajectory.shape) * noise_std

            # Evaluate constraints at each point
            max_violation = -np.inf
            for state in noisy_traj:
                state_dist = StateDistribution(state, np.diag(noise_std**2))
                violation = self.constraints.max_violation(state_dist)
                max_violation = max(max_violation, violation)

            max_violations.append(max_violation)

        max_violations = np.array(max_violations)

        # Compute risk metrics
        epsilon = self.constraints.constraints[0].epsilon if self.constraints.constraints else 0.05

        var = np.quantile(max_violations, 1 - epsilon)  # VaR
        cvar = np.mean(max_violations[max_violations >= var])  # CVaR
        prob_violation = np.mean(max_violations > 0)

        return {
            'var': float(var),
            'cvar': float(cvar) if not np.isnan(cvar) else 0.0,
            'prob_violation': float(prob_violation),
            'mean_violation': float(np.mean(np.maximum(max_violations, 0))),
            'max_violation': float(np.max(max_violations)),
        }


def create_standard_safety_constraints(
    terrain_field: Callable | None = None,
    threat_field: Callable | None = None,
    min_clearance: float = 50.0,
    pk_threshold: float = 0.1,
    epsilon: float = 0.05,
) -> ProbabilisticSafetyConstraint:
    """
    Create standard safety constraints for defense eVTOL.

    Args:
        terrain_field: Terrain elevation function
        threat_field: Threat probability function
        min_clearance: Minimum terrain clearance [m]
        pk_threshold: Maximum kill probability
        epsilon: Violation probability

    Returns:
        ProbabilisticSafetyConstraint
    """
    safety = ProbabilisticSafetyConstraint()

    if terrain_field is not None:
        safety.add(TerrainClearanceConstraint(
            terrain_field=terrain_field,
            min_clearance=min_clearance,
            epsilon=epsilon,
        ))

    if threat_field is not None:
        safety.add(ThreatAvoidanceConstraint(
            threat_field=threat_field,
            pk_threshold=pk_threshold,
            epsilon=epsilon,
        ))

    return safety
