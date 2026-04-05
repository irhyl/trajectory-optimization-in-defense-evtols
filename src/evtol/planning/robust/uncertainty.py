"""
Uncertainty Modeling and Propagation

This module provides mathematical models for uncertainty in:
- Wind fields (direction, magnitude)
- Threat positions and capabilities
- Sensor measurements
- Vehicle dynamics

Mathematical Foundation:
------------------------
State uncertainty is typically modeled as:
    x = x̄ + δx

where x̄ is the nominal state and δx ~ N(0, Σ) is Gaussian perturbation.

For nonlinear dynamics ẋ = f(x, u, w):
    Σ̇ = A·Σ + Σ·Aᵀ + Q

where A = ∂f/∂x is the Jacobian and Q is process noise covariance.

Uncertainty Propagation Methods:
1. Linearization (Extended Kalman-style)
2. Unscented Transform
3. Monte Carlo sampling

Author: Defense eVTOL Research Team
"""

from __future__ import annotations

import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from collections.abc import Callable
from scipy import linalg
import logging

logger = logging.getLogger(__name__)


class UncertaintyModel(ABC):
    """
    Abstract base class for uncertainty models.

    Defines the interface for sampling and computing statistics.
    """

    def __init__(self, name: str, dimension: int):
        self.name = name
        self.dimension = dimension

    @abstractmethod
    def sample(self, n_samples: int = 1) -> np.ndarray:
        """
        Draw samples from the uncertainty distribution.

        Args:
            n_samples: Number of samples to draw

        Returns:
            Array of shape (n_samples, dimension)
        """
        pass

    @abstractmethod
    def mean(self) -> np.ndarray:
        """Return mean of distribution."""
        pass

    @abstractmethod
    def covariance(self) -> np.ndarray:
        """Return covariance matrix."""
        pass

    def std(self) -> np.ndarray:
        """Return standard deviations."""
        return np.sqrt(np.diag(self.covariance()))

    def confidence_region(
        self,
        confidence: float = 0.95,
        n_points: int = 100,
    ) -> np.ndarray:
        """
        Compute confidence ellipse/ellipsoid boundary.

        For 2D, returns points on the confidence ellipse.

        Args:
            confidence: Confidence level (0-1)
            n_points: Number of boundary points

        Returns:
            Array of boundary points
        """
        from scipy import stats

        if self.dimension != 2:
            raise NotImplementedError("Confidence region only for 2D")

        # Chi-squared quantile for confidence region
        chi2_val = stats.chi2.ppf(confidence, df=2)

        mean = self.mean()
        cov = self.covariance()

        # Eigendecomposition for ellipse axes
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # Ellipse parameters
        angle = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])
        width = 2 * np.sqrt(chi2_val * eigenvalues[0])
        height = 2 * np.sqrt(chi2_val * eigenvalues[1])

        # Generate ellipse points
        t = np.linspace(0, 2 * np.pi, n_points)
        ellipse = np.zeros((n_points, 2))

        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)

        for i, theta in enumerate(t):
            x = width/2 * np.cos(theta)
            y = height/2 * np.sin(theta)

            # Rotate
            ellipse[i, 0] = mean[0] + x * cos_angle - y * sin_angle
            ellipse[i, 1] = mean[1] + x * sin_angle + y * cos_angle

        return ellipse


class GaussianUncertainty(UncertaintyModel):
    """
    Multivariate Gaussian uncertainty model.

    x ~ N(μ, Σ)

    Attributes:
        mu: Mean vector
        sigma: Covariance matrix
    """

    def __init__(
        self,
        mu: np.ndarray,
        sigma: np.ndarray,
        name: str = "gaussian",
    ):
        """
        Initialize Gaussian uncertainty.

        Args:
            mu: Mean vector (n,)
            sigma: Covariance matrix (n, n)
            name: Model name
        """
        mu = np.asarray(mu)
        sigma = np.asarray(sigma)

        super().__init__(name, len(mu))

        self.mu = mu
        self.sigma = sigma

        # Precompute Cholesky decomposition
        try:
            self._L = np.linalg.cholesky(sigma)
        except np.linalg.LinAlgError:
            # Add small regularization
            self._L = np.linalg.cholesky(sigma + 1e-6 * np.eye(len(mu)))

    def sample(self, n_samples: int = 1) -> np.ndarray:
        """Draw samples from Gaussian."""
        z = np.random.randn(n_samples, self.dimension)
        return self.mu + z @ self._L.T

    def mean(self) -> np.ndarray:
        return self.mu.copy()

    def covariance(self) -> np.ndarray:
        return self.sigma.copy()

    def pdf(self, x: np.ndarray) -> np.ndarray:
        """Compute probability density at x."""
        from scipy import stats
        return stats.multivariate_normal.pdf(x, self.mu, self.sigma)

    def mahalanobis(self, x: np.ndarray) -> float:
        """Compute Mahalanobis distance from mean."""
        diff = x - self.mu
        sigma_inv = np.linalg.inv(self.sigma)
        return np.sqrt(diff @ sigma_inv @ diff)

    @classmethod
    def from_samples(cls, samples: np.ndarray, name: str = "gaussian") -> GaussianUncertainty:
        """Fit Gaussian to samples."""
        mu = np.mean(samples, axis=0)
        sigma = np.cov(samples, rowvar=False)
        if sigma.ndim == 0:
            sigma = np.array([[sigma]])
        return cls(mu, sigma, name)


class UniformUncertainty(UncertaintyModel):
    """
    Multivariate uniform uncertainty model.

    x ~ U[a, b] (box)

    Attributes:
        lower: Lower bounds
        upper: Upper bounds
    """

    def __init__(
        self,
        lower: np.ndarray,
        upper: np.ndarray,
        name: str = "uniform",
    ):
        """
        Initialize uniform uncertainty.

        Args:
            lower: Lower bounds
            upper: Upper bounds
            name: Model name
        """
        lower = np.asarray(lower)
        upper = np.asarray(upper)

        super().__init__(name, len(lower))

        self.lower = lower
        self.upper = upper

    def sample(self, n_samples: int = 1) -> np.ndarray:
        """Draw samples from uniform distribution."""
        return np.random.uniform(self.lower, self.upper, (n_samples, self.dimension))

    def mean(self) -> np.ndarray:
        return (self.lower + self.upper) / 2

    def covariance(self) -> np.ndarray:
        """Covariance of uniform distribution."""
        # Var(U[a,b]) = (b-a)²/12
        variance = (self.upper - self.lower) ** 2 / 12
        return np.diag(variance)


class WindUncertainty(UncertaintyModel):
    """
    Wind field uncertainty model.

    Models uncertainty in wind speed and direction:
    - Speed: Log-normal (always positive, heavy tail)
    - Direction: Von Mises (circular)

    Attributes:
        mean_speed: Mean wind speed [m/s]
        speed_std: Wind speed standard deviation [m/s]
        mean_direction: Mean wind direction [rad]
        direction_std: Direction concentration parameter
    """

    def __init__(
        self,
        mean_speed: float = 5.0,
        speed_std: float = 2.0,
        mean_direction: float = 0.0,
        direction_kappa: float = 2.0,
        name: str = "wind",
    ):
        """
        Initialize wind uncertainty.

        Args:
            mean_speed: Mean wind speed [m/s]
            speed_std: Speed standard deviation [m/s]
            mean_direction: Mean direction [rad from North]
            direction_kappa: Von Mises concentration (higher = less spread)
            name: Model name
        """
        super().__init__(name, 2)  # 2D: (speed, direction) or (Vx, Vy)

        self.mean_speed = mean_speed
        self.speed_std = speed_std
        self.mean_direction = mean_direction
        self.direction_kappa = direction_kappa

        # Convert to log-normal parameters
        self._ln_sigma = np.sqrt(np.log(1 + (speed_std / mean_speed) ** 2))
        self._ln_mu = np.log(mean_speed) - 0.5 * self._ln_sigma ** 2

    def sample(self, n_samples: int = 1) -> np.ndarray:
        """
        Sample wind vectors (Vx, Vy).

        Returns:
            Array of shape (n_samples, 2) with wind components
        """
        # Sample speed from log-normal
        speeds = np.random.lognormal(self._ln_mu, self._ln_sigma, n_samples)

        # Sample direction from Von Mises
        directions = np.random.vonmises(self.mean_direction, self.direction_kappa, n_samples)

        # Convert to Cartesian
        vx = speeds * np.sin(directions)
        vy = speeds * np.cos(directions)

        return np.column_stack([vx, vy])

    def mean(self) -> np.ndarray:
        """Mean wind vector."""
        vx = self.mean_speed * np.sin(self.mean_direction)
        vy = self.mean_speed * np.cos(self.mean_direction)
        return np.array([vx, vy])

    def covariance(self) -> np.ndarray:
        """Approximate covariance (via sampling)."""
        samples = self.sample(10000)
        return np.cov(samples, rowvar=False)

    def probability_exceeds(self, threshold_speed: float, n_samples: int = 10000) -> float:
        """
        Compute P(wind_speed > threshold).

        Args:
            threshold_speed: Speed threshold [m/s]
            n_samples: Monte Carlo samples

        Returns:
            Exceedance probability
        """
        samples = self.sample(n_samples)
        speeds = np.linalg.norm(samples, axis=1)
        return np.mean(speeds > threshold_speed)


class ThreatUncertainty(UncertaintyModel):
    """
    Threat position and capability uncertainty.

    Models uncertainty in:
    - Position: Gaussian in 3D
    - Range capability: Log-normal
    - Detection probability: Beta

    Attributes:
        position_mean: Mean threat position
        position_std: Position uncertainty (m)
        range_mean: Mean engagement range (m)
        range_std: Range uncertainty (m)
    """

    def __init__(
        self,
        position_mean: np.ndarray,
        position_std: float = 100.0,
        range_mean: float = 5000.0,
        range_std: float = 500.0,
        name: str = "threat",
    ):
        """
        Initialize threat uncertainty.

        Args:
            position_mean: Mean threat position [lat, lon, alt]
            position_std: Position uncertainty [m]
            range_mean: Mean engagement range [m]
            range_std: Range uncertainty [m]
            name: Model name
        """
        super().__init__(name, 4)  # 4D: x, y, z, range

        self.position_mean = np.asarray(position_mean)
        self.position_std = position_std
        self.range_mean = range_mean
        self.range_std = range_std

        # Log-normal parameters for range
        self._range_ln_sigma = np.sqrt(np.log(1 + (range_std / range_mean) ** 2))
        self._range_ln_mu = np.log(range_mean) - 0.5 * self._range_ln_sigma ** 2

    def sample(self, n_samples: int = 1) -> np.ndarray:
        """
        Sample threat parameters.

        Returns:
            Array of shape (n_samples, 4) with [x, y, z, range]
        """
        # Position samples
        position_samples = np.random.normal(
            self.position_mean,
            self.position_std,
            (n_samples, 3),
        )

        # Range samples
        range_samples = np.random.lognormal(
            self._range_ln_mu,
            self._range_ln_sigma,
            (n_samples, 1),
        )

        return np.hstack([position_samples, range_samples])

    def mean(self) -> np.ndarray:
        return np.append(self.position_mean, self.range_mean)

    def covariance(self) -> np.ndarray:
        """Build covariance matrix."""
        cov = np.zeros((4, 4))
        # Position covariance
        cov[:3, :3] = np.eye(3) * self.position_std ** 2
        # Range variance (approximate)
        cov[3, 3] = self.range_std ** 2
        return cov

    def engagement_probability(
        self,
        target_position: np.ndarray,
        n_samples: int = 10000,
    ) -> tuple[float, float]:
        """
        Compute engagement probability with uncertainty.

        Args:
            target_position: Target position [lat, lon, alt]
            n_samples: Monte Carlo samples

        Returns:
            (mean_probability, std_probability)
        """
        samples = self.sample(n_samples)

        probabilities = []
        for sample in samples:
            threat_pos = sample[:3]
            threat_range = sample[3]

            # Distance to target (approximate, in meters)
            dlat = (target_position[0] - threat_pos[0]) * 110540
            dlon = (target_position[1] - threat_pos[1]) * 111320 * np.cos(np.radians(threat_pos[0]))
            dalt = target_position[2] - threat_pos[2]
            distance = np.sqrt(dlat**2 + dlon**2 + dalt**2)

            # Simple probability model
            if distance < threat_range:
                # Probability decreases with range
                prob = 1.0 - (distance / threat_range) ** 2
            else:
                prob = 0.0

            probabilities.append(prob)

        return float(np.mean(probabilities)), float(np.std(probabilities))


@dataclass
class StateDistribution:
    """
    Distribution over system states.

    Represents the uncertain state of the vehicle.

    Attributes:
        mean: Mean state vector
        covariance: State covariance matrix
        time: Time stamp
    """
    mean: np.ndarray
    covariance: np.ndarray
    time: float = 0.0

    @property
    def dimension(self) -> int:
        return len(self.mean)

    def sample(self, n_samples: int = 1) -> np.ndarray:
        """Draw state samples."""
        return np.random.multivariate_normal(self.mean, self.covariance, n_samples)

    def marginal(self, indices: list[int]) -> StateDistribution:
        """Extract marginal distribution for subset of states."""
        indices = np.asarray(indices)
        return StateDistribution(
            mean=self.mean[indices],
            covariance=self.covariance[np.ix_(indices, indices)],
            time=self.time,
        )

    def confidence_bound(self, index: int, sigma: float = 3.0) -> tuple[float, float]:
        """
        Get confidence bounds for a single state dimension.

        Args:
            index: State dimension index
            sigma: Number of standard deviations

        Returns:
            (lower, upper) bounds
        """
        std = np.sqrt(self.covariance[index, index])
        return (self.mean[index] - sigma * std, self.mean[index] + sigma * std)


class UncertaintyPropagation:
    """
    Propagate uncertainty through dynamics.

    Methods:
    1. Linear propagation (EKF-style)
    2. Unscented transform
    3. Monte Carlo

    Attributes:
        dynamics: System dynamics function x_dot = f(x, u, w)
        process_noise: Process noise covariance Q
    """

    def __init__(
        self,
        dynamics: Callable[[np.ndarray, np.ndarray], np.ndarray],
        process_noise: np.ndarray | None = None,
        method: str = "linear",
    ):
        """
        Initialize propagation.

        Args:
            dynamics: Dynamics function f(x, u) -> x_dot
            process_noise: Process noise covariance Q
            method: 'linear', 'unscented', or 'monte_carlo'
        """
        self.dynamics = dynamics
        self.process_noise = process_noise
        self.method = method

    def propagate(
        self,
        state_dist: StateDistribution,
        control: np.ndarray,
        dt: float,
    ) -> StateDistribution:
        """
        Propagate state distribution through dynamics.

        Args:
            state_dist: Current state distribution
            control: Control input
            dt: Time step

        Returns:
            Propagated state distribution
        """
        if self.method == "linear":
            return self._propagate_linear(state_dist, control, dt)
        elif self.method == "unscented":
            return self._propagate_unscented(state_dist, control, dt)
        elif self.method == "monte_carlo":
            return self._propagate_monte_carlo(state_dist, control, dt)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _propagate_linear(
        self,
        state_dist: StateDistribution,
        control: np.ndarray,
        dt: float,
    ) -> StateDistribution:
        """
        Linear propagation using Jacobian.

        Σ_new = A · Σ · Aᵀ + Q·dt

        where A = I + (∂f/∂x)·dt
        """
        x = state_dist.mean
        P = state_dist.covariance
        n = len(x)

        # Evaluate dynamics at mean
        x_dot = self.dynamics(x, control)

        # Numerical Jacobian
        A = self._numerical_jacobian(x, control)

        # Discrete-time state transition
        F = np.eye(n) + A * dt

        # Propagate mean
        x_new = x + x_dot * dt

        # Propagate covariance
        Q = self.process_noise if self.process_noise is not None else np.zeros((n, n))
        P_new = F @ P @ F.T + Q * dt

        return StateDistribution(x_new, P_new, state_dist.time + dt)

    def _numerical_jacobian(
        self,
        x: np.ndarray,
        u: np.ndarray,
        eps: float = 1e-6,
    ) -> np.ndarray:
        """Compute Jacobian numerically."""
        n = len(x)
        A = np.zeros((n, n))

        f0 = self.dynamics(x, u)

        for i in range(n):
            x_plus = x.copy()
            x_plus[i] += eps
            f_plus = self.dynamics(x_plus, u)
            A[:, i] = (f_plus - f0) / eps

        return A

    def _propagate_unscented(
        self,
        state_dist: StateDistribution,
        control: np.ndarray,
        dt: float,
    ) -> StateDistribution:
        """
        Unscented transform propagation.

        Uses sigma points to capture nonlinear effects.
        """
        x = state_dist.mean
        P = state_dist.covariance
        n = len(x)

        # Scaling parameters
        alpha = 1e-3
        beta = 2.0
        kappa = 0.0
        lam = alpha**2 * (n + kappa) - n

        # Sigma points
        sqrt_P = linalg.sqrtm((n + lam) * P)
        sigma_points = np.zeros((2 * n + 1, n))
        sigma_points[0] = x
        for i in range(n):
            sigma_points[i + 1] = x + sqrt_P[i]
            sigma_points[n + i + 1] = x - sqrt_P[i]

        # Weights
        Wm = np.zeros(2 * n + 1)
        Wc = np.zeros(2 * n + 1)
        Wm[0] = lam / (n + lam)
        Wc[0] = lam / (n + lam) + (1 - alpha**2 + beta)
        for i in range(1, 2 * n + 1):
            Wm[i] = 1 / (2 * (n + lam))
            Wc[i] = 1 / (2 * (n + lam))

        # Propagate sigma points
        prop_points = np.zeros_like(sigma_points)
        for i in range(2 * n + 1):
            x_dot = self.dynamics(sigma_points[i], control)
            prop_points[i] = sigma_points[i] + x_dot * dt

        # Compute mean and covariance
        x_new = np.sum(Wm[:, np.newaxis] * prop_points, axis=0)

        P_new = np.zeros((n, n))
        for i in range(2 * n + 1):
            diff = prop_points[i] - x_new
            P_new += Wc[i] * np.outer(diff, diff)

        # Add process noise
        if self.process_noise is not None:
            P_new += self.process_noise * dt

        return StateDistribution(x_new, P_new, state_dist.time + dt)

    def _propagate_monte_carlo(
        self,
        state_dist: StateDistribution,
        control: np.ndarray,
        dt: float,
        n_samples: int = 1000,
    ) -> StateDistribution:
        """
        Monte Carlo propagation.

        Most accurate but slowest.
        """
        x = state_dist.mean
        P = state_dist.covariance
        n = len(x)

        # Sample initial states
        samples = np.random.multivariate_normal(x, P, n_samples)

        # Propagate each sample
        prop_samples = np.zeros_like(samples)
        for i in range(n_samples):
            x_dot = self.dynamics(samples[i], control)
            prop_samples[i] = samples[i] + x_dot * dt

            # Add process noise
            if self.process_noise is not None:
                prop_samples[i] += np.random.multivariate_normal(
                    np.zeros(n),
                    self.process_noise * dt,
                )

        # Estimate new distribution
        x_new = np.mean(prop_samples, axis=0)
        P_new = np.cov(prop_samples, rowvar=False)

        return StateDistribution(x_new, P_new, state_dist.time + dt)

    def propagate_trajectory(
        self,
        initial_dist: StateDistribution,
        controls: np.ndarray,
        times: np.ndarray,
    ) -> list[StateDistribution]:
        """
        Propagate distribution along a trajectory.

        Args:
            initial_dist: Initial state distribution
            controls: Control sequence
            times: Time stamps

        Returns:
            List of state distributions at each time
        """
        distributions = [initial_dist]
        current = initial_dist

        for i in range(len(times) - 1):
            dt = times[i + 1] - times[i]
            control = controls[i] if i < len(controls) else controls[-1]

            current = self.propagate(current, control, dt)
            distributions.append(current)

        return distributions
