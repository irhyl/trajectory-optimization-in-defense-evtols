"""
Numerical Integrators for Vehicle Dynamics

This module provides integration schemes for propagating the vehicle state
forward in time. Includes:

1. RK4: Classic 4th-order Runge-Kutta (fixed step)
2. RK45: Adaptive Runge-Kutta-Fehlberg with error control
3. Symplectic: For energy-conserving long-duration simulation

For trajectory optimization, RK4 is typically sufficient and provides
a good balance of accuracy and computational cost.

For high-fidelity simulation or Monte Carlo, RK45 with error control
ensures accuracy while adapting to stiff dynamics.
"""

from __future__ import annotations
import numpy as np
from collections.abc import Callable
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


# Type alias for dynamics function
# f(t, x) -> x_dot
DynamicsFunc = Callable[[float, np.ndarray], np.ndarray]


@dataclass
class IntegrationResult:
    """Result of numerical integration."""
    t: np.ndarray           # Time points
    x: np.ndarray           # State trajectory [n_steps, n_states]
    dt_history: np.ndarray  # Step sizes used (for adaptive methods)
    n_evals: int            # Number of function evaluations
    success: bool           # Whether integration completed successfully
    message: str            # Status message


class RK4Integrator:
    """
    Classic 4th-order Runge-Kutta integrator (fixed step).

    The RK4 method:
        k1 = f(t, x)
        k2 = f(t + dt/2, x + dt/2 * k1)
        k3 = f(t + dt/2, x + dt/2 * k2)
        k4 = f(t + dt, x + dt * k3)
        x_new = x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

    Properties:
        - 4th order accuracy: error ~ O(dt^5)
        - 4 function evaluations per step
        - Simple and robust
        - No automatic step size control

    Attributes:
        dt: Fixed time step [s]
        normalize_quaternion: Whether to normalize quaternion after each step
        quaternion_indices: Indices of quaternion in state vector [6:10]
    """

    def __init__(
        self,
        dt: float = 0.01,
        normalize_quaternion: bool = True,
        quaternion_indices: tuple[int, int] = (6, 10),
    ):
        """
        Initialize RK4 integrator.

        Args:
            dt: Time step [s]
            normalize_quaternion: Normalize quaternion after each step
            quaternion_indices: (start, end) indices of quaternion in state
        """
        self.dt = dt
        self.normalize_quaternion = normalize_quaternion
        self.q_start = quaternion_indices[0]
        self.q_end = quaternion_indices[1]

    def step(
        self,
        f: DynamicsFunc,
        t: float,
        x: np.ndarray,
        dt: float = None,
    ) -> np.ndarray:
        """
        Perform single RK4 integration step.

        Args:
            f: Dynamics function f(t, x) -> x_dot
            t: Current time
            x: Current state
            dt: Time step (uses default if None)

        Returns:
            New state x(t + dt)
        """
        if dt is None:
            dt = self.dt

        # RK4 stages
        k1 = f(t, x)
        k2 = f(t + dt/2, x + dt/2 * k1)
        k3 = f(t + dt/2, x + dt/2 * k2)
        k4 = f(t + dt, x + dt * k3)

        # Update
        x_new = x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

        # Normalize quaternion if requested
        if self.normalize_quaternion:
            x_new = self._normalize_quaternion(x_new)

        return x_new

    def integrate(
        self,
        f: DynamicsFunc,
        x0: np.ndarray,
        t_span: tuple[float, float],
        dt: float = None,
        t_eval: np.ndarray = None,
    ) -> IntegrationResult:
        """
        Integrate over time span.

        Args:
            f: Dynamics function f(t, x) -> x_dot
            x0: Initial state
            t_span: (t_start, t_end)
            dt: Time step (uses default if None)
            t_eval: Optional specific times to return states at

        Returns:
            IntegrationResult with trajectory
        """
        if dt is None:
            dt = self.dt

        t_start, t_end = t_span
        n_steps = int(np.ceil((t_end - t_start) / dt))

        # Pre-allocate storage
        t_history = np.zeros(n_steps + 1)
        x_history = np.zeros((n_steps + 1, len(x0)))

        # Initial conditions
        t = t_start
        x = x0.copy()
        t_history[0] = t
        x_history[0] = x

        # Integration loop
        n_evals = 0
        for i in range(n_steps):
            # Adjust last step
            step_dt = min(dt, t_end - t)

            # RK4 step
            x = self.step(f, t, x, step_dt)
            t += step_dt
            n_evals += 4

            t_history[i + 1] = t
            x_history[i + 1] = x

        # Interpolate to t_eval if specified
        if t_eval is not None:
            x_eval = np.zeros((len(t_eval), len(x0)))
            for i in range(len(x0)):
                x_eval[:, i] = np.interp(t_eval, t_history, x_history[:, i])
            t_history = t_eval
            x_history = x_eval

        return IntegrationResult(
            t=t_history,
            x=x_history,
            dt_history=np.full(n_steps, dt),
            n_evals=n_evals,
            success=True,
            message="Integration completed successfully",
        )

    def _normalize_quaternion(self, x: np.ndarray) -> np.ndarray:
        """Normalize quaternion portion of state vector."""
        q = x[self.q_start:self.q_end]
        norm = np.linalg.norm(q)
        if norm > 1e-10:
            x[self.q_start:self.q_end] = q / norm
        return x


class RK45AdaptiveIntegrator:
    """
    Runge-Kutta-Fehlberg 4(5) with adaptive step size control.

    Uses embedded 4th and 5th order methods to estimate local truncation
    error and adapt step size to maintain accuracy.

    The RK45 method (Dormand-Prince coefficients):
        - 5th order solution for propagation
        - 4th order solution for error estimate
        - Error ~ |x5 - x4|

    Step size adaptation:
        dt_new = dt * min(max_factor, max(min_factor, safety * (tol/error)^0.2))

    Attributes:
        rtol: Relative tolerance
        atol: Absolute tolerance
        dt_min: Minimum step size
        dt_max: Maximum step size
        max_steps: Maximum number of steps
    """

    # Dormand-Prince coefficients
    A = np.array([
        [0, 0, 0, 0, 0, 0, 0],
        [1/5, 0, 0, 0, 0, 0, 0],
        [3/40, 9/40, 0, 0, 0, 0, 0],
        [44/45, -56/15, 32/9, 0, 0, 0, 0],
        [19372/6561, -25360/2187, 64448/6561, -212/729, 0, 0, 0],
        [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656, 0, 0],
        [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0],
    ])

    B5 = np.array([35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0])  # 5th order
    B4 = np.array([5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40])  # 4th order

    C = np.array([0, 1/5, 3/10, 4/5, 8/9, 1, 1])

    def __init__(
        self,
        rtol: float = 1e-6,
        atol: float = 1e-9,
        dt_init: float = 0.01,
        dt_min: float = 1e-8,
        dt_max: float = 1.0,
        max_steps: int = 100000,
        safety: float = 0.9,
        normalize_quaternion: bool = True,
        quaternion_indices: tuple[int, int] = (6, 10),
    ):
        """
        Initialize adaptive integrator.

        Args:
            rtol: Relative tolerance
            atol: Absolute tolerance
            dt_init: Initial step size
            dt_min: Minimum step size
            dt_max: Maximum step size
            max_steps: Maximum number of integration steps
            safety: Safety factor for step size adaptation
            normalize_quaternion: Normalize quaternion after each step
            quaternion_indices: (start, end) indices of quaternion
        """
        self.rtol = rtol
        self.atol = atol
        self.dt_init = dt_init
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.max_steps = max_steps
        self.safety = safety
        self.normalize_quaternion = normalize_quaternion
        self.q_start = quaternion_indices[0]
        self.q_end = quaternion_indices[1]

    def step(
        self,
        f: DynamicsFunc,
        t: float,
        x: np.ndarray,
        dt: float,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """
        Perform single RK45 step with error estimate.

        Args:
            f: Dynamics function
            t: Current time
            x: Current state
            dt: Proposed step size

        Returns:
            (x_new, error, dt_new)
        """
        n = len(x)
        k = np.zeros((7, n))

        # Compute stages
        k[0] = f(t, x)
        for i in range(1, 7):
            t_i = t + self.C[i] * dt
            x_i = x + dt * np.sum([self.A[i, j] * k[j] for j in range(i)], axis=0)
            k[i] = f(t_i, x_i)

        # 5th order solution (propagation)
        x5 = x + dt * np.sum([self.B5[i] * k[i] for i in range(7)], axis=0)

        # 4th order solution (error estimate)
        x4 = x + dt * np.sum([self.B4[i] * k[i] for i in range(7)], axis=0)

        # Error estimate
        error = x5 - x4

        # Error norm (scaled by tolerance)
        scale = self.atol + self.rtol * np.maximum(np.abs(x), np.abs(x5))
        error_norm = np.linalg.norm(error / scale) / np.sqrt(n)

        # Normalize quaternion
        if self.normalize_quaternion:
            x5 = self._normalize_quaternion(x5)

        return x5, error_norm, 7  # 7 function evaluations

    def integrate(
        self,
        f: DynamicsFunc,
        x0: np.ndarray,
        t_span: tuple[float, float],
        t_eval: np.ndarray = None,
        events: list[Callable] = None,
    ) -> IntegrationResult:
        """
        Integrate with adaptive step size.

        Args:
            f: Dynamics function f(t, x) -> x_dot
            x0: Initial state
            t_span: (t_start, t_end)
            t_eval: Optional specific times to output
            events: Optional event functions for termination

        Returns:
            IntegrationResult with trajectory
        """
        t_start, t_end = t_span
        direction = np.sign(t_end - t_start)

        # Storage (dynamically sized)
        t_history = [t_start]
        x_history = [x0.copy()]
        dt_history = []

        t = t_start
        x = x0.copy()
        dt = self.dt_init * direction
        n_evals = 0

        for _step in range(self.max_steps):
            # Check if finished
            if (direction > 0 and t >= t_end) or (direction < 0 and t <= t_end):
                break

            # Clamp step to not exceed t_end
            if direction * (t + dt) > direction * t_end:
                dt = t_end - t

            # Attempt step
            x_new, error_norm, evals = self.step(f, t, x, dt)
            n_evals += evals

            if error_norm <= 1.0:
                # Step accepted
                t += dt
                x = x_new

                t_history.append(t)
                x_history.append(x.copy())
                dt_history.append(abs(dt))

                # Check events
                if events:
                    for event in events:
                        if event(t, x):
                            return IntegrationResult(
                                t=np.array(t_history),
                                x=np.array(x_history),
                                dt_history=np.array(dt_history),
                                n_evals=n_evals,
                                success=True,
                                message="Event triggered",
                            )

            # Adapt step size
            if error_norm > 0:
                factor = self.safety * (1.0 / error_norm) ** 0.2
            else:
                factor = 5.0

            factor = max(0.2, min(5.0, factor))  # Limit change
            dt *= factor
            dt = direction * max(self.dt_min, min(self.dt_max, abs(dt)))

        # Check if completed
        success = (direction > 0 and t >= t_end) or (direction < 0 and t <= t_end)
        message = "Integration completed" if success else "Max steps reached"

        result = IntegrationResult(
            t=np.array(t_history),
            x=np.array(x_history),
            dt_history=np.array(dt_history) if dt_history else np.array([self.dt_init]),
            n_evals=n_evals,
            success=success,
            message=message,
        )

        # Interpolate to t_eval if specified
        if t_eval is not None:
            x_eval = np.zeros((len(t_eval), len(x0)))
            for i in range(len(x0)):
                x_eval[:, i] = np.interp(t_eval, result.t, result.x[:, i])
            result.t = t_eval
            result.x = x_eval

        return result

    def _normalize_quaternion(self, x: np.ndarray) -> np.ndarray:
        """Normalize quaternion portion of state vector."""
        q = x[self.q_start:self.q_end]
        norm = np.linalg.norm(q)
        if norm > 1e-10:
            x[self.q_start:self.q_end] = q / norm
        return x
