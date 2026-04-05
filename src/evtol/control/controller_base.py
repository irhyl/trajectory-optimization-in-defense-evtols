"""
Base controller classes and PID implementation.

Provides configurable PID controller with:
- Anti-windup (back-calculation method)
- Derivative filtering
- Output saturation
- Gain scheduling support
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np


@dataclass
class ControllerGains:
    """PID controller gains with limits."""
    Kp: float = 1.0          # Proportional gain
    Ki: float = 0.0          # Integral gain
    Kd: float = 0.0          # Derivative gain

    # Output limits
    output_min: float = -np.inf
    output_max: float = np.inf

    # Integrator limits (anti-windup)
    integrator_min: float = -np.inf
    integrator_max: float = np.inf

    # Derivative filter coefficient (0 = no filtering, 1 = full filtering)
    derivative_filter: float = 0.1

    def scale(self, factor: float) -> 'ControllerGains':
        """Return scaled gains for gain scheduling."""
        return ControllerGains(
            Kp=self.Kp * factor,
            Ki=self.Ki * factor,
            Kd=self.Kd * factor,
            output_min=self.output_min,
            output_max=self.output_max,
            integrator_min=self.integrator_min,
            integrator_max=self.integrator_max,
            derivative_filter=self.derivative_filter,
        )


class ControllerBase(ABC):
    """Abstract base class for all controllers."""

    def __init__(self, name: str = "controller"):
        self.name = name
        self._enabled = True

    @abstractmethod
    def reset(self) -> None:
        """Reset controller state."""
        pass

    @abstractmethod
    def compute(self, reference: float, measurement: float, dt: float) -> float:
        """Compute control output."""
        pass

    def enable(self) -> None:
        """Enable controller."""
        self._enabled = True

    def disable(self) -> None:
        """Disable controller and reset."""
        self._enabled = False
        self.reset()

    @property
    def enabled(self) -> bool:
        return self._enabled


class PIDController(ControllerBase):
    """
    PID Controller with anti-windup and derivative filtering.

    Features:
    - Back-calculation anti-windup
    - First-order derivative filter
    - Output and integrator saturation
    - Optional feedforward
    """

    def __init__(
        self,
        gains: ControllerGains,
        name: str = "pid",
    ):
        super().__init__(name)
        self.gains = gains

        # State variables
        self._integrator = 0.0
        self._prev_error = 0.0
        self._prev_derivative = 0.0
        self._prev_output = 0.0

    def reset(self) -> None:
        """Reset controller state."""
        self._integrator = 0.0
        self._prev_error = 0.0
        self._prev_derivative = 0.0
        self._prev_output = 0.0

    def compute(
        self,
        reference: float,
        measurement: float,
        dt: float,
        feedforward: float = 0.0,
    ) -> float:
        """
        Compute PID control output.

        Args:
            reference: Desired setpoint
            measurement: Current measured value
            dt: Time step (seconds)
            feedforward: Optional feedforward term

        Returns:
            Control output (saturated)
        """
        if not self._enabled or dt <= 0:
            return 0.0

        # Error
        error = reference - measurement

        # Proportional term
        P = self.gains.Kp * error

        # Integral term with anti-windup
        self._integrator += self.gains.Ki * error * dt
        self._integrator = np.clip(
            self._integrator,
            self.gains.integrator_min,
            self.gains.integrator_max,
        )
        I = self._integrator

        # Derivative term with filtering
        if dt > 0:
            raw_derivative = (error - self._prev_error) / dt
            # Low-pass filter on derivative
            alpha = self.gains.derivative_filter
            filtered_derivative = alpha * self._prev_derivative + (1 - alpha) * raw_derivative
            self._prev_derivative = filtered_derivative
            D = self.gains.Kd * filtered_derivative
        else:
            D = 0.0

        self._prev_error = error

        # Total output with feedforward
        output = P + I + D + feedforward

        # Output saturation
        output_saturated = np.clip(output, self.gains.output_min, self.gains.output_max)

        # Back-calculation anti-windup
        if self.gains.Ki > 0:
            saturation_error = output_saturated - output
            self._integrator += saturation_error * 0.1  # Anti-windup gain

        self._prev_output = output_saturated
        return output_saturated

    def set_gains(self, gains: ControllerGains) -> None:
        """Update controller gains (for gain scheduling)."""
        self.gains = gains

    def get_state(self) -> dict:
        """Get internal state for debugging."""
        return {
            'integrator': self._integrator,
            'prev_error': self._prev_error,
            'prev_derivative': self._prev_derivative,
            'prev_output': self._prev_output,
        }


class CascadedPID:
    """
    Cascaded PID controller (outer + inner loop).

    Typical use:
    - Outer loop: Position → Velocity command
    - Inner loop: Velocity → Acceleration/Force command
    """

    def __init__(
        self,
        outer_gains: ControllerGains,
        inner_gains: ControllerGains,
        name: str = "cascaded_pid",
    ):
        self.name = name
        self.outer = PIDController(outer_gains, f"{name}_outer")
        self.inner = PIDController(inner_gains, f"{name}_inner")

        # Rate limit on outer loop output (inner loop reference)
        self.inner_ref_limit = np.inf

    def reset(self) -> None:
        """Reset both loops."""
        self.outer.reset()
        self.inner.reset()

    def compute(
        self,
        outer_reference: float,
        outer_measurement: float,
        inner_measurement: float,
        dt: float,
        feedforward: float = 0.0,
    ) -> tuple[float, float]:
        """
        Compute cascaded control output.

        Args:
            outer_reference: Outer loop setpoint (e.g., position)
            outer_measurement: Outer loop measurement (e.g., position)
            inner_measurement: Inner loop measurement (e.g., velocity)
            dt: Time step
            feedforward: Feedforward to inner loop

        Returns:
            (control_output, inner_reference) - inner loop command
        """
        # Outer loop: generates reference for inner loop
        inner_reference = self.outer.compute(outer_reference, outer_measurement, dt)
        inner_reference = np.clip(inner_reference, -self.inner_ref_limit, self.inner_ref_limit)

        # Inner loop: generates control output
        output = self.inner.compute(inner_reference, inner_measurement, dt, feedforward)

        return output, inner_reference


@dataclass
class ControllerState:
    """Container for multi-axis controller states."""
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    thrust: float = 0.0

    def as_array(self) -> np.ndarray:
        return np.array([self.roll, self.pitch, self.yaw, self.thrust])

    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'ControllerState':
        return cls(roll=arr[0], pitch=arr[1], yaw=arr[2], thrust=arr[3])


@dataclass
class AttitudeCommand:
    """Commanded attitude angles and thrust."""
    roll_rad: float = 0.0       # φ - bank angle
    pitch_rad: float = 0.0      # θ - pitch angle
    yaw_rad: float = 0.0        # ψ - heading
    thrust_N: float = 0.0       # Total thrust command

    # Rate commands (optional, for rate mode)
    roll_rate_rps: float = 0.0   # p - roll rate
    pitch_rate_rps: float = 0.0  # q - pitch rate
    yaw_rate_rps: float = 0.0    # r - yaw rate


@dataclass
class MomentCommand:
    """Commanded moments for control allocation."""
    L: float = 0.0  # Roll moment (Nm)
    M: float = 0.0  # Pitch moment (Nm)
    N: float = 0.0  # Yaw moment (Nm)
    T: float = 0.0  # Total thrust (N)

    def as_array(self) -> np.ndarray:
        return np.array([self.T, self.L, self.M, self.N])


@dataclass
class VelocityCommand:
    """Commanded velocities."""
    vx: float = 0.0  # North velocity (m/s)
    vy: float = 0.0  # East velocity (m/s)
    vz: float = 0.0  # Down velocity (m/s) - negative = climb

    def as_array(self) -> np.ndarray:
        return np.array([self.vx, self.vy, self.vz])

    @property
    def horizontal_speed(self) -> float:
        return np.sqrt(self.vx**2 + self.vy**2)


@dataclass
class PositionCommand:
    """Commanded position."""
    x: float = 0.0        # North (m)
    y: float = 0.0        # East (m)
    z: float = 0.0        # Down (m) - negative = altitude
    heading: float = 0.0  # Yaw angle (rad)

    def as_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])

    @property
    def altitude(self) -> float:
        """Altitude above reference (positive up)."""
        return -self.z
