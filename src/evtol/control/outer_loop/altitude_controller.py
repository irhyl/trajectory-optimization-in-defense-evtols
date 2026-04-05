"""
Altitude Controller - Outer Loop.

Controls altitude (z) and climb rate (Vz).
Outputs thrust command and/or pitch adjustment.

Works in both hover and cruise modes with different control strategies.
"""

from dataclasses import dataclass

import numpy as np

from ..controller_base import (
    PIDController,
    ControllerGains,
)


@dataclass
class AltitudeGains:
    """Gains for altitude control."""
    altitude: ControllerGains = None      # Outer: altitude -> climb rate
    climb_rate: ControllerGains = None    # Inner: climb rate -> thrust

    # Limits
    max_climb_rate: float = 10.0   # m/s
    max_descent_rate: float = 8.0  # m/s (positive value)

    # Gravity compensation
    hover_thrust_n: float = 24525.0  # N — mg for 2500 kg vehicle

    def __post_init__(self):
        if self.altitude is None:
            # Outer loop: altitude error -> climb rate command
            # Conservative gains for smooth response
            self.altitude = ControllerGains(
                Kp=0.8,     # Climb rate (m/s) per meter altitude error
                Ki=0.1,     # Integral for steady-state accuracy
                Kd=0.2,     # Small derivative for damping
                output_min=-10.0, output_max=10.0,  # m/s climb rate limits
                integrator_min=-5.0, integrator_max=5.0,
                derivative_filter=0.8,
            )
        if self.climb_rate is None:
            # Inner loop: climb rate error -> thrust delta
            # Higher bandwidth than altitude loop
            self.climb_rate = ControllerGains(
                Kp=3000.0,   # N per m/s climb rate error (mg/10 per m/s)
                Ki=1500.0,   # Strong integral for steady-state
                Kd=200.0,    # Light derivative
                output_min=-15000.0, output_max=15000.0,  # ±60% hover thrust
                integrator_min=-8000.0, integrator_max=8000.0,
                derivative_filter=0.9,  # Heavy filtering
            )


class AltitudeController:
    """
    Altitude and climb rate controller.

    Cascaded structure:
    1. Outer loop: altitude error -> climb rate command
    2. Inner loop: climb rate error -> thrust command

    The output thrust is added to hover thrust (gravity compensation).

    Note: In NED frame, z is positive down, so:
    - Positive altitude error (below target) -> need to climb -> negative Vz
    - Thrust increase -> acceleration up -> negative acceleration in NED
    """

    def __init__(self, gains: AltitudeGains | None = None):
        self.gains = gains or AltitudeGains()

        self._alt_ctrl = PIDController(self.gains.altitude, "altitude")
        self._vz_ctrl = PIDController(self.gains.climb_rate, "climb_rate")

        # For direct climb rate commands
        self._direct_vz_mode = False

    def reset(self) -> None:
        """Reset controllers."""
        self._alt_ctrl.reset()
        self._vz_ctrl.reset()

    def compute(
        self,
        alt_cmd: float,
        alt: float,
        vz: float,
        dt: float,
    ) -> float:
        """
        Compute thrust command from altitude error.

        Args:
            alt_cmd: Commanded altitude [m, positive up]
            alt:     Current altitude  [m, positive up]
            vz:      Current vertical velocity in NED frame [m/s]; negative = climbing
            dt:      Time step [s]

        Returns:
            thrust_cmd: Total thrust command [N] including gravity compensation
        """
        # Convert NED Vz to climb rate (positive = climbing)
        climb_rate = -vz

        # Altitude loop: altitude error -> climb rate command
        climb_rate_cmd = self._alt_ctrl.compute(alt_cmd, alt, dt)

        # Limit climb/descent rate
        climb_rate_cmd = np.clip(
            climb_rate_cmd,
            -self.gains.max_descent_rate,
            self.gains.max_climb_rate,
        )

        # Climb rate loop: climb rate error -> thrust delta
        thrust_delta = self._vz_ctrl.compute(climb_rate_cmd, climb_rate, dt)

        # Total thrust = hover thrust + delta, clamped to [0.4, 1.6] × hover
        thrust_cmd = np.clip(
            self.gains.hover_thrust_n + thrust_delta,
            0.4 * self.gains.hover_thrust_n,
            1.6 * self.gains.hover_thrust_n,
        )

        return thrust_cmd

    def compute_direct(
        self,
        climb_rate_cmd: float,
        vz: float,
        dt: float,
    ) -> float:
        """
        Compute thrust command directly from a climb rate command (bypass altitude loop).

        Args:
            climb_rate_cmd: Desired climb rate [m/s, positive = climbing]
            vz:             Current vertical velocity in NED frame [m/s]
            dt:             Time step [s]

        Returns:
            thrust_cmd: Total thrust command [N]
        """
        climb_rate = -vz
        climb_rate_cmd = np.clip(
            climb_rate_cmd,
            -self.gains.max_descent_rate,
            self.gains.max_climb_rate,
        )
        thrust_delta = self._vz_ctrl.compute(climb_rate_cmd, climb_rate, dt)
        return np.clip(
            self.gains.hover_thrust_n + thrust_delta,
            0.4 * self.gains.hover_thrust_n,
            1.6 * self.gains.hover_thrust_n,
        )

    def compute_for_nacelle_angle(
        self,
        thrust_cmd: float,
        nacelle_angle_rad: float,
    ) -> float:
        """
        Adjust thrust command for nacelle angle.

        When nacelles are tilted, only sin(nacelle) of thrust is vertical.
        Need to increase total thrust to maintain vertical component.

        Args:
            thrust_cmd: Desired vertical thrust (N)
            nacelle_angle_rad: Current nacelle tilt (rad, 0=horizontal, 90°=vertical)

        Returns:
            adjusted_thrust: Rotor thrust to achieve desired vertical force
        """
        # Clamp to ≥ 0.2 so we never divide by near-zero at shallow nacelle angles.
        # At very small angles the wing provides most lift; this prevents
        # the rotor thrust command from diverging during transition.
        sin_nacelle = max(np.sin(nacelle_angle_rad), 0.2)
        return thrust_cmd / sin_nacelle

    def set_hover_thrust(self, hover_thrust_n: float) -> None:
        """Update hover thrust [N] for different weight configurations."""
        self.gains.hover_thrust_n = hover_thrust_n
