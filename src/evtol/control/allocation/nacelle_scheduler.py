"""
Nacelle Scheduler - Allocation Layer.

Schedules nacelle tilt angle based on flight phase and airspeed.
Manages the transition between hover (nacelle up) and cruise (nacelle forward).
"""

import numpy as np
from dataclasses import dataclass
from enum import Enum


class FlightMode(Enum):
    """Flight mode based on nacelle position."""
    HOVER = "hover"           # Nacelle 70-90°
    TRANSITION = "transition" # Nacelle 20-70°
    CRUISE = "cruise"         # Nacelle 0-20°


@dataclass
class NacelleConfig:
    """Configuration for nacelle scheduling."""
    # Angle limits (rad)
    min_angle: float = np.radians(0)    # Full forward (cruise)
    max_angle: float = np.radians(90)   # Full up (hover)

    # Rate limits
    max_rate: float = np.radians(15)    # rad/s

    # Transition thresholds (based on airspeed)
    hover_speed: float = 10.0           # m/s - below this, full hover mode
    cruise_speed: float = 35.0          # m/s - above this, full cruise mode

    # Stall protection
    min_speed_for_cruise: float = 25.0  # m/s - don't go below this in cruise mode

    # Blending region bounds
    hover_nacelle: float = np.radians(85)     # Nacelle angle in hover
    cruise_nacelle: float = np.radians(5)     # Nacelle angle in cruise


class NacelleScheduler:
    """
    Schedules nacelle tilt angle for tiltrotor eVTOL.

    The nacelle angle determines:
    - Thrust direction (vertical in hover, forward in cruise)
    - Control authority (roll/pitch via thrust in hover, via surfaces in cruise)
    - Aerodynamic efficiency

    Scheduling strategy:
    - Below hover_speed: maintain full hover nacelle (85-90°)
    - Above cruise_speed: maintain cruise nacelle (0-10°)
    - In between: linear interpolation with stall protection
    """

    def __init__(self, config: NacelleConfig | None = None):
        self.config = config or NacelleConfig()

        # Number of nacelles (typically 2 for quad tiltrotor)
        self.n_nacelles = 2

        # Current state
        self._nacelle_angles = np.full(self.n_nacelles, self.config.max_angle)
        self._target_angles = np.full(self.n_nacelles, self.config.max_angle)
        self._flight_mode = FlightMode.HOVER

    def reset(self) -> None:
        """Reset to hover position."""
        self._nacelle_angles = np.full(self.n_nacelles, self.config.max_angle)
        self._target_angles = np.full(self.n_nacelles, self.config.max_angle)
        self._flight_mode = FlightMode.HOVER

    def schedule(
        self,
        airspeed: float,
        altitude: float,
        dt: float,
        commanded_mode: FlightMode | None = None,
    ) -> np.ndarray:
        """
        Schedule nacelle angle based on airspeed.

        Args:
            airspeed: Current airspeed (m/s)
            altitude: Current altitude (m) - for ground proximity
            dt: Time step (s)
            commanded_mode: Optional mode override

        Returns:
            nacelle_angles: Array of nacelle angles (rad)
        """
        # Determine target based on airspeed
        if commanded_mode is not None:
            if commanded_mode == FlightMode.HOVER:
                target = self.config.hover_nacelle
            elif commanded_mode == FlightMode.CRUISE:
                target = self.config.cruise_nacelle
            else:
                target = self._compute_transition_angle(airspeed)
        else:
            target = self._compute_transition_angle(airspeed)

        # Ground proximity protection
        if altitude < 20.0:
            # Near ground, bias toward hover
            ground_factor = altitude / 20.0
            target = target * ground_factor + self.config.hover_nacelle * (1 - ground_factor)

        # Stall protection
        if airspeed < self.config.min_speed_for_cruise:
            min_safe_angle = self._compute_min_safe_angle(airspeed)
            target = max(target, min_safe_angle)

        # Update targets
        self._target_angles[:] = target

        # Rate limit the transition
        for i in range(self.n_nacelles):
            angle_error = self._target_angles[i] - self._nacelle_angles[i]
            max_change = self.config.max_rate * dt

            if abs(angle_error) > max_change:
                self._nacelle_angles[i] += np.sign(angle_error) * max_change
            else:
                self._nacelle_angles[i] = self._target_angles[i]

        # Update flight mode
        self._update_flight_mode()

        return self._nacelle_angles.copy()

    def _compute_transition_angle(self, airspeed: float) -> float:
        """Compute nacelle angle for transition region."""
        if airspeed <= self.config.hover_speed:
            return self.config.hover_nacelle
        elif airspeed >= self.config.cruise_speed:
            return self.config.cruise_nacelle
        else:
            # Linear interpolation
            t = (airspeed - self.config.hover_speed) / (
                self.config.cruise_speed - self.config.hover_speed
            )
            return (1 - t) * self.config.hover_nacelle + t * self.config.cruise_nacelle

    def _compute_min_safe_angle(self, airspeed: float) -> float:
        """
        Compute minimum safe nacelle angle to avoid stall.

        At low speeds, the wing cannot provide enough lift.
        Need sufficient thrust component for lift.
        """
        if airspeed < 10:
            return self.config.hover_nacelle
        elif airspeed < 20:
            # Need at least 60° nacelle
            return np.radians(60)
        elif airspeed < 30:
            # At least 30°
            return np.radians(30)
        else:
            return self.config.cruise_nacelle

    def _update_flight_mode(self) -> None:
        """Update flight mode based on nacelle position."""
        avg_angle = np.mean(self._nacelle_angles)

        if avg_angle >= np.radians(70):
            self._flight_mode = FlightMode.HOVER
        elif avg_angle <= np.radians(20):
            self._flight_mode = FlightMode.CRUISE
        else:
            self._flight_mode = FlightMode.TRANSITION

    def get_flight_mode(self) -> FlightMode:
        """Get current flight mode."""
        return self._flight_mode

    def get_nacelle_angles(self) -> np.ndarray:
        """Get current nacelle angles."""
        return self._nacelle_angles.copy()

    def get_thrust_fraction_vertical(self) -> float:
        """Get fraction of thrust that is vertical."""
        avg_angle = np.mean(self._nacelle_angles)
        return np.sin(avg_angle)

    def get_thrust_fraction_forward(self) -> float:
        """Get fraction of thrust that is forward."""
        avg_angle = np.mean(self._nacelle_angles)
        return np.cos(avg_angle)

    def force_hover(self) -> None:
        """Force hover mode (emergency)."""
        self._nacelle_angles[:] = self.config.hover_nacelle
        self._target_angles[:] = self.config.hover_nacelle
        self._flight_mode = FlightMode.HOVER

    def get_wing_lift_required(
        self,
        weight: float,
        nacelle_angle: float,
        thrust: float,
    ) -> float:
        """
        Calculate wing lift required for level flight.

        Args:
            weight: Aircraft weight (N)
            nacelle_angle: Current nacelle angle (rad)
            thrust: Total thrust (N)

        Returns:
            lift_required: Lift from wing needed (N)
        """
        thrust_vertical = thrust * np.sin(nacelle_angle)
        return weight - thrust_vertical
