"""
Flight Mode Manager - Mode State Machine.

Manages flight mode transitions and provides mode-specific control parameters.
"""

import numpy as np
from dataclasses import dataclass
from collections.abc import Callable
from enum import Enum, auto


class FlightMode(Enum):
    """Flight modes."""
    GROUND = auto()      # On ground
    HOVER = auto()       # Vertical flight, nacelles up
    TRANSITION = auto()  # Transitioning between hover and cruise
    CRUISE = auto()      # Forward flight, nacelles forward
    EMERGENCY = auto()   # Emergency mode


@dataclass
class ModeTransitionConfig:
    """Configuration for mode transitions."""
    # Hover to transition
    hover_to_transition_speed: float = 8.0    # m/s
    hover_to_transition_alt: float = 30.0     # m

    # Transition to cruise
    transition_to_cruise_speed: float = 30.0  # m/s
    transition_to_cruise_nacelle: float = np.radians(25)  # rad

    # Cruise to transition (deceleration)
    cruise_to_transition_speed: float = 25.0  # m/s

    # Transition to hover
    transition_to_hover_speed: float = 10.0   # m/s
    transition_to_hover_nacelle: float = np.radians(70)

    # Ground transitions
    ground_to_hover_alt: float = 2.0          # m
    hover_to_ground_alt: float = 1.0          # m
    hover_to_ground_vz: float = 0.5           # m/s descent


@dataclass
class FlightModeState:
    """Current flight mode state."""
    mode: FlightMode = FlightMode.GROUND
    time_in_mode: float = 0.0
    transition_progress: float = 0.0  # 0 = start, 1 = complete

    # Mode-specific flags
    can_transition_to_cruise: bool = False
    hover_established: bool = False
    landing_phase: bool = False


class FlightModeManager:
    """
    Flight mode state machine.

    Manages transitions between:
    - GROUND ↔ HOVER
    - HOVER ↔ TRANSITION
    - TRANSITION ↔ CRUISE
    - Any → EMERGENCY

    Each mode has different control gains, limits, and strategies.
    """

    def __init__(self, config: ModeTransitionConfig | None = None):
        self.config = config or ModeTransitionConfig()
        self.state = FlightModeState()

        self._mode_start_time = 0.0
        self._callbacks: dict[FlightMode, Callable] = {}

    def reset(self) -> None:
        """Reset to ground mode."""
        self.state = FlightModeState()
        self._mode_start_time = 0.0

    def update(
        self,
        t: float,
        altitude: float,
        airspeed: float,
        ground_speed: float,
        vz: float,
        nacelle_angle: float,
        dt: float,
    ) -> FlightModeState:
        """
        Update flight mode state machine.

        Args:
            t: Current time (s)
            altitude: Altitude above ground (m)
            airspeed: Airspeed (m/s)
            ground_speed: Ground speed (m/s)
            vz: Vertical velocity (m/s, negative = climbing)
            nacelle_angle: Average nacelle angle (rad)
            dt: Time step

        Returns:
            Updated flight mode state
        """
        self.state.time_in_mode = t - self._mode_start_time

        current_mode = self.state.mode
        new_mode = self._check_transitions(
            altitude, airspeed, ground_speed, vz, nacelle_angle
        )

        if new_mode != current_mode:
            self._transition_to(new_mode, t)

        # Update mode-specific state
        self._update_mode_state(altitude, airspeed, nacelle_angle)

        return self.state

    def _check_transitions(
        self,
        altitude: float,
        airspeed: float,
        ground_speed: float,
        vz: float,
        nacelle_angle: float,
    ) -> FlightMode:
        """Check for mode transitions."""
        mode = self.state.mode

        if mode == FlightMode.GROUND:
            # Ground → Hover when lifted off
            if altitude > self.config.ground_to_hover_alt:
                return FlightMode.HOVER

        elif mode == FlightMode.HOVER:
            # Hover → Ground when landed
            if altitude < self.config.hover_to_ground_alt and abs(vz) < self.config.hover_to_ground_vz:
                return FlightMode.GROUND

            # Hover → Transition when accelerating
            if airspeed > self.config.hover_to_transition_speed:
                return FlightMode.TRANSITION

        elif mode == FlightMode.TRANSITION:
            # Transition → Hover when slowed
            if airspeed < self.config.transition_to_hover_speed:
                return FlightMode.HOVER

            # Transition → Cruise when fast and nacelles tilted
            if (airspeed > self.config.transition_to_cruise_speed and
                nacelle_angle < self.config.transition_to_cruise_nacelle):
                return FlightMode.CRUISE

        elif mode == FlightMode.CRUISE:
            # Cruise → Transition when slowing
            if airspeed < self.config.cruise_to_transition_speed:
                return FlightMode.TRANSITION

        return mode

    def _transition_to(self, new_mode: FlightMode, t: float) -> None:
        """Execute mode transition."""
        old_mode = self.state.mode

        self.state.mode = new_mode
        self._mode_start_time = t
        self.state.time_in_mode = 0.0
        self.state.transition_progress = 0.0

        # Call callback if registered
        if new_mode in self._callbacks:
            self._callbacks[new_mode](old_mode, new_mode)

    def _update_mode_state(
        self,
        altitude: float,
        airspeed: float,
        nacelle_angle: float,
    ) -> None:
        """Update mode-specific state."""
        mode = self.state.mode

        if mode == FlightMode.HOVER:
            self.state.hover_established = self.state.time_in_mode > 2.0
            self.state.can_transition_to_cruise = altitude > self.config.hover_to_transition_alt

        elif mode == FlightMode.TRANSITION:
            # Transition progress based on speed
            speed_range = (
                self.config.transition_to_cruise_speed -
                self.config.hover_to_transition_speed
            )
            self.state.transition_progress = np.clip(
                (airspeed - self.config.hover_to_transition_speed) / speed_range,
                0.0, 1.0
            )

        elif mode == FlightMode.CRUISE:
            self.state.transition_progress = 1.0

    def force_mode(self, mode: FlightMode, t: float = 0.0) -> None:
        """Force transition to specific mode."""
        self._transition_to(mode, t)

    def emergency(self) -> None:
        """Enter emergency mode."""
        self.state.mode = FlightMode.EMERGENCY

    def register_callback(self, mode: FlightMode, callback: Callable) -> None:
        """Register callback for mode entry."""
        self._callbacks[mode] = callback

    @property
    def is_hover(self) -> bool:
        return self.state.mode == FlightMode.HOVER

    @property
    def is_cruise(self) -> bool:
        return self.state.mode == FlightMode.CRUISE

    @property
    def is_transition(self) -> bool:
        return self.state.mode == FlightMode.TRANSITION

    @property
    def is_ground(self) -> bool:
        return self.state.mode == FlightMode.GROUND

    def get_control_blending(self) -> float:
        """
        Get blending factor for control mixing.

        Returns value 0-1:
        - 0 = pure hover control
        - 1 = pure cruise control
        """
        mode = self.state.mode

        if mode == FlightMode.GROUND or mode == FlightMode.HOVER:
            return 0.0
        elif mode == FlightMode.CRUISE:
            return 1.0
        elif mode == FlightMode.TRANSITION:
            return self.state.transition_progress
        else:
            return 0.0
