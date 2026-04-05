"""
Mission Manager - Guidance Layer.

High-level mission state machine that sequences through mission phases.
Coordinates between planning layer outputs and control layer.
"""

import numpy as np
from dataclasses import dataclass
from collections.abc import Callable
from enum import Enum, auto


class MissionPhase(Enum):
    """Mission phases for defense eVTOL."""
    IDLE = auto()           # Pre-mission
    PREFLIGHT = auto()      # System checks
    TAKEOFF = auto()        # Vertical takeoff
    CLIMB = auto()          # Climb to cruise altitude
    TRANSITION = auto()     # Hover to cruise transition
    CRUISE = auto()         # Forward flight
    LOITER = auto()         # Hold position/orbit
    APPROACH = auto()       # Approach to landing
    LANDING = auto()        # Vertical landing
    SHUTDOWN = auto()       # Post-mission

    # Defense-specific phases
    INGRESS = auto()        # Low-altitude penetration
    STRIKE = auto()         # Target engagement
    EGRESS = auto()         # Return from target
    EVASION = auto()        # Threat evasion maneuver


@dataclass
class MissionConfig:
    """Mission configuration."""
    # Takeoff/landing
    takeoff_altitude: float = 50.0      # m
    landing_altitude: float = 5.0       # m for flare

    # Cruise
    cruise_altitude: float = 500.0      # m
    cruise_speed: float = 50.0          # m/s

    # Transition
    transition_altitude: float = 100.0  # m
    transition_speed: float = 25.0      # m/s

    # Loiter
    loiter_radius: float = 200.0        # m
    loiter_speed: float = 30.0          # m/s

    # Timeouts
    phase_timeout: float = 300.0        # s


@dataclass
class MissionState:
    """Current mission state."""
    phase: MissionPhase = MissionPhase.IDLE
    phase_start_time: float = 0.0
    phase_elapsed: float = 0.0

    # Waypoint tracking
    current_waypoint: int = 0
    total_waypoints: int = 0

    # Progress
    distance_to_target: float = float('inf')
    distance_traveled: float = 0.0

    # Status
    is_complete: bool = False
    abort_requested: bool = False

    # Phase outputs
    altitude_target: float = 0.0
    speed_target: float = 0.0
    heading_target: float = 0.0


class MissionManager:
    """
    Mission state machine and coordinator.

    Manages:
    - Mission phase sequencing
    - Phase transition logic
    - Target/waypoint management
    - Mission abort handling
    """

    def __init__(self, config: MissionConfig | None = None):
        self.config = config or MissionConfig()
        self.state = MissionState()

        self._phase_handlers: dict[MissionPhase, Callable] = {
            MissionPhase.IDLE: self._handle_idle,
            MissionPhase.PREFLIGHT: self._handle_preflight,
            MissionPhase.TAKEOFF: self._handle_takeoff,
            MissionPhase.CLIMB: self._handle_climb,
            MissionPhase.TRANSITION: self._handle_transition,
            MissionPhase.CRUISE: self._handle_cruise,
            MissionPhase.LOITER: self._handle_loiter,
            MissionPhase.APPROACH: self._handle_approach,
            MissionPhase.LANDING: self._handle_landing,
            MissionPhase.INGRESS: self._handle_ingress,
            MissionPhase.STRIKE: self._handle_strike,
            MissionPhase.EGRESS: self._handle_egress,
            MissionPhase.EVASION: self._handle_evasion,
        }

        self._mission_plan: list[MissionPhase] = []
        self._mission_idx: int = 0

    def reset(self) -> None:
        """Reset mission state."""
        self.state = MissionState()
        self._mission_idx = 0

    def set_mission_plan(self, phases: list[MissionPhase]) -> None:
        """
        Set mission phase sequence.

        Example: [TAKEOFF, CLIMB, TRANSITION, CRUISE, APPROACH, LANDING]
        """
        self._mission_plan = phases
        self._mission_idx = 0
        if len(phases) > 0:
            self._transition_to(phases[0])

    def set_standard_mission(self) -> None:
        """Set standard takeoff -> cruise -> land mission."""
        self.set_mission_plan([
            MissionPhase.PREFLIGHT,
            MissionPhase.TAKEOFF,
            MissionPhase.CLIMB,
            MissionPhase.TRANSITION,
            MissionPhase.CRUISE,
            MissionPhase.APPROACH,
            MissionPhase.LANDING,
        ])

    def set_defense_mission(self) -> None:
        """Set defense ingress/egress mission."""
        self.set_mission_plan([
            MissionPhase.PREFLIGHT,
            MissionPhase.TAKEOFF,
            MissionPhase.TRANSITION,
            MissionPhase.INGRESS,
            MissionPhase.STRIKE,
            MissionPhase.EGRESS,
            MissionPhase.APPROACH,
            MissionPhase.LANDING,
        ])

    def update(
        self,
        t: float,
        altitude: float,
        speed: float,
        heading: float,
        position: np.ndarray,
        dt: float,
    ) -> MissionState:
        """
        Update mission state machine.

        Args:
            t: Current time (s)
            altitude: Current altitude (m, positive up)
            speed: Current airspeed (m/s)
            heading: Current heading (rad)
            position: Current position [x, y, z] NED
            dt: Time step

        Returns:
            Updated mission state with targets
        """
        self.state.phase_elapsed = t - self.state.phase_start_time

        # Handle abort
        if self.state.abort_requested:
            self._handle_abort(altitude, speed)
            return self.state

        # Run phase handler
        handler = self._phase_handlers.get(self.state.phase)
        if handler:
            phase_complete = handler(t, altitude, speed, heading, position, dt)

            if phase_complete:
                self._advance_phase(t)

        return self.state

    def _transition_to(self, phase: MissionPhase) -> None:
        """Transition to new phase."""
        self.state.phase = phase
        self.state.phase_start_time = 0.0
        self.state.phase_elapsed = 0.0

    def _advance_phase(self, t: float) -> None:
        """Advance to next mission phase."""
        self._mission_idx += 1

        if self._mission_idx >= len(self._mission_plan):
            self.state.is_complete = True
            self.state.phase = MissionPhase.SHUTDOWN
        else:
            self.state.phase = self._mission_plan[self._mission_idx]
            self.state.phase_start_time = t
            self.state.phase_elapsed = 0.0

    def _handle_idle(self, *args) -> bool:
        """IDLE phase - waiting for mission start."""
        self.state.altitude_target = 0.0
        self.state.speed_target = 0.0
        return False  # Wait for explicit start

    def _handle_preflight(self, t, altitude, speed, heading, position, dt) -> bool:
        """PREFLIGHT - system checks."""
        self.state.altitude_target = 0.0
        self.state.speed_target = 0.0
        # Auto-complete after 2 seconds
        return self.state.phase_elapsed > 2.0

    def _handle_takeoff(self, t, altitude, speed, heading, position, dt) -> bool:
        """TAKEOFF - vertical climb to takeoff altitude."""
        self.state.altitude_target = self.config.takeoff_altitude
        self.state.speed_target = 0.0  # Hover
        self.state.heading_target = heading  # Hold heading

        # Complete when altitude reached
        return altitude >= self.config.takeoff_altitude - 5.0

    def _handle_climb(self, t, altitude, speed, heading, position, dt) -> bool:
        """CLIMB - climb to transition/cruise altitude."""
        self.state.altitude_target = self.config.transition_altitude
        self.state.speed_target = 5.0  # Slow forward

        return altitude >= self.config.transition_altitude - 10.0

    def _handle_transition(self, t, altitude, speed, heading, position, dt) -> bool:
        """TRANSITION - hover to cruise mode."""
        self.state.altitude_target = self.config.cruise_altitude
        self.state.speed_target = self.config.transition_speed

        # Complete when speed and altitude reached
        return (
            speed >= self.config.transition_speed - 5.0 and
            altitude >= self.config.cruise_altitude - 20.0
        )

    def _handle_cruise(self, t, altitude, speed, heading, position, dt) -> bool:
        """CRUISE - forward flight."""
        self.state.altitude_target = self.config.cruise_altitude
        self.state.speed_target = self.config.cruise_speed

        # Complete when all waypoints visited (checked externally)
        return False

    def _handle_loiter(self, t, altitude, speed, heading, position, dt) -> bool:
        """LOITER - hold position or orbit."""
        self.state.speed_target = self.config.loiter_speed
        # Loiter completes on external command
        return False

    def _handle_approach(self, t, altitude, speed, heading, position, dt) -> bool:
        """APPROACH - descend and slow for landing."""
        self.state.altitude_target = self.config.landing_altitude + 50.0
        self.state.speed_target = 10.0

        return (
            altitude <= self.config.landing_altitude + 60.0 and
            speed <= 12.0
        )

    def _handle_landing(self, t, altitude, speed, heading, position, dt) -> bool:
        """LANDING - vertical descent."""
        self.state.altitude_target = 0.0
        self.state.speed_target = 0.0

        return altitude <= 2.0

    def _handle_ingress(self, t, altitude, speed, heading, position, dt) -> bool:
        """INGRESS - low-altitude penetration."""
        # Low altitude, moderate speed
        self.state.altitude_target = 50.0
        self.state.speed_target = 40.0
        return False  # Complete on waypoint

    def _handle_strike(self, t, altitude, speed, heading, position, dt) -> bool:
        """STRIKE - target engagement at low altitude, reduced speed for precision."""
        self.state.altitude_target = 30.0   # NOE altitude for target engagement
        self.state.speed_target = 20.0      # Slow for precision
        # Strike phase completes when the mission plan externally advances the waypoint
        return False

    def _handle_egress(self, t, altitude, speed, heading, position, dt) -> bool:
        """EGRESS - return from target."""
        self.state.altitude_target = 100.0
        self.state.speed_target = self.config.cruise_speed
        return False  # Complete on waypoint

    def _handle_evasion(self, t, altitude, speed, heading, position, dt) -> bool:
        """EVASION - threat evasion maneuver: descend to NOE altitude, maximum speed."""
        self.state.altitude_target = 15.0                       # NOE — 15 m AGL
        self.state.speed_target = self.config.cruise_speed      # Maximum available speed
        # Evasion completes when threat is no longer imminent (external command)
        return False

    def _handle_abort(self, altitude: float, speed: float) -> None:
        """Handle mission abort - safe return."""
        if altitude > 100:
            self.state.phase = MissionPhase.APPROACH
        else:
            self.state.phase = MissionPhase.LANDING
        self.state.altitude_target = 0.0
        self.state.speed_target = 0.0

    def request_abort(self) -> None:
        """Request mission abort."""
        self.state.abort_requested = True

    def force_phase(self, phase: MissionPhase) -> None:
        """Force transition to specific phase."""
        self._transition_to(phase)

    def complete_waypoint(self) -> None:
        """Mark current waypoint as complete."""
        self.state.current_waypoint += 1

    @property
    def current_phase(self) -> MissionPhase:
        """Get current mission phase."""
        return self.state.phase
