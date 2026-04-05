"""
Flight Controller - Main Integration.

Top-level flight controller that integrates:
- Guidance (trajectory tracking, path following)
- Outer loop (position, velocity, altitude, heading)
- Inner loop (attitude, rate)
- Allocation (control mixing, nacelle scheduling)
- Mode management (hover, transition, cruise)

This is the single interface between the planning layer and vehicle model.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Any

# Guidance
from .guidance import TrajectoryTracker, PathFollower, MissionManager, MissionPhase
from .guidance.trajectory_tracker import TrajectoryPoint

# Outer loop
from .outer_loop import (
    AltitudeController,
)

# Inner loop

# Allocation
from .allocation import ControlMixer, NacelleScheduler, RotorCommand
from .allocation.nacelle_scheduler import FlightMode as NacelleFlightMode

# Modes
from .modes import FlightModeManager, HoverMode, TransitionMode, CruiseMode
from .modes.flight_mode import FlightMode

# Base
from .controller_base import MomentCommand


@dataclass
class FlightControllerConfig:
    """Configuration for flight controller."""
    # Vehicle parameters
    mass: float = 2500.0           # kg
    hover_thrust: float = 24525.0  # N (mg)

    # Control update rate
    control_dt: float = 0.01       # 100 Hz

    # Limits
    max_thrust: float = 40000.0    # N
    max_moment: np.ndarray = field(default_factory=lambda: np.array([20000, 30000, 15000]))

    # Guidance
    trajectory_lookahead: float = 0.5  # s

    # Mode thresholds
    hover_speed: float = 10.0      # m/s
    cruise_speed: float = 35.0     # m/s


@dataclass
class FlightControllerState:
    """Current state of flight controller."""
    # Mode
    flight_mode: FlightMode = FlightMode.GROUND
    mission_phase: MissionPhase = MissionPhase.IDLE

    # Commands
    thrust_cmd: float = 0.0
    moment_cmd: np.ndarray = field(default_factory=lambda: np.zeros(3))
    nacelle_angles: np.ndarray = field(default_factory=lambda: np.array([np.radians(90), np.radians(90)]))

    # Tracking errors
    position_error: np.ndarray = field(default_factory=lambda: np.zeros(3))
    velocity_error: np.ndarray = field(default_factory=lambda: np.zeros(3))
    attitude_error: np.ndarray = field(default_factory=lambda: np.zeros(3))
    cross_track_error: float = 0.0

    # Status
    trajectory_complete: bool = False
    in_transition: bool = False


class FlightController:
    """
    Main flight controller integrating all subsystems.

    Usage:
        controller = FlightController()
        controller.load_trajectory(trajectory)

        while not controller.is_complete:
            rotor_cmd = controller.update(vehicle_state, dt)
            vehicle.apply_command(rotor_cmd)
    """

    def __init__(self, config: FlightControllerConfig | None = None):
        self.config = config or FlightControllerConfig()
        self.state = FlightControllerState()

        # ===== Initialize Subsystems =====

        # Guidance
        self.trajectory_tracker = TrajectoryTracker()
        self.path_follower = PathFollower()
        self.mission_manager = MissionManager()

        # Mode controllers
        self.hover_mode = HoverMode()
        self.transition_mode = TransitionMode()
        self.cruise_mode = CruiseMode()

        # Mode manager
        self.mode_manager = FlightModeManager()

        # Allocation
        self.control_mixer = ControlMixer()
        self.nacelle_scheduler = NacelleScheduler()

        # Standalone controllers (for direct use)
        self.altitude_ctrl = AltitudeController()
        self.altitude_ctrl.set_hover_thrust(self.config.hover_thrust)

        # Current trajectory setpoint
        self._current_setpoint: TrajectoryPoint | None = None

        # Control mode
        self._use_trajectory = False
        self._use_path = False
        self._manual_mode = False

    def reset(self) -> None:
        """Reset all controllers to initial state."""
        self.state = FlightControllerState()

        self.trajectory_tracker = TrajectoryTracker()
        self.path_follower = PathFollower()
        self.mission_manager.reset()

        self.hover_mode.reset()
        self.transition_mode.reset()
        self.cruise_mode.reset()

        self.mode_manager.reset()
        self.nacelle_scheduler.reset()

        self._current_setpoint = None
        self._use_trajectory = False
        self._use_path = False

    def load_trajectory(
        self,
        times: np.ndarray,
        positions: np.ndarray,
        velocities: np.ndarray | None = None,
        headings: np.ndarray | None = None,
    ) -> None:
        """
        Load trajectory from planning layer.

        Args:
            times: Time vector (N,)
            positions: Position array (N, 3) [x, y, altitude]
            velocities: Optional velocity array (N, 3)
            headings: Optional heading array (N,)
        """
        self.trajectory_tracker.load_from_planning_output(
            times, positions, velocities, headings
        )
        self._use_trajectory = True
        self._use_path = False

        # Force HOVER mode to start trajectory tracking
        self.mode_manager.force_mode(FlightMode.HOVER, 0.0)
        self.state.flight_mode = FlightMode.HOVER

    def load_waypoints(
        self,
        waypoints: np.ndarray,
        speeds: np.ndarray | None = None,
    ) -> None:
        """
        Load waypoints for path following.

        Args:
            waypoints: Position array (N, 3) [x, y, altitude]
            speeds: Optional speed array (N,)
        """
        self.path_follower.load_from_array(waypoints, speeds)
        self._use_path = True
        self._use_trajectory = False

    def set_mission(self, mission_type: str = "standard") -> None:
        """
        Set mission type.

        Args:
            mission_type: "standard" or "defense"
        """
        if mission_type == "defense":
            self.mission_manager.set_defense_mission()
        else:
            self.mission_manager.set_standard_mission()

    def update(
        self,
        t: float,
        position: np.ndarray,      # [x, y, z] NED
        velocity: np.ndarray,      # [vx, vy, vz] NED
        attitude: np.ndarray,      # [roll, pitch, yaw] rad
        angular_rates: np.ndarray, # [p, q, r] rad/s
        dt: float,
    ) -> RotorCommand:
        """
        Update flight controller.

        Main entry point called each timestep.

        Args:
            t: Current time (s)
            position: Position in NED frame (m)
            velocity: Velocity in NED frame (m/s)
            attitude: Euler angles [roll, pitch, yaw] (rad)
            angular_rates: Angular rates [p, q, r] (rad/s)
            dt: Time step (s)

        Returns:
            RotorCommand with thrust/speed for each rotor
        """
        # Extract state
        x, y, z = position
        vx, vy, vz = velocity
        roll, pitch, yaw = attitude
        p, q, r = angular_rates

        altitude = -z  # NED to altitude
        airspeed = np.sqrt(vx**2 + vy**2 + vz**2)
        ground_speed = np.sqrt(vx**2 + vy**2)

        # ===== Update Mode Manager =====
        mode_state = self.mode_manager.update(
            t, altitude, airspeed, ground_speed, vz,
            np.mean(self.state.nacelle_angles), dt
        )
        self.state.flight_mode = mode_state.mode

        # ===== Get Setpoint from Guidance =====
        if self._use_trajectory:
            setpoint, cross_track = self.trajectory_tracker.get_setpoint(t, position)
            self._current_setpoint = setpoint
            self.state.cross_track_error = cross_track
            self.state.trajectory_complete = self.trajectory_tracker.is_completed

            # Extract commands
            x_cmd, y_cmd, z_cmd = setpoint.x, setpoint.y, setpoint.z
            alt_cmd = -z_cmd
            heading_cmd = setpoint.heading
            speed_cmd = np.sqrt(setpoint.vx**2 + setpoint.vy**2)

        elif self._use_path:
            heading_cmd, speed_cmd, alt_cmd, cross_track = self.path_follower.compute_guidance(
                x, y, z, yaw, airspeed
            )
            self.state.cross_track_error = cross_track
            self.state.trajectory_complete = self.path_follower.is_completed
            x_cmd, y_cmd = x, y  # Not used in path mode

        else:
            # Manual/hover mode - hold position
            x_cmd, y_cmd = x, y
            alt_cmd = max(altitude, 2.0)  # At least 2m
            heading_cmd = yaw
            speed_cmd = 0.0

        # ===== Mode-Specific Control =====
        flight_mode = self.state.flight_mode

        if flight_mode == FlightMode.GROUND:
            # On ground - check if we should takeoff
            if self._use_trajectory and alt_cmd > 2.0:
                # We have a trajectory commanding altitude - use hover control
                thrust_cmd, moment_cmd = self.hover_mode.compute(
                    x_cmd, y_cmd, alt_cmd, heading_cmd,
                    x, y, altitude, vx, vy, vz,
                    roll, pitch, yaw, p, q, r, dt
                )
                # Force mode transition
                self.state.flight_mode = FlightMode.HOVER
            else:
                # Truly on ground - zero commands
                thrust_cmd = 0.0
                moment_cmd = MomentCommand(L=0, M=0, N=0)

        elif flight_mode == FlightMode.HOVER:
            thrust_cmd, moment_cmd = self.hover_mode.compute(
                x_cmd, y_cmd, alt_cmd, heading_cmd,
                x, y, altitude, vx, vy, vz,
                roll, pitch, yaw, p, q, r, dt
            )

        elif flight_mode == FlightMode.TRANSITION:
            self.state.in_transition = True
            from .modes.transition_mode import TransitionState as _TS
            _ts = _TS(
                airspeed=airspeed, vx=vx, vy=vy, vz=vz,
                alt=altitude, roll=roll, pitch=pitch, yaw=yaw,
                p=p, q=q, r=r,
                nacelle_angle=float(np.mean(self.state.nacelle_angles)),
            )
            thrust_cmd, moment_cmd, _ = self.transition_mode.compute(
                speed_cmd, heading_cmd, alt_cmd, _ts, dt
            )

        elif flight_mode == FlightMode.CRUISE:
            self.state.in_transition = False
            thrust_cmd, moment_cmd = self.cruise_mode.compute(
                speed_cmd, heading_cmd, alt_cmd,
                airspeed, vz, altitude,
                roll, pitch, yaw, p, q, r, dt
            )

        else:  # EMERGENCY
            # Emergency - max thrust up, wings level
            thrust_cmd = self.config.hover_thrust
            moment_cmd = MomentCommand(L=-roll*5000, M=-pitch*5000, N=0)

        # ===== Apply Limits =====
        thrust_cmd = np.clip(thrust_cmd, 0, self.config.max_thrust)
        moments = np.array([moment_cmd.L, moment_cmd.M, moment_cmd.N])
        moments = np.clip(moments, -self.config.max_moment, self.config.max_moment)

        # ===== Update Nacelle Schedule =====
        nacelle_mode = None
        if flight_mode == FlightMode.HOVER:
            nacelle_mode = NacelleFlightMode.HOVER
        elif flight_mode == FlightMode.CRUISE:
            nacelle_mode = NacelleFlightMode.CRUISE

        nacelle_angles = self.nacelle_scheduler.schedule(
            airspeed, altitude, dt, nacelle_mode
        )
        self.state.nacelle_angles = nacelle_angles

        # ===== Control Allocation =====
        rotor_cmd = self.control_mixer.allocate(
            thrust_cmd, moments, nacelle_angles
        )

        # ===== Update State =====
        self.state.thrust_cmd = thrust_cmd
        self.state.moment_cmd = moments

        if self._current_setpoint:
            self.state.position_error = np.array([
                self._current_setpoint.x - x,
                self._current_setpoint.y - y,
                self._current_setpoint.z - z,
            ])

        return rotor_cmd

    def update_simple(
        self,
        t: float,
        x: float, y: float, alt: float,
        vx: float, vy: float, vz: float,
        roll: float, pitch: float, yaw: float,
        p: float, q: float, r: float,
        dt: float,
    ) -> tuple[float, np.ndarray, np.ndarray]:
        """
        Simplified update interface.

        Returns:
            (thrust, moments, nacelle_angles)
        """
        rotor_cmd = self.update(
            t,
            np.array([x, y, -alt]),
            np.array([vx, vy, vz]),
            np.array([roll, pitch, yaw]),
            np.array([p, q, r]),
            dt,
        )

        return (
            np.sum(rotor_cmd.thrusts),
            self.state.moment_cmd,
            rotor_cmd.nacelle_angles,
        )

    @property
    def is_complete(self) -> bool:
        """Check if trajectory/path following is complete."""
        return self.state.trajectory_complete

    @property
    def current_mode(self) -> FlightMode:
        """Get current flight mode."""
        return self.state.flight_mode

    def get_telemetry(self) -> dict[str, Any]:
        """Get telemetry data for logging."""
        return {
            "flight_mode": self.state.flight_mode.name,
            "thrust_cmd": self.state.thrust_cmd,
            "moments": self.state.moment_cmd.tolist(),
            "nacelle_angles_deg": np.degrees(self.state.nacelle_angles).tolist(),
            "position_error": self.state.position_error.tolist(),
            "cross_track_error": self.state.cross_track_error,
            "trajectory_complete": self.state.trajectory_complete,
        }
