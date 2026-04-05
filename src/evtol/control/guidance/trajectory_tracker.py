"""
Trajectory Tracker - Guidance Layer.

Tracks time-parameterized trajectories from the planning layer.
Provides position/velocity/acceleration setpoints at each timestep.

This is the interface between the planning layer and control layer.
"""

import numpy as np
from dataclasses import dataclass
from enum import Enum


@dataclass
class TrajectoryPoint:
    """Single trajectory point with all derivatives."""
    t: float           # Time (s)

    # Position (NED frame)
    x: float           # North (m)
    y: float           # East (m)
    z: float           # Down (m) - negative for altitude

    # Velocity (m/s)
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0

    # Acceleration (m/s²) - for feedforward
    ax: float = 0.0
    ay: float = 0.0
    az: float = 0.0

    # Heading (rad)
    heading: float = 0.0
    heading_rate: float = 0.0

    # Flight phase hints
    nacelle_angle: float = np.radians(90)  # 90° = hover, 0° = cruise


class TrackingMode(Enum):
    """Tracking mode."""
    POSITION = "position"         # Track position (slow, accurate)
    VELOCITY = "velocity"         # Track velocity (faster)
    ACCELERATION = "acceleration" # Track acceleration (aggressive)


@dataclass
class TrajectoryTrackerConfig:
    """Configuration for trajectory tracker."""
    lookahead_time: float = 0.5      # Seconds ahead to look
    interpolation_order: int = 3     # Cubic interpolation
    position_tolerance: float = 5.0  # m
    velocity_tolerance: float = 2.0  # m/s
    max_cross_track_error: float = 20.0  # m


class TrajectoryTracker:
    """
    Tracks time-parameterized trajectories from planning layer.

    Input: Trajectory as list of TrajectoryPoints
    Output: Current setpoint (position, velocity, acceleration, heading)

    Features:
    - Interpolation between waypoints
    - Lookahead for smooth tracking
    - Cross-track error monitoring
    - Trajectory completion detection
    """

    def __init__(self, config: TrajectoryTrackerConfig | None = None):
        self.config = config or TrajectoryTrackerConfig()

        self._trajectory: list[TrajectoryPoint] = []
        self._start_time: float = 0.0
        self._trajectory_time: float = 0.0
        self._completed: bool = False

    def load_trajectory(
        self,
        trajectory: list[TrajectoryPoint],
        start_time: float = 0.0,
    ) -> None:
        """
        Load a trajectory from the planning layer.

        Args:
            trajectory: List of trajectory points, sorted by time
            start_time: Simulation time when trajectory starts
        """
        self._trajectory = trajectory
        self._start_time = start_time
        self._trajectory_time = 0.0
        self._completed = False

        if len(trajectory) > 0:
            # Make trajectory times relative
            t0 = trajectory[0].t
            for pt in self._trajectory:
                pt.t = pt.t - t0

    def load_from_planning_output(
        self,
        times: np.ndarray,
        positions: np.ndarray,
        velocities: np.ndarray | None = None,
        headings: np.ndarray | None = None,
        start_time: float = 0.0,
    ) -> None:
        """
        Load trajectory from planning layer arrays.

        Args:
            times: Time vector (N,)
            positions: Position array (N, 3) [x, y, z] or (N, 3) [x, y, alt]
            velocities: Velocity array (N, 3) [vx, vy, vz]
            headings: Heading array (N,)
            start_time: Simulation start time
        """
        trajectory = []
        for i, t in enumerate(times):
            pt = TrajectoryPoint(
                t=t,
                x=positions[i, 0],
                y=positions[i, 1],
                z=-positions[i, 2] if positions[i, 2] > 0 else positions[i, 2],  # Convert alt to NED
            )

            if velocities is not None:
                pt.vx = velocities[i, 0]
                pt.vy = velocities[i, 1]
                pt.vz = -velocities[i, 2] if len(velocities[i]) > 2 else 0.0

            if headings is not None:
                pt.heading = headings[i]

            trajectory.append(pt)

        # Compute velocities if not provided
        if velocities is None and len(trajectory) > 1:
            for i in range(len(trajectory) - 1):
                dt = trajectory[i + 1].t - trajectory[i].t
                if dt > 0:
                    trajectory[i].vx = (trajectory[i + 1].x - trajectory[i].x) / dt
                    trajectory[i].vy = (trajectory[i + 1].y - trajectory[i].y) / dt
                    trajectory[i].vz = (trajectory[i + 1].z - trajectory[i].z) / dt
            # Last point same as previous
            if len(trajectory) > 1:
                trajectory[-1].vx = trajectory[-2].vx
                trajectory[-1].vy = trajectory[-2].vy
                trajectory[-1].vz = trajectory[-2].vz

        self.load_trajectory(trajectory, start_time)

    def get_setpoint(
        self,
        current_time: float,
        current_pos: np.ndarray | None = None,
    ) -> tuple[TrajectoryPoint, float]:
        """
        Get current trajectory setpoint.

        Args:
            current_time: Current simulation time
            current_pos: Current position [x, y, z] for cross-track calculation

        Returns:
            (setpoint, cross_track_error): Current setpoint and cross-track error
        """
        if len(self._trajectory) == 0:
            # No trajectory - hold at origin
            return TrajectoryPoint(t=0, x=0, y=0, z=0), 0.0

        # Trajectory time (time since start)
        traj_t = current_time - self._start_time + self.config.lookahead_time

        # Find segment
        if traj_t <= 0:
            setpoint = self._trajectory[0]
        elif traj_t >= self._trajectory[-1].t:
            setpoint = self._trajectory[-1]
            self._completed = True
        else:
            # Interpolate
            setpoint = self._interpolate(traj_t)

        # Cross-track error
        cross_track = 0.0
        if current_pos is not None:
            dx = current_pos[0] - setpoint.x
            dy = current_pos[1] - setpoint.y
            dz = current_pos[2] - setpoint.z
            cross_track = np.sqrt(dx**2 + dy**2 + dz**2)

        return setpoint, cross_track

    def _interpolate(self, t: float) -> TrajectoryPoint:
        """Interpolate trajectory at time t."""
        # Find segment
        for i in range(len(self._trajectory) - 1):
            if self._trajectory[i].t <= t <= self._trajectory[i + 1].t:
                break
        else:
            return self._trajectory[-1]

        pt0 = self._trajectory[i]
        pt1 = self._trajectory[i + 1]

        # Linear interpolation factor
        dt = pt1.t - pt0.t
        if dt <= 0:
            return pt0

        alpha = (t - pt0.t) / dt

        # Interpolate
        return TrajectoryPoint(
            t=t,
            x=pt0.x + alpha * (pt1.x - pt0.x),
            y=pt0.y + alpha * (pt1.y - pt0.y),
            z=pt0.z + alpha * (pt1.z - pt0.z),
            vx=pt0.vx + alpha * (pt1.vx - pt0.vx),
            vy=pt0.vy + alpha * (pt1.vy - pt0.vy),
            vz=pt0.vz + alpha * (pt1.vz - pt0.vz),
            heading=self._interp_angle(pt0.heading, pt1.heading, alpha),
            heading_rate=pt0.heading_rate + alpha * (pt1.heading_rate - pt0.heading_rate),
            nacelle_angle=pt0.nacelle_angle + alpha * (pt1.nacelle_angle - pt0.nacelle_angle),
        )

    @staticmethod
    def _interp_angle(a0: float, a1: float, alpha: float) -> float:
        """Interpolate angles handling wrap-around."""
        diff = a1 - a0
        while diff > np.pi:
            diff -= 2 * np.pi
        while diff < -np.pi:
            diff += 2 * np.pi
        return a0 + alpha * diff

    @property
    def is_completed(self) -> bool:
        """Check if trajectory is complete."""
        return self._completed

    @property
    def trajectory_length(self) -> int:
        """Number of waypoints in trajectory."""
        return len(self._trajectory)

    def get_total_duration(self) -> float:
        """Get total trajectory duration."""
        if len(self._trajectory) < 2:
            return 0.0
        return self._trajectory[-1].t - self._trajectory[0].t
