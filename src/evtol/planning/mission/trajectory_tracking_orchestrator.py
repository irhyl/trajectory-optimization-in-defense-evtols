"""
Trajectory Tracking Orchestrator

Integrates mission loading, trajectory tracking, and performance monitoring.
This module provides the high-level interface for waypoint-following missions
in closed-loop SITL simulation.

Workflow:
1. Load mission from JSON
2. Generate trajectory from waypoints
3. Load trajectory into tracker
4. Run closed-loop simulation with tracking feedback
5. Monitor cross-track error, heading error, altitude error
6. Log telemetry for post-mission analysis

Author: Defense eVTOL Research Team
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
import logging
from typing import Callable

from evtol.planning.mission.mission_loader import (
    MissionLoader,
    Waypoint,
    MissionConfig,
)
from evtol.control.guidance.trajectory_tracker import (
    TrajectoryTracker,
    TrajectoryTrackerConfig,
    TrajectoryPoint,
)

logger = logging.getLogger(__name__)


class TrackingPhase(Enum):
    """Current tracking phase."""
    TAKEOFF = "takeoff"
    TRANSITION = "transition"
    CRUISE = "cruise"
    DESCENT = "descent"
    LANDING = "landing"


@dataclass
class TrackingMetrics:
    """Real-time tracking metrics."""
    time: float
    phase: TrackingPhase
    position_error: float  # m (3D distance from target)
    cross_track_error: float  # m (perpendicular to path)
    altitude_error: float  # m
    heading_error: float  # rad
    velocity_error: float  # m/s
    
    # Current state
    current_pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    target_pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    current_vel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    target_vel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    current_heading: float = 0.0
    target_heading: float = 0.0
    
    # Performance aggregates
    max_position_error: float = 0.0
    mean_position_error: float = 0.0
    rms_cross_track_error: float = 0.0


@dataclass
class TrajectoryTrackingConfig:
    """Configuration for trajectory tracking orchestrator."""
    mission_config: MissionConfig = field(default_factory=MissionConfig)
    tracker_config: TrajectoryTrackerConfig = field(
        default_factory=TrajectoryTrackerConfig
    )
    dt: float = 0.02  # timestep in seconds (50 Hz)
    accumulate_metrics: bool = True


class TrajectoryTrackingOrchestrator:
    """High-level orchestrator for trajectory tracking missions."""

    def __init__(self, config: TrajectoryTrackingConfig | None = None):
        self.config = config or TrajectoryTrackingConfig()
        self.mission_loader = MissionLoader(self.config.mission_config)
        self.tracker = TrajectoryTracker(self.config.tracker_config)
        
        # Mission state
        self._trajectory: list[TrajectoryPoint] = []
        self._is_tracking = False
        self._current_phase = TrackingPhase.TAKEOFF
        
        # Metrics
        self._metrics_history: list[TrackingMetrics] = []
        self._position_errors: list[float] = []
        self._cross_track_errors: list[float] = []
        self._current_metrics: TrackingMetrics | None = None
        
        # Callbacks
        self._on_phase_change: Callable[[TrackingPhase], None] | None = None
        self._on_waypoint_reached: Callable[[int, Waypoint], None] | None = None

    def load_mission_from_file(self, filepath: Path | str) -> list[Waypoint]:
        """Load mission waypoints from JSON file."""
        waypoints = self.mission_loader.load_mission_json(filepath)
        self._trajectory = (
            self.mission_loader.generate_trajectory_from_waypoints(waypoints)
        )
        self.tracker.load_trajectory(self._trajectory, start_time=0.0)
        logger.info(f"Loaded mission from {filepath} with {len(self._trajectory)} trajectory points")
        return waypoints

    def load_mission_from_waypoints(self, waypoints: list[Waypoint]) -> None:
        """Load mission from waypoint list."""
        self._trajectory = (
            self.mission_loader.generate_trajectory_from_waypoints(waypoints)
        )
        self.tracker.load_trajectory(self._trajectory, start_time=0.0)
        logger.info(f"Loaded mission with {len(waypoints)} waypoints and {len(self._trajectory)} trajectory points")

    def start_tracking(self) -> None:
        """Start trajectory tracking."""
        self._is_tracking = True
        self._metrics_history.clear()
        logger.info("Trajectory tracking started")

    def stop_tracking(self) -> None:
        """Stop trajectory tracking."""
        self._is_tracking = False
        logger.info(f"Trajectory tracking stopped. Logged {len(self._metrics_history)} metrics")

    def update(
        self,
        current_time: float,
        current_pos_ned: np.ndarray,  # [north, east, down]
        current_vel_ned: np.ndarray,  # [vn, ve, vd]
        current_heading: float,  # rad
    ) -> tuple[TrajectoryPoint, TrackingMetrics]:
        """
        Update tracking state and get current setpoint.

        Args:
            current_time: Current simulation time (seconds)
            current_pos_ned: Current position in NED frame [north, east, down]
            current_vel_ned: Current velocity in NED frame [vn, ve, vd]
            current_heading: Current heading (radians)

        Returns:
            (setpoint, metrics) - Next setpoint to track and current metrics
        """

        if not self._is_tracking:
            raise RuntimeError("Tracking not started - call start_tracking() first")

        # Get current setpoint from trajectory tracker
        setpoint = self.tracker.get_current_setpoint(current_time)

        # Calculate tracking errors
        target_pos = np.array([setpoint.x, setpoint.y, setpoint.z])
        target_vel = np.array([setpoint.vx, setpoint.vy, setpoint.vz])

        # 3D position error
        pos_error_vec = current_pos_ned - target_pos
        position_error = np.linalg.norm(pos_error_vec)

        # Cross-track error (perpendicular distance from path)
        cross_track_error = self._calculate_cross_track_error(
            current_pos_ned, target_pos, target_vel
        )

        # Altitude error (z component)
        altitude_error = current_pos_ned[2] - target_pos[2]

        # Heading error (shortest angular distance)
        heading_error = self._wrap_angle(current_heading - setpoint.heading)

        # Velocity error
        vel_error_vec = current_vel_ned - target_vel
        velocity_error = np.linalg.norm(vel_error_vec)

        # Update phase based on trajectory progress
        self._update_phase(current_time, setpoint)

        # Create metrics object
        metrics = TrackingMetrics(
            time=current_time,
            phase=self._current_phase,
            position_error=position_error,
            cross_track_error=cross_track_error,
            altitude_error=altitude_error,
            heading_error=heading_error,
            velocity_error=velocity_error,
            current_pos=current_pos_ned.copy(),
            target_pos=target_pos.copy(),
            current_vel=current_vel_ned.copy(),
            target_vel=target_vel.copy(),
            current_heading=current_heading,
            target_heading=setpoint.heading,
        )

        # Accumulate metrics for post-flight analysis
        if self.config.accumulate_metrics:
            self._metrics_history.append(metrics)
            self._position_errors.append(position_error)
            self._cross_track_errors.append(cross_track_error)

            # Update aggregates
            if self._position_errors:
                metrics.max_position_error = max(self._position_errors)
                metrics.mean_position_error = np.mean(self._position_errors)
            if self._cross_track_errors:
                metrics.rms_cross_track_error = np.sqrt(
                    np.mean(np.array(self._cross_track_errors) ** 2)
                )

        self._current_metrics = metrics
        return setpoint, metrics

    def get_metrics_summary(self) -> dict:
        """Get summary of tracking metrics from current mission."""
        if not self._metrics_history:
            return {}

        position_errors = np.array(self._position_errors)
        cross_track_errors = np.array(self._cross_track_errors)
        times = np.array([m.time for m in self._metrics_history])

        return {
            "duration": times[-1] - times[0] if len(times) > 1 else 0,
            "num_samples": len(self._metrics_history),
            "position_error": {
                "max": float(np.max(position_errors)),
                "mean": float(np.mean(position_errors)),
                "std": float(np.std(position_errors)),
                "p95": float(np.percentile(position_errors, 95)),
            },
            "cross_track_error": {
                "max": float(np.max(cross_track_errors)),
                "mean": float(np.mean(cross_track_errors)),
                "rms": float(np.sqrt(np.mean(cross_track_errors ** 2))),
                "p95": float(np.percentile(cross_track_errors, 95)),
            },
            "trajectory_completion": self.tracker.is_completed(),
        }

    def get_metrics_history(self) -> list[TrackingMetrics]:
        """Get full history of tracking metrics."""
        return self._metrics_history.copy()

    def set_phase_change_callback(
        self, callback: Callable[[TrackingPhase], None]
    ) -> None:
        """Register callback for phase changes."""
        self._on_phase_change = callback

    def set_waypoint_reached_callback(
        self, callback: Callable[[int, Waypoint], None]
    ) -> None:
        """Register callback for waypoint reaches."""
        self._on_waypoint_reached = callback

    # Private methods

    def _calculate_cross_track_error(
        self, pos: np.ndarray, target: np.ndarray, path_velocity: np.ndarray
    ) -> float:
        """Calculate perpendicular distance from current position to path."""

        # Vector from target to current position
        error_vec = pos - target

        # Normalize path velocity to get path direction
        path_dir = path_velocity / (np.linalg.norm(path_velocity) + 1e-6)

        # Project error onto cross-track direction (perpendicular to path)
        along_track = np.dot(error_vec, path_dir)
        cross_track_vec = error_vec - along_track * path_dir

        return np.linalg.norm(cross_track_vec)

    def _update_phase(self, current_time: float, setpoint: TrajectoryPoint) -> None:
        """Update current tracking phase based on nacelle angle and time."""

        # Determine phase from nacelle angle (rough heuristic)
        nacelle_deg = np.degrees(setpoint.nacelle_angle)

        if nacelle_deg > 85:  # Hover
            new_phase = TrackingPhase.TAKEOFF
            if current_time > 10:  # After initial takeoff
                new_phase = TrackingPhase.CRUISE
        elif nacelle_deg > 45:
            new_phase = TrackingPhase.TRANSITION
        else:
            new_phase = TrackingPhase.CRUISE

        # Trigger callback on phase change
        if new_phase != self._current_phase and self._on_phase_change:
            self._on_phase_change(new_phase)

        self._current_phase = new_phase

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        """Wrap angle to [-π, π]."""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle

    def is_completed(self) -> bool:
        """Check if trajectory tracking is complete."""
        return self.tracker.is_completed()

    def get_progress(self) -> float:
        """Get progress through the trajectory as a fraction [0, 1]."""
        if not self._trajectory:
            return 0.0

        total_time = self._trajectory[-1].t - self._trajectory[0].t
        if total_time == 0:
            return 0.0

        elapsed_time = (
            self.tracker._trajectory_time
        )  # Access internal time (for demo)
        return min(1.0, max(0.0, elapsed_time / total_time))
