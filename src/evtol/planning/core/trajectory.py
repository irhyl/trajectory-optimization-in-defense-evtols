"""
Trajectory Representations

This module defines trajectory representations for path planning and optimization.
Trajectories are parameterized curves through the state space with associated
time profiles.

Mathematical Background
=======================

A trajectory is a mapping τ: [0, T] → X from time to state space.

We support multiple parameterizations:

1. **Waypoint-based**: Piecewise linear between waypoints
   τ(t) = xᵢ + (t - tᵢ)/(tᵢ₊₁ - tᵢ) · (xᵢ₊₁ - xᵢ)  for t ∈ [tᵢ, tᵢ₊₁]

2. **Polynomial**: Time-polynomial for each coordinate
   τⱼ(t) = Σₖ aⱼₖ tᵏ  (j-th coordinate)

3. **B-spline**: Smooth spline parameterization
   τ(t) = Σᵢ cᵢ Bᵢ,ₖ(t)  where Bᵢ,ₖ are B-spline basis functions

B-splines are preferred for optimization because:
- Smooth (Cᵏ⁻² continuous)
- Local support (sparse Jacobians)
- Convex hull property (constraint satisfaction)
- Efficient evaluation and differentiation

Author: Defense eVTOL Research Team
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from collections.abc import Iterator
from enum import Enum
from scipy.interpolate import splprep, splev
import logging

from .state import State, Pose, Velocity, CoordinateFrame

logger = logging.getLogger(__name__)


class TrajectoryType(Enum):
    """Trajectory parameterization type."""
    WAYPOINT = "waypoint"      # Piecewise linear
    POLYNOMIAL = "polynomial"  # Time polynomials
    BSPLINE = "bspline"        # B-spline curves


@dataclass
class TrajectorySegment:
    """
    A segment of a trajectory between two states.

    Segments are the atomic units of trajectories. Each segment has:
    - Start and end states
    - Duration
    - Optional velocity profile

    Attributes:
        start: Starting state
        end: Ending state
        duration: Segment duration in seconds
        segment_type: Type of segment ("cruise", "climb", "descent", "turn")
    """
    start: State
    end: State
    duration: float
    segment_type: str = "cruise"

    # Cached properties
    _distance: float | None = field(default=None, repr=False)

    def __post_init__(self):
        """Validate segment."""
        if self.duration <= 0:
            raise ValueError(f"Duration must be positive, got {self.duration}")

    @property
    def distance(self) -> float:
        """Distance of segment in meters."""
        if self._distance is None:
            self._distance = self.start.distance_to(self.end)
        return self._distance

    @property
    def average_speed(self) -> float:
        """Average speed over segment (m/s)."""
        return self.distance / self.duration if self.duration > 0 else 0.0

    @property
    def altitude_change(self) -> float:
        """Altitude change (end - start) in meters."""
        return self.end.pose.position[2] - self.start.pose.position[2]

    @property
    def energy_consumed(self) -> float:
        """Energy consumed (start - end) as fraction of capacity."""
        return self.start.energy - self.end.energy

    def evaluate(self, t: float) -> State:
        """
        Evaluate segment at time t (linear interpolation).

        Args:
            t: Time in [0, duration]

        Returns:
            Interpolated state
        """
        if t < 0 or t > self.duration:
            raise ValueError(f"t={t} outside segment duration [0, {self.duration}]")

        alpha = t / self.duration if self.duration > 0 else 0.0
        return self.start.interpolate(self.end, alpha)

    def is_feasible(self, constraints: dict) -> tuple[bool, list[str]]:
        """
        Check if segment satisfies kinodynamic constraints.

        Args:
            constraints: Dict with max_speed, max_climb_rate, etc.

        Returns:
            (is_feasible, list of violations)
        """
        violations = []

        # Speed check
        max_speed = constraints.get("max_speed", 60.0)
        if self.average_speed > max_speed:
            violations.append(f"Speed {self.average_speed:.1f} > max {max_speed}")

        # Climb rate check
        max_climb = constraints.get("max_climb_rate", 8.0)
        climb_rate = abs(self.altitude_change) / self.duration if self.duration > 0 else 0
        if climb_rate > max_climb:
            violations.append(f"Climb rate {climb_rate:.1f} > max {max_climb}")

        # Minimum altitude check
        min_alt = constraints.get("min_altitude", 50.0)
        if self.start.pose.position[2] < min_alt or self.end.pose.position[2] < min_alt:
            violations.append(f"Altitude below minimum {min_alt}m")

        return len(violations) == 0, violations


@dataclass
class Trajectory:
    """
    Complete trajectory through state space.

    A trajectory is an ordered sequence of states with timing information.
    It supports multiple parameterizations for evaluation and optimization.

    Attributes:
        states: List of trajectory states
        trajectory_type: Parameterization type
        metadata: Optional trajectory metadata (algorithm, cost, etc.)

    Properties:
        duration: Total trajectory duration
        distance: Total trajectory distance
        energy_consumed: Total energy consumption

    Example:
        >>> states = [State(...) for _ in range(10)]
        >>> traj = Trajectory(states)
        >>> print(f"Duration: {traj.duration}s, Distance: {traj.distance/1000:.1f}km")
    """
    states: list[State]
    trajectory_type: TrajectoryType = TrajectoryType.WAYPOINT
    metadata: dict = field(default_factory=dict)

    # Spline representation (computed on demand)
    _spline: tuple | None = field(default=None, repr=False)
    _spline_params: dict | None = field(default=None, repr=False)

    def __post_init__(self):
        """Validate trajectory."""
        if len(self.states) < 2:
            raise ValueError("Trajectory must have at least 2 states")

        # Ensure times are monotonically increasing
        times = [s.time for s in self.states]
        if not all(times[i] <= times[i+1] for i in range(len(times)-1)):
            raise ValueError("State times must be monotonically increasing")

    def __len__(self) -> int:
        """Number of states in trajectory."""
        return len(self.states)

    def __iter__(self) -> Iterator[State]:
        """Iterate over states."""
        return iter(self.states)

    def __getitem__(self, idx: int) -> State:
        """Get state by index."""
        return self.states[idx]

    @property
    def start_state(self) -> State:
        """First state of trajectory."""
        return self.states[0]

    @property
    def end_state(self) -> State:
        """Last state of trajectory."""
        return self.states[-1]

    @property
    def duration(self) -> float:
        """Total trajectory duration in seconds."""
        return self.states[-1].time - self.states[0].time

    @property
    def distance(self) -> float:
        """Total trajectory distance in meters."""
        total = 0.0
        for i in range(len(self.states) - 1):
            total += self.states[i].distance_to(self.states[i+1])
        return total

    @property
    def energy_consumed(self) -> float:
        """Total energy consumed as fraction of battery capacity."""
        return self.states[0].energy - self.states[-1].energy

    @property
    def average_speed(self) -> float:
        """Average speed over trajectory (m/s)."""
        return self.distance / self.duration if self.duration > 0 else 0.0

    @property
    def segments(self) -> list[TrajectorySegment]:
        """Generate trajectory segments."""
        segs = []
        for i in range(len(self.states) - 1):
            s0, s1 = self.states[i], self.states[i+1]
            duration = s1.time - s0.time

            # Ensure positive duration (fallback based on distance if time not set)
            if duration <= 0:
                dist = s0.distance_to(s1)
                duration = max(dist / 40.0, 0.1)  # Assume 40 m/s cruise, min 0.1s

            # Determine segment type based on altitude change
            alt_change = s1.pose.position[2] - s0.pose.position[2]
            if duration > 0:
                climb_rate = alt_change / duration
                if climb_rate > 1.0:
                    seg_type = "climb"
                elif climb_rate < -1.0:
                    seg_type = "descent"
                else:
                    seg_type = "cruise"
            else:
                seg_type = "cruise"

            segs.append(TrajectorySegment(s0, s1, duration, seg_type))

        return segs

    def evaluate(self, t: float) -> State:
        """
        Evaluate trajectory at time t.

        For waypoint trajectories: linear interpolation
        For spline trajectories: spline evaluation

        Args:
            t: Time in [t_start, t_end]

        Returns:
            State at time t
        """
        t0 = self.states[0].time
        t_end = self.states[-1].time

        if t < t0 or t > t_end:
            raise ValueError(f"t={t} outside trajectory [{t0}, {t_end}]")

        if self.trajectory_type == TrajectoryType.WAYPOINT:
            return self._evaluate_waypoint(t)
        elif self.trajectory_type == TrajectoryType.BSPLINE:
            return self._evaluate_bspline(t)
        else:
            raise NotImplementedError(f"Evaluation for {self.trajectory_type}")

    def _evaluate_waypoint(self, t: float) -> State:
        """Evaluate waypoint trajectory with linear interpolation."""
        # Find enclosing segment
        for i in range(len(self.states) - 1):
            if self.states[i].time <= t <= self.states[i+1].time:
                dt = self.states[i+1].time - self.states[i].time
                alpha = (t - self.states[i].time) / dt if dt > 0 else 0.0
                return self.states[i].interpolate(self.states[i+1], alpha)

        # Edge case: exactly at end
        return self.states[-1]

    def _evaluate_bspline(self, t: float) -> State:
        """Evaluate B-spline trajectory."""
        if self._spline is None:
            self._fit_bspline()

        # Normalize time to [0, 1] for spline parameter
        t_norm = (t - self.states[0].time) / self.duration

        # Evaluate spline for position
        pos = np.array(splev(t_norm, self._spline))

        # Interpolate other state variables
        idx = int(t_norm * (len(self.states) - 1))
        idx = max(0, min(idx, len(self.states) - 2))

        alpha = (t_norm * (len(self.states) - 1)) - idx
        base_state = self.states[idx].interpolate(self.states[idx + 1], alpha)

        # Create state with spline position
        return State(
            pose=Pose(
                pos[:3] if pos.shape[0] >= 3 else np.array([pos[0], pos[1], base_state.pose.position[2]]),
                base_state.pose.attitude,
                frame=base_state.pose.frame,
            ),
            velocity=base_state.velocity,
            energy=base_state.energy,
            time=t,
        )

    def _fit_bspline(self, degree: int = 3, smoothing: float = 0.0) -> None:
        """
        Fit B-spline to trajectory waypoints.

        Args:
            degree: Spline degree (3 = cubic)
            smoothing: Smoothing factor (0 = interpolating)
        """
        # Extract position coordinates
        positions = np.array([s.pose.position for s in self.states])

        # Fit B-spline
        try:
            tck, u = splprep(
                [positions[:, 0], positions[:, 1], positions[:, 2]],
                k=min(degree, len(self.states) - 1),
                s=smoothing,
            )
            self._spline = tck
            self._spline_params = {"degree": degree, "smoothing": smoothing}
        except Exception as e:
            logger.warning(f"B-spline fitting failed: {e}, using waypoints")
            self._spline = None

    def to_bspline(self, degree: int = 3, smoothing: float = 0.0) -> Trajectory:
        """
        Convert to B-spline trajectory.

        Args:
            degree: Spline degree
            smoothing: Smoothing factor

        Returns:
            New trajectory with B-spline type
        """
        new_traj = Trajectory(
            states=self.states.copy(),
            trajectory_type=TrajectoryType.BSPLINE,
            metadata=self.metadata.copy(),
        )
        new_traj._fit_bspline(degree, smoothing)
        return new_traj

    def resample(self, n_points: int) -> Trajectory:
        """
        Resample trajectory to fixed number of points.

        Args:
            n_points: Number of output points

        Returns:
            Resampled trajectory
        """
        times = np.linspace(self.states[0].time, self.states[-1].time, n_points)
        new_states = [self.evaluate(t) for t in times]

        return Trajectory(
            states=new_states,
            trajectory_type=self.trajectory_type,
            metadata=self.metadata.copy(),
        )

    def smooth(self, window_size: int = 5) -> Trajectory:
        """
        Apply moving average smoothing to trajectory.

        Args:
            window_size: Smoothing window size

        Returns:
            Smoothed trajectory
        """
        if window_size < 2 or window_size > len(self.states):
            return self

        positions = np.array([s.pose.position for s in self.states])

        # Simple moving average
        kernel = np.ones(window_size) / window_size
        smoothed = np.zeros_like(positions)

        for i in range(3):
            smoothed[:, i] = np.convolve(positions[:, i], kernel, mode='same')

        # Keep endpoints fixed
        smoothed[0] = positions[0]
        smoothed[-1] = positions[-1]

        # Create new states
        new_states = []
        for i, state in enumerate(self.states):
            new_states.append(State(
                pose=Pose(smoothed[i], state.pose.attitude, frame=state.pose.frame),
                velocity=state.velocity,
                energy=state.energy,
                time=state.time,
            ))

        return Trajectory(
            states=new_states,
            trajectory_type=self.trajectory_type,
            metadata={**self.metadata, "smoothed": True, "window_size": window_size},
        )

    def compute_velocities(self) -> None:
        """
        Compute velocities from position differences.

        Updates velocity field of each state based on finite differences.
        """
        for i in range(len(self.states)):
            if i == 0:
                # Forward difference
                dt = self.states[1].time - self.states[0].time
                if dt > 0:
                    dp = self.states[1].pose.position - self.states[0].pose.position
                    vel = dp / dt
                else:
                    vel = np.zeros(3)
            elif i == len(self.states) - 1:
                # Backward difference
                dt = self.states[-1].time - self.states[-2].time
                if dt > 0:
                    dp = self.states[-1].pose.position - self.states[-2].pose.position
                    vel = dp / dt
                else:
                    vel = np.zeros(3)
            else:
                # Central difference
                dt = self.states[i+1].time - self.states[i-1].time
                if dt > 0:
                    dp = self.states[i+1].pose.position - self.states[i-1].pose.position
                    vel = dp / dt
                else:
                    vel = np.zeros(3)

            self.states[i].velocity = Velocity(linear=vel)

    def is_feasible(self, constraints: dict) -> tuple[bool, list[str]]:
        """
        Check if entire trajectory satisfies constraints.

        Args:
            constraints: Constraint dictionary

        Returns:
            (is_feasible, list of all violations)
        """
        all_violations = []

        for i, segment in enumerate(self.segments):
            feasible, violations = segment.is_feasible(constraints)
            for v in violations:
                all_violations.append(f"Segment {i}: {v}")

        return len(all_violations) == 0, all_violations

    def get_positions(self) -> np.ndarray:
        """Get all positions as numpy array (N x 3)."""
        return np.array([s.pose.position for s in self.states])

    def get_times(self) -> np.ndarray:
        """Get all times as numpy array."""
        return np.array([s.time for s in self.states])

    def __repr__(self) -> str:
        return (
            f"Trajectory(n_states={len(self.states)}, duration={self.duration:.1f}s, "
            f"distance={self.distance/1000:.2f}km, type={self.trajectory_type.value})"
        )
