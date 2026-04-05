"""
Trajectory Smoothing Algorithms

This module implements various trajectory smoothing methods to reduce
jerkiness while maintaining path safety and feasibility.

Methods:
--------
1. B-Spline Smoothing
   - Fits B-splines to waypoints
   - Guarantees C² continuity (position, velocity, acceleration)
   - Control over smoothness vs. fit

2. Moving Average
   - Simple local averaging
   - Fast but can introduce lag

3. Savitzky-Golay Filter
   - Polynomial fitting in sliding window
   - Preserves local features better

Mathematical Foundation:
------------------------
B-splines are piecewise polynomials defined by:
    C(t) = Σᵢ Nᵢ,ₖ(t) Pᵢ

where Nᵢ,ₖ are basis functions of degree k and Pᵢ are control points.

Properties:
- Local support: each control point affects limited region
- Partition of unity: basis functions sum to 1
- Convex hull: curve lies within control point convex hull
- Variation diminishing: fewer oscillations than interpolating splines
"""

from __future__ import annotations

import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from scipy import interpolate
from scipy.signal import savgol_filter
import logging

from ..core.trajectory import Trajectory, TrajectorySegment
from ..core.state import State, Pose, Velocity

logger = logging.getLogger(__name__)


class TrajectorySmoothing(ABC):
    """
    Abstract base class for trajectory smoothing algorithms.

    Attributes:
        name: Algorithm identifier
        preserves_endpoints: Whether start/end are preserved exactly
    """

    def __init__(self, name: str, preserves_endpoints: bool = True):
        self.name = name
        self.preserves_endpoints = preserves_endpoints

    @abstractmethod
    def smooth(
        self,
        trajectory: Trajectory,
        **kwargs,
    ) -> Trajectory:
        """
        Smooth a trajectory.

        Args:
            trajectory: Input trajectory
            **kwargs: Algorithm-specific parameters

        Returns:
            Smoothed trajectory
        """
        pass

    def smooth_positions(
        self,
        positions: np.ndarray,
        times: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Smooth just the positions.

        Args:
            positions: Nx3 position array
            times: Optional time array

        Returns:
            Smoothed positions
        """
        # Default: create dummy trajectory and smooth
        states = []
        for i, pos in enumerate(positions):
            t = times[i] if times is not None else float(i)
            state = State(
                pose=Pose(
                    position=pos,
                    attitude=np.array([0.0, 0.0, 0.0]),
                ),
                velocity=Velocity(),
                time=t,
            )
            states.append(state)

        segments = []
        for i in range(len(states) - 1):
            seg = TrajectorySegment(
                start=states[i],
                end=states[i + 1],
                duration=states[i + 1].time - states[i].time,
            )
            segments.append(seg)

        traj = Trajectory(states=states)
        smoothed = self.smooth(traj)

        return smoothed.get_positions()


class BSplineSmoother(TrajectorySmoothing):
    """
    B-Spline based trajectory smoothing.

    Fits a B-spline curve through waypoints with controllable smoothness.
    The smoothness factor s controls the trade-off between fitting error
    and smoothness.

    Attributes:
        degree: Spline degree (3 = cubic, default)
        smoothness: Smoothing factor (0 = interpolation, larger = smoother)
        n_samples: Number of output samples
    """

    def __init__(
        self,
        degree: int = 3,
        smoothness: float = 0.0,
        n_samples: int | None = None,
    ):
        """
        Initialize B-spline smoother.

        Args:
            degree: Spline polynomial degree (1-5)
            smoothness: Smoothing factor (0 = exact interpolation)
            n_samples: Output sample count (None = same as input)
        """
        super().__init__("bspline", preserves_endpoints=True)
        self.degree = min(max(1, degree), 5)
        self.smoothness = max(0.0, smoothness)
        self.n_samples = n_samples

    def smooth(
        self,
        trajectory: Trajectory,
        **kwargs,
    ) -> Trajectory:
        """
        Smooth trajectory using B-splines.

        Args:
            trajectory: Input trajectory
            **kwargs: Override parameters (smoothness, n_samples)

        Returns:
            Smoothed trajectory
        """
        smoothness = kwargs.get('smoothness', self.smoothness)
        n_samples = kwargs.get('n_samples', self.n_samples)

        positions = trajectory.get_positions()
        times = trajectory.get_times()

        if len(positions) < self.degree + 1:
            logger.warning(f"Not enough points for degree {self.degree} spline")
            return trajectory

        # Parameterize by arc length (more uniform than time)
        arc_lengths = np.zeros(len(positions))
        for i in range(1, len(positions)):
            arc_lengths[i] = arc_lengths[i-1] + np.linalg.norm(positions[i] - positions[i-1])

        if arc_lengths[-1] < 1e-10:
            return trajectory

        arc_lengths /= arc_lengths[-1]  # Normalize to [0, 1]

        # Fit B-spline to each coordinate
        try:
            # Use splprep for parametric spline
            tck, u = interpolate.splprep(
                [positions[:, 0], positions[:, 1], positions[:, 2]],
                u=arc_lengths,
                k=self.degree,
                s=smoothness,
            )

            # Evaluate at new parameter values
            n_out = n_samples if n_samples else len(positions)
            u_new = np.linspace(0, 1, n_out)
            smooth_coords = interpolate.splev(u_new, tck)
            smooth_positions = np.column_stack(smooth_coords)

            # Interpolate times
            time_interp = interpolate.interp1d(
                arc_lengths, times, kind='linear', fill_value='extrapolate'
            )
            smooth_times = time_interp(u_new)

        except Exception as e:
            logger.warning(f"B-spline fitting failed: {e}, using fallback")
            return self._fallback_smooth(trajectory, n_samples)

        # Ensure endpoints are preserved
        if self.preserves_endpoints:
            smooth_positions[0] = positions[0]
            smooth_positions[-1] = positions[-1]
            smooth_times[0] = times[0]
            smooth_times[-1] = times[-1]

        # Compute velocities from finite differences
        velocities = self._compute_velocities(smooth_positions, smooth_times)

        # Build new trajectory
        return self._build_trajectory(smooth_positions, velocities, smooth_times)

    def _fallback_smooth(
        self,
        trajectory: Trajectory,
        n_samples: int | None,
    ) -> Trajectory:
        """Fallback to simple interpolation if spline fails."""
        positions = trajectory.get_positions()
        times = trajectory.get_times()

        n_out = n_samples if n_samples else len(positions)

        # Linear interpolation
        t_new = np.linspace(times[0], times[-1], n_out)

        smooth_positions = np.zeros((n_out, 3))
        for dim in range(3):
            interp = interpolate.interp1d(times, positions[:, dim], kind='linear')
            smooth_positions[:, dim] = interp(t_new)

        velocities = self._compute_velocities(smooth_positions, t_new)
        return self._build_trajectory(smooth_positions, velocities, t_new)

    def _compute_velocities(
        self,
        positions: np.ndarray,
        times: np.ndarray,
    ) -> np.ndarray:
        """Compute velocities using central differences."""
        n = len(positions)
        velocities = np.zeros_like(positions)

        # Forward difference for first point
        if n > 1:
            dt = times[1] - times[0]
            if dt > 1e-10:
                velocities[0] = (positions[1] - positions[0]) / dt

        # Central differences for interior
        for i in range(1, n - 1):
            dt = times[i + 1] - times[i - 1]
            if dt > 1e-10:
                velocities[i] = (positions[i + 1] - positions[i - 1]) / dt

        # Backward difference for last point
        if n > 1:
            dt = times[-1] - times[-2]
            if dt > 1e-10:
                velocities[-1] = (positions[-1] - positions[-2]) / dt

        return velocities

    def _build_trajectory(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        times: np.ndarray,
    ) -> Trajectory:
        """Build trajectory from smoothed data."""
        states = []
        for i in range(len(positions)):
            state = State(
                pose=Pose(
                    position=positions[i].copy(),
                    attitude=np.array([0.0, 0.0, 0.0]),  # Level flight
                ),
                velocity=Velocity(linear=velocities[i].copy()),
                time=float(times[i]),
            )
            states.append(state)

        segments = []
        for i in range(len(states) - 1):
            seg = TrajectorySegment(
                start=states[i],
                end=states[i + 1],
                duration=states[i + 1].time - states[i].time,
            )
            segments.append(seg)

        return Trajectory(states=states)

    def get_spline_representation(
        self,
        trajectory: Trajectory,
    ) -> tuple[tuple, np.ndarray]:
        """
        Get B-spline representation (tck, u) for a trajectory.

        Returns:
            (tck, u): Spline parameters from splprep
        """
        positions = trajectory.get_positions()
        arc_lengths = np.zeros(len(positions))
        for i in range(1, len(positions)):
            arc_lengths[i] = arc_lengths[i-1] + np.linalg.norm(positions[i] - positions[i-1])
        arc_lengths /= arc_lengths[-1] if arc_lengths[-1] > 0 else 1.0

        tck, u = interpolate.splprep(
            [positions[:, 0], positions[:, 1], positions[:, 2]],
            u=arc_lengths,
            k=self.degree,
            s=self.smoothness,
        )

        return tck, u


class MovingAverageSmoother(TrajectorySmoothing):
    """
    Simple moving average smoothing.

    Averages positions within a sliding window. Fast but can
    cause lag and corner rounding.

    Attributes:
        window_size: Number of points to average (must be odd)
    """

    def __init__(self, window_size: int = 5):
        """
        Initialize moving average smoother.

        Args:
            window_size: Window size (will be made odd if even)
        """
        super().__init__("moving_average", preserves_endpoints=True)
        self.window_size = window_size if window_size % 2 == 1 else window_size + 1

    def smooth(
        self,
        trajectory: Trajectory,
        **kwargs,
    ) -> Trajectory:
        """
        Smooth trajectory using moving average.

        Args:
            trajectory: Input trajectory
            **kwargs: Override window_size

        Returns:
            Smoothed trajectory
        """
        window = kwargs.get('window_size', self.window_size)
        half_window = window // 2

        positions = trajectory.get_positions()
        times = trajectory.get_times()
        n = len(positions)

        if n <= window:
            return trajectory

        smooth_positions = positions.copy()

        # Apply moving average to interior points
        for i in range(half_window, n - half_window):
            start = i - half_window
            end = i + half_window + 1
            smooth_positions[i] = np.mean(positions[start:end], axis=0)

        # Compute velocities
        velocities = np.zeros_like(smooth_positions)
        for i in range(n - 1):
            dt = times[i + 1] - times[i]
            if dt > 1e-10:
                velocities[i] = (smooth_positions[i + 1] - smooth_positions[i]) / dt
        velocities[-1] = velocities[-2] if n > 1 else np.zeros(3)

        # Build trajectory
        states = []
        for i in range(n):
            state = State(
                pose=Pose(
                    position=smooth_positions[i],
                    attitude=np.array([0.0, 0.0, 0.0]),
                ),
                velocity=Velocity(linear=velocities[i]),
                time=float(times[i]),
            )
            states.append(state)

        segments = []
        for i in range(n - 1):
            seg = TrajectorySegment(
                start=states[i],
                end=states[i + 1],
                duration=states[i + 1].time - states[i].time,
            )
            segments.append(seg)

        return Trajectory(states=states)


class SavitzkyGolaySmoother(TrajectorySmoothing):
    """
    Savitzky-Golay filter smoothing.

    Fits a polynomial to each window and uses the fitted value.
    Preserves local features better than moving average.

    Attributes:
        window_size: Window length (must be odd)
        poly_order: Polynomial order (must be < window_size)
    """

    def __init__(self, window_size: int = 7, poly_order: int = 3):
        """
        Initialize Savitzky-Golay smoother.

        Args:
            window_size: Window length (odd)
            poly_order: Polynomial order
        """
        super().__init__("savgol", preserves_endpoints=True)
        self.window_size = window_size if window_size % 2 == 1 else window_size + 1
        self.poly_order = min(poly_order, self.window_size - 1)

    def smooth(
        self,
        trajectory: Trajectory,
        **kwargs,
    ) -> Trajectory:
        """
        Smooth trajectory using Savitzky-Golay filter.

        Args:
            trajectory: Input trajectory
            **kwargs: Override parameters

        Returns:
            Smoothed trajectory
        """
        window = kwargs.get('window_size', self.window_size)
        order = kwargs.get('poly_order', self.poly_order)

        positions = trajectory.get_positions()
        times = trajectory.get_times()
        n = len(positions)

        if n < window:
            return trajectory

        # Apply Savitzky-Golay to each dimension
        smooth_positions = np.zeros_like(positions)
        for dim in range(3):
            smooth_positions[:, dim] = savgol_filter(
                positions[:, dim],
                window,
                order,
                mode='nearest',
            )

        # Preserve endpoints
        if self.preserves_endpoints:
            smooth_positions[0] = positions[0]
            smooth_positions[-1] = positions[-1]

        # Compute velocities
        velocities = np.zeros_like(smooth_positions)
        for i in range(n - 1):
            dt = times[i + 1] - times[i]
            if dt > 1e-10:
                velocities[i] = (smooth_positions[i + 1] - smooth_positions[i]) / dt
        velocities[-1] = velocities[-2] if n > 1 else np.zeros(3)

        # Build trajectory
        states = []
        for i in range(n):
            state = State(
                pose=Pose(
                    position=smooth_positions[i],
                    attitude=np.array([0.0, 0.0, 0.0]),
                ),
                velocity=Velocity(linear=velocities[i]),
                time=float(times[i]),
            )
            states.append(state)

        segments = []
        for i in range(n - 1):
            seg = TrajectorySegment(
                start=states[i],
                end=states[i + 1],
                duration=states[i + 1].time - states[i].time,
            )
            segments.append(seg)

        return Trajectory(states=states)


@dataclass
class SmoothingResult:
    """
    Result of trajectory smoothing.

    Attributes:
        original: Original trajectory
        smoothed: Smoothed trajectory
        max_deviation: Maximum deviation from original
        mean_deviation: Mean deviation from original
        algorithm: Smoothing algorithm used
    """
    original: Trajectory
    smoothed: Trajectory
    max_deviation: float = 0.0
    mean_deviation: float = 0.0
    algorithm: str = ""

    @classmethod
    def compute(
        cls,
        original: Trajectory,
        smoothed: Trajectory,
        algorithm: str = "",
    ) -> SmoothingResult:
        """Compute smoothing statistics."""
        orig_pos = original.get_positions()
        smooth_pos = smoothed.get_positions()

        # Resample if different lengths
        if len(orig_pos) != len(smooth_pos):
            # Use linear interpolation to compare
            orig_times = original.get_times()
            smooth_times = smoothed.get_times()

            common_times = np.linspace(
                max(orig_times[0], smooth_times[0]),
                min(orig_times[-1], smooth_times[-1]),
                100,
            )

            orig_interp = np.zeros((len(common_times), 3))
            smooth_interp = np.zeros((len(common_times), 3))

            for dim in range(3):
                f_orig = interpolate.interp1d(orig_times, orig_pos[:, dim])
                f_smooth = interpolate.interp1d(smooth_times, smooth_pos[:, dim])
                orig_interp[:, dim] = f_orig(common_times)
                smooth_interp[:, dim] = f_smooth(common_times)

            deviations = np.linalg.norm(orig_interp - smooth_interp, axis=1)
        else:
            deviations = np.linalg.norm(orig_pos - smooth_pos, axis=1)

        return cls(
            original=original,
            smoothed=smoothed,
            max_deviation=float(np.max(deviations)),
            mean_deviation=float(np.mean(deviations)),
            algorithm=algorithm,
        )


def smooth_trajectory(
    trajectory: Trajectory,
    method: str = 'bspline',
    **kwargs,
) -> Trajectory:
    """
    Convenience function to smooth a trajectory.

    Args:
        trajectory: Input trajectory
        method: 'bspline', 'moving_average', or 'savgol'
        **kwargs: Method-specific parameters

    Returns:
        Smoothed trajectory
    """
    if method == 'bspline':
        smoother = BSplineSmoother(
            degree=kwargs.get('degree', 3),
            smoothness=kwargs.get('smoothness', 0.0),
            n_samples=kwargs.get('n_samples'),
        )
    elif method == 'moving_average':
        smoother = MovingAverageSmoother(
            window_size=kwargs.get('window_size', 5),
        )
    elif method == 'savgol':
        smoother = SavitzkyGolaySmoother(
            window_size=kwargs.get('window_size', 7),
            poly_order=kwargs.get('poly_order', 3),
        )
    else:
        raise ValueError(f"Unknown smoothing method: {method}")

    return smoother.smooth(trajectory)
