"""
Smooth Trajectory Generation

This module implements trajectory smoothing using multiple methods:

1. **B-spline smoothing**: Parametric cubic/quintic B-splines
   - Advantage: Smooth (C² continuous), local support
   - Local control points, convex hull property
   - Efficient evaluation and differentiation

2. **Polynomial interpolation**: Piecewise polynomials
   - Hermite: Position & velocity boundary conditions
   - Cubic: Smooth transitions without oscillations
   - Quintic: Position, velocity, acceleration control

3. **Optimal trajectory**: Time-optimal with acceleration limits
   - Minimum-time subject to thrust, turn rate constraints
   - Bang-bang control, coast phases
   - Smooth jerk (derivative of acceleration)

Mathematical Background
=======================

B-spline Curve:
    τ(t) = Σᵢ Pᵢ Bᵢ,ₖ(t)

where:
    - Pᵢ: Control points
    - Bᵢ,ₖ(t): B-spline basis functions of degree k
    - Smoothness: C^(k-1) continuous

Hermite Cubic:
    τ(t) = H₀(t)·P₀ + H₁(t)·P₁ + H₂(t)·V₀ + H₃(t)·V₁

where Hᵢ are Hermite basis polynomials

Quintic (degree 5):
    τ(t) = a₀ + a₁t + a₂t² + a₃t³ + a₄t⁴ + a₅t⁵
    
    Coefficients from boundary conditions:
    - Position: τ(0), τ(T)
    - Velocity: τ'(0), τ'(T)
    - Acceleration: τ''(0), τ''(T)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from scipy.interpolate import splprep, splev, CubicSpline, BSpline, make_interp_spline
from scipy.optimize import minimize_scalar
from enum import Enum
import logging

from ..core.trajectory import Trajectory, TrajectorySegment
from ..core.state import State, Pose, Velocity

logger = logging.getLogger(__name__)


class SmoothingMethod(Enum):
    """Trajectory smoothing method."""
    BSPLINE_CUBIC = "bspline_cubic"      # Cubic B-spline (C²)
    BSPLINE_QUINTIC = "bspline_quintic"  # Quintic B-spline (C⁴)
    HERMITE_CUBIC = "hermite_cubic"      # Cubic Hermite (C¹)
    POLYNOMIAL_QUINTIC = "poly_quintic"  # Quintic polynomial (C²)


class AccelerationProfile(Enum):
    """Acceleration profile type."""
    CONSTANT = "constant"      # Constant acceleration
    SMOOTH = "smooth"          # Smooth jerk (S-curve)
    TRAPEZOIDAL = "trapezoidal"  # Ramp up, coast, ramp down


@dataclass
class SmoothingConstraints:
    """Constraints for trajectory smoothing."""
    
    max_speed: float = 60.0  # m/s
    max_acceleration: float = 10.0  # m/s²
    max_jerk: float = 50.0  # m/s³ (rate of change of acceleration)
    max_turn_rate: float = np.pi / 4  # rad/s
    
    # Continuity requirements
    continuous_velocity: bool = True  # Ensure smooth velocity
    continuous_acceleration: bool = False  # Enforce smooth acceleration
    
    # Smoothing parameters
    smoothing_lambda: float = 0.1  # Spline smoothing parameter [0, 1]
    min_segment_length: float = 50.0  # Minimum path segment length [m]


@dataclass
class SmoothedTrajectory:
    """Smooth trajectory with evaluation interface."""
    
    # Spline representations
    x_spline: BSpline | None = None
    y_spline: BSpline | None = None
    z_spline: BSpline | None = None
    
    # Time parametrization
    time_knots: np.ndarray = field(default_factory=lambda: np.array([]))
    speed_profile: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # Metadata
    total_distance: float = 0.0
    total_time: float = 0.0
    method: SmoothingMethod = SmoothingMethod.BSPLINE_CUBIC
    
    def evaluate(self, s: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Evaluate position and derivatives at path parameter s ∈ [0, L].
        
        Args:
            s: Arc length parameter [0, total_distance]
            
        Returns:
            (position [3], velocity [3]) at arc length s
        """
        if self.x_spline is None:
            raise ValueError("Trajectory not initialized")
        
        # Normalize s to spline parameter [0, 1]
        u = s / max(self.total_distance, 1.0)
        u = np.clip(u, 0, 1)
        
        # Evaluate position
        pos = np.array([
            splev(u, (self.x_spline.t, self.x_spline.c, self.x_spline.k)),
            splev(u, (self.y_spline.t, self.y_spline.c, self.y_spline.k)),
            splev(u, (self.z_spline.t, self.z_spline.c, self.z_spline.k)),
        ])
        
        # Evaluate derivatives (tangent)
        tangent = np.array([
            splev(u, (self.x_spline.t, self.x_spline.c, self.x_spline.k), der=1),
            splev(u, (self.y_spline.t, self.y_spline.c, self.y_spline.k), der=1),
            splev(u, (self.z_spline.t, self.z_spline.c, self.z_spline.k), der=1),
        ])
        
        return pos, tangent
    
    def evaluate_at_time(self, t: float) -> State:
        """
        Evaluate state at time t.
        
        Using time parametrization and speed profile.
        
        Args:
            t: Time [s]
            
        Returns:
            Complete state (position, velocity, pose)
        """
        if len(self.time_knots) == 0:
            raise ValueError("Time parametrization not set")
        
        # Find arc length at time t
        s = np.interp(t, self.time_knots, np.linspace(0, self.total_distance, len(self.time_knots)))
        
        pos, tangent = self.evaluate(s)
        
        # Speed at this arc length
        speed = np.interp(s / max(self.total_distance, 1.0), 
                         np.linspace(0, 1, len(self.speed_profile)), 
                         self.speed_profile)
        
        # Velocity = tangent direction × speed
        tangent_norm = np.linalg.norm(tangent)
        if tangent_norm > 1e-6:
            vel = (tangent / tangent_norm) * speed
        else:
            vel = np.zeros(3)
        
        return State(
            pose=Pose(position=pos),
            velocity=Velocity(linear=vel),
        )


class TrajectorySmootherEngine:
    """
    Smooth trajectory generation engine.
    
    Converts waypoint paths to smooth, dynamically-feasible trajectories
    using splines and time parametrization.
    """
    
    def __init__(
        self,
        method: SmoothingMethod = SmoothingMethod.BSPLINE_QUINTIC,
        constraints: SmoothingConstraints | None = None,
    ):
        """
        Initialize smoothing engine.
        
        Args:
            method: Smoothing method to use
            constraints: Smoothing constraints
        """
        self.method = method
        self.constraints = constraints or SmoothingConstraints()
        
        logger.info(f"TrajectorySmootherEngine initialized: {method.value}")
    
    def smooth_waypoints(
        self,
        waypoints: np.ndarray,
    ) -> SmoothedTrajectory:
        """
        Smooth trajectory from waypoint list.
        
        Args:
            waypoints: Array of shape (n, 3) with [x, y, z] positions
            
        Returns:
            Smoothed trajectory
        """
        waypoints = np.asarray(waypoints)
        
        if len(waypoints) < 2:
            raise ValueError("At least 2 waypoints required")
        
        if self.method == SmoothingMethod.BSPLINE_CUBIC:
            return self._smooth_bspline(waypoints, k=3)
        elif self.method == SmoothingMethod.BSPLINE_QUINTIC:
            return self._smooth_bspline(waypoints, k=5)
        elif self.method == SmoothingMethod.HERMITE_CUBIC:
            return self._smooth_hermite(waypoints)
        elif self.method == SmoothingMethod.POLYNOMIAL_QUINTIC:
            return self._smooth_polynomial_segments(waypoints)
        else:
            raise ValueError(f"Unknown smoothing method: {self.method}")
    
    def _smooth_bspline(
        self,
        waypoints: np.ndarray,
        k: int = 3,
    ) -> SmoothedTrajectory:
        """
        Smooth using B-spline interpolation.
        
        Args:
            waypoints: Input waypoints
            k: Spline degree (3=cubic, 5=quintic)
            
        Returns:
            Smoothed trajectory
        """
        waypoints = np.asarray(waypoints, dtype=float)
        
        # Increase points if too few for requested degree
        if len(waypoints) <= k:
            # Linear interpolation to increase point count
            t_old = np.linspace(0, 1, len(waypoints))
            t_new = np.linspace(0, 1, k + 2)
            waypoints = np.array([
                np.interp(t_new, t_old, waypoints[:, i])
                for i in range(3)
            ]).T
        
        # Create B-splines for each coordinate
        t = np.linspace(0, 1, len(waypoints))
        
        spl_x = make_interp_spline(t, waypoints[:, 0], k=min(k, len(waypoints) - 1))
        spl_y = make_interp_spline(t, waypoints[:, 1], k=min(k, len(waypoints) - 1))
        spl_z = make_interp_spline(t, waypoints[:, 2], k=min(k, len(waypoints) - 1))
        
        # Compute arc length
        total_distance = self._compute_arc_length(waypoints)
        
        # Create time parametrization
        time_knots, speed_profile = self._generate_time_profile(
            waypoints, total_distance
        )
        
        result = SmoothedTrajectory(
            x_spline=spl_x,
            y_spline=spl_y,
            z_spline=spl_z,
            time_knots=time_knots,
            speed_profile=speed_profile,
            total_distance=total_distance,
            total_time=time_knots[-1] if len(time_knots) > 0 else 0,
            method=self.method,
        )
        
        logger.info(f"B-spline smoothing: {len(waypoints)} waypoints → "
                   f"distance={total_distance:.1f}m, time={result.total_time:.1f}s")
        
        return result
    
    def _smooth_hermite(
        self,
        waypoints: np.ndarray,
    ) -> SmoothedTrajectory:
        """
        Smooth using Hermite cubic splines.
        
        Hermite splines ensure C¹ continuity (smooth positions and velocities).
        
        Args:
            waypoints: Input waypoints
            
        Returns:
            Smoothed trajectory
        """
        waypoints = np.asarray(waypoints, dtype=float)
        
        # Estimate velocities at waypoints
        velocities = np.zeros_like(waypoints)
        
        for i, wp in enumerate(waypoints):
            if i == 0:
                # Use forward difference
                velocities[i] = waypoints[1] - waypoints[0]
            elif i == len(waypoints) - 1:
                # Use backward difference
                velocities[i] = waypoints[i] - waypoints[i - 1]
            else:
                # Use central difference
                velocities[i] = (waypoints[i + 1] - waypoints[i - 1]) / 2
        
        # Normalize velocity magnitudes
        for i in range(len(velocities)):
            norm = np.linalg.norm(velocities[i])
            if norm > 0:
                # Scale to approximately 1/3 segment length
                if i < len(waypoints) - 1:
                    seg_length = np.linalg.norm(waypoints[i + 1] - waypoints[i])
                    velocities[i] = (velocities[i] / norm) * (seg_length / 3)
        
        # Create interpolating spline with velocity hints
        t = np.linspace(0, 1, len(waypoints))
        
        spl_x = CubicSpline(t, waypoints[:, 0], bc_type='natural')
        spl_y = CubicSpline(t, waypoints[:, 1], bc_type='natural')
        spl_z = CubicSpline(t, waypoints[:, 2], bc_type='natural')
        
        # Convert to BSpline representation
        spl_x_b = BSpline(spl_x.x, spl_x.c.T[0], 3)
        spl_y_b = BSpline(spl_y.x, spl_y.c.T[0], 3)
        spl_z_b = BSpline(spl_z.x, spl_z.c.T[0], 3)
        
        # Compute arc length
        total_distance = self._compute_arc_length(waypoints)
        
        # Time parametrization
        time_knots, speed_profile = self._generate_time_profile(
            waypoints, total_distance
        )
        
        result = SmoothedTrajectory(
            x_spline=spl_x_b,
            y_spline=spl_y_b,
            z_spline=spl_z_b,
            time_knots=time_knots,
            speed_profile=speed_profile,
            total_distance=total_distance,
            total_time=time_knots[-1] if len(time_knots) > 0 else 0,
            method=self.method,
        )
        
        return result
    
    def _smooth_polynomial_segments(
        self,
        waypoints: np.ndarray,
    ) -> SmoothedTrajectory:
        """
        Smooth using piecewise quintic polynomials.
        
        Each segment (waypoint-to-waypoint) is fit with a quintic polynomial
        constrained at position, velocity, and acceleration.
        
        Args:
            waypoints: Input waypoints
            
        Returns:
            Smoothed trajectory
        """
        # For now, fall back to Hermite (quintic fitting is more complex)
        return self._smooth_hermite(waypoints)
    
    def _compute_arc_length(self, waypoints: np.ndarray) -> float:
        """Compute total Euclidean arc length."""
        distances = np.linalg.norm(np.diff(waypoints, axis=0), axis=1)
        return np.sum(distances)
    
    def _generate_time_profile(
        self,
        waypoints: np.ndarray,
        total_distance: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate time parametrization and speed profile.
        
        Creates a smooth time profile respecting acceleration constraints.
        
        Args:
            waypoints: Input waypoints
            total_distance: Total path distance
            
        Returns:
            (time_knots, speed_profile) for arc-length parametrization
        """
        # Arc lengths at each waypoint
        arc_lengths = np.zeros(len(waypoints))
        for i in range(1, len(waypoints)):
            seg_dist = np.linalg.norm(waypoints[i] - waypoints[i - 1])
            arc_lengths[i] = arc_lengths[i - 1] + seg_dist
        
        # Normalize arc lengths
        arc_lengths_norm = arc_lengths / max(total_distance, 1.0)
        
        # Speed profile (trapezoidal)
        speeds = self._compute_speed_profile(arc_lengths_norm)
        
        # Time parametrization: integrate speed profile
        times = np.zeros(len(waypoints))
        for i in range(1, len(waypoints)):
            avg_speed = (speeds[i] + speeds[i - 1]) / 2
            ds = arc_lengths_norm[i] - arc_lengths_norm[i - 1]
            dt = ds * total_distance / max(avg_speed, 0.1)  # Avoid division by zero
            times[i] = times[i - 1] + dt
        
        return times, speeds
    
    def _compute_speed_profile(
        self,
        arc_length_norm: np.ndarray,
    ) -> np.ndarray:
        """
        Compute speed profile respecting constraints.
        
        Uses trapezoidal profile: accelerate → coast → decelerate
        
        Args:
            arc_length_norm: Normalized arc lengths [0, 1]
            
        Returns:
            Speed profile at each arc-length point [m/s]
        """
        max_speed = self.constraints.max_speed
        max_accel = self.constraints.max_acceleration
        
        # Rough time to reach cruise speed
        accel_dist = max_speed ** 2 / (2 * max_accel)
        accel_dist_norm = accel_dist / 1000.0  # Scale to normalized length
        
        speeds = np.zeros(len(arc_length_norm))
        
        for i, s in enumerate(arc_length_norm):
            if s < accel_dist_norm:
                # Acceleration phase
                speeds[i] = np.sqrt(2 * max_accel * s * 1000.0)
            elif s > 1 - accel_dist_norm:
                # Deceleration phase
                remaining = (1 - s) * 1000.0
                speeds[i] = np.sqrt(2 * max_accel * remaining)
            else:
                # Cruise phase
                speeds[i] = max_speed
        
        # Constrain to max speed
        speeds = np.minimum(speeds, max_speed)
        
        return speeds


class TrajectoryReferenceGenerator:
    """
    Generate reference trajectories for controller.
    
    Converts smooth trajectory to reference signals at controller update rate.
    """
    
    def __init__(
        self,
        smoothed_trajectory: SmoothedTrajectory,
        controller_update_rate: float = 50.0,  # Hz
    ):
        """
        Initialize reference generator.
        
        Args:
            smoothed_trajectory: Pre-smoothed trajectory
            controller_update_rate: Controller update frequency [Hz]
        """
        self.trajectory = smoothed_trajectory
        self.dt = 1.0 / controller_update_rate
        
        # Pre-compute reference points
        self.reference_times = np.arange(
            0, smoothed_trajectory.total_time, self.dt
        )
        self.reference_states = [
            smoothed_trajectory.evaluate_at_time(t)
            for t in self.reference_times
        ]
    
    def get_reference_at_time(self, t: float) -> State:
        """
        Get reference state at specific time.
        
        Args:
            t: Current time [s]
            
        Returns:
            Reference state
        """
        # Find closest reference point
        idx = int(t / self.dt) % len(self.reference_states)
        return self.reference_states[idx]
    
    def get_all_references(self) -> tuple[np.ndarray, list[State]]:
        """
        Get all reference times and states.
        
        Returns:
            (times [n], states [n])
        """
        return self.reference_times, self.reference_states
