"""
Velocity Profile Generation

This module generates time-optimal and kinodynamically feasible
velocity profiles along a given path.

Profiles:
---------
1. Trapezoidal Profile
   - Constant acceleration/deceleration
   - Maximum velocity cruise phase
   - Simple and widely used

2. S-Curve Profile
   - Bounded jerk for smooth motion
   - Reduces mechanical stress
   - Better for precision applications

3. Time-Optimal Profile
   - Maximizes velocity within constraints
   - Uses bang-bang or bang-singular-bang control
   - Accounts for path curvature

Mathematical Foundation:
------------------------
Time-optimal velocity profile satisfies:
    v(s) = min(v_max, √(a_max × ρ(s)))

where ρ(s) is the radius of curvature at arc length s.

For constrained motion:
    v² ≤ 2 × a_max × s (acceleration phase)
    v² ≤ 2 × a_max × (s_total - s) (deceleration phase)
"""

from __future__ import annotations

import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from collections.abc import Callable
import logging

from ..core.trajectory import Trajectory, TrajectorySegment
from ..core.state import State, Pose, Velocity

logger = logging.getLogger(__name__)


@dataclass
class ProfileConstraints:
    """
    Kinodynamic constraints for velocity profiles.

    Attributes:
        v_max: Maximum velocity [m/s]
        a_max: Maximum acceleration [m/s²]
        a_min: Maximum deceleration [m/s²] (positive value)
        j_max: Maximum jerk [m/s³]
        v_min: Minimum velocity [m/s]
    """
    v_max: float = 50.0  # m/s
    a_max: float = 5.0   # m/s²
    a_min: float = 5.0   # m/s² (deceleration magnitude)
    j_max: float = 10.0  # m/s³
    v_min: float = 0.0   # m/s

    def validate(self) -> bool:
        """Check constraint validity."""
        return (
            self.v_max > self.v_min >= 0 and
            self.a_max > 0 and
            self.a_min > 0 and
            self.j_max > 0
        )


@dataclass
class ProfilePoint:
    """
    Point along velocity profile.

    Attributes:
        s: Arc length [m]
        t: Time [s]
        v: Velocity [m/s]
        a: Acceleration [m/s²]
    """
    s: float
    t: float
    v: float
    a: float = 0.0


class VelocityProfile(ABC):
    """
    Abstract base class for velocity profiles.

    A velocity profile maps arc length s to velocity v(s),
    determining how fast to traverse each point along the path.
    """

    def __init__(self, name: str, constraints: ProfileConstraints):
        self.name = name
        self.constraints = constraints

    @abstractmethod
    def compute(
        self,
        path_length: float,
        v_start: float = 0.0,
        v_end: float = 0.0,
    ) -> list[ProfilePoint]:
        """
        Compute velocity profile.

        Args:
            path_length: Total path length [m]
            v_start: Initial velocity [m/s]
            v_end: Final velocity [m/s]

        Returns:
            List of profile points
        """
        pass

    def get_velocity(self, s: float, profile: list[ProfilePoint]) -> float:
        """
        Get velocity at arc length s.

        Args:
            s: Arc length
            profile: Computed profile

        Returns:
            Velocity at s
        """
        if len(profile) == 0:
            return 0.0

        # Binary search for interval
        if s <= profile[0].s:
            return profile[0].v
        if s >= profile[-1].s:
            return profile[-1].v

        for i in range(len(profile) - 1):
            if profile[i].s <= s <= profile[i + 1].s:
                # Linear interpolation
                t = (s - profile[i].s) / (profile[i + 1].s - profile[i].s)
                return profile[i].v + t * (profile[i + 1].v - profile[i].v)

        return profile[-1].v

    def total_time(self, profile: list[ProfilePoint]) -> float:
        """Get total traversal time."""
        return profile[-1].t if profile else 0.0

    def apply_to_trajectory(
        self,
        trajectory: Trajectory,
        profile: list[ProfilePoint],
    ) -> Trajectory:
        """
        Apply velocity profile to trajectory.

        Recomputes times based on the velocity profile.

        Args:
            trajectory: Input trajectory
            profile: Velocity profile

        Returns:
            Retimed trajectory
        """
        positions = trajectory.get_positions()
        n = len(positions)

        # Compute arc lengths
        arc_lengths = np.zeros(n)
        for i in range(1, n):
            arc_lengths[i] = arc_lengths[i-1] + np.linalg.norm(positions[i] - positions[i-1])

        # Compute times from profile
        new_times = np.zeros(n)
        new_times[0] = trajectory.get_times()[0]

        for i in range(1, n):
            s = arc_lengths[i]
            v = self.get_velocity(s, profile)
            v = max(v, 0.1)  # Avoid division by zero

            ds = arc_lengths[i] - arc_lengths[i-1]
            dt = ds / v
            new_times[i] = new_times[i-1] + dt

        # Compute velocities
        velocities = np.zeros((n, 3))
        for i in range(n):
            speed = self.get_velocity(arc_lengths[i], profile)
            if i < n - 1:
                direction = positions[i + 1] - positions[i]
                dist = np.linalg.norm(direction)
                if dist > 1e-10:
                    direction = direction / dist
                else:
                    direction = np.zeros(3)
            else:
                direction = velocities[i - 1] if i > 0 else np.zeros(3)
                direction = direction / (np.linalg.norm(direction) + 1e-10)

            velocities[i] = speed * direction

        # Build new trajectory
        states = []
        for i in range(n):
            state = State(
                pose=Pose(position=positions[i].copy()),
                velocity=Velocity(linear=velocities[i]),
                time=float(new_times[i]),
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


class TrapezoidalProfile(VelocityProfile):
    """
    Trapezoidal velocity profile.

    Three phases:
    1. Acceleration: v increases from v_start
    2. Cruise: constant v = v_max
    3. Deceleration: v decreases to v_end

    If path is too short, may become triangular (no cruise phase).
    """

    def __init__(self, constraints: ProfileConstraints | None = None):
        super().__init__("trapezoidal", constraints or ProfileConstraints())

    def compute(
        self,
        path_length: float,
        v_start: float = 0.0,
        v_end: float = 0.0,
        n_samples: int = 100,
    ) -> list[ProfilePoint]:
        """
        Compute trapezoidal velocity profile.

        Args:
            path_length: Total path length [m]
            v_start: Initial velocity [m/s]
            v_end: Final velocity [m/s]
            n_samples: Number of output samples

        Returns:
            List of profile points
        """
        c = self.constraints

        # Distance to accelerate to v_max
        s_acc = (c.v_max**2 - v_start**2) / (2 * c.a_max)
        # Distance to decelerate from v_max
        s_dec = (c.v_max**2 - v_end**2) / (2 * c.a_min)

        if s_acc + s_dec > path_length:
            # Triangular profile (no cruise phase)
            # Find peak velocity
            v_peak = np.sqrt(
                (2 * c.a_max * c.a_min * path_length + c.a_min * v_start**2 + c.a_max * v_end**2)
                / (c.a_max + c.a_min)
            )
            v_peak = min(v_peak, c.v_max)

            s_acc = (v_peak**2 - v_start**2) / (2 * c.a_max)
            s_dec = path_length - s_acc
        else:
            v_peak = c.v_max

        s_cruise_start = s_acc
        s_cruise_end = path_length - s_dec

        # Generate profile points
        profile = []

        # Time calculations
        t_acc = (v_peak - v_start) / c.a_max if c.a_max > 0 else 0
        t_cruise = (s_cruise_end - s_cruise_start) / v_peak if v_peak > 0 else 0
        (v_peak - v_end) / c.a_min if c.a_min > 0 else 0

        for i in range(n_samples):
            s = path_length * i / (n_samples - 1)

            if s <= s_cruise_start:
                # Acceleration phase
                v = np.sqrt(v_start**2 + 2 * c.a_max * s)
                a = c.a_max
                # Time: solve s = v_start * t + 0.5 * a * t²
                if c.a_max > 0:
                    t = (-v_start + np.sqrt(v_start**2 + 2 * c.a_max * s)) / c.a_max
                else:
                    t = s / v_start if v_start > 0 else 0

            elif s <= s_cruise_end:
                # Cruise phase
                v = v_peak
                a = 0.0
                t = t_acc + (s - s_cruise_start) / v_peak

            else:
                # Deceleration phase
                s_dec_traveled = s - s_cruise_end
                v = np.sqrt(v_peak**2 - 2 * c.a_min * s_dec_traveled)
                v = max(v, v_end)
                a = -c.a_min
                # Time
                if c.a_min > 0:
                    t = t_acc + t_cruise + (v_peak - v) / c.a_min
                else:
                    t = t_acc + t_cruise + s_dec_traveled / v_peak

            profile.append(ProfilePoint(s=s, t=t, v=v, a=a))

        return profile


class SCurveProfile(VelocityProfile):
    """
    S-curve velocity profile with bounded jerk.

    Seven phases:
    1. Jerk ramp-up (acceleration increasing)
    2. Constant acceleration
    3. Jerk ramp-down (acceleration decreasing to 0)
    4. Cruise (constant velocity)
    5. Jerk ramp-down (deceleration increasing)
    6. Constant deceleration
    7. Jerk ramp-up (deceleration decreasing to 0)

    Provides smooth motion with continuous acceleration.
    """

    def __init__(self, constraints: ProfileConstraints | None = None):
        super().__init__("s_curve", constraints or ProfileConstraints())

    def compute(
        self,
        path_length: float,
        v_start: float = 0.0,
        v_end: float = 0.0,
        n_samples: int = 100,
    ) -> list[ProfilePoint]:
        """
        Compute S-curve velocity profile.

        Args:
            path_length: Total path length [m]
            v_start: Initial velocity [m/s]
            v_end: Final velocity [m/s]
            n_samples: Number of output samples

        Returns:
            List of profile points
        """
        c = self.constraints

        # Time for jerk phase
        t_j = c.a_max / c.j_max

        # Distance during jerk phase: s = v*t + 0.5*a*t² + (1/6)*j*t³
        v_start * t_j + (1/6) * c.j_max * t_j**3

        # Velocity change during jerk phase
        dv_jerk = 0.5 * c.j_max * t_j**2

        # Time at constant acceleration
        dv_needed = c.v_max - v_start - 2 * dv_jerk
        if dv_needed > 0:
            dv_needed / c.a_max
        else:
            pass
            # Recalculate for limited acceleration

        # Use trapezoidal as approximation for now (full S-curve is complex)
        # This is a simplified version
        trap = TrapezoidalProfile(c)
        base_profile = trap.compute(path_length, v_start, v_end, n_samples)

        # Smooth the transitions using jerk limit
        profile = self._smooth_jerk(base_profile, c.j_max)

        return profile

    def _smooth_jerk(
        self,
        profile: list[ProfilePoint],
        j_max: float,
    ) -> list[ProfilePoint]:
        """Apply jerk limiting to profile."""
        if len(profile) < 3:
            return profile

        smoothed = [profile[0]]

        for i in range(1, len(profile)):
            prev = smoothed[-1]
            curr = profile[i]

            dt = curr.t - prev.t
            if dt < 1e-10:
                smoothed.append(curr)
                continue

            # Limit acceleration change (jerk)
            da = curr.a - prev.a
            da_max = j_max * dt

            if abs(da) > da_max:
                new_a = prev.a + np.sign(da) * da_max
            else:
                new_a = curr.a

            # Recompute velocity
            new_v = prev.v + 0.5 * (prev.a + new_a) * dt
            new_v = np.clip(new_v, 0, self.constraints.v_max)

            smoothed.append(ProfilePoint(
                s=curr.s,
                t=curr.t,
                v=new_v,
                a=new_a,
            ))

        return smoothed


class TimeOptimalProfile(VelocityProfile):
    """
    Time-optimal velocity profile accounting for path curvature.

    Uses the approach of:
    1. Forward pass: accelerate while respecting curvature
    2. Backward pass: decelerate to meet end conditions
    3. Merge: take minimum of forward and backward profiles

    The curvature constraint is:
        v² ≤ a_lat × ρ(s)

    where a_lat is lateral acceleration limit and ρ is radius of curvature.
    """

    def __init__(
        self,
        constraints: ProfileConstraints | None = None,
        a_lateral: float = 3.0,
    ):
        """
        Args:
            constraints: Kinodynamic constraints
            a_lateral: Maximum lateral acceleration [m/s²]
        """
        super().__init__("time_optimal", constraints or ProfileConstraints())
        self.a_lateral = a_lateral

    def compute(
        self,
        path_length: float,
        v_start: float = 0.0,
        v_end: float = 0.0,
        curvature: Callable[[float], float] | None = None,
        n_samples: int = 100,
    ) -> list[ProfilePoint]:
        """
        Compute time-optimal velocity profile.

        Args:
            path_length: Total path length [m]
            v_start: Initial velocity [m/s]
            v_end: Final velocity [m/s]
            curvature: Function κ(s) returning curvature at arc length s
            n_samples: Number of output samples

        Returns:
            List of profile points
        """
        c = self.constraints
        ds = path_length / (n_samples - 1)

        # Arc length samples
        s_samples = np.linspace(0, path_length, n_samples)

        # Velocity limit from curvature
        v_curvature = np.full(n_samples, c.v_max)
        if curvature is not None:
            for i, s in enumerate(s_samples):
                kappa = curvature(s)
                if kappa > 1e-6:
                    # v² = a_lat / κ
                    v_curv = np.sqrt(self.a_lateral / kappa)
                    v_curvature[i] = min(c.v_max, v_curv)

        # Forward pass (acceleration)
        v_forward = np.zeros(n_samples)
        v_forward[0] = v_start

        for i in range(1, n_samples):
            # v² = v₀² + 2as
            v_next = np.sqrt(v_forward[i-1]**2 + 2 * c.a_max * ds)
            v_forward[i] = min(v_next, v_curvature[i], c.v_max)

        # Backward pass (deceleration)
        v_backward = np.zeros(n_samples)
        v_backward[-1] = v_end

        for i in range(n_samples - 2, -1, -1):
            v_next = np.sqrt(v_backward[i+1]**2 + 2 * c.a_min * ds)
            v_backward[i] = min(v_next, v_curvature[i], c.v_max)

        # Merge: take minimum
        v_profile = np.minimum(v_forward, v_backward)
        v_profile = np.minimum(v_profile, v_curvature)

        # Compute times
        t_profile = np.zeros(n_samples)
        for i in range(1, n_samples):
            v_avg = 0.5 * (v_profile[i-1] + v_profile[i])
            if v_avg > 1e-10:
                t_profile[i] = t_profile[i-1] + ds / v_avg
            else:
                t_profile[i] = t_profile[i-1] + ds / 0.1

        # Compute accelerations
        a_profile = np.zeros(n_samples)
        for i in range(1, n_samples - 1):
            dt = t_profile[i+1] - t_profile[i-1]
            if dt > 1e-10:
                a_profile[i] = (v_profile[i+1] - v_profile[i-1]) / dt

        # Build profile
        profile = []
        for i in range(n_samples):
            profile.append(ProfilePoint(
                s=s_samples[i],
                t=t_profile[i],
                v=v_profile[i],
                a=a_profile[i],
            ))

        return profile

    def compute_for_trajectory(
        self,
        trajectory: Trajectory,
        v_start: float = 0.0,
        v_end: float = 0.0,
    ) -> list[ProfilePoint]:
        """
        Compute time-optimal profile for a trajectory.

        Extracts curvature from trajectory geometry.

        Args:
            trajectory: Input trajectory
            v_start: Initial velocity [m/s]
            v_end: Final velocity [m/s]

        Returns:
            List of profile points
        """
        positions = trajectory.get_positions()
        n = len(positions)

        # Compute arc lengths
        arc_lengths = np.zeros(n)
        for i in range(1, n):
            arc_lengths[i] = arc_lengths[i-1] + np.linalg.norm(positions[i] - positions[i-1])

        path_length = arc_lengths[-1]

        # Compute curvature at each point
        curvatures = np.zeros(n)
        for i in range(1, n - 1):
            # Three-point curvature estimate
            p0 = positions[i - 1]
            p1 = positions[i]
            p2 = positions[i + 1]

            # Vectors
            v1 = p1 - p0
            v2 = p2 - p1

            # Cross product magnitude (2D projection)
            cross = np.abs(v1[0] * v2[1] - v1[1] * v2[0])

            # Triangle area
            d1 = np.linalg.norm(v1)
            d2 = np.linalg.norm(v2)
            d3 = np.linalg.norm(p2 - p0)

            if d1 * d2 * d3 > 1e-10:
                # Menger curvature: κ = 4A / (abc)
                curvatures[i] = 4 * cross / (d1 * d2 * d3)
            else:
                curvatures[i] = 0.0

        curvatures[0] = curvatures[1] if n > 1 else 0.0
        curvatures[-1] = curvatures[-2] if n > 1 else 0.0

        # Create curvature interpolation function
        from scipy.interpolate import interp1d
        curvature_func = interp1d(
            arc_lengths, curvatures,
            kind='linear',
            fill_value=(curvatures[0], curvatures[-1]),
            bounds_error=False,
        )

        return self.compute(
            path_length,
            v_start,
            v_end,
            curvature=curvature_func,
            n_samples=n,
        )


def create_velocity_profile(
    path_length: float,
    profile_type: str = 'trapezoidal',
    constraints: ProfileConstraints | None = None,
    v_start: float = 0.0,
    v_end: float = 0.0,
    **kwargs,
) -> list[ProfilePoint]:
    """
    Convenience function to create a velocity profile.

    Args:
        path_length: Total path length [m]
        profile_type: 'trapezoidal', 's_curve', or 'time_optimal'
        constraints: Kinodynamic constraints
        v_start: Initial velocity [m/s]
        v_end: Final velocity [m/s]
        **kwargs: Additional profile-specific parameters

    Returns:
        List of profile points
    """
    constraints = constraints or ProfileConstraints()

    if profile_type == 'trapezoidal':
        profile = TrapezoidalProfile(constraints)
    elif profile_type == 's_curve':
        profile = SCurveProfile(constraints)
    elif profile_type == 'time_optimal':
        a_lateral = kwargs.get('a_lateral', 3.0)
        profile = TimeOptimalProfile(constraints, a_lateral)
    else:
        raise ValueError(f"Unknown profile type: {profile_type}")

    return profile.compute(path_length, v_start, v_end, **kwargs)
