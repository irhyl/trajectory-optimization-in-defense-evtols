"""
Trajectory Feasibility Checking

This module validates trajectory feasibility against kinodynamic
and environmental constraints.

Validation Levels:
------------------
1. Kinodynamic: velocity, acceleration, jerk limits
2. Geometric: turn radius, climb angle
3. Temporal: timing consistency
4. Environmental: terrain clearance, no-fly zones

Author: Defense eVTOL Research Team
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from collections.abc import Callable
from enum import Enum
import logging

from ..core.trajectory import Trajectory

logger = logging.getLogger(__name__)


class ViolationType(Enum):
    """Types of constraint violations."""
    MAX_VELOCITY = "max_velocity"
    MIN_VELOCITY = "min_velocity"
    MAX_ACCELERATION = "max_acceleration"
    MAX_DECELERATION = "max_deceleration"
    MAX_JERK = "max_jerk"
    MAX_TURN_RATE = "max_turn_rate"
    MAX_CLIMB_RATE = "max_climb_rate"
    MIN_TURN_RADIUS = "min_turn_radius"
    MAX_BANK_ANGLE = "max_bank_angle"
    TERRAIN_CLEARANCE = "terrain_clearance"
    NO_FLY_ZONE = "no_fly_zone"
    TIMING_CONSISTENCY = "timing_consistency"
    POSITION_CONTINUITY = "position_continuity"


@dataclass
class ConstraintViolation:
    """
    Record of a constraint violation.

    Attributes:
        type: Type of violation
        location: Index or arc length where violation occurs
        value: Actual value
        limit: Constraint limit
        severity: Severity measure (value/limit)
        message: Human-readable description
    """
    type: ViolationType
    location: float  # Arc length or index
    time: float  # Time of violation
    value: float
    limit: float
    severity: float = 0.0
    message: str = ""

    def __post_init__(self):
        if self.severity == 0.0 and self.limit != 0:
            self.severity = abs(self.value) / abs(self.limit)
        if not self.message:
            self.message = f"{self.type.value}: {self.value:.3f} exceeds {self.limit:.3f}"


@dataclass
class KinodynamicConstraints:
    """
    Kinodynamic constraint parameters.

    Attributes:
        v_max: Maximum velocity [m/s]
        v_min: Minimum velocity [m/s]
        a_max: Maximum acceleration [m/s²]
        a_min: Maximum deceleration [m/s²]
        j_max: Maximum jerk [m/s³]
        omega_max: Maximum angular velocity [rad/s]
        climb_max: Maximum climb rate [m/s]
        descent_max: Maximum descent rate [m/s]
        r_min: Minimum turn radius [m]
        bank_max: Maximum bank angle [rad]
    """
    v_max: float = 50.0
    v_min: float = 0.0
    a_max: float = 5.0
    a_min: float = 5.0
    j_max: float = 10.0
    omega_max: float = 0.5  # rad/s
    climb_max: float = 10.0  # m/s
    descent_max: float = 10.0  # m/s
    r_min: float = 50.0  # m
    bank_max: float = 0.52  # rad (~30 degrees)


@dataclass
class FeasibilityResult:
    """
    Result of feasibility check.

    Attributes:
        feasible: Whether trajectory is feasible
        violations: List of constraint violations
        max_severity: Maximum violation severity
        summary: Summary statistics
    """
    feasible: bool
    violations: list[ConstraintViolation] = field(default_factory=list)
    max_severity: float = 0.0
    summary: dict[str, int] = field(default_factory=dict)

    def __post_init__(self):
        if self.violations:
            self.max_severity = max(v.severity for v in self.violations)
            # Count by type
            for v in self.violations:
                type_name = v.type.value
                self.summary[type_name] = self.summary.get(type_name, 0) + 1

    def add_violation(self, violation: ConstraintViolation) -> None:
        """Add a violation and update statistics."""
        self.violations.append(violation)
        self.feasible = False
        self.max_severity = max(self.max_severity, violation.severity)
        type_name = violation.type.value
        self.summary[type_name] = self.summary.get(type_name, 0) + 1

    def report(self) -> str:
        """Generate human-readable report."""
        lines = ["Feasibility Report", "=" * 40]

        if self.feasible:
            lines.append("Status: FEASIBLE")
        else:
            lines.append(f"Status: INFEASIBLE ({len(self.violations)} violations)")
            lines.append(f"Maximum severity: {self.max_severity:.2f}")
            lines.append("")
            lines.append("Violations by type:")
            for type_name, count in sorted(self.summary.items()):
                lines.append(f"  {type_name}: {count}")

            if len(self.violations) <= 10:
                lines.append("")
                lines.append("Violation details:")
                for v in self.violations:
                    lines.append(f"  t={v.time:.2f}s: {v.message}")

        return "\n".join(lines)


class FeasibilityChecker:
    """
    Base class for feasibility checking.

    Provides common utilities for trajectory validation.
    """

    def __init__(self, name: str):
        self.name = name

    def check(self, trajectory: Trajectory) -> FeasibilityResult:
        """
        Check trajectory feasibility.

        Args:
            trajectory: Trajectory to validate

        Returns:
            FeasibilityResult
        """
        raise NotImplementedError

    def _compute_velocities(self, trajectory: Trajectory) -> np.ndarray:
        """Compute velocity magnitudes."""
        positions = trajectory.get_positions()
        times = trajectory.get_times()
        n = len(positions)

        velocities = np.zeros(n)
        for i in range(1, n):
            dt = times[i] - times[i - 1]
            if dt > 1e-10:
                dx = np.linalg.norm(positions[i] - positions[i - 1])
                velocities[i] = dx / dt
        velocities[0] = velocities[1] if n > 1 else 0.0

        return velocities

    def _compute_accelerations(self, trajectory: Trajectory) -> np.ndarray:
        """Compute acceleration magnitudes."""
        velocities = self._compute_velocities(trajectory)
        times = trajectory.get_times()
        n = len(velocities)

        accelerations = np.zeros(n)
        for i in range(1, n):
            dt = times[i] - times[i - 1]
            if dt > 1e-10:
                accelerations[i] = (velocities[i] - velocities[i - 1]) / dt

        return accelerations

    def _compute_arc_lengths(self, trajectory: Trajectory) -> np.ndarray:
        """Compute cumulative arc lengths."""
        positions = trajectory.get_positions()
        n = len(positions)

        arc_lengths = np.zeros(n)
        for i in range(1, n):
            arc_lengths[i] = arc_lengths[i - 1] + np.linalg.norm(positions[i] - positions[i - 1])

        return arc_lengths


class KinodynamicChecker(FeasibilityChecker):
    """
    Check kinodynamic feasibility.

    Validates velocity, acceleration, jerk, and turn rate constraints.
    """

    def __init__(self, constraints: KinodynamicConstraints | None = None):
        super().__init__("kinodynamic")
        self.constraints = constraints or KinodynamicConstraints()

    def check(self, trajectory: Trajectory) -> FeasibilityResult:
        """
        Check trajectory kinodynamic feasibility.

        Args:
            trajectory: Trajectory to validate

        Returns:
            FeasibilityResult with violations
        """
        result = FeasibilityResult(feasible=True)

        if len(trajectory) < 2:
            return result

        positions = trajectory.get_positions()
        times = trajectory.get_times()
        n = len(positions)

        # Compute kinematics
        velocities = self._compute_velocity_vectors(trajectory)
        speeds = np.linalg.norm(velocities, axis=1)
        accelerations = self._compute_acceleration_vectors(trajectory)
        np.linalg.norm(accelerations, axis=1)

        arc_lengths = self._compute_arc_lengths(trajectory)

        # Check velocity limits
        for i in range(n):
            if speeds[i] > self.constraints.v_max:
                result.add_violation(ConstraintViolation(
                    type=ViolationType.MAX_VELOCITY,
                    location=arc_lengths[i],
                    time=times[i],
                    value=speeds[i],
                    limit=self.constraints.v_max,
                ))

            if i > 0 and speeds[i] < self.constraints.v_min and arc_lengths[i] > 10:
                result.add_violation(ConstraintViolation(
                    type=ViolationType.MIN_VELOCITY,
                    location=arc_lengths[i],
                    time=times[i],
                    value=speeds[i],
                    limit=self.constraints.v_min,
                ))

        # Check acceleration limits
        for i in range(1, n):
            # Tangential acceleration
            if i < n - 1:
                dv = speeds[i] - speeds[i - 1]
                dt = times[i] - times[i - 1]
                if dt > 1e-10:
                    tangent_accel = dv / dt

                    if tangent_accel > self.constraints.a_max:
                        result.add_violation(ConstraintViolation(
                            type=ViolationType.MAX_ACCELERATION,
                            location=arc_lengths[i],
                            time=times[i],
                            value=tangent_accel,
                            limit=self.constraints.a_max,
                        ))

                    if tangent_accel < -self.constraints.a_min:
                        result.add_violation(ConstraintViolation(
                            type=ViolationType.MAX_DECELERATION,
                            location=arc_lengths[i],
                            time=times[i],
                            value=-tangent_accel,
                            limit=self.constraints.a_min,
                        ))

        # Check jerk limits
        jerks = self._compute_jerk(trajectory)
        for i in range(len(jerks)):
            if abs(jerks[i]) > self.constraints.j_max:
                result.add_violation(ConstraintViolation(
                    type=ViolationType.MAX_JERK,
                    location=arc_lengths[min(i, n - 1)],
                    time=times[min(i, n - 1)],
                    value=abs(jerks[i]),
                    limit=self.constraints.j_max,
                ))

        # Check climb rate
        for i in range(1, n):
            dt = times[i] - times[i - 1]
            if dt > 1e-10:
                climb_rate = (positions[i, 2] - positions[i - 1, 2]) / dt

                if climb_rate > self.constraints.climb_max:
                    result.add_violation(ConstraintViolation(
                        type=ViolationType.MAX_CLIMB_RATE,
                        location=arc_lengths[i],
                        time=times[i],
                        value=climb_rate,
                        limit=self.constraints.climb_max,
                    ))

                if climb_rate < -self.constraints.descent_max:
                    result.add_violation(ConstraintViolation(
                        type=ViolationType.MAX_CLIMB_RATE,
                        location=arc_lengths[i],
                        time=times[i],
                        value=-climb_rate,
                        limit=self.constraints.descent_max,
                        message=f"Descent rate {-climb_rate:.2f} exceeds {self.constraints.descent_max:.2f}",
                    ))

        # Check turn rate
        turn_rates = self._compute_turn_rates(trajectory)
        for i, omega in enumerate(turn_rates):
            if abs(omega) > self.constraints.omega_max:
                result.add_violation(ConstraintViolation(
                    type=ViolationType.MAX_TURN_RATE,
                    location=arc_lengths[min(i, n - 1)],
                    time=times[min(i, n - 1)],
                    value=abs(omega),
                    limit=self.constraints.omega_max,
                ))

        # Check turn radius
        curvatures = self._compute_curvatures(trajectory)
        for i, kappa in enumerate(curvatures):
            if kappa > 1e-6:
                radius = 1.0 / kappa
                if radius < self.constraints.r_min:
                    result.add_violation(ConstraintViolation(
                        type=ViolationType.MIN_TURN_RADIUS,
                        location=arc_lengths[min(i, n - 1)],
                        time=times[min(i, n - 1)],
                        value=radius,
                        limit=self.constraints.r_min,
                        message=f"Turn radius {radius:.1f}m below minimum {self.constraints.r_min:.1f}m",
                    ))

        return result

    def _compute_velocity_vectors(self, trajectory: Trajectory) -> np.ndarray:
        """Compute velocity vectors."""
        positions = trajectory.get_positions()
        times = trajectory.get_times()
        n = len(positions)

        velocities = np.zeros((n, 3))
        for i in range(1, n):
            dt = times[i] - times[i - 1]
            if dt > 1e-10:
                velocities[i] = (positions[i] - positions[i - 1]) / dt
        velocities[0] = velocities[1] if n > 1 else np.zeros(3)

        return velocities

    def _compute_acceleration_vectors(self, trajectory: Trajectory) -> np.ndarray:
        """Compute acceleration vectors."""
        velocities = self._compute_velocity_vectors(trajectory)
        times = trajectory.get_times()
        n = len(velocities)

        accelerations = np.zeros((n, 3))
        for i in range(1, n):
            dt = times[i] - times[i - 1]
            if dt > 1e-10:
                accelerations[i] = (velocities[i] - velocities[i - 1]) / dt

        return accelerations

    def _compute_jerk(self, trajectory: Trajectory) -> np.ndarray:
        """Compute jerk magnitudes."""
        accelerations = self._compute_acceleration_vectors(trajectory)
        times = trajectory.get_times()
        n = len(accelerations)

        jerks = np.zeros(n)
        for i in range(1, n):
            dt = times[i] - times[i - 1]
            if dt > 1e-10:
                da = np.linalg.norm(accelerations[i] - accelerations[i - 1])
                jerks[i] = da / dt

        return jerks

    def _compute_turn_rates(self, trajectory: Trajectory) -> np.ndarray:
        """Compute turn (yaw) rates."""
        positions = trajectory.get_positions()
        times = trajectory.get_times()
        n = len(positions)

        # Compute headings
        headings = np.zeros(n)
        for i in range(n - 1):
            dx = positions[i + 1, 0] - positions[i, 0]
            dy = positions[i + 1, 1] - positions[i, 1]
            headings[i] = np.arctan2(dy, dx)
        headings[-1] = headings[-2] if n > 1 else 0.0

        # Compute turn rates
        turn_rates = np.zeros(n)
        for i in range(1, n):
            dt = times[i] - times[i - 1]
            if dt > 1e-10:
                dh = headings[i] - headings[i - 1]
                # Handle wraparound
                while dh > np.pi:
                    dh -= 2 * np.pi
                while dh < -np.pi:
                    dh += 2 * np.pi
                turn_rates[i] = dh / dt

        return turn_rates

    def _compute_curvatures(self, trajectory: Trajectory) -> np.ndarray:
        """Compute path curvatures."""
        positions = trajectory.get_positions()
        n = len(positions)

        curvatures = np.zeros(n)
        for i in range(1, n - 1):
            p0 = positions[i - 1]
            p1 = positions[i]
            p2 = positions[i + 1]

            # Menger curvature
            v1 = p1 - p0
            v2 = p2 - p1

            # 2D projection for curvature
            cross = abs(v1[0] * v2[1] - v1[1] * v2[0])

            d1 = np.linalg.norm(v1)
            d2 = np.linalg.norm(v2)
            d3 = np.linalg.norm(p2 - p0)

            if d1 * d2 * d3 > 1e-10:
                curvatures[i] = 4 * cross / (d1 * d2 * d3)

        return curvatures


class EnvironmentalChecker(FeasibilityChecker):
    """
    Check environmental feasibility.

    Validates terrain clearance and no-fly zone avoidance.
    """

    def __init__(
        self,
        terrain_field: Callable[[float, float], float] | None = None,
        min_clearance: float = 50.0,
        no_fly_zones: list[tuple[np.ndarray, float]] | None = None,
    ):
        """
        Args:
            terrain_field: Function (lat, lon) -> elevation
            min_clearance: Minimum terrain clearance [m]
            no_fly_zones: List of (center, radius) tuples
        """
        super().__init__("environmental")
        self.terrain_field = terrain_field
        self.min_clearance = min_clearance
        self.no_fly_zones = no_fly_zones or []

    def check(self, trajectory: Trajectory) -> FeasibilityResult:
        """
        Check environmental feasibility.

        Args:
            trajectory: Trajectory to validate

        Returns:
            FeasibilityResult with violations
        """
        result = FeasibilityResult(feasible=True)

        positions = trajectory.get_positions()
        times = trajectory.get_times()
        arc_lengths = self._compute_arc_lengths(trajectory)

        for i, pos in enumerate(positions):
            # Check terrain clearance
            if self.terrain_field is not None:
                terrain_elev = self.terrain_field(pos[0], pos[1])
                clearance = pos[2] - terrain_elev

                if clearance < self.min_clearance:
                    result.add_violation(ConstraintViolation(
                        type=ViolationType.TERRAIN_CLEARANCE,
                        location=arc_lengths[i],
                        time=times[i],
                        value=clearance,
                        limit=self.min_clearance,
                        message=f"Terrain clearance {clearance:.1f}m below {self.min_clearance:.1f}m",
                    ))

            # Check no-fly zones
            for zone_idx, (center, radius) in enumerate(self.no_fly_zones):
                dist = np.linalg.norm(pos[:2] - center[:2])
                if dist < radius:
                    result.add_violation(ConstraintViolation(
                        type=ViolationType.NO_FLY_ZONE,
                        location=arc_lengths[i],
                        time=times[i],
                        value=dist,
                        limit=radius,
                        message=f"Inside no-fly zone {zone_idx} (dist={dist:.1f}m, radius={radius:.1f}m)",
                    ))

        return result


class ComprehensiveChecker:
    """
    Comprehensive feasibility checker combining multiple checks.
    """

    def __init__(
        self,
        kinodynamic: KinodynamicConstraints | None = None,
        terrain_field: Callable | None = None,
        min_clearance: float = 50.0,
        no_fly_zones: list | None = None,
    ):
        self.kinodynamic_checker = KinodynamicChecker(kinodynamic)
        self.environmental_checker = EnvironmentalChecker(
            terrain_field, min_clearance, no_fly_zones
        )

    def check(self, trajectory: Trajectory) -> FeasibilityResult:
        """
        Run all feasibility checks.

        Args:
            trajectory: Trajectory to validate

        Returns:
            Combined FeasibilityResult
        """
        result = FeasibilityResult(feasible=True)

        # Kinodynamic check
        kinodynamic_result = self.kinodynamic_checker.check(trajectory)
        for v in kinodynamic_result.violations:
            result.add_violation(v)

        # Environmental check
        environmental_result = self.environmental_checker.check(trajectory)
        for v in environmental_result.violations:
            result.add_violation(v)

        return result


def check_trajectory_feasibility(
    trajectory: Trajectory,
    constraints: KinodynamicConstraints | None = None,
    terrain_field: Callable | None = None,
    min_clearance: float = 50.0,
) -> FeasibilityResult:
    """
    Convenience function to check trajectory feasibility.

    Args:
        trajectory: Trajectory to validate
        constraints: Kinodynamic constraints
        terrain_field: Terrain elevation function
        min_clearance: Minimum terrain clearance

    Returns:
        FeasibilityResult
    """
    checker = ComprehensiveChecker(
        kinodynamic=constraints,
        terrain_field=terrain_field,
        min_clearance=min_clearance,
    )
    return checker.check(trajectory)
