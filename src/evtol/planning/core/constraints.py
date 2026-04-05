"""
Constraint Definitions

This module defines constraints for trajectory optimization. Constraints
represent physical, operational, and safety limits that trajectories must
satisfy.

Mathematical Formulation
========================

Constraints are expressed in three standard forms:

1. **Inequality Constraints** (most common):
   g(x) ≤ 0

   Examples:
   - Altitude limits: h_min - h(x) ≤ 0
   - Speed limits: ||v|| - v_max ≤ 0
   - Energy reserve: E_min - E(x) ≤ 0

2. **Equality Constraints**:
   h(x) = 0

   Examples:
   - Dynamics: ẋ = f(x, u)
   - Terminal state: x(T) = x_goal

3. **Chance Constraints** (for robust planning):
   P[g(x, ξ) ≤ 0] ≥ 1 - ε

   Examples:
   - Collision avoidance under position uncertainty
   - Threat avoidance under sensor uncertainty

Constraint Handling
-------------------

For optimization, constraints are handled via:
- Penalty methods: Add large penalty for violations
- Barrier methods: Add logarithmic barrier
- Projection: Project onto feasible set
- NSGA-III: Use constraint domination

Author: Defense eVTOL Research Team
"""

from __future__ import annotations

import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from collections.abc import Callable
from enum import Enum
import logging

from .trajectory import Trajectory
from .state import State

logger = logging.getLogger(__name__)


class ConstraintType(Enum):
    """Type of constraint."""
    INEQUALITY = "inequality"  # g(x) ≤ 0
    EQUALITY = "equality"      # h(x) = 0
    BOUND = "bound"            # x_min ≤ x ≤ x_max
    CHANCE = "chance"          # P[g(x) ≤ 0] ≥ 1 - ε


class ViolationSeverity(Enum):
    """Severity of constraint violation."""
    HARD = "hard"    # Must be satisfied (infeasible otherwise)
    SOFT = "soft"    # Penalized but trajectory can proceed
    INFO = "info"    # Informational only


@dataclass
class ConstraintViolation:
    """
    Record of a constraint violation.

    Attributes:
        constraint_name: Name of violated constraint
        violation_value: Amount of violation (0 = satisfied)
        severity: How serious the violation is
        location: Where in trajectory (index or time)
        message: Human-readable description
    """
    constraint_name: str
    violation_value: float
    severity: ViolationSeverity
    location: int | float | None = None
    message: str = ""

    @property
    def is_violated(self) -> bool:
        """Check if constraint is violated."""
        return self.violation_value > 0


class Constraint(ABC):
    """
    Abstract base class for constraints.

    Constraints evaluate trajectories/states and return:
    - Constraint value (negative = satisfied, positive = violated)
    - Violation details for debugging

    Subclasses must implement:
    - evaluate(): Check constraint on trajectory
    - evaluate_state(): Check constraint on single state
    """

    def __init__(
        self,
        name: str,
        constraint_type: ConstraintType = ConstraintType.INEQUALITY,
        severity: ViolationSeverity = ViolationSeverity.HARD,
    ):
        self.name = name
        self.constraint_type = constraint_type
        self.severity = severity

    @abstractmethod
    def evaluate(self, trajectory: Trajectory) -> list[ConstraintViolation]:
        """
        Evaluate constraint on entire trajectory.

        Args:
            trajectory: Input trajectory

        Returns:
            List of violations (empty if satisfied)
        """
        pass

    def evaluate_state(self, state: State) -> ConstraintViolation:
        """
        Evaluate constraint on single state.

        Args:
            state: Input state

        Returns:
            Violation (value=0 if satisfied)
        """
        # Default: not implemented for single states
        raise NotImplementedError(f"{self.name} doesn't support state evaluation")

    def get_margin(self, trajectory: Trajectory) -> float:
        """
        Get constraint margin (how far from violation).

        Args:
            trajectory: Input trajectory

        Returns:
            Margin (positive = satisfied, negative = violated)
        """
        violations = self.evaluate(trajectory)
        if not violations:
            return float('inf')  # No violations = infinite margin

        # Return minimum margin (worst violation)
        return -max(v.violation_value for v in violations)


# ============================================================================
# Kinodynamic Constraints
# ============================================================================

class MaxSpeedConstraint(Constraint):
    """
    Maximum speed constraint: ||v|| ≤ v_max
    """

    def __init__(
        self,
        max_speed: float,
        severity: ViolationSeverity = ViolationSeverity.HARD,
    ):
        super().__init__("max_speed", ConstraintType.INEQUALITY, severity)
        self.max_speed = max_speed

    def evaluate(self, trajectory: Trajectory) -> list[ConstraintViolation]:
        violations = []

        for i, segment in enumerate(trajectory.segments):
            speed = segment.average_speed
            if speed > self.max_speed:
                violations.append(ConstraintViolation(
                    constraint_name=self.name,
                    violation_value=speed - self.max_speed,
                    severity=self.severity,
                    location=i,
                    message=f"Speed {speed:.1f} m/s exceeds max {self.max_speed} m/s",
                ))

        return violations

    def evaluate_state(self, state: State) -> ConstraintViolation:
        speed = state.velocity.speed
        violation = max(0, speed - self.max_speed)
        return ConstraintViolation(
            constraint_name=self.name,
            violation_value=violation,
            severity=self.severity,
            message=f"Speed {speed:.1f} m/s" if violation > 0 else "",
        )


class MinMaxAltitudeConstraint(Constraint):
    """
    Altitude bounds constraint: h_min ≤ h ≤ h_max
    """

    def __init__(
        self,
        min_altitude: float = 30.0,
        max_altitude: float = 3000.0,
        severity: ViolationSeverity = ViolationSeverity.HARD,
    ):
        super().__init__("altitude_bounds", ConstraintType.BOUND, severity)
        self.min_altitude = min_altitude
        self.max_altitude = max_altitude

    def evaluate(self, trajectory: Trajectory) -> list[ConstraintViolation]:
        violations = []

        for i, state in enumerate(trajectory.states):
            alt = state.pose.position[2]

            if alt < self.min_altitude:
                violations.append(ConstraintViolation(
                    constraint_name=self.name,
                    violation_value=self.min_altitude - alt,
                    severity=self.severity,
                    location=i,
                    message=f"Altitude {alt:.0f}m below minimum {self.min_altitude}m",
                ))
            elif alt > self.max_altitude:
                violations.append(ConstraintViolation(
                    constraint_name=self.name,
                    violation_value=alt - self.max_altitude,
                    severity=self.severity,
                    location=i,
                    message=f"Altitude {alt:.0f}m above maximum {self.max_altitude}m",
                ))

        return violations


class MaxClimbRateConstraint(Constraint):
    """
    Maximum climb/descent rate: |ḣ| ≤ climb_max
    """

    def __init__(
        self,
        max_climb_rate: float = 8.0,
        max_descent_rate: float | None = None,
        severity: ViolationSeverity = ViolationSeverity.HARD,
    ):
        super().__init__("climb_rate", ConstraintType.INEQUALITY, severity)
        self.max_climb_rate = max_climb_rate
        self.max_descent_rate = max_descent_rate or max_climb_rate

    def evaluate(self, trajectory: Trajectory) -> list[ConstraintViolation]:
        violations = []

        for i, segment in enumerate(trajectory.segments):
            if segment.duration <= 0:
                continue

            climb_rate = segment.altitude_change / segment.duration

            if climb_rate > self.max_climb_rate:
                violations.append(ConstraintViolation(
                    constraint_name=self.name,
                    violation_value=climb_rate - self.max_climb_rate,
                    severity=self.severity,
                    location=i,
                    message=f"Climb rate {climb_rate:.1f} m/s exceeds max {self.max_climb_rate} m/s",
                ))
            elif climb_rate < -self.max_descent_rate:
                violations.append(ConstraintViolation(
                    constraint_name=self.name,
                    violation_value=-climb_rate - self.max_descent_rate,
                    severity=self.severity,
                    location=i,
                    message=f"Descent rate {-climb_rate:.1f} m/s exceeds max {self.max_descent_rate} m/s",
                ))

        return violations


class MaxTurnRateConstraint(Constraint):
    """
    Maximum turn rate constraint: |ψ̇| ≤ turn_max
    """

    def __init__(
        self,
        max_turn_rate: float = 30.0,  # degrees/second
        severity: ViolationSeverity = ViolationSeverity.SOFT,
    ):
        super().__init__("turn_rate", ConstraintType.INEQUALITY, severity)
        self.max_turn_rate = np.radians(max_turn_rate)  # Convert to rad/s

    def evaluate(self, trajectory: Trajectory) -> list[ConstraintViolation]:
        violations = []

        for i in range(len(trajectory) - 1):
            dt = trajectory.states[i+1].time - trajectory.states[i].time
            if dt <= 0:
                continue

            # Heading change
            yaw1 = trajectory.states[i].pose.attitude[2]
            yaw2 = trajectory.states[i+1].pose.attitude[2]

            # Handle wraparound
            dyaw = yaw2 - yaw1
            while dyaw > np.pi:
                dyaw -= 2 * np.pi
            while dyaw < -np.pi:
                dyaw += 2 * np.pi

            turn_rate = abs(dyaw) / dt

            if turn_rate > self.max_turn_rate:
                violations.append(ConstraintViolation(
                    constraint_name=self.name,
                    violation_value=turn_rate - self.max_turn_rate,
                    severity=self.severity,
                    location=i,
                    message=f"Turn rate {np.degrees(turn_rate):.1f}°/s exceeds max",
                ))

        return violations


class EnergyReserveConstraint(Constraint):
    """
    Minimum energy reserve constraint: E(t) ≥ E_min for all t

    Ensures the vehicle always has enough energy to:
    1. Complete the mission
    2. Handle emergencies
    3. Have reserve margin
    """

    def __init__(
        self,
        min_reserve: float = 0.15,  # 15% reserve
        severity: ViolationSeverity = ViolationSeverity.HARD,
    ):
        super().__init__("energy_reserve", ConstraintType.INEQUALITY, severity)
        self.min_reserve = min_reserve

    def evaluate(self, trajectory: Trajectory) -> list[ConstraintViolation]:
        violations = []

        for i, state in enumerate(trajectory.states):
            if state.energy < self.min_reserve:
                violations.append(ConstraintViolation(
                    constraint_name=self.name,
                    violation_value=self.min_reserve - state.energy,
                    severity=self.severity,
                    location=i,
                    message=f"Energy {state.energy*100:.1f}% below minimum {self.min_reserve*100:.0f}%",
                ))

        return violations


# ============================================================================
# Spatial Constraints
# ============================================================================

class TerrainClearanceConstraint(Constraint):
    """
    Terrain clearance constraint: h(x) - terrain(x, y) ≥ clearance

    Requires terrain elevation data.
    """

    def __init__(
        self,
        terrain_field: Callable[[float, float], float],  # (x, y) -> elevation
        min_clearance: float = 50.0,
        severity: ViolationSeverity = ViolationSeverity.HARD,
    ):
        super().__init__("terrain_clearance", ConstraintType.INEQUALITY, severity)
        self.terrain_field = terrain_field
        self.min_clearance = min_clearance

    def evaluate(self, trajectory: Trajectory) -> list[ConstraintViolation]:
        violations = []

        for i, state in enumerate(trajectory.states):
            pos = state.pose.position
            terrain_elev = self.terrain_field(pos[0], pos[1])
            clearance = pos[2] - terrain_elev

            if clearance < self.min_clearance:
                violations.append(ConstraintViolation(
                    constraint_name=self.name,
                    violation_value=self.min_clearance - clearance,
                    severity=self.severity,
                    location=i,
                    message=f"Clearance {clearance:.0f}m below minimum {self.min_clearance}m",
                ))

        return violations


class NoFlyZoneConstraint(Constraint):
    """
    No-fly zone avoidance: position must not enter restricted areas.

    Supports:
    - Cylindrical zones (circular with altitude range)
    - Polygonal zones
    """

    def __init__(
        self,
        zones: list[dict],  # [{center, radius, min_alt, max_alt, name}, ...]
        severity: ViolationSeverity = ViolationSeverity.HARD,
    ):
        super().__init__("no_fly_zone", ConstraintType.INEQUALITY, severity)
        self.zones = zones

    def evaluate(self, trajectory: Trajectory) -> list[ConstraintViolation]:
        violations = []

        for i, state in enumerate(trajectory.states):
            pos = state.pose.position

            for zone in self.zones:
                center = np.array(zone.get("center", [0, 0]))
                radius = zone.get("radius", 1000)
                min_alt = zone.get("min_alt", 0)
                max_alt = zone.get("max_alt", float('inf'))
                zone_name = zone.get("name", "unknown")

                # Check altitude
                if pos[2] < min_alt or pos[2] > max_alt:
                    continue  # Outside altitude range

                # Check horizontal distance
                dist = np.linalg.norm(pos[:2] - center[:2])
                if dist < radius:
                    violations.append(ConstraintViolation(
                        constraint_name=self.name,
                        violation_value=radius - dist,
                        severity=self.severity,
                        location=i,
                        message=f"Entered no-fly zone '{zone_name}' (penetration: {radius-dist:.0f}m)",
                    ))

        return violations


class ObstacleAvoidanceConstraint(Constraint):
    """
    Obstacle avoidance: maintain minimum distance from obstacles.

    Uses signed distance field (SDF) or explicit obstacle list.
    """

    def __init__(
        self,
        obstacle_field: Callable[[np.ndarray], float] | None = None,
        obstacles: list[dict] | None = None,  # [{position, radius}, ...]
        min_separation: float = 50.0,
        severity: ViolationSeverity = ViolationSeverity.HARD,
    ):
        super().__init__("obstacle_avoidance", ConstraintType.INEQUALITY, severity)
        self.obstacle_field = obstacle_field
        self.obstacles = obstacles or []
        self.min_separation = min_separation

    def _distance_to_obstacles(self, pos: np.ndarray) -> float:
        """Compute minimum distance to any obstacle."""
        if self.obstacle_field is not None:
            return self.obstacle_field(pos)

        if not self.obstacles:
            return float('inf')

        min_dist = float('inf')
        for obs in self.obstacles:
            obs_pos = np.array(obs["position"])
            obs_radius = obs.get("radius", 0)
            dist = np.linalg.norm(pos - obs_pos) - obs_radius
            min_dist = min(min_dist, dist)

        return min_dist

    def evaluate(self, trajectory: Trajectory) -> list[ConstraintViolation]:
        violations = []

        for i, state in enumerate(trajectory.states):
            dist = self._distance_to_obstacles(state.pose.position)

            if dist < self.min_separation:
                violations.append(ConstraintViolation(
                    constraint_name=self.name,
                    violation_value=self.min_separation - dist,
                    severity=self.severity,
                    location=i,
                    message=f"Obstacle clearance {dist:.0f}m below minimum {self.min_separation}m",
                ))

        return violations


# ============================================================================
# Threat Constraints
# ============================================================================

class ThreatAvoidanceConstraint(Constraint):
    """
    Threat avoidance: maintain minimum range from threat systems.

    Unlike cost (which minimizes exposure), this is a hard constraint
    on minimum safe distance.
    """

    def __init__(
        self,
        threats: list[dict],  # [{position, effective_range, type}, ...]
        safety_factor: float = 1.2,  # Multiple of effective range
        severity: ViolationSeverity = ViolationSeverity.HARD,
    ):
        super().__init__("threat_avoidance", ConstraintType.INEQUALITY, severity)
        self.threats = threats
        self.safety_factor = safety_factor

    def evaluate(self, trajectory: Trajectory) -> list[ConstraintViolation]:
        violations = []

        for i, state in enumerate(trajectory.states):
            pos = state.pose.position

            for threat in self.threats:
                threat_pos = np.array(threat["position"])
                eff_range = threat.get("effective_range", 5000)
                safe_range = eff_range * self.safety_factor

                dist = np.linalg.norm(pos - threat_pos)

                if dist < safe_range:
                    violations.append(ConstraintViolation(
                        constraint_name=self.name,
                        violation_value=safe_range - dist,
                        severity=self.severity,
                        location=i,
                        message=f"Within threat range ({dist:.0f}m < {safe_range:.0f}m)",
                    ))

        return violations


class MaxThreatExposureConstraint(Constraint):
    """
    Maximum cumulative threat exposure constraint.

    Limits total time spent in high-threat areas.
    """

    def __init__(
        self,
        threat_field: Callable[[np.ndarray], float],
        max_exposure: float = 10.0,  # seconds at threat level 1.0
        severity: ViolationSeverity = ViolationSeverity.SOFT,
    ):
        super().__init__("max_threat_exposure", ConstraintType.INEQUALITY, severity)
        self.threat_field = threat_field
        self.max_exposure = max_exposure

    def evaluate(self, trajectory: Trajectory) -> list[ConstraintViolation]:
        total_exposure = 0.0

        for segment in trajectory.segments:
            mid_pos = (segment.start.pose.position + segment.end.pose.position) / 2
            threat_level = self.threat_field(mid_pos)
            total_exposure += threat_level * segment.duration

        if total_exposure > self.max_exposure:
            return [ConstraintViolation(
                constraint_name=self.name,
                violation_value=total_exposure - self.max_exposure,
                severity=self.severity,
                message=f"Threat exposure {total_exposure:.1f}s exceeds max {self.max_exposure}s",
            )]

        return []


# ============================================================================
# Composite Constraint Set
# ============================================================================

@dataclass
class ConstraintSet:
    """
    Collection of constraints for trajectory optimization.

    Provides:
    - Batch evaluation of all constraints
    - Aggregated violation metrics
    - Feasibility checking

    Attributes:
        constraints: List of constraints
        name: Constraint set name
    """
    constraints: list[Constraint] = field(default_factory=list)
    name: str = "default"

    def add_constraint(self, constraint: Constraint) -> None:
        """Add a constraint to the set."""
        self.constraints.append(constraint)

    def evaluate(self, trajectory: Trajectory) -> list[ConstraintViolation]:
        """Evaluate all constraints and return all violations."""
        all_violations = []
        for constraint in self.constraints:
            violations = constraint.evaluate(trajectory)
            all_violations.extend(violations)
        return all_violations

    def is_feasible(self, trajectory: Trajectory) -> bool:
        """Check if trajectory satisfies all hard constraints."""
        for constraint in self.constraints:
            if constraint.severity == ViolationSeverity.HARD:
                violations = constraint.evaluate(trajectory)
                if any(v.is_violated for v in violations):
                    return False
        return True

    def get_violation_summary(self, trajectory: Trajectory) -> dict[str, float]:
        """Get summary of violation amounts by constraint."""
        summary = {}
        for constraint in self.constraints:
            violations = constraint.evaluate(trajectory)
            if violations:
                summary[constraint.name] = max(v.violation_value for v in violations)
            else:
                summary[constraint.name] = 0.0
        return summary

    def total_violation(self, trajectory: Trajectory) -> float:
        """Total constraint violation (for optimization)."""
        total = 0.0
        for violation in self.evaluate(trajectory):
            if violation.severity == ViolationSeverity.HARD:
                total += violation.violation_value * 100  # Heavy penalty
            elif violation.severity == ViolationSeverity.SOFT:
                total += violation.violation_value
        return total


def create_default_constraints() -> ConstraintSet:
    """
    Create default constraint set for defense eVTOL operations.

    Returns:
        ConstraintSet with standard kinodynamic and safety constraints
    """
    cs = ConstraintSet(name="defense_evtol_default")

    # Kinodynamic constraints
    cs.add_constraint(MaxSpeedConstraint(max_speed=60.0))
    cs.add_constraint(MinMaxAltitudeConstraint(min_altitude=30.0, max_altitude=3000.0))
    cs.add_constraint(MaxClimbRateConstraint(max_climb_rate=8.0))
    cs.add_constraint(MaxTurnRateConstraint(max_turn_rate=30.0))
    cs.add_constraint(EnergyReserveConstraint(min_reserve=0.15))

    return cs
