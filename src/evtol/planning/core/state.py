"""
Planning State Abstractions.

These classes represent the planning-domain state used by trajectory,
constraints, and planner algorithms. This module is intentionally lightweight
and focused on planning semantics (pose, velocity, time, energy).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class CoordinateFrame(Enum):
    """Coordinate frame for position representation."""

    NED = "ned"
    GEODETIC = "geodetic"
    ECEF = "ecef"


class AttitudeRepresentation(Enum):
    """Attitude representation convention."""

    EULER = "euler"
    QUATERNION = "quaternion"


@dataclass
class Pose:
    """Rigid-body pose for planning state."""

    position: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    attitude: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    frame: CoordinateFrame = CoordinateFrame.NED
    attitude_representation: AttitudeRepresentation = AttitudeRepresentation.EULER

    def __post_init__(self) -> None:
        self.position = np.asarray(self.position, dtype=float)
        self.attitude = np.asarray(self.attitude, dtype=float)

        if self.position.shape != (3,):
            raise ValueError(f"position must be shape (3,), got {self.position.shape}")

        if self.attitude_representation == AttitudeRepresentation.EULER and self.attitude.shape != (3,):
            raise ValueError(
                "attitude must be shape (3,) for Euler representation, "
                f"got {self.attitude.shape}"
            )

        if self.attitude_representation == AttitudeRepresentation.QUATERNION and self.attitude.shape != (4,):
            raise ValueError(
                "attitude must be shape (4,) for quaternion representation, "
                f"got {self.attitude.shape}"
            )


@dataclass
class Velocity:
    """Linear and angular velocity in body/inertial conventions."""

    linear: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    angular: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))

    def __post_init__(self) -> None:
        self.linear = np.asarray(self.linear, dtype=float)
        self.angular = np.asarray(self.angular, dtype=float)

        if self.linear.shape != (3,):
            raise ValueError(f"linear velocity must be shape (3,), got {self.linear.shape}")

        if self.angular.shape != (3,):
            raise ValueError(f"angular velocity must be shape (3,), got {self.angular.shape}")

    @property
    def speed(self) -> float:
        """Return magnitude of linear velocity."""
        return float(np.linalg.norm(self.linear))


@dataclass
class State:
    """Planning state on pose-velocity-time-energy manifold."""

    pose: Pose = field(default_factory=Pose)
    velocity: Velocity = field(default_factory=Velocity)
    energy: float = 1.0
    time: float = 0.0

    def __post_init__(self) -> None:
        self.energy = float(np.clip(self.energy, 0.0, 1.0))
        self.time = float(self.time)

    def distance_to(self, other: State) -> float:
        """Euclidean distance in position space."""
        return float(np.linalg.norm(self.pose.position - other.pose.position))

    def interpolate(self, other: State, alpha: float) -> State:
        """Linear interpolation between two states."""
        a = float(np.clip(alpha, 0.0, 1.0))

        pos = (1.0 - a) * self.pose.position + a * other.pose.position
        att = (1.0 - a) * self.pose.attitude + a * other.pose.attitude
        lin = (1.0 - a) * self.velocity.linear + a * other.velocity.linear
        ang = (1.0 - a) * self.velocity.angular + a * other.velocity.angular

        return State(
            pose=Pose(
                position=pos,
                attitude=att,
                frame=self.pose.frame,
                attitude_representation=self.pose.attitude_representation,
            ),
            velocity=Velocity(linear=lin, angular=ang),
            energy=(1.0 - a) * self.energy + a * other.energy,
            time=(1.0 - a) * self.time + a * other.time,
        )
