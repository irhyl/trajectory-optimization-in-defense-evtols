"""
Objective Functions for Multi-Objective Optimization

This module defines objective functions used in trajectory optimization.
Each objective is a callable that maps a trajectory to a scalar value
to be minimized.

Design Principles:
- Objectives are composable and modular
- Each objective has a clear physical interpretation
- Objectives support gradient computation where possible
- Normalization is handled for fair comparison

Author: Defense eVTOL Research Team
"""

from __future__ import annotations

import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from collections.abc import Callable
import logging

from ..core.trajectory import Trajectory

logger = logging.getLogger(__name__)


class ObjectiveFunction(ABC):
    """
    Abstract base class for objective functions.

    Objectives map trajectories to scalar values to be minimized.

    Attributes:
        name: Objective identifier
        weight: Weight for weighted-sum scalarization
        bounds: Expected (min, max) for normalization
    """

    def __init__(
        self,
        name: str,
        weight: float = 1.0,
        bounds: tuple[float, float] | None = None,
    ):
        self.name = name
        self.weight = weight
        self.bounds = bounds

    @abstractmethod
    def evaluate(self, trajectory: Trajectory) -> float:
        """
        Evaluate objective on trajectory.

        Args:
            trajectory: Input trajectory

        Returns:
            Objective value (to be minimized)
        """
        pass

    def __call__(self, trajectory: Trajectory) -> float:
        """Callable interface."""
        return self.evaluate(trajectory)

    def normalize(self, value: float) -> float:
        """
        Normalize value to [0, 1] range.

        Args:
            value: Raw objective value

        Returns:
            Normalized value
        """
        if self.bounds is None:
            return value

        min_val, max_val = self.bounds
        if max_val <= min_val:
            return 0.0

        normalized = (value - min_val) / (max_val - min_val)
        return np.clip(normalized, 0.0, 1.0)


class TimeObjective(ObjectiveFunction):
    """
    Minimize mission duration.

    J_time = T_final - T_initial
    """

    def __init__(self, weight: float = 1.0, bounds: tuple[float, float] | None = None):
        super().__init__("time", weight, bounds or (60.0, 3600.0))

    def evaluate(self, trajectory: Trajectory) -> float:
        return trajectory.duration


class EnergyObjective(ObjectiveFunction):
    """
    Minimize energy consumption.

    J_energy = E_initial - E_final (as fraction of battery)

    Or more detailed:
    J_energy = ∫ P(v, climb_rate) dt
    """

    def __init__(
        self,
        weight: float = 1.0,
        bounds: tuple[float, float] | None = None,
        power_model: Callable | None = None,
    ):
        super().__init__("energy", weight, bounds or (0.0, 1.0))
        self.power_model = power_model

    def evaluate(self, trajectory: Trajectory) -> float:
        if self.power_model is None:
            # Simple model: energy = distance × specific consumption
            return trajectory.energy_consumed

        # Detailed model: integrate power along trajectory
        total_energy = 0.0
        for segment in trajectory.segments:
            speed = segment.average_speed
            climb_rate = segment.altitude_change / segment.duration if segment.duration > 0 else 0
            power = self.power_model(speed, climb_rate)
            total_energy += power * segment.duration

        return total_energy


class PathLengthObjective(ObjectiveFunction):
    """
    Minimize total path length.

    J_length = ∫ ||v|| dt = total distance
    """

    def __init__(self, weight: float = 1.0, bounds: tuple[float, float] | None = None):
        super().__init__("path_length", weight, bounds or (0.0, 500000.0))

    def evaluate(self, trajectory: Trajectory) -> float:
        return trajectory.distance


class ThreatExposureObjective(ObjectiveFunction):
    """
    Minimize integrated threat exposure.

    J_threat = ∫ P_kill(x(t)) dt

    This measures cumulative exposure to threats along the path.
    """

    def __init__(
        self,
        threat_field: Callable[[float, float, float], float],
        weight: float = 1.0,
        bounds: tuple[float, float] | None = None,
    ):
        """
        Args:
            threat_field: Function (lat, lon, alt) -> threat probability
        """
        super().__init__("threat_exposure", weight, bounds or (0.0, 100.0))
        self.threat_field = threat_field

    def evaluate(self, trajectory: Trajectory) -> float:
        total_exposure = 0.0

        for segment in trajectory.segments:
            # Sample at segment midpoint
            mid_pos = (segment.start.pose.position + segment.end.pose.position) / 2
            threat = self.threat_field(mid_pos[0], mid_pos[1], mid_pos[2])

            # Integrate: probability × time
            total_exposure += threat * segment.duration

        return total_exposure


class DetectionObjective(ObjectiveFunction):
    """
    Minimize radar detection probability.

    J_detect = ∫ P_detect(x(t), heading(t)) dt

    Detection depends on:
    - Range to radars
    - RCS (aspect-dependent)
    - Terrain masking
    """

    def __init__(
        self,
        radar_positions: np.ndarray,
        rcs_model: Callable[[float], float] | None = None,
        weight: float = 1.0,
        bounds: tuple[float, float] | None = None,
    ):
        """
        Args:
            radar_positions: Nx3 array of radar positions
            rcs_model: Function (aspect_angle) -> RCS in m²
        """
        super().__init__("detection", weight, bounds or (0.0, 1.0))
        self.radar_positions = np.asarray(radar_positions)
        self.rcs_model = rcs_model or (lambda a: 0.5)  # Default 0.5 m²

    def evaluate(self, trajectory: Trajectory) -> float:
        total_detection = 0.0

        for segment in trajectory.segments:
            pos = (segment.start.pose.position + segment.end.pose.position) / 2

            # Heading
            direction = segment.end.pose.position - segment.start.pose.position
            heading = np.arctan2(direction[1], direction[0]) if np.linalg.norm(direction[:2]) > 0 else 0

            segment_detection = 0.0

            for radar_pos in self.radar_positions:
                # Range
                # Convert to approximate meters (for geodetic coordinates)
                dx = (pos[1] - radar_pos[1]) * 111320 * np.cos(np.radians(pos[0]))
                dy = (pos[0] - radar_pos[0]) * 110540
                dz = pos[2] - radar_pos[2]
                R = np.sqrt(dx**2 + dy**2 + dz**2)

                if R < 100:
                    R = 100  # Avoid singularity

                # Aspect angle
                to_radar = np.array([radar_pos[0] - pos[0], radar_pos[1] - pos[1]])
                if np.linalg.norm(to_radar) > 0:
                    radar_bearing = np.arctan2(to_radar[1], to_radar[0])
                    aspect = abs(heading - radar_bearing)
                else:
                    aspect = 0

                # RCS
                rcs = self.rcs_model(aspect)

                # Simplified radar equation (detection metric)
                detection = (rcs / (R ** 4)) * 1e12  # Scale factor
                segment_detection += detection

            total_detection += segment_detection * segment.duration

        return min(1.0, total_detection)


class SmoothnessObjective(ObjectiveFunction):
    """
    Minimize path jerkiness.

    J_smooth = Σ ||a_{i+1} - a_i||² (acceleration variation)

    Smooth paths are:
    - More efficient to track
    - More comfortable
    - Better for payload protection
    """

    def __init__(self, weight: float = 1.0, bounds: tuple[float, float] | None = None):
        super().__init__("smoothness", weight, bounds or (0.0, 1000.0))

    def evaluate(self, trajectory: Trajectory) -> float:
        if len(trajectory) < 3:
            return 0.0

        positions = trajectory.get_positions()
        times = trajectory.get_times()

        # Compute velocities
        dt = np.diff(times)
        dt = np.where(dt > 0, dt, 1.0)
        velocities = np.diff(positions, axis=0) / dt[:, np.newaxis]

        # Compute accelerations
        dt2 = (dt[:-1] + dt[1:]) / 2
        dt2 = np.where(dt2 > 0, dt2, 1.0)
        accelerations = np.diff(velocities, axis=0) / dt2[:, np.newaxis]

        # Sum squared acceleration changes (jerk metric)
        if len(accelerations) < 2:
            return 0.0

        jerk = np.diff(accelerations, axis=0)
        return float(np.sum(jerk ** 2))


class TerrainClearanceObjective(ObjectiveFunction):
    """
    Maximize (or enforce) terrain clearance.

    This is often a constraint, but can be an objective when
    trading off clearance against other factors.

    J = -min(clearance) (negative because we minimize)
    """

    def __init__(
        self,
        terrain_field: Callable[[float, float], float],
        min_clearance: float = 50.0,
        weight: float = 1.0,
    ):
        super().__init__("terrain_clearance", weight, (0.0, 1000.0))
        self.terrain_field = terrain_field
        self.min_clearance = min_clearance

    def evaluate(self, trajectory: Trajectory) -> float:
        min_clearance = float('inf')

        for state in trajectory.states:
            pos = state.pose.position
            terrain_elev = self.terrain_field(pos[0], pos[1])
            clearance = pos[2] - terrain_elev
            min_clearance = min(min_clearance, clearance)

        # Return negative of clearance (we want to maximize clearance)
        # Clip to avoid extreme values
        return max(0, self.min_clearance - min_clearance)


@dataclass
class ObjectiveSet:
    """
    Collection of objectives for multi-objective optimization.

    Provides:
    - Vector evaluation (for Pareto optimization)
    - Weighted sum (for single-objective scalarization)
    - Objective statistics

    Attributes:
        objectives: List of objective functions
    """
    objectives: list[ObjectiveFunction] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.objectives)

    def __iter__(self):
        return iter(self.objectives)

    def add(self, objective: ObjectiveFunction) -> None:
        """Add an objective."""
        self.objectives.append(objective)

    def evaluate_vector(self, trajectory: Trajectory) -> np.ndarray:
        """
        Evaluate all objectives and return as vector.

        Args:
            trajectory: Input trajectory

        Returns:
            Array of objective values
        """
        return np.array([obj.evaluate(trajectory) for obj in self.objectives])

    def evaluate_normalized(self, trajectory: Trajectory) -> np.ndarray:
        """
        Evaluate all objectives normalized to [0, 1].

        Args:
            trajectory: Input trajectory

        Returns:
            Array of normalized objective values
        """
        raw = self.evaluate_vector(trajectory)
        return np.array([
            self.objectives[i].normalize(raw[i])
            for i in range(len(self.objectives))
        ])

    def evaluate_weighted_sum(self, trajectory: Trajectory) -> float:
        """
        Evaluate weighted sum of objectives.

        Args:
            trajectory: Input trajectory

        Returns:
            Scalar weighted sum
        """
        total = 0.0
        for obj in self.objectives:
            value = obj.normalize(obj.evaluate(trajectory))
            total += obj.weight * value
        return total

    def evaluate_dict(self, trajectory: Trajectory) -> dict[str, float]:
        """
        Evaluate all objectives and return as dictionary.

        Args:
            trajectory: Input trajectory

        Returns:
            Dict mapping objective names to values
        """
        return {obj.name: obj.evaluate(trajectory) for obj in self.objectives}

    @property
    def names(self) -> list[str]:
        """Objective names."""
        return [obj.name for obj in self.objectives]

    @property
    def n_objectives(self) -> int:
        """Number of objectives."""
        return len(self.objectives)


def create_defense_objectives(
    threat_field: Callable | None = None,
    radar_positions: np.ndarray | None = None,
) -> ObjectiveSet:
    """
    Create standard objective set for defense eVTOL missions.

    Objectives:
    1. Time (weight=1.0)
    2. Energy (weight=0.8)
    3. Threat exposure (weight=1.5)
    4. Detection (weight=1.2)
    5. Smoothness (weight=0.3)

    Args:
        threat_field: Threat probability field
        radar_positions: Radar locations

    Returns:
        ObjectiveSet for defense missions
    """
    obj_set = ObjectiveSet()

    obj_set.add(TimeObjective(weight=1.0))
    obj_set.add(EnergyObjective(weight=0.8))
    obj_set.add(PathLengthObjective(weight=0.5))
    obj_set.add(SmoothnessObjective(weight=0.3))

    if threat_field is not None:
        obj_set.add(ThreatExposureObjective(threat_field, weight=1.5))

    if radar_positions is not None and len(radar_positions) > 0:
        obj_set.add(DetectionObjective(radar_positions, weight=1.2))

    return obj_set
