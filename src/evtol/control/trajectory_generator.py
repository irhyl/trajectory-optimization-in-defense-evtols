
"""
Trajectory Generation Module for eVTOL Control Layer
---------------------------------------------------
This module provides classes and methods to generate smooth, time-parameterized trajectories
for electric vertical takeoff and landing (eVTOL) vehicles. It supports trajectory generation
from a sequence of waypoints, ensuring that the resulting path respects vehicle velocity and
acceleration constraints. The generated trajectory includes position, velocity, and acceleration
profiles suitable for use by flight controllers.

Key Classes:
    - TrajectorySegment: Represents a single segment between two waypoints.
    - TrajectoryGenerator: Generates the full trajectory through all waypoints.

Example:
    generator = TrajectoryGenerator()
    waypoints = [np.array([0, 0, 100]), np.array([100, 50, 150]), ...]
    trajectory = generator.generate_trajectory(waypoints)
    # trajectory is a list of dicts with time, position, velocity, acceleration
"""

import numpy as np
import math
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class TrajectorySegment:
    """
    Represents a single segment of a trajectory between two waypoints.

    Attributes:
        start_time (float): Start time of the segment (seconds).
        end_time (float): End time of the segment (seconds).
        start_position (np.ndarray): Starting position [x, y, z].
        end_position (np.ndarray): Ending position [x, y, z].
        start_velocity (np.ndarray): Starting velocity vector.
        end_velocity (np.ndarray): Ending velocity vector.
        max_velocity (float): Maximum allowed velocity for the segment.
        max_acceleration (float): Maximum allowed acceleration for the segment.
    """
    start_time: float
    end_time: float
    start_position: np.ndarray
    end_position: np.ndarray
    start_velocity: np.ndarray
    end_velocity: np.ndarray
    max_velocity: float = 35.0
    max_acceleration: float = 3.0


class TrajectoryGenerator:
    """
    Generates time-parameterized, smooth trajectories through a sequence of waypoints.

    The generated trajectory is suitable for eVTOL flight controllers and includes
    position, velocity, and acceleration at each time step.
    """

    def __init__(
        self,
        max_velocity: float = 35.0,
        max_acceleration: float = 3.0,
        dt: float = 0.01
    ):
        """
        Initialize the TrajectoryGenerator.

        Args:
            max_velocity (float): Maximum velocity allowed (m/s).
            max_acceleration (float): Maximum acceleration allowed (m/s^2).
            dt (float): Time discretization step (seconds).
        """
        self.max_velocity = max_velocity
        self.max_acceleration = max_acceleration
        self.dt = dt

    def generate_trajectory(
        self,
        waypoints: List[np.ndarray],
        initial_velocity: Optional[np.ndarray] = None,
        final_velocity: Optional[np.ndarray] = None
    ) -> List[Dict]:
        """
        Generate a smooth trajectory through the provided waypoints.

        Args:
            waypoints (List[np.ndarray]): List of 3D position waypoints [x, y, z].
            initial_velocity (Optional[np.ndarray]): Initial velocity vector (default: hover).
            final_velocity (Optional[np.ndarray]): Final velocity vector (default: hover).

        Returns:
            List[Dict]: List of trajectory points, each a dict with keys:
                - 'time': float
                - 'position': np.ndarray
                - 'velocity': np.ndarray
                - 'acceleration': np.ndarray

        Raises:
            ValueError: If fewer than 2 waypoints are provided.
        """
        if len(waypoints) < 2:
            raise ValueError("At least two waypoints are required to generate a trajectory.")

        if initial_velocity is None:
            initial_velocity = np.zeros(3)
        if final_velocity is None:
            final_velocity = np.zeros(3)

        # Step 1: Compute time-optimal segments between waypoints
        segments = self._compute_segments(
            waypoints, initial_velocity, final_velocity
        )

        # Step 2: Sample the trajectory at regular intervals
        trajectory = self._sample_trajectory(segments)

        return trajectory

    def _compute_segments(
        self,
        waypoints: List[np.ndarray],
        initial_vel: np.ndarray,
        final_vel: np.ndarray
    ) -> List[TrajectorySegment]:
        """
        Compute trajectory segments between each pair of waypoints.

        Args:
            waypoints (List[np.ndarray]): List of 3D waypoints.
            initial_vel (np.ndarray): Initial velocity vector.
            final_vel (np.ndarray): Final velocity vector.

        Returns:
            List[TrajectorySegment]: List of trajectory segments.
        """
        segments = []
        current_time = 0.0
        current_vel = initial_vel.copy()

        for i in range(len(waypoints) - 1):
            start_pos = waypoints[i]
            end_pos = waypoints[i + 1]

            # Determine the target velocity for this segment
            if i == len(waypoints) - 2:
                target_vel = final_vel
            else:
                # Cruise velocity toward next waypoint
                direction = end_pos - start_pos
                distance = np.linalg.norm(direction)
                if distance > 0:
                    direction = direction / distance
                    target_vel = direction * min(self.max_velocity, distance / 2.0)
                else:
                    target_vel = np.zeros(3)

            # Compute segment duration using average velocity
            distance = np.linalg.norm(end_pos - start_pos)
            avg_velocity = np.linalg.norm(current_vel + target_vel) / 2
            if avg_velocity > 0:
                duration = distance / max(avg_velocity, 1.0)
            else:
                duration = distance / self.max_velocity

            segment = TrajectorySegment(
                start_time=current_time,
                end_time=current_time + duration,
                start_position=start_pos,
                end_position=end_pos,
                start_velocity=current_vel,
                end_velocity=target_vel,
                max_velocity=self.max_velocity,
                max_acceleration=self.max_acceleration
            )

            segments.append(segment)
            current_time += duration
            current_vel = target_vel

        return segments

    def _sample_trajectory(
        self,
        segments: List[TrajectorySegment]
    ) -> List[Dict]:
        """
        Sample the full trajectory at regular time intervals using cubic interpolation.

        Args:
            segments (List[TrajectorySegment]): List of trajectory segments.

        Returns:
            List[Dict]: List of trajectory points with time, position, velocity, acceleration.
        """
        trajectory = []

        for segment in segments:
            t_start = segment.start_time
            t_end = segment.end_time
            times = np.arange(t_start, t_end, self.dt)

            for t in times:
                # Normalized time within segment [0, 1]
                tau = (t - t_start) / (t_end - t_start)
                tau = np.clip(tau, 0.0, 1.0)

                # Cubic Hermite interpolation for smooth position and velocity
                h1 = 2 * tau ** 3 - 3 * tau ** 2 + 1
                h2 = tau ** 3 - 2 * tau ** 2 + tau
                h3 = -2 * tau ** 3 + 3 * tau ** 2
                h4 = tau ** 3 - tau ** 2

                duration = t_end - t_start

                position = (
                    h1 * segment.start_position +
                    h2 * duration * segment.start_velocity +
                    h3 * segment.end_position +
                    h4 * duration * segment.end_velocity
                )

                # Velocity (first derivative of position)
                h1_dot = (6 * tau ** 2 - 6 * tau) / duration
                h2_dot = (3 * tau ** 2 - 4 * tau + 1)
                h3_dot = (-6 * tau ** 2 + 6 * tau) / duration
                h4_dot = (3 * tau ** 2 - 2 * tau)

                velocity = (
                    h1_dot * segment.start_position +
                    h2_dot * segment.start_velocity +
                    h3_dot * segment.end_position +
                    h4_dot * segment.end_velocity
                )

                # Acceleration (second derivative of position)
                h1_ddot = (12 * tau - 6) / duration ** 2
                h2_ddot = (6 * tau - 4) / duration
                h3_ddot = (-12 * tau + 6) / duration ** 2
                h4_ddot = (6 * tau - 2) / duration

                acceleration = (
                    h1_ddot * segment.start_position +
                    h2_ddot * segment.start_velocity +
                    h3_ddot * segment.end_position +
                    h4_ddot * segment.end_velocity
                )

                trajectory.append({
                    'time': float(t),
                    'position': position,
                    'velocity': velocity,
                    'acceleration': acceleration
                })

        return trajectory


if __name__ == "__main__":
    # Example usage and demonstration
    logging.basicConfig(level=logging.INFO)

    generator = TrajectoryGenerator()

    # Define a set of 3D waypoints for the trajectory
    waypoints = [
        np.array([0.0, 0.0, 100.0]),
        np.array([100.0, 50.0, 150.0]),
        np.array([200.0, 100.0, 120.0]),
        np.array([300.0, 50.0, 100.0])
    ]

    # Generate the trajectory
    trajectory = generator.generate_trajectory(waypoints)

    print(f"Generated trajectory: {len(trajectory)} points")
    print(f"Duration: {trajectory[-1]['time']:.2f} seconds")
    print(f"\nSample points:")
    for i in [0, len(trajectory)//2, -1]:
        point = trajectory[i]
        print(f"  t={point['time']:.2f}s: pos={point['position']}, vel_mag={np.linalg.norm(point['velocity']):.2f}m/s")

