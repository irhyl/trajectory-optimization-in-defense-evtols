"""
Mission Loader - Load and parse mission files.

This module provides utilities to load mission waypoints from JSON files
and generate trajectories for the flight controller to track.

Supported mission formats:
- Waypoint-based missions (list of 3D points with timing)
- Segment-based missions (pre-computed trajectory segments)

A mission consists of:
1. Takeoff phase (static to hover)
2. Transition phase (hover to cruise if applicable)
3. Cruise/transit legs (waypoint-following)
4. Return/descent phase
5. Landing phase (cruise to static)

Author: Defense eVTOL Research Team
"""

from __future__ import annotations

import json
import numpy as np
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any
import logging

from evtol.control.guidance.trajectory_tracker import TrajectoryPoint
from evtol.planning.core.trajectory import TrajectorySegment, Trajectory
from evtol.planning.core.state import State, Pose, Velocity, CoordinateFrame

logger = logging.getLogger(__name__)


@dataclass
class Waypoint:
    """A single mission waypoint."""
    north: float        # m (NED)
    east: float         # m (NED)
    altitude: float     # m (positive = above ground)
    time: float | None = None  # seconds (optional - will be auto-computed)
    heading: float = 0.0        # radians (0 = north)
    speed: float = 10.0         # m/s (target speed for this waypoint)
    nacelle_angle: float | None = None  # radians (auto if None)
    hold_time: float = 0.0  # hold time at waypoint in seconds


@dataclass
class MissionConfig:
    """Mission configuration parameters."""
    vehicle_mass: float = 2500.0          # kg
    max_rotor_thrust: float = 50000.0     # N (per rotor)
    num_rotors: int = 4                   # number of rotors
    cruise_speed: float = 25.0             # m/s
    hover_altitude: float = 10.0           # m above ground
    transition_duration: float = 5.0       # seconds
    climb_rate: float = 3.0                # m/s
    descent_rate: float = 2.0              # m/s
    home_position: np.ndarray = None       # [north, east, altitude] in NED


class MissionLoader:
    """Load and parse mission files."""

    def __init__(self, config: MissionConfig | None = None):
        self.config = config or MissionConfig()
        if self.config.home_position is None:
            self.config.home_position = np.array([0.0, 0.0, 0.0])

    def load_mission_json(self, filepath: Path | str) -> list[Waypoint]:
        """
        Load mission waypoints from JSON file.

        JSON format:
        {
            "mission_name": "example_mission",
            "mission_config": {
                "cruise_speed": 25.0,
                "hover_altitude": 10.0
            },
            "waypoints": [
                {
                    "north": 0,
                    "east": 0,
                    "altitude": 10,
                    "heading": 0,
                    "speed": 25,
                    "hold_time": 5
                },
                ...
            ]
        }
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Mission file not found: {filepath}")

        with open(filepath, "r") as f:
            data = json.load(f)

        # Update config if provided
        if "mission_config" in data:
            config = data["mission_config"]
            for key, value in config.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)

        # Parse waypoints
        waypoints = []
        for wp_data in data.get("waypoints", []):
            wp = Waypoint(
                north=wp_data["north"],
                east=wp_data["east"],
                altitude=wp_data.get("altitude", 10.0),
                time=wp_data.get("time"),
                heading=wp_data.get("heading", 0.0),
                speed=wp_data.get("speed", self.config.cruise_speed),
                nacelle_angle=wp_data.get("nacelle_angle"),
                hold_time=wp_data.get("hold_time", 0.0),
            )
            waypoints.append(wp)

        logger.info(f"Loaded mission with {len(waypoints)} waypoints from {filepath}")
        return waypoints

    def generate_trajectory_from_waypoints(
        self,
        waypoints: list[Waypoint],
        include_takeoff: bool = True,
        include_landing: bool = True,
    ) -> list[TrajectoryPoint]:
        """
        Generate trajectory points from waypoints.

        This interpolates between waypoints with smooth acceleration profiles
        and automatic timing calculations.

        Args:
            waypoints: List of waypoints
            include_takeoff: Add takeoff phase before first waypoint
            include_landing: Add landing phase after last waypoint

        Returns:
            List of TrajectoryPoint objects for trajectory tracking
        """

        trajectory_points = []
        current_time = 0.0

        # Phase 1: Takeoff (static → hover at home position)
        if include_takeoff:
            trajectory_points.extend(
                self._generate_takeoff_phase(current_time)
            )
            current_time = trajectory_points[-1].t

        # Phase 2: Waypoint-to-waypoint navigation
        for i, wp in enumerate(waypoints):
            if i == 0:
                # First waypoint - transition from hover if needed
                traj_pts = self._generate_transition_to_waypoint(
                    trajectory_points[-1] if trajectory_points else None,
                    wp,
                    current_time,
                )
            else:
                # Subsequent waypoints
                prev_wp = waypoints[i - 1]
                traj_pts = self._generate_waypoint_leg(prev_wp, wp, current_time)

            trajectory_points.extend(traj_pts)
            current_time = trajectory_points[-1].t

        # Phase 3: Landing (hover at home → static)
        if include_landing:
            landing_pts = self._generate_landing_phase(current_time)
            trajectory_points.extend(landing_pts)

        logger.info(
            f"Generated trajectory with {len(trajectory_points)} points "
            f"over {current_time:.1f} seconds"
        )

        return trajectory_points

    def _generate_takeoff_phase(
        self, start_time: float, num_points: int = 20
    ) -> list[TrajectoryPoint]:
        """Generate takeoff phase: static → hover at home_position."""

        home = self.config.home_position
        hover_alt = home[2] - self.config.hover_altitude  # NED frame

        points = []
        duration = 5.0  # 5 second takeoff

        for i in range(num_points):
            t = start_time + (i / (num_points - 1)) * duration
            # Smooth ramp up using cubic easing
            progress = i / (num_points - 1)
            easing = progress**2 * (3 - 2 * progress)  # Smoothstep

            z = home[2] + easing * (hover_alt - home[2])

            points.append(
                TrajectoryPoint(
                    t=t,
                    x=home[0],
                    y=home[1],
                    z=z,
                    vx=0.0,
                    vy=0.0,
                    vz=(hover_alt - home[2]) / duration * progress,
                    ax=0.0,
                    ay=0.0,
                    az=0.0,
                    heading=0.0,
                    nacelle_angle=np.radians(90),  # Hover mode
                )
            )

        return points

    def _generate_transition_to_waypoint(
        self,
        current_point: TrajectoryPoint | None,
        waypoint: Waypoint,
        start_time: float,
        num_points: int = 30,
    ) -> list[TrajectoryPoint]:
        """Transition from current position to first waypoint."""

        if current_point is None:
            # Start from home at hover altitude
            start_pos = np.array(
                [
                    self.config.home_position[0],
                    self.config.home_position[1],
                    self.config.home_position[2] - self.config.hover_altitude,
                ]
            )
        else:
            start_pos = np.array([current_point.x, current_point.y, current_point.z])

        end_pos = np.array(
            [waypoint.north, waypoint.east, -waypoint.altitude]  # NED frame
        )

        distance = np.linalg.norm(end_pos - start_pos)
        speed = waypoint.speed if waypoint.speed > 0 else self.config.cruise_speed
        duration = distance / speed if speed > 0 else 5.0

        points = []
        for i in range(num_points):
            t = start_time + (i / (num_points - 1)) * duration
            progress = i / (num_points - 1)

            # Cubic easing for smooth acceleration/deceleration
            easing = progress**2 * (3 - 2 * progress)

            pos = start_pos + easing * (end_pos - start_pos)
            vel = (end_pos - start_pos) / duration if duration > 0 else np.zeros(3)

            # Determine nacelle angle (transition from hover to cruise if altitude is same)
            if waypoint.altitude == self.config.hover_altitude:
                # Hovering waypoint
                nacelle = np.radians(90)
            else:
                # Cruise waypoint - interpolate nacelle angle
                nacelle = np.radians(90) * (1 - easing) + np.radians(0) * easing

            points.append(
                TrajectoryPoint(
                    t=t,
                    x=pos[0],
                    y=pos[1],
                    z=pos[2],
                    vx=vel[0],
                    vy=vel[1],
                    vz=vel[2],
                    ax=0.0,
                    ay=0.0,
                    az=0.0,
                    heading=waypoint.heading,
                    nacelle_angle=nacelle,
                )
            )

        return points

    def _generate_waypoint_leg(
        self,
        prev_waypoint: Waypoint,
        waypoint: Waypoint,
        start_time: float,
        num_points: int = 50,
    ) -> list[TrajectoryPoint]:
        """Generate trajectory leg between two waypoints."""

        start_pos = np.array(
            [prev_waypoint.north, prev_waypoint.east, -prev_waypoint.altitude]
        )
        end_pos = np.array([waypoint.north, waypoint.east, -waypoint.altitude])

        distance = np.linalg.norm(end_pos - start_pos)
        speed = waypoint.speed if waypoint.speed > 0 else self.config.cruise_speed
        duration = distance / speed if speed > 0 else 5.0

        points = []

        # Main trajectory leg
        leg_points = int(num_points * 0.8)
        for i in range(leg_points):
            t = start_time + (i / (leg_points - 1)) * duration
            progress = i / (leg_points - 1)

            # Smooth easing
            easing = progress**2 * (3 - 2 * progress)

            pos = start_pos + easing * (end_pos - start_pos)
            vel = (end_pos - start_pos) / duration if duration > 0 else np.zeros(3)

            nacelle = np.radians(0)  # Cruise mode

            points.append(
                TrajectoryPoint(
                    t=t,
                    x=pos[0],
                    y=pos[1],
                    z=pos[2],
                    vx=vel[0],
                    vy=vel[1],
                    vz=vel[2],
                    ax=0.0,
                    ay=0.0,
                    az=0.0,
                    heading=waypoint.heading,
                    nacelle_angle=nacelle,
                )
            )

        # Hold at waypoint if specified
        if waypoint.hold_time > 0:
            hold_points = int(num_points * 0.2)
            for i in range(hold_points):
                t = start_time + duration + (i / (hold_points - 1)) * waypoint.hold_time
                points.append(
                    TrajectoryPoint(
                        t=t,
                        x=end_pos[0],
                        y=end_pos[1],
                        z=end_pos[2],
                        vx=0.0,
                        vy=0.0,
                        vz=0.0,
                        ax=0.0,
                        ay=0.0,
                        az=0.0,
                        heading=waypoint.heading,
                        nacelle_angle=np.radians(90) if waypoint.altitude == self.config.hover_altitude else np.radians(0),
                    )
                )

        return points

    def _generate_landing_phase(
        self, start_time: float, num_points: int = 20
    ) -> list[TrajectoryPoint]:
        """Generate landing phase: hover → static at home position."""

        home = self.config.home_position
        hover_alt = home[2] - self.config.hover_altitude  # NED frame

        points = []
        duration = 5.0  # 5 second landing

        for i in range(num_points):
            t = start_time + (i / (num_points - 1)) * duration
            progress = i / (num_points - 1)

            # Smooth ramp down using cubic easing
            easing = progress**2 * (3 - 2 * progress)

            z = hover_alt + easing * (home[2] - hover_alt)

            points.append(
                TrajectoryPoint(
                    t=t,
                    x=home[0],
                    y=home[1],
                    z=z,
                    vx=0.0,
                    vy=0.0,
                    vz=(home[2] - hover_alt) / duration if duration > 0 else 0.0,
                    ax=0.0,
                    ay=0.0,
                    az=0.0,
                    heading=0.0,
                    nacelle_angle=np.radians(90),  # Hover mode
                )
            )

        return points


def create_default_mission() -> dict[str, Any]:
    """Create a default mission configuration for testing."""
    return {
        "mission_name": "default_test_mission",
        "mission_config": {
            "cruise_speed": 25.0,
            "hover_altitude": 10.0,
            "transition_duration": 5.0,
        },
        "waypoints": [
            {
                "north": 0,
                "east": 0,
                "altitude": 10.0,
                "heading": 0.0,
                "speed": 10.0,
                "hold_time": 3.0,
            },
            {
                "north": 100,
                "east": 0,
                "altitude": 20.0,
                "heading": 0.0,
                "speed": 25.0,
                "hold_time": 2.0,
            },
            {
                "north": 100,
                "east": 100,
                "altitude": 20.0,
                "heading": 1.57,
                "speed": 25.0,
                "hold_time": 2.0,
            },
            {
                "north": 0,
                "east": 100,
                "altitude": 10.0,
                "heading": 3.14,
                "speed": 15.0,
                "hold_time": 3.0,
            },
            {
                "north": 0,
                "east": 0,
                "altitude": 10.0,
                "heading": 0.0,
                "speed": 10.0,
                "hold_time": 5.0,
            },
        ],
    }
