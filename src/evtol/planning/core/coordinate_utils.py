"""
Coordinate Frame Conversion Utilities

This module provides utilities for converting between geodetic (lat, lon, alt)
and local NED (North-East-Down) coordinate frames.

For path planning, we convert to a local meter-based frame where algorithms
like A*, RRT, and Dijkstra work correctly with Euclidean distances.

The local frame uses the mission start point as the origin:
- North (x): meters north of reference
- East (y): meters east of reference
- Down (z): meters below reference (or we use altitude directly as Up)

Author: Defense eVTOL Research Team
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass

# Earth radius for conversions
EARTH_RADIUS_M = 6371000.0
DEG_TO_M_LAT = 111000.0  # Approximate meters per degree latitude


@dataclass
class LocalFrameConverter:
    """
    Converts between geodetic (WGS84) and local NED coordinates.

    Uses a reference point as the origin of the local frame.
    The local frame is:
    - X: North (meters)
    - Y: East (meters)
    - Z: Altitude (meters, positive up - we use ENU-like for altitude)

    This is a simplified conversion suitable for areas < 500km across
    where Earth curvature effects are minimal.

    Example:
        >>> converter = LocalFrameConverter(ref_lat=33.5, ref_lon=77.0, ref_alt=4500.0)
        >>> local = converter.geodetic_to_local(34.0, 78.0, 5000.0)
        >>> print(f"North: {local[0]:.0f}m, East: {local[1]:.0f}m, Alt: {local[2]:.0f}m")
        North: 55500m, East: 92851m, Alt: 5000m
    """

    ref_lat: float  # Reference latitude (degrees)
    ref_lon: float  # Reference longitude (degrees)
    ref_alt: float  # Reference altitude (meters)

    def __post_init__(self):
        """Precompute conversion factors."""
        # Meters per degree longitude at reference latitude
        self.m_per_deg_lon = DEG_TO_M_LAT * np.cos(np.radians(self.ref_lat))
        self.m_per_deg_lat = DEG_TO_M_LAT

    def geodetic_to_local(
        self,
        lat: float,
        lon: float,
        alt: float
    ) -> np.ndarray:
        """
        Convert geodetic coordinates to local NED frame.

        Args:
            lat: Latitude in degrees
            lon: Longitude in degrees
            alt: Altitude in meters (above sea level)

        Returns:
            np.ndarray: [north, east, altitude] in meters
        """
        north = (lat - self.ref_lat) * self.m_per_deg_lat
        east = (lon - self.ref_lon) * self.m_per_deg_lon
        # Keep altitude as-is (positive up, not true NED which is positive down)
        return np.array([north, east, alt])

    def local_to_geodetic(
        self,
        north: float,
        east: float,
        alt: float
    ) -> np.ndarray:
        """
        Convert local NED frame to geodetic coordinates.

        Args:
            north: North offset in meters
            east: East offset in meters
            alt: Altitude in meters

        Returns:
            np.ndarray: [latitude, longitude, altitude] in degrees/meters
        """
        lat = self.ref_lat + north / self.m_per_deg_lat
        lon = self.ref_lon + east / self.m_per_deg_lon
        return np.array([lat, lon, alt])

    def geodetic_array_to_local(
        self,
        positions: np.ndarray
    ) -> np.ndarray:
        """
        Convert array of geodetic positions to local frame.

        Args:
            positions: Nx3 array of [lat, lon, alt]

        Returns:
            Nx3 array of [north, east, alt] in meters
        """
        local = np.zeros_like(positions)
        local[:, 0] = (positions[:, 0] - self.ref_lat) * self.m_per_deg_lat
        local[:, 1] = (positions[:, 1] - self.ref_lon) * self.m_per_deg_lon
        local[:, 2] = positions[:, 2]  # altitude stays in meters
        return local

    def local_array_to_geodetic(
        self,
        positions: np.ndarray
    ) -> np.ndarray:
        """
        Convert array of local positions to geodetic frame.

        Args:
            positions: Nx3 array of [north, east, alt] in meters

        Returns:
            Nx3 array of [lat, lon, alt]
        """
        geodetic = np.zeros_like(positions)
        geodetic[:, 0] = self.ref_lat + positions[:, 0] / self.m_per_deg_lat
        geodetic[:, 1] = self.ref_lon + positions[:, 1] / self.m_per_deg_lon
        geodetic[:, 2] = positions[:, 2]
        return geodetic

    def convert_bounds_to_local(
        self,
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float
    ) -> tuple[float, float, float, float]:
        """
        Convert geodetic bounds to local frame bounds.

        Args:
            lat_min, lat_max: Latitude bounds (degrees)
            lon_min, lon_max: Longitude bounds (degrees)

        Returns:
            (x_min, x_max, y_min, y_max) in meters
        """
        x_min = (lat_min - self.ref_lat) * self.m_per_deg_lat
        x_max = (lat_max - self.ref_lat) * self.m_per_deg_lat
        y_min = (lon_min - self.ref_lon) * self.m_per_deg_lon
        y_max = (lon_max - self.ref_lon) * self.m_per_deg_lon
        return (x_min, x_max, y_min, y_max)

    def distance(self, p1: np.ndarray, p2: np.ndarray) -> float:
        """
        Compute Euclidean distance in local frame.

        Args:
            p1, p2: Points in local frame [north, east, alt]

        Returns:
            Distance in meters
        """
        return np.linalg.norm(p2 - p1)


class LocalPlanningContext:
    """
    Context manager for planning in local coordinates.

    Handles conversion of all planning elements to/from local frame:
    - Start/goal states
    - Bounds
    - Trajectories
    - Cost grids

    Example:
        >>> with LocalPlanningContext(start, goal, bounds) as ctx:
        ...     # Plan in local frame
        ...     local_start, local_goal = ctx.local_start, ctx.local_goal
        ...     result = planner.plan(local_start, local_goal)
        ...     # Convert result back
        ...     geodetic_trajectory = ctx.to_geodetic_trajectory(result.trajectory)
    """

    def __init__(
        self,
        start_geodetic: np.ndarray,
        goal_geodetic: np.ndarray,
        bounds_geodetic: tuple[float, float, float, float],
        altitude_range: tuple[float, float] = (100, 5000),
    ):
        """
        Initialize planning context.

        Args:
            start_geodetic: [lat, lon, alt] start position
            goal_geodetic: [lat, lon, alt] goal position
            bounds_geodetic: (lat_min, lat_max, lon_min, lon_max)
            altitude_range: (min_alt, max_alt) in meters
        """
        # Use start as reference point
        self.converter = LocalFrameConverter(
            ref_lat=start_geodetic[0],
            ref_lon=start_geodetic[1],
            ref_alt=start_geodetic[2],
        )

        # Convert start and goal
        self.start_geodetic = start_geodetic
        self.goal_geodetic = goal_geodetic
        self.local_start = self.converter.geodetic_to_local(*start_geodetic)
        self.local_goal = self.converter.geodetic_to_local(*goal_geodetic)

        # Convert bounds to local
        lat_min, lat_max, lon_min, lon_max = bounds_geodetic
        self.local_bounds = self.converter.convert_bounds_to_local(
            lat_min, lat_max, lon_min, lon_max
        )

        # Altitude range (already in meters)
        self.altitude_min, self.altitude_max = altitude_range

    def to_local_position(self, geodetic: np.ndarray) -> np.ndarray:
        """Convert geodetic position to local."""
        return self.converter.geodetic_to_local(*geodetic)

    def to_geodetic_position(self, local: np.ndarray) -> np.ndarray:
        """Convert local position to geodetic."""
        return self.converter.local_to_geodetic(*local)

    def positions_to_geodetic(self, local_positions: np.ndarray) -> np.ndarray:
        """Convert array of local positions to geodetic."""
        return self.converter.local_array_to_geodetic(local_positions)

    def sample_cost_grid(
        self,
        cost_grid: np.ndarray,
        grid_bounds: tuple[float, float, float, float],
        local_position: np.ndarray,
    ) -> float:
        """
        Sample a geodetic cost grid at a local position.

        Args:
            cost_grid: 2D array of cost values
            grid_bounds: (lat_min, lat_max, lon_min, lon_max) of grid
            local_position: [north, east, alt] to sample at

        Returns:
            Cost value at position
        """
        # Convert local to geodetic for sampling
        geodetic = self.to_geodetic_position(local_position)
        lat, lon = geodetic[0], geodetic[1]

        # Sample grid
        lat_min, lat_max, lon_min, lon_max = grid_bounds
        rows, cols = cost_grid.shape

        # Normalized position
        lat_norm = (lat - lat_min) / (lat_max - lat_min)
        lon_norm = (lon - lon_min) / (lon_max - lon_min)

        # Grid indices
        i = int(lat_norm * (rows - 1))
        j = int(lon_norm * (cols - 1))

        # Clamp to bounds
        i = max(0, min(rows - 1, i))
        j = max(0, min(cols - 1, j))

        return float(cost_grid[i, j])
