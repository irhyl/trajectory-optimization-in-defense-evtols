"""
Mission Planning Module

High-level mission orchestration, multi-leg planning, and trajectory optimization.
"""

from __future__ import annotations

from typing import Dict, List, Optional
import logging

import numpy as np

from ..base import MissionPlanner as MissionPlannerBase
from ..base import Waypoint
from ..config import PlanningConfig

logger = logging.getLogger(__name__)


class MissionPlanner(MissionPlannerBase):
    """
    Mission-level planning and orchestration.

    Features:
    - Multi-leg mission planning
    - Waypoint optimization
    - Holding pattern generation
    - Trajectory smoothing
    - Flight envelope enforcement

    Example:
        >>> config = PlanningConfig()
        >>> planner = MissionPlanner(config)
        >>> origin = Waypoint(40.0, -74.0, 100)
        >>> destinations = [Waypoint(40.1, -74.1, 150), Waypoint(40.2, -74.2, 200)]
        >>> mission = planner.plan_mission(origin, destinations, "2024-01-01T12:00:00Z")
    """

    def __init__(self, config: PlanningConfig) -> None:
        """Initialize mission planner."""
        self.config = config
        
        # Flight envelope parameters
        self.min_turn_radius_m = float(config.get("vehicle.min_turn_radius_m", 50.0))
        self.max_altitude_rate_m_s = float(config.get("vehicle.max_altitude_rate_mps", 5.0))
        self.cruise_speed_m_s = float(config.get("energy.cruise_speed_mps", 35.0))

    def plan_mission(
        self,
        origin: Waypoint,
        destinations: List[Waypoint],
        time_iso: str,
        constraints: Optional[Dict] = None
    ) -> Dict:
        """
        Plan complete mission visiting multiple destinations.
        
        Args:
            origin: Starting waypoint
            destinations: List of destinations to visit
            time_iso: ISO timestamp for environment state
            constraints: Optional constraints
            
        Returns:
            Mission plan dict with routes, schedule, and metadata
        """
        if not destinations:
            raise ValueError("At least one destination required")
        
        logger.info(f"Planning mission with {len(destinations)} destinations")
        
        # Build waypoint sequence: origin -> destinations
        all_waypoints = [origin] + destinations
        
        # Simple mission structure: single route visiting all destinations in order
        routes = []
        current_pos = origin
        
        for dest in destinations:
            # In a real implementation, this would call a route planner
            # For now, create a simple linear interpolation route
            route_waypoints = self._interpolate_route(current_pos, dest)
            routes.append({
                "from": current_pos.__dict__,
                "to": dest.__dict__,
                "waypoints": [w.__dict__ for w in route_waypoints],
                "distance_km": self._compute_distance(route_waypoints),
            })
            current_pos = dest
        
        # Compute mission schedule
        total_distance_km = sum(r["distance_km"] for r in routes)
        total_time_s = (total_distance_km * 1000.0) / self.cruise_speed_m_s
        
        return {
            "type": "multi_leg_mission",
            "origin": origin.__dict__,
            "destinations": [d.__dict__ for d in destinations],
            "routes": routes,
            "total_distance_km": total_distance_km,
            "total_time_s": total_time_s,
            "num_legs": len(destinations),
            "metadata": {
                "cruise_speed_mps": self.cruise_speed_m_s,
                "min_turn_radius_m": self.min_turn_radius_m,
            }
        }

    def generate_holding_pattern(
        self,
        reference_waypoint: Waypoint,
        duration_s: float,
        pattern_type: str = "figure_eight"
    ) -> List[Waypoint]:
        """
        Generate holding pattern for loitering.
        
        Args:
            reference_waypoint: Center waypoint for pattern
            duration_s: Total duration in seconds
            pattern_type: Pattern type (figure_eight, circle, etc.)
            
        Returns:
            List of waypoints forming holding pattern
        """
        logger.info(f"Generating {pattern_type} holding pattern for {duration_s:.0f}s")
        
        # Holding pattern dimensions
        radius_m = 500.0  # 500m radius pattern
        
        if pattern_type == "circle":
            return self._generate_circle_pattern(reference_waypoint, radius_m, duration_s)
        elif pattern_type == "figure_eight":
            return self._generate_figure_eight_pattern(reference_waypoint, radius_m, duration_s)
        else:
            logger.warning(f"Unknown pattern type: {pattern_type}, using circle")
            return self._generate_circle_pattern(reference_waypoint, radius_m, duration_s)

    def smooth_trajectory(
        self,
        waypoints: List[Waypoint],
        min_turn_radius_m: float = 50.0,
        max_altitude_rate_m_s: float = 5.0
    ) -> List[Waypoint]:
        """
        Smooth waypoint sequence for realistic flight.
        
        Args:
            waypoints: Raw waypoint sequence
            min_turn_radius_m: Minimum turning radius
            max_altitude_rate_m_s: Maximum altitude change rate
            
        Returns:
            Smoothed waypoint sequence
        """
        if len(waypoints) < 3:
            return waypoints
        
        logger.info(f"Smoothing trajectory with {len(waypoints)} waypoints")
        
        # Convert to array for easier manipulation
        arr = np.array([[w.lat, w.lon, w.alt_m] for w in waypoints], dtype=np.float64)
        
        # Apply moving average filter for smoothing
        window_size = min(5, len(waypoints))
        if window_size < 3:
            return waypoints
        
        smoothed = np.zeros_like(arr)
        
        # Use simple moving average (no convolution to avoid edge issues)
        for i in range(len(arr)):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(arr), i + window_size // 2 + 1)
            
            smoothed[i] = np.mean(arr[start_idx:end_idx], axis=0)
        
        # Enforce altitude rate constraint
        for i in range(1, len(smoothed)):
            dt = 1.0  # Assume 1 second between waypoints
            max_alt_change = max_altitude_rate_m_s * dt
            
            alt_change = smoothed[i, 2] - smoothed[i - 1, 2]
            if abs(alt_change) > max_alt_change:
                smoothed[i, 2] = smoothed[i - 1, 2] + np.sign(alt_change) * max_alt_change
        
        return [Waypoint(float(lat), float(lon), float(alt))
                for lat, lon, alt in smoothed]

    # Private helper methods
    
    def _interpolate_route(self, start: Waypoint, goal: Waypoint, num_points: int = 25) -> List[Waypoint]:
        """Linear interpolation between two waypoints."""
        lats = np.linspace(start.lat, goal.lat, num_points)
        lons = np.linspace(start.lon, goal.lon, num_points)
        alts = np.linspace(start.alt_m, goal.alt_m, num_points)
        
        return [Waypoint(float(lat), float(lon), float(alt))
                for lat, lon, alt in zip(lats, lons, alts)]

    def _compute_distance(self, waypoints: List[Waypoint]) -> float:
        """Compute total distance along waypoint sequence."""
        if len(waypoints) < 2:
            return 0.0
        
        total_m = sum(
            waypoints[i].distance_to(waypoints[i + 1])
            for i in range(len(waypoints) - 1)
        )
        
        return total_m / 1000.0

    def _generate_circle_pattern(
        self,
        center: Waypoint,
        radius_m: float,
        duration_s: float
    ) -> List[Waypoint]:
        """Generate circular holding pattern."""
        # Number of points based on duration and cruise speed
        num_points = max(10, int(duration_s * self.cruise_speed_m_s / 1000))
        
        angles = np.linspace(0, 2 * np.pi, num_points)
        
        # Convert radius to lat/lon degrees (approximately)
        radius_deg = radius_m / 111000.0  # 1 degree ~ 111 km
        
        waypoints = []
        for angle in angles:
            lat = center.lat + radius_deg * np.sin(angle)
            lon = center.lon + radius_deg * np.cos(angle)
            waypoints.append(Waypoint(lat, lon, center.alt_m))
        
        return waypoints

    def _generate_figure_eight_pattern(
        self,
        center: Waypoint,
        radius_m: float,
        duration_s: float
    ) -> List[Waypoint]:
        """Generate figure-eight holding pattern."""
        num_points = max(20, int(duration_s * self.cruise_speed_m_s / 1000))
        
        t_values = np.linspace(0, 4 * np.pi, num_points)
        
        # Convert radius to lat/lon degrees
        radius_deg = radius_m / 111000.0
        
        waypoints = []
        for t in t_values:
            # Figure-eight parametric equations
            lat = center.lat + radius_deg * np.sin(t)
            lon = center.lon + radius_deg * np.sin(t) * np.cos(t)
            waypoints.append(Waypoint(lat, lon, center.alt_m))
        
        return waypoints
