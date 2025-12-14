"""
Risk Assessment and Management

Threat evaluation and risk-aware path planning for eVTOL operations.
"""

from __future__ import annotations

from typing import List, Optional
import logging

from ..base import RiskManager as RiskManagerBase
from ..base import Waypoint, RoutePlan
from ..config import PlanningConfig
from ..serving.perception_client import PerceptionClient

logger = logging.getLogger(__name__)


class RiskManager(RiskManagerBase):
    """
    Risk assessment and emergency planning.

    Features:
    - Threat evaluation along routes
    - Per-waypoint risk scoring
    - Contingency route planning
    - Integration with perception layer

    Example:
        >>> config = PlanningConfig()
        >>> manager = RiskManager(config)
        >>> risk_score = manager.evaluate_route_risk(waypoints)
    """

    def __init__(self, config: PlanningConfig) -> None:
        """Initialize risk manager."""
        self.config = config
        
        # Initialize perception client with graceful fallback for testing
        try:
            self.perception = PerceptionClient(config)
        except (ImportError, ValueError):
            logger.warning("Perception layer not available; using fake-mode for testing.")
            self.perception = PerceptionClient(config, use_fake=True)
        
        self.max_acceptable_risk = float(config.get("risk.max_risk_score", 0.7))

    def evaluate_route_risk(
        self,
        route: List[Waypoint],
        time_iso: Optional[str] = None
    ) -> float:
        """
        Evaluate overall risk score for a route [0-1].
        
        Args:
            route: List of waypoints
            time_iso: ISO timestamp for environment state
            
        Returns:
            Risk score from 0 (safe) to 1 (dangerous)
        """
        if len(route) < 2:
            return 0.0
        
        time_iso = time_iso or "1970-01-01T00:00:00Z"
        total_risk_weighted = 0.0
        total_distance_km = 0.0
        
        for i in range(len(route) - 1):
            a, b = route[i], route[i + 1]
            dist_km = a.distance_to(b) / 1000.0
            
            try:
                result = self.perception.query(a.lat, a.lon, a.alt_m, time_iso)
                risk = getattr(result, 'risk_score', 0.0)
                total_risk_weighted += risk * dist_km
            except:
                pass
            
            total_distance_km += dist_km
        
        if total_distance_km <= 0.0:
            return 0.0
        
        avg_risk = total_risk_weighted / total_distance_km
        return min(avg_risk, 1.0)

    def evaluate_waypoint_risk(
        self,
        waypoint: Waypoint,
        time_iso: Optional[str] = None
    ) -> float:
        """
        Evaluate risk score at a single waypoint.
        
        Args:
            waypoint: Waypoint to evaluate
            time_iso: ISO timestamp for environment state
            
        Returns:
            Risk score from 0 (safe) to 1 (dangerous)
        """
        time_iso = time_iso or "1970-01-01T00:00:00Z"
        
        try:
            result = self.perception.query(waypoint.lat, waypoint.lon, waypoint.alt_m, time_iso)
            risk = getattr(result, 'risk_score', 0.0)
            return min(risk, 1.0)
        except:
            return 0.0

    def plan_contingency_route(
        self,
        current_route: List[Waypoint],
        retreat_waypoint: Waypoint,
        time_iso: Optional[str] = None
    ) -> RoutePlan:
        """
        Plan contingency/emergency route to safe location.
        
        Args:
            current_route: Current route being executed
            retreat_waypoint: Safe waypoint to retreat to
            time_iso: ISO timestamp for environment state
            
        Returns:
            RoutePlan for emergency diversion
        """
        # Find current position (last waypoint of current route)
        if not current_route:
            raise ValueError("Current route cannot be empty")
        
        current_pos = current_route[-1]
        
        # Create straight-line emergency route
        num_points = max(5, int(current_pos.distance_to(retreat_waypoint) / 1000))
        
        emergency_route = []
        for i in range(num_points):
            t = i / (num_points - 1) if num_points > 1 else 0
            
            lat = current_pos.lat + t * (retreat_waypoint.lat - current_pos.lat)
            lon = current_pos.lon + t * (retreat_waypoint.lon - current_pos.lon)
            alt = current_pos.alt_m + t * (retreat_waypoint.alt_m - current_pos.alt_m)
            
            emergency_route.append(Waypoint(lat, lon, alt))
        
        # Compute metrics
        distance_km = sum(
            emergency_route[i].distance_to(emergency_route[i + 1]) / 1000.0
            for i in range(len(emergency_route) - 1)
        )
        
        risk_score = self.evaluate_route_risk(emergency_route, time_iso)
        
        # Estimate energy (simplified)
        energy_kwh = distance_km * 1.0  # Default 1 kWh/km
        
        # Estimate flight time
        cruise_speed_m_s = float(self.config.get("energy.cruise_speed_mps", 35.0))
        flight_time_s = (distance_km * 1000.0) / cruise_speed_m_s
        
        return RoutePlan(
            waypoints=emergency_route,
            distance_km=distance_km,
            energy_kwh=energy_kwh,
            risk_score=risk_score,
            flight_time_s=flight_time_s,
            metadata={"type": "emergency_diversion"}
        )
