"""
Energy Optimization Module

Battery modeling and energy cost estimation for eVTOL trajectory planning.
Provides energy feasibility constraints and optimization capabilities.
"""

from __future__ import annotations

from typing import List, Optional, Tuple
import logging

from ..base import EnergyOptimizer as EnergyOptimizerBase
from ..base import Waypoint
from ..config import PlanningConfig
from ..serving.perception_client import PerceptionClient

logger = logging.getLogger(__name__)


class EnergyOptimizer(EnergyOptimizerBase):
    """
    Energy optimization and battery management.

    Features:
    - Battery capacity modeling
    - Reserve fraction enforcement
    - Energy cost estimation
    - Range computation
    - Route energy optimization

    Example:
        >>> config = PlanningConfig()
        >>> optimizer = EnergyOptimizer(config)
        >>> route = [waypoint1, waypoint2, waypoint3]
        >>> energy_kwh = optimizer.estimate_route_energy(route)
    """

    def __init__(self, config: PlanningConfig) -> None:
        """Initialize energy optimizer."""
        self.config = config
        
        # Initialize perception client with graceful fallback for testing
        try:
            self.perception = PerceptionClient(config)
        except (ImportError, ValueError):
            logger.warning("Perception layer not available; using fake-mode for testing.")
            self.perception = PerceptionClient(config, use_fake=True)
        
        # Battery parameters
        self.battery_capacity_kwh = float(
            config.get("energy.battery_capacity_kwh", 120.0)
        )
        self.reserve_fraction = float(
            config.get("energy.reserve_fraction", 0.15)
        )
        self.usable_capacity_kwh = self.battery_capacity_kwh * (1.0 - self.reserve_fraction)
        
        # Performance parameters
        self.cruise_speed_m_s = float(
            config.get("energy.cruise_speed_mps", 35.0)
        )
        self.power_idle_kw = float(
            config.get("energy.power_idle_kw", 10.0)
        )

    def estimate_route_energy(
        self,
        route: List[Waypoint],
        time_iso: Optional[str] = None
    ) -> float:
        """
        Estimate total energy consumption for a route.
        
        Args:
            route: List of waypoints
            time_iso: ISO timestamp for environment state
            
        Returns:
            Energy consumption in kWh
        """
        if len(route) < 2:
            return 0.0
        
        time_iso = time_iso or "1970-01-01T00:00:00Z"
        total_energy_kwh = 0.0
        
        for i in range(len(route) - 1):
            a, b = route[i], route[i + 1]
            dist_m = a.distance_to(b)
            dist_km = dist_m / 1000.0
            
            try:
                # Query perception for energy cost at current position
                result = self.perception.query(a.lat, a.lon, a.alt_m, time_iso)
                energy_per_km = getattr(result, 'energy_cost_kwh_per_km', 1.0)
            except:
                # Fallback to default
                energy_per_km = 1.0
            
            total_energy_kwh += energy_per_km * dist_km
            
            # Check reserve limit
            if total_energy_kwh >= self.usable_capacity_kwh:
                logger.warning(f"Route exceeds usable capacity: {total_energy_kwh:.1f} kWh")
                return self.usable_capacity_kwh
        
        return min(total_energy_kwh, self.usable_capacity_kwh)

    def optimize_route_for_energy(
        self,
        route: List[Waypoint],
        max_energy_kwh: float,
        time_iso: Optional[str] = None
    ) -> List[Waypoint]:
        """
        Optimize route to fit within energy budget.
        
        Args:
            route: Original route
            max_energy_kwh: Maximum available energy
            time_iso: ISO timestamp for environment state
            
        Returns:
            Shortened route that fits budget, or original if already fits
        """
        time_iso = time_iso or "1970-01-01T00:00:00Z"
        
        # Find maximum reachable distance with given energy
        current_energy = 0.0
        for i in range(1, len(route)):
            segment_energy = self.estimate_route_energy(route[:i+1], time_iso)
            
            if segment_energy > max_energy_kwh:
                logger.info(f"Truncating route at waypoint {i} to fit energy budget")
                return route[:i]
            
            current_energy = segment_energy
        
        # Route fits within budget
        return route

    def get_range_for_energy(
        self,
        start: Waypoint,
        energy_kwh: float,
        time_iso: Optional[str] = None
    ) -> Tuple[float, float]:
        """
        Get range estimates for given energy.
        
        Args:
            start: Starting waypoint
            energy_kwh: Available energy in kWh
            time_iso: ISO timestamp for environment state
            
        Returns:
            Tuple of (min_range_km, max_range_km)
        """
        time_iso = time_iso or "1970-01-01T00:00:00Z"
        
        try:
            result = self.perception.query(start.lat, start.lon, start.alt_m, time_iso)
            energy_per_km = getattr(result, 'energy_cost_kwh_per_km', 1.0)
        except:
            energy_per_km = 1.0
        
        # Conservative estimate (30% margin)
        min_range_km = (energy_kwh * 0.7) / energy_per_km
        # Optimistic estimate
        max_range_km = energy_kwh / energy_per_km
        
        return (max(0.0, min_range_km), max_range_km)
