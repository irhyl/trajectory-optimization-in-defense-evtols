"""
Abstract Base Classes for Planning Layer

Defines the core interfaces for route planning, optimization, and mission planning.
All concrete planners must inherit from these base classes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class Waypoint:
    """Represents a 3D waypoint with geographic coordinates and altitude."""
    
    lat: float  # Latitude in degrees
    lon: float  # Longitude in degrees
    alt_m: float  # Altitude in meters above sea level
    
    def __hash__(self) -> int:
        """Hash based on rounded coordinates for grid-based algorithms."""
        return hash((round(self.lat, 6), round(self.lon, 6), round(self.alt_m, 1)))
    
    def __eq__(self, other: object) -> bool:
        """Equality based on rounded coordinates."""
        if not isinstance(other, Waypoint):
            return False
        return (round(self.lat, 6) == round(other.lat, 6) and
                round(self.lon, 6) == round(other.lon, 6) and
                round(self.alt_m, 1) == round(other.alt_m, 1))
    
    def __repr__(self) -> str:
        return f"Waypoint(lat={self.lat:.6f}, lon={self.lon:.6f}, alt={self.alt_m:.1f}m)"
    
    def distance_to(self, other: Waypoint) -> float:
        """Calculate 3D Euclidean distance to another waypoint in meters."""
        dlat = np.radians(other.lat - self.lat)
        dlon = np.radians(other.lon - self.lon)
        dalt = other.alt_m - self.alt_m
        
        # Haversine for lat/lon
        R = 6371000.0  # Earth radius in meters
        a = (np.sin(dlat / 2) ** 2 +
             np.cos(np.radians(self.lat)) * np.cos(np.radians(other.lat)) *
             np.sin(dlon / 2) ** 2)
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        horizontal = R * c
        
        # 3D distance
        return np.sqrt(horizontal ** 2 + dalt ** 2)


@dataclass
class RoutePlan:
    """Represents a complete route plan with waypoints and metadata."""
    
    waypoints: List[Waypoint]
    distance_km: float
    energy_kwh: float
    risk_score: float
    flight_time_s: float
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class RoutePlanner(ABC):
    """
    Abstract base class for route planning algorithms.
    
    Concrete implementations should handle:
    - Path planning (A*, Theta*, RRT*, etc.)
    - Dynamic feasibility constraints
    - Multi-objective cost optimization
    - Integration with perception layer
    """
    
    @abstractmethod
    def plan(
        self,
        start: Waypoint,
        goal: Waypoint,
        time_iso: str,
        constraints: Optional[Dict] = None
    ) -> RoutePlan:
        """
        Plan a route from start to goal waypoint.
        
        Args:
            start: Starting waypoint
            goal: Goal waypoint
            time_iso: ISO timestamp for environment state
            constraints: Optional dict with max_altitude, min_altitude, max_distance_km, etc.
            
        Returns:
            RoutePlan with waypoints and cost metrics
            
        Raises:
            ValueError: If no feasible path exists
        """
        pass


class EnergyOptimizer(ABC):
    """
    Abstract base class for energy optimization.
    
    Concrete implementations should handle:
    - Battery energy modeling
    - Trajectory energy cost estimation
    - Reserve management
    - Multi-segment energy planning
    """
    
    @abstractmethod
    def estimate_route_energy(
        self,
        route: List[Waypoint],
        time_iso: Optional[str] = None
    ) -> float:
        """
        Estimate total energy consumption for a route in kWh.
        
        Args:
            route: List of waypoints to follow
            time_iso: ISO timestamp for environment state
            
        Returns:
            Total energy consumption in kWh
        """
        pass
    
    @abstractmethod
    def optimize_route_for_energy(
        self,
        route: List[Waypoint],
        max_energy_kwh: float,
        time_iso: Optional[str] = None
    ) -> List[Waypoint]:
        """
        Optimize a route to minimize energy consumption.
        
        Args:
            route: Original route to optimize
            max_energy_kwh: Maximum available energy
            time_iso: ISO timestamp for environment state
            
        Returns:
            Optimized route that fits within energy budget
        """
        pass
    
    @abstractmethod
    def get_range_for_energy(
        self,
        start: Waypoint,
        energy_kwh: float,
        time_iso: Optional[str] = None
    ) -> Tuple[float, float]:
        """
        Get minimum and maximum achievable range for given energy.
        
        Args:
            start: Starting waypoint
            energy_kwh: Available energy in kWh
            time_iso: ISO timestamp for environment state
            
        Returns:
            Tuple of (min_range_km, max_range_km)
        """
        pass


class RiskManager(ABC):
    """
    Abstract base class for risk assessment and management.
    
    Concrete implementations should handle:
    - Threat evaluation along routes
    - Emergency diversion planning
    - Risk-aware path optimization
    - Contingency route planning
    """
    
    @abstractmethod
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
        pass
    
    @abstractmethod
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
        pass
    
    @abstractmethod
    def plan_contingency_route(
        self,
        current_route: List[Waypoint],
        retreat_waypoint: Waypoint,
        time_iso: Optional[str] = None
    ) -> RoutePlan:
        """
        Plan a contingency/emergency route to safe location.
        
        Args:
            current_route: Current route being executed
            retreat_waypoint: Safe waypoint to retreat to
            time_iso: ISO timestamp for environment state
            
        Returns:
            RoutePlan for emergency diversion
        """
        pass


class Optimizer(ABC):
    """
    Abstract base class for multi-objective optimization.
    
    Concrete implementations should handle:
    - Pareto frontier computation
    - Solution dominance checking
    - Knee point identification
    - Scalarization methods (weighted sum, Tchebycheff, etc.)
    """
    
    @abstractmethod
    def optimize(
        self,
        candidates: List[RoutePlan],
        objectives: Dict[str, float],
        constraints: Optional[Dict] = None
    ) -> List[RoutePlan]:
        """
        Optimize set of candidate routes based on objectives.
        
        Args:
            candidates: List of candidate routes to optimize
            objectives: Dict of {objective_name: weight} for multi-objective optimization
                       Common objectives: {distance, energy, risk, time}
            constraints: Optional constraints on objectives
            
        Returns:
            Pareto-optimal subset of candidates
        """
        pass


class MissionPlanner(ABC):
    """
    Abstract base class for mission-level planning.
    
    Concrete implementations should handle:
    - Multi-leg mission planning
    - Waypoint sequence optimization
    - Resource allocation
    - Flight envelope enforcement
    """
    
    @abstractmethod
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
        pass
    
    @abstractmethod
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
        pass
    
    @abstractmethod
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
        pass
