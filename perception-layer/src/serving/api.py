"""
Perception API for eVTOL Trajectory Planning

Provides interfaces for querying terrain, wind, threat, and energy data
for path planning and optimization algorithms.
"""

import math
import logging
from dataclasses import dataclass
from typing import Tuple, Optional, Union

logger = logging.getLogger(__name__)


@dataclass
class QueryPoint:
    """Data point for perception queries."""
    lat: float
    lon: float
    alt_m: float
    time_iso: str = "2024-01-01T12:00:00Z"


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate great-circle distance between two points on Earth.
    
    Args:
        lat1, lon1: First point (degrees)
        lat2, lon2: Second point (degrees)
        
    Returns:
        Distance in kilometers
    """
    R = 6371.0  # Earth radius in km
    
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)
    
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
    c = 2 * math.asin(math.sqrt(a))
    
    return R * c


def risk_score(point_or_lat: Union[float, QueryPoint], lon: Optional[float] = None, alt_m: Optional[float] = None, time_iso: str = "2024-01-01T12:00:00Z") -> float:
    """
    Evaluate threat/risk score at a given location.
    
    Can be called with either:
    - A QueryPoint object: risk_score(point)
    - Individual parameters: risk_score(lat, lon, alt_m, time_iso)
    
    Args:
        point_or_lat: QueryPoint object OR latitude in degrees
        lon: Longitude in degrees (ignored if first arg is QueryPoint)
        alt_m: Altitude in meters (ignored if first arg is QueryPoint)
        time_iso: ISO timestamp
        
    Returns:
        Risk score from 0.0 (safe) to 1.0 (high threat)
    """
    # Handle QueryPoint object
    if isinstance(point_or_lat, QueryPoint):
        lat = point_or_lat.lat
        lon = point_or_lat.lon
        alt_m = point_or_lat.alt_m
        time_iso = point_or_lat.time_iso
    else:
        lat = point_or_lat
    
    # Stub implementation: return 0.0 (safe)
    # In production, this would query actual threat models
    # (radar, hostile territories, weather hazards, etc.)
    logger.debug(f"Querying risk at ({lat:.4f}, {lon:.4f}, {alt_m}m)")
    return 0.0


def feasible(point_or_lat: Union[float, QueryPoint], lon: Optional[float] = None, alt_m: Optional[float] = None, time_iso: str = "2024-01-01T12:00:00Z") -> bool:
    """
    Check if a waypoint is feasible (within allowed airspace).
    
    Can be called with either:
    - A QueryPoint object: feasible(point)
    - Individual parameters: feasible(lat, lon, alt_m, time_iso)
    
    Args:
        point_or_lat: QueryPoint object OR latitude in degrees
        lon: Longitude in degrees (ignored if first arg is QueryPoint)
        alt_m: Altitude in meters (ignored if first arg is QueryPoint)
        time_iso: ISO timestamp
        
    Returns:
        True if waypoint is feasible, False otherwise
    """
    # Handle QueryPoint object
    if isinstance(point_or_lat, QueryPoint):
        lat = point_or_lat.lat
        lon = point_or_lat.lon
        alt_m = point_or_lat.alt_m
        time_iso = point_or_lat.time_iso
    else:
        lat = point_or_lat
    
    # Stub implementation: check altitude constraints
    # In production, this would check:
    # - Airspace restrictions
    # - Terrain clearance
    # - No-fly zones
    # - Altitude constraints
    min_alt = 50  # meters
    max_alt = 5000  # meters
    
    if alt_m is None:
        return False
    
    is_feasible = min_alt <= alt_m <= max_alt
    
    if not is_feasible:
        logger.debug(f"Waypoint ({lat:.4f}, {lon:.4f}, {alt_m}m) not feasible: altitude out of range")
    
    return is_feasible


def energy_cost_kwh_per_km(lat: float, lon: float, alt_m: Optional[float] = None, time_iso: str = "2024-01-01T12:00:00Z") -> float:
    """
    Estimate energy cost per kilometer at a given location.
    
    Accounts for:
    - Altitude (higher altitude = thinner air = more energy)
    - Wind conditions
    - Terrain slope and surface roughness
    
    Args:
        lat: Latitude in degrees
        lon: Longitude in degrees
        alt_m: Altitude in meters (default: sea level)
        time_iso: ISO timestamp
        
    Returns:
        Energy cost in kWh per km
    """
    # Stub implementation: return a constant value
    # In production, this would:
    # - Query wind model for wind speed/direction at location
    # - Query terrain model for slope and surface properties
    # - Use aerodynamic models to compute energy cost
    
    # Default to sea level if not provided
    if alt_m is None:
        alt_m = 0.0
    
    # Base energy cost (kWh/km at sea level, calm wind, flat terrain)
    base_cost = 1.0
    
    # Altitude adjustment (simplified: 10% increase per 1000m)
    altitude_factor = 1.0 + (alt_m / 1000.0) * 0.10
    
    total_cost = base_cost * altitude_factor
    
    logger.debug(f"Energy cost at ({lat:.4f}, {lon:.4f}, {alt_m}m): {total_cost:.3f} kWh/km")
    return total_cost
