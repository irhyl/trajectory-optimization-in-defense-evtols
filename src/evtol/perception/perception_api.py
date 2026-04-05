"""
Unified Perception API - Integration Layer

Provides single interface to all perception components:
- Terrain queries (elevation, slope, surface type)
- Wind assessment (speed, direction, turbulence)
- Obstacle detection (clearance, landing zones)
- Threat evaluation (probability, engagement risk)
- Fused risk/feasibility (integrated decision making)

Integrates with:
  * Planning layer: Trajectory optimization constraints
  * Control layer: Real-time hazard avoidance
  * Vehicle layer: Dynamics-aware perception queries
  * External APIs: Open-Meteo (terrain, wind data)

Architecture:
    ┌─────────────────────────────────┐
    │   External APIs                 │
    │ (Open-Meteo, Overpass, etc.)    │
    └────────────┬────────────────────┘
                 ↓
    ┌─────────────────────────────────┐
    │   Perception Module             │
    │ (Terrain, Wind, Obstacle,       │
    │  Threat, Fusion models)         │
    └────────────┬────────────────────┘
                 ↓
    ┌─────────────────────────────────┐
    │  UNIFIED PERCEPTION API (THIS)  │
    │  • Query interface              │
    │  • Caching layer                │
    │  • Real-time updates            │
    │  • Dataset export               │
    └────────────┬────────────────────┘
                 ↓
    ┌────┬──────────────┬──────────┬────┐
    ↓    ↓              ↓          ↓    ↓
  Plan Control Vehicle Logging  Datasets
"""

import logging
import numpy as np
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import threading

logger = logging.getLogger(__name__)

# QUERY INTERFACES
@dataclass
class TerrainQuery:
    """Terrain-specific query and response."""
    latitude: float
    longitude: float
    
    # Response fields
    elevation_m: float = 0.0
    slope_deg: float = 0.0
    roughness_m: float = 0.0
    surface_type: str = "unknown"
    landing_feasible: bool = False
    
    def __repr__(self) -> str:
        return f"Terrain(elev={self.elevation_m:.0f}m, slope={self.slope_deg:.1f}°, surface={self.surface_type})"


@dataclass
class WindQuery:
    """Wind-specific query and response."""
    altitude_m: float
    
    # Response fields
    wind_north_mps: float = 0.0
    wind_east_mps: float = 0.0
    wind_vertical_mps: float = 0.0
    wind_speed_mps: float = 0.0
    wind_direction_deg: float = 0.0
    gust_mps: float = 0.0
    turbulence_intensity: float = 0.0
    crosswind_coefficient: float = 0.0  # Normalized 0-1
    
    def __repr__(self) -> str:
        return f"Wind(speed={self.wind_speed_mps:.1f}m/s, direction={self.wind_direction_deg:.0f}°, gusts={self.gust_mps:.1f}m/s)"


@dataclass
class ObstacleQuery:
    """Obstacle-specific query and response."""
    latitude: float
    longitude: float
    altitude_m: float
    
    # Response fields
    nearest_obstacle_distance_m: float = float('inf')
    nearest_obstacle_height_m: float = 0.0
    clearance_m: float = float('inf')
    landing_zone_available: bool = False
    
    def __repr__(self) -> str:
        return f"Obstacle(distance={self.nearest_obstacle_distance_m:.0f}m, clearance={self.clearance_m:.1f}m)"


@dataclass
class ThreatQuery:
    """Threat-specific query and response."""
    latitude: float = 0.0
    longitude: float = 0.0
    altitude_m: float = 0.0
    
    # Response fields
    threat_probability: float = 0.0
    engagement_probability: float = 0.0
    threat_category: str = "none"  # SAM, aircraft, radar, etc.
    threat_range_m: Optional[float] = None
    threat_bearing_deg: Optional[float] = None
    threat_elevation_deg: Optional[float] = None
    
    def __repr__(self) -> str:
        return f"Threat(prob={self.threat_probability:.2f}, category={self.threat_category})"


@dataclass
class FusedPerceptionQuery:
    """Integrated multi-modal perception query and response."""
    latitude: float
    longitude: float
    altitude_m: float
    timestamp: datetime = None
    
    # Integrated scores [0, 1]
    risk_score: float = 0.5      # Combined hazard level
    feasibility_score: float = 0.5   # Path viability
    energy_cost_score: float = 0.5   # Relative energy cost
    
    # Component confidence
    all_models_valid: bool = True
    dominant_hazard: str = "none"  # Which component dominates
    
    # Recommendations
    recommended_altitude_m: Optional[float] = None
    recommended_heading_deg: Optional[float] = None
    replanning_urgent: bool = False
    
    def __repr__(self) -> str:
        return f"FusedPerception(risk={self.risk_score:.2f}, feasibility={self.feasibility_score:.2f}, cost={self.energy_cost_score:.2f})"


# =============================================================================
# UNIFIED PERCEPTION API
# =============================================================================

class UnifiedPerceptionAPI:
    """
    Single interface to all perception components.
    
    Thread-safe, cached, real-time updates.
    """
    
    def __init__(
        self,
        terrain_model = None,
        wind_model = None,
        obstacle_model = None,
        threat_model = None,
        fusion_model = None,
        cache_enabled: bool = True,
        cache_ttl_s: float = 1.0,
    ):
        """
        Initialize unified perception API.
        
        Args:
            *_model: Initialized perception component models
            cache_enabled: Enable query caching for performance
            cache_ttl_s: Cache time-to-live in seconds
        """
        self.terrain_model = terrain_model
        self.wind_model = wind_model
        self.obstacle_model = obstacle_model
        self.threat_model = threat_model
        self.fusion_model = fusion_model
        
        # Caching
        self.cache_enabled = cache_enabled
        self.cache_ttl_s = cache_ttl_s
        self._query_cache: Dict[str, Tuple[datetime, Any]] = {}
        self._cache_lock = threading.RLock()
        
        # Statistics
        self._query_count = 0
        self._cache_hits = 0
        
        logger.info("Unified Perception API initialized")
    
    # =========================================================================
    # INDIVIDUAL COMPONENT QUERIES
    # =========================================================================
    
    def query_terrain(self, latitude: float, longitude: float) -> TerrainQuery:
        """Query terrain at geographic location."""
        query = TerrainQuery(latitude=latitude, longitude=longitude)
        
        cache_key = f"terrain_{latitude:.4f}_{longitude:.4f}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached
        
        if self.terrain_model is None:
            logger.warning("Terrain model not initialized")
            return query
        
        try:
            # Query terrain model (implementation-specific)
            # This is a template that APIs should implement
            elevation = getattr(self.terrain_model, 'dem', None)
            if elevation is not None and hasattr(self.terrain_model, 'grid_bounds'):
                # Extract query point
                idx_lat, idx_lon = self._geographic_to_grid_indices(
                    latitude, longitude, self.terrain_model
                )
                query.elevation_m = float(elevation[idx_lat, idx_lon])
                query.slope_deg = self._compute_slope(elevation, idx_lat, idx_lon)
                # Classify surface type from slope and elevation
                if query.slope_deg < 5.0:
                    query.surface_type = "flat"
                elif query.slope_deg < 15.0:
                    query.surface_type = "rolling"
                elif query.slope_deg < 30.0:
                    query.surface_type = "hilly"
                else:
                    query.surface_type = "mountainous"
                
            logger.debug(f"Terrain query: {query}")
            self._cache_result(cache_key, query)
            
        except Exception as e:
            logger.warning(f"Terrain query failed: {e}")
        
        return query
    
    def query_wind(self, altitude_m: float) -> WindQuery:
        """Query wind field at altitude."""
        query = WindQuery(altitude_m=altitude_m)
        
        cache_key = f"wind_{altitude_m:.0f}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached
        
        if self.wind_model is None:
            logger.warning("Wind model not initialized")
            return query
        
        try:
            # Query wind model
            # This is a template that APIs should implement
            wind_velocity = getattr(self.wind_model, 'wind_velocity', None)
            wind_speed = getattr(self.wind_model, 'wind_speed', None)
            
            if wind_velocity is not None:
                # Extract velocity at altitude
                alt_idx = int(altitude_m / 100.0)  # Assume 100m grid spacing
                if alt_idx < wind_velocity.shape[0]:
                    v = wind_velocity[alt_idx]
                    query.wind_north_mps = float(v[0]) if len(v) > 0 else 0
                    query.wind_east_mps = float(v[1]) if len(v) > 1 else 0
                    query.wind_vertical_mps = float(v[2]) if len(v) > 2 else 0
            
            # Compute derived quantities
            query.wind_speed_mps = np.sqrt(
                query.wind_north_mps**2 + query.wind_east_mps**2
            )
            query.wind_direction_deg = (
                np.degrees(np.arctan2(query.wind_east_mps, query.wind_north_mps)) % 360
            )
            query.crosswind_coefficient = min(query.wind_speed_mps / 15.0, 1.0)
            
            logger.debug(f"Wind query: {query}")
            self._cache_result(cache_key, query)
            
        except Exception as e:
            logger.warning(f"Wind query failed: {e}")
        
        return query
    
    def query_obstacle(
        self, latitude: float, longitude: float, altitude_m: float
    ) -> ObstacleQuery:
        """Query obstacle clearance at location and altitude."""
        query = ObstacleQuery(
            latitude=latitude, longitude=longitude, altitude_m=altitude_m
        )
        
        cache_key = f"obstacle_{latitude:.4f}_{longitude:.4f}_{altitude_m:.0f}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached
        
        if self.obstacle_model is None:
            logger.warning("Obstacle model not initialized")
            return query
        
        try:
            # Query obstacle model
            # This is a template that APIs should implement
            clearance_map = getattr(self.obstacle_model, 'clearance_map', None)
            building_height = getattr(self.obstacle_model, 'building_height', None)
            
            if clearance_map is not None:
                idx_lat, idx_lon = self._geographic_to_grid_indices(
                    latitude, longitude, self.obstacle_model
                )
                query.clearance_m = float(clearance_map[idx_lat, idx_lon])
                query.nearest_obstacle_height_m = float(building_height[idx_lat, idx_lon]) if building_height is not None else 0.0
                
                # Landing feasible if adequate clearance
                query.landing_zone_available = query.clearance_m > 50  # Arbitrary threshold
            
            logger.debug(f"Obstacle query: {query}")
            self._cache_result(cache_key, query)
            
        except Exception as e:
            logger.warning(f"Obstacle query failed: {e}")
        
        return query
    
    def query_threat(
        self, latitude: float, longitude: float, altitude_m: float
    ) -> ThreatQuery:
        """Query threat assessment at location."""
        query = ThreatQuery(
            latitude=latitude, longitude=longitude, altitude_m=altitude_m
        )
        
        cache_key = f"threat_{latitude:.4f}_{longitude:.4f}_{altitude_m:.0f}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached
        
        if self.threat_model is None:
            logger.warning("Threat model not initialized")
            return query
        
        try:
            # Query threat model
            # This is a template that APIs should implement
            threat_field = getattr(self.threat_model, 'threat_field', None)
            
            if threat_field is not None:
                idx_lat, idx_lon = self._geographic_to_grid_indices(
                    latitude, longitude, self.threat_model
                )
                threat_val = threat_field[idx_lat, idx_lon]
                query.threat_probability = float(np.clip(threat_val, 0, 1))
                query.engagement_probability = query.threat_probability * 0.8  # Simple model
                query.threat_category = "SAM" if query.threat_probability > 0.5 else "none"
            
            logger.debug(f"Threat query: {query}")
            self._cache_result(cache_key, query)
            
        except Exception as e:
            logger.warning(f"Threat query failed: {e}")
        
        return query
    
    # =========================================================================
    # FUSED QUERY
    # =========================================================================
    
    def query_fused_perception(
        self,
        latitude: float,
        longitude: float,
        altitude_m: float,
        vehicle_state: Optional[Dict[str, float]] = None,
    ) -> FusedPerceptionQuery:
        """
        Query integrated perception assessment.
        
        Combines all component models into unified recommendation.
        
        Args:
            latitude, longitude: Geographic position
            altitude_m: Current altitude AGL
            vehicle_state: Optional vehicle state dict (velocity, etc.)
        
        Returns:
            FusedPerceptionQuery with integrated scores and recommendations
        """
        query = FusedPerceptionQuery(
            latitude=latitude,
            longitude=longitude,
            altitude_m=altitude_m,
            timestamp=datetime.utcnow(),
        )
        
        cache_key = f"fused_{latitude:.4f}_{longitude:.4f}_{altitude_m:.0f}"
        cached = self._get_cached(cache_key)
        if cached:
            self._cache_hits += 1
            return cached
        self._query_count += 1
        
        try:
            # Query all components
            terrain = self.query_terrain(latitude, longitude)
            wind = self.query_wind(altitude_m)
            obstacle = self.query_obstacle(latitude, longitude, altitude_m)
            threat = self.query_threat(latitude, longitude, altitude_m)
            
            # Compute fused scores
            # Risk: combination of threats and terrain
            risk_threat = threat.threat_probability * threat.engagement_probability
            risk_terrain = min(terrain.slope_deg / 45.0, 1.0)  # Steep = risky
            risk_wind = min(wind.wind_speed_mps / 15.0, 1.0)  # Windy = risky
            risk_obstacle = min(1.0 - obstacle.clearance_m / 500.0, 1.0)  # Low clearance = risky
            
            query.risk_score = float(np.clip(
                0.35 * risk_threat +
                0.25 * risk_terrain +
                0.2 * risk_wind +
                0.2 * risk_obstacle,
                0, 1
            ))
            
            # Feasibility: inverse of risk, adjusted for landing zones
            query.feasibility_score = float(np.clip(
                1.0 - query.risk_score,
                0,
                1.0,
            ))
            
            # Energy cost: combination of environmental factors
            query.energy_cost_score = float(np.clip(
                0.4 * min(wind.wind_speed_mps / 15.0, 1.0) +
                0.3 * min(terrain.slope_deg / 45.0, 1.0) +
                0.3 * min(terrain.elevation_m / 1000.0, 1.0),
                0,
                1.0,
            ))
            
            # Determine dominant hazard
            hazards = {
                'threat': risk_threat,
                'terrain': risk_terrain,
                'wind': risk_wind,
                'obstacle': risk_obstacle,
            }
            query.dominant_hazard = max(hazards, key=hazards.get)
            
            # Recommendations
            if query.risk_score > 0.7:
                query.replanning_urgent = True
            
            if wind.wind_speed_mps > 12:
                query.recommended_altitude_m = altitude_m + 100  # Climb above wind
            
            if threat.threat_probability > 0.5:
                # Recommend heading away from threat
                query.recommended_heading_deg = (threat.threat_bearing_deg + 180) % 360 if threat.threat_bearing_deg else None
            
            logger.debug(f"Fused query: {query}")
            self._cache_result(cache_key, query)
            
        except Exception as e:
            logger.warning(f"Fused query failed: {e}")
            query.all_models_valid = False
        
        return query
    
    # =========================================================================
    # BATCH QUERIES (for planning and analysis)
    # =========================================================================
    
    def query_trajectory(
        self,
        waypoints: List[Tuple[float, float]],
        altitude_profile: Optional[List[float]] = None,
    ) -> List[FusedPerceptionQuery]:
        """Query perception along planned trajectory."""
        if altitude_profile is None:
            altitude_profile = [200.0] * len(waypoints)
        
        results = []
        for (lat, lon), alt in zip(waypoints, altitude_profile):
            query = self.query_fused_perception(lat, lon, alt)
            results.append(query)
        
        logger.info(f"Queried trajectory with {len(results)} waypoints")
        return results
    
    def query_grid(
        self,
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float,
        altitude_m: float,
        grid_spacing_deg: float = 0.01,
    ) -> np.ndarray:
        """
        Query perception over geographic grid.
        
        Returns 2D array of risk scores.
        """
        lats = np.arange(lat_min, lat_max, grid_spacing_deg)
        lons = np.arange(lon_min, lon_max, grid_spacing_deg)
        
        risk_grid = np.zeros((len(lats), len(lons)))
        
        for i, lat in enumerate(lats):
            for j, lon in enumerate(lons):
                query = self.query_fused_perception(lat, lon, altitude_m)
                risk_grid[i, j] = query.risk_score
        
        logger.info(f"Generated {len(lats)}×{len(lons)} perception grid")
        return risk_grid
    
    # =========================================================================
    # CACHING
    # =========================================================================
    
    def _get_cached(self, key: str) -> Optional[Any]:
        """Get cached query result if fresh."""
        if not self.cache_enabled:
            return None
        
        with self._cache_lock:
            if key in self._query_cache:
                timestamp, value = self._query_cache[key]
                age_s = (datetime.utcnow() - timestamp).total_seconds()
                
                if age_s < self.cache_ttl_s:
                    self._cache_hits += 1
                    return value
                else:
                    del self._query_cache[key]
        
        return None
    
    def _cache_result(self, key: str, value: Any) -> None:
        """Cache query result."""
        if not self.cache_enabled:
            return
        
        with self._cache_lock:
            self._query_cache[key] = (datetime.utcnow(), value)
    
    def clear_cache(self) -> None:
        """Clear query cache."""
        with self._cache_lock:
            self._query_cache.clear()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        total = self._query_count + self._cache_hits
        hit_rate = 100 * self._cache_hits / total if total > 0 else 0
        
        return {
            "total_queries": self._query_count,
            "cache_hits": self._cache_hits,
            "hit_rate_pct": hit_rate,
            "cached_items": len(self._query_cache),
        }
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def _geographic_to_grid_indices(self, lat: float, lon: float, model: Any) -> Tuple[int, int]:
        """Convert geographic coordinates to grid indices."""
        if not hasattr(model, 'grid_bounds'):
            return 0, 0
        
        lat_min, lat_max, lon_min, lon_max = model.grid_bounds
        grid_shape = getattr(model, 'grid_size', 512)
        
        idx_lat = int((lat - lat_min) / (lat_max - lat_min) * (grid_shape - 1))
        idx_lon = int((lon - lon_min) / (lon_max - lon_min) * (grid_shape - 1))
        
        idx_lat = np.clip(idx_lat, 0, grid_shape - 1)
        idx_lon = np.clip(idx_lon, 0, grid_shape - 1)
        
        return idx_lat, idx_lon
    
    def _compute_slope(self, elevation: np.ndarray, i: int, j: int) -> float:
        """Compute local slope from elevation grid."""
        if i == 0 or i == elevation.shape[0] - 1 or j == 0 or j == elevation.shape[1] - 1:
            return 0.0
        
        dy = (elevation[i+1, j] - elevation[i-1, j]) / 2.0
        dx = (elevation[i, j+1] - elevation[i, j-1]) / 2.0
        
        slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
        return float(np.degrees(slope_rad))


# =============================================================================
# STANDALONE FUNCTIONS (for simpler use cases)
# =============================================================================

# Global API instance (lazily initialized)
_global_api: Optional[UnifiedPerceptionAPI] = None


def initialize_global_api(
    terrain_model=None,
    wind_model=None,
    obstacle_model=None,
    threat_model=None,
    fusion_model=None,
) -> UnifiedPerceptionAPI:
    """Initialize global unified perception API instance."""
    global _global_api
    _global_api = UnifiedPerceptionAPI(
        terrain_model=terrain_model,
        wind_model=wind_model,
        obstacle_model=obstacle_model,
        threat_model=threat_model,
        fusion_model=fusion_model,
    )
    return _global_api


def get_fused_perception(
    latitude: float,
    longitude: float,
    altitude_m: float,
) -> FusedPerceptionQuery:
    """Convenience function to query fused perception using global API."""
    global _global_api
    if _global_api is None:
        _global_api = UnifiedPerceptionAPI()
    
    return _global_api.query_fused_perception(latitude, longitude, altitude_m)


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.DEBUG)
    
    api = UnifiedPerceptionAPI()
    
    # Query at specific location
    query = api.query_fused_perception(40.7128, -74.0060, 300.0)
    print(f"Perception at NYC: {query}")
    print(f"  Cache stats: {api.get_cache_stats()}")
