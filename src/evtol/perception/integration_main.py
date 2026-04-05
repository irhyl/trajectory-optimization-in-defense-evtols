"""
Perception API Integration for Main Orchestrator

This module provides integration functions to connect the unified perception API
with the main flight control system (planning, control, vehicle layers).

Enables:
- Real-time perception queries in flight loop
- Risk-aware trajectory planning
- Threat-triggered evasive maneuvers
- Altitude/clearance constraints
"""

import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger("PerceptionIntegration")


@dataclass
class PerceptionQueryResult:
    """Result from unified perception query."""
    risk_score: float
    feasibility_score: float
    energy_cost_score: float
    threat_probability: float
    threat_type: str
    wind_speed_mps: float
    terrain_clearance_m: float
    landing_zone_available: bool
    dominant_hazard: str  # "threat", "wind", "terrain", "obstacle", "none"


class PerceptionIntegrationManager:
    """
    Orchestrates perception API usage across all system layers.
    
    Handles:
    - API initialization and model loading
    - Query batching and caching
    - Integration with planning/control/vehicle
    - Performance monitoring
    """
    
    def __init__(self, perception_api, enable_logging: bool = True):
        """
        Initialize perception integration.
        
        Args:
            perception_api: UnifiedPerceptionAPI instance
            enable_logging: Enable detailed logging
        """
        self.api = perception_api
        self.enable_logging = enable_logging
        self.query_count = 0
        self.cache_hit_count = 0
        
        if enable_logging:
            logger.info("Perception integration initialized")
    
    def query_vehicle_state_context(self, vehicle_state: Dict) -> PerceptionQueryResult:
        """
        Query perception for current vehicle state.
        
        Args:
            vehicle_state: {latitude, longitude, altitude_m, ...}
            
        Returns:
            PerceptionQueryResult with all relevant context
        """
        lat = vehicle_state.get("latitude", 0)
        lon = vehicle_state.get("longitude", 0)
        alt = vehicle_state.get("altitude_m", 100)
        
        # Query unified perception
        fused = self.api.query_fused_perception(lat, lon, alt)
        wind = self.api.query_wind(alt)
        obstacle = self.api.query_obstacle(lat, lon, alt)
        threat = self.api.query_threat(lat, lon, alt)
        
        self.query_count += 1
        
        # Determine dominant hazard
        # Use actual attribute names from the query dataclasses:
        #   FusedPerceptionQuery: risk_score, feasibility_score, energy_cost_score
        #   WindQuery:            turbulence_intensity, wind_speed_mps
        #   ObstacleQuery:        nearest_obstacle_distance_m, clearance_m, landing_zone_available
        #   ThreatQuery:          threat_probability, engagement_probability, threat_category
        components = {
            "threat": threat.threat_probability if threat else 0.0,
            "terrain": fused.risk_score,          # fused risk incorporates terrain
            "wind": wind.turbulence_intensity if wind else 0.0,
            "obstacle": 1.0 / max(obstacle.nearest_obstacle_distance_m, 1.0) if obstacle else 0.0,
        }
        dominant = max(components, key=components.get)
        if components[dominant] < 0.3:
            dominant = "none"

        result = PerceptionQueryResult(
            risk_score=fused.risk_score,
            feasibility_score=fused.feasibility_score,
            energy_cost_score=fused.energy_cost_score,
            threat_probability=threat.threat_probability if threat else 0.0,
            threat_type=threat.threat_category if threat else "none",
            wind_speed_mps=wind.wind_speed_mps if wind else 0.0,
            terrain_clearance_m=obstacle.clearance_m if obstacle else 1000.0,
            landing_zone_available=obstacle.landing_zone_available if obstacle else False,
            dominant_hazard=dominant,
        )
        
        return result
    
    def check_threat_and_recommend_evasion(self, vehicle_state: Dict) -> Tuple[bool, Dict]:
        """
        Check for immediate threats and recommend evasion.
        
        Args:
            vehicle_state: Current vehicle state
            
        Returns:
            (should_evade: bool, evasion_recommendation: Dict)
        """
        lat = vehicle_state.get("latitude", 0)
        lon = vehicle_state.get("longitude", 0)
        alt = vehicle_state.get("altitude_m", 100)
        current_heading = vehicle_state.get("heading_deg", 0)
        
        threat = self.api.query_threat(lat, lon, alt)
        
        if not threat or threat.engagement_probability < 0.6:
            return False, {}

        # Threat detected — recommend evasion perpendicular to bearing
        bearing = threat.threat_bearing_deg  # None if unknown
        evasion_dir = (bearing + 90) % 360 if bearing is not None else current_heading

        recommendation = {
            "evasion_required": True,
            "threat_type": threat.threat_category,
            "engagement_probability": threat.engagement_probability,
            "recommended_heading": evasion_dir,
            "recommended_altitude_change": 200,  # Climb 200m
            "recommended_speed_change": 5,  # Increase speed 5 m/s
            "urgency": "HIGH" if threat.engagement_probability > 0.8 else "MEDIUM",
        }
        
        if self.enable_logging:
            rng = f"{threat.threat_range_m:.0f}m" if threat.threat_range_m is not None else "unknown"
            logger.warning(f"THREAT DETECTED: {threat.threat_category} @ range {rng}, "
                           f"engagement prob {threat.engagement_probability:.2f}")
        
        return True, recommendation
    
    def get_planning_constraints(self, mission_waypoints: List[Tuple], 
                                current_state: Dict, num_samples: int = 20) -> Dict:
        """
        Sample perception at waypoints to get planning constraints.
        
        Args:
            mission_waypoints: List of (lat, lon) waypoints
            current_state: Current vehicle state (for altitude)
            num_samples: Number of waypoints to sample
            
        Returns:
            Dict with risk map and recommendations
        """
        altitude = current_state.get("altitude_m", 300)
        
        # Sample perceptions at waypoints
        waypoint_risks = []
        for i, (lat, lon) in enumerate(mission_waypoints[:num_samples]):
            fused = self.api.query_fused_perception(lat, lon, altitude)
            waypoint_risks.append({
                "waypoint_idx": i,
                "position": (lat, lon),
                "risk_score": fused.risk_score,
                "feasibility_score": fused.feasibility_score,
                "terrain_fraction": fused.terrain_fraction,
            })
        
        # Identify risky sections
        high_risk_waypoints = [w for w in waypoint_risks if w["risk_score"] > 0.7]
        avg_risk = sum(w["risk_score"] for w in waypoint_risks) / len(waypoint_risks)
        
        constraints = {
            "avg_risk_score": avg_risk,
            "num_high_risk_sections": len(high_risk_waypoints),
            "high_risk_waypoints": high_risk_waypoints,
            "recommend_altitude_increase": avg_risk > 0.5,
            "recommend_replanning": len(high_risk_waypoints) > 5,
        }
        
        return constraints
    
    def get_control_wind_compensation(self, altitude_m: float) -> Dict:
        """
        Get wind compensation values for control loop.
        
        Args:
            altitude_m: Current altitude
            
        Returns:
            Wind compensation data for attitude control
        """
        wind = self.api.query_wind(altitude_m)
        
        if not wind:
            return {
                "wind_north_mps": 0,
                "wind_east_mps": 0,
                "wind_down_mps": 0,
                "wind_speed_mps": 0,
                "turbulence_intensity": 0,
                "requires_damping_increase": False,
            }
        
        requires_damping = wind.turbulence_intensity > 0.3
        
        return {
            "wind_north_mps": wind.wind_north_mps,
            "wind_east_mps": wind.wind_east_mps,
            "wind_down_mps": wind.wind_vertical_mps,
            "wind_speed_mps": wind.wind_speed_mps,
            "wind_direction_deg": wind.wind_direction_deg,
            "turbulence_intensity": wind.turbulence_intensity,
            "requires_damping_increase": requires_damping,
            "gain_adjustment_factor": 0.8 if requires_damping else 1.0,  # Reduce gains in turbulence
        }
    
    def check_clearance_constraints(self, lat: float, lon: float, alt: float,
                                   min_clearance_m: float = 50) -> Tuple[bool, float]:
        """
        Check if altitude is safe given terrain/obstacles.
        
        Args:
            lat, lon: Position
            alt: Current altitude MSL
            min_clearance_m: Minimum required clearance
            
        Returns:
            (is_safe: bool, recommended_altitude_m: float)
        """
        terrain = self.api.query_terrain(lat, lon)
        obstacle = self.api.query_obstacle(lat, lon, alt)
        
        if not terrain or not obstacle:
            return True, alt
        
        # Minimum altitude = terrain + obstacles + clearance margin
        # ObstacleQuery has nearest_obstacle_height_m, not nearest_building_height_m
        bldg_height = obstacle.nearest_obstacle_height_m
        min_alt = terrain.elevation_m + bldg_height + min_clearance_m

        is_safe = alt > min_alt
        recommended_alt = max(alt, min_alt)

        if not is_safe and self.enable_logging:
            logger.warning(
                f"Clearance violation: alt={alt:.0f}m, required={min_alt:.0f}m "
                f"(terrain={terrain.elevation_m:.0f}m, obstacle_height={bldg_height:.0f}m)"
            )
        
        return is_safe, recommended_alt
    
    def check_landing_feasibility(self, landing_lat: float, landing_lon: float) -> Tuple[bool, str]:
        """
        Check if landing location is suitable.
        
        Args:
            landing_lat, landing_lon: Landing coordinates
            
        Returns:
            (is_feasible: bool, reason: str)
        """
        obstacle = self.api.query_obstacle(landing_lat, landing_lon, 0)
        terrain = self.api.query_terrain(landing_lat, landing_lon)
        
        if not obstacle or not terrain:
            return False, "Unable to query landing zone"
        
        if not obstacle.landing_zone_available:
            return False, "No suitable landing zone detected"

        # ObstacleQuery has nearest_obstacle_distance_m (not nearest_building_distance_m)
        if obstacle.nearest_obstacle_distance_m < 100:
            return False, (
                f"Too close to obstacles ({obstacle.nearest_obstacle_distance_m:.0f}m < 100m)"
            )

        # TerrainQuery may not expose slope_deg; use clearance as a proxy
        if obstacle.clearance_m < 20:
            return False, f"Insufficient terrain clearance ({obstacle.clearance_m:.1f}m)"
        
        return True, "Landing feasible"
    
    def get_trajectory_risk_profile(self, waypoints: List[Tuple], 
                                   altitudes: List[float]) -> Dict:
        """
        Get risk profile for entire trajectory.
        
        Args:
            waypoints: List of (lat, lon) waypoints
            altitudes: List of altitudes corresponding to waypoints
            
        Returns:
            Dict with trajectory risk analysis
        """
        if len(waypoints) != len(altitudes):
            raise ValueError("waypoints and altitudes must have same length")
        
        results = self.api.query_trajectory(waypoints, altitudes)
        
        if not results:
            return {"error": "No trajectory results"}
        
        risk_scores = [r.risk_score for r in results]
        feasibility_scores = [r.feasibility_score for r in results]
        
        analysis = {
            "trajectory_length": len(waypoints),
            "avg_risk_score": sum(risk_scores) / len(risk_scores),
            "max_risk_score": max(risk_scores),
            "avg_feasibility_score": sum(feasibility_scores) / len(feasibility_scores),
            "num_high_risk_segments": sum(1 for r in risk_scores if r > 0.7),
            "trajectory_feasible": min(feasibility_scores) > 0.4,
        }
        
        return analysis
    
    def get_statistics(self) -> Dict:
        """Get integration statistics."""
        return {
            "total_queries": self.query_count,
            "cache_hits": self.cache_hit_count,
            "hit_rate": self.cache_hit_count / self.query_count if self.query_count > 0 else 0,
        }


def initialize_perception_for_main(enable_cache: bool = True, 
                                   cache_ttl: int = 30) -> PerceptionIntegrationManager:
    """
    Initialize perception system for main orchestrator.
    
    This is the primary entry point for perception integration.
    
    Args:
        enable_cache: Enable caching for performance
        cache_ttl: Cache time-to-live in seconds
        
    Returns:
        PerceptionIntegrationManager ready for use
        
    Example:
        >>> perc_mgr = initialize_perception_for_main()
        >>> query = perc_mgr.query_vehicle_state_context({
        ...     "latitude": 40.7128,
        ...     "longitude": -74.0060,
        ...     "altitude_m": 300,
        ... })
        >>> print(f"Risk: {query.risk_score:.2f}")
    """
    try:
        # Use relative imports — avoids depending on sys.path having 'src' as root
        from .terrain import TerrainFieldModel
        from .wind import WindFieldModel
        from . import ObstacleDetectionModel, ThreatAssessmentModel, FusedIntelligenceModel
        from .perception_api import UnifiedPerceptionAPI
        
        logger.info("Initializing perception models...")
        
        # Initialize individual models
        terrain = TerrainFieldModel(grid_size=512)
        wind = WindFieldModel(grid_size=256)
        obstacle = ObstacleDetectionModel(grid_size=256)
        threat = ThreatAssessmentModel(grid_size=256)
        fusion = FusedIntelligenceModel()
        
        # Create unified API
        api = UnifiedPerceptionAPI(
            terrain_model=terrain,
            wind_model=wind,
            obstacle_model=obstacle,
            threat_model=threat,
            fusion_model=fusion,
            enable_cache=enable_cache,
            cache_ttl_seconds=cache_ttl,
        )
        
        logger.info("✓ Perception models initialized successfully")
        
        # Create integration manager
        manager = PerceptionIntegrationManager(api, enable_logging=True)
        
        logger.info("✓ Perception integration manager ready")
        
        return manager
        
    except ImportError as e:
        logger.error(f"Failed to import perception models: {e}")
        logger.warning("Perception system will operate in degraded mode")
        return None
    except Exception as e:
        logger.error(f"Error initializing perception: {e}")
        return None


# Integration functions for each layer

def integrate_perception_with_planning(planning_layer, perception_mgr: PerceptionIntegrationManager,
                                      mission_waypoints: List[Tuple], current_state: Dict):
    """
    Integrate perception with planning layer.
    
    Provides planning layer with:
    - Risk scores for trajectory optimization
    - Constraints and recommendations
    
    Args:
        planning_layer: PlanningLayer instance
        perception_mgr: PerceptionIntegrationManager
        mission_waypoints: List of waypoints to evaluate
        current_state: Current vehicle state
    """
    if not perception_mgr:
        logger.warning("Perception not available for planning integration")
        return
    
    # Get planning constraints from perception
    constraints = perception_mgr.get_planning_constraints(mission_waypoints, current_state)
    
    # Store in planning layer for use in replan_trajectory
    planning_layer.perception_constraints = constraints
    planning_layer.perception_mgr = perception_mgr
    
    logger.info(f"Planning layer integrated with perception (avg risk: {constraints['avg_risk_score']:.2f})")


def integrate_perception_with_control(control_layer, perception_mgr: PerceptionIntegrationManager):
    """
    Integrate perception with control layer.
    
    Provides control layer with:
    - Wind compensation for attitude control
    - Threat detection for evasive maneuvers
    - Damping adjustments for turbulence
    
    Args:
        control_layer: ControlLayer instance
        perception_mgr: PerceptionIntegrationManager
    """
    if not perception_mgr:
        logger.warning("Perception not available for control integration")
        return
    
    # Store perception manager in control layer
    control_layer.perception_mgr = perception_mgr
    
    logger.info("Control layer integrated with perception")


def integrate_perception_with_vehicle(vehicle_layer, perception_mgr: PerceptionIntegrationManager):
    """
    Integrate perception with vehicle layer.
    
    Provides vehicle layer with:
    - Altitude constraints based on terrain/obstacles
    - Real-time clearance monitoring
    - Landing feasibility checks
    
    Args:
        vehicle_layer: VehicleLayer instance
        perception_mgr: PerceptionIntegrationManager
    """
    if not perception_mgr:
        logger.warning("Perception not available for vehicle integration")
        return
    
    # Store perception manager in vehicle layer
    vehicle_layer.perception_mgr = perception_mgr
    
    logger.info("Vehicle layer integrated with perception")
