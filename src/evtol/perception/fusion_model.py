"""
Fused Intelligence Layer for eVTOL Perception.

Integrates terrain, wind, threat, and obstacle data into unified
risk scoring, feasibility assessment, and energy cost models.
"""

from dataclasses import dataclass, field
from typing import Tuple, Dict, Optional
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import zoom


def _resize_to_shape(arr: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    """Resize a 2D array to target shape using interpolation."""
    if arr.shape == target_shape:
        return arr
    
    # Calculate zoom factors
    zoom_factors = (target_shape[0] / arr.shape[0], target_shape[1] / arr.shape[1])
    return zoom(arr, zoom_factors, order=1)  # Bilinear interpolation


@dataclass
class FusionMetadata:
    """Metadata for fused intelligence layer."""
    grid_resolution_m: float = 30.0
    coverage_area_m: float = 50000.0
    grid_size: int = 1667
    fusion_version: str = "1.0"
    timestamp: str = "2025-01-15"
    component_weights: Dict[str, float] = field(default_factory=lambda: {
        'threat': 0.30,
        'obstacle': 0.30,
        'terrain': 0.20,
        'wind': 0.20
    })


class FusedIntelligenceModel:
    """Integrate all perception layers into unified risk/feasibility models."""
    
    def __init__(self, terrain_model, wind_model, threat_model, obstacle_model):
        """
        Initialize fusion model with all perception layers.
        
        Args:
            terrain_model: TerrainElevationMap instance
            wind_model: WindFieldModel instance
            threat_model: ThreatAssessmentModel instance
            obstacle_model: ObstacleDetectionModel instance
        """
        self.terrain = terrain_model
        self.wind = wind_model
        self.threat = threat_model
        self.obstacle = obstacle_model
        self.metadata = FusionMetadata()
        
        self.grid_size = obstacle_model.grid_size
        self.grid_resolution = obstacle_model.resolution_m
        
        # Compute fused layers
        self._compute_risk_map()
        self._compute_feasibility_map()
        self._compute_energy_cost_map()
        
    def _compute_risk_map(self) -> None:
        """Compute integrated risk map from all threat/obstacle sources."""
        weights = self.metadata.component_weights
        target_shape = (self.grid_size, self.grid_size)
        
        # Normalize threat map (0-1) and resize to common grid
        threat_raw = self.threat.threat_heatmap
        threat_resized = _resize_to_shape(threat_raw, target_shape)
        threat_normalized = threat_resized / (np.max(threat_resized) + 1e-6)
        
        # Normalize obstacle clearance (invert: high clearance = low risk)
        clearance_raw = self.obstacle.clearance_map
        clearance_resized = _resize_to_shape(clearance_raw, target_shape)
        clearance_normalized = 1.0 - (clearance_resized / (np.max(clearance_resized) + 1e-6))
        
        # Normalize terrain slope (high slope = high risk)
        terrain_raw = self.terrain.elevation
        terrain_resized = _resize_to_shape(terrain_raw, target_shape)
        terrain_slope = np.gradient(terrain_resized, axis=1)
        terrain_normalized = np.abs(terrain_slope) / (np.max(np.abs(terrain_slope)) + 1e-6)
        
        # Wind risk (high wind variability = risk) - use middle altitude layer (100m = index 1)
        wind_speed_2d = self.wind.wind_speed[1, :, :]
        wind_resized = _resize_to_shape(wind_speed_2d, target_shape)
        wind_normalized = wind_resized / (np.max(wind_resized) + 1e-6)
        
        # Fused risk: weighted combination
        self.risk_map = (
            weights['threat'] * threat_normalized +
            weights['obstacle'] * clearance_normalized +
            weights['terrain'] * terrain_normalized +
            weights['wind'] * wind_normalized
        )
        
        # Clip to valid range
        self.risk_map = np.clip(self.risk_map, 0.0, 1.0)
        
    def _compute_feasibility_map(self) -> None:
        """Compute path feasibility map based on all constraints."""
        target_shape = (self.grid_size, self.grid_size)
        feasibility = np.ones(target_shape, dtype=float)
        
        # Resize threat heatmap and remove high-threat areas
        threat_resized = _resize_to_shape(self.threat.threat_heatmap, target_shape)
        threat_penalty = np.where(threat_resized > 0.7, 0.0, 1.0)
        feasibility *= threat_penalty
        
        # Resize obstacle mask and remove obstacles
        obstacle_resized = _resize_to_shape(self.obstacle.building_mask.astype(float), target_shape)
        obstacle_penalty = np.where(obstacle_resized > 0.5, 0.0, 1.0)
        feasibility *= obstacle_penalty
        
        # Resize terrain and penalize high-slope terrain
        terrain_resized = _resize_to_shape(self.terrain.elevation, target_shape)
        terrain_slope = np.abs(np.gradient(terrain_resized, axis=1))
        slope_normalized = terrain_slope / (np.max(terrain_slope) + 1e-6)
        slope_penalty = 1.0 - (0.3 * slope_normalized)
        feasibility *= slope_penalty
        
        # Resize wind shear and penalize extreme wind shear
        wind_shear_resized = _resize_to_shape(self.wind.wind_shear, target_shape)
        shear_range = np.max(wind_shear_resized) - np.min(wind_shear_resized)
        shear_normalized = (wind_shear_resized - np.min(wind_shear_resized)) / (shear_range + 1e-6)
        shear_penalty = 1.0 - (0.2 * shear_normalized)
        feasibility *= shear_penalty
        
        self.feasibility_map = np.clip(feasibility, 0.0, 1.0)
        
    def _compute_energy_cost_map(self) -> None:
        """Compute energy cost map for trajectory optimization."""
        target_shape = (self.grid_size, self.grid_size)
        
        # Resize terrain elevation and compute elevation cost (climb cost)
        terrain_resized = _resize_to_shape(self.terrain.elevation, target_shape)
        elev_range = np.max(terrain_resized) - np.min(terrain_resized)
        dem_normalized = (terrain_resized - np.min(terrain_resized)) / (elev_range + 1e-6)
        elevation_cost = 0.3 * dem_normalized
        
        # Resize wind speed and compute wind energy cost
        wind_speed_2d = self.wind.wind_speed[1, :, :]
        wind_resized = _resize_to_shape(wind_speed_2d, target_shape)
        wind_cost = 0.2 * (wind_resized / 10.0)
        
        # Resize clearance map and compute obstacle avoidance cost
        clearance_resized = _resize_to_shape(self.obstacle.clearance_map, target_shape)
        clearance_cost = 0.3 * (clearance_resized / 150.0)
        
        # Compute terrain roughness cost
        terrain_roughness = np.abs(np.gradient(terrain_resized, axis=0)) + \
                           np.abs(np.gradient(terrain_resized, axis=1))
        roughness_normalized = terrain_roughness / (np.max(terrain_roughness) + 1e-6)
        roughness_cost = 0.2 * roughness_normalized
        
        self.energy_cost_map = elevation_cost + wind_cost + clearance_cost + roughness_cost
        self.energy_cost_map = np.clip(self.energy_cost_map, 0.0, 10.0)
        
    def validate_fused_data(self) -> Dict[str, bool]:
        """
        Validate fused intelligence maps.
        
        Returns:
            Dict with validation results
        """
        checks = {}
        
        # Check risk map bounds
        checks['risk_in_range'] = (np.all(self.risk_map >= 0.0) and 
                                   np.all(self.risk_map <= 1.0))
        
        # Check feasibility bounds
        checks['feasibility_in_range'] = (np.all(self.feasibility_map >= 0.0) and 
                                         np.all(self.feasibility_map <= 1.0))
        
        # Check for NaN values
        checks['no_nan_risk'] = not np.any(np.isnan(self.risk_map))
        checks['no_nan_feasibility'] = not np.any(np.isnan(self.feasibility_map))
        checks['no_nan_energy'] = not np.any(np.isnan(self.energy_cost_map))
        
        # Check data consistency
        checks['consistent_shapes'] = (self.risk_map.shape == self.feasibility_map.shape ==
                                      self.energy_cost_map.shape)
        
        return checks
    
    def get_path_cost(self, lat_indices: np.ndarray, lon_indices: np.ndarray,
                      altitude_m: float = 100.0) -> float:
        """
        Calculate total cost for a given path.
        
        Args:
            lat_indices: Array of latitude indices
            lon_indices: Array of longitude indices
            altitude_m: Flight altitude in meters
            
        Returns:
            Total path cost (normalized 0-10)
        """
        if len(lat_indices) == 0:
            return 0.0
        
        # Sample risk along path
        path_risk = []
        for lat_idx, lon_idx in zip(lat_indices, lon_indices):
            if 0 <= lat_idx < self.grid_size and 0 <= lon_idx < self.grid_size:
                # Risk adjusted by altitude vs clearance
                clearance = self.obstacle.clearance_map[int(lat_idx), int(lon_idx)]
                altitude_penalty = 0.0 if altitude_m > clearance else 1.0
                
                path_risk.append(self.risk_map[int(lat_idx), int(lon_idx)] + 
                               altitude_penalty * 0.5)
        
        if not path_risk:
            return 10.0  # Invalid path
        
        # Sum of risk + energy cost
        risk_cost = np.mean(path_risk) * 3.0
        energy_cost = np.sum([self.energy_cost_map[int(lat_idx), int(lon_idx)]
                             for lat_idx, lon_idx in zip(lat_indices, lon_indices)]) / len(lat_indices)
        
        return float(np.clip(risk_cost + energy_cost, 0.0, 10.0))
    
    def get_fused_query(self, lat_deg: float, lon_deg: float,
                       altitude_m: float = 100.0) -> Dict[str, float]:
        """
        Query fused intelligence at a specific location.
        
        Args:
            lat_deg: Latitude in degrees
            lon_deg: Longitude in degrees
            altitude_m: Altitude in meters
            
        Returns:
            Dict with risk, feasibility, energy cost scores
        """
        # Convert to grid indices
        lat_idx = int((lat_deg + 25.0) * 1667 / 50.0)
        lon_idx = int((lon_deg + 25.0) * 1667 / 50.0)
        
        # Bounds check
        if not (0 <= lat_idx < self.grid_size and 0 <= lon_idx < self.grid_size):
            return {
                'risk_score': 1.0,
                'feasibility_score': 0.0,
                'energy_cost': 10.0,
                'clearance_m': 0.0,
                'wind_speed_ms': 0.0,
                'threat_probability': 1.0,
                'valid': False
            }
        
        # Get terrain elevation
        terrain_elevation = float(self.terrain.dem[lat_idx, lon_idx])
        clearance_required = float(self.obstacle.clearance_map[lat_idx, lon_idx])
        
        # Altitude feasibility
        altitude_feasible = altitude_m >= clearance_required
        altitude_penalty = 0.0 if altitude_feasible else (clearance_required - altitude_m) / 100.0
        
        # Query each layer
        threat_prob = float(self.threat.get_threat_at_location(lat_idx, lon_idx, altitude_m))
        wind_speed = float(np.linalg.norm(self.wind.wind_velocity[:,:,1][lat_idx, lon_idx]))
        
        # Compute adjusted risk
        adjusted_risk = float(self.risk_map[lat_idx, lon_idx]) + altitude_penalty
        adjusted_feasibility = float(self.feasibility_map[lat_idx, lon_idx]) * \
                             (1.0 if altitude_feasible else 0.5)
        
        return {
            'risk_score': float(np.clip(adjusted_risk, 0.0, 1.0)),
            'feasibility_score': float(np.clip(adjusted_feasibility, 0.0, 1.0)),
            'energy_cost': float(self.energy_cost_map[lat_idx, lon_idx]),
            'clearance_m': clearance_required,
            'wind_speed_ms': wind_speed,
            'threat_probability': threat_prob,
            'terrain_elevation_m': terrain_elevation,
            'valid': True
        }
    
    def get_statistics(self) -> Dict[str, float]:
        """
        Get summary statistics for fused intelligence.
        
        Returns:
            Dict with statistical summaries
        """
        return {
            'mean_risk': float(np.mean(self.risk_map)),
            'max_risk': float(np.max(self.risk_map)),
            'min_risk': float(np.min(self.risk_map)),
            'mean_feasibility': float(np.mean(self.feasibility_map)),
            'high_risk_area_pct': float(100.0 * np.sum(self.risk_map > 0.7) / 
                                       self.risk_map.size),
            'feasible_area_pct': float(100.0 * np.sum(self.feasibility_map > 0.5) / 
                                      self.feasibility_map.size),
            'mean_energy_cost': float(np.mean(self.energy_cost_map)),
            'max_energy_cost': float(np.max(self.energy_cost_map)),
        }
    
    def to_dict(self) -> Dict:
        """Export fused model to dictionary."""
        return {
            'risk_map': self.risk_map.tolist(),
            'feasibility_map': self.feasibility_map.tolist(),
            'energy_cost_map': self.energy_cost_map.tolist(),
            'metadata': {
                'grid_resolution_m': self.metadata.grid_resolution_m,
                'grid_size': self.metadata.grid_size,
                'coverage_area_m': self.metadata.coverage_area_m,
                'fusion_version': self.metadata.fusion_version,
                'timestamp': self.metadata.timestamp,
                'component_weights': self.metadata.component_weights,
            },
            'statistics': self.get_statistics(),
        }
