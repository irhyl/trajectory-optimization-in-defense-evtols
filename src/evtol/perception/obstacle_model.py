"""
Obstacle Detection for eVTOL Trajectory Optimization.

Models building detection, clearance requirements, and landing zone identification
for collision avoidance and emergency landing planning.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import numpy as np
from scipy.ndimage import gaussian_filter, label, find_objects
from shapely.geometry import Polygon, box


@dataclass
class LandingZone:
    """Represents a potential landing zone."""
    zone_id: str
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    center_lat: float
    center_lon: float
    area_m2: float
    slope_deg: float
    feasibility_score: float  # 0-1, how suitable for landing


@dataclass
class ObstacleMetadata:
    """Metadata for obstacle detection."""
    resolution_m: float = 30.0
    coverage_north: float = 13.0
    coverage_south: float = 12.8
    coverage_east: float = 77.7
    coverage_west: float = 77.5
    generated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    processing_version: str = "1.0"


class ObstacleDetectionModel:
    """
    Models building and obstacle detection for collision avoidance.
    
    Generates:
    - Building presence masks
    - Building height maps
    - Clearance maps (minimum safe altitude)
    - Landing zone candidates
    """
    
    def __init__(self, grid_size: int = 1667, coverage_bounds: Optional[Tuple] = None):
        """
        Initialize obstacle detection model.
        
        Args:
            grid_size: Number of cells per dimension
            coverage_bounds: (north, south, east, west) in decimal degrees
        """
        self.grid_size = grid_size
        self.coverage_bounds = coverage_bounds or (13.0, 12.8, 77.7, 77.5)
        self.resolution_m = 30.0
        
        self.building_mask = None  # Binary: 0=no building, 1=building
        self.building_height = None  # Height above ground (meters)
        self.clearance_map = None  # Minimum safe altitude (meters)
        self.landing_zones: List[LandingZone] = []
        self.metadata = ObstacleMetadata()
        
        self.clearance_buffer_m = 50.0  # Safety buffer above obstacles
    
    def generate_realistic_obstacles(self, building_density: float = 0.08,
                                    urban_center: Tuple[int, int] = (700, 800)) -> None:
        """
        Generate realistic building and obstacle distribution.
        
        Args:
            building_density: Fraction of grid cells with buildings (0-1)
            urban_center: Grid coordinates of urban center (lat_idx, lon_idx)
        """
        # Initialize arrays
        self.building_mask = np.zeros((self.grid_size, self.grid_size))
        self.building_height = np.zeros((self.grid_size, self.grid_size))
        
        # Create urban cluster centered at specified location
        lat = np.arange(self.grid_size)
        lon = np.arange(self.grid_size)
        lat_grid, lon_grid = np.meshgrid(lon, lat, indexing='ij')
        
        # Distance from urban center
        distance_from_center = np.sqrt(
            (lat_grid - urban_center[0])**2 + (lon_grid - urban_center[1])**2
        )
        
        # Urban density decreases with distance (Gaussian falloff)
        urban_density_map = building_density * np.exp(
            -(distance_from_center**2) / (300**2)
        )
        
        # Random building placement based on density
        random_map = np.random.rand(self.grid_size, self.grid_size)
        self.building_mask = (random_map < urban_density_map).astype(int)
        
        # Building heights: taller near center, shorter at edges
        height_base = 15.0  # Base building height (meters)
        height_variation = 20.0 * urban_density_map
        
        # Random height variation
        self.building_height = np.where(
            self.building_mask > 0,
            height_base + height_variation * np.random.rand(self.grid_size, self.grid_size),
            0
        )
        
        # Smooth building heights for realism
        self.building_height = gaussian_filter(self.building_height, sigma=1.5)
        
        # Ensure buildings only where mask is 1
        self.building_height = self.building_height * self.building_mask
        
        # Compute clearance map
        self._compute_clearance_map()
        
        # Identify landing zones
        self._identify_landing_zones()
    
    def _compute_clearance_map(self) -> None:
        """Compute minimum safe altitude map."""
        if self.building_height is None:
            raise ValueError("Building height not computed. Call generate_realistic_obstacles() first.")
        
        # Clearance = building height + safety buffer
        self.clearance_map = self.building_height + self.clearance_buffer_m
        
        # Smooth for safer transitions
        self.clearance_map = gaussian_filter(self.clearance_map, sigma=2)
    
    def _identify_landing_zones(self, min_zone_size: float = 2500.0,
                               max_slope: float = 5.0) -> None:
        """
        Identify suitable landing zones (flat, obstacle-free areas).
        
        Args:
            min_zone_size: Minimum landing zone area (m²)
            max_slope: Maximum acceptable slope (degrees)
        """
        self.landing_zones = []
        
        # Find connected components of clear area
        clear_mask = (self.clearance_map < 100).astype(int)  # < 100m obstacle height
        labeled, num_features = label(clear_mask)
        
        # Get bounding boxes of each component
        slices = find_objects(labeled)
        
        zone_id = 1
        for slice_obj in slices:
            if slice_obj is None:
                continue
            
            lat_slice, lon_slice = slice_obj
            component_mask = labeled[lat_slice, lon_slice]
            
            # Count cells in this component
            component_size = np.sum(component_mask > 0)
            area_m2 = component_size * (self.resolution_m ** 2)
            
            # Only consider zones above minimum size
            if area_m2 < min_zone_size:
                continue
            
            # Calculate zone properties
            lat_indices = np.arange(lat_slice.start, lat_slice.stop)
            lon_indices = np.arange(lon_slice.start, lon_slice.stop)
            zone_mask = (component_mask > 0)
            
            # Mean slope in zone
            clearance_in_zone = self.clearance_map[lat_slice, lon_slice][zone_mask]
            slope = np.std(clearance_in_zone) / (self.resolution_m + 1e-6) * 180 / np.pi
            
            # Zone is suitable if slope is acceptable and no tall obstacles
            max_height_in_zone = np.max(clearance_in_zone)
            if slope <= max_slope and max_height_in_zone < 150:
                # Center of zone
                center_lat = np.mean(lat_indices[np.any(zone_mask, axis=1)])
                center_lon = np.mean(lon_indices[np.any(zone_mask, axis=0)])
                
                # Feasibility score: 0-1
                # Higher if flatter, smaller obstacles
                feasibility = 1.0 - (slope / 45.0) - (max_height_in_zone / 300.0)
                feasibility = np.clip(feasibility, 0.2, 1.0)
                
                zone = LandingZone(
                    zone_id=f"LZ_{zone_id:03d}",
                    lat_min=float(lat_indices[0]),
                    lat_max=float(lat_indices[-1]),
                    lon_min=float(lon_indices[0]),
                    lon_max=float(lon_indices[-1]),
                    center_lat=float(center_lat),
                    center_lon=float(center_lon),
                    area_m2=float(area_m2),
                    slope_deg=float(slope),
                    feasibility_score=float(feasibility)
                )
                
                self.landing_zones.append(zone)
                zone_id += 1
    
    def get_obstacle_at_location(self, lat_idx: float, lon_idx: float) -> Dict[str, float]:
        """
        Get obstacle information at a location.
        
        Args:
            lat_idx: Latitude grid index
            lon_idx: Longitude grid index
            
        Returns:
            Dictionary with obstacle metrics
        """
        if self.building_height is None:
            self.generate_realistic_obstacles()
        
        lat_idx = int(np.clip(lat_idx, 0, self.grid_size - 1))
        lon_idx = int(np.clip(lon_idx, 0, self.grid_size - 1))
        
        return {
            'building_present': bool(self.building_mask[lat_idx, lon_idx]),
            'building_height_m': float(self.building_height[lat_idx, lon_idx]),
            'min_safe_altitude_m': float(self.clearance_map[lat_idx, lon_idx]),
            'clearance_available_m': float(
                self.clearance_map[lat_idx, lon_idx] - self.clearance_buffer_m
            ),
        }
    
    def find_nearest_landing_zone(self, lat_idx: float, lon_idx: float,
                                 max_distance: float = 5000.0) -> Optional[Dict]:
        """
        Find nearest suitable landing zone to a location.
        
        Args:
            lat_idx: Current latitude grid index
            lon_idx: Current longitude grid index
            max_distance: Maximum search distance (meters)
            
        Returns:
            Landing zone info or None if no zone found
        """
        if not self.landing_zones:
            return None
        
        max_distance_cells = max_distance / self.resolution_m
        
        # Calculate distances to all zones
        distances = []
        for zone in self.landing_zones:
            dist = np.sqrt(
                (lat_idx - zone.center_lat)**2 + (lon_idx - zone.center_lon)**2
            ) * self.resolution_m
            distances.append(dist)
        
        # Find closest zone within range
        min_dist_idx = np.argmin(distances)
        min_dist = distances[min_dist_idx]
        
        if min_dist <= max_distance:
            zone = self.landing_zones[min_dist_idx]
            return {
                'zone_id': zone.zone_id,
                'distance_m': float(min_dist),
                'center_lat': zone.center_lat,
                'center_lon': zone.center_lon,
                'area_m2': zone.area_m2,
                'slope_deg': zone.slope_deg,
                'feasibility_score': zone.feasibility_score,
            }
        
        return None
    
    def validate_obstacle_data(self) -> Dict[str, bool]:
        """Validate obstacle maps."""
        if self.building_height is None:
            return {'obstacle_maps_generated': False}
        
        checks = {
            'no_nan_values': not np.any(np.isnan(self.clearance_map)),
            'positive_clearance': np.all(self.clearance_map >= 0),
            'realistic_heights': np.all(self.building_height <= 300),
            'landing_zones_identified': len(self.landing_zones) > 0,
        }
        return checks
    
    def get_statistics(self) -> Dict[str, float]:
        """Get obstacle statistics."""
        if self.building_height is None:
            return {}
        
        building_count = int(np.sum(self.building_mask))
        total_cells = self.building_mask.size
        
        return {
            'building_coverage_pct': float(100 * building_count / total_cells),
            'mean_building_height_m': float(np.mean(self.building_height[self.building_mask > 0])),
            'max_building_height_m': float(np.max(self.building_height)),
            'mean_clearance_m': float(np.mean(self.clearance_map)),
            'building_count': float(building_count),
            'landing_zones_count': float(len(self.landing_zones)),
            'total_landing_area_m2': float(sum(z.area_m2 for z in self.landing_zones)),
        }
    
    def to_dict(self) -> Dict:
        """Export obstacle model as dictionary."""
        return {
            'building_mask': self.building_mask,
            'building_height': self.building_height,
            'clearance_map': self.clearance_map,
            'landing_zones': [
                {
                    'zone_id': z.zone_id,
                    'center_lat': z.center_lat,
                    'center_lon': z.center_lon,
                    'area_m2': z.area_m2,
                    'slope_deg': z.slope_deg,
                    'feasibility_score': z.feasibility_score,
                }
                for z in self.landing_zones
            ],
            'metadata': {
                'resolution_m': self.metadata.resolution_m,
                'coverage_bounds': self.coverage_bounds,
                'generated_at': self.metadata.generated_at,
                'processing_version': self.metadata.processing_version,
            }
        }
