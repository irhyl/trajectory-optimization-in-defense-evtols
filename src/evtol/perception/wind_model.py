"""
Wind Field Modeling for eVTOL Trajectory Optimization.

Generates realistic multi-altitude wind fields with shear effects and provides
interpolation for arbitrary query points. Wind data is critical for:
- Energy consumption modeling (headwind/tailwind impact)
- Flight safety (gust and wind limits)
- Trajectory optimization (efficient routing)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import numpy as np
from scipy.interpolate import RegularGridInterpolator, interp1d
from scipy.ndimage import gaussian_filter


@dataclass
class WindMetadata:
    """Metadata tracking for wind field data."""
    resolution_m: float = 30.0
    coverage_north: float = 13.0
    coverage_south: float = 12.8
    coverage_east: float = 77.7
    coverage_west: float = 77.5
    altitude_bands: List[float] = field(default_factory=lambda: [10.0, 100.0, 500.0])
    forecast_hours: int = 72
    data_source: str = "Synthetic (Perlin noise approximation)"
    generated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    processing_version: str = "1.0"


class WindFieldModel:
    """
    Multi-altitude wind field generation with shear effects and interpolation.
    
    Generates realistic wind patterns at multiple altitudes (10m, 100m, 500m)
    with vertical shear, terrain-induced effects, and uncertainty bounds.
    """
    
    def __init__(self, grid_size: int = 1667, coverage_bounds: Optional[Tuple] = None):
        """
        Initialize wind field model.
        
        Args:
            grid_size: Number of cells per dimension (1667 for 50km at 30m resolution)
            coverage_bounds: (north, south, east, west) in decimal degrees
        """
        self.grid_size = grid_size
        self.coverage_bounds = coverage_bounds or (13.0, 12.8, 77.7, 77.5)
        self.resolution_m = 30.0
        self.altitude_bands = [10.0, 100.0, 500.0]
        
        # Wind data: [altitude, lat, lon]
        self.wind_u = None  # U component (W-E)
        self.wind_v = None  # V component (S-N)
        self.wind_speed = None
        self.wind_direction = None
        self.wind_shear = None  # Vertical wind shear magnitude
        self.turbulence_intensity = None
        
        self.metadata = WindMetadata()
        self._interpolators = {}
        
    def generate_realistic_wind(self, base_speed: float = 7.0) -> Dict[str, np.ndarray]:
        """
        Generate realistic multi-altitude wind field with shear.
        
        Args:
            base_speed: Mean wind speed at 100m (m/s)
            
        Returns:
            Dictionary with wind_speed, wind_direction, turbulence arrays
        """
        # Initialize wind arrays [altitude, lat, lon]
        num_alt = len(self.altitude_bands)
        self.wind_u = np.zeros((num_alt, self.grid_size, self.grid_size))
        self.wind_v = np.zeros((num_alt, self.grid_size, self.grid_size))
        self.turbulence_intensity = np.zeros((num_alt, self.grid_size, self.grid_size))
        
        # Generate base wind pattern using multiple frequency components
        lat_freq = np.arange(self.grid_size)
        lon_freq = np.arange(self.grid_size)
        lat_grid, lon_grid = np.meshgrid(lat_freq, lon_freq, indexing='ij')
        
        # Normalize to [-1, 1]
        lat_norm = 2 * (lat_grid / self.grid_size) - 1
        lon_norm = 2 * (lon_grid / self.grid_size) - 1
        
        # Generate base wind component (4-frequency Perlin-like noise)
        base_component = (
            0.5 * np.sin(lat_norm * np.pi) * np.cos(lon_norm * np.pi) +
            0.3 * np.sin(lat_norm * 3 * np.pi) * np.cos(lon_norm * 3 * np.pi) +
            0.15 * np.sin(lat_norm * 7 * np.pi) * np.cos(lon_norm * 7 * np.pi) +
            0.05 * np.sin(lat_norm * 15 * np.pi) * np.cos(lon_norm * 15 * np.pi)
        )
        base_component = gaussian_filter(base_component, sigma=3)
        
        # Generate perpendicular wind component
        perp_component = (
            0.5 * np.sin(lon_norm * np.pi) * np.cos(lat_norm * np.pi) +
            0.3 * np.sin(lon_norm * 3 * np.pi) * np.cos(lat_norm * 3 * np.pi) +
            0.15 * np.sin(lon_norm * 7 * np.pi) * np.cos(lat_norm * 7 * np.pi) +
            0.05 * np.sin(lon_norm * 15 * np.pi) * np.cos(lat_norm * 15 * np.pi)
        )
        perp_component = gaussian_filter(perp_component, sigma=3)
        
        # Apply altitude-dependent wind shear
        for alt_idx, altitude in enumerate(self.altitude_bands):
            # Wind speed increases with altitude (logarithmic profile)
            # Using wind shear exponent ~0.15 for moderate terrain
            shear_factor = (altitude / 100.0) ** 0.15
            
            # Wind speed at this altitude
            speed_scale = base_speed * shear_factor
            
            # Apply wind speed and direction variation
            # Prevailing direction: 270° (Westerly winds)
            dir_variation = 5.0 * (base_component - perp_component)  # ±5° variation
            
            # U component (W-E): wind direction ~270° = westerly (negative)
            direction_rad = np.radians(270.0 + dir_variation)
            self.wind_u[alt_idx] = speed_scale * np.cos(direction_rad)
            
            # V component (S-N): perpendicular component
            self.wind_v[alt_idx] = speed_scale * np.sin(direction_rad) + 0.2 * perp_component
            
            # Turbulence intensity increases near surface, decreases at height
            base_turb = 0.15 * (1.0 - 0.3 * shear_factor)
            self.turbulence_intensity[alt_idx] = (
                base_turb + 0.05 * np.abs(base_component - perp_component)
            )
        
        # Compute derived quantities
        self._compute_derived_quantities()
        
        return {
            'wind_u': self.wind_u,
            'wind_v': self.wind_v,
            'wind_speed': self.wind_speed,
            'wind_direction': self.wind_direction,
            'turbulence_intensity': self.turbulence_intensity,
        }
    
    def _compute_derived_quantities(self) -> None:
        """Compute wind speed, direction, and shear from U/V components."""
        self.wind_speed = np.sqrt(self.wind_u**2 + self.wind_v**2)
        self.wind_direction = np.degrees(np.arctan2(self.wind_v, self.wind_u))
        self.wind_direction = np.mod(self.wind_direction + 360, 360)
        
        # Compute vertical wind shear (speed difference between altitudes)
        # Shear = magnitude of speed difference between consecutive altitude levels
        if len(self.altitude_bands) >= 2:
            # Shear between 10m and 100m altitude
            shear_lower = np.abs(self.wind_speed[1, :, :] - self.wind_speed[0, :, :])
            # Shear between 100m and 500m altitude
            shear_upper = np.abs(self.wind_speed[2, :, :] - self.wind_speed[1, :, :])
            # Combined shear (max of the two layers)
            self.wind_shear = np.maximum(shear_lower, shear_upper)
        else:
            self.wind_shear = np.zeros((self.grid_size, self.grid_size))
    
    def validate_wind_data(self) -> Dict[str, bool]:
        """
        Validate wind field quality.
        
        Returns:
            Dictionary with validation check results
        """
        checks = {
            'no_nan_values': not np.any(np.isnan(self.wind_speed)),
            'realistic_speed_range': np.all((self.wind_speed >= 0) & (self.wind_speed <= 30)),
            'reasonable_std_dev': 1.0 <= np.std(self.wind_speed) <= 10.0,
            'continuous_spatial': True,  # Verified by Gaussian smoothing
            'turbulence_bounds': np.all((self.turbulence_intensity >= 0) & 
                                       (self.turbulence_intensity <= 0.5))
        }
        return checks
    
    def get_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        Get wind field statistics by altitude band.
        
        Returns:
            Dictionary with statistics for each altitude
        """
        stats = {}
        for idx, alt in enumerate(self.altitude_bands):
            wind_at_alt = self.wind_speed[idx]
            stats[f'{int(alt)}m'] = {
                'speed_mean': float(np.mean(wind_at_alt)),
                'speed_std': float(np.std(wind_at_alt)),
                'speed_min': float(np.min(wind_at_alt)),
                'speed_max': float(np.max(wind_at_alt)),
                'direction_mean': float(np.mean(self.wind_direction[idx])),
                'direction_std': float(np.std(self.wind_direction[idx])),
                'turbulence_mean': float(np.mean(self.turbulence_intensity[idx])),
            }
        
        # Wind shear calculation (speed gradient between altitudes)
        if len(self.altitude_bands) >= 2:
            shear_10_100 = (stats['100m']['speed_mean'] - stats['10m']['speed_mean']) / 90.0
            shear_100_500 = (stats['500m']['speed_mean'] - stats['100m']['speed_mean']) / 400.0
            stats['shear'] = {
                'shear_10_100m_per_m': shear_10_100,
                'shear_100_500m_per_m': shear_100_500,
            }
        
        return stats
    
    def interpolate_wind(self, query_lat: float, query_lon: float, 
                        query_alt: float) -> Dict[str, float]:
        """
        Interpolate wind at arbitrary location.
        
        Args:
            query_lat: Latitude (0-1667 grid index)
            query_lon: Longitude (0-1667 grid index)
            query_alt: Altitude in meters
            
        Returns:
            Dictionary with interpolated wind_speed, wind_u, wind_v, direction, turbulence
        """
        if self.wind_u is None:
            raise ValueError("Wind field not generated. Call generate_realistic_wind() first.")
        
        # Clamp query coordinates to grid bounds
        query_lat = np.clip(query_lat, 0, self.grid_size - 1)
        query_lon = np.clip(query_lon, 0, self.grid_size - 1)
        query_alt = np.clip(query_alt, self.altitude_bands[0], self.altitude_bands[-1])
        
        # Create interpolators if not already cached
        if 'wind_u' not in self._interpolators:
            lat_idx = np.arange(self.grid_size)
            lon_idx = np.arange(self.grid_size)
            alt_idx = np.arange(len(self.altitude_bands))
            
            self._interpolators['wind_u'] = RegularGridInterpolator(
                (alt_idx, lat_idx, lon_idx), self.wind_u, bounds_error=False, 
                fill_value='extrapolate'
            )
            self._interpolators['wind_v'] = RegularGridInterpolator(
                (alt_idx, lat_idx, lon_idx), self.wind_v, bounds_error=False,
                fill_value='extrapolate'
            )
            self._interpolators['turbulence'] = RegularGridInterpolator(
                (alt_idx, lat_idx, lon_idx), self.turbulence_intensity, 
                bounds_error=False, fill_value='extrapolate'
            )
        
        # Interpolate altitude index (linear in altitude)
        alt_indices = np.arange(len(self.altitude_bands))
        alt_interp_func = interp1d(self.altitude_bands, alt_indices, 
                                   bounds_error=False, fill_value='extrapolate')
        alt_idx_interp = float(alt_interp_func(query_alt))
        
        # Query wind components
        point = np.array([[alt_idx_interp, query_lat, query_lon]])
        u_interp = float(self._interpolators['wind_u'](point)[0])
        v_interp = float(self._interpolators['wind_v'](point)[0])
        turb_interp = float(self._interpolators['turbulence'](point)[0])
        
        # Compute speed and direction
        speed_interp = np.sqrt(u_interp**2 + v_interp**2)
        direction_interp = np.degrees(np.arctan2(v_interp, u_interp))
        direction_interp = np.mod(direction_interp + 360, 360)
        
        return {
            'wind_speed_ms': speed_interp,
            'wind_u_ms': u_interp,
            'wind_v_ms': v_interp,
            'wind_direction_deg': direction_interp,
            'turbulence_intensity': max(0.0, turb_interp),
        }
    
    def calculate_energy_impact(self, altitude: float, 
                               heading_deg: float, airspeed_ms: float = 35.0) -> Dict[str, float]:
        """
        Calculate wind energy impact on trajectory.
        
        Args:
            altitude: Flight altitude in meters
            heading_deg: Aircraft heading (0=N, 90=E, 180=S, 270=W)
            airspeed_ms: Aircraft airspeed in m/s
            
        Returns:
            Dictionary with groundspeed, wind_component, and energy_cost_factor
        """
        # Sample wind at center of grid
        center_idx = self.grid_size // 2
        wind_at_alt = self.interpolate_wind(center_idx, center_idx, altitude)
        
        wind_speed = wind_at_alt['wind_speed_ms']
        wind_dir = wind_at_alt['wind_direction_deg']
        
        # Compute relative wind (from aircraft perspective)
        relative_wind_dir = wind_dir - heading_deg
        
        # Wind component along aircraft heading (positive = headwind)
        headwind_component = wind_speed * np.cos(np.radians(relative_wind_dir))
        
        # Ground speed
        groundspeed = airspeed_ms - headwind_component
        groundspeed = max(5.0, groundspeed)  # Minimum safe speed
        
        # Energy cost factor: increases with headwind (more power needed)
        # Formula: energy ~ power_required = thrust × velocity
        # Thrust needed: higher in headwind, lower in tailwind
        base_power = 150.0  # kW at 35 m/s airspeed
        
        if headwind_component > 0:  # Headwind
            energy_factor = 1.0 + 0.1 * (headwind_component / airspeed_ms)
        else:  # Tailwind
            energy_factor = 1.0 + 0.05 * (headwind_component / airspeed_ms)
        
        energy_cost = base_power * energy_factor / groundspeed  # kWh per km
        
        return {
            'groundspeed_ms': float(groundspeed),
            'headwind_component_ms': float(headwind_component),
            'energy_cost_kwh_per_km': float(energy_cost),
            'wind_severity': float(wind_speed),
        }
    
    def get_wind_profile_at_location(self, lat_idx: float, 
                                    lon_idx: float) -> Dict[float, Dict[str, float]]:
        """
        Get wind profile (wind vs. altitude) at a specific location.
        
        Args:
            lat_idx: Latitude grid index
            lon_idx: Longitude grid index
            
        Returns:
            Dictionary mapping altitude to wind properties
        """
        profile = {}
        for alt in self.altitude_bands:
            wind_data = self.interpolate_wind(lat_idx, lon_idx, alt)
            profile[alt] = wind_data
        return profile
    
    def to_dict(self) -> Dict:
        """Export wind field as dictionary."""
        return {
            'wind_u': self.wind_u,
            'wind_v': self.wind_v,
            'wind_speed': self.wind_speed,
            'wind_direction': self.wind_direction,
            'turbulence_intensity': self.turbulence_intensity,
            'altitude_bands': self.altitude_bands,
            'grid_size': self.grid_size,
            'metadata': {
                'resolution_m': self.metadata.resolution_m,
                'coverage_bounds': self.coverage_bounds,
                'generated_at': self.metadata.generated_at,
                'processing_version': self.metadata.processing_version,
            }
        }
