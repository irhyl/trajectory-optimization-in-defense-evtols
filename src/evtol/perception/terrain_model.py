"""
Terrain Elevation Mapping Module

This module handles:
- Terrain elevation data generation/import
- Data quality validation
- Interpolation and smoothing
- Safety margin calculations
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple, Optional
from scipy.ndimage import median_filter
from scipy.interpolate import RectBivariateSpline


@dataclass
class TerrainMetadata:
    """Metadata for terrain datasets"""
    resolution_m: float  # Grid resolution in meters
    bounds_lat: Tuple[float, float]  # Latitude bounds (min, max)
    bounds_lon: Tuple[float, float]  # Longitude bounds (min, max)
    elevation_range: Tuple[float, float]  # Min and max elevation (m)
    accuracy_m: float  # Vertical accuracy in meters
    source: str  # Data source name
    timestamp: str  # When data was generated/collected


class TerrainElevationMap:
    """
    Terrain elevation mapping with data quality assurance.
    
    This class provides:
    - Realistic terrain generation with proper statistical properties
    - Data validation against typical real-world ranges
    - Interpolation for different resolutions
    - Safety margin calculations
    """
    
    def __init__(self, width_m: int = 50000, height_m: int = 50000, resolution_m: float = 30.0):
        """
        Initialize terrain elevation map.
        
        Args:
            width_m: Map width in meters
            height_m: Map height in meters
            resolution_m: Grid resolution in meters (typical: 30m SRTM, 90m GEBCO)
        """
        self.width_m = width_m
        self.height_m = height_m
        self.resolution_m = resolution_m
        
        # Calculate grid dimensions (ensure square grid)
        self.nx = int(width_m / resolution_m)
        self.ny = int(height_m / resolution_m)
        # Make grid square to avoid broadcasting issues
        grid_size = min(self.nx, self.ny)
        self.nx = grid_size
        self.ny = grid_size
        self.grid_size = grid_size
        
        # Initialize terrain array
        self.elevation = np.zeros((self.ny, self.nx))
        
        # Metadata
        self.metadata: Optional[TerrainMetadata] = None
        
    def generate_realistic_terrain(self, seed: int = 42) -> np.ndarray:
        """
        Generate realistic terrain using layered Perlin-like noise.
        
        This creates:
        - Base elevation (smooth variations)
        - Mountain peaks (localized highs)
        - Valleys (localized lows)
        - Ridge systems
        
        Statistical properties match real-world DEM data:
        - Mean elevation: 200-500m
        - Standard deviation: 100-200m
        - Skewness: slight positive (valleys are longer than peaks)
        """
        np.random.seed(seed)
        
        x = np.linspace(0, self.width_m / 1000, self.nx)  # in km
        y = np.linspace(0, self.height_m / 1000, self.ny)  # in km
        X, Y = np.meshgrid(x, y)
        
        # Layer 1: Base elevation (large wavelength, ~50km features)
        base = 300 + 150 * np.sin(X / 50) * np.cos(Y / 50)
        
        # Layer 2: Medium features (~15km wavelength)
        medium = 100 * np.sin(X / 15) * np.cos(Y / 15) + 80 * np.sin(X / 20) * np.cos(Y / 25)
        
        # Layer 3: Small features (~5km wavelength)
        small = 50 * np.sin(X / 5) * np.cos(Y / 5) + 30 * np.sin(X / 7) * np.cos(Y / 8)
        
        # Layer 4: Noise (sub-km features)
        noise = np.random.normal(0, 20, (self.ny, self.nx))
        
        # Combine layers with weights (higher frequencies have lower amplitudes)
        self.elevation = base + medium + small + noise
        
        # Add localized peaks
        self._add_peaks(num_peaks=15, peak_height_range=(400, 1200))
        
        # Add valleys
        self._add_valleys(num_valleys=8, valley_depth_range=(100, 300))
        
        # Ensure realistic range (clip to [50, 2000] meters)
        self.elevation = np.clip(self.elevation, 50, 2000)
        
        # Create metadata
        self.metadata = TerrainMetadata(
            resolution_m=self.resolution_m,
            bounds_lat=(40.0, 41.0),  # Example: New York region
            bounds_lon=(-74.0, -73.0),
            elevation_range=(float(self.elevation.min()), float(self.elevation.max())),
            accuracy_m=10.0,
            source="Synthetic_SRTM_like",
            timestamp="2025-11-23"
        )
        
        return self.elevation
    
    def _add_peaks(self, num_peaks: int = 10, peak_height_range: Tuple[float, float] = (300, 1000)):
        """Add mountain peaks to the terrain"""
        for _ in range(num_peaks):
            # Random peak location
            cx = np.random.randint(0, self.nx)
            cy = np.random.randint(0, self.ny)
            
            # Peak parameters
            height = np.random.uniform(peak_height_range[0], peak_height_range[1])
            radius = np.random.uniform(20, 100)  # radius in grid cells
            
            # Create Gaussian peak
            yy, xx = np.ogrid[:self.ny, :self.nx]
            dist = np.sqrt((xx - cx)**2 + (yy - cy)**2)
            peak = height * np.exp(-(dist**2) / (2 * (radius/3)**2))
            
            self.elevation += peak
    
    def _add_valleys(self, num_valleys: int = 5, valley_depth_range: Tuple[float, float] = (50, 300)):
        """Add valleys (low-elevation corridors) to terrain"""
        for _ in range(num_valleys):
            # Random valley location
            cx = np.random.randint(0, self.nx)
            cy = np.random.randint(0, self.ny)
            
            # Valley parameters
            depth = np.random.uniform(valley_depth_range[0], valley_depth_range[1])
            radius = np.random.uniform(30, 150)
            
            # Create Gaussian valley
            yy, xx = np.ogrid[:self.ny, :self.nx]
            dist = np.sqrt((xx - cx)**2 + (yy - cy)**2)
            valley = depth * np.exp(-(dist**2) / (2 * (radius/3)**2))
            
            self.elevation = np.maximum(self.elevation - valley, 0)
    
    def smooth_terrain(self, kernel_size: int = 3) -> np.ndarray:
        """
        Apply median filtering to reduce noise while preserving edges.
        
        Args:
            kernel_size: Median filter kernel size (use odd numbers)
        
        Returns:
            Smoothed elevation array
        """
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        self.elevation = median_filter(self.elevation, size=kernel_size)
        return self.elevation
    
    def validate_data_quality(self) -> dict:
        """
        Validate terrain data quality.
        
        Returns:
            Dictionary with validation results
        """
        results = {
            'is_valid': True,
            'checks': {},
            'warnings': []
        }
        
        # Check 1: No NaN or Inf values
        has_nan = np.any(np.isnan(self.elevation))
        has_inf = np.any(np.isinf(self.elevation))
        results['checks']['no_nan_inf'] = not (has_nan or has_inf)
        if has_nan or has_inf:
            results['is_valid'] = False
            results['warnings'].append("Data contains NaN or Inf values")
        
        # Check 2: Elevation in realistic range
        realistic_range = (0, 2500)
        out_of_range = np.sum((self.elevation < realistic_range[0]) | 
                             (self.elevation > realistic_range[1]))
        results['checks']['realistic_range'] = out_of_range == 0
        if out_of_range > 0:
            results['warnings'].append(
                f"{out_of_range} cells outside realistic elevation range {realistic_range}"
            )
        
        # Check 3: Reasonable standard deviation
        std = np.std(self.elevation)
        results['checks']['reasonable_std'] = 50 < std < 300
        if not results['checks']['reasonable_std']:
            results['warnings'].append(
                f"Standard deviation {std:.1f}m outside expected range (50-300m)"
            )
        
        # Check 4: Smooth gradients (no unrealistic cliffs)
        gradients = np.gradient(self.elevation)
        max_gradient = np.max(np.abs(gradients))
        # Max realistic gradient ≈ tan(45°) = 1.0 at 30m resolution = 30m height change
        results['checks']['smooth_gradients'] = max_gradient < 200
        if max_gradient >= 200:
            results['warnings'].append(
                f"Maximum gradient {max_gradient:.1f} m indicates possible numerical issues"
            )
        
        return results
    
    def calculate_safety_margins(self, terrain_buffer_m: float = 50.0) -> np.ndarray:
        """
        Calculate minimum safe altitude above terrain.
        
        Args:
            terrain_buffer_m: Safety buffer above terrain (default: 50m)
        
        Returns:
            Minimum safe altitude map (elevation + buffer)
        """
        return self.elevation + terrain_buffer_m
    
    def interpolate_elevation(self, lat: float, lon: float) -> float:
        """
        Interpolate elevation at a specific lat/lon point.
        
        Args:
            lat: Latitude
            lon: Longitude
        
        Returns:
            Interpolated elevation in meters
        """
        # Convert lat/lon to grid indices
        # Assuming bounds from metadata
        if self.metadata is None:
            return np.nan
        
        lat_min, lat_max = self.metadata.bounds_lat
        lon_min, lon_max = self.metadata.bounds_lon
        
        # Normalize to [0, 1]
        lat_norm = (lat - lat_min) / (lat_max - lat_min)
        lon_norm = (lon - lon_min) / (lon_max - lon_min)
        
        # Convert to grid indices
        y_idx = lat_norm * (self.ny - 1)
        x_idx = lon_norm * (self.nx - 1)
        
        # Bilinear interpolation
        x0, x1 = int(x_idx), int(x_idx) + 1
        y0, y1 = int(y_idx), int(y_idx) + 1
        
        # Handle boundaries
        x1 = min(x1, self.nx - 1)
        y1 = min(y1, self.ny - 1)
        
        # Interpolation weights
        wx = x_idx - int(x_idx)
        wy = y_idx - int(y_idx)
        
        # Bilinear interpolation
        z00 = self.elevation[y0, x0]
        z01 = self.elevation[y0, x1]
        z10 = self.elevation[y1, x0]
        z11 = self.elevation[y1, x1]
        
        z0 = z00 * (1 - wx) + z01 * wx
        z1 = z10 * (1 - wx) + z11 * wx
        
        return z0 * (1 - wy) + z1 * wy
    
    def get_statistics(self) -> dict:
        """
        Get statistical summary of terrain.
        
        Returns:
            Dictionary with terrain statistics
        """
        return {
            'mean_elevation_m': float(np.mean(self.elevation)),
            'std_elevation_m': float(np.std(self.elevation)),
            'min_elevation_m': float(np.min(self.elevation)),
            'max_elevation_m': float(np.max(self.elevation)),
            'median_elevation_m': float(np.median(self.elevation)),
            'skewness': float(self._calculate_skewness()),
            'grid_points': self.elevation.size,
            'grid_dimensions': (self.ny, self.nx),
            'coverage_m2': self.width_m * self.height_m
        }
    
    def _calculate_skewness(self) -> float:
        """Calculate skewness of elevation distribution"""
        mean = np.mean(self.elevation)
        std = np.std(self.elevation)
        third_moment = np.mean(((self.elevation - mean) / std) ** 3)
        return float(third_moment)
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Export terrain as pandas DataFrame.
        
        Returns:
            DataFrame with columns: x, y, elevation
        """
        x = np.linspace(0, self.width_m, self.nx)
        y = np.linspace(0, self.height_m, self.ny)
        X, Y = np.meshgrid(x, y)
        
        df = pd.DataFrame({
            'x_m': X.flatten(),
            'y_m': Y.flatten(),
            'elevation_m': self.elevation.flatten()
        })
        
        return df
