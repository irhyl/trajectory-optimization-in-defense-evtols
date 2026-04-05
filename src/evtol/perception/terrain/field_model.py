"""
Terrain Field Model - Grid-based Terrain with Derived Geomorphometric Products.

This module provides a comprehensive terrain representation with support for:
- Slope and aspect computation (Horn's method)
- Hillshade rendering
- Terrain Ruggedness Index (TRI)
- Curvature analysis (plan, profile)
- Line-of-sight analysis
- Viewshed computation
- Minimum safe altitude surfaces

References:
    - Horn, B.K.P. (1981), "Hill Shading and the Reflectance Map",
      Proceedings of the IEEE, 69(1), 14-47.
    - Riley, S.J. et al. (1999), "A Terrain Ruggedness Index that
      Quantifies Topographic Heterogeneity", Intermountain Journal of Sciences.
    - Zevenbergen, L.W. & Thorne, C.R. (1987), "Quantitative Analysis
      of Land Surface Topography", Earth Surface Processes and Landforms.

Author: Research-grade implementation for eVTOL trajectory optimization
Version: 1.0.0
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import numpy as np
from scipy import ndimage

from .data_provider import (
    TerrainDataProvider,
    TerrainProviderConfig,
)

logger = logging.getLogger(__name__)


class TerrainCategory(Enum):
    """
    Terrain category classification based on ICAO standards.

    Used for obstacle clearance calculations.

    Attributes:
        FLAT: Terrain slope < 2%
        ROLLING: Terrain slope 2-5%
        HILLY: Terrain slope 5-10%
        MOUNTAINOUS: Terrain slope > 10%
    """
    FLAT = "flat"
    ROLLING = "rolling"
    HILLY = "hilly"
    MOUNTAINOUS = "mountainous"


class LandCoverType(Enum):
    """
    Land cover classification for obstacle estimation.

    Attributes:
        WATER: Open water (0m obstacle height)
        BARE: Bare ground, desert (0m)
        GRASS: Grassland, agriculture (1-2m)
        SHRUB: Shrubland (3-5m)
        FOREST: Forest canopy (15-30m)
        URBAN_LOW: Low-density urban (10-15m)
        URBAN_HIGH: High-density urban (30-100m)
    """
    WATER = "water"
    BARE = "bare"
    GRASS = "grass"
    SHRUB = "shrub"
    FOREST = "forest"
    URBAN_LOW = "urban_low"
    URBAN_HIGH = "urban_high"


@dataclass
class TerrainMetadata:
    """
    Comprehensive metadata for terrain model.

    Attributes:
        bounds: Geographic bounds (north, south, east, west)
        resolution_m: Grid resolution in meters
        crs: Coordinate reference system
        source: Elevation data source
        vertical_datum: Vertical reference datum
        accuracy_m: Vertical accuracy estimate
        acquisition_date: Original data acquisition date
        processing_date: When model was created
        version: Model version string
        derived_products: List of computed derived products
        quality_metrics: Data quality information
    """
    bounds: tuple[float, float, float, float]
    resolution_m: float
    crs: str = "EPSG:4326"
    source: str = "SRTM"
    vertical_datum: str = "EGM96"
    accuracy_m: float = 16.0
    acquisition_date: datetime | None = None
    processing_date: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    version: str = "1.0.0"
    derived_products: list[str] = field(default_factory=list)
    quality_metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Export metadata as dictionary."""
        return {
            "bounds": list(self.bounds),
            "resolution_m": self.resolution_m,
            "crs": self.crs,
            "source": self.source,
            "vertical_datum": self.vertical_datum,
            "accuracy_m": self.accuracy_m,
            "acquisition_date": self.acquisition_date.isoformat() if self.acquisition_date else None,
            "processing_date": self.processing_date.isoformat(),
            "version": self.version,
            "derived_products": self.derived_products,
            "quality_metrics": self.quality_metrics,
        }


@dataclass
class TerrainProfile:
    """
    Elevation profile along a path.

    Attributes:
        distances_m: Cumulative distance along path in meters
        elevations_m: Elevation at each point in meters
        slopes_deg: Slope at each segment in degrees
        latitudes: Latitude at each point
        longitudes: Longitude at each point
        min_elevation_m: Minimum elevation along path
        max_elevation_m: Maximum elevation along path
        total_climb_m: Total ascent along path
        total_descent_m: Total descent along path
    """
    distances_m: np.ndarray
    elevations_m: np.ndarray
    slopes_deg: np.ndarray
    latitudes: np.ndarray
    longitudes: np.ndarray
    min_elevation_m: float = 0.0
    max_elevation_m: float = 0.0
    total_climb_m: float = 0.0
    total_descent_m: float = 0.0

    def __post_init__(self):
        """Compute derived statistics."""
        if len(self.elevations_m) > 0:
            self.min_elevation_m = float(np.min(self.elevations_m))
            self.max_elevation_m = float(np.max(self.elevations_m))

            # Compute climb and descent
            diffs = np.diff(self.elevations_m)
            self.total_climb_m = float(np.sum(diffs[diffs > 0]))
            self.total_descent_m = float(np.abs(np.sum(diffs[diffs < 0])))

    def to_dict(self) -> dict[str, Any]:
        """Export profile as dictionary."""
        return {
            "distances_m": self.distances_m.tolist(),
            "elevations_m": self.elevations_m.tolist(),
            "slopes_deg": self.slopes_deg.tolist(),
            "latitudes": self.latitudes.tolist(),
            "longitudes": self.longitudes.tolist(),
            "min_elevation_m": self.min_elevation_m,
            "max_elevation_m": self.max_elevation_m,
            "total_climb_m": self.total_climb_m,
            "total_descent_m": self.total_descent_m,
        }


@dataclass
class LineOfSightResult:
    """
    Result of line-of-sight analysis.

    Attributes:
        is_visible: Whether target is visible from observer
        observer: Observer position (lat, lon, alt_m)
        target: Target position (lat, lon, alt_m)
        blocking_point: First blocking point if not visible (lat, lon, elev)
        blocking_distance_m: Distance to blocking point
        clearance_m: Minimum clearance above terrain (negative if blocked)
        profile_distances: Distances along sight line
        profile_terrain: Terrain elevations along sight line
        profile_sight_line: Sight line elevations
    """
    is_visible: bool
    observer: tuple[float, float, float]
    target: tuple[float, float, float]
    blocking_point: tuple[float, float, float] | None = None
    blocking_distance_m: float | None = None
    clearance_m: float = 0.0
    profile_distances: np.ndarray | None = None
    profile_terrain: np.ndarray | None = None
    profile_sight_line: np.ndarray | None = None


@dataclass
class ViewshedResult:
    """
    Result of viewshed analysis.

    Attributes:
        visible: Boolean grid where True = visible from observer
        observer: Observer position (lat, lon, height_m)
        observer_height_m: Observer height above terrain
        visible_area_km2: Total visible area in square kilometers
        visible_fraction: Fraction of grid that is visible
        latitudes: Grid latitudes
        longitudes: Grid longitudes
    """
    visible: np.ndarray
    observer: tuple[float, float]
    observer_height_m: float
    visible_area_km2: float = 0.0
    visible_fraction: float = 0.0
    latitudes: np.ndarray = field(default_factory=lambda: np.array([]))
    longitudes: np.ndarray = field(default_factory=lambda: np.array([]))


class TerrainFieldModel:
    """
    Grid-based terrain model with derived geomorphometric products.

    This class provides a comprehensive terrain representation with:
    - Real elevation data from global APIs
    - Derived products (slope, aspect, hillshade, curvature)
    - Line-of-sight and viewshed analysis
    - Minimum safe altitude computation
    - Terrain classification

    Example:
        >>> model = TerrainFieldModel(
        ...     coverage_bounds=(28.7, 28.5, 77.3, 77.1),  # Delhi area
        ...     resolution_m=100
        ... )
        >>> model.initialize()
        >>>
        >>> # Get elevation at a point
        >>> elev = model.get_elevation(28.6, 77.2)
        >>> print(f"Elevation: {elev:.1f}m")
        >>>
        >>> # Compute slope
        >>> slope = model.compute_slope()
        >>> print(f"Mean slope: {np.mean(slope):.1f}°")
        >>>
        >>> # Check line of sight
        >>> los = model.check_line_of_sight(
        ...     observer=(28.6, 77.2, 100),  # lat, lon, height
        ...     target=(28.65, 77.25, 50)
        ... )
        >>> print(f"Visible: {los.is_visible}")

    Attributes:
        elevation: 2D elevation grid (n_lat, n_lon)
        latitudes: 1D array of latitudes (north to south)
        longitudes: 1D array of longitudes (west to east)
        metadata: Terrain metadata
        cell_size_m: Grid cell size in meters
    """

    def __init__(
        self,
        coverage_bounds: tuple[float, float, float, float],
        resolution_m: float = 100.0,
        provider_config: TerrainProviderConfig | None = None,
    ):
        """
        Initialize terrain field model.

        Args:
            coverage_bounds: (north, south, east, west) in decimal degrees
            resolution_m: Grid resolution in meters
            provider_config: Configuration for terrain data provider

        Note:
            Call initialize() to fetch elevation data.
        """
        self.coverage_bounds = coverage_bounds
        self.resolution_m = resolution_m

        # Initialize provider
        self._provider = TerrainDataProvider(config=provider_config)

        # Data arrays (populated by initialize())
        self.elevation: np.ndarray | None = None
        self.latitudes: np.ndarray | None = None
        self.longitudes: np.ndarray | None = None

        # Derived products (computed on demand)
        self._slope: np.ndarray | None = None
        self._aspect: np.ndarray | None = None
        self._hillshade: np.ndarray | None = None
        self._curvature_plan: np.ndarray | None = None
        self._curvature_profile: np.ndarray | None = None
        self._ruggedness: np.ndarray | None = None

        # Cell size in meters (computed after initialization)
        self._cell_size_x: float = 0.0
        self._cell_size_y: float = 0.0

        # Metadata
        self.metadata = TerrainMetadata(
            bounds=coverage_bounds,
            resolution_m=resolution_m,
        )

        self._initialized = False

        logger.info(f"TerrainFieldModel created for bounds {coverage_bounds}, resolution {resolution_m}m")

    def initialize(self) -> TerrainFieldModel:
        """
        Initialize model by fetching elevation data.

        Returns:
            Self for method chaining

        Raises:
            RuntimeError: If data fetch fails
        """
        logger.info("Initializing terrain model...")

        # Fetch elevation grid
        grid = self._provider.get_elevation_grid(
            bounds=self.coverage_bounds,
            resolution_m=self.resolution_m,
        )

        self.elevation = grid.elevation
        self.latitudes = grid.latitudes
        self.longitudes = grid.longitudes

        # Compute cell sizes in meters
        center_lat = (self.coverage_bounds[0] + self.coverage_bounds[1]) / 2
        meters_per_deg_lat = 111320.0
        meters_per_deg_lon = 111320.0 * np.cos(np.radians(center_lat))

        dlat = abs(self.latitudes[1] - self.latitudes[0]) if len(self.latitudes) > 1 else 0
        dlon = abs(self.longitudes[1] - self.longitudes[0]) if len(self.longitudes) > 1 else 0

        self._cell_size_y = dlat * meters_per_deg_lat
        self._cell_size_x = dlon * meters_per_deg_lon

        # Update metadata
        self.metadata.source = grid.source.value
        self.metadata.vertical_datum = grid.vertical_datum.value
        self.metadata.quality_metrics = grid.get_statistics()

        self._initialized = True

        logger.info(f"Terrain model initialized: {self.elevation.shape}, "
                   f"elevation range: {self.elevation.min():.0f}-{self.elevation.max():.0f}m")

        return self

    @property
    def shape(self) -> tuple[int, int]:
        """Grid shape (n_lat, n_lon)."""
        if self.elevation is None:
            return (0, 0)
        return self.elevation.shape

    @property
    def cell_size_m(self) -> tuple[float, float]:
        """Cell size in meters (dy, dx)."""
        return (self._cell_size_y, self._cell_size_x)

    def _ensure_initialized(self):
        """Ensure model is initialized."""
        if not self._initialized:
            raise RuntimeError("Model not initialized. Call initialize() first.")

    def get_elevation(self, latitude: float, longitude: float) -> float:
        """
        Get elevation at a point using bilinear interpolation.

        Args:
            latitude: WGS84 latitude in decimal degrees
            longitude: WGS84 longitude in decimal degrees

        Returns:
            Interpolated elevation in meters

        Raises:
            ValueError: If point is outside grid bounds
        """
        self._ensure_initialized()

        # Check bounds
        if not (self.latitudes[-1] <= latitude <= self.latitudes[0]):
            raise ValueError(f"Latitude {latitude} outside grid bounds [{self.latitudes[-1]}, {self.latitudes[0]}]")
        if not (self.longitudes[0] <= longitude <= self.longitudes[-1]):
            raise ValueError(f"Longitude {longitude} outside grid bounds [{self.longitudes[0]}, {self.longitudes[-1]}]")

        # Find grid indices
        lat_idx = np.interp(latitude, self.latitudes[::-1], np.arange(len(self.latitudes))[::-1])
        lon_idx = np.interp(longitude, self.longitudes, np.arange(len(self.longitudes)))

        # Bilinear interpolation
        i0, i1 = int(np.floor(lat_idx)), int(np.ceil(lat_idx))
        j0, j1 = int(np.floor(lon_idx)), int(np.ceil(lon_idx))

        # Clamp to valid indices
        i0, i1 = max(0, i0), min(self.elevation.shape[0] - 1, i1)
        j0, j1 = max(0, j0), min(self.elevation.shape[1] - 1, j1)

        # Handle edge cases
        if i0 == i1 and j0 == j1:
            return float(self.elevation[i0, j0])

        # Interpolation weights
        di = lat_idx - i0 if i1 > i0 else 0
        dj = lon_idx - j0 if j1 > j0 else 0

        # Bilinear interpolation
        elev = (
            self.elevation[i0, j0] * (1 - di) * (1 - dj) +
            self.elevation[i1, j0] * di * (1 - dj) +
            self.elevation[i0, j1] * (1 - di) * dj +
            self.elevation[i1, j1] * di * dj
        )

        return float(elev)

    def compute_slope(self, units: str = "degrees") -> np.ndarray:
        """
        Compute slope using Horn's method.

        Horn's method uses a 3x3 kernel for gradient estimation:

            dz/dx = ((z[i-1,j+1] + 2*z[i,j+1] + z[i+1,j+1]) -
                     (z[i-1,j-1] + 2*z[i,j-1] + z[i+1,j-1])) / (8 * cell_size_x)

        Reference:
            Horn, B.K.P. (1981), "Hill Shading and the Reflectance Map"

        Args:
            units: "degrees", "radians", or "percent"

        Returns:
            Slope array with same shape as elevation
        """
        self._ensure_initialized()

        if self._slope is not None:
            slope = self._slope.copy()
        else:
            # Horn's kernels
            kernel_x = np.array([
                [-1, 0, 1],
                [-2, 0, 2],
                [-1, 0, 1]
            ]) / (8 * self._cell_size_x)

            kernel_y = np.array([
                [-1, -2, -1],
                [0, 0, 0],
                [1, 2, 1]
            ]) / (8 * self._cell_size_y)

            # Compute gradients
            dzdx = ndimage.convolve(self.elevation, kernel_x, mode='reflect')
            dzdy = ndimage.convolve(self.elevation, kernel_y, mode='reflect')

            # Compute slope in radians
            slope_rad = np.arctan(np.sqrt(dzdx**2 + dzdy**2))
            self._slope = np.degrees(slope_rad)
            slope = self._slope.copy()

            # Update metadata
            if "slope" not in self.metadata.derived_products:
                self.metadata.derived_products.append("slope")

        # Convert units
        if units == "degrees":
            return slope
        elif units == "radians":
            return np.radians(slope)
        elif units == "percent":
            return np.tan(np.radians(slope)) * 100
        else:
            raise ValueError(f"Unknown units: {units}")

    def compute_aspect(self, convention: str = "compass") -> np.ndarray:
        """
        Compute aspect (slope direction) using Horn's method.

        Args:
            convention: "compass" (0=N, 90=E) or "math" (0=E, CCW positive)

        Returns:
            Aspect array in degrees (0-360)
        """
        self._ensure_initialized()

        if self._aspect is not None:
            aspect = self._aspect.copy()
        else:
            # Horn's kernels
            kernel_x = np.array([
                [-1, 0, 1],
                [-2, 0, 2],
                [-1, 0, 1]
            ]) / (8 * self._cell_size_x)

            kernel_y = np.array([
                [-1, -2, -1],
                [0, 0, 0],
                [1, 2, 1]
            ]) / (8 * self._cell_size_y)

            dzdx = ndimage.convolve(self.elevation, kernel_x, mode='reflect')
            dzdy = ndimage.convolve(self.elevation, kernel_y, mode='reflect')

            # Compute aspect (compass convention: 0=N, 90=E, 180=S, 270=W)
            aspect = np.degrees(np.arctan2(-dzdx, dzdy))
            aspect = (aspect + 360) % 360

            self._aspect = aspect

            if "aspect" not in self.metadata.derived_products:
                self.metadata.derived_products.append("aspect")

        if convention == "compass":
            return aspect
        elif convention == "math":
            # Convert to mathematical convention (0=E, CCW positive)
            return (90 - aspect + 360) % 360
        else:
            raise ValueError(f"Unknown convention: {convention}")

    def compute_hillshade(
        self,
        azimuth: float = 315.0,
        altitude: float = 45.0,
    ) -> np.ndarray:
        """
        Compute hillshade illumination.

        Hillshade simulates illumination of terrain from a light source.

        Args:
            azimuth: Light source azimuth in degrees (0=N, 90=E)
            altitude: Light source altitude in degrees above horizon

        Returns:
            Hillshade array (0-255)
        """
        self._ensure_initialized()

        # Get slope and aspect
        slope_rad = np.radians(self.compute_slope())
        aspect_rad = np.radians(self.compute_aspect())

        # Convert light source angles
        azimuth_rad = np.radians(azimuth)
        altitude_rad = np.radians(altitude)

        # Compute hillshade
        # Reference: ESRI hillshade algorithm
        hillshade = (
            np.sin(altitude_rad) * np.cos(slope_rad) +
            np.cos(altitude_rad) * np.sin(slope_rad) *
            np.cos(azimuth_rad - aspect_rad)
        )

        # Normalize to 0-255
        hillshade = np.clip(hillshade, 0, 1)
        self._hillshade = (hillshade * 255).astype(np.uint8)

        if "hillshade" not in self.metadata.derived_products:
            self.metadata.derived_products.append("hillshade")

        return self._hillshade

    def compute_ruggedness(self) -> np.ndarray:
        """
        Compute Terrain Ruggedness Index (TRI).

        TRI is the mean elevation difference between a cell and its 8 neighbors.

        Reference:
            Riley, S.J. et al. (1999), "A Terrain Ruggedness Index that
            Quantifies Topographic Heterogeneity"

        Returns:
            TRI array in meters
        """
        self._ensure_initialized()

        if self._ruggedness is not None:
            return self._ruggedness.copy()

        # Compute sum of squared differences with neighbors
        np.array([
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1]
        ])

        # Pad elevation for edge handling
        padded = np.pad(self.elevation, 1, mode='reflect')

        # Compute TRI
        tri = np.zeros_like(self.elevation)
        for i in range(self.elevation.shape[0]):
            for j in range(self.elevation.shape[1]):
                window = padded[i:i+3, j:j+3]
                center = window[1, 1]
                diffs = (window - center) ** 2
                diffs[1, 1] = 0  # Exclude center
                tri[i, j] = np.sqrt(np.sum(diffs))

        self._ruggedness = tri

        if "ruggedness" not in self.metadata.derived_products:
            self.metadata.derived_products.append("ruggedness")

        return self._ruggedness

    def compute_curvature(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute plan and profile curvature.

        - Plan curvature: Curvature in horizontal plane (affects flow convergence)
        - Profile curvature: Curvature in vertical plane (affects flow acceleration)

        Reference:
            Zevenbergen, L.W. & Thorne, C.R. (1987)

        Returns:
            Tuple of (plan_curvature, profile_curvature) arrays
        """
        self._ensure_initialized()

        if self._curvature_plan is not None and self._curvature_profile is not None:
            return self._curvature_plan.copy(), self._curvature_profile.copy()

        # Compute first and second derivatives
        dy, dx = self._cell_size_y, self._cell_size_x

        # First derivatives (Sobel-like)
        dzdx = np.gradient(self.elevation, dx, axis=1)
        dzdy = np.gradient(self.elevation, dy, axis=0)

        # Second derivatives
        d2zdx2 = np.gradient(dzdx, dx, axis=1)
        d2zdy2 = np.gradient(dzdy, dy, axis=0)
        d2zdxdy = np.gradient(dzdx, dy, axis=0)

        # Compute plan and profile curvature
        # Following Zevenbergen & Thorne (1987)
        p = dzdx ** 2
        q = dzdy ** 2

        # Profile curvature (in direction of maximum slope)
        profile = -2 * (d2zdx2 * p + 2 * d2zdxdy * dzdx * dzdy + d2zdy2 * q)
        profile = profile / ((p + q) * np.sqrt(1 + p + q) ** 3 + 1e-10)

        # Plan curvature (perpendicular to slope direction)
        plan = -2 * (d2zdx2 * q - 2 * d2zdxdy * dzdx * dzdy + d2zdy2 * p)
        plan = plan / ((p + q) ** 1.5 + 1e-10)

        self._curvature_plan = plan
        self._curvature_profile = profile

        if "curvature" not in self.metadata.derived_products:
            self.metadata.derived_products.append("curvature")

        return self._curvature_plan, self._curvature_profile

    def classify_terrain(self) -> np.ndarray:
        """
        Classify terrain into ICAO categories based on slope.

        Returns:
            Array of TerrainCategory values (as integers)
            0=FLAT, 1=ROLLING, 2=HILLY, 3=MOUNTAINOUS
        """
        self._ensure_initialized()

        slope_pct = self.compute_slope(units="percent")

        classification = np.zeros_like(slope_pct, dtype=np.int8)
        classification[slope_pct >= 2] = 1   # ROLLING
        classification[slope_pct >= 5] = 2   # HILLY
        classification[slope_pct >= 10] = 3  # MOUNTAINOUS

        return classification

    def get_minimum_safe_altitude(
        self,
        latitude: float,
        longitude: float,
        clearance_m: float = 150.0,
        buffer_radius_m: float = 1000.0,
    ) -> float:
        """
        Get minimum safe altitude at a point with terrain clearance.

        Args:
            latitude: WGS84 latitude
            longitude: WGS84 longitude
            clearance_m: Required terrain clearance in meters
            buffer_radius_m: Radius to check for highest terrain

        Returns:
            Minimum safe altitude in meters MSL
        """
        self._ensure_initialized()

        # Convert buffer to degrees
        center_lat = latitude
        buffer_lat = buffer_radius_m / 111320
        buffer_lon = buffer_radius_m / (111320 * np.cos(np.radians(center_lat)))

        # Find grid cells within buffer
        lat_mask = np.abs(self.latitudes - latitude) <= buffer_lat
        lon_mask = np.abs(self.longitudes - longitude) <= buffer_lon

        if not np.any(lat_mask) or not np.any(lon_mask):
            # Point outside grid, return elevation at point + clearance
            return self.get_elevation(latitude, longitude) + clearance_m

        # Get maximum terrain within buffer
        lat_indices = np.where(lat_mask)[0]
        lon_indices = np.where(lon_mask)[0]

        terrain_subset = self.elevation[
            lat_indices[0]:lat_indices[-1]+1,
            lon_indices[0]:lon_indices[-1]+1
        ]

        max_terrain = np.max(terrain_subset)

        return max_terrain + clearance_m

    def get_terrain_clearance_surface(
        self,
        clearance_m: float = 150.0,
    ) -> np.ndarray:
        """
        Compute terrain clearance surface (TCS).

        TCS represents minimum safe altitude at each grid point.

        Args:
            clearance_m: Required terrain clearance in meters

        Returns:
            TCS array in meters MSL
        """
        self._ensure_initialized()

        # Simple TCS: elevation + clearance
        # For more sophisticated TCS, consider local terrain variations
        return self.elevation + clearance_m

    def check_line_of_sight(
        self,
        observer: tuple[float, float, float],
        target: tuple[float, float, float],
        samples: int = 100,
    ) -> LineOfSightResult:
        """
        Check line of sight between observer and target.

        Args:
            observer: (latitude, longitude, height_above_ground_m)
            target: (latitude, longitude, height_above_ground_m)
            samples: Number of sample points along sight line

        Returns:
            LineOfSightResult with visibility status and details
        """
        self._ensure_initialized()

        obs_lat, obs_lon, obs_height = observer
        tgt_lat, tgt_lon, tgt_height = target

        # Get terrain elevations at observer and target
        obs_terrain = self.get_elevation(obs_lat, obs_lon)
        tgt_terrain = self.get_elevation(tgt_lat, tgt_lon)

        # Absolute altitudes
        obs_alt = obs_terrain + obs_height
        tgt_alt = tgt_terrain + tgt_height

        # Generate sample points along sight line
        t_values = np.linspace(0, 1, samples)
        sample_lats = obs_lat + t_values * (tgt_lat - obs_lat)
        sample_lons = obs_lon + t_values * (tgt_lon - obs_lon)

        # Get terrain at each sample
        terrain_elevations = np.array([
            self.get_elevation(lat, lon) for lat, lon in zip(sample_lats, sample_lons)
        ])

        # Compute sight line altitude at each sample
        sight_line = obs_alt + t_values * (tgt_alt - obs_alt)

        # Compute distances
        dlat = (tgt_lat - obs_lat) * 111320
        dlon = (tgt_lon - obs_lon) * 111320 * np.cos(np.radians((obs_lat + tgt_lat) / 2))
        total_dist = np.sqrt(dlat**2 + dlon**2)
        distances = t_values * total_dist

        # Check for blocking
        clearances = sight_line - terrain_elevations
        min_clearance_idx = np.argmin(clearances)
        min_clearance = clearances[min_clearance_idx]

        is_visible = min_clearance > 0

        # Find first blocking point if not visible
        blocking_point = None
        blocking_distance = None
        if not is_visible:
            blocked_indices = np.where(clearances <= 0)[0]
            if len(blocked_indices) > 0:
                first_blocked = blocked_indices[0]
                blocking_point = (
                    sample_lats[first_blocked],
                    sample_lons[first_blocked],
                    terrain_elevations[first_blocked]
                )
                blocking_distance = distances[first_blocked]

        return LineOfSightResult(
            is_visible=is_visible,
            observer=(obs_lat, obs_lon, obs_alt),
            target=(tgt_lat, tgt_lon, tgt_alt),
            blocking_point=blocking_point,
            blocking_distance_m=blocking_distance,
            clearance_m=float(min_clearance),
            profile_distances=distances,
            profile_terrain=terrain_elevations,
            profile_sight_line=sight_line,
        )

    def compute_viewshed(
        self,
        observer_lat: float,
        observer_lon: float,
        observer_height_m: float = 100.0,
        max_radius_m: float | None = None,
    ) -> ViewshedResult:
        """
        Compute viewshed (visible area) from an observer point.

        Args:
            observer_lat: Observer latitude
            observer_lon: Observer longitude
            observer_height_m: Observer height above terrain
            max_radius_m: Maximum analysis radius (None = entire grid)

        Returns:
            ViewshedResult with visibility grid
        """
        self._ensure_initialized()

        # Initialize visibility grid
        visible = np.zeros(self.elevation.shape, dtype=bool)

        # Observer terrain elevation
        obs_terrain = self.get_elevation(observer_lat, observer_lon)
        obs_terrain + observer_height_m

        # Convert max radius to degrees if specified
        max_radius_deg = None
        if max_radius_m is not None:
            max_radius_deg = max_radius_m / 111320

        # Check visibility for each cell
        for i, lat in enumerate(self.latitudes):
            for j, lon in enumerate(self.longitudes):
                # Skip if outside max radius
                if max_radius_deg is not None:
                    dist_deg = np.sqrt((lat - observer_lat)**2 + (lon - observer_lon)**2)
                    if dist_deg > max_radius_deg:
                        continue

                # Check line of sight to this cell (at ground level)
                los = self.check_line_of_sight(
                    observer=(observer_lat, observer_lon, observer_height_m),
                    target=(lat, lon, 0.0),
                    samples=50
                )
                visible[i, j] = los.is_visible

        # Compute visible area
        cell_area_km2 = (self._cell_size_x * self._cell_size_y) / 1e6
        visible_area = np.sum(visible) * cell_area_km2
        visible_fraction = np.mean(visible)

        return ViewshedResult(
            visible=visible,
            observer=(observer_lat, observer_lon),
            observer_height_m=observer_height_m,
            visible_area_km2=visible_area,
            visible_fraction=visible_fraction,
            latitudes=self.latitudes,
            longitudes=self.longitudes,
        )

    def get_terrain_profile(
        self,
        waypoints: list[tuple[float, float]],
        samples_per_segment: int = 50,
    ) -> TerrainProfile:
        """
        Get elevation profile along a path.

        Args:
            waypoints: List of (latitude, longitude) tuples
            samples_per_segment: Number of samples between waypoints

        Returns:
            TerrainProfile with elevations and derived data
        """
        self._ensure_initialized()

        if len(waypoints) < 2:
            raise ValueError("At least 2 waypoints required")

        # Generate sample points
        sample_lats = []
        sample_lons = []

        for i in range(len(waypoints) - 1):
            lat1, lon1 = waypoints[i]
            lat2, lon2 = waypoints[i + 1]

            for j in range(samples_per_segment):
                t = j / samples_per_segment
                sample_lats.append(lat1 + t * (lat2 - lat1))
                sample_lons.append(lon1 + t * (lon2 - lon1))

        # Add final point
        sample_lats.append(waypoints[-1][0])
        sample_lons.append(waypoints[-1][1])

        # Get elevations
        elevations = np.array([
            self.get_elevation(lat, lon)
            for lat, lon in zip(sample_lats, sample_lons)
        ])

        # Compute distances
        distances = [0.0]
        for i in range(1, len(sample_lats)):
            dlat = (sample_lats[i] - sample_lats[i-1]) * 111320
            dlon = (sample_lons[i] - sample_lons[i-1]) * 111320 * np.cos(np.radians(sample_lats[i]))
            dist = np.sqrt(dlat**2 + dlon**2)
            distances.append(distances[-1] + dist)

        distances = np.array(distances)

        # Compute slopes
        slopes = np.zeros(len(elevations))
        for i in range(1, len(elevations)):
            if distances[i] > distances[i-1]:
                slopes[i] = np.degrees(np.arctan(
                    (elevations[i] - elevations[i-1]) / (distances[i] - distances[i-1])
                ))

        return TerrainProfile(
            distances_m=distances,
            elevations_m=elevations,
            slopes_deg=slopes,
            latitudes=np.array(sample_lats),
            longitudes=np.array(sample_lons),
        )

    def get_statistics(self) -> dict[str, Any]:
        """Get terrain statistics."""
        self._ensure_initialized()

        stats = {
            "min_elevation_m": float(np.min(self.elevation)),
            "max_elevation_m": float(np.max(self.elevation)),
            "mean_elevation_m": float(np.mean(self.elevation)),
            "std_elevation_m": float(np.std(self.elevation)),
            "range_m": float(np.ptp(self.elevation)),
            "grid_shape": list(self.shape),
            "cell_size_m": list(self.cell_size_m),
        }

        # Add slope statistics if computed
        if self._slope is not None:
            stats["mean_slope_deg"] = float(np.mean(self._slope))
            stats["max_slope_deg"] = float(np.max(self._slope))

        return stats
