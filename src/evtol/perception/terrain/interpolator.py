"""
Terrain Interpolator - High-Precision Elevation Interpolation.

This module provides multiple interpolation methods for terrain data:
- Nearest neighbor (fast, discontinuous)
- Bilinear (default, smooth, fast)
- Bicubic (smoother, better gradients)
- Spline (highest quality, slowest)

Each method has tradeoffs between accuracy, smoothness, and speed.
For trajectory optimization, bilinear is typically sufficient.
For slope/gradient computation, bicubic provides better derivatives.

References:
    - Keys, R. (1981), "Cubic Convolution Interpolation for
      Digital Image Processing", IEEE Trans. ASSP.
    - Unser, M. (1999), "Splines: A Perfect Fit for Signal
      and Image Processing", IEEE Signal Processing Magazine.

Author: Research-grade implementation for eVTOL trajectory optimization
Version: 1.0.0
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
from scipy.interpolate import (
    RegularGridInterpolator,
    RectBivariateSpline,
)

logger = logging.getLogger(__name__)


class InterpolationMethod(Enum):
    """
    Available interpolation methods.

    Attributes:
        NEAREST: Nearest neighbor (fast, discontinuous)
        BILINEAR: Bilinear interpolation (default, smooth)
        BICUBIC: Bicubic interpolation (smoother gradients)
        SPLINE: B-spline interpolation (highest quality)
    """
    NEAREST = "nearest"
    BILINEAR = "bilinear"
    BICUBIC = "bicubic"
    SPLINE = "spline"


@dataclass
class InterpolatedElevation:
    """
    Result of elevation interpolation.

    Attributes:
        latitude: Query latitude
        longitude: Query longitude
        elevation_m: Interpolated elevation in meters
        gradient_lat: Elevation gradient in latitude direction (m/deg)
        gradient_lon: Elevation gradient in longitude direction (m/deg)
        slope_deg: Slope in degrees
        aspect_deg: Aspect in degrees (0=N, 90=E)
        uncertainty_m: Interpolation uncertainty estimate
        method: Interpolation method used
        is_extrapolated: Whether point is outside grid (extrapolated)
    """
    latitude: float
    longitude: float
    elevation_m: float
    gradient_lat: float = 0.0
    gradient_lon: float = 0.0
    slope_deg: float = 0.0
    aspect_deg: float = 0.0
    uncertainty_m: float = 1.0
    method: InterpolationMethod = InterpolationMethod.BILINEAR
    is_extrapolated: bool = False

    @property
    def gradient_magnitude(self) -> float:
        """Total gradient magnitude in m/deg."""
        return np.sqrt(self.gradient_lat**2 + self.gradient_lon**2)

    def to_dict(self) -> dict[str, Any]:
        """Export as dictionary."""
        return {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "elevation_m": self.elevation_m,
            "gradient_lat": self.gradient_lat,
            "gradient_lon": self.gradient_lon,
            "slope_deg": self.slope_deg,
            "aspect_deg": self.aspect_deg,
            "uncertainty_m": self.uncertainty_m,
            "method": self.method.value,
            "is_extrapolated": self.is_extrapolated,
        }


@dataclass
class TrajectoryTerrainInfo:
    """
    Terrain information along a trajectory.

    Attributes:
        waypoints: List of InterpolatedElevation for each waypoint
        distances_m: Cumulative distance along trajectory
        min_elevation_m: Minimum terrain elevation
        max_elevation_m: Maximum terrain elevation
        total_climb_m: Total terrain climb
        total_descent_m: Total terrain descent
        max_slope_deg: Maximum terrain slope encountered
    """
    waypoints: list[InterpolatedElevation]
    distances_m: np.ndarray
    min_elevation_m: float = 0.0
    max_elevation_m: float = 0.0
    total_climb_m: float = 0.0
    total_descent_m: float = 0.0
    max_slope_deg: float = 0.0

    def __post_init__(self):
        """Compute derived statistics."""
        if len(self.waypoints) > 0:
            elevations = np.array([w.elevation_m for w in self.waypoints])
            self.min_elevation_m = float(np.min(elevations))
            self.max_elevation_m = float(np.max(elevations))

            diffs = np.diff(elevations)
            self.total_climb_m = float(np.sum(diffs[diffs > 0]))
            self.total_descent_m = float(np.abs(np.sum(diffs[diffs < 0])))

            slopes = np.array([w.slope_deg for w in self.waypoints])
            self.max_slope_deg = float(np.max(np.abs(slopes)))

    def to_dict(self) -> dict[str, Any]:
        """Export as dictionary."""
        return {
            "n_waypoints": len(self.waypoints),
            "total_distance_m": float(self.distances_m[-1]) if len(self.distances_m) > 0 else 0,
            "min_elevation_m": self.min_elevation_m,
            "max_elevation_m": self.max_elevation_m,
            "total_climb_m": self.total_climb_m,
            "total_descent_m": self.total_descent_m,
            "max_slope_deg": self.max_slope_deg,
        }


class TerrainInterpolator:
    """
    High-precision terrain interpolator with multiple methods.

    This class provides smooth elevation interpolation between grid points,
    essential for trajectory optimization where precise terrain data is needed.

    Example:
        >>> interpolator = TerrainInterpolator(method=InterpolationMethod.BILINEAR)
        >>> interpolator.fit_grid(elevation, latitudes, longitudes)
        >>>
        >>> # Single point interpolation
        >>> result = interpolator.interpolate_point(28.6, 77.2)
        >>> print(f"Elevation: {result.elevation_m:.1f}m, Slope: {result.slope_deg:.1f}°")
        >>>
        >>> # Trajectory interpolation
        >>> path = [(28.6, 77.2), (28.65, 77.25), (28.7, 77.3)]
        >>> info = interpolator.interpolate_trajectory(path)
        >>> print(f"Total climb: {info.total_climb_m:.0f}m")

    Attributes:
        method: Interpolation method
        bounds: Grid bounds (min_lat, max_lat, min_lon, max_lon)
        cell_size_deg: Grid cell size in degrees
    """

    def __init__(
        self,
        method: InterpolationMethod = InterpolationMethod.BILINEAR,
        extrapolate: bool = False,
    ):
        """
        Initialize terrain interpolator.

        Args:
            method: Interpolation method to use
            extrapolate: Whether to allow extrapolation outside grid
        """
        self.method = method
        self.extrapolate = extrapolate

        # Grid data (set by fit_grid)
        self._elevation: np.ndarray | None = None
        self._latitudes: np.ndarray | None = None
        self._longitudes: np.ndarray | None = None

        # Interpolator objects
        self._interpolator: Any | None = None
        self._spline: RectBivariateSpline | None = None

        # Grid metadata
        self._bounds: tuple[float, float, float, float] | None = None
        self._cell_size: tuple[float, float] | None = None

        self._fitted = False

        logger.debug(f"TerrainInterpolator initialized with method={method.value}")

    @property
    def bounds(self) -> tuple[float, float, float, float] | None:
        """Grid bounds (min_lat, max_lat, min_lon, max_lon)."""
        return self._bounds

    @property
    def cell_size_deg(self) -> tuple[float, float] | None:
        """Cell size in degrees (dlat, dlon)."""
        return self._cell_size

    def fit_grid(
        self,
        elevation: np.ndarray,
        latitudes: np.ndarray,
        longitudes: np.ndarray,
    ) -> TerrainInterpolator:
        """
        Fit interpolator to elevation grid.

        Args:
            elevation: 2D elevation array, shape (n_lat, n_lon)
            latitudes: 1D array of latitudes (must be monotonic)
            longitudes: 1D array of longitudes (must be monotonic)

        Returns:
            Self for method chaining

        Raises:
            ValueError: If array dimensions don't match
        """
        # Validate dimensions
        if elevation.ndim != 2:
            raise ValueError(f"Elevation must be 2D, got shape {elevation.shape}")
        if len(latitudes) != elevation.shape[0]:
            raise ValueError(f"Latitude count {len(latitudes)} != elevation rows {elevation.shape[0]}")
        if len(longitudes) != elevation.shape[1]:
            raise ValueError(f"Longitude count {len(longitudes)} != elevation cols {elevation.shape[1]}")

        self._elevation = elevation.astype(np.float64)
        self._latitudes = latitudes.astype(np.float64)
        self._longitudes = longitudes.astype(np.float64)

        # Ensure latitudes are in ascending order for scipy interpolators
        if self._latitudes[0] > self._latitudes[-1]:
            self._latitudes = self._latitudes[::-1]
            self._elevation = self._elevation[::-1, :]

        # Compute bounds and cell size
        self._bounds = (
            float(self._latitudes.min()),
            float(self._latitudes.max()),
            float(self._longitudes.min()),
            float(self._longitudes.max()),
        )

        self._cell_size = (
            abs(self._latitudes[1] - self._latitudes[0]) if len(self._latitudes) > 1 else 0,
            abs(self._longitudes[1] - self._longitudes[0]) if len(self._longitudes) > 1 else 0,
        )

        # Create interpolator based on method
        bounds_error = not self.extrapolate
        fill_value = np.nan if not self.extrapolate else None

        if self.method == InterpolationMethod.NEAREST:
            self._interpolator = RegularGridInterpolator(
                (self._latitudes, self._longitudes),
                self._elevation,
                method='nearest',
                bounds_error=bounds_error,
                fill_value=fill_value,
            )

        elif self.method == InterpolationMethod.BILINEAR:
            self._interpolator = RegularGridInterpolator(
                (self._latitudes, self._longitudes),
                self._elevation,
                method='linear',
                bounds_error=bounds_error,
                fill_value=fill_value,
            )

        elif self.method == InterpolationMethod.BICUBIC:
            # Use cubic interpolation via map_coordinates
            self._interpolator = RegularGridInterpolator(
                (self._latitudes, self._longitudes),
                self._elevation,
                method='cubic',
                bounds_error=bounds_error,
                fill_value=fill_value,
            )

        elif self.method == InterpolationMethod.SPLINE:
            # B-spline interpolation (highest quality)
            self._spline = RectBivariateSpline(
                self._latitudes,
                self._longitudes,
                self._elevation,
                kx=3,  # Cubic spline
                ky=3,
            )

        self._fitted = True

        logger.info(f"Fitted interpolator: {elevation.shape}, bounds={self._bounds}")

        return self

    def _ensure_fitted(self):
        """Ensure interpolator is fitted."""
        if not self._fitted:
            raise RuntimeError("Interpolator not fitted. Call fit_grid() first.")

    def _is_in_bounds(self, latitude: float, longitude: float) -> bool:
        """Check if point is within grid bounds."""
        if self._bounds is None:
            return False
        min_lat, max_lat, min_lon, max_lon = self._bounds
        return (min_lat <= latitude <= max_lat) and (min_lon <= longitude <= max_lon)

    def _compute_gradient(
        self,
        latitude: float,
        longitude: float,
        delta: float = 0.0001,  # ~11m
    ) -> tuple[float, float]:
        """
        Compute elevation gradient at a point using central differences.

        Args:
            latitude: Query latitude
            longitude: Query longitude
            delta: Step size in degrees

        Returns:
            Tuple of (gradient_lat, gradient_lon) in m/deg
        """
        # Central difference for latitude gradient
        elev_north = self._interpolate_raw(latitude + delta, longitude)
        elev_south = self._interpolate_raw(latitude - delta, longitude)
        grad_lat = (elev_north - elev_south) / (2 * delta)

        # Central difference for longitude gradient
        elev_east = self._interpolate_raw(latitude, longitude + delta)
        elev_west = self._interpolate_raw(latitude, longitude - delta)
        grad_lon = (elev_east - elev_west) / (2 * delta)

        return (grad_lat, grad_lon)

    def _interpolate_raw(self, latitude: float, longitude: float) -> float:
        """Raw interpolation without metadata."""
        if self.method == InterpolationMethod.SPLINE:
            return float(self._spline(latitude, longitude, grid=False))
        else:
            return float(self._interpolator([[latitude, longitude]])[0])

    def interpolate_point(
        self,
        latitude: float,
        longitude: float,
        compute_gradients: bool = True,
    ) -> InterpolatedElevation:
        """
        Interpolate elevation at a single point.

        Args:
            latitude: WGS84 latitude in decimal degrees
            longitude: WGS84 longitude in decimal degrees
            compute_gradients: Whether to compute slope and aspect

        Returns:
            InterpolatedElevation with elevation and metadata

        Example:
            >>> result = interpolator.interpolate_point(28.6, 77.2)
            >>> print(f"Elevation: {result.elevation_m:.1f}m")
        """
        self._ensure_fitted()

        is_extrapolated = not self._is_in_bounds(latitude, longitude)

        # Interpolate elevation
        try:
            if self.method == InterpolationMethod.SPLINE:
                elevation = float(self._spline(latitude, longitude, grid=False))
            else:
                elevation = float(self._interpolator([[latitude, longitude]])[0])
        except ValueError as e:
            if self.extrapolate:
                # Clamp to bounds for extrapolation
                lat_clamped = np.clip(latitude, self._bounds[0], self._bounds[1])
                lon_clamped = np.clip(longitude, self._bounds[2], self._bounds[3])
                elevation = self._interpolate_raw(lat_clamped, lon_clamped)
            else:
                raise ValueError(f"Point ({latitude}, {longitude}) outside grid bounds") from e

        # Handle NaN (shouldn't happen with proper bounds checking)
        if np.isnan(elevation):
            elevation = 0.0

        # Compute gradients if requested
        gradient_lat = 0.0
        gradient_lon = 0.0
        slope_deg = 0.0
        aspect_deg = 0.0

        if compute_gradients and not is_extrapolated:
            try:
                gradient_lat, gradient_lon = self._compute_gradient(latitude, longitude)

                # Convert to slope and aspect
                # Gradients are in m/deg, convert to dimensionless for slope
                meters_per_deg_lat = 111320
                meters_per_deg_lon = 111320 * np.cos(np.radians(latitude))

                dz_dx = gradient_lon / meters_per_deg_lon  # dimensionless
                dz_dy = gradient_lat / meters_per_deg_lat

                slope_rad = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
                slope_deg = float(np.degrees(slope_rad))

                # Aspect (compass convention: 0=N, 90=E)
                aspect_rad = np.arctan2(-dz_dx, dz_dy)
                aspect_deg = float((np.degrees(aspect_rad) + 360) % 360)
            except Exception:
                pass  # Keep defaults if gradient computation fails

        # Estimate uncertainty
        # Uncertainty increases with:
        # - Cell size (coarser grid = more uncertainty)
        # - Slope (steeper terrain = more variability)
        # - Extrapolation
        base_uncertainty = 1.0  # meters
        if self._cell_size:
            cell_factor = max(self._cell_size) / 0.001  # Relative to ~100m cells
            base_uncertainty *= min(cell_factor, 5.0)
        if slope_deg > 10:
            base_uncertainty *= (1 + slope_deg / 45)
        if is_extrapolated:
            base_uncertainty *= 10

        return InterpolatedElevation(
            latitude=latitude,
            longitude=longitude,
            elevation_m=elevation,
            gradient_lat=gradient_lat,
            gradient_lon=gradient_lon,
            slope_deg=slope_deg,
            aspect_deg=aspect_deg,
            uncertainty_m=base_uncertainty,
            method=self.method,
            is_extrapolated=is_extrapolated,
        )

    def interpolate_points(
        self,
        latitudes: np.ndarray,
        longitudes: np.ndarray,
        compute_gradients: bool = True,
    ) -> list[InterpolatedElevation]:
        """
        Interpolate elevation at multiple points.

        Args:
            latitudes: Array of latitudes
            longitudes: Array of longitudes
            compute_gradients: Whether to compute slope and aspect

        Returns:
            List of InterpolatedElevation results
        """
        self._ensure_fitted()

        if len(latitudes) != len(longitudes):
            raise ValueError("Latitude and longitude arrays must have same length")

        return [
            self.interpolate_point(lat, lon, compute_gradients)
            for lat, lon in zip(latitudes, longitudes)
        ]

    def interpolate_trajectory(
        self,
        waypoints: list[tuple[float, float]],
        samples_per_segment: int = 10,
    ) -> TrajectoryTerrainInfo:
        """
        Interpolate terrain along a trajectory.

        Args:
            waypoints: List of (latitude, longitude) tuples
            samples_per_segment: Number of samples between waypoints

        Returns:
            TrajectoryTerrainInfo with terrain data along path

        Example:
            >>> path = [(28.6, 77.2), (28.65, 77.25), (28.7, 77.3)]
            >>> info = interpolator.interpolate_trajectory(path)
            >>> print(f"Total climb: {info.total_climb_m:.0f}m")
        """
        self._ensure_fitted()

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

        # Add final waypoint
        sample_lats.append(waypoints[-1][0])
        sample_lons.append(waypoints[-1][1])

        # Interpolate all points
        interpolated = self.interpolate_points(
            np.array(sample_lats),
            np.array(sample_lons),
            compute_gradients=True,
        )

        # Compute cumulative distances
        distances = [0.0]
        for i in range(1, len(sample_lats)):
            dlat = (sample_lats[i] - sample_lats[i-1]) * 111320
            dlon = (sample_lons[i] - sample_lons[i-1]) * 111320 * np.cos(np.radians(sample_lats[i]))
            dist = np.sqrt(dlat**2 + dlon**2)
            distances.append(distances[-1] + dist)

        return TrajectoryTerrainInfo(
            waypoints=interpolated,
            distances_m=np.array(distances),
        )

    def interpolate_grid(
        self,
        latitudes: np.ndarray,
        longitudes: np.ndarray,
    ) -> np.ndarray:
        """
        Interpolate elevation on a new grid.

        Useful for resampling terrain to a different resolution.

        Args:
            latitudes: 1D array of output latitudes
            longitudes: 1D array of output longitudes

        Returns:
            2D elevation array, shape (len(latitudes), len(longitudes))
        """
        self._ensure_fitted()

        # Create meshgrid
        lon_grid, lat_grid = np.meshgrid(longitudes, latitudes)
        points = np.column_stack([lat_grid.ravel(), lon_grid.ravel()])

        # Interpolate
        if self.method == InterpolationMethod.SPLINE:
            elevations = self._spline(latitudes, longitudes, grid=True)
        else:
            elevations = self._interpolator(points).reshape(len(latitudes), len(longitudes))

        return elevations

    def get_elevation_range(
        self,
        min_lat: float,
        max_lat: float,
        min_lon: float,
        max_lon: float,
        n_samples: int = 100,
    ) -> tuple[float, float]:
        """
        Estimate elevation range in a region.

        Args:
            min_lat: Minimum latitude
            max_lat: Maximum latitude
            min_lon: Minimum longitude
            max_lon: Maximum longitude
            n_samples: Number of sample points

        Returns:
            Tuple of (min_elevation, max_elevation) in meters
        """
        self._ensure_fitted()

        # Random sample points
        lats = np.random.uniform(min_lat, max_lat, n_samples)
        lons = np.random.uniform(min_lon, max_lon, n_samples)

        # Interpolate
        elevations = [self.interpolate_point(lat, lon, compute_gradients=False).elevation_m
                     for lat, lon in zip(lats, lons)]

        return (min(elevations), max(elevations))


# Convenience function
def create_interpolator(
    elevation: np.ndarray,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    method: InterpolationMethod = InterpolationMethod.BILINEAR,
) -> TerrainInterpolator:
    """
    Create and fit a terrain interpolator.

    Args:
        elevation: 2D elevation array
        latitudes: 1D array of latitudes
        longitudes: 1D array of longitudes
        method: Interpolation method

    Returns:
        Fitted TerrainInterpolator

    Example:
        >>> interpolator = create_interpolator(elevation, lats, lons)
        >>> elev = interpolator.interpolate_point(28.6, 77.2).elevation_m
    """
    interpolator = TerrainInterpolator(method=method)
    interpolator.fit_grid(elevation, latitudes, longitudes)
    return interpolator
