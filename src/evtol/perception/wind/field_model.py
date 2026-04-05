"""
Wind Field Model for eVTOL Trajectory Optimization.

This module provides the core physics-based wind field modeling with:
- Atmospheric Boundary Layer (ABL) wind profile computation
- Turbulence intensity and gust modeling
- Wind shear quantification (vertical and horizontal)
- Energy impact assessment for trajectory planning
- Terrain-induced wind modifications

Mathematical Foundation:
------------------------

1. LOGARITHMIC WIND PROFILE (Neutral ABL)

   The logarithmic wind profile describes wind speed variation with height
   in the surface layer (lowest 10% of ABL):

       U(z) = (u*/κ) × ln((z - d) / z₀)

   where:
       U(z) = wind speed at height z [m/s]
       u* = friction velocity [m/s]
       κ = von Kármán constant (0.41)
       z = height above ground [m]
       d = displacement height [m] (for urban/forest canopies)
       z₀ = aerodynamic roughness length [m]

2. POWER LAW WIND PROFILE (Engineering Approximation)

   Simpler approximation for engineering applications:

       U(z) = U_ref × (z / z_ref)^α

   where:
       α = wind shear exponent (terrain-dependent)

   Typical values of α:
       - Open water: 0.10
       - Open terrain: 0.14
       - Suburban: 0.22
       - Urban: 0.33

3. TURBULENCE INTENSITY

   Turbulence intensity is the ratio of wind speed standard deviation
   to mean wind speed:

       I(z) = σ_u / U(z)

   For neutral conditions (IEC 61400-1):

       I(z) = I_ref × (0.75 + 5.6/z)

   where I_ref is reference turbulence intensity.

4. WIND SHEAR

   Vertical wind shear (speed gradient with altitude):

       S_v = ∂U/∂z ≈ (U(z₂) - U(z₁)) / (z₂ - z₁)

   Wind shear exponent can be computed from two-level measurements:

       α = ln(U₂/U₁) / ln(z₂/z₁)

5. GUST FACTOR

   Ratio of maximum gust to mean wind:

       G = 1 + g_p × I

   where g_p is the peak factor (typically 3.0-3.5 for 3-second gusts).

6. ENERGY IMPACT (Headwind/Tailwind)

   Power required in wind:

       P = 0.5 × ρ × A × C_D × V_a³ + W × V_s

   where:
       V_a = airspeed
       V_s = sink rate

   Headwind increases ground distance per unit energy.
   Tailwind decreases ground distance per unit energy.

References:
-----------
[1] Stull, R. B. (1988). An Introduction to Boundary Layer Meteorology. Springer.
[2] Kaimal, J. C., & Finnigan, J. J. (1994). Atmospheric Boundary Layer Flows. Oxford.
[3] IEC 61400-1:2019. Wind energy generation systems - Design requirements.
[4] Manwell, J. F., McGowan, J. G., & Rogers, A. L. (2009). Wind Energy Explained. Wiley.
[5] Emeis, S. (2018). Wind Energy Meteorology. Springer.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
from scipy.interpolate import RegularGridInterpolator, interp1d
from scipy.ndimage import gaussian_filter

from .data_provider import (
    WindDataProvider,
    WindForecast,
)

# Configure module logger
logger = logging.getLogger(__name__)

# CONSTANTS AND ENUMS
class TerrainRoughness(Enum):
    """
    Terrain roughness categories with aerodynamic roughness lengths.

    Based on Davenport-Wieringa classification and IEC 61400-1.

    Attributes:
        value: (z₀ [m], α, description)
    """
    WATER = (0.0002, 0.10, "Open water, ice")
    OPEN = (0.03, 0.14, "Open terrain, few obstacles")
    FARMLAND = (0.10, 0.16, "Agricultural land, scattered buildings")
    SUBURBAN = (0.50, 0.22, "Suburban, regular obstacles")
    URBAN = (1.50, 0.33, "Urban, dense buildings")
    CITY_CENTER = (3.00, 0.40, "City center, high-rise buildings")

    @property
    def roughness_length(self) -> float:
        """Aerodynamic roughness length z₀ in meters."""
        return self.value[0]

    @property
    def shear_exponent(self) -> float:
        """Power law shear exponent α."""
        return self.value[1]

    @property
    def description(self) -> str:
        """Human-readable description."""
        return self.value[2]


class StabilityClass(Enum):
    """
    Atmospheric stability classes (Pasquill-Gifford).

    Stability affects vertical mixing and wind profile shape.

    Attributes:
        value: (stability parameter, turbulence multiplier)
    """
    A_VERY_UNSTABLE = (-2.0, 1.5, "Very unstable - strong convection")
    B_UNSTABLE = (-1.0, 1.3, "Unstable - moderate convection")
    C_SLIGHTLY_UNSTABLE = (-0.5, 1.1, "Slightly unstable")
    D_NEUTRAL = (0.0, 1.0, "Neutral - mechanical turbulence")
    E_SLIGHTLY_STABLE = (0.5, 0.8, "Slightly stable")
    F_STABLE = (1.0, 0.5, "Stable - suppressed turbulence")

    @property
    def stability_param(self) -> float:
        """Stability parameter (negative = unstable, positive = stable)."""
        return self.value[0]

    @property
    def turbulence_multiplier(self) -> float:
        """Multiplier for turbulence intensity."""
        return self.value[1]

    @property
    def description(self) -> str:
        """Human-readable description."""
        return self.value[2]


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class WindMetadata:
    """
    Metadata for wind field data provenance and quality tracking.

    Attributes:
        resolution_m: Spatial resolution in meters
        coverage_bounds: (north, south, east, west) in decimal degrees
        altitude_bands_m: List of altitude levels in meters
        temporal_resolution_s: Time step in seconds
        data_source: Origin of wind data
        model_run_time: NWP model initialization time
        fetch_time: When data was retrieved
        processing_version: Software version
        quality_metrics: Validation results
    """
    resolution_m: float
    coverage_bounds: tuple[float, float, float, float]
    altitude_bands_m: list[float]
    temporal_resolution_s: int = 3600
    data_source: str = "Open-Meteo"
    model_run_time: datetime | None = None
    fetch_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    processing_version: str = "2.0.0"
    quality_metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Export metadata as dictionary."""
        return {
            "resolution_m": self.resolution_m,
            "coverage_bounds": {
                "north": self.coverage_bounds[0],
                "south": self.coverage_bounds[1],
                "east": self.coverage_bounds[2],
                "west": self.coverage_bounds[3],
            },
            "altitude_bands_m": self.altitude_bands_m,
            "temporal_resolution_s": self.temporal_resolution_s,
            "data_source": self.data_source,
            "model_run_time": self.model_run_time.isoformat() if self.model_run_time else None,
            "fetch_time": self.fetch_time.isoformat(),
            "processing_version": self.processing_version,
            "quality_metrics": self.quality_metrics,
        }


@dataclass
class WindComponents:
    """
    Complete wind vector components at a point.

    Attributes:
        u_ms: U component (west-to-east) in m/s
        v_ms: V component (south-to-north) in m/s
        w_ms: W component (vertical, positive = upward) in m/s
        speed_ms: Horizontal wind speed in m/s
        direction_deg: Wind direction (FROM) in degrees (0=N, 90=E)
        vertical_speed_ms: Vertical velocity in m/s
        turbulence_intensity: σ_u / U ratio (dimensionless)
        gust_speed_ms: Maximum gust speed in m/s
    """
    u_ms: float
    v_ms: float
    w_ms: float = 0.0
    speed_ms: float = field(init=False)
    direction_deg: float = field(init=False)
    turbulence_intensity: float = 0.1
    gust_speed_ms: float = field(init=False)

    def __post_init__(self) -> None:
        """Compute derived quantities."""
        self.speed_ms = np.sqrt(self.u_ms**2 + self.v_ms**2)
        # Meteorological convention: direction wind is FROM
        direction = np.degrees(np.arctan2(-self.u_ms, -self.v_ms))
        self.direction_deg = (direction + 360) % 360
        # Gust = mean + 3σ (99.7th percentile)
        self.gust_speed_ms = self.speed_ms * (1 + 3.0 * self.turbulence_intensity)

    def to_dict(self) -> dict[str, float]:
        """Export as dictionary."""
        return {
            "u_ms": self.u_ms,
            "v_ms": self.v_ms,
            "w_ms": self.w_ms,
            "speed_ms": self.speed_ms,
            "direction_deg": self.direction_deg,
            "turbulence_intensity": self.turbulence_intensity,
            "gust_speed_ms": self.gust_speed_ms,
        }


@dataclass
class BoundaryLayerProfile:
    """
    Atmospheric Boundary Layer wind profile.

    Represents the vertical structure of wind in the ABL.

    Attributes:
        altitudes_m: Array of altitudes
        wind_speed_ms: Wind speed at each altitude
        wind_direction_deg: Direction at each altitude
        turbulence_intensity: TI at each altitude
        shear_exponent: Computed shear exponent
        boundary_layer_height_m: Estimated ABL height
        surface_roughness: Terrain roughness category
        stability: Atmospheric stability class
    """
    altitudes_m: np.ndarray
    wind_speed_ms: np.ndarray
    wind_direction_deg: np.ndarray
    turbulence_intensity: np.ndarray
    shear_exponent: float
    boundary_layer_height_m: float
    surface_roughness: TerrainRoughness
    stability: StabilityClass

    def get_speed_at_altitude(self, altitude_m: float) -> float:
        """Interpolate wind speed at arbitrary altitude."""
        interp = interp1d(
            self.altitudes_m, self.wind_speed_ms,
            kind='linear', bounds_error=False, fill_value='extrapolate'
        )
        return float(interp(altitude_m))

    def get_direction_at_altitude(self, altitude_m: float) -> float:
        """Interpolate wind direction at arbitrary altitude."""
        # Handle circular interpolation for direction
        dirs_rad = np.radians(self.wind_direction_deg)
        sin_interp = interp1d(self.altitudes_m, np.sin(dirs_rad), bounds_error=False, fill_value='extrapolate')
        cos_interp = interp1d(self.altitudes_m, np.cos(dirs_rad), bounds_error=False, fill_value='extrapolate')
        direction = np.degrees(np.arctan2(sin_interp(altitude_m), cos_interp(altitude_m)))
        return float((direction + 360) % 360)

    def to_dict(self) -> dict[str, Any]:
        """Export profile as dictionary."""
        return {
            "altitudes_m": self.altitudes_m.tolist(),
            "wind_speed_ms": self.wind_speed_ms.tolist(),
            "wind_direction_deg": self.wind_direction_deg.tolist(),
            "turbulence_intensity": self.turbulence_intensity.tolist(),
            "shear_exponent": self.shear_exponent,
            "boundary_layer_height_m": self.boundary_layer_height_m,
            "surface_roughness": self.surface_roughness.name,
            "stability": self.stability.name,
        }


# =============================================================================
# WIND FIELD MODEL
# =============================================================================

class WindFieldModel:
    """
    Production-grade wind field model for eVTOL trajectory optimization.

    This class integrates real meteorological data with physics-based
    modeling of the atmospheric boundary layer to provide accurate
    wind estimates for trajectory planning.

    Features:
        - Real data integration via WindDataProvider (Open-Meteo API)
        - Logarithmic and power-law wind profiles
        - Turbulence intensity modeling (IEC 61400-1)
        - Wind shear quantification
        - Terrain roughness effects
        - Atmospheric stability corrections
        - Energy impact assessment for trajectory optimization

    Usage:
        >>> model = WindFieldModel(
        ...     coverage_bounds=(13.0, 12.8, 77.7, 77.5),
        ...     altitude_bands=[10, 50, 100, 200, 500],
        ...     resolution_m=100.0,
        ... )
        >>> model.initialize()
        >>> wind = model.get_wind_at_point(12.9, 77.6, 150.0)
        >>> print(f"Wind: {wind.speed_ms:.1f} m/s from {wind.direction_deg:.0f}°")

    Attributes:
        coverage_bounds: (north, south, east, west) in decimal degrees
        altitude_bands: Target altitudes in meters
        resolution_m: Spatial resolution in meters
        terrain_roughness: Default terrain category
        metadata: Data provenance and quality info
    """

    # Physical constants
    VON_KARMAN = 0.41  # von Kármán constant
    AIR_DENSITY_SL = 1.225  # kg/m³ at sea level
    GRAVITY = 9.80665  # m/s²

    def __init__(
        self,
        coverage_bounds: tuple[float, float, float, float],
        altitude_bands: list[float] | None = None,
        resolution_m: float = 100.0,
        terrain_roughness: TerrainRoughness = TerrainRoughness.SUBURBAN,
        stability: StabilityClass = StabilityClass.D_NEUTRAL,
        cache_dir: Path | None = None,
    ):
        """
        Initialize wind field model.

        Args:
            coverage_bounds: (north, south, east, west) in decimal degrees
            altitude_bands: Altitude levels in meters (default: [10, 50, 100, 200, 500])
            resolution_m: Spatial grid resolution in meters
            terrain_roughness: Default terrain roughness category
            stability: Default atmospheric stability class
            cache_dir: Directory for caching API responses
        """
        self.coverage_bounds = coverage_bounds
        self.altitude_bands = altitude_bands or [10.0, 50.0, 100.0, 200.0, 500.0]
        self.resolution_m = resolution_m
        self.terrain_roughness = terrain_roughness
        self.stability = stability

        # Compute grid dimensions
        north, south, east, west = coverage_bounds
        self.lat_extent_m = (north - south) * 111000  # ~111 km per degree
        self.lon_extent_m = (east - west) * 111000 * np.cos(np.radians((north + south) / 2))

        self.n_lat = max(2, int(self.lat_extent_m / resolution_m))
        self.n_lon = max(2, int(self.lon_extent_m / resolution_m))
        self.n_alt = len(self.altitude_bands)

        # Create coordinate grids
        self.latitudes = np.linspace(south, north, self.n_lat)
        self.longitudes = np.linspace(west, east, self.n_lon)
        self.altitudes = np.array(self.altitude_bands)

        # Initialize data arrays (will be populated by initialize())
        # Shape: [altitude, lat, lon]
        self.wind_u: np.ndarray | None = None  # U component (m/s)
        self.wind_v: np.ndarray | None = None  # V component (m/s)
        self.wind_speed: np.ndarray | None = None  # Speed (m/s)
        self.wind_direction: np.ndarray | None = None  # Direction (degrees)
        self.turbulence_intensity: np.ndarray | None = None  # TI (0-1)
        self.gust_speed: np.ndarray | None = None  # Gust speed (m/s)
        self.wind_shear: np.ndarray | None = None  # Vertical shear (1/s)

        # Interpolators (lazy initialization)
        self._interpolators: dict[str, RegularGridInterpolator] = {}

        # Data provider
        if cache_dir is None:
            cache_dir = Path(__file__).resolve().parents[4] / "outputs" / "perception" / "wind" / "cache"
        self._data_provider = WindDataProvider(cache_dir=cache_dir)

        # Forecast data
        self._forecast: WindForecast | None = None

        # Metadata
        self.metadata = WindMetadata(
            resolution_m=resolution_m,
            coverage_bounds=coverage_bounds,
            altitude_bands_m=self.altitude_bands,
        )

        self._initialized = False
        logger.info(
            "WindFieldModel created: %sx%s grid, %s altitude levels",
            self.n_lat,
            self.n_lon,
            self.n_alt,
        )

    # =========================================================================
    # INITIALIZATION AND DATA LOADING
    # =========================================================================

    def initialize(self, forecast_hours: int = 24) -> None:
        """
        Initialize model by fetching real wind data.

        This method:
        1. Fetches wind forecast from Open-Meteo API
        2. Applies boundary layer physics corrections
        3. Computes derived quantities (turbulence, shear)
        4. Validates data quality

        Args:
            forecast_hours: Hours of forecast to fetch (default: 24)

        Raises:
            RuntimeError: If data fetch fails
        """
        logger.info("Initializing WindFieldModel with real meteorological data...")

        # Fetch forecast data
        try:
            self._forecast = self._data_provider.fetch_forecast(
                bounds=self.coverage_bounds,
                altitude_bands=self.altitude_bands,
                forecast_hours=forecast_hours,
            )
            logger.info("Fetched forecast: %s", self._forecast.grid_shape)
        except Exception as e:
            logger.error("Failed to fetch wind data: %s", e)
            raise RuntimeError(f"Wind data initialization failed: {e}") from e

        # Use first time step for static field (or implement temporal selection)
        time_idx = 0

        # Initialize arrays
        self.wind_u = np.zeros((self.n_alt, self.n_lat, self.n_lon))
        self.wind_v = np.zeros((self.n_alt, self.n_lat, self.n_lon))

        # Map forecast data to our grid
        self._map_forecast_to_grid(time_idx)

        # Apply boundary layer corrections
        self._apply_boundary_layer_physics()

        # Compute derived quantities
        self._compute_derived_quantities()

        # Build interpolators
        self._build_interpolators()

        # Validate
        self._validate_data()

        # Update metadata
        self.metadata.data_source = self._forecast.source.value
        self.metadata.model_run_time = self._forecast.model_run
        self.metadata.fetch_time = self._forecast.fetch_time

        self._initialized = True
        logger.info("WindFieldModel initialization complete")

    def _map_forecast_to_grid(self, time_idx: int = 0) -> None:
        """
        Map forecast data to model grid via interpolation.

        Args:
            time_idx: Time index to use from forecast
        """
        forecast = self._forecast

        # Create interpolators for forecast grid
        for alt_idx, altitude in enumerate(self.altitude_bands):
            # Find nearest level in forecast
            fc_level_idx = forecast.get_level_index(altitude)

            # Get forecast data at this level and time
            fc_u = forecast.wind_u[time_idx, fc_level_idx, :, :]
            fc_v = forecast.wind_v[time_idx, fc_level_idx, :, :]

            # Create interpolator from forecast grid
            fc_lat_interp = RegularGridInterpolator(
                (forecast.latitudes, forecast.longitudes),
                fc_u,
                method='linear',
                bounds_error=False,
                fill_value=None,
            )
            fc_lon_interp = RegularGridInterpolator(
                (forecast.latitudes, forecast.longitudes),
                fc_v,
                method='linear',
                bounds_error=False,
                fill_value=None,
            )

            # Interpolate to our grid
            lat_grid, lon_grid = np.meshgrid(self.latitudes, self.longitudes, indexing='ij')
            points = np.column_stack([lat_grid.ravel(), lon_grid.ravel()])

            self.wind_u[alt_idx, :, :] = fc_lat_interp(points).reshape(self.n_lat, self.n_lon)
            self.wind_v[alt_idx, :, :] = fc_lon_interp(points).reshape(self.n_lat, self.n_lon)

    def _apply_boundary_layer_physics(self) -> None:
        """
        Apply atmospheric boundary layer corrections to wind field.

        This method:
        1. Adjusts wind profile based on surface roughness
        2. Applies stability corrections
        3. Ensures physical consistency across altitudes
        """
        z0 = self.terrain_roughness.roughness_length
        alpha = self.terrain_roughness.shear_exponent
        stability_mult = self.stability.turbulence_multiplier

        # Reference altitude (typically 100m for power law)
        z_ref = 100.0
        ref_idx = np.argmin(np.abs(self.altitudes - z_ref))

        # Get reference speed field at z_ref and preserve spatial structure.
        ref_speed = np.sqrt(
            self.wind_u[ref_idx, :, :] ** 2 + self.wind_v[ref_idx, :, :] ** 2
        )
        ref_speed = np.maximum(ref_speed, 0.01)

        # Apply power law correction for each altitude
        for alt_idx, altitude in enumerate(self.altitudes):
            if altitude < 1.0:
                altitude = 1.0  # Minimum height

            # Power law scaling factor
            if altitude != z_ref:
                scale_factor = (altitude / z_ref) ** alpha

                # Adjust for stability (unstable = more mixing = more uniform profile)
                if self.stability.stability_param < 0:
                    # Unstable: reduce shear
                    scale_factor = scale_factor ** (1.0 / stability_mult)
                elif self.stability.stability_param > 0:
                    # Stable: enhance shear
                    scale_factor = scale_factor ** stability_mult

                # Apply scaling while preserving direction
                current_speed = np.sqrt(
                    self.wind_u[alt_idx, :, :]**2 +
                    self.wind_v[alt_idx, :, :]**2
                )
                target_speed = ref_speed * scale_factor

                # Avoid division by zero
                current_speed = np.maximum(current_speed, 0.01)
                adjustment = target_speed / current_speed

                self.wind_u[alt_idx, :, :] *= adjustment
                self.wind_v[alt_idx, :, :] *= adjustment

        # Apply spatial smoothing for physical consistency
        for alt_idx in range(self.n_alt):
            self.wind_u[alt_idx, :, :] = gaussian_filter(
                self.wind_u[alt_idx, :, :], sigma=1.0
            )
            self.wind_v[alt_idx, :, :] = gaussian_filter(
                self.wind_v[alt_idx, :, :], sigma=1.0
            )

        logger.debug("Applied ABL corrections: z0=%.3fm, alpha=%.2f", z0, alpha)

    def _compute_derived_quantities(self) -> None:
        """
        Compute wind speed, direction, turbulence, shear, and gusts.
        """
        # Wind speed and direction
        self.wind_speed = np.sqrt(self.wind_u**2 + self.wind_v**2)
        direction = np.degrees(np.arctan2(-self.wind_u, -self.wind_v))
        self.wind_direction = (direction + 360) % 360

        # Turbulence intensity (IEC 61400-1 model)
        # I(z) = I_ref × (0.75 + 5.6/z) × stability_multiplier
        I_ref = 0.14  # Reference TI for Category B (medium turbulence)
        stability_mult = self.stability.turbulence_multiplier

        self.turbulence_intensity = np.zeros_like(self.wind_speed)
        for alt_idx, altitude in enumerate(self.altitudes):
            z = max(altitude, 1.0)  # Avoid division by zero
            ti_base = I_ref * (0.75 + 5.6 / z)
            self.turbulence_intensity[alt_idx, :, :] = np.clip(
                ti_base * stability_mult, 0.05, 0.50
            )

        # Gust speed (3-second gusts)
        # G = U × (1 + g_p × I) where g_p ≈ 3.0
        gust_factor = 1.0 + 3.0 * self.turbulence_intensity
        self.gust_speed = self.wind_speed * gust_factor

        # Vertical wind shear (dU/dz)
        self.wind_shear = np.zeros((self.n_lat, self.n_lon))
        if self.n_alt >= 2:
            # Compute shear between lowest and next level
            dU = self.wind_speed[1, :, :] - self.wind_speed[0, :, :]
            dz = self.altitudes[1] - self.altitudes[0]
            self.wind_shear = np.abs(dU / dz)

        logger.debug("Computed derived quantities: speed, direction, TI, gusts, shear")

    def _build_interpolators(self) -> None:
        """Build RegularGridInterpolators for efficient point queries."""
        self._interpolators.clear()

        # Axes: (altitude, latitude, longitude)
        axes = (self.altitudes, self.latitudes, self.longitudes)

        self._interpolators['wind_u'] = RegularGridInterpolator(
            axes, self.wind_u, method='linear', bounds_error=False, fill_value=None
        )
        self._interpolators['wind_v'] = RegularGridInterpolator(
            axes, self.wind_v, method='linear', bounds_error=False, fill_value=None
        )
        self._interpolators['wind_speed'] = RegularGridInterpolator(
            axes, self.wind_speed, method='linear', bounds_error=False, fill_value=None
        )
        # Interpolate direction in unit-circle space to avoid 0/360 wrap artifacts.
        direction_rad = np.radians(self.wind_direction)
        self._interpolators['wind_direction_sin'] = RegularGridInterpolator(
            axes, np.sin(direction_rad), method='linear', bounds_error=False, fill_value=None
        )
        self._interpolators['wind_direction_cos'] = RegularGridInterpolator(
            axes, np.cos(direction_rad), method='linear', bounds_error=False, fill_value=None
        )
        self._interpolators['turbulence_intensity'] = RegularGridInterpolator(
            axes, self.turbulence_intensity, method='linear', bounds_error=False, fill_value=None
        )
        self._interpolators['gust_speed'] = RegularGridInterpolator(
            axes, self.gust_speed, method='linear', bounds_error=False, fill_value=None
        )

        logger.debug("Built interpolators for point queries")

    def _validate_data(self) -> dict[str, bool]:
        """
        Validate wind field data quality.

        Returns:
            Dictionary of validation check results
        """
        checks = {
            "no_nan_wind_u": not np.any(np.isnan(self.wind_u)),
            "no_nan_wind_v": not np.any(np.isnan(self.wind_v)),
            "wind_speed_realistic": np.all(self.wind_speed <= 80.0),  # < 80 m/s
            "wind_speed_positive": np.all(self.wind_speed >= 0),
            "turbulence_bounds": np.all((self.turbulence_intensity >= 0) &
                                        (self.turbulence_intensity <= 0.6)),
            "direction_bounds": np.all((self.wind_direction >= 0) &
                                       (self.wind_direction <= 360)),
        }

        self.metadata.quality_metrics = checks

        if not all(checks.values()):
            failed = [k for k, v in checks.items() if not v]
            logger.warning("Validation failures: %s", failed)
        else:
            logger.info("All validation checks passed")

        return checks

    # =========================================================================
    # POINT QUERIES
    # =========================================================================

    def get_wind_at_point(
        self,
        latitude: float,
        longitude: float,
        altitude_m: float,
    ) -> WindComponents:
        """
        Get interpolated wind components at arbitrary point.

        Uses trilinear interpolation for smooth wind field queries.

        Args:
            latitude: Latitude in decimal degrees
            longitude: Longitude in decimal degrees
            altitude_m: Altitude in meters AGL

        Returns:
            WindComponents at the query point

        Raises:
            ValueError: If model not initialized
        """
        if not self._initialized:
            raise ValueError("Model not initialized. Call initialize() first.")

        # Clamp to valid ranges
        altitude_m = np.clip(altitude_m, self.altitudes[0], self.altitudes[-1])
        latitude = np.clip(latitude, self.latitudes[0], self.latitudes[-1])
        longitude = np.clip(longitude, self.longitudes[0], self.longitudes[-1])

        point = np.array([[altitude_m, latitude, longitude]])

        u = float(self._interpolators['wind_u'](point)[0])
        v = float(self._interpolators['wind_v'](point)[0])
        ti = float(self._interpolators['turbulence_intensity'](point)[0])

        return WindComponents(
            u_ms=u,
            v_ms=v,
            w_ms=0.0,  # Vertical velocity not modeled
            turbulence_intensity=ti,
        )

    def get_wind_profile(
        self,
        latitude: float,
        longitude: float,
    ) -> BoundaryLayerProfile:
        """
        Get vertical wind profile at a location.

        Args:
            latitude: Latitude in decimal degrees
            longitude: Longitude in decimal degrees

        Returns:
            BoundaryLayerProfile with wind vs. altitude
        """
        if not self._initialized:
            raise ValueError("Model not initialized. Call initialize() first.")

        speeds = np.zeros(self.n_alt)
        directions = np.zeros(self.n_alt)
        ti = np.zeros(self.n_alt)

        for alt_idx, altitude in enumerate(self.altitudes):
            wind = self.get_wind_at_point(latitude, longitude, altitude)
            speeds[alt_idx] = wind.speed_ms
            directions[alt_idx] = wind.direction_deg
            ti[alt_idx] = wind.turbulence_intensity

        # Compute shear exponent from profile
        if len(self.altitudes) >= 2:
            # Use lowest two levels
            alpha = np.log(speeds[1] / max(speeds[0], 0.1)) / np.log(self.altitudes[1] / self.altitudes[0])
            alpha = np.clip(alpha, 0.05, 0.50)
        else:
            alpha = self.terrain_roughness.shear_exponent

        # Estimate boundary layer height (simple heuristic)
        # ABL height increases with wind speed and instability
        abl_height = 500.0 + 50.0 * np.mean(speeds)
        if self.stability.stability_param < 0:
            abl_height *= 1.5  # Deeper ABL in unstable conditions
        elif self.stability.stability_param > 0:
            abl_height *= 0.5  # Shallower ABL in stable conditions

        return BoundaryLayerProfile(
            altitudes_m=self.altitudes.copy(),
            wind_speed_ms=speeds,
            wind_direction_deg=directions,
            turbulence_intensity=ti,
            shear_exponent=float(alpha),
            boundary_layer_height_m=float(abl_height),
            surface_roughness=self.terrain_roughness,
            stability=self.stability,
        )

    # =========================================================================
    # ENERGY IMPACT ASSESSMENT
    # =========================================================================

    def calculate_energy_impact(
        self,
        latitude: float,
        longitude: float,
        altitude_m: float,
        heading_deg: float,
        airspeed_ms: float = 35.0,
    ) -> dict[str, float]:
        """
        Calculate wind impact on eVTOL energy consumption.

        This method computes the energy penalty/benefit of wind for a
        given flight heading and airspeed.

        Physics:
            - Headwind: Increases power required, reduces groundspeed
            - Tailwind: Reduces power required, increases groundspeed
            - Crosswind: Requires crab angle, slight energy penalty

        Energy Model:
            Power = P_base × (1 + k × W_headwind / V_air)
            Energy/km = Power / V_ground

        Args:
            latitude: Query latitude
            longitude: Query longitude
            altitude_m: Query altitude in meters
            heading_deg: Aircraft heading (0=N, 90=E, 180=S, 270=W)
            airspeed_ms: True airspeed in m/s (default: 35 m/s)

        Returns:
            Dictionary with energy impact metrics:
                - groundspeed_ms: Resulting ground speed
                - headwind_component_ms: Headwind (+) or tailwind (-)
                - crosswind_component_ms: Crosswind component
                - energy_factor: Multiplier vs. no-wind condition
                - energy_kwh_per_km: Energy consumption per km
        """
        wind = self.get_wind_at_point(latitude, longitude, altitude_m)

        # Compute relative wind angle
        # Wind direction is FROM, heading is TO
        relative_angle_deg = wind.direction_deg - heading_deg
        relative_angle_rad = np.radians(relative_angle_deg)

        # Decompose wind into headwind and crosswind
        # Positive headwind = wind opposing motion
        headwind = wind.speed_ms * np.cos(relative_angle_rad)
        crosswind = wind.speed_ms * np.sin(relative_angle_rad)

        # Ground speed (simplified, ignoring crab angle)
        groundspeed = airspeed_ms - headwind
        groundspeed = max(groundspeed, 5.0)  # Minimum for safety

        # Energy model
        # Base power at cruise (typical eVTOL)
        P_base_kw = 150.0  # kW

        # Energy factor: increases with headwind, decreases with tailwind
        if headwind > 0:
            # Headwind: need more thrust
            energy_factor = 1.0 + 0.12 * (headwind / airspeed_ms)
        else:
            # Tailwind: need less thrust
            energy_factor = 1.0 + 0.06 * (headwind / airspeed_ms)

        # Crosswind penalty (need to crab, slightly less efficient)
        crosswind_penalty = 1.0 + 0.02 * (abs(crosswind) / airspeed_ms) ** 2
        energy_factor *= crosswind_penalty

        # Energy per km
        energy_kwh_per_km = (P_base_kw * energy_factor) / (groundspeed * 3.6)  # Convert to kWh/km

        return {
            "groundspeed_ms": float(groundspeed),
            "headwind_component_ms": float(headwind),
            "crosswind_component_ms": float(crosswind),
            "energy_factor": float(energy_factor),
            "energy_kwh_per_km": float(energy_kwh_per_km),
            "wind_speed_ms": float(wind.speed_ms),
            "wind_direction_deg": float(wind.direction_deg),
            "turbulence_intensity": float(wind.turbulence_intensity),
        }

    # =========================================================================
    # STATISTICS AND EXPORT
    # =========================================================================

    def get_statistics(self) -> dict[str, Any]:
        """
        Get comprehensive wind field statistics.

        Returns:
            Dictionary with statistics by altitude and overall
        """
        if not self._initialized:
            raise ValueError("Model not initialized.")

        stats = {
            "overall": {
                "speed_mean_ms": float(np.mean(self.wind_speed)),
                "speed_std_ms": float(np.std(self.wind_speed)),
                "speed_min_ms": float(np.min(self.wind_speed)),
                "speed_max_ms": float(np.max(self.wind_speed)),
                "gust_max_ms": float(np.max(self.gust_speed)),
                "turbulence_mean": float(np.mean(self.turbulence_intensity)),
                "shear_mean_per_s": float(np.mean(self.wind_shear)),
            },
            "by_altitude": {},
        }

        for alt_idx, altitude in enumerate(self.altitudes):
            speed_layer = self.wind_speed[alt_idx, :, :]
            dir_layer = self.wind_direction[alt_idx, :, :]
            ti_layer = self.turbulence_intensity[alt_idx, :, :]

            stats["by_altitude"][f"{int(altitude)}m"] = {
                "speed_mean_ms": float(np.mean(speed_layer)),
                "speed_std_ms": float(np.std(speed_layer)),
                "speed_min_ms": float(np.min(speed_layer)),
                "speed_max_ms": float(np.max(speed_layer)),
                "direction_mean_deg": float(np.mean(dir_layer)),
                "direction_std_deg": float(np.std(dir_layer)),
                "turbulence_mean": float(np.mean(ti_layer)),
                "turbulence_max": float(np.max(ti_layer)),
            }

        return stats

    def to_dict(self) -> dict[str, Any]:
        """
        Export wind field as dictionary (metadata only, not arrays).

        For full array export, use WindOutputManager.
        """
        return {
            "metadata": self.metadata.to_dict(),
            "grid_shape": {
                "n_altitude": self.n_alt,
                "n_latitude": self.n_lat,
                "n_longitude": self.n_lon,
            },
            "altitudes_m": self.altitudes.tolist(),
            "latitudes": self.latitudes.tolist(),
            "longitudes": self.longitudes.tolist(),
            "terrain_roughness": self.terrain_roughness.name,
            "stability": self.stability.name,
            "statistics": self.get_statistics() if self._initialized else None,
            "initialized": self._initialized,
        }
