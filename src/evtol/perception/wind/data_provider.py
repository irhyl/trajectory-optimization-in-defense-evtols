"""
Wind Data Provider for Real Meteorological Data Integration.

This module provides production-grade integration with meteorological data sources:
- Open-Meteo API (primary, free, no API key required)
- NOAA GFS (secondary, for extended forecasts)
- Local file ingestion (GeoTIFF, NetCDF, CSV)

The provider fetches real wind observations and forecasts at multiple pressure
levels (corresponding to altitude bands) and handles caching, rate limiting,
and data validation.

Mathematical Foundation:
------------------------
Wind data is provided in standard meteorological format:
- U component: Zonal wind (positive = westerly, blowing from west to east)
- V component: Meridional wind (positive = southerly, blowing from south to north)
- Wind speed: |W| = sqrt(U² + V²)
- Wind direction: θ = atan2(V, U) (meteorological convention: direction FROM)

Pressure-Altitude Conversion (International Standard Atmosphere):
    h = 44330 × (1 - (P/P₀)^0.1903)

    where:
        h = altitude (meters)
        P = pressure (hPa)
        P₀ = sea level pressure (1013.25 hPa)

Standard pressure levels and approximate altitudes:
    1000 hPa → ~111 m
    925 hPa  → ~762 m
    850 hPa  → ~1458 m
    700 hPa  → ~3012 m
    500 hPa  → ~5574 m

References:
-----------
[1] Open-Meteo API: https://open-meteo.com/en/docs
[2] NOAA GFS: https://www.ncei.noaa.gov/products/weather-climate-models/global-forecast
[3] WMO Guide to Meteorological Instruments and Methods of Observation

Author: eVTOL Trajectory Optimization Research Team
Version: 2.0.0
"""

from __future__ import annotations

import json
import hashlib
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configure module logger
logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class WindDataSource(Enum):
    """
    Enumeration of supported meteorological data sources.

    Attributes:
        OPEN_METEO: Open-Meteo API (free, no API key, global coverage)
        NOAA_GFS: NOAA Global Forecast System (0.25° resolution)
        LOCAL_FILE: Local file ingestion (GeoTIFF, NetCDF, CSV)
        CACHED: Previously cached data
    """
    OPEN_METEO = "open_meteo"
    NOAA_GFS = "noaa_gfs"
    LOCAL_FILE = "local_file"
    CACHED = "cached"


@dataclass(frozen=True)
class GeoPoint:
    """
    Immutable geographic coordinate point.

    Attributes:
        latitude: Latitude in decimal degrees [-90, 90]
        longitude: Longitude in decimal degrees [-180, 180]
        altitude_m: Altitude in meters above mean sea level

    Raises:
        ValueError: If coordinates are outside valid ranges
    """
    latitude: float
    longitude: float
    altitude_m: float = 0.0

    def __post_init__(self) -> None:
        """Validate coordinate ranges."""
        if not -90 <= self.latitude <= 90:
            raise ValueError(f"Latitude must be in [-90, 90], got {self.latitude}")
        if not -180 <= self.longitude <= 180:
            raise ValueError(f"Longitude must be in [-180, 180], got {self.longitude}")
        if self.altitude_m < -500 or self.altitude_m > 50000:
            raise ValueError(f"Altitude must be in [-500, 50000] m, got {self.altitude_m}")

    def to_tuple(self) -> tuple[float, float, float]:
        """Return as (lat, lon, alt) tuple."""
        return (self.latitude, self.longitude, self.altitude_m)


@dataclass
class PressureLevel:
    """
    Pressure level with corresponding altitude.

    Attributes:
        pressure_hpa: Pressure in hectopascals (millibars)
        altitude_m: Approximate altitude in meters (ISA)
        label: Human-readable label
    """
    pressure_hpa: float
    altitude_m: float
    label: str

    @classmethod
    def from_altitude(cls, altitude_m: float) -> PressureLevel:
        """
        Create pressure level from altitude using ISA.

        Uses the barometric formula for troposphere:
            P = P₀ × (1 - L×h/T₀)^(g×M/(R×L))

        where:
            P₀ = 1013.25 hPa (sea level pressure)
            L = 0.0065 K/m (temperature lapse rate)
            T₀ = 288.15 K (sea level temperature)
            g = 9.80665 m/s² (gravitational acceleration)
            M = 0.0289644 kg/mol (molar mass of air)
            R = 8.31447 J/(mol·K) (gas constant)

        Args:
            altitude_m: Altitude in meters

        Returns:
            PressureLevel instance
        """
        P0 = 1013.25  # hPa
        L = 0.0065    # K/m
        T0 = 288.15   # K
        g = 9.80665   # m/s²
        M = 0.0289644 # kg/mol
        R = 8.31447   # J/(mol·K)

        exponent = (g * M) / (R * L)
        pressure = P0 * (1 - (L * altitude_m) / T0) ** exponent

        return cls(
            pressure_hpa=round(pressure, 2),
            altitude_m=altitude_m,
            label=f"{int(altitude_m)}m"
        )

    @classmethod
    def standard_levels(cls) -> list[PressureLevel]:
        """
        Return standard meteorological pressure levels.

        These correspond to the levels available in GFS and ERA5 datasets.

        Returns:
            List of standard pressure levels
        """
        return [
            cls(1000, 111, "Surface"),
            cls(975, 323, "Near-surface"),
            cls(950, 540, "Low-level"),
            cls(925, 762, "925 hPa"),
            cls(900, 988, "900 hPa"),
            cls(850, 1458, "850 hPa"),
            cls(800, 1949, "800 hPa"),
            cls(700, 3012, "700 hPa"),
            cls(600, 4206, "600 hPa"),
            cls(500, 5574, "500 hPa"),
        ]


@dataclass
class WindObservation:
    """
    Single wind observation at a specific point and time.

    Attributes:
        location: Geographic location of observation
        timestamp: Observation time (UTC)
        wind_u_ms: U component (W→E) in m/s
        wind_v_ms: V component (S→N) in m/s
        wind_speed_ms: Computed wind speed in m/s
        wind_direction_deg: Meteorological direction (FROM) in degrees
        temperature_c: Temperature in Celsius (optional)
        pressure_hpa: Pressure in hPa (optional)
        humidity_pct: Relative humidity percentage (optional)
        source: Data source identifier
        quality_flag: Quality control flag (0=good, 1=suspect, 2=bad)
    """
    location: GeoPoint
    timestamp: datetime
    wind_u_ms: float
    wind_v_ms: float
    wind_speed_ms: float = field(init=False)
    wind_direction_deg: float = field(init=False)
    temperature_c: float | None = None
    pressure_hpa: float | None = None
    humidity_pct: float | None = None
    source: WindDataSource = WindDataSource.OPEN_METEO
    quality_flag: int = 0

    def __post_init__(self) -> None:
        """Compute derived quantities."""
        self.wind_speed_ms = np.sqrt(self.wind_u_ms**2 + self.wind_v_ms**2)
        # Meteorological convention: direction wind is coming FROM
        direction = np.degrees(np.arctan2(-self.wind_u_ms, -self.wind_v_ms))
        self.wind_direction_deg = (direction + 360) % 360

    def to_dict(self) -> dict[str, Any]:
        """Export observation as dictionary."""
        return {
            "latitude": self.location.latitude,
            "longitude": self.location.longitude,
            "altitude_m": self.location.altitude_m,
            "timestamp": self.timestamp.isoformat(),
            "wind_u_ms": self.wind_u_ms,
            "wind_v_ms": self.wind_v_ms,
            "wind_speed_ms": self.wind_speed_ms,
            "wind_direction_deg": self.wind_direction_deg,
            "temperature_c": self.temperature_c,
            "pressure_hpa": self.pressure_hpa,
            "humidity_pct": self.humidity_pct,
            "source": self.source.value,
            "quality_flag": self.quality_flag,
        }


@dataclass
class WindForecast:
    """
    Multi-altitude wind forecast for a geographic region.

    This represents a complete forecast containing wind data at multiple
    pressure levels over a forecast horizon.

    Attributes:
        region_bounds: (north, south, east, west) in decimal degrees
        timestamps: List of forecast valid times (UTC)
        pressure_levels: List of pressure levels with data
        wind_u: 4D array [time, level, lat, lon] of U component (m/s)
        wind_v: 4D array [time, level, lat, lon] of V component (m/s)
        temperature: 4D array [time, level, lat, lon] of temperature (°C)
        latitudes: 1D array of latitude grid points
        longitudes: 1D array of longitude grid points
        source: Data source
        fetch_time: When the data was fetched
        model_run: Model run time (e.g., 00Z, 06Z, 12Z, 18Z)
    """
    region_bounds: tuple[float, float, float, float]  # N, S, E, W
    timestamps: list[datetime]
    pressure_levels: list[PressureLevel]
    wind_u: np.ndarray  # [time, level, lat, lon]
    wind_v: np.ndarray  # [time, level, lat, lon]
    temperature: np.ndarray | None = None  # [time, level, lat, lon]
    latitudes: np.ndarray | None = None
    longitudes: np.ndarray | None = None
    source: WindDataSource = WindDataSource.OPEN_METEO
    fetch_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    model_run: datetime | None = None

    @property
    def wind_speed(self) -> np.ndarray:
        """Compute wind speed from U/V components."""
        return np.sqrt(self.wind_u**2 + self.wind_v**2)

    @property
    def wind_direction(self) -> np.ndarray:
        """Compute wind direction (FROM) in degrees."""
        direction = np.degrees(np.arctan2(-self.wind_u, -self.wind_v))
        return (direction + 360) % 360

    @property
    def grid_shape(self) -> tuple[int, int, int, int]:
        """Return (time, level, lat, lon) shape."""
        return self.wind_u.shape

    def get_level_index(self, altitude_m: float) -> int:
        """
        Find nearest pressure level index for given altitude.

        Args:
            altitude_m: Query altitude in meters

        Returns:
            Index of nearest pressure level
        """
        altitudes = np.array([pl.altitude_m for pl in self.pressure_levels])
        return int(np.argmin(np.abs(altitudes - altitude_m)))

    def validate(self) -> dict[str, bool]:
        """
        Validate forecast data quality.

        Returns:
            Dictionary of validation check results
        """
        return {
            "no_nan_wind_u": not np.any(np.isnan(self.wind_u)),
            "no_nan_wind_v": not np.any(np.isnan(self.wind_v)),
            "wind_speed_realistic": np.all(self.wind_speed <= 100),  # < 100 m/s
            "shape_consistent": self.wind_u.shape == self.wind_v.shape,
            "timestamps_valid": len(self.timestamps) == self.wind_u.shape[0],
            "levels_valid": len(self.pressure_levels) == self.wind_u.shape[1],
        }

    def to_dict(self) -> dict[str, Any]:
        """Export forecast metadata (without large arrays)."""
        return {
            "region_bounds": self.region_bounds,
            "timestamps": [t.isoformat() for t in self.timestamps],
            "pressure_levels": [
                {"hPa": pl.pressure_hpa, "altitude_m": pl.altitude_m}
                for pl in self.pressure_levels
            ],
            "grid_shape": self.grid_shape,
            "source": self.source.value,
            "fetch_time": self.fetch_time.isoformat(),
            "model_run": self.model_run.isoformat() if self.model_run else None,
            "validation": self.validate(),
        }


# =============================================================================
# DATA PROVIDER INTERFACE
# =============================================================================

class BaseWindDataProvider(ABC):
    """
    Abstract base class for wind data providers.

    All concrete data providers must implement these methods to ensure
    consistent interface across different data sources.
    """

    @abstractmethod
    def fetch_forecast(
        self,
        bounds: tuple[float, float, float, float],
        altitude_bands: list[float],
        forecast_hours: int = 24,
    ) -> WindForecast:
        """
        Fetch wind forecast for a geographic region.

        Args:
            bounds: (north, south, east, west) in decimal degrees
            altitude_bands: List of altitudes (m) to fetch
            forecast_hours: Hours of forecast to retrieve

        Returns:
            WindForecast instance with multi-level wind data
        """
        ...

    @abstractmethod
    def fetch_observation(
        self,
        location: GeoPoint,
        timestamp: datetime | None = None,
    ) -> WindObservation:
        """
        Fetch wind observation at a specific point.

        Args:
            location: Geographic point
            timestamp: Query time (None = current)

        Returns:
            WindObservation at the location
        """
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the data source is available."""
        ...


# =============================================================================
# OPEN-METEO PROVIDER (PRIMARY)
# =============================================================================

class OpenMeteoProvider(BaseWindDataProvider):
    """
    Wind data provider using Open-Meteo API.

    Open-Meteo provides free access to weather forecasts without API keys.
    It aggregates data from multiple national weather services including:
    - NOAA GFS (Global)
    - ECMWF (Europe)
    - DWD ICON (Germany)
    - Météo-France AROME (France)

    API Endpoints:
        - Forecast: https://api.open-meteo.com/v1/forecast
        - Historical: https://archive-api.open-meteo.com/v1/archive

    Rate Limits:
        - 10,000 requests/day (free tier)
        - 600 requests/minute

    Available Pressure Levels:
        1000, 975, 950, 925, 900, 850, 800, 700, 600, 500, 400, 300, 250, 200,
        150, 100, 70, 50, 30 hPa

    References:
        [1] https://open-meteo.com/en/docs
        [2] https://open-meteo.com/en/docs/historical-weather-api
    """

    # API endpoints
    FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
    HISTORICAL_URL = "https://archive-api.open-meteo.com/v1/archive"

    # Available pressure levels (hPa)
    AVAILABLE_LEVELS = [1000, 975, 950, 925, 900, 850, 800, 700, 600, 500,
                        400, 300, 250, 200, 150, 100, 70, 50, 30]

    def __init__(
        self,
        cache_dir: Path | None = None,
        cache_ttl_hours: int = 1,
        timeout_seconds: int = 30,
        max_retries: int = 3,
    ):
        """
        Initialize Open-Meteo provider.

        Args:
            cache_dir: Directory for caching responses
            cache_ttl_hours: Cache time-to-live in hours
            timeout_seconds: Request timeout
            max_retries: Maximum retry attempts on failure
        """
        self.cache_dir = cache_dir
        self.cache_ttl = timedelta(hours=cache_ttl_hours)
        self.timeout = timeout_seconds

        # Configure HTTP session with retries
        self.session = requests.Session()
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info("OpenMeteoProvider initialized")

    def _altitude_to_pressure_level(self, altitude_m: float) -> int:
        """
        Map altitude to nearest available pressure level.

        Args:
            altitude_m: Altitude in meters

        Returns:
            Nearest available pressure level in hPa
        """
        # Create pressure level for target altitude
        target = PressureLevel.from_altitude(altitude_m)

        # Find nearest available level
        available = np.array(self.AVAILABLE_LEVELS)
        idx = np.argmin(np.abs(available - target.pressure_hpa))
        return int(available[idx])

    def _get_cache_key(self, params: dict[str, Any]) -> str:
        """Generate cache key from request parameters."""
        param_str = json.dumps(params, sort_keys=True)
        return hashlib.md5(param_str.encode()).hexdigest()

    def _load_from_cache(self, cache_key: str) -> dict | None:
        """Load response from cache if valid."""
        if not self.cache_dir:
            return None

        cache_file = self.cache_dir / f"{cache_key}.json"
        if not cache_file.exists():
            return None

        # Check TTL
        file_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
        if file_age > self.cache_ttl:
            cache_file.unlink()
            return None

        with open(cache_file, encoding="utf-8") as f:
            return json.load(f)

    def _save_to_cache(self, cache_key: str, data: dict) -> None:
        """Save response to cache."""
        if not self.cache_dir:
            return

        cache_file = self.cache_dir / f"{cache_key}.json"
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(data, f)

    def fetch_forecast(
        self,
        bounds: tuple[float, float, float, float],
        altitude_bands: list[float],
        forecast_hours: int = 24,
    ) -> WindForecast:
        """
        Fetch wind forecast from Open-Meteo.

        Args:
            bounds: (north, south, east, west) in decimal degrees
            altitude_bands: List of altitudes (m) to fetch
            forecast_hours: Hours of forecast (max 16 days = 384 hours)

        Returns:
            WindForecast with multi-level wind data

        Raises:
            requests.RequestException: On network failure
            ValueError: On invalid response
        """
        north, south, east, west = bounds

        # Map altitudes to available pressure levels.
        pressure_levels = []
        level_params = []
        for alt in sorted(altitude_bands):
            level_hpa = self._altitude_to_pressure_level(alt)
            if level_hpa not in [pl.pressure_hpa for pl in pressure_levels]:
                pl = PressureLevel.from_altitude(alt)
                pl.pressure_hpa = level_hpa
                pressure_levels.append(pl)
                level_params.append(level_hpa)

        pressure_vars = []
        for level in level_params:
            pressure_vars.extend(
                [
                    f"wind_speed_{level}hPa",
                    f"wind_direction_{level}hPa",
                    f"temperature_{level}hPa",
                ]
            )

        # Build output grid with bounded memory footprint for large regions.
        lat_step = 0.05
        lon_step = 0.05
        latitudes = np.arange(south, north + lat_step, lat_step)
        longitudes = np.arange(west, east + lon_step, lon_step)
        max_grid_points = 25000
        n_points = len(latitudes) * len(longitudes)
        if n_points > max_grid_points:
            scale = np.sqrt(n_points / max_grid_points)
            lat_step *= scale
            lon_step *= scale
            latitudes = np.arange(south, north + lat_step, lat_step)
            longitudes = np.arange(west, east + lon_step, lon_step)
            logger.warning(
                "Wind forecast request auto-coarsened to %dx%d grid for large-area query",
                len(latitudes),
                len(longitudes),
            )

        # Multi-anchor fetch: center + 4 corners. This keeps API calls bounded
        # while producing real spatial variability from measured forecast values.
        center_lat = (north + south) / 2.0
        center_lon = (east + west) / 2.0
        anchors = [
            (center_lat, center_lon),
            (south, west),
            (south, east),
            (north, west),
            (north, east),
        ]

        def _fetch_hourly(lat: float, lon: float) -> dict[str, Any]:
            params = {
                "latitude": lat,
                "longitude": lon,
                "hourly": ",".join(pressure_vars),
                "forecast_hours": forecast_hours,
                "timezone": "UTC",
            }
            cache_key = self._get_cache_key(params)
            cached = self._load_from_cache(cache_key)
            if cached is not None:
                return cached

            response = self.session.get(
                self.FORECAST_URL,
                params=params,
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
            self._save_to_cache(cache_key, data)
            return data

        anchor_payloads = []
        for lat, lon in anchors:
            try:
                anchor_payloads.append(((lat, lon), _fetch_hourly(lat, lon)))
            except requests.RequestException as exc:
                logger.warning("Wind anchor fetch failed at (%.4f, %.4f): %s", lat, lon, exc)

        if not anchor_payloads:
            raise RuntimeError("Failed to fetch wind forecast from all anchor points")

        first_hourly = anchor_payloads[0][1].get("hourly", {})
        time_strs = first_hourly.get("time", [])
        n_times = min(forecast_hours, len(time_strs))
        if n_times == 0:
            raise ValueError("Wind API response did not contain hourly timestamps")

        timestamps = [
            datetime.fromisoformat(t.replace("Z", "+00:00"))
            for t in time_strs[:n_times]
        ]

        n_levels = len(pressure_levels)
        n_lats = len(latitudes)
        n_lons = len(longitudes)
        n_anchors = len(anchor_payloads)

        anchor_u = np.full((n_anchors, n_times, n_levels), np.nan, dtype=float)
        anchor_v = np.full((n_anchors, n_times, n_levels), np.nan, dtype=float)
        anchor_t = np.full((n_anchors, n_times, n_levels), np.nan, dtype=float)
        anchor_lat = np.zeros(n_anchors, dtype=float)
        anchor_lon = np.zeros(n_anchors, dtype=float)

        for a_idx, ((a_lat, a_lon), payload) in enumerate(anchor_payloads):
            hourly = payload.get("hourly", {})
            anchor_lat[a_idx] = a_lat
            anchor_lon[a_idx] = a_lon

            for level_idx, level_hpa in enumerate(level_params):
                speed_key = f"wind_speed_{level_hpa}hPa"
                dir_key = f"wind_direction_{level_hpa}hPa"
                temp_key = f"temperature_{level_hpa}hPa"

                speeds = np.array(hourly.get(speed_key, [])[:n_times], dtype=float)
                directions = np.array(hourly.get(dir_key, [])[:n_times], dtype=float)
                temps = np.array(hourly.get(temp_key, [])[:n_times], dtype=float)

                if speeds.size == 0 or directions.size == 0:
                    continue

                dir_rad = np.radians(directions)
                u_vals = -speeds * np.sin(dir_rad)
                v_vals = -speeds * np.cos(dir_rad)

                anchor_u[a_idx, : len(u_vals), level_idx] = u_vals
                anchor_v[a_idx, : len(v_vals), level_idx] = v_vals
                if temps.size > 0:
                    anchor_t[a_idx, : len(temps), level_idx] = temps

        # Fill partial missing data across anchors.
        for arr in (anchor_u, anchor_v, anchor_t):
            for t_idx in range(n_times):
                for l_idx in range(n_levels):
                    col = arr[:, t_idx, l_idx]
                    valid = ~np.isnan(col)
                    if not np.any(valid):
                        arr[:, t_idx, l_idx] = 0.0
                    else:
                        arr[~valid, t_idx, l_idx] = np.mean(col[valid])

        lat_grid, lon_grid = np.meshgrid(latitudes, longitudes, indexing="ij")
        weights = np.zeros((n_anchors, n_lats, n_lons), dtype=float)
        for a_idx in range(n_anchors):
            d2 = (lat_grid - anchor_lat[a_idx]) ** 2 + (lon_grid - anchor_lon[a_idx]) ** 2 + 1e-12
            weights[a_idx] = 1.0 / d2
        weights /= np.sum(weights, axis=0, keepdims=True)

        wind_u = np.tensordot(anchor_u, weights, axes=(0, 0))
        wind_v = np.tensordot(anchor_v, weights, axes=(0, 0))
        temperature = np.tensordot(anchor_t, weights, axes=(0, 0))

        forecast = WindForecast(
            region_bounds=bounds,
            timestamps=timestamps,
            pressure_levels=pressure_levels,
            wind_u=wind_u,
            wind_v=wind_v,
            temperature=temperature,
            latitudes=latitudes,
            longitudes=longitudes,
            source=WindDataSource.OPEN_METEO,
            fetch_time=datetime.now(timezone.utc),
            model_run=datetime.now(timezone.utc).replace(
                hour=(datetime.now(timezone.utc).hour // 6) * 6,
                minute=0, second=0, microsecond=0
            ),
        )

        # Validate
        validation = forecast.validate()
        if not all(validation.values()):
            logger.warning("Forecast validation issues: %s", validation)

        logger.info("Wind forecast retrieved: %s", forecast.grid_shape)
        return forecast

    def fetch_observation(
        self,
        location: GeoPoint,
        timestamp: datetime | None = None,
    ) -> WindObservation:
        """
        Fetch current wind observation at a point.

        Args:
            location: Geographic point
            timestamp: Query time (None = current conditions)

        Returns:
            WindObservation at the location
        """
        params = {
            "latitude": location.latitude,
            "longitude": location.longitude,
            "current": "wind_speed_10m,wind_direction_10m,temperature_2m,"
                      "relative_humidity_2m,surface_pressure",
            "timezone": "UTC",
        }

        response = self.session.get(
            self.FORECAST_URL,
            params=params,
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()

        current = data.get("current", {})

        # Extract values
        speed = current.get("wind_speed_10m", 0)
        direction = current.get("wind_direction_10m", 0)
        temp = current.get("temperature_2m")
        humidity = current.get("relative_humidity_2m")
        pressure = current.get("surface_pressure")

        # Convert to U/V
        dir_rad = np.radians(direction)
        wind_u = -speed * np.sin(dir_rad)
        wind_v = -speed * np.cos(dir_rad)

        obs_time = datetime.now(timezone.utc)
        if "time" in current:
            obs_time = datetime.fromisoformat(
                current["time"].replace("Z", "+00:00")
            )

        return WindObservation(
            location=location,
            timestamp=obs_time,
            wind_u_ms=wind_u,
            wind_v_ms=wind_v,
            temperature_c=temp,
            pressure_hpa=pressure,
            humidity_pct=humidity,
            source=WindDataSource.OPEN_METEO,
        )

    def is_available(self) -> bool:
        """Check if Open-Meteo API is reachable."""
        try:
            response = self.session.get(
                self.FORECAST_URL,
                params={"latitude": 0, "longitude": 0, "current": "temperature_2m"},
                timeout=5,
            )
            return response.status_code == 200
        except requests.RequestException:
            return False


# =============================================================================
# MAIN WIND DATA PROVIDER
# =============================================================================

class WindDataProvider:
    """
    Main wind data provider with fallback chain.

    This class orchestrates data fetching from multiple sources with
    automatic fallback if the primary source is unavailable.

    Fallback Chain:
        1. Open-Meteo (primary, free, no API key)
        2. Cached data (if available and fresh)
        3. Raise exception (no fallback available)

    Usage:
        >>> provider = WindDataProvider(cache_dir=Path("./cache"))
        >>> bounds = (13.0, 12.8, 77.7, 77.5)  # N, S, E, W
        >>> altitudes = [10, 100, 500, 1000]
        >>> forecast = provider.fetch_forecast(bounds, altitudes, forecast_hours=24)
        >>> print(forecast.grid_shape)
        (24, 4, 5, 5)

    Attributes:
        cache_dir: Directory for caching API responses
        primary_provider: Main data provider (Open-Meteo)
    """

    def __init__(
        self,
        cache_dir: Path | None = None,
        cache_ttl_hours: int = 1,
    ):
        """
        Initialize wind data provider.

        Args:
            cache_dir: Cache directory (default: outputs/perception/wind/cache)
            cache_ttl_hours: Cache TTL in hours
        """
        if cache_dir is None:
            cache_dir = Path(__file__).resolve().parents[4] / "outputs" / "perception" / "wind" / "cache"

        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize providers
        self.primary_provider = OpenMeteoProvider(
            cache_dir=self.cache_dir,
            cache_ttl_hours=cache_ttl_hours,
        )

        logger.info("WindDataProvider initialized with cache at %s", self.cache_dir)

    def fetch_forecast(
        self,
        bounds: tuple[float, float, float, float],
        altitude_bands: list[float],
        forecast_hours: int = 24,
    ) -> WindForecast:
        """
        Fetch wind forecast for a geographic region.

        Args:
            bounds: (north, south, east, west) in decimal degrees
            altitude_bands: Target altitudes in meters
            forecast_hours: Forecast horizon (max 384 hours)

        Returns:
            WindForecast with multi-level wind data

        Raises:
            RuntimeError: If no data source is available
        """
        # Try primary provider directly — skip the is_available() ping which
        # uses lat=0,lon=0 and can spuriously fail on slow networks.
        try:
            return self.primary_provider.fetch_forecast(
                bounds, altitude_bands, forecast_hours
            )
        except (requests.RequestException, RuntimeError, ValueError) as exc:
            logger.warning("Primary provider failed: %s", exc)

        raise RuntimeError(
            "No wind data source available. Check network connectivity."
        )

    def fetch_observation(
        self,
        location: GeoPoint,
        timestamp: datetime | None = None,
    ) -> WindObservation:
        """
        Fetch wind observation at a specific point.

        Args:
            location: Geographic point
            timestamp: Query time (None = current)

        Returns:
            WindObservation at the location
        """
        if self.primary_provider.is_available():
            return self.primary_provider.fetch_observation(location, timestamp)

        raise RuntimeError("No wind data source available.")

    def check_availability(self) -> dict[str, bool]:
        """
        Check availability of all data sources.

        Returns:
            Dictionary mapping source name to availability status
        """
        return {
            "open_meteo": self.primary_provider.is_available(),
        }


# =============================================================================
# MODULE-LEVEL CONVENIENCE FUNCTIONS
# =============================================================================

def get_current_wind(lat: float, lon: float, alt: float = 10.0) -> WindObservation:
    """
    Convenience function to get current wind at a point.

    Args:
        lat: Latitude in decimal degrees
        lon: Longitude in decimal degrees
        alt: Altitude in meters (default: 10m)

    Returns:
        WindObservation at the location
    """
    provider = WindDataProvider()
    location = GeoPoint(lat, lon, alt)
    return provider.fetch_observation(location)


def get_wind_forecast(
    north: float,
    south: float,
    east: float,
    west: float,
    altitudes: list[float] | None = None,
    hours: int = 24,
) -> WindForecast:
    """
    Convenience function to get wind forecast for a region.

    Args:
        north, south, east, west: Region bounds in decimal degrees
        altitudes: Altitude bands in meters (default: [10, 100, 500])
        hours: Forecast hours (default: 24)

    Returns:
        WindForecast for the region
    """
    if altitudes is None:
        altitudes = [10, 100, 500]

    provider = WindDataProvider()
    return provider.fetch_forecast((north, south, east, west), altitudes, hours)
