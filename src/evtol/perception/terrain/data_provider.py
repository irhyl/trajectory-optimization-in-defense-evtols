"""
Terrain Data Provider - Real Elevation Data from Global APIs.

This module provides real-time elevation data from multiple global APIs,
with intelligent caching, error handling, and quality metadata.

Data Sources:
    - Open-Meteo Elevation API (Primary): SRTM-based, 30m resolution, global
    - OpenTopography SRTM (Fallback): NASA SRTM, 30m/90m resolution
    - Open-Elevation API (Backup): Community-maintained, global coverage

References:
    - NASA SRTM: https://www2.jpl.nasa.gov/srtm/
    - Open-Meteo: https://open-meteo.com/en/docs/elevation-api
    - SRTM Accuracy: Rodriguez et al. (2006), "An Assessment of the SRTM
      Topographic Products", Photogrammetric Engineering & Remote Sensing

Author: Research-grade implementation for eVTOL trajectory optimization
Version: 1.0.0
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


class ElevationSource(Enum):
    """
    Available elevation data sources with their characteristics.

    Attributes:
        SRTM: NASA Shuttle Radar Topography Mission (30m/90m)
        ASTER: NASA ASTER GDEM v3 (30m)
        COPERNICUS: ESA Copernicus DEM GLO-30 (30m)
        OPEN_METEO: Open-Meteo API (SRTM-based, 30m)
        OPEN_ELEVATION: Community API (mixed sources)
    """
    SRTM = "srtm"
    ASTER = "aster"
    COPERNICUS = "copernicus"
    OPEN_METEO = "open_meteo"
    OPEN_ELEVATION = "open_elevation"


class VerticalDatum(Enum):
    """
    Vertical reference datums for elevation measurements.

    Attributes:
        EGM96: Earth Gravitational Model 1996 (SRTM default)
        EGM2008: Earth Gravitational Model 2008 (improved)
        WGS84: WGS84 ellipsoid height (geometric)
        MSL: Mean Sea Level (local)
    """
    EGM96 = "EGM96"
    EGM2008 = "EGM2008"
    WGS84 = "WGS84"
    MSL = "MSL"


@dataclass
class ElevationPoint:
    """
    Single elevation measurement with full metadata.

    Attributes:
        latitude: WGS84 latitude in decimal degrees
        longitude: WGS84 longitude in decimal degrees
        elevation_m: Elevation above vertical datum in meters
        source: Data source identifier
        resolution_m: Native horizontal resolution in meters
        uncertainty_m: Vertical accuracy (1-sigma) in meters
        vertical_datum: Reference datum for elevation
        timestamp: When the data was retrieved
        is_interpolated: Whether value was interpolated from neighbors
        quality_flag: Data quality indicator (0=good, 1=suspect, 2=void filled)

    Note:
        SRTM absolute vertical accuracy: ±16m (90% LE)
        SRTM relative vertical accuracy: ±6m (90% LE)
        Reference: Rodriguez et al. (2006)
    """
    latitude: float
    longitude: float
    elevation_m: float
    source: ElevationSource = ElevationSource.OPEN_METEO
    resolution_m: float = 30.0
    uncertainty_m: float = 16.0
    vertical_datum: VerticalDatum = VerticalDatum.EGM96
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    is_interpolated: bool = False
    quality_flag: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Export point data as dictionary."""
        return {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "elevation_m": self.elevation_m,
            "source": self.source.value,
            "resolution_m": self.resolution_m,
            "uncertainty_m": self.uncertainty_m,
            "vertical_datum": self.vertical_datum.value,
            "timestamp": self.timestamp.isoformat(),
            "is_interpolated": self.is_interpolated,
            "quality_flag": self.quality_flag,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ElevationPoint:
        """Create ElevationPoint from dictionary."""
        return cls(
            latitude=data["latitude"],
            longitude=data["longitude"],
            elevation_m=data["elevation_m"],
            source=ElevationSource(data.get("source", "open_meteo")),
            resolution_m=data.get("resolution_m", 30.0),
            uncertainty_m=data.get("uncertainty_m", 16.0),
            vertical_datum=VerticalDatum(data.get("vertical_datum", "EGM96")),
            timestamp=datetime.fromisoformat(data["timestamp"]) if "timestamp" in data else datetime.now(timezone.utc),
            is_interpolated=data.get("is_interpolated", False),
            quality_flag=data.get("quality_flag", 0),
        )


@dataclass
class ElevationGrid:
    """
    Grid of elevation data with full geospatial metadata.

    Attributes:
        elevation: 2D array of elevations, shape (n_lat, n_lon)
        latitudes: 1D array of latitude values (north to south)
        longitudes: 1D array of longitude values (west to east)
        source: Data source identifier
        resolution_m: Grid cell size in meters
        bounds: (north, south, east, west) in decimal degrees
        vertical_datum: Reference datum for elevations
        uncertainty_m: Grid-wide vertical accuracy estimate
        void_mask: Boolean mask where True indicates void/invalid data
        timestamp: When the grid was retrieved
    """
    elevation: np.ndarray
    latitudes: np.ndarray
    longitudes: np.ndarray
    source: ElevationSource = ElevationSource.OPEN_METEO
    resolution_m: float = 30.0
    bounds: tuple[float, float, float, float] = (0, 0, 0, 0)
    vertical_datum: VerticalDatum = VerticalDatum.EGM96
    uncertainty_m: float = 16.0
    void_mask: np.ndarray | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self):
        """Validate grid dimensions."""
        if self.elevation.ndim != 2:
            raise ValueError(f"Elevation must be 2D, got shape {self.elevation.shape}")
        if len(self.latitudes) != self.elevation.shape[0]:
            raise ValueError(f"Latitude count {len(self.latitudes)} != elevation rows {self.elevation.shape[0]}")
        if len(self.longitudes) != self.elevation.shape[1]:
            raise ValueError(f"Longitude count {len(self.longitudes)} != elevation cols {self.elevation.shape[1]}")

    @property
    def shape(self) -> tuple[int, int]:
        """Grid shape (n_lat, n_lon)."""
        return self.elevation.shape

    @property
    def cell_size_deg(self) -> tuple[float, float]:
        """Cell size in degrees (dlat, dlon)."""
        dlat = abs(self.latitudes[1] - self.latitudes[0]) if len(self.latitudes) > 1 else 0
        dlon = abs(self.longitudes[1] - self.longitudes[0]) if len(self.longitudes) > 1 else 0
        return (dlat, dlon)

    def get_statistics(self) -> dict[str, float]:
        """Compute grid statistics."""
        valid = self.elevation[~np.isnan(self.elevation)] if self.void_mask is None else self.elevation[~self.void_mask]
        return {
            "min_elevation_m": float(np.min(valid)),
            "max_elevation_m": float(np.max(valid)),
            "mean_elevation_m": float(np.mean(valid)),
            "std_elevation_m": float(np.std(valid)),
            "range_m": float(np.ptp(valid)),
            "void_percentage": float(np.sum(self.void_mask) / self.void_mask.size * 100) if self.void_mask is not None else 0.0,
        }

    def to_dict(self) -> dict[str, Any]:
        """Export grid metadata as dictionary (without arrays)."""
        stats = self.get_statistics()
        return {
            "shape": list(self.shape),
            "bounds": self.bounds,
            "resolution_m": self.resolution_m,
            "cell_size_deg": list(self.cell_size_deg),
            "source": self.source.value,
            "vertical_datum": self.vertical_datum.value,
            "uncertainty_m": self.uncertainty_m,
            "timestamp": self.timestamp.isoformat(),
            "statistics": stats,
        }


@dataclass
class TerrainProviderConfig:
    """
    Configuration for terrain data provider.

    Attributes:
        cache_dir: Directory for caching API responses
        cache_ttl_hours: Cache time-to-live in hours
        timeout_s: Request timeout in seconds
        max_retries: Maximum retry attempts for failed requests
        batch_size: Maximum points per API request
        primary_source: Primary elevation data source
        fallback_sources: Ordered list of fallback sources
        rate_limit_delay_s: Delay between API calls (rate limiting)
    """
    cache_dir: Path = field(default_factory=lambda: Path("outputs/perception/terrain/cache"))
    cache_ttl_hours: float = 168.0  # 1 week - terrain doesn't change
    timeout_s: float = 30.0
    max_retries: int = 3
    batch_size: int = 100
    primary_source: ElevationSource = ElevationSource.OPEN_METEO
    fallback_sources: list[ElevationSource] = field(default_factory=lambda: [ElevationSource.OPEN_ELEVATION])
    rate_limit_delay_s: float = 0.1
    max_grid_points: int = 120000

    def __post_init__(self):
        """Ensure cache directory exists."""
        self.cache_dir = Path(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)


class TerrainDataProvider:
    """
    Real elevation data provider with global coverage.

    This class provides elevation data from real APIs with:
    - Intelligent caching (terrain is static)
    - Multiple fallback sources
    - Batch querying for efficiency
    - Full uncertainty quantification
    - Quality metadata

    Example:
        >>> provider = TerrainDataProvider()
        >>> point = provider.get_elevation(28.6, 77.2)
        >>> print(f"Elevation: {point.elevation_m:.1f}m ± {point.uncertainty_m:.1f}m")

        >>> grid = provider.get_elevation_grid(
        ...     bounds=(28.7, 28.5, 77.3, 77.1),
        ...     resolution_m=100
        ... )
        >>> print(f"Grid shape: {grid.shape}, range: {grid.elevation.min():.0f}-{grid.elevation.max():.0f}m")

    Note:
        Open-Meteo Elevation API:
        - No API key required
        - Global coverage (SRTM + ASTER)
        - ~30m resolution
        - Up to 100 points per request
        - Rate limit: reasonable use expected
    """

    # API endpoints
    OPEN_METEO_URL = "https://api.open-meteo.com/v1/elevation"
    OPEN_ELEVATION_URL = "https://api.open-elevation.com/api/v1/lookup"

    # SRTM coverage limits
    SRTM_LAT_MIN = -60.0
    SRTM_LAT_MAX = 60.0

    def __init__(self, config: TerrainProviderConfig | None = None):
        """
        Initialize terrain data provider.

        Args:
            config: Provider configuration. If None, uses defaults.
        """
        self.config = config or TerrainProviderConfig()
        self._session = self._create_session()
        self._cache: dict[str, Any] = {}
        self._load_cache()

        logger.info(f"TerrainDataProvider initialized, cache: {self.config.cache_dir}")

    def _create_session(self) -> requests.Session:
        """Create HTTP session with retry logic."""
        session = requests.Session()

        retry_strategy = Retry(
            total=self.config.max_retries,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        return session

    def _get_cache_key(self, lat: float, lon: float, resolution_m: float = 30.0) -> str:
        """Generate cache key for a location."""
        # Round to ~10m precision for cache efficiency
        lat_key = round(lat, 4)
        lon_key = round(lon, 4)
        key_str = f"{lat_key}_{lon_key}_{resolution_m}"
        return hashlib.md5(key_str.encode()).hexdigest()[:16]

    def _get_grid_cache_key(self, bounds: tuple, resolution_m: float) -> str:
        """Generate cache key for a grid."""
        key_str = f"{bounds}_{resolution_m}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def _load_cache(self):
        """Load cache from disk."""
        cache_file = self.config.cache_dir / "elevation_cache.json"
        if cache_file.exists():
            try:
                with open(cache_file) as f:
                    self._cache = json.load(f)
                logger.debug(f"Loaded {len(self._cache)} cached elevation points")
            except (OSError, json.JSONDecodeError) as e:
                logger.warning(f"Could not load cache: {e}")
                self._cache = {}

    def _save_cache(self):
        """Save cache to disk."""
        cache_file = self.config.cache_dir / "elevation_cache.json"
        try:
            with open(cache_file, "w") as f:
                json.dump(self._cache, f)
        except OSError as e:
            logger.warning(f"Could not save cache: {e}")

    def _is_cache_valid(self, cache_entry: dict) -> bool:
        """Check if cache entry is still valid."""
        if "timestamp" not in cache_entry:
            return False
        cached_time = datetime.fromisoformat(cache_entry["timestamp"])
        age_hours = (datetime.now(timezone.utc) - cached_time).total_seconds() / 3600
        return age_hours < self.config.cache_ttl_hours

    def _fetch_open_meteo(
        self,
        latitudes: list[float],
        longitudes: list[float],
    ) -> list[float | None]:
        """
        Fetch elevations from Open-Meteo API.

        Args:
            latitudes: List of latitudes
            longitudes: List of longitudes

        Returns:
            List of elevations (None for failures)
        """
        try:
            # Format coordinates for API
            lat_str = ",".join(f"{lat:.6f}" for lat in latitudes)
            lon_str = ",".join(f"{lon:.6f}" for lon in longitudes)

            response = self._session.get(
                self.OPEN_METEO_URL,
                params={"latitude": lat_str, "longitude": lon_str},
                timeout=self.config.timeout_s,
            )
            response.raise_for_status()

            data = response.json()
            elevations = data.get("elevation", [])

            # Handle single point response
            if isinstance(elevations, (int, float)):
                elevations = [elevations]

            return elevations

        except requests.exceptions.RequestException as e:
            logger.error(f"Open-Meteo API error: {e}")
            return [None] * len(latitudes)

    def _fetch_open_elevation(
        self,
        latitudes: list[float],
        longitudes: list[float],
    ) -> list[float | None]:
        """
        Fetch elevations from Open-Elevation API (fallback).

        Args:
            latitudes: List of latitudes
            longitudes: List of longitudes

        Returns:
            List of elevations (None for failures)
        """
        try:
            locations = [
                {"latitude": lat, "longitude": lon}
                for lat, lon in zip(latitudes, longitudes)
            ]

            response = self._session.post(
                self.OPEN_ELEVATION_URL,
                json={"locations": locations},
                timeout=self.config.timeout_s,
            )
            response.raise_for_status()

            data = response.json()
            results = data.get("results", [])

            return [r.get("elevation") for r in results]

        except requests.exceptions.RequestException as e:
            logger.error(f"Open-Elevation API error: {e}")
            return [None] * len(latitudes)

    def get_elevation(
        self,
        latitude: float,
        longitude: float,
        use_cache: bool = True,
    ) -> ElevationPoint:
        """
        Get elevation for a single point.

        Args:
            latitude: WGS84 latitude in decimal degrees
            longitude: WGS84 longitude in decimal degrees
            use_cache: Whether to use cached values

        Returns:
            ElevationPoint with elevation and metadata

        Raises:
            ValueError: If coordinates are invalid

        Example:
            >>> point = provider.get_elevation(28.6139, 77.2090)  # Delhi
            >>> print(f"Elevation: {point.elevation_m:.1f}m")
        """
        # Validate coordinates
        if not -90 <= latitude <= 90:
            raise ValueError(f"Latitude must be in [-90, 90], got {latitude}")
        if not -180 <= longitude <= 180:
            raise ValueError(f"Longitude must be in [-180, 180], got {longitude}")

        # Check cache
        cache_key = self._get_cache_key(latitude, longitude)
        if use_cache and cache_key in self._cache:
            cached = self._cache[cache_key]
            if self._is_cache_valid(cached):
                logger.debug(f"Cache hit for ({latitude}, {longitude})")
                return ElevationPoint.from_dict(cached)

        # Fetch from API
        elevations = self._fetch_open_meteo([latitude], [longitude])

        if elevations[0] is None:
            # Try fallback
            logger.info("Trying fallback source: Open-Elevation")
            elevations = self._fetch_open_elevation([latitude], [longitude])

        elevation = elevations[0] if elevations[0] is not None else 0.0

        # Determine quality
        is_void = elevation == 0.0 and not (abs(latitude) < 1 and abs(longitude) < 1)  # Sea level unlikely inland
        quality_flag = 2 if is_void else 0

        # Determine uncertainty based on source and location
        # SRTM accuracy varies by terrain
        uncertainty = 16.0  # Default SRTM accuracy
        if abs(latitude) > 50:  # Higher latitudes have less coverage
            uncertainty = 20.0

        point = ElevationPoint(
            latitude=latitude,
            longitude=longitude,
            elevation_m=elevation,
            source=ElevationSource.OPEN_METEO,
            resolution_m=30.0,
            uncertainty_m=uncertainty,
            quality_flag=quality_flag,
        )

        # Update cache
        if use_cache:
            self._cache[cache_key] = point.to_dict()
            self._save_cache()

        return point

    def get_elevation_batch(
        self,
        latitudes: list[float],
        longitudes: list[float],
        use_cache: bool = True,
    ) -> list[ElevationPoint]:
        """
        Get elevations for multiple points efficiently.

        Args:
            latitudes: List of WGS84 latitudes
            longitudes: List of WGS84 longitudes
            use_cache: Whether to use cached values

        Returns:
            List of ElevationPoints

        Note:
            Points are batched for API efficiency.
            Cached points are returned immediately.
        """
        if len(latitudes) != len(longitudes):
            raise ValueError("Latitude and longitude lists must have same length")

        results: list[ElevationPoint | None] = [None] * len(latitudes)
        uncached_indices = []
        uncached_lats = []
        uncached_lons = []

        # Check cache first
        for i, (lat, lon) in enumerate(zip(latitudes, longitudes)):
            cache_key = self._get_cache_key(lat, lon)
            if use_cache and cache_key in self._cache:
                cached = self._cache[cache_key]
                if self._is_cache_valid(cached):
                    results[i] = ElevationPoint.from_dict(cached)
                    continue
            uncached_indices.append(i)
            uncached_lats.append(lat)
            uncached_lons.append(lon)

        logger.debug(f"Batch query: {len(latitudes)} points, {len(uncached_indices)} uncached")

        # Fetch uncached points in batches
        for batch_start in range(0, len(uncached_lats), self.config.batch_size):
            batch_end = min(batch_start + self.config.batch_size, len(uncached_lats))
            batch_lats = uncached_lats[batch_start:batch_end]
            batch_lons = uncached_lons[batch_start:batch_end]
            batch_indices = uncached_indices[batch_start:batch_end]

            # Fetch from API
            elevations = self._fetch_open_meteo(batch_lats, batch_lons)

            # Check for failures and try fallback
            failed_indices = [i for i, e in enumerate(elevations) if e is None]
            if failed_indices:
                logger.info(f"Retrying {len(failed_indices)} points with fallback")
                failed_lats = [batch_lats[i] for i in failed_indices]
                failed_lons = [batch_lons[i] for i in failed_indices]
                fallback_elevations = self._fetch_open_elevation(failed_lats, failed_lons)
                for i, elev in zip(failed_indices, fallback_elevations):
                    elevations[i] = elev

            # Create points
            for i, (lat, lon, elev) in enumerate(zip(batch_lats, batch_lons, elevations)):
                elev = elev if elev is not None else 0.0
                point = ElevationPoint(
                    latitude=lat,
                    longitude=lon,
                    elevation_m=elev,
                    source=ElevationSource.OPEN_METEO,
                    resolution_m=30.0,
                    uncertainty_m=16.0,
                    quality_flag=2 if elev == 0.0 else 0,
                )
                results[batch_indices[i]] = point

                # Update cache
                if use_cache:
                    cache_key = self._get_cache_key(lat, lon)
                    self._cache[cache_key] = point.to_dict()

            # Rate limiting
            if batch_end < len(uncached_lats):
                time.sleep(self.config.rate_limit_delay_s)

        # Save cache
        if use_cache and uncached_indices:
            self._save_cache()

        return results

    def get_elevation_grid(
        self,
        bounds: tuple[float, float, float, float],
        resolution_m: float = 100.0,
        use_cache: bool = True,
    ) -> ElevationGrid:
        """
        Get elevation grid for a geographic region.

        Args:
            bounds: (north, south, east, west) in decimal degrees
            resolution_m: Desired grid resolution in meters
            use_cache: Whether to use cached values

        Returns:
            ElevationGrid with 2D elevation array

        Example:
            >>> grid = provider.get_elevation_grid(
            ...     bounds=(28.7, 28.5, 77.3, 77.1),  # Delhi area
            ...     resolution_m=100
            ... )
            >>> print(f"Shape: {grid.shape}, Range: {grid.elevation.min():.0f}-{grid.elevation.max():.0f}m")

        Note:
            - Resolution is converted to degrees using latitude
            - Grid is oriented north-south (first row is northernmost)
            - API batching is used for efficiency
        """
        north, south, east, west = bounds

        # Validate bounds
        if north <= south:
            raise ValueError(f"North ({north}) must be greater than south ({south})")
        if east <= west:
            raise ValueError(f"East ({east}) must be greater than west ({west})")

        # Convert resolution to degrees
        # At equator: 1° ≈ 111,320m latitude, varies for longitude
        center_lat = (north + south) / 2
        meters_per_deg_lat = 111320.0
        meters_per_deg_lon = max(111320.0 * np.cos(np.radians(center_lat)), 1e-6)

        dlat = resolution_m / meters_per_deg_lat
        dlon = resolution_m / meters_per_deg_lon

        # Create grid coordinates
        latitudes = np.arange(north, south - dlat/2, -dlat)  # North to south
        longitudes = np.arange(west, east + dlon/2, dlon)   # West to east

        n_lat = len(latitudes)
        n_lon = len(longitudes)
        n_points = n_lat * n_lon

        # Keep memory and API volume bounded for very large regions.
        if n_points > self.config.max_grid_points:
            scale = np.sqrt(n_points / self.config.max_grid_points)
            dlat *= scale
            dlon *= scale
            latitudes = np.arange(north, south - dlat/2, -dlat)
            longitudes = np.arange(west, east + dlon/2, dlon)
            n_lat = len(latitudes)
            n_lon = len(longitudes)
            n_points = n_lat * n_lon
            logger.warning(
                "Requested terrain grid exceeded max points; auto-coarsened to %dx%d (%d points)",
                n_lat,
                n_lon,
                n_points,
            )

        logger.info(f"Creating elevation grid: {n_lat}x{n_lon} = {n_points} points")

        # Check grid cache
        grid_cache_key = self._get_grid_cache_key(bounds, resolution_m)
        grid_cache_file = self.config.cache_dir / f"grid_{grid_cache_key}.npz"

        if use_cache and grid_cache_file.exists():
            try:
                cached = np.load(grid_cache_file)
                logger.info(f"Loaded cached grid from {grid_cache_file.name}")
                return ElevationGrid(
                    elevation=cached["elevation"],
                    latitudes=cached["latitudes"],
                    longitudes=cached["longitudes"],
                    source=ElevationSource.OPEN_METEO,
                    resolution_m=resolution_m,
                    bounds=bounds,
                )
            except Exception as e:
                logger.warning(f"Could not load cached grid: {e}")

        # Create flat coordinate lists for batch query
        lats_flat = []
        lons_flat = []
        for lat in latitudes:
            for lon in longitudes:
                lats_flat.append(lat)
                lons_flat.append(lon)

        # Fetch all points
        points = self.get_elevation_batch(lats_flat, lons_flat, use_cache=use_cache)

        # Reshape to grid
        elevation = np.array([p.elevation_m for p in points]).reshape(n_lat, n_lon)

        # Create void mask for zero elevations (potential voids)
        void_mask = (elevation == 0.0)

        # Create grid
        grid = ElevationGrid(
            elevation=elevation,
            latitudes=latitudes,
            longitudes=longitudes,
            source=ElevationSource.OPEN_METEO,
            resolution_m=resolution_m,
            bounds=bounds,
            void_mask=void_mask,
        )

        # Cache the grid
        if use_cache:
            try:
                np.savez_compressed(
                    grid_cache_file,
                    elevation=elevation,
                    latitudes=latitudes,
                    longitudes=longitudes,
                )
                logger.info(f"Cached grid to {grid_cache_file.name}")
            except Exception as e:
                logger.warning(f"Could not cache grid: {e}")

        return grid

    def get_elevation_profile(
        self,
        waypoints: list[tuple[float, float]],
        samples_per_segment: int = 10,
        use_cache: bool = True,
    ) -> tuple[list[ElevationPoint], np.ndarray]:
        """
        Get elevation profile along a path.

        Args:
            waypoints: List of (latitude, longitude) tuples
            samples_per_segment: Number of samples between waypoints
            use_cache: Whether to use cached values

        Returns:
            Tuple of:
                - List of ElevationPoints along path
                - Cumulative distance array in meters

        Example:
            >>> path = [(28.6, 77.2), (28.7, 77.3), (28.8, 77.2)]
            >>> points, distances = provider.get_elevation_profile(path)
            >>> print(f"Profile: {len(points)} points, {distances[-1]/1000:.1f} km")
        """
        if len(waypoints) < 2:
            raise ValueError("At least 2 waypoints required for profile")

        # Generate sample points along path
        sample_lats = []
        sample_lons = []

        for i in range(len(waypoints) - 1):
            lat1, lon1 = waypoints[i]
            lat2, lon2 = waypoints[i + 1]

            for j in range(samples_per_segment):
                t = j / samples_per_segment
                lat = lat1 + t * (lat2 - lat1)
                lon = lon1 + t * (lon2 - lon1)
                sample_lats.append(lat)
                sample_lons.append(lon)

        # Add final point
        sample_lats.append(waypoints[-1][0])
        sample_lons.append(waypoints[-1][1])

        # Fetch elevations
        points = self.get_elevation_batch(sample_lats, sample_lons, use_cache=use_cache)

        # Compute cumulative distances
        distances = [0.0]
        for i in range(1, len(points)):
            dlat = points[i].latitude - points[i-1].latitude
            dlon = points[i].longitude - points[i-1].longitude

            # Haversine approximation for short distances
            lat_m = dlat * 111320
            lon_m = dlon * 111320 * np.cos(np.radians(points[i].latitude))
            dist = np.sqrt(lat_m**2 + lon_m**2)
            distances.append(distances[-1] + dist)

        return points, np.array(distances)

    def get_terrain_statistics(
        self,
        bounds: tuple[float, float, float, float],
        resolution_m: float = 100.0,
    ) -> dict[str, Any]:
        """
        Compute terrain statistics for a region.

        Args:
            bounds: (north, south, east, west) in decimal degrees
            resolution_m: Analysis resolution in meters

        Returns:
            Dictionary of terrain statistics
        """
        grid = self.get_elevation_grid(bounds, resolution_m)
        stats = grid.get_statistics()

        # Add derived statistics
        stats["bounds"] = bounds
        stats["resolution_m"] = resolution_m
        stats["grid_shape"] = list(grid.shape)
        stats["source"] = grid.source.value

        return stats


# Module-level convenience functions
_default_provider: TerrainDataProvider | None = None


def get_default_provider() -> TerrainDataProvider:
    """Get or create the default terrain data provider."""
    global _default_provider
    if _default_provider is None:
        _default_provider = TerrainDataProvider()
    return _default_provider


def get_elevation(latitude: float, longitude: float) -> ElevationPoint:
    """
    Get elevation for a single point using default provider.

    Args:
        latitude: WGS84 latitude in decimal degrees
        longitude: WGS84 longitude in decimal degrees

    Returns:
        ElevationPoint with elevation and metadata

    Example:
        >>> from evtol.perception.terrain import get_elevation
        >>> point = get_elevation(28.6139, 77.2090)  # Delhi
        >>> print(f"Elevation: {point.elevation_m:.1f}m")
    """
    return get_default_provider().get_elevation(latitude, longitude)


def get_elevation_grid(
    bounds: tuple[float, float, float, float],
    resolution_m: float = 100.0,
) -> ElevationGrid:
    """
    Get elevation grid for a region using default provider.

    Args:
        bounds: (north, south, east, west) in decimal degrees
        resolution_m: Grid resolution in meters

    Returns:
        ElevationGrid with 2D elevation array

    Example:
        >>> grid = get_elevation_grid((28.7, 28.5, 77.3, 77.1), resolution_m=100)
        >>> print(f"Grid: {grid.shape}, Range: {grid.elevation.min():.0f}-{grid.elevation.max():.0f}m")
    """
    return get_default_provider().get_elevation_grid(bounds, resolution_m)
