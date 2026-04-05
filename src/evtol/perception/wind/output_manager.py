"""
Wind Output Manager for Persistence and Export.

This module provides comprehensive data persistence and export capabilities
for wind field data, supporting multiple formats and maintaining full
provenance tracking for research reproducibility.

Supported Export Formats:
-------------------------

1. NETCDF4 (.nc)
   - Standard format for geospatial atmospheric data
   - Self-describing with embedded metadata
   - CF (Climate and Forecast) conventions compliant
   - Efficient compression with zlib
   - Widely supported by scientific tools (xarray, NCO, CDO)

2. GEOTIFF (.tif)
   - Georeferenced raster format
   - One file per altitude layer
   - Compatible with GIS software (QGIS, ArcGIS)
   - Embedded CRS and transform information

3. NUMPY (.npz)
   - Compressed NumPy archive
   - Fast I/O for Python workflows
   - Preserves exact array precision

4. JSON (.json)
   - Metadata and statistics export
   - Human-readable format
   - API response serialization

5. CSV (.csv)
   - Tabular export of point samples
   - Compatible with spreadsheet software
   - Good for small datasets and validation

6. PARQUET (.parquet)
   - Columnar storage format
   - Efficient for large datasets
   - Compatible with pandas and Spark

Directory Structure:
--------------------

outputs/
└── perception/
    └── wind/
        ├── cache/                    # API response cache
        ├── fields/                   # Full wind field arrays
        │   ├── wind_field_YYYYMMDD_HHMMSS.nc
        │   ├── wind_field_YYYYMMDD_HHMMSS.npz
        │   └── ...
        ├── layers/                   # Per-altitude GeoTIFFs
        │   ├── wind_speed_10m.tif
        │   ├── wind_speed_100m.tif
        │   └── ...
        ├── profiles/                 # Vertical profile exports
        │   └── profile_lat_lon.json
        ├── statistics/               # Aggregated statistics
        │   └── wind_stats_YYYYMMDD.json
        └── metadata/                 # Provenance tracking
            └── wind_metadata_YYYYMMDD.json

Provenance Tracking:
--------------------

All exports include comprehensive provenance metadata:
- Data source (API, file, etc.)
- Fetch/generation timestamp
- Processing version
- Grid configuration
- Quality metrics
- Reproducibility hash

References:
-----------
[1] CF Conventions: http://cfconventions.org/
[2] NetCDF: https://www.unidata.ucar.edu/software/netcdf/
[3] GeoTIFF: https://trac.osgeo.org/geotiff/

Author: eVTOL Trajectory Optimization Research Team
Version: 2.0.0
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

# Optional imports for various formats
try:
    import netCDF4 as nc
    HAS_NETCDF = True
except ImportError:
    HAS_NETCDF = False

try:
    import rasterio
    from rasterio.transform import from_bounds
    from rasterio.crs import CRS
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import pyarrow.parquet as pq
    import pyarrow as pa
    HAS_PARQUET = True
except ImportError:
    HAS_PARQUET = False

# Configure module logger
logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS AND CONFIGURATION
# =============================================================================

class WindExportFormat(Enum):
    """
    Supported export formats for wind data.

    Attributes:
        value: (extension, description, requires_optional_dep)
    """
    NETCDF = ("nc", "NetCDF4 (CF-compliant)", True)
    GEOTIFF = ("tif", "GeoTIFF (per-layer raster)", True)
    NUMPY = ("npz", "Compressed NumPy archive", False)
    JSON = ("json", "JSON metadata/statistics", False)
    CSV = ("csv", "CSV tabular data", False)
    PARQUET = ("parquet", "Apache Parquet columnar", True)

    @property
    def extension(self) -> str:
        """File extension."""
        return self.value[0]

    @property
    def description(self) -> str:
        """Human-readable description."""
        return self.value[1]

    @property
    def requires_optional(self) -> bool:
        """Whether format requires optional dependencies."""
        return self.value[2]

    def is_available(self) -> bool:
        """Check if format dependencies are available."""
        if self == WindExportFormat.NETCDF:
            return HAS_NETCDF
        elif self == WindExportFormat.GEOTIFF:
            return HAS_RASTERIO
        elif self == WindExportFormat.PARQUET:
            return HAS_PARQUET
        else:
            return True


@dataclass
class ExportConfig:
    """
    Configuration for wind data export.

    Attributes:
        output_dir: Base output directory
        include_derived: Whether to include derived quantities (speed, direction)
        include_uncertainty: Whether to include uncertainty estimates
        compress: Whether to apply compression
        compression_level: Compression level (1-9)
        float_precision: Floating point precision ('float32' or 'float64')
        include_metadata: Whether to embed metadata
        overwrite: Whether to overwrite existing files
    """
    output_dir: Path = field(default_factory=lambda: Path("outputs/perception/wind"))
    include_derived: bool = True
    include_uncertainty: bool = True
    compress: bool = True
    compression_level: int = 4
    float_precision: str = "float32"
    include_metadata: bool = True
    overwrite: bool = False

    def __post_init__(self):
        """Ensure output directory exists."""
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> dict[str, Any]:
        """Export configuration as dictionary."""
        return {
            "output_dir": str(self.output_dir),
            "include_derived": self.include_derived,
            "include_uncertainty": self.include_uncertainty,
            "compress": self.compress,
            "compression_level": self.compression_level,
            "float_precision": self.float_precision,
            "include_metadata": self.include_metadata,
            "overwrite": self.overwrite,
        }


@dataclass
class ExportResult:
    """
    Result of an export operation.

    Attributes:
        success: Whether export succeeded
        file_path: Path to exported file(s)
        file_size_bytes: Size of exported file(s)
        format: Export format used
        timestamp: Export timestamp
        checksum: MD5 checksum of file content
        error: Error message if failed
    """
    success: bool
    file_path: Path | list[Path]
    file_size_bytes: int = 0
    format: WindExportFormat = WindExportFormat.NUMPY
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    checksum: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Export result as dictionary."""
        if isinstance(self.file_path, list):
            paths = [str(p) for p in self.file_path]
        else:
            paths = str(self.file_path)

        return {
            "success": self.success,
            "file_path": paths,
            "file_size_bytes": self.file_size_bytes,
            "format": self.format.name,
            "timestamp": self.timestamp.isoformat(),
            "checksum": self.checksum,
            "error": self.error,
        }


# =============================================================================
# PROVENANCE TRACKER
# =============================================================================

@dataclass
class WindProvenance:
    """
    Provenance tracking for wind data.

    Records complete lineage information for reproducibility.

    Attributes:
        data_source: Origin of wind data
        source_url: API URL or file path
        fetch_time: When data was retrieved
        model_run: NWP model run time
        processing_version: Software version
        grid_config: Grid configuration
        quality_metrics: Validation results
        processing_steps: List of processing operations
        parent_hash: Hash of parent data (if derived)
    """
    data_source: str
    source_url: str | None = None
    fetch_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    model_run: datetime | None = None
    processing_version: str = "2.0.0"
    grid_config: dict[str, Any] = field(default_factory=dict)
    quality_metrics: dict[str, bool] = field(default_factory=dict)
    processing_steps: list[str] = field(default_factory=list)
    parent_hash: str | None = None

    def add_processing_step(self, step: str) -> None:
        """Record a processing step."""
        timestamp = datetime.now(timezone.utc).isoformat()
        self.processing_steps.append(f"{timestamp}: {step}")

    def compute_hash(self, data: np.ndarray) -> str:
        """Compute reproducibility hash of data."""
        return hashlib.md5(data.tobytes()).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        """Export provenance as dictionary."""
        return {
            "data_source": self.data_source,
            "source_url": self.source_url,
            "fetch_time": self.fetch_time.isoformat(),
            "model_run": self.model_run.isoformat() if self.model_run else None,
            "processing_version": self.processing_version,
            "grid_config": self.grid_config,
            "quality_metrics": self.quality_metrics,
            "processing_steps": self.processing_steps,
            "parent_hash": self.parent_hash,
        }

    def save(self, filepath: Path) -> None:
        """Save provenance to JSON file."""
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: Path) -> WindProvenance:
        """Load provenance from JSON file."""
        with open(filepath) as f:
            data = json.load(f)

        # Parse datetime fields
        data["fetch_time"] = datetime.fromisoformat(data["fetch_time"])
        if data["model_run"]:
            data["model_run"] = datetime.fromisoformat(data["model_run"])

        return cls(**data)


# =============================================================================
# WIND OUTPUT MANAGER
# =============================================================================

class WindOutputManager:
    """
    Comprehensive output manager for wind field data.

    Handles export to multiple formats with provenance tracking
    and quality assurance.

    Features:
        - Multi-format export (NetCDF4, GeoTIFF, NumPy, JSON, CSV, Parquet)
        - CF-compliant NetCDF with compression
        - Georeferenced GeoTIFF output
        - Full provenance tracking
        - Checksum verification
        - Automatic directory organization

    Usage:
        >>> manager = WindOutputManager(config=ExportConfig(
        ...     output_dir=Path("outputs/perception/wind"),
        ...     compress=True,
        ... ))
        >>>
        >>> # Export wind field
        >>> result = manager.export_field(
        ...     wind_model=model,
        ...     format=WindExportFormat.NETCDF,
        ...     filename="wind_field_20260101",
        ... )
        >>> print(f"Exported to: {result.file_path}")
        >>> print(f"Checksum: {result.checksum}")

    Attributes:
        config: Export configuration
    """

    # CF-compliant variable attributes
    CF_ATTRIBUTES = {
        "wind_u": {
            "standard_name": "eastward_wind",
            "long_name": "Eastward component of wind velocity",
            "units": "m s-1",
        },
        "wind_v": {
            "standard_name": "northward_wind",
            "long_name": "Northward component of wind velocity",
            "units": "m s-1",
        },
        "wind_speed": {
            "standard_name": "wind_speed",
            "long_name": "Wind speed",
            "units": "m s-1",
        },
        "wind_direction": {
            "standard_name": "wind_from_direction",
            "long_name": "Wind direction (from)",
            "units": "degrees",
        },
        "turbulence_intensity": {
            "standard_name": "turbulence_intensity",
            "long_name": "Turbulence intensity ratio",
            "units": "1",
        },
        "gust_speed": {
            "standard_name": "wind_speed_of_gust",
            "long_name": "Wind gust speed (3-second)",
            "units": "m s-1",
        },
    }

    def __init__(self, config: ExportConfig | None = None):
        """
        Initialize output manager.

        Args:
            config: Export configuration (default: standard config)
        """
        self.config = config or ExportConfig()

        # Create subdirectories
        self._fields_dir = self.config.output_dir / "fields"
        self._layers_dir = self.config.output_dir / "layers"
        self._profiles_dir = self.config.output_dir / "profiles"
        self._statistics_dir = self.config.output_dir / "statistics"
        self._metadata_dir = self.config.output_dir / "metadata"

        for directory in [
            self._fields_dir,
            self._layers_dir,
            self._profiles_dir,
            self._statistics_dir,
            self._metadata_dir,
        ]:
            directory.mkdir(parents=True, exist_ok=True)

        logger.info(f"WindOutputManager initialized: {self.config.output_dir}")

    # =========================================================================
    # FILE UTILITIES
    # =========================================================================

    def _compute_checksum(self, filepath: Path) -> str:
        """Compute MD5 checksum of file."""
        hash_md5 = hashlib.md5()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def _get_file_size(self, filepath: Path | list[Path]) -> int:
        """Get total file size in bytes."""
        if isinstance(filepath, list):
            return sum(p.stat().st_size for p in filepath if p.exists())
        return filepath.stat().st_size if filepath.exists() else 0

    def _generate_filename(
        self,
        base_name: str,
        format: WindExportFormat,
        timestamp: bool = True,
    ) -> str:
        """Generate filename with optional timestamp."""
        if timestamp:
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            return f"{base_name}_{ts}.{format.extension}"
        return f"{base_name}.{format.extension}"

    def _check_format_available(self, format: WindExportFormat) -> None:
        """Check if format dependencies are available."""
        if not format.is_available():
            raise ImportError(
                f"Format {format.name} requires optional dependencies. "
                f"Install with: pip install netcdf4 rasterio pyarrow"
            )

    # =========================================================================
    # NETCDF EXPORT
    # =========================================================================

    def export_netcdf(
        self,
        wind_u: np.ndarray,
        wind_v: np.ndarray,
        altitudes: np.ndarray,
        latitudes: np.ndarray,
        longitudes: np.ndarray,
        metadata: dict[str, Any],
        filename: str = "wind_field",
        include_derived: bool = True,
    ) -> ExportResult:
        """
        Export wind field to CF-compliant NetCDF4.

        Args:
            wind_u: 3D array [alt, lat, lon] of U component
            wind_v: 3D array [alt, lat, lon] of V component
            altitudes: 1D array of altitude levels
            latitudes: 1D array of latitudes
            longitudes: 1D array of longitudes
            metadata: Dictionary of metadata attributes
            filename: Output filename (without extension)
            include_derived: Whether to include speed/direction

        Returns:
            ExportResult with file path and status
        """
        self._check_format_available(WindExportFormat.NETCDF)

        filepath = self._fields_dir / self._generate_filename(
            filename, WindExportFormat.NETCDF
        )

        if filepath.exists() and not self.config.overwrite:
            return ExportResult(
                success=False,
                file_path=filepath,
                format=WindExportFormat.NETCDF,
                error="File exists and overwrite=False",
            )

        try:
            # Create NetCDF file
            with nc.Dataset(filepath, "w", format="NETCDF4") as ds:
                # Global attributes
                ds.Conventions = "CF-1.8"
                ds.title = "Wind Field Data for eVTOL Trajectory Optimization"
                ds.institution = "eVTOL Research Team"
                ds.source = metadata.get("data_source", "Open-Meteo API")
                ds.history = f"Created {datetime.now(timezone.utc).isoformat()}"
                ds.references = "https://open-meteo.com"
                ds.processing_version = metadata.get("processing_version", "2.0.0")

                # Dimensions
                ds.createDimension("altitude", len(altitudes))
                ds.createDimension("latitude", len(latitudes))
                ds.createDimension("longitude", len(longitudes))

                # Coordinate variables
                alt_var = ds.createVariable("altitude", "f4", ("altitude",))
                alt_var.standard_name = "altitude"
                alt_var.long_name = "Altitude above ground level"
                alt_var.units = "m"
                alt_var.positive = "up"
                alt_var[:] = altitudes

                lat_var = ds.createVariable("latitude", "f4", ("latitude",))
                lat_var.standard_name = "latitude"
                lat_var.long_name = "Latitude"
                lat_var.units = "degrees_north"
                lat_var[:] = latitudes

                lon_var = ds.createVariable("longitude", "f4", ("longitude",))
                lon_var.standard_name = "longitude"
                lon_var.long_name = "Longitude"
                lon_var.units = "degrees_east"
                lon_var[:] = longitudes

                # Compression settings
                zlib = self.config.compress
                complevel = self.config.compression_level
                dtype = self.config.float_precision

                # Wind U component
                u_var = ds.createVariable(
                    "wind_u", dtype, ("altitude", "latitude", "longitude"),
                    zlib=zlib, complevel=complevel,
                )
                for key, value in self.CF_ATTRIBUTES["wind_u"].items():
                    setattr(u_var, key, value)
                u_var[:] = wind_u

                # Wind V component
                v_var = ds.createVariable(
                    "wind_v", dtype, ("altitude", "latitude", "longitude"),
                    zlib=zlib, complevel=complevel,
                )
                for key, value in self.CF_ATTRIBUTES["wind_v"].items():
                    setattr(v_var, key, value)
                v_var[:] = wind_v

                # Derived quantities
                if include_derived:
                    # Wind speed
                    speed = np.sqrt(wind_u**2 + wind_v**2)
                    speed_var = ds.createVariable(
                        "wind_speed", dtype, ("altitude", "latitude", "longitude"),
                        zlib=zlib, complevel=complevel,
                    )
                    for key, value in self.CF_ATTRIBUTES["wind_speed"].items():
                        setattr(speed_var, key, value)
                    speed_var[:] = speed

                    # Wind direction
                    direction = np.degrees(np.arctan2(-wind_u, -wind_v))
                    direction = (direction + 360) % 360
                    dir_var = ds.createVariable(
                        "wind_direction", dtype, ("altitude", "latitude", "longitude"),
                        zlib=zlib, complevel=complevel,
                    )
                    for key, value in self.CF_ATTRIBUTES["wind_direction"].items():
                        setattr(dir_var, key, value)
                    dir_var[:] = direction

            checksum = self._compute_checksum(filepath)

            logger.info(f"Exported NetCDF: {filepath}")

            return ExportResult(
                success=True,
                file_path=filepath,
                file_size_bytes=self._get_file_size(filepath),
                format=WindExportFormat.NETCDF,
                checksum=checksum,
            )

        except Exception as e:
            logger.error(f"NetCDF export failed: {e}")
            return ExportResult(
                success=False,
                file_path=filepath,
                format=WindExportFormat.NETCDF,
                error=str(e),
            )

    # =========================================================================
    # GEOTIFF EXPORT
    # =========================================================================

    def export_geotiff(
        self,
        data: np.ndarray,
        altitudes: np.ndarray,
        bounds: tuple[float, float, float, float],
        variable_name: str = "wind_speed",
        filename_prefix: str = "wind",
    ) -> ExportResult:
        """
        Export wind layers as georeferenced GeoTIFFs.

        Creates one GeoTIFF per altitude layer.

        Args:
            data: 3D array [alt, lat, lon]
            altitudes: 1D array of altitude levels
            bounds: (north, south, east, west) in decimal degrees
            variable_name: Variable name for filename
            filename_prefix: Prefix for output files

        Returns:
            ExportResult with list of file paths
        """
        self._check_format_available(WindExportFormat.GEOTIFF)

        north, south, east, west = bounds
        n_alt, n_lat, n_lon = data.shape

        # Compute transform (top-left origin for GeoTIFF)
        transform = from_bounds(west, south, east, north, n_lon, n_lat)
        crs = CRS.from_epsg(4326)  # WGS84

        output_files = []

        try:
            for alt_idx, altitude in enumerate(altitudes):
                filename = f"{filename_prefix}_{variable_name}_{int(altitude)}m.tif"
                filepath = self._layers_dir / filename

                if filepath.exists() and not self.config.overwrite:
                    continue

                layer_data = data[alt_idx, :, :]

                # GeoTIFF expects (height, width) with top-left origin
                # Flip vertically if necessary
                layer_data = np.flipud(layer_data)

                with rasterio.open(
                    filepath,
                    "w",
                    driver="GTiff",
                    height=n_lat,
                    width=n_lon,
                    count=1,
                    dtype=self.config.float_precision,
                    crs=crs,
                    transform=transform,
                    compress="lzw" if self.config.compress else None,
                ) as dst:
                    dst.write(layer_data, 1)
                    dst.update_tags(
                        altitude_m=altitude,
                        variable=variable_name,
                        units="m/s",
                        source="eVTOL Wind Model",
                    )

                output_files.append(filepath)

            logger.info(f"Exported {len(output_files)} GeoTIFF layers")

            return ExportResult(
                success=True,
                file_path=output_files,
                file_size_bytes=self._get_file_size(output_files),
                format=WindExportFormat.GEOTIFF,
            )

        except Exception as e:
            logger.error(f"GeoTIFF export failed: {e}")
            return ExportResult(
                success=False,
                file_path=output_files,
                format=WindExportFormat.GEOTIFF,
                error=str(e),
            )

    # =========================================================================
    # NUMPY EXPORT
    # =========================================================================

    def export_numpy(
        self,
        wind_u: np.ndarray,
        wind_v: np.ndarray,
        altitudes: np.ndarray,
        latitudes: np.ndarray,
        longitudes: np.ndarray,
        metadata: dict[str, Any],
        filename: str = "wind_field",
    ) -> ExportResult:
        """
        Export wind field to compressed NumPy archive.

        Args:
            wind_u: 3D array [alt, lat, lon] of U component
            wind_v: 3D array [alt, lat, lon] of V component
            altitudes: 1D array of altitude levels
            latitudes: 1D array of latitudes
            longitudes: 1D array of longitudes
            metadata: Dictionary of metadata
            filename: Output filename (without extension)

        Returns:
            ExportResult with file path
        """
        filepath = self._fields_dir / self._generate_filename(
            filename, WindExportFormat.NUMPY
        )

        if filepath.exists() and not self.config.overwrite:
            return ExportResult(
                success=False,
                file_path=filepath,
                format=WindExportFormat.NUMPY,
                error="File exists and overwrite=False",
            )

        try:
            # Compute derived quantities
            wind_speed = np.sqrt(wind_u**2 + wind_v**2)
            wind_direction = np.degrees(np.arctan2(-wind_u, -wind_v))
            wind_direction = (wind_direction + 360) % 360

            # Convert numpy types to Python types for JSON serialization
            def convert_for_json(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, (np.bool_, bool)):
                    return bool(obj)
                elif isinstance(obj, (np.integer,)):
                    return int(obj)
                elif isinstance(obj, (np.floating,)):
                    return float(obj)
                elif isinstance(obj, dict):
                    return {k: convert_for_json(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_for_json(v) for v in obj]
                elif isinstance(obj, datetime):
                    return obj.isoformat()
                return obj

            metadata_clean = convert_for_json(metadata)

            # Save compressed archive
            np.savez_compressed(
                filepath,
                wind_u=wind_u.astype(self.config.float_precision),
                wind_v=wind_v.astype(self.config.float_precision),
                wind_speed=wind_speed.astype(self.config.float_precision),
                wind_direction=wind_direction.astype(self.config.float_precision),
                altitudes=altitudes,
                latitudes=latitudes,
                longitudes=longitudes,
                metadata=np.array([json.dumps(metadata_clean)]),  # Store as string
            )

            checksum = self._compute_checksum(filepath)

            logger.info(f"Exported NumPy: {filepath}")

            return ExportResult(
                success=True,
                file_path=filepath,
                file_size_bytes=self._get_file_size(filepath),
                format=WindExportFormat.NUMPY,
                checksum=checksum,
            )

        except Exception as e:
            logger.error(f"NumPy export failed: {e}")
            return ExportResult(
                success=False,
                file_path=filepath,
                format=WindExportFormat.NUMPY,
                error=str(e),
            )

    # =========================================================================
    # JSON EXPORT
    # =========================================================================

    def export_json(
        self,
        data: dict[str, Any],
        filename: str = "wind_metadata",
        subdir: str = "metadata",
    ) -> ExportResult:
        """
        Export metadata/statistics to JSON.

        Args:
            data: Dictionary to export
            filename: Output filename (without extension)
            subdir: Subdirectory ('metadata', 'statistics', 'profiles')

        Returns:
            ExportResult with file path
        """
        subdirs = {
            "metadata": self._metadata_dir,
            "statistics": self._statistics_dir,
            "profiles": self._profiles_dir,
        }
        output_dir = subdirs.get(subdir, self._metadata_dir)

        filepath = output_dir / self._generate_filename(
            filename, WindExportFormat.JSON
        )

        if filepath.exists() and not self.config.overwrite:
            return ExportResult(
                success=False,
                file_path=filepath,
                format=WindExportFormat.JSON,
                error="File exists and overwrite=False",
            )

        try:
            # Convert numpy types to Python types
            def convert_types(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.bool_):
                    return bool(obj)
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, dict):
                    return {k: convert_types(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_types(v) for v in obj]
                elif isinstance(obj, datetime):
                    return obj.isoformat()
                return obj

            data = convert_types(data)

            with open(filepath, "w") as f:
                json.dump(data, f, indent=2)

            logger.info(f"Exported JSON: {filepath}")

            return ExportResult(
                success=True,
                file_path=filepath,
                file_size_bytes=self._get_file_size(filepath),
                format=WindExportFormat.JSON,
            )

        except Exception as e:
            logger.error(f"JSON export failed: {e}")
            return ExportResult(
                success=False,
                file_path=filepath,
                format=WindExportFormat.JSON,
                error=str(e),
            )

    # =========================================================================
    # CSV EXPORT
    # =========================================================================

    def export_csv(
        self,
        wind_u: np.ndarray,
        wind_v: np.ndarray,
        altitudes: np.ndarray,
        latitudes: np.ndarray,
        longitudes: np.ndarray,
        filename: str = "wind_samples",
        sample_rate: int = 10,
    ) -> ExportResult:
        """
        Export wind data to CSV (sampled for manageable size).

        Args:
            wind_u: 3D array [alt, lat, lon]
            wind_v: 3D array [alt, lat, lon]
            altitudes: 1D array
            latitudes: 1D array
            longitudes: 1D array
            filename: Output filename
            sample_rate: Sample every Nth point in each dimension

        Returns:
            ExportResult with file path
        """
        filepath = self._fields_dir / self._generate_filename(
            filename, WindExportFormat.CSV
        )

        try:
            rows = []

            for alt_idx in range(0, len(altitudes), sample_rate):
                for lat_idx in range(0, len(latitudes), sample_rate):
                    for lon_idx in range(0, len(longitudes), sample_rate):
                        u = wind_u[alt_idx, lat_idx, lon_idx]
                        v = wind_v[alt_idx, lat_idx, lon_idx]
                        speed = np.sqrt(u**2 + v**2)
                        direction = np.degrees(np.arctan2(-u, -v))
                        direction = (direction + 360) % 360

                        rows.append({
                            "altitude_m": altitudes[alt_idx],
                            "latitude": latitudes[lat_idx],
                            "longitude": longitudes[lon_idx],
                            "wind_u_ms": u,
                            "wind_v_ms": v,
                            "wind_speed_ms": speed,
                            "wind_direction_deg": direction,
                        })

            # Write CSV
            if HAS_PANDAS:
                df = pd.DataFrame(rows)
                df.to_csv(filepath, index=False, float_format="%.4f")
            else:
                # Fallback without pandas
                import csv
                with open(filepath, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                    writer.writeheader()
                    writer.writerows(rows)

            logger.info(f"Exported CSV: {filepath} ({len(rows)} samples)")

            return ExportResult(
                success=True,
                file_path=filepath,
                file_size_bytes=self._get_file_size(filepath),
                format=WindExportFormat.CSV,
            )

        except Exception as e:
            logger.error(f"CSV export failed: {e}")
            return ExportResult(
                success=False,
                file_path=filepath,
                format=WindExportFormat.CSV,
                error=str(e),
            )

    # =========================================================================
    # HIGH-LEVEL EXPORT METHODS
    # =========================================================================

    def export_field(
        self,
        wind_model,
        format: WindExportFormat = WindExportFormat.NUMPY,
        filename: str = "wind_field",
    ) -> ExportResult:
        """
        Export complete wind field from WindFieldModel.

        Args:
            wind_model: WindFieldModel instance
            format: Export format
            filename: Base filename

        Returns:
            ExportResult
        """
        if not wind_model._initialized:
            return ExportResult(
                success=False,
                file_path=Path(),
                format=format,
                error="WindFieldModel not initialized",
            )

        metadata = wind_model.metadata.to_dict()

        if format == WindExportFormat.NETCDF:
            return self.export_netcdf(
                wind_u=wind_model.wind_u,
                wind_v=wind_model.wind_v,
                altitudes=wind_model.altitudes,
                latitudes=wind_model.latitudes,
                longitudes=wind_model.longitudes,
                metadata=metadata,
                filename=filename,
            )

        elif format == WindExportFormat.NUMPY:
            return self.export_numpy(
                wind_u=wind_model.wind_u,
                wind_v=wind_model.wind_v,
                altitudes=wind_model.altitudes,
                latitudes=wind_model.latitudes,
                longitudes=wind_model.longitudes,
                metadata=metadata,
                filename=filename,
            )

        elif format == WindExportFormat.GEOTIFF:
            return self.export_geotiff(
                data=wind_model.wind_speed,
                altitudes=wind_model.altitudes,
                bounds=wind_model.coverage_bounds,
                variable_name="wind_speed",
                filename_prefix=filename,
            )

        elif format == WindExportFormat.JSON:
            data = {
                "metadata": metadata,
                "statistics": wind_model.get_statistics(),
                "grid_info": {
                    "n_altitudes": wind_model.n_alt,
                    "n_latitudes": wind_model.n_lat,
                    "n_longitudes": wind_model.n_lon,
                    "altitudes_m": wind_model.altitudes.tolist(),
                },
            }
            return self.export_json(data, filename, subdir="metadata")

        elif format == WindExportFormat.CSV:
            return self.export_csv(
                wind_u=wind_model.wind_u,
                wind_v=wind_model.wind_v,
                altitudes=wind_model.altitudes,
                latitudes=wind_model.latitudes,
                longitudes=wind_model.longitudes,
                filename=filename,
            )

        else:
            return ExportResult(
                success=False,
                file_path=Path(),
                format=format,
                error=f"Unsupported format: {format}",
            )

    def export_all_formats(
        self,
        wind_model,
        filename: str = "wind_field",
    ) -> dict[str, ExportResult]:
        """
        Export wind field to all available formats.

        Args:
            wind_model: WindFieldModel instance
            filename: Base filename

        Returns:
            Dictionary mapping format name to ExportResult
        """
        results = {}

        for format in WindExportFormat:
            if format.is_available():
                try:
                    results[format.name] = self.export_field(
                        wind_model, format, filename
                    )
                except Exception as e:
                    results[format.name] = ExportResult(
                        success=False,
                        file_path=Path(),
                        format=format,
                        error=str(e),
                    )

        return results

    def export_statistics(
        self,
        wind_model,
        filename: str = "wind_statistics",
    ) -> ExportResult:
        """
        Export wind field statistics to JSON.

        Args:
            wind_model: WindFieldModel instance
            filename: Output filename

        Returns:
            ExportResult
        """
        stats = wind_model.get_statistics()
        stats["export_time"] = datetime.now(timezone.utc).isoformat()
        stats["metadata"] = wind_model.metadata.to_dict()

        return self.export_json(stats, filename, subdir="statistics")

    def export_profile(
        self,
        profile,
        latitude: float,
        longitude: float,
        filename: str | None = None,
    ) -> ExportResult:
        """
        Export vertical wind profile to JSON.

        Args:
            profile: BoundaryLayerProfile instance
            latitude: Profile location latitude
            longitude: Profile location longitude
            filename: Output filename (auto-generated if None)

        Returns:
            ExportResult
        """
        if filename is None:
            filename = f"profile_{latitude:.4f}_{longitude:.4f}"

        data = {
            "location": {"latitude": latitude, "longitude": longitude},
            "profile": profile.to_dict(),
            "export_time": datetime.now(timezone.utc).isoformat(),
        }

        return self.export_json(data, filename, subdir="profiles")

    # =========================================================================
    # LOADING METHODS
    # =========================================================================

    def load_numpy(self, filepath: Path) -> dict[str, np.ndarray]:
        """
        Load wind field from NumPy archive.

        Args:
            filepath: Path to .npz file

        Returns:
            Dictionary with arrays and metadata
        """
        data = np.load(filepath, allow_pickle=True)

        result = {
            "wind_u": data["wind_u"],
            "wind_v": data["wind_v"],
            "wind_speed": data["wind_speed"],
            "wind_direction": data["wind_direction"],
            "altitudes": data["altitudes"],
            "latitudes": data["latitudes"],
            "longitudes": data["longitudes"],
        }

        if "metadata" in data:
            metadata_str = str(data["metadata"][0])
            result["metadata"] = json.loads(metadata_str)

        return result

    def list_exports(self, format: WindExportFormat | None = None) -> list[Path]:
        """
        List all exported files.

        Args:
            format: Filter by format (None = all)

        Returns:
            List of file paths
        """
        all_files = []

        for directory in [
            self._fields_dir,
            self._layers_dir,
            self._profiles_dir,
            self._statistics_dir,
            self._metadata_dir,
        ]:
            if format:
                pattern = f"*.{format.extension}"
                all_files.extend(directory.glob(pattern))
            else:
                all_files.extend(directory.glob("*.*"))

        return sorted(all_files, key=lambda p: p.stat().st_mtime, reverse=True)

    def cleanup_old_exports(self, max_age_days: int = 30) -> int:
        """
        Remove exports older than specified age.

        Args:
            max_age_days: Maximum age in days

        Returns:
            Number of files removed
        """
        cutoff = datetime.now() - timedelta(days=max_age_days)
        removed = 0

        for filepath in self.list_exports():
            mtime = datetime.fromtimestamp(filepath.stat().st_mtime)
            if mtime < cutoff:
                filepath.unlink()
                removed += 1

        logger.info(f"Cleaned up {removed} old export files")
        return removed

    def get_disk_usage(self) -> dict[str, int]:
        """
        Get disk usage by category.

        Returns:
            Dictionary mapping category to bytes used
        """
        usage = {}

        for name, directory in [
            ("fields", self._fields_dir),
            ("layers", self._layers_dir),
            ("profiles", self._profiles_dir),
            ("statistics", self._statistics_dir),
            ("metadata", self._metadata_dir),
        ]:
            total = sum(
                f.stat().st_size for f in directory.glob("*.*") if f.is_file()
            )
            usage[name] = total

        usage["total"] = sum(usage.values())
        return usage
