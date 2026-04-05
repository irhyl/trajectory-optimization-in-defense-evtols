"""
Terrain Output Manager - Multi-format Terrain Data Persistence.

This module provides comprehensive export capabilities for terrain data:
- NumPy compressed arrays (.npz)
- GeoTIFF raster files (.tif) - industry standard
- JSON metadata files
- ESRI ASCII Grid (.asc)
- XYZ point cloud format
- NetCDF for scientific applications

All exports include full provenance metadata for reproducibility.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class TerrainExportFormat(Enum):
    """
    Supported terrain export formats.

    Attributes:
        NUMPY: NumPy compressed format (.npz)
        GEOTIFF: GeoTIFF raster (.tif)
        JSON: JSON metadata/statistics
        ASCII_GRID: ESRI ASCII Grid (.asc)
        XYZ: XYZ point cloud format
        NETCDF: NetCDF4 scientific format (.nc)
    """
    NUMPY = "npz"
    GEOTIFF = "tif"
    JSON = "json"
    ASCII_GRID = "asc"
    XYZ = "xyz"
    NETCDF = "nc"


@dataclass
class TerrainProvenance:
    """
    Provenance information for terrain data exports.

    Attributes:
        source: Original data source (e.g., "SRTM", "Open-Meteo")
        source_resolution_m: Native resolution of source data
        processing_steps: List of processing operations applied
        software_version: Version of export software
        export_timestamp: When export was created
        checksum: MD5 checksum of primary data
        bounds: Geographic bounds (north, south, east, west)
        crs: Coordinate reference system
        vertical_datum: Vertical reference datum
    """
    source: str = "SRTM"
    source_resolution_m: float = 30.0
    processing_steps: list[str] = field(default_factory=list)
    software_version: str = "1.0.0"
    export_timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    checksum: str | None = None
    bounds: tuple[float, float, float, float] | None = None
    crs: str = "EPSG:4326"
    vertical_datum: str = "EGM96"

    def to_dict(self) -> dict[str, Any]:
        """Export provenance as dictionary."""
        return {
            "source": self.source,
            "source_resolution_m": self.source_resolution_m,
            "processing_steps": self.processing_steps,
            "software_version": self.software_version,
            "export_timestamp": self.export_timestamp.isoformat(),
            "checksum": self.checksum,
            "bounds": list(self.bounds) if self.bounds else None,
            "crs": self.crs,
            "vertical_datum": self.vertical_datum,
        }


@dataclass
class ExportConfig:
    """
    Configuration for terrain exports.

    Attributes:
        output_dir: Base output directory
        include_derived: Whether to export derived products (slope, aspect)
        compress: Whether to use compression
        compression_level: Compression level (1-9)
        float_precision: Floating point precision
        overwrite: Whether to overwrite existing files
        create_subdirs: Whether to create subdirectories
    """
    output_dir: Path = field(default_factory=lambda: Path("outputs/perception/terrain"))
    include_derived: bool = True
    compress: bool = True
    compression_level: int = 4
    float_precision: str = "float32"
    overwrite: bool = False
    create_subdirs: bool = True

    def __post_init__(self):
        """Ensure output directory exists."""
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if self.create_subdirs:
            for subdir in ["grids", "derived", "profiles", "viewsheds", "metadata"]:
                (self.output_dir / subdir).mkdir(exist_ok=True)


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
        checksum: MD5 checksum
        error: Error message if failed
    """
    success: bool
    file_path: Path | list[Path]
    file_size_bytes: int = 0
    format: TerrainExportFormat = TerrainExportFormat.NUMPY
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
            "format": self.format.value,
            "timestamp": self.timestamp.isoformat(),
            "checksum": self.checksum,
            "error": self.error,
        }


class TerrainOutputManager:
    """
    Multi-format terrain data persistence manager.

    This class provides comprehensive export capabilities for terrain data
    with full provenance tracking and multiple format support.

    Example:
        >>> config = ExportConfig(output_dir=Path("outputs/perception/terrain"))
        >>> manager = TerrainOutputManager(config=config)
        >>>
        >>> # Export elevation grid
        >>> result = manager.export_numpy(
        ...     elevation=model.elevation,
        ...     latitudes=model.latitudes,
        ...     longitudes=model.longitudes,
        ...     metadata=model.metadata.to_dict(),
        ...     filename="delhi_elevation"
        ... )
        >>> print(f"Exported: {result.file_path}")
        >>>
        >>> # Export GeoTIFF
        >>> result = manager.export_geotiff(
        ...     elevation=model.elevation,
        ...     bounds=model.coverage_bounds,
        ...     filename="delhi_dem"
        ... )
    """

    def __init__(self, config: ExportConfig | None = None):
        """
        Initialize output manager.

        Args:
            config: Export configuration. If None, uses defaults.
        """
        self.config = config or ExportConfig()

        # Subdirectory paths
        self._grids_dir = self.config.output_dir / "grids"
        self._derived_dir = self.config.output_dir / "derived"
        self._profiles_dir = self.config.output_dir / "profiles"
        self._viewsheds_dir = self.config.output_dir / "viewsheds"
        self._metadata_dir = self.config.output_dir / "metadata"

        logger.info(f"TerrainOutputManager initialized, output_dir: {self.config.output_dir}")

    def _generate_filename(
        self,
        base_name: str,
        format: TerrainExportFormat,
        include_timestamp: bool = True,
    ) -> str:
        """Generate filename with optional timestamp."""
        if include_timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            return f"{base_name}_{timestamp}.{format.value}"
        return f"{base_name}.{format.value}"

    def _compute_checksum(self, data: np.ndarray) -> str:
        """Compute MD5 checksum of array data."""
        return hashlib.md5(data.tobytes()).hexdigest()

    def _get_file_size(self, filepath: Path) -> int:
        """Get file size in bytes."""
        return filepath.stat().st_size if filepath.exists() else 0

    def _convert_for_json(self, obj: Any) -> Any:
        """Convert numpy types to Python native types for JSON serialization."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: self._convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_for_json(v) for v in obj]
        elif isinstance(obj, datetime):
            return obj.isoformat()
        return obj

    def export_numpy(
        self,
        elevation: np.ndarray,
        latitudes: np.ndarray,
        longitudes: np.ndarray,
        metadata: dict[str, Any] | None = None,
        filename: str = "terrain_elevation",
        derived: dict[str, np.ndarray] | None = None,
    ) -> ExportResult:
        """
        Export terrain data to NumPy compressed format.

        Args:
            elevation: 2D elevation array
            latitudes: 1D latitude array
            longitudes: 1D longitude array
            metadata: Optional metadata dictionary
            filename: Output filename (without extension)
            derived: Optional derived products (slope, aspect, etc.)

        Returns:
            ExportResult with file path and status
        """
        filepath = self._grids_dir / self._generate_filename(
            filename, TerrainExportFormat.NUMPY
        )

        if filepath.exists() and not self.config.overwrite:
            return ExportResult(
                success=False,
                file_path=filepath,
                format=TerrainExportFormat.NUMPY,
                error="File exists and overwrite=False",
            )

        try:
            # Prepare save dict
            save_dict = {
                "elevation": elevation.astype(self.config.float_precision),
                "latitudes": latitudes.astype(np.float64),
                "longitudes": longitudes.astype(np.float64),
            }

            # Add derived products
            if derived:
                for name, data in derived.items():
                    save_dict[name] = data.astype(self.config.float_precision)

            # Save with compression
            if self.config.compress:
                np.savez_compressed(filepath, **save_dict)
            else:
                np.savez(filepath, **save_dict)

            # Compute checksum
            checksum = self._compute_checksum(elevation)

            # Save metadata sidecar
            if metadata:
                meta_filepath = self._metadata_dir / f"{filename}_metadata.json"
                metadata["checksum"] = checksum
                metadata["file"] = str(filepath.name)

                with open(meta_filepath, "w") as f:
                    json.dump(self._convert_for_json(metadata), f, indent=2)

            logger.info(f"Exported NumPy: {filepath}")

            return ExportResult(
                success=True,
                file_path=filepath,
                file_size_bytes=self._get_file_size(filepath),
                format=TerrainExportFormat.NUMPY,
                checksum=checksum,
            )

        except Exception as e:
            logger.error(f"NumPy export failed: {e}")
            return ExportResult(
                success=False,
                file_path=filepath,
                format=TerrainExportFormat.NUMPY,
                error=str(e),
            )

    def export_geotiff(
        self,
        elevation: np.ndarray,
        bounds: tuple[float, float, float, float],
        filename: str = "terrain_dem",
        crs: str = "EPSG:4326",
    ) -> ExportResult:
        """
        Export terrain to GeoTIFF format.

        Requires rasterio library.

        Args:
            elevation: 2D elevation array
            bounds: (north, south, east, west) in decimal degrees
            filename: Output filename (without extension)
            crs: Coordinate reference system

        Returns:
            ExportResult with file path and status
        """
        filepath = self._grids_dir / self._generate_filename(
            filename, TerrainExportFormat.GEOTIFF
        )

        if filepath.exists() and not self.config.overwrite:
            return ExportResult(
                success=False,
                file_path=filepath,
                format=TerrainExportFormat.GEOTIFF,
                error="File exists and overwrite=False",
            )

        try:
            import rasterio
            from rasterio.transform import from_bounds

            north, south, east, west = bounds
            height, width = elevation.shape

            # Compute transform
            transform = from_bounds(west, south, east, north, width, height)

            # Write GeoTIFF
            with rasterio.open(
                filepath,
                "w",
                driver="GTiff",
                height=height,
                width=width,
                count=1,
                dtype=elevation.dtype,
                crs=crs,
                transform=transform,
                compress="lzw" if self.config.compress else None,
            ) as dst:
                dst.write(elevation, 1)
                dst.update_tags(
                    TIFFTAG_IMAGEDESCRIPTION="Terrain Elevation Model",
                    TIFFTAG_SOFTWARE="eVTOL Trajectory Optimization",
                )

            logger.info(f"Exported GeoTIFF: {filepath}")

            return ExportResult(
                success=True,
                file_path=filepath,
                file_size_bytes=self._get_file_size(filepath),
                format=TerrainExportFormat.GEOTIFF,
                checksum=self._compute_checksum(elevation),
            )

        except ImportError:
            logger.warning("rasterio not installed, skipping GeoTIFF export")
            return ExportResult(
                success=False,
                file_path=filepath,
                format=TerrainExportFormat.GEOTIFF,
                error="rasterio not installed",
            )
        except Exception as e:
            logger.error(f"GeoTIFF export failed: {e}")
            return ExportResult(
                success=False,
                file_path=filepath,
                format=TerrainExportFormat.GEOTIFF,
                error=str(e),
            )

    def export_ascii_grid(
        self,
        elevation: np.ndarray,
        bounds: tuple[float, float, float, float],
        filename: str = "terrain_dem",
        nodata_value: float = -9999.0,
    ) -> ExportResult:
        """
        Export terrain to ESRI ASCII Grid format.

        ASCII Grid is widely compatible with GIS software.

        Args:
            elevation: 2D elevation array
            bounds: (north, south, east, west) in decimal degrees
            filename: Output filename (without extension)
            nodata_value: Value for no-data cells

        Returns:
            ExportResult with file path and status
        """
        filepath = self._grids_dir / self._generate_filename(
            filename, TerrainExportFormat.ASCII_GRID
        )

        if filepath.exists() and not self.config.overwrite:
            return ExportResult(
                success=False,
                file_path=filepath,
                format=TerrainExportFormat.ASCII_GRID,
                error="File exists and overwrite=False",
            )

        try:
            north, south, east, west = bounds
            nrows, ncols = elevation.shape
            cellsize_lat = (north - south) / nrows
            cellsize_lon = (east - west) / ncols
            cellsize = (cellsize_lat + cellsize_lon) / 2  # Average for square cells

            with open(filepath, "w") as f:
                # Write header
                f.write(f"ncols {ncols}\n")
                f.write(f"nrows {nrows}\n")
                f.write(f"xllcorner {west}\n")
                f.write(f"yllcorner {south}\n")
                f.write(f"cellsize {cellsize:.10f}\n")
                f.write(f"NODATA_value {nodata_value}\n")

                # Write data (row by row, north to south)
                for row in elevation:
                    row_str = " ".join(f"{v:.2f}" if not np.isnan(v) else str(nodata_value) for v in row)
                    f.write(row_str + "\n")

            logger.info(f"Exported ASCII Grid: {filepath}")

            return ExportResult(
                success=True,
                file_path=filepath,
                file_size_bytes=self._get_file_size(filepath),
                format=TerrainExportFormat.ASCII_GRID,
            )

        except Exception as e:
            logger.error(f"ASCII Grid export failed: {e}")
            return ExportResult(
                success=False,
                file_path=filepath,
                format=TerrainExportFormat.ASCII_GRID,
                error=str(e),
            )

    def export_xyz(
        self,
        elevation: np.ndarray,
        latitudes: np.ndarray,
        longitudes: np.ndarray,
        filename: str = "terrain_points",
        decimals: int = 2,
    ) -> ExportResult:
        """
        Export terrain to XYZ point cloud format.

        Args:
            elevation: 2D elevation array
            latitudes: 1D latitude array
            longitudes: 1D longitude array
            filename: Output filename (without extension)
            decimals: Decimal places for elevation

        Returns:
            ExportResult with file path and status
        """
        filepath = self._grids_dir / self._generate_filename(
            filename, TerrainExportFormat.XYZ
        )

        if filepath.exists() and not self.config.overwrite:
            return ExportResult(
                success=False,
                file_path=filepath,
                format=TerrainExportFormat.XYZ,
                error="File exists and overwrite=False",
            )

        try:
            with open(filepath, "w") as f:
                f.write("# X (Longitude), Y (Latitude), Z (Elevation m)\n")
                for i, lat in enumerate(latitudes):
                    for j, lon in enumerate(longitudes):
                        elev = elevation[i, j]
                        if not np.isnan(elev):
                            f.write(f"{lon:.6f} {lat:.6f} {elev:.{decimals}f}\n")

            logger.info(f"Exported XYZ: {filepath}")

            return ExportResult(
                success=True,
                file_path=filepath,
                file_size_bytes=self._get_file_size(filepath),
                format=TerrainExportFormat.XYZ,
            )

        except Exception as e:
            logger.error(f"XYZ export failed: {e}")
            return ExportResult(
                success=False,
                file_path=filepath,
                format=TerrainExportFormat.XYZ,
                error=str(e),
            )

    def export_json(
        self,
        data: dict[str, Any],
        filename: str = "terrain_metadata",
        subdir: str = "metadata",
    ) -> ExportResult:
        """
        Export metadata/statistics to JSON.

        Args:
            data: Dictionary to export
            filename: Output filename (without extension)
            subdir: Subdirectory (metadata, profiles, viewsheds)

        Returns:
            ExportResult with file path
        """
        subdirs = {
            "metadata": self._metadata_dir,
            "profiles": self._profiles_dir,
            "viewsheds": self._viewsheds_dir,
            "grids": self._grids_dir,
        }
        output_dir = subdirs.get(subdir, self._metadata_dir)

        filepath = output_dir / self._generate_filename(
            filename, TerrainExportFormat.JSON
        )

        if filepath.exists() and not self.config.overwrite:
            return ExportResult(
                success=False,
                file_path=filepath,
                format=TerrainExportFormat.JSON,
                error="File exists and overwrite=False",
            )

        try:
            with open(filepath, "w") as f:
                json.dump(self._convert_for_json(data), f, indent=2)

            logger.info(f"Exported JSON: {filepath}")

            return ExportResult(
                success=True,
                file_path=filepath,
                file_size_bytes=self._get_file_size(filepath),
                format=TerrainExportFormat.JSON,
            )

        except Exception as e:
            logger.error(f"JSON export failed: {e}")
            return ExportResult(
                success=False,
                file_path=filepath,
                format=TerrainExportFormat.JSON,
                error=str(e),
            )

    def export_derived_products(
        self,
        slope: np.ndarray | None = None,
        aspect: np.ndarray | None = None,
        hillshade: np.ndarray | None = None,
        ruggedness: np.ndarray | None = None,
        bounds: tuple[float, float, float, float] = None,
        filename_prefix: str = "terrain",
    ) -> list[ExportResult]:
        """
        Export derived terrain products as GeoTIFFs.

        Args:
            slope: Slope array in degrees
            aspect: Aspect array in degrees
            hillshade: Hillshade array (0-255)
            ruggedness: Terrain Ruggedness Index
            bounds: Geographic bounds
            filename_prefix: Prefix for output files

        Returns:
            List of ExportResults
        """
        results = []

        products = [
            ("slope", slope),
            ("aspect", aspect),
            ("hillshade", hillshade),
            ("ruggedness", ruggedness),
        ]

        for name, data in products:
            if data is not None:
                filepath = self._derived_dir / f"{filename_prefix}_{name}.tif"

                try:
                    import rasterio
                    from rasterio.transform import from_bounds

                    north, south, east, west = bounds
                    height, width = data.shape
                    transform = from_bounds(west, south, east, north, width, height)

                    with rasterio.open(
                        filepath,
                        "w",
                        driver="GTiff",
                        height=height,
                        width=width,
                        count=1,
                        dtype=data.dtype,
                        crs="EPSG:4326",
                        transform=transform,
                        compress="lzw" if self.config.compress else None,
                    ) as dst:
                        dst.write(data, 1)

                    results.append(ExportResult(
                        success=True,
                        file_path=filepath,
                        file_size_bytes=self._get_file_size(filepath),
                        format=TerrainExportFormat.GEOTIFF,
                    ))

                except ImportError:
                    # Fallback to NumPy
                    numpy_path = self._derived_dir / f"{filename_prefix}_{name}.npz"
                    np.savez_compressed(numpy_path, data=data)
                    results.append(ExportResult(
                        success=True,
                        file_path=numpy_path,
                        file_size_bytes=self._get_file_size(numpy_path),
                        format=TerrainExportFormat.NUMPY,
                    ))

                except Exception as e:
                    results.append(ExportResult(
                        success=False,
                        file_path=filepath,
                        format=TerrainExportFormat.GEOTIFF,
                        error=str(e),
                    ))

        return results

    def export_profile(
        self,
        profile_data: dict[str, Any],
        filename: str = "terrain_profile",
    ) -> ExportResult:
        """
        Export terrain profile data.

        Args:
            profile_data: Profile dictionary with distances, elevations, etc.
            filename: Output filename

        Returns:
            ExportResult
        """
        return self.export_json(profile_data, filename, subdir="profiles")

    def export_viewshed(
        self,
        visible: np.ndarray,
        observer: tuple[float, float],
        observer_height_m: float,
        bounds: tuple[float, float, float, float],
        filename: str = "viewshed",
    ) -> ExportResult:
        """
        Export viewshed analysis result.

        Args:
            visible: Boolean visibility grid
            observer: Observer (lat, lon)
            observer_height_m: Observer height
            bounds: Grid bounds
            filename: Output filename

        Returns:
            ExportResult
        """
        # Save as NumPy
        filepath = self._viewsheds_dir / self._generate_filename(
            filename, TerrainExportFormat.NUMPY
        )

        try:
            np.savez_compressed(
                filepath,
                visible=visible.astype(np.uint8),
                observer_lat=observer[0],
                observer_lon=observer[1],
                observer_height_m=observer_height_m,
                bounds=np.array(bounds),
            )

            # Also save metadata JSON
            meta = {
                "observer": {"latitude": observer[0], "longitude": observer[1]},
                "observer_height_m": observer_height_m,
                "bounds": list(bounds),
                "visible_cells": int(np.sum(visible)),
                "total_cells": int(visible.size),
                "visible_fraction": float(np.mean(visible)),
            }
            self.export_json(meta, f"{filename}_meta", subdir="viewsheds")

            return ExportResult(
                success=True,
                file_path=filepath,
                file_size_bytes=self._get_file_size(filepath),
                format=TerrainExportFormat.NUMPY,
            )

        except Exception as e:
            return ExportResult(
                success=False,
                file_path=filepath,
                format=TerrainExportFormat.NUMPY,
                error=str(e),
            )

    def create_export_report(
        self,
        results: list[ExportResult],
        filename: str = "export_report",
    ) -> ExportResult:
        """
        Create a summary report of all exports.

        Args:
            results: List of export results
            filename: Report filename

        Returns:
            ExportResult for the report file
        """
        report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_exports": len(results),
            "successful": sum(1 for r in results if r.success),
            "failed": sum(1 for r in results if not r.success),
            "total_size_bytes": sum(r.file_size_bytes for r in results if r.success),
            "exports": [r.to_dict() for r in results],
        }

        return self.export_json(report, filename, subdir="metadata")
