"""
Terrain Perception Package - Grid-based Terrain Analysis for eVTOL Operations.

This package provides comprehensive terrain analysis capabilities:
- Real elevation data from global APIs (Open-Meteo, SRTM)
- Derived geomorphometric products (slope, aspect, curvature)
- High-precision interpolation (bilinear, bicubic, spline)
- Line-of-sight and viewshed analysis
- Multi-format data export (NumPy, GeoTIFF, ASCII Grid)
- Publication-quality visualization

Architecture:
    ┌─────────────────────────────────────────────────────────────────┐
    │                      TerrainFieldModel                          │
    │  (Grid-based terrain with elevation and derived products)       │
    └─────────────────────────────────────────────────────────────────┘
                              ▲
                              │ initializes from
                              │
    ┌─────────────────────────────────────────────────────────────────┐
    │                    TerrainDataProvider                          │
    │  (Real elevation data from Open-Meteo/SRTM APIs)               │
    └─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │                   TerrainInterpolator                           │
    │  (Bilinear/Bicubic/Spline interpolation for smooth queries)    │
    └─────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
    ┌─────────────────────┐       ┌─────────────────────┐
    │ TerrainOutputManager│       │ TerrainVisualization│
    │ (NumPy/GeoTIFF/JSON)│       │ (Contours/3D/Profiles)│
    └─────────────────────┘       └─────────────────────┘

Example:
    >>> from evtol.perception.terrain import (
    ...     TerrainFieldModel,
    ...     TerrainInterpolator,
    ...     get_elevation,
    ... )
    >>>
    >>> # Quick elevation lookup
    >>> elev = get_elevation(28.6, 77.2)
    >>> print(f"Delhi elevation: {elev.elevation_m:.0f}m")
    >>>
    >>> # Full terrain model
    >>> model = TerrainFieldModel(
    ...     coverage_bounds=(28.7, 28.5, 77.3, 77.1),
    ...     resolution_m=100
    ... )
    >>> model.initialize()
    >>>
    >>> # Compute derived products
    >>> slope = model.compute_slope()
    >>> hillshade = model.compute_hillshade()
    >>>
    >>> # Check line of sight
    >>> los = model.check_line_of_sight(
    ...     observer=(28.6, 77.2, 100),
    ...     target=(28.65, 77.25, 50)
    ... )
    >>> print(f"Visible: {los.is_visible}")

Data Sources:
    - Open-Meteo Elevation API (SRTM-based, 30m, global)
    - NASA SRTM (30m/90m, 60°N-56°S)
    - Accuracy: ±16m absolute, ±6m relative (SRTM spec)

References:
    - Rodriguez et al. (2006), "An Assessment of the SRTM
      Topographic Products", Photogrammetric Eng. & Remote Sensing
    - Horn, B.K.P. (1981), "Hill Shading and the Reflectance Map",
      Proceedings of the IEEE

Author: Research-grade implementation for eVTOL trajectory optimization
Version: 1.0.0
"""

from __future__ import annotations

__version__ = "1.0.0"
__author__ = "eVTOL Research Team"

# Data Provider - Real elevation API integration
from .data_provider import (
    TerrainDataProvider,
    TerrainProviderConfig,
    ElevationPoint,
    ElevationGrid,
    ElevationSource,
    VerticalDatum,
    get_elevation,
    get_elevation_grid,
    get_default_provider,
)

# Field Model - Grid-based terrain with derived products
from .field_model import (
    TerrainFieldModel,
    TerrainMetadata,
    TerrainProfile,
    TerrainCategory,
    LandCoverType,
    LineOfSightResult,
    ViewshedResult,
)

# Interpolator - High-precision elevation interpolation
from .interpolator import (
    TerrainInterpolator,
    InterpolatedElevation,
    TrajectoryTerrainInfo,
    InterpolationMethod,
    create_interpolator,
)

# Output Manager - Multi-format persistence
from .output_manager import (
    TerrainOutputManager,
    TerrainExportFormat,
    ExportConfig,
    ExportResult,
    TerrainProvenance,
)

# Visualization - Publication-quality plots
try:
    from .visualization import (
        TerrainContourPlotter,
        TerrainProfilePlotter,
        TerrainViewshedPlotter,
        Terrain3DPlotter,
        TerrainColorScheme,
        PlotStyle,
        plot_terrain_overview,
        save_terrain_visualization_suite,
    )
except Exception:  # pragma: no cover - optional plotting deps
    TerrainContourPlotter = None  # type: ignore
    TerrainProfilePlotter = None  # type: ignore
    TerrainViewshedPlotter = None  # type: ignore
    Terrain3DPlotter = None  # type: ignore
    TerrainColorScheme = None  # type: ignore
    PlotStyle = None  # type: ignore
    plot_terrain_overview = None  # type: ignore
    save_terrain_visualization_suite = None  # type: ignore

# Public API
__all__ = [
    # Version
    "__version__",

    # Data Provider
    "TerrainDataProvider",
    "TerrainProviderConfig",
    "ElevationPoint",
    "ElevationGrid",
    "ElevationSource",
    "VerticalDatum",
    "get_elevation",
    "get_elevation_grid",
    "get_default_provider",

    # Field Model
    "TerrainFieldModel",
    "TerrainMetadata",
    "TerrainProfile",
    "TerrainCategory",
    "LandCoverType",
    "LineOfSightResult",
    "ViewshedResult",

    # Interpolator
    "TerrainInterpolator",
    "InterpolatedElevation",
    "TrajectoryTerrainInfo",
    "InterpolationMethod",
    "create_interpolator",

    # Output Manager
    "TerrainOutputManager",
    "TerrainExportFormat",
    "ExportConfig",
    "ExportResult",
    "TerrainProvenance",

    # Visualization
    "TerrainContourPlotter",
    "TerrainProfilePlotter",
    "TerrainViewshedPlotter",
    "Terrain3DPlotter",
    "TerrainColorScheme",
    "PlotStyle",
    "plot_terrain_overview",
    "save_terrain_visualization_suite",
]
