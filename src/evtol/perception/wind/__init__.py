"""
Wind Field Modeling Package for eVTOL Trajectory Optimization.

This package provides research-grade wind field modeling with:
- Real meteorological data integration (Open-Meteo, NOAA GFS)
- Multi-altitude wind field interpolation
- Energy impact assessment for trajectory planning
- Comprehensive output management and persistence

Modules:
    data_provider: Fetches real wind data from meteorological APIs
    field_model: Core physics-based wind field computations
    interpolator: Spatial-temporal interpolation algorithms
    output_manager: Persistence and export functionality

References:
    [1] Stull, R. B. (1988). An Introduction to Boundary Layer Meteorology.
    [2] Kaimal, J. C., & Finnigan, J. J. (1994). Atmospheric Boundary Layer Flows.
    [3] Open-Meteo API Documentation: https://open-meteo.com/en/docs

Author: eVTOL Trajectory Optimization Research Team
Version: 2.0.0 (Real Data Integration)
"""

from .data_provider import (
    WindDataProvider,
    WindDataSource,
    WindObservation,
    WindForecast,
)
from .field_model import (
    WindFieldModel,
    WindMetadata,
    WindComponents,
    BoundaryLayerProfile,
)
from .interpolator import (
    WindInterpolator,
    InterpolationMethod,
    TemporalInterpolator,
)
from .output_manager import (
    WindOutputManager,
    WindExportFormat,
    ExportConfig,
    ExportResult,
    WindProvenance,
)
try:
    from .visualization import (
        WindField2DPlotter,
        WindProfilePlotter,
        WindRosePlotter,
        EnergyImpactPlotter,
        WindField3DPlotter,
        WindColorScheme,
        PlotStyle,
        plot_wind_field,
        save_wind_visualization_suite,
    )
except Exception:  # pragma: no cover - optional plotting deps
    WindField2DPlotter = None  # type: ignore
    WindProfilePlotter = None  # type: ignore
    WindRosePlotter = None  # type: ignore
    EnergyImpactPlotter = None  # type: ignore
    WindField3DPlotter = None  # type: ignore
    WindColorScheme = None  # type: ignore
    PlotStyle = None  # type: ignore
    plot_wind_field = None  # type: ignore
    save_wind_visualization_suite = None  # type: ignore

__all__ = [
    # Data Provider
    "WindDataProvider",
    "WindDataSource",
    "WindObservation",
    "WindForecast",
    # Field Model
    "WindFieldModel",
    "WindMetadata",
    "WindComponents",
    "BoundaryLayerProfile",
    # Interpolator
    "WindInterpolator",
    "InterpolationMethod",
    "TemporalInterpolator",
    # Output Manager
    "WindOutputManager",
    "WindExportFormat",
    "ExportConfig",
    "ExportResult",
    "WindProvenance",
    # Visualization
    "WindField2DPlotter",
    "WindProfilePlotter",
    "WindRosePlotter",
    "EnergyImpactPlotter",
    "WindField3DPlotter",
    "WindColorScheme",
    "PlotStyle",
    "plot_wind_field",
    "save_wind_visualization_suite",
]

__version__ = "2.0.0"
__author__ = "eVTOL Trajectory Optimization Research Team"
