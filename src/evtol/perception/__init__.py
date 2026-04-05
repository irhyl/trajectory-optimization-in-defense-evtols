"""
Perception and Environment Layer for eVTOL Trajectory Optimization

This package provides spatiotemporal maps for eVTOL planning including:
- Terrain elevation and geomorphometry (terrain package - real API data)
- Atmospheric conditions (wind package - real API data)
- Threat and risk assessment (threat_model)
- Obstacle detection (obstacle_model)
- Data fusion and uncertainty quantification (fusion_model)

The layer produces accurate, versioned, georeferenced, and fast-to-query
maps for the planner via the FusedIntelligenceModel interface.

Real Data Sources (No Synthetic Data):
    - Terrain: Open-Meteo Elevation API (SRTM-based, 30m resolution, global)
    - Wind: Open-Meteo Weather API (GFS/HRES, hourly, 5-day forecast)

Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │                    PERCEPTION LAYER                         │
    ├─────────────┬─────────────┬───────────────┬─────────────────┤
    │   terrain/  │   wind/     │ threat_model  │ obstacle_model  │
    │             │             │               │                 │
    │ Real SRTM   │ Real Meteo  │ Intelligence  │ LIDAR/Sensor    │
    │ Elevation   │ Forecast    │ Assessment    │ Integration     │
    ├─────────────┴─────────────┴───────────────┴─────────────────┤
    │                     fusion_model                            │
    │              Unified Risk & Cost Mapping                    │
    └─────────────────────────────────────────────────────────────┘
"""

# =============================================================================
# TERRAIN PACKAGE (Real Elevation Data from Open-Meteo)
# =============================================================================
from .terrain import (
    # Data provider
    TerrainDataProvider,
    ElevationPoint,
    ElevationGrid,
    ElevationSource,
    VerticalDatum,
    TerrainProviderConfig,
    get_elevation,
    get_elevation_grid,
    get_default_provider,
    # Field model
    TerrainFieldModel,
    TerrainMetadata,
    TerrainProfile,
    TerrainCategory,
    LandCoverType,
    LineOfSightResult,
    ViewshedResult,
    # Interpolator
    TerrainInterpolator,
    InterpolatedElevation,
    TrajectoryTerrainInfo,
    InterpolationMethod as TerrainInterpolationMethod,
    create_interpolator as create_terrain_interpolator,
    # Output manager
    TerrainOutputManager,
    TerrainExportFormat,
    TerrainProvenance,
    ExportConfig as TerrainExportConfig,
    ExportResult as TerrainExportResult,
    # Visualization
    TerrainContourPlotter,
    TerrainProfilePlotter,
    TerrainViewshedPlotter,
    Terrain3DPlotter,
    TerrainColorScheme,
    PlotStyle as TerrainPlotStyle,
    plot_terrain_overview,
    save_terrain_visualization_suite,
)

# =============================================================================
# WIND PACKAGE (Real Meteorological Data from Open-Meteo)
# =============================================================================
from .wind import (
    # Core models
    WindFieldModel,
    WindDataProvider,
    WindInterpolator,
    WindOutputManager,
    # Data structures
    WindMetadata,
    WindComponents,
    BoundaryLayerProfile,
    WindObservation,
    WindForecast,
    # Configuration
    WindExportFormat,
    ExportConfig as WindExportConfig,
    InterpolationMethod as WindInterpolationMethod,
    # Visualization
    WindField2DPlotter,
    WindProfilePlotter,
    WindRosePlotter,
    EnergyImpactPlotter,
    WindColorScheme,
    plot_wind_field,
    save_wind_visualization_suite,
)

# =============================================================================
# OBSTACLE PACKAGE (Multi-source obstacle detection)
# =============================================================================
from .obstacle import (
    # Enumerations
    ObstacleCategory,
    ObstacleMobility,
    ObstacleSource,
    AlertLevel,
    TrackStatus,
    GeometryType,
    # Base types
    Obstacle,
    ObstacleState,
    # Static obstacles
    Building,
    Tower,
    PowerLine,
    PowerPylon,
    Chimney,
    ReligiousStructure,
    WindTurbine,
    Bridge,
    # Dynamic obstacles
    Aircraft,
    AircraftState,
    Helicopter,
    Drone,
    Bird,
    BirdFlock,
    # Tethered
    TetheredObstacle,
    Aerostat,
    CableCar,
    ConstructionCrane,
    # Temporary
    TemporaryObstacle,
    NOTAMRestriction,
    # Environmental
    EnvironmentalHazard,
    WeatherHazard,
    VisibilityHazard,
    # Geometry
    BoundingSphere,
    BoundingCylinder,
    BoundingBox,
    OrientedBoundingBox,
    ConvexHull2D,
    CompositeGeometry,
    LineSegment3D,
    Polyline3D,
    # Conflict
    ConflictAlert,
    ConflictZone,
    SeparationMinima,
    DEFAULT_SEPARATION_MINIMA,
    # Tracking
    Track,
    TrackHistory,
    KalmanState,
    # Configuration
    ObstacleConfig,
    ProviderConfig,
    TrackerConfig,
    ConflictConfig,
)

# =============================================================================
# THREAT, OBSTACLE, AND FUSION API COMPATIBILITY ALIASES
# =============================================================================
# Keep legacy names mapped to current package implementations.
try:
    from .threat import ThreatField as ThreatAssessmentModel
except Exception:
    ThreatAssessmentModel = None  # type: ignore

try:
    from .obstacle import ObstacleDataProvider as ObstacleDetectionModel
except Exception:
    ObstacleDetectionModel = None  # type: ignore

# Legacy placeholder retained for compatibility.
LandingZone = None  # type: ignore

try:
    from .fusion import EnvironmentModel as FusedIntelligenceModel
    from .fusion import DatasetMetadata as FusionMetadata
except Exception:
    FusedIntelligenceModel = None  # type: ignore
    FusionMetadata = None  # type: ignore

# =============================================================================
# VISUALIZATION MODULES
# =============================================================================
# =============================================================================
# PHASE 2F: SENSOR FUSION & THREAT DETECTION (Real-time integration)
# =============================================================================
from .sensor_fusion import (
    # Enumerations
    ThreatType,
    ThreatLevel,
    # Measurement types
    RadarMeasurement,
    RFMeasurement,
    # Track representation
    TrackedThreat,
    # Fusion components
    SensorFuser,
    PathThreatAnalyzer,
    ReplanTriggerLogic,
)

from .fusion_orchestrator import (
    # Orchestrator
    SensorFusionOrchestrator,
    # Output
    ThreatMap,
)

try:
    from .threat_visualization import (
        plot_threat_heatmap_matplotlib,
        plot_sam_coverage_matplotlib,
        plot_safe_corridors_matplotlib,
        plot_threat_statistics_matplotlib,
        plot_threat_3d_altitude_analysis_plotly
    )
except ImportError:
    plot_threat_heatmap_matplotlib = None  # type: ignore
    plot_sam_coverage_matplotlib = None  # type: ignore
    plot_safe_corridors_matplotlib = None  # type: ignore
    plot_threat_statistics_matplotlib = None  # type: ignore
    plot_threat_3d_altitude_analysis_plotly = None  # type: ignore

try:
    from .obstacle_visualization import (
        plot_building_detection_matplotlib,
        plot_landing_zones_matplotlib,
        plot_clearance_profile_matplotlib,
        plot_obstacle_statistics_matplotlib
    )
except ImportError:
    plot_building_detection_matplotlib = None  # type: ignore
    plot_landing_zones_matplotlib = None  # type: ignore
    plot_clearance_profile_matplotlib = None  # type: ignore
    plot_obstacle_statistics_matplotlib = None  # type: ignore

try:
    from .fusion_visualization import (
        plot_fused_risk_map_matplotlib,
        plot_feasibility_and_risk_matplotlib,
        plot_energy_cost_map_matplotlib as plot_fusion_energy_cost_map_matplotlib,
        plot_component_contribution_matplotlib,
        plot_fused_statistics_matplotlib,
        plot_fused_3d_heatmap_plotly,
        plot_comparison_before_after_fusion
    )
except ImportError:
    plot_fused_risk_map_matplotlib = None  # type: ignore
    plot_feasibility_and_risk_matplotlib = None  # type: ignore
    plot_fusion_energy_cost_map_matplotlib = None  # type: ignore
    plot_component_contribution_matplotlib = None  # type: ignore
    plot_fused_statistics_matplotlib = None  # type: ignore
    plot_fused_3d_heatmap_plotly = None  # type: ignore
    plot_comparison_before_after_fusion = None  # type: ignore

# =============================================================================
# PUBLIC API
# =============================================================================
__all__ = [
    # -------------------------------------------------------------------------
    # Terrain package (real data)
    # -------------------------------------------------------------------------
    # Data provider
    "TerrainDataProvider",
    "ElevationPoint",
    "ElevationGrid",
    "ElevationSource",
    "VerticalDatum",
    "TerrainProviderConfig",
    "get_elevation",
    "get_elevation_grid",
    "get_default_provider",
    # Field model
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
    "TerrainInterpolationMethod",
    "create_terrain_interpolator",
    # Output manager
    "TerrainOutputManager",
    "TerrainExportFormat",
    "TerrainProvenance",
    "TerrainExportConfig",
    "TerrainExportResult",
    # Visualization
    "TerrainContourPlotter",
    "TerrainProfilePlotter",
    "TerrainViewshedPlotter",
    "Terrain3DPlotter",
    "TerrainColorScheme",
    "TerrainPlotStyle",
    "plot_terrain_overview",
    "save_terrain_visualization_suite",

    # -------------------------------------------------------------------------
    # Wind package (real data)
    # -------------------------------------------------------------------------
    "WindFieldModel",
    "WindDataProvider",
    "WindInterpolator",
    "WindOutputManager",
    "WindMetadata",
    "WindComponents",
    "BoundaryLayerProfile",
    "WindObservation",
    "WindForecast",
    "WindExportFormat",
    "WindExportConfig",
    "WindInterpolationMethod",
    "WindField2DPlotter",
    "WindProfilePlotter",
    "WindRosePlotter",
    "EnergyImpactPlotter",
    "WindColorScheme",
    "plot_wind_field",
    "save_wind_visualization_suite",

    # -------------------------------------------------------------------------
    # Threat, Obstacle, Fusion models
    # -------------------------------------------------------------------------
    "ThreatAssessmentModel",
    "ObstacleDetectionModel",
    "FusedIntelligenceModel",
    "LandingZone",
    "FusionMetadata",

    # -------------------------------------------------------------------------
    # Phase 2F: Sensor Fusion & Threat Detection
    # -------------------------------------------------------------------------
    # Enumerations
    "ThreatType",
    "ThreatLevel",
    # Measurement types
    "RadarMeasurement",
    "RFMeasurement",
    # Track representation
    "TrackedThreat",
    # Fusion components
    "SensorFuser",
    "PathThreatAnalyzer",
    "ReplanTriggerLogic",
    # Orchestrator
    "SensorFusionOrchestrator",
    # Output
    "ThreatMap",

    # -------------------------------------------------------------------------
    # Visualization functions
    # -------------------------------------------------------------------------
    # Threat
    "plot_threat_heatmap_matplotlib",
    "plot_sam_coverage_matplotlib",
    "plot_safe_corridors_matplotlib",
    "plot_threat_statistics_matplotlib",
    "plot_threat_3d_altitude_analysis_plotly",
    # Obstacle
    "plot_building_detection_matplotlib",
    "plot_landing_zones_matplotlib",
    "plot_clearance_profile_matplotlib",
    "plot_obstacle_statistics_matplotlib",
    # Fusion
    "plot_fused_risk_map_matplotlib",
    "plot_feasibility_and_risk_matplotlib",
    "plot_fusion_energy_cost_map_matplotlib",
    "plot_component_contribution_matplotlib",
    "plot_fused_statistics_matplotlib",
    "plot_fused_3d_heatmap_plotly",
    "plot_comparison_before_after_fusion",
]

# Version
__version__ = "2.0.0"


def get_perception_info() -> dict:
    """Return information about the perception layer."""
    return {
        "version": __version__,
        "packages": {
            "terrain": {
                "description": "Real elevation data from Open-Meteo/SRTM",
                "data_source": "Open-Meteo Elevation API",
                "resolution_m": 30,
                "coverage": "Global (-60° to 60° latitude)",
            },
            "wind": {
                "description": "Real meteorological data from Open-Meteo",
                "data_source": "Open-Meteo Weather API (GFS/HRES)",
                "forecast_hours": 120,
                "coverage": "Global",
            },
        },
        "real_data_policy": "No synthetic data - all values from real APIs",
    }

