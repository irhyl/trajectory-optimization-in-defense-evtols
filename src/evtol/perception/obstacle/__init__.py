"""
Obstacle Detection and Tracking Package for Defense eVTOL Operations.

This package provides comprehensive obstacle detection, tracking, and conflict
detection capabilities for eVTOL trajectory optimization in defense scenarios.

Data Sources:
    - OpenStreetMap (Overpass API): Static infrastructure (buildings, towers, power lines)
    - OpenSky Network: Real-time aircraft tracking (ADS-B)
    - NOTAM feeds: Temporary flight restrictions and obstacles

Key Capabilities:
    - Multi-source obstacle data fusion
    - Real-time Kalman filter tracking for dynamic obstacles
    - Trajectory prediction (CV, CTRV models)
    - Conflict detection (CPA, TTC computation)
    - Spatial indexing for efficient queries

India-Specific Considerations:
    - High-tension power lines (national grid)
    - Telecom towers (600k+ nationwide)
    - Religious structures (temples, minarets)
    - Seasonal hazards (kites, bird migration, dust storms)
    - Border aerostats and restricted zones
"""

# Enumerations
from .obstacle_types import (
    ObstacleCategory,
    ObstacleMobility,
    ObstacleSource,
    AlertLevel,
    TrackStatus,
    GeometryType,
)

# Core Data Types
from .obstacle_types import (
    # Base obstacle representation
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

    # Tethered/Suspended
    TetheredObstacle,
    Aerostat,
    CableCar,
    ConstructionCrane,

    # Temporary/NOTAM
    TemporaryObstacle,
    NOTAMRestriction,

    # Environmental
    EnvironmentalHazard,
    WeatherHazard,
    VisibilityHazard,
)

# Geometry Primitives
from .obstacle_types import (
    BoundingSphere,
    BoundingCylinder,
    BoundingBox,
    OrientedBoundingBox,
    ConvexHull2D,
    CompositeGeometry,
    LineSegment3D,
    Polyline3D,
)

# Conflict Detection
from .obstacle_types import (
    ConflictAlert,
    ConflictZone,
    SeparationMinima,
    DEFAULT_SEPARATION_MINIMA,
)

# Track Management
from .obstacle_types import (
    Track,
    TrackHistory,
    KalmanState,
)

# Configuration
from .obstacle_types import (
    ObstacleConfig,
    ProviderConfig,
    TrackerConfig,
    ConflictConfig,
)

# Data Providers
from .data_provider import (
    # OSM Provider
    OSMDataProvider,
    OSMProviderConfig,
    ObstacleCache,
    ObstacleDataProvider,
    get_osm_provider,
    fetch_static_obstacles,

    # OpenSky Provider (Live Aircraft)
    OpenSkyDataProvider,
    OpenSkyProviderConfig,
    get_opensky_provider,
    fetch_live_aircraft,
    fetch_india_airspace,
)

# Geometry Operations
from .geometry import (
    # Coordinate transformations
    ENUFrame,
    ecef_to_lla,

    # Distance calculations
    haversine_distance,
    distance_3d,
    slant_range,
    point_to_line_distance,
    point_to_polyline_distance,

    # Bounding volume operations
    BoundingVolumeOps,

    # CPA/TTC
    CPAResult,
    compute_cpa,
    TTCResult,
    compute_ttc_spheres,

    # Spatial queries
    SpatialQuery,
    create_corridor_query,

    # Utilities
    meters_per_degree_lat,
    meters_per_degree_lon,
    degrees_to_meters,
    meters_to_degrees,
    validate_coordinates,
    validate_india_bounds,
)

# Tracker (Kalman Filter)
from .tracker import (
    # Core filter
    KalmanFilter,
    MotionModel,
    ProcessNoise,

    # Track management
    TrackManager,
    TrackManagerConfig,
    ManagedTrack,

    # Convenience functions
    create_tracker,
    track_aircraft_stream,
)

# Conflict Detection & Prediction
from .conflict_detector import (
    # Configuration
    ConflictDetectorConfig,
    OwnshipState,

    # Trajectory Prediction
    PredictedState,
    TrajectoryPredictor,

    # Conflict Analysis
    ConflictAnalysis,
    ConflictDetector,

    # Convenience functions
    create_conflict_detector,
    check_immediate_conflicts,
)

__version__ = "1.0.0"

__all__ = [
    # Enumerations
    "ObstacleCategory",
    "ObstacleMobility",
    "ObstacleSource",
    "AlertLevel",
    "TrackStatus",
    "GeometryType",

    # Base Types
    "Obstacle",
    "ObstacleState",

    # Static Obstacles
    "Building",
    "Tower",
    "PowerLine",
    "PowerPylon",
    "Chimney",
    "ReligiousStructure",
    "WindTurbine",
    "Bridge",

    # Dynamic Obstacles
    "Aircraft",
    "AircraftState",
    "Helicopter",
    "Drone",
    "Bird",
    "BirdFlock",

    # Tethered
    "TetheredObstacle",
    "Aerostat",
    "CableCar",
    "ConstructionCrane",

    # Temporary
    "TemporaryObstacle",
    "NOTAMRestriction",

    # Environmental
    "EnvironmentalHazard",
    "WeatherHazard",
    "VisibilityHazard",

    # Geometry
    "BoundingSphere",
    "BoundingCylinder",
    "BoundingBox",
    "OrientedBoundingBox",
    "ConvexHull2D",
    "CompositeGeometry",
    "LineSegment3D",
    "Polyline3D",

    # Conflict Detection
    "ConflictAlert",
    "ConflictZone",
    "SeparationMinima",
    "DEFAULT_SEPARATION_MINIMA",

    # Tracking
    "Track",
    "TrackHistory",
    "KalmanState",

    # Configuration
    "ObstacleConfig",
    "ProviderConfig",
    "TrackerConfig",
    "ConflictConfig",

    # Data Providers
    "OSMDataProvider",
    "OSMProviderConfig",
    "ObstacleCache",
    "ObstacleDataProvider",
    "get_osm_provider",
    "fetch_static_obstacles",

    # OpenSky Live Aircraft
    "OpenSkyDataProvider",
    "OpenSkyProviderConfig",
    "get_opensky_provider",
    "fetch_live_aircraft",
    "fetch_india_airspace",

    # Geometry Operations
    "ENUFrame",
    "ecef_to_lla",
    "haversine_distance",
    "distance_3d",
    "slant_range",
    "point_to_line_distance",
    "point_to_polyline_distance",
    "BoundingVolumeOps",
    "CPAResult",
    "compute_cpa",
    "TTCResult",
    "compute_ttc_spheres",
    "SpatialQuery",
    "create_corridor_query",
    "meters_per_degree_lat",
    "meters_per_degree_lon",
    "degrees_to_meters",
    "meters_to_degrees",
    "validate_coordinates",
    "validate_india_bounds",

    # Tracker (Kalman Filter)
    "KalmanFilter",
    "MotionModel",
    "ProcessNoise",
    "TrackManager",
    "TrackManagerConfig",
    "ManagedTrack",
    "create_tracker",
    "track_aircraft_stream",

    # Conflict Detection & Prediction
    "ConflictDetectorConfig",
    "OwnshipState",
    "PredictedState",
    "TrajectoryPredictor",
    "ConflictAnalysis",
    "ConflictDetector",
    "create_conflict_detector",
    "check_immediate_conflicts",
]
