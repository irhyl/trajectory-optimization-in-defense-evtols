"""
Obstacle Type Definitions for Defense eVTOL Operations.

This module defines comprehensive dataclasses and enumerations for representing
all obstacle categories relevant to low-altitude eVTOL flight in India.

Categories:
    - Static Infrastructure: Buildings, towers, power lines
    - Dynamic Aerial: Aircraft, drones, birds
    - Tethered/Suspended: Aerostats, cable cars, cranes
    - Temporary: NOTAM restrictions, construction
    - Environmental: Weather, visibility hazards

Mathematical Framework:
    - State vectors: [px, py, pz, vx, vy, vz, heading, turn_rate]
    - Bounding volumes: Sphere, Cylinder, OBB, Convex Hull
    - Uncertainty: Covariance matrices for Kalman filtering

Author: Trajectory Optimization Team
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any

import numpy as np
from numpy.typing import NDArray


# =============================================================================
# ENUMERATIONS
# =============================================================================

class ObstacleCategory(Enum):
    """Primary obstacle categorization."""

    # Static Infrastructure
    BUILDING = auto()
    TOWER = auto()
    POWER_LINE = auto()
    POWER_PYLON = auto()
    CHIMNEY = auto()
    RELIGIOUS_STRUCTURE = auto()
    WIND_TURBINE = auto()
    BRIDGE = auto()
    ANTENNA = auto()

    # Dynamic Aerial
    AIRCRAFT = auto()
    HELICOPTER = auto()
    DRONE = auto()
    BIRD = auto()
    BIRD_FLOCK = auto()

    # Tethered/Suspended
    AEROSTAT = auto()
    CABLE_CAR = auto()
    CONSTRUCTION_CRANE = auto()
    ZIP_LINE = auto()
    TETHERED_BALLOON = auto()

    # Temporary
    NOTAM_RESTRICTION = auto()
    TEMPORARY_STRUCTURE = auto()
    EVENT_INFRASTRUCTURE = auto()

    # Environmental
    WEATHER_HAZARD = auto()
    VISIBILITY_HAZARD = auto()
    SMOKE_PLUME = auto()
    DUST_STORM = auto()

    # Unknown/Unclassified
    UNKNOWN = auto()


class ObstacleMobility(Enum):
    """Obstacle mobility classification."""

    STATIC = auto()          # Fixed position (buildings, towers)
    QUASI_STATIC = auto()    # Slow-moving or oscillating (cranes, tethered)
    DYNAMIC = auto()         # Moving obstacles (aircraft, birds)
    TRANSIENT = auto()       # Temporary/episodic (weather, events)


class ObstacleSource(Enum):
    """Data source for obstacle information."""

    OPENSTREETMAP = auto()   # OSM via Overpass API
    OPENSKY = auto()         # OpenSky Network ADS-B
    NOTAM = auto()           # NOTAM feeds
    EBIRD = auto()           # Bird observation data
    MANUAL = auto()          # Manually entered
    SENSOR = auto()          # On-board sensors
    PREDICTED = auto()       # Trajectory prediction
    FUSED = auto()           # Multi-source fusion


class AlertLevel(Enum):
    """Conflict alert severity levels."""

    NONE = 0           # No conflict
    ADVISORY = 1       # Awareness only (< 2km, < 5min)
    CAUTION = 2        # Monitor closely (< 500m, < 30s)
    WARNING = 3        # Prepare maneuver (< 200m, < 15s)
    CRITICAL = 4       # Immediate evasion (< 50m, < 5s)

    def __lt__(self, other: AlertLevel) -> bool:
        return self.value < other.value

    def __le__(self, other: AlertLevel) -> bool:
        return self.value <= other.value


class TrackStatus(Enum):
    """Track lifecycle status."""

    TENTATIVE = auto()    # New track, not yet confirmed
    CONFIRMED = auto()    # Confirmed track with stable updates
    COASTING = auto()     # No recent updates, predicted state
    LOST = auto()         # Track lost, pending deletion
    DELETED = auto()      # Track marked for removal


class GeometryType(Enum):
    """Bounding geometry types."""

    SPHERE = auto()
    CYLINDER = auto()
    BOX = auto()
    ORIENTED_BOX = auto()
    CONVEX_HULL = auto()
    COMPOSITE = auto()
    LINE_SEGMENT = auto()
    POLYLINE = auto()


# =============================================================================
# GEOMETRY PRIMITIVES
# =============================================================================

@dataclass
class BoundingSphere:
    """Spherical bounding volume for fast collision checks."""

    center_lat: float           # Latitude (degrees)
    center_lon: float           # Longitude (degrees)
    center_alt_m: float         # Altitude MSL (meters)
    radius_m: float             # Radius (meters)

    geometry_type: GeometryType = field(default=GeometryType.SPHERE, repr=False)

    def contains_point(self, lat: float, lon: float, alt_m: float) -> bool:
        """Check if point is inside the sphere (approximate)."""
        # Approximate distance using Haversine for horizontal + vertical
        from math import radians, sin, cos, sqrt, atan2

        R = 6371000  # Earth radius in meters

        lat1, lat2 = radians(self.center_lat), radians(lat)
        lon1, lon2 = radians(self.center_lon), radians(lon)

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        horizontal_dist = 2 * R * atan2(sqrt(a), sqrt(1-a))

        vertical_dist = abs(alt_m - self.center_alt_m)

        total_dist = sqrt(horizontal_dist**2 + vertical_dist**2)
        return total_dist <= self.radius_m

    def buffer(self, margin_m: float) -> BoundingSphere:
        """Return expanded sphere with safety margin."""
        return BoundingSphere(
            center_lat=self.center_lat,
            center_lon=self.center_lon,
            center_alt_m=self.center_alt_m,
            radius_m=self.radius_m + margin_m
        )


@dataclass
class BoundingCylinder:
    """Cylindrical bounding volume for towers and buildings."""

    center_lat: float           # Latitude (degrees)
    center_lon: float           # Longitude (degrees)
    base_alt_m: float           # Base altitude MSL (meters)
    radius_m: float             # Horizontal radius (meters)
    height_m: float             # Vertical extent (meters)

    geometry_type: GeometryType = field(default=GeometryType.CYLINDER, repr=False)

    @property
    def top_alt_m(self) -> float:
        """Top of cylinder altitude."""
        return self.base_alt_m + self.height_m

    def contains_point(self, lat: float, lon: float, alt_m: float) -> bool:
        """Check if point is inside the cylinder."""
        # Vertical check
        if alt_m < self.base_alt_m or alt_m > self.top_alt_m:
            return False

        # Horizontal check (approximate)
        from math import radians, sin, cos, sqrt, atan2

        R = 6371000
        lat1, lat2 = radians(self.center_lat), radians(lat)
        lon1, lon2 = radians(self.center_lon), radians(lon)

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        horizontal_dist = 2 * R * atan2(sqrt(a), sqrt(1-a))

        return horizontal_dist <= self.radius_m

    def buffer(self, h_margin_m: float, v_margin_m: float) -> BoundingCylinder:
        """Return expanded cylinder with safety margins."""
        return BoundingCylinder(
            center_lat=self.center_lat,
            center_lon=self.center_lon,
            base_alt_m=self.base_alt_m - v_margin_m,
            radius_m=self.radius_m + h_margin_m,
            height_m=self.height_m + 2 * v_margin_m
        )


@dataclass
class BoundingBox:
    """Axis-aligned bounding box (AABB)."""

    min_lat: float
    max_lat: float
    min_lon: float
    max_lon: float
    min_alt_m: float
    max_alt_m: float

    geometry_type: GeometryType = field(default=GeometryType.BOX, repr=False)

    @property
    def center(self) -> tuple[float, float, float]:
        """Center point of the box."""
        return (
            (self.min_lat + self.max_lat) / 2,
            (self.min_lon + self.max_lon) / 2,
            (self.min_alt_m + self.max_alt_m) / 2
        )

    def contains_point(self, lat: float, lon: float, alt_m: float) -> bool:
        """Check if point is inside the box."""
        return (
            self.min_lat <= lat <= self.max_lat and
            self.min_lon <= lon <= self.max_lon and
            self.min_alt_m <= alt_m <= self.max_alt_m
        )

    def intersects(self, other: BoundingBox) -> bool:
        """Check if two boxes intersect."""
        return (
            self.min_lat <= other.max_lat and self.max_lat >= other.min_lat and
            self.min_lon <= other.max_lon and self.max_lon >= other.min_lon and
            self.min_alt_m <= other.max_alt_m and self.max_alt_m >= other.min_alt_m
        )

    def buffer(self, margin_deg: float, v_margin_m: float) -> BoundingBox:
        """Return expanded box with safety margins."""
        return BoundingBox(
            min_lat=self.min_lat - margin_deg,
            max_lat=self.max_lat + margin_deg,
            min_lon=self.min_lon - margin_deg,
            max_lon=self.max_lon + margin_deg,
            min_alt_m=self.min_alt_m - v_margin_m,
            max_alt_m=self.max_alt_m + v_margin_m
        )


@dataclass
class OrientedBoundingBox:
    """Oriented bounding box for elongated structures like aircraft."""

    center_lat: float
    center_lon: float
    center_alt_m: float

    # Half-extents along local axes (meters)
    half_length: float      # Forward/backward
    half_width: float       # Left/right
    half_height: float      # Up/down

    # Orientation (radians)
    heading: float          # Yaw angle from north
    pitch: float = 0.0      # Pitch angle
    roll: float = 0.0       # Roll angle

    geometry_type: GeometryType = field(default=GeometryType.ORIENTED_BOX, repr=False)


@dataclass
class ConvexHull2D:
    """2D convex hull for building footprints (extruded vertically)."""

    vertices_lat: NDArray[np.float64]   # Latitude array
    vertices_lon: NDArray[np.float64]   # Longitude array
    base_alt_m: float
    height_m: float

    geometry_type: GeometryType = field(default=GeometryType.CONVEX_HULL, repr=False)

    @property
    def top_alt_m(self) -> float:
        return self.base_alt_m + self.height_m

    @property
    def centroid(self) -> tuple[float, float]:
        """Centroid of the footprint."""
        return float(np.mean(self.vertices_lat)), float(np.mean(self.vertices_lon))


@dataclass
class LineSegment3D:
    """3D line segment for power lines, cables."""

    start_lat: float
    start_lon: float
    start_alt_m: float

    end_lat: float
    end_lon: float
    end_alt_m: float

    radius_m: float = 5.0   # Buffer radius for collision checking

    geometry_type: GeometryType = field(default=GeometryType.LINE_SEGMENT, repr=False)

    @property
    def length_m(self) -> float:
        """Approximate length of segment."""
        from math import radians, sin, cos, sqrt, atan2

        R = 6371000
        lat1, lat2 = radians(self.start_lat), radians(self.end_lat)
        lon1, lon2 = radians(self.start_lon), radians(self.end_lon)

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        horizontal = 2 * R * atan2(sqrt(a), sqrt(1-a))
        vertical = abs(self.end_alt_m - self.start_alt_m)

        return sqrt(horizontal**2 + vertical**2)


@dataclass
class Polyline3D:
    """3D polyline for multi-segment cables, routes."""

    latitudes: NDArray[np.float64]
    longitudes: NDArray[np.float64]
    altitudes_m: NDArray[np.float64]

    radius_m: float = 5.0   # Buffer radius

    geometry_type: GeometryType = field(default=GeometryType.POLYLINE, repr=False)

    @property
    def num_segments(self) -> int:
        return len(self.latitudes) - 1

    def get_segment(self, index: int) -> LineSegment3D:
        """Get a specific segment as LineSegment3D."""
        return LineSegment3D(
            start_lat=float(self.latitudes[index]),
            start_lon=float(self.longitudes[index]),
            start_alt_m=float(self.altitudes_m[index]),
            end_lat=float(self.latitudes[index + 1]),
            end_lon=float(self.longitudes[index + 1]),
            end_alt_m=float(self.altitudes_m[index + 1]),
            radius_m=self.radius_m
        )


@dataclass
class CompositeGeometry:
    """Composite geometry combining multiple primitives."""

    components: list[
        BoundingSphere | BoundingCylinder | BoundingBox |
        OrientedBoundingBox | ConvexHull2D | LineSegment3D | Polyline3D
    ]

    geometry_type: GeometryType = field(default=GeometryType.COMPOSITE, repr=False)

    def get_bounding_box(self) -> BoundingBox:
        """Compute axis-aligned bounding box enclosing all components."""
        min_lat = min_lon = min_alt = float('inf')
        max_lat = max_lon = max_alt = float('-inf')

        for comp in self.components:
            if isinstance(comp, BoundingSphere):
                # Approximate conversion from meters to degrees
                deg_offset = comp.radius_m / 111000
                min_lat = min(min_lat, comp.center_lat - deg_offset)
                max_lat = max(max_lat, comp.center_lat + deg_offset)
                min_lon = min(min_lon, comp.center_lon - deg_offset)
                max_lon = max(max_lon, comp.center_lon + deg_offset)
                min_alt = min(min_alt, comp.center_alt_m - comp.radius_m)
                max_alt = max(max_alt, comp.center_alt_m + comp.radius_m)
            elif isinstance(comp, BoundingCylinder):
                deg_offset = comp.radius_m / 111000
                min_lat = min(min_lat, comp.center_lat - deg_offset)
                max_lat = max(max_lat, comp.center_lat + deg_offset)
                min_lon = min(min_lon, comp.center_lon - deg_offset)
                max_lon = max(max_lon, comp.center_lon + deg_offset)
                min_alt = min(min_alt, comp.base_alt_m)
                max_alt = max(max_alt, comp.top_alt_m)
            elif isinstance(comp, BoundingBox):
                min_lat = min(min_lat, comp.min_lat)
                max_lat = max(max_lat, comp.max_lat)
                min_lon = min(min_lon, comp.min_lon)
                max_lon = max(max_lon, comp.max_lon)
                min_alt = min(min_alt, comp.min_alt_m)
                max_alt = max(max_alt, comp.max_alt_m)
            # Add other geometry types as needed

        return BoundingBox(
            min_lat=min_lat, max_lat=max_lat,
            min_lon=min_lon, max_lon=max_lon,
            min_alt_m=min_alt, max_alt_m=max_alt
        )


# =============================================================================
# BASE OBSTACLE TYPES
# =============================================================================

@dataclass
class ObstacleState:
    """Kinematic state of an obstacle at a specific time."""

    timestamp: datetime

    # Position (WGS84)
    latitude: float             # Degrees
    longitude: float            # Degrees
    altitude_m: float           # MSL meters

    # Velocity (m/s in ENU frame)
    velocity_east: float = 0.0
    velocity_north: float = 0.0
    velocity_up: float = 0.0

    # Orientation
    heading_deg: float = 0.0    # True heading (degrees from north)
    pitch_deg: float = 0.0
    roll_deg: float = 0.0

    # Rates
    turn_rate_deg_s: float = 0.0
    climb_rate_m_s: float = 0.0

    # Uncertainty (1-sigma)
    position_uncertainty_m: float = 0.0
    velocity_uncertainty_m_s: float = 0.0

    @property
    def speed_m_s(self) -> float:
        """Ground speed magnitude."""
        return float(np.sqrt(
            self.velocity_east**2 +
            self.velocity_north**2
        ))

    @property
    def speed_3d_m_s(self) -> float:
        """3D speed magnitude."""
        return float(np.sqrt(
            self.velocity_east**2 +
            self.velocity_north**2 +
            self.velocity_up**2
        ))

    @property
    def velocity_vector(self) -> NDArray[np.float64]:
        """Velocity as numpy array [E, N, U]."""
        return np.array([
            self.velocity_east,
            self.velocity_north,
            self.velocity_up
        ])

    @property
    def position_lla(self) -> tuple[float, float, float]:
        """Position as (lat, lon, alt) tuple."""
        return (self.latitude, self.longitude, self.altitude_m)


@dataclass
class Obstacle:
    """Base obstacle representation."""

    # Identity
    obstacle_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    description: str = ""

    # Classification
    category: ObstacleCategory = ObstacleCategory.UNKNOWN
    mobility: ObstacleMobility = ObstacleMobility.STATIC
    source: ObstacleSource = ObstacleSource.MANUAL

    # State
    state: ObstacleState | None = None

    # Geometry
    geometry: BoundingSphere | BoundingCylinder | BoundingBox | OrientedBoundingBox | ConvexHull2D | CompositeGeometry | LineSegment3D | Polyline3D | None = None

    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: datetime | None = None

    # Source-specific data
    osm_id: int | None = None
    icao24: str | None = None          # Aircraft hex code
    callsign: str | None = None        # Aircraft callsign
    notam_id: str | None = None

    # Risk assessment
    collision_risk: float = 0.0           # 0-1 probability
    severity: float = 1.0                 # Impact severity multiplier

    # Flags
    is_cooperative: bool = True           # Has transponder/ADS-B
    is_confirmed: bool = True
    is_active: bool = True

    @property
    def is_expired(self) -> bool:
        """Check if obstacle has expired."""
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at

    @property
    def age_seconds(self) -> float:
        """Time since last update."""
        return (datetime.utcnow() - self.updated_at).total_seconds()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "obstacle_id": self.obstacle_id,
            "name": self.name,
            "category": self.category.name,
            "mobility": self.mobility.name,
            "source": self.source.name,
            "position": self.state.position_lla if self.state else None,
            "is_active": self.is_active,
            "collision_risk": self.collision_risk,
        }


# =============================================================================
# STATIC INFRASTRUCTURE OBSTACLES
# =============================================================================

@dataclass
class Building(Obstacle):
    """Building obstacle with footprint and height."""

    category: ObstacleCategory = field(default=ObstacleCategory.BUILDING)
    mobility: ObstacleMobility = field(default=ObstacleMobility.STATIC)

    # Building-specific
    height_m: float = 0.0
    levels: int = 0
    footprint_area_m2: float = 0.0
    building_type: str = ""              # residential, commercial, industrial

    # Rooftop hazards
    has_antenna: bool = False
    antenna_height_m: float = 0.0
    has_helipad: bool = False


@dataclass
class Tower(Obstacle):
    """Generic tower (telecom, observation, etc.)."""

    category: ObstacleCategory = field(default=ObstacleCategory.TOWER)
    mobility: ObstacleMobility = field(default=ObstacleMobility.STATIC)

    height_m: float = 0.0
    tower_type: str = ""                 # telecom, observation, water, radio
    has_guy_wires: bool = False
    guy_wire_radius_m: float = 0.0       # Horizontal extent of guy wires
    is_lit: bool = False                 # Has obstruction lighting


@dataclass
class PowerPylon(Obstacle):
    """Power transmission pylon/tower."""

    category: ObstacleCategory = field(default=ObstacleCategory.POWER_PYLON)
    mobility: ObstacleMobility = field(default=ObstacleMobility.STATIC)

    height_m: float = 0.0
    voltage_kv: float = 0.0
    pylon_type: str = ""                 # lattice, monopole, H-frame
    num_circuits: int = 1


@dataclass
class PowerLine(Obstacle):
    """Power transmission line (conductor between pylons)."""

    category: ObstacleCategory = field(default=ObstacleCategory.POWER_LINE)
    mobility: ObstacleMobility = field(default=ObstacleMobility.QUASI_STATIC)  # Sag varies

    voltage_kv: float = 0.0
    num_conductors: int = 1

    # Sag characteristics
    max_sag_m: float = 0.0               # Maximum sag at mid-span
    span_length_m: float = 0.0

    # Connected pylons
    start_pylon_id: str | None = None
    end_pylon_id: str | None = None


@dataclass
class Chimney(Obstacle):
    """Industrial chimney/smokestack."""

    category: ObstacleCategory = field(default=ObstacleCategory.CHIMNEY)
    mobility: ObstacleMobility = field(default=ObstacleMobility.STATIC)

    height_m: float = 0.0
    diameter_m: float = 0.0
    is_active: bool = True               # Emitting smoke/gases
    plume_height_m: float = 0.0          # Additional height for thermal plume


@dataclass
class ReligiousStructure(Obstacle):
    """Temple, mosque, church with tall elements."""

    category: ObstacleCategory = field(default=ObstacleCategory.RELIGIOUS_STRUCTURE)
    mobility: ObstacleMobility = field(default=ObstacleMobility.STATIC)

    structure_type: str = ""             # temple, mosque, church, gurudwara
    max_height_m: float = 0.0
    spire_height_m: float = 0.0          # Gopuram, minaret, steeple

    # Cultural sensitivity
    no_fly_buffer_m: float = 100.0       # Recommended avoidance radius


@dataclass
class WindTurbine(Obstacle):
    """Wind turbine with rotating blades."""

    category: ObstacleCategory = field(default=ObstacleCategory.WIND_TURBINE)
    mobility: ObstacleMobility = field(default=ObstacleMobility.QUASI_STATIC)  # Rotating

    hub_height_m: float = 0.0
    rotor_diameter_m: float = 0.0

    @property
    def tip_height_m(self) -> float:
        """Maximum height when blade is vertical."""
        return self.hub_height_m + self.rotor_diameter_m / 2

    # Operational state
    is_rotating: bool = True
    rotation_speed_rpm: float = 0.0


@dataclass
class Bridge(Obstacle):
    """Bridge structure with clearance info."""

    category: ObstacleCategory = field(default=ObstacleCategory.BRIDGE)
    mobility: ObstacleMobility = field(default=ObstacleMobility.STATIC)

    deck_height_m: float = 0.0           # Height of deck above ground/water
    clearance_below_m: float = 0.0       # Vertical clearance under bridge
    span_length_m: float = 0.0
    width_m: float = 0.0

    # Superstructure
    has_towers: bool = False
    tower_height_m: float = 0.0
    has_cables: bool = False             # Suspension/cable-stayed


# =============================================================================
# DYNAMIC AERIAL OBSTACLES
# =============================================================================

@dataclass
class AircraftState(ObstacleState):
    """Extended state for aircraft with flight-specific data."""

    # Flight data
    on_ground: bool = False
    squawk: str | None = None         # Transponder code

    # Performance
    indicated_airspeed_kt: float = 0.0
    true_airspeed_kt: float = 0.0
    mach: float = 0.0

    # Intent
    selected_altitude_m: float | None = None
    selected_heading_deg: float | None = None


@dataclass
class Aircraft(Obstacle):
    """Fixed-wing or rotary-wing manned aircraft."""

    category: ObstacleCategory = field(default=ObstacleCategory.AIRCRAFT)
    mobility: ObstacleMobility = field(default=ObstacleMobility.DYNAMIC)
    source: ObstacleSource = field(default=ObstacleSource.OPENSKY)

    # Aircraft identification
    icao24: str = ""                     # 24-bit ICAO address (hex)
    callsign: str = ""
    registration: str = ""
    aircraft_type: str = ""              # ICAO type designator

    # Category
    wake_category: str = ""              # L/M/H/J (light/medium/heavy/super)
    is_military: bool = False

    # Dimensions (for OBB)
    length_m: float = 20.0
    wingspan_m: float = 25.0
    height_m: float = 5.0

    # Origin/Destination
    origin_icao: str = ""
    destination_icao: str = ""


@dataclass
class Helicopter(Obstacle):
    """Rotorcraft with hover capability."""

    category: ObstacleCategory = field(default=ObstacleCategory.HELICOPTER)
    mobility: ObstacleMobility = field(default=ObstacleMobility.DYNAMIC)

    icao24: str = ""
    callsign: str = ""
    helicopter_type: str = ""

    # Rotor dimensions
    main_rotor_diameter_m: float = 15.0

    # Operational
    is_hovering: bool = False
    is_medical: bool = False             # HEMS flight


@dataclass
class Drone(Obstacle):
    """Unmanned aerial vehicle (UAV/UAS)."""

    category: ObstacleCategory = field(default=ObstacleCategory.DRONE)
    mobility: ObstacleMobility = field(default=ObstacleMobility.DYNAMIC)

    # Remote ID (if available)
    remote_id: str = ""
    operator_id: str = ""

    # Classification
    drone_class: str = ""                # C0-C4 (EU), Part 107 (US)
    weight_kg: float = 0.0

    # Dimensions
    max_dimension_m: float = 0.5

    # Capabilities
    max_altitude_m: float = 120.0        # Legal limit
    max_speed_m_s: float = 20.0
    max_endurance_min: float = 30.0

    # Cooperation status
    is_cooperative: bool = False         # Usually false for unknown drones


@dataclass
class Bird(Obstacle):
    """Individual large bird (raptor, vulture, etc.)."""

    category: ObstacleCategory = field(default=ObstacleCategory.BIRD)
    mobility: ObstacleMobility = field(default=ObstacleMobility.DYNAMIC)
    source: ObstacleSource = field(default=ObstacleSource.EBIRD)

    species: str = ""
    wingspan_m: float = 1.0
    mass_kg: float = 2.0

    # Behavior
    is_soaring: bool = False
    is_migrating: bool = False


@dataclass
class BirdFlock(Obstacle):
    """Flock of birds modeled as a volume."""

    category: ObstacleCategory = field(default=ObstacleCategory.BIRD_FLOCK)
    mobility: ObstacleMobility = field(default=ObstacleMobility.DYNAMIC)

    species: str = ""
    estimated_count: int = 0

    # Flock dimensions
    flock_radius_m: float = 50.0
    flock_height_m: float = 20.0

    # Movement
    migration_direction_deg: float = 0.0


# =============================================================================
# TETHERED / SUSPENDED OBSTACLES
# =============================================================================

@dataclass
class TetheredObstacle(Obstacle):
    """Base class for tethered obstacles."""

    mobility: ObstacleMobility = field(default=ObstacleMobility.QUASI_STATIC)

    # Tether properties
    tether_length_m: float = 0.0
    tether_anchor_lat: float = 0.0
    tether_anchor_lon: float = 0.0
    tether_anchor_alt_m: float = 0.0

    # Movement envelope
    max_horizontal_offset_m: float = 0.0  # Due to wind

    @property
    def max_altitude_m(self) -> float:
        """Maximum altitude the tethered object can reach."""
        return self.tether_anchor_alt_m + self.tether_length_m


@dataclass
class Aerostat(TetheredObstacle):
    """Tethered balloon/aerostat (e.g., border surveillance)."""

    category: ObstacleCategory = field(default=ObstacleCategory.AEROSTAT)

    # Aerostat dimensions
    envelope_length_m: float = 30.0
    envelope_diameter_m: float = 10.0

    # Operational
    operational_altitude_m: float = 3000.0
    is_military: bool = True
    has_radar: bool = False


@dataclass
class CableCar(TetheredObstacle):
    """Aerial tramway/cable car system."""

    category: ObstacleCategory = field(default=ObstacleCategory.CABLE_CAR)

    # Cable properties
    cable_height_min_m: float = 0.0
    cable_height_max_m: float = 0.0
    cable_length_m: float = 0.0

    # Stations
    station_coords: list[tuple[float, float, float]] = field(default_factory=list)

    # Operation
    is_operational: bool = True
    cabin_interval_m: float = 100.0


@dataclass
class ConstructionCrane(TetheredObstacle):
    """Construction crane (tower or mobile)."""

    category: ObstacleCategory = field(default=ObstacleCategory.CONSTRUCTION_CRANE)

    crane_type: str = ""                 # tower, mobile, crawler

    # Dimensions
    tower_height_m: float = 50.0
    jib_length_m: float = 60.0
    jib_height_m: float = 55.0           # Height at tip

    # Movement
    can_rotate: bool = True
    rotation_range_deg: float = 360.0


# =============================================================================
# TEMPORARY OBSTACLES
# =============================================================================

@dataclass
class TemporaryObstacle(Obstacle):
    """Generic temporary obstacle with validity period."""

    mobility: ObstacleMobility = field(default=ObstacleMobility.TRANSIENT)

    valid_from: datetime = field(default_factory=datetime.utcnow)
    valid_until: datetime | None = None

    @property
    def is_currently_valid(self) -> bool:
        """Check if obstacle is currently active."""
        now = datetime.utcnow()
        if now < self.valid_from:
            return False
        if self.valid_until and now > self.valid_until:
            return False
        return True


@dataclass
class NOTAMRestriction(TemporaryObstacle):
    """NOTAM-based flight restriction or obstacle."""

    category: ObstacleCategory = field(default=ObstacleCategory.NOTAM_RESTRICTION)
    source: ObstacleSource = field(default=ObstacleSource.NOTAM)

    # NOTAM identification
    notam_id: str = ""
    notam_type: str = ""                 # N (new), R (replace), C (cancel)

    # Affected area
    fir: str = ""                        # Flight Information Region

    # Restriction details
    restriction_type: str = ""           # obstacle, airspace, activity
    lower_limit_m: float = 0.0
    upper_limit_m: float = float('inf')

    # Schedule
    is_permanent: bool = False
    schedule: str = ""                   # e.g., "SR-SS" (sunrise to sunset)

    # Raw text
    notam_text: str = ""


# =============================================================================
# ENVIRONMENTAL HAZARDS
# =============================================================================

@dataclass
class EnvironmentalHazard(Obstacle):
    """Base class for environmental/weather hazards."""

    mobility: ObstacleMobility = field(default=ObstacleMobility.TRANSIENT)

    severity_level: int = 1              # 1-5 scale
    is_forecast: bool = False            # Predicted vs observed


@dataclass
class WeatherHazard(EnvironmentalHazard):
    """Weather-related flight hazard."""

    category: ObstacleCategory = field(default=ObstacleCategory.WEATHER_HAZARD)

    hazard_type: str = ""                # thunderstorm, icing, turbulence

    # Intensity
    precipitation_rate_mm_hr: float = 0.0
    wind_speed_m_s: float = 0.0
    turbulence_intensity: str = ""       # light, moderate, severe

    # Vertical extent
    cloud_base_m: float = 0.0
    cloud_top_m: float = 0.0

    # Movement
    movement_direction_deg: float = 0.0
    movement_speed_m_s: float = 0.0


@dataclass
class VisibilityHazard(EnvironmentalHazard):
    """Visibility-reducing hazard."""

    category: ObstacleCategory = field(default=ObstacleCategory.VISIBILITY_HAZARD)

    hazard_type: str = ""                # fog, haze, smoke, dust

    visibility_m: float = 0.0
    ceiling_m: float | None = None    # Cloud ceiling if applicable

    # For smoke/dust
    source_description: str = ""         # e.g., "stubble burning"
    particle_concentration: float = 0.0  # PM2.5 or similar


# =============================================================================
# CONFLICT DETECTION
# =============================================================================

@dataclass
class SeparationMinima:
    """Minimum separation standards for conflict detection."""

    horizontal_m: float
    vertical_m: float
    time_s: float

    alert_level: AlertLevel = AlertLevel.NONE


# Default separation minima for defense eVTOL
DEFAULT_SEPARATION_MINIMA: dict[AlertLevel, SeparationMinima] = {
    AlertLevel.ADVISORY: SeparationMinima(
        horizontal_m=2000.0, vertical_m=500.0, time_s=300.0,
        alert_level=AlertLevel.ADVISORY
    ),
    AlertLevel.CAUTION: SeparationMinima(
        horizontal_m=500.0, vertical_m=200.0, time_s=30.0,
        alert_level=AlertLevel.CAUTION
    ),
    AlertLevel.WARNING: SeparationMinima(
        horizontal_m=200.0, vertical_m=100.0, time_s=15.0,
        alert_level=AlertLevel.WARNING
    ),
    AlertLevel.CRITICAL: SeparationMinima(
        horizontal_m=50.0, vertical_m=30.0, time_s=5.0,
        alert_level=AlertLevel.CRITICAL
    ),
}


@dataclass
class ConflictAlert:
    """Alert for detected conflict with an obstacle."""

    alert_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    # Conflict parties
    own_position: tuple[float, float, float] = (0.0, 0.0, 0.0)  # lat, lon, alt
    obstacle_id: str = ""
    obstacle_category: ObstacleCategory = ObstacleCategory.UNKNOWN

    # Alert classification
    alert_level: AlertLevel = AlertLevel.NONE

    # Conflict metrics
    horizontal_distance_m: float = float('inf')
    vertical_distance_m: float = float('inf')
    slant_distance_m: float = float('inf')

    # Time-based metrics
    time_to_cpa_s: float = float('inf')  # Time to Closest Point of Approach
    distance_at_cpa_m: float = float('inf')
    time_to_collision_s: float | None = None  # Only if collision predicted

    # Recommended action
    recommended_action: str = ""          # climb, descend, turn_left, etc.

    # Timestamps
    detected_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: datetime | None = None


@dataclass
class ConflictZone:
    """Spatial region representing a conflict area."""

    zone_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    # Associated obstacle
    obstacle_id: str = ""

    # Zone geometry (typically expanded obstacle geometry)
    geometry: BoundingSphere | BoundingCylinder | BoundingBox | None = None

    # Alert level for this zone
    alert_level: AlertLevel = AlertLevel.NONE

    # Validity
    valid_from: datetime = field(default_factory=datetime.utcnow)
    valid_until: datetime | None = None


# =============================================================================
# TRACK MANAGEMENT
# =============================================================================

@dataclass
class KalmanState:
    """Kalman filter state for dynamic obstacle tracking."""

    # State vector [px, py, pz, vx, vy, vz, heading, turn_rate]
    state: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(8)
    )

    # State covariance matrix (8x8)
    covariance: NDArray[np.float64] = field(
        default_factory=lambda: np.eye(8) * 100.0
    )

    # Process noise
    process_noise: NDArray[np.float64] = field(
        default_factory=lambda: np.eye(8) * 0.1
    )

    # Measurement noise
    measurement_noise: NDArray[np.float64] = field(
        default_factory=lambda: np.eye(4) * 10.0  # [px, py, pz, heading]
    )

    @property
    def position(self) -> NDArray[np.float64]:
        """Position [px, py, pz]."""
        return self.state[:3]

    @property
    def velocity(self) -> NDArray[np.float64]:
        """Velocity [vx, vy, vz]."""
        return self.state[3:6]

    @property
    def heading(self) -> float:
        """Heading angle."""
        return float(self.state[6])

    @property
    def turn_rate(self) -> float:
        """Turn rate."""
        return float(self.state[7])

    @property
    def position_uncertainty(self) -> float:
        """1-sigma position uncertainty (RMS of position covariance)."""
        return float(np.sqrt(np.trace(self.covariance[:3, :3]) / 3))

    @property
    def velocity_uncertainty(self) -> float:
        """1-sigma velocity uncertainty."""
        return float(np.sqrt(np.trace(self.covariance[3:6, 3:6]) / 3))


@dataclass
class TrackHistory:
    """Historical states for a track."""

    timestamps: list[datetime] = field(default_factory=list)
    states: list[ObstacleState] = field(default_factory=list)

    # Maximum history length
    max_length: int = 100

    def add(self, timestamp: datetime, state: ObstacleState) -> None:
        """Add a state to history."""
        self.timestamps.append(timestamp)
        self.states.append(state)

        # Trim if exceeding max length
        if len(self.timestamps) > self.max_length:
            self.timestamps = self.timestamps[-self.max_length:]
            self.states = self.states[-self.max_length:]

    @property
    def duration_s(self) -> float:
        """Duration of track history in seconds."""
        if len(self.timestamps) < 2:
            return 0.0
        return (self.timestamps[-1] - self.timestamps[0]).total_seconds()


@dataclass
class Track:
    """Dynamic obstacle track with Kalman filter state."""

    track_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    # Associated obstacle
    obstacle: Obstacle | None = None

    # Track status
    status: TrackStatus = TrackStatus.TENTATIVE

    # Kalman filter state
    kalman: KalmanState | None = None

    # Track history
    history: TrackHistory = field(default_factory=TrackHistory)

    # Track quality metrics
    update_count: int = 0
    missed_count: int = 0

    # Confirmation thresholds
    confirm_threshold: int = 3           # Updates to confirm
    delete_threshold: int = 5            # Misses to delete

    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_update: datetime = field(default_factory=datetime.utcnow)

    @property
    def age_s(self) -> float:
        """Track age in seconds."""
        return (datetime.utcnow() - self.created_at).total_seconds()

    @property
    def staleness_s(self) -> float:
        """Time since last update in seconds."""
        return (datetime.utcnow() - self.last_update).total_seconds()

    def confirm(self) -> None:
        """Confirm the track."""
        self.status = TrackStatus.CONFIRMED

    def coast(self) -> None:
        """Set track to coasting (no recent updates)."""
        self.status = TrackStatus.COASTING
        self.missed_count += 1

    def should_delete(self) -> bool:
        """Check if track should be deleted."""
        return self.missed_count >= self.delete_threshold


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class ProviderConfig:
    """Configuration for data providers."""

    # API settings
    base_url: str = ""
    api_key: str = ""
    timeout_s: float = 30.0
    max_retries: int = 3

    # Rate limiting
    requests_per_minute: int = 60

    # Caching
    cache_enabled: bool = True
    cache_ttl_s: float = 300.0           # 5 minutes default

    # Geographic bounds (India)
    default_bounds: tuple[float, float, float, float] = (
        35.0, 6.0, 97.0, 68.0            # north, south, east, west
    )


@dataclass
class TrackerConfig:
    """Configuration for obstacle tracker."""

    # Kalman filter settings
    process_noise_position: float = 1.0
    process_noise_velocity: float = 0.5
    measurement_noise_position: float = 10.0
    measurement_noise_velocity: float = 2.0

    # Track management
    confirm_threshold: int = 3
    delete_threshold: int = 5
    coast_timeout_s: float = 30.0

    # Association
    max_association_distance_m: float = 500.0
    max_association_velocity_diff_m_s: float = 50.0

    # Prediction
    max_prediction_horizon_s: float = 60.0
    prediction_step_s: float = 1.0


@dataclass
class ConflictConfig:
    """Configuration for conflict detection."""

    # Separation minima (use defaults or customize)
    separation_minima: dict[AlertLevel, SeparationMinima] = field(
        default_factory=lambda: DEFAULT_SEPARATION_MINIMA.copy()
    )

    # Time horizons
    tactical_horizon_s: float = 30.0     # Immediate collision avoidance
    strategic_horizon_s: float = 300.0   # Route planning

    # Alert settings
    alert_persistence_s: float = 10.0    # Minimum time between repeated alerts

    # Safety buffers
    horizontal_buffer_m: float = 20.0    # Added to all obstacles
    vertical_buffer_m: float = 10.0


@dataclass
class ObstacleConfig:
    """Master configuration for obstacle module."""

    # Sub-configurations
    provider: ProviderConfig = field(default_factory=ProviderConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    conflict: ConflictConfig = field(default_factory=ConflictConfig)

    # Update rates
    static_refresh_interval_s: float = 3600.0   # 1 hour for static
    dynamic_update_interval_s: float = 5.0      # 5 seconds for aircraft

    # Spatial indexing
    use_rtree: bool = True
    rtree_page_size: int = 4096

    # Output
    output_dir: str = "outputs/perception/obstacle"
    log_conflicts: bool = True
    save_tracks: bool = True
