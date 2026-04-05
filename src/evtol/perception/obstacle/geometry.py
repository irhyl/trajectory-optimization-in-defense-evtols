"""
Geometry Operations for Obstacle Collision Detection.

This module provides geometric primitives and operations for:
    - Distance calculations (point-to-point, point-to-line, point-to-polygon)
    - Bounding volume intersection tests
    - Closest Point of Approach (CPA) computation
    - Time to Collision (TTC) estimation
    - Coordinate transformations (LLA ↔ ENU)

Mathematical Framework:
    - WGS84 ellipsoid for geodetic calculations
    - Local ENU (East-North-Up) tangent plane for 3D operations
    - Vincenty/Haversine for great-circle distances

Performance Considerations:
    - Hierarchical bounding volume tests (sphere → cylinder → OBB)
    - SIMD-friendly numpy operations
    - Spatial indexing support (R-tree compatible)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
import numpy as np
from numpy.typing import NDArray

from .obstacle_types import (
    BoundingSphere,
    BoundingCylinder,
    BoundingBox,
    OrientedBoundingBox,
    Polyline3D,
)


# =============================================================================
# CONSTANTS


# WGS84 Ellipsoid Parameters
WGS84_A = 6378137.0  # Semi-major axis (m)
WGS84_B = 6356752.314245  # Semi-minor axis (m)
WGS84_F = 1 / 298.257223563  # Flattening
WGS84_E2 = 2 * WGS84_F - WGS84_F ** 2  # First eccentricity squared

# Average Earth radius for quick calculations
EARTH_RADIUS_M = 6371000.0

# Small epsilon for floating-point comparisons
EPS = 1e-10


# =============================================================================
# TYPE ALIASES
# =============================================================================

Vec3 = NDArray[np.float64]  # 3D vector [x, y, z]
Mat3 = NDArray[np.float64]  # 3x3 matrix
Point3D = tuple[float, float, float]  # (x, y, z) or (lat, lon, alt)


# =============================================================================
# COORDINATE TRANSFORMATIONS
# =============================================================================

@dataclass
class ENUFrame:
    """
    Local East-North-Up reference frame.

    Centered at an origin point (lat, lon, alt), provides conversions
    between geodetic (LLA) and local Cartesian (ENU) coordinates.

    The ENU frame:
        - X (East): Positive towards east
        - Y (North): Positive towards north
        - Z (Up): Positive upwards (opposite to gravity)

    Attributes:
        origin_lat: Latitude of frame origin (degrees)
        origin_lon: Longitude of frame origin (degrees)
        origin_alt: Altitude of frame origin (meters MSL)
    """
    origin_lat: float
    origin_lon: float
    origin_alt: float = 0.0

    # Cached trigonometric values
    _sin_lat: float = field(init=False, repr=False)
    _cos_lat: float = field(init=False, repr=False)
    _sin_lon: float = field(init=False, repr=False)
    _cos_lon: float = field(init=False, repr=False)

    def __post_init__(self):
        lat_rad = math.radians(self.origin_lat)
        lon_rad = math.radians(self.origin_lon)
        self._sin_lat = math.sin(lat_rad)
        self._cos_lat = math.cos(lat_rad)
        self._sin_lon = math.sin(lon_rad)
        self._cos_lon = math.cos(lon_rad)

    def lla_to_ecef(self, lat: float, lon: float, alt: float) -> Vec3:
        """Convert LLA to ECEF coordinates."""
        lat_rad = math.radians(lat)
        lon_rad = math.radians(lon)

        sin_lat = math.sin(lat_rad)
        cos_lat = math.cos(lat_rad)
        sin_lon = math.sin(lon_rad)
        cos_lon = math.cos(lon_rad)

        # Radius of curvature in the prime vertical
        N = WGS84_A / math.sqrt(1 - WGS84_E2 * sin_lat ** 2)

        x = (N + alt) * cos_lat * cos_lon
        y = (N + alt) * cos_lat * sin_lon
        z = (N * (1 - WGS84_E2) + alt) * sin_lat

        return np.array([x, y, z], dtype=np.float64)

    def ecef_to_enu(self, ecef: Vec3) -> Vec3:
        """Convert ECEF to local ENU coordinates."""
        # Get ECEF of origin
        origin_ecef = self.lla_to_ecef(
            self.origin_lat, self.origin_lon, self.origin_alt
        )

        # Delta from origin
        dx = ecef[0] - origin_ecef[0]
        dy = ecef[1] - origin_ecef[1]
        dz = ecef[2] - origin_ecef[2]

        # Rotation matrix ECEF → ENU
        east = -self._sin_lon * dx + self._cos_lon * dy
        north = (-self._sin_lat * self._cos_lon * dx
                 - self._sin_lat * self._sin_lon * dy
                 + self._cos_lat * dz)
        up = (self._cos_lat * self._cos_lon * dx
              + self._cos_lat * self._sin_lon * dy
              + self._sin_lat * dz)

        return np.array([east, north, up], dtype=np.float64)

    def lla_to_enu(self, lat: float, lon: float, alt: float) -> Vec3:
        """Convert LLA directly to local ENU coordinates."""
        ecef = self.lla_to_ecef(lat, lon, alt)
        return self.ecef_to_enu(ecef)

    def enu_to_ecef(self, enu: Vec3) -> Vec3:
        """Convert local ENU to ECEF coordinates."""
        origin_ecef = self.lla_to_ecef(
            self.origin_lat, self.origin_lon, self.origin_alt
        )

        east, north, up = enu[0], enu[1], enu[2]

        # Inverse rotation matrix ENU → ECEF
        dx = (-self._sin_lon * east
              - self._sin_lat * self._cos_lon * north
              + self._cos_lat * self._cos_lon * up)
        dy = (self._cos_lon * east
              - self._sin_lat * self._sin_lon * north
              + self._cos_lat * self._sin_lon * up)
        dz = self._cos_lat * north + self._sin_lat * up

        return np.array([
            origin_ecef[0] + dx,
            origin_ecef[1] + dy,
            origin_ecef[2] + dz
        ], dtype=np.float64)

    def enu_to_lla(self, enu: Vec3) -> tuple[float, float, float]:
        """Convert local ENU to LLA coordinates."""
        ecef = self.enu_to_ecef(enu)
        return ecef_to_lla(ecef)


def ecef_to_lla(ecef: Vec3) -> tuple[float, float, float]:
    """
    Convert ECEF to LLA using iterative method.

    Returns:
        (latitude_deg, longitude_deg, altitude_m)
    """
    x, y, z = ecef[0], ecef[1], ecef[2]

    # Longitude
    lon = math.atan2(y, x)

    # Iterative latitude/altitude
    p = math.sqrt(x ** 2 + y ** 2)
    lat = math.atan2(z, p * (1 - WGS84_E2))

    for _ in range(10):  # Converges quickly
        sin_lat = math.sin(lat)
        N = WGS84_A / math.sqrt(1 - WGS84_E2 * sin_lat ** 2)
        lat_new = math.atan2(z + WGS84_E2 * N * sin_lat, p)

        if abs(lat_new - lat) < 1e-12:
            break
        lat = lat_new

    sin_lat = math.sin(lat)
    N = WGS84_A / math.sqrt(1 - WGS84_E2 * sin_lat ** 2)
    alt = p / math.cos(lat) - N

    return math.degrees(lat), math.degrees(lon), alt


# =============================================================================
# DISTANCE CALCULATIONS
# =============================================================================

def haversine_distance(
    lat1: float, lon1: float,
    lat2: float, lon2: float,
) -> float:
    """
    Calculate great-circle distance using Haversine formula.

    Args:
        lat1, lon1: First point (degrees)
        lat2, lon2: Second point (degrees)

    Returns:
        Distance in meters
    """
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)

    a = (math.sin(dlat / 2) ** 2 +
         math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return EARTH_RADIUS_M * c


def distance_3d(
    lat1: float, lon1: float, alt1: float,
    lat2: float, lon2: float, alt2: float,
) -> float:
    """
    Calculate 3D distance between two LLA points.

    Combines horizontal (Haversine) and vertical components.

    Returns:
        Distance in meters
    """
    horizontal = haversine_distance(lat1, lon1, lat2, lon2)
    vertical = abs(alt2 - alt1)
    return math.sqrt(horizontal ** 2 + vertical ** 2)


def slant_range(
    lat1: float, lon1: float, alt1: float,
    lat2: float, lon2: float, alt2: float,
) -> float:
    """
    Calculate slant range (direct line-of-sight distance).

    Uses ECEF for accuracy over long distances.

    Returns:
        Distance in meters
    """
    frame = ENUFrame(lat1, lon1, alt1)
    enu = frame.lla_to_enu(lat2, lon2, alt2)
    return float(np.linalg.norm(enu))


def point_to_line_distance(
    point: Vec3,
    line_start: Vec3,
    line_end: Vec3,
) -> tuple[float, Vec3]:
    """
    Calculate minimum distance from point to line segment.

    Args:
        point: Query point [x, y, z]
        line_start: Line segment start
        line_end: Line segment end

    Returns:
        (distance, closest_point)
    """
    line_vec = line_end - line_start
    line_len_sq = np.dot(line_vec, line_vec)

    if line_len_sq < EPS:
        # Degenerate line (single point)
        return float(np.linalg.norm(point - line_start)), line_start.copy()

    # Project point onto infinite line
    t = np.dot(point - line_start, line_vec) / line_len_sq
    t = max(0.0, min(1.0, t))  # Clamp to segment

    closest = line_start + t * line_vec
    distance = float(np.linalg.norm(point - closest))

    return distance, closest


def point_to_polyline_distance(
    point: Vec3,
    polyline: list[Vec3],
) -> tuple[float, Vec3, int]:
    """
    Calculate minimum distance from point to polyline.

    Args:
        point: Query point
        polyline: List of vertices

    Returns:
        (distance, closest_point, segment_index)
    """
    min_dist = float('inf')
    closest_point = polyline[0] if polyline else np.zeros(3)
    closest_segment = 0

    for i in range(len(polyline) - 1):
        dist, closest = point_to_line_distance(point, polyline[i], polyline[i + 1])
        if dist < min_dist:
            min_dist = dist
            closest_point = closest
            closest_segment = i

    return min_dist, closest_point, closest_segment


# =============================================================================
# BOUNDING VOLUME OPERATIONS
# =============================================================================

class BoundingVolumeOps:
    """Operations on bounding volumes for collision detection."""

    @staticmethod
    def sphere_contains_point(sphere: BoundingSphere, point: Vec3) -> bool:
        """Test if sphere contains point (ENU coordinates)."""
        center = np.array([0, 0, 0], dtype=np.float64)  # Sphere is at origin in local frame
        return float(np.linalg.norm(point - center)) <= sphere.radius_m

    @staticmethod
    def sphere_sphere_intersect(
        s1: BoundingSphere, p1: Vec3,
        s2: BoundingSphere, p2: Vec3,
    ) -> bool:
        """Test if two spheres intersect."""
        dist = float(np.linalg.norm(p1 - p2))
        return dist <= (s1.radius_m + s2.radius_m)

    @staticmethod
    def sphere_sphere_distance(
        s1: BoundingSphere, p1: Vec3,
        s2: BoundingSphere, p2: Vec3,
    ) -> float:
        """Calculate distance between sphere surfaces (negative if overlapping)."""
        center_dist = float(np.linalg.norm(p1 - p2))
        return center_dist - s1.radius_m - s2.radius_m

    @staticmethod
    def cylinder_contains_point(
        cylinder: BoundingCylinder,
        point: Vec3,
        cylinder_base: Vec3,
    ) -> bool:
        """Test if vertical cylinder contains point."""
        # Horizontal distance from axis
        dx = point[0] - cylinder_base[0]
        dy = point[1] - cylinder_base[1]
        horiz_dist = math.sqrt(dx ** 2 + dy ** 2)

        if horiz_dist > cylinder.radius_m:
            return False

        # Vertical bounds
        z_min = cylinder_base[2]
        z_max = cylinder_base[2] + cylinder.height_m

        return z_min <= point[2] <= z_max

    @staticmethod
    def aabb_contains_point(box: BoundingBox, point: Vec3) -> bool:
        """Test if axis-aligned bounding box contains point."""
        return (box.min_lat <= point[0] <= box.max_lat and
                box.min_lon <= point[1] <= box.max_lon and
                box.min_alt_m <= point[2] <= box.max_alt_m)

    @staticmethod
    def aabb_aabb_intersect(b1: BoundingBox, b2: BoundingBox) -> bool:
        """Test if two AABBs intersect."""
        return (b1.min_lat <= b2.max_lat and b1.max_lat >= b2.min_lat and
                b1.min_lon <= b2.max_lon and b1.max_lon >= b2.min_lon and
                b1.min_alt_m <= b2.max_alt_m and b1.max_alt_m >= b2.min_alt_m)

    @staticmethod
    def obb_contains_point(
        obb: OrientedBoundingBox,
        point_enu: Vec3,
        obb_center_enu: Vec3,
    ) -> bool:
        """
        Test if oriented bounding box contains point.

        Both point and OBB center must be in same ENU frame.
        """
        # Vector from OBB center to point
        d = point_enu - obb_center_enu

        # Rotate to OBB local frame (undo heading rotation)
        cos_h = math.cos(-obb.heading)
        sin_h = math.sin(-obb.heading)

        local_x = d[0] * cos_h - d[1] * sin_h
        local_y = d[0] * sin_h + d[1] * cos_h
        local_z = d[2]

        # Check against half-extents
        return (abs(local_x) <= obb.half_length and
                abs(local_y) <= obb.half_width and
                abs(local_z) <= obb.half_height)

    @staticmethod
    def obb_to_aabb(obb: OrientedBoundingBox) -> BoundingBox:
        """Convert OBB to enclosing AABB (conservative)."""
        # Maximum extent in any direction
        cos_h = abs(math.cos(obb.heading))
        sin_h = abs(math.sin(obb.heading))

        max_horiz = (obb.half_length * cos_h + obb.half_width * sin_h +
                     obb.half_length * sin_h + obb.half_width * cos_h)

        # Convert to lat/lon degrees (approximate)
        m_per_deg_lat = 111320.0
        m_per_deg_lon = 111320.0 * math.cos(math.radians(obb.center_lat))

        dlat = max_horiz / m_per_deg_lat
        dlon = max_horiz / m_per_deg_lon

        return BoundingBox(
            min_lat=obb.center_lat - dlat,
            max_lat=obb.center_lat + dlat,
            min_lon=obb.center_lon - dlon,
            max_lon=obb.center_lon + dlon,
            min_alt_m=obb.center_alt_m - obb.half_height,
            max_alt_m=obb.center_alt_m + obb.half_height,
        )


# =============================================================================
# CLOSEST POINT OF APPROACH (CPA)
# =============================================================================

@dataclass
class CPAResult:
    """Result of Closest Point of Approach calculation."""

    time_to_cpa: float  # Seconds until CPA (negative = past)
    distance_at_cpa: float  # Distance at CPA (meters)
    horizontal_distance: float  # Horizontal component (meters)
    vertical_distance: float  # Vertical component (meters)

    # Positions at CPA
    position1_at_cpa: Vec3  # First object position (ENU)
    position2_at_cpa: Vec3  # Second object position (ENU)

    # Validity
    is_converging: bool  # Objects moving towards each other
    is_valid: bool = True  # Calculation succeeded


def compute_cpa(
    pos1: Vec3, vel1: Vec3,
    pos2: Vec3, vel2: Vec3,
    max_time: float = 300.0,
) -> CPAResult:
    """
    Compute Closest Point of Approach between two moving objects.

    Uses linear motion model to find time and distance of closest approach.

    Args:
        pos1, vel1: Position and velocity of first object (ENU, m, m/s)
        pos2, vel2: Position and velocity of second object (ENU, m, m/s)
        max_time: Maximum lookahead time (seconds)

    Returns:
        CPAResult with time, distance, and positions
    """
    # Relative position and velocity
    dpos = pos2 - pos1
    dvel = vel2 - vel1

    # Current distance
    current_dist = float(np.linalg.norm(dpos))

    # Relative speed squared
    dvel_sq = np.dot(dvel, dvel)

    if dvel_sq < EPS:
        # Objects moving in parallel or stationary
        return CPAResult(
            time_to_cpa=0.0,
            distance_at_cpa=current_dist,
            horizontal_distance=float(np.linalg.norm(dpos[:2])),
            vertical_distance=abs(dpos[2]),
            position1_at_cpa=pos1.copy(),
            position2_at_cpa=pos2.copy(),
            is_converging=False,
        )

    # Time to CPA: t = -(dpos · dvel) / |dvel|²
    t_cpa = -np.dot(dpos, dvel) / dvel_sq

    # Check if converging (t_cpa > 0 means CPA is in future)
    is_converging = t_cpa > 0

    # Clamp to valid range
    t_cpa = max(0.0, min(t_cpa, max_time))

    # Positions at CPA
    pos1_cpa = pos1 + vel1 * t_cpa
    pos2_cpa = pos2 + vel2 * t_cpa

    # Distance at CPA
    dpos_cpa = pos2_cpa - pos1_cpa
    dist_cpa = float(np.linalg.norm(dpos_cpa))
    horiz_dist = float(np.linalg.norm(dpos_cpa[:2]))
    vert_dist = abs(dpos_cpa[2])

    return CPAResult(
        time_to_cpa=t_cpa,
        distance_at_cpa=dist_cpa,
        horizontal_distance=horiz_dist,
        vertical_distance=vert_dist,
        position1_at_cpa=pos1_cpa,
        position2_at_cpa=pos2_cpa,
        is_converging=is_converging,
    )


# =============================================================================
# TIME TO COLLISION (TTC)
# =============================================================================

@dataclass
class TTCResult:
    """Result of Time to Collision calculation."""

    time_to_collision: float  # Seconds (inf if no collision)
    collision_point: Vec3 | None  # Point of collision (ENU)
    will_collide: bool  # Within prediction horizon

    # Separation at various times
    separation_now: float
    min_separation: float
    time_of_min_separation: float


def compute_ttc_spheres(
    pos1: Vec3, vel1: Vec3, radius1: float,
    pos2: Vec3, vel2: Vec3, radius2: float,
    max_time: float = 300.0,
) -> TTCResult:
    """
    Compute Time to Collision for two spherical objects.

    Solves quadratic equation for sphere intersection.

    Args:
        pos1, vel1, radius1: First object state
        pos2, vel2, radius2: Second object state
        max_time: Maximum lookahead (seconds)

    Returns:
        TTCResult with collision time and details
    """
    # Relative state
    dpos = pos2 - pos1
    dvel = vel2 - vel1
    combined_radius = radius1 + radius2

    current_sep = float(np.linalg.norm(dpos)) - combined_radius

    # Already colliding?
    if current_sep <= 0:
        return TTCResult(
            time_to_collision=0.0,
            collision_point=(pos1 + pos2) / 2,
            will_collide=True,
            separation_now=current_sep,
            min_separation=current_sep,
            time_of_min_separation=0.0,
        )

    # Quadratic coefficients: |dpos + t*dvel|² = R²
    a = np.dot(dvel, dvel)
    b = 2 * np.dot(dpos, dvel)
    c = np.dot(dpos, dpos) - combined_radius ** 2

    # Get CPA for min separation
    cpa = compute_cpa(pos1, vel1, pos2, vel2, max_time)

    if a < EPS:
        # Parallel motion, no collision unless already overlapping
        return TTCResult(
            time_to_collision=float('inf'),
            collision_point=None,
            will_collide=False,
            separation_now=current_sep,
            min_separation=cpa.distance_at_cpa - combined_radius,
            time_of_min_separation=cpa.time_to_cpa,
        )

    discriminant = b ** 2 - 4 * a * c

    if discriminant < 0:
        # No intersection
        return TTCResult(
            time_to_collision=float('inf'),
            collision_point=None,
            will_collide=False,
            separation_now=current_sep,
            min_separation=cpa.distance_at_cpa - combined_radius,
            time_of_min_separation=cpa.time_to_cpa,
        )

    # Two intersection times
    sqrt_disc = math.sqrt(discriminant)
    t1 = (-b - sqrt_disc) / (2 * a)
    t2 = (-b + sqrt_disc) / (2 * a)

    # First positive time is collision
    ttc = float('inf')
    if t1 > 0:
        ttc = t1
    elif t2 > 0:
        ttc = t2

    if ttc <= max_time:
        # Collision point
        collision_pos1 = pos1 + vel1 * ttc
        collision_pos2 = pos2 + vel2 * ttc
        collision_point = (collision_pos1 + collision_pos2) / 2

        return TTCResult(
            time_to_collision=ttc,
            collision_point=collision_point,
            will_collide=True,
            separation_now=current_sep,
            min_separation=0.0,
            time_of_min_separation=ttc,
        )

    return TTCResult(
        time_to_collision=float('inf'),
        collision_point=None,
        will_collide=False,
        separation_now=current_sep,
        min_separation=max(0, cpa.distance_at_cpa - combined_radius),
        time_of_min_separation=cpa.time_to_cpa,
    )


# =============================================================================
# GEOMETRY TO BOUNDING VOLUME CONVERSION
# =============================================================================

def obstacle_to_sphere(
    lat: float, lon: float, alt: float,
    dimensions: tuple[float, float, float],
) -> tuple[BoundingSphere, Vec3]:
    """
    Create bounding sphere for obstacle.

    Args:
        lat, lon, alt: Obstacle center
        dimensions: (length, width, height) in meters

    Returns:
        (BoundingSphere, center_enu) - sphere and center for ENU frame
    """
    # Enclosing sphere radius
    radius = math.sqrt(sum(d ** 2 for d in dimensions)) / 2

    sphere = BoundingSphere(
        center_lat=lat,
        center_lon=lon,
        center_alt_m=alt,
        radius_m=radius,
    )

    return sphere, np.array([0, 0, 0], dtype=np.float64)


def polyline_to_bounding_box(polyline: Polyline3D) -> BoundingBox:
    """Create AABB enclosing polyline."""
    if not polyline.vertices:
        return BoundingBox(0, 0, 0, 0, 0, 0)

    lats = [v[0] for v in polyline.vertices]
    lons = [v[1] for v in polyline.vertices]
    alts = [v[2] for v in polyline.vertices]

    return BoundingBox(
        min_lat=min(lats),
        max_lat=max(lats),
        min_lon=min(lons),
        max_lon=max(lons),
        min_alt_m=min(alts),
        max_alt_m=max(alts) + polyline.clearance_m,
    )


# =============================================================================
# SPATIAL QUERY UTILITIES
# =============================================================================

@dataclass
class SpatialQuery:
    """
    Spatial query for finding obstacles in a region.

    Supports point, sphere, and box queries.
    """

    # Query center
    center_lat: float
    center_lon: float
    center_alt_m: float

    # Query radius (meters)
    radius_m: float

    # Optional altitude band
    min_alt_m: float | None = None
    max_alt_m: float | None = None

    def to_bounding_box(self) -> BoundingBox:
        """Convert query to approximate bounding box."""
        m_per_deg_lat = 111320.0
        m_per_deg_lon = 111320.0 * math.cos(math.radians(self.center_lat))

        dlat = self.radius_m / m_per_deg_lat
        dlon = self.radius_m / m_per_deg_lon

        return BoundingBox(
            min_lat=self.center_lat - dlat,
            max_lat=self.center_lat + dlat,
            min_lon=self.center_lon - dlon,
            max_lon=self.center_lon + dlon,
            min_alt_m=self.min_alt_m or (self.center_alt_m - self.radius_m),
            max_alt_m=self.max_alt_m or (self.center_alt_m + self.radius_m),
        )

    def contains_point(self, lat: float, lon: float, alt: float) -> bool:
        """Check if point is within query region."""
        dist = distance_3d(
            self.center_lat, self.center_lon, self.center_alt_m,
            lat, lon, alt
        )

        if dist > self.radius_m:
            return False

        if self.min_alt_m is not None and alt < self.min_alt_m:
            return False

        if self.max_alt_m is not None and alt > self.max_alt_m:
            return False

        return True


def create_corridor_query(
    start_lat: float, start_lon: float, start_alt: float,
    end_lat: float, end_lon: float, end_alt: float,
    corridor_width_m: float,
) -> list[SpatialQuery]:
    """
    Create queries covering a flight corridor.

    Samples along trajectory to create overlapping spheres.

    Args:
        start_*: Corridor start point
        end_*: Corridor end point
        corridor_width_m: Half-width of corridor

    Returns:
        List of SpatialQuery objects covering corridor
    """
    # Distance along corridor
    total_dist = distance_3d(
        start_lat, start_lon, start_alt,
        end_lat, end_lon, end_alt
    )

    # Number of samples (overlap by 50%)
    step = corridor_width_m * 1.5
    num_samples = max(2, int(total_dist / step) + 1)

    queries = []
    for i in range(num_samples):
        t = i / (num_samples - 1) if num_samples > 1 else 0.5

        # Interpolate position
        lat = start_lat + t * (end_lat - start_lat)
        lon = start_lon + t * (end_lon - start_lon)
        alt = start_alt + t * (end_alt - start_alt)

        queries.append(SpatialQuery(
            center_lat=lat,
            center_lon=lon,
            center_alt_m=alt,
            radius_m=corridor_width_m,
        ))

    return queries


# =============================================================================
# GEOMETRY VALIDATION
# =============================================================================

def validate_coordinates(lat: float, lon: float, alt: float) -> bool:
    """Validate LLA coordinates are within valid ranges."""
    if not (-90 <= lat <= 90):
        return False
    if not (-180 <= lon <= 180):
        return False
    if not (-1000 <= alt <= 100000):  # -1km to 100km
        return False
    return True


def validate_india_bounds(lat: float, lon: float) -> bool:
    """Check if coordinates are within India."""
    return (6.0 <= lat <= 35.5) and (68.0 <= lon <= 97.5)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def meters_per_degree_lat(lat: float) -> float:
    """Get meters per degree of latitude at given latitude."""
    return 111320.0


def meters_per_degree_lon(lat: float) -> float:
    """Get meters per degree of longitude at given latitude."""
    return 111320.0 * math.cos(math.radians(lat))


def degrees_to_meters(dlat: float, dlon: float, ref_lat: float) -> tuple[float, float]:
    """Convert coordinate deltas to meters."""
    m_lat = dlat * meters_per_degree_lat(ref_lat)
    m_lon = dlon * meters_per_degree_lon(ref_lat)
    return m_lat, m_lon


def meters_to_degrees(m_north: float, m_east: float, ref_lat: float) -> tuple[float, float]:
    """Convert meter offsets to coordinate deltas."""
    dlat = m_north / meters_per_degree_lat(ref_lat)
    dlon = m_east / meters_per_degree_lon(ref_lat)
    return dlat, dlon
