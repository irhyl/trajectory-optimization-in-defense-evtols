"""
Threat Field Model for Defense eVTOL Trajectory Optimization.

This module implements spatial threat cost fields that discretize the
operational airspace into a grid-based risk representation suitable
for path planning algorithms.

Mathematical Framework:

    1. Grid-Based Discretization:
        - 3D voxel grid (lat, lon, alt)
        - Configurable resolution (100m - 1km)
        - Multi-resolution support for efficiency

    2. Threat Cost Computation:
        For each grid cell, compute composite threat cost:

        C(x,y,z) = Σ_i w_i × T_i(x,y,z)

        Where:
            - T_i = Individual threat contribution
            - w_i = Threat weight (priority/severity)

    3. Cost Components:
        - Detection probability: P_d(range, RCS, terrain)
        - Engagement probability: P_e(weapon envelope)
        - Kill probability: P_k(distance, altitude)
        - Exposure time cost: f(velocity through cell)

    4. Field Operations:
        - Gradient computation for steepest descent
        - Path integration for cumulative risk
        - Level set extraction for iso-risk contours

Integration with Path Planning:
    - A* / Dijkstra: Cell cost = C(x,y,z)
    - RRT*: Edge cost = ∫ C(path) ds
    - MPC: Running cost in objective function

Output Format:
    - NumPy arrays for fast computation
    - GeoTIFF for GIS visualization
    - NetCDF for 3D volumetric data

References:
    [1] LaValle, S. "Planning Algorithms" Ch. 8
    [2] Ruz, J. "Path Planning using Potential Fields"
    [3] Bortoff, S. "Path Planning for UAVs"

Author: Defense eVTOL Trajectory Optimization System
Version: 1.0.0
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from typing import Any
from collections.abc import Callable
import numpy as np
from numpy.typing import NDArray

from .threat_types import (
    ThreatStatus,
    ThreatSystem,
)
from .detection_model import RadarDetectionModel
from .engagement_model import EngagementModel


logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

# Earth radius for coordinate conversions
EARTH_RADIUS_M = 6371000.0

# Default altitude layers (meters MSL)
DEFAULT_ALTITUDE_LAYERS = [
    50, 100, 150, 200, 300, 500, 750, 1000, 1500, 2000, 3000, 5000
]


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class ThreatFieldConfig:
    """Configuration for threat field generation."""

    # Grid resolution
    horizontal_resolution_m: float = 500.0    # Grid cell size (horizontal)
    vertical_resolution_m: float = 100.0      # Altitude layer spacing

    # Altitude range
    min_altitude_m: float = 50.0
    max_altitude_m: float = 5000.0
    altitude_layers: list[float] = field(default_factory=DEFAULT_ALTITUDE_LAYERS.copy)

    # Computation settings
    use_terrain_masking: bool = True
    include_detection_cost: bool = True
    include_engagement_cost: bool = True
    include_kill_probability: bool = True

    # Cost weighting
    detection_weight: float = 0.2
    engagement_weight: float = 0.3
    kill_weight: float = 0.5

    # Platform parameters (for cost computation)
    platform_rcs_dbsm: float = -10.0   # eVTOL RCS in dBsm
    platform_speed_m_s: float = 50.0    # Nominal speed for exposure time

    # Risk thresholds
    acceptable_risk: float = 0.01       # 1% acceptable instantaneous risk
    high_risk_threshold: float = 0.10   # 10% = high risk
    no_go_threshold: float = 0.50       # 50% = no-go zone

    # Performance
    parallel_computation: bool = True
    cache_enabled: bool = True


@dataclass
class GridBounds:
    """Geographic bounds for threat field grid."""

    min_lat: float
    max_lat: float
    min_lon: float
    max_lon: float
    min_alt_m: float
    max_alt_m: float

    @property
    def center_lat(self) -> float:
        return (self.min_lat + self.max_lat) / 2

    @property
    def center_lon(self) -> float:
        return (self.min_lon + self.max_lon) / 2

    @property
    def lat_span(self) -> float:
        return self.max_lat - self.min_lat

    @property
    def lon_span(self) -> float:
        return self.max_lon - self.min_lon

    @property
    def alt_span(self) -> float:
        return self.max_alt_m - self.min_alt_m

    def contains(self, lat: float, lon: float, alt: float) -> bool:
        """Check if point is within bounds."""
        return (
            self.min_lat <= lat <= self.max_lat and
            self.min_lon <= lon <= self.max_lon and
            self.min_alt_m <= alt <= self.max_alt_m
        )


# =============================================================================
# GRID CELL
# =============================================================================

@dataclass
class GridCell:
    """Single cell in the threat field grid."""

    # Grid indices
    i: int  # Latitude index
    j: int  # Longitude index
    k: int  # Altitude index

    # Geographic position (cell center)
    latitude: float
    longitude: float
    altitude_m: float

    # Threat costs (0-1 scale)
    detection_cost: float = 0.0
    engagement_cost: float = 0.0
    kill_probability: float = 0.0

    # Composite cost
    total_cost: float = 0.0

    # Risk classification
    is_no_go: bool = False
    is_high_risk: bool = False

    # Contributing threats
    threat_contributions: dict[str, float] = field(default_factory=dict)

    # Terrain masking factor (0=fully masked, 1=fully exposed)
    terrain_exposure: float = 1.0


@dataclass
class CostGradient:
    """Gradient of threat cost at a point."""

    # Gradient components
    d_lat: float    # Cost change per degree latitude
    d_lon: float    # Cost change per degree longitude
    d_alt: float    # Cost change per meter altitude

    # Gradient magnitude
    magnitude: float = 0.0

    # Steepest descent direction (unit vector in ENU)
    descent_east: float = 0.0
    descent_north: float = 0.0
    descent_up: float = 0.0


# =============================================================================
# THREAT FIELD
# =============================================================================

class ThreatField:
    """
    3D grid-based threat cost field for path planning.

    Discretizes operational airspace into cells, computing threat
    costs based on detection probability, engagement envelopes,
    and kill probability from all contributing threat systems.

    Example:
        >>> # Create threat field for region
        >>> bounds = GridBounds(
        ...     min_lat=28.0, max_lat=29.0,
        ...     min_lon=76.0, max_lon=78.0,
        ...     min_alt_m=50, max_alt_m=3000
        ... )
        >>> field = ThreatField(bounds, threats, config)
        >>>
        >>> # Query cost at position
        >>> cost = field.get_cost(28.5, 77.0, 500)
        >>>
        >>> # Get gradient for path planning
        >>> gradient = field.get_gradient(28.5, 77.0, 500)
        >>>
        >>> # Integrate cost along path
        >>> path_risk = field.integrate_path(waypoints)

    Inputs:
        bounds: Geographic bounds for the field
        threats: List of ThreatSystem objects
        config: ThreatFieldConfig
        detection_model: Optional DetectionModel
        engagement_model: Optional EngagementModel

    Outputs:
        - 3D cost array: shape (n_lat, n_lon, n_alt)
        - Cost queries at arbitrary positions
        - Gradient vectors for optimization
        - Path integration for trajectory evaluation
    """

    def __init__(
        self,
        bounds: GridBounds,
        threats: list[ThreatSystem],
        config: ThreatFieldConfig | None = None,
        detection_model: RadarDetectionModel | None = None,
        engagement_model: EngagementModel | None = None,
    ):
        self.bounds = bounds
        self.threats = threats
        self.config = config or ThreatFieldConfig()

        # Models
        self.detection_model = detection_model or RadarDetectionModel()
        self.engagement_model = engagement_model or EngagementModel()

        # Compute grid dimensions
        self._compute_grid_dimensions()

        # Initialize cost arrays
        self._initialize_arrays()

        # Build field
        self._is_computed = False

        logger.info(
            "ThreatField initialized: %dx%dx%d grid, %d threats",
            self.n_lat,
            self.n_lon,
            self.n_alt,
            len(threats),
        )

    def _compute_grid_dimensions(self):
        """Compute grid size based on bounds and resolution."""
        # Meters per degree at center latitude
        center_lat = self.bounds.center_lat
        m_per_deg_lat = 111132.92
        m_per_deg_lon = 111132.92 * math.cos(math.radians(center_lat))

        self.m_per_deg_lat = m_per_deg_lat
        self.m_per_deg_lon = m_per_deg_lon

        # Grid dimensions
        lat_extent_m = self.bounds.lat_span * m_per_deg_lat
        lon_extent_m = self.bounds.lon_span * m_per_deg_lon

        self.n_lat = max(2, int(lat_extent_m / self.config.horizontal_resolution_m) + 1)
        self.n_lon = max(2, int(lon_extent_m / self.config.horizontal_resolution_m) + 1)
        self.n_alt = len(self.config.altitude_layers)

        # Create coordinate arrays
        self.lat_coords = np.linspace(self.bounds.min_lat, self.bounds.max_lat, self.n_lat)
        self.lon_coords = np.linspace(self.bounds.min_lon, self.bounds.max_lon, self.n_lon)
        self.alt_coords = np.array(self.config.altitude_layers)

        # Grid spacing
        self.d_lat = self.lat_coords[1] - self.lat_coords[0] if self.n_lat > 1 else 0.01
        self.d_lon = self.lon_coords[1] - self.lon_coords[0] if self.n_lon > 1 else 0.01

    def _initialize_arrays(self):
        """Initialize cost arrays."""
        shape = (self.n_lat, self.n_lon, self.n_alt)

        self.detection_cost = np.zeros(shape, dtype=np.float32)
        self.engagement_cost = np.zeros(shape, dtype=np.float32)
        self.kill_probability = np.zeros(shape, dtype=np.float32)
        self.total_cost = np.zeros(shape, dtype=np.float32)
        self.terrain_exposure = np.ones(shape, dtype=np.float32)

        # Risk classification masks
        self.no_go_mask = np.zeros(shape, dtype=bool)
        self.high_risk_mask = np.zeros(shape, dtype=bool)

        # Threat contribution tracking
        self.threat_contributions: dict[str, NDArray] = {}

    def compute(self, progress_callback: Callable[[float], None] | None = None):
        """
        Compute threat costs for all grid cells.

        Args:
            progress_callback: Optional callback for progress updates
        """
        logger.info("Computing threat field...")

        total_cells = self.n_lat * self.n_lon * self.n_alt
        processed = 0

        # Initialize per-threat contribution arrays
        for threat in self.threats:
            self.threat_contributions[threat.threat_id] = np.zeros(
                (self.n_lat, self.n_lon, self.n_alt), dtype=np.float32
            )

        # Iterate over all cells
        for i, lat in enumerate(self.lat_coords):
            for j, lon in enumerate(self.lon_coords):
                for k, alt in enumerate(self.alt_coords):
                    self._compute_cell(i, j, k, lat, lon, alt)

                    processed += 1
                    if progress_callback and processed % 1000 == 0:
                        progress_callback(processed / total_cells)

        # Compute composite cost
        self._compute_composite_cost()

        # Classify risk zones
        self._classify_risk_zones()

        self._is_computed = True
        logger.info("Threat field computed: %d cells processed", total_cells)

    def _compute_cell(
        self,
        i: int, j: int, k: int,
        lat: float, lon: float, alt: float
    ):
        """Compute threat costs for a single cell."""

        # Accumulate costs from all threats
        cell_detection = 0.0
        cell_engagement = 0.0
        cell_kill = 0.0

        for threat in self.threats:
            # Skip inactive threats
            if threat.status != ThreatStatus.ACTIVE:
                continue

            # Compute range to threat
            range_m = self._compute_range(lat, lon, alt, threat)

            # Skip if outside max detection range (using max_range_km_1sqm converted to meters)
            max_range_m = threat.radar.max_range_km_1sqm * 1000.0 if threat.radar else 100000.0
            if range_m > max_range_m * 1.5:
                continue

            # Detection probability
            # Check radar has valid frequency (avoid division by zero)
            if self.config.include_detection_cost and threat.radar and threat.radar.frequency_ghz > 0:
                try:
                    # Convert RCS from dBsm to m² for API call
                    rcs_sqm = 10 ** (self.config.platform_rcs_dbsm / 10.0)
                    det_result = self.detection_model.calculate_detection_probability(
                        radar=threat.radar,
                        target_range_m=range_m,
                        target_altitude_m=alt,
                        target_rcs_sqm=rcs_sqm,
                    )
                    p_detect = det_result.get('detection_probability', 0.0) if isinstance(det_result, dict) else 0.0
                except (ZeroDivisionError, ValueError):
                    p_detect = 0.0
            else:
                p_detect = 0.0

            # Engagement probability
            if self.config.include_engagement_cost:
                eng_result = self.engagement_model.calculate_engagement_probability(
                    threat=threat,
                    target_lat=lat,
                    target_lon=lon,
                    target_alt_m=alt,
                )
                p_engage = eng_result.get('engagement_probability', 0.0) if isinstance(eng_result, dict) else 0.0
            else:
                p_engage = 0.0
                eng_result = {}

            # Kill probability
            if self.config.include_kill_probability:
                if isinstance(eng_result, dict) and 'kill_probability' in eng_result:
                    p_kill = eng_result['kill_probability']
                else:
                    # Simplified Pk model
                    p_kill = self._compute_simple_pk(threat, range_m, alt)
            else:
                p_kill = 0.0

            # Store per-threat contribution
            threat_cost = (
                self.config.detection_weight * p_detect +
                self.config.engagement_weight * p_engage +
                self.config.kill_weight * p_kill
            )
            self.threat_contributions[threat.threat_id][i, j, k] = threat_cost

            # Accumulate (using probabilistic OR for multiple threats)
            cell_detection = 1 - (1 - cell_detection) * (1 - p_detect)
            cell_engagement = 1 - (1 - cell_engagement) * (1 - p_engage)
            cell_kill = 1 - (1 - cell_kill) * (1 - p_kill)

        # Store results
        self.detection_cost[i, j, k] = cell_detection
        self.engagement_cost[i, j, k] = cell_engagement
        self.kill_probability[i, j, k] = cell_kill

    def _compute_range(
        self,
        lat: float, lon: float, alt: float,
        threat: ThreatSystem
    ) -> float:
        """Compute 3D range to threat."""
        # Haversine for horizontal distance
        lat1, lon1 = math.radians(lat), math.radians(lon)
        lat2, lon2 = math.radians(threat.latitude), math.radians(threat.longitude)

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))

        horizontal_m = EARTH_RADIUS_M * c
        vertical_m = alt - threat.altitude_m

        return math.sqrt(horizontal_m**2 + vertical_m**2)

    def _compute_simple_pk(
        self,
        threat: ThreatSystem,
        range_m: float,
        altitude_m: float
    ) -> float:
        """Simplified kill probability model."""
        # Check if threat has missile specification
        if not threat.missile:
            # Use base_pk from threat system with range attenuation
            if threat.envelope:
                max_range_m = threat.envelope.engagement_range_km * 1000.0
                if range_m > max_range_m:
                    return 0.0
                range_factor = 1.0 - (range_m / max_range_m) ** 2
                return max(0.0, min(1.0, threat.base_pk * range_factor))
            return 0.0

        missile = threat.missile

        # Check altitude limits using envelope
        if threat.envelope:
            if altitude_m > threat.envelope.max_altitude_m or altitude_m < threat.envelope.min_altitude_m:
                return 0.0
            max_range_m = threat.envelope.engagement_range_km * 1000.0
            if range_m > max_range_m:
                return 0.0
        else:
            max_range_m = missile.max_range_km * 1000.0 if hasattr(missile, 'max_range_km') else 50000.0
            if range_m > max_range_m:
                return 0.0

        # Base Pk with range attenuation
        range_factor = 1.0 - (range_m / max_range_m) ** 2
        single_shot_pk = getattr(missile, 'pk', threat.base_pk)
        pk = single_shot_pk * range_factor

        return max(0.0, min(1.0, pk))

    def _compute_composite_cost(self):
        """Compute weighted composite cost."""
        self.total_cost = (
            self.config.detection_weight * self.detection_cost +
            self.config.engagement_weight * self.engagement_cost +
            self.config.kill_weight * self.kill_probability
        )

        # Apply terrain exposure
        self.total_cost *= self.terrain_exposure

    def _classify_risk_zones(self):
        """Classify cells by risk level."""
        self.no_go_mask = self.total_cost >= self.config.no_go_threshold
        self.high_risk_mask = (
            (self.total_cost >= self.config.high_risk_threshold) &
            ~self.no_go_mask
        )

        # Log statistics
        no_go_pct = 100 * np.mean(self.no_go_mask)
        high_risk_pct = 100 * np.mean(self.high_risk_mask)

        logger.info("Risk classification: %.1f%% no-go, %.1f%% high-risk", no_go_pct, high_risk_pct)

    # =========================================================================
    # QUERY METHODS
    # =========================================================================

    def get_cost(self, lat: float, lon: float, alt: float) -> float:
        """
        Get interpolated threat cost at position.

        Args:
            lat: Latitude (degrees)
            lon: Longitude (degrees)
            alt: Altitude (meters MSL)

        Returns:
            Interpolated threat cost (0-1)
        """
        if not self._is_computed:
            raise RuntimeError("Threat field not computed. Call compute() first.")

        return float(self._trilinear_interpolate(
            self.total_cost, lat, lon, alt
        ))

    def get_cost_components(
        self,
        lat: float, lon: float, alt: float
    ) -> dict[str, float]:
        """Get individual cost components at position."""
        if not self._is_computed:
            raise RuntimeError("Threat field not computed. Call compute() first.")

        return {
            'detection': float(self._trilinear_interpolate(self.detection_cost, lat, lon, alt)),
            'engagement': float(self._trilinear_interpolate(self.engagement_cost, lat, lon, alt)),
            'kill_probability': float(self._trilinear_interpolate(self.kill_probability, lat, lon, alt)),
            'total': float(self._trilinear_interpolate(self.total_cost, lat, lon, alt)),
        }

    def _trilinear_interpolate(
        self,
        array: NDArray,
        lat: float, lon: float, alt: float
    ) -> float:
        """Trilinear interpolation in the grid."""
        # Normalize coordinates to grid indices
        i_f = (lat - self.bounds.min_lat) / self.d_lat
        j_f = (lon - self.bounds.min_lon) / self.d_lon

        # Find altitude index
        k_f = np.interp(alt, self.alt_coords, np.arange(len(self.alt_coords)))

        # Clamp to valid range
        i_f = np.clip(i_f, 0, self.n_lat - 1.001)
        j_f = np.clip(j_f, 0, self.n_lon - 1.001)
        k_f = np.clip(k_f, 0, self.n_alt - 1.001)

        # Integer indices
        i0, j0, k0 = int(i_f), int(j_f), int(k_f)
        i1, j1, k1 = min(i0 + 1, self.n_lat - 1), min(j0 + 1, self.n_lon - 1), min(k0 + 1, self.n_alt - 1)

        # Fractional parts
        di, dj, dk = i_f - i0, j_f - j0, k_f - k0

        # Trilinear interpolation
        c00 = array[i0, j0, k0] * (1 - di) + array[i1, j0, k0] * di
        c01 = array[i0, j0, k1] * (1 - di) + array[i1, j0, k1] * di
        c10 = array[i0, j1, k0] * (1 - di) + array[i1, j1, k0] * di
        c11 = array[i0, j1, k1] * (1 - di) + array[i1, j1, k1] * di

        c0 = c00 * (1 - dj) + c10 * dj
        c1 = c01 * (1 - dj) + c11 * dj

        return c0 * (1 - dk) + c1 * dk

    def get_gradient(self, lat: float, lon: float, alt: float) -> CostGradient:
        """
        Compute cost gradient at position.

        Uses central differences for gradient approximation.

        Args:
            lat, lon, alt: Position

        Returns:
            CostGradient with descent direction
        """
        if not self._is_computed:
            raise RuntimeError("Threat field not computed. Call compute() first.")

        # Step sizes
        h_lat = self.d_lat / 2
        h_lon = self.d_lon / 2
        h_alt = (self.alt_coords[1] - self.alt_coords[0]) / 2 if self.n_alt > 1 else 50.0

        # Central differences
        d_lat = (self.get_cost(lat + h_lat, lon, alt) -
                 self.get_cost(lat - h_lat, lon, alt)) / (2 * h_lat)

        d_lon = (self.get_cost(lat, lon + h_lon, alt) -
                 self.get_cost(lat, lon - h_lon, alt)) / (2 * h_lon)

        d_alt = (self.get_cost(lat, lon, alt + h_alt) -
                 self.get_cost(lat, lon, alt - h_alt)) / (2 * h_alt)

        # Convert to ENU gradient
        d_east = d_lon / self.m_per_deg_lon
        d_north = d_lat / self.m_per_deg_lat
        d_up = d_alt

        magnitude = math.sqrt(d_east**2 + d_north**2 + d_up**2)

        # Steepest descent (negative gradient)
        if magnitude > 1e-10:
            descent_east = -d_east / magnitude
            descent_north = -d_north / magnitude
            descent_up = -d_up / magnitude
        else:
            descent_east = descent_north = descent_up = 0.0

        return CostGradient(
            d_lat=d_lat,
            d_lon=d_lon,
            d_alt=d_alt,
            magnitude=magnitude,
            descent_east=descent_east,
            descent_north=descent_north,
            descent_up=descent_up,
        )

    def is_no_go(self, lat: float, lon: float, alt: float) -> bool:
        """Check if position is in no-go zone."""
        cost = self.get_cost(lat, lon, alt)
        return cost >= self.config.no_go_threshold

    def is_high_risk(self, lat: float, lon: float, alt: float) -> bool:
        """Check if position is high-risk."""
        cost = self.get_cost(lat, lon, alt)
        return (cost >= self.config.high_risk_threshold and
                cost < self.config.no_go_threshold)

    # =========================================================================
    # PATH OPERATIONS
    # =========================================================================

    def integrate_path(
        self,
        waypoints: list[tuple[float, float, float]],
        method: str = 'trapezoidal'
    ) -> float:
        """
        Integrate threat cost along a path.

        Args:
            waypoints: List of (lat, lon, alt) tuples
            method: Integration method ('trapezoidal', 'simpson', 'midpoint')

        Returns:
            Cumulative threat exposure (cost × distance)
        """
        if len(waypoints) < 2:
            return 0.0

        total_exposure = 0.0

        for i in range(len(waypoints) - 1):
            lat1, lon1, alt1 = waypoints[i]
            lat2, lon2, alt2 = waypoints[i + 1]

            # Segment distance
            segment_dist = self._compute_range(
                lat1, lon1, alt1,
                type('Threat', (), {'latitude': lat2, 'longitude': lon2, 'altitude_m': alt2})()
            )

            if method == 'trapezoidal':
                cost1 = self.get_cost(lat1, lon1, alt1)
                cost2 = self.get_cost(lat2, lon2, alt2)
                segment_exposure = 0.5 * (cost1 + cost2) * segment_dist

            elif method == 'midpoint':
                mid_lat = (lat1 + lat2) / 2
                mid_lon = (lon1 + lon2) / 2
                mid_alt = (alt1 + alt2) / 2
                cost_mid = self.get_cost(mid_lat, mid_lon, mid_alt)
                segment_exposure = cost_mid * segment_dist

            elif method == 'simpson':
                mid_lat = (lat1 + lat2) / 2
                mid_lon = (lon1 + lon2) / 2
                mid_alt = (alt1 + alt2) / 2
                cost1 = self.get_cost(lat1, lon1, alt1)
                cost2 = self.get_cost(lat2, lon2, alt2)
                cost_mid = self.get_cost(mid_lat, mid_lon, mid_alt)
                segment_exposure = (cost1 + 4*cost_mid + cost2) / 6 * segment_dist

            else:
                raise ValueError(f"Unknown integration method: {method}")

            total_exposure += segment_exposure

        return total_exposure

    def compute_survival_probability(
        self,
        waypoints: list[tuple[float, float, float]],
        speed_m_s: float | None = None
    ) -> float:
        """
        Compute probability of surviving a path.

        Uses time-integrated kill probability:
        P(survival) = exp(-∫ λ(x) dt)

        where λ is instantaneous kill rate.

        Args:
            waypoints: Path waypoints
            speed_m_s: Platform speed (uses config default if None)

        Returns:
            Survival probability (0-1)
        """
        speed = speed_m_s or self.config.platform_speed_m_s

        total_hazard = 0.0

        for i in range(len(waypoints) - 1):
            lat1, lon1, alt1 = waypoints[i]
            lat2, lon2, alt2 = waypoints[i + 1]

            # Segment distance and time
            dist = self._compute_range(
                lat1, lon1, alt1,
                type('T', (), {'latitude': lat2, 'longitude': lon2, 'altitude_m': alt2})()
            )
            time_s = dist / speed

            # Average kill probability along segment
            pk1 = float(self._trilinear_interpolate(self.kill_probability, lat1, lon1, alt1))
            pk2 = float(self._trilinear_interpolate(self.kill_probability, lat2, lon2, alt2))
            avg_pk = (pk1 + pk2) / 2

            # Convert to hazard rate (instantaneous kill rate)
            # Assume Pk is per-engagement, typical engagement window = 30s
            engagement_window_s = 30.0
            hazard_rate = avg_pk / engagement_window_s

            total_hazard += hazard_rate * time_s

        return math.exp(-total_hazard)

    # =========================================================================
    # EXPORT METHODS
    # =========================================================================

    def to_numpy(self) -> dict[str, NDArray]:
        """Export field as NumPy arrays."""
        return {
            'total_cost': self.total_cost.copy(),
            'detection_cost': self.detection_cost.copy(),
            'engagement_cost': self.engagement_cost.copy(),
            'kill_probability': self.kill_probability.copy(),
            'lat_coords': self.lat_coords.copy(),
            'lon_coords': self.lon_coords.copy(),
            'alt_coords': self.alt_coords.copy(),
            'no_go_mask': self.no_go_mask.copy(),
            'high_risk_mask': self.high_risk_mask.copy(),
        }

    def to_dict(self) -> dict[str, Any]:
        """Export field as dictionary for JSON serialization."""
        return {
            'bounds': {
                'min_lat': self.bounds.min_lat,
                'max_lat': self.bounds.max_lat,
                'min_lon': self.bounds.min_lon,
                'max_lon': self.bounds.max_lon,
                'min_alt_m': self.bounds.min_alt_m,
                'max_alt_m': self.bounds.max_alt_m,
            },
            'grid': {
                'n_lat': self.n_lat,
                'n_lon': self.n_lon,
                'n_alt': self.n_alt,
                'd_lat': self.d_lat,
                'd_lon': self.d_lon,
            },
            'config': {
                'horizontal_resolution_m': self.config.horizontal_resolution_m,
                'detection_weight': self.config.detection_weight,
                'engagement_weight': self.config.engagement_weight,
                'kill_weight': self.config.kill_weight,
            },
            'statistics': {
                'mean_cost': float(np.mean(self.total_cost)),
                'max_cost': float(np.max(self.total_cost)),
                'min_cost': float(np.min(self.total_cost)),
                'no_go_fraction': float(np.mean(self.no_go_mask)),
                'high_risk_fraction': float(np.mean(self.high_risk_mask)),
            },
            'threats': [t.threat_id for t in self.threats],
        }

    def get_slice(
        self,
        altitude_m: float,
    ) -> tuple[NDArray, NDArray, NDArray]:
        """
        Get 2D cost slice at specified altitude.

        Args:
            altitude_m: Altitude for slice

        Returns:
            (lat_grid, lon_grid, cost_grid) meshgrids
        """
        # Find nearest altitude layer
        k = int(np.argmin(np.abs(self.alt_coords - altitude_m)))

        lat_grid, lon_grid = np.meshgrid(self.lat_coords, self.lon_coords, indexing='ij')
        cost_grid = self.total_cost[:, :, k]

        return lat_grid, lon_grid, cost_grid

    def get_statistics(self) -> dict[str, Any]:
        """Get field statistics."""
        return {
            'grid_size': (self.n_lat, self.n_lon, self.n_alt),
            'total_cells': self.n_lat * self.n_lon * self.n_alt,
            'bounds': {
                'lat': (self.bounds.min_lat, self.bounds.max_lat),
                'lon': (self.bounds.min_lon, self.bounds.max_lon),
                'alt': (self.bounds.min_alt_m, self.bounds.max_alt_m),
            },
            'cost_stats': {
                'mean': float(np.mean(self.total_cost)),
                'std': float(np.std(self.total_cost)),
                'min': float(np.min(self.total_cost)),
                'max': float(np.max(self.total_cost)),
                'median': float(np.median(self.total_cost)),
            },
            'risk_zones': {
                'no_go_cells': int(np.sum(self.no_go_mask)),
                'no_go_percent': 100 * float(np.mean(self.no_go_mask)),
                'high_risk_cells': int(np.sum(self.high_risk_mask)),
                'high_risk_percent': 100 * float(np.mean(self.high_risk_mask)),
            },
            'threats': {
                'count': len(self.threats),
                'active': sum(1 for t in self.threats if t.status == ThreatStatus.ACTIVE),
            },
        }


# =============================================================================
# MULTI-RESOLUTION THREAT FIELD
# =============================================================================

class MultiResolutionThreatField:
    """
    Multi-resolution threat field for efficient queries.

    Uses octree-like structure with coarse global grid and
    fine grids in high-threat areas.
    """

    def __init__(
        self,
        bounds: GridBounds,
        threats: list[ThreatSystem],
        coarse_resolution_m: float = 2000.0,
        fine_resolution_m: float = 200.0,
    ):
        self.bounds = bounds
        self.threats = threats
        self.coarse_resolution = coarse_resolution_m
        self.fine_resolution = fine_resolution_m

        # Coarse global field
        coarse_config = ThreatFieldConfig(
            horizontal_resolution_m=coarse_resolution_m
        )
        self.coarse_field = ThreatField(bounds, threats, coarse_config)

        # Fine fields for high-threat regions
        self.fine_fields: dict[str, ThreatField] = {}

        self._is_computed = False

    def compute(self):
        """Compute multi-resolution field."""
        # First compute coarse field
        self.coarse_field.compute()

        # Identify high-threat regions needing refinement
        self._identify_refinement_regions()

        # Compute fine fields
        for region_id, fine_field in self.fine_fields.items():
            logger.info(f"Computing fine field: {region_id}")
            fine_field.compute()

        self._is_computed = True

    def _identify_refinement_regions(self):
        """Identify regions needing higher resolution."""
        # Find high-cost areas in coarse grid

        for threat in self.threats:
            if threat.status != ThreatStatus.ACTIVE:
                continue

            # Create fine grid around each threat
            margin = threat.radar.max_range_km_1sqm / 111.0  # degrees (km to degrees)

            fine_bounds = GridBounds(
                min_lat=max(self.bounds.min_lat, threat.latitude - margin),
                max_lat=min(self.bounds.max_lat, threat.latitude + margin),
                min_lon=max(self.bounds.min_lon, threat.longitude - margin),
                max_lon=min(self.bounds.max_lon, threat.longitude + margin),
                min_alt_m=self.bounds.min_alt_m,
                max_alt_m=self.bounds.max_alt_m,
            )

            fine_config = ThreatFieldConfig(
                horizontal_resolution_m=self.fine_resolution
            )

            self.fine_fields[threat.threat_id] = ThreatField(
                fine_bounds, [threat], fine_config
            )

    def get_cost(self, lat: float, lon: float, alt: float) -> float:
        """Get cost using appropriate resolution."""
        if not self._is_computed:
            raise RuntimeError("Field not computed")

        # Check if in any fine field
        for _region_id, fine_field in self.fine_fields.items():
            if fine_field.bounds.contains(lat, lon, alt):
                return fine_field.get_cost(lat, lon, alt)

        # Fall back to coarse field
        return self.coarse_field.get_cost(lat, lon, alt)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_threat_field(
    center_lat: float,
    center_lon: float,
    radius_km: float,
    threats: list[ThreatSystem],
    resolution_m: float = 500.0,
) -> ThreatField:
    """
    Create threat field centered on a point.

    Args:
        center_lat: Center latitude
        center_lon: Center longitude
        radius_km: Radius in kilometers
        threats: List of threat systems
        resolution_m: Grid resolution

    Returns:
        Configured ThreatField
    """
    margin = radius_km / 111.0  # Approximate degrees

    bounds = GridBounds(
        min_lat=center_lat - margin,
        max_lat=center_lat + margin,
        min_lon=center_lon - margin,
        max_lon=center_lon + margin,
        min_alt_m=50.0,
        max_alt_m=5000.0,
    )

    config = ThreatFieldConfig(horizontal_resolution_m=resolution_m)

    return ThreatField(bounds, threats, config)


def evaluate_route(
    waypoints: list[tuple[float, float, float]],
    threat_field: ThreatField,
    speed_m_s: float = 50.0,
) -> dict[str, Any]:
    """
    Evaluate a route against threat field.

    Args:
        waypoints: Route waypoints
        threat_field: Computed threat field
        speed_m_s: Platform speed

    Returns:
        Route evaluation metrics
    """
    # Path integration
    total_exposure = threat_field.integrate_path(waypoints)
    survival_prob = threat_field.compute_survival_probability(waypoints, speed_m_s)

    # Per-waypoint analysis
    waypoint_costs = []
    max_cost = 0.0
    no_go_violations = 0

    for lat, lon, alt in waypoints:
        cost = threat_field.get_cost(lat, lon, alt)
        waypoint_costs.append(cost)
        max_cost = max(max_cost, cost)
        if threat_field.is_no_go(lat, lon, alt):
            no_go_violations += 1

    return {
        'total_exposure': total_exposure,
        'survival_probability': survival_prob,
        'max_instantaneous_cost': max_cost,
        'mean_cost': sum(waypoint_costs) / len(waypoint_costs),
        'no_go_violations': no_go_violations,
        'waypoint_costs': waypoint_costs,
        'is_feasible': no_go_violations == 0,
    }
