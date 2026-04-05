"""
Unified Environment Model for Defense eVTOL Operations.

This module provides a fused representation of the operational environment
by integrating terrain, wind, obstacle, and threat models into a single
coherent framework for trajectory planning and optimization.

Mathematical Framework:

    Total Environment Cost:

        C_total(x, t) = Σᵢ wᵢ · Cᵢ(x, t)

    Where:
        x = (lat, lon, alt) - 3D position
        t = time
        wᵢ = weight for cost component i
        Cᵢ = individual cost function (terrain, wind, obstacle, threat)

    Normalized cost with risk-aware weighting:

        C_norm(x, t) = Σᵢ wᵢ · σᵢ(Cᵢ(x, t))

    Where σᵢ is a normalization function (e.g., sigmoid, softmax)

    Survival Probability Integration:

        P_survival(path) = ∏ⱼ (1 - P_kill(xⱼ))

    Where xⱼ are waypoints along the path

References:
    [1] LaValle, S. "Planning Algorithms" - Chapter 8
    [2] Choset et al. "Principles of Robot Motion"
    [3] Kochenderfer, M. "Decision Making Under Uncertainty"

Author: Defense eVTOL Research Team
"""

from __future__ import annotations

import logging
import math
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import (
    Any
)
import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

# Earth parameters
EARTH_RADIUS_M = 6371000.0
METERS_PER_DEG_LAT = 111320.0

# Default altitude range for eVTOL operations (meters MSL)
MIN_OPERATIONAL_ALTITUDE_M = 30.0
MAX_OPERATIONAL_ALTITUDE_M = 5000.0

# Cost normalization bounds
COST_MIN = 0.0
COST_MAX = 1.0


# =============================================================================
# ENUMERATIONS
# =============================================================================

class CostComponent(Enum):
    """Individual cost components in the environment model."""

    TERRAIN = auto()        # Terrain traversability/elevation cost
    WIND = auto()           # Wind/atmospheric cost
    OBSTACLE = auto()       # Static and dynamic obstacles
    THREAT = auto()         # Military threat (detection/engagement)
    ENERGY = auto()         # Energy consumption cost
    TIME = auto()           # Time/distance cost
    NOISE = auto()          # Acoustic signature cost
    COMBINED = auto()       # Fused total cost


class NormalizationMethod(Enum):
    """Cost normalization methods."""

    LINEAR = auto()         # Linear min-max scaling
    SIGMOID = auto()        # Sigmoid transformation
    SOFTMAX = auto()        # Softmax (relative scaling)
    LOG = auto()            # Logarithmic scaling
    EXPONENTIAL = auto()    # Exponential emphasis on high costs
    QUANTILE = auto()       # Quantile-based normalization


class FusionMethod(Enum):
    """Methods for fusing multiple cost components."""

    WEIGHTED_SUM = auto()           # Linear weighted sum
    WEIGHTED_PRODUCT = auto()       # Product of weighted costs
    MAX_COST = auto()               # Maximum cost dominates
    PROBABILISTIC = auto()          # Probability-based fusion
    LEXICOGRAPHIC = auto()          # Priority-ordered fusion
    PARETO = auto()                 # Multi-objective Pareto
    RISK_SENSITIVE = auto()         # CVaR-based risk-sensitive


class QueryMode(Enum):
    """Environment query modes."""

    POINT = auto()          # Single point query
    PATH = auto()           # Path/trajectory query
    GRID = auto()           # Grid region query
    BATCH = auto()          # Batch of points


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class CostWeights:
    """
    Weighting factors for different cost components.

    Weights are normalized internally to sum to 1.0.
    Higher weights give more importance to that cost component.

    For defense operations, threat weight is typically highest.
    """

    terrain: float = 0.15
    wind: float = 0.10
    obstacle: float = 0.20
    threat: float = 0.40
    energy: float = 0.10
    time: float = 0.05

    # Risk-sensitivity parameter (0 = risk-neutral, 1 = risk-averse)
    risk_sensitivity: float = 0.7

    # Minimum survivability threshold
    min_survival_prob: float = 0.7

    def __post_init__(self):
        """Validate and normalize weights."""
        self._validate()

    def _validate(self):
        """Ensure weights are valid."""
        weights = [
            self.terrain, self.wind, self.obstacle,
            self.threat, self.energy, self.time
        ]
        if any(w < 0 for w in weights):
            raise ValueError("Weights cannot be negative")
        if sum(weights) == 0:
            raise ValueError("At least one weight must be positive")

    @property
    def normalized(self) -> dict[str, float]:
        """Get normalized weights summing to 1.0."""
        total = (
            self.terrain + self.wind + self.obstacle +
            self.threat + self.energy + self.time
        )
        return {
            'terrain': self.terrain / total,
            'wind': self.wind / total,
            'obstacle': self.obstacle / total,
            'threat': self.threat / total,
            'energy': self.energy / total,
            'time': self.time / total,
        }

    def to_array(self) -> NDArray[np.float64]:
        """Convert to numpy array (ordered)."""
        norm = self.normalized
        return np.array([
            norm['terrain'], norm['wind'], norm['obstacle'],
            norm['threat'], norm['energy'], norm['time']
        ])

    @classmethod
    def defense_priority(cls) -> CostWeights:
        """Weights prioritizing threat avoidance."""
        return cls(
            terrain=0.10,
            wind=0.05,
            obstacle=0.15,
            threat=0.55,
            energy=0.10,
            time=0.05,
            risk_sensitivity=0.8,
            min_survival_prob=0.8,
        )

    @classmethod
    def balanced(cls) -> CostWeights:
        """Balanced weights for general operations."""
        return cls(
            terrain=0.15,
            wind=0.10,
            obstacle=0.20,
            threat=0.25,
            energy=0.15,
            time=0.15,
            risk_sensitivity=0.5,
            min_survival_prob=0.6,
        )

    @classmethod
    def energy_priority(cls) -> CostWeights:
        """Weights prioritizing energy efficiency."""
        return cls(
            terrain=0.15,
            wind=0.25,
            obstacle=0.15,
            threat=0.15,
            energy=0.25,
            time=0.05,
            risk_sensitivity=0.4,
            min_survival_prob=0.5,
        )


@dataclass
class FusedCost:
    """
    Complete fused cost at a single point or for a path segment.

    Contains both the total cost and individual components
    for analysis and debugging.
    """

    # Total fused cost (0-1, normalized)
    total: float = 0.0

    # Individual component costs (0-1 each)
    terrain: float = 0.0
    wind: float = 0.0
    obstacle: float = 0.0
    threat: float = 0.0
    energy: float = 0.0
    time: float = 0.0

    # Risk metrics
    detection_probability: float = 0.0
    engagement_probability: float = 0.0
    kill_probability: float = 0.0
    survival_probability: float = 1.0

    # Constraint violations
    is_feasible: bool = True
    constraint_violations: list[str] = field(default_factory=list)

    # Gradients (for optimization)
    gradient_lat: float = 0.0
    gradient_lon: float = 0.0
    gradient_alt: float = 0.0

    # Uncertainty
    uncertainty: float = 0.0
    confidence: float = 1.0

    # Explainability and source reliability
    component_confidence: dict[str, float] = field(default_factory=dict)
    source_status: dict[str, str] = field(default_factory=dict)

    @property
    def components(self) -> dict[str, float]:
        """Get all cost components as dict."""
        return {
            'terrain': self.terrain,
            'wind': self.wind,
            'obstacle': self.obstacle,
            'threat': self.threat,
            'energy': self.energy,
            'time': self.time,
        }

    @property
    def gradient(self) -> NDArray[np.float64]:
        """Get gradient as numpy array."""
        return np.array([
            self.gradient_lat,
            self.gradient_lon,
            self.gradient_alt
        ])

    def to_array(self) -> NDArray[np.float64]:
        """Convert to feature array for ML."""
        return np.array([
            self.total,
            self.terrain,
            self.wind,
            self.obstacle,
            self.threat,
            self.energy,
            self.time,
            self.detection_probability,
            self.engagement_probability,
            self.kill_probability,
            self.survival_probability,
            float(self.is_feasible),
            self.uncertainty,
            self.confidence,
        ])

    @classmethod
    def feature_names(cls) -> list[str]:
        """Get feature names for ML."""
        return [
            'total_cost',
            'terrain_cost',
            'wind_cost',
            'obstacle_cost',
            'threat_cost',
            'energy_cost',
            'time_cost',
            'detection_prob',
            'engagement_prob',
            'kill_prob',
            'survival_prob',
            'is_feasible',
            'uncertainty',
            'confidence',
        ]


@dataclass
class EnvironmentQuery:
    """
    Query to the environment model.

    Can represent a single point, a path, or a grid region.
    """

    # Query mode
    mode: QueryMode = QueryMode.POINT

    # Single point (for POINT mode)
    latitude: float | None = None
    longitude: float | None = None
    altitude_m: float | None = None

    # Path (for PATH mode)
    waypoints: list[tuple[float, float, float]] | None = None

    # Grid bounds (for GRID mode)
    min_lat: float | None = None
    max_lat: float | None = None
    min_lon: float | None = None
    max_lon: float | None = None
    min_alt: float | None = None
    max_alt: float | None = None
    resolution_m: float = 100.0

    # Batch points (for BATCH mode)
    points: NDArray[np.float64] | None = None

    # Temporal
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    # Vehicle parameters
    velocity_mps: float = 30.0
    heading_deg: float = 0.0
    rcs_sqm: float = 0.5

    # Options
    include_gradients: bool = False
    include_uncertainty: bool = False
    components_only: bool = False


@dataclass
class EnvironmentResponse:
    """
    Response from environment model query.
    """

    # Query metadata
    query_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    query_mode: QueryMode = QueryMode.POINT
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    # Results
    cost: FusedCost | None = None                    # Single point
    path_costs: list[FusedCost] | None = None        # Path
    grid_costs: NDArray[np.float64] | None = None    # Grid
    batch_costs: list[FusedCost] | None = None       # Batch

    # Path metrics (for PATH mode)
    total_path_cost: float = 0.0
    cumulative_survival: float = 1.0
    path_feasible: bool = True

    # Grid metadata (for GRID mode)
    grid_shape: tuple[int, int, int] | None = None
    grid_bounds: dict[str, float] | None = None
    grid_layers: dict[str, NDArray[np.float64]] | None = None

    # Computation info
    computation_time_ms: float = 0.0
    cache_hit: bool = False

    # Data quality and provenance metadata
    data_quality: dict[str, Any] = field(default_factory=dict)


@dataclass
class EnvironmentState:
    """
    Current state of the environment.

    Captures time-varying conditions that affect costs.
    """

    # Temporal
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    # Weather conditions
    visibility_km: float = 10.0
    cloud_ceiling_m: float = 3000.0
    precipitation_mmhr: float = 0.0

    # Threat alert level (affects threat model)
    alert_level: str = "ELEVATED"

    # Scenario intensity
    scenario_intensity: str = "HEIGHTENED"

    # Active regions
    active_theaters: list[str] = field(default_factory=lambda: ["WESTERN", "NORTHERN"])

    # Dynamic updates
    threat_updates: list[dict[str, Any]] = field(default_factory=list)
    obstacle_updates: list[dict[str, Any]] = field(default_factory=list)

    def advance_time(self, delta: timedelta) -> EnvironmentState:
        """Create new state with advanced time."""
        return EnvironmentState(
            timestamp=self.timestamp + delta,
            visibility_km=self.visibility_km,
            cloud_ceiling_m=self.cloud_ceiling_m,
            precipitation_mmhr=self.precipitation_mmhr,
            alert_level=self.alert_level,
            scenario_intensity=self.scenario_intensity,
            active_theaters=self.active_theaters.copy(),
            threat_updates=self.threat_updates.copy(),
            obstacle_updates=self.obstacle_updates.copy(),
        )


@dataclass
class EnvironmentConfig:
    """
    Configuration for the Environment Model.
    """

    # Cost weights
    weights: CostWeights = field(default_factory=CostWeights.defense_priority)

    # Fusion settings
    fusion_method: FusionMethod = FusionMethod.WEIGHTED_SUM
    normalization: NormalizationMethod = NormalizationMethod.SIGMOID

    # Threat model settings
    threat_scenario: str = "western_heightened"
    include_detection: bool = True
    include_engagement: bool = True
    include_kill: bool = True
    threat_min_coverage_radius_km: float = 120.0
    threat_unknown_penalty: float = 0.08

    # Terrain settings
    terrain_source: str = "open_meteo"
    terrain_resolution_m: float = 30.0

    # Wind settings
    wind_source: str = "open_meteo"
    wind_fetch_hours: int = 24
    local_api_margin_deg: float = 0.2
    wind_max_retries: int = 2
    wind_circuit_breaker_threshold: int = 3
    wind_circuit_breaker_cooldown_s: float = 120.0

    # Obstacle settings
    include_static_obstacles: bool = True
    include_dynamic_obstacles: bool = True
    max_obstacle_query_interval_s: float = 120.0
    obstacle_max_retries: int = 1
    obstacle_circuit_breaker_threshold: int = 3
    obstacle_circuit_breaker_cooldown_s: float = 120.0

    # Grid settings
    default_grid_resolution_m: float = 100.0
    max_grid_points: int = 30000
    batch_chunk_size: int = 2048
    altitude_layers: list[float] = field(default_factory=lambda: [
        50, 100, 200, 300, 500, 750, 1000, 1500, 2000, 3000
    ])

    # Caching
    enable_cache: bool = True
    cache_size_mb: int = 256

    # Parallel processing
    num_workers: int = 4


# =============================================================================
# ENVIRONMENT MODEL
# =============================================================================

class EnvironmentModel:
    """
    Unified environment model fusing all perception components.

    This is the main interface for the planning layer to query
    environmental costs and constraints.

    Example:
        >>> config = EnvironmentConfig()
        >>> env = EnvironmentModel(config)
        >>>
        >>> # Query single point
        >>> query = EnvironmentQuery(
        ...     mode=QueryMode.POINT,
        ...     latitude=28.5, longitude=77.0, altitude_m=500
        ... )
        >>> response = env.query(query)
        >>> print(f"Cost: {response.cost.total:.3f}")
        >>>
        >>> # Query path
        >>> waypoints = [(28.0, 77.0, 200), (28.5, 77.5, 500), (29.0, 78.0, 300)]
        >>> path_query = EnvironmentQuery(
        ...     mode=QueryMode.PATH,
        ...     waypoints=waypoints
        ... )
        >>> path_response = env.query(path_query)
        >>> print(f"Survival: {path_response.cumulative_survival:.2%}")
    """

    def __init__(
        self,
        config: EnvironmentConfig | None = None,
        terrain_model: Any | None = None,
        wind_model: Any | None = None,
        obstacle_model: Any | None = None,
        threat_model: Any | None = None,
    ):
        """
        Initialize environment model.

        Args:
            config: Configuration settings
            terrain_model: Optional pre-initialized terrain model
            wind_model: Optional pre-initialized wind model
            obstacle_model: Optional pre-initialized obstacle model
            threat_model: Optional pre-initialized threat model
        """
        self.config = config or EnvironmentConfig()

        # Store or create models
        self._terrain_model = terrain_model
        self._wind_model = wind_model
        self._obstacle_model = obstacle_model
        self._threat_model = threat_model
        self._dynamic_obstacle_model: Any | None = None
        self._threat_systems: list[Any] = []

        # Current state
        self._state = EnvironmentState()

        # Cache
        self._cache: dict[str, FusedCost] = {}
        self._cache_hits = 0
        self._cache_misses = 0
        self._wind_forecast_cache: dict[str, Any] = {}
        self._last_obstacle_fetch: datetime | None = None
        self._obstacle_snapshot: list[Any] = []

        # Provider resilience state (production hardening).
        self._wind_failure_count = 0
        self._obstacle_failure_count = 0
        self._wind_circuit_open_until: datetime | None = None
        self._obstacle_circuit_open_until: datetime | None = None

        # Initialize models if not provided
        self._initialize_models()

        logger.info("EnvironmentModel initialized")

    def _initialize_models(self):
        """Initialize perception models."""
        if self._terrain_model is None:
            try:
                from ..terrain.data_provider import TerrainDataProvider

                self._terrain_model = TerrainDataProvider()
                logger.info("Environment fusion connected to TerrainDataProvider")
            except (ImportError, RuntimeError, ValueError) as exc:
                logger.warning("Terrain provider unavailable, using fallback terrain cost: %s", exc)

        if self._wind_model is None:
            try:
                from ..wind.data_provider import WindDataProvider

                self._wind_model = WindDataProvider()
                logger.info("Environment fusion connected to WindDataProvider")
            except (ImportError, RuntimeError, ValueError) as exc:
                logger.warning("Wind provider unavailable, using fallback wind cost: %s", exc)

        if self._obstacle_model is None and self.config.include_static_obstacles:
            try:
                from ..obstacle.data_provider import get_osm_provider

                self._obstacle_model = get_osm_provider()
                logger.info("Environment fusion connected to OSM obstacle provider")
            except (ImportError, RuntimeError, ValueError) as exc:
                logger.warning("Obstacle provider unavailable, using fallback obstacle cost: %s", exc)

        if (
            self._dynamic_obstacle_model is None
            and self._obstacle_model is None
            and self.config.include_dynamic_obstacles
        ):
            try:
                from ..obstacle.data_provider import get_opensky_provider

                self._dynamic_obstacle_model = get_opensky_provider(cache_ttl_s=10.0)
                logger.info("Environment fusion connected to OpenSky dynamic obstacle provider")
            except (ImportError, RuntimeError, ValueError) as exc:
                logger.warning("Dynamic obstacle provider unavailable, using static-only obstacle cost: %s", exc)

        if self._threat_model is None:
            try:
                from ..threat.threat_aggregator import create_aggregator

                self._threat_systems = self._build_default_threat_systems()
                self._threat_model = create_aggregator(self._threat_systems)
                logger.info(
                    "Environment fusion connected to ThreatAggregator with %d seeded threats",
                    len(self._threat_systems),
                )
            except (ImportError, RuntimeError, ValueError) as exc:
                logger.warning("Threat aggregator unavailable, using fallback threat model: %s", exc)

    def _build_default_threat_systems(self) -> list[Any]:
        """Build deterministic baseline threat systems for scenario coverage."""
        from ..threat.threat_types import create_threat_from_template

        scenario = self.config.threat_scenario.lower()

        if "western" in scenario:
            center_lat, center_lon = 32.8, 76.6
        elif "northern" in scenario:
            center_lat, center_lon = 34.3, 74.9
        else:
            center_lat, center_lon = 31.5, 76.0

        seeds = [
            ("SA-11", center_lat + 0.25, center_lon + 0.20, 20.0),
            ("SA-15", center_lat - 0.15, center_lon + 0.10, 45.0),
            ("SA-18", center_lat + 0.05, center_lon - 0.18, 90.0),
            ("ZSU-23-4", center_lat - 0.22, center_lon - 0.12, 130.0),
        ]

        if "crisis" in scenario or "heightened" in scenario:
            seeds.extend([
                ("HQ-16", center_lat + 0.4, center_lon - 0.3, 60.0),
                ("SA-6", center_lat - 0.35, center_lon + 0.28, 10.0),
            ])

        return [
            create_threat_from_template(
                designator=designator,
                latitude=lat,
                longitude=lon,
                heading_deg=heading,
            )
            for designator, lat, lon, heading in seeds
        ]

    def query(self, query: EnvironmentQuery) -> EnvironmentResponse:
        """
        Query the environment model.

        Args:
            query: Environment query specification

        Returns:
            EnvironmentResponse with costs and metrics
        """
        import time
        start_time = time.time()

        if query.mode == QueryMode.POINT:
            response = self._query_point(query)
        elif query.mode == QueryMode.PATH:
            response = self._query_path(query)
        elif query.mode == QueryMode.GRID:
            response = self._query_grid(query)
        elif query.mode == QueryMode.BATCH:
            response = self._query_batch(query)
        else:
            raise ValueError(f"Unknown query mode: {query.mode}")

        response.computation_time_ms = (time.time() - start_time) * 1000
        return response

    def _query_point(self, query: EnvironmentQuery) -> EnvironmentResponse:
        """Query a single point."""
        lat = query.latitude
        lon = query.longitude
        alt = query.altitude_m

        # Check cache
        tkey = query.timestamp.replace(minute=0, second=0, microsecond=0).isoformat()
        cache_key = (
            f"{lat:.6f},{lon:.6f},{alt:.1f}|"
            f"{query.velocity_mps:.1f}|{query.heading_deg:.1f}|{query.rcs_sqm:.3f}|{tkey}"
        )
        if self.config.enable_cache and cache_key in self._cache:
            self._cache_hits += 1
            cached = self._cache[cache_key]
            return EnvironmentResponse(
                query_mode=QueryMode.POINT,
                cost=cached,
                cache_hit=True,
                data_quality={
                    'component_confidence': cached.component_confidence,
                    'source_status': cached.source_status,
                },
            )
        self._cache_misses += 1

        # Compute costs
        cost = self._compute_fused_cost(
            lat, lon, alt,
            query.timestamp,
            query.velocity_mps,
            query.heading_deg,
            query.rcs_sqm,
            include_gradients=query.include_gradients,
            include_uncertainty=query.include_uncertainty,
        )

        # Cache result
        if self.config.enable_cache:
            self._cache[cache_key] = cost

        return EnvironmentResponse(
            query_mode=QueryMode.POINT,
            timestamp=query.timestamp,
            cost=cost,
            data_quality={
                'component_confidence': cost.component_confidence,
                'source_status': cost.source_status,
            },
        )

    def _query_path(self, query: EnvironmentQuery) -> EnvironmentResponse:
        """Query a path/trajectory."""
        waypoints = query.waypoints
        path_costs = []
        cumulative_survival = 1.0
        total_cost = 0.0
        all_feasible = True

        for _i, (lat, lon, alt) in enumerate(waypoints):
            cost = self._compute_fused_cost(
                lat, lon, alt,
                query.timestamp,
                query.velocity_mps,
                query.heading_deg,
                query.rcs_sqm,
            )
            path_costs.append(cost)
            total_cost += cost.total
            cumulative_survival *= cost.survival_probability
            if not cost.is_feasible:
                all_feasible = False

        # Normalize path cost
        avg_cost = total_cost / len(waypoints) if waypoints else 0.0

        return EnvironmentResponse(
            query_mode=QueryMode.PATH,
            timestamp=query.timestamp,
            path_costs=path_costs,
            total_path_cost=avg_cost,
            cumulative_survival=cumulative_survival,
            path_feasible=all_feasible,
            data_quality={
                'mean_confidence': float(np.mean([c.confidence for c in path_costs])) if path_costs else 1.0,
                'mean_uncertainty': float(np.mean([c.uncertainty for c in path_costs])) if path_costs else 0.0,
            },
        )

    def _query_grid(self, query: EnvironmentQuery) -> EnvironmentResponse:
        """Query a 3D grid region."""
        # Calculate grid dimensions
        lat_range = query.max_lat - query.min_lat
        lon_range = query.max_lon - query.min_lon
        res_deg = query.resolution_m / METERS_PER_DEG_LAT

        n_lat = max(1, int(lat_range / res_deg))
        n_lon = max(1, int(lon_range / res_deg))
        n_alt = len(self.config.altitude_layers)

        n_points = n_lat * n_lon * n_alt
        if n_points > self.config.max_grid_points:
            scale = (n_points / self.config.max_grid_points) ** (1 / 3)
            n_lat = max(1, int(n_lat / scale))
            n_lon = max(1, int(n_lon / scale))
            logger.warning(
                "Environment grid request auto-coarsened to %dx%dx%d to satisfy max_grid_points",
                n_lat,
                n_lon,
                n_alt,
            )

        # Create planner-ready grids with explainability layers.
        grid = np.zeros((n_lat, n_lon, n_alt))
        confidence_grid = np.ones((n_lat, n_lon, n_alt))
        uncertainty_grid = np.zeros((n_lat, n_lon, n_alt))
        threat_grid = np.zeros((n_lat, n_lon, n_alt))
        obstacle_grid = np.zeros((n_lat, n_lon, n_alt))

        for i, lat in enumerate(np.linspace(query.min_lat, query.max_lat, n_lat)):
            for j, lon in enumerate(np.linspace(query.min_lon, query.max_lon, n_lon)):
                for k, alt in enumerate(self.config.altitude_layers):
                    if query.min_alt <= alt <= query.max_alt:
                        cost = self._compute_fused_cost(
                            lat, lon, alt,
                            query.timestamp,
                            query.velocity_mps,
                            query.heading_deg,
                            query.rcs_sqm,
                            include_uncertainty=True,
                        )
                        grid[i, j, k] = cost.total
                        confidence_grid[i, j, k] = cost.confidence
                        uncertainty_grid[i, j, k] = cost.uncertainty
                        threat_grid[i, j, k] = cost.threat
                        obstacle_grid[i, j, k] = cost.obstacle

        return EnvironmentResponse(
            query_mode=QueryMode.GRID,
            timestamp=query.timestamp,
            grid_costs=grid,
            grid_shape=(n_lat, n_lon, n_alt),
            grid_bounds={
                'min_lat': query.min_lat,
                'max_lat': query.max_lat,
                'min_lon': query.min_lon,
                'max_lon': query.max_lon,
                'min_alt': query.min_alt,
                'max_alt': query.max_alt,
            },
            grid_layers={
                'confidence': confidence_grid,
                'uncertainty': uncertainty_grid,
                'threat_cost': threat_grid,
                'obstacle_cost': obstacle_grid,
            },
            data_quality={
                'mean_confidence': float(np.mean(confidence_grid)),
                'mean_uncertainty': float(np.mean(uncertainty_grid)),
                'threat_cost_mean': float(np.mean(threat_grid)),
                'obstacle_cost_mean': float(np.mean(obstacle_grid)),
            },
        )

    def _query_batch(self, query: EnvironmentQuery) -> EnvironmentResponse:
        """Query a batch of points."""
        points = query.points
        batch_costs = []

        if points is None:
            return EnvironmentResponse(
                query_mode=QueryMode.BATCH,
                timestamp=query.timestamp,
                batch_costs=[],
            )

        chunk = max(1, self.config.batch_chunk_size)
        for start in range(0, len(points), chunk):
            stop = min(start + chunk, len(points))
            for i in range(start, stop):
                lat, lon, alt = points[i, 0], points[i, 1], points[i, 2]
                cost = self._compute_fused_cost(
                    lat,
                    lon,
                    alt,
                    query.timestamp,
                    query.velocity_mps,
                    query.heading_deg,
                    query.rcs_sqm,
                )
                batch_costs.append(cost)

        return EnvironmentResponse(
            query_mode=QueryMode.BATCH,
            timestamp=query.timestamp,
            batch_costs=batch_costs,
            data_quality={
                'mean_confidence': float(np.mean([c.confidence for c in batch_costs])) if batch_costs else 1.0,
                'mean_uncertainty': float(np.mean([c.uncertainty for c in batch_costs])) if batch_costs else 0.0,
            },
        )

    def _wind_cache_key(self, bounds: tuple[float, float, float, float], timestamp: datetime) -> str:
        north, south, east, west = bounds
        tkey = timestamp.replace(minute=0, second=0, microsecond=0).isoformat()
        return f"{north:.3f}:{south:.3f}:{east:.3f}:{west:.3f}:{tkey}"

    def _is_circuit_open(self, open_until: datetime | None, now: datetime) -> bool:
        return open_until is not None and open_until > now

    def _record_wind_result(self, success: bool, now: datetime) -> None:
        if success:
            self._wind_failure_count = 0
            self._wind_circuit_open_until = None
            return
        self._wind_failure_count += 1
        if self._wind_failure_count >= self.config.wind_circuit_breaker_threshold:
            self._wind_circuit_open_until = now + timedelta(seconds=self.config.wind_circuit_breaker_cooldown_s)

    def _record_obstacle_result(self, success: bool, now: datetime) -> None:
        if success:
            self._obstacle_failure_count = 0
            self._obstacle_circuit_open_until = None
            return
        self._obstacle_failure_count += 1
        if self._obstacle_failure_count >= self.config.obstacle_circuit_breaker_threshold:
            self._obstacle_circuit_open_until = now + timedelta(seconds=self.config.obstacle_circuit_breaker_cooldown_s)

    def _get_or_fetch_wind_forecast(
        self,
        lat: float,
        lon: float,
        timestamp: datetime,
    ) -> Any | None:
        if self._wind_model is None:
            return None

        margin = max(0.05, self.config.local_api_margin_deg)
        bounds = (lat + margin, lat - margin, lon + margin, lon - margin)
        cache_key = self._wind_cache_key(bounds, timestamp)
        if cache_key in self._wind_forecast_cache:
            return self._wind_forecast_cache[cache_key]

        now = datetime.now(timezone.utc)
        if self._is_circuit_open(self._wind_circuit_open_until, now):
            logger.warning(
                "Wind provider circuit open until %s, using fallback wind cost",
                self._wind_circuit_open_until,
            )
            return None

        for attempt in range(max(1, self.config.wind_max_retries + 1)):
            try:
                forecast = self._wind_model.fetch_forecast(
                    bounds=bounds,
                    altitude_bands=self.config.altitude_layers,
                    forecast_hours=self.config.wind_fetch_hours,
                )
                self._wind_forecast_cache[cache_key] = forecast
                self._record_wind_result(success=True, now=now)
                return forecast
            except (RuntimeError, ValueError, AttributeError, TimeoutError, ConnectionError) as exc:
                logger.warning(
                    "Wind forecast fetch failed at (%.4f, %.4f), attempt %d/%d: %s",
                    lat,
                    lon,
                    attempt + 1,
                    max(1, self.config.wind_max_retries + 1),
                    exc,
                )
            except Exception as exc:
                logger.warning(
                    "Unexpected wind provider error at (%.4f, %.4f), attempt %d/%d: %s",
                    lat,
                    lon,
                    attempt + 1,
                    max(1, self.config.wind_max_retries + 1),
                    exc,
                )

        self._record_wind_result(success=False, now=now)
        return None

    def _refresh_obstacles(self, lat: float, lon: float, timestamp: datetime) -> None:
        if self._obstacle_model is None and self._dynamic_obstacle_model is None:
            self._obstacle_snapshot = []
            return

        if self._last_obstacle_fetch is not None:
            age = (timestamp - self._last_obstacle_fetch).total_seconds()
            if age < self.config.max_obstacle_query_interval_s:
                return

        now = datetime.now(timezone.utc)
        if self._is_circuit_open(self._obstacle_circuit_open_until, now):
            logger.warning(
                "Obstacle providers circuit open until %s, keeping previous snapshot",
                self._obstacle_circuit_open_until,
            )
            return

        margin = max(0.03, self.config.local_api_margin_deg)
        bounds = (lat + margin, lat - margin, lon + margin, lon - margin)
        combined: list[Any] = []

        had_failure = False

        for attempt in range(max(1, self.config.obstacle_max_retries + 1)):
            combined = []
            had_failure = False

            try:
                if self._obstacle_model is not None:
                    combined.extend(self._obstacle_model.fetch_obstacles(bounds))
            except (RuntimeError, ValueError, AttributeError, TimeoutError, ConnectionError) as exc:
                had_failure = True
                logger.warning(
                    "Static obstacle fetch failed near (%.4f, %.4f), attempt %d/%d: %s",
                    lat,
                    lon,
                    attempt + 1,
                    max(1, self.config.obstacle_max_retries + 1),
                    exc,
                )
            except Exception as exc:
                had_failure = True
                logger.warning(
                    "Unexpected static obstacle provider error near (%.4f, %.4f), attempt %d/%d: %s",
                    lat,
                    lon,
                    attempt + 1,
                    max(1, self.config.obstacle_max_retries + 1),
                    exc,
                )

            try:
                if self._dynamic_obstacle_model is not None:
                    combined.extend(self._dynamic_obstacle_model.fetch_obstacles(bounds))
            except (RuntimeError, ValueError, AttributeError, TimeoutError, ConnectionError) as exc:
                had_failure = True
                logger.warning(
                    "Dynamic obstacle fetch failed near (%.4f, %.4f), attempt %d/%d: %s",
                    lat,
                    lon,
                    attempt + 1,
                    max(1, self.config.obstacle_max_retries + 1),
                    exc,
                )
            except Exception as exc:
                had_failure = True
                logger.warning(
                    "Unexpected dynamic obstacle provider error near (%.4f, %.4f), attempt %d/%d: %s",
                    lat,
                    lon,
                    attempt + 1,
                    max(1, self.config.obstacle_max_retries + 1),
                    exc,
                )

            if not had_failure:
                break

        self._obstacle_snapshot = combined
        self._last_obstacle_fetch = timestamp
        self._record_obstacle_result(success=not had_failure, now=now)

    def _derive_uncertainty(
        self,
        alt: float,
        terrain_conf: float,
        wind_conf: float,
        obstacle_conf: float,
        threat_conf: float,
    ) -> tuple[float, float]:
        """Derive deterministic uncertainty/confidence from component confidences."""
        mean_conf = np.mean([terrain_conf, wind_conf, obstacle_conf, threat_conf])
        altitude_penalty = min(0.12, max(0.0, alt / 12000.0))
        uncertainty = float(np.clip((1.0 - mean_conf) + altitude_penalty, 0.0, 1.0))
        confidence = float(np.clip(1.0 - uncertainty, 0.0, 1.0))
        return uncertainty, confidence

    def _compute_fused_cost(
        self,
        lat: float,
        lon: float,
        alt: float,
        timestamp: datetime,
        velocity: float,
        heading: float,
        rcs: float,
        include_gradients: bool = False,
        include_uncertainty: bool = False,
    ) -> FusedCost:
        """
        Compute fused cost at a single point.

        This is the core fusion algorithm.
        """
        # Get individual costs
        terrain_result = self._get_terrain_cost(lat, lon, alt)
        wind_result = self._get_wind_cost(lat, lon, alt, velocity, heading, timestamp)
        obstacle_result = self._get_obstacle_cost(lat, lon, alt, timestamp)
        threat_result = self._get_threat_cost(lat, lon, alt, rcs)

        terrain_cost = terrain_result['cost']
        wind_cost = wind_result['cost']
        obstacle_cost = obstacle_result['cost']

        threat_cost = threat_result['cost']
        p_detect = threat_result['detection_prob']
        p_engage = threat_result['engagement_prob']
        p_kill = threat_result['kill_prob']
        p_survive = 1.0 - p_kill

        # Energy and time costs (simplified)
        energy_cost = self._estimate_energy_cost(lat, lon, alt, velocity)
        time_cost = 0.1  # Constant for now

        # Check constraints
        is_feasible = True
        violations = []

        if alt < MIN_OPERATIONAL_ALTITUDE_M:
            is_feasible = False
            violations.append(f"Below min altitude: {alt:.0f}m < {MIN_OPERATIONAL_ALTITUDE_M:.0f}m")

        if obstacle_cost > 0.9:
            is_feasible = False
            violations.append("Obstacle collision")

        if p_survive < self.config.weights.min_survival_prob:
            is_feasible = False
            violations.append(f"Survival prob too low: {p_survive:.2%}")

        # Apply fusion
        weights = self.config.weights.normalized

        if self.config.fusion_method == FusionMethod.WEIGHTED_SUM:
            total = (
                weights['terrain'] * terrain_cost +
                weights['wind'] * wind_cost +
                weights['obstacle'] * obstacle_cost +
                weights['threat'] * threat_cost +
                weights['energy'] * energy_cost +
                weights['time'] * time_cost
            )
        elif self.config.fusion_method == FusionMethod.MAX_COST:
            total = max(terrain_cost, wind_cost, obstacle_cost, threat_cost)
        elif self.config.fusion_method == FusionMethod.PROBABILISTIC:
            # Risk-based: total = 1 - survival_probability
            total = 1.0 - p_survive
        else:
            # Default to weighted sum
            total = (
                weights['terrain'] * terrain_cost +
                weights['wind'] * wind_cost +
                weights['obstacle'] * obstacle_cost +
                weights['threat'] * threat_cost +
                weights['energy'] * energy_cost +
                weights['time'] * time_cost
            )

        # Apply normalization
        total = self._normalize_cost(total)

        # Compute gradients if requested
        grad_lat, grad_lon, grad_alt = 0.0, 0.0, 0.0
        if include_gradients:
            grad_lat, grad_lon, grad_alt = self._compute_gradient(
                lat, lon, alt, timestamp, velocity, heading, rcs
            )

        # Compute uncertainty if requested
        uncertainty, confidence = self._derive_uncertainty(
            alt=alt,
            terrain_conf=float(terrain_result['confidence']),
            wind_conf=float(wind_result['confidence']),
            obstacle_conf=float(obstacle_result['confidence']),
            threat_conf=float(threat_result['confidence']),
        )

        if not include_uncertainty:
            uncertainty = 0.0
            confidence = 1.0

        return FusedCost(
            total=total,
            terrain=terrain_cost,
            wind=wind_cost,
            obstacle=obstacle_cost,
            threat=threat_cost,
            energy=energy_cost,
            time=time_cost,
            detection_probability=p_detect,
            engagement_probability=p_engage,
            kill_probability=p_kill,
            survival_probability=p_survive,
            is_feasible=is_feasible,
            constraint_violations=violations,
            gradient_lat=grad_lat,
            gradient_lon=grad_lon,
            gradient_alt=grad_alt,
            uncertainty=uncertainty,
            confidence=confidence,
            component_confidence={
                'terrain': float(terrain_result['confidence']),
                'wind': float(wind_result['confidence']),
                'obstacle': float(obstacle_result['confidence']),
                'threat': float(threat_result['confidence']),
            },
            source_status={
                'terrain': str(terrain_result['source']),
                'wind': str(wind_result['source']),
                'obstacle': str(obstacle_result['source']),
                'threat': str(threat_result['source']),
            },
        )

    def _get_terrain_cost(self, lat: float, lon: float, alt: float) -> dict[str, Any]:
        """Get terrain cost at position."""
        if self._terrain_model is None:
            # Conservative fallback when terrain API is unavailable.
            clearance = alt - 150.0
            if clearance < 50:
                return {'cost': 1.0, 'confidence': 0.55, 'source': 'fallback'}
            if clearance < 100:
                return {'cost': 0.8, 'confidence': 0.55, 'source': 'fallback'}
            if clearance < 200:
                return {'cost': 0.4, 'confidence': 0.55, 'source': 'fallback'}
            return {'cost': 0.1, 'confidence': 0.55, 'source': 'fallback'}

        try:
            point = self._terrain_model.get_elevation(lat, lon)
            terrain_elevation = float(point.elevation_m)
            clearance = alt - terrain_elevation
            slope_factor = 0.0
            source = 'provider'
            confidence = 0.95
        except (RuntimeError, ValueError, AttributeError) as exc:
            logger.warning("Terrain query failed at (%.4f, %.4f): %s", lat, lon, exc)
            clearance = alt - 150.0
            slope_factor = 0.1
            source = 'fallback_after_provider_error'
            confidence = 0.6

        if clearance < 50:
            clearance_cost = 1.0
        elif clearance < 100:
            clearance_cost = 0.8
        elif clearance < 200:
            clearance_cost = 0.4
        else:
            clearance_cost = 0.1

        return {
            'cost': 0.3 * slope_factor + 0.7 * clearance_cost,
            'confidence': confidence,
            'source': source,
        }

    def _get_wind_cost(
        self, lat: float, lon: float, alt: float,
        velocity: float, heading: float, timestamp: datetime
    ) -> dict[str, Any]:
        """Get wind cost at position."""
        forecast = self._get_or_fetch_wind_forecast(lat, lon, timestamp)

        if forecast is None:
            # Conservative fallback if no forecast is available.
            wind_speed = 8.0 + max(0.0, alt / 1200.0)
            wind_dir = (heading + 45.0) % 360.0
            source = 'fallback'
            confidence = 0.6
        else:
            if not forecast.timestamps:
                return {'cost': 0.2, 'confidence': 0.55, 'source': 'provider_no_timestamps'}
            def _align_timestamp(t: datetime) -> datetime:
                if t.tzinfo is None and timestamp.tzinfo is not None:
                    return t.replace(tzinfo=timestamp.tzinfo)
                if t.tzinfo is not None and timestamp.tzinfo is None:
                    return t.replace(tzinfo=None)
                return t

            t_idx = int(np.argmin([abs((_align_timestamp(t) - timestamp).total_seconds()) for t in forecast.timestamps]))
            l_idx = forecast.get_level_index(alt)
            lat_idx = int(np.argmin(np.abs(forecast.latitudes - lat)))
            lon_idx = int(np.argmin(np.abs(forecast.longitudes - lon)))

            u = float(forecast.wind_u[t_idx, l_idx, lat_idx, lon_idx])
            v = float(forecast.wind_v[t_idx, l_idx, lat_idx, lon_idx])
            wind_speed = math.sqrt(u * u + v * v)
            wind_dir = (math.degrees(math.atan2(-u, -v)) + 360.0) % 360.0
            source = 'provider'
            confidence = 0.92

        # Headwind component
        angle_diff = abs(wind_dir - heading)
        if angle_diff > 180:
            angle_diff = 360 - angle_diff

        headwind = wind_speed * math.cos(math.radians(angle_diff))
        crosswind = abs(wind_speed * math.sin(math.radians(angle_diff)))

        # Cost based on wind impact
        if headwind > 0:  # Headwind
            headwind_cost = min(1.0, headwind / 30.0)
        else:  # Tailwind
            headwind_cost = 0.0

        crosswind_cost = min(1.0, crosswind / 20.0)

        # Wind sensitivity scales with nominal mission speed.
        velocity_factor = min(1.0, max(0.2, velocity / 45.0))

        # Mild turbulence proxy from wind magnitude.
        turbulence = min(0.35, 0.05 + 0.01 * wind_speed)

        return {
            'cost': velocity_factor * (0.4 * headwind_cost + 0.4 * crosswind_cost + 0.2 * turbulence),
            'confidence': confidence,
            'source': source,
        }

    def _get_obstacle_cost(self, lat: float, lon: float, alt: float, timestamp: datetime) -> dict[str, Any]:
        """Get obstacle cost at position."""
        self._refresh_obstacles(lat, lon, timestamp)
        obstacles = self._obstacle_snapshot

        if not obstacles:
            return {'cost': 0.05, 'confidence': 0.5, 'source': 'no_obstacles_available'}

        min_horizontal = float("inf")
        min_vertical = float("inf")

        for obs in obstacles:
            if obs.state is None:
                continue
            obs_lat, obs_lon, obs_alt = obs.state.position_lla
            horizontal_m = self._haversine_km(lat, lon, obs_lat, obs_lon) * 1000.0
            vertical_m = abs(alt - obs_alt)
            if horizontal_m < min_horizontal:
                min_horizontal = horizontal_m
            if vertical_m < min_vertical:
                min_vertical = vertical_m

        if not np.isfinite(min_horizontal):
            return {'cost': 0.05, 'confidence': 0.5, 'source': 'obstacles_without_state'}

        horizontal_cost = max(0.0, 1.0 - min_horizontal / 500.0)
        vertical_cost = max(0.0, 1.0 - min_vertical / 120.0)
        dynamic_count = 0
        freshness_scores = []
        uncertainty_scores = []
        for obs in obstacles:
            if getattr(obs, 'mobility', None) is not None and str(obs.mobility).endswith('DYNAMIC'):
                dynamic_count += 1
            if getattr(obs, 'state', None) is not None:
                state_ts = getattr(obs.state, 'timestamp', None)
                if isinstance(state_ts, datetime):
                    age_s = max(0.0, (timestamp.replace(tzinfo=None) - state_ts.replace(tzinfo=None)).total_seconds())
                    freshness_scores.append(max(0.0, min(1.0, 1.0 - age_s / 300.0)))
                uncertainty_m = float(getattr(obs.state, 'position_uncertainty_m', 20.0))
                uncertainty_scores.append(max(0.0, min(1.0, uncertainty_m / 100.0)))

        freshness = float(np.mean(freshness_scores)) if freshness_scores else 0.7
        mean_uncertainty = float(np.mean(uncertainty_scores)) if uncertainty_scores else 0.2
        dynamic_bonus = 0.1 if dynamic_count > 0 else 0.0
        confidence = float(np.clip(0.55 + 0.35 * freshness - 0.25 * mean_uncertainty + dynamic_bonus, 0.35, 0.98))

        source = 'static+dynamic_provider' if dynamic_count > 0 else 'static_provider'
        return {
            'cost': float(min(1.0, 0.7 * horizontal_cost + 0.3 * vertical_cost)),
            'confidence': confidence,
            'source': source,
        }

    def _get_threat_cost(
        self, lat: float, lon: float, alt: float, rcs: float
    ) -> dict[str, Any]:
        """Get threat cost at position."""
        if self._threat_model is not None and hasattr(self._threat_model, 'get_aggregated_risk'):
            try:
                agg = self._threat_model.get_aggregated_risk(
                    latitude=lat,
                    longitude=lon,
                    altitude_m=alt,
                    platform_rcs_dbsm=10.0 * math.log10(max(rcs, 1e-6)),
                )
                p_detect = float(agg.detection_probability)
                p_engage = float(agg.engagement_probability)
                p_kill = float(agg.kill_probability)
                source = 'threat_aggregator'
                confidence = 0.92
            except (RuntimeError, ValueError, AttributeError, TypeError) as exc:
                logger.warning("Threat aggregator failed, using fallback model: %s", exc)
                p_detect = 0.0
                p_engage = 0.0
                p_kill = 0.0
                source = 'fallback_after_aggregator_error'
                confidence = 0.55
        else:
            p_detect = 0.0
            p_engage = 0.0
            p_kill = 0.0
            source = 'fallback'
            confidence = 0.55

        nearest_km = float('inf')
        if self._threat_systems:
            nearest_km = min(
                self._haversine_km(lat, lon, t.latitude, t.longitude)
                for t in self._threat_systems
            )

        if nearest_km > self.config.threat_min_coverage_radius_km:
            # Apply conservative floor where modeled threat coverage is sparse.
            p_kill = max(p_kill, self.config.threat_unknown_penalty)
            p_engage = max(p_engage, 0.75 * self.config.threat_unknown_penalty)
            p_detect = max(p_detect, 0.5 * self.config.threat_unknown_penalty)
            confidence = min(confidence, 0.65)
            source = f"{source}+coverage_penalty"

        threat_cost = 0.3 * p_detect + 0.3 * p_engage + 0.4 * p_kill

        return {
            'cost': float(threat_cost),
            'detection_prob': float(p_detect),
            'engagement_prob': float(p_engage),
            'kill_prob': float(p_kill),
            'confidence': float(confidence),
            'source': source,
        }

    def _estimate_energy_cost(
        self, _lat: float, _lon: float, alt: float, velocity: float
    ) -> float:
        """Estimate energy consumption cost."""
        # Simplified energy model
        # Higher altitude = more power for climb
        # Higher velocity = more power

        alt_factor = min(1.0, alt / 3000)
        vel_factor = min(1.0, velocity / 50)

        return 0.5 * alt_factor + 0.5 * vel_factor

    def _normalize_cost(self, cost: float) -> float:
        """Normalize cost to [0, 1] range."""
        if self.config.normalization == NormalizationMethod.LINEAR:
            return max(0.0, min(1.0, cost))
        elif self.config.normalization == NormalizationMethod.SIGMOID:
            # Sigmoid: 0.5 at cost=0.5, asymptotes at 0 and 1
            return 1.0 / (1.0 + math.exp(-10 * (cost - 0.5)))
        elif self.config.normalization == NormalizationMethod.EXPONENTIAL:
            return 1.0 - math.exp(-3 * cost)
        else:
            return max(0.0, min(1.0, cost))

    def _compute_gradient(
        self, lat: float, lon: float, alt: float,
        timestamp: datetime, velocity: float, heading: float, rcs: float
    ) -> tuple[float, float, float]:
        """Compute cost gradient using finite differences."""
        delta_lat = 0.001  # ~111m
        delta_lon = 0.001
        delta_alt = 10.0   # 10m

        cost_center = self._compute_fused_cost(
            lat, lon, alt, timestamp, velocity, heading, rcs
        ).total

        # Latitude gradient
        cost_lat_plus = self._compute_fused_cost(
            lat + delta_lat, lon, alt, timestamp, velocity, heading, rcs
        ).total
        grad_lat = (cost_lat_plus - cost_center) / delta_lat

        # Longitude gradient
        cost_lon_plus = self._compute_fused_cost(
            lat, lon + delta_lon, alt, timestamp, velocity, heading, rcs
        ).total
        grad_lon = (cost_lon_plus - cost_center) / delta_lon

        # Altitude gradient
        cost_alt_plus = self._compute_fused_cost(
            lat, lon, alt + delta_alt, timestamp, velocity, heading, rcs
        ).total
        grad_alt = (cost_alt_plus - cost_center) / delta_alt

        return grad_lat, grad_lon, grad_alt

    def _estimate_uncertainty(
        self, lat: float, lon: float, alt: float
    ) -> float:
        """Estimate uncertainty in cost estimate."""
        altitude_factor = min(0.12, max(0.0, alt / 10000.0))
        distance_factor = min(0.2, abs(lat) * 0.001 + abs(lon) * 0.001)
        return float(np.clip(0.08 + altitude_factor + distance_factor, 0.0, 1.0))

    def _haversine_km(
        self, lat1: float, lon1: float, lat2: float, lon2: float
    ) -> float:
        """Calculate distance between two points in km."""
        R = 6371  # Earth radius in km

        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lat = math.radians(lat2 - lat1)
        delta_lon = math.radians(lon2 - lon1)

        a = (math.sin(delta_lat / 2) ** 2 +
             math.cos(lat1_rad) * math.cos(lat2_rad) *
             math.sin(delta_lon / 2) ** 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        return R * c

    @property
    def state(self) -> EnvironmentState:
        """Get current environment state."""
        return self._state

    def update_state(self, new_state: EnvironmentState):
        """Update environment state."""
        self._state = new_state
        # Clear cache on state update
        self._cache.clear()
        logger.info("Environment state updated at %s", new_state.timestamp)

    def get_statistics(self) -> dict[str, Any]:
        """Get model statistics."""
        total_queries = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total_queries if total_queries > 0 else 0.0

        return {
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'cache_hit_rate': hit_rate,
            'cache_size': len(self._cache),
            'provider_resilience': {
                'wind_failures': self._wind_failure_count,
                'wind_circuit_open_until': self._wind_circuit_open_until.isoformat() if self._wind_circuit_open_until else None,
                'obstacle_failures': self._obstacle_failure_count,
                'obstacle_circuit_open_until': self._obstacle_circuit_open_until.isoformat() if self._obstacle_circuit_open_until else None,
            },
            'config': {
                'fusion_method': self.config.fusion_method.name,
                'normalization': self.config.normalization.name,
            },
        }

    def build_planning_cost_grid(
        self,
        bounds: tuple[float, float, float, float],
        altitude_layers: list[float] | None = None,
        resolution_m: float | None = None,
        timestamp: datetime | None = None,
    ) -> dict[str, Any]:
        """
        Build planner-ready 3D cost tensor and metadata.

        Returns a dictionary that can be passed directly to planning
        algorithms requiring grid-based cost fields.
        """
        altitude_layers = altitude_layers or self.config.altitude_layers
        resolution_m = resolution_m or self.config.default_grid_resolution_m
        timestamp = timestamp or datetime.now(timezone.utc)

        query = EnvironmentQuery(
            mode=QueryMode.GRID,
            min_lat=bounds[1],
            max_lat=bounds[0],
            min_lon=bounds[3],
            max_lon=bounds[2],
            min_alt=min(altitude_layers),
            max_alt=max(altitude_layers),
            resolution_m=resolution_m,
            timestamp=timestamp,
            include_uncertainty=True,
        )
        response = self.query(query)

        return {
            "cost_tensor": response.grid_costs,
            "grid_shape": response.grid_shape,
            "grid_bounds": response.grid_bounds,
            "altitude_layers": altitude_layers,
            "resolution_m": resolution_m,
            "timestamp": timestamp.isoformat(),
            "grid_layers": response.grid_layers,
            "data_quality": response.data_quality,
        }

    def export_perception_snapshot(
        self,
        bounds: tuple[float, float, float, float],
        altitude_layers: list[float] | None = None,
        resolution_m: float | None = None,
        output_dir: str | Path = "outputs/perception/canonical",
        timestamp: datetime | None = None,
    ) -> dict[str, str]:
        """Export canonical perception outputs directly from model APIs."""
        altitude_layers = altitude_layers or self.config.altitude_layers
        timestamp = timestamp or datetime.now(timezone.utc)

        handoff = self.build_planning_cost_grid(
            bounds=bounds,
            altitude_layers=altitude_layers,
            resolution_m=resolution_m,
            timestamp=timestamp,
        )

        # Refresh local obstacle snapshot for export metadata.
        center_lat = (bounds[0] + bounds[1]) / 2.0
        center_lon = (bounds[2] + bounds[3]) / 2.0
        self._refresh_obstacles(center_lat, center_lon, timestamp)

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        fusion_file = out / "fusion_planning_handoff.json"
        threat_file = out / "threat_model_snapshot.json"
        obstacle_file = out / "obstacle_model_snapshot.json"

        layers_summary = {}
        if handoff.get("grid_layers"):
            for key, value in handoff["grid_layers"].items():
                layers_summary[key] = {
                    "min": float(np.min(value)),
                    "max": float(np.max(value)),
                    "mean": float(np.mean(value)),
                }

        with open(fusion_file, 'w', encoding='utf-8') as f:
            json.dump(
                {
                    "timestamp": timestamp.isoformat(),
                    "grid_shape": list(handoff["grid_shape"]),
                    "grid_bounds": handoff["grid_bounds"],
                    "altitude_layers": handoff["altitude_layers"],
                    "resolution_m": handoff["resolution_m"],
                    "cost_tensor_stats": {
                        "min": float(np.min(handoff["cost_tensor"])),
                        "max": float(np.max(handoff["cost_tensor"])),
                        "mean": float(np.mean(handoff["cost_tensor"])),
                    },
                    "layer_stats": layers_summary,
                    "data_quality": handoff.get("data_quality", {}),
                    "provenance": "EnvironmentModel.build_planning_cost_grid",
                },
                f,
                indent=2,
            )

        with open(threat_file, 'w', encoding='utf-8') as f:
            json.dump(
                {
                    "timestamp": timestamp.isoformat(),
                    "scenario": self.config.threat_scenario,
                    "threat_count": len(self._threat_systems),
                    "threat_ids": [t.threat_id for t in self._threat_systems],
                    "coverage_radius_km": self.config.threat_min_coverage_radius_km,
                    "unknown_area_penalty": self.config.threat_unknown_penalty,
                    "provenance": "ThreatAggregator.get_aggregated_risk",
                },
                f,
                indent=2,
            )

        with open(obstacle_file, 'w', encoding='utf-8') as f:
            json.dump(
                {
                    "timestamp": timestamp.isoformat(),
                    "obstacle_count": len(self._obstacle_snapshot),
                    "dynamic_obstacles": int(
                        sum(
                            1
                            for o in self._obstacle_snapshot
                            if getattr(o, 'mobility', None) is not None
                            and str(o.mobility).endswith('DYNAMIC')
                        )
                    ),
                    "sources": sorted({str(getattr(o, 'source', 'UNKNOWN')) for o in self._obstacle_snapshot}),
                    "provenance": "Obstacle providers (OSM/OpenSky)",
                },
                f,
                indent=2,
            )

        return {
            "fusion": str(fusion_file),
            "threat": str(threat_file),
            "obstacle": str(obstacle_file),
        }


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_environment_model(
    theater: str = "combined",
    scenario: str = "heightened",
    weights: CostWeights | None = None,
) -> EnvironmentModel:
    """
    Create an environment model for a specific theater and scenario.

    Args:
        theater: Theater region ("western", "northern", "combined")
        scenario: Scenario intensity ("peacetime", "heightened", "crisis")
        weights: Optional cost weights

    Returns:
        Configured EnvironmentModel
    """
    config = EnvironmentConfig(
        weights=weights or CostWeights.defense_priority(),
        threat_scenario=f"{theater}_{scenario}",
    )

    return EnvironmentModel(config)
