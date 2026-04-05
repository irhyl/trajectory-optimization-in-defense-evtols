"""
Cost Functions for Defense eVTOL Trajectory Planning.

This module provides a comprehensive library of cost functions that map
environmental conditions to traversability costs for path planning.

Mathematical Framework:

    Individual Cost Functions:

        C_terrain(x) = f(elevation, slope, roughness)
        C_wind(x) = g(windspeed, direction, turbulence)
        C_obstacle(x) = h(distance, velocity, type)
        C_threat(x) = ψ(P_d, P_e, P_k)

    Composite Cost:

        C_total(x) = Σᵢ wᵢ · Cᵢ(x)

    Risk-Aware Cost (CVaR):

        CVaR_α(C) = E[C | C ≥ VaR_α(C)]

    Survival-Based Cost:

        C_survival(path) = 1 - ∏ⱼ (1 - P_kill(xⱼ))

Design Principles:
    1. All costs normalized to [0, 1] range
    2. Composable via weighted sum or product
    3. Differentiable for gradient-based optimization
    4. Support for uncertainty quantification

Author: Defense eVTOL Research Team
"""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Protocol
)
import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

# Physical constants
EARTH_RADIUS_M = 6371000.0
G = 9.81  # Gravitational acceleration (m/s²)
AIR_DENSITY_SL = 1.225  # kg/m³ at sea level

# Cost function parameters
MAX_SLOPE_DEG = 45.0
MAX_WIND_SPEED_MPS = 30.0
MAX_TURBULENCE = 5.0
SAFE_OBSTACLE_DISTANCE_M = 100.0
MAX_THREAT_RANGE_KM = 200.0

# eVTOL parameters
DEFAULT_CRUISE_VELOCITY_MPS = 30.0
DEFAULT_RCS_SQM = 0.5


# =============================================================================
# ENUMERATIONS
# =============================================================================

class CostType(Enum):
    """Types of cost functions."""

    TERRAIN = auto()
    WIND = auto()
    OBSTACLE = auto()
    THREAT = auto()
    ENERGY = auto()
    TIME = auto()
    NOISE = auto()
    COMPOSITE = auto()


class GradientMethod(Enum):
    """Methods for gradient computation."""

    ANALYTICAL = auto()      # Closed-form gradient
    FINITE_DIFF = auto()     # Finite difference approximation
    AUTODIFF = auto()        # Automatic differentiation


# =============================================================================
# PROTOCOLS
# =============================================================================

class CostFunctionProtocol(Protocol):
    """Protocol for cost functions."""

    def compute(
        self,
        lat: float,
        lon: float,
        alt: float,
        **kwargs
    ) -> float:
        """Compute cost at a position."""
        ...

    def gradient(
        self,
        lat: float,
        lon: float,
        alt: float,
        **kwargs
    ) -> NDArray[np.float64]:
        """Compute gradient at a position."""
        ...


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class CostResult:
    """Result of a cost function evaluation."""

    cost: float                                  # Primary cost value [0, 1]
    gradient: NDArray[np.float64] | None = None  # Gradient (lat, lon, alt)
    hessian: NDArray[np.float64] | None = None   # Hessian matrix
    uncertainty: float = 0.0                     # Uncertainty estimate
    components: dict[str, float] = field(default_factory=dict)  # Breakdown
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TerrainParams:
    """Parameters for terrain cost function."""

    slope_weight: float = 0.4          # Weight for slope component
    elevation_weight: float = 0.3      # Weight for elevation/clearance
    roughness_weight: float = 0.3      # Weight for terrain roughness

    max_slope_deg: float = 45.0        # Maximum traversable slope
    min_clearance_m: float = 50.0      # Minimum terrain clearance
    preferred_clearance_m: float = 200.0  # Preferred clearance

    # Terrain type penalties (0 = no penalty, 1 = impassable)
    water_penalty: float = 0.2
    forest_penalty: float = 0.1
    urban_penalty: float = 0.3
    mountain_penalty: float = 0.4


@dataclass
class WindParams:
    """Parameters for wind cost function."""

    headwind_weight: float = 0.4       # Weight for headwind component
    crosswind_weight: float = 0.3      # Weight for crosswind
    turbulence_weight: float = 0.3     # Weight for turbulence

    max_headwind_mps: float = 25.0     # Maximum penetrable headwind
    max_crosswind_mps: float = 15.0    # Maximum safe crosswind
    max_turbulence: float = 5.0        # Maximum turbulence intensity

    # Energy impact factor
    headwind_energy_factor: float = 0.1  # Extra energy per m/s headwind


@dataclass
class ObstacleParams:
    """Parameters for obstacle cost function."""

    collision_distance_m: float = 10.0      # Hard collision threshold
    danger_distance_m: float = 50.0         # Danger zone
    safe_distance_m: float = 200.0          # Safe clearance

    # Cost function shape
    decay_rate: float = 0.02                # Exponential decay rate

    # Dynamic obstacle parameters
    prediction_horizon_s: float = 10.0      # Future prediction window
    uncertainty_growth_rate: float = 0.1    # Uncertainty over time


@dataclass
class ThreatParams:
    """Parameters for threat cost function."""

    detection_weight: float = 0.3      # Weight for detection probability
    engagement_weight: float = 0.3     # Weight for engagement probability
    kill_weight: float = 0.4           # Weight for kill probability

    # Survival threshold
    min_survival_prob: float = 0.7     # Minimum acceptable survival

    # Risk sensitivity (0 = neutral, 1 = very risk-averse)
    risk_sensitivity: float = 0.7

    # RCS factor
    rcs_reference_sqm: float = 1.0     # Reference RCS

    # Terrain masking benefit
    terrain_masking_benefit: float = 0.3


# =============================================================================
# BASE COST FUNCTION
# =============================================================================

class CostFunction(ABC):
    """
    Abstract base class for all cost functions.

    Cost functions map positions (and optionally velocities, time)
    to scalar cost values in the range [0, 1].
    """

    def __init__(
        self,
        name: str,
        cost_type: CostType,
        gradient_method: GradientMethod = GradientMethod.FINITE_DIFF,
    ):
        self.name = name
        self.cost_type = cost_type
        self.gradient_method = gradient_method
        self._call_count = 0

    @abstractmethod
    def compute(
        self,
        lat: float,
        lon: float,
        alt: float,
        **kwargs
    ) -> float:
        """
        Compute cost at a position.

        Args:
            lat: Latitude (degrees)
            lon: Longitude (degrees)
            alt: Altitude (meters MSL)
            **kwargs: Additional parameters

        Returns:
            Cost value in [0, 1]
        """
        pass

    def gradient(
        self,
        lat: float,
        lon: float,
        alt: float,
        delta: float = 1e-5,
        **kwargs
    ) -> NDArray[np.float64]:
        """
        Compute gradient of cost function.

        Args:
            lat, lon, alt: Position
            delta: Step size for finite differences
            **kwargs: Additional parameters

        Returns:
            Gradient array [dC/dlat, dC/dlon, dC/dalt]
        """
        if self.gradient_method == GradientMethod.ANALYTICAL:
            return self._analytical_gradient(lat, lon, alt, **kwargs)
        else:
            return self._finite_diff_gradient(lat, lon, alt, delta, **kwargs)

    def _finite_diff_gradient(
        self,
        lat: float,
        lon: float,
        alt: float,
        delta: float,
        **kwargs
    ) -> NDArray[np.float64]:
        """Compute gradient using central finite differences."""
        grad = np.zeros(3)

        # Latitude gradient
        c_plus = self.compute(lat + delta, lon, alt, **kwargs)
        c_minus = self.compute(lat - delta, lon, alt, **kwargs)
        grad[0] = (c_plus - c_minus) / (2 * delta)

        # Longitude gradient
        c_plus = self.compute(lat, lon + delta, alt, **kwargs)
        c_minus = self.compute(lat, lon - delta, alt, **kwargs)
        grad[1] = (c_plus - c_minus) / (2 * delta)

        # Altitude gradient (different scale)
        alt_delta = delta * 111320  # Convert degrees to meters approx
        c_plus = self.compute(lat, lon, alt + alt_delta, **kwargs)
        c_minus = self.compute(lat, lon, alt - alt_delta, **kwargs)
        grad[2] = (c_plus - c_minus) / (2 * alt_delta)

        return grad

    def _analytical_gradient(
        self,
        lat: float,
        lon: float,
        alt: float,
        **kwargs
    ) -> NDArray[np.float64]:
        """Override in subclasses for analytical gradients; falls back to finite differences."""
        return self._finite_diff_gradient(lat, lon, alt, 1e-5, **kwargs)

    def evaluate(
        self,
        lat: float,
        lon: float,
        alt: float,
        compute_gradient: bool = False,
        compute_hessian: bool = False,
        **kwargs
    ) -> CostResult:
        """
        Full evaluation including gradient and uncertainty.

        Args:
            lat, lon, alt: Position
            compute_gradient: Whether to compute gradient
            compute_hessian: Whether to compute Hessian
            **kwargs: Additional parameters

        Returns:
            CostResult with all requested quantities
        """
        self._call_count += 1

        cost = self.compute(lat, lon, alt, **kwargs)

        gradient = None
        if compute_gradient:
            gradient = self.gradient(lat, lon, alt, **kwargs)

        hessian = None
        if compute_hessian:
            hessian = self._compute_hessian(lat, lon, alt, **kwargs)

        return CostResult(
            cost=cost,
            gradient=gradient,
            hessian=hessian,
            metadata={
                'cost_type': self.cost_type.name,
                'call_count': self._call_count,
            }
        )

    def _compute_hessian(
        self,
        lat: float,
        lon: float,
        alt: float,
        delta: float = 1e-4,
        **kwargs
    ) -> NDArray[np.float64]:
        """Compute Hessian matrix using finite differences."""
        H = np.zeros((3, 3))

        # Second derivatives
        for i, (d1, _name1) in enumerate([(delta, 'lat'), (delta, 'lon'), (delta * 111320, 'alt')]):
            for j, (d2, _name2) in enumerate([(delta, 'lat'), (delta, 'lon'), (delta * 111320, 'alt')]):
                if i <= j:
                    pos = [lat, lon, alt]
                    pos[i] += d1
                    pos[j] += d2
                    c_pp = self.compute(*pos, **kwargs)

                    pos = [lat, lon, alt]
                    pos[i] += d1
                    pos[j] -= d2
                    c_pm = self.compute(*pos, **kwargs)

                    pos = [lat, lon, alt]
                    pos[i] -= d1
                    pos[j] += d2
                    c_mp = self.compute(*pos, **kwargs)

                    pos = [lat, lon, alt]
                    pos[i] -= d1
                    pos[j] -= d2
                    c_mm = self.compute(*pos, **kwargs)

                    H[i, j] = (c_pp - c_pm - c_mp + c_mm) / (4 * d1 * d2)
                    H[j, i] = H[i, j]

        return H


# =============================================================================
# TERRAIN COST FUNCTION
# =============================================================================

class TerrainCostFunction(CostFunction):
    """
    Cost function based on terrain characteristics.

    Considers:
    - Terrain elevation and clearance
    - Terrain slope
    - Surface roughness
    - Terrain type (water, forest, urban, etc.)

    Mathematical Model:
        C_terrain = w_s·σ(slope) + w_e·ε(clearance) + w_r·ρ(roughness)

    Where σ, ε, ρ are normalized sub-costs.
    """

    def __init__(
        self,
        params: TerrainParams | None = None,
        terrain_model: Any | None = None,
    ):
        super().__init__(
            name="terrain_cost",
            cost_type=CostType.TERRAIN,
        )
        self.params = params or TerrainParams()
        self._terrain_model = terrain_model

    def compute(
        self,
        lat: float,
        lon: float,
        alt: float,
        **kwargs
    ) -> float:
        """Compute terrain cost at position."""

        # Get terrain data (simulated if no model)
        terrain_data = self._get_terrain_data(lat, lon)

        elevation = terrain_data['elevation']
        slope = terrain_data['slope']
        roughness = terrain_data['roughness']
        terrain_type = terrain_data.get('terrain_type', 'open')

        # Clearance cost
        clearance = alt - elevation
        if clearance < self.params.min_clearance_m:
            clearance_cost = 1.0  # Critical - too low
        elif clearance < self.params.preferred_clearance_m:
            # Linear interpolation in danger zone
            clearance_cost = 1.0 - (clearance - self.params.min_clearance_m) / (
                self.params.preferred_clearance_m - self.params.min_clearance_m
            )
        else:
            # Above preferred clearance
            clearance_cost = 0.0

        # Slope cost (affects maneuverability near terrain)
        slope_normalized = min(1.0, slope / self.params.max_slope_deg)
        slope_cost = slope_normalized ** 2  # Quadratic penalty

        # Roughness cost
        roughness_cost = min(1.0, roughness)

        # Terrain type penalty
        type_penalties = {
            'water': self.params.water_penalty,
            'forest': self.params.forest_penalty,
            'urban': self.params.urban_penalty,
            'mountain': self.params.mountain_penalty,
            'open': 0.0,
        }
        type_penalty = type_penalties.get(terrain_type, 0.0)

        # Combine components
        total = (
            self.params.elevation_weight * clearance_cost +
            self.params.slope_weight * slope_cost +
            self.params.roughness_weight * roughness_cost +
            0.2 * type_penalty  # Additional penalty
        )

        return min(1.0, max(0.0, total))

    def _get_terrain_data(self, lat: float, lon: float) -> dict[str, Any]:
        """Get terrain data at position."""
        if self._terrain_model is not None:
            # Use actual terrain model
            return self._terrain_model.query(lat, lon)

        # Simulated terrain data
        np.random.seed(int((lat * 1000 + lon * 100) % 10000))

        # Base elevation with spatial variation
        base_elev = 200 + 300 * np.sin(lat * 10) + 200 * np.cos(lon * 10)
        elevation = max(0, base_elev + 50 * np.random.randn())

        # Slope
        slope = 5 + 15 * np.random.random()

        # Roughness
        roughness = 0.2 + 0.3 * np.random.random()

        # Terrain type
        types = ['open', 'open', 'open', 'forest', 'water', 'urban', 'mountain']
        terrain_type = np.random.choice(types)

        return {
            'elevation': elevation,
            'slope': slope,
            'roughness': roughness,
            'terrain_type': terrain_type,
        }


# =============================================================================
# WIND COST FUNCTION
# =============================================================================

class WindCostFunction(CostFunction):
    """
    Cost function based on wind conditions.

    Considers:
    - Headwind/tailwind component (affects energy and speed)
    - Crosswind component (affects stability)
    - Turbulence intensity
    - Gust factor

    Mathematical Model:
        C_wind = w_h·H(v_h) + w_c·X(v_x) + w_t·T(I_t)

    Where:
        H = headwind penalty function
        X = crosswind penalty function
        T = turbulence penalty function
    """

    def __init__(
        self,
        params: WindParams | None = None,
        wind_model: Any | None = None,
    ):
        super().__init__(
            name="wind_cost",
            cost_type=CostType.WIND,
        )
        self.params = params or WindParams()
        self._wind_model = wind_model

    def compute(
        self,
        lat: float,
        lon: float,
        alt: float,
        velocity: float = DEFAULT_CRUISE_VELOCITY_MPS,
        heading: float = 0.0,
        **kwargs
    ) -> float:
        """Compute wind cost at position."""

        # Get wind data
        wind_data = self._get_wind_data(lat, lon, alt)

        wind_speed = wind_data['speed']
        wind_dir = wind_data['direction']
        turbulence = wind_data['turbulence']
        gust = wind_data.get('gust', wind_speed * 1.2)

        # Calculate wind components relative to heading
        angle_diff_rad = math.radians(wind_dir - heading)
        headwind = wind_speed * math.cos(angle_diff_rad)
        crosswind = abs(wind_speed * math.sin(angle_diff_rad))

        # Headwind cost
        if headwind > 0:  # True headwind
            headwind_cost = min(1.0, headwind / self.params.max_headwind_mps)
        else:  # Tailwind (beneficial)
            headwind_cost = 0.0

        # Crosswind cost
        crosswind_cost = min(1.0, crosswind / self.params.max_crosswind_mps)
        crosswind_cost = crosswind_cost ** 1.5  # Super-linear penalty

        # Turbulence cost
        turbulence_cost = min(1.0, turbulence / self.params.max_turbulence)
        turbulence_cost = turbulence_cost ** 2  # Quadratic penalty

        # Gust factor penalty
        gust_factor = gust / wind_speed if wind_speed > 0 else 1.0
        gust_penalty = max(0, (gust_factor - 1.5) / 0.5) * 0.2

        # Combine components
        total = (
            self.params.headwind_weight * headwind_cost +
            self.params.crosswind_weight * crosswind_cost +
            self.params.turbulence_weight * turbulence_cost +
            gust_penalty
        )

        return min(1.0, max(0.0, total))

    def _get_wind_data(
        self, lat: float, lon: float, alt: float
    ) -> dict[str, Any]:
        """Get wind data at position."""
        if self._wind_model is not None:
            return self._wind_model.query(lat, lon, alt)

        # Simulated wind data
        np.random.seed(int((lat * 1000 + lon * 100 + alt) % 10000))

        # Wind speed increases with altitude
        base_speed = 5 + (alt / 1000) * 8
        speed = max(0, base_speed + 3 * np.random.randn())

        # Wind direction (prevailing westerlies in India)
        direction = 270 + 30 * np.random.randn()
        direction = direction % 360

        # Turbulence (higher near ground, in mountains)
        turbulence = 1.0 + 2.0 * np.random.random()
        if alt < 500:
            turbulence *= 1.5

        # Gust
        gust = speed * (1.2 + 0.3 * np.random.random())

        return {
            'speed': speed,
            'direction': direction,
            'turbulence': turbulence,
            'gust': gust,
        }


# =============================================================================
# OBSTACLE COST FUNCTION
# =============================================================================

class ObstacleCostFunction(CostFunction):
    """
    Cost function based on obstacle proximity.

    Considers:
    - Static obstacles (buildings, towers, terrain features)
    - Dynamic obstacles (other aircraft, birds)
    - Predicted future positions

    Mathematical Model:
        C_obstacle = max_i { ψ(d_i) }

    Where:
        ψ(d) = exp(-λ·(d - d_safe)) for d > d_collision
        ψ(d) = 1.0 for d ≤ d_collision
    """

    def __init__(
        self,
        params: ObstacleParams | None = None,
        obstacle_model: Any | None = None,
    ):
        super().__init__(
            name="obstacle_cost",
            cost_type=CostType.OBSTACLE,
        )
        self.params = params or ObstacleParams()
        self._obstacle_model = obstacle_model

    def compute(
        self,
        lat: float,
        lon: float,
        alt: float,
        **kwargs
    ) -> float:
        """Compute obstacle cost at position."""

        # Get nearby obstacles
        obstacles = self._get_obstacles(lat, lon, alt)

        if not obstacles:
            return 0.0

        max_cost = 0.0

        for obstacle in obstacles:
            distance = obstacle['distance']

            if distance <= self.params.collision_distance_m:
                # Collision zone
                cost = 1.0
            elif distance <= self.params.danger_distance_m:
                # Danger zone - high cost
                t = (distance - self.params.collision_distance_m) / (
                    self.params.danger_distance_m - self.params.collision_distance_m
                )
                cost = 0.8 * (1 - t) + 0.3 * t
            elif distance <= self.params.safe_distance_m:
                # Caution zone
                t = (distance - self.params.danger_distance_m) / (
                    self.params.safe_distance_m - self.params.danger_distance_m
                )
                cost = 0.3 * (1 - t)
            else:
                # Safe zone
                excess = distance - self.params.safe_distance_m
                cost = 0.1 * math.exp(-self.params.decay_rate * excess)

            max_cost = max(max_cost, cost)

        return min(1.0, max(0.0, max_cost))

    def _get_obstacles(
        self, lat: float, lon: float, alt: float
    ) -> list[dict[str, Any]]:
        """Get obstacles near position."""
        if self._obstacle_model is not None:
            return self._obstacle_model.query_nearby(lat, lon, alt)

        # Simulated obstacles
        np.random.seed(int((lat * 1000 + lon * 100) % 10000))

        obstacles = []
        n_obstacles = np.random.poisson(2)

        for i in range(n_obstacles):
            distance = 50 + 500 * np.random.random()
            height = 50 + 200 * np.random.random()

            obstacles.append({
                'id': f'obs_{i}',
                'distance': distance,
                'height': height,
                'type': np.random.choice(['tower', 'building', 'terrain']),
            })

        return obstacles


# =============================================================================
# THREAT COST FUNCTION
# =============================================================================

class ThreatCostFunction(CostFunction):
    """
    Cost function based on military threat exposure.

    This is the most complex cost function, incorporating:
    - Detection probability by radar systems
    - Engagement probability (tracking, missile launch)
    - Kill probability (missile hit)
    - Terrain masking effects
    - Electronic warfare considerations

    Mathematical Model:
        C_threat = w_d·P_d + w_e·P_e + w_k·P_k

    Where:
        P_d = detection probability
        P_e = engagement probability (given detection)
        P_k = kill probability (given engagement)

    Risk-adjusted cost using CVaR:
        C_threat_CVaR = E[C | C ≥ VaR_α(C)]
    """

    def __init__(
        self,
        params: ThreatParams | None = None,
        threat_model: Any | None = None,
    ):
        super().__init__(
            name="threat_cost",
            cost_type=CostType.THREAT,
        )
        self.params = params or ThreatParams()
        self._threat_model = threat_model

    def compute(
        self,
        lat: float,
        lon: float,
        alt: float,
        rcs: float = DEFAULT_RCS_SQM,
        velocity: float = DEFAULT_CRUISE_VELOCITY_MPS,
        terrain_masking: bool = True,
        **kwargs
    ) -> float:
        """Compute threat cost at position."""

        # Get threat assessment
        threat_data = self._get_threat_data(lat, lon, alt, rcs)

        p_detect = threat_data['detection_prob']
        p_engage = threat_data['engagement_prob']
        p_kill = threat_data['kill_prob']

        # Apply terrain masking if enabled
        if terrain_masking:
            masking_factor = self._get_terrain_masking(lat, lon, alt)
            p_detect *= (1 - masking_factor * self.params.terrain_masking_benefit)
            p_engage *= (1 - masking_factor * self.params.terrain_masking_benefit * 0.5)

        # Basic threat cost
        base_cost = (
            self.params.detection_weight * p_detect +
            self.params.engagement_weight * p_engage +
            self.params.kill_weight * p_kill
        )

        # Survival probability
        p_survive = 1.0 - p_kill

        # Apply risk sensitivity
        if self.params.risk_sensitivity > 0:
            # Risk-sensitive transformation
            # Higher risk sensitivity → higher penalty for threat
            risk_factor = 1.0 + self.params.risk_sensitivity * (1 - p_survive)
            base_cost *= risk_factor

        # Check survival threshold
        if p_survive < self.params.min_survival_prob:
            base_cost = min(1.0, base_cost + 0.3)  # Additional penalty

        return min(1.0, max(0.0, base_cost))

    def compute_detailed(
        self,
        lat: float,
        lon: float,
        alt: float,
        rcs: float = DEFAULT_RCS_SQM,
        **kwargs
    ) -> dict[str, float]:
        """Compute detailed threat assessment."""
        threat_data = self._get_threat_data(lat, lon, alt, rcs)

        cost = self.compute(lat, lon, alt, rcs=rcs, **kwargs)

        return {
            'cost': cost,
            'detection_prob': threat_data['detection_prob'],
            'engagement_prob': threat_data['engagement_prob'],
            'kill_prob': threat_data['kill_prob'],
            'survival_prob': 1.0 - threat_data['kill_prob'],
            'dominant_threat': threat_data.get('dominant_threat', 'unknown'),
        }

    def _get_threat_data(
        self, lat: float, lon: float, alt: float, rcs: float
    ) -> dict[str, float]:
        """Get threat assessment at position."""
        if self._threat_model is not None:
            return self._threat_model.assess_threat(lat, lon, alt, rcs)

        # Simulated threat assessment
        np.random.seed(int((lat * 1000 + lon * 100 + alt * 10) % 10000))

        # Distance to nearest threat (simulated)
        # Western border threat zones
        if 68.0 <= lon <= 76.0 and 23.0 <= lat <= 35.0:
            base_threat = 0.6  # Higher threat near western border
        elif 76.0 <= lon <= 95.0 and 27.0 <= lat <= 35.0:
            base_threat = 0.5  # Northern border
        else:
            base_threat = 0.1  # Interior

        # Altitude factor (lower = harder to detect)
        if alt < 50:
            alt_factor = 0.3
        elif alt < 100:
            alt_factor = 0.5
        elif alt < 300:
            alt_factor = 0.7
        else:
            alt_factor = 1.0

        # RCS factor
        rcs_factor = (rcs / self.params.rcs_reference_sqm) ** 0.25

        # Calculate probabilities
        p_detect = base_threat * alt_factor * rcs_factor
        p_detect = min(0.95, max(0.0, p_detect + 0.1 * np.random.randn()))

        p_engage = p_detect * (0.6 + 0.2 * np.random.random())
        p_kill = p_engage * (0.5 + 0.2 * np.random.random())

        return {
            'detection_prob': p_detect,
            'engagement_prob': p_engage,
            'kill_prob': p_kill,
            'dominant_threat': 'simulated_sam',
        }

    def _get_terrain_masking(
        self, lat: float, lon: float, alt: float
    ) -> float:
        """Get terrain masking factor (0 = no masking, 1 = full masking)."""
        # Simulated terrain masking
        np.random.seed(int((lat * 1000 + lon * 100) % 10000))

        # Lower altitude = more masking potential
        if alt < 50:
            base_masking = 0.8
        elif alt < 100:
            base_masking = 0.6
        elif alt < 200:
            base_masking = 0.4
        elif alt < 500:
            base_masking = 0.2
        else:
            base_masking = 0.0

        # Random terrain factor
        terrain_factor = 0.5 + 0.5 * np.random.random()

        return base_masking * terrain_factor


# =============================================================================
# ENERGY COST FUNCTION
# =============================================================================

class EnergyCostFunction(CostFunction):
    """
    Cost function based on energy consumption.

    Considers:
    - Altitude (potential energy)
    - Velocity (kinetic energy, drag)
    - Wind (headwind increases consumption)
    - Payload

    Mathematical Model:
        C_energy = P_req / P_max

    Where:
        P_req = P_hover + P_climb + P_cruise + P_wind
    """

    def __init__(
        self,
        max_power_kw: float = 150.0,
        hover_power_kw: float = 80.0,
        efficiency: float = 0.85,
    ):
        super().__init__(
            name="energy_cost",
            cost_type=CostType.ENERGY,
        )
        self.max_power_kw = max_power_kw
        self.hover_power_kw = hover_power_kw
        self.efficiency = efficiency

    def compute(
        self,
        lat: float,
        lon: float,
        alt: float,
        velocity: float = DEFAULT_CRUISE_VELOCITY_MPS,
        climb_rate: float = 0.0,
        headwind: float = 0.0,
        **kwargs
    ) -> float:
        """Compute energy cost at position."""

        # Base hover power (adjusted for altitude)
        air_density = AIR_DENSITY_SL * math.exp(-alt / 8500)
        density_factor = (AIR_DENSITY_SL / air_density) ** 0.5
        hover_power = self.hover_power_kw * density_factor

        # Cruise power (increases with velocity^3 due to drag)
        cruise_power = 10 * (velocity / 30) ** 3

        # Climb power
        mass_kg = 500  # Approximate mass
        climb_power = (mass_kg * G * climb_rate) / (1000 * self.efficiency)

        # Headwind power
        headwind_power = 5 * (headwind / 10) ** 2

        # Total power
        total_power = hover_power + cruise_power + max(0, climb_power) + headwind_power

        # Normalize to cost
        cost = total_power / self.max_power_kw

        return min(1.0, max(0.0, cost))


# =============================================================================
# COMPOSITE COST FUNCTION
# =============================================================================

class CompositeCostFunction(CostFunction):
    """
    Combines multiple cost functions with weights.

    Supports different fusion methods:
    - Weighted sum
    - Weighted product
    - Maximum
    - Pareto (returns vector)
    """

    def __init__(
        self,
        components: list[tuple[CostFunction, float]],
        fusion_method: str = "weighted_sum",
    ):
        """
        Initialize composite cost function.

        Args:
            components: List of (cost_function, weight) tuples
            fusion_method: "weighted_sum", "weighted_product", "max"
        """
        super().__init__(
            name="composite_cost",
            cost_type=CostType.COMPOSITE,
        )
        self.components = components
        self.fusion_method = fusion_method

        # Normalize weights
        total_weight = sum(w for _, w in components)
        self.normalized_weights = [w / total_weight for _, w in components]

    def compute(
        self,
        lat: float,
        lon: float,
        alt: float,
        **kwargs
    ) -> float:
        """Compute composite cost at position."""

        costs = []
        for (func, _), weight in zip(self.components, self.normalized_weights):
            cost = func.compute(lat, lon, alt, **kwargs)
            costs.append((cost, weight))

        if self.fusion_method == "weighted_sum":
            total = sum(c * w for c, w in costs)
        elif self.fusion_method == "weighted_product":
            total = 1.0
            for c, w in costs:
                total *= (c ** w)
        elif self.fusion_method == "max":
            total = max(c for c, _ in costs)
        else:
            total = sum(c * w for c, w in costs)

        return min(1.0, max(0.0, total))

    def compute_components(
        self,
        lat: float,
        lon: float,
        alt: float,
        **kwargs
    ) -> dict[str, float]:
        """Get individual component costs."""
        result = {}
        for func, _weight in self.components:
            cost = func.compute(lat, lon, alt, **kwargs)
            result[func.name] = cost

        result['total'] = self.compute(lat, lon, alt, **kwargs)
        return result

    def gradient(
        self,
        lat: float,
        lon: float,
        alt: float,
        **kwargs
    ) -> NDArray[np.float64]:
        """Compute gradient of composite cost."""
        total_grad = np.zeros(3)

        for (func, _), weight in zip(self.components, self.normalized_weights):
            grad = func.gradient(lat, lon, alt, **kwargs)
            total_grad += weight * grad

        return total_grad


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_defense_cost_function(
    terrain_weight: float = 0.15,
    wind_weight: float = 0.10,
    obstacle_weight: float = 0.20,
    threat_weight: float = 0.40,
    energy_weight: float = 0.15,
) -> CompositeCostFunction:
    """
    Create a composite cost function optimized for defense operations.

    Args:
        terrain_weight: Weight for terrain cost
        wind_weight: Weight for wind cost
        obstacle_weight: Weight for obstacle cost
        threat_weight: Weight for threat cost
        energy_weight: Weight for energy cost

    Returns:
        CompositeCostFunction configured for defense operations
    """
    components = [
        (TerrainCostFunction(), terrain_weight),
        (WindCostFunction(), wind_weight),
        (ObstacleCostFunction(), obstacle_weight),
        (ThreatCostFunction(), threat_weight),
        (EnergyCostFunction(), energy_weight),
    ]

    return CompositeCostFunction(
        components=components,
        fusion_method="weighted_sum",
    )


def create_reconnaissance_cost_function() -> CompositeCostFunction:
    """Create cost function optimized for reconnaissance missions."""
    # Higher emphasis on stealth and threat avoidance
    components = [
        (TerrainCostFunction(), 0.15),
        (WindCostFunction(), 0.05),
        (ObstacleCostFunction(), 0.15),
        (ThreatCostFunction(
            params=ThreatParams(risk_sensitivity=0.9)
        ), 0.55),
        (EnergyCostFunction(), 0.10),
    ]

    return CompositeCostFunction(
        components=components,
        fusion_method="weighted_sum",
    )


def create_logistics_cost_function() -> CompositeCostFunction:
    """Create cost function optimized for logistics/cargo missions."""
    # Balance between efficiency and safety
    components = [
        (TerrainCostFunction(), 0.20),
        (WindCostFunction(), 0.15),
        (ObstacleCostFunction(), 0.25),
        (ThreatCostFunction(
            params=ThreatParams(risk_sensitivity=0.5)
        ), 0.20),
        (EnergyCostFunction(), 0.20),
    ]

    return CompositeCostFunction(
        components=components,
        fusion_method="weighted_sum",
    )
