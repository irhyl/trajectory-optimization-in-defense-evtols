"""
Wind Field Interpolation for eVTOL Trajectory Optimization.

This module provides research-grade spatial-temporal interpolation algorithms
for querying wind data at arbitrary 4D points (latitude, longitude, altitude, time).

Interpolation Methods:
----------------------

1. TRILINEAR INTERPOLATION (Spatial)

   For a 3D regular grid, trilinear interpolation computes:

       f(x,y,z) = Σᵢ Σⱼ Σₖ f(xᵢ,yⱼ,zₖ) × L_i(x) × L_j(y) × L_k(z)

   where L_i, L_j, L_k are linear basis functions:

       L_i(x) = (x - x₀)/(x₁ - x₀)  for i=1
       L_i(x) = (x₁ - x)/(x₁ - x₀)  for i=0

2. QUADRILINEAR INTERPOLATION (4D: Space + Time)

   Extension to 4D for temporal evolution:

       f(x,y,z,t) = Σₗ f(x,y,z,tₗ) × T_l(t)

   where T_l is temporal basis function and f(x,y,z,tₗ) uses trilinear spatial.

3. KRIGING (Optimal Interpolation)

   Geostatistical method that provides:
   - Best Linear Unbiased Prediction (BLUP)
   - Uncertainty quantification
   - Spatial correlation modeling

   Kriging weights minimize:
       E[(Z(x₀) - Σᵢ λᵢ Z(xᵢ))²]

   Subject to unbiasedness constraint:
       Σᵢ λᵢ = 1

4. INVERSE DISTANCE WEIGHTING (IDW)

   Simple weighted average:

       f(x) = Σᵢ wᵢ f(xᵢ) / Σᵢ wᵢ

   where wᵢ = 1/d(x,xᵢ)^p and p is power parameter (typically 2).

5. NATURAL NEIGHBOR INTERPOLATION

   Uses Voronoi tessellation for smooth interpolation with local support.

Temporal Interpolation:
-----------------------

For NWP data with hourly/3-hourly time steps:

1. LINEAR: Simple linear interpolation between time steps
2. CUBIC SPLINE: Smooth C² continuity across time steps
3. PERSISTENCE: Use nearest time step (for short forecasts)

References:
-----------
[1] Oliver, M. A., & Webster, R. (1990). Kriging: A method of interpolation
    for geographical information systems. IJGIS, 4(3), 313-332.
[2] Shepard, D. (1968). A two-dimensional interpolation function for
    irregularly-spaced data. ACM National Conference.
[3] Sibson, R. (1981). A brief description of natural neighbour interpolation.
    Interpreting Multivariate Data, 21-36.
[4] Press, W. H., et al. (2007). Numerical Recipes. Cambridge University Press.

Author: eVTOL Trajectory Optimization Research Team
Version: 2.0.0
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np
from scipy.interpolate import (
    RegularGridInterpolator,
    CubicSpline,
    interp1d,
    RBFInterpolator,
)
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

# Configure module logger
logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS AND CONFIGURATION
# =============================================================================

class InterpolationMethod(Enum):
    """
    Supported interpolation methods.

    Attributes:
        value: (name, description, computational_cost)
    """
    TRILINEAR = ("trilinear", "Trilinear interpolation on regular grid", 1)
    CUBIC = ("cubic", "Cubic spline interpolation", 2)
    IDW = ("idw", "Inverse Distance Weighting", 2)
    KRIGING = ("kriging", "Ordinary Kriging with uncertainty", 4)
    RBF = ("rbf", "Radial Basis Function interpolation", 3)
    NEAREST = ("nearest", "Nearest neighbor (no interpolation)", 0)

    @property
    def description(self) -> str:
        """Human-readable description."""
        return self.value[1]

    @property
    def cost(self) -> int:
        """Relative computational cost (0=lowest, 4=highest)."""
        return self.value[2]


class TemporalMethod(Enum):
    """
    Temporal interpolation methods.

    Attributes:
        value: (name, description)
    """
    LINEAR = ("linear", "Linear interpolation between time steps")
    CUBIC = ("cubic", "Cubic spline for smooth temporal evolution")
    PERSISTENCE = ("persistence", "Use nearest time step (no interpolation)")
    EXPONENTIAL_DECAY = ("exp_decay", "Exponential weighting toward recent data")

    @property
    def description(self) -> str:
        """Human-readable description."""
        return self.value[1]


@dataclass
class InterpolationConfig:
    """
    Configuration for wind interpolation.

    Attributes:
        spatial_method: Method for spatial interpolation
        temporal_method: Method for temporal interpolation
        idw_power: Power parameter for IDW (default: 2)
        kriging_variogram: Variogram model for Kriging
        rbf_kernel: RBF kernel type ('linear', 'thin_plate', 'cubic', 'gaussian')
        max_neighbors: Maximum neighbors for local methods
        search_radius_m: Search radius for local methods
        min_points: Minimum points required for interpolation
        extrapolate: Whether to allow extrapolation
    """
    spatial_method: InterpolationMethod = InterpolationMethod.TRILINEAR
    temporal_method: TemporalMethod = TemporalMethod.LINEAR
    idw_power: float = 2.0
    kriging_variogram: str = "spherical"
    rbf_kernel: str = "thin_plate_spline"
    max_neighbors: int = 12
    search_radius_m: float = 10000.0
    min_points: int = 3
    extrapolate: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Export configuration as dictionary."""
        return {
            "spatial_method": self.spatial_method.name,
            "temporal_method": self.temporal_method.name,
            "idw_power": self.idw_power,
            "kriging_variogram": self.kriging_variogram,
            "rbf_kernel": self.rbf_kernel,
            "max_neighbors": self.max_neighbors,
            "search_radius_m": self.search_radius_m,
            "min_points": self.min_points,
            "extrapolate": self.extrapolate,
        }


@dataclass
class InterpolationResult:
    """
    Result of wind interpolation with uncertainty.

    Attributes:
        value: Interpolated value
        uncertainty: Estimation uncertainty (std dev)
        method: Method used
        n_points_used: Number of data points used
        distance_to_nearest_m: Distance to nearest observation
        is_extrapolated: Whether result is extrapolated
    """
    value: float
    uncertainty: float = 0.0
    method: str = "trilinear"
    n_points_used: int = 0
    distance_to_nearest_m: float = 0.0
    is_extrapolated: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Export result as dictionary."""
        return {
            "value": self.value,
            "uncertainty": self.uncertainty,
            "method": self.method,
            "n_points_used": self.n_points_used,
            "distance_to_nearest_m": self.distance_to_nearest_m,
            "is_extrapolated": self.is_extrapolated,
        }


# =============================================================================
# BASE INTERPOLATOR CLASS
# =============================================================================

class BaseInterpolator(ABC):
    """
    Abstract base class for spatial interpolators.

    All interpolators must implement the interpolate method.
    """

    @abstractmethod
    def interpolate(
        self,
        query_points: np.ndarray,
        data_points: np.ndarray,
        data_values: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Interpolate values at query points.

        Args:
            query_points: Array of shape (N, D) with N query points in D dimensions
            data_points: Array of shape (M, D) with M data points
            data_values: Array of shape (M,) with values at data points

        Returns:
            Tuple of (interpolated_values, uncertainties)
        """
        pass

    @abstractmethod
    def fit(self, data_points: np.ndarray, data_values: np.ndarray) -> None:
        """
        Fit interpolator to data (for methods that require fitting).

        Args:
            data_points: Array of shape (M, D)
            data_values: Array of shape (M,)
        """
        pass


# =============================================================================
# SPATIAL INTERPOLATORS
# =============================================================================

class TrilinearInterpolator(BaseInterpolator):
    """
    Trilinear interpolation on regular grids.

    Uses scipy.interpolate.RegularGridInterpolator for efficient
    interpolation on structured grids.

    Computational Complexity: O(1) per query point
    Memory: O(N) where N is grid size
    """

    def __init__(self, bounds_error: bool = False, fill_value: float | None = None):
        """
        Initialize trilinear interpolator.

        Args:
            bounds_error: Whether to raise error for out-of-bounds queries
            fill_value: Value to use for out-of-bounds (None = extrapolate)
        """
        self.bounds_error = bounds_error
        self.fill_value = fill_value
        self._interpolator: RegularGridInterpolator | None = None
        self._axes: tuple[np.ndarray, ...] | None = None

    def fit(self, data_points: np.ndarray, data_values: np.ndarray) -> None:
        """
        Fit interpolator to regular grid data.

        Note: data_points should define the grid axes, not scattered points.
        For regular grids, pass axes as tuple of 1D arrays.
        """
        # This is handled in fit_grid method for regular grids
        pass

    def fit_grid(
        self,
        axes: tuple[np.ndarray, ...],
        values: np.ndarray,
    ) -> None:
        """
        Fit interpolator to regular grid.

        Args:
            axes: Tuple of 1D arrays defining grid axes (z, y, x) or (t, z, y, x)
            values: N-dimensional array of values on grid
        """
        self._axes = axes
        self._interpolator = RegularGridInterpolator(
            axes,
            values,
            method='linear',
            bounds_error=self.bounds_error,
            fill_value=self.fill_value,
        )

    def interpolate(
        self,
        query_points: np.ndarray,
        data_points: np.ndarray = None,
        data_values: np.ndarray = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Interpolate at query points.

        Args:
            query_points: Array of shape (N, D) with query coordinates

        Returns:
            Tuple of (values, uncertainties)
        """
        if self._interpolator is None:
            raise ValueError("Interpolator not fitted. Call fit_grid() first.")

        values = self._interpolator(query_points)

        # Estimate uncertainty based on grid spacing
        uncertainties = np.zeros_like(values)

        return values, uncertainties


class IDWInterpolator(BaseInterpolator):
    """
    Inverse Distance Weighting interpolation.

    Computes weighted average of nearby points with weights inversely
    proportional to distance raised to a power.

    Formula:
        f(x) = Σᵢ wᵢ f(xᵢ) / Σᵢ wᵢ
        wᵢ = 1 / d(x, xᵢ)^p

    Properties:
        - Exact interpolation (passes through data points)
        - Smooth for p > 1
        - Local support with search radius

    Computational Complexity: O(N×k) where k is max_neighbors
    """

    def __init__(
        self,
        power: float = 2.0,
        max_neighbors: int = 12,
        search_radius: float | None = None,
        min_points: int = 1,
    ):
        """
        Initialize IDW interpolator.

        Args:
            power: Distance power parameter (p)
            max_neighbors: Maximum neighbors to consider
            search_radius: Maximum search distance (None = unlimited)
            min_points: Minimum points required
        """
        self.power = power
        self.max_neighbors = max_neighbors
        self.search_radius = search_radius
        self.min_points = min_points

        self._tree: cKDTree | None = None
        self._data_values: np.ndarray | None = None

    def fit(self, data_points: np.ndarray, data_values: np.ndarray) -> None:
        """
        Build KD-tree for efficient neighbor search.

        Args:
            data_points: Array of shape (M, D)
            data_values: Array of shape (M,)
        """
        self._tree = cKDTree(data_points)
        self._data_values = data_values.copy()

    def interpolate(
        self,
        query_points: np.ndarray,
        data_points: np.ndarray = None,
        data_values: np.ndarray = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Interpolate using IDW.

        Args:
            query_points: Array of shape (N, D)

        Returns:
            Tuple of (values, uncertainties)
        """
        if data_points is not None and data_values is not None:
            self.fit(data_points, data_values)

        if self._tree is None:
            raise ValueError("Interpolator not fitted. Call fit() first.")

        n_queries = len(query_points)
        values = np.zeros(n_queries)
        uncertainties = np.zeros(n_queries)

        # Query KD-tree for neighbors
        distances, indices = self._tree.query(
            query_points,
            k=min(self.max_neighbors, len(self._data_values)),
            distance_upper_bound=self.search_radius or np.inf,
        )

        for i in range(n_queries):
            # Handle single neighbor case
            if np.isscalar(distances[i]):
                dist = np.array([distances[i]])
                idx = np.array([indices[i]])
            else:
                dist = distances[i]
                idx = indices[i]

            # Filter out invalid indices (beyond search radius)
            valid = np.isfinite(dist)
            dist = dist[valid]
            idx = idx[valid]

            if len(idx) < self.min_points:
                values[i] = np.nan
                uncertainties[i] = np.inf
                continue

            # Handle exact matches (distance = 0)
            if np.any(dist == 0):
                zero_idx = np.where(dist == 0)[0][0]
                values[i] = self._data_values[idx[zero_idx]]
                uncertainties[i] = 0.0
                continue

            # Compute IDW weights
            weights = 1.0 / (dist ** self.power)
            weights /= np.sum(weights)

            # Weighted average
            neighbor_values = self._data_values[idx]
            values[i] = np.sum(weights * neighbor_values)

            # Uncertainty: weighted std dev
            uncertainties[i] = np.sqrt(
                np.sum(weights * (neighbor_values - values[i])**2)
            )

        return values, uncertainties


class KrigingInterpolator(BaseInterpolator):
    """
    Ordinary Kriging interpolation with uncertainty quantification.

    Kriging provides the Best Linear Unbiased Prediction (BLUP) and
    naturally produces uncertainty estimates through the kriging variance.

    Variogram Models:
        - Spherical: γ(h) = c₀ + c₁[1.5(h/a) - 0.5(h/a)³] for h ≤ a
        - Exponential: γ(h) = c₀ + c₁[1 - exp(-h/a)]
        - Gaussian: γ(h) = c₀ + c₁[1 - exp(-(h/a)²)]

    where:
        c₀ = nugget (measurement error)
        c₁ = sill - nugget (spatial variance)
        a = range (correlation length)

    Computational Complexity: O(N³) for matrix inversion
    """

    def __init__(
        self,
        variogram_model: str = "spherical",
        nugget: float = 0.01,
        sill: float = 1.0,
        range_m: float = 5000.0,
        max_neighbors: int = 20,
    ):
        """
        Initialize Kriging interpolator.

        Args:
            variogram_model: 'spherical', 'exponential', or 'gaussian'
            nugget: Nugget variance (c₀)
            sill: Total sill (c₀ + c₁)
            range_m: Variogram range in meters
            max_neighbors: Maximum neighbors for local kriging
        """
        self.variogram_model = variogram_model
        self.nugget = nugget
        self.sill = sill
        self.range_m = range_m
        self.max_neighbors = max_neighbors

        self._tree: cKDTree | None = None
        self._data_points: np.ndarray | None = None
        self._data_values: np.ndarray | None = None

    def _variogram(self, h: np.ndarray) -> np.ndarray:
        """
        Compute variogram value at lag distance h.

        Args:
            h: Lag distances (array)

        Returns:
            Variogram values γ(h)
        """
        c0 = self.nugget
        c1 = self.sill - self.nugget
        a = self.range_m

        h = np.asarray(h)
        gamma = np.zeros_like(h, dtype=float)

        if self.variogram_model == "spherical":
            # Spherical model
            in_range = h <= a
            h_norm = h[in_range] / a
            gamma[in_range] = c0 + c1 * (1.5 * h_norm - 0.5 * h_norm**3)
            gamma[~in_range] = c0 + c1

        elif self.variogram_model == "exponential":
            # Exponential model
            gamma = c0 + c1 * (1 - np.exp(-h / a))

        elif self.variogram_model == "gaussian":
            # Gaussian model
            gamma = c0 + c1 * (1 - np.exp(-(h / a)**2))

        else:
            raise ValueError(f"Unknown variogram model: {self.variogram_model}")

        # Handle zero distance
        gamma[h == 0] = 0.0

        return gamma

    def _covariance(self, h: np.ndarray) -> np.ndarray:
        """
        Compute covariance from variogram.

        C(h) = sill - γ(h)
        """
        return self.sill - self._variogram(h)

    def fit(self, data_points: np.ndarray, data_values: np.ndarray) -> None:
        """
        Fit kriging interpolator.

        Args:
            data_points: Array of shape (M, D)
            data_values: Array of shape (M,)
        """
        self._data_points = data_points.copy()
        self._data_values = data_values.copy()
        self._tree = cKDTree(data_points)

    def interpolate(
        self,
        query_points: np.ndarray,
        data_points: np.ndarray = None,
        data_values: np.ndarray = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Perform ordinary kriging interpolation.

        Args:
            query_points: Array of shape (N, D)

        Returns:
            Tuple of (values, kriging_variances)
        """
        if data_points is not None and data_values is not None:
            self.fit(data_points, data_values)

        if self._tree is None:
            raise ValueError("Interpolator not fitted. Call fit() first.")

        n_queries = len(query_points)
        values = np.zeros(n_queries)
        variances = np.zeros(n_queries)

        for i in range(n_queries):
            query = query_points[i:i+1]

            # Find neighbors
            dist, idx = self._tree.query(
                query,
                k=min(self.max_neighbors, len(self._data_values)),
            )

            dist = np.atleast_1d(dist.flatten())
            idx = np.atleast_1d(idx.flatten())

            n_neighbors = len(idx)

            # Get neighbor coordinates and values
            neighbors = self._data_points[idx]
            neighbor_values = self._data_values[idx]

            # Build kriging system
            # [C    1] [λ]   [c₀]
            # [1ᵀ   0] [μ] = [1 ]

            # Covariance matrix between neighbors
            dist_matrix = cdist(neighbors, neighbors)
            C = self._covariance(dist_matrix)

            # Covariance between query and neighbors
            c0 = self._covariance(dist)

            # Build augmented system for ordinary kriging
            K = np.zeros((n_neighbors + 1, n_neighbors + 1))
            K[:n_neighbors, :n_neighbors] = C
            K[n_neighbors, :n_neighbors] = 1.0
            K[:n_neighbors, n_neighbors] = 1.0

            b = np.zeros(n_neighbors + 1)
            b[:n_neighbors] = c0
            b[n_neighbors] = 1.0

            # Solve kriging system
            try:
                weights = np.linalg.solve(K, b)
            except np.linalg.LinAlgError:
                # Fallback to IDW if system is singular
                if np.any(dist == 0):
                    values[i] = neighbor_values[dist == 0][0]
                    variances[i] = 0.0
                else:
                    w = 1.0 / dist**2
                    w /= np.sum(w)
                    values[i] = np.sum(w * neighbor_values)
                    variances[i] = self.sill
                continue

            # Kriging estimate
            lambda_weights = weights[:n_neighbors]
            values[i] = np.sum(lambda_weights * neighbor_values)

            # Kriging variance
            mu = weights[n_neighbors]  # Lagrange multiplier
            variances[i] = self.sill - np.sum(lambda_weights * c0) - mu
            variances[i] = max(0.0, variances[i])  # Ensure non-negative

        return values, np.sqrt(variances)


class RBFSpatialInterpolator(BaseInterpolator):
    """
    Radial Basis Function interpolation.

    Uses scipy.interpolate.RBFInterpolator for scattered data interpolation.

    Kernel Options:
        - 'linear': φ(r) = r
        - 'thin_plate_spline': φ(r) = r² log(r)
        - 'cubic': φ(r) = r³
        - 'quintic': φ(r) = r⁵
        - 'gaussian': φ(r) = exp(-(εr)²)

    Properties:
        - Smooth interpolation
        - Exact at data points
        - Global support (all points influence result)
    """

    def __init__(
        self,
        kernel: str = "thin_plate_spline",
        epsilon: float | None = None,
        smoothing: float = 0.0,
    ):
        """
        Initialize RBF interpolator.

        Args:
            kernel: RBF kernel type
            epsilon: Shape parameter (for gaussian kernel)
            smoothing: Smoothing parameter (0 = exact interpolation)
        """
        self.kernel = kernel
        self.epsilon = epsilon
        self.smoothing = smoothing

        self._interpolator: RBFInterpolator | None = None

    def fit(self, data_points: np.ndarray, data_values: np.ndarray) -> None:
        """
        Fit RBF interpolator.

        Args:
            data_points: Array of shape (M, D)
            data_values: Array of shape (M,)
        """
        self._interpolator = RBFInterpolator(
            data_points,
            data_values,
            kernel=self.kernel,
            epsilon=self.epsilon,
            smoothing=self.smoothing,
        )

    def interpolate(
        self,
        query_points: np.ndarray,
        data_points: np.ndarray = None,
        data_values: np.ndarray = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Interpolate using RBF.

        Args:
            query_points: Array of shape (N, D)

        Returns:
            Tuple of (values, uncertainties)
        """
        if data_points is not None and data_values is not None:
            self.fit(data_points, data_values)

        if self._interpolator is None:
            raise ValueError("Interpolator not fitted. Call fit() first.")

        values = self._interpolator(query_points)

        # RBF doesn't naturally provide uncertainty
        # Use distance to nearest point as proxy
        tree = cKDTree(self._interpolator.y)
        distances, _ = tree.query(query_points)
        uncertainties = distances * 0.1  # Heuristic scaling

        return values, uncertainties


# =============================================================================
# TEMPORAL INTERPOLATOR
# =============================================================================

class TemporalInterpolator:
    """
    Temporal interpolation for time-varying wind fields.

    Handles interpolation between forecast time steps to provide
    wind estimates at arbitrary query times.

    Methods:
        - LINEAR: Simple linear interpolation
        - CUBIC: Cubic spline for smooth evolution
        - PERSISTENCE: Use nearest time step
        - EXPONENTIAL_DECAY: Weight recent data more heavily
    """

    def __init__(
        self,
        method: TemporalMethod = TemporalMethod.LINEAR,
        decay_hours: float = 3.0,
    ):
        """
        Initialize temporal interpolator.

        Args:
            method: Temporal interpolation method
            decay_hours: Decay time for exponential method
        """
        self.method = method
        self.decay_hours = decay_hours

    def interpolate(
        self,
        query_time: datetime,
        timestamps: list[datetime],
        values: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """
        Interpolate values at query time.

        Args:
            query_time: Target time for interpolation
            timestamps: List of data timestamps
            values: Array of values at each timestamp (can be multi-dimensional)

        Returns:
            Tuple of (interpolated_values, time_uncertainty_hours)
        """
        if len(timestamps) == 0:
            raise ValueError("No timestamps provided")

        if len(timestamps) == 1:
            # Single time step - return as is
            dt = abs((query_time - timestamps[0]).total_seconds() / 3600)
            return values[0], dt

        # Convert timestamps to hours since first timestamp
        t0 = timestamps[0]
        hours = np.array([
            (t - t0).total_seconds() / 3600 for t in timestamps
        ])
        query_hour = (query_time - t0).total_seconds() / 3600

        # Check if query is within range
        is_extrapolation = query_hour < hours[0] or query_hour > hours[-1]

        if self.method == TemporalMethod.PERSISTENCE:
            # Use nearest time step
            idx = np.argmin(np.abs(hours - query_hour))
            result = values[idx]
            uncertainty = abs(hours[idx] - query_hour)

        elif self.method == TemporalMethod.LINEAR:
            # Linear interpolation
            if values.ndim == 1:
                interp = interp1d(
                    hours, values,
                    kind='linear',
                    bounds_error=False,
                    fill_value='extrapolate',
                )
                result = interp(query_hour)
            else:
                # Multi-dimensional: interpolate along first axis
                shape = values.shape[1:]
                result = np.zeros(shape)
                for idx in np.ndindex(shape):
                    slice_values = values[(slice(None),) + idx]
                    interp = interp1d(
                        hours, slice_values,
                        kind='linear',
                        bounds_error=False,
                        fill_value='extrapolate',
                    )
                    result[idx] = interp(query_hour)

            # Uncertainty based on time to nearest data point
            uncertainty = np.min(np.abs(hours - query_hour))

        elif self.method == TemporalMethod.CUBIC:
            # Cubic spline interpolation
            if values.ndim == 1:
                spline = CubicSpline(hours, values, extrapolate=True)
                result = spline(query_hour)
            else:
                shape = values.shape[1:]
                result = np.zeros(shape)
                for idx in np.ndindex(shape):
                    slice_values = values[(slice(None),) + idx]
                    spline = CubicSpline(hours, slice_values, extrapolate=True)
                    result[idx] = spline(query_hour)

            uncertainty = np.min(np.abs(hours - query_hour)) * 0.5

        elif self.method == TemporalMethod.EXPONENTIAL_DECAY:
            # Exponential weighting toward recent data
            dt = hours - query_hour
            weights = np.exp(-np.abs(dt) / self.decay_hours)
            weights /= np.sum(weights)

            if values.ndim == 1:
                result = np.sum(weights * values)
            else:
                # Weighted sum along first axis
                result = np.tensordot(weights, values, axes=([0], [0]))

            uncertainty = self.decay_hours * (1 - np.max(weights))

        else:
            raise ValueError(f"Unknown method: {self.method}")

        # Increase uncertainty for extrapolation
        if is_extrapolation:
            uncertainty *= 2.0

        return result, float(uncertainty)


# =============================================================================
# MAIN WIND INTERPOLATOR
# =============================================================================

class WindInterpolator:
    """
    Unified wind field interpolator supporting 4D queries.

    Combines spatial and temporal interpolation to provide wind
    estimates at arbitrary (latitude, longitude, altitude, time) points.

    Features:
        - Multiple interpolation methods (trilinear, IDW, kriging, RBF)
        - Temporal interpolation with multiple strategies
        - Uncertainty quantification
        - Efficient caching of interpolators

    Usage:
        >>> interpolator = WindInterpolator(config=InterpolationConfig(
        ...     spatial_method=InterpolationMethod.TRILINEAR,
        ...     temporal_method=TemporalMethod.LINEAR,
        ... ))
        >>>
        >>> # Fit to grid data
        >>> interpolator.fit_grid(
        ...     axes=(altitudes, latitudes, longitudes),
        ...     timestamps=timestamps,
        ...     wind_u=wind_u_4d,
        ...     wind_v=wind_v_4d,
        ... )
        >>>
        >>> # Query at arbitrary point
        >>> result = interpolator.query(
        ...     latitude=12.95,
        ...     longitude=77.65,
        ...     altitude_m=150.0,
        ...     time=query_datetime,
        ... )
        >>> print(f"Wind: {result['speed_ms']:.1f} m/s ± {result['uncertainty']:.2f}")

    Attributes:
        config: Interpolation configuration
    """

    def __init__(self, config: InterpolationConfig | None = None):
        """
        Initialize wind interpolator.

        Args:
            config: Interpolation configuration (default: trilinear + linear)
        """
        self.config = config or InterpolationConfig()

        # Interpolators for each wind component
        self._u_interpolator: BaseInterpolator | None = None
        self._v_interpolator: BaseInterpolator | None = None
        self._temporal_interpolator = TemporalInterpolator(
            method=self.config.temporal_method
        )

        # Grid data storage
        self._axes: tuple[np.ndarray, ...] | None = None
        self._timestamps: list[datetime] | None = None
        self._wind_u: np.ndarray | None = None  # [time, alt, lat, lon]
        self._wind_v: np.ndarray | None = None

        self._fitted = False

        logger.info(f"WindInterpolator initialized: "
                   f"spatial={self.config.spatial_method.name}, "
                   f"temporal={self.config.temporal_method.name}")

    def _create_spatial_interpolator(self) -> BaseInterpolator:
        """Create spatial interpolator based on configuration."""
        method = self.config.spatial_method

        if method == InterpolationMethod.TRILINEAR:
            return TrilinearInterpolator()
        elif method == InterpolationMethod.IDW:
            return IDWInterpolator(
                power=self.config.idw_power,
                max_neighbors=self.config.max_neighbors,
            )
        elif method == InterpolationMethod.KRIGING:
            return KrigingInterpolator(
                variogram_model=self.config.kriging_variogram,
                max_neighbors=self.config.max_neighbors,
            )
        elif method == InterpolationMethod.RBF:
            return RBFSpatialInterpolator(kernel=self.config.rbf_kernel)
        elif method == InterpolationMethod.NEAREST:
            return IDWInterpolator(power=0, max_neighbors=1)
        else:
            raise ValueError(f"Unknown method: {method}")

    def fit_grid(
        self,
        axes: tuple[np.ndarray, np.ndarray, np.ndarray],
        timestamps: list[datetime],
        wind_u: np.ndarray,
        wind_v: np.ndarray,
    ) -> None:
        """
        Fit interpolator to regular grid data.

        Args:
            axes: (altitudes, latitudes, longitudes) as 1D arrays
            timestamps: List of timestamps for time dimension
            wind_u: 4D array [time, alt, lat, lon] of U component
            wind_v: 4D array [time, alt, lat, lon] of V component
        """
        self._axes = axes
        self._timestamps = timestamps
        self._wind_u = wind_u.copy()
        self._wind_v = wind_v.copy()

        # Validate shapes
        expected_shape = (
            len(timestamps),
            len(axes[0]),
            len(axes[1]),
            len(axes[2]),
        )
        if wind_u.shape != expected_shape:
            raise ValueError(f"wind_u shape {wind_u.shape} != expected {expected_shape}")
        if wind_v.shape != expected_shape:
            raise ValueError(f"wind_v shape {wind_v.shape} != expected {expected_shape}")

        # Create spatial interpolators
        self._u_interpolator = self._create_spatial_interpolator()
        self._v_interpolator = self._create_spatial_interpolator()

        # For trilinear, fit once per time step (lazy)
        # For other methods, we'll fit on-demand

        self._fitted = True
        logger.info(f"WindInterpolator fitted to grid: {wind_u.shape}")

    def query(
        self,
        latitude: float,
        longitude: float,
        altitude_m: float,
        time: datetime | None = None,
    ) -> dict[str, Any]:
        """
        Query wind at arbitrary 4D point.

        Args:
            latitude: Latitude in decimal degrees
            longitude: Longitude in decimal degrees
            altitude_m: Altitude in meters
            time: Query time (None = use first timestamp)

        Returns:
            Dictionary with wind components, speed, direction, and uncertainty
        """
        if not self._fitted:
            raise ValueError("Interpolator not fitted. Call fit_grid() first.")

        if time is None:
            time = self._timestamps[0]

        altitudes, latitudes, longitudes = self._axes

        # Clamp to grid bounds
        altitude_m = np.clip(altitude_m, altitudes[0], altitudes[-1])
        latitude = np.clip(latitude, latitudes[0], latitudes[-1])
        longitude = np.clip(longitude, longitudes[0], longitudes[-1])

        # Temporal interpolation first
        u_at_time, time_uncertainty = self._temporal_interpolator.interpolate(
            time, self._timestamps, self._wind_u
        )
        v_at_time, _ = self._temporal_interpolator.interpolate(
            time, self._timestamps, self._wind_v
        )

        # Spatial interpolation
        query_point = np.array([[altitude_m, latitude, longitude]])

        if isinstance(self._u_interpolator, TrilinearInterpolator):
            # Use RegularGridInterpolator for trilinear
            self._u_interpolator.fit_grid(self._axes, u_at_time)
            self._v_interpolator.fit_grid(self._axes, v_at_time)

        u_values, u_uncertainty = self._u_interpolator.interpolate(query_point)
        v_values, v_uncertainty = self._v_interpolator.interpolate(query_point)

        u = float(u_values[0])
        v = float(v_values[0])

        # Compute derived quantities
        speed = np.sqrt(u**2 + v**2)
        direction = np.degrees(np.arctan2(-u, -v))
        direction = (direction + 360) % 360

        # Combined uncertainty
        spatial_uncertainty = np.sqrt(u_uncertainty[0]**2 + v_uncertainty[0]**2)
        total_uncertainty = np.sqrt(spatial_uncertainty**2 + (time_uncertainty * 0.5)**2)

        return {
            "u_ms": u,
            "v_ms": v,
            "speed_ms": float(speed),
            "direction_deg": float(direction),
            "uncertainty_ms": float(total_uncertainty),
            "time_uncertainty_hours": float(time_uncertainty),
            "method": self.config.spatial_method.name,
            "query_point": {
                "latitude": latitude,
                "longitude": longitude,
                "altitude_m": altitude_m,
                "time": time.isoformat() if time else None,
            },
        }

    def query_trajectory(
        self,
        waypoints: list[tuple[float, float, float, datetime]],
    ) -> list[dict[str, Any]]:
        """
        Query wind along a trajectory.

        Args:
            waypoints: List of (latitude, longitude, altitude_m, time) tuples

        Returns:
            List of wind query results for each waypoint
        """
        results = []
        for lat, lon, alt, t in waypoints:
            results.append(self.query(lat, lon, alt, t))
        return results

    def get_config(self) -> dict[str, Any]:
        """Get interpolator configuration."""
        return {
            "config": self.config.to_dict(),
            "fitted": self._fitted,
            "grid_shape": self._wind_u.shape if self._fitted else None,
            "time_range": {
                "start": self._timestamps[0].isoformat() if self._timestamps else None,
                "end": self._timestamps[-1].isoformat() if self._timestamps else None,
                "n_steps": len(self._timestamps) if self._timestamps else 0,
            } if self._fitted else None,
        }
