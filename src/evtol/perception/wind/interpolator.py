"""
Wind Field Interpolator.

Provides spatial and temporal interpolation of wind data across the
mission theatre.  Mirrors the terrain interpolator interface so that
both can be used interchangeably by the cost-field fusion module.

Methods
-------
NEAREST    : Nearest-grid-cell lookup (fast, discontinuous)
TRILINEAR  : Trilinear interpolation in (x, y, z) (default)
TEMPORAL   : Time-interpolation between two forecast snapshots

References
----------
Stull, R.B. (1988). An Introduction to Boundary Layer Meteorology.
Kaimal, J.C. & Finnigan, J.J. (1994). Atmospheric Boundary Layer Flows.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
from scipy.interpolate import RegularGridInterpolator

logger = logging.getLogger(__name__)


class WindInterpolationMethod(Enum):
    NEAREST   = "nearest"
    TRILINEAR = "linear"   # maps to scipy 'linear'
    TEMPORAL  = "temporal"


@dataclass
class WindGrid:
    """Structured wind data grid for interpolation."""
    # Coordinate axes (all 1-D, monotone increasing)
    x_m: np.ndarray       # North positions (m)
    y_m: np.ndarray       # East positions  (m)
    z_m: np.ndarray       # Altitudes       (m, positive up)
    # Wind components on grid: shape (Nx, Ny, Nz)
    u:   np.ndarray       # East  component (m/s)
    v:   np.ndarray       # North component (m/s)
    w:   np.ndarray       # Vertical        (m/s)

    def __post_init__(self):
        expected = (len(self.x_m), len(self.y_m), len(self.z_m))
        for name, arr in (("u", self.u), ("v", self.v), ("w", self.w)):
            if arr.shape != expected:
                raise ValueError(
                    f"Wind component '{name}' shape {arr.shape} "
                    f"does not match grid dimensions {expected}"
                )


class WindInterpolator:
    """
    Trilinear / nearest-neighbour wind interpolation over a 3-D grid.

    Parameters
    ----------
    grid : WindGrid
        Structured wind field on a regular (x, y, z) grid.
    method : WindInterpolationMethod
        Interpolation method (default: TRILINEAR).
    """

    def __init__(
        self,
        grid:   WindGrid,
        method: WindInterpolationMethod = WindInterpolationMethod.TRILINEAR,
    ):
        self.grid   = grid
        self.method = method
        self._build_interpolators()

    def _build_interpolators(self) -> None:
        axes = (self.grid.x_m, self.grid.y_m, self.grid.z_m)
        m    = self.method.value if self.method != WindInterpolationMethod.NEAREST else "nearest"
        # RegularGridInterpolator expects 'linear' or 'nearest'
        scipy_method = "nearest" if self.method == WindInterpolationMethod.NEAREST else "linear"
        self._u_interp = RegularGridInterpolator(axes, self.grid.u, method=scipy_method, bounds_error=False, fill_value=0.0)
        self._v_interp = RegularGridInterpolator(axes, self.grid.v, method=scipy_method, bounds_error=False, fill_value=0.0)
        self._w_interp = RegularGridInterpolator(axes, self.grid.w, method=scipy_method, bounds_error=False, fill_value=0.0)
        logger.debug("WindInterpolator built (%s)", scipy_method)

    def query(self, x: float, y: float, z: float) -> tuple[float, float, float]:
        """
        Interpolate wind components at position (x, y, z).

        Args:
            x: North position (m)
            y: East position  (m)
            z: Altitude       (m, positive up)

        Returns:
            (u, v, w): Wind components (m/s) — east, north, vertical
        """
        pt = np.array([[x, y, z]])
        u  = float(self._u_interp(pt)[0])
        v  = float(self._v_interp(pt)[0])
        w  = float(self._w_interp(pt)[0])
        return u, v, w

    def query_speed(self, x: float, y: float, z: float) -> float:
        """Return wind speed magnitude (m/s) at position."""
        u, v, w = self.query(x, y, z)
        return float(np.sqrt(u**2 + v**2 + w**2))

    def query_array(self, positions: np.ndarray) -> np.ndarray:
        """
        Vectorised query for multiple positions.

        Args:
            positions: (N, 3) array of [x, y, z] positions.

        Returns:
            (N, 3) array of [u, v, w] wind components.
        """
        u = self._u_interp(positions)
        v = self._v_interp(positions)
        w = self._w_interp(positions)
        return np.stack([u, v, w], axis=-1)

    def headwind_component(
        self,
        x: float, y: float, z: float,
        heading_rad: float,
    ) -> float:
        """
        Compute headwind component along vehicle heading.

        Positive = headwind (opposing motion), Negative = tailwind.

        Args:
            heading_rad: Vehicle heading from North, clockwise (rad).

        Returns:
            Headwind component (m/s).
        """
        u, v, _ = self.query(x, y, z)
        # Vehicle unit vector (NED north=x, east=y)
        vx = np.cos(heading_rad)
        vy = np.sin(heading_rad)
        # Wind in NED (v=north, u=east)
        return float(-(v * vx + u * vy))


class TemporalWindInterpolator:
    """
    Interpolates between two wind snapshots at times t0 and t1.

    Used for time-varying wind forecasts.
    """

    def __init__(
        self,
        grid_t0: WindGrid,
        grid_t1: WindGrid,
        t0: float,
        t1: float,
    ):
        if abs(t1 - t0) < 1e-6:
            raise ValueError("t0 and t1 must be different")
        self._i0 = WindInterpolator(grid_t0)
        self._i1 = WindInterpolator(grid_t1)
        self._t0 = t0
        self._t1 = t1

    def query(self, x: float, y: float, z: float, t: float) -> tuple[float, float, float]:
        """Temporally-interpolated wind at (x, y, z, t)."""
        alpha = float(np.clip((t - self._t0) / (self._t1 - self._t0), 0.0, 1.0))
        u0, v0, w0 = self._i0.query(x, y, z)
        u1, v1, w1 = self._i1.query(x, y, z)
        return (
            (1 - alpha) * u0 + alpha * u1,
            (1 - alpha) * v0 + alpha * v1,
            (1 - alpha) * w0 + alpha * w1,
        )
