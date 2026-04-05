"""
Wind Field Visualization Module for eVTOL Trajectory Optimization.

This module provides research-grade visualization capabilities for wind field
data, including 2D/3D vector fields, vertical profiles, wind roses, and
energy impact analysis plots.

Visualization Types:
--------------------

1. 2D WIND FIELD PLOTS
   - Vector field (quiver) plots with speed colormap
   - Streamline plots showing flow patterns
   - Contour maps of wind speed/direction
   - Wind barbs (meteorological convention)

2. 3D WIND FIELD PLOTS
   - Volumetric vector field visualization
   - Isosurface plots of wind speed
   - 3D streamlines/streamtubes

3. VERTICAL PROFILE PLOTS
   - Boundary layer profile visualization
   - Power-law fit comparison
   - Turbulence intensity profiles

4. WIND ROSES
   - Directional frequency histograms
   - Speed-binned wind roses
   - Sector-based analysis

5. ENERGY IMPACT PLOTS
   - Headwind/tailwind analysis
   - Power consumption overlay
   - Trajectory optimization visualization

Color Schemes:
--------------
- Minimalist pastel palette consistent with other perception modules
- Colorblind-friendly options
- Publication-ready styling

References:
-----------
[1] Tufte, E. (2001). The Visual Display of Quantitative Information.
[2] Matplotlib Documentation: https://matplotlib.org/stable/
[3] Plotly Documentation: https://plotly.com/python/

Author: eVTOL Trajectory Optimization Research Team
Version: 2.0.0
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Matplotlib imports
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.figure import Figure
from matplotlib.axes import Axes

# Optional Plotly for interactive plots
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

# Configure module logger
logger = logging.getLogger(__name__)


# =============================================================================
# COLOR SCHEMES
# =============================================================================

class WindColorScheme:
    """
    Minimalist pastel color schemes for wind visualization.

    Designed for:
    - Clarity and readability
    - Colorblind accessibility
    - Publication quality
    - Consistency with terrain/threat visualizations
    """

    # Pastel wind speed palette (light → dark blue-purple)
    PASTEL_SPEED = [
        "#F0F7FF",  # Lightest (calm)
        "#DCE8F5",
        "#C8D9EB",
        "#B4CAE1",
        "#A0BBD7",
        "#8CACCD",
        "#789DC3",
        "#648EB9",
        "#507FAF",
        "#3C70A5",
        "#28619B",  # Darkest (strong wind)
    ]

    # Pastel diverging for headwind/tailwind
    PASTEL_DIVERGING = [
        "#D4867F",  # Headwind (red-ish)
        "#E0A099",
        "#ECB9B3",
        "#F5D2CD",
        "#FEECEA",  # Neutral
        "#E8F4E8",
        "#D2E8D2",
        "#BBDCBC",
        "#A4D0A6",
        "#8DC490",  # Tailwind (green-ish)
    ]

    # Pastel turbulence intensity
    PASTEL_TURBULENCE = [
        "#F5F5DC",  # Low turbulence (cream)
        "#F0E6C8",
        "#EBD7B4",
        "#E6C8A0",
        "#E1B98C",
        "#DCAA78",
        "#D79B64",
        "#D28C50",
        "#CD7D3C",
        "#C86E28",  # High turbulence (orange)
    ]

    # Plotly-compatible colorscales
    PLOTLY_SPEED = [
        [0.0, "#F0F7FF"],
        [0.1, "#DCE8F5"],
        [0.2, "#C8D9EB"],
        [0.3, "#B4CAE1"],
        [0.4, "#A0BBD7"],
        [0.5, "#8CACCD"],
        [0.6, "#789DC3"],
        [0.7, "#648EB9"],
        [0.8, "#507FAF"],
        [0.9, "#3C70A5"],
        [1.0, "#28619B"],
    ]

    @classmethod
    def create_speed_cmap(cls) -> LinearSegmentedColormap:
        """Create matplotlib colormap for wind speed."""
        return LinearSegmentedColormap.from_list("wind_speed", cls.PASTEL_SPEED)

    @classmethod
    def create_diverging_cmap(cls) -> LinearSegmentedColormap:
        """Create matplotlib colormap for headwind/tailwind."""
        return LinearSegmentedColormap.from_list("wind_diverging", cls.PASTEL_DIVERGING)

    @classmethod
    def create_turbulence_cmap(cls) -> LinearSegmentedColormap:
        """Create matplotlib colormap for turbulence."""
        return LinearSegmentedColormap.from_list("turbulence", cls.PASTEL_TURBULENCE)

    @classmethod
    def get_direction_colors(cls, n_sectors: int = 16) -> list[str]:
        """Get evenly spaced colors for wind direction sectors."""
        cmap = plt.cm.twilight_shifted
        return [mcolors.rgb2hex(cmap(i / n_sectors)) for i in range(n_sectors)]


@dataclass
class PlotStyle:
    """
    Configuration for plot styling.

    Attributes:
        figsize: Figure size in inches
        dpi: Resolution for raster output
        font_family: Font family for labels
        title_size: Title font size
        label_size: Axis label font size
        tick_size: Tick label font size
        linewidth: Default line width
        alpha: Default transparency
        grid_alpha: Grid line transparency
    """
    figsize: tuple[float, float] = (12, 10)
    dpi: int = 150
    font_family: str = "sans-serif"
    title_size: int = 14
    label_size: int = 12
    tick_size: int = 10
    linewidth: float = 1.5
    alpha: float = 0.9
    grid_alpha: float = 0.3

    def apply(self, ax: Axes) -> None:
        """Apply style to matplotlib axes."""
        ax.tick_params(labelsize=self.tick_size)
        ax.grid(True, alpha=self.grid_alpha, linestyle="--")
        ax.set_facecolor("white")


# =============================================================================
# 2D WIND FIELD VISUALIZATION
# =============================================================================

class WindField2DPlotter:
    """
    2D wind field visualization using matplotlib.

    Creates publication-quality plots of horizontal wind fields
    at specified altitude levels.

    Usage:
        >>> plotter = WindField2DPlotter(style=PlotStyle(figsize=(14, 10)))
        >>> fig = plotter.plot_vector_field(
        ...     wind_u=model.wind_u[5],  # 5th altitude layer
        ...     wind_v=model.wind_v[5],
        ...     latitudes=model.latitudes,
        ...     longitudes=model.longitudes,
        ...     title="Wind Field at 100m AGL",
        ... )
        >>> fig.savefig("wind_field_100m.png", dpi=300)
    """

    def __init__(self, style: PlotStyle | None = None):
        """Initialize plotter with style configuration."""
        self.style = style or PlotStyle()
        self.cmap_speed = WindColorScheme.create_speed_cmap()
        self.cmap_diverging = WindColorScheme.create_diverging_cmap()

    def plot_vector_field(
        self,
        wind_u: np.ndarray,
        wind_v: np.ndarray,
        latitudes: np.ndarray,
        longitudes: np.ndarray,
        title: str = "Wind Vector Field",
        skip: int = 2,
        scale: float = 50.0,
        show_colorbar: bool = True,
    ) -> Figure:
        """
        Create quiver plot of wind vectors.

        Args:
            wind_u: 2D array [lat, lon] of U component (m/s)
            wind_v: 2D array [lat, lon] of V component (m/s)
            latitudes: 1D array of latitudes
            longitudes: 1D array of longitudes
            title: Plot title
            skip: Skip factor for arrow density
            scale: Arrow scale factor
            show_colorbar: Whether to show colorbar

        Returns:
            Matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=self.style.figsize, dpi=self.style.dpi)

        # Compute wind speed for coloring
        speed = np.sqrt(wind_u**2 + wind_v**2)

        # Create meshgrid
        lon_grid, lat_grid = np.meshgrid(longitudes, latitudes)

        # Subsample for clarity
        lon_sub = lon_grid[::skip, ::skip]
        lat_sub = lat_grid[::skip, ::skip]
        u_sub = wind_u[::skip, ::skip]
        v_sub = wind_v[::skip, ::skip]
        speed_sub = speed[::skip, ::skip]

        # Background contour of wind speed
        contour = ax.contourf(
            lon_grid, lat_grid, speed,
            levels=20,
            cmap=self.cmap_speed,
            alpha=0.7,
        )

        # Quiver plot
        ax.quiver(
            lon_sub, lat_sub, u_sub, v_sub,
            speed_sub,
            cmap=self.cmap_speed,
            scale=scale,
            width=0.003,
            headwidth=4,
            headlength=5,
            alpha=self.style.alpha,
        )

        if show_colorbar:
            cbar = plt.colorbar(contour, ax=ax, shrink=0.8, pad=0.02)
            cbar.set_label("Wind Speed (m/s)", fontsize=self.style.label_size)

        ax.set_xlabel("Longitude (°)", fontsize=self.style.label_size)
        ax.set_ylabel("Latitude (°)", fontsize=self.style.label_size)
        ax.set_title(title, fontsize=self.style.title_size, fontweight="bold")
        self.style.apply(ax)

        plt.tight_layout()
        return fig

    def plot_streamlines(
        self,
        wind_u: np.ndarray,
        wind_v: np.ndarray,
        latitudes: np.ndarray,
        longitudes: np.ndarray,
        title: str = "Wind Streamlines",
        density: float = 1.5,
        linewidth_scale: float = 2.0,
    ) -> Figure:
        """
        Create streamline plot showing flow patterns.

        Args:
            wind_u: 2D array [lat, lon] of U component
            wind_v: 2D array [lat, lon] of V component
            latitudes: 1D array of latitudes
            longitudes: 1D array of longitudes
            title: Plot title
            density: Streamline density
            linewidth_scale: Line width scaling factor

        Returns:
            Matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=self.style.figsize, dpi=self.style.dpi)

        speed = np.sqrt(wind_u**2 + wind_v**2)
        lon_grid, lat_grid = np.meshgrid(longitudes, latitudes)

        # Background speed contour
        contour = ax.contourf(
            lon_grid, lat_grid, speed,
            levels=20,
            cmap=self.cmap_speed,
            alpha=0.5,
        )

        # Streamplot
        lw = linewidth_scale * speed / speed.max()
        ax.streamplot(
            longitudes, latitudes, wind_u, wind_v,
            color=speed,
            cmap=self.cmap_speed,
            density=density,
            linewidth=lw,
            arrowsize=1.5,
        )

        cbar = plt.colorbar(contour, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label("Wind Speed (m/s)", fontsize=self.style.label_size)

        ax.set_xlabel("Longitude (°)", fontsize=self.style.label_size)
        ax.set_ylabel("Latitude (°)", fontsize=self.style.label_size)
        ax.set_title(title, fontsize=self.style.title_size, fontweight="bold")
        self.style.apply(ax)

        plt.tight_layout()
        return fig

    def plot_wind_barbs(
        self,
        wind_u: np.ndarray,
        wind_v: np.ndarray,
        latitudes: np.ndarray,
        longitudes: np.ndarray,
        title: str = "Wind Barbs",
        skip: int = 3,
    ) -> Figure:
        """
        Create wind barb plot (meteorological convention).

        Wind barbs show:
        - Short barb: 5 knots
        - Long barb: 10 knots
        - Triangle: 50 knots

        Args:
            wind_u: 2D array [lat, lon] of U component (m/s)
            wind_v: 2D array [lat, lon] of V component (m/s)
            latitudes: 1D array of latitudes
            longitudes: 1D array of longitudes
            title: Plot title
            skip: Skip factor for barb density

        Returns:
            Matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=self.style.figsize, dpi=self.style.dpi)

        speed = np.sqrt(wind_u**2 + wind_v**2)
        lon_grid, lat_grid = np.meshgrid(longitudes, latitudes)

        # Background contour
        contour = ax.contourf(
            lon_grid, lat_grid, speed,
            levels=15,
            cmap=self.cmap_speed,
            alpha=0.6,
        )

        # Convert m/s to knots for barbs (1 m/s ≈ 1.944 knots)
        u_knots = wind_u[::skip, ::skip] * 1.944
        v_knots = wind_v[::skip, ::skip] * 1.944

        ax.barbs(
            lon_grid[::skip, ::skip],
            lat_grid[::skip, ::skip],
            u_knots, v_knots,
            length=6,
            barbcolor="#2c3e50",
            flagcolor="#2c3e50",
            linewidth=0.8,
        )

        cbar = plt.colorbar(contour, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label("Wind Speed (m/s)", fontsize=self.style.label_size)

        ax.set_xlabel("Longitude (°)", fontsize=self.style.label_size)
        ax.set_ylabel("Latitude (°)", fontsize=self.style.label_size)
        ax.set_title(title, fontsize=self.style.title_size, fontweight="bold")
        self.style.apply(ax)

        plt.tight_layout()
        return fig

    def plot_speed_contour(
        self,
        wind_u: np.ndarray,
        wind_v: np.ndarray,
        latitudes: np.ndarray,
        longitudes: np.ndarray,
        title: str = "Wind Speed Contours",
        levels: int = 20,
        show_labels: bool = True,
    ) -> Figure:
        """
        Create contour plot of wind speed.

        Args:
            wind_u: 2D array [lat, lon] of U component
            wind_v: 2D array [lat, lon] of V component
            latitudes: 1D array of latitudes
            longitudes: 1D array of longitudes
            title: Plot title
            levels: Number of contour levels
            show_labels: Whether to show contour labels

        Returns:
            Matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=self.style.figsize, dpi=self.style.dpi)

        speed = np.sqrt(wind_u**2 + wind_v**2)
        lon_grid, lat_grid = np.meshgrid(longitudes, latitudes)

        # Filled contours
        cf = ax.contourf(
            lon_grid, lat_grid, speed,
            levels=levels,
            cmap=self.cmap_speed,
            alpha=0.85,
        )

        # Line contours
        cs = ax.contour(
            lon_grid, lat_grid, speed,
            levels=10,
            colors="white",
            linewidths=0.5,
            alpha=0.7,
        )

        if show_labels:
            ax.clabel(cs, inline=True, fontsize=8, fmt="%.1f")

        cbar = plt.colorbar(cf, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label("Wind Speed (m/s)", fontsize=self.style.label_size)

        ax.set_xlabel("Longitude (°)", fontsize=self.style.label_size)
        ax.set_ylabel("Latitude (°)", fontsize=self.style.label_size)
        ax.set_title(title, fontsize=self.style.title_size, fontweight="bold")
        self.style.apply(ax)

        plt.tight_layout()
        return fig


# =============================================================================
# VERTICAL PROFILE VISUALIZATION
# =============================================================================

class WindProfilePlotter:
    """
    Vertical wind profile visualization.

    Creates plots of boundary layer wind profiles, turbulence
    intensity, and power-law comparisons.

    Usage:
        >>> plotter = WindProfilePlotter()
        >>> fig = plotter.plot_profile(
        ...     altitudes=profile.altitudes,
        ...     wind_speed=profile.wind_speeds,
        ...     turbulence=profile.turbulence_intensities,
        ...     reference_height=80,
        ... )
    """

    def __init__(self, style: PlotStyle | None = None):
        """Initialize plotter with style configuration."""
        self.style = style or PlotStyle()

    def plot_profile(
        self,
        altitudes: np.ndarray,
        wind_speed: np.ndarray,
        turbulence_intensity: np.ndarray | None = None,
        reference_height: float = 10.0,
        title: str = "Atmospheric Boundary Layer Profile",
    ) -> Figure:
        """
        Plot vertical wind profile with optional turbulence.

        Args:
            altitudes: 1D array of heights (m)
            wind_speed: 1D array of wind speeds (m/s)
            turbulence_intensity: 1D array of TI values (optional)
            reference_height: Reference height for power-law fit
            title: Plot title

        Returns:
            Matplotlib Figure
        """
        n_plots = 2 if turbulence_intensity is not None else 1
        fig, axes = plt.subplots(
            1, n_plots,
            figsize=(self.style.figsize[0], self.style.figsize[1] * 0.8),
            dpi=self.style.dpi,
            sharey=True,
        )

        if n_plots == 1:
            axes = [axes]

        # Wind speed profile
        ax1 = axes[0]
        ax1.plot(
            wind_speed, altitudes,
            "o-",
            color="#3498db",
            linewidth=2,
            markersize=6,
            label="Wind Speed",
        )

        # Power-law fit overlay
        ref_idx = np.argmin(np.abs(altitudes - reference_height))
        u_ref = wind_speed[ref_idx]
        alpha = 0.14  # Typical over open terrain

        z_fit = np.linspace(altitudes.min(), altitudes.max(), 100)
        u_fit = u_ref * (z_fit / reference_height) ** alpha

        ax1.plot(
            u_fit, z_fit,
            "--",
            color="#e74c3c",
            linewidth=1.5,
            alpha=0.7,
            label=f"Power-law (α={alpha})",
        )

        ax1.axhline(
            reference_height, color="#95a5a6",
            linestyle=":", linewidth=1,
            label=f"Reference ({reference_height}m)",
        )

        ax1.set_xlabel("Wind Speed (m/s)", fontsize=self.style.label_size)
        ax1.set_ylabel("Altitude (m)", fontsize=self.style.label_size)
        ax1.legend(loc="upper left", fontsize=9)
        self.style.apply(ax1)

        # Turbulence intensity profile
        if turbulence_intensity is not None:
            ax2 = axes[1]
            ax2.fill_betweenx(
                altitudes,
                0, turbulence_intensity * 100,
                color="#f39c12",
                alpha=0.4,
            )
            ax2.plot(
                turbulence_intensity * 100, altitudes,
                "s-",
                color="#d35400",
                linewidth=2,
                markersize=5,
                label="Turbulence Intensity",
            )

            ax2.set_xlabel("Turbulence Intensity (%)", fontsize=self.style.label_size)
            ax2.legend(loc="upper right", fontsize=9)
            self.style.apply(ax2)

        fig.suptitle(title, fontsize=self.style.title_size, fontweight="bold")
        plt.tight_layout()
        return fig

    def plot_multi_profile(
        self,
        profiles: list[tuple[np.ndarray, np.ndarray, str]],
        title: str = "Wind Profile Comparison",
    ) -> Figure:
        """
        Compare multiple wind profiles.

        Args:
            profiles: List of (altitudes, speeds, label) tuples
            title: Plot title

        Returns:
            Matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=self.style.figsize, dpi=self.style.dpi)

        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(profiles)))

        for (altitudes, speeds, label), color in zip(profiles, colors):
            ax.plot(
                speeds, altitudes,
                "o-",
                color=color,
                linewidth=2,
                markersize=5,
                label=label,
            )

        ax.set_xlabel("Wind Speed (m/s)", fontsize=self.style.label_size)
        ax.set_ylabel("Altitude (m)", fontsize=self.style.label_size)
        ax.set_title(title, fontsize=self.style.title_size, fontweight="bold")
        ax.legend(loc="upper left", fontsize=10)
        self.style.apply(ax)

        plt.tight_layout()
        return fig


# =============================================================================
# WIND ROSE VISUALIZATION
# =============================================================================

class WindRosePlotter:
    """
    Wind rose (directional histogram) visualization.

    Creates polar plots showing wind direction frequency
    and speed distributions.

    Usage:
        >>> plotter = WindRosePlotter()
        >>> fig = plotter.plot_wind_rose(
        ...     directions=wind_directions,
        ...     speeds=wind_speeds,
        ...     title="Wind Rose - 100m AGL",
        ... )
    """

    def __init__(
        self,
        style: PlotStyle | None = None,
        n_sectors: int = 16,
        n_speed_bins: int = 5,
    ):
        """
        Initialize wind rose plotter.

        Args:
            style: Plot styling configuration
            n_sectors: Number of directional sectors
            n_speed_bins: Number of speed bins
        """
        self.style = style or PlotStyle()
        self.n_sectors = n_sectors
        self.n_speed_bins = n_speed_bins
        self.sector_width = 360 / n_sectors

    def plot_wind_rose(
        self,
        directions: np.ndarray,
        speeds: np.ndarray,
        title: str = "Wind Rose",
        speed_bins: list[float] | None = None,
    ) -> Figure:
        """
        Create wind rose plot.

        Args:
            directions: 1D array of wind directions (degrees from N)
            speeds: 1D array of wind speeds (m/s)
            title: Plot title
            speed_bins: Custom speed bin edges

        Returns:
            Matplotlib Figure
        """
        fig = plt.figure(figsize=self.style.figsize, dpi=self.style.dpi)
        ax = fig.add_subplot(111, projection="polar")

        # Convert to radians (meteorological convention: 0=N, clockwise)
        np.radians(90 - directions)  # Convert to math convention

        # Speed bins
        if speed_bins is None:
            max_speed = np.ceil(speeds.max())
            speed_bins = np.linspace(0, max_speed, self.n_speed_bins + 1)

        # Sector bins
        sector_edges = np.linspace(0, 360, self.n_sectors + 1)
        sector_centers = (sector_edges[:-1] + sector_edges[1:]) / 2
        sector_centers_rad = np.radians(90 - sector_centers)

        # Count by sector and speed bin
        colors = WindColorScheme.PASTEL_SPEED[::2][:len(speed_bins)-1]

        bottom = np.zeros(self.n_sectors)

        for i in range(len(speed_bins) - 1):
            mask = (speeds >= speed_bins[i]) & (speeds < speed_bins[i+1])

            counts = np.zeros(self.n_sectors)
            for j in range(self.n_sectors):
                sector_mask = (
                    (directions >= sector_edges[j]) &
                    (directions < sector_edges[j+1])
                )
                counts[j] = np.sum(mask & sector_mask)

            # Normalize to percentage
            counts_pct = counts / len(directions) * 100

            ax.bar(
                sector_centers_rad,
                counts_pct,
                width=np.radians(self.sector_width * 0.9),
                bottom=bottom,
                color=colors[i] if i < len(colors) else colors[-1],
                edgecolor="white",
                linewidth=0.5,
                label=f"{speed_bins[i]:.0f}-{speed_bins[i+1]:.0f} m/s",
            )
            bottom += counts_pct

        # Configure polar axes
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)  # Clockwise
        ax.set_thetagrids(
            np.arange(0, 360, 45),
            ["N", "NE", "E", "SE", "S", "SW", "W", "NW"],
        )

        ax.set_title(title, fontsize=self.style.title_size, fontweight="bold", pad=20)
        ax.legend(loc="lower left", bbox_to_anchor=(1.1, 0), fontsize=9)

        plt.tight_layout()
        return fig

    def plot_direction_histogram(
        self,
        directions: np.ndarray,
        title: str = "Wind Direction Distribution",
    ) -> Figure:
        """
        Create simple directional histogram.

        Args:
            directions: 1D array of wind directions (degrees)
            title: Plot title

        Returns:
            Matplotlib Figure
        """
        fig, ax = plt.subplots(
            figsize=self.style.figsize,
            dpi=self.style.dpi,
            subplot_kw={"projection": "polar"},
        )

        # Histogram
        theta = np.radians(90 - directions)
        bins = np.linspace(-np.pi, np.pi, self.n_sectors + 1)

        counts, _ = np.histogram(theta, bins=bins)
        counts_pct = counts / len(directions) * 100

        # Center bins
        bin_centers = (bins[:-1] + bins[1:]) / 2

        ax.bar(
            bin_centers,
            counts_pct,
            width=2 * np.pi / self.n_sectors * 0.9,
            color="#6ab0de",
            edgecolor="white",
            alpha=0.8,
        )

        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_thetagrids(np.arange(0, 360, 45), ["N", "NE", "E", "SE", "S", "SW", "W", "NW"])

        ax.set_title(title, fontsize=self.style.title_size, fontweight="bold", pad=20)

        plt.tight_layout()
        return fig


# =============================================================================
# ENERGY IMPACT VISUALIZATION
# =============================================================================

class EnergyImpactPlotter:
    """
    Visualization of wind energy impact on trajectory.

    Shows headwind/tailwind effects, power consumption overlay,
    and trajectory optimization context.

    Usage:
        >>> plotter = EnergyImpactPlotter()
        >>> fig = plotter.plot_trajectory_impact(
        ...     trajectory=planned_path,
        ...     wind_model=wind_model,
        ...     vehicle_speed=30,
        ... )
    """

    def __init__(self, style: PlotStyle | None = None):
        """Initialize plotter with style configuration."""
        self.style = style or PlotStyle()
        self.cmap_diverging = WindColorScheme.create_diverging_cmap()

    def plot_headwind_field(
        self,
        wind_u: np.ndarray,
        wind_v: np.ndarray,
        heading: float,
        latitudes: np.ndarray,
        longitudes: np.ndarray,
        title: str = "Headwind/Tailwind Analysis",
    ) -> Figure:
        """
        Plot headwind/tailwind for a given heading.

        Args:
            wind_u: 2D array [lat, lon] of U component
            wind_v: 2D array [lat, lon] of V component
            heading: Vehicle heading in degrees (0=N, clockwise)
            latitudes: 1D array of latitudes
            longitudes: 1D array of longitudes
            title: Plot title

        Returns:
            Matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=self.style.figsize, dpi=self.style.dpi)

        # Compute headwind component (positive = headwind, negative = tailwind)
        heading_rad = np.radians(heading)
        # Vehicle direction vector (unit)
        veh_x = np.sin(heading_rad)
        veh_y = np.cos(heading_rad)

        # Headwind = wind component opposing vehicle direction
        headwind = -(wind_u * veh_x + wind_v * veh_y)

        lon_grid, lat_grid = np.meshgrid(longitudes, latitudes)

        # Symmetric colorbar centered at zero
        max_val = max(abs(headwind.min()), abs(headwind.max()))
        norm = Normalize(vmin=-max_val, vmax=max_val)

        contour = ax.contourf(
            lon_grid, lat_grid, headwind,
            levels=20,
            cmap=self.cmap_diverging,
            norm=norm,
            alpha=0.85,
        )

        cbar = plt.colorbar(contour, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label("Headwind (+) / Tailwind (-) (m/s)", fontsize=self.style.label_size)

        # Add heading indicator
        ax.annotate(
            f"Heading: {heading:.0f}°",
            xy=(0.95, 0.95),
            xycoords="axes fraction",
            ha="right", va="top",
            fontsize=11,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        ax.set_xlabel("Longitude (°)", fontsize=self.style.label_size)
        ax.set_ylabel("Latitude (°)", fontsize=self.style.label_size)
        ax.set_title(title, fontsize=self.style.title_size, fontweight="bold")
        self.style.apply(ax)

        plt.tight_layout()
        return fig

    def plot_energy_factor(
        self,
        wind_u: np.ndarray,
        wind_v: np.ndarray,
        vehicle_speed: float,
        latitudes: np.ndarray,
        longitudes: np.ndarray,
        title: str = "Energy Factor Map",
    ) -> Figure:
        """
        Plot energy factor (ratio vs still air).

        Energy factor > 1 means more energy needed (headwind)
        Energy factor < 1 means less energy needed (tailwind)

        Args:
            wind_u: 2D array [lat, lon] of U component
            wind_v: 2D array [lat, lon] of V component
            vehicle_speed: Nominal vehicle airspeed (m/s)
            latitudes: 1D array of latitudes
            longitudes: 1D array of longitudes
            title: Plot title

        Returns:
            Matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=self.style.figsize, dpi=self.style.dpi)

        wind_speed = np.sqrt(wind_u**2 + wind_v**2)

        # Simplified energy factor based on wind/vehicle ratio
        # Assuming worst-case headwind scenario
        energy_factor = (vehicle_speed + wind_speed) / vehicle_speed

        lon_grid, lat_grid = np.meshgrid(longitudes, latitudes)

        contour = ax.contourf(
            lon_grid, lat_grid, energy_factor,
            levels=20,
            cmap=self.cmap_diverging,
            alpha=0.85,
        )

        # 1.0 contour line
        ax.contour(
            lon_grid, lat_grid, energy_factor,
            levels=[1.0],
            colors=["black"],
            linewidths=[2],
            linestyles=["--"],
        )

        cbar = plt.colorbar(contour, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label("Energy Factor (vs still air)", fontsize=self.style.label_size)

        ax.set_xlabel("Longitude (°)", fontsize=self.style.label_size)
        ax.set_ylabel("Latitude (°)", fontsize=self.style.label_size)
        ax.set_title(title, fontsize=self.style.title_size, fontweight="bold")
        self.style.apply(ax)

        plt.tight_layout()
        return fig


# =============================================================================
# 3D VISUALIZATION (PLOTLY)
# =============================================================================

class WindField3DPlotter:
    """
    Interactive 3D wind field visualization using Plotly.

    Creates volumetric visualizations of wind fields with
    interactive rotation and zoom.

    Requires: plotly

    Usage:
        >>> plotter = WindField3DPlotter()
        >>> fig = plotter.plot_3d_vectors(
        ...     wind_u=model.wind_u,
        ...     wind_v=model.wind_v,
        ...     altitudes=model.altitudes,
        ...     latitudes=model.latitudes,
        ...     longitudes=model.longitudes,
        ... )
        >>> fig.show()  # Opens in browser
    """

    def __init__(self):
        """Initialize 3D plotter."""
        if not HAS_PLOTLY:
            raise ImportError("Plotly required for 3D visualization. Install with: pip install plotly")

    def plot_3d_vectors(
        self,
        wind_u: np.ndarray,
        wind_v: np.ndarray,
        altitudes: np.ndarray,
        latitudes: np.ndarray,
        longitudes: np.ndarray,
        skip: int = 3,
        title: str = "3D Wind Field",
    ) -> go.Figure:
        """
        Create 3D cone plot of wind vectors.

        Args:
            wind_u: 3D array [alt, lat, lon] of U component
            wind_v: 3D array [alt, lat, lon] of V component
            altitudes: 1D array of altitudes
            latitudes: 1D array of latitudes
            longitudes: 1D array of longitudes
            skip: Skip factor for vector density
            title: Plot title

        Returns:
            Plotly Figure
        """
        # Create meshgrid
        lon_grid, lat_grid, alt_grid = np.meshgrid(
            longitudes[::skip],
            latitudes[::skip],
            altitudes[::skip],
            indexing="ij",
        )

        # Subsample wind components
        u_sub = wind_u[::skip, ::skip, ::skip].T
        v_sub = wind_v[::skip, ::skip, ::skip].T
        w_sub = np.zeros_like(u_sub)  # No vertical component

        # Flatten for cone plot
        x = lon_grid.flatten()
        y = lat_grid.flatten()
        z = alt_grid.flatten()
        u = u_sub.flatten()
        v = v_sub.flatten()
        w = w_sub.flatten()

        np.sqrt(u**2 + v**2)

        fig = go.Figure(data=go.Cone(
            x=x, y=y, z=z,
            u=u, v=v, w=w,
            colorscale=WindColorScheme.PLOTLY_SPEED,
            sizemode="absolute",
            sizeref=5,
            colorbar=dict(title="Wind Speed (m/s)"),
        ))

        fig.update_layout(
            title=dict(text=title, font=dict(size=18)),
            scene=dict(
                xaxis_title="Longitude (°)",
                yaxis_title="Latitude (°)",
                zaxis_title="Altitude (m)",
                aspectmode="manual",
                aspectratio=dict(x=1.5, y=1, z=0.8),
            ),
            margin=dict(l=0, r=0, t=40, b=0),
        )

        return fig

    def plot_isosurface(
        self,
        wind_speed: np.ndarray,
        altitudes: np.ndarray,
        latitudes: np.ndarray,
        longitudes: np.ndarray,
        iso_values: list[float] = None,
        title: str = "Wind Speed Isosurfaces",
    ) -> go.Figure:
        """
        Create isosurface plot of wind speed.

        Args:
            wind_speed: 3D array [alt, lat, lon] of wind speed
            altitudes: 1D array of altitudes
            latitudes: 1D array of latitudes
            longitudes: 1D array of longitudes
            iso_values: Wind speeds for isosurfaces
            title: Plot title

        Returns:
            Plotly Figure
        """
        if iso_values is None:
            iso_values = [5, 10, 15]
        lon_grid, lat_grid, alt_grid = np.meshgrid(
            longitudes, latitudes, altitudes, indexing="ij"
        )

        fig = go.Figure(data=go.Isosurface(
            x=lon_grid.flatten(),
            y=lat_grid.flatten(),
            z=alt_grid.flatten(),
            value=wind_speed.T.flatten(),
            isomin=min(iso_values),
            isomax=max(iso_values),
            surface_count=len(iso_values),
            colorscale=WindColorScheme.PLOTLY_SPEED,
            caps=dict(x_show=False, y_show=False, z_show=False),
            opacity=0.6,
            colorbar=dict(title="Wind Speed (m/s)"),
        ))

        fig.update_layout(
            title=dict(text=title, font=dict(size=18)),
            scene=dict(
                xaxis_title="Longitude (°)",
                yaxis_title="Latitude (°)",
                zaxis_title="Altitude (m)",
            ),
        )

        return fig


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def plot_wind_field(
    wind_model,
    altitude_idx: int = 0,
    plot_type: str = "vectors",
    output_path: Path | None = None,
    **kwargs,
) -> Figure:
    """
    Quick plot of wind field from WindFieldModel.

    Args:
        wind_model: WindFieldModel instance
        altitude_idx: Altitude layer index
        plot_type: "vectors", "streamlines", "barbs", or "contour"
        output_path: Optional path to save figure
        **kwargs: Additional arguments for plotter

    Returns:
        Matplotlib Figure
    """
    plotter = WindField2DPlotter()

    wind_u = wind_model.wind_u[altitude_idx]
    wind_v = wind_model.wind_v[altitude_idx]
    altitude = wind_model.altitudes[altitude_idx]

    title = kwargs.pop("title", f"Wind Field at {altitude:.0f}m AGL")

    if plot_type == "vectors":
        fig = plotter.plot_vector_field(
            wind_u, wind_v,
            wind_model.latitudes, wind_model.longitudes,
            title=title, **kwargs,
        )
    elif plot_type == "streamlines":
        fig = plotter.plot_streamlines(
            wind_u, wind_v,
            wind_model.latitudes, wind_model.longitudes,
            title=title, **kwargs,
        )
    elif plot_type == "barbs":
        fig = plotter.plot_wind_barbs(
            wind_u, wind_v,
            wind_model.latitudes, wind_model.longitudes,
            title=title, **kwargs,
        )
    elif plot_type == "contour":
        fig = plotter.plot_speed_contour(
            wind_u, wind_v,
            wind_model.latitudes, wind_model.longitudes,
            title=title, **kwargs,
        )
    else:
        raise ValueError(f"Unknown plot type: {plot_type}")

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved figure: {output_path}")

    return fig


def save_wind_visualization_suite(
    wind_model,
    output_dir: Path,
    altitude_indices: list[int] | None = None,
) -> list[Path]:
    """
    Generate and save a complete visualization suite.

    Args:
        wind_model: WindFieldModel instance
        output_dir: Directory to save figures
        altitude_indices: Altitude layers to plot (default: first, middle, last)

    Returns:
        List of saved file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if altitude_indices is None:
        n_alt = len(wind_model.altitudes)
        altitude_indices = [0, n_alt // 2, n_alt - 1]

    saved_files = []

    for alt_idx in altitude_indices:
        altitude = wind_model.altitudes[alt_idx]

        for plot_type in ["vectors", "streamlines", "contour"]:
            filename = f"wind_{plot_type}_{int(altitude)}m.png"
            filepath = output_dir / filename

            fig = plot_wind_field(
                wind_model, alt_idx, plot_type, output_path=filepath
            )
            plt.close(fig)
            saved_files.append(filepath)

    logger.info(f"Saved {len(saved_files)} visualization files to {output_dir}")
    return saved_files
