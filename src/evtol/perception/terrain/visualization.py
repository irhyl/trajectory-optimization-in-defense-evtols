"""
This module provides comprehensive visualization capabilities:
- 2D contour maps
- Hillshade rendering
- Slope and aspect maps
- Elevation profiles
- 3D surface plots
- Cross-section views
- Viewshed visualization

All plots use research-grade color schemes and styling.

References:
    - Brewer, C.A. (2003), "A Transition in Improving Maps: The
      ColorBrewer Example", Cartography and Geographic Information Science.
    - Patterson, T. & Jenny, B. (2011), "The Development and Rationale
      of Cross-blended Hypsometric Tints", Cartographic Perspectives.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Import matplotlib with fallback
try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("matplotlib not available, visualization disabled")

# Import plotly for 3D (optional)
try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


class TerrainColorScheme:
    """
    Color schemes for terrain visualization.

    Includes hypsometric tints following cartographic best practices.
    """

    # Hypsometric tint (elevation-based coloring)
    # Based on Patterson & Jenny (2011)
    HYPSOMETRIC = [
        (0, 0.4, 0.2),      # Dark green (low elevation)
        (0.4, 0.6, 0.2),    # Green
        (0.6, 0.7, 0.3),    # Yellow-green
        (0.8, 0.8, 0.4),    # Yellow
        (0.9, 0.7, 0.4),    # Tan
        (0.8, 0.6, 0.4),    # Brown
        (0.7, 0.5, 0.4),    # Dark brown
        (0.9, 0.9, 0.9),    # Light gray (high elevation)
        (1.0, 1.0, 1.0),    # White (peaks)
    ]

    # Terrain colormap (similar to ESRI)
    TERRAIN = [
        (0.2, 0.4, 0.2),    # Dark green
        (0.4, 0.6, 0.3),    # Green
        (0.6, 0.7, 0.4),    # Yellow-green
        (0.8, 0.8, 0.5),    # Yellow
        (0.9, 0.7, 0.4),    # Orange
        (0.7, 0.5, 0.3),    # Brown
        (0.5, 0.4, 0.3),    # Dark brown
        (0.8, 0.8, 0.8),    # Gray
    ]

    # Slope colors (green=flat, red=steep)
    SLOPE = [
        (0.2, 0.7, 0.3),    # Green (flat)
        (0.6, 0.8, 0.3),    # Yellow-green
        (0.9, 0.9, 0.3),    # Yellow
        (0.9, 0.6, 0.2),    # Orange
        (0.8, 0.3, 0.2),    # Red (steep)
    ]

    # Aspect colors (circular)
    ASPECT = [
        (0.2, 0.4, 0.8),    # North (blue)
        (0.2, 0.7, 0.5),    # NE (cyan)
        (0.4, 0.8, 0.3),    # East (green)
        (0.8, 0.8, 0.2),    # SE (yellow)
        (0.9, 0.5, 0.2),    # South (orange)
        (0.8, 0.3, 0.3),    # SW (red)
        (0.6, 0.2, 0.6),    # West (purple)
        (0.3, 0.3, 0.7),    # NW (indigo)
        (0.2, 0.4, 0.8),    # North again (wrap)
    ]

    @classmethod
    def get_hypsometric_cmap(cls, name: str = "terrain_hypsometric"):
        """Create matplotlib colormap for hypsometric tints."""
        if not MATPLOTLIB_AVAILABLE:
            return None
        return mcolors.LinearSegmentedColormap.from_list(name, cls.HYPSOMETRIC)

    @classmethod
    def get_slope_cmap(cls, name: str = "terrain_slope"):
        """Create matplotlib colormap for slope."""
        if not MATPLOTLIB_AVAILABLE:
            return None
        return mcolors.LinearSegmentedColormap.from_list(name, cls.SLOPE)

    @classmethod
    def get_aspect_cmap(cls, name: str = "terrain_aspect"):
        """Create matplotlib colormap for aspect (circular)."""
        if not MATPLOTLIB_AVAILABLE:
            return None
        return mcolors.LinearSegmentedColormap.from_list(name, cls.ASPECT)


@dataclass
class PlotStyle:
    """
    Configuration for plot styling.

    Attributes:
        figsize: Figure size (width, height) in inches
        dpi: Figure resolution
        title_fontsize: Title font size
        label_fontsize: Axis label font size
        colorbar_shrink: Colorbar size relative to plot
        contour_levels: Number of contour levels
        grid_alpha: Grid transparency
        background_color: Background color
    """
    figsize: tuple[float, float] = (12, 10)
    dpi: int = 150
    title_fontsize: int = 14
    label_fontsize: int = 12
    colorbar_shrink: float = 0.8
    contour_levels: int = 20
    grid_alpha: float = 0.3
    background_color: str = "#f5f5f5"


class TerrainContourPlotter:
    """
    2D contour map plotter for terrain data.

    Example:
        >>> plotter = TerrainContourPlotter()
        >>> fig = plotter.plot_elevation(
        ...     elevation=model.elevation,
        ...     latitudes=model.latitudes,
        ...     longitudes=model.longitudes,
        ...     title="Delhi Elevation"
        ... )
        >>> fig.savefig("elevation_map.png")
    """

    def __init__(self, style: PlotStyle | None = None):
        """Initialize with optional style configuration."""
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("matplotlib required for visualization")
        self.style = style or PlotStyle()

    def plot_elevation(
        self,
        elevation: np.ndarray,
        latitudes: np.ndarray,
        longitudes: np.ndarray,
        title: str = "Terrain Elevation",
        contour_interval_m: float | None = None,
        hillshade: np.ndarray | None = None,
        show_colorbar: bool = True,
        ax: Axes | None = None,
    ) -> Figure:
        """
        Plot elevation contour map.

        Args:
            elevation: 2D elevation array
            latitudes: 1D latitude array
            longitudes: 1D longitude array
            title: Plot title
            contour_interval_m: Contour interval (auto if None)
            hillshade: Optional hillshade for shading
            show_colorbar: Whether to show colorbar
            ax: Existing axes to plot on

        Returns:
            matplotlib Figure
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=self.style.figsize, dpi=self.style.dpi)
        else:
            fig = ax.figure

        # Create meshgrid
        lon_grid, lat_grid = np.meshgrid(longitudes, latitudes)

        # Plot hillshade background if provided
        if hillshade is not None:
            ax.imshow(
                hillshade,
                extent=[longitudes.min(), longitudes.max(),
                       latitudes.min(), latitudes.max()],
                cmap='gray',
                alpha=0.5,
                aspect='auto',
            )

        # Determine contour levels
        if contour_interval_m is not None:
            levels = np.arange(
                np.floor(elevation.min() / contour_interval_m) * contour_interval_m,
                np.ceil(elevation.max() / contour_interval_m) * contour_interval_m + contour_interval_m,
                contour_interval_m
            )
        else:
            levels = self.style.contour_levels

        # Get colormap
        cmap = TerrainColorScheme.get_hypsometric_cmap()

        # Filled contours
        cf = ax.contourf(lon_grid, lat_grid, elevation, levels=levels, cmap=cmap, alpha=0.8)

        # Contour lines
        cs = ax.contour(lon_grid, lat_grid, elevation, levels=levels, colors='k', linewidths=0.3, alpha=0.5)

        # Labels on some contours
        ax.clabel(cs, inline=True, fontsize=8, fmt='%.0f')

        # Colorbar
        if show_colorbar:
            plt.colorbar(cf, ax=ax, shrink=self.style.colorbar_shrink, label='Elevation (m)')

        # Labels and title
        ax.set_xlabel('Longitude (°)', fontsize=self.style.label_fontsize)
        ax.set_ylabel('Latitude (°)', fontsize=self.style.label_fontsize)
        ax.set_title(title, fontsize=self.style.title_fontsize)

        # Grid
        ax.grid(True, alpha=self.style.grid_alpha)
        ax.set_aspect('equal')

        plt.tight_layout()
        return fig

    def plot_hillshade(
        self,
        hillshade: np.ndarray,
        latitudes: np.ndarray,
        longitudes: np.ndarray,
        elevation: np.ndarray | None = None,
        title: str = "Hillshade",
        ax: Axes | None = None,
    ) -> Figure:
        """
        Plot hillshade map.

        Args:
            hillshade: Hillshade array (0-255)
            latitudes: 1D latitude array
            longitudes: 1D longitude array
            elevation: Optional elevation for contour overlay
            title: Plot title
            ax: Existing axes

        Returns:
            matplotlib Figure
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=self.style.figsize, dpi=self.style.dpi)
        else:
            fig = ax.figure

        # Plot hillshade
        ax.imshow(
            hillshade,
            extent=[longitudes.min(), longitudes.max(),
                   latitudes.min(), latitudes.max()],
            cmap='gray',
            aspect='auto',
            origin='upper',
        )

        # Add contour overlay if elevation provided
        if elevation is not None:
            lon_grid, lat_grid = np.meshgrid(longitudes, latitudes)
            ax.contour(lon_grid, lat_grid, elevation, levels=10, colors='brown', linewidths=0.5, alpha=0.5)

        ax.set_xlabel('Longitude (°)', fontsize=self.style.label_fontsize)
        ax.set_ylabel('Latitude (°)', fontsize=self.style.label_fontsize)
        ax.set_title(title, fontsize=self.style.title_fontsize)

        plt.tight_layout()
        return fig

    def plot_slope(
        self,
        slope: np.ndarray,
        latitudes: np.ndarray,
        longitudes: np.ndarray,
        title: str = "Slope",
        max_slope: float = 45.0,
        ax: Axes | None = None,
    ) -> Figure:
        """
        Plot slope map.

        Args:
            slope: Slope array in degrees
            latitudes: 1D latitude array
            longitudes: 1D longitude array
            title: Plot title
            max_slope: Maximum slope for colormap scaling
            ax: Existing axes

        Returns:
            matplotlib Figure
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=self.style.figsize, dpi=self.style.dpi)
        else:
            fig = ax.figure

        cmap = TerrainColorScheme.get_slope_cmap()

        im = ax.imshow(
            slope,
            extent=[longitudes.min(), longitudes.max(),
                   latitudes.min(), latitudes.max()],
            cmap=cmap,
            vmin=0,
            vmax=max_slope,
            aspect='auto',
            origin='upper',
        )

        plt.colorbar(im, ax=ax, shrink=self.style.colorbar_shrink, label='Slope (°)')

        ax.set_xlabel('Longitude (°)', fontsize=self.style.label_fontsize)
        ax.set_ylabel('Latitude (°)', fontsize=self.style.label_fontsize)
        ax.set_title(title, fontsize=self.style.title_fontsize)

        plt.tight_layout()
        return fig

    def plot_aspect(
        self,
        aspect: np.ndarray,
        latitudes: np.ndarray,
        longitudes: np.ndarray,
        title: str = "Aspect",
        ax: Axes | None = None,
    ) -> Figure:
        """
        Plot aspect map.

        Args:
            aspect: Aspect array in degrees (0-360)
            latitudes: 1D latitude array
            longitudes: 1D longitude array
            title: Plot title
            ax: Existing axes

        Returns:
            matplotlib Figure
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=self.style.figsize, dpi=self.style.dpi)
        else:
            fig = ax.figure

        cmap = TerrainColorScheme.get_aspect_cmap()

        im = ax.imshow(
            aspect,
            extent=[longitudes.min(), longitudes.max(),
                   latitudes.min(), latitudes.max()],
            cmap=cmap,
            vmin=0,
            vmax=360,
            aspect='auto',
            origin='upper',
        )

        cbar = plt.colorbar(im, ax=ax, shrink=self.style.colorbar_shrink, label='Aspect (°)')
        cbar.set_ticks([0, 90, 180, 270, 360])
        cbar.set_ticklabels(['N', 'E', 'S', 'W', 'N'])

        ax.set_xlabel('Longitude (°)', fontsize=self.style.label_fontsize)
        ax.set_ylabel('Latitude (°)', fontsize=self.style.label_fontsize)
        ax.set_title(title, fontsize=self.style.title_fontsize)

        plt.tight_layout()
        return fig


class TerrainProfilePlotter:
    """
    Elevation profile plotter.

    Example:
        >>> plotter = TerrainProfilePlotter()
        >>> fig = plotter.plot_profile(
        ...     distances=profile.distances_m,
        ...     elevations=profile.elevations_m,
        ...     title="Mission Profile"
        ... )
    """

    def __init__(self, style: PlotStyle | None = None):
        """Initialize with optional style configuration."""
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("matplotlib required for visualization")
        self.style = style or PlotStyle()

    def plot_profile(
        self,
        distances: np.ndarray,
        elevations: np.ndarray,
        title: str = "Elevation Profile",
        flight_altitude: np.ndarray | None = None,
        clearance_m: float = 150.0,
        ax: Axes | None = None,
    ) -> Figure:
        """
        Plot elevation profile with optional flight path.

        Args:
            distances: Distance along path in meters
            elevations: Terrain elevation in meters
            title: Plot title
            flight_altitude: Optional flight altitude profile
            clearance_m: Required terrain clearance
            ax: Existing axes

        Returns:
            matplotlib Figure
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6), dpi=self.style.dpi)
        else:
            fig = ax.figure

        # Convert to km
        distances_km = distances / 1000

        # Plot terrain
        ax.fill_between(distances_km, 0, elevations, color='#8B4513', alpha=0.7, label='Terrain')
        ax.plot(distances_km, elevations, 'k-', linewidth=1)

        # Plot minimum safe altitude
        min_safe = elevations + clearance_m
        ax.plot(distances_km, min_safe, 'r--', linewidth=1, label=f'Min Safe Alt (+{clearance_m}m)')

        # Plot flight path if provided
        if flight_altitude is not None:
            ax.plot(distances_km, flight_altitude, 'b-', linewidth=2, label='Flight Path')

            # Shade clearance
            ax.fill_between(
                distances_km,
                elevations,
                np.minimum(flight_altitude, elevations + clearance_m),
                color='red',
                alpha=0.3,
                where=flight_altitude < min_safe,
                label='Clearance Violation'
            )

        # Labels
        ax.set_xlabel('Distance (km)', fontsize=self.style.label_fontsize)
        ax.set_ylabel('Elevation (m)', fontsize=self.style.label_fontsize)
        ax.set_title(title, fontsize=self.style.title_fontsize)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=self.style.grid_alpha)

        # Set y-axis to start from 0 or slightly below min elevation
        ax.set_ylim(bottom=max(0, elevations.min() - 50))

        plt.tight_layout()
        return fig

    def plot_cross_section(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
        elevation: np.ndarray,
        latitudes: np.ndarray,
        longitudes: np.ndarray,
        n_samples: int = 100,
        title: str = "Cross Section",
        ax: Axes | None = None,
    ) -> Figure:
        """
        Plot terrain cross-section between two points.

        Args:
            start: Start point (lat, lon)
            end: End point (lat, lon)
            elevation: 2D elevation grid
            latitudes: Grid latitudes
            longitudes: Grid longitudes
            n_samples: Number of sample points
            title: Plot title
            ax: Existing axes

        Returns:
            matplotlib Figure
        """
        from .interpolator import TerrainInterpolator, InterpolationMethod

        # Create interpolator
        interp = TerrainInterpolator(method=InterpolationMethod.BILINEAR)
        interp.fit_grid(elevation, latitudes, longitudes)

        # Sample points along line
        t = np.linspace(0, 1, n_samples)
        sample_lats = start[0] + t * (end[0] - start[0])
        sample_lons = start[1] + t * (end[1] - start[1])

        # Get elevations
        elevations = np.array([
            interp.interpolate_point(lat, lon, compute_gradients=False).elevation_m
            for lat, lon in zip(sample_lats, sample_lons)
        ])

        # Compute distances
        dlat = (end[0] - start[0]) * 111320
        dlon = (end[1] - start[1]) * 111320 * np.cos(np.radians((start[0] + end[0]) / 2))
        total_dist = np.sqrt(dlat**2 + dlon**2)
        distances = t * total_dist

        return self.plot_profile(distances, elevations, title=title, ax=ax)


class TerrainViewshedPlotter:
    """
    Viewshed visualization plotter.
    """

    def __init__(self, style: PlotStyle | None = None):
        """Initialize with optional style configuration."""
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("matplotlib required for visualization")
        self.style = style or PlotStyle()

    def plot_viewshed(
        self,
        visible: np.ndarray,
        latitudes: np.ndarray,
        longitudes: np.ndarray,
        observer: tuple[float, float],
        hillshade: np.ndarray | None = None,
        title: str = "Viewshed Analysis",
        ax: Axes | None = None,
    ) -> Figure:
        """
        Plot viewshed map.

        Args:
            visible: Boolean visibility grid
            latitudes: Grid latitudes
            longitudes: Grid longitudes
            observer: Observer (lat, lon)
            hillshade: Optional hillshade background
            title: Plot title
            ax: Existing axes

        Returns:
            matplotlib Figure
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=self.style.figsize, dpi=self.style.dpi)
        else:
            fig = ax.figure

        extent = [longitudes.min(), longitudes.max(), latitudes.min(), latitudes.max()]

        # Plot hillshade background
        if hillshade is not None:
            ax.imshow(hillshade, extent=extent, cmap='gray', alpha=0.7, aspect='auto', origin='upper')

        # Create viewshed colormap (invisible=transparent, visible=green)
        cmap = mcolors.ListedColormap(['none', '#00ff0080'])

        # Plot viewshed
        ax.imshow(visible.astype(float), extent=extent, cmap=cmap, aspect='auto', origin='upper')

        # Mark observer
        ax.plot(observer[1], observer[0], 'r*', markersize=15, label='Observer')

        ax.set_xlabel('Longitude (°)', fontsize=self.style.label_fontsize)
        ax.set_ylabel('Latitude (°)', fontsize=self.style.label_fontsize)
        ax.set_title(title, fontsize=self.style.title_fontsize)
        ax.legend()

        plt.tight_layout()
        return fig


class Terrain3DPlotter:
    """
    3D terrain surface plotter using Plotly.
    """

    def __init__(self):
        """Initialize 3D plotter."""
        if not PLOTLY_AVAILABLE:
            raise ImportError("plotly required for 3D visualization")

    def plot_surface(
        self,
        elevation: np.ndarray,
        latitudes: np.ndarray,
        longitudes: np.ndarray,
        title: str = "3D Terrain Surface",
        exaggeration: float = 1.0,
        colorscale: str = "Earth",
    ) -> go.Figure:
        """
        Create interactive 3D terrain surface.

        Args:
            elevation: 2D elevation array
            latitudes: 1D latitude array
            longitudes: 1D longitude array
            title: Plot title
            exaggeration: Vertical exaggeration factor
            colorscale: Plotly colorscale name

        Returns:
            Plotly Figure
        """
        # Create meshgrid
        lon_grid, lat_grid = np.meshgrid(longitudes, latitudes)

        # Apply vertical exaggeration
        z = elevation * exaggeration

        fig = go.Figure(data=[
            go.Surface(
                x=lon_grid,
                y=lat_grid,
                z=z,
                colorscale=colorscale,
                colorbar=dict(title="Elevation (m)"),
            )
        ])

        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title="Longitude (°)",
                yaxis_title="Latitude (°)",
                zaxis_title="Elevation (m)",
                aspectmode='manual',
                aspectratio=dict(x=1, y=1, z=0.3 * exaggeration),
            ),
            width=900,
            height=700,
        )

        return fig

    def plot_surface_with_path(
        self,
        elevation: np.ndarray,
        latitudes: np.ndarray,
        longitudes: np.ndarray,
        path_lats: np.ndarray,
        path_lons: np.ndarray,
        path_alts: np.ndarray,
        title: str = "3D Terrain with Flight Path",
        exaggeration: float = 1.0,
    ) -> go.Figure:
        """
        Create 3D terrain with flight path overlay.

        Args:
            elevation: 2D elevation array
            latitudes: 1D latitude array
            longitudes: 1D longitude array
            path_lats: Path latitudes
            path_lons: Path longitudes
            path_alts: Path altitudes
            title: Plot title
            exaggeration: Vertical exaggeration

        Returns:
            Plotly Figure
        """
        fig = self.plot_surface(elevation, latitudes, longitudes, title, exaggeration)

        # Add flight path
        fig.add_trace(go.Scatter3d(
            x=path_lons,
            y=path_lats,
            z=path_alts * exaggeration,
            mode='lines+markers',
            line=dict(color='red', width=4),
            marker=dict(size=4),
            name='Flight Path',
        ))

        return fig


# Convenience functions
def plot_terrain_overview(
    elevation: np.ndarray,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    slope: np.ndarray | None = None,
    hillshade: np.ndarray | None = None,
    title: str = "Terrain Overview",
    save_path: Path | None = None,
) -> Figure:
    """
    Create multi-panel terrain overview figure.

    Args:
        elevation: 2D elevation array
        latitudes: 1D latitude array
        longitudes: 1D longitude array
        slope: Optional slope array
        hillshade: Optional hillshade array
        title: Overall title
        save_path: Optional path to save figure

    Returns:
        matplotlib Figure
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib required")

    n_panels = 2 + (1 if slope is not None else 0) + (1 if hillshade is not None else 0)
    fig, axes = plt.subplots(1, min(n_panels, 2), figsize=(14, 6), dpi=150)

    if n_panels == 1:
        axes = [axes]

    plotter = TerrainContourPlotter()

    # Elevation
    plotter.plot_elevation(elevation, latitudes, longitudes,
                          title="Elevation", hillshade=hillshade, ax=axes[0])

    # Hillshade or slope
    if hillshade is not None:
        plotter.plot_hillshade(hillshade, latitudes, longitudes,
                              elevation=elevation, title="Hillshade", ax=axes[1])
    elif slope is not None:
        plotter.plot_slope(slope, latitudes, longitudes, title="Slope", ax=axes[1])

    fig.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def save_terrain_visualization_suite(
    elevation: np.ndarray,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    slope: np.ndarray | None = None,
    aspect: np.ndarray | None = None,
    hillshade: np.ndarray | None = None,
    output_dir: Path = Path("outputs/perception/terrain/figures"),
    prefix: str = "terrain",
) -> list[Path]:
    """
    Save complete set of terrain visualizations.

    Args:
        elevation: 2D elevation array
        latitudes: 1D latitude array
        longitudes: 1D longitude array
        slope: Optional slope array
        aspect: Optional aspect array
        hillshade: Optional hillshade array
        output_dir: Output directory
        prefix: Filename prefix

    Returns:
        List of saved file paths
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib required")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_files = []
    plotter = TerrainContourPlotter()

    # Elevation contour
    fig = plotter.plot_elevation(elevation, latitudes, longitudes, hillshade=hillshade)
    path = output_dir / f"{prefix}_elevation.png"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    saved_files.append(path)

    # Hillshade
    if hillshade is not None:
        fig = plotter.plot_hillshade(hillshade, latitudes, longitudes, elevation=elevation)
        path = output_dir / f"{prefix}_hillshade.png"
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        saved_files.append(path)

    # Slope
    if slope is not None:
        fig = plotter.plot_slope(slope, latitudes, longitudes)
        path = output_dir / f"{prefix}_slope.png"
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        saved_files.append(path)

    # Aspect
    if aspect is not None:
        fig = plotter.plot_aspect(aspect, latitudes, longitudes)
        path = output_dir / f"{prefix}_aspect.png"
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        saved_files.append(path)

    logger.info(f"Saved {len(saved_files)} terrain visualizations to {output_dir}")

    return saved_files
