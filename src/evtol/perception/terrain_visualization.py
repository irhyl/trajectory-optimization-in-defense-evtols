"""
Terrain Visualization Module with Minimalist Pastel Aesthetics

This module provides plotting functions for terrain elevation maps
with a minimalist, calming pastel color palette.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.figure import Figure
from matplotlib.colors import LinearSegmentedColormap
import plotly.graph_objects as go
from typing import Tuple, Optional


class PastelColorScheme:
    """Minimalist pastel color schemes for terrain visualization"""
    
    # Pastel color palette (RGB normalized to [0, 1])
    PASTEL_TERRAIN = [
        (0.94, 0.92, 0.85),  # Light beige (low elevation)
        (0.90, 0.91, 0.82),  # Pale green
        (0.86, 0.89, 0.79),  # Soft green
        (0.82, 0.87, 0.77),  # Muted green
        (0.78, 0.85, 0.75),  # Subtle green
        (0.74, 0.82, 0.74),  # Gentle green
        (0.70, 0.80, 0.72),  # Soft sage
        (0.66, 0.78, 0.71),  # Pale sage
        (0.92, 0.88, 0.80),  # Warm beige
        (0.88, 0.82, 0.75),  # Tan
        (0.84, 0.77, 0.70),  # Light brown
    ]
    
    # Plotly-compatible pastel colors
    PLOTLY_PASTEL = [
        '#F0EDD9',  # Lightest
        '#E8E5CF',
        '#E0DCC5',
        '#D8D3BB',
        '#D0CAB1',
        '#C8C1A7',
        '#C0B89D',
        '#B8AF93',
        '#B0A689',
        '#A89D7F',
        '#A09475',  # Darkest
    ]
    
    @staticmethod
    def create_mpl_colormap():
        """Create matplotlib colormap with pastel colors"""
        return LinearSegmentedColormap.from_list(
            'pastel_terrain',
            PastelColorScheme.PASTEL_TERRAIN
        )
    
    @staticmethod
    def get_plotly_colorscale():
        """Get Plotly-compatible colorscale"""
        return [
            [i / (len(PastelColorScheme.PLOTLY_PASTEL) - 1), color]
            for i, color in enumerate(PastelColorScheme.PLOTLY_PASTEL)
        ]


def plot_terrain_2d_matplotlib(
    elevation: np.ndarray,
    title: str = "Terrain Elevation Map",
    figsize: Tuple[int, int] = (12, 10),
    cbar_label: str = "Elevation (m)"
) -> Figure:
    """
    Create a 2D terrain elevation map using matplotlib
    
    Args:
        elevation: 2D array of elevation values
        title: Plot title
        figsize: Figure size in inches
        cbar_label: Colorbar label
    
    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize, facecolor='white')
    
    # Create pastel colormap
    cmap = PastelColorScheme.create_mpl_colormap()
    
    # Plot with contourf for smooth appearance
    im = ax.imshow(
        elevation,
        cmap=cmap,
        aspect='auto',
        origin='lower',
        interpolation='bilinear',
        alpha=0.95
    )
    
    # Add subtle contours
    levels = np.linspace(elevation.min(), elevation.max(), 15)
    contours = ax.contour(
        elevation,
        levels=levels,
        colors='white',
        alpha=0.2,
        linewidths=0.5,
        origin='lower'
    )
    
    # Minimal styling
    ax.set_facecolor('#FEFDFB')  # Very light off-white
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['left'].set_color('#CCCCCC')
    ax.spines['bottom'].set_color('#CCCCCC')
    
    # Labels and title
    ax.set_xlabel('X Index', fontsize=11, color='#333333', fontweight='500')
    ax.set_ylabel('Y Index', fontsize=11, color='#333333', fontweight='500')
    ax.set_title(title, fontsize=14, color='#333333', fontweight='600', pad=20)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, label=cbar_label, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=9, colors='#333333')
    cbar.set_label(cbar_label, fontsize=10, color='#333333', fontweight='500')
    
    # Grid
    ax.grid(True, alpha=0.1, linestyle='--', linewidth=0.5, color='#999999')
    ax.set_axisbelow(True)
    
    # Tick styling
    ax.tick_params(colors='#333333', labelsize=9)
    
    plt.tight_layout()
    return fig


def plot_terrain_3d_plotly(
    elevation: np.ndarray,
    title: str = "Terrain Elevation Map (3D)",
    height: int = 700
) -> go.Figure:
    """
    Create an interactive 3D terrain elevation map using Plotly.
    
    Args:
        elevation: 2D array of elevation values
        title: Plot title
        height: Figure height in pixels
    
    Returns:
        Plotly figure object
    """
    # Create coordinate arrays
    ny, nx = elevation.shape
    x = np.arange(nx)
    y = np.arange(ny)
    
    # Create 3D surface plot
    fig = go.Figure(data=[
        go.Surface(
            z=elevation,
            x=x,
            y=y,
            colorscale=PastelColorScheme.get_plotly_colorscale(),
            colorbar=dict(
                title="Elevation (m)",
                thickness=15,
                len=0.7,
                x=1.02,
                tickfont=dict(size=10, color='#333333'),
                titlefont=dict(size=11, color='#333333'),
                bordercolor='#CCCCCC',
                borderwidth=1
            ),
            contours=dict(
                z=dict(
                    show=True,
                    usecolormap=True,
                    highlightcolor='#FFFFFF',
                    project=dict(z=True)
                )
            ),
            showscale=True,
            name='Elevation'
        )
    ])
    
    # Update layout for minimalist design
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=16, color='#333333', family='Arial'),
            x=0.5,
            xanchor='center'
        ),
        scene=dict(
            xaxis=dict(
                title='X Index',
                backgroundcolor='#F5F5F5',
                gridcolor='#E0E0E0',
                showbackground=True,
                zerolinecolor='#CCCCCC',
                titlefont=dict(size=11, color='#333333'),
                tickfont=dict(size=9, color='#333333')
            ),
            yaxis=dict(
                title='Y Index',
                backgroundcolor='#F5F5F5',
                gridcolor='#E0E0E0',
                showbackground=True,
                zerolinecolor='#CCCCCC',
                titlefont=dict(size=11, color='#333333'),
                tickfont=dict(size=9, color='#333333')
            ),
            zaxis=dict(
                title='Elevation (m)',
                backgroundcolor='#F5F5F5',
                gridcolor='#E0E0E0',
                showbackground=True,
                zerolinecolor='#CCCCCC',
                titlefont=dict(size=11, color='#333333'),
                tickfont=dict(size=9, color='#333333')
            ),
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.3)
            ),
            aspectmode='cube'
        ),
        height=height,
        margin=dict(l=0, r=0, b=0, t=40),
        paper_bgcolor='#FFFFFF',
        plot_bgcolor='#FAFAFA',
        hovermode='closest'
    )
    
    return fig


def plot_terrain_statistics(
    elevation: np.ndarray,
    figsize: Tuple[int, int] = (14, 5)
) -> Figure:
    """
    Create a figure with terrain statistics visualizations.
    
    Args:
        elevation: 2D array of elevation values
        figsize: Figure size in inches
    
    Returns:
        Matplotlib figure object
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize, facecolor='white')
    
    # 1. Histogram
    ax = axes[0]
    ax.hist(elevation.flatten(), bins=50, color='#B8AF93', alpha=0.7, edgecolor='#999999', linewidth=0.5)
    ax.set_xlabel('Elevation (m)', fontsize=11, color='#333333', fontweight='500')
    ax.set_ylabel('Frequency', fontsize=11, color='#333333', fontweight='500')
    ax.set_title('Elevation Distribution', fontsize=12, color='#333333', fontweight='600')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.2, linestyle='--', axis='y')
    ax.set_facecolor('#FEFDFB')
    
    # 2. Gradient magnitude
    ax = axes[1]
    grad_x, grad_y = np.gradient(elevation)
    gradient_mag = np.sqrt(grad_x**2 + grad_y**2)
    im = ax.imshow(gradient_mag, cmap='Greys', aspect='auto', origin='lower', alpha=0.85)
    ax.set_xlabel('X Index', fontsize=11, color='#333333', fontweight='500')
    ax.set_ylabel('Y Index', fontsize=11, color='#333333', fontweight='500')
    ax.set_title('Terrain Slope Magnitude', fontsize=12, color='#333333', fontweight='600')
    cbar = plt.colorbar(im, ax=ax, label='Slope (m/cell)')
    ax.set_facecolor('#FEFDFB')
    
    # 3. Cumulative distribution
    ax = axes[2]
    sorted_elev = np.sort(elevation.flatten())
    cumsum = np.arange(1, len(sorted_elev) + 1) / len(sorted_elev)
    ax.plot(sorted_elev, cumsum, color='#A89D7F', linewidth=2, alpha=0.8)
    ax.fill_between(sorted_elev, cumsum, alpha=0.2, color='#A89D7F')
    ax.set_xlabel('Elevation (m)', fontsize=11, color='#333333', fontweight='500')
    ax.set_ylabel('Cumulative Probability', fontsize=11, color='#333333', fontweight='500')
    ax.set_title('Elevation CDF', fontsize=12, color='#333333', fontweight='600')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.2, linestyle='--')
    ax.set_facecolor('#FEFDFB')
    
    plt.tight_layout()
    return fig


def plot_cross_section(
    elevation: np.ndarray,
    row_idx: Optional[int] = None,
    figsize: Tuple[int, int] = (12, 5)
) -> Figure:
    """
    Plot a cross-section of terrain elevation.
    
    Args:
        elevation: 2D array of elevation values
        row_idx: Row index for cross-section (default: middle row)
        figsize: Figure size in inches
    
    Returns:
        Matplotlib figure object
    """
    if row_idx is None:
        row_idx = elevation.shape[0] // 2
    
    fig, ax = plt.subplots(figsize=figsize, facecolor='white')
    
    # Extract cross-section
    cross_section = elevation[row_idx, :]
    x = np.arange(len(cross_section))
    
    # Plot with pastel color
    ax.fill_between(x, cross_section, alpha=0.3, color='#B8AF93')
    ax.plot(x, cross_section, color='#A89D7F', linewidth=2.5, label='Elevation Profile')
    
    # Styling
    ax.set_xlabel('X Index', fontsize=12, color='#333333', fontweight='500')
    ax.set_ylabel('Elevation (m)', fontsize=12, color='#333333', fontweight='500')
    ax.set_title(f'Terrain Cross-Section (Row {row_idx})', fontsize=13, color='#333333', fontweight='600', pad=15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.2, linestyle='--', axis='y')
    ax.set_facecolor('#FEFDFB')
    ax.legend(loc='upper right', framealpha=0.9, fancybox=True)
    
    plt.tight_layout()
    return fig
