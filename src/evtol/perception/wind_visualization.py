"""
Wind Field Visualization for eVTOL Perception Layer.

Provides production-grade visualizations of wind fields with:
- Vector field plots (quiver, streamlines)
- Wind profile comparisons across altitudes
- Wind shear and turbulence visualization
- 3D altitude layer analysis with pastel aesthetics
"""

from typing import Dict, Tuple, Optional, List
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class PastelWindColorScheme:
    """Pastel color scheme for wind visualization (consistent with terrain)."""
    
    def __init__(self):
        """Initialize pastel color palette."""
        # 9-color pastel scale for wind speed (0-15 m/s)
        self.colors = [
            '#F0EDD9',  # Very light beige (0 m/s)
            '#E8D5C4',  # Light beige
            '#DFC5AE',  # Soft beige
            '#D4B8A0',  # Tan
            '#C9A890',  # Warm tan
            '#BD9D82',  # Muted gold
            '#B09475',  # Warm brown
            '#A08A68',  # Deeper brown
            '#8E7A5C',  # Dark brown (15+ m/s)
        ]
        self.cmap = LinearSegmentedColormap.from_list('wind_pastel', self.colors)
        
        # Plotly colorscale
        self.plotly_colorscale = [
            [i / 8, self.colors[i]] for i in range(len(self.colors))
        ]
    
    def get_color(self, value: float, vmin: float = 0, vmax: float = 15) -> str:
        """Get pastel color for wind speed value."""
        normalized = np.clip((value - vmin) / (vmax - vmin), 0, 1)
        return self.colors[int(normalized * (len(self.colors) - 1))]


def plot_wind_vector_field_matplotlib(wind_speed: np.ndarray, 
                                     wind_u: np.ndarray, 
                                     wind_v: np.ndarray,
                                     altitude_idx: int = 1,
                                     skip: int = 5) -> Figure:
    """
    Plot wind vector field with quiver and streamlines.
    
    Args:
        wind_speed: 3D wind speed array [altitude, lat, lon]
        wind_u: 3D U component [altitude, lat, lon]
        wind_v: 3D V component [altitude, lat, lon]
        altitude_idx: Which altitude layer to plot (0=10m, 1=100m, 2=500m)
        skip: Plot every nth vector (to avoid clutter)
        
    Returns:
        Matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Extract altitude layer
    u_layer = wind_u[altitude_idx]
    v_layer = wind_v[altitude_idx]
    speed_layer = wind_speed[altitude_idx]
    
    # Create grid coordinates
    lat = np.arange(speed_layer.shape[0])
    lon = np.arange(speed_layer.shape[1])
    lat_grid, lon_grid = np.meshgrid(lon, lat, indexing='ij')
    
    # Plot 1: Vector field (quiver)
    color_scheme = PastelWindColorScheme()
    magnitude = np.sqrt(u_layer**2 + v_layer**2)
    
    q = ax1.quiver(
        lat_grid[::skip, ::skip], lon_grid[::skip, ::skip],
        u_layer[::skip, ::skip], v_layer[::skip, ::skip],
        magnitude[::skip, ::skip],
        cmap=color_scheme.cmap,
        scale=500,
        scale_units='inches',
        angles='xy',
        width=0.004,
        alpha=0.8
    )
    
    ax1.set_xlabel('Latitude Index', fontsize=11, color='#333333')
    ax1.set_ylabel('Longitude Index', fontsize=11, color='#333333')
    ax1.set_title(f'Wind Vector Field (Quiver) - {100 if altitude_idx == 1 else [10, 100, 500][altitude_idx]}m Altitude', 
                  fontsize=12, fontweight='bold', color='#333333', pad=15)
    ax1.set_facecolor('#FAFAF8')
    ax1.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
    cbar1 = plt.colorbar(q, ax=ax1, label='Wind Speed (m/s)')
    cbar1.ax.tick_params(labelsize=9)
    
    # Plot 2: Streamlines
    levels = np.linspace(magnitude.min(), magnitude.max(), 12)
    contour = ax2.contourf(lat_grid, lon_grid, magnitude, levels=levels, 
                           cmap=color_scheme.cmap, alpha=0.85)
    ax2.streamplot(lat_grid, lon_grid, u_layer, v_layer, 
                   color='#5C4A42', linewidth=1.0, density=1.5, arrowsize=1.5)
    
    ax2.set_xlabel('Latitude Index', fontsize=11, color='#333333')
    ax2.set_ylabel('Longitude Index', fontsize=11, color='#333333')
    ax2.set_title(f'Wind Field (Streamlines) - {100 if altitude_idx == 1 else [10, 100, 500][altitude_idx]}m Altitude',
                  fontsize=12, fontweight='bold', color='#333333', pad=15)
    ax2.set_facecolor('#FAFAF8')
    ax2.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
    cbar2 = plt.colorbar(contour, ax=ax2, label='Wind Speed (m/s)')
    cbar2.ax.tick_params(labelsize=9)
    
    plt.tight_layout()
    return fig


def plot_wind_profile_matplotlib(wind_model) -> Figure:
    """
    Plot wind speed and direction vs. altitude.
    
    Args:
        wind_model: WindFieldModel instance
        
    Returns:
        Matplotlib figure
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))
    
    stats = wind_model.get_statistics()
    altitudes = list(wind_model.altitude_bands)
    speeds = [stats[f'{int(alt)}m']['speed_mean'] for alt in altitudes]
    speeds_std = [stats[f'{int(alt)}m']['speed_std'] for alt in altitudes]
    directions = [stats[f'{int(alt)}m']['direction_mean'] for alt in altitudes]
    turbulence = [stats[f'{int(alt)}m']['turbulence_mean'] for alt in altitudes]
    
    # Plot 1: Wind Speed vs Altitude
    ax1.plot(speeds, altitudes, marker='o', markersize=10, linewidth=2.5, 
            color='#B09475', label='Mean Wind Speed')
    ax1.fill_betweenx(altitudes, 
                      np.array(speeds) - np.array(speeds_std),
                      np.array(speeds) + np.array(speeds_std),
                      alpha=0.3, color='#D4B8A0', label='±1 Std Dev')
    ax1.set_xlabel('Wind Speed (m/s)', fontsize=11, color='#333333')
    ax1.set_ylabel('Altitude (m)', fontsize=11, color='#333333')
    ax1.set_title('Wind Speed Profile', fontsize=12, fontweight='bold', color='#333333', pad=15)
    ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax1.set_facecolor('#FAFAF8')
    ax1.legend(fontsize=9, loc='best')
    
    # Plot 2: Wind Direction vs Altitude
    ax2.barh(altitudes, [1]*len(altitudes), color='#D4B8A0', alpha=0.6, edgecolor='#B09475', linewidth=2)
    for alt, direction in zip(altitudes, directions):
        ax2.text(0.5, alt, f'{direction:.0f}°', ha='center', va='center', 
                fontweight='bold', fontsize=10, color='#333333')
    ax2.set_xlabel('Wind Direction', fontsize=11, color='#333333')
    ax2.set_ylabel('Altitude (m)', fontsize=11, color='#333333')
    ax2.set_title('Mean Wind Direction', fontsize=12, fontweight='bold', color='#333333', pad=15)
    ax2.set_xlim(0, 1)
    ax2.set_xticks([])
    ax2.set_facecolor('#FAFAF8')
    
    # Plot 3: Turbulence Intensity vs Altitude
    ax3.plot(turbulence, altitudes, marker='s', markersize=10, linewidth=2.5,
            color='#8E7A5C', label='Turbulence Intensity')
    ax3.fill_betweenx(altitudes, 0, turbulence, alpha=0.3, color='#D4B8A0')
    ax3.set_xlabel('Turbulence Intensity', fontsize=11, color='#333333')
    ax3.set_ylabel('Altitude (m)', fontsize=11, color='#333333')
    ax3.set_title('Turbulence Profile', fontsize=12, fontweight='bold', color='#333333', pad=15)
    ax3.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax3.set_facecolor('#FAFAF8')
    ax3.set_xlim(0, 0.25)
    
    plt.tight_layout()
    return fig


def plot_wind_3d_layers_plotly(wind_model) -> go.Figure:
    """
    Plot interactive 3D wind layers comparison.
    
    Args:
        wind_model: WindFieldModel instance
        
    Returns:
        Plotly figure
    """
    color_scheme = PastelWindColorScheme()
    altitudes = wind_model.altitude_bands
    
    # Create subplots for each altitude
    fig = make_subplots(
        rows=1, cols=3,
        specs=[[{"type": "heatmap"}, {"type": "heatmap"}, {"type": "heatmap"}]],
        subplot_titles=[f'{int(alt)}m Altitude' for alt in altitudes],
        horizontal_spacing=0.1
    )
    
    for col_idx, alt_idx in enumerate([0, 1, 2]):
        wind_layer = wind_model.wind_speed[alt_idx]
        
        heatmap = go.Heatmap(
            z=wind_layer,
            colorscale=color_scheme.plotly_colorscale,
            colorbar=dict(x=0.32 + col_idx*0.33, len=0.4),
            showscale=(col_idx == 2),
            name=f'{int(altitudes[alt_idx])}m',
            hovertemplate='Lat: %{x}<br>Lon: %{y}<br>Wind: %{z:.1f} m/s<extra></extra>'
        )
        fig.add_trace(heatmap, row=1, col=col_idx+1)
    
    fig.update_xaxes(title_text='Latitude Index', row=1, col=1)
    fig.update_xaxes(title_text='Latitude Index', row=1, col=2)
    fig.update_xaxes(title_text='Latitude Index', row=1, col=3)
    fig.update_yaxes(title_text='Longitude Index', row=1, col=1)
    
    fig.update_layout(
        title_text='Wind Speed at Multiple Altitudes',
        title_font=dict(size=16, color='#333333'),
        height=500,
        showlegend=False,
        paper_bgcolor='#F9F8F6',
        plot_bgcolor='#FAFAF8',
        font=dict(size=11, color='#333333'),
    )
    
    return fig


def plot_wind_shear_matplotlib(wind_model) -> Figure:
    """
    Plot wind shear (speed change with altitude).
    
    Args:
        wind_model: WindFieldModel instance
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    stats = wind_model.get_statistics()
    altitudes = list(wind_model.altitude_bands)
    speeds = [stats[f'{int(alt)}m']['speed_mean'] for alt in altitudes]
    
    # Plot wind speed profile
    ax.plot(altitudes, speeds, marker='o', markersize=12, linewidth=3,
           color='#B09475', label='Mean Wind Speed', zorder=3)
    
    # Add shading between altitudes to show shear
    for i in range(len(altitudes)-1):
        ax.fill_between([altitudes[i], altitudes[i+1]], 0, 
                        [speeds[i], speeds[i+1]], 
                        alpha=0.15, color='#D4B8A0')
    
    # Logarithmic wind profile (theoretical reference)
    alt_dense = np.linspace(altitudes[0], altitudes[-1], 100)
    z0 = 0.1  # Roughness length for moderate terrain
    z_ref = 10.0
    u_ref = speeds[0]
    u_log = u_ref * np.log(alt_dense / z0) / np.log(z_ref / z0)
    ax.plot(alt_dense, u_log, '--', color='#8E7A5C', linewidth=2, 
           alpha=0.6, label='Log Wind Profile (Reference)')
    
    ax.set_xlabel('Altitude (m)', fontsize=12, color='#333333', fontweight='bold')
    ax.set_ylabel('Wind Speed (m/s)', fontsize=12, color='#333333', fontweight='bold')
    ax.set_title('Wind Shear Profile (Speed Change with Altitude)', 
                fontsize=13, fontweight='bold', color='#333333', pad=15)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_facecolor('#FAFAF8')
    ax.legend(fontsize=10, loc='best')
    
    # Add text annotations for shear rates
    if 'shear' in stats:
        shear_info = stats['shear']
        shear_text = (f"Shear (10-100m): {shear_info['shear_10_100m_per_m']:.4f} (1/m)\n"
                     f"Shear (100-500m): {shear_info['shear_100_500m_per_m']:.4f} (1/m)")
        ax.text(0.98, 0.05, shear_text, transform=ax.transAxes,
               fontsize=9, verticalalignment='bottom', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='#F0EDD9', alpha=0.8, edgecolor='#B09475'))
    
    plt.tight_layout()
    return fig


def plot_energy_cost_map_matplotlib(wind_model) -> Figure:
    """
    Plot energy cost due to wind across the domain.
    
    Args:
        wind_model: WindFieldModel instance
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    color_scheme = PastelWindColorScheme()
    heading_options = [0, 90, 180]  # North, East, South
    heading_labels = ['North (0°)', 'East (90°)', 'South (180°)']
    
    for ax, heading, label in zip(axes, heading_options, heading_labels):
        # Calculate energy cost for each grid point at 100m altitude
        energy_map = np.zeros((wind_model.grid_size, wind_model.grid_size))
        
        for i in range(0, wind_model.grid_size, 10):  # Subsample for speed
            for j in range(0, wind_model.grid_size, 10):
                energy_data = wind_model.calculate_energy_impact(100, heading)
                energy_map[i, j] = energy_data['energy_cost_kwh_per_km']
        
        # Interpolate to full resolution for smooth visualization
        from scipy.ndimage import zoom
        energy_map_full = zoom(energy_map, 10, order=1)
        
        im = ax.imshow(energy_map_full, cmap='RdYlGn_r', origin='lower', 
                       vmin=0.15, vmax=0.35, alpha=0.85)
        ax.set_title(f'Energy Cost - Flying {label}', fontsize=12, 
                    fontweight='bold', color='#333333', pad=10)
        ax.set_xlabel('Latitude Index', fontsize=11, color='#333333')
        ax.set_ylabel('Longitude Index', fontsize=11, color='#333333')
        ax.set_facecolor('#FAFAF8')
        ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
        plt.colorbar(im, ax=ax, label='Energy (kWh/km)')
    
    plt.suptitle('Energy Cost Map at 100m Altitude for Different Flight Directions', 
                fontsize=14, fontweight='bold', color='#333333', y=1.00)
    plt.tight_layout()
    return fig
