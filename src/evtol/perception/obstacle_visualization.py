"""
Obstacle Detection Visualization for eVTOL Perception Layer.

Visualizes building maps, clearance zones, and landing zones with
minimalist pastel aesthetics.
"""

from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.figure import Figure
from matplotlib.colors import LinearSegmentedColormap
import plotly.graph_objects as go


class ObstacleColorScheme:
    """Color scheme for obstacle visualization."""
    
    def __init__(self):
        """Initialize obstacle color palette."""
        # Pastel colors for obstacle height
        self.colors = [
            '#E8F5E9',  # Very light (no obstacle)
            '#C8E6C9',  # Light green
            '#A5D6A7',  # Green
            '#81C784',  # Medium green
            '#66BB6A',  # Darker green
            '#FDD835',  # Yellow
            '#FB8C00',  # Orange
            '#E65100',  # Deep orange
            '#BF360C',  # Dark red
        ]
        self.cmap = LinearSegmentedColormap.from_list('obstacle_height', self.colors)


def plot_building_detection_matplotlib(obstacle_model) -> Figure:
    """
    Plot building presence and height maps.
    
    Args:
        obstacle_model: ObstacleDetectionModel instance
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    color_scheme = ObstacleColorScheme()
    
    # Plot 1: Building mask
    ax = axes[0, 0]
    im1 = ax.imshow(obstacle_model.building_mask, cmap='gray', origin='lower', alpha=0.85)
    ax.set_title('Building Presence Mask (Binary)', fontsize=12, fontweight='bold',
                color='#333333', pad=15)
    ax.set_xlabel('Latitude Index', fontsize=11, color='#333333')
    ax.set_ylabel('Longitude Index', fontsize=11, color='#333333')
    ax.set_facecolor('#F5F5F5')
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
    cbar1 = plt.colorbar(im1, ax=ax, label='Building Present')
    cbar1.set_ticks([0.25, 0.75])
    cbar1.set_ticklabels(['No', 'Yes'])
    
    # Plot 2: Building height map
    ax = axes[0, 1]
    im2 = ax.imshow(obstacle_model.building_height, cmap=color_scheme.cmap, 
                   origin='lower', vmin=0, vmax=50, alpha=0.85)
    ax.set_title('Building Height Map', fontsize=12, fontweight='bold',
                color='#333333', pad=15)
    ax.set_xlabel('Latitude Index', fontsize=11, color='#333333')
    ax.set_ylabel('Longitude Index', fontsize=11, color='#333333')
    ax.set_facecolor('#F5F5F5')
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
    cbar2 = plt.colorbar(im2, ax=ax, label='Height (m)')
    
    # Plot 3: Clearance map
    ax = axes[1, 0]
    clearance_display = obstacle_model.clearance_map.copy()
    clearance_display[clearance_display > 150] = 150  # Cap for visualization
    im3 = ax.imshow(clearance_display, cmap='RdYlGn_r', origin='lower',
                   vmin=0, vmax=150, alpha=0.85)
    ax.set_title('Minimum Safe Altitude Map (Clearance)', fontsize=12, fontweight='bold',
                color='#333333', pad=15)
    ax.set_xlabel('Latitude Index', fontsize=11, color='#333333')
    ax.set_ylabel('Longitude Index', fontsize=11, color='#333333')
    ax.set_facecolor('#F5F5F5')
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
    cbar3 = plt.colorbar(im3, ax=ax, label='Minimum Altitude (m)')
    
    # Plot 4: Height distribution histogram
    ax = axes[1, 1]
    heights = obstacle_model.building_height[obstacle_model.building_mask > 0]
    if len(heights) > 0:
        ax.hist(heights, bins=40, color='#FF9800', alpha=0.7, edgecolor='#E65100')
        ax.axvline(np.mean(heights), color='#E65100', linestyle='--', linewidth=2.5,
                  label=f'Mean: {np.mean(heights):.1f} m')
        ax.set_xlabel('Building Height (m)', fontsize=11, color='#333333')
        ax.set_ylabel('Frequency', fontsize=11, color='#333333')
        ax.set_title('Distribution of Building Heights', fontsize=12, fontweight='bold',
                    color='#333333', pad=15)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y', linestyle='--', linewidth=0.5)
        ax.set_facecolor('#F5F5F5')
    
    plt.tight_layout()
    return fig


def plot_landing_zones_matplotlib(obstacle_model) -> Figure:
    """
    Plot identified landing zones.
    
    Args:
        obstacle_model: ObstacleDetectionModel instance
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Landing zones on clearance map
    ax = axes[0]
    clearance_display = obstacle_model.clearance_map.copy()
    clearance_display[clearance_display > 150] = 150
    im1 = ax.imshow(clearance_display, cmap='RdYlGn_r', origin='lower',
                   vmin=0, vmax=150, alpha=0.7)
    
    # Draw landing zones
    for zone in obstacle_model.landing_zones:
        # Rectangle for zone bounds
        rect = patches.Rectangle(
            (zone.lon_min, zone.lat_min),
            zone.lon_max - zone.lon_min,
            zone.lat_max - zone.lat_min,
            linewidth=2, edgecolor='#00AA00', facecolor='none', alpha=0.8
        )
        ax.add_patch(rect)
        
        # Center point
        ax.plot(zone.center_lon, zone.center_lat, marker='*', markersize=15,
               color='#00AA00', markeredgecolor='black', markeredgewidth=1)
    
    ax.set_title('Landing Zones (Green Boxes)', fontsize=12, fontweight='bold',
                color='#333333', pad=15)
    ax.set_xlabel('Latitude Index', fontsize=11, color='#333333')
    ax.set_ylabel('Longitude Index', fontsize=11, color='#333333')
    ax.set_facecolor('#F5F5F5')
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
    cbar1 = plt.colorbar(im1, ax=ax, label='Clearance (m)')
    
    # Plot 2: Landing zone quality
    ax = axes[1]
    
    if obstacle_model.landing_zones:
        zone_ids = [f"LZ{z.zone_id.split('_')[1]}" for z in obstacle_model.landing_zones[:20]]
        feasibility = [z.feasibility_score for z in obstacle_model.landing_zones[:20]]
        areas = [z.area_m2 / 1000 for z in obstacle_model.landing_zones[:20]]  # Convert to km²
        
        colors_bar = ['#66BB6A' if f >= 0.7 else '#FDD835' if f >= 0.5 else '#FF9800'
                     for f in feasibility]
        
        bars = ax.barh(zone_ids, feasibility, color=colors_bar, alpha=0.8, edgecolor='#333333')
        
        # Add area labels
        for i, (bar, area) in enumerate(zip(bars, areas)):
            width = bar.get_width()
            ax.text(width - 0.05, bar.get_y() + bar.get_height()/2, 
                   f'{area:.2f}k m²', ha='right', va='center', fontsize=8,
                   fontweight='bold', color='white')
        
        ax.set_xlabel('Feasibility Score (0-1)', fontsize=11, color='#333333')
        ax.set_title(f'Landing Zone Quality (Top {min(20, len(obstacle_model.landing_zones))})',
                    fontsize=12, fontweight='bold', color='#333333', pad=15)
        ax.set_xlim(0, 1)
        ax.grid(True, alpha=0.3, axis='x', linestyle='--', linewidth=0.5)
        ax.set_facecolor('#F5F5F5')
    
    plt.tight_layout()
    return fig


def plot_clearance_profile_matplotlib(obstacle_model, lat_idx: int = None,
                                     lon_idx: int = None) -> Figure:
    """
    Plot clearance profile along a cross-section.
    
    Args:
        obstacle_model: ObstacleDetectionModel instance
        lat_idx: Latitude index for cross-section (default: center)
        lon_idx: Longitude index for cross-section (default: center)
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    if lat_idx is None:
        lat_idx = obstacle_model.grid_size // 2
    if lon_idx is None:
        lon_idx = obstacle_model.grid_size // 2
    
    # Cross-section along latitude
    lat_profile = obstacle_model.clearance_map[lat_idx, :]
    building_profile = obstacle_model.building_height[lat_idx, :]
    
    ax = axes[0]
    x = np.arange(len(lat_profile))
    ax.fill_between(x, 0, building_profile, alpha=0.6, color='#FF9800', label='Building Height')
    ax.plot(lat_profile, color='#D32F2F', linewidth=2.5, label='Min Safe Altitude')
    ax.set_ylabel('Altitude (m)', fontsize=11, color='#333333', fontweight='bold')
    ax.set_title(f'Clearance Profile - Latitude Cross-Section (Row {lat_idx})',
                fontsize=12, fontweight='bold', color='#333333', pad=15)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.legend(fontsize=10, loc='best')
    ax.set_facecolor('#F5F5F5')
    
    # Cross-section along longitude
    lon_profile = obstacle_model.clearance_map[:, lon_idx]
    building_profile_lon = obstacle_model.building_height[:, lon_idx]
    
    ax = axes[1]
    x = np.arange(len(lon_profile))
    ax.fill_between(x, 0, building_profile_lon, alpha=0.6, color='#FF9800', label='Building Height')
    ax.plot(lon_profile, color='#D32F2F', linewidth=2.5, label='Min Safe Altitude')
    ax.set_xlabel('Longitude Index', fontsize=11, color='#333333', fontweight='bold')
    ax.set_ylabel('Altitude (m)', fontsize=11, color='#333333', fontweight='bold')
    ax.set_title(f'Clearance Profile - Longitude Cross-Section (Column {lon_idx})',
                fontsize=12, fontweight='bold', color='#333333', pad=15)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.legend(fontsize=10, loc='best')
    ax.set_facecolor('#F5F5F5')
    
    plt.tight_layout()
    return fig


def plot_obstacle_statistics_matplotlib(obstacle_model) -> Figure:
    """
    Plot obstacle statistics and summary.
    
    Args:
        obstacle_model: ObstacleDetectionModel instance
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    stats = obstacle_model.get_statistics()
    
    # Plot 1: Coverage and counts
    ax = axes[0, 0]
    categories = ['Building\nCoverage', 'Landing\nZones']
    values = [stats['building_coverage_pct'], len(obstacle_model.landing_zones)]
    max_values = [100, max(50, len(obstacle_model.landing_zones))]
    
    bars = ax.bar(categories, values, color=['#FF9800', '#66BB6A'], alpha=0.8, edgecolor='#333333')
    for bar, val, max_val in zip(bars, values, max_values):
        height = bar.get_height()
        label = f'{val:.1f}%' if val == stats['building_coverage_pct'] else f'{int(val)}'
        ax.text(bar.get_x() + bar.get_width()/2., height,
               label, ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_ylabel('Value', fontsize=11, color='#333333')
    ax.set_title('Building Coverage & Landing Zone Count', fontsize=12, fontweight='bold',
                color='#333333', pad=10)
    ax.set_facecolor('#F5F5F5')
    ax.grid(True, alpha=0.3, axis='y', linestyle='--', linewidth=0.5)
    
    # Plot 2: Building heights
    ax = axes[0, 1]
    height_data = [
        stats['mean_building_height_m'],
        stats['max_building_height_m'],
        stats['mean_clearance_m']
    ]
    labels = ['Mean\nBuilding Height', 'Max\nBuilding Height', 'Mean\nMin Safe Altitude']
    colors = ['#FF9800', '#D32F2F', '#0066CC']
    
    bars = ax.bar(labels, height_data, color=colors, alpha=0.8, edgecolor='#333333')
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.1f}m', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_ylabel('Altitude (m)', fontsize=11, color='#333333')
    ax.set_title('Building & Clearance Heights', fontsize=12, fontweight='bold',
                color='#333333', pad=10)
    ax.set_facecolor('#F5F5F5')
    ax.grid(True, alpha=0.3, axis='y', linestyle='--', linewidth=0.5)
    
    # Plot 3: Landing zone distribution
    ax = axes[1, 0]
    if obstacle_model.landing_zones:
        feasibility_scores = [z.feasibility_score for z in obstacle_model.landing_zones]
        ax.hist(feasibility_scores, bins=15, color='#66BB6A', alpha=0.7, edgecolor='#2E7D32')
        ax.axvline(np.mean(feasibility_scores), color='#D32F2F', linestyle='--', 
                  linewidth=2.5, label=f'Mean: {np.mean(feasibility_scores):.2f}')
        ax.set_xlabel('Feasibility Score', fontsize=11, color='#333333')
        ax.set_ylabel('Number of Zones', fontsize=11, color='#333333')
        ax.set_title('Landing Zone Feasibility Distribution', fontsize=12, fontweight='bold',
                    color='#333333', pad=10)
        ax.legend(fontsize=10)
        ax.set_facecolor('#F5F5F5')
        ax.grid(True, alpha=0.3, axis='y', linestyle='--', linewidth=0.5)
    
    # Plot 4: Statistics text
    ax = axes[1, 1]
    ax.axis('off')
    
    stats_text = f"""
OBSTACLE & LANDING ZONE STATISTICS

Building Coverage:       {stats['building_coverage_pct']:.1f}%
Building Count:          {int(stats['building_count'])}

Heights:
  Mean:                  {stats['mean_building_height_m']:.1f} m
  Maximum:               {stats['max_building_height_m']:.1f} m
  
Clearance:
  Mean Min Safe Alt:     {stats['mean_clearance_m']:.1f} m
  
Landing Zones:
  Count:                 {int(stats['landing_zones_count'])}
  Total Area:            {stats['total_landing_area_m2']/1e6:.2f} km²
  Avg Zone Size:         {stats['total_landing_area_m2']/max(1, stats['landing_zones_count'])/1000:.1f} k m²
    """
    
    ax.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
           verticalalignment='center', bbox=dict(boxstyle='round',
           facecolor='#E8F5E9', alpha=0.8, edgecolor='#66BB6A', linewidth=2))
    
    plt.tight_layout()
    return fig
