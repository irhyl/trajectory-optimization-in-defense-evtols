"""
Fused Intelligence Visualization for eVTOL Perception Layer.

Comprehensive visualizations of integrated risk, feasibility, and energy
cost assessments combining all perception layers.
"""

from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.figure import Figure
from matplotlib.colors import LinearSegmentedColormap
import plotly.graph_objects as go


class FusionColorScheme:
    """Color schemes for fusion layer visualization."""
    
    def __init__(self):
        """Initialize fusion color palettes."""
        # Risk: green to red
        self.risk_colors = ['#66BB6A', '#FDD835', '#FF9800', '#E65100', '#BF360C']
        self.risk_cmap = LinearSegmentedColormap.from_list('risk', self.risk_colors)
        
        # Feasibility: red to green
        self.feasibility_colors = ['#BF360C', '#FF9800', '#FDD835', '#A5D6A7', '#66BB6A']
        self.feasibility_cmap = LinearSegmentedColormap.from_list('feasibility', 
                                                                  self.feasibility_colors)
        
        # Energy: green to red
        self.energy_colors = ['#66BB6A', '#81C784', '#FDD835', '#FF9800', '#E65100']
        self.energy_cmap = LinearSegmentedColormap.from_list('energy', self.energy_colors)


def plot_fused_risk_map_matplotlib(fusion_model) -> Figure:
    """
    Plot integrated risk assessment map.
    
    Args:
        fusion_model: FusedIntelligenceModel instance
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(14, 11))
    
    color_scheme = FusionColorScheme()
    
    # Main risk map
    im = ax.imshow(fusion_model.risk_map, cmap=color_scheme.risk_cmap, origin='lower',
                  vmin=0.0, vmax=1.0, alpha=0.85)
    
    # Contour lines
    contours = ax.contour(fusion_model.risk_map, levels=[0.3, 0.5, 0.7], 
                         colors=['green', 'orange', 'red'], alpha=0.4, linewidths=1.5)
    ax.clabel(contours, inline=True, fontsize=9, fmt='%0.1f')
    
    ax.set_title('Integrated Risk Assessment Map', fontsize=14, fontweight='bold',
                color='#333333', pad=20)
    ax.set_xlabel('Longitude Index', fontsize=12, color='#333333', fontweight='bold')
    ax.set_ylabel('Latitude Index', fontsize=12, color='#333333', fontweight='bold')
    ax.set_facecolor('#F5F5F5')
    ax.grid(True, alpha=0.1, linestyle='--', linewidth=0.5)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, label='Risk Score (0=Safe, 1=Dangerous)')
    cbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
    
    plt.tight_layout()
    return fig


def plot_feasibility_and_risk_matplotlib(fusion_model) -> Figure:
    """
    Plot feasibility map overlaid with risk areas.
    
    Args:
        fusion_model: FusedIntelligenceModel instance
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    color_scheme = FusionColorScheme()
    
    # Plot 1: Feasibility
    ax = axes[0]
    im1 = ax.imshow(fusion_model.feasibility_map, cmap=color_scheme.feasibility_cmap,
                   origin='lower', vmin=0.0, vmax=1.0, alpha=0.85)
    ax.set_title('Path Feasibility Map', fontsize=13, fontweight='bold',
                color='#333333', pad=15)
    ax.set_xlabel('Longitude Index', fontsize=11, color='#333333')
    ax.set_ylabel('Latitude Index', fontsize=11, color='#333333')
    ax.set_facecolor('#F5F5F5')
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
    cbar1 = plt.colorbar(im1, ax=ax, label='Feasibility (0=Infeasible, 1=Feasible)')
    cbar1.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
    
    # Plot 2: Risk with contours
    ax = axes[1]
    im2 = ax.imshow(fusion_model.risk_map, cmap=color_scheme.risk_cmap, origin='lower',
                   vmin=0.0, vmax=1.0, alpha=0.85)
    
    # Overlay feasibility boundaries
    feasible_boundary = np.where(fusion_model.feasibility_map > 0.5, 1.0, 0.0)
    ax.contour(feasible_boundary, levels=[0.5], colors=['lime'], linewidths=2.5, alpha=0.7)
    
    ax.set_title('Risk Map with Feasibility Boundary (Green)', fontsize=13, fontweight='bold',
                color='#333333', pad=15)
    ax.set_xlabel('Longitude Index', fontsize=11, color='#333333')
    ax.set_ylabel('Latitude Index', fontsize=11, color='#333333')
    ax.set_facecolor('#F5F5F5')
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
    cbar2 = plt.colorbar(im2, ax=ax, label='Risk Score (0=Safe, 1=Dangerous)')
    cbar2.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
    
    plt.tight_layout()
    return fig


def plot_energy_cost_map_matplotlib(fusion_model) -> Figure:
    """
    Plot energy cost distribution for trajectory planning.
    
    Args:
        fusion_model: FusedIntelligenceModel instance
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(14, 11))
    
    color_scheme = FusionColorScheme()
    
    # Energy cost map
    im = ax.imshow(fusion_model.energy_cost_map, cmap=color_scheme.energy_cmap,
                  origin='lower', vmin=0.0, vmax=5.0, alpha=0.85)
    
    # Contour lines
    contours = ax.contour(fusion_model.energy_cost_map, levels=[1.0, 2.0, 3.0],
                         colors=['green', 'orange', 'red'], alpha=0.3, linewidths=1.5)
    ax.clabel(contours, inline=True, fontsize=9, fmt='%0.1f')
    
    ax.set_title('Energy Cost Map for Trajectory Optimization', fontsize=14, fontweight='bold',
                color='#333333', pad=20)
    ax.set_xlabel('Longitude Index', fontsize=12, color='#333333', fontweight='bold')
    ax.set_ylabel('Latitude Index', fontsize=12, color='#333333', fontweight='bold')
    ax.set_facecolor('#F5F5F5')
    ax.grid(True, alpha=0.1, linestyle='--', linewidth=0.5)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, label='Energy Cost (0=Efficient, 5+=Expensive)')
    cbar.set_ticks([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    
    plt.tight_layout()
    return fig


def plot_component_contribution_matplotlib(fusion_model) -> Figure:
    """
    Plot contribution of each perception component to fused risk.
    
    Args:
        fusion_model: FusedIntelligenceModel instance
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 13))
    
    color_scheme = FusionColorScheme()
    
    # Normalize individual components
    threat_norm = fusion_model.threat.combined_threat_map / \
                 np.max(fusion_model.threat.combined_threat_map)
    obstacle_norm = 1.0 - (fusion_model.obstacle.clearance_map / 
                           np.max(fusion_model.obstacle.clearance_map))
    terrain_slope = np.abs(np.gradient(fusion_model.terrain.dem, axis=1))
    terrain_norm = terrain_slope / np.max(terrain_slope)
    wind_norm = fusion_model.wind.wind_magnitude[:,:,1] / \
               np.max(fusion_model.wind.wind_magnitude)
    
    components = [
        ('Threat Component (30%)', threat_norm, color_scheme.risk_cmap),
        ('Obstacle Component (30%)', obstacle_norm, color_scheme.risk_cmap),
        ('Terrain Component (20%)', terrain_norm, color_scheme.risk_cmap),
        ('Wind Component (20%)', wind_norm, color_scheme.risk_cmap),
    ]
    
    for i, (ax, (title, data, cmap)) in enumerate(zip(axes.flat, components)):
        im = ax.imshow(data, cmap=cmap, origin='lower', vmin=0.0, vmax=1.0, alpha=0.85)
        ax.set_title(title, fontsize=12, fontweight='bold', color='#333333', pad=10)
        ax.set_xlabel('Longitude Index', fontsize=10, color='#333333')
        ax.set_ylabel('Latitude Index', fontsize=10, color='#333333')
        ax.set_facecolor('#F5F5F5')
        ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_ticks([0.0, 0.5, 1.0])
    
    plt.suptitle('Fused Risk Components - Individual Contributions', fontsize=14,
                fontweight='bold', color='#333333', y=0.995)
    plt.tight_layout()
    return fig


def plot_fused_statistics_matplotlib(fusion_model) -> Figure:
    """
    Plot comprehensive fused statistics dashboard.
    
    Args:
        fusion_model: FusedIntelligenceModel instance
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    stats = fusion_model.get_statistics()
    
    # Plot 1: Risk distribution
    ax = axes[0, 0]
    risk_counts, risk_bins, _ = ax.hist(fusion_model.risk_map.flatten(), bins=40,
                                        color='#FF9800', alpha=0.7, edgecolor='#E65100')
    ax.axvline(stats['mean_risk'], color='#D32F2F', linestyle='--', linewidth=2.5,
              label=f"Mean: {stats['mean_risk']:.3f}")
    ax.axvline(0.7, color='red', linestyle=':', linewidth=2, alpha=0.5,
              label='High Risk (0.7)')
    ax.set_xlabel('Risk Score', fontsize=11, color='#333333')
    ax.set_ylabel('Frequency', fontsize=11, color='#333333')
    ax.set_title('Risk Score Distribution', fontsize=12, fontweight='bold',
                color='#333333', pad=10)
    ax.legend(fontsize=10)
    ax.set_facecolor('#F5F5F5')
    ax.grid(True, alpha=0.3, axis='y', linestyle='--', linewidth=0.5)
    
    # Plot 2: Feasibility distribution
    ax = axes[0, 1]
    ax.hist(fusion_model.feasibility_map.flatten(), bins=40, color='#66BB6A', alpha=0.7,
           edgecolor='#2E7D32')
    ax.axvline(stats['mean_feasibility'], color='#D32F2F', linestyle='--', linewidth=2.5,
              label=f"Mean: {stats['mean_feasibility']:.3f}")
    ax.axvline(0.5, color='orange', linestyle=':', linewidth=2, alpha=0.5,
              label='Threshold (0.5)')
    ax.set_xlabel('Feasibility Score', fontsize=11, color='#333333')
    ax.set_ylabel('Frequency', fontsize=11, color='#333333')
    ax.set_title('Feasibility Score Distribution', fontsize=12, fontweight='bold',
                color='#333333', pad=10)
    ax.legend(fontsize=10)
    ax.set_facecolor('#F5F5F5')
    ax.grid(True, alpha=0.3, axis='y', linestyle='--', linewidth=0.5)
    
    # Plot 3: Area breakdown by risk level
    ax = axes[1, 0]
    risk_levels = {
        'Low (0-0.3)': np.sum(fusion_model.risk_map <= 0.3),
        'Medium (0.3-0.5)': np.sum((fusion_model.risk_map > 0.3) & (fusion_model.risk_map <= 0.5)),
        'High (0.5-0.7)': np.sum((fusion_model.risk_map > 0.5) & (fusion_model.risk_map <= 0.7)),
        'Critical (>0.7)': np.sum(fusion_model.risk_map > 0.7),
    }
    colors_pie = ['#66BB6A', '#FDD835', '#FF9800', '#D32F2F']
    
    wedges, texts, autotexts = ax.pie(risk_levels.values(), labels=risk_levels.keys(),
                                       colors=colors_pie, autopct='%1.1f%%', startangle=90)
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    ax.set_title('Area Distribution by Risk Level', fontsize=12, fontweight='bold',
                color='#333333', pad=10)
    
    # Plot 4: Statistics text
    ax = axes[1, 1]
    ax.axis('off')
    
    stats_text = f"""
FUSED INTELLIGENCE STATISTICS

Risk Assessment:
  Mean Risk Score:       {stats['mean_risk']:.4f}
  Max Risk Score:        {stats['max_risk']:.4f}
  High Risk Area:        {stats['high_risk_area_pct']:.1f}%

Feasibility:
  Mean Feasibility:      {stats['mean_feasibility']:.4f}
  Feasible Area:         {stats['feasible_area_pct']:.1f}%
  
Energy Costs:
  Mean Cost:             {stats['mean_energy_cost']:.2f}
  Max Cost:              {stats['max_energy_cost']:.2f}

Component Weights:
  Threat:                {fusion_model.metadata.component_weights['threat']*100:.0f}%
  Obstacle:              {fusion_model.metadata.component_weights['obstacle']*100:.0f}%
  Terrain:               {fusion_model.metadata.component_weights['terrain']*100:.0f}%
  Wind:                  {fusion_model.metadata.component_weights['wind']*100:.0f}%
    """
    
    ax.text(0.05, 0.5, stats_text, fontsize=11, family='monospace',
           verticalalignment='center', bbox=dict(boxstyle='round',
           facecolor='#E3F2FD', alpha=0.8, edgecolor='#0066CC', linewidth=2))
    
    plt.tight_layout()
    return fig


def plot_fused_3d_heatmap_plotly(fusion_model) -> go.Figure:
    """
    Interactive 3D visualization of risk landscape.
    
    Args:
        fusion_model: FusedIntelligenceModel instance
        
    Returns:
        Plotly figure
    """
    # Create 3D surface
    fig = go.Figure(data=[go.Surface(z=fusion_model.risk_map, colorscale='RdYlGn_r',
                                     colorbar=dict(title='Risk Score'))])
    
    fig.update_layout(
        title='3D Risk Surface - Fused Intelligence Layer',
        scene=dict(
            xaxis_title='Longitude Index',
            yaxis_title='Latitude Index',
            zaxis_title='Risk Score',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.3))
        ),
        width=1000,
        height=800,
        font=dict(family='Arial', size=11, color='#333333'),
        paper_bgcolor='#F5F5F5',
        plot_bgcolor='#FFFFFF'
    )
    
    return fig


def plot_comparison_before_after_fusion(fusion_model) -> Figure:
    """
    Compare individual perception layers vs fused result.
    
    Args:
        fusion_model: FusedIntelligenceModel instance
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    color_scheme = FusionColorScheme()
    
    # Normalize components for comparison
    threat_norm = fusion_model.threat.combined_threat_map / \
                 np.max(fusion_model.threat.combined_threat_map)
    obstacle_norm = 1.0 - (fusion_model.obstacle.clearance_map / 
                           np.max(fusion_model.obstacle.clearance_map))
    wind_norm = fusion_model.wind.wind_magnitude[:,:,1] / \
               np.max(fusion_model.wind.wind_magnitude)
    
    components = [
        ('Threat Risk', threat_norm),
        ('Obstacle Risk', obstacle_norm),
        ('Wind Risk', wind_norm),
    ]
    
    # Top row: Individual components
    for i, (ax, (title, data)) in enumerate(zip(axes[0, :], components)):
        im = ax.imshow(data, cmap=color_scheme.risk_cmap, origin='lower',
                      vmin=0.0, vmax=1.0, alpha=0.85)
        ax.set_title(f'{title}', fontsize=11, fontweight='bold', color='#333333', pad=10)
        ax.set_xlabel('Longitude', fontsize=10, color='#333333')
        ax.set_ylabel('Latitude', fontsize=10, color='#333333')
        ax.set_facecolor('#F5F5F5')
        ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Bottom row: Fused and derivatives
    ax = axes[1, 0]
    im = ax.imshow(fusion_model.risk_map, cmap=color_scheme.risk_cmap, origin='lower',
                  vmin=0.0, vmax=1.0, alpha=0.85)
    ax.set_title('Fused Risk', fontsize=11, fontweight='bold', color='#333333', pad=10)
    ax.set_xlabel('Longitude', fontsize=10, color='#333333')
    ax.set_ylabel('Latitude', fontsize=10, color='#333333')
    ax.set_facecolor('#F5F5F5')
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    ax = axes[1, 1]
    im = ax.imshow(fusion_model.feasibility_map, cmap=color_scheme.feasibility_cmap,
                  origin='lower', vmin=0.0, vmax=1.0, alpha=0.85)
    ax.set_title('Feasibility', fontsize=11, fontweight='bold', color='#333333', pad=10)
    ax.set_xlabel('Longitude', fontsize=10, color='#333333')
    ax.set_ylabel('Latitude', fontsize=10, color='#333333')
    ax.set_facecolor('#F5F5F5')
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    ax = axes[1, 2]
    im = ax.imshow(fusion_model.energy_cost_map, cmap=color_scheme.energy_cmap,
                  origin='lower', vmin=0.0, vmax=5.0, alpha=0.85)
    ax.set_title('Energy Cost', fontsize=11, fontweight='bold', color='#333333', pad=10)
    ax.set_xlabel('Longitude', fontsize=10, color='#333333')
    ax.set_ylabel('Latitude', fontsize=10, color='#333333')
    ax.set_facecolor('#F5F5F5')
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.suptitle('Perception Layer Integration - Individual vs Fused Intelligence',
                fontsize=14, fontweight='bold', color='#333333', y=0.995)
    plt.tight_layout()
    return fig
