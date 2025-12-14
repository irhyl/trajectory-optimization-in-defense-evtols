"""
Threat Assessment Visualization for eVTOL Perception Layer.

Visualizes threat detection maps, SAM coverage zones, and safe corridors
with minimalist pastel aesthetics.
"""

from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
import matplotlib.patches as patches
import plotly.graph_objects as go


class ThreatColorScheme:
    """Threat-specific color scheme (red-yellow-green for danger levels)."""
    
    def __init__(self):
        """Initialize threat color palette."""
        # Red-yellow-green for threat levels
        self.colors = [
            '#E8F5E9',  # Very light green (safe)
            '#C8E6C9',  # Light green
            '#A5D6A7',  # Green
            '#7CB342',  # Yellow-green
            '#FDD835',  # Yellow
            '#FB8C00',  # Orange
            '#E65100',  # Deep orange
            '#D32F2F',  # Red
            '#B71C1C',  # Dark red (high threat)
        ]
        self.cmap = LinearSegmentedColormap.from_list('threat', self.colors)


def plot_threat_heatmap_matplotlib(threat_model) -> Figure:
    """
    Plot combined threat heatmap with contours.
    
    Args:
        threat_model: ThreatAssessmentModel instance
        
    Returns:
        Matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    threat_data = threat_model.threat_heatmap
    color_scheme = ThreatColorScheme()
    
    # Plot 1: Heatmap with contours
    im1 = ax1.imshow(threat_data, cmap=color_scheme.cmap, origin='lower', 
                     vmin=0, vmax=1, alpha=0.85)
    
    contours = ax1.contour(threat_data, levels=[0.3, 0.5, 0.7], 
                           colors=['#558B2F', '#F57F17', '#C62828'], 
                           linewidths=[1.5, 2, 2.5], alpha=0.7)
    ax1.clabel(contours, inline=True, fontsize=9, fmt='%.1f')
    
    # Mark threat sources
    for threat in threat_model.threat_sources:
        marker_color = '#FF0000' if 'SAM' in threat.threat_type.value else '#0066CC'
        ax1.plot(threat.lon_idx, threat.lat_idx, marker='*', markersize=20, 
                color=marker_color, markeredgecolor='black', markeredgewidth=1.5,
                zorder=5, label=threat.threat_id)
    
    ax1.set_xlabel('Latitude Index', fontsize=11, color='#333333')
    ax1.set_ylabel('Longitude Index', fontsize=11, color='#333333')
    ax1.set_title('Combined Threat Heatmap (Radar + SAM)', fontsize=12, 
                 fontweight='bold', color='#333333', pad=15)
    ax1.set_facecolor('#F5F5F5')
    ax1.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
    cbar1 = plt.colorbar(im1, ax=ax1, label='Threat Level (0=Safe, 1=Dangerous)')
    cbar1.ax.tick_params(labelsize=9)
    
    # Plot 2: Radar detection probability
    im2 = ax2.imshow(threat_model.radar_detection_prob, cmap='RdYlGn_r', 
                     origin='lower', vmin=0, vmax=1, alpha=0.85)
    
    ax2.set_xlabel('Latitude Index', fontsize=11, color='#333333')
    ax2.set_ylabel('Longitude Index', fontsize=11, color='#333333')
    ax2.set_title('Radar Detection Probability', fontsize=12, 
                 fontweight='bold', color='#333333', pad=15)
    ax2.set_facecolor('#F5F5F5')
    ax2.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
    cbar2 = plt.colorbar(im2, ax=ax2, label='Detection Probability')
    cbar2.ax.tick_params(labelsize=9)
    
    # Add radar sites
    for threat in threat_model.threat_sources:
        if 'RADAR' in threat.threat_type.value:
            circle = patches.Circle((threat.lon_idx, threat.lat_idx), 
                                   threat.range_m / threat_model.resolution_m,
                                   fill=False, edgecolor='#0066CC', 
                                   linewidth=1.5, linestyle='--', alpha=0.5)
            ax2.add_patch(circle)
    
    plt.tight_layout()
    return fig


def plot_sam_coverage_matplotlib(threat_model) -> Figure:
    """
    Plot SAM engagement range map.
    
    Args:
        threat_model: ThreatAssessmentModel instance
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    sam_data = threat_model.sam_range_map / 1000  # Convert to km
    
    im = ax.imshow(sam_data, cmap='YlOrRd', origin='lower', alpha=0.85)
    
    # Draw SAM coverage circles
    for threat in threat_model.threat_sources:
        if 'SAM' in threat.threat_type.value:
            circle = patches.Circle((threat.lon_idx, threat.lat_idx),
                                   threat.range_m / threat_model.resolution_m,
                                   fill=False, edgecolor='#D32F2F', 
                                   linewidth=2.5, linestyle='-', alpha=0.8)
            ax.add_patch(circle)
            
            # SAM site marker
            ax.plot(threat.lon_idx, threat.lat_idx, marker='s', markersize=15,
                   color='#D32F2F', markeredgecolor='black', markeredgewidth=1.5,
                   zorder=5, label=threat.threat_id)
    
    ax.set_xlabel('Latitude Index', fontsize=12, color='#333333', fontweight='bold')
    ax.set_ylabel('Longitude Index', fontsize=12, color='#333333', fontweight='bold')
    ax.set_title('SAM Engagement Range Coverage', fontsize=13, 
                fontweight='bold', color='#333333', pad=15)
    ax.set_facecolor('#F5F5F5')
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
    ax.legend(fontsize=10, loc='upper right')
    
    cbar = plt.colorbar(im, ax=ax, label='Effective Range (km)')
    cbar.ax.tick_params(labelsize=10)
    
    plt.tight_layout()
    return fig


def plot_safe_corridors_matplotlib(threat_model, threshold: float = 0.3) -> Figure:
    """
    Plot identified safe corridors for path planning.
    
    Args:
        threat_model: ThreatAssessmentModel instance
        threshold: Threat threshold for safe areas
        
    Returns:
        Matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    safe_mask = threat_model.find_safe_corridors(threshold)
    threat_data = threat_model.threat_heatmap
    
    # Plot 1: Safe corridors overlay
    color_scheme = ThreatColorScheme()
    im1 = ax1.imshow(threat_data, cmap=color_scheme.cmap, origin='lower', 
                     vmin=0, vmax=1, alpha=0.7)
    
    # Overlay safe areas
    safe_overlay = np.ma.masked_where(safe_mask == 0, safe_mask)
    ax1.imshow(safe_overlay, cmap='Greens', origin='lower', alpha=0.4, vmin=0, vmax=1)
    
    ax1.set_xlabel('Latitude Index', fontsize=11, color='#333333')
    ax1.set_ylabel('Longitude Index', fontsize=11, color='#333333')
    ax1.set_title(f'Safe Corridors (Threat < {threshold})', fontsize=12,
                 fontweight='bold', color='#333333', pad=15)
    ax1.set_facecolor('#F5F5F5')
    ax1.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
    
    # Plot 2: Binary safe/threatened map
    im2 = ax2.imshow(safe_mask, cmap='RdYlGn', origin='lower', alpha=0.85, vmin=0, vmax=1)
    
    # Add threat sources
    for threat in threat_model.threat_sources:
        marker = '*' if 'SAM' in threat.threat_type.value else '^'
        color = '#D32F2F' if 'SAM' in threat.threat_type.value else '#0066CC'
        ax2.plot(threat.lon_idx, threat.lat_idx, marker=marker, markersize=18,
                color=color, markeredgecolor='black', markeredgewidth=1.5, zorder=5)
    
    ax2.set_xlabel('Latitude Index', fontsize=11, color='#333333')
    ax2.set_ylabel('Longitude Index', fontsize=11, color='#333333')
    ax2.set_title(f'Safe (Green) vs. Threatened (Red) Areas', fontsize=12,
                 fontweight='bold', color='#333333', pad=15)
    ax2.set_facecolor('#F5F5F5')
    ax2.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
    
    cbar = plt.colorbar(im2, ax=ax2, label='Safety Status')
    cbar.set_ticks([0.25, 0.75])
    cbar.set_ticklabels(['Threatened', 'Safe'])
    
    plt.tight_layout()
    return fig


def plot_threat_statistics_matplotlib(threat_model) -> Figure:
    """
    Plot threat statistics and area analysis.
    
    Args:
        threat_model: ThreatAssessmentModel instance
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    threat_data = threat_model.threat_heatmap.flatten()
    
    # Plot 1: Histogram of threat levels
    ax = axes[0, 0]
    ax.hist(threat_data, bins=50, color='#FF6F00', alpha=0.7, edgecolor='#D32F2F')
    ax.axvline(np.mean(threat_data), color='#D32F2F', linestyle='--', linewidth=2, 
              label=f'Mean: {np.mean(threat_data):.2f}')
    ax.set_xlabel('Threat Level', fontsize=11, color='#333333')
    ax.set_ylabel('Frequency', fontsize=11, color='#333333')
    ax.set_title('Distribution of Threat Levels', fontsize=12, fontweight='bold', 
                color='#333333', pad=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--', linewidth=0.5)
    ax.set_facecolor('#F5F5F5')
    
    # Plot 2: Cumulative threat
    ax = axes[0, 1]
    threat_sorted = np.sort(threat_data)
    cumulative = np.arange(1, len(threat_sorted) + 1) / len(threat_sorted)
    ax.plot(threat_sorted, cumulative, linewidth=2.5, color='#D32F2F')
    ax.fill_between(threat_sorted, 0, cumulative, alpha=0.3, color='#FF6F00')
    ax.set_xlabel('Threat Level', fontsize=11, color='#333333')
    ax.set_ylabel('Cumulative Probability', fontsize=11, color='#333333')
    ax.set_title('Cumulative Threat Distribution', fontsize=12, fontweight='bold',
                color='#333333', pad=10)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_facecolor('#F5F5F5')
    
    # Plot 3: Area by threat level
    ax = axes[1, 0]
    threat_levels = ['Very Safe\n(<0.2)', 'Safe\n(0.2-0.4)', 'Moderate\n(0.4-0.6)',
                    'High\n(0.6-0.8)', 'Critical\n(>0.8)']
    area_pcts = [
        100 * np.sum(threat_data < 0.2) / len(threat_data),
        100 * np.sum((threat_data >= 0.2) & (threat_data < 0.4)) / len(threat_data),
        100 * np.sum((threat_data >= 0.4) & (threat_data < 0.6)) / len(threat_data),
        100 * np.sum((threat_data >= 0.6) & (threat_data < 0.8)) / len(threat_data),
        100 * np.sum(threat_data >= 0.8) / len(threat_data),
    ]
    colors_bar = ['#66BB6A', '#FDD835', '#FB8C00', '#E65100', '#D32F2F']
    bars = ax.bar(threat_levels, area_pcts, color=colors_bar, alpha=0.8, edgecolor='#333333')
    ax.set_ylabel('Area Coverage (%)', fontsize=11, color='#333333')
    ax.set_title('Area Distribution by Threat Level', fontsize=12, fontweight='bold',
                color='#333333', pad=10)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--', linewidth=0.5)
    ax.set_facecolor('#F5F5F5')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Plot 4: Statistics text
    ax = axes[1, 1]
    ax.axis('off')
    
    stats = threat_model.get_statistics()
    stats_text = f"""
THREAT STATISTICS

Mean Threat Level:        {stats['mean_threat']:.3f}
Max Threat Level:         {stats['max_threat']:.3f}
Std Deviation:            {stats['std_threat']:.3f}

High Threat Areas:        {stats['area_high_threat_pct']:.1f}%
Low Threat Areas:         {stats['area_low_threat_pct']:.1f}%

Threat Sources:           {len(threat_model.threat_sources)}
- Radar Sites:            {sum(1 for t in threat_model.threat_sources if 'RADAR' in t.threat_type.value)}
- SAM Systems:            {sum(1 for t in threat_model.threat_sources if 'SAM' in t.threat_type.value)}
    """
    
    ax.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
           verticalalignment='center', bbox=dict(boxstyle='round', 
           facecolor='#E8F5E9', alpha=0.8, edgecolor='#558B2F', linewidth=2))
    
    plt.tight_layout()
    return fig


def plot_threat_3d_altitude_analysis_plotly(threat_model) -> go.Figure:
    """
    Interactive 3D visualization of threat vs altitude.
    
    Args:
        threat_model: ThreatAssessmentModel instance
        
    Returns:
        Plotly figure
    """
    altitudes = np.linspace(0, 5000, 20)
    threat_by_alt = []
    
    for alt in altitudes:
        center_idx = threat_model.grid_size // 2
        threat_at_alt = threat_model.get_threat_at_location(center_idx, center_idx, alt)
        threat_by_alt.append(threat_at_alt['combined_threat_score'])
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=altitudes,
        y=threat_by_alt,
        mode='lines+markers',
        name='Threat Level',
        line=dict(color='#D32F2F', width=3),
        marker=dict(size=8, color='#D32F2F', symbol='circle')
    ))
    
    fig.add_hrect(y0=0, y1=0.3, fillcolor='#66BB6A', opacity=0.2, 
                  annotation_text='Low Threat', annotation_position='right')
    fig.add_hrect(y0=0.3, y1=0.7, fillcolor='#FDD835', opacity=0.2,
                  annotation_text='Medium Threat', annotation_position='right')
    fig.add_hrect(y0=0.7, y1=1, fillcolor='#D32F2F', opacity=0.2,
                  annotation_text='High Threat', annotation_position='right')
    
    fig.update_layout(
        title='Threat Level vs. Altitude (Center of Domain)',
        xaxis_title='Altitude (m)',
        yaxis_title='Threat Level (0-1)',
        template='plotly_white',
        hovermode='closest',
        font=dict(size=11, color='#333333'),
        plot_bgcolor='#F9F8F6',
        paper_bgcolor='#F9F8F6',
        height=500,
    )
    
    return fig
