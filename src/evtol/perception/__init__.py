"""
Perception and Environment Layer for eVTOL Trajectory Optimization

This package provides spatiotemporal maps for eVTOL planning including:
- Terrain and obstacle data (terrain_model, obstacle_model)
- Atmospheric conditions (wind_model)
- Threat and risk assessment (threat_model)
- Data fusion and uncertainty quantification (fusion_model)

The layer produces accurate, versioned, georeferenced, and fast-to-query
maps for the planner via the FusedIntelligenceModel interface.
"""

# Import production perception modules
from .terrain_model import TerrainElevationMap, TerrainMetadata
from .wind_model import WindFieldModel, WindMetadata
from .threat_model import ThreatAssessmentModel
from .obstacle_model import ObstacleDetectionModel, LandingZone
from .fusion_model import FusedIntelligenceModel, FusionMetadata

# Import visualization modules
from .terrain_visualization import (
    plot_terrain_2d_matplotlib,
    plot_terrain_3d_plotly,
    plot_terrain_statistics,
    plot_cross_section
)
from .wind_visualization import (
    plot_wind_vector_field_matplotlib,
    plot_wind_profile_matplotlib,
    plot_wind_3d_layers_plotly,
    plot_wind_shear_matplotlib,
    plot_energy_cost_map_matplotlib
)
from .threat_visualization import (
    plot_threat_heatmap_matplotlib,
    plot_sam_coverage_matplotlib,
    plot_safe_corridors_matplotlib,
    plot_threat_statistics_matplotlib,
    plot_threat_3d_altitude_analysis_plotly
)
from .obstacle_visualization import (
    plot_building_detection_matplotlib,
    plot_landing_zones_matplotlib,
    plot_clearance_profile_matplotlib,
    plot_obstacle_statistics_matplotlib
)
from .fusion_visualization import (
    plot_fused_risk_map_matplotlib,
    plot_feasibility_and_risk_matplotlib,
    plot_energy_cost_map_matplotlib as plot_fusion_energy_cost_map_matplotlib,
    plot_component_contribution_matplotlib,
    plot_fused_statistics_matplotlib,
    plot_fused_3d_heatmap_plotly,
    plot_comparison_before_after_fusion
)

# Public API - Production modules
__all__ = [
    # Core models
    "TerrainElevationMap",
    "WindFieldModel",
    "ThreatAssessmentModel",
    "ObstacleDetectionModel",
    "FusedIntelligenceModel",
    # Metadata
    "TerrainMetadata",
    "WindMetadata",
    "FusionMetadata",
    # Terrain visualization
    "plot_terrain_2d_matplotlib",
    "plot_terrain_3d_plotly",
    "plot_terrain_statistics",
    "plot_cross_section",
    # Wind visualization
    "plot_wind_vector_field_matplotlib",
    "plot_wind_profile_matplotlib",
    "plot_wind_3d_layers_plotly",
    "plot_wind_shear_matplotlib",
    "plot_energy_cost_map_matplotlib",
    # Threat visualization
    "plot_threat_heatmap_matplotlib",
    "plot_sam_coverage_matplotlib",
    "plot_safe_corridors_matplotlib",
    "plot_threat_statistics_matplotlib",
    "plot_threat_3d_altitude_analysis_plotly",
    # Obstacle visualization
    "plot_building_detection_matplotlib",
    "plot_landing_zones_matplotlib",
    "plot_clearance_profile_matplotlib",
    "plot_obstacle_statistics_matplotlib",
    # Fusion visualization
    "plot_fused_risk_map_matplotlib",
    "plot_feasibility_and_risk_matplotlib",
    "plot_fusion_energy_cost_map_matplotlib",
    "plot_component_contribution_matplotlib",
    "plot_fused_statistics_matplotlib",
    "plot_fused_3d_heatmap_plotly",
    "plot_comparison_before_after_fusion",
    # Data structures
    "LandingZone"
]

# Quick setup function
def setup_perception_layer(config_path: str = None, log_level: str = "INFO"):
    """Quick setup of the perception layer with logging."""
    from .utils.config import Config
    import logging
    
    # Load configuration
    config = Config(config_path)
    
    # Setup logging
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return config, logger

