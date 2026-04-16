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

# Sub-package imports are done lazily in each sub-package's own module.
# Eager imports from this __init__ are intentionally omitted because
# several top-level model files (terrain_model, wind_model, etc.) have not
# been ported to the current sub-package layout.  The dataset pipeline
# scripts import from sub-packages directly, e.g.:
#   from evtol.perception.terrain.data_provider import TerrainDataProvider
# and do not require anything from this __init__.

__all__: list[str] = []

