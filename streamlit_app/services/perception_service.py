"""Perception layer service for Streamlit front-end."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

# Ensure the core src package is importable
ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from evtol.perception import (  # type: ignore  # pylint: disable=import-error
        TerrainElevationMap,
        WindFieldModel,
        ThreatAssessmentModel,
        ObstacleDetectionModel,
        FusedIntelligenceModel,
    )
except ImportError as e:
    print(f"Warning: Could not import perception models: {e}")


class PerceptionService:
    """Wraps all perception layer models with caching and query APIs."""

    def __init__(self, grid_size: int = 512, seed: int = 42):
        """
        Initialize perception layer with all models.

        Args:
            grid_size: Grid resolution (512 for balanced performance/fidelity)
            seed: Random seed for reproducible generation
        """
        self.grid_size = grid_size
        self.seed = seed
        self._models_loaded = False
        self._models: Dict[str, Any] = {}

    def load_all_models(self) -> Dict[str, Any]:
        """
        Lazy-load all perception models (expensive, done once).

        Returns:
            Dictionary with all initialized models
        """
        if self._models_loaded:
            return self._models

        # Terrain - smaller area for memory efficiency
        terrain = TerrainElevationMap(width_m=25600, height_m=25600, resolution_m=50.0)
        terrain.generate_realistic_terrain(seed=self.seed)

        # Wind
        wind = WindFieldModel(grid_size=self.grid_size)
        wind.generate_realistic_wind(base_speed=7.0)

        # Threat
        threat = ThreatAssessmentModel(grid_size=self.grid_size)
        threat.generate_realistic_threats()

        # Obstacles - reduced building density to prevent memory issues
        obstacle = ObstacleDetectionModel(grid_size=self.grid_size)
        obstacle.generate_realistic_obstacles(building_density=0.02)

        # Fusion
        fusion = FusedIntelligenceModel(terrain, wind, threat, obstacle)

        self._models = {
            "terrain": terrain,
            "wind": wind,
            "threat": threat,
            "obstacle": obstacle,
            "fusion": fusion,
        }
        self._models_loaded = True

        return self._models

    def get_terrain_stats(self) -> Dict[str, float]:
        """Get terrain statistics."""
        models = self.load_all_models()
        terrain = models["terrain"]
        dem = terrain.elevation

        return {
            "min_elevation_m": float(np.min(dem)),
            "max_elevation_m": float(np.max(dem)),
            "mean_elevation_m": float(np.mean(dem)),
            "std_elevation_m": float(np.std(dem)),
            "grid_size": self.grid_size,
        }

    def get_wind_stats(self, altitude_idx: int = 1) -> Dict[str, float]:
        """Get wind statistics at specified altitude (default: 100m)."""
        models = self.load_all_models()
        wind = models["wind"]

        if wind.wind_speed is None:
            return {}

        speed_at_alt = wind.wind_speed[altitude_idx, :, :]

        return {
            "min_speed_ms": float(np.min(speed_at_alt)),
            "max_speed_ms": float(np.max(speed_at_alt)),
            "mean_speed_ms": float(np.mean(speed_at_alt)),
            "std_speed_ms": float(np.std(speed_at_alt)),
            "altitude_m": float(wind.altitude_bands[altitude_idx]),
        }

    def get_threat_stats(self) -> Dict[str, Any]:
        """Get threat assessment statistics."""
        models = self.load_all_models()
        threat = models["threat"]

        if threat.threat_heatmap is None:
            return {}

        return {
            "min_threat_score": float(np.min(threat.threat_heatmap)),
            "max_threat_score": float(np.max(threat.threat_heatmap)),
            "mean_threat_score": float(np.mean(threat.threat_heatmap)),
            "high_threat_area_pct": float(
                100.0 * np.sum(threat.threat_heatmap > 0.7) / threat.threat_heatmap.size
            ),
            "num_threat_sources": len(threat.threat_sources),
        }

    def get_obstacle_stats(self) -> Dict[str, Any]:
        """Get obstacle detection statistics."""
        models = self.load_all_models()
        obstacle = models["obstacle"]

        if obstacle.building_mask is None:
            return {}

        return {
            "building_coverage_pct": float(
                100.0 * np.sum(obstacle.building_mask > 0) / obstacle.building_mask.size
            ),
            "max_building_height_m": float(np.max(obstacle.building_height)),
            "mean_clearance_m": float(np.mean(obstacle.clearance_map)),
            "min_clearance_m": float(np.min(obstacle.clearance_map)),
            "num_landing_zones": len(obstacle.landing_zones),
        }

    def get_fused_stats(self) -> Dict[str, Any]:
        """Get fused intelligence statistics."""
        models = self.load_all_models()
        fusion = models["fusion"]

        return fusion.get_statistics()

    def query_location(self, lat_deg: float, lon_deg: float, altitude_m: float = 100.0) -> Dict[str, Any]:
        """
        Query fused intelligence at a specific location.

        Args:
            lat_deg: Latitude in degrees
            lon_deg: Longitude in degrees
            altitude_m: Altitude in meters

        Returns:
            Dictionary with risk, feasibility, energy cost, etc.
        """
        models = self.load_all_models()
        fusion = models["fusion"]

        return fusion.get_fused_query(lat_deg, lon_deg, altitude_m)

    def get_corridor_summary(self, waypoints: list[Dict[str, float]]) -> Dict[str, Any]:
        """
        Analyze a corridor (list of waypoints) and return summary stats.

        Args:
            waypoints: List of dicts with 'lat', 'lon', 'alt_m' keys

        Returns:
            Summary statistics along the corridor
        """
        if not waypoints:
            return {}

        models = self.load_all_models()
        fusion = models["fusion"]

        corridor_risk = []
        corridor_feasibility = []
        corridor_energy = []

        for wp in waypoints:
            query = fusion.get_fused_query(
                wp.get("lat"),
                wp.get("lon"),
                wp.get("alt_m", 100.0),
            )
            if query.get("valid"):
                corridor_risk.append(query.get("risk_score", 0.5))
                corridor_feasibility.append(query.get("feasibility_score", 0.5))
                corridor_energy.append(query.get("energy_cost", 5.0))

        if not corridor_risk:
            return {}

        return {
            "mean_risk_score": float(np.mean(corridor_risk)),
            "max_risk_score": float(np.max(corridor_risk)),
            "mean_feasibility": float(np.mean(corridor_feasibility)),
            "min_feasibility": float(np.min(corridor_feasibility)),
            "mean_energy_cost": float(np.mean(corridor_energy)),
            "max_energy_cost": float(np.max(corridor_energy)),
            "waypoint_count": len(waypoints),
        }

    def get_models(self) -> Dict[str, Any]:
        """Return raw model objects for advanced visualization."""
        return self.load_all_models()


__all__ = ["PerceptionService"]
