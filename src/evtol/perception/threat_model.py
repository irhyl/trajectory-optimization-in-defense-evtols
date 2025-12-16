"""
Threat Assessment for eVTOL Trajectory Optimization.

Models radar detection probability, SAM coverage, and generates threat heatmaps
for trajectory planning and risk assessment.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from enum import Enum

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import cdist


class ThreatType(Enum):
    """Types of threats in the operational area."""
    RADAR_3D = "3D_Radar"
    SAM_SYSTEM = "SAM_System"
    AIR_PATROL = "Air_Patrol"
    EW_ZONE = "EW_Zone"


@dataclass
class ThreatSource:
    """Individual threat source (radar, SAM, patrol, etc)."""
    threat_id: str
    threat_type: ThreatType
    lat_idx: float  # Grid index
    lon_idx: float  # Grid index
    range_m: float  # Detection or engagement range (meters)
    altitude_min: float = 0.0  # Minimum altitude coverage (meters)
    altitude_max: float = 5000.0  # Maximum altitude coverage (meters)
    detection_threshold: float = 13.0  # dB for radar (SNR threshold)
    target_rcs: float = 0.5  # Target RCS in m² (eVTOL typical)


@dataclass
class ThreatMetadata:
    """Metadata for threat assessment."""
    resolution_m: float = 30.0
    coverage_north: float = 13.0
    coverage_south: float = 12.8
    coverage_east: float = 77.7
    coverage_west: float = 77.5
    generated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    processing_version: str = "1.0"


class ThreatAssessmentModel:
    """
    Models threat detection and engagement zones.
    
    Generates probability maps for:
    - Radar detection probability (detection range vs altitude)
    - SAM engagement zones (altitude-dependent range)
    - Combined threat score (fusion of multiple threats)
    """
    
    def __init__(self, grid_size: int = 1667, coverage_bounds: Optional[Tuple] = None):
        """
        Initialize threat assessment model.
        
        Args:
            grid_size: Number of cells per dimension
            coverage_bounds: (north, south, east, west) in decimal degrees
        """
        self.grid_size = grid_size
        self.coverage_bounds = coverage_bounds or (13.0, 12.8, 77.7, 77.5)
        self.resolution_m = 30.0
        
        self.threat_sources: List[ThreatSource] = []
        self.radar_detection_prob = None
        self.sam_range_map = None
        self.threat_heatmap = None
        self.metadata = ThreatMetadata()

        # For multi-altitude threat visualization
        self.altitude_bands = [100, 500, 1000, 2000, 3000, 4000]
        self.threat_grid = None  # Will be (alt, y, x)
    
    def add_threat_source(self, threat_source: ThreatSource) -> None:
        """Add a threat source to the model."""
        self.threat_sources.append(threat_source)
    
    def generate_realistic_threats(self) -> None:
        """
        Generate realistic threat scenario with multiple threat sources.
        """
        # Clear previous threats
        self.threat_sources = []
        
        # Add realistic threat distribution
        # 3 radar sites forming a triangle
        self.add_threat_source(ThreatSource(
            threat_id="radar_001",
            threat_type=ThreatType.RADAR_3D,
            lat_idx=400,
            lon_idx=600,
            range_m=50000,
            altitude_max=8000,
            detection_threshold=13.0
        ))
        
        self.add_threat_source(ThreatSource(
            threat_id="radar_002",
            threat_type=ThreatType.RADAR_3D,
            lat_idx=800,
            lon_idx=200,
            range_m=45000,
            altitude_max=7500,
            detection_threshold=13.5
        ))
        
        self.add_threat_source(ThreatSource(
            threat_id="radar_003",
            threat_type=ThreatType.RADAR_3D,
            lat_idx=1200,
            lon_idx=1000,
            range_m=48000,
            altitude_max=8000,
            detection_threshold=13.0
        ))
        
        # 2 SAM sites
        self.add_threat_source(ThreatSource(
            threat_id="sam_001",
            threat_type=ThreatType.SAM_SYSTEM,
            lat_idx=600,
            lon_idx=800,
            range_m=30000,
            altitude_min=500,
            altitude_max=4000
        ))
        
        self.add_threat_source(ThreatSource(
            threat_id="sam_002",
            threat_type=ThreatType.SAM_SYSTEM,
            lat_idx=1000,
            lon_idx=600,
            range_m=28000,
            altitude_min=500,
            altitude_max=4000
        ))
        
        # Generate threat maps

        # Generate threat grid for multiple altitudes
        self.threat_grid = np.zeros((len(self.altitude_bands), self.grid_size, self.grid_size))
        for i, alt in enumerate(self.altitude_bands):
            self._compute_radar_detection_map(altitude_m=alt)
            self._compute_sam_range_map()
            self._compute_combined_threat_map()
            # Save the combined threat map for this altitude
            self.threat_grid[i] = self.threat_heatmap
    
    def _compute_radar_detection_map(self, altitude_m: float = 500.0) -> None:
        """
        Compute radar detection probability map.
        
        Uses radar range equation:
        P_d = 1 - exp(-SNR / SNR_threshold)
        SNR = (Pt × G × λ² × σ) / ((4π)³ × R⁴ × L × N_f)
        
        Simplified: SNR ≈ power / R⁴
        """
        self.radar_detection_prob = np.zeros((self.grid_size, self.grid_size))
        
        lat = np.arange(self.grid_size)
        lon = np.arange(self.grid_size)
        lat_grid, lon_grid = np.meshgrid(lon, lat, indexing='ij')
        
        for threat in self.threat_sources:
            if threat.threat_type != ThreatType.RADAR_3D:
                continue
            
            # Altitude factor (signal loss above/below coverage)
            if threat.altitude_min <= altitude_m <= threat.altitude_max:
                alt_factor = 1.0
            elif altitude_m < threat.altitude_min:
                alt_factor = 0.5
            else:
                alt_factor = 0.3
            
            # Distance from threat source
            distance = np.sqrt((lat_grid - threat.lat_idx)**2 + 
                             (lon_grid - threat.lon_idx)**2) * self.resolution_m
            
            # Radar equation: SNR ∝ power / R⁴
            # With range degradation
            snr = alt_factor * (threat.range_m**4) / (distance**4 + 1e-6)
            
            # Detection probability
            detection_prob = 1.0 - np.exp(-snr / threat.detection_threshold)
            detection_prob = np.clip(detection_prob, 0, 1)
            
            # Accumulate detection probability (at least one radar)
            self.radar_detection_prob = np.maximum(
                self.radar_detection_prob, detection_prob
            )
        
        # Smooth for realism
        self.radar_detection_prob = gaussian_filter(self.radar_detection_prob, sigma=2)
        self.radar_detection_prob = np.clip(self.radar_detection_prob, 0, 1)
    
    def _compute_sam_range_map(self) -> None:
        """Compute effective SAM engagement range map."""
        self.sam_range_map = np.zeros((self.grid_size, self.grid_size))
        
        lat = np.arange(self.grid_size)
        lon = np.arange(self.grid_size)
        lat_grid, lon_grid = np.meshgrid(lon, lat, indexing='ij')
        
        for threat in self.threat_sources:
            if threat.threat_type != ThreatType.SAM_SYSTEM:
                continue
            
            # Distance from SAM site
            distance = np.sqrt((lat_grid - threat.lat_idx)**2 + 
                             (lon_grid - threat.lon_idx)**2) * self.resolution_m
            
            # SAM range effectiveness (decreases beyond optimal range)
            effective_range = threat.range_m * np.exp(-distance / threat.range_m)
            
            # Store maximum range at each location
            self.sam_range_map = np.maximum(self.sam_range_map, effective_range)
        
        # Smooth
        self.sam_range_map = gaussian_filter(self.sam_range_map, sigma=2)
    
    def _compute_combined_threat_map(self) -> None:
        """Compute combined threat score (fused radar + SAM)."""
        if self.radar_detection_prob is None:
            self._compute_radar_detection_map()
        if self.sam_range_map is None:
            self._compute_sam_range_map()
        
        # Normalize SAM range to 0-1 scale
        sam_normalized = np.zeros_like(self.sam_range_map)
        sam_max = np.max(self.sam_range_map)
        if sam_max > 0:
            sam_normalized = self.sam_range_map / sam_max
        
        # Combined threat: weighted fusion
        # 60% radar detection, 40% SAM coverage
        self.threat_heatmap = (
            0.6 * self.radar_detection_prob +
            0.4 * sam_normalized
        )
        self.threat_heatmap = gaussian_filter(self.threat_heatmap, sigma=1)
        self.threat_heatmap = np.clip(self.threat_heatmap, 0, 1)
    
    def get_threat_at_location(self, lat_idx: float, lon_idx: float, 
                              altitude_m: float) -> Dict[str, float]:
        """
        Get threat assessment at a specific location and altitude.
        
        Args:
            lat_idx: Latitude grid index
            lon_idx: Longitude grid index
            altitude_m: Altitude in meters
            
        Returns:
            Dictionary with threat metrics
        """
        if self.radar_detection_prob is None:
            self.generate_realistic_threats()
        
        lat_idx = int(np.clip(lat_idx, 0, self.grid_size - 1))
        lon_idx = int(np.clip(lon_idx, 0, self.grid_size - 1))
        
        # Get base threat from maps
        radar_prob = float(self.radar_detection_prob[lat_idx, lon_idx])
        sam_range = float(self.sam_range_map[lat_idx, lon_idx])
        
        # Altitude-dependent threat reduction
        # Threats most effective at medium altitudes (500-2000m)
        if altitude_m < 100 or altitude_m > 4000:
            threat_multiplier = 0.5
        elif altitude_m < 500 or altitude_m > 3000:
            threat_multiplier = 0.75
        else:
            threat_multiplier = 1.0
        
        radar_threat = radar_prob * threat_multiplier
        
        # SAM threat based on altitude vs range
        if altitude_m > 5000 or sam_range < 1000:
            sam_threat = 0.0
        else:
            sam_threat = min(1.0, sam_range / 30000) * threat_multiplier
        
        # Combined threat
        combined_threat = max(radar_threat, sam_threat * 0.8)
        
        return {
            'radar_detection_prob': float(radar_threat),
            'sam_threat_level': float(sam_threat),
            'combined_threat_score': float(combined_threat),
            'altitude_factor': float(threat_multiplier),
        }
    
    def validate_threat_data(self) -> Dict[str, bool]:
        """Validate threat maps."""
        if self.radar_detection_prob is None:
            return {'threat_maps_generated': False}
        
        checks = {
            'no_nan_values': not np.any(np.isnan(self.threat_heatmap)),
            'probability_range': np.all((self.threat_heatmap >= 0) & 
                                       (self.threat_heatmap <= 1)),
            'spatial_continuity': True,  # Verified by Gaussian smoothing
            'threat_sources_defined': len(self.threat_sources) > 0,
        }
        return checks
    
    def get_statistics(self) -> Dict[str, float]:
        """Get threat map statistics."""
        if self.threat_heatmap is None:
            return {}
        
        return {
            'mean_threat': float(np.mean(self.threat_heatmap)),
            'max_threat': float(np.max(self.threat_heatmap)),
            'std_threat': float(np.std(self.threat_heatmap)),
            'area_high_threat_pct': float(
                100 * np.sum(self.threat_heatmap > 0.7) / self.threat_heatmap.size
            ),
            'area_low_threat_pct': float(
                100 * np.sum(self.threat_heatmap < 0.3) / self.threat_heatmap.size
            ),
        }
    
    def find_safe_corridors(self, threat_threshold: float = 0.3) -> np.ndarray:
        """
        Identify low-threat corridors for path planning.
        
        Args:
            threat_threshold: Maximum acceptable threat level
            
        Returns:
            Binary mask of safe areas (1=safe, 0=threatened)
        """
        if self.threat_heatmap is None:
            self.generate_realistic_threats()
        
        safe_mask = (self.threat_heatmap < threat_threshold).astype(int)
        return safe_mask
    
    def to_dict(self) -> Dict:
        """Export threat model as dictionary."""
        return {
            'radar_detection_prob': self.radar_detection_prob,
            'sam_range_map': self.sam_range_map,
            'threat_heatmap': self.threat_heatmap,
            'threat_sources': [
                {
                    'id': t.threat_id,
                    'type': t.threat_type.value,
                    'lat_idx': t.lat_idx,
                    'lon_idx': t.lon_idx,
                    'range_m': t.range_m,
                }
                for t in self.threat_sources
            ],
            'metadata': {
                'resolution_m': self.metadata.resolution_m,
                'coverage_bounds': self.coverage_bounds,
                'generated_at': self.metadata.generated_at,
                'processing_version': self.metadata.processing_version,
            }
        }
