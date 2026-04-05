"""
Threat Analysis Module for Defense eVTOL Planning

This module implements threat detection and avoidance for mission planning.
It provides:

1. Threat field model (air defense, radar)
2. Trajectory risk computation
3. Threat-aware path scoring for RRT*
4. NSGA-III Pareto objective (threat exposure)

Threat Models
=============

1. **Air Defense (SAM/SHORAD)**:
   - Engagement zone: sphere of radius R_kill
   - Effective zone: sphere of radius R_eff (50% LOP)
   - Probability of kill: PK(r) = exp(-(r/R_eff))
   - Multiple systems increase cumulative probability

2. **Radar Detection**:
   - Detection range: radar cross-section (RCS) dependent
   - Stealth factor: 10 dB gain from altitude/terrain masking
   - Cumulative detection: if seen by any radar

3. **Enemy Fighter Coverage**:
   - Engagement envelope: Mach-dependent
   - Turn performance: g-limited turn radius
   - Combat maneuver zone

Risk Integration:
- Cumulative PK: 1 - ∏(1 - PK_i) for each threat
- Temporal: higher risk in sustained engagement zones
- Spatial: integrated along entire trajectory

Author: Defense eVTOL Research Team
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from collections.abc import Callable
import logging

from .trajectory import Trajectory
from .state import State, Pose

logger = logging.getLogger(__name__)


class ThreatType(Enum):
    """Types of threats."""
    SAM = "sam"  # Surface-to-air missile
    AAA = "aaa"  # Anti-aircraft artillery
    RADAR = "radar"  # Radar detection
    FIGHTER = "fighter"  # Enemy aerial vehicle


@dataclass
class ThreatZone:
    """
    Definition of a single threat.
    
    Attributes:
        zone_id: Unique identifier
        position: 3D position [x, y, z] in mission frame [m]
        threat_type: Type of threat
        kill_radius: Radius of certain kill [m]
        effective_radius: Radius of 50% PK [m]
        altitude_weight: Influence of altitude on threat (0-1)
        rcs_target: Radar cross-section in dBsm
        detection_enabled: Whether detection is active
        engagement_enabled: Whether engagement is possible
        threat_level: Subjective threat severity (1-10)
    """
    zone_id: str
    position: np.ndarray
    threat_type: ThreatType = ThreatType.SAM
    kill_radius: float = 500.0  # m
    effective_radius: float = 2000.0  # m
    altitude_weight: float = 0.7  # Altitude reduces threat
    rcs_target: float = 1.0  # dBsm
    detection_enabled: bool = True
    engagement_enabled: bool = True
    threat_level: int = 5
    
    def __post_init__(self):
        self.position = np.asarray(self.position)
    
    def engagement_probability(self, distance: float, altitude: float) -> float:
        """
        Compute probability of kill at given distance and altitude.
        
        Args:
            distance: Distance from threat [m]
            altitude: Target altitude relative to threat [m]
            
        Returns:
            P(kill) in [0, 1]
        """
        if not self.engagement_enabled or distance < 0:
            return 0.0
        
        if distance < self.kill_radius:
            return 1.0
        
        if distance > self.effective_radius:
            return 0.0
        
        # Exponential falloff from kill zone to effective zone
        norm_dist = (distance - self.kill_radius) / (self.effective_radius - self.kill_radius)
        pk = np.exp(-3 * norm_dist)  # 3x exponent gives ~5% PK at eff_radius
        
        # Altitude degradation (higher is safer)
        altitude_factor = max(0, 1 - (altitude / 1000))  # Degrades above 1km
        pk *= (altitude_factor * self.altitude_weight + (1 - self.altitude_weight))
        
        return pk
    
    def detection_range(self, rcs: float = 0.0) -> float:
        """
        Compute detection range for radar threat.
        
        Args:
            rcs: Target RCS in dBsm
            
        Returns:
            Detection range [m]
        """
        if self.threat_type != ThreatType.RADAR:
            return self.effective_radius
        
        # Radar range equation: R = (P * G * σ / (4π S_min))^(1/4)
        # Typical: 100 km at 1 dBsm RCS
        base_range = 100_000  # m at 0 dBsm
        rcs_diff = rcs - self.rcs_target
        range_mult = 10 ** (rcs_diff / 40)  # 40 dB per decade
        
        return base_range * range_mult
    
    def distance_to_position(self, position: np.ndarray) -> float:
        """Compute distance from threat to position."""
        delta = np.asarray(position) - self.position
        return np.linalg.norm(delta)


@dataclass
class ThreatField:
    """
    Collection of threats in mission area.
    
    Represents the overall threat environment.
    """
    threats: dict[str, ThreatZone] = field(default_factory=dict)
    reference_altitude: float = 0.0  # Altitude reference [m]
    threat_density: float = 0.1  # Threats per km²
    time_varying: bool = False  # Whether threats move over time
    
    def add_threat(self, threat: ThreatZone) -> None:
        """Add threat to field."""
        self.threats[threat.zone_id] = threat
    
    def remove_threat(self, zone_id: str) -> None:
        """Remove threat from field."""
        if zone_id in self.threats:
            del self.threats[zone_id]
    
    def get_nearby_threats(
        self,
        position: np.ndarray,
        radius: float = 5000.0,
    ) -> list[ThreatZone]:
        """
        Get threats within radius of position.
        
        Args:
            position: Center position
            radius: Search radius [m]
            
        Returns:
            List of nearby threats
        """
        nearby = []
        position = np.asarray(position)
        
        for threat in self.threats.values():
            dist = np.linalg.norm(threat.position - position)
            if dist < radius:
                nearby.append(threat)
        
        return nearby


class ThreatAnalyzer:
    """
    Threat analysis for trajectory optimization.
    
    Computes:
    - Point-wise threat exposure along trajectory
    - Cumulative risk (integrated threat)
    - Path scores for RRT* goal scoring
    - Pareto objectives for multi-objective optimization
    """
    
    def __init__(
        self,
        threat_field: ThreatField,
        aircraft_rcs: float = 1.0,
    ):
        """
        Initialize threat analyzer.
        
        Args:
            threat_field: Threat environment
            aircraft_rcs: Aircraft RCS in dBsm
        """
        self.field = threat_field
        self.aircraft_rcs = aircraft_rcs
    
    def analyze_trajectory(
        self,
        trajectory: Trajectory,
    ) -> dict:
        """
        Comprehensive threat analysis of trajectory.
        
        Args:
            trajectory: Trajectory to analyze
            
        Returns:
            Dict with analysis results:
            {
                'max_threat': float,  # Worst point along trajectory
                'mean_threat': float,  # Average threat exposure
                'cumulative_threat': float,  # Integrated threat
                'pk_cumulative': float,  # Cumulative PK
                'time_in_danger': float,  # Time in high-threat zone
                'threat_events': list,  # Specific threat encounters
            }
        """
        max_threat = 0.0
        mean_threat = 0.0
        cumulative_threat = 0.0
        time_in_danger = 0.0
        threat_events = []
        
        point_threats = []
        
        for segment in trajectory.segments:
            # Sample segment at multiple points
            n_samples = max(5, int(segment.duration / 2))
            for i in range(n_samples + 1):
                t_frac = i / max(1, n_samples)
                t = segment.start.time + t_frac * segment.duration
                
                # Interpolate state
                pos = segment.start.pose.position * (1 - t_frac) + \
                      segment.end.pose.position * t_frac
                alt = pos[2] - self.field.reference_altitude
                
                # Compute threat at this point
                point_threat = self._evaluate_point(pos)
                point_threats.append(point_threat)
                
                # Track statistics
                cumulative_threat += point_threat * (segment.duration / n_samples)
                max_threat = max(max_threat, point_threat)
                
                if point_threat > 0.5:
                    time_in_danger += segment.duration / n_samples
                
                # Check for specific threats
                if point_threat > 0.3:
                    for threat_id, threat in self.field.threats.items():
                        pk = threat.engagement_probability(
                            threat.distance_to_position(pos),
                            alt,
                        )
                        if pk > 0.1:
                            threat_events.append({
                                'threat_id': threat_id,
                                'time': t,
                                'position': pos.copy(),
                                'pk': pk,
                            })
        
        if point_threats:
            mean_threat = np.mean(point_threats)
        
        # Cumulative PK from all threats
        pk_cumulative = self._cumulative_pk(threat_events)
        
        return {
            'max_threat': max_threat,
            'mean_threat': mean_threat,
            'cumulative_threat': cumulative_threat,
            'time_in_danger': time_in_danger,
            'pk_cumulative': pk_cumulative,
            'threat_events': threat_events,
        }
    
    def _evaluate_point(self, position: np.ndarray) -> float:
        """
        Evaluate cumulative threat at single point.
        
        Args:
            position: 3D position
            
        Returns:
            Threat level [0, 1]
        """
        pk_total = 0.0
        alt = position[2] - self.field.reference_altitude
        
        for threat in self.field.threats.values():
            if not threat.engagement_enabled:
                continue
            
            distance = threat.distance_to_position(position)
            pk = threat.engagement_probability(distance, alt)
            
            # Cumulative PK
            pk_total = pk_total + pk * (1 - pk_total)
        
        return pk_total
    
    def _cumulative_pk(self, threat_events: list) -> float:
        """Compute cumulative probability of kill from events."""
        if not threat_events:
            return 0.0
        
        # Group by threat and use worst PK
        worst_by_threat = {}
        for event in threat_events:
            threat_id = event['threat_id']
            pk = event['pk']
            if threat_id not in worst_by_threat:
                worst_by_threat[threat_id] = pk
            else:
                worst_by_threat[threat_id] = max(worst_by_threat[threat_id], pk)
        
        # Combine across threats
        pk_total = 0.0
        for pk in worst_by_threat.values():
            pk_total = pk_total + pk * (1 - pk_total)
        
        return pk_total
    
    def threat_objective(self, trajectory: Trajectory) -> float:
        """
        Objective function for NSGA-III (minimize threat).
        
        Args:
            trajectory: Candidate trajectory
            
        Returns:
            Threat metric (to be minimized). Lower is better.
        """
        analysis = self.analyze_trajectory(trajectory)
        
        # Weighted combination of threat metrics
        # - Primary: cumulative threat integrated over path
        # - Secondary: peak threat avoidance
        # - Tertiary: time in danger zones
        
        threat_score = (
            analysis['cumulative_threat'] * 0.5 +
            analysis['max_threat'] * 100 * 0.3 +
            analysis['time_in_danger'] * 10 * 0.2
        )
        
        return threat_score
    
    def is_safe_passage(
        self,
        trajectory: Trajectory,
        max_pk: float = 0.2,
    ) -> tuple[bool, str]:
        """
        Check if trajectory meets safety criteria.
        
        Args:
            trajectory: Candidate trajectory
            max_pk: Maximum acceptable cumulative PK
            
        Returns:
            (safe, reason)
        """
        analysis = self.analyze_trajectory(trajectory)
        
        if analysis['pk_cumulative'] > max_pk:
            return False, f"Cumulative PK {analysis['pk_cumulative']:.1%} exceeds limit"
        
        if analysis['max_threat'] > 0.8:
            return False, "Peak threat too high"
        
        return True, "Safe passage"
