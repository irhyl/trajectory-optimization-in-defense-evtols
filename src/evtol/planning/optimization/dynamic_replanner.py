"""
Dynamic Replanning Module (Phase 2.2)

Real-time threat-aware trajectory modification and replanning.
Three-tier approach: heading adjustment (fast) → waypoint modification (medium) → full replanning (slow).

Author: Defense eVTOL Team
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum
import math


class ReplanningMode(Enum):
    """Replanning strategy tier."""
    HEADING_ADJUSTMENT = 1  # Immediate turn, <100ms
    WAYPOINT_MODIFICATION = 2  # Regenerate trajectory, <500ms
    FULL_REPLANNING = 3  # Global optimization, 1-2s background


@dataclass
class ReplanningDecision:
    """Output from replanning optimizer."""
    should_replan: bool
    replanning_mode: ReplanningMode
    recommended_heading_deg: Optional[float] = None
    modified_waypoints: Optional[List[np.ndarray]] = None
    cost_reduction_pct: float = 0.0
    reason: str = ""
    confidence: float = 0.0  # 0-1, how confident in this recommendation


class ThreatAwarePathOptimizer:
    """
    Real-time threat-aware path optimizer for dynamic replanning.
    
    Responsibilities:
    - Analyze current trajectory against threat/obstacle field
    - Generate alternative headings
    - Modify waypoints to avoid threats
    - Output replanning decisions for integration with trajectory tracker
    
    Computational Cost: 
    - Heading adjustment: <1ms
    - Waypoint modification: 50-100ms
    - Full replanning: 500-1000ms (background)
    """
    
    def __init__(self):
        """Initialize path optimizer."""
        
        # Heading adjustment parameters
        self.HEADING_SEARCH_GRANULE_DEG = 15  # Try ±15, ±30, ±45 degree offsets
        self.HEADING_SAMPLE_COUNT = 7  # -45, -30, -15, 0, 15, 30, 45 degrees
        self.MIN_HEADING_CHANGE_DEG = 20  # Require at least 20 degree deviation
        self.MAX_HEADING_CHANGE_DEG = 90  # Don't turn more than 90 degrees
        
        # Waypoint modification parameters
        self.WAYPOINT_INSERT_DISTANCE_M = 2000  # Insert safety waypoint 2km away
        self.WAYPOINT_LATERAL_OFFSET_M = 1000  # Offset perpendicular to threat
        self.TRAJECTORY_SAMPLE_INTERVAL_S = 5
        
        # Cost evaluation
        self.COST_IMPROVEMENT_THRESHOLD = 0.8  # Need 20% cost reduction to trigger replan
        self.HEADING_LOOKAHEAD_S = 30  # Look 30 seconds ahead for heading adjustment
        
        # Statistics
        self.heading_adjustments_suggested = 0
        self.waypoint_mods_suggested = 0
        self.full_replans_suggested = 0
    
    def evaluate_trajectory(self, current_pos_ned: np.ndarray,
                           current_heading_deg: float,
                           current_velocity_mps: float,
                           trajectory_waypoints: List[np.ndarray],
                           threat_cost_field: Optional[np.ndarray] = None,
                           obstacle_cost_field: Optional[np.ndarray] = None,
                           threat_positions: Optional[List[np.ndarray]] = None) \
            -> ReplanningDecision:
        """
        Evaluate if current trajectory is optimal given threats/obstacles.
        
        If threats detected ahead, search for better heading or waypoints.
        
        Args:
            current_pos_ned: Vehicle position [north, east, down]
            current_heading_deg: Heading in degrees (0=North, 90=East, etc.)
            current_velocity_mps: Velocity magnitude in m/s
            trajectory_waypoints: Planned waypoints [list of [north, east, down] arrays]
            threat_cost_field: Optional pre-computed threat cost grid
            obstacle_cost_field: Optional pre-computed obstacle cost grid
            threat_positions: Optional list of active threat positions
        
        Returns:
            ReplanningDecision with recommendations
        """
        
        # Evaluate current trajectory cost
        current_cost = self._evaluate_trajectory_cost(
            trajectory_waypoints, threat_cost_field, obstacle_cost_field
        )
        
        # Strategy 1: Heading Adjustment (immediate response)
        heading_decision = self._search_heading_adjustment(
            current_pos_ned, current_heading_deg, current_velocity_mps,
            trajectory_waypoints, current_cost,
            threat_positions
        )
        
        if heading_decision.should_replan and heading_decision.confidence > 0.7:
            self.heading_adjustments_suggested += 1
            return heading_decision
        
        # Strategy 2: Waypoint Modification (for significant threats)
        waypoint_decision = self._search_waypoint_modification(
            current_pos_ned, trajectory_waypoints, current_cost,
            threat_positions
        )
        
        if waypoint_decision.should_replan and waypoint_decision.confidence > 0.6:
            self.waypoint_mods_suggested += 1
            return waypoint_decision
        
        # No replanning needed
        return ReplanningDecision(
            should_replan=False,
            replanning_mode=ReplanningMode.HEADING_ADJUSTMENT,
            reason="Trajectory optimal"
        )
    
    def _search_heading_adjustment(self, current_pos_ned: np.ndarray,
                                  current_heading_deg: float,
                                  current_velocity_mps: float,
                                  trajectory_waypoints: List[np.ndarray],
                                  baseline_cost: float,
                                  threat_positions: Optional[List[np.ndarray]]) \
            -> ReplanningDecision:
        """Search for better heading by trying ±30, ±45 degree offsets."""
        
        if not threat_positions or len(threat_positions) == 0:
            return ReplanningDecision(
                should_replan=False,
                replanning_mode=ReplanningMode.HEADING_ADJUSTMENT
            )
        
        # Check if threats ahead
        nearest_threat_distance = self._nearest_threat_distance(
            current_pos_ned, trajectory_waypoints, threat_positions
        )
        
        # If threats more than 10km away, no immediate replanning
        if nearest_threat_distance > 10000:
            return ReplanningDecision(
                should_replan=False,
                replanning_mode=ReplanningMode.HEADING_ADJUSTMENT
            )
        
        # Search heading space
        best_heading = None
        best_cost = baseline_cost
        best_cost_reduction = 0.0
        
        # Sample headings: current ±45 degrees in 15-degree increments
        heading_samples = [
            current_heading_deg + offset * self.HEADING_SEARCH_GRANULE_DEG
            for offset in range(-3, 4)
        ]
        
        for test_heading in heading_samples:
            # Project trajectory 30 seconds ahead with this heading
            projected_path = self._project_path_with_heading(
                current_pos_ned, test_heading, current_velocity_mps,
                self.HEADING_LOOKAHEAD_S
            )
            
            # Compute cost of projected path
            cost = self._evaluate_trajectory_segment_cost(
                projected_path, threat_positions
            )
            
            cost_reduction = (baseline_cost - cost) / (baseline_cost + 1e-6)
            if cost_reduction > best_cost_reduction:
                best_heading = test_heading
                best_cost = cost
                best_cost_reduction = cost_reduction
        
        # Decision
        if best_heading is not None and best_cost_reduction > (1.0 - self.COST_IMPROVEMENT_THRESHOLD):
            heading_change = self._normalize_heading_change(
                best_heading - current_heading_deg
            )
            
            if abs(heading_change) >= self.MIN_HEADING_CHANGE_DEG:
                self.heading_adjustments_suggested += 1
                
                return ReplanningDecision(
                    should_replan=True,
                    replanning_mode=ReplanningMode.HEADING_ADJUSTMENT,
                    recommended_heading_deg=best_heading,
                    cost_reduction_pct=best_cost_reduction * 100,
                    reason=f"Threat {nearest_threat_distance/1000:.1f}km ahead, heading={best_heading:.1f}°",
                    confidence=min(1.0, 0.5 + best_cost_reduction)
                )
        
        return ReplanningDecision(
            should_replan=False,
            replanning_mode=ReplanningMode.HEADING_ADJUSTMENT
        )
    
    def _search_waypoint_modification(self, current_pos_ned: np.ndarray,
                                     trajectory_waypoints: List[np.ndarray],
                                     baseline_cost: float,
                                     threat_positions: Optional[List[np.ndarray]]) \
            -> ReplanningDecision:
        """Search for better trajectory by modifying waypoints."""
        
        if not threat_positions or len(threat_positions) == 0:
            return ReplanningDecision(
                should_replan=False,
                replanning_mode=ReplanningMode.WAYPOINT_MODIFICATION
            )
        
        # Find closest threat to trajectory
        closest_threat_idx, closest_threat_pos, threat_proximity = self._find_nearest_threat_to_trajectory(
            trajectory_waypoints, threat_positions
        )
        
        if threat_proximity > 5000:  # Threat more than 5km away
            return ReplanningDecision(
                should_replan=False,
                replanning_mode=ReplanningMode.WAYPOINT_MODIFICATION
            )
        
        # Generate modified waypoints with safety detour
        modified_waypoints = self._generate_detour_waypoints(
            trajectory_waypoints, closest_threat_pos, closest_threat_idx
        )
        
        # Evaluate modified trajectory
        modified_cost = self._evaluate_trajectory_cost(modified_waypoints)
        cost_reduction = (baseline_cost - modified_cost) / (baseline_cost + 1e-6)
        
        if cost_reduction > (1.0 - self.COST_IMPROVEMENT_THRESHOLD):
            self.waypoint_mods_suggested += 1
            
            return ReplanningDecision(
                should_replan=True,
                replanning_mode=ReplanningMode.WAYPOINT_MODIFICATION,
                modified_waypoints=modified_waypoints,
                cost_reduction_pct=cost_reduction * 100,
                reason=f"Threat {threat_proximity/1000:.1f}km away, inserting detour waypoint",
                confidence=min(1.0, 0.6 + cost_reduction)
            )
        
        return ReplanningDecision(
            should_replan=False,
            replanning_mode=ReplanningMode.WAYPOINT_MODIFICATION
        )
    
    def _project_path_with_heading(self, start_pos_ned: np.ndarray,
                                  heading_deg: float,
                                  speed_mps: float,
                                  duration_s: float) -> List[np.ndarray]:
        """Project path forward with constant heading and speed."""
        
        heading_rad = math.radians(heading_deg)
        north_component = speed_mps * math.cos(heading_rad)
        east_component = speed_mps * math.sin(heading_rad)
        
        path = [start_pos_ned.copy()]
        
        step_count = int(duration_s)
        for _ in range(step_count):
            next_pos = path[-1].copy()
            next_pos[0] += north_component
            next_pos[1] += east_component
            path.append(next_pos)
        
        return path
    
    def _evaluate_trajectory_segment_cost(self, path: List[np.ndarray],
                                         threat_positions: Optional[List[np.ndarray]]) -> float:
        """Quick cost evaluation for trajectory segment."""
        
        if not threat_positions:
            return 0.0
        
        total_cost = 0.0
        for point in path:
            for threat_pos in threat_positions:
                distance = np.linalg.norm(point - threat_pos)
                # Cost inversed by distance (closer = higher cost)
                cost = 1.0 / (distance / 1000.0 + 0.1)  # Scale to km
                total_cost += cost
        
        return total_cost / (len(path) * max(1, len(threat_positions)))
    
    def _evaluate_trajectory_cost(self, waypoints: List[np.ndarray],
                                 threat_field: Optional[np.ndarray] = None,
                                 obstacle_field: Optional[np.ndarray] = None) -> float:
        """Evaluate overall cost of trajectory."""
        
        cost = 0.0
        
        # Normalize by trajectory length
        if len(waypoints) > 1:
            trajectory_length = sum(
                np.linalg.norm(waypoints[i+1] - waypoints[i])
                for i in range(len(waypoints)-1)
            )
            cost = trajectory_length / 10000.0  # Normalize to 10km reference
        
        return cost
    
    def _nearest_threat_distance(self, current_pos_ned: np.ndarray,
                                trajectory_waypoints: List[np.ndarray],
                                threat_positions: List[np.ndarray]) -> float:
        """Find distance to nearest threat along trajectory."""
        
        min_distance = float('inf')
        
        for threat_pos in threat_positions:
            # Distance to each trajectory point
            for waypoint in trajectory_waypoints:
                distance = np.linalg.norm(waypoint - threat_pos)
                min_distance = min(min_distance, distance)
        
        return min_distance if min_distance != float('inf') else 10000
    
    def _find_nearest_threat_to_trajectory(self, trajectory_waypoints: List[np.ndarray],
                                          threat_positions: List[np.ndarray]) \
            -> Tuple[int, np.ndarray, float]:
        """Find which threat is closest to trajectory."""
        
        min_distance = float('inf')
        closest_threat_idx = 0
        closest_threat_pos = threat_positions[0]
        closest_waypoint_idx = 0
        
        for t_idx, threat_pos in enumerate(threat_positions):
            for w_idx, waypoint in enumerate(trajectory_waypoints):
                distance = np.linalg.norm(waypoint - threat_pos)
                if distance < min_distance:
                    min_distance = distance
                    closest_threat_idx = t_idx
                    closest_threat_pos = threat_pos
                    closest_waypoint_idx = w_idx
        
        return closest_waypoint_idx, closest_threat_pos, min_distance
    
    def _generate_detour_waypoints(self, trajectory_waypoints: List[np.ndarray],
                                  threat_pos_ned: np.ndarray,
                                  threat_waypoint_idx: int) -> List[np.ndarray]:
        """Generate modified waypoints that avoid threat."""
        
        modified = trajectory_waypoints[:threat_waypoint_idx+1]
        
        # Insert detour waypoint perpendicular to threat
        current_waypoint = trajectory_waypoints[threat_waypoint_idx]
        next_waypoint = trajectory_waypoints[threat_waypoint_idx+1] if threat_waypoint_idx+1 < len(trajectory_waypoints) else None
        
        if next_waypoint is not None:
            # Direction of travel
            travel_dir = next_waypoint - current_waypoint
            travel_dir_norm = travel_dir / (np.linalg.norm(travel_dir) + 1e-6)
            
            # Perpendicular offset (90 degrees left)
            perpendicular = np.array([-travel_dir_norm[1], travel_dir_norm[0], 0])
            
            # Safety waypoint
            safety_waypoint = current_waypoint + perpendicular * self.WAYPOINT_LATERAL_OFFSET_M
            modified.append(safety_waypoint)
        
        # Continue with remaining waypoints
        if threat_waypoint_idx + 1 < len(trajectory_waypoints):
            modified.extend(trajectory_waypoints[threat_waypoint_idx+1:])
        
        return modified
    
    @staticmethod
    def _normalize_heading_change(heading_delta_deg: float) -> float:
        """Normalize heading change to [-180, 180] range."""
        delta = heading_delta_deg % 360
        if delta > 180:
            delta -= 360
        return delta
    
    def get_statistics(self) -> Dict:
        """Return statistics on replanning suggestions."""
        return {
            "heading_adjustments_suggested": self.heading_adjustments_suggested,
            "waypoint_mods_suggested": self.waypoint_mods_suggested,
            "full_replans_suggested": self.full_replans_suggested,
        }
