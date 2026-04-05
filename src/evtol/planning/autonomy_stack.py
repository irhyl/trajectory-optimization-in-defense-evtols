"""
Phase 2G: Autonomous Decision-Making & Mission Adaptation

Integrates threat awareness (Phase 2F) with real-time mission adaptation:

- **Autonomous Mission Replanning**: Dynamic route adjustment to avoid threats
- **Contingency Management**: Handle sensor anomalies, vehicle failures
- **Closed-Loop Decision-Making**: Integrate threat/mission data into planning
- **Learning & Adaptation**: Improve threat classification and trajectory planning

**Philosophy**: Transform detected threats into autonomous actions without human intervention.

**Architecture**

```
Phase 2F: Threat Detection
    ├─ Active Threats (positions, velocities, confidence)
    └─ Risk Assessment (risk scores, threat levels)
    
        ↓
    
Phase 2G: Autonomous Decision  ← THIS MODULE
    ├─ Mission Replanner: Compute avoidance routes
    ├─ Contingency Manager: Handle failures
    ├─ Decision Logic: When/how to replan
    └─ Learning Module: Improve over time
    
        ↓
    
Phase 2E: Execution Loop
    ├─ Updated trajectory commands
    └─ Adaptive control gains
    
        ↓
    
Phase 2B: Motor Commands
    └─ Vehicle actuation
```

**Execution Flow**

```
Autonomous Cycle (Phase 2G):
    
0. Receive threat_map from Phase 2F
   
1. DecisionLogic: Should we replan?
   - Threat level increased?
   - Risk trajectory unsafe?
   - Vehicle state anomaly?
   
2. If YES → AutonomousMissionReplanner:
   - Generate avoidance trajectories
   - Evaluate alternatives (distance, energy, time)
   - Select best trajectory
   
3. If anomaly detected → ContingencyManager:
   - Classify failure type
   - Execute contingency behavior
   - Log for learning
   
4. Update mission state
   - New waypoints
   - Updated execution priority
   - Contingency status
   
5. Pass to Phase 2E for execution
```

**References**

[1] Kavraki et al. (1996): "Probabilistic Roadmaps for Path Planning in High-Dimensional Spaces"
[2] Hart et al. (1968): "A Formal Basis for the Heuristic Determination of Minimum Cost Paths"
[3] Bellman (1957): "Dynamic Programming"
[4] Sutton & Barto (2018): "Reinforcement Learning: An Introduction"
"""

from __future__ import annotations
import logging
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional, List, Dict, Tuple, Any
from collections import deque

logger = logging.getLogger(__name__)


class ContingencyType(Enum):
    """Classification of contingencies (failures/anomalies)."""
    SENSOR_LOSS = "sensor_loss"                    # GPS loss, radar failure
    MOTOR_FAULT = "motor_fault"                    # Motor malfunction
    POWER_ANOMALY = "power_anomaly"                # Battery, electrical
    CONTROL_FAILURE = "control_failure"            # Attitude hold failure
    STRUCTURAL_DAMAGE = "structural_damage"        # Vehicle damage
    ENVIRONMENTAL = "environmental"                # Weather, turbulence
    UNKNOWN = "unknown"


class ContingencyLevel(Enum):
    """Severity of contingency."""
    ADVISORY = "advisory"         # Minor issue, continue mission
    WARNING = "warning"            # Moderate issue, prepare for recovery
    CRITICAL = "critical"          # Severe issue, must act now
    EMERGENCY = "emergency"        # Vehicle integrity at risk


class ReplanDecision(Enum):
    """Replanning decision types."""
    CONTINUE = "continue"          # Stick with current plan
    REPLAN = "replan"              # Compute new trajectory
    DIVERT = "divert"              # Emergency route to safety
    DENIABLE_REPLAN = "deniable"   # Subtle avoidance without backtracking


@dataclass
class MissionState:
    """Current mission status."""
    mission_id: str
    waypoints: np.ndarray              # [N, 3] planned waypoints
    current_waypoint_idx: int = 0
    
    # Execution state
    start_time: datetime = field(default_factory=datetime.now)
    mission_time_remaining_s: float = 300.0
    
    # Mission properties
    mission_priority: str = "default"  # low, default, high, emergency
    is_autonomous: bool = True
    can_divert: bool = True
    
    # Contingency state
    active_contingencies: List[Tuple[ContingencyType, ContingencyLevel]] = field(default_factory=list)
    contingency_history: List[Dict[str, Any]] = field(default_factory=list)
    
    def get_elapsed_time_s(self) -> float:
        """Time since mission start."""
        return (datetime.now() - self.start_time).total_seconds()
    
    def get_progress_fraction(self) -> float:
        """Mission progress [0, 1]."""
        if not self.waypoints.size:
            return 0.0
        return min(1.0, self.current_waypoint_idx / len(self.waypoints))


@dataclass
class ReplanRequest:
    """Request for mission replanning."""
    reason: str                        # Why replan?
    threat_map: Optional[Dict] = None  # Threat data from Phase 2F
    contingency: Optional[Tuple] = None  # (ContingencyType, ContingencyLevel)
    vehicle_state: Optional[np.ndarray] = None  # Current state if available
    timeout_s: float = 5.0             # Max time to compute new plan


@dataclass
class ReplannedTrajectory:
    """New trajectory computed by autonomy replanner."""
    waypoints: np.ndarray              # [N, 3] new planned waypoints
    altitudes: np.ndarray              # [N] altitude constraints
    speeds: np.ndarray                 # [N] speed recommendations
    
    # Metadata
    reason: str                        # Why was this plan computed?
    distance_vs_original_pct: float    # +X% longer than original
    energy_vs_original_pct: float      # +X% more energy needed
    time_vs_original_pct: float        # +X% longer flight time
    
    # Evaluation
    threat_avoidance_margin: float     # Minimum distance to threats (m)
    feasibility_score: float           # [0, 1] how feasible is this plan?
    safety_score: float                # [0, 1] how safe is this plan?
    
    timestamp: datetime = field(default_factory=datetime.now)


class AutonomousMissionReplanner:
    """
    Computes new mission trajectories to avoid threats.
    
    **Algorithm**:
    - Threat-aware path planning
    - Multi-objective optimization (distance, energy, time, safety)
    - Lazy replanning (compute only when needed)
    """
    
    def __init__(
        self,
        max_detour_pct: float = 50.0,        # Max path elongation
        min_altitude_m: float = 50.0,
        max_altitude_m: float = 300.0,
    ):
        self.max_detour = max_detour_pct
        self.min_alt = min_altitude_m
        self.max_alt = max_altitude_m
        
        self.replan_history = deque(maxlen=100)
    
    def compute_evasion_route(
        self,
        current_position: np.ndarray,        # [x, y, z]
        original_waypoints: np.ndarray,      # [N, 3]
        threat_positions: List[np.ndarray],  # List of threat [x, y, z]
        threat_radii: List[float],           # Threat avoidance radius (m)
    ) -> ReplannedTrajectory:
        """
        Compute evasion route around threats.
        
        **Algorithm**:
        1. Identify waypoints near threats
        2. Insert avoidance waypoints
        3. Evaluate path metrics
        4. Return plan with feasibility/safety scores
        
        Args:
            current_position: [x, y, z]
            original_waypoints: [N, 3] original plan
            threat_positions: List of threat centers
            threat_radii: Avoidance radius per threat
        
        Returns:
            ReplannedTrajectory with new waypoints
        """
        if not threat_positions:
            # No threats, return original plan
            return ReplannedTrajectory(
                waypoints=original_waypoints.copy(),
                altitudes=np.ones(len(original_waypoints)) * 150.0,
                speeds=np.ones(len(original_waypoints)) * 15.0,
                reason="no_threats",
                distance_vs_original_pct=0.0,
                energy_vs_original_pct=0.0,
                time_vs_original_pct=0.0,
                threat_avoidance_margin=999.0,
                feasibility_score=1.0,
                safety_score=1.0,
            )
        
        # Find waypoints threatened
        threatened_indices = []
        for i, wp in enumerate(original_waypoints):
            for threat_pos, threat_rad in zip(threat_positions, threat_radii):
                dist = np.linalg.norm(wp - threat_pos)
                if dist < threat_rad * 1.5:  # Trigger radius ~1.5x nominal
                    threatened_indices.append(i)
                    break
        
        if not threatened_indices:
            # No waypoints threatened
            return ReplannedTrajectory(
                waypoints=original_waypoints.copy(),
                altitudes=np.ones(len(original_waypoints)) * 150.0,
                speeds=np.ones(len(original_waypoints)) * 15.0,
                reason="no_waypoints_threatened",
                distance_vs_original_pct=0.0,
                energy_vs_original_pct=0.0,
                time_vs_original_pct=0.0,
                threat_avoidance_margin=999.0,
                feasibility_score=1.0,
                safety_score=1.0,
            )
        
        # Insert avoidance waypoints
        new_waypoints = list(original_waypoints)
        
        for threatened_idx in threatened_indices:
            if threatened_idx >= len(new_waypoints):
                break
            
            wp = new_waypoints[threatened_idx]
            
            # Compute avoidance vector (escape away from closest threat)
            closest_threat = min(
                zip(threat_positions, threat_radii),
                key=lambda x: np.linalg.norm(wp - x[0])
            )
            escape_vec = wp - closest_threat[0]
            escape_vec_normalized = escape_vec / (np.linalg.norm(escape_vec) + 1e-6)
            
            # Create avoidance waypoint
            avoidance_wp = wp + escape_vec_normalized * (closest_threat[1] * 1.5)
            
            # Insert before threatened waypoint
            new_waypoints.insert(threatened_idx, avoidance_wp)
        
        # Convert to ndarray
        new_waypoints_array = np.array(new_waypoints)
        
        # Compute metrics
        original_distance = np.sum([
            np.linalg.norm(new_waypoints_array[i+1] - new_waypoints_array[i])
            for i in range(len(new_waypoints_array) - 1)
        ])
        
        original_plan_distance = np.sum([
            np.linalg.norm(original_waypoints[i+1] - original_waypoints[i])
            for i in range(len(original_waypoints) - 1)
        ])
        
        distance_pct = (original_distance - original_plan_distance) / max(original_plan_distance, 1.0) * 100.0
        
        # Minimum distance to any threat
        min_threat_dist = min([
            min([np.linalg.norm(wp - tp) for wp in new_waypoints_array])
            for tp in threat_positions
        ])
        
        # Feasibility check
        feasibility = min(1.0, max(0.0, 1.0 - distance_pct / self.max_detour))
        
        # Safety score (higher if farther from threats)
        safety = np.clip((min_threat_dist - threat_radii[0]) / 1000.0, 0.0, 1.0)
        
        return ReplannedTrajectory(
            waypoints=new_waypoints_array,
            altitudes=np.ones(len(new_waypoints_array)) * 150.0,
            speeds=np.ones(len(new_waypoints_array)) * 15.0,
            reason="threat_avoidance",
            distance_vs_original_pct=distance_pct,
            energy_vs_original_pct=distance_pct * 0.8,  # Rough estimation
            time_vs_original_pct=distance_pct * 1.1,
            threat_avoidance_margin=min_threat_dist,
            feasibility_score=feasibility,
            safety_score=safety,
        )


class ContingencyManager:
    """
    Handles vehicle failures and sensor anomalies.
    
    **Contingency Types**:
    - Sensor loss (GPS, radar, imaging)
    - Motor faults (stuck motor, reduced thrust)
    - Power anomalies (battery low, electrical failure)
    - Control failures (cannot hold attitude)
    - Environmental (extreme wind, icing)
    """
    
    def __init__(self):
        self.contingency_history = deque(maxlen=1000)
        self.recovery_count = {}  # Track successful recoveries
    
    def detect_anomaly(
        self,
        vehicle_state: np.ndarray,         # [x, y, z, vx, vy, vz, ...]
        control_command: np.ndarray,       # Expected motor commands
        actual_motor_status: Optional[Dict] = None,  # Actual motor feedback
    ) -> Tuple[Optional[ContingencyType], ContingencyLevel]:
        """
        Detect vehicle or sensor anomalies.
        
        Args:
            vehicle_state: Current state vector
            control_command: Expected motor thrust
            actual_motor_status: Motor feedback (if available)
        
        Returns:
            (ContingencyType, ContingencyLevel) or (None, None)
        """
        # Check for attitude instability
        if vehicle_state.size >= 12:
            # Assuming state includes roll, pitch, roll_rate, pitch_rate
            attitude_rates = vehicle_state[9:11]
            if np.any(np.abs(attitude_rates) > 60.0):  # >60°/s
                return (ContingencyType.CONTROL_FAILURE, ContingencyLevel.WARNING)
        
        # Check for motor mismatch (if feedback available)
        if actual_motor_status is not None:
            for motor_id, status in actual_motor_status.items():
                if status.get('failed', False):
                    return (ContingencyType.MOTOR_FAULT, ContingencyLevel.CRITICAL)
                
                # Check for thrust mismatch
                expected = control_command[motor_id] if motor_id < len(control_command) else 0
                actual = status.get('thrust', 0)
                mismatch = abs(expected - actual) / max(expected, 1.0)
                
                if mismatch > 0.5:  # >50% mismatch
                    return (ContingencyType.MOTOR_FAULT, ContingencyLevel.WARNING)
        
        # No detectable anomaly
        return (None, None)
    
    def compute_recovery_behavior(
        self,
        contingency_type: ContingencyType,
        contingency_level: ContingencyLevel,
        mission_state: MissionState,
    ) -> Dict[str, Any]:
        """
        Compute recovery behavior for detected contingency.
        
        Args:
            contingency_type: Type of failure
            contingency_level: Severity
            mission_state: Current mission context
        
        Returns:
            Dictionary with recovery action
        """
        recovery_action = {
            'type': contingency_type.value,
            'level': contingency_level.value,
            'timestamp': datetime.now().isoformat(),
            'action': 'unknown',
            'priority': 'normal',
        }
        
        if contingency_type == ContingencyType.MOTOR_FAULT:
            if contingency_level == ContingencyLevel.CRITICAL:
                recovery_action['action'] = 'emergency_landing'
                recovery_action['priority'] = 'critical'
            elif contingency_level == ContingencyLevel.WARNING:
                recovery_action['action'] = 'reduce_maneuver_rate'
                recovery_action['priority'] = 'high'
        
        elif contingency_type == ContingencyType.CONTROL_FAILURE:
            recovery_action['action'] = 'increase_control_loop_rate'
            recovery_action['priority'] = 'high'
        
        elif contingency_type == ContingencyType.SENSOR_LOSS:
            recovery_action['action'] = 'switch_to_fallback_sensor'
            recovery_action['priority'] = 'high'
        
        elif contingency_type == ContingencyType.POWER_ANOMALY:
            if contingency_level == ContingencyLevel.CRITICAL:
                recovery_action['action'] = 'emergency_landing'
                recovery_action['priority'] = 'critical'
            else:
                recovery_action['action'] = 'reduce_power_load'
                recovery_action['priority'] = 'high'
        
        return recovery_action


class DecisionLogic:
    """
    Autonomous decision-making for mission control.
    
    **Decision Factors**:
    - Threat level from Phase 2F
    - Current mission state
    - Vehicle health status
    - Time constraints
    - Energy constraints
    """
    
    def __init__(
        self,
        threat_threshold: float = 0.7,      # Trigger replan at this risk
        urgency_threshold: float = 0.5,     # Switch to urgent planning
    ):
        self.threat_threshold = threat_threshold
        self.urgency_threshold = urgency_threshold
        
        self.decision_history = deque(maxlen=1000)
    
    def should_replan(
        self,
        mission_state: MissionState,
        threat_map: Optional[Dict] = None,
        contingency: Optional[Tuple] = None,
    ) -> Tuple[ReplanDecision, str]:
        """
        Determine if mission should be replanned.
        
        Args:
            mission_state: Current mission context
            threat_map: Threat data from Phase 2F
            contingency: Active contingency (type, level)
        
        Returns:
            (ReplanDecision, reason_string)
        """
        reason = "no_trigger"
        decision = ReplanDecision.CONTINUE
        
        # Check for critical contingencies
        if contingency is not None:
            cont_type, cont_level = contingency
            
            if cont_level == ContingencyLevel.CRITICAL:
                return (ReplanDecision.DIVERT, f"Critical contingency: {cont_type.value}")
            elif cont_level == ContingencyLevel.WARNING:
                return (ReplanDecision.REPLAN, f"Warning contingency: {cont_type.value}")
        
        # Check threat level
        if threat_map is not None:
            highest_risk = threat_map.get('highest_risk', 0.0)
            
            if highest_risk > 0.8:  # Very high risk
                return (ReplanDecision.DIVERT, f"Emergency threat avoidance (risk={highest_risk:.2f})")
            elif highest_risk > self.threat_threshold:
                return (ReplanDecision.REPLAN, f"Threat avoidance needed (risk={highest_risk:.2f})")
            elif highest_risk > self.urgency_threshold:
                return (ReplanDecision.DENIABLE_REPLAN, f"Subtle adjustment (risk={highest_risk:.2f})")
        
        # Check time constraints
        if mission_state.mission_time_remaining_s < 30.0:
            return (ReplanDecision.CONTINUE, "Near end of mission, no replanning")
        
        return (decision, reason)
    
    def record_decision(self, decision: ReplanDecision, reason: str, metadata: Dict = None):
        """Log decision for learning."""
        self.decision_history.append({
            'timestamp': datetime.now(),
            'decision': decision.value,
            'reason': reason,
            'metadata': metadata or {},
        })


class AdaptiveController:
    """
    Adapts control gains based on threat level and mission state.
    
    **Adaptation Rules**:
    - High threat → aggressive control (short settling time)
    - Low threat → conservative control (fuel efficient)
    - Emergency → maximum safety
    """
    
    @staticmethod
    def compute_adaptive_gains(
        threat_level: str,              # green, yellow, red, black
        vehicle_maneuvering: bool = False,
    ) -> Dict[str, float]:
        """
        Compute control gains for current threat level.
        
        Args:
            threat_level: Threat severity
            vehicle_maneuvering: Is vehicle currently maneuvering?
        
        Returns:
            Dictionary of control gains
        """
        base_gains = {
            'kp_attitude': 1.0,
            'kd_attitude': 0.3,
            'kp_velocity': 0.5,
            'kd_velocity': 0.2,
            'ki_attitude': 0.05,
        }
        
        # Scale gains based on threat
        threat_multipliers = {
            'green': 0.8,      # Conservative
            'yellow': 1.0,     # Nominal
            'red': 1.3,        # Aggressive
            'black': 1.5,      # Emergency
        }
        
        multiplier = threat_multipliers.get(threat_level, 1.0)
        
        # Additional scaling if maneuvering
        if vehicle_maneuvering:
            multiplier *= 1.1
        
        # Apply multiplier
        adaptive_gains = {k: v * multiplier for k, v in base_gains.items()}
        
        return adaptive_gains
