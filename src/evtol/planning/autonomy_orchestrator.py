"""
Phase 2G: Autonomy Orchestrator - Integration with Phase 2E Control Loop

Bridges threat-aware autonomous decisions into the 50 Hz execution loop:

- Receives threat maps from Phase 2F
- Triggers autonomous replanning when needed
- Manages contingency responses
- Adapts control gains based on mission state
- Coordinates with Phase 2E execution loop

**Integration Architecture**

```
Phase 2F: Threat Detection (20 Hz)
    ├─ ThreatMap (active_threats, risk_scores)
    └─ Replanning trigger
    
        ↓
    
Phase 2G: Autonomy Orchestrator (variable, <100ms)
    ├─ DecisionLogic: Should autonomously replan?
    ├─ AutonomousMissionReplanner: Compute new trajectory
    └─ ContingencyManager: Handle failures
    
        ↓
    
Phase 2E: Execution Loop (50 Hz)
    └─ Updated trajectory, adaptive gains
```

**Decision Flow**

```
receive threat_map from Phase 2F:
    
    if threat_map.replanning_required:
        decision, reason = decision_logic.should_replan(mission_state, threat_map)
        
        if decision == REPLAN:
            new_trajectory = replanner.compute_evasion_route(
                current_pos, original_waypoints, threats, radii
            )
            
            mission_state.waypoints = new_trajectory.waypoints
            mission_state.is_autonomous = True
            
            prepare_phase2e_update({
                'new_trajectory': new_trajectory,
                'adaptive_gains': adaptive_gains,
            })
    
    elif contingency_detected:
        recovery = contingency_mgr.compute_recovery_behavior(cont_type, cont_level)
        prepare_phase2e_update({'recovery_action': recovery})
```
"""

import logging
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, Optional, Any, List, Tuple
from collections import deque

from .autonomy_stack import (
    MissionState, ReplanRequest, ReplannedTrajectory, ReplanDecision,
    AutonomousMissionReplanner, ContingencyManager, DecisionLogic, AdaptiveController,
    ContingencyType, ContingencyLevel,
)

logger = logging.getLogger(__name__)


@dataclass
class AutonomyOutputCommand:
    """Command output from autonomy stack to Phase 2E."""
    
    timestamp: datetime
    command_type: str                      # 'replan', 'contingency_recovery', 'adaptive_gain_update', 'continue'
    
    # Replanning
    new_trajectory: Optional[ReplannedTrajectory] = None
    reason: str = ""
    
    # Contingency
    recovery_action: Optional[Dict[str, Any]] = None
    
    # Adaptive control
    adaptive_gains: Optional[Dict[str, float]] = None
    
    # Metadata
    urgency: str = "nominal"               # nominal, urgent, emergency
    metadata: Dict[str, Any] = field(default_factory=dict)


class AutonomyOrchestrator:
    """
    Orchestrates Phase 2G autonomous decision-making.
    
    **Responsibilities**:
    1. Monitor threat maps from Phase 2F
    2. Detect vehicle anomalies
    3. Generate autonomous decisions
    4. Compute replanning requests
    5. Manage contingency recovery
    6. Coordinate with Phase 2E execution loop
    
    **Timing**: Asynchronous (triggered by Phase 2F updates, <100ms completion)
    """
    
    def __init__(
        self,
        threat_replan_threshold: float = 0.7,
        max_replan_frequency_s: float = 5.0,
        enable_learning: bool = True,
    ):
        """
        Initialize autonomy orchestrator.
        
        Args:
            threat_replan_threshold: Trigger replanning at this risk score
            max_replan_frequency_s: Limit replanning to once per N seconds
            enable_learning: Track decisions for learning
        """
        # Core components
        self.replanner = AutonomousMissionReplanner()
        self.contingency_mgr = ContingencyManager()
        self.decision_logic = DecisionLogic(threat_threshold=threat_replan_threshold)
        self.adaptive_controller = AdaptiveController()
        
        # Configuration
        self.max_replan_freq = max_replan_frequency_s
        self.enable_learning = enable_learning
        
        # State tracking
        self.mission_state: Optional[MissionState] = None
        self.last_replan_time = datetime.now() - timedelta(seconds=max_replan_frequency_s)
        self.command_history = deque(maxlen=1000)
        self.cycle_count = 0
        
        logger.info("Autonomy Orchestrator initialized")
    
    def initialize_mission(self, mission_id: str, waypoints: np.ndarray) -> MissionState:
        """
        Initialize mission for autonomous control.
        
        Args:
            mission_id: Unique mission identifier
            waypoints: [N, 3] planned trajectory
        
        Returns:
            MissionState object
        """
        self.mission_state = MissionState(
            mission_id=mission_id,
            waypoints=waypoints.copy(),
        )
        
        logger.info(f"Mission initialized: {mission_id}, {len(waypoints)} waypoints")
        return self.mission_state
    
    def process_threat_map(
        self,
        threat_map: Dict[str, Any],           # From Phase 2F SensorFusionOrchestrator
        vehicle_state: np.ndarray,            # [x, y, z, vx, vy, vz]
    ) -> AutonomyOutputCommand:
        """
        Process threat map and generate autonomous decision.
        
        **Execution Sequence** (<100ms):
        
        1. Parse threat_map (+1ms)
        2. Evaluate decision (+2ms)
        3. If replan needed:
           a. Check rate limit (+1ms)
           b. Compute evasion route (+20ms)
           c. Evaluate feasibility (+5ms)
           d. Generate command (+2ms)
        4. If contingency:
           a. Classify failure (+2ms)
           b. Compute recovery (+2ms)
        5. Compute adaptive gains (+1ms)
        6. Return command (<1ms)
        
        Total: ~5ms nominal, ~30ms with replanning
        
        Args:
            threat_map: Threat picture from Phase 2F
            vehicle_state: Current vehicle state [x, y, z, vx, vy, vz]
        
        Returns:
            AutonomyOutputCommand for Phase 2E
        """
        command_start = datetime.now()
        self.cycle_count += 1
        
        if self.mission_state is None:
            return AutonomyOutputCommand(
                timestamp=command_start,
                command_type='continue',
                reason='no_mission_initialized',
            )
        
        # Step 1: Extract threat data
        active_threats = threat_map.get('threats', [])
        highest_risk = threat_map.get('highest_risk', 0.0)
        current_threat_level = self._risk_to_threat_level(highest_risk)
        
        # Step 2: Evaluate decision
        decision, reason = self.decision_logic.should_replan(
            mission_state=self.mission_state,
            threat_map=threat_map,
        )
        
        # Log decision
        if self.enable_learning:
            self.decision_logic.record_decision(
                decision, reason,
                metadata={
                    'threat_level': current_threat_level,
                    'active_threats': len(active_threats),
                    'highest_risk': highest_risk,
                },
            )
        
        # Step 3: Route based on decision
        if decision == ReplanDecision.REPLAN or decision == ReplanDecision.DIVERT:
            command = self._generate_replan_command(
                decision, reason, threat_map, vehicle_state,
            )
        elif decision == ReplanDecision.DENIABLE_REPLAN:
            # Subtle adjustment, not full replan
            command = self._generate_subtle_adjustment(reason, threat_map)
        else:
            # Continue with current plan
            command = AutonomyOutputCommand(
                timestamp=command_start,
                command_type='continue',
                reason=reason,
                adaptive_gains=self.adaptive_controller.compute_adaptive_gains(current_threat_level),
            )
        
        # Store in history
        self.command_history.append(command)
        
        execution_time = (datetime.now() - command_start).total_seconds() * 1000.0
        logger.debug(f"Autonomy cycle {self.cycle_count}: decision={decision.value}, time={execution_time:.1f}ms")
        
        return command
    
    def _risk_to_threat_level(self, risk_score: float) -> str:
        """Convert risk score to threat level."""
        if risk_score < 0.3:
            return "green"
        elif risk_score < 0.6:
            return "yellow"
        elif risk_score < 0.8:
            return "red"
        else:
            return "black"
    
    def _generate_replan_command(
        self,
        decision: ReplanDecision,
        reason: str,
        threat_map: Dict,
        vehicle_state: np.ndarray,
    ) -> AutonomyOutputCommand:
        """Generate autonomous replanning command."""
        
        # Check rate limiting
        time_since_last_replan = (datetime.now() - self.last_replan_time).total_seconds()
        
        if time_since_last_replan < self.max_replan_freq:
            logger.debug(f"Replanning rate limited (next in {self.max_replan_freq - time_since_last_replan:.1f}s)")
            # Fall back to adaptive gains only
            threat_level = self._risk_to_threat_level(threat_map.get('highest_risk', 0.0))
            return AutonomyOutputCommand(
                timestamp=datetime.now(),
                command_type='adaptive_gain_update',
                reason=f"Rate limited: {reason}",
                adaptive_gains=self.adaptive_controller.compute_adaptive_gains(threat_level),
                urgency="urgent" if threat_map.get('highest_risk', 0.0) > 0.7 else "nominal",
            )
        
        # Extract threat positions
        threat_positions = [
            np.array(t['position']) for t in threat_map.get('threats', [])
        ]
        threat_radii = [
            100.0 for _ in threat_positions  # Nominal threat radius
        ]
        
        # Compute evasion route
        current_pos = vehicle_state[:3]
        original_waypoints = self.mission_state.waypoints
        
        new_trajectory = self.replanner.compute_evasion_route(
            current_position=current_pos,
            original_waypoints=original_waypoints,
            threat_positions=threat_positions,
            threat_radii=threat_radii,
        )
        
        # Evaluate feasibility
        if new_trajectory.feasibility_score < 0.3:
            logger.warning(f"Computed trajectory has low feasibility: {new_trajectory.feasibility_score:.2f}")
        
        # Update mission state
        self.mission_state.waypoints = new_trajectory.waypoints
        self.last_replan_time = datetime.now()
        
        # Compute adaptive gains
        threat_level = self._risk_to_threat_level(threat_map.get('highest_risk', 0.0))
        
        urgency = "emergency" if decision == ReplanDecision.DIVERT else "urgent"
        
        return AutonomyOutputCommand(
            timestamp=datetime.now(),
            command_type='replan',
            new_trajectory=new_trajectory,
            reason=reason,
            adaptive_gains=self.adaptive_controller.compute_adaptive_gains(threat_level, vehicle_maneuvering=True),
            urgency=urgency,
            metadata={
                'decision': decision.value,
                'detour_pct': new_trajectory.distance_vs_original_pct,
                'safety_score': new_trajectory.safety_score,
                'avoidance_margin': new_trajectory.threat_avoidance_margin,
            },
        )
    
    def _generate_subtle_adjustment(self, reason: str, threat_map: Dict) -> AutonomyOutputCommand:
        """Generate subtle trajectory adjustment without full replanning."""
        
        # Adjust speed and altitude to slip past threat
        threat_level = self._risk_to_threat_level(threat_map.get('highest_risk', 0.0))
        
        return AutonomyOutputCommand(
            timestamp=datetime.now(),
            command_type='adaptive_gain_update',
            reason=f"Subtle adjustment: {reason}",
            adaptive_gains=self.adaptive_controller.compute_adaptive_gains(threat_level, vehicle_maneuvering=True),
            urgency="urgent",
            metadata={
                'adjustment_type': 'altitude_speed_profile',
                'target_altitude_delta_m': -25.0,  # Descend 25m
                'target_speed_delta_mps': 2.0,     # Speed up 2 m/s
            },
        )
    
    def process_contingency(
        self,
        vehicle_state: np.ndarray,                 # Full state vector
        control_command: np.ndarray,               # Motor commands sent
        actual_motor_status: Optional[Dict] = None,
    ) -> Optional[AutonomyOutputCommand]:
        """
        Process vehicle anomalies and generate recovery commands.
        
        Args:
            vehicle_state: Current state
            control_command: Expected motor commands
            actual_motor_status: Feedback from Phase 2B
        
        Returns:
            AutonomyOutputCommand if contingency detected, else None
        """
        # Detect anomaly
        cont_type, cont_level = self.contingency_mgr.detect_anomaly(
            vehicle_state, control_command, actual_motor_status,
        )
        
        if cont_type is None:
            return None
        
        # Compute recovery
        recovery_action = self.contingency_mgr.compute_recovery_behavior(
            cont_type, cont_level, self.mission_state,
        )
        
        # Log contingency
        if self.mission_state:
            self.mission_state.active_contingencies.append((cont_type, cont_level))
            self.mission_state.contingency_history.append({
                'timestamp': datetime.now().isoformat(),
                'type': cont_type.value,
                'level': cont_level.value,
                'recovery': recovery_action,
            })
        
        logger.warning(f"Contingency detected: {cont_type.value} ({cont_level.value})")
        logger.info(f"Recovery action: {recovery_action['action']}")
        
        # Generate command
        return AutonomyOutputCommand(
            timestamp=datetime.now(),
            command_type='contingency_recovery',
            recovery_action=recovery_action,
            reason=f"Contingency: {cont_type.value}",
            urgency="emergency" if cont_level == ContingencyLevel.CRITICAL else "urgent",
        )
    
    def get_mission_summary(self) -> Dict[str, Any]:
        """Get current mission status summary."""
        if self.mission_state is None:
            return {'status': 'no_mission'}
        
        return {
            'status': 'active',
            'mission_id': self.mission_state.mission_id,
            'elapsed_time_s': self.mission_state.get_elapsed_time_s(),
            'progress_fraction': self.mission_state.get_progress_fraction(),
            'waypoints_total': len(self.mission_state.waypoints),
            'current_waypoint': self.mission_state.current_waypoint_idx,
            'is_autonomous': self.mission_state.is_autonomous,
            'active_contingencies': len(self.mission_state.active_contingencies),
            'cycles_completed': self.cycle_count,
        }
