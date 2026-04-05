"""
Phase 2F: Sensor Fusion & Threat Detection - Integration with Phase 2E

Orchestrates sensor fusion into the 50 Hz control loop:

- Radar/RF sensor input processing
- Threat track management and prediction
- Path-threat risk assessment
- Replanning trigger decision logic
- Integration into Phase 2E execution loop

**Integration Architecture**

```
Phase 1: Sensor Collection
    ↓
Radar Provider, RF Detector
    ↓
[New] Phase 2F Orchestrator ← This module
    ↓
    ├─ SensorFuser: Track fusion
    ├─ PathThreatAnalyzer: Risk computation
    └─ ReplanTrigger: Replanning decision
    ↓
Outputs: Threat map, replanning decisions
    ↓
Phase 2E Control Loop
    ├─ OnlineReplanner (triggered by Phase 2F)
    └─ ExecutionController (updated commands)
```

**Execution**

From Phase 2E integration_loop:

```python
sensor_orchestrator = SensorFusionOrchestrator(...)

# In 50 Hz loop, Phase 2F slice:
threat_map = sensor_orchestrator.process_cycle(
    radar_measurements=radar_buffer,
    rf_measurements=rf_buffer,
    vehicle_state=current_state,
    planned_trajectory=trajectory,
    mission_time_remaining_s=time_left,
)

# Decision from Phase 2F
if threat_map.replanning_required:
    trigger_phase2e_replanner()
```

**Threat Output Format**

```python
threat_map = {
    'timestamp': datetime,
    'active_threats': [TrackedThreat, ...],        # All non-stale tracks
    'threat_risks': [0.3, 0.7, 0.2, ...],          # Risk scores per threat
    'highest_risk': 0.7,                            # Max risk
    'highest_risk_threat_id': 'threat_0002',
    'replanning_required': True,                    # From ReplanTrigger
    'replan_reason': 'Critical threat threat_0002',
    'recommended_avoidance_heading': 285.0,        # deg
    'fusion_metrics': {
        'radar_measurements_processed': 12,
        'rf_measurements_processed': 3,
        'active_tracks': 3,
        'stale_tracks_pruned': 2,
        'new_tracks_created': 1,
    },
}
```
"""

import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from collections import deque

from .sensor_fusion import (
    SensorFuser, PathThreatAnalyzer, ReplanTriggerLogic,
    RadarMeasurement, RFMeasurement, ThreatLevel, ThreatType,
)

logger = logging.getLogger(__name__)


@dataclass
class ThreatMap:
    """Threat picture for Phase 2E integration."""
    
    timestamp: datetime
    active_threats: List[Any]                 # TrackedThreat objects
    threat_risks: List[float]                 # Risk scores [0, 1]
    highest_risk: float = 0.0
    highest_risk_threat_id: Optional[str] = None
    
    # Replanning decision
    replanning_required: bool = False
    replan_reason: str = ""
    recommended_avoidance_heading: Optional[float] = None
    
    # Metrics
    fusion_metrics: Dict[str, Any] = field(default_factory=dict)
    
    def get_threat_level(self) -> ThreatLevel:
        """Convert risk score to threat level."""
        if self.highest_risk < 0.3:
            return ThreatLevel.GREEN
        elif self.highest_risk < 0.6:
            return ThreatLevel.YELLOW
        elif self.highest_risk < 0.8:
            return ThreatLevel.RED
        else:
            return ThreatLevel.BLACK


class SensorFusionOrchestrator:
    """
    Orchestrates Phase 2F sensor fusion to feed threat data into Phase 2E execution loop.
    
    **Execution Rates**:
    - Sensor input: Variable (radar 10 Hz, RF 1 Hz, imaging event-based)
    - Fusion cycle: 20 Hz (for 50 Hz control loop compatibility)
    - Replanning trigger: Event-driven
    
    **Responsibilities**:
    1. Buffer sensor measurements between cycles
    2. Run sensor fusion (track association, Kalman filtering)
    3. Compute path-threat risk scores
    4. Trigger replanning decisions
    5. Maintain threat map for Phase 2E
    """
    
    def __init__(
        self,
        fusion_rate_hz: float = 20.0,           # Sensor fusion cycle rate
        radar_agg_window_ms: int = 100,         # Accumulate radar for 100ms
        rf_agg_window_ms: int = 500,            # Accumulate RF for 500ms
    ):
        """
        Initialize Phase 2F orchestrator.
        
        Args:
            fusion_rate_hz: Sensor fusion update rate
            radar_agg_window_ms: Radar measurement integration window
            rf_agg_window_ms: RF measurement integration window
        """
        self.fusion_rate = fusion_rate_hz
        self.radar_window = radar_agg_window_ms / 1000.0
        self.rf_window = rf_agg_window_ms / 1000.0
        
        # Fusion components
        self.sensor_fuser = SensorFuser(max_track_age_s=30.0, gate_threshold_m=100.0)
        self.path_analyzer = PathThreatAnalyzer()
        self.replan_logic = ReplanTriggerLogic(
            risk_threshold_alert=0.6,
            risk_threshold_critical=0.8,
        )
        
        # Measurement buffering
        self.radar_queue = deque()              # [RadarMeasurement, ...]
        self.rf_queue = deque()                 # [RFMeasurement, ...]
        
        # State tracking
        self.threat_map_history = deque(maxlen=1000)
        self.cycle_counter = 0
        self.last_fusion_time = datetime.now()
        
        logger.info(f"Initialized Phase 2F SensorFusionOrchestrator @ {fusion_rate_hz} Hz")
    
    def add_radar_measurement(self, measurement: RadarMeasurement):
        """
        Queue a single radar detection.
        
        Args:
            measurement: RadarMeasurement object
        """
        self.radar_queue.append(measurement)
    
    def add_radar_measurements(self, measurements: List[RadarMeasurement]):
        """Queue batch of radar detections."""
        self.radar_queue.extend(measurements)
    
    def add_rf_measurement(self, measurement: RFMeasurement):
        """Queue a single RF detection."""
        self.rf_queue.append(measurement)
    
    def add_rf_measurements(self, measurements: List[RFMeasurement]):
        """Queue batch of RF detections."""
        self.rf_queue.extend(measurements)
    
    def process_cycle(
        self,
        vehicle_state: np.ndarray,          # [x, y, z, vx, vy, vz]
        planned_trajectory: np.ndarray,     # [N, 3] waypoints
        mission_time_remaining_s: float = 300.0,
    ) -> ThreatMap:
        """
        Run one sensor fusion cycle.
        
        **Execution Steps** (must complete in <50ms for 20Hz):
        
        1. Predict all tracks forward (+1ms)
        2. Process buffered radar measurements (+3ms)
        3. Process buffered RF measurements (+1ms)
        4. Prune stale tracks (+1ms)
        5. Compute risk scores (+2ms)
        6. Check replanning triggers (+1ms)
        7. Generate threat map output (<1ms)
        
        Total: ~10ms budget, leaves time for Phase 2E/2B padding
        
        Args:
            vehicle_state: [x, y, z, vx, vy, vz] in earth frame
            planned_trajectory: [N, 3] array of waypoint positions
            mission_time_remaining_s: Time left in mission for risk assessment
        
        Returns:
            ThreatMap object with threat picture and replanning decision
        """
        cycle_start = datetime.now()
        self.cycle_counter += 1
        
        vehicle_pos = vehicle_state[:3]
        
        # Step 1: Predict all tracks forward
        time_since_last = (cycle_start - self.last_fusion_time).total_seconds()
        self.sensor_fuser.predict_all_tracks(time_step_s=time_since_last)
        
        # Step 2: Process buffered radar measurements
        radar_batch = list(self.radar_queue)
        self.radar_queue.clear()
        
        if radar_batch:
            logger.debug(f"Processing {len(radar_batch)} radar measurements")
            self.sensor_fuser.update_radar_measurements(radar_batch, vehicle_pos)
        
        # Step 3: Process buffered RF measurements
        rf_batch = list(self.rf_queue)
        self.rf_queue.clear()
        
        if rf_batch:
            logger.debug(f"Processing {len(rf_batch)} RF measurements")
            self.sensor_fuser.update_rf_measurements(rf_batch, vehicle_pos)
        
        # Step 4: Prune stale tracks
        self.sensor_fuser.prune_stale_tracks()
        
        # Step 5: Get active threats and compute risk scores
        active_threats = self.sensor_fuser.get_active_tracks()
        threat_risks = []
        avoidance_headings = []
        
        for threat in active_threats:
            risk, closest_point = self.path_analyzer.compute_threat_risk(
                threat=threat,
                trajectory_waypoints=planned_trajectory,
                time_remaining_s=mission_time_remaining_s,
            )
            threat_risks.append(risk)
            
            # Compute avoidance heading
            heading = self.path_analyzer.compute_avoidance_heading(
                current_position=vehicle_pos,
                threat_position=threat.position,
                current_heading=0.0,  # In practice, get from vehicle state
            )
            avoidance_headings.append(heading)
        
        # Step 6: Check replanning triggers
        should_replan, replan_reason = self.replan_logic.should_trigger_replan(
            active_threats=active_threats,
            threat_risks=threat_risks,
        )
        
        # Step 7: Determine highest risk and recommended action
        highest_risk = max(threat_risks) if threat_risks else 0.0
        highest_risk_idx = threat_risks.index(highest_risk) if threat_risks else -1
        highest_risk_threat_id = (
            active_threats[highest_risk_idx].threat_id
            if highest_risk_idx >= 0 else None
        )
        recommended_heading = (
            avoidance_headings[highest_risk_idx]
            if highest_risk_idx >= 0 else None
        )
        
        # Build threat map
        threat_map = ThreatMap(
            timestamp=cycle_start,
            active_threats=active_threats,
            threat_risks=threat_risks,
            highest_risk=highest_risk,
            highest_risk_threat_id=highest_risk_threat_id,
            replanning_required=should_replan,
            replan_reason=replan_reason,
            recommended_avoidance_heading=recommended_heading,
            fusion_metrics={
                'radar_measurements_processed': len(radar_batch),
                'rf_measurements_processed': len(rf_batch),
                'active_tracks': len(active_threats),
                'total_tracks': len(self.sensor_fuser.tracks),
                'new_threat_level': 'pending',  # Filled in below after construction
            },
        )
        
        threat_map.fusion_metrics['new_threat_level'] = threat_map.get_threat_level().value
        
        # Store in history for analysis
        self.threat_map_history.append(threat_map)
        self.last_fusion_time = cycle_start
        
        # Log summary
        cycle_time = (datetime.now() - cycle_start).total_seconds() * 1000.0
        logger.debug(
            f"Phase 2F Cycle {self.cycle_counter}: "
            f"tracks={len(active_threats)}, risk={highest_risk:.2f}, "
            f"replan={should_replan}, time={cycle_time:.1f}ms"
        )
        
        return threat_map
    
    def get_threat_picture_summary(self) -> Dict[str, Any]:
        """
        Get summary of current threat picture.
        
        Returns:
            Dictionary with threat levels, counts, trends
        """
        if not self.threat_map_history:
            return {'status': 'no_data'}
        
        latest = self.threat_map_history[-1]
        
        # Trend analysis (last 10 cycles)
        recent_risks = [
            m.highest_risk
            for m in list(self.threat_map_history)[-10:]
        ]
        
        trend = "increasing" if (recent_risks[-1] > np.mean(recent_risks[:-1])) else "decreasing"
        threat_level = latest.get_threat_level()
        
        return {
            'status': 'ok',
            'timestamp': latest.timestamp.isoformat(),
            'threat_level': threat_level.value,
            'highest_risk': latest.highest_risk,
            'highest_risk_threat': latest.highest_risk_threat_id,
            'active_threats': len(latest.active_threats),
            'trend': trend,
            'replanning_required': latest.replanning_required,
            'replan_reason': latest.replan_reason,
        }
    
    def get_threat_map_for_phase2e(self) -> Optional[Dict[str, Any]]:
        """
        Extract threat map for Phase 2E integration.
        
        Returns:
            Dictionary suitable for Phase 2E replan trigger
        """
        if not self.threat_map_history:
            return None
        
        latest = self.threat_map_history[-1]
        
        return {
            'timestamp': latest.timestamp,
            'threats': [
                {
                    'id': t.threat_id,
                    'type': t.threat_type.value,
                    'position': t.position.tolist(),
                    'velocity': t.velocity.tolist(),
                    'confidence': t.confidence,
                    'risk': r,
                }
                for t, r in zip(latest.active_threats, latest.threat_risks)
            ],
            'replanning_required': latest.replanning_required,
            'replan_reason': latest.replan_reason,
            'recommended_avoidance_heading': latest.recommended_avoidance_heading,
        }
