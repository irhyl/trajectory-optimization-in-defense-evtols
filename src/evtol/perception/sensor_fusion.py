"""
Phase 2F: Sensor Integration & Threat Detection

Fuses sensor data (radar, imaging, RF) from Phase 1 perception with Phase 2E execution
for real-time threat detection and environmental awareness:

- WiFi/RF signal fusion: Detects electronic warfare threats
- Radar track fusion: Moving target detection, classification
- Imaging analysis: Fixed threat identification
- Trajectory-aware threat scoring: Risk assessment based on future path
- Real-time replanning triggers: Updates from sensor fusion

**Architecture**

```
Phase 1: Sensor Collection
├─ Radar Provider
├─ RF Detector
├─ Imaging (Passive)
└─ GNSS/Inertial

        ↓ Fused measurements
    
Phase 2F: Sensor Fusion ← NEW (THIS MODULE)
├─ ThreatTracker: Multi-target fusion
├─ SignalFuser: RF signal correlation
├─ Path-Threat Analyzer: Risk assessment
└─ Replanning Trigger Logic

        ↓ Threat map + alerts
    
Phase 2C/2E: Planning/Execution
├─ OnlineReplanner: Route around threats
└─ ExecutionController: Adjust commands

        ↓ Updated trajectory
    
Vehicle Execution
```

**Threat Detection Chain**

```
Detection → Classification → Fusion → Risk Assessment → Action
   ↓           ↓               ↓          ↓               ↓
Radar      Threat vs    Multi-frame   Path-based    Replan
RF sensor  Clutter     tracking       scoring       Alert
```

**References**

[1] Bar-Shalom et al. (2011): "Estimation with Applications to Tracking and Navigation"
[2] Stone et al. (2016): "Cooperative multi-agent systems for signal detection"
[3] Blom & Bar-Shalom (1988): "The interacting multiple model algorithm for systems with Markovian switching coefficients"
"""

from __future__ import annotations
import logging
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional, List, Dict, Tuple, Any
from scipy.spatial.distance import cdist
from collections import deque

logger = logging.getLogger(__name__)


class ThreatType(Enum):
    """Classification of detected threats."""
    STATIONARY_RADAR = "stationary_radar"       # Fixed radar site
    MOVING_RADAR = "moving_radar"                # Vehicle-mounted radar
    MISSILE_LAUNCH_SITE = "missile_site"         # SAM launcher
    AIRCRAFT = "aircraft"                        # Manned aircraft
    UAV = "uav"                                  # Hostile drone
    RF_EMITTER = "rf_emitter"                    # Jamming/comms
    UNKNOWN = "unknown"


class ThreatLevel(Enum):
    """Threat severity level."""
    GREEN = "green"          # No threat
    YELLOW = "yellow"        # Precaution advised
    RED = "red"              # Imminent danger
    BLACK = "black"          # Critical


@dataclass
class RadarMeasurement:
    """Single radar detection."""
    range_m: float                 # Distance to target
    azimuth_deg: float             # Horizontal angle
    elevation_deg: float           # Vertical angle
    radial_velocity_mps: float     # Closing velocity
    signal_strength_dbm: float     # Power received
    
    # Classification
    rcs_m2: float = 0.0            # Estimated RCS (m²)
    
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_cartesian_body(self, vehicle_position: np.ndarray) -> np.ndarray:
        """Convert to body-frame Cartesian coordinates."""
        # Spherical to Cartesian
        az_rad = np.radians(self.azimuth_deg)
        el_rad = np.radians(self.elevation_deg)
        
        x = self.range_m * np.cos(el_rad) * np.cos(az_rad)
        y = self.range_m * np.cos(el_rad) * np.sin(az_rad)
        z = self.range_m * np.sin(el_rad)
        
        return np.array([x, y, z])


@dataclass
class RFMeasurement:
    """RF signal detection (radar, jam, comms)."""
    frequency_mhz: float           # Center frequency
    power_dbm: float               # Signal strength
    direction_deg: float           # Bearing (azimuth)
    
    # Classification
    emission_type: str = "unknown"  # RADAR, JAM, COMMS, etc.
    threat_index: float = 0.3      # [0, 1] estimated threat level
    
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class TrackedThreat:
    """Kalman-filtered threat track."""
    threat_id: str                 # Unique identifier
    threat_type: ThreatType
    
    # Position (earth frame)
    position: np.ndarray           # [x, y, z] (m)
    velocity: np.ndarray           # [vx, vy, vz] (m/s)
    
    # Uncertainty
    position_error: np.ndarray     # [σx, σy, σz] (m)
    velocity_error: np.ndarray     # [σvx, σvy, σvz] (m/s)
    
    # Confidence
    detection_count: int = 1       # Number of detections
    confidence: float = 0.5        # [0, 1]
    
    # Time tracking
    timestamp_created: datetime = field(default_factory=datetime.now)
    timestamp_updated: datetime = field(default_factory=datetime.now)
    time_since_update: float = 0.0 # Seconds
    
    def predict_position_at_time(self, t_seconds: float) -> np.ndarray:
        """Extrapolate position to future time."""
        return self.position + self.velocity * t_seconds
    
    def is_stale(self, max_age_s: float = 30.0) -> bool:
        """Check if track is too old to trust."""
        return self.time_since_update > max_age_s


class SensorFuser:
    """
    Fuses multi-source sensor data (radar, RF, imaging).
    
    **Fusion Algorithm**:
    - Track association: Nearest-neighbor with gating
    - Kalman filtering: Per-track state estimation
    - Confidence update: Bayesian combination
    """
    
    def __init__(self, max_track_age_s: float = 30.0, gate_threshold_m: float = 100.0):
        """
        Initialize sensor fusion engine.
        
        Args:
            max_track_age_s: Max time to keep stale tracks
            gate_threshold_m: Association gate distance
        """
        self.tracks: Dict[str, TrackedThreat] = {}
        self.max_track_age = max_track_age_s
        self.gate_threshold = gate_threshold_m
        
        self.measurement_history = deque(maxlen=1000)
        self.track_counter = 0
    
    def update_radar_measurements(self, measurements: List[RadarMeasurement], vehicle_pos: np.ndarray):
        """
        Process new radar detections.
        
        Args:
            measurements: List of radar returns
            vehicle_pos: Vehicle position (earth frame)
        """
        for meas in measurements:
            # Convert to earth frame
            meas_pos = vehicle_pos + meas.to_cartesian_body(vehicle_pos)
            
            # Try to associate with existing track
            associated_id = self._find_nearest_track(meas_pos, max_dist=self.gate_threshold)
            
            if associated_id:
                # Update existing track
                track = self.tracks[associated_id]
                self._update_track_kalman(track, meas_pos, meas.signal_strength_dbm)
                track.timestamp_updated = datetime.now()
                track.detection_count += 1
            else:
                # Create new track
                self._create_new_track(meas_pos, ThreatType.UNKNOWN, meas)
            
            self.measurement_history.append((datetime.now(), meas_pos, "RADAR"))
    
    def update_rf_measurements(self, measurements: List[RFMeasurement], vehicle_pos: np.ndarray):
        """
        Process RF sensor detections (radar, jam, comms).
        
        Args:
            measurements: List of RF detections
            vehicle_pos: Vehicle position (earth frame)
        """
        for meas in measurements:
            # RF bearings are less precise, use broader gates
            associated_id = self._find_nearest_track(vehicle_pos, max_dist=1000.0)
            
            if associated_id:
                # Correlate with radar track
                track = self.tracks[associated_id]
                
                # Update threat classification based on RF type
                if meas.emission_type == "RADAR" and meas.threat_index > 0.7:
                    track.threat_type = ThreatType.STATIONARY_RADAR
                    track.confidence = min(1.0, track.confidence + 0.1)
                elif meas.emission_type == "JAM":
                    track.threat_type = ThreatType.RF_EMITTER
                    track.confidence = min(1.0, track.confidence + 0.1)
            
            self.measurement_history.append((datetime.now(), vehicle_pos, "RF"))
    
    def _find_nearest_track(self, measurement_pos: np.ndarray, max_dist: float = 100.0) -> Optional[str]:
        """Find closest track within gate threshold."""
        if not self.tracks:
            return None
        
        track_positions = np.array([t.position for t in self.tracks.values()])
        distances = np.linalg.norm(track_positions - measurement_pos, axis=1)
        
        min_idx = np.argmin(distances)
        min_dist = distances[min_idx]
        
        if min_dist < max_dist:
            return list(self.tracks.keys())[min_idx]
        
        return None
    
    def _create_new_track(self, position: np.ndarray, threat_type: ThreatType, meas: Any):
        """Initialize new tracked threat."""
        track_id = f"threat_{self.track_counter:04d}"
        self.track_counter += 1
        
        track = TrackedThreat(
            threat_id=track_id,
            threat_type=threat_type,
            position=position.copy(),
            velocity=np.zeros(3),  # Unknown velocity initially
            position_error=np.array([100.0, 100.0, 100.0]),  # High uncertainty
            velocity_error=np.array([10.0, 10.0, 10.0]),
            confidence=0.3,  # Low confidence from single detection
        )
        
        self.tracks[track_id] = track
        logger.info(f"New track created: {track_id} at {position}")
    
    def _update_track_kalman(self, track: TrackedThreat, measurement: np.ndarray, signal_strength: float):
        """
        Simple Kalman update (predictor-corrector).
        
        **Simplified Kalman Filter**:
        - Predict: x̂_pred = F·x̂_prev
        - Correct: x̂ = x̂_pred + K·(z - H·x̂_pred)
        
        where K is Kalman gain (simplified)
        """
        # Process noise (velocity change)
        process_noise = 0.5  # m/s² std
        
        # Measurement noise (based on signal strength)
        meas_noise = 50.0 / (1.0 + 0.1 * np.clip(signal_strength, -100, 0))  # Better signal → lower noise
        
        # Simple gain (in practice use full covariance update)
        K = track.position_error[0] / (track.position_error[0] + meas_noise)
        
        # Update position
        residual = measurement - track.position
        track.position = track.position + K * residual
        
        # Update uncertainty (reduced by measurement)
        track.position_error = track.position_error * (1.0 - K)
        
        # Estimate velocity from successive measurements
        time_delta = (datetime.now() - track.timestamp_updated).total_seconds()
        if time_delta > 0.1:  # Only update if time has advanced
            velocity_meas = residual / max(time_delta, 0.01)
            # Low-pass filter velocity estimate
            alpha = 0.3
            track.velocity = (1.0 - alpha) * track.velocity + alpha * velocity_meas
    
    def predict_all_tracks(self, time_step_s: float = 0.02):
        """Predict all tracks forward in time."""
        now = datetime.now()
        
        for track in self.tracks.values():
            # Age the track
            track.time_since_update = (now - track.timestamp_updated).total_seconds()
            
            # Predict position
            track.position = track.position + track.velocity * time_step_s
            
            # Increase uncertainty with time (process noise)
            process_noise_growth = 0.1  # m/cycle
            track.position_error = np.sqrt(
                track.position_error**2 + process_noise_growth**2
            )
    
    def get_active_tracks(self) -> List[TrackedThreat]:
        """Return all tracks that are not stale."""
        return [
            t for t in self.tracks.values()
            if not t.is_stale(self.max_track_age)
        ]
    
    def prune_stale_tracks(self):
        """Remove tracks older than max_age."""
        stale_ids = [
            tid for tid, track in self.tracks.items()
            if track.is_stale(self.max_track_age)
        ]
        
        for tid in stale_ids:
            logger.debug(f"Pruning stale track {tid}")
            del self.tracks[tid]


class PathThreatAnalyzer:
    """
    Analyzes threat risk relative to planned trajectory.
    
    **Risk Assessment**:
    - Threat position vs planned waypoints
    - Minimum distance to threat over trajectory
    - Time-to-closest-approach (if threat moving)
    - Sensor cross-section (RCS) indicating threat type
    """
    
    @staticmethod
    def compute_threat_risk(
        threat: TrackedThreat,
        trajectory_waypoints: np.ndarray,
        time_remaining_s: float,
    ) -> Tuple[float, np.ndarray]:
        """
        Compute risk score based on trajectory proximity.
        
        Args:
            threat: Tracked threat
            trajectory_waypoints: [N, 3] planned path
            time_remaining_s: Time available for mission
        
        Returns:
            (risk_score [0, 1], closest_point on trajectory)
        """
        # Find minimum distance between threat and trajectory
        distances = np.linalg.norm(trajectory_waypoints - threat.position, axis=1)
        closest_idx = np.argmin(distances)
        min_distance = distances[closest_idx]
        closest_point = trajectory_waypoints[closest_idx]
        
        # Compute time-to-closest-approach
        if np.linalg.norm(threat.velocity) > 0.1:
            # Threat is moving
            threat_to_closest = closest_point - threat.position
            closing_angle = np.dot(threat_to_closest, threat.velocity)
            
            if closing_angle > 0:
                # Threat approaching
                relative_speed = np.dot(threat.velocity, threat_to_closest) / (np.linalg.norm(threat_to_closest) + 1e-6)
                time_to_ca = np.linalg.norm(threat_to_closest) / max(relative_speed, 0.1)
            else:
                # Threat moving away
                time_to_ca = 999.0
        else:
            time_to_ca = 999.0
        
        # Risk factors
        # 1. Distance to closest approach (closer = higher risk)
        distance_risk = max(0.0, 1.0 - min_distance / 1000.0)  # 1000m baseline
        
        # 2. Time-to-closest-approach (sooner = higher risk)
        time_risk = max(0.0, 1.0 - time_to_ca / time_remaining_s) if time_remaining_s > 0 else 0.0
        
        # 3. Threat confidence (lower conf = higher uncertainty penalty)
        confidence_risk = 1.0 - threat.confidence
        
        # Combine factors (weighted average)
        risk = 0.5 * distance_risk + 0.3 * time_risk + 0.2 * confidence_risk
        
        return np.clip(risk, 0.0, 1.0), closest_point
    
    @staticmethod
    def compute_avoidance_heading(
        current_position: np.ndarray,
        threat_position: np.ndarray,
        current_heading: float,
        min_turn_angle_deg: float = 30.0,
    ) -> float:
        """
        Compute heading to avoid threat.
        
        Args:
            current_position: [x, y, z]
            threat_position: [x, y, z]
            current_heading: Current heading (deg)
            min_turn_angle_deg: Minimum turn angle
        
        Returns:
            Recommended heading (deg) to avoid threat
        """
        # Vector from threat to current position (escape direction)
        escape_vec = current_position - threat_position
        
        # Heading of escape vector (2D projection)
        escape_heading = np.degrees(np.arctan2(escape_vec[1], escape_vec[0]))
        
        # Add perpendicular offset for more aggressive avoidance
        perpendicular = escape_heading + 90.0
        
        # Select perpendicular that minimizes turn angle
        turn_perp = abs(perpendicular - current_heading)
        if turn_perp > 180:
            turn_perp = 360 - turn_perp
        
        turn_escape = abs(escape_heading - current_heading)
        if turn_escape > 180:
            turn_escape = 360 - turn_escape
        
        # Choose direction with smaller turn
        recommended_heading = perpendicular if turn_perp < turn_escape else escape_heading
        
        return recommended_heading % 360.0


class ReplanTriggerLogic:
    """
    Determines when sensor fusion should trigger replanning.
    """
    
    def __init__(
        self,
        risk_threshold_alert: float = 0.6,
        risk_threshold_critical: float = 0.8,
        time_to_closest_limit_s: float = 60.0,
    ):
        self.risk_alert = risk_threshold_alert
        self.risk_critical = risk_threshold_critical
        self.time_to_ca_limit = time_to_closest_limit_s
    
    def should_trigger_replan(
        self,
        active_threats: List[TrackedThreat],
        threat_risks: List[float],
    ) -> Tuple[bool, str]:
        """
        Determine if replanning should be triggered.
        
        Args:
            active_threats: List of tracked threats
            threat_risks: Risk scores computed by PathThreatAnalyzer
        
        Returns:
            (should_replan: bool, reason: str)
        """
        if not active_threats:
            return False, "No threats detected"
        
        # Check for critical threats
        for threat, risk in zip(active_threats, threat_risks):
            if risk > self.risk_critical:
                return True, f"Critical threat {threat.threat_id} (risk={risk:.2f})"
            
            if risk > self.risk_alert and threat.threat_type in [
                ThreatType.MISSILE_LAUNCH_SITE,
                ThreatType.MOVING_RADAR,
            ]:
                return True, f"High-priority threat {threat.threat_id}"
        
        # Check for cluster of medium threats
        medium_threats = [r for r in threat_risks if r > 0.5]
        if len(medium_threats) >= 2:
            return True, f"Multiple medium threats ({len(medium_threats)} detected)"
        
        return False, "No replan trigger conditions met"
