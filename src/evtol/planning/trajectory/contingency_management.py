"""
Contingency Management for Mission Execution

This module implements real-time contingency detection and response:

1. **Anomaly Detection**:
   - Trajectory tracking error exceeds threshold
   - Energy consumption exceeds predicted model
   - Threat exposure increases unexpectedly
   - Vehicle performance degradation

2. **Contingency Triggers**:
   - Engine failure (loss of rotor/motor)
   - Battery degradation (voltage sag, capacity loss)
   - Structural damage (wing laminar flow loss)
   - Environmental hazards (wind gust, microbursts)

3. **Response Actions**:
   - Path replanning to alternate landing site
   - Energy-optimal reduced-thrust operation
   - Abort mission / return-to-base
   - Emergency landing protocol

4. **Prediction & Prevention**:
   - Battery state-of-health trending
   - Motor/rotor temperature monitoring
   - Fuel economy prediction vs. baseline
   - Contingency threshold adaptation

Mathematical Framework
======================

Anomaly Detection:
An anomaly is detected when:
    |x_actual - x_predicted| > k·σ

where:
    - σ: Standard deviation of measurement noise
    - k: Detection threshold (typically 3-4 for 99.7% confidence)

Severity Scoring:
    Severity = (Error / Threshold)^p

where p = 2 emphasizes large deviations

Accumulation:
    Risk(t) = w₁·Severity₁(t) + w₂·Severity₂(t) + ... + wₙ·Severityₙ(t)

Triggers when Risk(t) > Threshold_risk

Author: Defense eVTOL Research Team
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from enum import Enum, auto
from collections.abc import Callable
from datetime import datetime, timedelta
import logging

from ..core.trajectory import Trajectory
from ..core.state import State, Pose, Velocity

logger = logging.getLogger(__name__)


class ContingencyLevel(Enum):
    """Severity level of contingency."""
    NOMINAL = auto()         # All systems normal
    WARNING = auto()         # Minor anomaly detected
    ALERT = auto()           # Significant degradation
    CRITICAL = auto()        # Immediate response required
    EMERGENCY = auto()       # Life-critical


class AnomalyType(Enum):
    """Type of detected anomaly."""
    TRACKING_ERROR = "tracking_error"       # Position deviation
    ENERGY_OVERRUN = "energy_overrun"       # Excessive battery drain
    THERMAL = "thermal"                     # Temperature high
    STRUCTURAL = "structural"               # Damage detected
    CONTROL = "control"                     # Actuator failure
    ENVIRONMENTAL = "environmental"         # Wind, weather
    THREAT = "threat"                       # Threat exposure
    UNKNOWN = "unknown"


class ResponseAction(Enum):
    """Action to take in response."""
    NONE = auto()
    ADJUST_HEADING = auto()      # Minor path adjustment
    REDUCE_SPEED = auto()        # Lower cruise speed
    CLIMB_ALTITUDE = auto()      # Gain altitude
    RETURN_TO_BASE = auto()      # Abort and return
    EMERGENCY_LAND = auto()      # Immediate landing
    REPLAN_MISSION = auto()      # Full re-optimization


@dataclass
class AnomalyDetection:
    """Anomaly detection result."""
    
    detected: bool = False
    anomaly_type: AnomalyType = AnomalyType.UNKNOWN
    severity: float = 0.0  # [0, 1]
    message: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Trend information
    trend: float = 0.0  # Rate of change
    duration: float = 0.0  # How long anomaly has been present [s]


@dataclass
class ContingencyEvent:
    """Contingency event triggered."""
    
    level: ContingencyLevel = ContingencyLevel.NOMINAL
    primary_anomaly: AnomalyType = AnomalyType.UNKNOWN
    recommended_action: ResponseAction = ResponseAction.NONE
    confidence: float = 0.0  # [0, 1]
    
    # Timeline
    detected_time: datetime = field(default_factory=datetime.now)
    response_deadline: float = 30.0  # Seconds until critical
    
    # Details
    details: dict = field(default_factory=dict)


class AnomalyDetector:
    """
    Real-time anomaly detection for flight operations.
    
    Monitors actual trajectory versus predicted model to detect
    degradation or unexpected behavior.
    """
    
    def __init__(
        self,
        prediction_model: Callable,
        detection_threshold: float = 3.0,  # σ threshold
        integration_window: float = 30.0,  # s
    ):
        """
        Initialize anomaly detector.
        
        Args:
            prediction_model: Function that predicts state from reference
            detection_threshold: Standard deviation threshold for detection
            integration_window: Time window for anomaly accumulation
        """
        self.predict = prediction_model
        self.k_threshold = detection_threshold
        self.integration_window = integration_window
        
        # History buffers
        self.error_history: list[float] = []
        self.anomaly_history: list[AnomalyDetection] = []
        
        # Noise model (estimated from data)
        self.position_noise_sigma = 1.0  # m
        self.energy_noise_sigma = 500.0  # Wh
        self.thermal_noise_sigma = 2.0   # °C
        
        logger.info(f"AnomalyDetector initialized: k_threshold={detection_threshold}")
    
    def detect_tracking_error(
        self,
        actual_state: State,
        reference_state: State,
    ) -> AnomalyDetection:
        """
        Detect position tracking error.
        
        Args:
            actual_state: Measured vehicle state
            reference_state: Reference trajectory state
            
        Returns:
            Anomaly detection result
        """
        pos_actual = actual_state.pose.position
        pos_ref = reference_state.pose.position
        
        # Euclidean distance error
        error = np.linalg.norm(pos_actual - pos_ref)
        self.error_history.append(error)
        
        # Keep window
        max_samples = int(self.integration_window * 50)  # Assume 50 Hz
        if len(self.error_history) > max_samples:
            self.error_history.pop(0)
        
        # Compute statistics
        if len(self.error_history) > 1:
            mean_error = np.mean(self.error_history)
            std_error = np.std(self.error_history)
        else:
            mean_error = error
            std_error = self.position_noise_sigma
        
        # Severity: how many sigma above baseline?
        threshold_error = self.k_threshold * std_error
        severity = max(0, (error - threshold_error) / std_error)
        
        detected = error > threshold_error
        
        # Trend
        if len(self.error_history) > 10:
            recent_mean = np.mean(self.error_history[-10:])
            older_mean = np.mean(self.error_history[:-10])
            trend = recent_mean - older_mean
        else:
            trend = 0.0
        
        return AnomalyDetection(
            detected=detected,
            anomaly_type=AnomalyType.TRACKING_ERROR,
            severity=min(1.0, severity / self.k_threshold),
            message=f"Position error: {error:.1f}m (threshold: {threshold_error:.1f}m)",
            trend=trend,
        )
    
    def detect_energy_anomaly(
        self,
        actual_energy: float,
        predicted_energy: float,
        energy_sigma: float = 500.0,
    ) -> AnomalyDetection:
        """
        Detect excessive energy consumption.
        
        Args:
            actual_energy: Measured battery SOC [Wh]
            predicted_energy: Model prediction [Wh]
            energy_sigma: Noise standard deviation [Wh]
            
        Returns:
            Anomaly detection result
        """
        # Energy overrun
        overrun = predicted_energy - actual_energy
        
        threshold = self.k_threshold * energy_sigma
        severity = max(0, (overrun - threshold) / energy_sigma)
        
        detected = overrun > threshold
        
        return AnomalyDetection(
            detected=detected,
            anomaly_type=AnomalyType.ENERGY_OVERRUN,
            severity=min(1.0, severity / self.k_threshold),
            message=f"Energy overrun: {overrun:.0f} Wh",
        )
    
    def detect_thermal_anomaly(
        self,
        motor_temp: float,
        threshold_temp: float = 80.0,
    ) -> AnomalyDetection:
        """
        Detect motor thermal issues.
        
        Args:
            motor_temp: Current motor temperature [°C]
            threshold_temp: Maximum allowable [°C]
            
        Returns:
            Anomaly detection result
        """
        overtemp = motor_temp - threshold_temp
        detected = overtemp > 0
        
        # Severity scales with temperature
        if threshold_temp > 0:
            severity = overtemp / 20.0  # 20°C above threshold = severe
        else:
            severity = 0.0
        
        return AnomalyDetection(
            detected=detected,
            anomaly_type=AnomalyType.THERMAL,
            severity=min(1.0, severity),
            message=f"Motor temperature: {motor_temp:.1f}°C (limit: {threshold_temp:.1f}°C)",
        )
    
    def detect_threat_anomaly(
        self,
        current_threat: float,
        baseline_threat: float,
        threshold_increase: float = 0.3,  # 30% increase
    ) -> AnomalyDetection:
        """
        Detect unexpected threat exposure increase.
        
        Args:
            current_threat: Current threat level [0, 1]
            baseline_threat: Expected threat level [0, 1]
            threshold_increase: Sensitivity to increase
            
        Returns:
            Anomaly detection result
        """
        threat_increase = current_threat - baseline_threat
        detected = threat_increase > threshold_increase
        
        # Severity proportional to increase
        if threshold_increase > 0:
            severity = threat_increase / threshold_increase
        else:
            severity = 0.0
        
        return AnomalyDetection(
            detected=detected,
            anomaly_type=AnomalyType.THREAT,
            severity=min(1.0, severity),
            message=f"Threat increase: {threat_increase:.1%} (baseline: {baseline_threat:.1%})",
        )


class ContingencyTrigger:
    """
    Contingency trigger logic.
    
    Combines multiple anomalies to determine if contingency response needed.
    """
    
    def __init__(self):
        """Initialize contingency trigger."""
        
        # Anomaly weights (sum to 1.0)
        self.weights = {
            AnomalyType.TRACKING_ERROR: 0.25,
            AnomalyType.ENERGY_OVERRUN: 0.30,
            AnomalyType.THERMAL: 0.25,
            AnomalyType.THREAT: 0.15,
            AnomalyType.CONTROL: 0.05,
        }
        
        # Severity thresholds for each level
        self.thresholds = {
            ContingencyLevel.WARNING: 0.3,
            ContingencyLevel.ALERT: 0.6,
            ContingencyLevel.CRITICAL: 0.85,
            ContingencyLevel.EMERGENCY: 0.95,
        }
        
        # Recent anomalies
        self.recent_anomalies: list[AnomalyDetection] = []
    
    def evaluate(
        self,
        anomalies: list[AnomalyDetection],
    ) -> ContingencyEvent:
        """
        Evaluate anomalies and determine response.
        
        Args:
            anomalies: List of detected anomalies
            
        Returns:
            Contingency event with recommended action
        """
        self.recent_anomalies = anomalies
        
        # Compute composite risk score
        risk_score = 0.0
        primary_anomaly = AnomalyType.UNKNOWN
        max_severity = 0.0
        
        for anomaly in anomalies:
            if not anomaly.detected:
                continue
            
            weight = self.weights.get(anomaly.anomaly_type, 0.0)
            risk_score += anomaly.severity * weight
            
            if anomaly.severity > max_severity:
                max_severity = anomaly.severity
                primary_anomaly = anomaly.anomaly_type
        
        # Determine level
        level = ContingencyLevel.NOMINAL
        for lv, threshold in sorted(self.thresholds.items(), 
                                   key=lambda x: x[1], reverse=True):
            if risk_score >= threshold:
                level = lv
                break
        
        # Select response action
        action = self._select_action(level, primary_anomaly)
        
        # Response deadline (minutes available before critical)
        response_deadline = 30.0 if level == ContingencyLevel.ALERT else \
                           5.0 if level == ContingencyLevel.CRITICAL else \
                           0.5 if level == ContingencyLevel.EMERGENCY else \
                           float('inf')
        
        return ContingencyEvent(
            level=level,
            primary_anomaly=primary_anomaly,
            recommended_action=action,
            confidence=min(1.0, risk_score),
            response_deadline=response_deadline,
            details={
                'risk_score': risk_score,
                'anomaly_count': len([a for a in anomalies if a.detected]),
            },
        )
    
    def _select_action(
        self,
        level: ContingencyLevel,
        anomaly: AnomalyType,
    ) -> ResponseAction:
        """Select response action based on level and type."""
        
        if level == ContingencyLevel.NOMINAL:
            return ResponseAction.NONE
        
        if level == ContingencyLevel.WARNING:
            if anomaly == AnomalyType.THERMAL:
                return ResponseAction.REDUCE_SPEED
            elif anomaly == AnomalyType.ENERGY_OVERRUN:
                return ResponseAction.REDUCE_SPEED
            else:
                return ResponseAction.ADJUST_HEADING
        
        if level == ContingencyLevel.ALERT:
            if anomaly == AnomalyType.THREAT:
                return ResponseAction.CLIMB_ALTITUDE
            elif anomaly == AnomalyType.ENERGY_OVERRUN:
                return ResponseAction.RETURN_TO_BASE
            else:
                return ResponseAction.REPLAN_MISSION
        
        if level in (ContingencyLevel.CRITICAL, ContingencyLevel.EMERGENCY):
            return ResponseAction.RETURN_TO_BASE
        
        return ResponseAction.NONE


class ContingencyManager:
    """
    Master contingency management system.
    
    Orchestrates anomaly detection, triggering, and response.
    """
    
    def __init__(
        self,
        detector: AnomalyDetector,
        trigger: ContingencyTrigger,
    ):
        """
        Initialize contingency manager.
        
        Args:
            detector: Anomaly detection engine
            trigger: Contingency trigger logic
        """
        self.detector = detector
        self.trigger = trigger
        
        self.current_event: ContingencyEvent | None = None
        self.event_history: list[ContingencyEvent] = []
    
    def update(
        self,
        actual_state: State,
        reference_state: State,
        battery_energy: float,
        predicted_energy: float,
        motor_temp: float = 70.0,
        threat_level: float = 0.1,
    ) -> ContingencyEvent:
        """
        Update contingency status (call once per control cycle).
        
        Args:
            actual_state: Current vehicle state
            reference_state: Reference trajectory
            battery_energy: Battery state-of-charge [Wh]
            predicted_energy: Model prediction [Wh]
            motor_temp: Motor temperature [°C]
            threat_level: Current threat exposure [0, 1]
            
        Returns:
            Contingency event (may be nominal)
        """
        # Detect anomalies
        anomalies = [
            self.detector.detect_tracking_error(actual_state, reference_state),
            self.detector.detect_energy_anomaly(battery_energy, predicted_energy),
            self.detector.detect_thermal_anomaly(motor_temp),
            self.detector.detect_threat_anomaly(threat_level, 0.1),
        ]
        
        # Evaluate and trigger
        event = self.trigger.evaluate(anomalies)
        
        # Track history
        if event.level != ContingencyLevel.NOMINAL:
            self.event_history.append(event)
            
            if self.current_event is None or event.level > self.current_event.level:
                self.current_event = event
                logger.warning(
                    f"Contingency {event.level.name}: {event.primary_anomaly.value} "
                    f"(action: {event.recommended_action.name})"
                )
        
        return event
    
    def reset(self):
        """Reset contingency manager after resolution."""
        self.current_event = None
        self.detector.error_history.clear()
        self.detector.anomaly_history.clear()
        logger.info("Contingency manager reset")
