"""
Stall Protection - Phase 2.3

Monitors airspeed against mode-specific envelopes and executes recovery
maneuvers when stall is imminent or occurring.

Includes:
- Minimum airspeed enforcement (mode-dependent)
- Safety alerting (Yellow/Orange/Red)
- Automatic recovery procedures
- Recovery maneuver logging
"""

from dataclasses import dataclass, field
from typing import Optional
from enum import Enum, auto
import numpy as np


class RecoveryPhase(Enum):
    """Phases of stall recovery maneuver."""
    INACTIVE = auto()
    PITCH_DOWN = auto()           # Initial pitch down
    THRUST_INCREASE = auto()      # Increase total thrust
    STABILIZE = auto()            # Stabilize airspeed
    RETURN_TO_NORMAL = auto()     # Transition back to normal control


@dataclass
class StallRecoveryProfile:
    """Recovery maneuver profile and timing."""
    phase: RecoveryPhase = RecoveryPhase.INACTIVE
    phase_time_s: float = 0.0
    total_time_s: float = 0.0
    
    max_total_time_s: float = 5.0  # Max duration
    
    # Recovery commands
    pitch_command_rad: float = 0.0     # Target pitch (negative = down)
    thrust_command_n: float = 0.0      # Target total thrust
    
    # Phase timing
    pitch_down_duration_s: float = 0.5
    thrust_up_duration_s: float = 1.0
    stabilize_duration_s: float = 2.0
    recovery_waypoint_insert_offset_m: float = 200.0
    
    # Logging
    trigger_airspeed_ms: float = 0.0
    trigger_time_s: float = 0.0


@dataclass
class StallEnvelopeMargins:
    """Safety margins for stall envelope."""
    # Minimum airspeed thresholds
    hover_min_ms: float = 0.0
    transition_min_ms: float = 5.0
    cruise_min_ms: float = 12.0
    
    # Safety alerting margins (above minimum)
    yellow_alert_margin_ms: float = 1.0    # Advisory
    orange_alert_margin_ms: float = 0.5    # Warning
    
    def get_alert_threshold(self, alert_level: str, mode: str) -> float:
        """Get airspeed threshold for alert level and mode."""
        v_min = getattr(self, f'{mode.lower()}_min_ms')
        if alert_level == 'YELLOW':
            return v_min + self.yellow_alert_margin_ms
        elif alert_level == 'ORANGE':
            return v_min + self.orange_alert_margin_ms
        else:
            return v_min


@dataclass
class PitchEnvelopeConstraint:
    """Pitch limitations during different modes."""
    hover_max_pitch_rad: float = np.radians(10)
    transition_max_pitch_rad: float = np.radians(25)
    cruise_max_pitch_rad: float = np.radians(15)
    
    def get_max_pitch(self, mode: str) -> float:
        """Get maximum pitch for mode."""
        return getattr(self, f'{mode.lower()}_max_pitch_rad')


class StallDetector:
    """
    Monitors vehicle state for stall conditions.
    
    Uses airspeed, mode, and pitch to determine stall risk.
    """
    
    def __init__(self, envelope: Optional[StallEnvelopeMargins] = None):
        self.envelope = envelope or StallEnvelopeMargins()
        self._alert_state = 'GREEN'
        self._stall_imminent = False
        self._stall_active = False
    
    def update(
        self,
        airspeed_ms: float,
        pitch_rad: float,
        current_mode: str,
        dt: float = 0.01
    ) -> dict:
        """
        Update stall detection state.
        
        Args:
            airspeed_ms: Current airspeed (m/s)
            pitch_rad: Current pitch (rad)
            current_mode: Current flight mode ('HOVER', 'TRANSITION', 'CRUISE')
            dt: Time step (s)
            
        Returns:
            Dictionary with detection status
        """
        
        # Get minimum airspeed for mode
        v_min_key = f'{current_mode.lower()}_min_ms'
        v_min = getattr(self.envelope, v_min_key, 0.0)
        
        # Apply pitch correction for transition mode
        if current_mode == 'TRANSITION':
            pitch_deg = np.degrees(pitch_rad)
            v_min += 0.1 * max(0, pitch_deg)
        
        # Determine alert level
        if airspeed_ms < v_min:
            alert_state = 'RED'
            stall_active = True
            stall_imminent = True
        elif airspeed_ms < v_min + self.envelope.orange_alert_margin_ms:
            alert_state = 'ORANGE'
            stall_active = False
            stall_imminent = True
        elif airspeed_ms < v_min + self.envelope.yellow_alert_margin_ms:
            alert_state = 'YELLOW'
            stall_active = False
            stall_imminent = False
        else:
            alert_state = 'GREEN'
            stall_active = False
            stall_imminent = False
        
        self._alert_state = alert_state
        self._stall_imminent = stall_imminent
        self._stall_active = stall_active
        
        return {
            'alert_state': alert_state,
            'stall_imminent': stall_imminent,
            'stall_active': stall_active,
            'airspeed_ms': airspeed_ms,
            'v_min_ms': v_min,
            'margin_ms': airspeed_ms - v_min,
        }


class StallProtectionController:
    """
    Executes stall recovery maneuvers.
    
    When stall is detected:
    1. Pitch down immediately (-5°)
    2. Increase thrust (+15%)
    3. Monitor airspeed recovery
    4. Return to normal operations
    """
    
    def __init__(
        self,
        envelope: Optional[StallEnvelopeMargins] = None,
        pitch_constraint: Optional[PitchEnvelopeConstraint] = None
    ):
        self.envelope = envelope or StallEnvelopeMargins()
        self.pitch_constraint = pitch_constraint or PitchEnvelopeConstraint()
        
        self.recovery_profile = StallRecoveryProfile()
        self._recovering = False
        self._recovery_start_time = 0.0
    
    def check_recovery_needed(
        self,
        airspeed_ms: float,
        pitch_rad: float,
        mode: str,
        total_thrust_n: float,
        mass_kg: float = 2500.0,
    ) -> bool:
        """Check if recovery is needed."""
        # Re-use the persistent detector stored on this instance to avoid
        # allocating a new StallDetector object on every control cycle.
        if not hasattr(self, '_detector'):
            self._detector = StallDetector(self.envelope)
        detection = self._detector.update(airspeed_ms, pitch_rad, mode)
        
        if detection['stall_active']:
            self._recovery_start_time = 0.0  # Will be set in compute_recovery
            self._recovering = True
            return True
        
        return False
    
    def compute_recovery_command(
        self,
        t: float,
        airspeed_ms: float,
        pitch_rad: float,
        mode: str,
        total_thrust_n: float,
        mass_kg: float = 2500.0,
        max_thrust_n: float = 40000.0,
    ) -> tuple[float, float, RecoveryPhase]:
        """
        Compute recovery command.
        
        Args:
            t: Current time (s)
            airspeed_ms: Current airspeed (m/s)
            pitch_rad: Current pitch (rad)
            mode: Current flight mode
            total_thrust_n: Current total thrust (N)
            mass_kg: Vehicle mass (kg)
            max_thrust_n: Maximum available thrust (N)
            
        Returns:
            (pitch_cmd_rad, thrust_cmd_n, phase)
        """
        
        if not self._recovering:
            return 0.0, 0.0, RecoveryPhase.INACTIVE
        
        if self.recovery_profile.phase == RecoveryPhase.INACTIVE:
            # Initialize recovery
            self.recovery_profile.phase = RecoveryPhase.PITCH_DOWN
            self.recovery_profile.phase_time_s = 0.0
            self.recovery_profile.trigger_airspeed_ms = airspeed_ms
            self.recovery_profile.trigger_time_s = t
        
        # Update phase timing
        self.recovery_profile.phase_time_s += 0.01  # Assume 100 Hz update
        self.recovery_profile.total_time_s += 0.01
        
        # Compute recovery commands based on phase
        pitch_cmd = 0.0
        thrust_cmd = 0.0
        
        if self.recovery_profile.phase_time_s < self.recovery_profile.pitch_down_duration_s:
            # Phase 1: Pitch down
            self.recovery_profile.phase = RecoveryPhase.PITCH_DOWN
            pitch_cmd = np.radians(-5.0)  # Pitch down 5°
            thrust_cmd = total_thrust_n + mass_kg * 9.81 * 0.15  # +15% thrust
        
        elif self.recovery_profile.phase_time_s < (
            self.recovery_profile.pitch_down_duration_s + 
            self.recovery_profile.thrust_up_duration_s
        ):
            # Phase 2: Continue thrust increase
            self.recovery_profile.phase = RecoveryPhase.THRUST_INCREASE
            pitch_cmd = np.radians(-3.0)  # Reduce pitch-down gradually
            thrust_cmd = total_thrust_n + mass_kg * 9.81 * 0.20  # +20% thrust
        
        elif self.recovery_profile.phase_time_s < (
            self.recovery_profile.pitch_down_duration_s +
            self.recovery_profile.thrust_up_duration_s +
            self.recovery_profile.stabilize_duration_s
        ):
            # Phase 3: Stabilize
            self.recovery_profile.phase = RecoveryPhase.STABILIZE
            pitch_cmd = 0.0  # Return to level
            thrust_cmd = total_thrust_n + mass_kg * 9.81 * 0.10  # +10% thrust
        
        else:
            # Recovery complete
            self.recovery_profile.phase = RecoveryPhase.RETURN_TO_NORMAL
            self._recovering = False
            pitch_cmd = 0.0
            thrust_cmd = 0.0
        
        # Clamp commands
        thrust_cmd = min(thrust_cmd, max_thrust_n * 0.95)  # 95% max for safety
        
        self.recovery_profile.pitch_command_rad = pitch_cmd
        self.recovery_profile.thrust_command_n = thrust_cmd
        
        return pitch_cmd, thrust_cmd, self.recovery_profile.phase
    
    def is_recovering(self) -> bool:
        """Check if recovery is active."""
        return self._recovering
    
    def get_recovery_status(self) -> dict:
        """Get detailed recovery status."""
        return {
            'is_recovering': self._recovering,
            'phase': self.recovery_profile.phase.name,
            'phase_time_s': self.recovery_profile.phase_time_s,
            'total_time_s': self.recovery_profile.total_time_s,
            'pitch_cmd_rad': self.recovery_profile.pitch_command_rad,
            'thrust_cmd_n': self.recovery_profile.thrust_command_n,
            'trigger_airspeed_ms': self.recovery_profile.trigger_airspeed_ms,
        }


class ComprehensiveStallProtection:
    """
    Complete stall protection system combining detection and recovery.
    
    Provides:
    - Continuous airspeed monitoring
    - Multi-level safety alerting
    - Automatic recovery when needed
    - Comprehensive logging
    """
    
    def __init__(
        self,
        envelope: Optional[StallEnvelopeMargins] = None,
        pitch_constraint: Optional[PitchEnvelopeConstraint] = None,
    ):
        self.detector = StallDetector(envelope)
        self.controller = StallProtectionController(envelope, pitch_constraint)
        self.envelope = envelope or StallEnvelopeMargins()
        
        # Event logging
        self.alert_events: list[tuple[float, str, str]] = []  # (time, mode, alert_level)
        self.recovery_events: list[tuple[float, float, float]] = []  # (time, airspeed, thrust)
    
    def update(
        self,
        t: float,
        airspeed_ms: float,
        pitch_rad: float,
        mode: str,
        total_thrust_n: float,
        mass_kg: float = 2500.0,
        max_thrust_n: float = 40000.0,
        dt: float = 0.01,
    ) -> dict:
        """
        Complete stall protection update.
        
        Returns:
            Dictionary with status, recommendations, and commands
        """
        
        # Step 1: Detect stall condition
        detection = self.detector.update(airspeed_ms, pitch_rad, mode, dt)
        alert_state = detection['alert_state']
        
        # Log alert state changes
        if len(self.alert_events) == 0 or self.alert_events[-1][2] != alert_state:
            self.alert_events.append((t, mode, alert_state))
        
        # Step 2: Compute recovery (if needed)
        pitch_cmd = 0.0
        thrust_cmd = 0.0
        phase = RecoveryPhase.INACTIVE
        
        if alert_state == 'RED' or self.controller.is_recovering():
            pitch_cmd, thrust_cmd, phase = self.controller.compute_recovery_command(
                t, airspeed_ms, pitch_rad, mode, total_thrust_n, mass_kg, max_thrust_n
            )
            self.recovery_events.append((t, airspeed_ms, thrust_cmd))
        
        # Step 3: Generate safety recommendation
        recommendation = None
        if alert_state == 'RED':
            recommendation = "STALL CRITICAL: Recovery in progress"
        elif alert_state == 'ORANGE':
            recommendation = f"WARNING: Airspeed {airspeed_ms:.1f} m/s near minimum"
        elif alert_state == 'YELLOW':
            recommendation = f"ADVISORY: Approaching stall margin"
        
        return {
            'alert_state': alert_state,
            'stall_imminent': detection['stall_imminent'],
            'stall_active': detection['stall_active'],
            'airspeed_ms': airspeed_ms,
            'v_min_recommended': detection['v_min_ms'],
            'airspeed_margin_ms': detection['margin_ms'],
            'recovery_active': self.controller.is_recovering(),
            'recovery_phase': phase.name,
            'pitch_command_rad': pitch_cmd,
            'thrust_command_n': thrust_cmd,
            'recommendation': recommendation,
            'recovery_status': self.controller.get_recovery_status(),
        }


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    'ComprehensiveStallProtection',
    'StallDetector',
    'StallProtectionController',
    'StallRecoveryProfile',
    'StallEnvelopeMargins',
    'PitchEnvelopeConstraint',
    'RecoveryPhase',
]
