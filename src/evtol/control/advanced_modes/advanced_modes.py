"""
Advanced Flight Modes - Phase 2.3

Implements:
1. Intelligent mode selection based on threats & energy
2. Smooth hover-transition-cruise blending
3. Stall protection with recovery
4. Energy-optimal mode recommendation

This module enhances the basic flight mode system with threat awareness,
energy optimization, and advanced transition control.
"""

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from datetime import datetime


class FlightModeState(Enum):
    """Advanced flight mode enumeration."""
    HOVER = auto()
    TRANSITION = auto()
    CRUISE = auto()
    EMERGENCY = auto()


class SafetyLevel(Enum):
    """Stall and safety alerting."""
    GREEN = auto()        # All clear
    YELLOW = auto()       # Advisory (approaching limit)
    ORANGE = auto()       # Warning (near limit)
    RED = auto()          # Critical (limit exceeded)


@dataclass
class AirspeedEnvelope:
    """Airspeed limits for each flight mode."""
    hover_min: float = 0.0          # m/s
    transition_min: float = 5.0     # m/s (with pitch compensation)
    cruise_min: float = 12.0        # m/s (stall margin)
    cruise_max: float = 30.0        # m/s
    
    # Safety margins
    yellow_margin: float = 1.0      # Advisory threshold above min
    orange_margin: float = 0.5      # Warning threshold above min


@dataclass
class FlightModeTransitionConfig:
    """Configuration for advanced flight mode transitions."""
    # Transition thresholds
    hover_to_transition_speed_min: float = 5.0      # m/s
    transition_to_cruise_speed_min: float = 14.0    # m/s
    transition_to_cruise_speed_max: float = 22.0    # m/s
    
    # Transition timing
    acceleration_ramp_time: float = 10.0            # s (TRANSITION acceleration)
    deceleration_ramp_time: float = 5.0             # s (CRUISE to TRANSITION)
    
    # Motor mix blending
    rotor_to_forward_transition_duration: float = 5.0  # s
    
    # Safety thresholds
    min_altitude_for_cruise: float = 50.0           # m
    energy_reserve_threshold: float = 0.2           # 20% battery reserve
    

@dataclass
class MissionContext:
    """Mission-level context for mode decisions."""
    threat_level: float = 0.0           # [0,1] threat severity
    threat_imminent: bool = False       # Critical immediate threat
    remaining_distance_m: float = 1000.0
    remaining_time_s: float = 3600.0
    battery_soc: float = 1.0            # [0,1] state of charge
    altitude_m: float = 100.0
    wind_speed_ms: float = 0.0
    

@dataclass
class ModeInputState:
    """Current vehicle state."""
    airspeed_ms: float = 0.0
    altitude_m: float = 100.0
    pitch_rad: float = 0.0
    roll_rad: float = 0.0
    vz_ms: float = 0.0                 # Vertical velocity
    total_thrust_N: float = 0.0
    rotor_power_fraction: float = 0.0  # [0,1] rotor thrust / total
    forward_motor_power_fraction: float = 0.0


@dataclass
class AdvancedFlightModeState:
    """State tracking for advanced flight modes."""
    current_mode: FlightModeState = FlightModeState.HOVER
    target_mode: FlightModeState = FlightModeState.HOVER
    transition_progress: float = 0.0        # [0,1]
    mode_time_elapsed_s: float = 0.0
    
    # Safety state
    safety_level: SafetyLevel = SafetyLevel.GREEN
    stall_recovery_active: bool = False
    stall_recovery_time_remaining_s: float = 0.0
    
    # Recommendation
    recommended_mode: FlightModeState = FlightModeState.CRUISE
    mode_confidence: float = 0.0            # [0,1]
    energy_efficiency_score: float = 0.0


@dataclass
class BlendingCurve:
    """S-curve blending function generator."""
    start_time_s: float = 0.0
    duration_s: float = 10.0
    
    def alpha(self, t: float) -> float:
        """
        S-curve blending function [0,1].
        
        Uses cosine-based smooth polynomial for jerk-free blending.
        """
        if t < self.start_time_s:
            return 0.0
        if t > self.start_time_s + self.duration_s:
            return 1.0
        
        tau = (t - self.start_time_s) / self.duration_s  # [0,1]
        # Smooth S-curve: avoids jerky transitions
        return tau - (np.sin(2 * np.pi * tau) / (2 * np.pi))
    
    def d_alpha_dt(self, t: float) -> float:
        """Rate of change of alpha (for rate limiting)."""
        if t < self.start_time_s or t > self.start_time_s + self.duration_s:
            return 0.0
        
        tau = (t - self.start_time_s) / self.duration_s
        # Derivative of S-curve
        return (1.0 - np.cos(2 * np.pi * tau)) / self.duration_s


@dataclass
class MotorCommand:
    """Blended motor command during transitions."""
    rotor_thrust_frac: float = 0.0      # [0,1] rotor contribution
    forward_thrust_frac: float = 1.0    # [0,1] forward motor contribution
    rotor_rpm_cmd: float = 0.0
    forward_rpm_cmd: float = 0.0
    
    def __post_init__(self):
        """Validate fractions sum to 1."""
        total = self.rotor_thrust_frac + self.forward_thrust_frac
        if abs(total - 1.0) > 0.01:
            self.rotor_thrust_frac /= total
            self.forward_thrust_frac /= total


@dataclass
class ThreatMap:
    """Threat data from Phase 2.2 perception."""
    threats_detected: int = 0
    critical_threat_range_m: float = float('inf')
    avg_threat_level: float = 0.0  # [0,1]
    
    def is_imminent_threat(self, threat_range_threshold_m: float = 3000.0) -> bool:
        """Check if threats are imminent."""
        return self.critical_threat_range_m < threat_range_threshold_m


class AdvancedFlightModeManager:
    """
    Main advanced flight mode manager.
    
    Coordinates:
    - Mode selection (based on threats + energy)
    - Transition control (smooth blending)
    - Stall protection (safety enforcement)
    - Energy optimization (efficiency recommendations)
    """
    
    def __init__(self, config: Optional[FlightModeTransitionConfig] = None):
        self.config = config or FlightModeTransitionConfig()
        self.state = AdvancedFlightModeState()
        self.airspeed_envelope = AirspeedEnvelope()
        
        # Transition blending curve
        self._blend_curve: Optional[BlendingCurve] = None
        self._transition_start_time_s = 0.0
        
        # Safety monitoring
        self._stall_recovery_start_time = 0.0
        self._mode_history: list[tuple[float, FlightModeState]] = []
    
    def reset(self) -> None:
        """Reset to hover mode."""
        self.state = AdvancedFlightModeState(
            current_mode=FlightModeState.HOVER,
            target_mode=FlightModeState.HOVER
        )
        self._blend_curve = None
        self._mode_history = []
    
    def update(
        self,
        t: float,
        vehicle_state: ModeInputState,
        mission_context: MissionContext,
        threat_map: Optional[ThreatMap] = None,
        dt: float = 0.01
    ) -> AdvancedFlightModeState:
        """
        Update flight mode state machine.
        
        Args:
            t: Current time (s)
            vehicle_state: Current vehicle state
            mission_context: Mission-level context
            threat_map: Perceived threats
            dt: Time step (s)
            
        Returns:
            Updated AdvancedFlightModeState
        """
        threat_map = threat_map or ThreatMap()
        
        # Step 1: Update mode time and progression
        self.state.mode_time_elapsed_s += dt
        
        # Step 2: Handle active stall recovery
        if self.state.stall_recovery_active:
            self._update_stall_recovery(t, vehicle_state, dt)
        
        # Step 3: Check for stall conditions
        self._check_stall_envelope(vehicle_state, mission_context)
        
        # Step 4: Update transition blending
        if self.state.transition_progress < 1.0:
            self._update_transition_blend(t, vehicle_state, dt)
        else:
            self.state.current_mode = self.state.target_mode
        
        # Step 5: Select target mode (if not in transition)
        if self.state.transition_progress >= 1.0:
            self.state.target_mode = self._select_target_mode(
                vehicle_state, mission_context, threat_map
            )
            
            # Initiate transition if mode changed
            if self.state.target_mode != self.state.current_mode:
                self._initiate_mode_transition(t, vehicle_state, self.state.target_mode)
        
        # Step 6: Energy and efficiency analysis
        self._analyze_energy_efficiency(vehicle_state, mission_context)
        
        # Step 7: Log mode history
        if len(self._mode_history) == 0 or self._mode_history[-1][1] != self.state.current_mode:
            self._mode_history.append((t, self.state.current_mode))
        
        return self.state
    
    def _select_target_mode(
        self,
        vehicle_state: ModeInputState,
        mission_context: MissionContext,
        threat_map: ThreatMap
    ) -> FlightModeState:
        """
        Select optimal target flight mode.
        
        Decision hierarchy:
        1. If imminent threat → HOVER (immediate evasion)
        2. If high threat + adequate altitude → TRANSITION (speed up)
        3. If energy abundant + long distance → CRUISE (efficient)
        4. If stall risk → TRANSITION (safety first)
        5. Otherwise → maintain current mode
        """
        
        # Emergency: imminent threat
        if threat_map.is_imminent_threat(3000.0) and mission_context.threat_imminent:
            return FlightModeState.EMERGENCY if self.state.current_mode == FlightModeState.CRUISE else FlightModeState.HOVER
        
        # Threat level thresholds
        high_threat = threat_map.avg_threat_level > 0.6
        medium_threat = threat_map.avg_threat_level > 0.3
        
        # Energy levels
        low_energy = mission_context.battery_soc < self.config.energy_reserve_threshold
        abundant_energy = mission_context.battery_soc > 0.7
        
        # Decision logic
        if high_threat and vehicle_state.altitude_m > self.config.min_altitude_for_cruise:
            # Speed up from hover when threatened
            return FlightModeState.TRANSITION if self.state.current_mode == FlightModeState.HOVER else self.state.current_mode
        
        elif low_energy:
            # Cruise is most efficient
            return FlightModeState.CRUISE if vehicle_state.airspeed_ms > 12.0 else FlightModeState.TRANSITION
        
        elif abundant_energy and mission_context.remaining_distance_m > 5000.0:
            # Long range: cruise for efficiency
            return FlightModeState.CRUISE
        
        elif medium_threat or mission_context.remaining_distance_m < 500.0:
            # Slow for threat/precision
            return FlightModeState.TRANSITION if vehicle_state.airspeed_ms > 10.0 else FlightModeState.HOVER
        
        else:
            # Maintain current mode
            return self.state.current_mode
    
    def _initiate_mode_transition(
        self,
        t: float,
        vehicle_state: ModeInputState,
        target_mode: FlightModeState
    ) -> None:
        """Initiate transition to target mode."""
        
        # Determine transition duration
        if self.state.current_mode == FlightModeState.CRUISE and target_mode == FlightModeState.TRANSITION:
            duration = self.config.deceleration_ramp_time
        elif self.state.current_mode == FlightModeState.TRANSITION and target_mode == FlightModeState.CRUISE:
            duration = self.config.acceleration_ramp_time
        else:
            duration = 2.0  # Quick transitions for other mode pairs
        
        self._blend_curve = BlendingCurve(start_time_s=t, duration_s=duration)
        self._transition_start_time_s = t
        self.state.transition_progress = 0.0
        self.state.target_mode = target_mode
    
    def _update_transition_blend(
        self,
        t: float,
        vehicle_state: ModeInputState,
        dt: float
    ) -> None:
        """Update smooth transition blending."""
        if self._blend_curve is None:
            return
        
        alpha = self._blend_curve.alpha(t)
        self.state.transition_progress = alpha
        
        # Rate limiting on transition progress (prevent jerky changes)
        d_alpha = self._blend_curve.d_alpha_dt(t)
        max_rate = 0.2  # per second
        if d_alpha > max_rate:
            self.state.transition_progress = min(alpha, self.state.transition_progress + max_rate * dt)
    
    def _check_stall_envelope(
        self,
        vehicle_state: ModeInputState,
        mission_context: MissionContext
    ) -> None:
        """Monitor airspeed against stall envelope."""
        
        airspeed = vehicle_state.airspeed_ms
        pitch = vehicle_state.pitch_rad
        current_mode = self.state.current_mode
        
        # Determine minimum airspeed for current mode
        if current_mode == FlightModeState.HOVER:
            v_min = self.airspeed_envelope.hover_min
        elif current_mode == FlightModeState.TRANSITION:
            # Pitch-dependent stall margin
            pitch_deg = np.degrees(pitch)
            v_min = self.airspeed_envelope.transition_min + 0.1 * max(0, pitch_deg)
        else:  # CRUISE
            v_min = self.airspeed_envelope.cruise_min
        
        # Safety level assessment
        if airspeed < v_min:
            self.state.safety_level = SafetyLevel.RED
            if not self.state.stall_recovery_active:
                self.state.stall_recovery_active = True
        elif airspeed < v_min + self.airspeed_envelope.orange_margin:
            self.state.safety_level = SafetyLevel.ORANGE
        elif airspeed < v_min + self.airspeed_envelope.yellow_margin:
            self.state.safety_level = SafetyLevel.YELLOW
        else:
            self.state.safety_level = SafetyLevel.GREEN
            self.state.stall_recovery_active = False
    
    def _update_stall_recovery(
        self,
        t: float,
        vehicle_state: ModeInputState,
        dt: float
    ) -> None:
        """Handle stall recovery maneuver."""

        recovery_duration = 3.0  # seconds

        # Record start time on first entry (stall_recovery_active was just set True)
        if self._stall_recovery_start_time == 0.0:
            self._stall_recovery_start_time = t

        elapsed = t - self._stall_recovery_start_time
        if elapsed < recovery_duration:
            self.state.stall_recovery_time_remaining_s = recovery_duration - elapsed
        else:
            self.state.stall_recovery_active = False
            self.state.stall_recovery_time_remaining_s = 0.0
    
    def _analyze_energy_efficiency(
        self,
        vehicle_state: ModeInputState,
        mission_context: MissionContext
    ) -> None:
        """Analyze and score energy efficiency for each mode."""

        # Power estimates from VehicleEnergyModel at typical operating conditions
        # (hover: actuator-disk induced power at sea level, cruise: drag power at V_opt)
        # These are conservative design-point values; per-mission accuracy requires
        # a full motor+rotor model call.
        hover_power = 83.0      # kW — from dataset: mean power_hover_elec_W / 1000
        cruise_power = 211.0    # kW — from dataset: mean power_cruise_elec_W / 1000
        # Note: cruise has higher absolute power but far higher speed → more efficient per km

        # Hours of flight available at these power levels
        total_energy_kwh = mission_context.battery_soc * 50.0  # 50 kWh pack
        
        hover_hours = total_energy_kwh / hover_power if hover_power > 0 else float('inf')
        cruise_hours = total_energy_kwh / cruise_power if cruise_power > 0 else float('inf')
        
        # Efficiency score (higher is better)
        if vehicle_state.airspeed_ms > 14.0:
            self.state.energy_efficiency_score = cruise_hours / (cruise_hours + hover_hours)
            self.state.recommended_mode = FlightModeState.CRUISE
            self.state.mode_confidence = 0.85
        elif mission_context.remaining_distance_m > 5000.0:
            self.state.energy_efficiency_score = 0.7
            self.state.recommended_mode = FlightModeState.CRUISE
            self.state.mode_confidence = 0.7
        else:
            self.state.energy_efficiency_score = 0.4
            self.state.recommended_mode = FlightModeState.TRANSITION
            self.state.mode_confidence = 0.6
    
    def get_motor_blend_command(self) -> MotorCommand:
        """
        Get blended motor command for current transition state.
        
        Returns:
            MotorCommand with rotor/forward motor mix
        """
        progress = self.state.transition_progress
        
        if self.state.current_mode == FlightModeState.HOVER:
            return MotorCommand(rotor_thrust_frac=1.0, forward_thrust_frac=0.0)
        
        elif self.state.current_mode == FlightModeState.CRUISE:
            return MotorCommand(rotor_thrust_frac=0.05, forward_thrust_frac=0.95)
        
        else:  # TRANSITION
            # Blend between hover and cruise based on progress
            rotor_frac = 1.0 - progress * 0.95  # 100% → 5%
            fwd_frac = progress * 0.95           # 0% → 95%
            return MotorCommand(rotor_thrust_frac=rotor_frac, forward_thrust_frac=fwd_frac)
    
    def get_safety_recommendation(self) -> Optional[str]:
        """Get safety recommendation based on current state."""
        if self.state.safety_level == SafetyLevel.RED:
            return "STALL RECOVERY: Pitch down, increase thrust"
        elif self.state.safety_level == SafetyLevel.ORANGE:
            return "WARNING: Approaching stall, be prepared to recover"
        elif self.state.stall_recovery_active:
            return f"RECOVERY in progress: {self.state.stall_recovery_time_remaining_s:.1f}s remaining"
        return None


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    'AdvancedFlightModeManager',
    'FlightModeState',
    'SafetyLevel',
    'AdvancedFlightModeState',
    'BlendingCurve',
    'MotorCommand',
    'ThreatMap',
    'ModeInputState',
    'MissionContext',
    'AirspeedEnvelope',
    'FlightModeTransitionConfig',
]
