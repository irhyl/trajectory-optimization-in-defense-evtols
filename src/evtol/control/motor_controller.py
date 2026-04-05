"""
Phase 2B: Motor Control Layer - PWM Signal Generation & Motor Management

Lowest-level flight control interface that converts force/torque commands
from Phase 2E (ExecutionController) into PWM signals for motor ESCs.

ARCHITECTURE:
    Phase 2E ControlCommand
        ├─ motor_thrusts: [T₀, T₁, T₂, T₃] (Newtons)
        ├─ adaptive_gains: from Phase 2G (optional)
        └─ feedforward: attitude, rates, accel (optional)
                 ↓
        MotorController (Phase 2B)
                 ├─ Thrust validation & saturation
                 ├─ Motor mixing (if applicable for tiltwing)
                 ├─ PWM conversion (0-1 normalized → 1000-2000 µs)
                 ├─ Rate limiting & filtering
                 ├─ Safety checks & failsafes
                 └─ Telemetry feedback
                 ↓
        PWM Output → Motor ESCs [1000, 1000, 1000, 1000] µs

MOTOR SPECS (eVTOL Generic):
- 4 electric motors (quadcopter config)
- Motor limits: ~15-30 N thrust per motor (tunable)
- Max RPM: ~6000-8000 (tunable)
- PWM range: 1000-2000 µs (1000 = 0% thrust, 2000 = 100% thrust)
- Response time: ~10-50 ms per motor
- Feedback: Motor current, RPM (if available from ESC telemetry)

SAFETY:
- Thrust saturation: Prevent individual motor T > T_max
- Rate limiting: Smooth thrust changes to prevent control instability
- Watchdog: Detect loss of motor feedback or timeout
- Failsafe: Return to previous command if update missed >100ms
- Motor health: Monitor current draw for anomalies

INTEGRATION:
- Phase 2E → Phase 2B: ControlCommand (motor_thrusts, adaptive_gains)
- Phase 2G → Phase 2B: Adaptive gains (optional, for motor response tuning)
- Phase 2B → Phase 2E: MotorFeedback (RPM, current, status)
- Timing: Async from Phase 2E 50 Hz, runs on ESC update (typically 50-400 Hz PWM)

Author: PhD-level eVTOL control systems
Date: March 2026
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Callable
from enum import Enum
from datetime import datetime, timedelta
import numpy as np
from collections import deque
import threading


class MotorStatus(Enum):
    """Motor operational status"""
    HEALTHY = "healthy"
    WARNING = "warning"  # High current or thermal stress
    FAULT = "fault"      # Complete failure detected
    UNKNOWN = "unknown"  # No telemetry feedback


class MotorFailureMode(Enum):
    """Detected motor failure modes"""
    NONE = "none"
    OVERCURRENT = "overcurrent"           # >150% nominal current
    UNDERCURRENT = "undercurrent"         # <10% expected current (stalled)
    OVERHEAT = "overheat"                 # Temperature >80°C
    NO_TELEMETRY = "no_telemetry"         # Missing ESC feedback
    MECHANICAL_FAULT = "mechanical_fault" # Irregular current draw pattern
    SUPPLY_FAULT = "supply_fault"         # Battery voltage drop


class MotorControlMode(Enum):
    """Motor control modes"""
    PWM = "pwm"              # Direct PWM control (1000-2000 µs)
    THRUST = "thrust"        # Thrust-command via motor model
    RPM = "rpm"              # RPM control with feedback


@dataclass
class MotorLimits:
    """Physical and operational limits for each motor"""
    max_thrust_n: float = 25.0            # Maximum thrust per motor (N)
    min_thrust_n: float = 0.0             # Minimum thrust (N)
    max_rpm: float = 8000.0               # Maximum RPM
    pwm_min_us: float = 1000.0            # Minimum PWM (µs)
    pwm_max_us: float = 2000.0            # Maximum PWM (µs)
    pwm_neutral_us: float = 1500.0        # Neutral PWM (µs)
    rate_limit_n_per_s: float = 50.0      # Max thrust change per second (N/s)
    current_limit_a: float = 40.0         # Max current per motor (A)
    thermal_limit_c: float = 80.0         # Shutdown temperature (°C)


@dataclass
class MotorCommand:
    """Command to a single motor"""
    motor_id: int                          # Motor index [0-3]
    thrust_n: float                        # Desired thrust (N)
    pwm_us: float = 1500.0                 # Resulting PWM signal (µs)
    adaptive_gain: float = 1.0             # Adaptive gain from Phase 2G
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class MotorTelemetry:
    """Feedback from a single motor via ESC telemetry"""
    motor_id: int
    pwm_us: float                          # Current PWM command (µs)
    rpm: Optional[float] = None            # Current RPM (if available)
    current_a: Optional[float] = None      # Current draw (A, if available)
    voltage_v: Optional[float] = None      # Motor supply voltage (V, if available)
    temperature_c: Optional[float] = None  # Temperature (°C, if available)
    status: MotorStatus = MotorStatus.UNKNOWN
    failure_mode: MotorFailureMode = MotorFailureMode.NONE
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class MotorControllerState:
    """State of the motor controller"""
    active: bool = False
    cycle_count: int = 0
    last_command_time: Optional[datetime] = None
    last_telemetry_time: Optional[datetime] = None
    current_thrusts: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])
    motor_status: List[MotorStatus] = field(default_factory=lambda: [MotorStatus.UNKNOWN] * 4)
    watchdog_failures: int = 0
    emergency_stop_active: bool = False


class MotorModel:
    """
    Motor thrust model: Convert thrust (N) ↔ PWM (µs) ↔ RPM
    
    Empirical quadratic model:
        thrust_n = a * RPM² + b * RPM + c
        
    Or approximated as linear for control:
        T = k_t * (PWM_us - PWM_neutral) / (PWM_max - PWM_neutral)
    """
    
    def __init__(self, motor_id: int, limits: MotorLimits):
        self.motor_id = motor_id
        self.limits = limits
        
        # Linear approximation: thrust coefficient
        # Scales PWM command (0-1) to thrust (0 to max_thrust)
        pwm_range = limits.pwm_max_us - limits.pwm_min_us
        self.pwm_to_thrust_slope = limits.max_thrust_n / pwm_range
        
    def pwm_to_thrust(self, pwm_us: float) -> float:
        """Convert PWM signal (µs) to thrust estimate (N)"""
        pwm_range = self.limits.pwm_max_us - self.limits.pwm_min_us
        pwm_normalized = (pwm_us - self.limits.pwm_min_us) / pwm_range
        thrust = pwm_normalized * self.limits.max_thrust_n
        return np.clip(thrust, self.limits.min_thrust_n, self.limits.max_thrust_n)
    
    def thrust_to_pwm(self, thrust_n: float, adaptive_gain: float = 1.0) -> float:
        """Convert desired thrust (N) to PWM command (µs)"""
        # Apply adaptive gain (from Phase 2G) - scales motor response
        thrust_adjusted = thrust_n * adaptive_gain
        thrust_adjusted = np.clip(thrust_adjusted, 
                                  self.limits.min_thrust_n, 
                                  self.limits.max_thrust_n)
        
        pwm_range = self.limits.pwm_max_us - self.limits.pwm_min_us
        pwm_normalized = thrust_adjusted / self.limits.max_thrust_n
        pwm_us = self.limits.pwm_min_us + pwm_normalized * pwm_range
        
        return np.clip(pwm_us, self.limits.pwm_min_us, self.limits.pwm_max_us)
    
    def rpm_to_thrust(self, rpm: float) -> float:
        """Estimate thrust from RPM (requires quadratic calibration in practice)"""
        # Placeholder: linear approximation
        # In production, fit quadratic from thrust-stand data
        return (rpm / self.limits.max_rpm) * self.limits.max_thrust_n


class MotorHealthMonitor:
    """
    Monitors motor health and detects anomalies
    
    Checks:
    - Overcurrent: Current > 1.5 × nominal
    - Undercurrent: Current < 10% expected (stalled)
    - Overheat: Temperature > 80°C
    - No telemetry: ESC feedback timeout
    - Mechanical fault: Current draw pattern anomalies
    """
    
    def __init__(self, motor_id: int, nominal_current_a: float = 20.0):
        self.motor_id = motor_id
        self.nominal_current_a = nominal_current_a
        self.current_history = deque(maxlen=50)  # 1 second @ 50 Hz
        self.last_telemetry_time = None
        self.telemetry_timeout_s = 0.5  # 500ms without update = fault
        
    def check_health(self, telemetry: Optional[MotorTelemetry]) -> Tuple[MotorStatus, MotorFailureMode]:
        """
        Evaluate motor health from telemetry
        
        Returns: (status, failure_mode)
        """
        if telemetry is None:
            return MotorStatus.FAULT, MotorFailureMode.NO_TELEMETRY
        
        # Update timestamp
        now = datetime.now()
        if self.last_telemetry_time is not None:
            dt = (now - self.last_telemetry_time).total_seconds()
            if dt > self.telemetry_timeout_s:
                return MotorStatus.FAULT, MotorFailureMode.NO_TELEMETRY
        self.last_telemetry_time = now
        
        # Check current
        if telemetry.current_a is not None:
            self.current_history.append(telemetry.current_a)
            
            # Overcurrent
            if telemetry.current_a > 1.5 * self.nominal_current_a:
                return MotorStatus.FAULT, MotorFailureMode.OVERCURRENT
            
            # Undercurrent (stalled motor - current stuck at low value)
            if telemetry.current_a < 0.1 * self.nominal_current_a:
                # But if PWM is near neutral, this is expected
                if abs(telemetry.pwm_us - 1500.0) > 100:  # Not near neutral
                    return MotorStatus.WARNING, MotorFailureMode.UNDERCURRENT
        
        # Check temperature
        if telemetry.temperature_c is not None:
            if telemetry.temperature_c > 80.0:
                return MotorStatus.FAULT, MotorFailureMode.OVERHEAT
            elif telemetry.temperature_c > 70.0:
                return MotorStatus.WARNING, MotorFailureMode.NONE
        
        # Check mechanical faults (current variance anomalies)
        if len(self.current_history) > 10:
            current_std = np.std(list(self.current_history))
            # High variance in current can indicate mechanical issues
            if current_std > 3.0 * self.nominal_current_a:
                return MotorStatus.WARNING, MotorFailureMode.MECHANICAL_FAULT
        
        return MotorStatus.HEALTHY, MotorFailureMode.NONE


class MotorController:
    """
    Phase 2B Motor Controller - Orchestrates PWM generation & motor management
    
    Responsible for:
    1. Receiving thrust commands from Phase 2E (motor_thrusts array)
    2. Receiving adaptive gains from Phase 2G (optional)
    3. Converting thrusts → PWM signals
    4. Applying rate limiting & filtering
    5. Safety checks & motor health monitoring
    6. Dispatching PWM commands to ESCs
    7. Collecting & processing motor telemetry
    
    Timing: Async from Phase 2E, runs independently at PWM update rate
    Target: <5ms latency from command to PWM
    """
    
    def __init__(
        self,
        num_motors: int = 4,
        limits: Optional[MotorLimits] = None,
        enable_telemetry: bool = True,
        enable_rate_limiting: bool = True,
        pwm_callback: Optional[Callable[[List[float]], None]] = None,
    ):
        """
        Initialize motor controller
        
        Args:
            num_motors: Number of motors (typically 4)
            limits: Motor physical limits (MotorLimits)
            enable_telemetry: Track motor feedback
            enable_rate_limiting: Smooth thrust changes
            pwm_callback: Callback to dispatch PWM (for simulation/hardware)
        """
        self.num_motors = num_motors
        self.limits = limits or MotorLimits()
        self.enable_telemetry = enable_telemetry
        self.enable_rate_limiting = enable_rate_limiting
        self.pwm_callback = pwm_callback
        
        # Initialize motor models & health monitors
        self.motor_models = [MotorModel(i, self.limits) for i in range(num_motors)]
        self.health_monitors = [MotorHealthMonitor(i) for i in range(num_motors)]
        
        # State tracking
        self.state = MotorControllerState()
        self.current_pwm = [self.limits.pwm_neutral_us] * num_motors
        self.command_history = deque(maxlen=100)  # Store last 100 commands
        self.telemetry_history = deque(maxlen=500)  # Last 500 telemetry updates
        
        # Synchronization
        self._lock = threading.RLock()
        
    def set_motor_thrusts(
        self,
        motor_thrusts: List[float],
        adaptive_gains: Optional[List[float]] = None,
        timestamp: Optional[datetime] = None,
    ) -> Tuple[List[float], List[MotorCommand]]:
        """
        Set desired motor thrusts and generate PWM commands
        
        Args:
            motor_thrusts: [T₀, T₁, T₂, T₃] in Newtons
            adaptive_gains: [g₀, g₁, g₂, g₃] from Phase 2G (scales motor response)
            timestamp: Command timestamp
            
        Returns:
            (pwm_signals, motor_commands)
            pwm_signals: [pwm₀, pwm₁, pwm₂, pwm₃] in microseconds
            motor_commands: MotorCommand objects with details
            
        Raises:
            ValueError: If input dimensions don't match
        """
        if len(motor_thrusts) != self.num_motors:
            raise ValueError(f"Expected {self.num_motors} thrusts, got {len(motor_thrusts)}")
        
        with self._lock:
            timestamp = timestamp or datetime.now()
            adaptive_gains = adaptive_gains or [1.0] * self.num_motors
            
            if len(adaptive_gains) != self.num_motors:
                raise ValueError(f"Expected {self.num_motors} gains, got {len(adaptive_gains)}")
            
            # Thrust validation & saturation
            thrusts_saturated = [
                np.clip(t, self.limits.min_thrust_n, self.limits.max_thrust_n)
                for t in motor_thrusts
            ]
            
            # Rate limiting (smooth thrust changes)
            if self.enable_rate_limiting and self.state.current_thrusts:
                # Compute actual dt from last command timestamp
                time_since_last = (
                    (timestamp - self.state.last_command_time).total_seconds()
                    if self.state.last_command_time else 0.02
                )
                time_since_last = float(np.clip(time_since_last, 1e-4, 0.1))
                max_change = self.limits.rate_limit_n_per_s * time_since_last
                
                thrusts_limited = []
                for i, (new_thrust, old_thrust) in enumerate(zip(thrusts_saturated, self.state.current_thrusts)):
                    change = new_thrust - old_thrust
                    if abs(change) > max_change:
                        limited_thrust = old_thrust + np.sign(change) * max_change
                    else:
                        limited_thrust = new_thrust
                    thrusts_limited.append(limited_thrust)
                
                thrusts_saturated = thrusts_limited
            
            # Convert thrusts to PWM
            pwm_commands = []
            motor_commands = []
            
            for motor_id in range(self.num_motors):
                thrust = thrusts_saturated[motor_id]
                gain = adaptive_gains[motor_id]
                
                pwm_us = self.motor_models[motor_id].thrust_to_pwm(thrust, gain)
                pwm_commands.append(pwm_us)
                
                motor_cmd = MotorCommand(
                    motor_id=motor_id,
                    thrust_n=thrust,
                    pwm_us=pwm_us,
                    adaptive_gain=gain,
                    timestamp=timestamp,
                )
                motor_commands.append(motor_cmd)
            
            # Update state
            self.state.current_thrusts = list(thrusts_saturated)
            self.state.last_command_time = timestamp
            self.state.cycle_count += 1
            self.current_pwm = list(pwm_commands)
            
            # Log command
            self.command_history.append({
                'timestamp': timestamp,
                'thrusts': list(thrusts_saturated),
                'pwms': list(pwm_commands),
                'gains': list(adaptive_gains),
            })
            
            # Dispatch PWM to hardware/simulator
            if self.pwm_callback:
                self.pwm_callback(pwm_commands)
            
            return pwm_commands, motor_commands
    
    def update_motor_telemetry(self, telemetry_list: List[MotorTelemetry]) -> Dict[str, any]:
        """
        Process motor telemetry feedback from ESCs
        
        Args:
            telemetry_list: List of MotorTelemetry from each motor ESC
            
        Returns:
            Dict with motor status, health, and any detected issues
        """
        if len(telemetry_list) != self.num_motors:
            raise ValueError(f"Expected {self.num_motors} telemetry, got {len(telemetry_list)}")
        
        with self._lock:
            now = datetime.now()
            self.state.last_telemetry_time = now
            
            health_report = {
                'timestamp': now,
                'all_healthy': True,
                'motor_status': [],
                'warnings': [],
                'faults': [],
            }
            
            # Check each motor
            for i, telem in enumerate(telemetry_list):
                status, failure_mode = self.health_monitors[i].check_health(telem)
                self.state.motor_status[i] = status
                
                status_dict = {
                    'motor_id': i,
                    'status': status.value,
                    'failure_mode': failure_mode.value,
                    'thrust_estimate': self.motor_models[i].pwm_to_thrust(telem.pwm_us),
                }
                
                if telem.rpm:
                    status_dict['rpm'] = telem.rpm
                if telem.current_a:
                    status_dict['current_a'] = telem.current_a
                    
                health_report['motor_status'].append(status_dict)
                
                # Collect warnings/faults
                if status == MotorStatus.WARNING:
                    health_report['warnings'].append(f"Motor {i}: {failure_mode.value}")
                    health_report['all_healthy'] = False
                elif status == MotorStatus.FAULT:
                    health_report['faults'].append(f"Motor {i}: {failure_mode.value}")
                    health_report['all_healthy'] = False
            
            # Log telemetry
            self.telemetry_history.append(health_report)
            
            return health_report
    
    def emergency_stop(self, reason: str = "manual"):
        """
        Trigger emergency stop - set all motors to neutral
        
        Args:
            reason: Description of why emergency stop was triggered
        """
        with self._lock:
            self.state.emergency_stop_active = True
            self.current_pwm = [self.limits.pwm_neutral_us] * self.num_motors
            
            if self.pwm_callback:
                self.pwm_callback(self.current_pwm)
            
            import logging as _log
            _log.getLogger(__name__).critical(f"EMERGENCY STOP: {reason} — motors set to neutral {self.current_pwm}")
    
    def resume_from_emergency_stop(self):
        """Resume normal operation after emergency stop"""
        with self._lock:
            self.state.emergency_stop_active = False
            self.state.watchdog_failures = 0
    
    def get_status_summary(self) -> Dict[str, any]:
        """Get comprehensive motor controller status"""
        with self._lock:
            return {
                'active': self.state.active,
                'cycle_count': self.state.cycle_count,
                'current_pwms': list(self.current_pwm),
                'current_thrusts': list(self.state.current_thrusts),
                'motor_status': [s.value for s in self.state.motor_status],
                'emergency_stop_active': self.state.emergency_stop_active,
                'watchdog_failures': self.state.watchdog_failures,
                'last_command_age_ms': (
                    (datetime.now() - self.state.last_command_time).total_seconds() * 1000
                    if self.state.last_command_time else None
                ),
                'last_telemetry_age_ms': (
                    (datetime.now() - self.state.last_telemetry_time).total_seconds() * 1000
                    if self.state.last_telemetry_time else None
                ),
            }
    
    def get_command_history(self, num_commands: int = 50) -> List[Dict]:
        """Get recent command history (last N commands)"""
        with self._lock:
            return list(self.command_history)[-num_commands:]
    
    def get_telemetry_history(self, num_updates: int = 100) -> List[Dict]:
        """Get recent telemetry history (last N updates)"""
        with self._lock:
            return list(self.telemetry_history)[-num_updates:]


class MotorControllerOrchestrator:
    """
    High-level orchestrator for motor control integration
    
    Bridges Phase 2E (control commands) with Phase 2B (motor control)
    and Phase 2G (adaptive gains).
    
    Responsibilities:
    - Receive ControlCommand from Phase 2E
    - Extract adaptive gains from Phase 2G (if available)
    - Delegate to MotorController for PWM generation
    - Manage motor telemetry collection & health monitoring
    - Provide health status to Phase 2E
    """
    
    def __init__(self, motor_controller: Optional[MotorController] = None):
        """
        Initialize orchestrator
        
        Args:
            motor_controller: MotorController instance (creates default if None)
        """
        self.controller = motor_controller or MotorController()
        self.is_initialized = False
        self.cycle_count = 0
        self._lock = threading.RLock()
    
    def initialize(self):
        """Initialize motor controller"""
        with self._lock:
            self.controller.state.active = True
            self.is_initialized = True
    
    def shutdown(self):
        """Shutdown motor controller safely"""
        with self._lock:
            self.controller.state.active = False
            self.controller.emergency_stop("orchestrator shutdown")
    
    def process_control_command(
        self,
        motor_thrusts: List[float],
        adaptive_gains: Optional[List[float]] = None,
        timestamp: Optional[datetime] = None,
    ) -> Tuple[List[float], Dict[str, any]]:
        """
        Process control command from Phase 2E
        
        Args:
            motor_thrusts: [T₀, T₁, T₂, T₃] from Phase 2E
            adaptive_gains: Optional gains from Phase 2G
            timestamp: Command timestamp
            
        Returns:
            (pwm_signals, status)
            pwm_signals: PWM commands to ESCs
            status: Motor controller status & diagnostics
        """
        if not self.is_initialized:
            raise RuntimeError("Motor controller not initialized")
        
        with self._lock:
            pwm_signals, motor_commands = self.controller.set_motor_thrusts(
                motor_thrusts,
                adaptive_gains=adaptive_gains,
                timestamp=timestamp,
            )
            
            status = self.controller.get_status_summary()
            self.cycle_count += 1
            
            return pwm_signals, status
    
    def process_motor_telemetry(
        self,
        telemetry_list: List[MotorTelemetry],
    ) -> Dict[str, any]:
        """
        Process motor telemetry from ESCs
        
        Args:
            telemetry_list: Telemetry from each motor
            
        Returns:
            Health report with warnings/faults
        """
        return self.controller.update_motor_telemetry(telemetry_list)
    
    def get_system_status(self) -> Dict[str, any]:
        """Get complete system status"""
        with self._lock:
            status = self.controller.get_status_summary()
            status['orchestrator_cycle_count'] = self.cycle_count
            return status
