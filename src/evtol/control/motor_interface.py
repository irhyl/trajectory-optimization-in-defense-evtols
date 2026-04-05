"""
Motor Interface & PWM Conversion Utilities

Hardware abstraction layer for PWM signal generation and motor interface.
Provides low-level motor control primitives and hardware-agnostic interfaces.

Components:
1. PWMConverter: Bidirectional PWM ↔ Thrust conversion
2. MotorESCInterface: Abstract interface for ESC hardware
3. SimulatedESC: Simulation backend for testing
4. MotorCommandQueue: Thread-safe command queueing
"""

from dataclasses import dataclass
from typing import Optional, Callable, List, Dict
from enum import Enum
import numpy as np
from datetime import datetime
import threading


class ESCType(Enum):
    """Supported ESC types"""
    GENERIC_PWM = "generic_pwm"      # Standard 1000-2000 µs PWM
    ONESHOT42 = "oneshot42"          # OneShot125 protocol (faster)
    ONESHOT125 = "oneshot125"        # OneShot125 protocol
    SERIALWIRE = "serialwire"        # Serial protocol (future)
    CAN_PROTOCOL = "can_protocol"    # CAN bus (future)


@dataclass
class PWMSignal:
    """A single PWM signal"""
    motor_id: int
    pulse_width_us: float     # Pulse width in microseconds (1000-2000)
    frequency_hz: float = 50.0  # PWM frequency (50-400 Hz typical)
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class PWMConverter:
    """
    Bidirectional converter: Thrust ↔ PWM ↔ RPM
    
    Physical model:
        PWM (1000-2000 µs) ↔ Throttle (0-100%) ↔ Thrust (0-Tmax)
    
    Linear approximation (production uses motor thrust curve):
        thrust(pwm) = (pwm - 1000) / 1000 * Tmax
        pwm(thrust) = 1000 + (thrust / Tmax) * 1000
    """
    
    def __init__(
        self,
        motor_id: int,
        max_thrust_n: float = 25.0,
        pwm_min_us: float = 1000.0,
        pwm_max_us: float = 2000.0,
        pwm_neutral_us: float = 1500.0,
    ):
        self.motor_id = motor_id
        self.max_thrust_n = max_thrust_n
        self.pwm_min_us = pwm_min_us
        self.pwm_max_us = pwm_max_us
        self.pwm_neutral_us = pwm_neutral_us
        
        self.pwm_range = pwm_max_us - pwm_min_us
        self.thrust_to_pwm_slope = self.pwm_range / max_thrust_n
        
    def pwm_to_throttle(self, pwm_us: float) -> float:
        """
        Convert PWM (µs) to throttle percentage (0-1)
        
        Args:
            pwm_us: PWM pulse width in microseconds
            
        Returns:
            Throttle 0.0 (0%) to 1.0 (100%)
        """
        throttle = (pwm_us - self.pwm_min_us) / self.pwm_range
        return np.clip(throttle, 0.0, 1.0)
    
    def throttle_to_pwm(self, throttle: float) -> float:
        """
        Convert throttle (0-1) to PWM (µs)
        
        Args:
            throttle: Normalized throttle 0.0-1.0
            
        Returns:
            PWM pulse width in microseconds
        """
        throttle = np.clip(throttle, 0.0, 1.0)
        pwm_us = self.pwm_min_us + throttle * self.pwm_range
        return pwm_us
    
    def thrust_to_pwm(self, thrust_n: float) -> float:
        """
        Convert thrust (N) to PWM (µs)
        
        Args:
            thrust_n: Desired thrust in Newtons
            
        Returns:
            PWM pulse width in microseconds
        """
        thrust_n = np.clip(thrust_n, 0.0, self.max_thrust_n)
        throttle = thrust_n / self.max_thrust_n
        return self.throttle_to_pwm(throttle)
    
    def pwm_to_thrust(self, pwm_us: float) -> float:
        """
        Convert PWM (µs) to estimated thrust (N)
        
        Args:
            pwm_us: PWM pulse width in microseconds
            
        Returns:
            Estimated thrust in Newtons
        """
        throttle = self.pwm_to_throttle(pwm_us)
        return throttle * self.max_thrust_n
    
    def rpm_to_thrust(self, rpm: float, kv: float = 2800.0) -> float:
        """
        Estimate thrust from RPM (requires motor characterization)
        
        Args:
            rpm: Motor RPM
            kv: Motor KV rating (RPM per Volt) - typical quadcopter: 2000-3500 KV
            
        Returns:
            Estimated thrust (requires empirical calibration)
        """
        # Placeholder: thrust ∝ RPM²
        # Actual relationship: thrust = C_t * rpm² (aerodynamic)
        # Typical for 10" propeller: C_t ≈ 5e-6
        C_t = 5e-6
        thrust = C_t * (rpm ** 2)
        return min(thrust, self.max_thrust_n)
    
    def thrust_to_rpm(self, thrust_n: float, kv: float = 2800.0) -> float:
        """
        Estimate required RPM for desired thrust
        
        Args:
            thrust_n: Desired thrust (N)
            kv: Motor KV rating
            
        Returns:
            Required RPM (inverse of rpm_to_thrust)
        """
        C_t = 5e-6
        rpm = np.sqrt(max(thrust_n, 0.0) / C_t)
        return rpm


class MotorESCInterface:
    """
    Abstract interface for motor ESC hardware
    
    Defines the contract for ESC backends (PWM GPIO, serial, CAN, etc.)
    """
    
    def __init__(self, motor_id: int, esc_type: ESCType = ESCType.GENERIC_PWM):
        self.motor_id = motor_id
        self.esc_type = esc_type
        self.is_connected = False
    
    def connect(self) -> bool:
        """Connect to ESC hardware"""
        raise NotImplementedError
    
    def disconnect(self) -> bool:
        """Disconnect from ESC hardware"""
        raise NotImplementedError
    
    def set_pwm(self, pwm_us: float) -> bool:
        """
        Set motor PWM signal
        
        Args:
            pwm_us: PWM pulse width in microseconds
            
        Returns:
            True if successful
        """
        raise NotImplementedError
    
    def read_telemetry(self) -> Optional[Dict[str, float]]:
        """
        Read ESC telemetry (RPM, current, voltage, temp)
        
        Returns:
            Dict with available telemetry or None if not available
        """
        raise NotImplementedError
    
    def get_status(self) -> Dict[str, any]:
        """Get ESC status"""
        return {
            'motor_id': self.motor_id,
            'esc_type': self.esc_type.value,
            'is_connected': self.is_connected,
        }


class SimulatedESC(MotorESCInterface):
    """
    Simulated ESC for testing and development
    
    Models motor behavior:
    - PWM input → motor dynamics (1st order lag)
    - Current draw proportional to load
    - Temperature rise with duty cycle
    """
    
    def __init__(
        self,
        motor_id: int,
        max_thrust_n: float = 25.0,
        response_time_ms: float = 20.0,
        nominal_current_a: float = 20.0,
    ):
        super().__init__(motor_id, ESCType.GENERIC_PWM)
        self.pwm_converter = PWMConverter(motor_id, max_thrust_n)
        self.response_time = response_time_ms / 1000.0
        self.nominal_current_a = nominal_current_a
        
        # Simulated state
        self.pwm_command = 1500.0                    # Current PWM
        self.rpm = 0.0                               # Simulated RPM
        self.current_a = 0.0                         # Simulated current
        self.temperature_c = 25.0                    # Simulated temperature
        self.last_update = datetime.now()
        self._lock = threading.RLock()
        
        self.is_connected = True  # Simulated always connected
    
    def set_pwm(self, pwm_us: float) -> bool:
        """Set PWM command"""
        with self._lock:
            pwm_us = np.clip(pwm_us, 1000.0, 2000.0)
            self.pwm_command = pwm_us
            self._update_motor_state()
            return True
    
    def _update_motor_state(self):
        """Update simulated motor state based on current PWM"""
        # Update RPM with 1st order lag
        throttle = self.pwm_converter.pwm_to_throttle(self.pwm_command)
        max_rpm = 8000.0
        target_rpm = throttle * max_rpm
        
        dt = (datetime.now() - self.last_update).total_seconds()
        tau = self.response_time
        self.rpm = self.rpm * np.exp(-dt / tau) + target_rpm * (1 - np.exp(-dt / tau))
        
        # Update current (proportional to RPM, with load component)
        base_current = 2.0  # Idle current (A)
        load_current = (self.rpm / 8000.0) * 18.0  # Load up to 20A at max RPM
        self.current_a = base_current + load_current
        
        # Update temperature
        power_w = self.current_a * 12.0  # Assume 12V battery
        heat_dissipation = 5.0 + (self.rpm / 8000.0) * 20.0  # Cooling increases with RPM
        dt_deg_per_sec = (power_w - heat_dissipation) / 100.0
        self.temperature_c += dt_deg_per_sec * dt
        self.temperature_c = max(25.0, self.temperature_c - 0.01 * dt)  # Passive cooling
        
        self.last_update = datetime.now()
    
    def read_telemetry(self) -> Dict[str, float]:
        """Read simulated motor telemetry"""
        with self._lock:
            self._update_motor_state()
            return {
                'pwm_us': self.pwm_command,
                'rpm': self.rpm,
                'current_a': self.current_a,
                'voltage_v': 12.0,
                'temperature_c': self.temperature_c,
            }
    
    def get_status(self) -> Dict[str, any]:
        """Get status"""
        with self._lock:
            self._update_motor_state()
            status = super().get_status()
            status.update({
                'pwm_us': self.pwm_command,
                'rpm': self.rpm,
                'current_a': self.current_a,
                'temperature_c': self.temperature_c,
            })
            return status


class MotorCommandQueue:
    """
    Thread-safe queue for motor PWM commands
    
    Decouples command generation (Phase 2E at 50 Hz) from PWM output
    (which may run at different frequency, e.g., 100-400 Hz).
    """
    
    def __init__(self, max_queue_size: int = 1000):
        self.queue = []
        self.max_size = max_queue_size
        self._lock = threading.RLock()
        self.total_commands = 0
        self.dropped_commands = 0
    
    def enqueue(self, pwm_commands: List[float], timestamp: datetime = None):
        """
        Add PWM command to queue
        
        Args:
            pwm_commands: List of PWM values for all motors
            timestamp: Command timestamp
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        with self._lock:
            if len(self.queue) >= self.max_size:
                self.queue.pop(0)  # Drop oldest
                self.dropped_commands += 1
            
            self.queue.append({
                'timestamp': timestamp,
                'commands': list(pwm_commands),
            })
            self.total_commands += 1
    
    def dequeue(self) -> Optional[Dict]:
        """
        Get next PWM command from queue
        
        Returns:
            Command dict with 'timestamp' and 'commands', or None if empty
        """
        with self._lock:
            if self.queue:
                return self.queue.pop(0)
            return None
    
    def peek(self) -> Optional[Dict]:
        """Peek at next command without removing"""
        with self._lock:
            if self.queue:
                return self.queue[0]
            return None
    
    def is_empty(self) -> bool:
        """Check if queue is empty"""
        with self._lock:
            return len(self.queue) == 0
    
    def get_size(self) -> int:
        """Get current queue size"""
        with self._lock:
            return len(self.queue)
    
    def get_statistics(self) -> Dict[str, int]:
        """Get queue statistics"""
        with self._lock:
            return {
                'current_size': len(self.queue),
                'max_size': self.max_size,
                'total_enqueued': self.total_commands,
                'total_dropped': self.dropped_commands,
                'drop_rate': (
                    self.dropped_commands / self.total_commands
                    if self.total_commands > 0 else 0.0
                ),
            }


class MotorControlInterface:
    """
    User-facing interface for motor control
    
    Provides clean API for Phase 2E to control motors:
    - set_motor_thrusts(thrusts, gains)
    - get_motor_status()
    - emergency_stop()
    """
    
    def __init__(self, num_motors: int = 4):
        self.num_motors = num_motors
        self.esc_interfaces = [
            SimulatedESC(i) for i in range(num_motors)
        ]
        self.pwm_converters = [
            PWMConverter(i) for i in range(num_motors)
        ]
        self.command_queue = MotorCommandQueue()
        self.last_pwm = [1500.0] * num_motors
        
    def set_motor_thrusts(
        self,
        motor_thrusts: List[float],
        adaptive_gains: Optional[List[float]] = None,
    ) -> List[float]:
        """
        Set motor thrusts and generate PWM
        
        Args:
            motor_thrusts: [T₀, T₁, T₂, T₃] in Newtons
            adaptive_gains: Optional gains from Phase 2G
            
        Returns:
            PWM commands [pwm₀, pwm₁, pwm₂, pwm₃]
        """
        if len(motor_thrusts) != self.num_motors:
            raise ValueError(f"Expected {self.num_motors} thrusts")
        
        adaptive_gains = adaptive_gains or [1.0] * self.num_motors
        
        pwm_commands = []
        for i, (thrust, gain) in enumerate(zip(motor_thrusts, adaptive_gains)):
            thrust_adjusted = thrust * gain
            pwm_us = self.pwm_converters[i].thrust_to_pwm(thrust_adjusted)
            pwm_commands.append(pwm_us)
            self.esc_interfaces[i].set_pwm(pwm_us)
        
        self.last_pwm = list(pwm_commands)
        self.command_queue.enqueue(pwm_commands)
        
        return pwm_commands
    
    def get_motor_telemetry(self) -> List[Dict[str, float]]:
        """Read telemetry from all motors"""
        telemetry = []
        for esc in self.esc_interfaces:
            telemetry.append(esc.read_telemetry())
        return telemetry
    
    def get_motor_status(self) -> List[Dict[str, any]]:
        """Get status of all motors"""
        return [esc.get_status() for esc in self.esc_interfaces]
    
    def emergency_stop(self):
        """Immediately stop all motors"""
        for esc in self.esc_interfaces:
            esc.set_pwm(1000.0)  # Neutral/minimum throttle
        self.last_pwm = [1000.0] * self.num_motors
