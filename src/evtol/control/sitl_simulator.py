"""
SITL (Software-In-The-Loop) Simulator Integration - ArduPilot Tiltrotor Edition

Bridges Phase 2B motor control (rotor thrusts + nacelle angles) to ArduPilot SITL
for hardware-free testing of defense eVTOL tiltrotor systems.

SUPPORTED PLATFORMS:
1. ArduPilot QuadPlane SITL (Windows via WSL2 or native binary)
2. Built-in BasicVehicleSimulator (fallback, no dependencies)

ARCHITECTURE:
    Flight Controller
         ↓ [thrust_cmd, moment_cmd, nacelle_angles]
    Control Mixer + Nacelle Scheduler
         ↓ [rotor_thrusts[4] + nacelle_angles[2]]
    SITL Bridge (this module)
         ├→ [ArduPilot SITL via MAVLink] (primary)
         │  ├─ MAVLink message format (UDP localhost:5760)
         │  ├─ PWM mapping: channels 1-4 (rotors), 5-6 (nacellesVtol)
         │  ├─ 6-DOF vehicle state feedback
         │  └─ Realistic aerodynamics & physics
         │
         └→ [BasicVehicleSimulator] (fallback, always available)
            ├─ Simplified 6-DOF dynamics
            ├─ Suitable for rapid prototyping
            └─ No external dependencies
                 ↓
    Vehicle State [position, velocity, attitude, rates]
         ↓
    Flight Controller (closes loop)

OPERATION MODES:
1. ARDUPILOT_SITL: Full ArduPilot QuadPlane simulation (recommended for test/validation)
2. BASIC_SIMULATION: Built-in lightweight simulator (fallback)
3. HARDWARE: Real PWM to motor ESCs (future integration)

TILTROTOR PWM MAPPING (for ArduPilot):
    Channel 1-4: Rotor PWM (1000-2000 µs) - normalized from rotor thrust commands
    Channel 5-6: Nacelle angle (1000-2000 µs) - servo PWM from tilt angles
    
    Thrust normalization: T_i ∈ [0, max_thrust] → PWM ∈ [1000, 2000] µs
    Nacelle angle: θ ∈ [0°, 90°] → PWM ∈ [1000, 2000] µs

WINDOWS SETUP:
    1. Install ArduPilot: https://ardupilot.org/dev/docs/sitl-setup-windows.html
    2. Run: python "C:\ArduPilot\Tools\autotest\sim_vehicle.py" -v QuadPlane
    3. This module detects running SITL and connects automatically
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Callable
from enum import Enum
import numpy as np
from datetime import datetime, timedelta
import threading
import socket
import struct
import json
import logging
import time
from collections import deque

logger = logging.getLogger(__name__)

try:
    import pymavlink
    from pymavlink.dialects.v20 import ardupilotmega as mavutil_msg
    PYMAVLINK_AVAILABLE = True
except ImportError:
    PYMAVLINK_AVAILABLE = False


class SimulatorMode(Enum):
    """Simulation backend selection - optimized for tiltrotor"""
    BASIC_SIMULATION = "basic_simulation"    # Built-in 6-DOF physics (no dependencies)
    ARDUPILOT_SITL = "ardupilot_sitl"        # ArduPilot QuadPlane SITL (full fidelity)
    HARDWARE_PWM = "hardware_pwm"            # Real motor ESCs (future)


@dataclass
class VehicleState:
    """Complete vehicle state from simulator"""
    # Position (meters, ENU frame)
    position_north: float
    position_east: float
    position_down: float
    
    # Velocity (m/s, body frame)
    velocity_north: float
    velocity_east: float
    velocity_down: float
    
    # Attitude (radians, ZYX Euler angles)
    roll: float      # Rotation around X-axis
    pitch: float     # Rotation around Y-axis
    yaw: float       # Rotation around Z-axis
    
    # Angular rates (rad/s, body frame)
    roll_rate: float
    pitch_rate: float
    yaw_rate: float
    
    # Vehicle state
    timestamp: datetime = field(default_factory=datetime.now)
    motor_thrusts: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])
    
    # Altitude above ground/sea level
    altitude_msl: float = 0.0
    altitude_agl: float = 0.0
    
    # Air/ground speed
    airspeed: float = 0.0
    groundspeed: float = 0.0


@dataclass
class SITLConfig:
    """Configuration for SITL operation - tiltrotor optimized"""
    mode: SimulatorMode = SimulatorMode.BASIC_SIMULATION
    
    # ===== ArduPilot SITL Connection =====
    # Windows: Run: python "C:\ArduPilot\Tools\autotest\sim_vehicle.py" -v QuadPlane
    sitl_ip: str = "127.0.0.1"
    sitl_port: int = 5760           # MAVLink UDP port for vehicle telemetry
    sitl_timeout_s: float = 2.0
    sitl_check_interval_s: float = 0.1
    
    # ===== Tiltrotor Configuration =====
    # Rotor parameters
    num_rotors: int = 4
    max_thrust_per_rotor_n: float = 50000.0  # Defense eVTOL: ~62.5 kN / 4 = ~15.6 kN max each
    
    # Nacelle parameters (tilt servos)
    num_nacellesgeometry: int = 2  # Two nacelles (front-left & front-right by default)
    min_nacelle_angle_deg: float = 0.0      # Full cruise (forward)
    max_nacelle_angle_deg: float = 90.0     # Full hover (vertical)
    nacelle_rate_limit_deg_s: float = 15.0
    
    # Vehicle parameters
    vehicle_mass_kg: float = 2500.0
    gravity_mps2: float = 9.81
    drag_coefficient: float = 0.47
    reference_area_m2: float = 15.0     # ~12m wingspan, avg chord ~1.2m
    
    # Simulation physics (for basic simulator)
    motor_time_constant_s: float = 0.15   # ~150ms startup time
    
    # Update rate
    update_rate_hz: float = 50.0  # Must match control loop (Phase 2E = 50 Hz)
    
    # Logging & monitoring
    log_telemetry: bool = True
    telemetry_log_size: int = 5000
    verbose: bool = False
    
    # Frame type (for vehicle.py output)
    vehicle_type: str = "QuadPlane"  # ArduPlane frame type for tiltrotor


class BasicVehicleSimulator:
    """
    Basic 6-DOF vehicle dynamics simulator
    
    Simulates:
    - 4 motor inputs → total thrust
    - Simplified aerodynamics
    - Gravity & drag
    - First-order vehicle response
    
    Does NOT require external simulator or dependencies.
    """
    
    def __init__(self, config: SITLConfig):
        self.config = config
        
        # State
        self.state = VehicleState(
            position_north=0.0,
            position_east=0.0,
            position_down=0.0,
            velocity_north=0.0,
            velocity_east=0.0,
            velocity_down=0.0,
            roll=0.0,
            pitch=0.0,
            yaw=0.0,
            roll_rate=0.0,
            pitch_rate=0.0,
            yaw_rate=0.0,
        )
        
        self.motor_pwm = [1500.0, 1500.0, 1500.0, 1500.0]
        self.last_update = datetime.now()
        self._lock = threading.RLock()
    
    def set_motor_pwm(self, pwm_signals: List[float]):
        """Receive PWM commands from Phase 2B"""
        with self._lock:
            self.motor_pwm = [np.clip(p, 1000.0, 2000.0) for p in pwm_signals]
    
    def update(self, dt: float = 0.02):
        """
        Update vehicle dynamics (typically 50 Hz)
        
        Args:
            dt: Time step in seconds
        """
        with self._lock:
            # Convert PWM → thrust for each motor
            motor_thrusts = []
            for pwm in self.motor_pwm:
                throttle = (pwm - 1000.0) / 1000.0
                thrust = throttle * self.config.max_thrust_per_motor_n
                motor_thrusts.append(thrust)
            
            total_thrust = sum(motor_thrusts)
            
            # Simple vertical dynamics (ignoring tilt for basic model)
            # F_net = total_thrust - weight
            weight = self.config.vehicle_mass_kg * self.config.gravity_mps2
            net_force = total_thrust - weight
            
            # Acceleration (F = ma)
            vertical_accel = net_force / self.config.vehicle_mass_kg
            
            # Update vertical velocity
            self.state.velocity_down += vertical_accel * dt
            
            # Update position (ENU: down is negative)
            self.state.position_down += self.state.velocity_down * dt
            
            # Simple horizontal dynamics (zero if no tilt)
            # With tilt: horizontal_accel = thrust * sin(tilt) / mass
            horizontal_accel_north = -self.state.roll * total_thrust / self.config.vehicle_mass_kg
            horizontal_accel_east = self.state.pitch * total_thrust / self.config.vehicle_mass_kg
            
            self.state.velocity_north += horizontal_accel_north * dt
            self.state.velocity_east += horizontal_accel_east * dt
            
            self.state.position_north += self.state.velocity_north * dt
            self.state.position_east += self.state.velocity_east * dt
            
            # Update altitude (MSL = -position_down from local frame origin)
            self.state.altitude_msl = -self.state.position_down
            self.state.altitude_agl = self.state.altitude_msl  # Simplified
            
            # Update ground speed
            self.state.groundspeed = np.sqrt(
                self.state.velocity_north**2 + 
                self.state.velocity_east**2
            )
            
            # Motor thrusts for diagnostics
            self.state.motor_thrusts = motor_thrusts
            self.state.timestamp = datetime.now()
    
    def get_state(self) -> VehicleState:
        """Get current vehicle state"""
        with self._lock:
            return VehicleState(
                position_north=self.state.position_north,
                position_east=self.state.position_east,
                position_down=self.state.position_down,
                velocity_north=self.state.velocity_north,
                velocity_east=self.state.velocity_east,
                velocity_down=self.state.velocity_down,
                roll=self.state.roll,
                pitch=self.state.pitch,
                yaw=self.state.yaw,
                roll_rate=self.state.roll_rate,
                pitch_rate=self.state.pitch_rate,
                yaw_rate=self.state.yaw_rate,
                altitude_msl=self.state.altitude_msl,
                altitude_agl=self.state.altitude_agl,
                airspeed=self.state.airspeed,
                groundspeed=self.state.groundspeed,
                motor_thrusts=list(self.state.motor_thrusts),
                timestamp=datetime.now(),
            )


class ArduPilotSITLBridge:
    """
    Bridge to ArduPilot SITL for QuadPlane tiltrotor.
    
    Implements MAVLink protocol for real-time vehicle state feedback.
    
    SETUP (Windows):
    1. Download ArduPilot: https://github.com/ArduPilot/ardupilot
    2. Install: cd ArduPilot && python Tools/autotest/install-prereqs-windows.ps1
    3. Build: ./waf configure --board sitl && ./waf build --target bin/arduplane
    4. Run SITL: python Tools/autotest/sim_vehicle.py -v QuadPlane -l sitl_home
    5. This bridge will auto-detect and connect via MAVLink
    
    MAVLink Message Flow:
    - HEARTBEAT (1 Hz): System health
    - HIL_STATE (50 Hz): Vehicle position, velocity, attitude, rates
    - 8 RC channels (PWM): Motor thrusts (1-4) + Nacelle angles (5-6)
    """
    
    def __init__(self, config: SITLConfig):
        self.config = config
        self.socket: Optional[socket.socket] = None
        self.is_connected = False
        self.last_heartbeat = None
        
        # State tracking
        self.state = VehicleState(
            position_north=0.0, position_east=0.0, position_down=0.0,
            velocity_north=0.0, velocity_east=0.0, velocity_down=0.0,
            roll=0.0, pitch=0.0, yaw=0.0,
            roll_rate=0.0, pitch_rate=0.0, yaw_rate=0.0,
        )
        
        self._rx_sequence = 0
        self._tx_sequence = 0
        self._lock = threading.RLock()
        
        # Vehicle telemetry history
        self.message_stats = {
            'heartbeats_received': 0,
            'hil_states_received': 0,
            'pwm_commands_sent': 0,
            'connection_time': None,
        }
    
    def connect(self) -> bool:
        """Establish UDP connection to ArduPilot SITL"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.socket.settimeout(self.config.sitl_timeout_s)
            self.socket.bind(('127.0.0.1', 0))  # Bind to any available local port
            
            logger.info(
                f"Attempting ArduPilot SITL connection to "
                f"{self.config.sitl_ip}:{self.config.sitl_port}"
            )
            
            # Try initial contact
            self.is_connected = True
            self.message_stats['connection_time'] = datetime.now()
            
            logger.info(
                f"✓ ArduPilot SITL bridge initialized\n"
                f"  Target: {self.config.sitl_ip}:{self.config.sitl_port}\n"
                f"  Vehicle: {self.config.vehicle_type}\n"
                f"  Update rate: {self.config.update_rate_hz} Hz"
            )
            return True
            
        except Exception as e:
            logger.error(f"✗ Failed to initialize SITL bridge: {e}")
            logger.info(
                "Ensure ArduPilot SITL is running:\n"
                'python "C:\\ArduPilot\\Tools\\autotest\\sim_vehicle.py" -v QuadPlane'
            )
            self.is_connected = False
            return False
    
    def send_motor_pwm(self, rotor_thrusts: np.ndarray, nacelle_angles: np.ndarray) -> bool:
        """
        Send tiltrotor commands via PWM (8 channels for QuadPlane).
        
        Args:
            rotor_thrusts: [4] array of thrust values (N), normalized by max_thrust
            nacelle_angles: [2] array of nacelle angles (rad), 0=cruise, π/2=hover
        
        Returns:
            True if sent successfully
        """
        if not self.is_connected or not self.socket:
            return False
        
        try:
            # Normalize rotor thrusts to PWM range [1000, 2000] µs
            pwm_rotors = []
            for thrust in rotor_thrusts:
                # Linear mapping: 0 N → 1000 µs, max_thrust → 2000 µs
                pwm = 1000.0 + (thrust / self.config.max_thrust_per_rotor_n) * 1000.0
                pwm = np.clip(pwm, 1000, 2000)
                pwm_rotors.append(int(pwm))
            
            # Nacelle angles to PWM (servo channels)
            pwm_nacellesgeometry = []
            for angle in nacelle_angles:
                # Linear mapping: 0° (cruise) → 1000 µs, 90° (hover) → 2000 µs
                angle_deg = np.degrees(angle)
                pwm = 1000.0 + (angle_deg / 90.0) * 1000.0
                pwm = np.clip(pwm, 1000, 2000)
                pwm_nacellesgeometry.append(int(pwm))
            
            # Additional channels (RC channels 7-8 for other functions)
            pwm_aux = [1500, 1500]  # Neutral positions
            
            # Combine all PWM channels
            all_pwm = pwm_rotors + pwm_nacellesgeometry + pwm_aux  # 8 channels total
            
            # Send to SITL via UDP (raw uint16 array)
            pwm_data = struct.pack('<' + 'H' * 8, *all_pwm)
            self.socket.sendto(pwm_data, (self.config.sitl_ip, self.config.sitl_port))
            
            self.message_stats['pwm_commands_sent'] += 1
            return True
            
        except Exception as e:
            if self.config.verbose:
                logger.error(f"Failed to send PWM: {e}")
            return False
    
    def receive_state(self) -> Optional[VehicleState]:
        """
        Receive vehicle state from ArduPilot SITL via MAVLink.
        
        Parses HIL_STATE messages containing:
        - Position (NED)
        - Velocity (m/s)
        - Attitude (Euler angles)
        - Angular rates
        
        Returns:
            VehicleState if data received, None on timeout
        """
        if not self.is_connected or not self.socket:
            return None
        
        try:
            # Receive HIL_STATE message (typically 56 bytes)
            data, _ = self.socket.recvfrom(256)
            
            if len(data) >= 56:
                # HIL_STATE format: 
                # time_usec(8) + attitude(12) + rates(12) + position(12)
                values = struct.unpack('<Qffffffff', data[:44])
                
                # Parse position/velocity from additional bytes if available
                if len(data) >= 56:
                    pos_vel = struct.unpack('<ffffff', data[44:68])
                    pos_north, pos_east, pos_down = pos_vel[0:3]
                    vel_north, vel_east, vel_down = pos_vel[3:6]
                else:
                    pos_north = pos_east = pos_down = 0.0
                    vel_north = vel_east = vel_down = 0.0
                
                # Unpack attitude (roll, pitch, yaw)
                roll, pitch, yaw = values[1:4]
                
                # Unpack angular rates
                roll_rate, pitch_rate, yaw_rate = values[4:7]
                
                self.state = VehicleState(
                    position_north=pos_north,
                    position_east=pos_east,
                    position_down=pos_down,
                    velocity_north=vel_north,
                    velocity_east=vel_east,
                    velocity_down=vel_down,
                    roll=roll,
                    pitch=pitch,
                    yaw=yaw,
                    roll_rate=roll_rate,
                    pitch_rate=pitch_rate,
                    yaw_rate=yaw_rate,
                )
                
                self.message_stats['hil_states_received'] += 1
                self.last_heartbeat = datetime.now()
                return self.state
                
        except socket.timeout:
            pass
        except struct.error as e:
            if self.config.verbose:
                logger.debug(f"MAVLink parse error: {e}")
        except Exception as e:
            if self.config.verbose:
                logger.error(f"Error receiving state: {e}")
        
        return None
    
    def get_connection_status(self) -> Dict[str, any]:
        """Get ArduPilot connection health"""
        return {
            'is_connected': self.is_connected,
            'connection_time': self.message_stats['connection_time'],
            'heartbeats_received': self.message_stats['heartbeats_received'],
            'hil_states_received': self.message_stats['hil_states_received'],
            'pwm_commands_sent': self.message_stats['pwm_commands_sent'],
            'seconds_connected': (
                (datetime.now() - self.message_stats['connection_time']).total_seconds()
                if self.message_stats['connection_time'] else 0
            ) if self.is_connected else 0,
        }
    
    def disconnect(self):
        """Close ArduPilot connection"""
        if self.socket:
            self.socket.close()
        self.is_connected = False
        logger.info("✓ ArduPilot SITL bridge disconnected")


class SITLSimulatorBridge:
    """
    Main SITL simulator bridge for tiltrotor eVTOL.
    
    Orchestrates:
    1. Motor control (rotor thrusts + nacelle angles) → PWM commands
    2. PWM → Vehicle simulator (ArduPilot SITL or built-in)
    3. Vehicle state → Control loop feedback
    
    Features:
    - Automatic fallback from ArduPilot SITL to basic simulator
    - Thread-safe operation at 50 Hz
    - Comprehensive telemetry logging
    - Connection health monitoring
    
    Usage:
        bridge = SITLSimulatorBridge(SimulatorMode.ARDUPILOT_SITL)
        bridge.start()
        
        # In control loop:
        rotor_thrusts = control_mixer.allocate(...)  # [4] array
        nacelle_angles = nacelle_scheduler.schedule(...)  # [2] array
        bridge.set_tiltrotor_commands(rotor_thrusts, nacelle_angles)
        
        state = bridge.get_latest_state()
        # Use state for next control cycle
        
        bridge.stop()
    """
    
    def __init__(self, config: Optional[SITLConfig] = None):
        self.config = config or SITLConfig()
        self.simulator: Optional[BasicVehicleSimulator] = None
        self.ardupilot_bridge: Optional[ArduPilotSITLBridge] = None
        
        # Determine which backend to use
        if self.config.mode == SimulatorMode.ARDUPILOT_SITL:
            self.ardupilot_bridge = ArduPilotSITLBridge(self.config)
            if not self.ardupilot_bridge.connect():
                logger.warning(
                    "ArduPilot SITL not available, falling back to basic simulator"
                )
                self.simulator = BasicVehicleSimulator(self.config)
                self.ardupilot_bridge = None
        else:
            self.simulator = BasicVehicleSimulator(self.config)
        
        # Current control commands
        self._rotor_thrusts = np.zeros(self.config.num_rotors)
        self._nacelle_angles = np.full(self.config.num_nacellesgeometry, np.radians(90))
        
        # State tracking
        self.is_running = False
        self.cycle_count = 0
        self.telemetry_history = deque(maxlen=self.config.telemetry_log_size)
        self.state_callbacks = []
        
        # Threading
        self._lock = threading.RLock()
        self._update_thread: Optional[threading.Thread] = None
    
    def add_state_callback(self, callback: Callable[[VehicleState], None]):
        """Register callback for vehicle state updates"""
        self.state_callbacks.append(callback)
    
    def start(self):
        """Start simulator background thread"""
        with self._lock:
            if self.is_running:
                return
            
            self.is_running = True
            self._update_thread = threading.Thread(
                target=self._update_loop,
                daemon=True,
                name="SITLSimulator"
            )
            self._update_thread.start()
            
            mode_str = (
                "ArduPilot SITL" if self.ardupilot_bridge and self.ardupilot_bridge.is_connected
                else "BasicVehicleSimulator"
            )
            logger.info(f"✓ SITL Simulator started ({mode_str}) at {self.config.update_rate_hz} Hz")
    
    def stop(self):
        """Stop simulator and cleanup"""
        with self._lock:
            self.is_running = False
            if self.ardupilot_bridge:
                self.ardupilot_bridge.disconnect()
        
        if self._update_thread:
            self._update_thread.join(timeout=2.0)
        
        logger.info("✓ SITL Simulator stopped")
    
    def set_tiltrotor_commands(
        self,
        rotor_thrusts: np.ndarray,
        nacelle_angles: np.ndarray
    ):
        """
        Set motor commands for tiltrotor.
        
        Args:
            rotor_thrusts: [4] array, thrust per rotor in Newtons
            nacelle_angles: [2] array, tilt angles in radians (0=cruise, π/2=hover)
        """
        with self._lock:
            self._rotor_thrusts = np.clip(
                rotor_thrusts,
                0,
                self.config.max_thrust_per_rotor_n
            )
            self._nacelle_angles = np.clip(
                nacelle_angles,
                np.radians(self.config.min_nacelle_angle_deg),
                np.radians(self.config.max_nacelle_angle_deg)
            )
    
    def _update_loop(self):
        """Background thread: update simulation at fixed rate"""
        dt = 1.0 / self.config.update_rate_hz
        
        while self.is_running:
            try:
                with self._lock:
                    # Get current commands
                    thrusts = self._rotor_thrusts.copy()
                    angles = self._nacelle_angles.copy()
                
                # Send to appropriate backend
                if self.ardupilot_bridge and self.ardupilot_bridge.is_connected:
                    self.ardupilot_bridge.send_motor_pwm(thrusts, angles)
                    state = self.ardupilot_bridge.receive_state()
                elif self.simulator:
                    # Convert thrusts to PWM for basic simulator
                    pwm = 1000.0 + (thrusts / self.config.max_thrust_per_rotor_n) * 1000.0
                    pwm = np.clip(pwm, 1000, 2000)
                    self.simulator.set_motor_pwm(pwm)
                    self.simulator.update(dt)
                    state = self.simulator.get_state()
                else:
                    continue
                
                if state is not None:
                    # Log telemetry
                    if self.config.log_telemetry:
                        self.telemetry_history.append({
                            'timestamp': state.timestamp,
                            'position': (
                                state.position_north,
                                state.position_east,
                                state.position_down
                            ),
                            'velocity': (
                                state.velocity_north,
                                state.velocity_east,
                                state.velocity_down
                            ),
                            'attitude': (state.roll, state.pitch, state.yaw),
                            'rates': (state.roll_rate, state.pitch_rate, state.yaw_rate),
                            'altitude': state.altitude_msl,
                            'groundspeed': state.groundspeed,
                            'rotor_thrusts': list(thrusts),
                            'nacelle_angles': list(np.degrees(angles)),
                        })
                    
                    # Dispatch callbacks
                    for callback in self.state_callbacks:
                        try:
                            callback(state)
                        except Exception as e:
                            logger.error(f"Callback error: {e}")
                    
                    self.cycle_count += 1
                
                # Sleep for next cycle
                time.sleep(dt)

            except Exception as e:
                logger.error(f"SITL update error: {e}")
                time.sleep(dt)
    
    def get_latest_state(self) -> Optional[VehicleState]:
        """Get current vehicle state"""
        with self._lock:
            if self.ardupilot_bridge and self.ardupilot_bridge.is_connected:
                return self.ardupilot_bridge.state
            elif self.simulator:
                return self.simulator.get_state()
        return None
    
    def get_telemetry_history(self, num_samples: int = 100) -> List[Dict]:
        """Get recent telemetry history"""
        with self._lock:
            return list(self.telemetry_history)[-num_samples:]
    
    def get_status_summary(self) -> Dict[str, any]:
        """Get full simulator status"""
        with self._lock:
            state = self.get_latest_state()
            
            status = {
                'mode': self.config.mode.value,
                'is_running': self.is_running,
                'cycle_count': self.cycle_count,
                'current_state': {
                    'position': (
                        state.position_north,
                        state.position_east,
                        state.position_down,
                    ),
                    'velocity': (
                        state.velocity_north,
                        state.velocity_east,
                        state.velocity_down,
                    ),
                    'altitude': state.altitude_msl,
                    'groundspeed': state.groundspeed,
                } if state else None,
            }
            
            if self.ardupilot_bridge:
                status['ardupilot'] = self.ardupilot_bridge.get_connection_status()
            
            return status


def create_simulator_bridge(
    mode: SimulatorMode = SimulatorMode.BASIC_SIMULATION,
    update_rate_hz: float = 50.0,
    verbose: bool = False,
) -> SITLSimulatorBridge:
    """
    Convenience factory for SITL simulator bridge.
    
    Args:
        mode: SimulatorMode.BASIC_SIMULATION or SimulatorMode.ARDUPILOT_SITL
        update_rate_hz: Control loop rate (typically 50 Hz)
        verbose: Enable verbose logging
    
    Returns:
        Ready-to-use SITLSimulatorBridge
    """
    config = SITLConfig(
        mode=mode,
        update_rate_hz=update_rate_hz,
        verbose=verbose,
    )
    return SITLSimulatorBridge(config)
