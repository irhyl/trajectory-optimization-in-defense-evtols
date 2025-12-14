"""
SITL Connector Module - Real ArduPilot/PX4 SITL Integration

Provides MAVLink-based connection to Software-In-The-Loop simulators
for realistic flight simulation and telemetry.
"""

import threading
import time
from dataclasses import dataclass, field
from typing import Optional, Callable, List, Dict, Any
from collections import deque
import math

try:
    from pymavlink import mavutil
    PYMAVLINK_AVAILABLE = True
except ImportError:
    PYMAVLINK_AVAILABLE = False
    print("Warning: pymavlink not installed. Install with: pip install pymavlink")


@dataclass
class Telemetry:
    """Real-time telemetry data from the vehicle."""
    timestamp: float = 0.0
    
    # Position
    latitude: float = 0.0
    longitude: float = 0.0
    altitude_msl: float = 0.0  # Altitude above mean sea level (m)
    altitude_rel: float = 0.0  # Altitude relative to home (m)
    
    # Velocity
    vx: float = 0.0  # m/s North
    vy: float = 0.0  # m/s East
    vz: float = 0.0  # m/s Down
    groundspeed: float = 0.0  # m/s
    airspeed: float = 0.0  # m/s
    
    # Attitude (radians)
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    
    # Rates (rad/s)
    rollspeed: float = 0.0
    pitchspeed: float = 0.0
    yawspeed: float = 0.0
    
    # Battery
    battery_voltage: float = 0.0
    battery_current: float = 0.0
    battery_remaining: int = 100  # Percentage
    
    # System
    armed: bool = False
    mode: str = "UNKNOWN"
    gps_fix: int = 0
    satellites: int = 0
    hdop: float = 99.99
    
    # Heartbeat
    system_status: str = "UNKNOWN"
    is_connected: bool = False


@dataclass
class Waypoint:
    """Mission waypoint definition."""
    seq: int
    lat: float
    lon: float
    alt: float  # Relative altitude in meters
    command: int = 16  # MAV_CMD_NAV_WAYPOINT
    hold_time: float = 0.0  # Seconds to hold at waypoint
    accept_radius: float = 5.0  # Acceptance radius in meters


@dataclass
class SITLConfig:
    """SITL connection configuration."""
    connection_string: str = "tcp:127.0.0.1:5760"
    baud_rate: int = 115200
    source_system: int = 255
    source_component: int = 0
    autoreconnect: bool = True
    heartbeat_timeout: float = 3.0


class SITLConnector:
    """
    Real SITL Connector for ArduPilot/PX4 integration.
    
    Provides:
    - MAVLink connection management
    - Real-time telemetry streaming
    - Mission upload and management
    - Vehicle control commands
    - Arming/disarming
    - Mode changes
    """
    
    def __init__(self, config: Optional[SITLConfig] = None):
        self.config = config or SITLConfig()
        self.connection: Optional[Any] = None
        self.telemetry = Telemetry()
        self.telemetry_history: deque = deque(maxlen=1000)
        
        self._running = False
        self._telemetry_thread: Optional[threading.Thread] = None
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._last_heartbeat = 0.0
        
        self._telemetry_callbacks: List[Callable[[Telemetry], None]] = []
        self._connection_callbacks: List[Callable[[bool], None]] = []
        
        self.mission_items: List[Waypoint] = []
        self.current_mission_item = 0
        
    @property
    def is_available(self) -> bool:
        """Check if pymavlink is available."""
        return PYMAVLINK_AVAILABLE
    
    @property
    def is_connected(self) -> bool:
        """Check if connected to SITL."""
        if not self.connection:
            return False
        return (time.time() - self._last_heartbeat) < self.config.heartbeat_timeout
    
    def connect(self) -> bool:
        """
        Establish connection to SITL.
        
        Returns:
            True if connection successful, False otherwise.
        """
        if not PYMAVLINK_AVAILABLE:
            print("Error: pymavlink not installed")
            return False
            
        try:
            print(f"Connecting to SITL at {self.config.connection_string}...")
            
            self.connection = mavutil.mavlink_connection(
                self.config.connection_string,
                baud=self.config.baud_rate,
                source_system=self.config.source_system,
                source_component=self.config.source_component
            )
            
            # Wait for heartbeat
            print("Waiting for heartbeat...")
            msg = self.connection.wait_heartbeat(timeout=10)
            
            if msg:
                self._last_heartbeat = time.time()
                self.telemetry.is_connected = True
                print(f"Connected! System: {self.connection.target_system}, Component: {self.connection.target_component}")
                
                # Start telemetry thread
                self._start_telemetry_loop()
                
                # Request data streams
                self._request_data_streams()
                
                # Notify callbacks
                for cb in self._connection_callbacks:
                    cb(True)
                
                return True
            else:
                print("No heartbeat received")
                return False
                
        except Exception as e:
            print(f"Connection failed: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from SITL."""
        self._running = False
        
        if self._telemetry_thread and self._telemetry_thread.is_alive():
            self._telemetry_thread.join(timeout=2.0)
        
        if self.connection:
            self.connection.close()
            self.connection = None
        
        self.telemetry.is_connected = False
        
        for cb in self._connection_callbacks:
            cb(False)
        
        print("Disconnected from SITL")
    
    def _request_data_streams(self):
        """Request data streams from the vehicle."""
        if not self.connection:
            return
            
        # Request all data streams at 10Hz
        self.connection.mav.request_data_stream_send(
            self.connection.target_system,
            self.connection.target_component,
            mavutil.mavlink.MAV_DATA_STREAM_ALL,
            10,  # 10 Hz
            1    # Start
        )
    
    def _start_telemetry_loop(self):
        """Start the telemetry receiving thread."""
        self._running = True
        self._telemetry_thread = threading.Thread(target=self._telemetry_loop, daemon=True)
        self._telemetry_thread.start()
    
    def _telemetry_loop(self):
        """Main telemetry receiving loop."""
        while self._running and self.connection:
            try:
                msg = self.connection.recv_match(blocking=True, timeout=0.1)
                if msg:
                    self._process_message(msg)
            except Exception as e:
                print(f"Telemetry error: {e}")
                if not self.config.autoreconnect:
                    break
    
    def _process_message(self, msg):
        """Process incoming MAVLink message."""
        msg_type = msg.get_type()
        
        if msg_type == 'HEARTBEAT':
            self._last_heartbeat = time.time()
            self.telemetry.is_connected = True
            self.telemetry.armed = (msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED) != 0
            self.telemetry.mode = mavutil.mode_string_v10(msg)
            self.telemetry.system_status = self._get_system_status(msg.system_status)
            
        elif msg_type == 'GLOBAL_POSITION_INT':
            self.telemetry.latitude = msg.lat / 1e7
            self.telemetry.longitude = msg.lon / 1e7
            self.telemetry.altitude_msl = msg.alt / 1000.0
            self.telemetry.altitude_rel = msg.relative_alt / 1000.0
            self.telemetry.vx = msg.vx / 100.0
            self.telemetry.vy = msg.vy / 100.0
            self.telemetry.vz = msg.vz / 100.0
            
        elif msg_type == 'ATTITUDE':
            self.telemetry.roll = msg.roll
            self.telemetry.pitch = msg.pitch
            self.telemetry.yaw = msg.yaw
            self.telemetry.rollspeed = msg.rollspeed
            self.telemetry.pitchspeed = msg.pitchspeed
            self.telemetry.yawspeed = msg.yawspeed
            
        elif msg_type == 'VFR_HUD':
            self.telemetry.groundspeed = msg.groundspeed
            self.telemetry.airspeed = msg.airspeed
            
        elif msg_type == 'SYS_STATUS':
            self.telemetry.battery_voltage = msg.voltage_battery / 1000.0
            self.telemetry.battery_current = msg.current_battery / 100.0
            self.telemetry.battery_remaining = msg.battery_remaining
            
        elif msg_type == 'GPS_RAW_INT':
            self.telemetry.gps_fix = msg.fix_type
            self.telemetry.satellites = msg.satellites_visible
            self.telemetry.hdop = msg.eph / 100.0 if msg.eph != 65535 else 99.99
        
        elif msg_type == 'MISSION_CURRENT':
            self.current_mission_item = msg.seq
        
        # Update timestamp and store in history
        self.telemetry.timestamp = time.time()
        self.telemetry_history.append(self._copy_telemetry())
        
        # Notify callbacks
        for cb in self._telemetry_callbacks:
            try:
                cb(self.telemetry)
            except Exception as e:
                print(f"Callback error: {e}")
    
    def _copy_telemetry(self) -> Telemetry:
        """Create a copy of current telemetry."""
        return Telemetry(
            timestamp=self.telemetry.timestamp,
            latitude=self.telemetry.latitude,
            longitude=self.telemetry.longitude,
            altitude_msl=self.telemetry.altitude_msl,
            altitude_rel=self.telemetry.altitude_rel,
            vx=self.telemetry.vx,
            vy=self.telemetry.vy,
            vz=self.telemetry.vz,
            groundspeed=self.telemetry.groundspeed,
            airspeed=self.telemetry.airspeed,
            roll=self.telemetry.roll,
            pitch=self.telemetry.pitch,
            yaw=self.telemetry.yaw,
            rollspeed=self.telemetry.rollspeed,
            pitchspeed=self.telemetry.pitchspeed,
            yawspeed=self.telemetry.yawspeed,
            battery_voltage=self.telemetry.battery_voltage,
            battery_current=self.telemetry.battery_current,
            battery_remaining=self.telemetry.battery_remaining,
            armed=self.telemetry.armed,
            mode=self.telemetry.mode,
            gps_fix=self.telemetry.gps_fix,
            satellites=self.telemetry.satellites,
            hdop=self.telemetry.hdop,
            system_status=self.telemetry.system_status,
            is_connected=self.telemetry.is_connected
        )
    
    def _get_system_status(self, status: int) -> str:
        """Convert system status to string."""
        status_map = {
            0: "UNINIT",
            1: "BOOT",
            2: "CALIBRATING",
            3: "STANDBY",
            4: "ACTIVE",
            5: "CRITICAL",
            6: "EMERGENCY",
            7: "POWEROFF",
            8: "FLIGHT_TERMINATION"
        }
        return status_map.get(status, "UNKNOWN")
    
    # ==================== Vehicle Control ====================
    
    def arm(self, force: bool = False) -> bool:
        """
        Arm the vehicle.
        
        Args:
            force: Force arming even if pre-arm checks fail.
            
        Returns:
            True if command sent successfully.
        """
        if not self.connection:
            return False
            
        try:
            self.connection.mav.command_long_send(
                self.connection.target_system,
                self.connection.target_component,
                mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                0,  # Confirmation
                1,  # Arm
                21196 if force else 0,  # Force arm magic number
                0, 0, 0, 0, 0
            )
            
            # Wait for ACK
            ack = self.connection.recv_match(type='COMMAND_ACK', blocking=True, timeout=3)
            if ack and ack.result == mavutil.mavlink.MAV_RESULT_ACCEPTED:
                print("Vehicle armed")
                return True
            else:
                print(f"Arm failed: {ack.result if ack else 'No ACK'}")
                return False
                
        except Exception as e:
            print(f"Arm error: {e}")
            return False
    
    def disarm(self, force: bool = False) -> bool:
        """Disarm the vehicle."""
        if not self.connection:
            return False
            
        try:
            self.connection.mav.command_long_send(
                self.connection.target_system,
                self.connection.target_component,
                mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                0,
                0,  # Disarm
                21196 if force else 0,
                0, 0, 0, 0, 0
            )
            
            ack = self.connection.recv_match(type='COMMAND_ACK', blocking=True, timeout=3)
            if ack and ack.result == mavutil.mavlink.MAV_RESULT_ACCEPTED:
                print("Vehicle disarmed")
                return True
            return False
            
        except Exception as e:
            print(f"Disarm error: {e}")
            return False
    
    def set_mode(self, mode: str) -> bool:
        """
        Set flight mode.
        
        Args:
            mode: Mode name (e.g., "GUIDED", "AUTO", "LOITER", "RTL")
            
        Returns:
            True if mode change successful.
        """
        if not self.connection:
            return False
            
        try:
            mode_id = self.connection.mode_mapping().get(mode.upper())
            if mode_id is None:
                print(f"Unknown mode: {mode}")
                return False
            
            self.connection.mav.set_mode_send(
                self.connection.target_system,
                mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                mode_id
            )
            
            # Wait for mode change
            time.sleep(0.5)
            return self.telemetry.mode.upper() == mode.upper()
            
        except Exception as e:
            print(f"Mode change error: {e}")
            return False
    
    def takeoff(self, altitude: float = 10.0) -> bool:
        """
        Command takeoff to specified altitude.
        
        Args:
            altitude: Target altitude in meters (relative).
            
        Returns:
            True if command sent successfully.
        """
        if not self.connection:
            return False
            
        try:
            self.connection.mav.command_long_send(
                self.connection.target_system,
                self.connection.target_component,
                mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
                0,
                0, 0, 0, 0, 0, 0,
                altitude
            )
            
            ack = self.connection.recv_match(type='COMMAND_ACK', blocking=True, timeout=3)
            if ack and ack.result == mavutil.mavlink.MAV_RESULT_ACCEPTED:
                print(f"Takeoff to {altitude}m commanded")
                return True
            return False
            
        except Exception as e:
            print(f"Takeoff error: {e}")
            return False
    
    def land(self) -> bool:
        """Command landing at current position."""
        if not self.connection:
            return False
            
        try:
            self.connection.mav.command_long_send(
                self.connection.target_system,
                self.connection.target_component,
                mavutil.mavlink.MAV_CMD_NAV_LAND,
                0,
                0, 0, 0, 0, 0, 0, 0
            )
            
            ack = self.connection.recv_match(type='COMMAND_ACK', blocking=True, timeout=3)
            return ack and ack.result == mavutil.mavlink.MAV_RESULT_ACCEPTED
            
        except Exception as e:
            print(f"Land error: {e}")
            return False
    
    def return_to_launch(self) -> bool:
        """Command return to launch."""
        return self.set_mode("RTL")
    
    def goto(self, lat: float, lon: float, alt: float) -> bool:
        """
        Go to specified position.
        
        Args:
            lat: Latitude in degrees
            lon: Longitude in degrees
            alt: Altitude in meters (relative)
            
        Returns:
            True if command sent successfully.
        """
        if not self.connection:
            return False
            
        try:
            self.connection.mav.mission_item_int_send(
                self.connection.target_system,
                self.connection.target_component,
                0,  # Sequence
                mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
                mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
                2,  # Current = 2 means guided mode target
                0,  # Autocontinue
                0, 0, 0, 0,  # Params
                int(lat * 1e7),
                int(lon * 1e7),
                alt
            )
            print(f"Goto: {lat:.6f}, {lon:.6f}, {alt}m")
            return True
            
        except Exception as e:
            print(f"Goto error: {e}")
            return False
    
    # ==================== Mission Management ====================
    
    def upload_mission(self, waypoints: List[Waypoint]) -> bool:
        """
        Upload mission to vehicle.
        
        Args:
            waypoints: List of waypoints to upload.
            
        Returns:
            True if mission upload successful.
        """
        if not self.connection:
            return False
            
        try:
            # Clear existing mission
            self.connection.mav.mission_clear_all_send(
                self.connection.target_system,
                self.connection.target_component
            )
            time.sleep(0.5)
            
            # Send mission count
            self.connection.mav.mission_count_send(
                self.connection.target_system,
                self.connection.target_component,
                len(waypoints)
            )
            
            # Send each waypoint
            for i, wp in enumerate(waypoints):
                # Wait for request
                msg = self.connection.recv_match(type='MISSION_REQUEST', blocking=True, timeout=5)
                if not msg or msg.seq != i:
                    print(f"Mission upload failed at waypoint {i}")
                    return False
                
                # Send waypoint
                self.connection.mav.mission_item_int_send(
                    self.connection.target_system,
                    self.connection.target_component,
                    i,
                    mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
                    wp.command,
                    1 if i == 0 else 0,  # Current
                    1,  # Autocontinue
                    wp.hold_time,
                    wp.accept_radius,
                    0, 0,  # Params
                    int(wp.lat * 1e7),
                    int(wp.lon * 1e7),
                    wp.alt
                )
            
            # Wait for ACK
            ack = self.connection.recv_match(type='MISSION_ACK', blocking=True, timeout=5)
            if ack and ack.type == mavutil.mavlink.MAV_MISSION_ACCEPTED:
                print(f"Mission uploaded: {len(waypoints)} waypoints")
                self.mission_items = waypoints
                return True
            
            return False
            
        except Exception as e:
            print(f"Mission upload error: {e}")
            return False
    
    def start_mission(self) -> bool:
        """Start the uploaded mission."""
        if not self.connection:
            return False
            
        try:
            # Set mode to AUTO
            if not self.set_mode("AUTO"):
                print("Failed to set AUTO mode")
                return False
            
            # Arm if needed
            if not self.telemetry.armed:
                if not self.arm():
                    print("Failed to arm")
                    return False
            
            # Start mission
            self.connection.mav.command_long_send(
                self.connection.target_system,
                self.connection.target_component,
                mavutil.mavlink.MAV_CMD_MISSION_START,
                0,
                0, 0, 0, 0, 0, 0, 0
            )
            
            print("Mission started")
            return True
            
        except Exception as e:
            print(f"Mission start error: {e}")
            return False
    
    # ==================== Callbacks ====================
    
    def add_telemetry_callback(self, callback: Callable[[Telemetry], None]):
        """Register a callback for telemetry updates."""
        self._telemetry_callbacks.append(callback)
    
    def add_connection_callback(self, callback: Callable[[bool], None]):
        """Register a callback for connection state changes."""
        self._connection_callbacks.append(callback)
    
    def get_telemetry_history(self, n: int = 100) -> List[Telemetry]:
        """Get recent telemetry history."""
        return list(self.telemetry_history)[-n:]


# ==================== Demo Mode (No SITL Required) ====================

class DemoSITLConnector(SITLConnector):
    """
    Demo SITL Connector that simulates a vehicle without actual SITL.
    Useful for UI testing and demonstration.
    """
    
    def __init__(self, config: Optional[SITLConfig] = None):
        super().__init__(config)
        self._demo_running = False
        self._demo_thread: Optional[threading.Thread] = None
        self._demo_start_time = 0.0
        self._home_lat = 35.363261
        self._home_lon = -117.060185
        self._target_alt = 0.0
        
    @property
    def is_available(self) -> bool:
        return True  # Always available for demo
    
    def connect(self) -> bool:
        """Simulate connection."""
        print("Starting demo SITL simulation...")
        self.telemetry.is_connected = True
        self.telemetry.latitude = self._home_lat
        self.telemetry.longitude = self._home_lon
        self.telemetry.mode = "STABILIZE"
        self.telemetry.system_status = "STANDBY"
        self.telemetry.battery_remaining = 100
        self.telemetry.battery_voltage = 16.8
        self.telemetry.gps_fix = 3
        self.telemetry.satellites = 12
        self.telemetry.hdop = 0.9
        
        self._demo_running = True
        self._demo_start_time = time.time()
        self._demo_thread = threading.Thread(target=self._demo_loop, daemon=True)
        self._demo_thread.start()
        
        for cb in self._connection_callbacks:
            cb(True)
        
        print("Demo SITL connected!")
        return True
    
    def disconnect(self):
        """Stop demo simulation."""
        self._demo_running = False
        if self._demo_thread:
            self._demo_thread.join(timeout=1.0)
        self.telemetry.is_connected = False
        
        for cb in self._connection_callbacks:
            cb(False)
        
        print("Demo SITL disconnected")
    
    def _demo_loop(self):
        """Generate simulated telemetry."""
        while self._demo_running:
            t = time.time() - self._demo_start_time
            
            # Simulate flight dynamics
            if self.telemetry.armed and self.telemetry.mode == "GUIDED":
                # Climbing/descending toward target altitude
                alt_error = self._target_alt - self.telemetry.altitude_rel
                climb_rate = max(-5, min(5, alt_error * 0.5))
                self.telemetry.altitude_rel += climb_rate * 0.1
                self.telemetry.vz = -climb_rate
                
                # Add some movement
                self.telemetry.latitude += 0.00001 * math.sin(t * 0.1)
                self.telemetry.longitude += 0.00001 * math.cos(t * 0.1)
                self.telemetry.groundspeed = 5.0 + math.sin(t * 0.5)
            
            # Simulate attitude
            self.telemetry.roll = 0.05 * math.sin(t * 0.3)
            self.telemetry.pitch = 0.03 * math.cos(t * 0.4)
            self.telemetry.yaw = (t * 0.1) % (2 * math.pi)
            
            # Simulate battery drain
            if self.telemetry.armed:
                self.telemetry.battery_remaining = max(0, 100 - t * 0.1)
                self.telemetry.battery_current = 25.0 + 5 * math.sin(t)
            
            self.telemetry.timestamp = time.time()
            self.telemetry.altitude_msl = self.telemetry.altitude_rel + 100
            
            # Store and notify
            self.telemetry_history.append(self._copy_telemetry())
            for cb in self._telemetry_callbacks:
                try:
                    cb(self.telemetry)
                except Exception:
                    pass
            
            time.sleep(0.1)  # 10 Hz update
    
    def arm(self, force: bool = False) -> bool:
        if self.telemetry.mode in ["GUIDED", "AUTO", "LOITER"]:
            self.telemetry.armed = True
            print("Demo: Vehicle armed")
            return True
        print("Demo: Cannot arm in current mode")
        return False
    
    def disarm(self, force: bool = False) -> bool:
        self.telemetry.armed = False
        print("Demo: Vehicle disarmed")
        return True
    
    def set_mode(self, mode: str) -> bool:
        valid_modes = ["STABILIZE", "GUIDED", "AUTO", "LOITER", "RTL", "LAND"]
        if mode.upper() in valid_modes:
            self.telemetry.mode = mode.upper()
            print(f"Demo: Mode set to {mode}")
            return True
        return False
    
    def takeoff(self, altitude: float = 10.0) -> bool:
        if not self.telemetry.armed:
            print("Demo: Must arm before takeoff")
            return False
        self._target_alt = altitude
        print(f"Demo: Taking off to {altitude}m")
        return True
    
    def land(self) -> bool:
        self._target_alt = 0
        self.telemetry.mode = "LAND"
        print("Demo: Landing")
        return True
    
    def goto(self, lat: float, lon: float, alt: float) -> bool:
        self.telemetry.latitude = lat
        self.telemetry.longitude = lon
        self._target_alt = alt
        print(f"Demo: Going to {lat:.6f}, {lon:.6f}, {alt}m")
        return True


def get_connector(use_demo: bool = False) -> SITLConnector:
    """
    Factory function to get appropriate SITL connector.
    
    Args:
        use_demo: If True, return demo connector. If False, return real connector
                  (falls back to demo if pymavlink not available).
    
    Returns:
        SITLConnector instance.
    """
    if use_demo or not PYMAVLINK_AVAILABLE:
        return DemoSITLConnector()
    return SITLConnector()
