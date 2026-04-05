"""
Guidance Layer - Waypoint Navigation and Path Following

Decoupled from fast control loops (0.1 Hz) - operates as slow outer layer
providing velocity references to cascaded control system.

Design:
1. Waypoint arrival detection (spherical acceptance radius)
2. L-1 lateral guidance law (proven stable for aircraft)
3. Threat avoidance (reactive steering)
4. Velocity profile (smooth acceleration/deceleration)

References:
[1] Beard, R.W., McLain, T.W. (2012). "Small UAS Theory and Practice."
    Chapter 3 (guidance laws).
    
[2] Nelson, R.C. (1998). "Flight Stability and Automatic Control." 2nd ed.
    Chapters 5-6 (autopilot structures).

[3] Park, S. et al. (2007). "Experimental Adaptive Backstepping Lateral
    Control of an Unmanned Aircraft." AIAA Journal of Guidance.

Author: Defense eVTOL Research Team
License: MIT
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)


class NavigationMode(Enum):
    """Guidance system operating modes."""
    IDLE = "idle"
    WAYPOINT = "waypoint"
    LOITER = "loiter"
    RETURN_HOME = "return_home"
    THREAT_AVOIDANCE = "threat_avoidance"
    EMERGENCY = "emergency"


@dataclass
class Waypoint:
    """Single navigation waypoint."""
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))  # [m] NED frame
    velocity_desired: float = 10.0  # [m/s]
    loiter_time: float = 0.0  # [s] time to spend at waypoint (0 = pass-through)
    accept_radius: float = 50.0  # [m] when to consider waypoint "reached"
    
    def __post_init__(self):
        self.position = np.array(self.position, dtype=float)


@dataclass
class ThreatZone:
    """Threat zone for avoidance."""
    center: np.ndarray = field(default_factory=lambda: np.zeros(3))
    radius: float = 1000.0  # [m]
    priority: int = 1  # Higher = more dangerous
    
    def __post_init__(self):
        self.center = np.array(self.center, dtype=float)


@dataclass
class GuidanceState:
    """Guidance system state and outputs."""
    mode: NavigationMode = NavigationMode.IDLE
    
    # Waypoint tracking
    current_waypoint_idx: int = 0
    waypoint_list: List[Waypoint] = field(default_factory=list)
    
    # Outputs to control system
    velocity_reference: np.ndarray = field(default_factory=lambda: np.zeros(3))  # [m/s]
    position_error: np.ndarray = field(default_factory=lambda: np.zeros(3))  # [m]
    
    # Status
    is_at_waypoint: bool = False
    loiter_time_remaining: float = 0.0
    
    # Threat avoidance
    threat_zones: List[ThreatZone] = field(default_factory=list)
    avoidance_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))


class GuidanceSystem:
    """
    Guidance layer for waypoint navigation with threat avoidance.
    
    Operates at 0.1 Hz (decoupled from fast control).
    Provides velocity references to cascaded control system.
    
    Usage:
        guidance = GuidanceSystem()
        guidance.set_waypoints([p1, p2, p3])
        v_ref = guidance.update(position_current, velocity_current, dt=10.0)
    """
    
    def __init__(
        self,
        L1_distance: float = 150.0,  # [m] L-1 look-ahead distance
        max_velocity: float = 25.0,  # [m/s]
        home_position: np.ndarray = None,
    ):
        """
        Initialize guidance system.
        
        Args:
            L1_distance: L-1 look-ahead distance [m]
            max_velocity: Maximum commanded velocity [m/s]
            home_position: Home position for RTH [m] in NED
        """
        self.L1_distance = L1_distance
        self.max_velocity = max_velocity
        self.home_position = home_position or np.zeros(3)
        
        self.state = GuidanceState()
        
        logger.info(
            f"GuidanceSystem initialized: L1={L1_distance}m, "
            f"v_max={max_velocity}m/s"
        )
    
    def set_waypoints(self, waypoints: List[Waypoint]):
        """
        Set list of waypoints for mission.
        
        Args:
            waypoints: List of Waypoint objects
        """
        self.state.waypoint_list = waypoints
        self.state.current_waypoint_idx = 0
        logger.info(f"Loaded {len(waypoints)} waypoints")
    
    def add_threat_zone(self, threat: ThreatZone):
        """Add threat zone for avoidance."""
        self.state.threat_zones.append(threat)
        logger.warning(f"Added threat zone at {threat.center} (radius={threat.radius}m)")
    
    def update(
        self,
        position_current: np.ndarray,
        velocity_current: np.ndarray,
        dt: float = 10.0,  # Slow loop (0.1 Hz = 10 s)
    ) -> np.ndarray:
        """
        Execute guidance update cycle.
        
        Args:
            position_current: Current position [m] in NED frame
            velocity_current: Current velocity [m/s]
            dt: Time step [s]
        
        Returns:
            Velocity reference [m/s] for control system
        """
        position_current = np.array(position_current, dtype=float)
        velocity_current = np.array(velocity_current, dtype=float)
        
        # Check for threats and compute avoidance
        avoidance_vel = self._compute_threat_avoidance(position_current)
        
        # Waypoint navigation
        if len(self.state.waypoint_list) == 0:
            self.state.mode = NavigationMode.IDLE
            v_ref = np.array([0.0, 0.0, 0.0])
        else:
            wp = self.state.waypoint_list[self.state.current_waypoint_idx]
            
            # Distance to current waypoint
            delta_pos = wp.position - position_current
            distance_to_wp = np.linalg.norm(delta_pos)
            
            # Check if at waypoint
            if distance_to_wp < wp.accept_radius:
                self.state.is_at_waypoint = True
                self.state.loiter_time_remaining -= dt
                
                if self.state.loiter_time_remaining <= 0:
                    # Move to next waypoint
                    self.state.current_waypoint_idx += 1
                    if self.state.current_waypoint_idx >= len(self.state.waypoint_list):
                        # Mission complete
                        self.state.current_waypoint_idx = 0
                        self.state.mode = NavigationMode.IDLE
                        v_ref = np.array([0.0, 0.0, 0.0])
                    else:
                        # Set up next waypoint
                        next_wp = self.state.waypoint_list[self.state.current_waypoint_idx]
                        self.state.loiter_time_remaining = next_wp.loiter_time
                else:
                    # Loitering at current waypoint
                    self.state.mode = NavigationMode.LOITER
                    v_ref = np.array([0.0, 0.0, 0.0])
            else:
                self.state.is_at_waypoint = False
                self.state.mode = NavigationMode.WAYPOINT
                
                # L-1 guidance law
                v_ref = self._compute_l1_guidance(
                    position_current,
                    delta_pos,
                    wp.velocity_desired
                )
                
                # Apply threat avoidance steering
                if np.linalg.norm(avoidance_vel) > 0.1:
                    self.state.mode = NavigationMode.THREAT_AVOIDANCE
                    v_ref = v_ref + 0.3 * avoidance_vel  # Blend avoidance
            
            self.state.position_error = delta_pos
        
        # Saturate to max velocity
        v_norm = np.linalg.norm(v_ref)
        if v_norm > self.max_velocity:
            v_ref = (v_ref / v_norm) * self.max_velocity
        
        self.state.velocity_reference = v_ref
        
        return v_ref
    
    def _compute_l1_guidance(
        self,
        position: np.ndarray,
        delta_pos: np.ndarray,
        v_desired: float,
    ) -> np.ndarray:
        """
        L-1 lateral guidance law.
        
        Proven stable navigation law that computes desired velocity
        point toward a look-ahead point on the path.
        
        Args:
            position: Current position [m]
            delta_pos: Vector from current position to waypoint [m]
            v_desired: Desired velocity magnitude [m/s]
        
        Returns:
            Velocity reference [m/s]
        """
        # Normalize path direction
        distance_to_wp = np.linalg.norm(delta_pos)
        
        if distance_to_wp < 1.0:
            # At waypoint
            path_direction = delta_pos / (distance_to_wp + 1e-6)
        else:
            path_direction = delta_pos / distance_to_wp
        
        # Cross-track error (perpendicular distance from straight-line path)
        # For single-segment path: cross-track error is implicit in delta_pos
        
        # Compute look-ahead point at distance L1_distance along path
        look_ahead_distance = min(self.L1_distance, distance_to_wp)
        look_ahead_point = position + path_direction * look_ahead_distance
        
        # Desired velocity points toward look-ahead point
        vector_to_lookahead = look_ahead_point - position
        angle_to_lookahead = np.arctan2(
            vector_to_lookahead[1],
            vector_to_lookahead[0]
        )
        
        # Command velocity
        v_north = v_desired * np.cos(angle_to_lookahead)
        v_east = v_desired * np.sin(angle_to_lookahead)
        
        # Climb/descent to target altitude
        altitude_error = delta_pos[2]  # Positive = below target (NED)
        v_down = np.clip(altitude_error * 0.1, -2.0, 2.0)  # Proportional gain
        
        v_ref = np.array([v_north, v_east, v_down])
        
        return v_ref
    
    def _compute_threat_avoidance(
        self,
        position: np.ndarray,
    ) -> np.ndarray:
        """
        Compute avoidance velocity based on threat zones.
        
        Uses simple repulsive potential field.
        
        Args:
            position: Current position [m]
        
        Returns:
            Avoidance velocity [m/s] (zero if no threats nearby)
        """
        avoidance_vel = np.zeros(3)
        
        for threat in self.state.threat_zones:
            delta_threat = position - threat.center
            distance_to_threat = np.linalg.norm(delta_threat)
            
            # Check if in threat sphere + buffer
            if distance_to_threat < threat.radius * 1.5:
                # Repulsive force increases closer to threat
                repulsion_magnitude = 5.0 / (distance_to_threat + 1.0)  # [m/s]
                
                if distance_to_threat > 0.1:
                    repulsion_direction = delta_threat / distance_to_threat
                else:
                    # At threat center - random escape
                    repulsion_direction = np.array([1.0, 0.0, 0.0])
                
                repulsion_vec = repulsion_direction * repulsion_magnitude
                
                # Weight by threat priority
                avoidance_vel += repulsion_vec * threat.priority
        
        return avoidance_vel
    
    def return_to_home(self):
        """Switch to return-to-home mode."""
        self.state.mode = NavigationMode.RETURN_HOME
        logger.info("Returning to home position")
    
    def get_mission_progress(self) -> Tuple[int, int]:
        """
        Get current mission progress.
        
        Returns:
            (current_waypoint_index, total_waypoints)
        """
        return self.state.current_waypoint_idx, len(self.state.waypoint_list)


class TrajectoryGenerator:
    """
    Generate smooth velocity profiles for guidance commands.
    
    Handles acceleration/deceleration to respect vehicle limits.
    """
    
    def __init__(
        self,
        max_acceleration: float = 2.0,  # [m/s²]
        max_deceleration: float = 1.0,  # [m/s²]
    ):
        """
        Initialize trajectory generator.
        
        Args:
            max_acceleration: Maximum acceleration [m/s²]
            max_deceleration: Maximum deceleration [m/s²]
        """
        self.max_accel = max_acceleration
        self.max_decel = max_deceleration
        
        # Current velocity profile state
        self.velocity_profile = np.zeros(3)
    
    def generate_velocity_profile(
        self,
        v_ref: np.ndarray,
        v_actual: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        """
        Generate smooth velocity profile ramping from v_actual toward v_ref.
        
        Respects maximum acceleration/deceleration limits.
        
        Args:
            v_ref: Desired velocity [m/s]
            v_actual: Current velocity [m/s]
            dt: Time step [s]
        
        Returns:
            Commanded velocity profile [m/s]
        """
        v_ref = np.array(v_ref, dtype=float)
        v_actual = np.array(v_actual, dtype=float)
        
        v_error = v_ref - v_actual
        
        # Component-wise acceleration limiting
        v_cmd = np.zeros(3)
        
        for i in range(3):
            if abs(v_error[i]) < 0.1:
                # Already close, no ramping
                v_cmd[i] = v_ref[i]
            elif v_error[i] > 0:
                # Need to accelerate
                max_delta_v = self.max_accel * dt
                v_cmd[i] = v_actual[i] + np.clip(v_error[i], 0, max_delta_v)
            else:
                # Need to decelerate
                max_delta_v = self.max_decel * dt
                v_cmd[i] = v_actual[i] - np.clip(-v_error[i], 0, max_delta_v)
        
        self.velocity_profile = v_cmd
        
        return v_cmd
