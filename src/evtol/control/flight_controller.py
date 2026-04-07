"""
Flight Control Module

Implements basic flight control for eVTOL trajectory tracking.
Includes PID controllers for position, velocity, and attitude.
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ControlGains:
    """PID control gains"""
    kp: float = 1.0
    ki: float = 0.1
    kd: float = 0.5


@dataclass
class ControlState:
    """Control state variables"""
    integral_error: np.ndarray = None
    previous_error: np.ndarray = None
    
    def __post_init__(self):
        if self.integral_error is None:
            self.integral_error = np.zeros(3)
        if self.previous_error is None:
            self.previous_error = np.zeros(3)


class PIDController:
    """Generic PID controller."""
    
    def __init__(self, gains: ControlGains, dt: float = 0.01):
        """
        Initialize PID controller.
        
        Args:
            gains: PID gains
            dt: Control timestep
        """
        self.gains = gains
        self.dt = dt
        self.state = ControlState()
        
        # Anti-windup limits
        self.integral_limit = 10.0
    
    def compute(
        self,
        error: np.ndarray,
        derivative: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute PID control output.
        
        Args:
            error: Current error
            derivative: Error derivative (computed if None)
            
        Returns:
            Control output
        """
        # Proportional term
        p_term = self.gains.kp * error
        
        # Integral term with anti-windup
        self.state.integral_error += error * self.dt
        self.state.integral_error = np.clip(
            self.state.integral_error,
            -self.integral_limit,
            self.integral_limit
        )
        i_term = self.gains.ki * self.state.integral_error
        
        # Derivative term
        if derivative is None:
            derivative = (error - self.state.previous_error) / self.dt
        d_term = self.gains.kd * derivative
        
        # Update state
        self.state.previous_error = error.copy()
        
        # Total control
        control = p_term + i_term + d_term
        
        return control
    
    def reset(self):
        """Reset controller state."""
        self.state = ControlState()


class FlightController:
    """
    Hierarchical flight controller for eVTOL.
    
    Structure:
    - Outer loop: Position control
    - Inner loop: Velocity control
    - Innermost loop: Attitude control
    """
    
    def __init__(self, dt: float = 0.01):
        """
        Initialize flight controller.
        
        Args:
            dt: Control timestep in seconds
        """
        self.dt = dt
        
        # Position controller (outer loop)
        self.position_controller = PIDController(
            ControlGains(kp=2.0, ki=0.1, kd=1.0),
            dt=dt
        )
        
        # Velocity controller (middle loop)
        self.velocity_controller = PIDController(
            ControlGains(kp=1.5, ki=0.2, kd=0.5),
            dt=dt
        )
        
        # Attitude controller (inner loop)
        self.attitude_controller = PIDController(
            ControlGains(kp=3.0, ki=0.1, kd=0.8),
            dt=dt
        )
        
        # Altitude controller (separate)
        self.altitude_controller = PIDController(
            ControlGains(kp=2.5, ki=0.15, kd=1.2),
            dt=dt
        )
        
        logger.info("Flight controller initialized")
    
    def compute_control(
        self,
        current_position: np.ndarray,  # [x, y, z]
        current_velocity: np.ndarray,  # [vx, vy, vz]
        current_attitude: np.ndarray,  # [roll, pitch, yaw]
        target_position: np.ndarray,
        target_velocity: Optional[np.ndarray] = None,
        target_attitude: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute control commands.
        
        Args:
            current_position: Current position [x, y, z] in meters
            current_velocity: Current velocity [vx, vy, vz] in m/s
            current_attitude: Current attitude [roll, pitch, yaw] in radians
            target_position: Target position
            target_velocity: Target velocity (None for hover)
            target_attitude: Target attitude (None for computed)
            
        Returns:
            (thrust_commands, attitude_commands)
        """
        # Position error
        position_error = target_position - current_position
        
        # Desired velocity from position controller
        desired_velocity = self.position_controller.compute(position_error)
        
        # Use target velocity if provided
        if target_velocity is not None:
            desired_velocity = target_velocity
        
        # Velocity error
        velocity_error = desired_velocity - current_velocity
        
        # Desired acceleration from velocity controller
        desired_accel = self.velocity_controller.compute(velocity_error)
        
        # Convert desired acceleration to thrust and attitude
        thrust, desired_attitude = self._accel_to_thrust_attitude(
            desired_accel,
            current_attitude
        )
        
        # Use target attitude if provided
        if target_attitude is not None:
            desired_attitude = target_attitude
        
        # Attitude error
        attitude_error = desired_attitude - current_attitude
        
        # Normalize angles to [-π, π]
        attitude_error = np.arctan2(np.sin(attitude_error), np.cos(attitude_error))
        
        # Attitude control
        attitude_commands = self.attitude_controller.compute(attitude_error)
        
        return thrust, attitude_commands
    
    def _accel_to_thrust_attitude(
        self,
        desired_accel: np.ndarray,
        current_attitude: np.ndarray
    ) -> Tuple[float, np.ndarray]:
        """
        Convert desired acceleration to thrust and attitude.
        
        Uses simplified model: thrust compensates for gravity plus desired accel.
        Attitude provides horizontal acceleration.
        
        Args:
            desired_accel: Desired acceleration [ax, ay, az]
            current_attitude: Current attitude [roll, pitch, yaw]
            
        Returns:
            (thrust, desired_attitude)
        """
        g = 9.81  # m/s²
        
        # Total thrust magnitude (vertical)
        thrust = g + desired_accel[2]  # Hover + vertical accel
        
        # Desired pitch and roll from horizontal acceleration
        # Small angle approximation: ax ≈ g*tan(pitch)
        desired_pitch = np.arctan2(desired_accel[0], g)
        desired_roll = np.arctan2(-desired_accel[1], g)
        
        # Keep current yaw
        desired_yaw = current_attitude[2]
        
        desired_attitude = np.array([desired_roll, desired_pitch, desired_yaw])
        
        return thrust, desired_attitude
    
    def compute_trajectory_tracking(
        self,
        current_state: dict,
        trajectory_point: dict,
        next_point: Optional[dict] = None
    ) -> dict:
        """
        Compute controls for trajectory tracking.
        
        Args:
            current_state: Current vehicle state
            trajectory_point: Current trajectory target
            next_point: Next trajectory point (for feedforward)
            
        Returns:
            Control commands dictionary
        """
        current_pos = np.array(current_state['position'])
        current_vel = np.array(current_state['velocity'])
        current_att = np.array(current_state['attitude'])
        
        target_pos = np.array(trajectory_point['position'])
        target_vel = np.array(trajectory_point.get('velocity', [0, 0, 0]))
        
        # Compute control
        thrust, attitude_cmd = self.compute_control(
            current_pos, current_vel, current_att,
            target_pos, target_vel
        )
        
        return {
            'thrust': float(thrust),
            'roll_cmd': float(attitude_cmd[0]),
            'pitch_cmd': float(attitude_cmd[1]),
            'yaw_cmd': float(attitude_cmd[2]),
            'timestamp': trajectory_point.get('time', 0.0)
        }
    
    def reset(self):
        """Reset all controllers."""
        self.position_controller.reset()
        self.velocity_controller.reset()
        self.attitude_controller.reset()
        self.altitude_controller.reset()


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    controller = FlightController(dt=0.01)
    
    # Current state
    current_pos = np.array([0.0, 0.0, 100.0])
    current_vel = np.array([5.0, 0.0, 0.0])
    current_att = np.array([0.0, 0.0, 0.0])
    
    # Target state
    target_pos = np.array([10.0, 5.0, 100.0])
    
    # Compute control
    thrust, attitude_cmd = controller.compute_control(
        current_pos, current_vel, current_att,
        target_pos
    )
    
    print("Control Output:")
    print(f"  Thrust: {thrust:.2f} m/s²")
    print(f"  Roll: {np.degrees(attitude_cmd[0]):.2f}°")
    print(f"  Pitch: {np.degrees(attitude_cmd[1]):.2f}°")
    print(f"  Yaw: {np.degrees(attitude_cmd[2]):.2f}°")



