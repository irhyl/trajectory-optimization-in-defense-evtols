"""
Control Backend Module - Bridges Real Control Layer with Streamlit

This module provides a clean interface between the Streamlit UI and
the actual flight control and trajectory tracking implementation.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import sys
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

try:
    from evtol.control.flight_controller import FlightController, PIDController, ControlGains
    from evtol.control.trajectory_generator import TrajectoryGenerator
    CONTROL_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import control modules: {e}")
    CONTROL_AVAILABLE = False


@dataclass
class ControlConfig:
    """Configuration for control system simulation."""
    # Simulation parameters
    dt: float = 0.01  # Control timestep (seconds)
    simulation_time: float = 30.0  # Total simulation time
    
    # Position controller gains
    pos_kp: float = 2.0
    pos_ki: float = 0.1
    pos_kd: float = 1.0
    
    # Velocity controller gains
    vel_kp: float = 1.5
    vel_ki: float = 0.2
    vel_kd: float = 0.5
    
    # Attitude controller gains
    att_kp: float = 3.0
    att_ki: float = 0.1
    att_kd: float = 0.8
    
    # Altitude controller gains
    alt_kp: float = 2.5
    alt_ki: float = 0.15
    alt_kd: float = 1.2
    
    # Trajectory parameters
    max_velocity: float = 35.0  # m/s
    max_acceleration: float = 3.0  # m/s²
    
    # Waypoints for trajectory
    waypoints: List[List[float]] = field(default_factory=lambda: [
        [0.0, 0.0, 100.0],
        [500.0, 200.0, 150.0],
        [1000.0, 500.0, 120.0],
        [1500.0, 300.0, 180.0],
        [2000.0, 0.0, 100.0]
    ])
    
    # Disturbance parameters
    wind_disturbance: bool = True
    wind_strength: float = 2.0  # m/s


class ControlSimulator:
    """Simulates the flight control system with real algorithms."""
    
    def __init__(self, config: ControlConfig):
        self.config = config
        
        if CONTROL_AVAILABLE:
            # Initialize real controllers
            self.flight_controller = FlightController(dt=config.dt)
            self.trajectory_generator = TrajectoryGenerator(
                max_velocity=config.max_velocity,
                max_acceleration=config.max_acceleration,
                dt=config.dt
            )
            
            # Update controller gains
            self._update_controller_gains()
        else:
            self.flight_controller = None
            self.trajectory_generator = None
        
        # Simulation state
        self.trajectory = None
        self.simulation_results = None
    
    def _update_controller_gains(self):
        """Update controller gains from config."""
        if not self.flight_controller:
            return
        
        self.flight_controller.position_controller.gains = ControlGains(
            kp=self.config.pos_kp,
            ki=self.config.pos_ki,
            kd=self.config.pos_kd
        )
        self.flight_controller.velocity_controller.gains = ControlGains(
            kp=self.config.vel_kp,
            ki=self.config.vel_ki,
            kd=self.config.vel_kd
        )
        self.flight_controller.attitude_controller.gains = ControlGains(
            kp=self.config.att_kp,
            ki=self.config.att_ki,
            kd=self.config.att_kd
        )
        self.flight_controller.altitude_controller.gains = ControlGains(
            kp=self.config.alt_kp,
            ki=self.config.alt_ki,
            kd=self.config.alt_kd
        )
    
    def generate_trajectory(self) -> List[Dict]:
        """Generate trajectory through waypoints."""
        if not CONTROL_AVAILABLE:
            return self._generate_dummy_trajectory()
        
        waypoints = [np.array(wp) for wp in self.config.waypoints]
        
        self.trajectory = self.trajectory_generator.generate_trajectory(
            waypoints,
            initial_velocity=np.zeros(3),
            final_velocity=np.zeros(3)
        )
        
        return self.trajectory
    
    def _generate_dummy_trajectory(self) -> List[Dict]:
        """Generate dummy trajectory when real module unavailable."""
        num_points = int(self.config.simulation_time / self.config.dt)
        trajectory = []
        
        for i in range(num_points):
            t = i * self.config.dt
            # Simple linear interpolation through waypoints
            progress = t / self.config.simulation_time
            wp_idx = min(int(progress * (len(self.config.waypoints) - 1)), len(self.config.waypoints) - 2)
            local_progress = (progress * (len(self.config.waypoints) - 1)) - wp_idx
            
            start_wp = np.array(self.config.waypoints[wp_idx])
            end_wp = np.array(self.config.waypoints[wp_idx + 1])
            
            position = start_wp + local_progress * (end_wp - start_wp)
            velocity = (end_wp - start_wp) / (self.config.simulation_time / (len(self.config.waypoints) - 1))
            
            trajectory.append({
                'time': t,
                'position': position.tolist(),
                'velocity': velocity.tolist(),
                'acceleration': [0.0, 0.0, 0.0]
            })
        
        self.trajectory = trajectory
        return trajectory
    
    def run_simulation(self) -> Dict[str, Any]:
        """Run full control simulation."""
        if self.trajectory is None:
            self.generate_trajectory()
        
        if not CONTROL_AVAILABLE:
            return self._run_dummy_simulation()
        
        # Initialize state
        current_pos = np.array(self.trajectory[0]['position'])
        current_vel = np.zeros(3)
        current_att = np.zeros(3)  # roll, pitch, yaw
        
        # Results storage
        results = {
            'time': [],
            'target_pos': [],
            'actual_pos': [],
            'target_vel': [],
            'actual_vel': [],
            'position_error': [],
            'velocity_error': [],
            'altitude_error': [],
            'thrust': [],
            'roll_cmd': [],
            'pitch_cmd': [],
            'yaw_cmd': [],
            'roll_actual': [],
            'pitch_actual': [],
            'yaw_actual': [],
            'p_term': [],
            'i_term': [],
            'd_term': [],
            'tracking_success': []
        }
        
        # Reset controller
        self.flight_controller.reset()
        
        for i, traj_point in enumerate(self.trajectory):
            t = traj_point['time']
            target_pos = np.array(traj_point['position'])
            target_vel = np.array(traj_point.get('velocity', [0, 0, 0]))
            
            # Add wind disturbance
            if self.config.wind_disturbance:
                wind = self.config.wind_strength * np.array([
                    np.sin(t * 0.5),
                    np.cos(t * 0.3),
                    np.sin(t * 0.2) * 0.5
                ])
            else:
                wind = np.zeros(3)
            
            # Compute control
            thrust, attitude_cmd = self.flight_controller.compute_control(
                current_pos, current_vel, current_att,
                target_pos, target_vel
            )
            
            # Simple dynamics update (for demonstration)
            # In reality, this would come from vehicle dynamics
            dt = self.config.dt
            
            # Update attitude (simplified first-order response)
            attitude_rate = (attitude_cmd - current_att) * 5.0
            current_att = current_att + attitude_rate * dt
            
            # Convert attitude to acceleration
            g = 9.81
            accel = np.array([
                g * np.tan(current_att[1]),  # pitch -> x accel
                -g * np.tan(current_att[0]),  # roll -> y accel
                thrust - g  # vertical accel
            ]) + wind * 0.1  # Add wind effect
            
            # Update velocity and position
            current_vel = current_vel + accel * dt
            current_pos = current_pos + current_vel * dt
            
            # Compute errors
            pos_error = np.linalg.norm(target_pos - current_pos)
            vel_error = np.linalg.norm(target_vel - current_vel)
            alt_error = abs(target_pos[2] - current_pos[2])
            
            # Tracking success (position error < threshold)
            tracking_success = 100.0 * max(0, 1 - pos_error / 50.0)
            
            # Get PID terms from position controller
            p_term = self.flight_controller.position_controller.gains.kp * pos_error
            i_term = np.linalg.norm(self.flight_controller.position_controller.state.integral_error)
            d_term = np.linalg.norm(self.flight_controller.position_controller.state.previous_error) * \
                     self.flight_controller.position_controller.gains.kd
            
            # Store results
            results['time'].append(t)
            results['target_pos'].append(target_pos.tolist())
            results['actual_pos'].append(current_pos.tolist())
            results['target_vel'].append(target_vel.tolist())
            results['actual_vel'].append(current_vel.tolist())
            results['position_error'].append(pos_error)
            results['velocity_error'].append(vel_error)
            results['altitude_error'].append(alt_error)
            results['thrust'].append(thrust)
            results['roll_cmd'].append(np.degrees(attitude_cmd[0]))
            results['pitch_cmd'].append(np.degrees(attitude_cmd[1]))
            results['yaw_cmd'].append(np.degrees(attitude_cmd[2]))
            results['roll_actual'].append(np.degrees(current_att[0]))
            results['pitch_actual'].append(np.degrees(current_att[1]))
            results['yaw_actual'].append(np.degrees(current_att[2]))
            results['p_term'].append(p_term)
            results['i_term'].append(i_term)
            results['d_term'].append(d_term)
            results['tracking_success'].append(tracking_success)
        
        self.simulation_results = results
        return results
    
    def _run_dummy_simulation(self) -> Dict[str, Any]:
        """Run dummy simulation when real module unavailable."""
        num_points = len(self.trajectory)
        t = np.linspace(0, self.config.simulation_time, num_points)
        
        results = {
            'time': t.tolist(),
            'position_error': (5.0 * np.exp(-t / 10) + 0.5 * np.random.randn(num_points)).tolist(),
            'velocity_error': (2.0 * np.exp(-t / 8) + 0.2 * np.random.randn(num_points)).tolist(),
            'altitude_error': (3.0 * np.exp(-t / 12) + 0.3 * np.random.randn(num_points)).tolist(),
            'thrust': (9.81 + 0.5 * np.sin(t * 0.5)).tolist(),
            'roll_cmd': (np.sin(t * 0.3) * 5).tolist(),
            'pitch_cmd': (np.cos(t * 0.3) * 3).tolist(),
            'yaw_cmd': (t * 2).tolist(),
            'p_term': (np.abs(np.sin(t * 0.5) * 2)).tolist(),
            'i_term': (0.5 * (1 - np.exp(-t / 5))).tolist(),
            'd_term': (np.abs(np.cos(t * 0.5) * 0.5)).tolist(),
            'tracking_success': (90 + 10 * (1 - np.exp(-t / 5)) + 2 * np.random.randn(num_points)).tolist()
        }
        
        self.simulation_results = results
        return results


# Singleton instance
_control_simulator: Optional[ControlSimulator] = None


def get_control_simulator(config: Optional[ControlConfig] = None) -> ControlSimulator:
    """Get or create control simulator instance."""
    global _control_simulator
    
    if _control_simulator is None or config is not None:
        if config is None:
            config = ControlConfig()
        _control_simulator = ControlSimulator(config)
    
    return _control_simulator


def generate_pid_data(config: Optional[ControlConfig] = None) -> pd.DataFrame:
    """Generate PID control data for visualization."""
    simulator = get_control_simulator(config)
    
    if simulator.simulation_results is None:
        simulator.run_simulation()
    
    results = simulator.simulation_results
    
    df = pd.DataFrame({
        'Time': results['time'],
        'P_term': results['p_term'],
        'I_term': results['i_term'],
        'D_term': results['d_term'],
        'Thrust': results['thrust']
    })
    
    return df


def generate_trajectory_tracking_data(config: Optional[ControlConfig] = None) -> pd.DataFrame:
    """Generate trajectory tracking data for visualization."""
    simulator = get_control_simulator(config)
    
    if simulator.simulation_results is None:
        simulator.run_simulation()
    
    results = simulator.simulation_results
    
    df = pd.DataFrame({
        'Time': results['time'],
        'Position Error (m)': results['position_error'],
        'Velocity Error (m/s)': results['velocity_error'],
        'Altitude Error (m)': results['altitude_error'],
        'Tracking Success (%)': results['tracking_success']
    })
    
    return df


def generate_attitude_data(config: Optional[ControlConfig] = None) -> pd.DataFrame:
    """Generate attitude control data for visualization."""
    simulator = get_control_simulator(config)
    
    if simulator.simulation_results is None:
        simulator.run_simulation()
    
    results = simulator.simulation_results
    
    # Handle case where actual values might not be present
    if 'roll_actual' not in results:
        results['roll_actual'] = results['roll_cmd']
        results['pitch_actual'] = results['pitch_cmd']
        results['yaw_actual'] = results['yaw_cmd']
    
    df = pd.DataFrame({
        'Time': results['time'],
        'Roll Command (deg)': results['roll_cmd'],
        'Pitch Command (deg)': results['pitch_cmd'],
        'Yaw Command (deg)': results['yaw_cmd'],
        'Roll Actual (deg)': results['roll_actual'],
        'Pitch Actual (deg)': results['pitch_actual'],
        'Yaw Actual (deg)': results['yaw_actual'],
        'Roll Error (deg)': [c - a for c, a in zip(results['roll_cmd'], results['roll_actual'])],
        'Pitch Error (deg)': [c - a for c, a in zip(results['pitch_cmd'], results['pitch_actual'])],
        'Yaw Error (deg)': [c - a for c, a in zip(results['yaw_cmd'], results['yaw_actual'])]
    })
    
    return df


def generate_performance_metrics(config: Optional[ControlConfig] = None) -> Dict[str, Any]:
    """Generate performance metrics from control simulation."""
    simulator = get_control_simulator(config)
    
    if simulator.simulation_results is None:
        simulator.run_simulation()
    
    results = simulator.simulation_results
    
    pos_errors = np.array(results['position_error'])
    vel_errors = np.array(results['velocity_error'])
    tracking = np.array(results['tracking_success'])
    
    # Compute performance metrics
    metrics = {
        'position_hold_accuracy': f"{np.mean(pos_errors[-100:]):.2f}m" if len(pos_errors) > 100 else f"{np.mean(pos_errors):.2f}m",
        'max_position_error': f"{np.max(pos_errors):.2f}m",
        'mean_position_error': f"{np.mean(pos_errors):.2f}m",
        'max_velocity_error': f"{np.max(vel_errors):.2f}m/s",
        'mean_velocity_error': f"{np.mean(vel_errors):.2f}m/s",
        'tracking_success_rate': f"{np.mean(tracking):.1f}%",
        'rise_time': self._estimate_rise_time(pos_errors),
        'settling_time': self._estimate_settling_time(pos_errors),
        'overshoot': self._estimate_overshoot(pos_errors),
        'steady_state_error': f"{np.mean(pos_errors[-50:]):.2f}m" if len(pos_errors) > 50 else f"{np.mean(pos_errors):.2f}m"
    }
    
    return metrics


def _estimate_rise_time(errors: np.ndarray) -> str:
    """Estimate rise time from error profile."""
    if len(errors) < 10:
        return "N/A"
    
    # Find when error drops to 10% of initial
    initial = errors[0]
    threshold = initial * 0.1
    
    for i, e in enumerate(errors):
        if e < threshold:
            return f"{i * 0.01:.2f}s"
    
    return f"{len(errors) * 0.01:.2f}s"


def _estimate_settling_time(errors: np.ndarray) -> str:
    """Estimate settling time from error profile."""
    if len(errors) < 10:
        return "N/A"
    
    # Find when error stays below 2% of initial
    initial = errors[0]
    threshold = max(initial * 0.02, 0.5)
    
    for i in range(len(errors) - 10, -1, -1):
        if errors[i] > threshold:
            return f"{(i + 10) * 0.01:.2f}s"
    
    return f"{0.1:.2f}s"


def _estimate_overshoot(errors: np.ndarray) -> str:
    """Estimate overshoot from error profile."""
    if len(errors) < 10:
        return "0%"
    
    # Look for local minimum then maximum
    initial = errors[0]
    min_error = np.min(errors[:len(errors)//2])
    max_after_min = np.max(errors[np.argmin(errors[:len(errors)//2]):])
    
    if min_error < 1.0 and max_after_min > min_error:
        overshoot = (max_after_min - min_error) / max(initial, 1.0) * 100
        return f"{overshoot:.1f}%"
    
    return "0%"


def generate_step_response_data(controller_type: str = "altitude",
                                 config: Optional[ControlConfig] = None) -> pd.DataFrame:
    """Generate step response data for a specific controller."""
    dt = 0.01
    t = np.arange(0, 5, dt)
    
    # Different response characteristics for different controllers
    if controller_type == "altitude":
        # Altitude: slightly underdamped
        wn = 2.0  # natural frequency
        zeta = 0.7  # damping ratio
        wd = wn * np.sqrt(1 - zeta**2)
        response = 1 - np.exp(-zeta * wn * t) * (np.cos(wd * t) + (zeta * wn / wd) * np.sin(wd * t))
    elif controller_type == "velocity":
        # Velocity: faster response
        wn = 3.0
        zeta = 0.8
        wd = wn * np.sqrt(1 - zeta**2)
        response = 1 - np.exp(-zeta * wn * t) * (np.cos(wd * t) + (zeta * wn / wd) * np.sin(wd * t))
    elif controller_type == "pitch":
        # Pitch: critically damped
        wn = 4.0
        zeta = 1.0
        response = 1 - (1 + wn * t) * np.exp(-wn * t)
    elif controller_type == "roll":
        # Roll: similar to pitch
        wn = 4.5
        zeta = 0.95
        wd = wn * np.sqrt(1 - zeta**2) if zeta < 1 else 0
        if zeta < 1:
            response = 1 - np.exp(-zeta * wn * t) * (np.cos(wd * t) + (zeta * wn / wd) * np.sin(wd * t))
        else:
            response = 1 - (1 + wn * t) * np.exp(-wn * t)
    else:  # yaw
        # Yaw: slower response
        wn = 2.5
        zeta = 0.85
        wd = wn * np.sqrt(1 - zeta**2)
        response = 1 - np.exp(-zeta * wn * t) * (np.cos(wd * t) + (zeta * wn / wd) * np.sin(wd * t))
    
    df = pd.DataFrame({
        'Time (s)': t,
        'Response': response,
        'Setpoint': np.ones_like(t)
    })
    
    return df


def generate_gains_config() -> pd.DataFrame:
    """Generate gains configuration table."""
    config = ControlConfig()
    
    df = pd.DataFrame({
        "Channel": ["Altitude", "Velocity", "Pitch/Roll", "Position"],
        "P Gain": [config.alt_kp, config.vel_kp, config.att_kp, config.pos_kp],
        "I Gain": [config.alt_ki, config.vel_ki, config.att_ki, config.pos_ki],
        "D Gain": [config.alt_kd, config.vel_kd, config.att_kd, config.pos_kd]
    })
    
    return df


def reset_simulator():
    """Reset the control simulator."""
    global _control_simulator
    _control_simulator = None


# Export functions for Streamlit
def export_pid_data(config: Optional[ControlConfig] = None) -> str:
    """Export PID data as CSV string."""
    df = generate_pid_data(config)
    return df.to_csv(index=False)


def export_tracking_data(config: Optional[ControlConfig] = None) -> str:
    """Export trajectory tracking data as CSV string."""
    df = generate_trajectory_tracking_data(config)
    return df.to_csv(index=False)


def export_attitude_data(config: Optional[ControlConfig] = None) -> str:
    """Export attitude control data as CSV string."""
    df = generate_attitude_data(config)
    return df.to_csv(index=False)


if __name__ == "__main__":
    # Test the control backend
    print("Testing Control Backend...")
    
    config = ControlConfig()
    simulator = get_control_simulator(config)
    
    print("\nGenerating trajectory...")
    trajectory = simulator.generate_trajectory()
    print(f"Generated {len(trajectory)} trajectory points")
    
    print("\nRunning simulation...")
    results = simulator.run_simulation()
    print(f"Simulation complete with {len(results['time'])} timesteps")
    
    print("\nGenerating PID data...")
    pid_df = generate_pid_data()
    print(pid_df.head())
    
    print("\nGenerating tracking data...")
    tracking_df = generate_trajectory_tracking_data()
    print(tracking_df.head())
    
    print("\nPerformance metrics:")
    metrics = generate_performance_metrics()
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    
    print("\nControl backend test complete!")
