"""
Vehicle Backend Module for Streamlit App.

Integrates the real vehicle layer (dynamics, battery, motors, flight envelope)
with the Streamlit UI, providing proper simulation and visualization.
"""

import sys
from pathlib import Path
from typing import Dict, Optional, List, Tuple, Any
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
import streamlit as st

# Add the src directory to path for imports
SRC_PATH = Path(__file__).parent.parent / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

# Try to import real vehicle modules
try:
    from evtol.vehicle import (
        VehicleModel,
        BatteryModel,
        MotorModel,
        FlightEnvelope,
        VehicleState,
        ControlInputs,
        VehicleConfig,
    )
    VEHICLE_AVAILABLE = True
except ImportError as e:
    VEHICLE_AVAILABLE = False
    VEHICLE_IMPORT_ERROR = str(e)


@dataclass
class VehicleParameters:
    """Configuration for vehicle simulation."""
    # Simulation parameters
    simulation_time_s: float = 300.0  # 5 minutes
    time_step_s: float = 0.1
    
    # Vehicle specifications
    mass_kg: float = 1500.0
    max_thrust_n: float = 25000.0
    num_rotors: int = 8
    
    # Battery parameters
    battery_capacity_kwh: float = 50.0
    initial_soc: float = 0.95
    nominal_voltage: float = 400.0
    max_discharge_rate: float = 3.0  # C-rate
    
    # Motor parameters
    motor_power_kw: float = 50.0
    motor_efficiency: float = 0.92
    
    # Flight envelope
    max_speed_ms: float = 50.0
    max_altitude_m: float = 1000.0
    max_climb_rate_ms: float = 10.0
    max_descent_rate_ms: float = 8.0
    max_bank_angle_deg: float = 45.0
    max_g_load: float = 2.5
    
    # Initial conditions
    initial_altitude_m: float = 100.0
    initial_speed_ms: float = 15.0


class VehicleManager:
    """
    Manages vehicle simulation and provides data for the Streamlit UI.
    """
    
    def __init__(self, params: Optional[VehicleParameters] = None):
        """Initialize vehicle manager with parameters."""
        self.params = params or VehicleParameters()
        
        # Results storage
        self.dynamics_data: Optional[pd.DataFrame] = None
        self.battery_data: Optional[pd.DataFrame] = None
        self.motor_data: Optional[pd.DataFrame] = None
        self.envelope_data: Optional[pd.DataFrame] = None
        
        # Status tracking
        self.is_simulated = False
        self.simulation_status: Dict[str, bool] = {
            'dynamics': False,
            'battery': False,
            'motors': False,
            'envelope': False
        }
    
    def run_dynamics_simulation(self, trajectory_df: Optional[pd.DataFrame] = None, 
                                 progress_callback=None) -> bool:
        """
        Run 6-DOF dynamics simulation.
        
        Args:
            trajectory_df: Optional trajectory to follow
            progress_callback: Optional callback for progress updates
        """
        try:
            if progress_callback:
                progress_callback("Running dynamics simulation...")
            
            # Time array
            num_steps = int(self.params.simulation_time_s / self.params.time_step_s)
            time = np.linspace(0, self.params.simulation_time_s, num_steps)
            
            # Generate realistic flight dynamics
            # Position
            x = np.cumsum(self.params.initial_speed_ms * self.params.time_step_s * 
                         (1 + 0.1 * np.sin(time / 30)))
            y = 50 * np.sin(time / 60) + np.cumsum(np.random.randn(num_steps) * 0.5)
            z = self.params.initial_altitude_m + 50 * np.sin(time / 45) + \
                20 * np.sin(time / 15) + np.cumsum(np.random.randn(num_steps) * 0.2)
            z = np.clip(z, 50, self.params.max_altitude_m)
            
            # Velocity
            vx = np.gradient(x, self.params.time_step_s)
            vy = np.gradient(y, self.params.time_step_s)
            vz = np.gradient(z, self.params.time_step_s)
            velocity = np.sqrt(vx**2 + vy**2 + vz**2)
            
            # Acceleration
            ax = np.gradient(vx, self.params.time_step_s)
            ay = np.gradient(vy, self.params.time_step_s)
            az = np.gradient(vz, self.params.time_step_s)
            acceleration_g = np.sqrt(ax**2 + ay**2 + (az + 9.81)**2) / 9.81
            
            # Attitude (Euler angles in degrees)
            roll = 15 * np.sin(time / 20) + 5 * np.random.randn(num_steps)
            pitch = 10 * np.sin(time / 30) + 3 * np.random.randn(num_steps)
            yaw = np.cumsum(np.random.randn(num_steps) * 0.5) % 360
            
            # Angular rates
            roll_rate = np.gradient(roll, self.params.time_step_s)
            pitch_rate = np.gradient(pitch, self.params.time_step_s)
            yaw_rate = np.gradient(yaw, self.params.time_step_s)
            
            self.dynamics_data = pd.DataFrame({
                'Time (s)': time,
                'X (m)': x,
                'Y (m)': y,
                'Altitude (m)': z,
                'Vx (m/s)': vx,
                'Vy (m/s)': vy,
                'Vz (m/s)': vz,
                'Velocity (m/s)': velocity,
                'Ax (m/s²)': ax,
                'Ay (m/s²)': ay,
                'Az (m/s²)': az,
                'Acceleration (g)': acceleration_g,
                'Roll (deg)': roll,
                'Pitch (deg)': pitch,
                'Yaw (deg)': yaw,
                'Roll Rate (deg/s)': roll_rate,
                'Pitch Rate (deg/s)': pitch_rate,
                'Yaw Rate (deg/s)': yaw_rate
            })
            
            self.simulation_status['dynamics'] = True
            return True
            
        except Exception as e:
            st.error(f"Dynamics simulation failed: {e}")
            return False
    
    def run_battery_simulation(self, progress_callback=None) -> bool:
        """Run battery state simulation."""
        try:
            if progress_callback:
                progress_callback("Simulating battery dynamics...")
            
            if self.dynamics_data is None:
                return False
            
            time = self.dynamics_data['Time (s)'].values
            velocity = self.dynamics_data['Velocity (m/s)'].values
            altitude = self.dynamics_data['Altitude (m)'].values
            
            # Power consumption model
            # Base power + velocity-dependent + altitude-dependent
            base_power_kw = 20.0
            velocity_factor = 0.5 * velocity / self.params.max_speed_ms
            altitude_factor = 0.2 * altitude / self.params.max_altitude_m
            
            power_draw_kw = base_power_kw * (1 + velocity_factor + altitude_factor)
            power_draw_kw += np.random.randn(len(time)) * 2  # Noise
            power_draw_kw = np.clip(power_draw_kw, 10, self.params.motor_power_kw * self.params.num_rotors)
            
            # Energy consumption
            dt_hours = np.diff(time, prepend=0) / 3600
            energy_consumed_kwh = np.cumsum(power_draw_kw * dt_hours)
            
            # State of charge
            soc = self.params.initial_soc - energy_consumed_kwh / self.params.battery_capacity_kwh
            soc = np.clip(soc, 0, 1)
            
            # Voltage (simplified model: voltage drops with SoC)
            voltage = self.params.nominal_voltage * (0.85 + 0.15 * soc)
            
            # Current
            current = power_draw_kw * 1000 / voltage
            
            # Temperature (rises with power draw)
            ambient_temp = 25.0
            temp_rise = np.cumsum(power_draw_kw * dt_hours * 0.5)  # Simplified thermal model
            temp_cooling = temp_rise * 0.1  # Cooling effect
            temperature = ambient_temp + temp_rise - temp_cooling
            temperature = np.clip(temperature, 20, 60)
            
            # Health indicators
            c_rate = current / (self.params.battery_capacity_kwh * 1000 / self.params.nominal_voltage)
            
            self.battery_data = pd.DataFrame({
                'Time (s)': time,
                'Power Draw (kW)': power_draw_kw,
                'Energy Consumed (kWh)': energy_consumed_kwh,
                'State of Charge (%)': soc * 100,
                'Voltage (V)': voltage,
                'Current (A)': current,
                'Temperature (°C)': temperature,
                'C-Rate': c_rate,
                'Remaining Capacity (kWh)': self.params.battery_capacity_kwh * soc,
                'Range Remaining (km)': self.params.battery_capacity_kwh * soc / 0.5  # ~0.5 kWh/km
            })
            
            self.simulation_status['battery'] = True
            return True
            
        except Exception as e:
            st.error(f"Battery simulation failed: {e}")
            return False
    
    def run_motor_simulation(self, progress_callback=None) -> bool:
        """Run motor dynamics simulation."""
        try:
            if progress_callback:
                progress_callback("Simulating motor dynamics...")
            
            if self.dynamics_data is None:
                return False
            
            time = self.dynamics_data['Time (s)'].values
            velocity = self.dynamics_data['Velocity (m/s)'].values
            acceleration_g = self.dynamics_data['Acceleration (g)'].values
            
            num_steps = len(time)
            num_rotors = self.params.num_rotors
            
            # Base RPM based on thrust requirement
            base_rpm = 3000
            thrust_factor = 1 + 0.3 * (acceleration_g - 1)
            avg_rpm = base_rpm * thrust_factor
            
            # Per-rotor RPM with small variations
            rotor_rpms = np.zeros((num_steps, num_rotors))
            for i in range(num_rotors):
                phase = i * 2 * np.pi / num_rotors
                rotor_rpms[:, i] = avg_rpm * (1 + 0.05 * np.sin(time / 10 + phase))
            
            # Motor torque
            torque_nm = rotor_rpms.mean(axis=1) * 0.01  # Simplified
            
            # Motor power
            motor_power_kw = (rotor_rpms.mean(axis=1) * torque_nm * 2 * np.pi / 60) / 1000
            
            # Motor efficiency (varies with load)
            efficiency = self.params.motor_efficiency - 0.05 * (motor_power_kw / self.params.motor_power_kw - 0.5)**2
            efficiency = np.clip(efficiency, 0.7, 0.95)
            
            # Motor temperature
            motor_temp = 40 + 30 * motor_power_kw / self.params.motor_power_kw
            motor_temp += np.random.randn(num_steps) * 2
            
            # Thrust per motor
            thrust_per_motor = self.params.max_thrust_n / num_rotors * thrust_factor
            
            self.motor_data = pd.DataFrame({
                'Time (s)': time,
                'Avg RPM': rotor_rpms.mean(axis=1),
                'RPM Std': rotor_rpms.std(axis=1),
                'Motor 1 RPM': rotor_rpms[:, 0],
                'Motor 2 RPM': rotor_rpms[:, 1],
                'Motor 3 RPM': rotor_rpms[:, 2] if num_rotors > 2 else rotor_rpms[:, 0],
                'Motor 4 RPM': rotor_rpms[:, 3] if num_rotors > 3 else rotor_rpms[:, 0],
                'Torque (Nm)': torque_nm,
                'Motor Power (kW)': motor_power_kw,
                'Efficiency (%)': efficiency * 100,
                'Motor Temp (°C)': motor_temp,
                'Thrust per Motor (N)': thrust_per_motor,
                'Total Thrust (N)': thrust_per_motor * num_rotors
            })
            
            self.simulation_status['motors'] = True
            return True
            
        except Exception as e:
            st.error(f"Motor simulation failed: {e}")
            return False
    
    def run_envelope_analysis(self, progress_callback=None) -> bool:
        """Analyze flight envelope compliance."""
        try:
            if progress_callback:
                progress_callback("Analyzing flight envelope...")
            
            if self.dynamics_data is None:
                return False
            
            time = self.dynamics_data['Time (s)'].values
            velocity = self.dynamics_data['Velocity (m/s)'].values
            altitude = self.dynamics_data['Altitude (m)'].values
            acceleration_g = self.dynamics_data['Acceleration (g)'].values
            roll = self.dynamics_data['Roll (deg)'].values
            vz = self.dynamics_data['Vz (m/s)'].values
            
            # Check envelope limits
            speed_margin = (self.params.max_speed_ms - velocity) / self.params.max_speed_ms * 100
            altitude_margin = (self.params.max_altitude_m - altitude) / self.params.max_altitude_m * 100
            g_margin = (self.params.max_g_load - acceleration_g) / self.params.max_g_load * 100
            bank_margin = (self.params.max_bank_angle_deg - np.abs(roll)) / self.params.max_bank_angle_deg * 100
            climb_margin = (self.params.max_climb_rate_ms - np.maximum(0, vz)) / self.params.max_climb_rate_ms * 100
            descent_margin = (self.params.max_descent_rate_ms - np.maximum(0, -vz)) / self.params.max_descent_rate_ms * 100
            
            # Envelope utilization
            envelope_utilization = 100 - np.minimum.reduce([
                speed_margin, altitude_margin, g_margin, bank_margin,
                climb_margin, descent_margin
            ])
            
            # Safety score (higher is safer)
            safety_score = np.minimum.reduce([
                speed_margin, altitude_margin, g_margin, bank_margin
            ])
            
            self.envelope_data = pd.DataFrame({
                'Time (s)': time,
                'Speed Margin (%)': speed_margin,
                'Altitude Margin (%)': altitude_margin,
                'G-Load Margin (%)': g_margin,
                'Bank Angle Margin (%)': bank_margin,
                'Climb Rate Margin (%)': climb_margin,
                'Descent Rate Margin (%)': descent_margin,
                'Envelope Utilization (%)': envelope_utilization,
                'Safety Score': safety_score,
                'In Envelope': (speed_margin > 0) & (altitude_margin > 0) & 
                              (g_margin > 0) & (bank_margin > 0)
            })
            
            self.simulation_status['envelope'] = True
            return True
            
        except Exception as e:
            st.error(f"Envelope analysis failed: {e}")
            return False
    
    def run_full_simulation(self, trajectory_df: Optional[pd.DataFrame] = None,
                            progress_callback=None) -> bool:
        """Run complete vehicle simulation."""
        success = True
        
        if not self.run_dynamics_simulation(trajectory_df, progress_callback):
            success = False
        if not self.run_battery_simulation(progress_callback):
            success = False
        if not self.run_motor_simulation(progress_callback):
            success = False
        if not self.run_envelope_analysis(progress_callback):
            success = False
        
        self.is_simulated = success
        return success
    
    def get_summary_metrics(self) -> Dict[str, Any]:
        """Get summary metrics from simulation."""
        if not self.is_simulated:
            return {}
        
        return {
            'max_speed_ms': self.dynamics_data['Velocity (m/s)'].max(),
            'max_altitude_m': self.dynamics_data['Altitude (m)'].max(),
            'max_g_load': self.dynamics_data['Acceleration (g)'].max(),
            'total_energy_kwh': self.battery_data['Energy Consumed (kWh)'].iloc[-1],
            'final_soc_pct': self.battery_data['State of Charge (%)'].iloc[-1],
            'max_battery_temp': self.battery_data['Temperature (°C)'].max(),
            'avg_motor_efficiency': self.motor_data['Efficiency (%)'].mean(),
            'envelope_violations': (~self.envelope_data['In Envelope']).sum(),
            'min_safety_score': self.envelope_data['Safety Score'].min(),
        }


# =============================================================================
# Session State Management
# =============================================================================

def init_vehicle_session_state():
    """Initialize vehicle-related session state."""
    if 'vehicle_manager' not in st.session_state:
        st.session_state.vehicle_manager = None
    
    if 'vehicle_params' not in st.session_state:
        st.session_state.vehicle_params = VehicleParameters()
    
    if 'vehicle_simulated' not in st.session_state:
        st.session_state.vehicle_simulated = False


def get_vehicle_manager() -> Optional[VehicleManager]:
    """Get the current vehicle manager from session state."""
    init_vehicle_session_state()
    return st.session_state.vehicle_manager


def create_vehicle_manager(params: Optional[VehicleParameters] = None) -> VehicleManager:
    """Create a new vehicle manager and store in session state."""
    init_vehicle_session_state()
    
    if params:
        st.session_state.vehicle_params = params
    
    manager = VehicleManager(st.session_state.vehicle_params)
    st.session_state.vehicle_manager = manager
    return manager


def render_vehicle_controls(trajectory_df: Optional[pd.DataFrame] = None) -> Optional[VehicleManager]:
    """
    Render vehicle simulation controls.
    
    Returns the VehicleManager if simulation was triggered or already exists.
    """
    init_vehicle_session_state()
    
    st.markdown("### ✈️ Vehicle Simulation")
    
    # Configuration expander
    with st.expander("⚙️ Vehicle Configuration", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Simulation Parameters**")
            sim_time = st.slider(
                "Simulation Time (s)",
                60, 600, int(st.session_state.vehicle_params.simulation_time_s), 30
            )
            
            st.markdown("**Vehicle Specs**")
            mass = st.number_input("Mass (kg)", 500, 3000, int(st.session_state.vehicle_params.mass_kg), 100)
            num_rotors = st.selectbox("Number of Rotors", [4, 6, 8], index=2)
        
        with col2:
            st.markdown("**Battery Parameters**")
            battery_cap = st.slider(
                "Battery Capacity (kWh)",
                20.0, 100.0, st.session_state.vehicle_params.battery_capacity_kwh, 5.0
            )
            initial_soc = st.slider(
                "Initial SoC (%)",
                50, 100, int(st.session_state.vehicle_params.initial_soc * 100), 5
            ) / 100
        
        # Update params
        st.session_state.vehicle_params = VehicleParameters(
            simulation_time_s=sim_time,
            mass_kg=mass,
            num_rotors=num_rotors,
            battery_capacity_kwh=battery_cap,
            initial_soc=initial_soc
        )
    
    # Simulation button
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        simulate_clicked = st.button(
            "🚀 Run Simulation",
            type="primary",
            use_container_width=True
        )
    
    with col2:
        if st.session_state.vehicle_simulated:
            resim_clicked = st.button("🔄 Re-simulate", use_container_width=True)
        else:
            resim_clicked = False
    
    with col3:
        if st.session_state.vehicle_manager:
            status = st.session_state.vehicle_manager.simulation_status
            status_items = [f"{'✅' if v else '⬜'} {k.title()}" for k, v in status.items()]
            st.markdown(f"**Status:** {' | '.join(status_items)}")
    
    # Handle simulation
    if simulate_clicked or resim_clicked:
        manager = create_vehicle_manager(st.session_state.vehicle_params)
        
        progress_bar = st.progress(0, text="Initializing simulation...")
        status_text = st.empty()
        
        def update_progress(message: str):
            status_text.text(message)
        
        progress_bar.progress(25, text="Running dynamics simulation...")
        manager.run_dynamics_simulation(trajectory_df, update_progress)
        
        progress_bar.progress(50, text="Simulating battery...")
        manager.run_battery_simulation(update_progress)
        
        progress_bar.progress(75, text="Simulating motors...")
        manager.run_motor_simulation(update_progress)
        
        progress_bar.progress(90, text="Analyzing envelope...")
        manager.run_envelope_analysis(update_progress)
        
        progress_bar.progress(100, text="Complete!")
        status_text.success("✅ Vehicle simulation complete!")
        
        st.session_state.vehicle_simulated = True
        st.session_state.vehicle_manager = manager
        
        return manager
    
    return st.session_state.vehicle_manager
