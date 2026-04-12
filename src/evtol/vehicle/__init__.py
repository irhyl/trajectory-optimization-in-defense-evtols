"""
Vehicle Layer Package
This package provides comprehensive vehicle dynamics and energy management
for eVTOL aircraft simulation and optimization.

Main Components:
- VehicleModel: Main vehicle dynamics and simulation
- BatteryModel: Advanced battery state modeling
- MotorModel: Electric motor dynamics
- FlightEnvelope: Constraint checking and validation
- FaultInjector: Fault injection and modeling
- VehicleAPI: REST API for remote access

Example Usage:
    from vehicle_layer import VehicleModel, VehicleConfig, VehicleState, ControlInputs
    
    # Load configuration
    config = VehicleConfig("config/vehicle_config.yaml")
    
    # Create vehicle model
    vehicle = VehicleModel(config)
    
    # Set initial state
    initial_state = VehicleState(
        position=np.array([0.0, 0.0, 100.0]),
        velocity=np.array([0.0, 0.0, 0.0]),
        attitude=np.array([0.0, 0.0, 0.0]),
        angular_velocity=np.array([0.0, 0.0, 0.0]),
        battery_soc=0.8,
        battery_temperature=20.0,
        battery_voltage=400.0,
        rotor_rpm=np.array([1000, 1000, 1000, 1000]),
        control_surface_deflections=np.array([0.0, 0.0, 0.0]),
        time=0.0
    )
    
    # Create control inputs
    controls = ControlInputs(
        main_rotor_rpm=np.array([1000, 1000, 1000, 1000]),
        tail_rotor_rpm=1200,
        lift_fan_rpm=np.array([800, 800]),
        propeller_rpm=np.array([0, 0]),
        elevator_deflection=0.0,
        aileron_deflection=0.0,
        rudder_deflection=0.0,
        throttle=0.7,
        collective=0.5
    )
    
    # Run simulation
    trajectory = vehicle.simulate(initial_state, controls, 0.01, 60.0)
"""

# Core vehicle components
from .dynamics import VehicleModel
from .energy.battery_model import BatteryModel
from .actuators.motor_model import MotorModel
from .constraints.flight_envelope import FlightEnvelope
from .faults.fault_injector import FaultInjector

# Data structures
from .dynamics import VehicleState, ControlInputs

# Utilities
from .utils.config import VehicleConfig
from .utils.data_loader import DataLoader

# Integration
from .integration.rk4_integrator import RK4Integrator

# Serving
try:
    from .serving.api import VehicleAPI
except ImportError:
    # uvicorn not available, API not loaded
    VehicleAPI = None

# Version information
__version__ = "1.0.0"
__author__ = "eVTOL Defense System Team"
__email__ = "team@evtol-defense.com"

# Package metadata
__all__ = [
    # Core components
    "VehicleModel",
    "BatteryModel", 
    "MotorModel",
    "FlightEnvelope",
    "FaultInjector",
    
    # Data structures
    "VehicleState",
    "ControlInputs",
    
    # Utilities
    "VehicleConfig",
    "DataLoader",
    
    # Integration
    "RK4Integrator",
    
    # Serving
    "VehicleAPI",
    
    # Metadata
    "__version__",
    "__author__",
    "__email__",
]