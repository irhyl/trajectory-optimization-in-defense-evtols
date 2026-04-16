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

# Core vehicle components — only import from modules that actually exist
from .vehicle_model import TiltrotorVehicle, TiltrotorVehicle as VehicleModel
from .vehicle_model import ControlInputs, VehicleOutput, TiltrotorConfig
try:
    from .energy.battery_model import BatteryModel
except ImportError:
    BatteryModel = None

try:
    from .propulsion.motor_model import MotorModel
except ImportError:
    MotorModel = None

# Data structures
from .dynamics import VehicleState

# Optional modules (may not exist in all deployment configurations)
try:
    from .config import VehicleConfig
except ImportError:
    VehicleConfig = None

# Version information
__version__ = "1.0.0"
__author__ = "eVTOL Defense System Team"
__email__ = "team@evtol-defense.com"

__all__ = [
    # Core vehicle
    "TiltrotorVehicle",
    "VehicleModel",          # alias for TiltrotorVehicle
    "TiltrotorConfig",
    "ControlInputs",
    "VehicleOutput",
    # Energy
    "BatteryModel",
    # Propulsion
    "MotorModel",
    # Dynamics state
    "VehicleState",
    # Config
    "VehicleConfig",
    # Metadata
    "__version__",
    "__author__",
    "__email__",
]