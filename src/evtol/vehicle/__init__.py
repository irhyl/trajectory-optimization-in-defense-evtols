"""
Vehicle Layer - Defense eVTOL Tiltrotor Dynamics

This package provides comprehensive 6-DoF vehicle dynamics modeling for
a defense-oriented electric tiltrotor aircraft. The model captures:

1. **Rigid Body Dynamics**: Full 6-DoF equations of motion with quaternion
   attitude representation for singularity-free rotation.

2. **Tiltrotor Propulsion**: Nacelle tilt dynamics, BEMT-based rotor
   aerodynamics, PMSM motor models, and power distribution.

3. **Aerodynamics**: Wing lift/drag, fuselage contributions, transition
   flight modeling, and ground effect.

4. **Energy Systems**: Electrochemical battery model with thermal dynamics,
   power budget management, and reserve calculations.

5. **Signature Models**: Radar cross-section (RCS), infrared (IR), and
   acoustic signature prediction for survivability analysis.

Mathematical Framework
======================

State Vector (13 states + auxiliaries):
    x = [p_N, p_E, p_D,           # Position (NED)
         u, v, w,                  # Body velocity
         q0, q1, q2, q3,           # Quaternion attitude
         p, q, r]                  # Angular rates

Control Vector:
    u = [δ_nacelle,               # Nacelle tilt angle
         Ω_L, Ω_R,                # Left/right rotor speeds
         θ_col_L, θ_col_R,        # Collective pitch
         δ_e, δ_a, δ_r]           # Control surfaces

Equations of Motion:
    ṗ = R(q) · V_B                           (Position kinematics)
    m(V̇_B + ω × V_B) = F_B                   (Translational dynamics)
    q̇ = ½ q ⊗ ω                              (Attitude kinematics)
    I·ω̇ + ω × (I·ω) = M_B                    (Rotational dynamics)

Reference Frames:
    - NED: North-East-Down (inertial)
    - Body: Forward-Right-Down (aircraft)
    - Nacelle: Rotated by nacelle angle from body
    - Rotor: Aligned with rotor shaft

Author: Defense eVTOL Research Team
Version: 1.0.0
"""

from .config import (
    VehicleConfig,
    TiltrotorConfig,
    PropulsionConfig,
    AerodynamicsConfig,
    BatteryConfig,
    SignatureConfig,
)

# Standalone vehicle model (no dependencies)
from .vehicle_model import (
    TiltrotorVehicle,
    ControlInputs,
    VehicleOutput,
    VehicleState,
    FlightPhase,
)


# Optional: Import subpackage models if available
try:
    from .dynamics.rigid_body import RigidBodyDynamics
    from .dynamics.integrator import RK4Integrator, RK45AdaptiveIntegrator
except ImportError:
    pass
try:
    from .propulsion.motor_model import PMSMMotor, MotorState
    from .propulsion.rotor_model import BEMTRotor, RotorState
    from .propulsion.nacelle import NacelleDynamics, NacelleState
except ImportError:
    pass

try:
    from .signatures.rcs import RCSModel
    from .signatures.infrared import IRModel
    from .signatures.acoustic import AcousticModel
except ImportError:
    pass

__version__ = "1.0.0"
__author__ = "Defense eVTOL Research Team"

__all__ = [
    # Main standalone model
    "TiltrotorVehicle",
    "TiltrotorConfig",
    "ControlInputs",
    "VehicleOutput",
    "VehicleState",
    "FlightPhase",
]
