"""
Propulsion Subpackage - Tiltrotor Propulsion System

This subpackage contains comprehensive propulsion models for
tiltrotor eVTOL aircraft:

Motor Model (motor_model.py):
- PMSMMotor: Permanent Magnet Synchronous Motor with dq-frame control
- MotorState: Motor operating state (torque, power, efficiency, thermal)
- Includes field weakening, efficiency mapping, thermal dynamics

Rotor Model (rotor_model.py):
- BEMTRotor: Blade Element Momentum Theory aerodynamic model
- RotorState: Rotor operating state (thrust, torque, power, inflow)
- Supports hover, forward flight, axial climb/descent, ground effect

Nacelle Dynamics (nacelle.py):
- NacelleDynamics: Nacelle tilting mechanism with rate limiting
- NacelleState: Nacelle angle, rate, actuator state
- TransitionController: Coordinated transition management

Power System (power_system.py):
- PowerDistribution: DC bus and load management
- PowerSystem: Integrated power system with inverters
- InverterModel: DC-AC conversion with losses
- Load shedding and fault protection
"""

from .motor_model import PMSMMotor, MotorState
from .rotor_model import BEMTRotor, RotorState
from .nacelle import NacelleDynamics, NacelleState, TransitionController, TiltSchedule
from .power_system import (
    PowerDistribution,
    PowerSystem,
    PowerSystemState,
    InverterModel,
    PowerLoad,
    BusState,
    LoadPriority,
)

__all__ = [
    # Motor
    "PMSMMotor",
    "MotorState",
    # Rotor
    "BEMTRotor",
    "RotorState",
    # Nacelle
    "NacelleDynamics",
    "NacelleState",
    "TransitionController",
    "TiltSchedule",
    # Power
    "PowerDistribution",
    "PowerSystem",
    "PowerSystemState",
    "InverterModel",
    "PowerLoad",
    "BusState",
    "LoadPriority",
]
