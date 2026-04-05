"""
Aerodynamics Subpackage - Tiltrotor Aerodynamic Models

This subpackage contains aerodynamic models for tiltrotor eVTOL:

Wing Model (wing_model.py):
- WingAerodynamics: Finite wing with flaps and ailerons
- Lift/drag/moment computation
- Ground effect, dynamic stall

Fuselage Model (fuselage_model.py):
- FuselageAerodynamics: Body drag and side forces
- Nacelle interference

Transition Blending (transition.py):
- TransitionAerodynamics: Blends rotor/wing lift
- Conversion corridor management
"""

from .wing_model import WingAerodynamics, WingState
from .fuselage_model import FuselageAerodynamics, FuselageState
from .transition import TransitionAerodynamics

__all__ = [
    "WingAerodynamics",
    "WingState",
    "FuselageAerodynamics",
    "FuselageState",
    "TransitionAerodynamics",
]
