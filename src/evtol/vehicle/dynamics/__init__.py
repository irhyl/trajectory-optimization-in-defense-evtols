"""
6-DoF Vehicle Dynamics Module

This module implements the complete 6-DoF rigid body dynamics for eVTOL aircraft,
including translational and rotational motion, force and moment calculations,
and kinematic transformations.
"""

from .state import VehicleState
from .rigid_body import RigidBodyDynamics

__all__ = [
    "VehicleState",
    "RigidBodyDynamics",
]

