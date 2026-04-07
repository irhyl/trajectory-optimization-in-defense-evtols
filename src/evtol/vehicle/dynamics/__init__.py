"""
6-DoF Vehicle Dynamics Module

This module implements the complete 6-DoF rigid body dynamics for eVTOL aircraft,
including translational and rotational motion, force and moment calculations,
and kinematic transformations.
"""

from .vehicle_model import VehicleModel

__all__ = [
    "VehicleModel",
]

