"""
evtol.core
==========
Canonical cross-layer types shared across planning, vehicle, control,
and perception sub-packages.

Exports
-------
VehicleState  – full 6-DoF dynamics state (from vehicle.dynamics.state)
FlightPhase   – enum of discrete flight phases
"""

from .state import FlightPhase, VehicleState

__all__ = ["FlightPhase", "VehicleState"]
