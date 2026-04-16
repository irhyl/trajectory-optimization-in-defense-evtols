"""
evtol.core.state
================
Canonical shared types for cross-layer state representation.

This module is the single source of truth for:

* **FlightPhase** – discrete phase enum used by vehicle_model, config, and
  control layers to communicate what mode the vehicle is in.
* **VehicleState** – the full 6-DoF dynamics state.  Re-exported from
  ``evtol.vehicle.dynamics.state`` so that any layer can do::

      from evtol.core.state import VehicleState, FlightPhase

  without creating a circular import through the vehicle sub-package.

Layer-specific state types
---------------------------
Each sub-package defines its own lightweight state dataclass for
internal use; they are *not* the canonical VehicleState:

* ``evtol.control.sitl_simulator.SITLState``
  Hardware-telemetry struct (ENU, Euler angles, flat scalars).
* ``evtol.control.advanced_modes.advanced_modes.ModeInputState``
  Minimal inputs needed by the advanced flight-mode selector.

These are intentionally separate — converting to/from VehicleState is the
responsibility of the layer boundary (e.g. SITLSimulator.get_state()).
"""

from __future__ import annotations

import importlib.util
import pathlib
from enum import Enum, auto

# Re-export the canonical dynamics-layer VehicleState so other layers can
# import it from here without a direct vehicle.dynamics dependency.
#
# We load via file path (not package import) to break the circular import that
# would otherwise occur:
#   core.state → vehicle.__init__ → vehicle_model → core.state  (cycle!)
# Importing the .py file directly bypasses vehicle/__init__ entirely.
_state_file = pathlib.Path(__file__).parent.parent / "vehicle" / "dynamics" / "state.py"
_spec = importlib.util.spec_from_file_location("evtol.vehicle.dynamics.state", _state_file)
_dynamics_state_mod = importlib.util.module_from_spec(_spec)
import sys as _sys
if "evtol.vehicle.dynamics.state" not in _sys.modules:
    _sys.modules["evtol.vehicle.dynamics.state"] = _dynamics_state_mod
    _spec.loader.exec_module(_dynamics_state_mod)
VehicleState = _sys.modules["evtol.vehicle.dynamics.state"].VehicleState  # noqa: F401
del _state_file, _spec, _dynamics_state_mod, _sys


class FlightPhase(Enum):
    """Discrete flight phases for a tilt-rotor eVTOL.

    Transitions follow the schedule::

        GROUND → HOVER → TRANSITION_TO_CRUISE → CRUISE
               ← TRANSITION_TO_HOVER ← CRUISE
               ← DESCENT ← HOVER
               → GROUND  (landing roll-out)
    """

    GROUND = auto()               # On the ground, rotors idle or off
    HOVER = auto()                # Vertical flight, nacelles at 90°
    TRANSITION_TO_CRUISE = auto() # Nacelles tilting from 90° → 0°
    CRUISE = auto()               # Wing-borne flight, nacelles at 0°
    TRANSITION_TO_HOVER = auto()  # Nacelles tilting from 0° → 90°
    DESCENT = auto()              # Controlled descent before landing


__all__ = ["FlightPhase", "VehicleState"]
