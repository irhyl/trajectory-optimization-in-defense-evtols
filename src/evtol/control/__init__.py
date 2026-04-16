"""
Control layer for eVTOL trajectory tracking.
"""

from .flight_controller import (
    ControlGains,
    PIDController,
    FlightController,
)

# trajectory_generator.py does not exist in this package; TrajectoryGenerator
# functionality lives in guidance/trajectory_tracker.py and cascaded_control.py.
try:
    from .guidance.trajectory_tracker import TrajectoryTracker
except ImportError:
    TrajectoryTracker = None

__all__ = [
    "ControlGains",
    "PIDController",
    "FlightController",
    "TrajectoryTracker",
]



