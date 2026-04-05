"""
Mission Planning Module

This module provides mission-level trajectory planning:
- Multi-leg mission planning
- Waypoint sequencing
- Contingency handling
- Mission monitoring and replanning

Mission Architecture:
--------------------
A mission consists of:
1. Mission objectives (targets, waypoints, timing)
2. Mission phases (takeoff, cruise, loiter, landing)
3. Constraints (no-fly zones, fuel limits, time windows)
4. Contingency plans (alternate landing sites, abort procedures)

Author: Defense eVTOL Research Team
"""

from .planner import (
    MissionPhase,
    MissionLeg,
    MissionObjective,
    MissionPlan,
    MissionPlanner,
)
from .contingency import (
    ContingencyType,
    ContingencyPlan,
    ContingencyTrigger,
    ContingencyManager,
)
from .mission_planner import (
    MissionPriority,
    MissionObjective as MissionObjectiveNew,
    MissionPlan as MissionPlanNew,
    MissionLeg as MissionLegNew,
    MissionPlanner as MissionPlannerNew,
)

__all__ = [
    # Planning
    'MissionPhase',
    'MissionLeg',
    'MissionObjective',
    'MissionPlan',
    'MissionPlanner',
    # Contingency
    'ContingencyType',
    'ContingencyPlan',
    'ContingencyTrigger',
    'ContingencyManager',
    # Mission Planning Engine
    'MissionPriority',
    'MissionObjectiveNew',
    'MissionPlanNew',
    'MissionLegNew',
    'MissionPlannerNew',
]
