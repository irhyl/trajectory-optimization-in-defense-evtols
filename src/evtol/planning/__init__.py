"""
Trajectory Planning Layer

Core module for route planning, optimization, and mission planning.
Provides abstract interfaces and concrete implementations for eVTOL path planning.
"""

from .base import (
    Waypoint,
    RoutePlan,
    RoutePlanner,
    EnergyOptimizer,
    RiskManager,
    Optimizer,
    MissionPlanner,
)
from .config import PlanningConfig, setup_planning_layer
from .routing.planner import AStarPlanner
from .routing.graph_router import GraphRoutePlanner
from .energy.optimizer import EnergyOptimizer as EnergyOptimizerImpl
from .risk.assessment import RiskManager as RiskManagerImpl
from .optimization.pareto import ParetoFrontier
from .mission.planning import MissionPlanner as MissionPlannerImpl

__all__ = [
    # Core abstractions
    "Waypoint",
    "RoutePlan",
    "RoutePlanner",
    "EnergyOptimizer",
    "RiskManager",
    "Optimizer",
    "MissionPlanner",
    # Configuration
    "PlanningConfig",
    "setup_planning_layer",
    # Implementations
    "AStarPlanner",
    "GraphRoutePlanner",
    "EnergyOptimizerImpl",
    "RiskManagerImpl",
    "ParetoFrontier",
    "MissionPlannerImpl",
]


