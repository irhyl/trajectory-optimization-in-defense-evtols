"""
Path Planning Algorithms

This module provides graph-based and sampling-based path planning algorithms
for trajectory optimization. All algorithms inherit from a common base class
and work with the unified state/trajectory representations.

Algorithms
==========

**Graph-Based:**
- A*: Multi-objective heuristic search
- Theta*: Any-angle path planning
- Dijkstra: Baseline shortest path

**Sampling-Based:**
- RRT: Rapidly-exploring Random Tree
- RRT*: Asymptotically optimal RRT
- Informed RRT*: Focused sampling in goal region

**Optimization-Based:**
- Direct Collocation: Transcribe to NLP
- Shooting Methods: Forward integration

All algorithms accept:
- Start and goal states
- Cost function (modular, composable)
- Constraint set
- Environment fields (terrain, wind, threat)

And produce:
- Trajectory (or set of Pareto-optimal trajectories)
- Planning statistics (nodes expanded, time, etc.)

Author: Defense eVTOL Research Team
"""

from .base import PathPlanner, PlanningResult, PlanningConfig

# Graph-based methods
from .graph import AStarPlanner, ThetaStarPlanner, DijkstraPlanner

# Sampling-based methods
from .sampling import RRTPlanner, RRTStarPlanner

__all__ = [
    # Base
    "PathPlanner",
    "PlanningResult",
    "PlanningConfig",
    # Graph-based
    "AStarPlanner",
    "ThetaStarPlanner",
    "DijkstraPlanner",
    # Sampling-based
    "RRTPlanner",
    "RRTStarPlanner",
]
