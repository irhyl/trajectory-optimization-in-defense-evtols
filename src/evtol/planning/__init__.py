"""
Planning Layer - Multi-Objective Trajectory Optimization for Defense eVTOLs

This package implements a hierarchical planning architecture for autonomous
trajectory generation in contested environments. The design follows a
mathematically rigorous approach suitable for PhD-level research.

Architecture Overview
=====================

The planning layer is organized into the following subpackages:

1. **core/** - Fundamental abstractions
   - State space definitions (SE(3) manifold)
   - Trajectory representations (polynomial, B-spline)
   - Cost function framework (modular, composable)
   - Constraint definitions (equality, inequality, chance)

2. **algorithms/** - Path planning algorithms
   - Graph-based: A*, Theta*, Dijkstra
   - Sampling-based: RRT, RRT*, Informed-RRT*
   - Optimization-based: Direct collocation, shooting

3. **optimization/** - Multi-objective optimization
   - Objective function definitions
   - NSGA-III many-objective optimizer
   - Pareto frontier computation
   - Weighted-sum scalarization

4. **trajectory/** - Trajectory post-processing
   - B-spline smoothing
   - Time-optimal velocity profiles
   - Kinodynamic feasibility checking

5. **robust/** - Robust planning under uncertainty
   - Uncertainty propagation models
   - Chance-constrained optimization
   - Distributionally robust optimization

6. **mission/** - Mission-level orchestration
   - Multi-leg mission planning
   - Contingency and replanning

7. External environment and service interfaces are provided by
   the top-level core and perception layers.

Mathematical Foundation
=======================

State Space:
    x ∈ SE(3) × ℝ³ × ℝ⁺ = {(p, R, v, E)}
    where p ∈ ℝ³ is position, R ∈ SO(3) is attitude,
    v ∈ ℝ³ is velocity, E ∈ ℝ⁺ is energy state

Trajectory:
    τ: [0, T] → SE(3) × ℝ³
    Parameterized as B-spline: τ(t) = Σᵢ cᵢ Bᵢ,ₖ(t)

Optimization Problem:
    min     J(τ) = [J_time(τ), J_energy(τ), J_risk(τ), J_exposure(τ)]ᵀ
    τ ∈ T

    s.t.    τ(0) = x₀, τ(T) = xf                    (boundary)
            ḣ(τ(t)) ≥ h_terrain(τ(t)) + ε           (terrain clearance)
            ||τ̈(t)|| ≤ a_max                        (dynamics)
            P[g(τ, ξ) ≤ 0] ≥ 1 - α                  (chance constraints)

Reference
=========
For theoretical background, see:
- LaValle, S.M. "Planning Algorithms" (2006)
- Boyd & Vandenberghe "Convex Optimization" (2004)
- Deb, K. "Multi-Objective Optimization using Evolutionary Algorithms" (2001)

Author: Defense eVTOL Research Team
Version: 2.0.0
Date: January 2026
"""

__version__ = "2.0.0"
__author__ = "Defense eVTOL Research Team"

# Core exports
from .core.state import State, Pose, Velocity, CoordinateFrame
from .core.trajectory import Trajectory, TrajectorySegment, TrajectoryType
from .core.cost import CostFunction, CompositeCost
from .core.constraints import Constraint, ConstraintSet

# Algorithm exports
from .algorithms.base import PathPlanner, PlanningResult, PlanningConfig
from .algorithms.graph import AStarPlanner, ThetaStarPlanner, DijkstraPlanner
from .algorithms.sampling import RRTPlanner, RRTStarPlanner

# Optimization exports
from .optimization.objectives import (
    ObjectiveFunction,
    ObjectiveSet,
    TimeObjective,
    EnergyObjective,
    ThreatExposureObjective,
    DetectionObjective,
    create_defense_objectives,
)
from .optimization.nsga3 import NSGA3Optimizer, NSGA3Config
from .optimization.pareto import ParetoFrontier, ParetoSolution

# Trajectory processing exports
from .trajectory.smoothing import smooth_trajectory, SmoothingResult
from .trajectory.velocity_profile import create_velocity_profile, ProfileConstraints
from .trajectory.feasibility import check_trajectory_feasibility, FeasibilityResult

# Robust planning exports
from .robust.uncertainty import UncertaintyModel, GaussianUncertainty, UncertaintyPropagation
from .robust.chance_constraints import ChanceConstraint, RobustTrajectoryOptimizer

# Mission exports
from .mission.planner import MissionPlanner, MissionPlan, MissionLeg
from .mission.contingency import ContingencyManager, ContingencyType, ContingencyPlan

# Perception-Planning integration exports
from .perception_integration import (
    PerceptionState,
    PlanningConstraintSet,
    PerceptionDataLoader,
    PerceptionToPlanningMapper,
    PerceptionPlanningBridge,
    verify_perception_planning_integration,
)

# Phase 2G: Autonomy Stack exports
from .autonomy_stack import (
    MissionState,
    ReplanRequest,
    ReplannedTrajectory,
    ReplanDecision,
    AutonomousMissionReplanner,
    ContingencyManager as autonomy_ContingencyManager,
    ContingencyType as autonomy_ContingencyType,
    ContingencyLevel,
    DecisionLogic,
    AdaptiveController,
)
from .autonomy_orchestrator import (
    AutonomyOrchestrator,
    AutonomyOutputCommand,
)

__all__ = [
    # Core
    "State",
    "Pose",
    "Velocity",
    "CoordinateFrame",
    "Trajectory",
    "TrajectorySegment",
    "TrajectoryType",
    "CostFunction",
    "CompositeCost",
    "Constraint",
    "ConstraintSet",
    # Algorithms
    "PathPlanner",
    "PlanningResult",
    "PlanningConfig",
    "AStarPlanner",
    "ThetaStarPlanner",
    "DijkstraPlanner",
    "RRTPlanner",
    "RRTStarPlanner",
    # Optimization
    "ObjectiveFunction",
    "ObjectiveSet",
    "TimeObjective",
    "EnergyObjective",
    "ThreatExposureObjective",
    "DetectionObjective",
    "create_defense_objectives",
    "NSGA3Optimizer",
    "NSGA3Config",
    "ParetoFrontier",
    "ParetoSolution",
    # Trajectory
    "smooth_trajectory",
    "SmoothingResult",
    "create_velocity_profile",
    "ProfileConstraints",
    "check_trajectory_feasibility",
    "FeasibilityResult",
    # Robust
    "UncertaintyModel",
    "GaussianUncertainty",
    "UncertaintyPropagation",
    "ChanceConstraint",
    "RobustTrajectoryOptimizer",
    # Mission
    "MissionPlanner",
    "MissionPlan",
    "MissionLeg",
    "ContingencyManager",
    "ContingencyType",
    "ContingencyPlan",
    # Perception-Planning Integration
    "PerceptionState",
    "PlanningConstraintSet",
    "PerceptionDataLoader",
    "PerceptionToPlanningMapper",
    "PerceptionPlanningBridge",
    "verify_perception_planning_integration",
    # Phase 2G: Autonomy Stack
    "MissionState",
    "ReplanRequest",
    "ReplannedTrajectory",
    "ReplanDecision",
    "AutonomousMissionReplanner",
    "autonomy_ContingencyManager",
    "autonomy_ContingencyType",
    "ContingencyLevel",
    "DecisionLogic",
    "AdaptiveController",
    "AutonomyOrchestrator",
    "AutonomyOutputCommand",
]
