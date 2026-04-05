"""
Core Module

This module provides the fundamental building blocks for trajectory planning:

- **State**: Full vehicle state on SE(3) × ℝ³ × ℝ⁺
- **Pose**: Rigid body pose (position + attitude)
- **Velocity**: Linear and angular velocity
- **Trajectory**: Parameterized path through state space
- **CostFunction**: Objective function framework
- **Constraint**: Constraint definitions and checking

Mathematical Foundation
=======================

The state space is:
    X = SE(3) × ℝ³ × ℝ⁺ = {(p, R, v, E)}

where:
    - p ∈ ℝ³: Position (geodetic or NED)
    - R ∈ SO(3): Attitude (rotation matrix, or Euler/quaternion)
    - v ∈ ℝ³: Velocity vector
    - E ∈ [0, 1]: Energy state (battery SOC)

Author: Defense eVTOL Research Team
"""

from .state import (
    CoordinateFrame,
    AttitudeRepresentation,
    Pose,
    Velocity,
    State,
)

from .trajectory import (
    TrajectoryType,
    TrajectorySegment,
    Trajectory,
)

from .cost import (
    ObjectiveType,
    CostComponent,
    CostFunction,
    TimeCost,
    EnergyCost,
    PathLengthCost,
    ThreatExposureCost,
    SmoothnessCost,
    TerminalCost,
    DetectionCost,
    CompositeCost,
    create_default_cost,
    create_stealth_cost,
)

from .constraints import (
    ConstraintType,
    ViolationSeverity,
    ConstraintViolation,
    Constraint,
    MaxSpeedConstraint,
    MinMaxAltitudeConstraint,
    MaxClimbRateConstraint,
    MaxTurnRateConstraint,
    EnergyReserveConstraint,
    TerrainClearanceConstraint,
    NoFlyZoneConstraint,
    ObstacleAvoidanceConstraint,
    ThreatAvoidanceConstraint,
    MaxThreatExposureConstraint,
    ConstraintSet,
    create_default_constraints,
)

from .energy_evaluator import (
    EnergyMode,
    EnergyProfile,
    EnergyConstraints,
    EnergyEvaluator,
)

from .threat_analyzer import (
    ThreatType,
    ThreatZone,
    ThreatField,
    ThreatAnalyzer,
)

__all__ = [
    # State module
    "CoordinateFrame",
    "AttitudeRepresentation",
    "Pose",
    "Velocity",
    "State",
    # Trajectory module
    "TrajectoryType",
    "TrajectorySegment",
    "Trajectory",
    # Cost module
    "ObjectiveType",
    "CostComponent",
    "CostFunction",
    "TimeCost",
    "EnergyCost",
    "PathLengthCost",
    "ThreatExposureCost",
    "SmoothnessCost",
    "TerminalCost",
    "DetectionCost",
    "CompositeCost",
    "create_default_cost",
    "create_stealth_cost",
    # Constraints module
    "ConstraintType",
    "ViolationSeverity",
    "ConstraintViolation",
    "Constraint",
    "MaxSpeedConstraint",
    "MinMaxAltitudeConstraint",
    "MaxClimbRateConstraint",
    "MaxTurnRateConstraint",
    "EnergyReserveConstraint",
    "TerrainClearanceConstraint",
    "NoFlyZoneConstraint",
    "ObstacleAvoidanceConstraint",
    "ThreatAvoidanceConstraint",
    "MaxThreatExposureConstraint",
    "ConstraintSet",
    "create_default_constraints",
    # Energy module
    "EnergyMode",
    "EnergyProfile",
    "EnergyConstraints",
    "EnergyEvaluator",
    # Threat module
    "ThreatType",
    "ThreatZone",
    "ThreatField",
    "ThreatAnalyzer",
]
