"""
Trajectory Processing Module

This module provides trajectory post-processing capabilities:
- B-spline smoothing (Phase 2D)
- Time-optimal velocity profiles (Phase 2D)
- Kinodynamic feasibility validation
- Contingency management (Phase 2D)
- Execution engine (Phase 2D)

Author: Defense eVTOL Research Team
"""

from .smoothing import (
    TrajectorySmoothing,
    BSplineSmoother,
    MovingAverageSmoother,
    SavitzkyGolaySmoother,
)
from .velocity_profile import (
    VelocityProfile,
    TrapezoidalProfile,
    SCurveProfile,
    TimeOptimalProfile,
)
from .feasibility import (
    FeasibilityChecker,
    KinodynamicChecker,
    ConstraintViolation,
)
from .smooth_trajectory_generator import (
    SmoothingMethod,
    AccelerationProfile,
    SmoothingConstraints,
    SmoothedTrajectory,
    TrajectorySmootherEngine,
    TrajectoryReferenceGenerator,
)
from .contingency_management import (
    ContingencyLevel,
    AnomalyType,
    ResponseAction,
    AnomalyDetection,
    ContingencyEvent,
    AnomalyDetector,
    ContingencyTrigger,
    ContingencyManager,
)
from .execution_engine import (
    ExecutionPhase,
    ExecutionMetrics,
    TrajectoryExecutionEngine,
    ExecutionValidator,
)
from .differential_flatness_analyzer import (
    ControlMode,
    ThrustCommand,
    TrajectoryDerivatives,
    ControlGains,
    FlatnessAnalysisResult,
    DifferentialFlatnessAnalyzer,
    OptimalControlSynthesizer,
)
from .learning_adaptation import (
    MissionOutcome,
    MissionMetrics,
    AnomalyHistoryPoint,
    LearnedParameters,
    MissionRecorder,
    ROCAnalyzer,
    ParameterLearner,
)
from .online_replanning_interface import (
    ReplanTrigger,
    ReplanRequest,
    ReplanResult,
    BlendedTrajectory,
    FeasibilityChecker,
    OnlineReplanner,
)

__all__ = [
    # Smoothing (legacy)
    'TrajectorySmoothing',
    'BSplineSmoother',
    'MovingAverageSmoother',
    'SavitzkyGolaySmoother',
    # Velocity (legacy)
    'VelocityProfile',
    'TrapezoidalProfile',
    'SCurveProfile',
    'TimeOptimalProfile',
    # Feasibility (legacy)
    'FeasibilityChecker',
    'KinodynamicChecker',
    'ConstraintViolation',
    # Phase 2D: Smooth Trajectory Generation
    'SmoothingMethod',
    'AccelerationProfile',
    'SmoothingConstraints',
    'SmoothedTrajectory',
    'TrajectorySmootherEngine',
    'TrajectoryReferenceGenerator',
    # Phase 2D: Contingency Management
    'ContingencyLevel',
    'AnomalyType',
    'ResponseAction',
    'AnomalyDetection',
    'ContingencyEvent',
    'AnomalyDetector',
    'ContingencyTrigger',
    'ContingencyManager',
    # Phase 2D: Execution Engine
    'ExecutionPhase',
    'ExecutionMetrics',
    'TrajectoryExecutionEngine',
    'ExecutionValidator',
    # Phase 2E: Differential Flatness Control
    'ControlMode',
    'ThrustCommand',
    'TrajectoryDerivatives',
    'ControlGains',
    'FlatnessAnalysisResult',
    'DifferentialFlatnessAnalyzer',
    'OptimalControlSynthesizer',
    # Phase 2E: Learning-Based Adaptation
    'MissionOutcome',
    'MissionMetrics',
    'AnomalyHistoryPoint',
    'LearnedParameters',
    'MissionRecorder',
    'ROCAnalyzer',
    'ParameterLearner',
    # Phase 2E: Online Replanning
    'ReplanTrigger',
    'ReplanRequest',
    'ReplanResult',
    'BlendedTrajectory',
    'OnlineReplanner',
]
