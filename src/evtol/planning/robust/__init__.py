"""
Robust Planning Module

This module provides robust planning capabilities under uncertainty:
- Uncertainty modeling and propagation
- Chance-constrained optimization
- Risk-aware path planning

These methods are essential for defense applications where:
1. Wind and atmospheric conditions are uncertain
2. Threat locations/capabilities may be imprecise
3. Sensor data has measurement noise
4. Mission success must be guaranteed with high probability

Author: Defense eVTOL Research Team
"""

from .uncertainty import (
    UncertaintyModel,
    GaussianUncertainty,
    UniformUncertainty,
    WindUncertainty,
    ThreatUncertainty,
    UncertaintyPropagation,
)
from .chance_constraints import (
    ChanceConstraint,
    ProbabilisticSafetyConstraint,
    RobustTrajectoryOptimizer,
    ChanceConstrainedPlanner,
)

__all__ = [
    # Uncertainty
    'UncertaintyModel',
    'GaussianUncertainty',
    'UniformUncertainty',
    'WindUncertainty',
    'ThreatUncertainty',
    'UncertaintyPropagation',
    # Chance constraints
    'ChanceConstraint',
    'ProbabilisticSafetyConstraint',
    'RobustTrajectoryOptimizer',
    'ChanceConstrainedPlanner',
]
