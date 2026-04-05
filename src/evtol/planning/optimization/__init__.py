"""
Multi-Objective Optimization Module

This module provides multi-objective trajectory optimization using
evolutionary algorithms, specifically NSGA-III for many-objective problems.

Multi-Objective Formulation
===========================

Defense eVTOL trajectory planning has multiple competing objectives:

    min F(τ) = [f₁(τ), f₂(τ), ..., fₘ(τ)]

where:
- f₁: Mission time
- f₂: Energy consumption
- f₃: Threat exposure (survivability)
- f₄: Radar detection probability
- f₅: Path smoothness

These objectives are often conflicting:
- Fast paths may have higher threat exposure
- Low-threat paths may consume more energy
- Smooth paths may be longer

Solution: Find the Pareto frontier of non-dominated solutions.

A solution x* is Pareto-optimal if there exists no x such that:
    fᵢ(x) ≤ fᵢ(x*) ∀i and fⱼ(x) < fⱼ(x*) for some j

NSGA-III
========

NSGA-III (Non-dominated Sorting Genetic Algorithm III) is designed for
many-objective optimization (4+ objectives). Key features:

1. Non-dominated sorting: Rank solutions by dominance
2. Reference point association: Use structured reference points on
   the normalized objective hyperplane
3. Niche preservation: Maintain diversity via reference point proximity

This produces a well-distributed Pareto frontier that decision-makers
can use to select trajectories based on mission priorities.

Author: Defense eVTOL Research Team
"""

from .objectives import (
    ObjectiveFunction,
    TimeObjective,
    EnergyObjective,
    ThreatExposureObjective,
    DetectionObjective,
    SmoothnessObjective,
    PathLengthObjective,
    ObjectiveSet,
    create_defense_objectives,
)

from .pareto import (
    dominates,
    is_pareto_optimal,
    find_pareto_indices,
    fast_non_dominated_sort,
    ParetoFrontier,
    ParetoSolution,
    crowding_distance,
)

from .nsga3 import (
    NSGA3Optimizer,
    NSGA3Config,
    Individual,
    Population,
    OptimizationResult,
)

__all__ = [
    # Objectives
    "ObjectiveFunction",
    "TimeObjective",
    "EnergyObjective",
    "ThreatExposureObjective",
    "DetectionObjective",
    "SmoothnessObjective",
    "PathLengthObjective",
    "ObjectiveSet",
    "create_defense_objectives",
    # Pareto
    "dominates",
    "is_pareto_optimal",
    "find_pareto_frontier",
    "ParetoFrontier",
    "crowding_distance",
    # NSGA-III
    "NSGA3Optimizer",
    "NSGA3Config",
    "Individual",
    "Population",
    "OptimizationResult",
]
