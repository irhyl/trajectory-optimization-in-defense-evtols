"""
Pareto Frontier Computation and Analysis

This module implements Pareto dominance, non-dominated sorting,
and frontier analysis for multi-objective trajectory optimization.

Mathematical Foundation:
------------------------
A solution x dominates solution y (x ≻ y) if:
1. x is no worse than y in all objectives
2. x is strictly better than y in at least one objective

The Pareto frontier (or Pareto set) is the set of all non-dominated solutions.

Key Algorithms:
- Fast Non-Dominated Sorting (O(MN²) where M=objectives, N=population)
- Crowding Distance Assignment (for diversity preservation)
- Reference Point Based Selection (for NSGA-III)

Author: Defense eVTOL Research Team
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


def dominates(f1: np.ndarray, f2: np.ndarray) -> bool:
    """
    Check if f1 Pareto-dominates f2 (all objectives minimized).

    f1 ≻ f2 iff:
        ∀i: f1[i] ≤ f2[i]  AND  ∃j: f1[j] < f2[j]

    Args:
        f1: Objective vector 1
        f2: Objective vector 2

    Returns:
        True if f1 dominates f2
    """
    f1 = np.asarray(f1)
    f2 = np.asarray(f2)

    # f1 must be no worse in all objectives
    all_leq = np.all(f1 <= f2)
    # f1 must be strictly better in at least one
    any_lt = np.any(f1 < f2)

    return all_leq and any_lt


def is_pareto_optimal(
    f: np.ndarray,
    F: np.ndarray,
    strict: bool = True,
) -> bool:
    """
    Check if objective vector f is Pareto optimal within set F.

    Args:
        f: Objective vector to check
        F: Matrix of objective vectors (population)
        strict: If True, require strict non-dominance

    Returns:
        True if f is non-dominated
    """
    f = np.asarray(f)
    F = np.asarray(F)

    for other in F:
        if np.array_equal(f, other):
            continue
        if dominates(other, f):
            return False

    return True


def find_pareto_indices(F: np.ndarray) -> np.ndarray:
    """
    Find indices of non-dominated solutions (Pareto frontier).

    Uses O(N²) pairwise comparison. For larger populations,
    use fast_non_dominated_sort for efficiency.

    Args:
        F: Nxm matrix of objective values (N solutions, m objectives)

    Returns:
        Array of indices of non-dominated solutions
    """
    F = np.asarray(F)
    n = len(F)
    is_dominated = np.zeros(n, dtype=bool)

    for i in range(n):
        if is_dominated[i]:
            continue
        for j in range(n):
            if i == j or is_dominated[j]:
                continue
            if dominates(F[j], F[i]):
                is_dominated[i] = True
                break

    return np.where(~is_dominated)[0]


def fast_non_dominated_sort(F: np.ndarray) -> list[list[int]]:
    """
    Fast non-dominated sorting from NSGA-II.

    Partitions population into fronts F₁, F₂, ..., where:
    - F₁ is the Pareto frontier
    - F₂ is the frontier after removing F₁
    - etc.

    Time Complexity: O(MN²) where M = objectives, N = population

    Args:
        F: Nxm matrix of objective values

    Returns:
        List of fronts, each front is a list of solution indices
    """
    F = np.asarray(F)
    n = len(F)

    if n == 0:
        return []

    # S[i] = set of solutions that i dominates
    S = [[] for _ in range(n)]
    # n[i] = number of solutions that dominate i
    domination_count = np.zeros(n, dtype=int)

    # First front
    fronts = [[]]

    # Compare all pairs
    for i in range(n):
        for j in range(i + 1, n):
            if dominates(F[i], F[j]):
                S[i].append(j)
                domination_count[j] += 1
            elif dominates(F[j], F[i]):
                S[j].append(i)
                domination_count[i] += 1

    # Find first front (non-dominated solutions)
    for i in range(n):
        if domination_count[i] == 0:
            fronts[0].append(i)

    # Generate subsequent fronts
    current_front = 0
    while fronts[current_front]:
        next_front = []
        for i in fronts[current_front]:
            for j in S[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    next_front.append(j)

        current_front += 1
        if next_front:
            fronts.append(next_front)
        else:
            break

    return fronts


def crowding_distance(F: np.ndarray, front: list[int]) -> np.ndarray:
    """
    Compute crowding distance for solutions in a front.

    Crowding distance measures solution density in objective space.
    Higher values indicate more isolated solutions (more diverse).

    Algorithm (per objective):
    1. Sort front by objective value
    2. Boundary solutions get infinite distance
    3. Interior solutions: distance += (neighbor_i+1 - neighbor_i-1) / range

    Args:
        F: Objective value matrix
        front: List of solution indices in the front

    Returns:
        Array of crowding distances for each solution in front
    """
    n = len(front)
    if n <= 2:
        return np.full(n, np.inf)

    distances = np.zeros(n)
    m = F.shape[1]  # Number of objectives

    for obj in range(m):
        # Get objective values for this front
        values = F[front, obj]

        # Sort by this objective
        sorted_indices = np.argsort(values)

        # Boundary points get infinite distance
        distances[sorted_indices[0]] = np.inf
        distances[sorted_indices[-1]] = np.inf

        # Range for normalization
        obj_range = values[sorted_indices[-1]] - values[sorted_indices[0]]
        if obj_range < 1e-10:
            continue

        # Interior points
        for i in range(1, n - 1):
            idx = sorted_indices[i]
            prev_idx = sorted_indices[i - 1]
            next_idx = sorted_indices[i + 1]
            distances[idx] += (values[next_idx] - values[prev_idx]) / obj_range

    return distances


def crowding_distance_assignment(F: np.ndarray, front: list[int]) -> np.ndarray:
    """
    Assign crowding distances to all solutions in a front.

    This is the NSGA-II version that handles edge cases.

    Args:
        F: Objective value matrix
        front: List of solution indices

    Returns:
        Crowding distances for solutions in front order
    """
    return crowding_distance(F, front)


@dataclass
class ParetoSolution:
    """
    A solution on the Pareto frontier.

    Attributes:
        index: Index in original population
        objectives: Objective vector
        decision: Decision variables (e.g., trajectory)
        crowding: Crowding distance
        front: Pareto front rank (0 = best)
    """
    index: int
    objectives: np.ndarray
    decision: object | None = None
    crowding: float = np.inf
    front: int = 0

    def dominates(self, other: ParetoSolution) -> bool:
        """Check if this solution dominates other."""
        return dominates(self.objectives, other.objectives)


@dataclass
class ParetoFrontier:
    """
    Pareto frontier analysis and visualization.

    Maintains the non-dominated set and provides:
    - Adding new solutions
    - Extracting knee points
    - Computing hypervolume
    - Visualization
    """
    solutions: list[ParetoSolution] = field(default_factory=list)
    n_objectives: int = 0
    ideal_point: np.ndarray | None = None  # Best in each objective
    nadir_point: np.ndarray | None = None  # Worst in frontier

    def __len__(self) -> int:
        return len(self.solutions)

    def __iter__(self):
        return iter(self.solutions)

    @classmethod
    def from_population(
        cls,
        F: np.ndarray,
        decisions: list | None = None,
    ) -> ParetoFrontier:
        """
        Create Pareto frontier from population objectives.

        Args:
            F: Nxm objective matrix
            decisions: Optional decision variables for each solution

        Returns:
            ParetoFrontier containing non-dominated solutions
        """
        F = np.asarray(F)
        pareto_indices = find_pareto_indices(F)

        solutions = []
        for idx in pareto_indices:
            decision = decisions[idx] if decisions is not None else None
            sol = ParetoSolution(
                index=int(idx),
                objectives=F[idx].copy(),
                decision=decision,
                front=0,
            )
            solutions.append(sol)

        # Compute crowding distances
        if len(solutions) > 0:
            front_indices = list(range(len(solutions)))
            front_F = np.array([s.objectives for s in solutions])
            distances = crowding_distance(front_F, front_indices)
            for i, dist in enumerate(distances):
                solutions[i].crowding = float(dist)

        frontier = cls(
            solutions=solutions,
            n_objectives=F.shape[1] if len(F) > 0 else 0,
        )

        # Compute ideal and nadir
        if len(solutions) > 0:
            objectives = frontier.get_objectives()
            frontier.ideal_point = np.min(objectives, axis=0)
            frontier.nadir_point = np.max(objectives, axis=0)

        return frontier

    def add_solution(self, objectives: np.ndarray, decision: object = None) -> bool:
        """
        Add a solution if it's non-dominated.

        Args:
            objectives: Objective vector
            decision: Decision variables

        Returns:
            True if solution was added (is non-dominated)
        """
        objectives = np.asarray(objectives)

        # Check if dominated by existing solutions
        for sol in self.solutions:
            if dominates(sol.objectives, objectives):
                return False

        # Remove solutions dominated by new one
        self.solutions = [
            sol for sol in self.solutions
            if not dominates(objectives, sol.objectives)
        ]

        # Add new solution
        new_sol = ParetoSolution(
            index=len(self.solutions),
            objectives=objectives,
            decision=decision,
            front=0,
        )
        self.solutions.append(new_sol)

        # Update ideal/nadir
        self._update_bounds()

        return True

    def _update_bounds(self) -> None:
        """Update ideal and nadir points."""
        if len(self.solutions) == 0:
            self.ideal_point = None
            self.nadir_point = None
            return

        objectives = self.get_objectives()
        self.ideal_point = np.min(objectives, axis=0)
        self.nadir_point = np.max(objectives, axis=0)

    def get_objectives(self) -> np.ndarray:
        """Get objective matrix for frontier solutions."""
        if len(self.solutions) == 0:
            return np.array([])
        return np.array([sol.objectives for sol in self.solutions])

    def get_decisions(self) -> list:
        """Get decision variables for frontier solutions."""
        return [sol.decision for sol in self.solutions]

    def normalize(self, objectives: np.ndarray) -> np.ndarray:
        """
        Normalize objectives to [0, 1] using ideal/nadir.

        Args:
            objectives: Objective vector(s)

        Returns:
            Normalized objectives
        """
        if self.ideal_point is None or self.nadir_point is None:
            return objectives

        range_vec = self.nadir_point - self.ideal_point
        range_vec = np.where(range_vec > 1e-10, range_vec, 1.0)

        return (objectives - self.ideal_point) / range_vec

    def find_knee_point(self) -> ParetoSolution | None:
        """
        Find the knee point (maximum trade-off solution).

        The knee point maximizes the minimum marginal improvement,
        representing the best balance among objectives.

        Uses the geometric knee identification method.

        Returns:
            Knee solution or None if frontier is empty
        """
        if len(self.solutions) < 3:
            return self.solutions[0] if self.solutions else None

        objectives = self.get_objectives()
        normalized = self.normalize(objectives)

        # Distance to the ideal-nadir line
        # For each point, compute orthogonal distance to the line
        # connecting (0,0,...) to (1,1,...)

        # Direction vector of line (normalized)
        n_obj = normalized.shape[1]
        line_dir = np.ones(n_obj) / np.sqrt(n_obj)

        # For each point, project onto line and find perpendicular distance
        distances = np.zeros(len(normalized))
        for i, point in enumerate(normalized):
            # Projection onto line
            proj_length = np.dot(point, line_dir)
            projection = proj_length * line_dir

            # Perpendicular distance
            perp = point - projection
            distances[i] = np.linalg.norm(perp)

        # Knee is the point with maximum distance from the line
        # But we want the point that's "inward" (lower values)
        # So we use a weighted distance
        weighted_dist = distances * (1 - np.mean(normalized, axis=1))

        knee_idx = np.argmax(weighted_dist)
        return self.solutions[knee_idx]

    def find_extreme_solutions(self) -> list[ParetoSolution]:
        """
        Find extreme solutions (best in each objective).

        Returns:
            List of extreme solutions
        """
        if len(self.solutions) == 0:
            return []

        objectives = self.get_objectives()
        extremes = []

        for obj_idx in range(objectives.shape[1]):
            best_idx = np.argmin(objectives[:, obj_idx])
            extremes.append(self.solutions[best_idx])

        return extremes

    def compute_hypervolume(
        self,
        reference_point: np.ndarray | None = None,
    ) -> float:
        """
        Compute hypervolume indicator (S-metric).

        The hypervolume is the measure of objective space dominated
        by the Pareto frontier with respect to a reference point.

        Uses inclusion-exclusion for 2D, Monte Carlo for higher dimensions.

        Args:
            reference_point: Reference point (default: nadir * 1.1)

        Returns:
            Hypervolume indicator value
        """
        if len(self.solutions) == 0:
            return 0.0

        objectives = self.get_objectives()
        n_obj = objectives.shape[1]

        if reference_point is None:
            if self.nadir_point is not None:
                reference_point = self.nadir_point * 1.1
            else:
                reference_point = np.max(objectives, axis=0) * 1.1

        reference_point = np.asarray(reference_point)

        if n_obj == 2:
            return self._hypervolume_2d(objectives, reference_point)
        else:
            return self._hypervolume_monte_carlo(objectives, reference_point)

    def _hypervolume_2d(
        self,
        objectives: np.ndarray,
        ref_point: np.ndarray,
    ) -> float:
        """
        Exact hypervolume for 2 objectives.

        O(N log N) algorithm using sorted sweeping.
        """
        # Filter points that dominate reference
        valid = np.all(objectives < ref_point, axis=1)
        if not np.any(valid):
            return 0.0

        objectives = objectives[valid]

        # Sort by first objective
        sorted_idx = np.argsort(objectives[:, 0])
        sorted_obj = objectives[sorted_idx]

        # Sweep line algorithm
        hv = 0.0
        prev_y = ref_point[1]

        for i in range(len(sorted_obj)):
            x = sorted_obj[i, 0]
            y = sorted_obj[i, 1]

            if i < len(sorted_obj) - 1:
                width = sorted_obj[i + 1, 0] - x
            else:
                width = ref_point[0] - x

            height = prev_y - y
            if height > 0 and width > 0:
                hv += width * height

            prev_y = min(prev_y, y)

        return float(hv)

    def _hypervolume_monte_carlo(
        self,
        objectives: np.ndarray,
        ref_point: np.ndarray,
        n_samples: int = 100000,
    ) -> float:
        """
        Monte Carlo hypervolume estimation for higher dimensions.

        Args:
            objectives: Pareto front objectives
            ref_point: Reference point
            n_samples: Number of MC samples

        Returns:
            Estimated hypervolume
        """
        # Compute ideal point (minimum in each dimension)
        ideal = np.min(objectives, axis=0)

        # Box volume
        box_volume = np.prod(ref_point - ideal)

        if box_volume <= 0:
            return 0.0

        # Sample uniformly in bounding box
        samples = np.random.uniform(
            ideal, ref_point, (n_samples, len(ref_point))
        )

        # Count dominated samples
        dominated = 0
        for sample in samples:
            # Check if dominated by any Pareto point
            for obj in objectives:
                if np.all(obj <= sample):
                    dominated += 1
                    break

        return box_volume * (dominated / n_samples)

    def spacing(self) -> float:
        """
        Compute spacing metric (distribution uniformity).

        Lower spacing indicates more uniform distribution.

        Returns:
            Spacing metric value
        """
        if len(self.solutions) < 2:
            return 0.0

        objectives = self.get_objectives()
        n = len(objectives)

        # Compute minimum distance for each point
        min_distances = np.zeros(n)
        for i in range(n):
            dists = np.linalg.norm(objectives - objectives[i], axis=1)
            dists[i] = np.inf  # Exclude self
            min_distances[i] = np.min(dists)

        mean_dist = np.mean(min_distances)
        variance = np.sum((min_distances - mean_dist) ** 2) / n

        return float(np.sqrt(variance))

    def spread(self) -> float:
        """
        Compute spread metric (extent of frontier).

        Higher spread indicates better coverage of objective space.

        Returns:
            Spread metric value
        """
        if len(self.solutions) < 2:
            return 0.0

        objectives = self.get_objectives()
        ranges = np.max(objectives, axis=0) - np.min(objectives, axis=0)

        return float(np.prod(np.maximum(ranges, 1e-10)) ** (1 / len(ranges)))
