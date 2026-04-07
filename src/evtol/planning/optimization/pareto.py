"""
Multi-Objective Pareto Optimization

Implements Pareto frontier computation for multi-objective route planning.
Provides k-best diverse routes with different trade-offs between objectives.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class Solution:
    """A solution in the multi-objective space"""
    route_id: int
    objectives: Dict[str, float]  # {objective_name: value}
    metadata: Dict = None
    
    def dominates(self, other: 'Solution') -> bool:
        """Check if this solution Pareto-dominates another."""
        better_in_any = False
        
        for obj_name in self.objectives:
            if obj_name not in other.objectives:
                continue
            
            # Assuming minimization for all objectives
            if self.objectives[obj_name] > other.objectives[obj_name]:
                return False  # Worse in this objective
            elif self.objectives[obj_name] < other.objectives[obj_name]:
                better_in_any = True
        
        return better_in_any


class ParetoFrontier:
    """
    Pareto frontier computation and management.
    
    Finds non-dominated solutions in multi-objective space.
    """
    
    def __init__(self):
        """Initialize Pareto frontier."""
        self.solutions: List[Solution] = []
        self.pareto_front: List[Solution] = []
    
    def add_solution(self, solution: Solution):
        """Add a solution and update Pareto front."""
        self.solutions.append(solution)
        self._update_pareto_front(solution)
    
    def add_solutions(self, solutions: List[Solution]):
        """Add multiple solutions and recompute Pareto front."""
        self.solutions.extend(solutions)
        self._recompute_pareto_front()
    
    def _update_pareto_front(self, new_solution: Solution):
        """Incrementally update Pareto front with new solution."""
        # Check if new solution is dominated by any existing Pareto solution
        dominated = False
        to_remove = []
        
        for existing in self.pareto_front:
            if existing.dominates(new_solution):
                dominated = True
                break
            elif new_solution.dominates(existing):
                to_remove.append(existing)
        
        # Remove dominated solutions
        for sol in to_remove:
            self.pareto_front.remove(sol)
        
        # Add new solution if not dominated
        if not dominated:
            self.pareto_front.append(new_solution)
    
    def _recompute_pareto_front(self):
        """Recompute entire Pareto front from scratch."""
        self.pareto_front = []
        
        for solution in self.solutions:
            is_dominated = False
            
            for other in self.solutions:
                if other is solution:
                    continue
                if other.dominates(solution):
                    is_dominated = True
                    break
            
            if not is_dominated:
                self.pareto_front.append(solution)
    
    def get_pareto_front(self) -> List[Solution]:
        """Get current Pareto front."""
        return self.pareto_front.copy()
    
    def get_knee_point(self) -> Optional[Solution]:
        """
        Find knee point in Pareto front.
        
        The knee point represents best trade-off between objectives.
        """
        if len(self.pareto_front) < 2:
            return self.pareto_front[0] if self.pareto_front else None
        
        # Normalize objectives
        obj_names = list(self.pareto_front[0].objectives.keys())
        
        # Get ranges
        ranges = {}
        for obj_name in obj_names:
            values = [s.objectives[obj_name] for s in self.pareto_front]
            ranges[obj_name] = (min(values), max(values))
        
        # Normalize and compute distance from ideal point
        best_distance = float('inf')
        knee_solution = None
        
        for solution in self.pareto_front:
            # Normalize objectives
            normalized = []
            for obj_name in obj_names:
                min_val, max_val = ranges[obj_name]
                if max_val > min_val:
                    norm_val = (solution.objectives[obj_name] - min_val) / (max_val - min_val)
                else:
                    norm_val = 0.5
                normalized.append(norm_val)
            
            # Distance from ideal point (0, 0, ..., 0)
            distance = np.sqrt(sum(v**2 for v in normalized))
            
            if distance < best_distance:
                best_distance = distance
                knee_solution = solution
        
        return knee_solution
    
    def get_extreme_points(self) -> Dict[str, Solution]:
        """
        Get extreme points in each objective.
        
        Returns:
            Dictionary mapping objective name to best solution for that objective
        """
        if not self.pareto_front:
            return {}
        
        extremes = {}
        obj_names = list(self.pareto_front[0].objectives.keys())
        
        for obj_name in obj_names:
            best_solution = min(
                self.pareto_front,
                key=lambda s: s.objectives[obj_name]
            )
            extremes[obj_name] = best_solution
        
        return extremes


class DiverseRouteSelector:
    """
    Select k diverse routes from Pareto front.
    
    Uses diversity metrics to ensure selected routes are sufficiently different.
    """
    
    def __init__(self, diversity_threshold: float = 0.3):
        """
        Initialize diverse route selector.
        
        Args:
            diversity_threshold: Minimum diversity score between routes (0-1)
        """
        self.diversity_threshold = diversity_threshold
    
    def select_diverse_routes(
        self,
        solutions: List[Solution],
        k: int,
        diversity_metric: str = "objective_space"
    ) -> List[Solution]:
        """
        Select k diverse solutions.
        
        Args:
            solutions: List of candidate solutions
            k: Number of solutions to select
            diversity_metric: Metric for diversity ("objective_space", "spread")
            
        Returns:
            List of k diverse solutions
        """
        if len(solutions) <= k:
            return solutions
        
        selected = []
        remaining = solutions.copy()
        
        # Start with knee point or first solution
        if solutions:
            pf = ParetoFrontier()
            pf.add_solutions(solutions)
            knee = pf.get_knee_point()
            if knee:
                selected.append(knee)
                remaining.remove(knee)
        
        # Greedily select most diverse solutions
        while len(selected) < k and remaining:
            if diversity_metric == "objective_space":
                next_solution = self._select_most_diverse_objective_space(
                    selected, remaining
                )
            else:
                next_solution = self._select_most_diverse_spread(
                    selected, remaining
                )
            
            selected.append(next_solution)
            remaining.remove(next_solution)
        
        return selected
    
    def _select_most_diverse_objective_space(
        self,
        selected: List[Solution],
        candidates: List[Solution]
    ) -> Solution:
        """Select solution most diverse in objective space."""
        if not selected:
            return candidates[0]
        
        best_diversity = -1
        best_candidate = None
        
        for candidate in candidates:
            # Compute minimum distance to any selected solution
            min_distance = float('inf')
            
            for selected_sol in selected:
                distance = self._objective_distance(candidate, selected_sol)
                min_distance = min(min_distance, distance)
            
            if min_distance > best_diversity:
                best_diversity = min_distance
                best_candidate = candidate
        
        return best_candidate
    
    def _select_most_diverse_spread(
        self,
        selected: List[Solution],
        candidates: List[Solution]
    ) -> Solution:
        """Select solution that maximizes spread."""
        # Simple version: maximize distance to centroid of selected
        if not selected:
            return candidates[0]
        
        # Compute centroid of selected solutions
        obj_names = list(selected[0].objectives.keys())
        centroid = {
            obj: np.mean([s.objectives[obj] for s in selected])
            for obj in obj_names
        }
        
        # Find candidate farthest from centroid
        best_distance = -1
        best_candidate = None
        
        for candidate in candidates:
            distance = 0.0
            for obj in obj_names:
                distance += (candidate.objectives[obj] - centroid[obj]) ** 2
            distance = np.sqrt(distance)
            
            if distance > best_distance:
                best_distance = distance
                best_candidate = candidate
        
        return best_candidate
    
    def _objective_distance(self, sol1: Solution, sol2: Solution) -> float:
        """Compute Euclidean distance in objective space."""
        distance = 0.0
        
        for obj_name in sol1.objectives:
            if obj_name in sol2.objectives:
                diff = sol1.objectives[obj_name] - sol2.objectives[obj_name]
                distance += diff ** 2
        
        return np.sqrt(distance)


def weighted_sum_scalarization(
    objectives: Dict[str, float],
    weights: Dict[str, float]
) -> float:
    """
    Scalarize multiple objectives using weighted sum.
    
    Args:
        objectives: Dictionary of objective values
        weights: Dictionary of weights (should sum to 1.0)
        
    Returns:
        Scalar cost
    """
    total = 0.0
    
    for obj_name, obj_value in objectives.items():
        weight = weights.get(obj_name, 0.0)
        total += weight * obj_value
    
    return total


def tchebycheff_scalarization(
    objectives: Dict[str, float],
    weights: Dict[str, float],
    ideal_point: Dict[str, float]
) -> float:
    """
    Scalarize using Tchebycheff method.
    
    Args:
        objectives: Dictionary of objective values
        weights: Dictionary of weights
        ideal_point: Ideal (utopia) point
        
    Returns:
        Scalar cost
    """
    max_weighted_diff = 0.0
    
    for obj_name, obj_value in objectives.items():
        weight = weights.get(obj_name, 1.0)
        ideal = ideal_point.get(obj_name, 0.0)
        weighted_diff = weight * abs(obj_value - ideal)
        max_weighted_diff = max(max_weighted_diff, weighted_diff)
    
    return max_weighted_diff


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create some sample solutions
    solutions = [
        Solution(1, {"time": 100, "energy": 50, "risk": 0.2}),
        Solution(2, {"time": 90, "energy": 60, "risk": 0.3}),
        Solution(3, {"time": 110, "energy": 45, "risk": 0.15}),
        Solution(4, {"time": 95, "energy": 55, "risk": 0.25}),
        Solution(5, {"time": 120, "energy": 40, "risk": 0.1}),
    ]
    
    # Compute Pareto front
    pf = ParetoFrontier()
    pf.add_solutions(solutions)
    
    print(f"Pareto front size: {len(pf.get_pareto_front())}")
    
    for sol in pf.get_pareto_front():
        print(f"  Route {sol.route_id}: time={sol.objectives['time']:.0f}, "
              f"energy={sol.objectives['energy']:.0f}, risk={sol.objectives['risk']:.2f}")
    
    # Find knee point
    knee = pf.get_knee_point()
    if knee:
        print(f"\nKnee point: Route {knee.route_id}")
    
    # Get extreme points
    extremes = pf.get_extreme_points()
    print(f"\nExtreme points:")
    for obj, sol in extremes.items():
        print(f"  Best {obj}: Route {sol.route_id}")
    
    # Select diverse routes
    selector = DiverseRouteSelector()
    diverse = selector.select_diverse_routes(pf.get_pareto_front(), k=3)
    
    print(f"\nDiverse routes (k=3):")
    for sol in diverse:
        print(f"  Route {sol.route_id}")

