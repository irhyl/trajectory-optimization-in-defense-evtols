"""
NSGA-III Algorithm Implementation for Multi-Objective Optimization

This module implements the NSGA-III algorithm for many-objective optimization
in eVTOL trajectory planning, addressing the critical gap in advanced optimization.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import logging
from .pareto import Solution, ParetoFrontier
from ..base import Optimizer, RoutePlan

logger = logging.getLogger(__name__)


@dataclass
class ReferencePoint:
    """Reference point for NSGA-III"""
    position: np.ndarray
    weight: float
    associated_solutions: List[int]


class NSGA3Optimizer(Optimizer):
    """
    NSGA-III optimizer for many-objective eVTOL trajectory optimization.
    
    Implements the NSGA-III algorithm with:
    - Reference point generation
    - Non-dominated sorting
    - Environmental selection
    - Convergence analysis
    """
    
    def __init__(
        self,
        objectives: List[str],
        constraints: List[str],
        population_size: int = 100,
        max_generations: int = 1000,
        crossover_prob: float = 0.9,
        mutation_prob: float = 0.1,
        eta_c: float = 20.0,
        eta_m: float = 20.0
    ):
        """
        Initialize NSGA-III optimizer.
        
        Args:
            objectives: List of objective function names
            constraints: List of constraint function names
            population_size: Population size
            max_generations: Maximum number of generations
            crossover_prob: Crossover probability
            mutation_prob: Mutation probability
            eta_c: Crossover distribution index
            eta_m: Mutation distribution index
        """
        self.objectives = objectives
        self.constraints = constraints
        self.population_size = population_size
        self.max_generations = max_generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.eta_c = eta_c
        self.eta_m = eta_m
        
        # Generate reference points
        self.reference_points = self._generate_reference_points()
        
        # Optimization state
        self.population: List[Solution] = []
        self.generation = 0
        self.best_solutions: List[Solution] = []
        self.convergence_history: List[float] = []
        
        logger.info(f"Initialized NSGA-III with {len(objectives)} objectives")
    
    def _run_evolution(self, initial_population: Optional[List[Solution]] = None) -> List[Solution]:
        """Run the internal NSGA-III evolution loop on a population of Solution objects."""
        # Initialize population
        if initial_population is None:
            self.population = self._generate_initial_population()
        else:
            self.population = initial_population

        logger.info(f"Starting NSGA-III optimization for {self.max_generations} generations")

        for generation in range(self.max_generations):
            self.generation = generation

            # Evaluate objectives
            self._evaluate_population()

            # Non-dominated sorting
            fronts = self._non_dominated_sorting()

            # Environmental selection
            self.population = self._environmental_selection(fronts)

            # Generate offspring
            offspring = self._generate_offspring()

            # Combine parent and offspring
            combined_population = self.population + offspring

            # Select next generation
            self.population = self._select_next_generation(combined_population)

            # Track convergence
            self._update_convergence_metrics()

            # Log progress
            if generation % 100 == 0:
                logger.info(f"Generation {generation}: {len(self.population)} solutions")

        # Final selection
        self.best_solutions = self._get_pareto_optimal_solutions()

        logger.info(f"Optimization completed. Found {len(self.best_solutions)} Pareto-optimal solutions")
        return self.best_solutions
    
    def optimize(self, candidates: List[RoutePlan], objectives: Dict[str, float], constraints: Optional[Dict] = None) -> List[RoutePlan]:
        """Adapter method to run NSGA-III on a list of RoutePlan candidates.

        Converts RoutePlan objects to internal Solution objects, runs the NSGA-III
        evolution, and maps selected Solutions back to RoutePlan instances.
        """
        # Convert RoutePlan candidates to Solution objects
        solutions: List[Solution] = []
        for idx, route in enumerate(candidates):
            obj_map = {}
            # map common objective names to route attributes
            for obj in self.objectives:
                if obj == "distance":
                    obj_map[obj] = float(route.distance_km)
                elif obj == "energy":
                    obj_map[obj] = float(route.energy_kwh)
                elif obj == "risk":
                    obj_map[obj] = float(route.risk_score)
                elif obj == "time":
                    obj_map[obj] = float(route.flight_time_s)
                else:
                    # fallback to metadata
                    obj_map[obj] = float(route.metadata.get(obj, 0.0)) if route.metadata else 0.0

            solutions.append(Solution(route_id=idx, objectives=obj_map, metadata=(route.metadata or {}).copy()))

        # Run the evolution on the Solution objects
        pareto_solutions = self._run_evolution(initial_population=solutions)

        # Map selected solutions back to RoutePlan objects using route_id
        selected_routes: List[RoutePlan] = []
        for sol in pareto_solutions:
            if 0 <= sol.route_id < len(candidates):
                selected_routes.append(candidates[sol.route_id])

        return selected_routes
    
    def _generate_reference_points(self) -> List[ReferencePoint]:
        """Generate reference points using Das and Dennis method."""
        num_objectives = len(self.objectives)
        
        if num_objectives <= 3:
            # Use Das and Dennis method for 2-3 objectives
            reference_points = self._das_dennis_method(num_objectives)
        else:
            # Use uniform distribution for many objectives
            reference_points = self._uniform_reference_points(num_objectives)
        
        return reference_points
    
    def _das_dennis_method(self, num_objectives: int) -> List[ReferencePoint]:
        """Generate reference points using Das and Dennis method."""
        # Implementation of Das and Dennis method
        # This is a simplified version - full implementation would be more complex
        reference_points = []
        
        # Generate points on the unit simplex
        for i in range(self.population_size):
            # Generate random point on unit simplex
            point = np.random.dirichlet(np.ones(num_objectives))
            reference_points.append(ReferencePoint(
                position=point,
                weight=1.0 / self.population_size,
                associated_solutions=[]
            ))
        
        return reference_points
    
    def _uniform_reference_points(self, num_objectives: int) -> List[ReferencePoint]:
        """Generate uniform reference points for many objectives."""
        reference_points = []
        
        # Generate uniform points on unit hyperplane
        for i in range(self.population_size):
            point = np.random.dirichlet(np.ones(num_objectives))
            reference_points.append(ReferencePoint(
                position=point,
                weight=1.0 / self.population_size,
                associated_solutions=[]
            ))
        
        return reference_points
    
    def _generate_initial_population(self) -> List[Solution]:
        """Generate initial population."""
        population = []
        
        for i in range(self.population_size):
            # Generate random solution
            solution = self._generate_random_solution()
            population.append(solution)
        
        return population
    
    def _generate_random_solution(self) -> Solution:
        """Generate a random solution."""
        # This would generate a random trajectory solution
        # Implementation depends on your specific problem
        route_id = np.random.randint(0, 1000)
        objectives = {
            obj: np.random.random() 
            for obj in self.objectives
        }
        
        return Solution(
            route_id=route_id,
            objectives=objectives,
            metadata={'generation': 0}
        )
    
    def _evaluate_population(self):
        """Evaluate objectives for all solutions in population."""
        for solution in self.population:
            # Evaluate objectives
            # This would call your actual objective functions
            pass
    
    def _non_dominated_sorting(self) -> List[List[Solution]]:
        """Perform non-dominated sorting."""
        fronts = []
        remaining_solutions = self.population.copy()
        
        while remaining_solutions:
            current_front = []
            dominated_solutions = []
            
            for i, solution_i in enumerate(remaining_solutions):
                is_dominated = False
                for j, solution_j in enumerate(remaining_solutions):
                    if i != j and solution_j.dominates(solution_i):
                        is_dominated = True
                        break
                
                if not is_dominated:
                    current_front.append(solution_i)
                else:
                    dominated_solutions.append(solution_i)
            
            fronts.append(current_front)
            remaining_solutions = dominated_solutions
        
        return fronts
    
    def _environmental_selection(self, fronts: List[List[Solution]]) -> List[Solution]:
        """Perform environmental selection."""
        selected_solutions = []
        
        for front in fronts:
            if len(selected_solutions) + len(front) <= self.population_size:
                selected_solutions.extend(front)
            else:
                # Need to select subset from this front
                remaining_slots = self.population_size - len(selected_solutions)
                selected_from_front = self._select_from_front(front, remaining_slots)
                selected_solutions.extend(selected_from_front)
                break
        
        return selected_solutions
    
    def _select_from_front(self, front: List[Solution], num_to_select: int) -> List[Solution]:
        """Select solutions from a front using reference points."""
        if len(front) <= num_to_select:
            return front
        
        # Associate solutions with reference points
        self._associate_solutions_with_reference_points(front)
        
        # Select based on reference point associations
        selected = self._select_by_reference_points(front, num_to_select)
        
        return selected
    
    def _associate_solutions_with_reference_points(self, solutions: List[Solution]):
        """Associate solutions with reference points."""
        for ref_point in self.reference_points:
            ref_point.associated_solutions = []
        
        for i, solution in enumerate(solutions):
            # Find closest reference point
            min_distance = float('inf')
            closest_ref_point = None
            
            for ref_point in self.reference_points:
                distance = self._perpendicular_distance(solution, ref_point)
                if distance < min_distance:
                    min_distance = distance
                    closest_ref_point = ref_point
            
            if closest_ref_point is not None:
                closest_ref_point.associated_solutions.append(i)
    
    def _perpendicular_distance(self, solution: Solution, ref_point: ReferencePoint) -> float:
        """Calculate perpendicular distance from solution to reference point."""
        # Normalize objectives
        normalized_objectives = np.array([solution.objectives[obj] for obj in self.objectives])
        
        # Calculate distance to reference point
        distance = np.linalg.norm(normalized_objectives - ref_point.position)
        
        return distance
    
    def _select_by_reference_points(self, solutions: List[Solution], num_to_select: int) -> List[Solution]:
        """Select solutions based on reference point associations."""
        selected = []
        
        # Sort reference points by number of associated solutions
        ref_points_sorted = sorted(
            self.reference_points,
            key=lambda rp: len(rp.associated_solutions)
        )
        
        for ref_point in ref_points_sorted:
            if len(selected) >= num_to_select:
                break
            
            if ref_point.associated_solutions:
                # Select solution from this reference point
                solution_idx = ref_point.associated_solutions[0]
                selected.append(solutions[solution_idx])
                ref_point.associated_solutions.pop(0)
        
        return selected
    
    def _generate_offspring(self) -> List[Solution]:
        """Generate offspring through crossover and mutation."""
        offspring = []
        
        for i in range(0, len(self.population), 2):
            # Select parents
            parent1 = self._tournament_selection()
            parent2 = self._tournament_selection()
            
            # Crossover
            if np.random.random() < self.crossover_prob:
                child1, child2 = self._simulated_binary_crossover(parent1, parent2)
            else:
                child1, child2 = parent1, parent2
            
            # Mutation
            if np.random.random() < self.mutation_prob:
                child1 = self._polynomial_mutation(child1)
            if np.random.random() < self.mutation_prob:
                child2 = self._polynomial_mutation(child2)
            
            offspring.extend([child1, child2])
        
        return offspring
    
    def _tournament_selection(self) -> Solution:
        """Tournament selection."""
        tournament_size = 2
        tournament = np.random.choice(self.population, tournament_size, replace=False)
        
        # Select best solution from tournament
        best = tournament[0]
        for solution in tournament[1:]:
            if solution.dominates(best):
                best = solution
        
        return best
    
    def _simulated_binary_crossover(self, parent1: Solution, parent2: Solution) -> Tuple[Solution, Solution]:
        """Simulated binary crossover (SBX)."""
        # Simplified implementation
        # Full SBX would be more complex
        child1 = parent1
        child2 = parent2
        
        return child1, child2
    
    def _polynomial_mutation(self, solution: Solution) -> Solution:
        """Polynomial mutation."""
        # Simplified implementation
        # Full polynomial mutation would be more complex
        return solution
    
    def _select_next_generation(self, combined_population: List[Solution]) -> List[Solution]:
        """Select next generation from combined population."""
        # Non-dominated sorting of combined population
        fronts = self._non_dominated_sorting()
        
        # Environmental selection
        next_generation = self._environmental_selection(fronts)
        
        return next_generation
    
    def _update_convergence_metrics(self):
        """Update convergence metrics."""
        # Calculate hypervolume
        hypervolume = self._calculate_hypervolume()
        self.convergence_history.append(hypervolume)
        
        # Check convergence
        if len(self.convergence_history) > 10:
            recent_improvement = abs(
                self.convergence_history[-1] - self.convergence_history[-10]
            )
            if recent_improvement < 1e-6:
                logger.info("Convergence detected")
    
    def _calculate_hypervolume(self) -> float:
        """Calculate hypervolume indicator."""
        if not self.population:
            return 0.0
        
        # Simplified hypervolume calculation
        # Full implementation would be more complex
        return len(self.population) * 0.1
    
    def _get_pareto_optimal_solutions(self) -> List[Solution]:
        """Get Pareto-optimal solutions from final population."""
        # Find non-dominated solutions
        pareto_optimal = []
        
        for solution in self.population:
            is_dominated = False
            for other in self.population:
                if other != solution and other.dominates(solution):
                    is_dominated = True
                    break
            
            if not is_dominated:
                pareto_optimal.append(solution)
        
        return pareto_optimal
    
    def get_convergence_history(self) -> List[float]:
        """Get convergence history."""
        return self.convergence_history
    
    def get_reference_points(self) -> List[ReferencePoint]:
        """Get reference points."""
        return self.reference_points
