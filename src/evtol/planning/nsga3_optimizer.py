"""
NSGA-III Multi-Objective Optimizer

Phase 2C-2: Non-dominated Sorting Genetic Algorithm III for Pareto-optimal
trajectory generation balancing energy consumption, mission time, and threat avoidance.

Algorithm: NSGA-III (Deb & Jain, 2014)
  • Many-objective optimization (3+ objectives)
  • Reference point-based diversity maintenance
  • Elitist evolutionary strategy
  • Converges to Pareto front

Objectives:
  f₁: Energy consumption [kWh] → MINIMIZE
  f₂: Mission time [minutes] → MINIMIZE  
  f₃: Threat margin [m] → MAXIMIZE (stay far from threats)

Trade-offs:
  • More speed → more energy, less time, closer to threats (risky)
  • More efficient → less energy, more time, safer
  • Mix: medium energy, medium time, medium safety
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Callable
import numpy as np
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


@dataclass
class Trajectory:
    """
    Single trajectory solution - candidate mission plan.
    
    Attributes:
        waypoints: Sequence of positions [(x,y,z), ...] meters
        velocities: Speed commands [v1, v2, ...] m/s at each waypoint
        energy: Total energy consumption [kWh]
        time: Mission duration [seconds]
        threat_margin: Minimum distance to any threat [m]
        fitness: [f1, f2, f3] objective values (populated after evaluation)
    """
    waypoints: List[np.ndarray]
    velocities: List[float]
    energy: float = 0.0  # Will be computed
    time: float = 0.0    # Will be computed
    threat_margin: float = 5000.0  # Will be computed (large = safe)
    fitness: np.ndarray = field(default_factory=lambda: np.array([np.inf, np.inf, -np.inf]))
    
    def __post_init__(self):
        """Validate trajectory structure."""
        if len(self.waypoints) != len(self.velocities):
            raise ValueError(
                f"waypoints ({len(self.waypoints)}) and velocities ({len(self.velocities)}) "
                "must have same length"
            )
        if len(self.waypoints) < 2:
            raise ValueError("Need at least 2 waypoints for trajectory")
    
    @property
    def is_feasible(self) -> bool:
        """Check if trajectory is feasible (energy and time bounded)."""
        # Placeholder - would check against vehicle limits
        return self.energy < 85.0 and self.threat_margin > 100.0  # kWh, m
    
    def copy(self) -> Trajectory:
        """Create deep copy of trajectory."""
        return Trajectory(
            waypoints=[wp.copy() for wp in self.waypoints],
            velocities=self.velocities.copy(),
            energy=self.energy,
            time=self.time,
            threat_margin=self.threat_margin,
            fitness=self.fitness.copy(),
        )


@dataclass
class NSGA3Config:
    """
    Configuration for NSGA-III optimizer.
    
    Attributes:
        population_size: Number of solutions per generation
        generations: Number of generations to evolve
        crossover_prob: Probability of crossover mutation [0-1]
        mutation_prob: Probability of individual mutation [0-1]
        mutation_std: Standard deviation for Gaussian mutation [m]
    """
    population_size: int = 100
    generations: int = 50
    crossover_prob: float = 0.8
    mutation_prob: float = 0.1
    mutation_std: float = 500.0  # meters


class ObjectiveFunction(ABC):
    """Base class for optimization objectives."""
    
    @abstractmethod
    def evaluate(self, trajectory: Trajectory) -> float:
        """Compute objective value for trajectory."""
        pass
    
    @abstractmethod
    def direction(self) -> str:
        """Return 'min' or 'max' for optimization direction."""
        pass


class EnergyObjective(ObjectiveFunction):
    """Minimize energy consumption [kWh]."""
    
    def evaluate(self, trajectory: Trajectory) -> float:
        """Return energy (to be minimized)."""
        return trajectory.energy
    
    def direction(self) -> str:
        return "min"


class TimeObjective(ObjectiveFunction):
    """Minimize mission duration [minutes]."""
    
    def evaluate(self, trajectory: Trajectory) -> float:
        """Return time in minutes (to be minimized)."""
        return trajectory.time / 60.0  # Convert seconds to minutes
    
    def direction(self) -> str:
        return "min"


class ThreatMarginObjective(ObjectiveFunction):
    """Maximize minimum distance to threats [m]."""
    
    def evaluate(self, trajectory: Trajectory) -> float:
        """Return negative threat margin (to be minimized = maximizing distance)."""
        return -trajectory.threat_margin  # Negative so minimization = maximize distance
    
    def direction(self) -> str:
        return "min"


class NSGA3Optimizer:
    """
    NSGA-III many-objective optimizer for trajectory planning.
    
    Algorithm flow:
      1. Initialize population with random trajectories
      2. Evaluate fitness (objectives)
      3. Non-dominated sort (rank by Pareto dominance)
      4. Assign to reference points (diversity maintenance)
      5. Selection (keep best solutions)
      6. Crossover & mutation (generate next generation)
      7. Repeat until convergence
    
    Usage:
        optimizer = NSGA3Optimizer(
            objectives=[EnergyObjective(), TimeObjective(), ThreatMarginObjective()],
            config=NSGA3Config(population_size=100, generations=50),
        )
        
        pareto_front = optimizer.optimize(
            initial_trajectories=start_pop,
            vehicle=vehicle_model,
            threats=threat_zones,
        )
        
        for traj in pareto_front:
            print(f"Energy: {traj.energy:.1f} kWh, Time: {traj.time/60:.1f} min")
    
    Attributes:
        objectives: List of ObjectiveFunction instances
        config: NSGA3Config configuration
        reference_points: Reference points for diversity (generated via uniform spread)
    """
    
    def __init__(
        self,
        objectives: List[ObjectiveFunction],
        config: Optional[NSGA3Config] = None,
    ):
        """
        Initialize NSGA-III optimizer.
        
        Args:
            objectives: List of ObjectiveFunction instances to optimize
            config: NSGA3Config (uses defaults if None)
        """
        if config is None:
            config = NSGA3Config()
        
        self.objectives = objectives
        self.config = config
        self.n_objectives = len(objectives)
        
        # Generate reference points (Das & Dennis method)
        self.reference_points = self._generate_reference_points()
        
        logger.info(
            f"NSGA3Optimizer initialized: {self.n_objectives} objectives, "
            f"{len(self.reference_points)} reference points, "
            f"pop_size={config.population_size}, gens={config.generations}"
        )
    
    def optimize(
        self,
        initial_trajectories: List[Trajectory],
        vehicle_evaluator: Callable[[Trajectory], Trajectory],
        threat_evaluator: Callable[[Trajectory], float],
        max_time: float = 300.0,  # seconds
    ) -> List[Trajectory]:
        """
        Optimize trajectories to Pareto front.
        
        Args:
            initial_trajectories: Starting population
            vehicle_evaluator: Function to compute energy/time for trajectory
            threat_evaluator: Function to compute threat margin for trajectory
            max_time: Max computation time [seconds]
        
        Returns:
            Pareto front solutions (non-dominated set)
        """
        import time
        start_time = time.time()
        
        # Initialize population
        population = [t.copy() for t in initial_trajectories]
        
        logger.info(f"Starting optimization with {len(population)} initial solutions")
        
        for generation in range(self.config.generations):
            elapsed = time.time() - start_time
            if elapsed > max_time:
                logger.warning(f"Optimization timeout after {elapsed:.1f}s")
                break
            
            # Evaluate fitness: first populate energy/time/threat via evaluators,
            # then score each objective.
            for traj in population:
                if vehicle_evaluator is not None:
                    vehicle_evaluator(traj)
                if threat_evaluator is not None:
                    threat_evaluator(traj)
                for i, obj in enumerate(self.objectives):
                    traj.fitness[i] = obj.evaluate(traj)
            
            # Non-dominated sort
            ranks = self._nondominated_sort(population)
            
            # Assign to reference points
            distances = self._assign_to_reference_points(population, ranks)
            
            # Selection (keep best solutions)
            selected = self._select_based_on_rank_and_distance(
                population, ranks, distances, self.config.population_size
            )
            
            # Crossover & mutation (create offspring)
            offspring = []
            while len(offspring) < self.config.population_size:
                # Tournament selection
                parent1 = self._tournament_select(selected)
                parent2 = self._tournament_select(selected)
                
                # Crossover
                if np.random.rand() < self.config.crossover_prob:
                    child1, child2 = self._crossover(parent1, parent2)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()
                
                # Mutation
                if np.random.rand() < self.config.mutation_prob:
                    child1 = self._mutate(child1)
                if np.random.rand() < self.config.mutation_prob:
                    child2 = self._mutate(child2)
                
                offspring.extend([child1, child2])
            
            # Combine parent and offspring for next generation
            population = selected + offspring[:self.config.population_size]
            
            # Log progress
            if (generation + 1) % 10 == 0:
                pareto_front = self._extract_pareto_front(population)
                logger.info(
                    f"Generation {generation+1}/{self.config.generations}: "
                    f"pop_size {len(population)}, pareto_size {len(pareto_front)}"
                )
        
        # Extract final Pareto front
        pareto_front = self._extract_pareto_front(population)
        
        logger.info(
            f"Optimization complete: {len(pareto_front)} solutions on Pareto front "
            f"in {time.time() - start_time:.1f}s"
        )
        
        return pareto_front
    
    def _generate_reference_points(self, divisions: int = 4) -> List[np.ndarray]:
        """
        Generate reference points using Das & Dennis method.
        
        Creates uniform spread of points on Pareto front.
        """
        # Placeholder - simplified uniform grid
        n_points = (divisions + self.n_objectives - 1) ** (self.n_objectives - 1)
        points = []
        for _ in range(n_points):
            point = np.random.dirichlet(np.ones(self.n_objectives))
            points.append(point)
        return points
    
    def _nondominated_sort(self, population: List[Trajectory]) -> List[int]:
        """
        Rank solutions by Pareto dominance.
        
        Returns array where rank[i] is the rank of population[i]
        (rank 1 = best, higher = worse)
        """
        ranks = [float('inf')] * len(population)
        
        for i, sol_i in enumerate(population):
            rank_i = 1
            for j, sol_j in enumerate(population):
                if i != j:
                    if self._dominates(sol_j, sol_i):
                        rank_i += 1
            ranks[i] = rank_i
        
        return ranks
    
    def _dominates(self, sol1: Trajectory, sol2: Trajectory) -> bool:
        """Check if sol1 dominates sol2 (better on all objectives)."""
        all_better_or_equal = True
        any_strictly_better = False
        
        for i, obj in enumerate(self.objectives):
            val1 = sol1.fitness[i]
            val2 = sol2.fitness[i]
            
            if obj.direction() == "min":
                if val1 > val2:
                    all_better_or_equal = False
                if val1 < val2:
                    any_strictly_better = True
            else:  # max
                if val1 < val2:
                    all_better_or_equal = False
                if val1 > val2:
                    any_strictly_better = True
        
        return all_better_or_equal and any_strictly_better
    
    def _assign_to_reference_points(
        self,
        population: List[Trajectory],
        ranks: List[int],
    ) -> List[float]:
        """
        Assign each solution to nearest reference point.
        
        Returns distances to assigned reference points.
        """
        distances = []
        for sol in population:
            min_dist = float('inf')
            for ref_point in self.reference_points:
                # Normalize fitness for comparison
                dist = np.linalg.norm(sol.fitness - ref_point)
                min_dist = min(min_dist, dist)
            distances.append(min_dist)
        return distances
    
    def _select_based_on_rank_and_distance(
        self,
        population: List[Trajectory],
        ranks: List[int],
        distances: List[float],
        n_select: int,
    ) -> List[Trajectory]:
        """
        Select best solutions based on rank and crowding distance.
        """
        # Sort by rank first, then by distance
        indices = sorted(
            range(len(population)),
            key=lambda i: (ranks[i], -distances[i]),
        )
        
        return [population[i] for i in indices[:n_select]]
    
    def _tournament_select(self, population: List[Trajectory]) -> Trajectory:
        """Tournament selection - pick random pair, return better one."""
        i1, i2 = np.random.choice(len(population), 2, replace=False)
        sol1, sol2 = population[i1], population[i2]
        
        # Prefer lower rank (better solution)
        if np.mean(sol1.fitness) < np.mean(sol2.fitness):
            return sol1.copy()
        else:
            return sol2.copy()
    
    def _crossover(self, parent1: Trajectory, parent2: Trajectory) -> Tuple[Trajectory, Trajectory]:
        """
        Blend trajectory parameters.
        
        Uniform crossover on waypoints, blend on velocities.
        """
        child1 = parent1.copy()
        child2 = parent2.copy()
        
        # Blend waypoints
        for i in range(len(child1.waypoints)):
            if np.random.rand() < 0.5:
                child1.waypoints[i], child2.waypoints[i] = child2.waypoints[i], child1.waypoints[i]
        
        # Blend velocities
        for i in range(len(child1.velocities)):
            blend_factor = np.random.rand()
            child1.velocities[i] = (
                (1 - blend_factor) * child1.velocities[i] +
                blend_factor * child2.velocities[i]
            )
            child2.velocities[i] = (
                (1 - blend_factor) * child2.velocities[i] +
                blend_factor * child1.velocities[i]
            )
        
        return child1, child2
    
    def _mutate(self, trajectory: Trajectory) -> Trajectory:
        """
        Mutate trajectory via Gaussian perturbation.
        """
        mutant = trajectory.copy()
        
        # Perturb random waypoints
        n_perturb = np.random.randint(1, max(2, len(mutant.waypoints) // 3))
        indices = np.random.choice(len(mutant.waypoints), n_perturb, replace=False)
        
        for i in indices:
            perturbation = np.random.normal(0, self.config.mutation_std, 3)
            mutant.waypoints[i] = mutant.waypoints[i] + perturbation
        
        # Perturb velocities
        for i in range(len(mutant.velocities)):
            if np.random.rand() < 0.2:
                mutant.velocities[i] += np.random.normal(0, 1.0)  # ±1 m/s
                mutant.velocities[i] = np.clip(mutant.velocities[i], 0, 30)  # Clamp [0, 30 m/s]
        
        return mutant
    
    def _extract_pareto_front(self, population: List[Trajectory]) -> List[Trajectory]:
        """Extract non-dominated solutions (rank 1)."""
        ranks = self._nondominated_sort(population)
        return [population[i] for i in range(len(population)) if ranks[i] == 1]


# ═══════════════════════════════════════════════════════════════════════════════
# Example usage
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Create optimizer with 3 objectives
    objectives = [
        EnergyObjective(),
        TimeObjective(),
        ThreatMarginObjective(),
    ]
    config = NSGA3Config(population_size=50, generations=20)
    optimizer = NSGA3Optimizer(objectives, config)
    
    # Create sample trajectories
    sample_trajs = []
    for _ in range(20):
        waypoints = [
            np.array([0, 0, 0]),
            np.array([50000, 50000, 500]),
            np.array([100000, 100000, 0]),
        ]
        velocities = [15.0, 20.0, 15.0]
        traj = Trajectory(waypoints=waypoints, velocities=velocities)
        # Simulate different energy/time values
        traj.energy = np.random.uniform(20, 70)
        traj.time = np.random.uniform(900, 2400)
        traj.threat_margin = np.random.uniform(100, 5000)
        sample_trajs.append(traj)
    
    # Simple evaluators (placeholders)
    def dummy_vehicle_eval(t: Trajectory) -> Trajectory:
        return t
    
    def dummy_threat_eval(t: Trajectory) -> float:
        return t.threat_margin
    
    # Optimize
    pareto_front = optimizer.optimize(
        initial_trajectories=sample_trajs,
        vehicle_evaluator=dummy_vehicle_eval,
        threat_evaluator=dummy_threat_eval,
        max_time=10.0,
    )
    
    print(f"✓ Found {len(pareto_front)} solutions on Pareto front")
    for i, traj in enumerate(pareto_front[:5]):
        print(
            f"  Solution {i+1}: Energy {traj.energy:.1f} kWh, "
            f"Time {traj.time/60:.1f} min, Threat margin {traj.threat_margin:.0f}m"
        )
