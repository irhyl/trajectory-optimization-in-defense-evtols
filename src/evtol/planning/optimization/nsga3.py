"""
NSGA-III: Non-dominated Sorting Genetic Algorithm III

This module implements NSGA-III, a many-objective evolutionary algorithm
that uses reference point based selection for maintaining diversity.

Mathematical Foundation:
------------------------
NSGA-III extends NSGA-II for many-objective optimization (>3 objectives):

1. Non-dominated sorting remains the same
2. Instead of crowding distance, uses reference point association
3. Reference points form a structured simplex in objective space
4. Solutions are associated with nearest reference point
5. Selection favors solutions associated with less crowded reference points

Key Features:
- Handles many objectives (>3) effectively
- Maintains diversity through reference point distribution
- Converges to well-distributed Pareto frontier

Reference:
Deb, K., & Jain, H. (2014). An Evolutionary Many-Objective Optimization
Algorithm Using Reference-Point-Based Nondominated Sorting Approach,
Part I: Solving Problems With Box Constraints. IEEE TEVC.

Author: Defense eVTOL Research Team
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from collections.abc import Callable
import logging
from copy import deepcopy
import time

from ..core.trajectory import Trajectory
from .objectives import ObjectiveSet
from .pareto import (
    fast_non_dominated_sort,
    crowding_distance,
    ParetoFrontier,
)

logger = logging.getLogger(__name__)


@dataclass
class NSGA3Config:
    """
    Configuration for NSGA-III optimizer.

    Attributes:
        n_generations: Number of generations
        population_size: Population size (should be divisible by 4)
        n_divisions: Divisions for reference point generation
        crossover_prob: SBX crossover probability
        crossover_eta: SBX distribution index
        mutation_prob: Polynomial mutation probability
        mutation_eta: Polynomial mutation distribution index
        seed: Random seed for reproducibility
    """
    n_generations: int = 100
    population_size: int = 100
    n_divisions: int = 12
    crossover_prob: float = 0.9
    crossover_eta: float = 20.0
    mutation_prob: float = None  # Default: 1/n_vars
    mutation_eta: float = 20.0
    seed: int | None = None
    verbose: bool = True
    checkpoint_interval: int = 10


@dataclass
class Individual:
    """
    Individual in the population.

    Represents a candidate trajectory with its objective values.

    Attributes:
        genes: Decision variable encoding (waypoint sequence)
        trajectory: Decoded trajectory
        objectives: Objective vector
        rank: Pareto front rank
        crowding: Crowding distance (for NSGA-II fallback)
        ref_point: Associated reference point index
        niche_count: Number of solutions sharing reference point
    """
    genes: np.ndarray
    trajectory: Trajectory | None = None
    objectives: np.ndarray | None = None
    rank: int = 0
    crowding: float = np.inf
    ref_point: int = -1
    niche_count: int = 0

    def copy(self) -> Individual:
        """Create a deep copy."""
        ind = Individual(
            genes=self.genes.copy(),
            trajectory=deepcopy(self.trajectory),
            objectives=self.objectives.copy() if self.objectives is not None else None,
            rank=self.rank,
            crowding=self.crowding,
            ref_point=self.ref_point,
            niche_count=self.niche_count,
        )
        return ind


@dataclass
class Population:
    """
    Population of individuals.

    Provides operations for population management and statistics.
    """
    individuals: list[Individual] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.individuals)

    def __iter__(self):
        return iter(self.individuals)

    def __getitem__(self, idx: int) -> Individual:
        return self.individuals[idx]

    def add(self, individual: Individual) -> None:
        """Add an individual."""
        self.individuals.append(individual)

    def get_objectives(self) -> np.ndarray:
        """Get objective matrix."""
        return np.array([ind.objectives for ind in self.individuals])

    def get_genes(self) -> np.ndarray:
        """Get gene matrix."""
        return np.array([ind.genes for ind in self.individuals])

    def sort_by_rank_crowding(self) -> None:
        """Sort by rank (ascending) then crowding (descending)."""
        self.individuals.sort(key=lambda x: (x.rank, -x.crowding))

    def get_pareto_frontier(self) -> list[Individual]:
        """Get individuals on Pareto frontier (rank 0)."""
        return [ind for ind in self.individuals if ind.rank == 0]


@dataclass
class OptimizationResult:
    """
    Result of NSGA-III optimization.

    Attributes:
        pareto_frontier: Final Pareto frontier
        population: Final population
        history: Objective history per generation
        runtime: Total runtime in seconds
        n_evaluations: Total function evaluations
    """
    pareto_frontier: ParetoFrontier
    population: Population
    history: dict[str, list[float]] = field(default_factory=dict)
    runtime: float = 0.0
    n_evaluations: int = 0
    best_individual: Individual | None = None

    def summary(self) -> str:
        """Generate optimization summary."""
        lines = [
            "=" * 50,
            "NSGA-III Optimization Results",
            "=" * 50,
            f"Pareto frontier size: {len(self.pareto_frontier)}",
            f"Total evaluations: {self.n_evaluations}",
            f"Runtime: {self.runtime:.2f} seconds",
        ]

        if self.pareto_frontier.ideal_point is not None:
            lines.append(f"Ideal point: {self.pareto_frontier.ideal_point}")
        if self.pareto_frontier.nadir_point is not None:
            lines.append(f"Nadir point: {self.pareto_frontier.nadir_point}")

        knee = self.pareto_frontier.find_knee_point()
        if knee is not None:
            lines.append(f"Knee point objectives: {knee.objectives}")

        lines.append("=" * 50)
        return "\n".join(lines)


class NSGA3Optimizer:
    """
    NSGA-III Multi-Objective Optimizer.

    Optimizes trajectories for multiple objectives (time, energy, threat, etc.)
    using the NSGA-III evolutionary algorithm with reference point based selection.

    Usage:
        optimizer = NSGA3Optimizer(objective_set, config)
        result = optimizer.optimize(
            initial_population,
            decoder=trajectory_decoder,
            bounds=gene_bounds,
        )

    Attributes:
        objectives: ObjectiveSet defining optimization objectives
        config: NSGA3Config with algorithm parameters
        reference_points: Generated reference points on unit simplex
    """

    def __init__(
        self,
        objectives: ObjectiveSet,
        config: NSGA3Config | None = None,
    ):
        """
        Initialize NSGA-III optimizer.

        Args:
            objectives: Set of objective functions
            config: Algorithm configuration
        """
        self.objectives = objectives
        self.config = config or NSGA3Config()
        self.n_objectives = len(objectives)

        # Generate reference points
        self.reference_points = self._generate_reference_points(
            self.n_objectives,
            self.config.n_divisions,
        )

        # Set random seed
        if self.config.seed is not None:
            np.random.seed(self.config.seed)

        # Statistics
        self._n_evaluations = 0
        self._generation = 0
        self._history: dict[str, list[float]] = {
            'hypervolume': [],
            'spacing': [],
            'pareto_size': [],
        }
        for obj in objectives:
            self._history[f'min_{obj.name}'] = []
            self._history[f'mean_{obj.name}'] = []

    def _generate_reference_points(
        self,
        n_objectives: int,
        n_divisions: int,
    ) -> np.ndarray:
        """
        Generate uniformly distributed reference points on unit simplex.

        Uses Das and Dennis's systematic approach:
        H = C(n_objectives + n_divisions - 1, n_divisions)

        Args:
            n_objectives: Number of objectives
            n_divisions: Number of divisions per dimension

        Returns:
            H x n_objectives array of reference points
        """
        # Generate all combinations that sum to n_divisions
        def generate_combinations(n_obj, n_div, current=None):
            if current is None:
                current = []
            if n_obj == 1:
                yield current + [n_div]
            else:
                for i in range(n_div + 1):
                    yield from generate_combinations(n_obj - 1, n_div - i, current + [i])

        points = np.array(list(generate_combinations(n_objectives, n_divisions)))
        points = points / n_divisions  # Normalize to [0, 1]

        logger.info(f"Generated {len(points)} reference points for {n_objectives} objectives")
        return points

    def optimize(
        self,
        decoder: Callable[[np.ndarray], Trajectory],
        bounds: tuple[np.ndarray, np.ndarray],
        initial_genes: np.ndarray | None = None,
        constraint_check: Callable[[Trajectory], bool] | None = None,
    ) -> OptimizationResult:
        """
        Run NSGA-III optimization.

        Args:
            decoder: Function to decode genes to trajectory
            bounds: (lower, upper) bounds for genes
            initial_genes: Optional initial population genes
            constraint_check: Optional constraint checker

        Returns:
            OptimizationResult with Pareto frontier
        """
        start_time = time.time()
        lower_bounds, upper_bounds = bounds
        n_vars = len(lower_bounds)

        # Set default mutation probability
        if self.config.mutation_prob is None:
            mutation_prob = 1.0 / n_vars
        else:
            mutation_prob = self.config.mutation_prob

        # Initialize population
        population = self._initialize_population(
            initial_genes,
            decoder,
            bounds,
            constraint_check,
        )

        if self.config.verbose:
            logger.info(f"Starting NSGA-III with {len(population)} individuals")

        # Main evolutionary loop
        for gen in range(self.config.n_generations):
            self._generation = gen

            # Generate offspring
            offspring = self._create_offspring(
                population,
                decoder,
                bounds,
                mutation_prob,
                constraint_check,
            )

            # Combine parent and offspring
            combined = Population(population.individuals + offspring.individuals)

            # Non-dominated sorting
            objectives = combined.get_objectives()
            fronts = fast_non_dominated_sort(objectives)

            # Assign ranks
            for rank, front in enumerate(fronts):
                for idx in front:
                    combined[idx].rank = rank

            # Select next generation using reference point association
            population = self._reference_point_selection(
                combined,
                fronts,
                objectives,
            )

            # Record statistics
            self._record_statistics(population)

            # Checkpoint logging
            if self.config.verbose and (gen + 1) % self.config.checkpoint_interval == 0:
                pareto_size = len([ind for ind in population if ind.rank == 0])
                logger.info(f"Generation {gen + 1}/{self.config.n_generations} - "
                          f"Pareto size: {pareto_size}")

        # Build final result
        runtime = time.time() - start_time

        # Create Pareto frontier
        pareto_individuals = [ind for ind in population if ind.rank == 0]
        F = np.array([ind.objectives for ind in pareto_individuals])
        decisions = [ind.trajectory for ind in pareto_individuals]
        pareto_frontier = ParetoFrontier.from_population(F, decisions)

        # Find best individual (knee point or minimum weighted sum)
        knee = pareto_frontier.find_knee_point()
        best_individual = None
        if knee is not None:
            for ind in pareto_individuals:
                if np.allclose(ind.objectives, knee.objectives):
                    best_individual = ind
                    break

        result = OptimizationResult(
            pareto_frontier=pareto_frontier,
            population=population,
            history=self._history.copy(),
            runtime=runtime,
            n_evaluations=self._n_evaluations,
            best_individual=best_individual,
        )

        if self.config.verbose:
            logger.info(result.summary())

        return result

    def _initialize_population(
        self,
        initial_genes: np.ndarray | None,
        decoder: Callable,
        bounds: tuple[np.ndarray, np.ndarray],
        constraint_check: Callable | None = None,
    ) -> Population:
        """
        Initialize the population.

        Args:
            initial_genes: Optional seed genes
            decoder: Gene to trajectory decoder
            bounds: Gene bounds
            constraint_check: Constraint checker

        Returns:
            Initial population
        """
        lower, upper = bounds
        n_vars = len(lower)
        population = Population()

        # Add initial genes if provided
        if initial_genes is not None:
            for genes in initial_genes:
                if len(population) >= self.config.population_size:
                    break
                ind = self._create_individual(genes, decoder, constraint_check)
                if ind is not None:
                    population.add(ind)
                    len(population)

        # Fill remaining with random individuals
        attempts = 0
        max_attempts = self.config.population_size * 10

        while len(population) < self.config.population_size and attempts < max_attempts:
            genes = np.random.uniform(lower, upper, n_vars)
            ind = self._create_individual(genes, decoder, constraint_check)
            if ind is not None:
                population.add(ind)
            attempts += 1

        if len(population) < self.config.population_size:
            logger.warning(f"Only initialized {len(population)} valid individuals")

        return population

    def _create_individual(
        self,
        genes: np.ndarray,
        decoder: Callable,
        constraint_check: Callable | None = None,
    ) -> Individual | None:
        """
        Create and evaluate an individual.

        Args:
            genes: Gene vector
            decoder: Gene to trajectory decoder
            constraint_check: Optional constraint checker

        Returns:
            Individual or None if invalid
        """
        try:
            trajectory = decoder(genes)

            # Check constraints
            if constraint_check is not None and not constraint_check(trajectory):
                return None

            # Evaluate objectives
            objectives = self.objectives.evaluate_vector(trajectory)
            self._n_evaluations += 1

            return Individual(
                genes=genes.copy(),
                trajectory=trajectory,
                objectives=objectives,
            )
        except Exception as e:
            logger.debug(f"Failed to create individual: {e}")
            return None

    def _create_offspring(
        self,
        population: Population,
        decoder: Callable,
        bounds: tuple[np.ndarray, np.ndarray],
        mutation_prob: float,
        constraint_check: Callable | None = None,
    ) -> Population:
        """
        Create offspring through crossover and mutation.

        Uses:
        - Binary tournament selection
        - SBX (Simulated Binary Crossover)
        - Polynomial mutation

        Args:
            population: Parent population
            decoder: Gene to trajectory decoder
            bounds: Gene bounds
            mutation_prob: Mutation probability
            constraint_check: Constraint checker

        Returns:
            Offspring population
        """
        offspring = Population()
        n_offspring = self.config.population_size
        lower, upper = bounds

        while len(offspring) < n_offspring:
            # Binary tournament selection
            parent1 = self._tournament_select(population)
            parent2 = self._tournament_select(population)

            # SBX crossover
            if np.random.random() < self.config.crossover_prob:
                child1_genes, child2_genes = self._sbx_crossover(
                    parent1.genes, parent2.genes,
                    lower, upper,
                    self.config.crossover_eta,
                )
            else:
                child1_genes = parent1.genes.copy()
                child2_genes = parent2.genes.copy()

            # Polynomial mutation
            child1_genes = self._polynomial_mutation(
                child1_genes, lower, upper,
                mutation_prob, self.config.mutation_eta,
            )
            child2_genes = self._polynomial_mutation(
                child2_genes, lower, upper,
                mutation_prob, self.config.mutation_eta,
            )

            # Create individuals
            for genes in [child1_genes, child2_genes]:
                if len(offspring) >= n_offspring:
                    break
                ind = self._create_individual(genes, decoder, constraint_check)
                if ind is not None:
                    offspring.add(ind)

        return offspring

    def _tournament_select(self, population: Population, k: int = 2) -> Individual:
        """
        Binary tournament selection.

        Selects k random individuals and returns the best
        (lowest rank, highest crowding).

        Args:
            population: Population to select from
            k: Tournament size

        Returns:
            Selected individual
        """
        contestants = np.random.choice(len(population), size=k, replace=False)

        best = population[contestants[0]]
        for idx in contestants[1:]:
            other = population[idx]
            if other.rank < best.rank:
                best = other
            elif other.rank == best.rank and other.crowding > best.crowding:
                best = other

        return best

    def _sbx_crossover(
        self,
        parent1: np.ndarray,
        parent2: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        eta: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Simulated Binary Crossover (SBX).

        Creates two offspring from two parents using a probability
        distribution that mimics single-point crossover on binary strings.

        Args:
            parent1: First parent genes
            parent2: Second parent genes
            lower: Lower bounds
            upper: Upper bounds
            eta: Distribution index (higher = children closer to parents)

        Returns:
            Two child gene vectors
        """
        child1 = parent1.copy()
        child2 = parent2.copy()

        for i in range(len(parent1)):
            if np.random.random() > 0.5:
                continue

            if abs(parent1[i] - parent2[i]) < 1e-14:
                continue

            y1 = min(parent1[i], parent2[i])
            y2 = max(parent1[i], parent2[i])

            rand = np.random.random()

            # Compute beta
            beta = 1.0 + 2.0 * (y1 - lower[i]) / (y2 - y1)
            alpha = 2.0 - beta ** (-(eta + 1))
            if rand <= 1.0 / alpha:
                betaq = (rand * alpha) ** (1.0 / (eta + 1))
            else:
                betaq = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1))

            c1 = 0.5 * ((y1 + y2) - betaq * (y2 - y1))

            beta = 1.0 + 2.0 * (upper[i] - y2) / (y2 - y1)
            alpha = 2.0 - beta ** (-(eta + 1))
            if rand <= 1.0 / alpha:
                betaq = (rand * alpha) ** (1.0 / (eta + 1))
            else:
                betaq = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1))

            c2 = 0.5 * ((y1 + y2) + betaq * (y2 - y1))

            # Bound check
            child1[i] = np.clip(c1, lower[i], upper[i])
            child2[i] = np.clip(c2, lower[i], upper[i])

            # Random swap
            if np.random.random() > 0.5:
                child1[i], child2[i] = child2[i], child1[i]

        return child1, child2

    def _polynomial_mutation(
        self,
        genes: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        prob: float,
        eta: float,
    ) -> np.ndarray:
        """
        Polynomial mutation.

        Mutates genes with polynomial distribution, respecting bounds.

        Args:
            genes: Gene vector
            lower: Lower bounds
            upper: Upper bounds
            prob: Mutation probability per gene
            eta: Distribution index

        Returns:
            Mutated genes
        """
        mutant = genes.copy()

        for i in range(len(genes)):
            if np.random.random() >= prob:
                continue

            y = genes[i]
            delta1 = (y - lower[i]) / (upper[i] - lower[i])
            delta2 = (upper[i] - y) / (upper[i] - lower[i])

            rand = np.random.random()

            if rand < 0.5:
                xy = 1.0 - delta1
                val = 2.0 * rand + (1.0 - 2.0 * rand) * (xy ** (eta + 1))
                deltaq = val ** (1.0 / (eta + 1)) - 1.0
            else:
                xy = 1.0 - delta2
                val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (xy ** (eta + 1))
                deltaq = 1.0 - val ** (1.0 / (eta + 1))

            y = y + deltaq * (upper[i] - lower[i])
            mutant[i] = np.clip(y, lower[i], upper[i])

        return mutant

    def _reference_point_selection(
        self,
        combined: Population,
        fronts: list[list[int]],
        objectives: np.ndarray,
    ) -> Population:
        """
        Select next generation using reference point association.

        This is the core NSGA-III selection mechanism:
        1. Add fronts until population fills or exceeds size
        2. If last front causes overflow, use niching to select

        Args:
            combined: Combined parent+offspring population
            fronts: Non-dominated sort result
            objectives: Objective matrix

        Returns:
            Selected population
        """
        pop_size = self.config.population_size
        selected = Population()

        # Add complete fronts until we exceed population size
        last_front_idx = 0
        for front_idx, front in enumerate(fronts):
            if len(selected) + len(front) <= pop_size:
                for idx in front:
                    selected.add(combined[idx])
                last_front_idx = front_idx + 1
            else:
                break

        # If population not full, select from last front using niching
        remaining = pop_size - len(selected)
        if remaining > 0 and last_front_idx < len(fronts):
            last_front = fronts[last_front_idx]

            # Normalize objectives
            ideal = np.min(objectives, axis=0)
            nadir = np.max(objectives, axis=0)
            range_vec = nadir - ideal
            range_vec = np.where(range_vec > 1e-10, range_vec, 1.0)
            normalized = (objectives - ideal) / range_vec

            # Associate all solutions with reference points
            associations = self._associate_reference_points(
                normalized, self.reference_points
            )

            # Count niche for each reference point (from already selected)
            niche_count = np.zeros(len(self.reference_points), dtype=int)
            for idx in range(len(selected)):
                ref_idx = associations[idx]
                niche_count[ref_idx] += 1

            # Select from last front based on niche count
            front_associations = [associations[idx] for idx in last_front]
            selected_from_front = self._niche_selection(
                last_front,
                front_associations,
                niche_count,
                remaining,
                normalized,
            )

            for idx in selected_from_front:
                selected.add(combined[idx])

        # Assign crowding distances for backup selection
        selected_objectives = selected.get_objectives()
        selected_indices = list(range(len(selected)))
        distances = crowding_distance(selected_objectives, selected_indices)
        for i, dist in enumerate(distances):
            selected[i].crowding = dist

        return selected

    def _associate_reference_points(
        self,
        normalized: np.ndarray,
        reference_points: np.ndarray,
    ) -> np.ndarray:
        """
        Associate each solution with its nearest reference point.

        Uses perpendicular distance to reference line.

        Args:
            normalized: Normalized objective matrix
            reference_points: Reference point matrix

        Returns:
            Array of reference point indices for each solution
        """
        associations = np.zeros(len(normalized), dtype=int)

        for i, obj in enumerate(normalized):
            min_dist = np.inf
            nearest_ref = 0

            for j, ref in enumerate(reference_points):
                # Perpendicular distance to reference line
                ref_norm = np.linalg.norm(ref)
                if ref_norm < 1e-10:
                    continue

                # Project point onto reference line
                proj_length = np.dot(obj, ref) / ref_norm
                projection = proj_length * ref / ref_norm

                # Perpendicular distance
                dist = np.linalg.norm(obj - projection)

                if dist < min_dist:
                    min_dist = dist
                    nearest_ref = j

            associations[i] = nearest_ref

        return associations

    def _niche_selection(
        self,
        front: list[int],
        associations: list[int],
        niche_count: np.ndarray,
        k: int,
        normalized: np.ndarray,
    ) -> list[int]:
        """
        Select k solutions from front using niching.

        Preferentially selects from less crowded reference points.

        Args:
            front: Indices of solutions in last front
            associations: Reference point associations for front
            niche_count: Current niche counts
            k: Number to select
            normalized: Normalized objectives

        Returns:
            List of selected indices
        """
        selected = []
        available = list(range(len(front)))
        niche_count = niche_count.copy()

        while len(selected) < k and available:
            # Find reference points with minimum niche count among available
            available_refs = set(associations[i] for i in available)
            if not available_refs:
                break

            min_count = min(niche_count[r] for r in available_refs)
            min_refs = [r for r in available_refs if niche_count[r] == min_count]

            # Randomly select one of these reference points
            chosen_ref = np.random.choice(min_refs)

            # Find solutions associated with this reference point
            candidates = [i for i in available if associations[i] == chosen_ref]

            if not candidates:
                continue

            # Select the one with smallest perpendicular distance
            best_candidate = candidates[0]
            if len(candidates) > 1:
                best_dist = np.inf
                ref = self.reference_points[chosen_ref]
                ref_norm = np.linalg.norm(ref)

                for i in candidates:
                    idx = front[i]
                    obj = normalized[idx]

                    if ref_norm > 1e-10:
                        proj_length = np.dot(obj, ref) / ref_norm
                        projection = proj_length * ref / ref_norm
                        dist = np.linalg.norm(obj - projection)
                    else:
                        dist = np.linalg.norm(obj)

                    if dist < best_dist:
                        best_dist = dist
                        best_candidate = i

            selected.append(front[best_candidate])
            available.remove(best_candidate)
            niche_count[chosen_ref] += 1

        return selected

    def _record_statistics(self, population: Population) -> None:
        """Record generation statistics."""
        pareto = [ind for ind in population if ind.rank == 0]

        if len(pareto) > 0:
            F = np.array([ind.objectives for ind in pareto])
            frontier = ParetoFrontier.from_population(F)

            self._history['hypervolume'].append(frontier.compute_hypervolume())
            self._history['spacing'].append(frontier.spacing())
            self._history['pareto_size'].append(len(pareto))

            for i, obj in enumerate(self.objectives):
                self._history[f'min_{obj.name}'].append(float(np.min(F[:, i])))
                self._history[f'mean_{obj.name}'].append(float(np.mean(F[:, i])))


def create_trajectory_decoder(
    waypoint_template: list[np.ndarray],
    altitude_range: tuple[float, float] = (50.0, 500.0),
    time_scale: float = 1.0,
) -> Callable[[np.ndarray], Trajectory]:
    """
    Create a gene-to-trajectory decoder.

    Gene encoding:
    - Every 4 values: (lat_offset, lon_offset, alt, time_delta)

    Args:
        waypoint_template: Template waypoint positions
        altitude_range: (min, max) altitude
        time_scale: Time scaling factor

    Returns:
        Decoder function
    """
    from ..core.state import State, Pose, Velocity
    from ..core.trajectory import Trajectory

    def decoder(genes: np.ndarray) -> Trajectory:
        n_waypoints = len(genes) // 4
        states = []
        current_time = 0.0

        for i in range(n_waypoints):
            base = i * 4

            if i < len(waypoint_template):
                template = waypoint_template[i]
                lat = template[0] + genes[base] * 0.01  # Small offset
                lon = template[1] + genes[base + 1] * 0.01
            else:
                lat = genes[base] * 0.1
                lon = genes[base + 1] * 0.1

            alt = altitude_range[0] + genes[base + 2] * (altitude_range[1] - altitude_range[0])
            time_delta = max(1.0, genes[base + 3] * time_scale * 100)
            current_time += time_delta

            position = np.array([lat, lon, alt])
            pose = Pose(position=position)
            velocity = Velocity(linear=np.array([30.0, 0.0, 0.0]))

            state = State(
                pose=pose,
                velocity=velocity,
                energy=1.0 - current_time / 10000,
                time=current_time,
            )
            states.append(state)

        return Trajectory(states=states)

    return decoder
