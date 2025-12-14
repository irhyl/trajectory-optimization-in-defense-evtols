"""
A* Based Route Planner

Implements A* pathfinding with multi-objective optimization.
Integrates with perception layer for cost and feasibility queries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set
import heapq
import logging

import numpy as np

from ..base import RoutePlanner, Waypoint, RoutePlan
from ..config import PlanningConfig
from ..serving.perception_client import PerceptionClient

logger = logging.getLogger(__name__)


@dataclass(order=True)
class _AStarNode:
    """Internal node for A* search."""
    
    f_cost: float  # Total cost (g + h)
    waypoint: Waypoint = field(compare=False)
    g_cost: float = field(compare=False)
    h_cost: float = field(compare=False)
    parent: Optional[_AStarNode] = field(default=None, compare=False)


class AStarPlanner(RoutePlanner):
    """
    A* based route planner with multi-objective optimization.

    Features:
    - A* pathfinding with admissible heuristic
    - Multi-objective cost computation
    - Dynamic feasibility constraints
    - Perception layer integration
    - Automatic fallback to straight-line routing
    - Path smoothing for realistic trajectories

    Example:
        >>> config = PlanningConfig()
        >>> planner = AStarPlanner(config)
        >>> start = Waypoint(40.7128, -74.0060, 100)
        >>> goal = Waypoint(34.0522, -118.2437, 100)
        >>> plan = planner.plan(start, goal, "2024-01-01T12:00:00Z")
        >>> print(f"Distance: {plan.distance_km:.1f} km")
    """

    def __init__(self, config: PlanningConfig) -> None:
        """Initialize A* planner."""
        self.config = config
        
        # Initialize perception client with graceful fallback for testing
        try:
            self.perception = PerceptionClient(config)
        except (ImportError, ValueError):
            logger.warning("Perception layer not available; using fake-mode for testing.")
            self.perception = PerceptionClient(config, use_fake=True)
        
        # Grid resolution for search space (degrees)
        self.grid_resolution_deg = float(config.get("routing.grid_resolution_deg", 0.01))
        
        # Cost weights for multi-objective optimization
        self.cost_weights = config.get("routing.objective_weights", {
            "distance": 0.3,
            "energy": 0.3,
            "risk": 0.3,
            "time": 0.1
        })
        
        # Search parameters
        self.max_iterations = int(config.get("routing.max_iterations", 1000))
        self.goal_tolerance_km = float(config.get("routing.goal_tolerance_km", 0.5))
        
        # Smoothing
        self.smoothing_window = int(config.get("routing.smoothing_window", 5))

    def plan(
        self,
        start: Waypoint,
        goal: Waypoint,
        time_iso: str,
        constraints: Optional[Dict] = None
    ) -> RoutePlan:
        """
        Find optimal route from start to goal using A*.
        
        Args:
            start: Starting waypoint
            goal: Goal waypoint
            time_iso: ISO timestamp for environment state
            constraints: Optional dict with max_altitude, min_altitude, etc.
            
        Returns:
            RoutePlan with waypoints and cost metrics
            
        Raises:
            ValueError: If route is clearly infeasible
        """
        logger.info(f"Planning route from {start} to {goal}")
        
        # Run A* search
        waypoints = self._astar_search(start, goal, time_iso, constraints)
        
        if not waypoints:
            # Fallback to straight line
            logger.warning("A* search failed, using straight-line fallback")
            waypoints = self._straight_line_fallback(start, goal)
        
        # Smooth the path
        waypoints = self._smooth_path(waypoints)
        
        # Compute route metrics
        distance_km = self._compute_distance(waypoints)
        energy_kwh = self._compute_energy(waypoints, time_iso)
        risk_score = self._compute_risk(waypoints, time_iso)
        flight_time_s = self._compute_flight_time(waypoints)
        
        return RoutePlan(
            waypoints=waypoints,
            distance_km=distance_km,
            energy_kwh=energy_kwh,
            risk_score=risk_score,
            flight_time_s=flight_time_s,
            metadata={
                "algorithm": "A*",
                "grid_resolution_deg": self.grid_resolution_deg,
                "cost_weights": self.cost_weights
            }
        )

    def _astar_search(
        self,
        start: Waypoint,
        goal: Waypoint,
        time_iso: str,
        constraints: Optional[Dict]
    ) -> List[Waypoint]:
        """Run A* search algorithm."""
        
        # Initialize start node
        start_node = _AStarNode(
            f_cost=self._heuristic(start, goal),
            waypoint=start,
            g_cost=0.0,
            h_cost=self._heuristic(start, goal),
            parent=None
        )
        
        open_set = [start_node]
        closed_set: Set[tuple] = set()
        g_scores: Dict[tuple, float] = {(start.lat, start.lon, start.alt_m): 0.0}
        
        iterations = 0
        
        while open_set and iterations < self.max_iterations:
            iterations += 1
            
            current_node = heapq.heappop(open_set)
            current_wp = current_node.waypoint
            current_key = (current_wp.lat, current_wp.lon, current_wp.alt_m)
            
            # Goal check
            if self._is_goal(current_wp, goal):
                logger.info(f"Found path in {iterations} iterations")
                return self._reconstruct_path(current_node)
            
            # Skip if already visited
            if current_key in closed_set:
                continue
            closed_set.add(current_key)
            
            # Expand neighbors
            neighbors = self._get_neighbors(current_wp, constraints)
            
            for neighbor_wp in neighbors:
                neighbor_key = (neighbor_wp.lat, neighbor_wp.lon, neighbor_wp.alt_m)
                
                if neighbor_key in closed_set:
                    continue
                
                # Compute edge cost
                edge_cost = self._compute_edge_cost(current_wp, neighbor_wp, time_iso)
                
                # Skip infeasible edges
                if edge_cost < 0:
                    continue
                
                tentative_g = current_node.g_cost + edge_cost
                
                # Check if this is a better path
                if neighbor_key not in g_scores or tentative_g < g_scores[neighbor_key]:
                    g_scores[neighbor_key] = tentative_g
                    h_cost = self._heuristic(neighbor_wp, goal)
                    f_cost = tentative_g + h_cost
                    
                    neighbor_node = _AStarNode(
                        f_cost=f_cost,
                        waypoint=neighbor_wp,
                        g_cost=tentative_g,
                        h_cost=h_cost,
                        parent=current_node
                    )
                    heapq.heappush(open_set, neighbor_node)
        
        logger.warning(f"A* search failed after {iterations} iterations")
        return []

    def _get_neighbors(
        self,
        waypoint: Waypoint,
        constraints: Optional[Dict] = None
    ) -> List[Waypoint]:
        """Generate neighbor waypoints for expansion."""
        
        neighbors = []
        
        # 8-connected grid in lat/lon
        for dlat in [-1, 0, 1]:
            for dlon in [-1, 0, 1]:
                if dlat == 0 and dlon == 0:
                    continue
                
                new_lat = waypoint.lat + dlat * self.grid_resolution_deg
                new_lon = waypoint.lon + dlon * self.grid_resolution_deg
                
                # Maintain altitude (could add altitude variations here)
                new_alt = waypoint.alt_m
                
                # Check constraints
                if constraints:
                    min_alt = constraints.get("min_altitude_m", 50)
                    max_alt = constraints.get("max_altitude_m", 5000)
                    if not (min_alt <= new_alt <= max_alt):
                        continue
                
                neighbors.append(Waypoint(new_lat, new_lon, new_alt))
        
        return neighbors

    def _compute_edge_cost(
        self,
        from_wp: Waypoint,
        to_wp: Waypoint,
        time_iso: str
    ) -> float:
        """
        Compute multi-objective cost for edge.
        
        Returns:
            Cost value, or -1 if infeasible
        """
        try:
            # Query perception layer
            result = self.perception.query(
                to_wp.lat, to_wp.lon, to_wp.alt_m, time_iso
            )
            
            # Check feasibility
            if not getattr(result, 'feasible', True) is False:
                # Compute distance
                dist_km = to_wp.distance_to(from_wp) / 1000.0  # Convert m to km
                
                # Component costs
                distance_cost = dist_km
                energy_cost = getattr(result, 'energy_cost_kwh_per_km', 1.0) * dist_km
                risk_cost = getattr(result, 'risk_score', 0.0) * dist_km
                
                # Time cost (assume 35 m/s cruise speed)
                time_cost = (dist_km * 1000) / 35.0 / 3600.0
                
                # Weighted combination
                total_cost = (
                    self.cost_weights.get("distance", 0.3) * distance_cost +
                    self.cost_weights.get("energy", 0.3) * energy_cost +
                    self.cost_weights.get("risk", 0.3) * risk_cost +
                    self.cost_weights.get("time", 0.1) * time_cost
                )
                
                return total_cost
            else:
                return -1.0
                
        except Exception as e:
            logger.debug(f"Perception query failed: {e}")
            return 10.0

    def _heuristic(self, waypoint: Waypoint, goal: Waypoint) -> float:
        """
        Admissible heuristic using straight-line distance.
        
        Guarantees A* optimality by underestimating true cost.
        """
        dist_m = waypoint.distance_to(goal)
        dist_km = dist_m / 1000.0
        
        # Minimum cost per km (from weights)
        min_weight = min(self.cost_weights.values())
        
        return dist_km * min_weight

    def _is_goal(self, waypoint: Waypoint, goal: Waypoint) -> bool:
        """Check if waypoint is within goal tolerance."""
        dist_m = waypoint.distance_to(goal)
        dist_km = dist_m / 1000.0
        return dist_km < self.goal_tolerance_km

    def _reconstruct_path(self, node: _AStarNode) -> List[Waypoint]:
        """Reconstruct path from goal node to start."""
        path = []
        current = node
        
        while current is not None:
            path.append(current.waypoint)
            current = current.parent
        
        path.reverse()
        return path

    def _straight_line_fallback(self, start: Waypoint, goal: Waypoint) -> List[Waypoint]:
        """Create straight-line path as fallback."""
        num_points = max(25, int(start.distance_to(goal) / 1000))
        
        lats = np.linspace(start.lat, goal.lat, num_points)
        lons = np.linspace(start.lon, goal.lon, num_points)
        alts = np.linspace(start.alt_m, goal.alt_m, num_points)
        
        return [Waypoint(float(lat), float(lon), float(alt))
                for lat, lon, alt in zip(lats, lons, alts)]

    def _smooth_path(self, waypoints: List[Waypoint]) -> List[Waypoint]:
        """Smooth waypoint sequence using moving average."""
        
        window = self.smoothing_window
        if window < 3 or len(waypoints) <= window:
            return waypoints
        
        arr = np.array([[w.lat, w.lon, w.alt_m] for w in waypoints])
        kernel = np.ones(window) / window
        
        smoothed = np.vstack([
            np.convolve(arr[:, i], kernel, mode="same") for i in range(3)
        ]).T
        
        return [Waypoint(float(lat), float(lon), float(alt))
                for lat, lon, alt in smoothed]

    def _compute_distance(self, waypoints: List[Waypoint]) -> float:
        """Compute total distance along waypoint sequence."""
        if len(waypoints) < 2:
            return 0.0
        
        total_m = sum(
            waypoints[i].distance_to(waypoints[i + 1])
            for i in range(len(waypoints) - 1)
        )
        
        return total_m / 1000.0  # Convert to km

    def _compute_energy(self, waypoints: List[Waypoint], time_iso: str) -> float:
        """Estimate total energy consumption."""
        if len(waypoints) < 2:
            return 0.0
        
        total_energy = 0.0
        
        for i in range(len(waypoints) - 1):
            a, b = waypoints[i], waypoints[i + 1]
            dist_km = a.distance_to(b) / 1000.0
            
            try:
                result = self.perception.query(a.lat, a.lon, a.alt_m, time_iso)
                e_per_km = getattr(result, 'energy_cost_kwh_per_km', 1.0)
                total_energy += e_per_km * dist_km
            except:
                total_energy += 1.0 * dist_km  # Default fallback
        
        return total_energy

    def _compute_risk(self, waypoints: List[Waypoint], time_iso: str) -> float:
        """Compute average risk along route."""
        if len(waypoints) < 2:
            return 0.0
        
        total_risk_weighted = 0.0
        total_distance = 0.0
        
        for i in range(len(waypoints) - 1):
            a, b = waypoints[i], waypoints[i + 1]
            dist_km = a.distance_to(b) / 1000.0
            
            try:
                result = self.perception.query(a.lat, a.lon, a.alt_m, time_iso)
                risk = getattr(result, 'risk_score', 0.0)
                total_risk_weighted += risk * dist_km
            except:
                pass
            
            total_distance += dist_km
        
        if total_distance <= 0:
            return 0.0
        
        return min(total_risk_weighted / total_distance, 1.0)

    def _compute_flight_time(self, waypoints: List[Waypoint]) -> float:
        """Estimate flight time assuming constant cruise speed."""
        distance_km = self._compute_distance(waypoints)
        cruise_speed_m_s = float(self.config.get("energy.cruise_speed_mps", 35.0))
        
        # Convert km to m and compute time
        time_s = (distance_km * 1000.0) / cruise_speed_m_s
        
        return time_s


