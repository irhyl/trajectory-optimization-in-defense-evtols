"""
RRT* (Rapidly-exploring Random Tree*) Sampling-Based Planning

Implements RRT* for asymptotically-optimal path planning in high-dimensional
configuration spaces with probabilistic completeness and guaranteed convergence
to optimal solutions as iterations approach infinity.

References:
    - "Sampling-based Algorithms for Optimal Motion Planning" (Karaman & Frazzoli, 2011)
    - Time complexity: O(n log n) per iteration for KD-tree, O(n) per iteration for naive
    - Space complexity: O(n) for tree storage
"""

import numpy as np
import logging
import math
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from ..base import Waypoint, RoutePlanner, RoutePlan
from ..config import PlanningConfig

logger = logging.getLogger(__name__)


@dataclass
class RRTNode:
    """Node in RRT* tree"""
    waypoint: Waypoint
    cost: float  # Cost from root
    parent: Optional['RRTNode'] = None
    children: List['RRTNode'] = None
    
    def __post_init__(self):
        if self.children is None:
            self.children = []


class RRTStar(RoutePlanner):
    """
    RRT* (Asymptotically Optimal RRT) for high-dimensional trajectory planning.
    
    Key features:
    - Probabilistic completeness
    - Asymptotic optimality
    - Efficient sampling-based approach
    - Works well in high-dimensional spaces
    - Handles complex constraints naturally
    """
    
    def __init__(self, config: PlanningConfig):
        """Initialize RRT* planner"""
        self.config = config
        self.max_iterations = 1000
        self.goal_bias = 0.1  # Probability of sampling goal
        self.step_size = 0.01  # Grid step in degrees
        self.rewire_radius_factor = 40.0  # For rewiring neighbors
        
    def plan(
        self,
        start: Waypoint,
        goal: Waypoint,
        time_iso: str,
        constraints: Optional[Dict] = None,
        max_iterations: Optional[int] = None
    ) -> RoutePlan:
        """
        Find asymptotically-optimal path using RRT*.
        
        Args:
            start: Start waypoint
            goal: Goal waypoint
            time_iso: ISO timestamp
            constraints: Optional constraints
            max_iterations: Maximum iterations (overrides default)
            
        Returns:
            List of waypoints from start to goal
        """
        max_iter = max_iterations or self.max_iterations
        logger.info(f"RRT* planning with {max_iter} iterations")
        
        # Initialize tree with start node
        start_node = RRTNode(waypoint=start, cost=0.0, parent=None)
        nodes = [start_node]
        
        goal_node = None
        best_goal_cost = float('inf')
        
        # Main RRT* loop
        for iteration in range(max_iter):
            # Sample random point
            random_wp = self._sample_waypoint(goal, time_iso, constraints)
            
            # Find nearest node in tree
            nearest_node = self._nearest_node(random_wp, nodes)
            
            # Extend tree towards random point
            new_wp = self._steer(nearest_node.waypoint, random_wp, constraints)
            
            # Check collision (simplified - in production would use perception layer)
            if not self._is_collision_free(nearest_node.waypoint, new_wp, constraints):
                continue
            
            # Calculate cost to new node
            edge_cost = self._distance(nearest_node.waypoint, new_wp)
            new_cost = nearest_node.cost + edge_cost
            
            # Find neighbors within rewiring radius
            neighbors = self._find_neighbors(new_wp, nodes, constraints)
            
            # Find best parent among neighbors
            best_parent = nearest_node
            best_cost = new_cost
            
            for neighbor_node in neighbors:
                if self._is_collision_free(neighbor_node.waypoint, new_wp, constraints):
                    neighbor_cost = neighbor_node.cost + self._distance(neighbor_node.waypoint, new_wp)
                    
                    if neighbor_cost < best_cost:
                        best_parent = neighbor_node
                        best_cost = neighbor_cost
            
            # Create new node
            new_node = RRTNode(waypoint=new_wp, cost=best_cost, parent=best_parent)
            best_parent.children.append(new_node)
            nodes.append(new_node)
            
            # Rewire neighbors
            for neighbor_node in neighbors:
                if neighbor_node == best_parent:
                    continue
                
                new_neighbor_cost = new_node.cost + self._distance(new_node.waypoint, neighbor_node.waypoint)
                
                if new_neighbor_cost < neighbor_node.cost:
                    if self._is_collision_free(new_node.waypoint, neighbor_node.waypoint, constraints):
                        # Rewire: update neighbor's parent
                        if neighbor_node.parent:
                            neighbor_node.parent.children.remove(neighbor_node)
                        
                        neighbor_node.parent = new_node
                        new_node.children.append(neighbor_node)
                        neighbor_node.cost = new_neighbor_cost
            
            # Check if we reached goal
            if self._is_close_to_goal(new_wp, goal):
                if self._is_collision_free(new_wp, goal, constraints):
                    goal_cost = new_cost + self._distance(new_wp, goal)
                    
                    if goal_cost < best_goal_cost:
                        goal_node = RRTNode(waypoint=goal, cost=goal_cost, parent=new_node)
                        new_node.children.append(goal_node)
                        best_goal_cost = goal_cost
                        logger.info(f"Found goal at iteration {iteration}, cost: {goal_cost:.3f}")
            
            if iteration % 100 == 0:
                logger.debug(f"RRT* iteration {iteration}, tree size: {len(nodes)}")
        
        # Extract and return best path
        if goal_node:
            path = self._extract_path(goal_node)
            logger.info(f"RRT* found path with {len(path)} waypoints, cost: {best_goal_cost:.3f}")
            return self._to_route_plan(path, time_iso)
        else:
            logger.warning("RRT* failed to find path to goal")
            # Return best path towards goal as RoutePlan
            if nodes:
                closest_node = min(nodes, key=lambda n: self._distance(n.waypoint, goal))
                path = self._extract_path(closest_node)
                return self._to_route_plan(path, time_iso)

            return RoutePlan(waypoints=[], distance_km=0.0, energy_kwh=0.0, risk_score=0.0, flight_time_s=0.0, metadata={"algorithm": "RRT*", "status": "failed"})
    
    def _sample_waypoint(
        self,
        goal: Waypoint,
        time_iso: str,
        constraints: Optional[Dict]
    ) -> Waypoint:
        """Sample random waypoint (with goal bias)"""
        if np.random.random() < self.goal_bias:
            return goal
        
        # Random sampling in state space
        # Bounds for India
        lat = np.random.uniform(8, 36)
        lon = np.random.uniform(65, 97)
        
        # Altitude respecting constraints
        min_alt = constraints.get('min_altitude_m', 50) if constraints else 50
        max_alt = constraints.get('max_altitude_m', 5000) if constraints else 5000
        alt = np.random.uniform(min_alt, max_alt)
        
        return Waypoint(lat, lon, alt)
    
    def _nearest_node(self, waypoint: Waypoint, nodes: List[RRTNode]) -> RRTNode:
        """Find nearest node in tree (naive O(n), could use KD-tree for O(log n))"""
        nearest = nodes[0]
        min_dist = self._distance(waypoint, nearest.waypoint)
        
        for node in nodes[1:]:
            dist = self._distance(waypoint, node.waypoint)
            if dist < min_dist:
                min_dist = dist
                nearest = node
        
        return nearest
    
    def _steer(
        self,
        from_wp: Waypoint,
        to_wp: Waypoint,
        constraints: Optional[Dict]
    ) -> Waypoint:
        """Steer from one waypoint towards another by step_size"""
        dist = self._distance(from_wp, to_wp)
        
        if dist < self.step_size:
            return to_wp
        
        # Unit direction vector
        dlat = (to_wp.lat - from_wp.lat) / dist
        dlon = (to_wp.lon - from_wp.lon) / dist
        dalt = (to_wp.alt_m - from_wp.alt_m) / dist
        
        # Step in direction
        new_lat = from_wp.lat + dlat * self.step_size
        new_lon = from_wp.lon + dlon * self.step_size
        new_alt = from_wp.alt_m + dalt * self.step_size
        
        # Respect altitude constraints
        if constraints:
            min_alt = constraints.get('min_altitude_m', 50)
            max_alt = constraints.get('max_altitude_m', 5000)
            new_alt = max(min_alt, min(max_alt, new_alt))
        
        return Waypoint(new_lat, new_lon, new_alt)
    
    def _is_collision_free(
        self,
        wp1: Waypoint,
        wp2: Waypoint,
        constraints: Optional[Dict]
    ) -> bool:
        """Check if path between waypoints is collision-free"""
        # Simplified collision check
        # In production, would query perception layer for obstacles
        
        if constraints:
            min_alt = constraints.get('min_altitude_m', 50)
            max_alt = constraints.get('max_altitude_m', 5000)
            
            # Intermediate point check
            if wp1.alt_m < min_alt or wp1.alt_m > max_alt:
                return False
            if wp2.alt_m < min_alt or wp2.alt_m > max_alt:
                return False
        
        return True
    
    def _find_neighbors(
        self,
        waypoint: Waypoint,
        nodes: List[RRTNode],
        constraints: Optional[Dict],
        radius_scale: float = 1.0
    ) -> List[RRTNode]:
        """Find neighbors within rewiring radius"""
        # Rewiring radius decreases with tree size (theoretical property of RRT*)
        dimension = 3  # lat, lon, alt
        tree_size = len(nodes)
        
        radius = self.rewire_radius_factor * math.sqrt(math.log(tree_size) / tree_size)
        radius *= radius_scale
        
        neighbors = []
        for node in nodes:
            dist = self._distance(waypoint, node.waypoint)
            if dist < radius:
                neighbors.append(node)
        
        return neighbors
    
    def _is_close_to_goal(self, waypoint: Waypoint, goal: Waypoint, tolerance: float = 0.01) -> bool:
        """Check if waypoint is close to goal"""
        dist = self._distance(waypoint, goal)
        return dist < tolerance
    
    def _distance(self, wp1: Waypoint, wp2: Waypoint) -> float:
        """Euclidean distance in (lat, lon, alt) space"""
        lat_diff = (wp2.lat - wp1.lat) * 111.0  # km per degree
        lon_diff = (wp2.lon - wp1.lon) * 111.0 * math.cos(math.radians(wp1.lat))
        alt_diff = (wp2.alt_m - wp1.alt_m) / 1000.0  # Convert to km
        
        distance = math.sqrt(lat_diff**2 + lon_diff**2 + alt_diff**2)
        return distance
    
    def _extract_path(self, node: RRTNode) -> List[Waypoint]:
        """Extract path from root to given node"""
        path = []
        current = node
        
        while current is not None:
            path.insert(0, current.waypoint)
            current = current.parent
        
        return path
    
    def _to_route_plan(self, waypoints: List[Waypoint], time_iso: str) -> RoutePlan:
        """Convert waypoint list to RoutePlan with simple metrics."""
        if not waypoints:
            return RoutePlan(waypoints=[], distance_km=0.0, energy_kwh=0.0, risk_score=0.0, flight_time_s=0.0, metadata={"algorithm": "RRT*"})

        total_m = 0.0
        for i in range(len(waypoints) - 1):
            # Use Waypoint.distance_to for robust geodesic distance (returns meters)
            total_m += waypoints[i].distance_to(waypoints[i + 1])
        distance_km = total_m / 1000.0

        default_e_per_km = float(self.config.get("energy.default_per_km", 1.0)) if hasattr(self.config, 'get') else 1.0
        energy_kwh = distance_km * default_e_per_km

        # Placeholder risk
        risk_score = 0.0

        cruise_speed_m_s = float(self.config.get("energy.cruise_speed_mps", 35.0)) if hasattr(self.config, 'get') else 35.0
        flight_time_s = (distance_km * 1000.0) / cruise_speed_m_s if cruise_speed_m_s > 0 else 0.0

        return RoutePlan(
            waypoints=waypoints,
            distance_km=distance_km,
            energy_kwh=energy_kwh,
            risk_score=risk_score,
            flight_time_s=flight_time_s,
            metadata={"algorithm": "RRT*"}
        )
