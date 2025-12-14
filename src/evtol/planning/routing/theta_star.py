"""
Theta* Any-Angle Pathfinding Algorithm

Implements Theta* for any-angle movement on grids with reduced path cost
compared to standard A* by allowing diagonal movement without grid locking.

References:
    - "Theta*: Any-Angle Path Planning on Grids" (Nash et al., 2007)
    - Time complexity: O((V+E) log V), often better than A* in practice
"""

import heapq
import math
import logging
from typing import List, Dict, Optional, Set, Tuple
from dataclasses import dataclass, field
from ..base import Waypoint, RoutePlanner, RoutePlan
from ..config import PlanningConfig

logger = logging.getLogger(__name__)


@dataclass
class ThetaNode:
    """Node for Theta* search"""
    f_cost: float
    waypoint: Waypoint
    g_cost: float
    h_cost: float
    parent: Optional['ThetaNode'] = None
    
    def __lt__(self, other):
        return self.f_cost < other.f_cost


class ThetaStar(RoutePlanner):
    """
    Theta* pathfinding algorithm for any-angle trajectory planning.
    
    Key differences from A*:
    - Allows any-angle movement (not grid-locked)
    - Uses line-of-sight checks between non-adjacent nodes
    - Produces smoother, shorter paths with fewer waypoints
    """
    
    def __init__(self, config: PlanningConfig):
        """Initialize Theta* planner"""
        self.config = config
        self.grid_resolution = 0.01  # degrees
        self.max_iterations = 1000
        self.grid_width = 100
        self.grid_height = 100
        
    def plan(
        self,
        start: Waypoint,
        goal: Waypoint,
        time_iso: str,
        constraints: Optional[Dict] = None
    ) -> RoutePlan:
        """
        Find any-angle optimal path using Theta*.
        
        Args:
            start: Start waypoint
            goal: Goal waypoint
            time_iso: ISO timestamp for environment state
            constraints: Optional constraints (altitude, clearance, etc.)
            
        Returns:
            List of waypoints from start to goal
        """
        logger.info(f"Theta* planning from ({start.lat:.4f}, {start.lon:.4f}) to ({goal.lat:.4f}, {goal.lon:.4f})")
        
        # Priority queue: (f_cost, node)
        open_set = []
        start_node = ThetaNode(
            f_cost=0.0,
            waypoint=start,
            g_cost=0.0,
            h_cost=self._heuristic(start, goal),
            parent=None
        )
        heapq.heappush(open_set, start_node)
        
        # Tracking
        closed_set: Set[Tuple[float, float]] = set()
        g_scores: Dict[Tuple[float, float], float] = {(start.lat, start.lon): 0.0}
        parent_map: Dict[Tuple[float, float], ThetaNode] = {(start.lat, start.lon): start_node}
        
        iterations = 0
        
        while open_set and iterations < self.max_iterations:
            iterations += 1
            current = heapq.heappop(open_set)
            
            # Goal reached
            if self._is_goal(current.waypoint, goal):
                logger.info(f"Theta* found path in {iterations} iterations")
                path = self._reconstruct_path(current)
                return self._to_route_plan(path, time_iso)
            
            closed_key = (current.waypoint.lat, current.waypoint.lon)
            if closed_key in closed_set:
                continue
            closed_set.add(closed_key)
            
            # Get neighbors (8-connected with any-angle option)
            neighbors = self._get_neighbors(current.waypoint, goal, constraints)
            
            for neighbor_wp in neighbors:
                neighbor_key = (neighbor_wp.lat, neighbor_wp.lon)
                
                if neighbor_key in closed_set:
                    continue
                
                # Theta* key step: line-of-sight check
                if current.parent is not None:
                    parent_wp = current.parent.waypoint
                    
                    # Check if we can go directly from parent to neighbor (any-angle)
                    if self._line_of_sight(parent_wp, neighbor_wp, constraints):
                        # Direct path cost
                        tentative_g = current.parent.g_cost + self._distance(parent_wp, neighbor_wp)
                        parent_for_neighbor = current.parent
                    else:
                        # Path through current node
                        tentative_g = current.g_cost + self._distance(current.waypoint, neighbor_wp)
                        parent_for_neighbor = current
                else:
                    # Initial path
                    tentative_g = current.g_cost + self._distance(current.waypoint, neighbor_wp)
                    parent_for_neighbor = current
                
                # Check if this path is better
                if neighbor_key not in g_scores or tentative_g < g_scores[neighbor_key]:
                    g_scores[neighbor_key] = tentative_g
                    h_cost = self._heuristic(neighbor_wp, goal)
                    f_cost = tentative_g + h_cost
                    
                    neighbor_node = ThetaNode(
                        f_cost=f_cost,
                        waypoint=neighbor_wp,
                        g_cost=tentative_g,
                        h_cost=h_cost,
                        parent=parent_for_neighbor
                    )
                    parent_map[neighbor_key] = neighbor_node
                    heapq.heappush(open_set, neighbor_node)
        
        logger.warning(f"Theta* failed to find path after {iterations} iterations")
        return RoutePlan(waypoints=[], distance_km=0.0, energy_kwh=0.0, risk_score=0.0, flight_time_s=0.0, metadata={"algorithm": "Theta*", "status": "failed"})
    
    def _line_of_sight(self, wp1: Waypoint, wp2: Waypoint, constraints: Optional[Dict]) -> bool:
        """
        Check if there's a clear line of sight between two waypoints.
        
        Uses Bresenham-style line checking on grid with collision checks.
        """
        # For now, assume no obstacles between points
        # In production, would check against obstacle map from perception layer
        
        # Altitude constraint check
        if constraints:
            min_alt = constraints.get('min_altitude_m', 50)
            max_alt = constraints.get('max_altitude_m', 5000)
            
            if wp1.alt_m < min_alt or wp1.alt_m > max_alt:
                return False
            if wp2.alt_m < min_alt or wp2.alt_m > max_alt:
                return False
        
        return True
    
    def _get_neighbors(
        self,
        wp: Waypoint,
        goal: Waypoint,
        constraints: Optional[Dict]
    ) -> List[Waypoint]:
        """Get neighbor waypoints (8-connected grid)"""
        neighbors = []
        
        # 8-connected neighbors (horizontal, vertical, diagonal)
        directions = [
            (0.01, 0, 0), (0, 0.01, 0), (-0.01, 0, 0), (0, -0.01, 0),
            (0.01, 0.01, 0), (0.01, -0.01, 0), (-0.01, 0.01, 0), (-0.01, -0.01, 0)
        ]
        
        for dlat, dlon, dalt in directions:
            new_lat = wp.lat + dlat
            new_lon = wp.lon + dlon
            new_alt = wp.alt_m + dalt
            
            # Bounds check (rough bounds for India)
            if 8 <= new_lat <= 36 and 65 <= new_lon <= 97:
                neighbor = Waypoint(new_lat, new_lon, new_alt)
                neighbors.append(neighbor)
        
        return neighbors
    
    def _heuristic(self, current: Waypoint, goal: Waypoint) -> float:
        """Euclidean distance heuristic"""
        lat_diff = (goal.lat - current.lat) * 111.0  # km per degree latitude
        lon_diff = (goal.lon - current.lon) * 111.0 * math.cos(math.radians(current.lat))
        alt_diff = (goal.alt_m - current.alt_m) / 1000.0
        
        distance = math.sqrt(lat_diff**2 + lon_diff**2 + alt_diff**2)
        return distance
    
    def _distance(self, wp1: Waypoint, wp2: Waypoint) -> float:
        """Euclidean distance between waypoints"""
        lat_diff = (wp2.lat - wp1.lat) * 111.0
        lon_diff = (wp2.lon - wp1.lon) * 111.0 * math.cos(math.radians(wp1.lat))
        alt_diff = (wp2.alt_m - wp1.alt_m) / 1000.0
        
        distance = math.sqrt(lat_diff**2 + lon_diff**2 + alt_diff**2)
        return distance
    
    def _is_goal(self, current: Waypoint, goal: Waypoint, tolerance: float = 0.001) -> bool:
        """Check if current waypoint is at goal"""
        lat_err = abs(current.lat - goal.lat)
        lon_err = abs(current.lon - goal.lon)
        alt_err = abs(current.alt_m - goal.alt_m)
        
        return lat_err < tolerance and lon_err < tolerance and alt_err < 10
    
    def _reconstruct_path(self, node: ThetaNode) -> List[Waypoint]:
        """Reconstruct path from goal to start following parent pointers"""
        path = []
        current = node
        
        while current is not None:
            path.insert(0, current.waypoint)
            current = current.parent
        
        # Ensure start and goal are in path
        if path:
            logger.info(f"Theta* path: {len(path)} waypoints")
        
        return path
    
    def _to_route_plan(self, waypoints: List[Waypoint], time_iso: str) -> RoutePlan:
        """Convert waypoint list to RoutePlan with simple cost estimates."""
        if not waypoints:
            return RoutePlan(waypoints=[], distance_km=0.0, energy_kwh=0.0, risk_score=0.0, flight_time_s=0.0, metadata={"algorithm": "Theta*"})

        # Compute distance using Waypoint.distance_to (meters -> km)
        total_m = 0.0
        for i in range(len(waypoints) - 1):
            total_m += waypoints[i].distance_to(waypoints[i + 1])
        distance_km = total_m / 1000.0

        # Energy estimate: config-provided default per-km or fallback
        default_e_per_km = float(self.config.get("energy.default_per_km", 1.0)) if hasattr(self.config, 'get') else 1.0
        energy_kwh = distance_km * default_e_per_km

        # Risk: placeholder average (no perception client here)
        risk_score = 0.0

        # Flight time: use cruise speed from config or sensible default
        cruise_speed_m_s = float(self.config.get("energy.cruise_speed_mps", 35.0)) if hasattr(self.config, 'get') else 35.0
        flight_time_s = (distance_km * 1000.0) / cruise_speed_m_s if cruise_speed_m_s > 0 else 0.0

        return RoutePlan(
            waypoints=waypoints,
            distance_km=distance_km,
            energy_kwh=energy_kwh,
            risk_score=risk_score,
            flight_time_s=flight_time_s,
            metadata={"algorithm": "Theta*"}
        )
