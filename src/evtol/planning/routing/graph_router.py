from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import math
import networkx as nx

from ..config import PlanningConfig
from ..serving.perception_client import PerceptionClient, PerceptionQuery
from ..base import Waypoint, RoutePlanner


@dataclass
class GridBounds:
    min_lat: float
    min_lon: float
    max_lat: float
    max_lon: float


class GraphRoutePlanner:
    """Grid-based multi-objective routing using NetworkX with alternative routes.

    Builds a lat/lon grid over AOI, assigns edge weights from time/energy/risk, and
    computes k-shortest paths using edge weights. This is a baseline implementation
    intended to be swapped with a more sophisticated graph.
    """

    def __init__(self, config: PlanningConfig) -> None:
        self.config = config
        self.perception = PerceptionClient(config)

    def _edge_cost(self, a: Tuple[float, float], b: Tuple[float, float], alt_m: float, time_iso: str) -> float:
        # Compute segment cost using configured objective weights
        weights = self.config.get("routing.objective_weights", {"time": 0.4, "energy": 0.3, "risk": 0.3})
        cruise_speed_mps = float(self.config.get("energy.cruise_speed_mps", 35.0))
        query = PerceptionQuery(a[0], a[1], alt_m, time_iso)
        result = self.perception.query(query)

        # Distance
        d_km = self._haversine_km(a[0], a[1], b[0], b[1])
        # Time
        time_s = (d_km * 1000.0) / max(1e-6, cruise_speed_mps)
        time_norm = time_s / 3600.0
        # Energy
        e_kwh = result.energy_cost_kwh_per_km * d_km
        # Risk
        r = result.risk_score

        return weights.get("time", 0.4) * time_norm + weights.get("energy", 0.3) * e_kwh + weights.get("risk", 0.3) * r
    
    def _haversine_km(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate Haversine distance in kilometers"""
        R = 6371.0  # Earth radius in km
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = (math.sin(dlat/2)**2 + 
             math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * 
             math.sin(dlon/2)**2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        return R * c

    def _grid_points(self, bounds: GridBounds, lat_steps: int, lon_steps: int) -> List[Tuple[float, float]]:
        lats = [bounds.min_lat + i * (bounds.max_lat - bounds.min_lat) / max(1, lat_steps - 1) for i in range(lat_steps)]
        lons = [bounds.min_lon + j * (bounds.max_lon - bounds.min_lon) / max(1, lon_steps - 1) for j in range(lon_steps)]
        return [(lat, lon) for lat in lats for lon in lons]

    def _nearest_grid_node(self, G: nx.Graph, lat: float, lon: float) -> Tuple[float, float]:
        best = None
        best_d = float("inf")
        for n_lat, n_lon in G.nodes:
            d = (n_lat - lat) ** 2 + (n_lon - lon) ** 2
            if d < best_d:
                best_d = d
                best = (n_lat, n_lon)
        assert best is not None
        return best

    def plan_with_alternatives(
        self,
        bounds: GridBounds,
        start_lat: float,
        start_lon: float,
        goal_lat: float,
        goal_lon: float,
        alt_m: float,
        time_iso: str,
        lat_steps: int | None = None,
        lon_steps: int | None = None,
        k: int | None = None,
    ) -> List[List[Waypoint]]:
        """Helper: build grid from config defaults and return up to k routes.

        Falls back to config keys:
          - routing.allow_alternatives (bool)
          - routing.num_alternatives (int)
        """
        lat_steps_i = int(lat_steps) if lat_steps is not None else int(self.config.get("grid.lat_steps", 21))
        lon_steps_i = int(lon_steps) if lon_steps is not None else int(self.config.get("grid.lon_steps", 21))
        G = self.build_grid_graph(bounds, lat_steps=lat_steps_i, lon_steps=lon_steps_i, alt_m=alt_m, time_iso=time_iso)

        allow = bool(self.config.get("routing.allow_alternatives", True))
        k_eff = int(k) if k is not None else int(self.config.get("routing.num_alternatives", 3))
        k_eff = max(1, k_eff if allow else 1)
        return self.k_shortest_routes(G, start_lat, start_lon, goal_lat, goal_lon, k=k_eff)

    def build_grid_graph(
        self,
        bounds: GridBounds,
        lat_steps: int,
        lon_steps: int,
        alt_m: float,
        time_iso: str,
    ) -> nx.Graph:
        G = nx.Graph()
        points = self._grid_points(bounds, lat_steps, lon_steps)
        for p in points:
            G.add_node(p)

        # 8-neighborhood connectivity
        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

        def idx(i: int, j: int) -> int:
            return i * lon_steps + j

        for i in range(lat_steps):
            for j in range(lon_steps):
                a = points[idx(i, j)]
                for di, dj in directions:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < lat_steps and 0 <= nj < lon_steps:
                        b = points[idx(ni, nj)]
                        # feasibility check
                        query = PerceptionQuery(a[0], a[1], alt_m, time_iso)
                        result = self.perception.query(query)
                        if not result.feasible:
                            continue
                        cost = self._edge_cost(a, b, alt_m, time_iso)
                        G.add_edge(a, b, weight=cost)
        return G

    def shortest_route(
        self,
        G: nx.Graph,
        start_lat: float,
        start_lon: float,
        goal_lat: float,
        goal_lon: float,
    ) -> List[Waypoint]:
        s = self._nearest_grid_node(G, start_lat, start_lon)
        t = self._nearest_grid_node(G, goal_lat, goal_lon)
        path = nx.shortest_path(G, source=s, target=t, weight="weight")
        return [Waypoint(lat=p[0], lon=p[1], alt_m=0.0) for p in path]

    def k_shortest_routes(
        self,
        G: nx.Graph,
        start_lat: float,
        start_lon: float,
        goal_lat: float,
        goal_lon: float,
        k: int = 3,
    ) -> List[List[Waypoint]]:
        s = self._nearest_grid_node(G, start_lat, start_lon)
        t = self._nearest_grid_node(G, goal_lat, goal_lon)
        gen = nx.shortest_simple_paths(G, s, t, weight="weight")
        routes: List[List[Waypoint]] = []
        for _ in range(k):
            try:
                path = next(gen)
            except StopIteration:
                break
            routes.append([Waypoint(lat=p[0], lon=p[1], alt_m=0.0) for p in path])
        return routes




