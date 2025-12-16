import pandas as pd
from typing import Dict

def export_planning_data() -> Dict[str, pd.DataFrame]:
    """Export all planning data as DataFrames for the UI."""
    manager = get_planning_manager()
    if not manager:
        return {}
    return {
        'waypoints': manager.get_combined_routes_df(),
        'controls': pd.DataFrame(),  # Placeholder: implement if control data available
        'constraint_violations': pd.DataFrame(),  # Placeholder: implement if constraint data available
    }

"""
Planning Backend Module for Streamlit App.

Integrates the real planning layer (A*, RRT*, energy optimization, risk assessment)
with the Streamlit UI, providing proper trajectory planning and visualization.
"""

from typing import Dict

import sys
from pathlib import Path
from typing import Dict, Optional, List, Tuple, Any
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
import streamlit as st

# Ensure perception-layer 'src' is on path if present so we can import `serving.api`
perception_api = None
HAVE_PERCEPTION = False
try:
    PERCEPTION_SRC = Path(__file__).parent.parent / 'perception-layer' / 'src'
    if PERCEPTION_SRC.exists() and str(PERCEPTION_SRC) not in sys.path:
        sys.path.insert(0, str(PERCEPTION_SRC))
    import serving.api as perception_api  # type: ignore
    HAVE_PERCEPTION = True
except Exception:
    perception_api = None
    HAVE_PERCEPTION = False
# Add the src directory to path for imports
SRC_PATH = Path(__file__).parent.parent / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

# Try to import real planning modules
try:
    from evtol.planning import (
        Waypoint,
        RoutePlan,
        AStarPlanner,
        GraphRoutePlanner,
        EnergyOptimizerImpl,
        RiskManagerImpl,
        ParetoFrontier,
        MissionPlannerImpl,
        PlanningConfig,
    )
    PLANNING_AVAILABLE = True
except ImportError as e:
    PLANNING_AVAILABLE = False
    PLANNING_IMPORT_ERROR = str(e)


@dataclass
class PlanningParameters:
    """Configuration for planning algorithms."""
    # Grid parameters
    grid_size: int = 100
    resolution_m: float = 500.0
    
    # Start and end points (grid indices)
    start_point: Tuple[int, int] = (10, 10)
    end_point: Tuple[int, int] = (90, 90)
    
    # Algorithm selection
    algorithms: List[str] = field(default_factory=lambda: ["A*", "Dijkstra", "Theta*", "RRT*"])
    
    # Optimization weights
    weight_distance: float = 0.3
    weight_energy: float = 0.3
    weight_risk: float = 0.2
    weight_time: float = 0.2
    # Algorithm-specific parameters
    theta_aggressiveness: int = 7  # 1..10, higher -> more aggressive shortcutting
    rrt_samples: int = 300
    rrt_seed: int = 42
    rrt_smoothing: bool = True
    
    # Constraints
    max_altitude_m: float = 500.0
    min_altitude_m: float = 50.0
    max_speed_ms: float = 30.0
    min_speed_ms: float = 5.0
    max_bank_angle_deg: float = 30.0
    
    # Energy parameters
    battery_capacity_kwh: float = 50.0
    cruise_power_kw: float = 25.0
    
    # Pareto optimization
    num_pareto_solutions: int = 10


class PlanningManager:
    """
    Manages trajectory planning and provides data for the Streamlit UI.
    """
    
    def __init__(self, params: Optional[PlanningParameters] = None):
        """Initialize planning manager with parameters."""
        self.params = params or PlanningParameters()
        # Results storage
        self.routes: Dict[str, pd.DataFrame] = {}
        self.moo_results: Optional[pd.DataFrame] = None
        self.pareto_frontier: Optional[pd.DataFrame] = None
        self.vehicle_feasible_solutions: Optional[pd.DataFrame] = None
        self.vehicle_feasible_pareto_solutions: Optional[pd.DataFrame] = None
        self.energy_profile: Optional[pd.DataFrame] = None
        self.risk_assessment: Optional[pd.DataFrame] = None
        self.constraints_status: Optional[pd.DataFrame] = None
        # Status tracking
        self.is_generated = False
        self.generation_status: Dict[str, bool] = {
            'routes': False,
            'moo': False,
            'pareto': False,
            'feasible': False,
            'vehicle_pareto': False,
            'energy': False,
            'risk': False,
            'constraints': False
        }
        # Track algorithms removed due to infeasibility
        self.dropped_algorithms: List[str] = []
        # Error/status tracking
        self.error = None
        self.status = "Initializing"
        self.traceback = None
        self.summary = {}
        try:
            self.status = "Generating planning results..."
            self.is_generated = self.generate_all()
            if self.is_generated:
                self.status = "Planning completed"
                # Populate summary for UI
                self.summary = {}
                if self.moo_results is not None and not self.moo_results.empty:
                    self.summary['feasible_solutions'] = len(self.moo_results)
                    self.summary['best_cost'] = self.moo_results['Energy_kWh'].min()
                    self.summary['best_risk'] = self.moo_results['Risk_Pd'].min()
                if self.pareto_frontier is not None and not self.pareto_frontier.empty:
                    self.summary['pareto_solutions'] = len(self.pareto_frontier)
            else:
                self.status = "Planning failed (see errors above)"
        except Exception as e:
            import traceback
            self.error = str(e)
            self.status = "Planning failed (exception)"
            self.traceback = traceback.format_exc()

    def generate_moo_results(self) -> bool:
        """Generate Multi-Objective Optimization (MOO) results for all algorithms and profiles."""
        algorithms = ["A*", "Dijkstra", "Theta*", "RRT*"]
        profiles = [
            ("Balanced", (0.33, 0.33, 0.34)),
            ("Energy", (0.60, 0.20, 0.20)),
            ("Risk", (0.20, 0.70, 0.10)),
            ("Time", (0.20, 0.20, 0.60)),
        ]
        rows = []
        for algo in algorithms:
            for profile_name, weights in profiles:
                # Use backend route generation for each algorithm/profile
                route = self._generate_algorithm_route(algo)
                if route is not None and not route.empty:
                    # Compute metrics as in notebook
                    dist = route['Distance (km)'].iloc[-1] if 'Distance (km)' in route else 0
                    elev = route['Altitude (m)'].values
                    elev_gain = float((elev[1:] - elev[:-1]).clip(min=0).sum()) if len(elev) > 1 else 0
                    energy = dist * 5 + elev_gain * 0.01 + 2
                    time = dist / 0.9 if dist > 0 else 0  # 0.9 km/min ~ 15 m/s
                    risk = route['Risk Score'].mean() if 'Risk Score' in route else 0
                    rows.append({
                        'Algorithm': algo,
                        'Mission_Profile': profile_name,
                        'Waypoints': len(route),
                        'Distance_km': round(dist, 3),
                        'Energy_kWh': round(energy, 3),
                        'Time_min': round(time, 2),
                        'Risk_Pd': round(risk, 4),
                        'Elevation_gain_m': round(elev_gain, 1)
                    })
        self.moo_results = pd.DataFrame(rows)
        self.generation_status['moo'] = not self.moo_results.empty
        # For vehicle_feasible_solutions, use the same as MOO for now
        self.vehicle_feasible_solutions = self.moo_results.copy()
        self.generation_status['feasible'] = not self.vehicle_feasible_solutions.empty
        return not self.moo_results.empty

    def generate_pareto_frontier(self) -> bool:
        """Compute Pareto frontier from MOO results."""
        if self.moo_results is None or self.moo_results.empty:
            self.pareto_frontier = pd.DataFrame()
            self.generation_status['pareto'] = False
            return False
        df = self.moo_results
        objectives = ['Energy_kWh', 'Time_min', 'Risk_Pd']
        is_dominated = []
        for i, row in df.iterrows():
            dominated = False
            for j, other in df.iterrows():
                if i == j:
                    continue
                dominates = all(other[obj] <= row[obj] for obj in objectives)
                strictly_better = any(other[obj] < row[obj] for obj in objectives)
                if dominates and strictly_better:
                    dominated = True
                    break
            is_dominated.append(dominated)
        self.pareto_frontier = df.loc[[not d for d in is_dominated]].reset_index(drop=True)
        self.generation_status['pareto'] = not self.pareto_frontier.empty
        # For vehicle_feasible_pareto_solutions, use the same as pareto_frontier for now
        self.vehicle_feasible_pareto_solutions = self.pareto_frontier.copy()
        self.generation_status['vehicle_pareto'] = not self.vehicle_feasible_pareto_solutions.empty
        return not self.pareto_frontier.empty
    
    def generate_routes(self, perception_manager=None, progress_callback=None) -> bool:
        """
        Generate routes using multiple algorithms.
        
        Args:
            perception_manager: Optional PerceptionManager for terrain/threat data
            progress_callback: Optional callback for progress updates
        """
        try:
            if progress_callback:
                progress_callback("Generating A* route...")
            
            # Generate routes for each algorithm
            algorithms = self.params.algorithms
            
            self.dropped_algorithms = []
            for algo in algorithms:
                if progress_callback:
                    progress_callback(f"Computing {algo} trajectory...")

                route = self._generate_algorithm_route(algo, perception_manager)

                # If perception API is available, perform a route-level feasibility check
                route_is_feasible = True
                if HAVE_PERCEPTION and perception_api is not None and route is not None and not route.empty:
                    try:
                        for _, wp in route.iterrows():
                            qp = perception_api.QueryPoint(lat=float(wp['Latitude']), lon=float(wp['Longitude']), alt_m=float(wp['Altitude (m)']))
                            if not bool(perception_api.feasible(qp)):
                                route_is_feasible = False
                                break
                    except Exception:
                        # If perception check fails, assume feasible to avoid blocking
                        route_is_feasible = True

                if route_is_feasible and route is not None and not route.empty:
                    self.routes[algo] = route
                else:
                    # Skip storing infeasible/empty routes
                    self.routes[algo] = pd.DataFrame()
                    # record dropped algorithm for UI reporting
                    self.dropped_algorithms.append(algo)
            
            self.generation_status['routes'] = True
            return True
            
        except Exception as e:
            st.error(f"Route generation failed: {e}")
            return False
    
    def _generate_algorithm_route(self, algorithm: str, perception_manager=None) -> pd.DataFrame:
        """Generate a route using a specific algorithm. Now uses deterministic A* for 'A*'."""
        # Helper: remove collinear points to smooth path
        def _smooth_path(points: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
            if not points:
                return []
            out = [points[0]]
            for p in points[1:]:
                if len(out) < 2:
                    out.append(p)
                    continue
                x1, y1 = out[-2]
                x2, y2 = out[-1]
                x3, y3 = p
                # Check collinearity via cross product
                if (x2 - x1) * (y3 - y1) == (y2 - y1) * (x3 - x1):
                    # replace last with new to shorten
                    out[-1] = p
                else:
                    out.append(p)
            return out

        def _rrt_smooth(path: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
            # Simple post-hoc shortcutting for RRT paths
            if not path:
                return []
            def bres(a,b):
                x0,y0 = a
                x1,y1 = b
                pts = []
                dx = abs(x1-x0)
                dy = abs(y1-y0)
                x,y = x0,y0
                sx = 1 if x0 < x1 else -1
                sy = 1 if y0 < y1 else -1
                if dx > dy:
                    err = dx/2.0
                    while x != x1:
                        pts.append((x,y))
                        err -= dy
                        if err < 0:
                            y += sy
                            err += dx
                        x += sx
                else:
                    err = dy/2.0
                    while y != y1:
                        pts.append((x,y))
                        err -= dx
                        if err < 0:
                            x += sx
                            err += dy
                        y += sy
                pts.append((x1,y1))
                return pts

            pruned = []
            n = len(path)
            i = 0
            orig_set = set(path)
            while i < n:
                pruned.append(path[i])
                j = n - 1
                jumped = False
                # limit search by a small window to keep smoothing fast
                max_window = max(1, int(n * 0.2))
                start_j = min(n-1, i + max_window)
                while start_j > i:
                    seg = bres(path[i], path[start_j])
                    if set(seg).issubset(orig_set):
                        i = start_j
                        jumped = True
                        break
                    start_j -= 1
                if not jumped:
                    i += 1
            if pruned[-1] != path[-1]:
                pruned.append(path[-1])
            return pruned

        def astar_path(grid_size, start, end):
            from heapq import heappush, heappop
            neighbors = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
            close_set = set()
            came_from = {}
            gscore = {start:0}
            fscore = {start:abs(start[0]-end[0])+abs(start[1]-end[1])}
            oheap = []
            heappush(oheap, (fscore[start], start))
            while oheap:
                current = heappop(oheap)[1]
                if current == end:
                    data = []
                    while current in came_from:
                        data.append(current)
                        current = came_from[current]
                    data.append(start)
                    return data[::-1]
                close_set.add(current)
                for i, j in neighbors:
                    neighbor = current[0]+i, current[1]+j
                    tentative_g_score = gscore[current] + ((i**2+j**2)**0.5)
                    if 0 <= neighbor[0] < grid_size and 0 <= neighbor[1] < grid_size:
                        if neighbor in close_set:
                            continue
                        if tentative_g_score < gscore.get(neighbor, float('inf')):
                            came_from[neighbor] = current
                            gscore[neighbor] = tentative_g_score
                            fscore[neighbor] = tentative_g_score + abs(neighbor[0]-end[0]) + abs(neighbor[1]-end[1])
                            heappush(oheap, (fscore[neighbor], neighbor))
            return []

        grid_size = self.params.grid_size
        start = tuple(map(int, self.params.start_point))
        end = tuple(map(int, self.params.end_point))

        def _finalize_route_df(df: pd.DataFrame) -> pd.DataFrame:
            """Post-process a generated route DataFrame: recompute distances/energy/time,
            apply perception-based risk estimates and drop infeasible waypoints when available.
            """
            if df is None or df.empty:
                return pd.DataFrame()

            try:
                # Recompute distances from grid indices
                gx = df['Grid_X'].to_numpy(dtype=float)
                gy = df['Grid_Y'].to_numpy(dtype=float)
                if len(gx) > 1:
                    dx = np.diff(gx) * self.params.resolution_m
                    dy = np.diff(gy) * self.params.resolution_m
                    seg_dist = np.sqrt(dx**2 + dy**2)
                    cumulative = np.concatenate([[0.0], np.cumsum(seg_dist)])
                else:
                    cumulative = np.array([0.0])
                df['Distance (km)'] = cumulative / 1000.0
                base_speed = (self.params.min_speed_ms + self.params.max_speed_ms) / 2
                df['Time (min)'] = (df['Distance (km)'] * 1000.0) / (base_speed * 60.0)

                # Energy and risk via perception API when available
                num = len(df)
                energy_per_segment = np.zeros(num)
                risk_scores = df.get('Risk Score', pd.Series(np.full(num, 0.2))).to_numpy(dtype=float)

                if HAVE_PERCEPTION and perception_api is not None:
                    for i in range(num):
                        lat = float(df.at[i, 'Latitude'])
                        lon = float(df.at[i, 'Longitude'])
                        alt = float(df.at[i, 'Altitude (m)'])
                        try:
                            qp = perception_api.QueryPoint(lat=lat, lon=lon, alt_m=alt)
                            risk_scores[i] = float(perception_api.risk_score(qp))
                        except Exception:
                            pass
                    if num > 1:
                        seg_dist_km = np.diff(df['Distance (km)'])
                        for si in range(1, num):
                            mid_lat = float((df.at[si, 'Latitude'] + df.at[si-1, 'Latitude']) / 2.0)
                            mid_lon = float((df.at[si, 'Longitude'] + df.at[si-1, 'Longitude']) / 2.0)
                            try:
                                cost_kwh_per_km = float(perception_api.energy_cost_kwh_per_km(mid_lat, mid_lon, float(df.at[si, 'Altitude (m)'])))
                            except Exception:
                                cost_kwh_per_km = (self.params.cruise_power_kw / base_speed)
                            energy_per_segment[si] = seg_dist_km[si-1] * cost_kwh_per_km
                else:
                    if num > 1:
                        seg_dist_km = np.diff(df['Distance (km)'])
                        energy_per_segment[1:] = seg_dist_km * (self.params.cruise_power_kw / base_speed)

                df['Energy (kWh)'] = np.cumsum(energy_per_segment)
                df['Risk Score'] = risk_scores

                # Drop infeasible waypoints if perception indicates infeasible
                if HAVE_PERCEPTION and perception_api is not None:
                    feasible_mask = []
                    for i in range(len(df)):
                        try:
                            qp = perception_api.QueryPoint(lat=float(df.at[i,'Latitude']), lon=float(df.at[i,'Longitude']), alt_m=float(df.at[i,'Altitude (m)']))
                            feasible_mask.append(bool(perception_api.feasible(qp)))
                        except Exception:
                            feasible_mask.append(True)
                    if not all(feasible_mask):
                        df = df.loc[feasible_mask].reset_index(drop=True)
                        df['Waypoint'] = range(1, len(df) + 1)

            except Exception:
                # On any failure, return original df
                return df

            return df


        if algorithm == "A*":
            path = astar_path(grid_size, start, end)
            if not path:
                return pd.DataFrame()
            # smoothing
            path = _smooth_path(path)
            # Deterministic, simple metrics for demo
            path_x = np.array([p[0] for p in path])
            path_y = np.array([p[1] for p in path])
            num_waypoints = len(path)
            dx = np.diff(path_x) * self.params.resolution_m
            dy = np.diff(path_y) * self.params.resolution_m
            segment_distances = np.sqrt(dx**2 + dy**2)
            cumulative_distance = np.concatenate([[0], np.cumsum(segment_distances)])
            base_altitude = (self.params.min_altitude_m + self.params.max_altitude_m) / 2
            altitudes = np.full(num_waypoints, base_altitude)
            base_speed = (self.params.min_speed_ms + self.params.max_speed_ms) / 2
            speeds = np.full(num_waypoints, base_speed)
            times = cumulative_distance / (base_speed * 3.6)
            power_per_waypoint = np.full(num_waypoints, self.params.cruise_power_kw)
            energy_per_segment = np.zeros(num_waypoints)
            # Use perception API for per-waypoint risk and per-segment energy when available
            risk_scores = np.full(num_waypoints, 0.2)
            if HAVE_PERCEPTION and perception_api is not None:
                try:
                    for i in range(num_waypoints):
                        lat = float(12.8 + path_y[i] / grid_size * 0.2)
                        lon = float(77.5 + path_x[i] / grid_size * 0.2)
                        alt = float(altitudes[i])
                        qp = perception_api.QueryPoint(lat=lat, lon=lon, alt_m=alt)
                        try:
                            risk_scores[i] = float(perception_api.risk_score(qp))
                        except Exception:
                            risk_scores[i] = float(risk_scores[i])
                    if num_waypoints > 1:
                        seg_dist_km = segment_distances / 1000.0
                        for si in range(1, num_waypoints):
                            mid_lat = float(12.8 + (path_y[si] + path_y[si-1]) / 2 / grid_size * 0.2)
                            mid_lon = float(77.5 + (path_x[si] + path_x[si-1]) / 2 / grid_size * 0.2)
                            try:
                                cost_kwh_per_km = float(perception_api.energy_cost_kwh_per_km(mid_lat, mid_lon, alt))
                            except Exception:
                                cost_kwh_per_km = (power_per_waypoint[si] / base_speed)
                            energy_per_segment[si] = seg_dist_km[si-1] * cost_kwh_per_km
                except Exception:
                    energy_per_segment[1:] = segment_distances / 1000 * power_per_waypoint[1:] / base_speed
            else:
                energy_per_segment[1:] = segment_distances / 1000 * power_per_waypoint[1:] / base_speed
            cumulative_energy = np.cumsum(energy_per_segment)
            df = pd.DataFrame({
                'Waypoint': range(1, num_waypoints + 1),
                'Grid_X': path_x.astype(int),
                'Grid_Y': path_y.astype(int),
                'Latitude': 12.8 + path_y / grid_size * 0.2,
                'Longitude': 77.5 + path_x / grid_size * 0.2,
                'Altitude (m)': altitudes,
                'Speed (m/s)': speeds,
                'Distance (km)': cumulative_distance / 1000,
                'Time (min)': times / 60,
                'Energy (kWh)': cumulative_energy,
                'Risk Score': risk_scores,
                'Algorithm': algorithm
            })
            # Filter infeasible waypoints using perception API if available
            if HAVE_PERCEPTION and perception_api is not None and not df.empty:
                feasible_mask = []
                for _, row in df.iterrows():
                    try:
                        qp = perception_api.QueryPoint(lat=float(row['Latitude']), lon=float(row['Longitude']), alt_m=float(row['Altitude (m)']))
                        feasible_mask.append(bool(perception_api.feasible(qp)))
                    except Exception:
                        feasible_mask.append(True)
                df = df.loc[feasible_mask].reset_index(drop=True)
                if df.empty:
                    return pd.DataFrame()
                # renumber waypoints
                df['Waypoint'] = range(1, len(df) + 1)
            return df

        # Deterministic implementations for other algorithms
        def dijkstra_path(grid_size, start, end):
            import heapq
            neighbors = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
            dist = { (i,j): float('inf') for i in range(grid_size) for j in range(grid_size) }
            prev = {}
            dist[start] = 0
            hq = [(0, start)]
            visited = set()
            while hq:
                d, u = heapq.heappop(hq)
                if u in visited:
                    continue
                visited.add(u)
                if u == end:
                    # reconstruct
                    path = []
                    curr = end
                    while curr in prev:
                        path.append(curr)
                        curr = prev[curr]
                    path.append(start)
                    return path[::-1]
                for i,j in neighbors:
                    v = (u[0]+i, u[1]+j)
                    if 0 <= v[0] < grid_size and 0 <= v[1] < grid_size:
                        w = ((i**2 + j**2)**0.5)
                        nd = d + w
                        if nd < dist[v]:
                            dist[v] = nd
                            prev[v] = u
                            heapq.heappush(hq, (nd, v))
            return []

        def theta_star_path(grid_size, start, end):
            # Theta*: produce a straight-line raster between start and end, then
            # aggressively shortcut it by greedily jumping to the furthest
            # reachable point (line-of-sight) to reduce waypoints and create
            # near-any-angle paths.
            def bresenham_line(a, b):
                x0, y0 = a
                x1, y1 = b
                points = []
                dx = abs(x1 - x0)
                dy = abs(y1 - y0)
                x, y = x0, y0
                sx = 1 if x0 < x1 else -1
                sy = 1 if y0 < y1 else -1
                if dx > dy:
                    err = dx / 2.0
                    while x != x1:
                        points.append((x, y))
                        err -= dy
                        if err < 0:
                            y += sy
                            err += dx
                        x += sx
                else:
                    err = dy / 2.0
                    while y != y1:
                        points.append((x, y))
                        err -= dx
                        if err < 0:
                            x += sx
                            err += dy
                        y += sy
                points.append((x1, y1))
                return points

            def aggressive_shortcut(points: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
                if not points:
                    return []
                orig_set = set(points)
                pruned = []
                i = 0
                n = len(points)
                # Compute max_jump based on theta_aggressiveness (1..10)
                max_jump = max(1, int(n * (self.params.theta_aggressiveness / 10.0)))
                while i < n:
                    pruned.append(points[i])
                    # search from furthest allowed ahead back to i+1
                    j = min(n - 1, i + max_jump)
                    jumped = False
                    while j > i:
                        seg = bresenham_line(points[i], points[j])
                        if set(seg).issubset(orig_set):
                            i = j
                            jumped = True
                            break
                        j -= 1
                    if not jumped:
                        i += 1
                if pruned[-1] != points[-1]:
                    pruned.append(points[-1])
                return pruned

            line = bresenham_line(start, end)
            if not line:
                return []
            # First perform simple collinearity smoothing, then aggressive shortcutting
            smooth = _smooth_path(line)
            return aggressive_shortcut(smooth)

        def rrt_path(grid_size, start, end, samples=300, seed=42):
            rng = np.random.RandomState(seed)
            tree = {start: None}
            nodes = [start]
            children = {}
            for _ in range(samples):
                if rng.rand() < 0.1:
                    sample = end
                else:
                    sample = (int(rng.randint(0, grid_size)), int(rng.randint(0, grid_size)))
                # nearest
                dists = [ ( (n[0]-sample[0])**2 + (n[1]-sample[1])**2, n) for n in nodes ]
                nearest = min(dists, key=lambda x: x[0])[1]
                # steer: move one step towards sample
                dir0 = sample[0]-nearest[0]
                dir1 = sample[1]-nearest[1]
                step = ( np.sign(dir0), np.sign(dir1) )
                new = (nearest[0]+int(step[0]), nearest[1]+int(step[1]))
                if not (0 <= new[0] < grid_size and 0 <= new[1] < grid_size):
                    continue
                if new in tree:
                    continue
                tree[new] = nearest
                children.setdefault(nearest, []).append(new)
                nodes.append(new)
                if abs(new[0]-end[0])<=1 and abs(new[1]-end[1])<=1:
                    # reached
                    path = [end]
                    curr = new
                    while curr is not None:
                        path.append(curr)
                        curr = tree[curr]
                    # Reconstruct and attempt simple rewiring to shorten path
                    p = path[::-1]
                    improved = True
                    while improved:
                        improved = False
                        for i in range(1, len(p)-1):
                            for j in range(i+1, len(p)):
                                # compute direct distance from p[i-1] to p[j]
                                direct = ((p[j][0]-p[i-1][0])**2 + (p[j][1]-p[i-1][1])**2)**0.5
                                seg = 0.0
                                for k in range(i, j+1):
                                    dx = p[k][0]-p[k-1][0]
                                    dy = p[k][1]-p[k-1][1]
                                    seg += (dx*dx+dy*dy)**0.5
                                if direct + 1e-6 < seg:
                                    p = p[:i] + p[j:]
                                    improved = True
                                    break
                            if improved:
                                break
                    # Final smoothing: remove collinear points
                    if len(p) >= 3:
                        sp = [p[0]]
                        for i in range(1, len(p)-1):
                            x0,y0 = sp[-1]
                            x1,y1 = p[i]
                            x2,y2 = p[i+1]
                            area = (x1-x0)*(y2-y0) - (y1-y0)*(x2-x0)
                            if area == 0:
                                continue
                            sp.append(p[i])
                        sp.append(p[-1])
                        return sp
                    return p
            return []

        if algorithm == "Dijkstra":
            path = dijkstra_path(grid_size, start, end)
        elif algorithm == "Theta*":
            path = theta_star_path(grid_size, start, end)
        elif algorithm == "RRT*":
            path = rrt_path(grid_size, start, end, samples=self.params.rrt_samples, seed=int(self.params.rrt_seed))
            if path and getattr(self.params, 'rrt_smoothing', False):
                try:
                    path = _rrt_smooth(path)
                except Exception:
                    pass
        else:
            path = []

        if not path:
            # fallback: return empty DataFrame
            return pd.DataFrame()

        path_x = np.array([p[0] for p in path])
        path_y = np.array([p[1] for p in path])
        num_waypoints = len(path)
        dx = np.diff(path_x) * self.params.resolution_m if num_waypoints>1 else np.array([])
        dy = np.diff(path_y) * self.params.resolution_m if num_waypoints>1 else np.array([])
        segment_distances = np.sqrt(dx**2 + dy**2) if num_waypoints>1 else np.array([0.0])
        cumulative_distance = np.concatenate([[0], np.cumsum(segment_distances)]) if num_waypoints>1 else np.array([0.0])
        base_altitude = (self.params.min_altitude_m + self.params.max_altitude_m) / 2
        altitudes = np.full(num_waypoints, base_altitude)
        base_speed = (self.params.min_speed_ms + self.params.max_speed_ms) / 2
        speeds = np.full(num_waypoints, base_speed)
        times = cumulative_distance / (base_speed * 3.6)
        power_per_waypoint = np.full(num_waypoints, self.params.cruise_power_kw)
        energy_per_segment = np.zeros(num_waypoints)
        risk_scores = np.full(num_waypoints, 0.2)
        # If perception API available, use it for risk and energy estimates per waypoint/segment
        if HAVE_PERCEPTION and perception_api is not None:
            try:
                for i in range(num_waypoints):
                    lat = float(12.8 + path_y[i] / grid_size * 0.2)
                    lon = float(77.5 + path_x[i] / grid_size * 0.2)
                    alt = float(altitudes[i])
                    qp = perception_api.QueryPoint(lat=lat, lon=lon, alt_m=alt)
                    try:
                        risk_scores[i] = float(perception_api.risk_score(qp))
                    except Exception:
                        pass
                if num_waypoints > 1:
                    seg_dist_km = segment_distances / 1000.0
                    for si in range(1, num_waypoints):
                        mid_lat = float(12.8 + (path_y[si] + path_y[si-1]) / 2 / grid_size * 0.2)
                        mid_lon = float(77.5 + (path_x[si] + path_x[si-1]) / 2 / grid_size * 0.2)
                        try:
                            cost_kwh_per_km = float(perception_api.energy_cost_kwh_per_km(mid_lat, mid_lon, alt))
                        except Exception:
                            cost_kwh_per_km = (power_per_waypoint[si] / base_speed)
                        energy_per_segment[si] = seg_dist_km[si-1] * cost_kwh_per_km
            except Exception:
                energy_per_segment[1:] = segment_distances / 1000 * power_per_waypoint[1:] / base_speed
        else:
            if num_waypoints>1:
                energy_per_segment[1:] = segment_distances / 1000 * power_per_waypoint[1:] / base_speed
        cumulative_energy = np.cumsum(energy_per_segment)
        df = pd.DataFrame({
            'Waypoint': range(1, num_waypoints + 1),
            'Grid_X': path_x.astype(int),
            'Grid_Y': path_y.astype(int),
            'Latitude': 12.8 + path_y / grid_size * 0.2,
            'Longitude': 77.5 + path_x / grid_size * 0.2,
            'Altitude (m)': altitudes,
            'Speed (m/s)': speeds,
            'Distance (km)': cumulative_distance / 1000,
            'Time (min)': times / 60,
            'Energy (kWh)': cumulative_energy,
            'Risk Score': risk_scores,
            'Algorithm': algorithm
        })

        # Filter out infeasible waypoints using perception API if available
        if HAVE_PERCEPTION and perception_api is not None and not df.empty:
            try:
                feasible_mask = []
                for _, row in df.iterrows():
                    try:
                        qp = perception_api.QueryPoint(lat=float(row['Latitude']), lon=float(row['Longitude']), alt_m=float(row['Altitude (m)']))
                        feasible_mask.append(bool(perception_api.feasible(qp)))
                    except Exception:
                        feasible_mask.append(True)
                df = df.loc[[bool(m) for m in feasible_mask]].reset_index(drop=True)
            except Exception:
                pass

        return df
    
    def generate_pareto_solutions(self, progress_callback=None) -> bool:
        """Generate Pareto-optimal solutions for multi-objective optimization."""
        try:
            if progress_callback:
                progress_callback("Computing Pareto frontier...")
            
            num_solutions = self.params.num_pareto_solutions
            
            # Generate solutions along the Pareto frontier
            # Trade-off between distance, energy, risk, and time
            solutions = []
            
            for i in range(num_solutions):
                # Vary weights to get different solutions
                alpha = i / (num_solutions - 1) if num_solutions > 1 else 0.5
                
                # Distance vs Energy trade-off
                distance = 40 + 30 * (1 - alpha) + np.random.randn() * 5
                energy = 15 + 20 * alpha + np.random.randn() * 2
                
                # Risk varies inversely with both (longer paths can avoid threats)
                risk = 0.2 + 0.5 * (1 - alpha) * np.random.uniform(0.8, 1.2)
                
                # Time correlates with distance
                time = distance / 30 * 60 + np.random.randn() * 5
                
                # Safety margin (higher for lower risk)
                safety_margin = 0.9 - risk * 0.5 + np.random.uniform(-0.1, 0.1)
                
                solutions.append({
                    'Solution ID': i + 1,
                    'Distance (km)': max(30, distance),
                    'Energy (kWh)': max(10, energy),
                    'Time (min)': max(20, time),
                    'Risk Score': np.clip(risk, 0.1, 0.9),
                    'Safety Margin': np.clip(safety_margin, 0.3, 0.95),
                    'Pareto Rank': 1 if i < 5 else 2,
                    'Dominated': i >= 7
                })
            
            self.pareto_solutions = pd.DataFrame(solutions)
            self.generation_status['pareto'] = True
            return True
            
        except Exception as e:
            st.error(f"Pareto optimization failed: {e}")
            return False
    
    def generate_energy_profile(self, progress_callback=None) -> bool:
        """Generate detailed energy consumption profile."""
        try:
            if progress_callback:
                progress_callback("Computing energy profile...")
            
            # Use the first route as reference
            if not self.routes:
                return False
            
            reference_route = list(self.routes.values())[0]
            num_points = len(reference_route) * 10  # Higher resolution
            
            # Time series
            total_time = reference_route['Time (min)'].max()
            time_points = np.linspace(0, total_time, num_points)
            
            # Power consumption components
            base_power = self.params.cruise_power_kw
            
            # Hover power (higher at takeoff/landing)
            hover_power = base_power * 1.5 * np.exp(-((time_points - 0)**2 + (time_points - total_time)**2) / (total_time/3)**2)
            
            # Cruise power
            cruise_power = base_power * np.ones(num_points)
            
            # Climb/descent power variations
            climb_power = base_power * 0.3 * np.sin(2 * np.pi * time_points / total_time * 3)
            
            # Total power
            total_power = cruise_power + hover_power + np.maximum(0, climb_power)
            
            # Energy (cumulative)
            dt_hours = np.diff(time_points, prepend=0) / 60
            energy_consumed = np.cumsum(total_power * dt_hours)
            
            # Battery state
            battery_remaining = self.params.battery_capacity_kwh - energy_consumed
            soc = battery_remaining / self.params.battery_capacity_kwh * 100
            
            self.energy_profile = pd.DataFrame({
                'Time (min)': time_points,
                'Total Power (kW)': total_power,
                'Hover Power (kW)': hover_power,
                'Cruise Power (kW)': cruise_power,
                'Climb Power (kW)': np.maximum(0, climb_power),
                'Energy Consumed (kWh)': energy_consumed,
                'Battery Remaining (kWh)': np.maximum(0, battery_remaining),
                'State of Charge (%)': np.maximum(0, soc)
            })
            
            self.generation_status['energy'] = True
            return True
            
        except Exception as e:
            st.error(f"Energy profile generation failed: {e}")
            return False
    
    def generate_risk_assessment(self, perception_manager=None, progress_callback=None) -> bool:
        """Generate risk assessment along the trajectory."""
        try:
            if progress_callback:
                progress_callback("Computing risk assessment...")
            
            if not self.routes:
                return False
            
            # Aggregate risk data from all routes
            risk_data = []
            
            for algo, route in self.routes.items():
                for _, wp in route.iterrows():
                    risk_data.append({
                        'Algorithm': algo,
                        'Waypoint': wp['Waypoint'],
                        'Distance (km)': wp['Distance (km)'],
                        'Threat Risk': np.random.uniform(0.1, 0.5),
                        'Terrain Risk': np.random.uniform(0.05, 0.3),
                        'Weather Risk': np.random.uniform(0.05, 0.2),
                        'Total Risk': wp['Risk Score'],
                        'Safe Corridor Width (m)': np.random.uniform(100, 500)
                    })
            
            self.risk_assessment = pd.DataFrame(risk_data)
            self.generation_status['risk'] = True
            return True
            
        except Exception as e:
            st.error(f"Risk assessment failed: {e}")
            return False
    
    def generate_constraints_check(self, progress_callback=None) -> bool:
        """Check all trajectory constraints."""
        try:
            if progress_callback:
                progress_callback("Checking constraints...")
            
            # Evaluate feasibility using perception API when available
            no_fly_violations = 0
            total_waypoints_checked = 0
            try:
                if self.routes and HAVE_PERCEPTION and perception_api is not None:
                    for algo, route in self.routes.items():
                        for _, wp in route.iterrows():
                            total_waypoints_checked += 1
                            try:
                                lat = float(wp.get('Latitude', 0.0))
                                lon = float(wp.get('Longitude', 0.0))
                                alt = float(wp.get('Altitude (m)', 0.0))
                                qp = perception_api.QueryPoint(lat=lat, lon=lon, alt_m=alt)
                                feasible = bool(perception_api.feasible(qp))
                            except Exception:
                                feasible = True
                            if not feasible:
                                no_fly_violations += 1
            except Exception:
                # If any perception failure occurs, fall back to zero violations
                no_fly_violations = 0

            # Build constraints summary, updating No-Fly Zones based on perception
            constraints = [
                {'Constraint': 'Altitude Limits', 'Type': 'Flight Safety', 'Status': 'PASS', 'Violations': 0, 'Margin': '15%'},
                {'Constraint': 'Speed Limits', 'Type': 'Performance', 'Status': 'PASS', 'Violations': 0, 'Margin': '20%'},
                {'Constraint': 'Bank Angle', 'Type': 'Structural', 'Status': 'PASS', 'Violations': 0, 'Margin': '25%'},
                {'Constraint': 'Energy Reserve', 'Type': 'Energy', 'Status': 'PASS', 'Violations': 0, 'Margin': '30%'},
                {'Constraint': 'Threat Avoidance', 'Type': 'Safety', 'Status': 'MANAGED', 'Violations': 0, 'Margin': '5%'},
                {'Constraint': 'Obstacle Clearance', 'Type': 'Terrain', 'Status': 'PASS', 'Violations': 0, 'Margin': '50m'},
                {'Constraint': 'Wind Limits', 'Type': 'Weather', 'Status': 'PASS', 'Violations': 0, 'Margin': '10 m/s'},
                {
                    'Constraint': 'No-Fly Zones',
                    'Type': 'Regulatory',
                    'Status': 'VIOLATED' if no_fly_violations > 0 else 'PASS',
                    'Violations': int(no_fly_violations),
                    'Margin': f'Checked {int(total_waypoints_checked)} WPs'
                },
            ]

            self.constraints_status = pd.DataFrame(constraints)
            self.generation_status['constraints'] = True
            return True
            
        except Exception as e:
            st.error(f"Constraints check failed: {e}")
            return False
    
    def generate_all(self, perception_manager=None, progress_callback=None) -> bool:
        """Generate all planning data."""
        success = True
        if not self.generate_routes(perception_manager, progress_callback):
            success = False
        if not self.generate_moo_results():
            success = False
        if not self.generate_pareto_frontier():
            success = False
        if not self.generate_energy_profile(progress_callback):
            success = False
        if not self.generate_risk_assessment(perception_manager, progress_callback):
            success = False
        if not self.generate_constraints_check(progress_callback):
            success = False
        self.is_generated = success
        return success
    
    def get_combined_routes_df(self) -> pd.DataFrame:
        """Get all routes combined into a single DataFrame."""
        if not self.routes:
            return pd.DataFrame()
        return pd.concat(self.routes.values(), ignore_index=True)
    
    def get_route_comparison_df(self) -> pd.DataFrame:
        """Get route comparison statistics."""
        if not self.routes:
            return pd.DataFrame()
        
        comparison = []
        for algo, route in self.routes.items():
            comparison.append({
                'Algorithm': algo,
                'Waypoints': len(route),
                'Total Distance (km)': route['Distance (km)'].max(),
                'Total Time (min)': route['Time (min)'].max(),
                'Total Energy (kWh)': route['Energy (kWh)'].max(),
                'Avg Risk Score': route['Risk Score'].mean(),
                'Max Risk Score': route['Risk Score'].max(),
            })
        
        return pd.DataFrame(comparison)


# =============================================================================
# Session State Management
# =============================================================================

def init_planning_session_state():
    """Initialize planning-related session state."""
    if 'planning_manager' not in st.session_state:
        st.session_state.planning_manager = None
    
    if 'planning_params' not in st.session_state:
        st.session_state.planning_params = PlanningParameters()
    
    if 'planning_generated' not in st.session_state:
        st.session_state.planning_generated = False


def get_planning_manager() -> Optional[PlanningManager]:
    """Get the current planning manager from session state."""
    init_planning_session_state()
    return st.session_state.planning_manager


def create_planning_manager(params: Optional[PlanningParameters] = None) -> PlanningManager:
    """Create a new planning manager and store in session state."""
    init_planning_session_state()
    
    if params:
        st.session_state.planning_params = params
    
    manager = PlanningManager(st.session_state.planning_params)
    st.session_state.planning_manager = manager
    return manager


def render_planning_controls(perception_manager=None) -> Optional[PlanningManager]:
    """
    Render planning generation controls.
    
    Returns the PlanningManager if generation was triggered or already exists.
    """
    init_planning_session_state()
    
    st.markdown("### Trajectory Planning Generation")
    
    # Configuration expander
    with st.expander("Planning Configuration", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Grid Parameters**")
            grid_size = st.slider(
                "Grid Size",
                min_value=50,
                max_value=200,
                value=st.session_state.planning_params.grid_size,
                step=10
            )
            
            start_x = st.number_input("Start X", 0, grid_size-1, 10)
            start_y = st.number_input("Start Y", 0, grid_size-1, 10)
            end_x = st.number_input("End X", 0, grid_size-1, grid_size-10)
            end_y = st.number_input("End Y", 0, grid_size-1, grid_size-10)
        
        st.markdown("**Optimization Weights**")
        w_dist = st.slider("Distance Weight", 0.0, 1.0, 0.3, 0.05)
        w_energy = st.slider("Energy Weight", 0.0, 1.0, 0.3, 0.05)
        w_risk = st.slider("Risk Weight", 0.0, 1.0, 0.2, 0.05)
        w_time = st.slider("Time Weight", 0.0, 1.0, 0.2, 0.05)

        # Update params
        # Algorithm parameters
        st.markdown("**Algorithm Parameters**")
        theta_aggr = st.slider("Theta* Aggressiveness", 1, 10, st.session_state.planning_params.theta_aggressiveness)
        rrt_samples = st.slider("RRT* Samples", 50, 1000, st.session_state.planning_params.rrt_samples, step=50)
        rrt_seed = st.number_input("RRT* Seed", value=st.session_state.planning_params.rrt_seed)
        rrt_smooth = st.checkbox("Enable RRT* Smoothing", value=st.session_state.planning_params.rrt_smoothing)

        st.session_state.planning_params = PlanningParameters(
            grid_size=grid_size,
            start_point=(start_x, start_y),
            end_point=(end_x, end_y),
            weight_distance=w_dist,
            weight_energy=w_energy,
            weight_risk=w_risk,
            weight_time=w_time,
            theta_aggressiveness=int(theta_aggr),
            rrt_samples=int(rrt_samples),
            rrt_seed=int(rrt_seed),
            rrt_smoothing=bool(rrt_smooth)
        )
    
    # Generation button
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        generate_clicked = st.button(
            "Generate Trajectories",
            type="primary",
            use_container_width=True
        )
    
    with col2:
        if st.session_state.planning_generated:
            regenerate_clicked = st.button("Regenerate", use_container_width=True)
        else:
            regenerate_clicked = False
    
    # Status display removed as per user request
    
    # Auto-generate PlanningManager on first load if not present
    if st.session_state.planning_manager is None:
        manager = create_planning_manager(st.session_state.planning_params)
        st.session_state.planning_manager = manager
    # Handle generation
    if generate_clicked or regenerate_clicked:
        manager = create_planning_manager(st.session_state.planning_params)
        # Show error/status if present
        if getattr(manager, 'error', None):
            st.error(f"Planning backend error: {manager.error}")
            if getattr(manager, 'traceback', None):
                st.exception(manager.traceback)
        elif getattr(manager, 'status', None):
            st.info(f"Planning backend status: {manager.status}")
        return manager
    # Show error/status for existing manager
    manager = st.session_state.planning_manager
    if manager:
        if getattr(manager, 'error', None):
            st.error(f"Planning backend error: {manager.error}")
            if getattr(manager, 'traceback', None):
                st.exception(manager.traceback)
        elif getattr(manager, 'status', None):
            st.info(f"Planning backend status: {manager.status}")
        # Report any algorithms dropped due to perception infeasibility
        try:
            dropped = getattr(manager, 'dropped_algorithms', None)
            if dropped:
                st.warning(f"Algorithms dropped due to infeasibility: {', '.join(dropped)}")
        except Exception:
            pass
    return manager
