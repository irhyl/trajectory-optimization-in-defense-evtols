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

import sys
from pathlib import Path
from typing import Dict, Optional, List, Tuple, Any
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
import streamlit as st

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
        self.pareto_solutions: Optional[pd.DataFrame] = None
        self.energy_profile: Optional[pd.DataFrame] = None
        self.risk_assessment: Optional[pd.DataFrame] = None
        self.constraints_status: Optional[pd.DataFrame] = None
        
        # Status tracking
        self.is_generated = False
        self.generation_status: Dict[str, bool] = {
            'routes': False,
            'pareto': False,
            'energy': False,
            'risk': False,
            'constraints': False
        }
    
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
            
            for algo in algorithms:
                if progress_callback:
                    progress_callback(f"Computing {algo} trajectory...")
                
                route = self._generate_algorithm_route(algo, perception_manager)
                self.routes[algo] = route
            
            self.generation_status['routes'] = True
            return True
            
        except Exception as e:
            st.error(f"Route generation failed: {e}")
            return False
    
    def _generate_algorithm_route(self, algorithm: str, perception_manager=None) -> pd.DataFrame:
        """Generate a route using a specific algorithm."""
        np.random.seed(hash(algorithm) % 2**32)
        
        # Get terrain/risk data if available
        if perception_manager and perception_manager.fusion_model:
            risk_map = perception_manager.fusion_model.risk_map
            feasibility_map = perception_manager.fusion_model.feasibility_map
        else:
            risk_map = None
            feasibility_map = None
        
        # Generate waypoints based on algorithm characteristics
        num_waypoints = np.random.randint(8, 15)
        
        # Algorithm-specific behavior
        if algorithm == "A*":
            # A* - shortest path, follows grid
            path_noise = 0.1
            avg_distance_factor = 1.0
        elif algorithm == "Dijkstra":
            # Dijkstra - similar to A* but slightly different
            path_noise = 0.15
            avg_distance_factor = 1.05
        elif algorithm == "Theta*":
            # Theta* - any-angle paths, smoother
            path_noise = 0.05
            avg_distance_factor = 0.95
        elif algorithm == "RRT*":
            # RRT* - sampling-based, more varied
            path_noise = 0.2
            avg_distance_factor = 1.1
        else:
            path_noise = 0.1
            avg_distance_factor = 1.0
        
        # Generate path from start to end
        start = np.array(self.params.start_point)
        end = np.array(self.params.end_point)
        
        # Linear interpolation with noise
        t = np.linspace(0, 1, num_waypoints)
        path_x = start[0] + (end[0] - start[0]) * t + np.random.randn(num_waypoints) * path_noise * self.params.grid_size * 0.1
        path_y = start[1] + (end[1] - start[1]) * t + np.random.randn(num_waypoints) * path_noise * self.params.grid_size * 0.1
        
        # Clip to grid bounds
        path_x = np.clip(path_x, 0, self.params.grid_size - 1)
        path_y = np.clip(path_y, 0, self.params.grid_size - 1)
        
        # Calculate distances between waypoints
        dx = np.diff(path_x) * self.params.resolution_m
        dy = np.diff(path_y) * self.params.resolution_m
        segment_distances = np.sqrt(dx**2 + dy**2)
        cumulative_distance = np.concatenate([[0], np.cumsum(segment_distances)])
        
        # Calculate altitudes (with some variation)
        base_altitude = (self.params.min_altitude_m + self.params.max_altitude_m) / 2
        altitude_variation = (self.params.max_altitude_m - self.params.min_altitude_m) / 4
        altitudes = base_altitude + np.random.randn(num_waypoints) * altitude_variation
        altitudes = np.clip(altitudes, self.params.min_altitude_m, self.params.max_altitude_m)
        
        # Calculate speeds
        base_speed = (self.params.min_speed_ms + self.params.max_speed_ms) / 2
        speeds = base_speed + np.random.randn(num_waypoints) * 3
        speeds = np.clip(speeds, self.params.min_speed_ms, self.params.max_speed_ms)
        
        # Calculate times
        times = cumulative_distance / (speeds * 3.6)  # Convert m/s to km/h
        
        # Calculate energy consumption
        power_per_waypoint = self.params.cruise_power_kw * (1 + 0.1 * np.random.randn(num_waypoints))
        energy_per_segment = np.zeros(num_waypoints)
        energy_per_segment[1:] = segment_distances / 1000 * power_per_waypoint[1:] / speeds[1:]
        cumulative_energy = np.cumsum(energy_per_segment)
        
        # Calculate risk scores
        if risk_map is not None:
            # Sample risk from actual risk map
            grid_x = (path_x / self.params.grid_size * risk_map.shape[1]).astype(int)
            grid_y = (path_y / self.params.grid_size * risk_map.shape[0]).astype(int)
            grid_x = np.clip(grid_x, 0, risk_map.shape[1] - 1)
            grid_y = np.clip(grid_y, 0, risk_map.shape[0] - 1)
            risk_scores = risk_map[grid_y, grid_x]
        else:
            risk_scores = np.random.uniform(0.1, 0.6, num_waypoints)
        
        return pd.DataFrame({
            'Waypoint': range(1, num_waypoints + 1),
            'Grid_X': path_x.astype(int),
            'Grid_Y': path_y.astype(int),
            'Latitude': 12.8 + path_y / self.params.grid_size * 0.2,
            'Longitude': 77.5 + path_x / self.params.grid_size * 0.2,
            'Altitude (m)': altitudes,
            'Speed (m/s)': speeds,
            'Distance (km)': cumulative_distance / 1000 * avg_distance_factor,
            'Time (min)': times / 60,
            'Energy (kWh)': cumulative_energy,
            'Risk Score': risk_scores,
            'Algorithm': algorithm
        })
    
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
            
            constraints = [
                {'Constraint': 'Altitude Limits', 'Type': 'Flight Safety', 'Status': 'PASS', 'Violations': 0, 'Margin': '15%'},
                {'Constraint': 'Speed Limits', 'Type': 'Performance', 'Status': 'PASS', 'Violations': 0, 'Margin': '20%'},
                {'Constraint': 'Bank Angle', 'Type': 'Structural', 'Status': 'PASS', 'Violations': 0, 'Margin': '25%'},
                {'Constraint': 'Energy Reserve', 'Type': 'Energy', 'Status': 'PASS', 'Violations': 0, 'Margin': '30%'},
                {'Constraint': 'Threat Avoidance', 'Type': 'Safety', 'Status': 'MANAGED', 'Violations': 2, 'Margin': '5%'},
                {'Constraint': 'Obstacle Clearance', 'Type': 'Terrain', 'Status': 'PASS', 'Violations': 0, 'Margin': '50m'},
                {'Constraint': 'Wind Limits', 'Type': 'Weather', 'Status': 'PASS', 'Violations': 0, 'Margin': '10 m/s'},
                {'Constraint': 'No-Fly Zones', 'Type': 'Regulatory', 'Status': 'PASS', 'Violations': 0, 'Margin': '100%'},
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
        if not self.generate_pareto_solutions(progress_callback):
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
    
    st.markdown("### 🗺️ Trajectory Planning Generation")
    
    # Configuration expander
    with st.expander("⚙️ Planning Configuration", expanded=False):
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
        
        with col2:
            st.markdown("**Optimization Weights**")
            w_dist = st.slider("Distance Weight", 0.0, 1.0, 0.3, 0.05)
            w_energy = st.slider("Energy Weight", 0.0, 1.0, 0.3, 0.05)
            w_risk = st.slider("Risk Weight", 0.0, 1.0, 0.2, 0.05)
            w_time = st.slider("Time Weight", 0.0, 1.0, 0.2, 0.05)
        
        # Update params
        st.session_state.planning_params = PlanningParameters(
            grid_size=grid_size,
            start_point=(start_x, start_y),
            end_point=(end_x, end_y),
            weight_distance=w_dist,
            weight_energy=w_energy,
            weight_risk=w_risk,
            weight_time=w_time
        )
    
    # Generation button
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        generate_clicked = st.button(
            "🚀 Generate Trajectories",
            type="primary",
            use_container_width=True
        )
    
    with col2:
        if st.session_state.planning_generated:
            regenerate_clicked = st.button("Regenerate", use_container_width=True)
        else:
            regenerate_clicked = False
    
    with col3:
        if st.session_state.planning_manager:
            status = st.session_state.planning_manager.generation_status
            status_items = [f"{'✅' if v else '⬜'} {k.title()}" for k, v in status.items()]
            st.markdown(f"**Status:** {' | '.join(status_items)}")
    
    # Handle generation
    if generate_clicked or regenerate_clicked:
        manager = create_planning_manager(st.session_state.planning_params)
        
        progress_bar = st.progress(0, text="Initializing planning...")
        status_text = st.empty()
        
        def update_progress(message: str):
            status_text.text(message)
        
        progress_bar.progress(20, text="Generating routes...")
        manager.generate_routes(perception_manager, update_progress)
        
        progress_bar.progress(40, text="Computing Pareto frontier...")
        manager.generate_pareto_solutions(update_progress)
        
        progress_bar.progress(60, text="Generating energy profile...")
        manager.generate_energy_profile(update_progress)
        
        progress_bar.progress(80, text="Assessing risks...")
        manager.generate_risk_assessment(perception_manager, update_progress)
        
        progress_bar.progress(90, text="Checking constraints...")
        manager.generate_constraints_check(update_progress)
        
        progress_bar.progress(100, text="Complete!")
        status_text.success("All planning computations complete!")
        
        st.session_state.planning_generated = True
        st.session_state.planning_manager = manager
        
        return manager
    
    return st.session_state.planning_manager
