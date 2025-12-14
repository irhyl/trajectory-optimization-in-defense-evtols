"""
Utility functions for Streamlit app.

Provides CSV export, data generation, and helper functions.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
import io
import zipfile


class CSVExporter:
    """Handles CSV export functionality."""
    
    @staticmethod
    def export_dataframe(df: pd.DataFrame) -> str:
        """Export dataframe to CSV string."""
        return df.to_csv(index=False)
    
    @staticmethod
    def export_multi_dataframes(dfs: Dict[str, pd.DataFrame], base_filename: str) -> Dict[str, str]:
        """Export multiple dataframes."""
        exports = {}
        for name, df in dfs.items():
            exports[f"{base_filename}_{name}.csv"] = df.to_csv(index=False)
        return exports
    
    @staticmethod
    def create_zip_export(dataframes: Dict[str, pd.DataFrame]) -> bytes:
        """Create a ZIP file with multiple CSVs."""
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
            for name, df in dataframes.items():
                csv_data = df.to_csv(index=False)
                zf.writestr(f"{name}.csv", csv_data)
        
        zip_buffer.seek(0)
        return zip_buffer.getvalue()


class DataGenerator:
    """Generates sample data for visualization."""
    
    @staticmethod
    def generate_mission_data(num_waypoints: int = 10) -> pd.DataFrame:
        """Generate sample mission waypoint data."""
        np.random.seed(42)
        
        distances = np.cumsum(np.random.uniform(0.5, 2, num_waypoints))
        times = np.cumsum(np.random.uniform(30, 120, num_waypoints))
        
        return pd.DataFrame({
            'WP_ID': range(1, num_waypoints + 1),
            'Latitude': np.linspace(15.2, 15.4, num_waypoints),
            'Longitude': np.linspace(74.1, 74.3, num_waypoints),
            'Altitude (m)': np.linspace(100, 500, num_waypoints),
            'Distance (m)': distances * 1000,
            'Time (s)': times,
            'Cumulative Distance (km)': distances
        })
    
    @staticmethod
    def generate_perception_data() -> Dict:
        """Generate perception layer data."""
        np.random.seed(42)
        
        # Terrain data
        terrain = pd.DataFrame({
            'Latitude': np.random.uniform(15.2, 15.4, 100),
            'Longitude': np.random.uniform(74.1, 74.3, 100),
            'Elevation (m)': np.random.uniform(100, 500, 100),
            'Slope (deg)': np.random.uniform(0, 45, 100),
            'Roughness': np.random.uniform(0, 1, 100),
        })
        
        # Wind data
        wind = pd.DataFrame({
            'Altitude (m)': [10, 100, 500] * 1,
            'Wind Speed (m/s)': np.random.uniform(3, 15, 3),
            'Wind Direction (deg)': np.random.uniform(0, 360, 3),
            'Turbulence': np.random.uniform(0.05, 0.3, 3),
        })
        
        # Threat data
        threat = pd.DataFrame({
            'Threat Type': ['Radar', 'Radar', 'SAM', 'Patrol'],
            'Latitude': np.random.uniform(15.2, 15.4, 4),
            'Longitude': np.random.uniform(74.1, 74.3, 4),
            'Range (m)': np.random.uniform(10000, 50000, 4),
            'Threat Level': ['High', 'Medium', 'High', 'Low'],
        })
        
        # Obstacle data
        obstacle = pd.DataFrame({
            'Building ID': range(1, 51),
            'Latitude': np.random.uniform(15.2, 15.4, 50),
            'Longitude': np.random.uniform(74.1, 74.3, 50),
            'Height (m)': np.random.uniform(10, 100, 50),
            'Footprint (m2)': np.random.uniform(100, 5000, 50),
        })
        
        return {
            'terrain': terrain,
            'wind': wind,
            'threat': threat,
            'obstacle': obstacle
        }
    
    @staticmethod
    def generate_planning_data() -> Dict:
        """Generate planning layer data."""
        np.random.seed(42)
        
        # Route comparison
        routes = pd.DataFrame({
            'Algorithm': ['A*'] * 5 + ['Dijkstra'] * 5 + ['Theta*'] * 5 + ['RRT*'] * 5,
            'Distance (km)': np.random.uniform(50, 200, 20),
            'Time (min)': np.random.uniform(20, 100, 20),
            'Safety Margin': np.random.uniform(0.3, 0.9, 20),
        })
        
        # Multi-objective solutions
        multi_objective = pd.DataFrame({
            'Solution ID': range(1, 11),
            'Distance (km)': np.random.uniform(50, 150, 10),
            'Time (min)': np.random.uniform(20, 80, 10),
            'Energy Efficiency': np.random.uniform(0.7, 0.95, 10),
            'Risk Score': np.random.uniform(0.1, 0.8, 10),
        })
        
        # Constraints
        constraints = pd.DataFrame({
            'Constraint Type': ['Altitude', 'Speed', 'Energy', 'Threat', 'Obstacle'],
            'Active': [True, True, False, True, True],
            'Violations': [0, 0, 0, 1, 0],
            'Status': ['PASS', 'PASS', 'PASS', 'MANAGED', 'PASS'],
        })
        
        return {
            'routes': routes,
            'multi_objective': multi_objective,
            'constraints': constraints
        }
    
    @staticmethod
    def generate_vehicle_data() -> Dict:
        """Generate vehicle layer data."""
        np.random.seed(42)
        time = np.arange(360)
        
        # Dynamics
        dynamics = pd.DataFrame({
            'Altitude (m)': 100 + 200 * np.sin(time / 100) + np.random.normal(0, 5, 360),
            'Velocity (m/s)': 20 + 5 * np.sin(time / 50) + np.random.normal(0, 0.5, 360),
            'Pitch (deg)': 5 + 10 * np.sin(time / 100) + np.random.normal(0, 1, 360),
            'Roll (deg)': 3 * np.sin(time / 80) + np.random.normal(0, 0.5, 360),
            'Yaw (deg)': np.linspace(0, 360, 360),
            'Acceleration (g)': 1 + 0.5 * np.sin(time / 100) + np.random.normal(0, 0.1, 360),
        })
        
        # Battery
        battery = pd.DataFrame({
            'State of Charge (%)': 100 - (time / 360 * 65) + np.random.normal(0, 1, 360),
            'Voltage (V)': 400 - (time / 360 * 50) + np.random.normal(0, 2, 360),
            'Current (A)': 100 + 20 * np.sin(time / 50) + np.random.normal(0, 5, 360),
            'Temperature (C)': 25 + (time / 360 * 20) + np.random.normal(0, 1, 360),
            'Cycle Count': np.ones(360) * 150,
            'Degradation (%)': np.ones(360) * 5,
            'Power (W)': 40000 + 5000 * np.sin(time / 50) + np.random.normal(0, 1000, 360),
        })
        
        # Motors
        motors = pd.DataFrame({
            'RPM': 2000 + 500 * np.sin(time / 50) + np.random.normal(0, 50, 360),
            'Thrust (N)': 100 + 30 * np.sin(time / 50) + np.random.normal(0, 5, 360),
            'Power (W)': 10000 + 2000 * np.sin(time / 50) + np.random.normal(0, 500, 360),
            'Efficiency (%)': 85 + 5 * np.sin(time / 100) + np.random.normal(0, 1, 360),
        })
        
        return {
            'dynamics': dynamics,
            'battery': battery,
            'motors': motors
        }
    
    @staticmethod
    def generate_control_data() -> Dict:
        """Generate control layer data."""
        np.random.seed(42)
        time = np.arange(360)
        
        # PID control
        pid = pd.DataFrame({
            'P_term': 10 * np.sin(time / 100) + np.random.normal(0, 0.5, 360),
            'I_term': 5 * np.sin(time / 200) + np.random.normal(0, 0.2, 360),
            'D_term': 3 * np.cos(time / 100) + np.random.normal(0, 0.3, 360),
        })
        
        # Trajectory tracking
        tracking = pd.DataFrame({
            'Position Error (m)': 5 * np.sin(time / 100) + np.random.normal(0, 1, 360),
            'Velocity Error (m/s)': 1 * np.sin(time / 150) + np.random.normal(0, 0.2, 360),
            'Altitude Error (m)': 3 * np.sin(time / 80) + np.random.normal(0, 0.5, 360),
            'Tracking Success (%)': 95 + 4 * np.sin(time / 100) + np.random.normal(0, 1, 360),
        })
        
        # Attitude control data
        attitude = pd.DataFrame({
            'Pitch Command (deg)': 5 + 8 * np.sin(time / 80) + np.random.normal(0, 0.5, 360),
            'Pitch Actual (deg)': 5 + 8 * np.sin(time / 80) + np.random.normal(0, 0.8, 360),
            'Roll Command (deg)': 3 * np.sin(time / 60) + np.random.normal(0, 0.3, 360),
            'Roll Actual (deg)': 3 * np.sin(time / 60) + np.random.normal(0, 0.5, 360),
            'Yaw Command (deg)': np.linspace(0, 360, 360) + np.random.normal(0, 1, 360),
            'Yaw Actual (deg)': np.linspace(0, 360, 360) + np.random.normal(0, 1.5, 360),
        })
        
        # Control performance metrics
        performance = pd.DataFrame({
            'Time (s)': time * 0.1,
            'Stability Index': 0.95 + 0.04 * np.sin(time / 50) + np.random.normal(0, 0.01, 360),
            'Control Effort': 50 + 15 * np.sin(time / 100) + np.random.normal(0, 3, 360),
            'Responsiveness Score': 0.92 + 0.06 * np.sin(time / 80) + np.random.normal(0, 0.02, 360),
        })
        
        return {
            'pid': pid,
            'trajectory_tracking': tracking,
            'attitude': attitude,
            'performance': performance
        }


def format_number(value: float, decimals: int = 2) -> str:
    """Format a number with specified decimal places."""
    return f"{value:.{decimals}f}"


def format_large_number(value: float) -> str:
    """Format large numbers with K/M/B suffixes."""
    if abs(value) >= 1_000_000_000:
        return f"{value / 1_000_000_000:.1f}B"
    elif abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    elif abs(value) >= 1_000:
        return f"{value / 1_000:.1f}K"
    return f"{value:.2f}"


# ============================================================================
# Standalone generation functions for Mission Parameters workflow
# ============================================================================

def generate_mission_data(config: Dict) -> pd.DataFrame:
    """
    Generate mission data from configuration parameters.
    
    Args:
        config: Dictionary with mission parameters
        
    Returns:
        DataFrame with waypoint data
    """
    np.random.seed(42)
    
    num_waypoints = config.get("num_waypoints", 8)
    distance_km = config.get("distance_km", 25.0)
    cruise_altitude = config.get("cruise_altitude", 300)
    cruise_speed = config.get("cruise_speed", 50)
    
    # Generate waypoints along a path
    latitudes = np.linspace(13.0, 13.1, num_waypoints)
    longitudes = np.linspace(77.5, 77.6, num_waypoints)
    
    # Altitude profile: climb to cruise, maintain, descend
    altitudes = np.ones(num_waypoints) * cruise_altitude
    altitudes[0] = 50  # Start at 50m
    altitudes[-1] = 50  # End at 50m
    altitudes[1:int(num_waypoints*0.3)] = np.linspace(50, cruise_altitude, int(num_waypoints*0.3)-1)
    altitudes[int(num_waypoints*0.7):] = np.linspace(cruise_altitude, 50, num_waypoints - int(num_waypoints*0.7))
    
    # Distance between waypoints
    wp_distances = np.full(num_waypoints - 1, distance_km / (num_waypoints - 1))
    distance_to_next = np.append(wp_distances, 0)
    
    # Duration to next waypoint (in minutes)
    duration_to_next = (distance_to_next / cruise_speed * 60)
    
    return pd.DataFrame({
        'waypoint_id': range(1, num_waypoints + 1),
        'latitude': latitudes,
        'longitude': longitudes,
        'altitude': altitudes,
        'speed': np.full(num_waypoints, cruise_speed),
        'distance_to_next': distance_to_next,
        'duration_to_next': duration_to_next,
        'timestamp': pd.date_range(start='2024-01-01 12:00:00', periods=num_waypoints, freq='5min')
    })


def generate_perception_data(mission_data: pd.DataFrame) -> pd.DataFrame:
    """
    Generate perception layer data based on mission.
    
    Args:
        mission_data: Mission waypoint data
        
    Returns:
        DataFrame with perception analysis
    """
    np.random.seed(42)
    
    num_points = len(mission_data)
    
    return pd.DataFrame({
        'waypoint_id': mission_data['waypoint_id'],
        'terrain_elevation': mission_data['altitude'] + np.random.uniform(-20, 50, num_points),
        'slope': np.random.uniform(0, 45, num_points),
        'wind_speed': np.random.uniform(5, 25, num_points),
        'wind_direction': np.random.uniform(0, 360, num_points),
        'threat_level': np.random.uniform(0, 1, num_points),
        'obstacle_count': np.random.randint(0, 10, num_points),
        'visibility': np.random.uniform(5, 20, num_points),
    })


def generate_planning_data(mission_data: pd.DataFrame, perception_data: pd.DataFrame) -> pd.DataFrame:
    """
    Generate planning layer data based on mission and perception.
    
    Args:
        mission_data: Mission waypoint data
        perception_data: Perception analysis data
        
    Returns:
        DataFrame with planning analysis
    """
    np.random.seed(42)
    
    num_points = len(mission_data)
    
    return pd.DataFrame({
        'waypoint_id': mission_data['waypoint_id'],
        'planned_altitude': mission_data['altitude'],
        'planned_speed': mission_data['speed'],
        'safety_margin': np.random.uniform(0.3, 0.9, num_points),
        'energy_required': mission_data['duration_to_next'] * np.random.uniform(5, 15, num_points),
        'risk_score': perception_data['threat_level'],
        'path_deviation': np.random.uniform(-5, 5, num_points),
        'clearance_altitude': perception_data['terrain_elevation'] + np.random.uniform(50, 100, num_points),
    })


def generate_vehicle_data(mission_data: pd.DataFrame, planning_data: pd.DataFrame) -> pd.DataFrame:
    """
    Generate vehicle layer data based on mission and planning.
    
    Args:
        mission_data: Mission waypoint data
        planning_data: Planning analysis data
        
    Returns:
        DataFrame with vehicle simulation data
    """
    np.random.seed(42)
    
    num_points = len(mission_data)
    
    return pd.DataFrame({
        'waypoint_id': mission_data['waypoint_id'],
        'actual_altitude': planning_data['planned_altitude'] + np.random.normal(0, 5, num_points),
        'actual_speed': planning_data['planned_speed'] + np.random.normal(0, 2, num_points),
        'pitch': np.random.uniform(-10, 10, num_points),
        'roll': np.random.uniform(-5, 5, num_points),
        'yaw': np.linspace(0, 360, num_points),
        'battery_soc': 100 - (np.arange(num_points) / num_points * 40),
        'motor_rpm': np.random.uniform(1000, 3000, num_points),
        'temperature': 25 + np.random.uniform(0, 25, num_points),
    })


def generate_control_data(mission_data: pd.DataFrame, vehicle_data: pd.DataFrame) -> pd.DataFrame:
    """
    Generate control layer data based on mission and vehicle.
    
    Args:
        mission_data: Mission waypoint data
        vehicle_data: Vehicle simulation data
        
    Returns:
        DataFrame with control system data
    """
    np.random.seed(42)
    
    num_points = len(mission_data)
    
    return pd.DataFrame({
        'waypoint_id': mission_data['waypoint_id'],
        'command_altitude': mission_data['altitude'],
        'command_speed': mission_data['speed'],
        'position_error': np.random.uniform(-10, 10, num_points),
        'velocity_error': np.random.uniform(-2, 2, num_points),
        'altitude_error': np.random.uniform(-5, 5, num_points),
        'heading_error': np.random.uniform(-10, 10, num_points),
        'pid_output': np.random.uniform(-1, 1, num_points),
        'control_mode': ['CRUISE'] * num_points,
        'tracking_success': 95 + np.random.uniform(-5, 5, num_points),
    })
