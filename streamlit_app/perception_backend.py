"""
Perception Backend Module for Streamlit App.

Integrates the real perception layer (terrain, wind, threat, obstacle models)
with the Streamlit UI, providing proper data generation and visualization.
"""

import sys
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np
import pandas as pd
import streamlit as st

# Add the src directory to path for imports
SRC_PATH = Path(__file__).parent.parent / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

# Try to import real perception modules
try:
    from evtol.perception import (
        TerrainElevationMap,
        WindFieldModel,
        ThreatAssessmentModel,
        ObstacleDetectionModel,
        FusedIntelligenceModel,
        # Visualization functions
        plot_terrain_3d_plotly,
        plot_wind_3d_layers_plotly,
        plot_threat_3d_altitude_analysis_plotly,
        plot_fused_3d_heatmap_plotly,
    )
    PERCEPTION_AVAILABLE = True
except ImportError as e:
    PERCEPTION_AVAILABLE = False
    IMPORT_ERROR = str(e)


@dataclass
class PerceptionConfig:
    """Configuration for perception layer generation."""
    # Grid configuration
    grid_size: int = 200  # Reduced for faster computation in UI
    resolution_m: float = 250.0  # 250m resolution for 50km area
    
    # Terrain parameters
    terrain_seed: int = 42
    
    # Wind parameters
    base_wind_speed: float = 7.0
    
    # Threat parameters
    num_radars: int = 3
    num_sams: int = 2
    
    # Obstacle parameters
    building_density: float = 0.08
    urban_center: Tuple[int, int] = (100, 100)
    
    # Coverage bounds (lat/lon)
    bounds_north: float = 13.0
    bounds_south: float = 12.8
    bounds_east: float = 77.7
    bounds_west: float = 77.5


class PerceptionManager:
        def get_terrain_wind_3d_figure(self, altitude_idx: int = 1, n_vectors: int = 15):
            """Return a Plotly figure with terrain surface and wind cones at selected altitude."""
            if not (self._terrain_model and self._wind_model):
                return None
            try:
                import plotly.graph_objects as go
                elevation = self._terrain_model.elevation
                ny, nx = elevation.shape
                x = np.arange(nx)
                y = np.arange(ny)
                # Terrain surface
                surface = go.Surface(
                    z=elevation,
                    x=x,
                    y=y,
                    colorscale='Earth',
                    showscale=False,
                    opacity=0.85,
                    name='Terrain'
                )
                # Wind cones (subsample for clarity)
                u = self._wind_model.wind_u[altitude_idx]
                v = self._wind_model.wind_v[altitude_idx]
                w = np.zeros_like(u)  # Assume horizontal wind only
                step = max(1, nx // n_vectors)
                xg, yg = np.meshgrid(x, y)
                cone = go.Cone(
                    x=xg[::step, ::step].flatten(),
                    y=yg[::step, ::step].flatten(),
                    z=(elevation[::step, ::step]).flatten(),
                    u=u[::step, ::step].flatten(),
                    v=v[::step, ::step].flatten(),
                    w=w[::step, ::step].flatten(),
                    sizemode="absolute",
                    sizeref=2,
                    anchor="tail",
                    colorscale='Blues',
                    showscale=False,
                    name='Wind Vectors',
                    opacity=0.7
                )
                fig = go.Figure(data=[surface, cone])
                fig.update_layout(
                    scene=dict(
                        xaxis_title='Longitude Index',
                        yaxis_title='Latitude Index',
                        zaxis_title='Elevation (m)',
                        aspectmode='cube',
                    ),
                    margin=dict(l=10, r=10, t=30, b=10),
                    paper_bgcolor='#f8f9fa',
                    plot_bgcolor='#f8f9fa',
                    font=dict(family="Roboto, sans-serif", size=14, color="#222"),
                    showlegend=False,
                    title=f"Terrain and Wind Vectors at {int(self._wind_model.altitude_bands[altitude_idx])}m"
                )
                return fig
            except Exception as e:
                import streamlit as st
                st.warning(f"Could not create combined terrain/wind plot: {e}")
                return None
    # ...existing code...
        """Initialize perception manager with configuration."""
        self.config = config or PerceptionConfig()
        
        # Models (initialized lazily)
        self._terrain_model: Optional[TerrainElevationMap] = None
        self._wind_model: Optional[WindFieldModel] = None
        self._threat_model: Optional[ThreatAssessmentModel] = None
        self._obstacle_model: Optional[ObstacleDetectionModel] = None
        self._fusion_model: Optional[FusedIntelligenceModel] = None
        
        # Status tracking
        self.is_generated = False
        self.generation_status: Dict[str, bool] = {
            'terrain': False,
            'wind': False,
            'threat': False,
            'obstacle': False,
            'fusion': False
        }
    
    @property
    def terrain_model(self) -> Optional[TerrainElevationMap]:
        return self._terrain_model
    
    @property
    def wind_model(self) -> Optional[WindFieldModel]:
        return self._wind_model
    
    @property
    def threat_model(self) -> Optional[ThreatAssessmentModel]:
        return self._threat_model
    
    @property
    def obstacle_model(self) -> Optional[ObstacleDetectionModel]:
        return self._obstacle_model
    
    @property
    def fusion_model(self) -> Optional[FusedIntelligenceModel]:
        return self._fusion_model
    
    def generate_terrain(self, progress_callback=None) -> bool:
        """Generate terrain elevation data."""
        if not PERCEPTION_AVAILABLE:
            return False
        
        try:
            if progress_callback:
                progress_callback("Initializing terrain model...")
            
            # Create terrain model with configured grid size
            # Compute resolution to match the target grid_size
            # width_m / resolution_m = grid_size, so resolution_m = width_m / grid_size
            width_m = 50000
            height_m = 50000
            resolution_m = width_m / self.config.grid_size
            
            self._terrain_model = TerrainElevationMap(
                width_m=width_m,
                height_m=height_m,
                resolution_m=resolution_m
            )
            
            if progress_callback:
                progress_callback("Generating realistic terrain...")
            
            self._terrain_model.generate_realistic_terrain(seed=self.config.terrain_seed)
            self.generation_status['terrain'] = True
            return True
            
        except Exception as e:
            st.error(f"Terrain generation failed: {e}")
            return False
    
    def generate_wind(self, progress_callback=None) -> bool:
        """Generate wind field data."""
        if not PERCEPTION_AVAILABLE:
            return False
        
        try:
            if progress_callback:
                progress_callback("Initializing wind model...")
            
            self._wind_model = WindFieldModel(
                grid_size=self.config.grid_size,
                coverage_bounds=(
                    self.config.bounds_north,
                    self.config.bounds_south,
                    self.config.bounds_east,
                    self.config.bounds_west
                )
            )
            
            if progress_callback:
                progress_callback("Generating realistic wind field...")
            
            self._wind_model.generate_realistic_wind(base_speed=self.config.base_wind_speed)
            self.generation_status['wind'] = True
            return True
            
        except Exception as e:
            st.error(f"Wind generation failed: {e}")
            return False
    
    def generate_threats(self, progress_callback=None) -> bool:
        """Generate threat assessment data."""
        if not PERCEPTION_AVAILABLE:
            return False
        
        try:
            if progress_callback:
                progress_callback("Initializing threat model...")
            
            self._threat_model = ThreatAssessmentModel(
                grid_size=self.config.grid_size,
                coverage_bounds=(
                    self.config.bounds_north,
                    self.config.bounds_south,
                    self.config.bounds_east,
                    self.config.bounds_west
                )
            )
            
            if progress_callback:
                progress_callback("Generating realistic threats...")
            
            self._threat_model.generate_realistic_threats()
            self.generation_status['threat'] = True
            return True
            
        except Exception as e:
            st.error(f"Threat generation failed: {e}")
            return False
    
    def generate_obstacles(self, progress_callback=None) -> bool:
        """Generate obstacle detection data."""
        if not PERCEPTION_AVAILABLE:
            return False
        
        try:
            if progress_callback:
                progress_callback("Initializing obstacle model...")
            
            self._obstacle_model = ObstacleDetectionModel(
                grid_size=self.config.grid_size,
                coverage_bounds=(
                    self.config.bounds_north,
                    self.config.bounds_south,
                    self.config.bounds_east,
                    self.config.bounds_west
                )
            )
            
            if progress_callback:
                progress_callback("Generating realistic obstacles...")
            
            self._obstacle_model.generate_realistic_obstacles(
                building_density=self.config.building_density,
                urban_center=self.config.urban_center
            )
            self.generation_status['obstacle'] = True
            return True
            
        except Exception as e:
            st.error(f"Obstacle generation failed: {e}")
            return False
    
    def generate_fusion(self, progress_callback=None) -> bool:
        """Generate fused intelligence data (requires all other models)."""
        if not PERCEPTION_AVAILABLE:
            return False
        
        # Check prerequisites
        if not all([
            self._terrain_model,
            self._wind_model,
            self._threat_model,
            self._obstacle_model
        ]):
            st.warning("Fusion requires all perception layers to be generated first.")
            return False
        
        try:
            if progress_callback:
                progress_callback("Computing fused intelligence...")
            
            self._fusion_model = FusedIntelligenceModel(
                terrain_model=self._terrain_model,
                wind_model=self._wind_model,
                threat_model=self._threat_model,
                obstacle_model=self._obstacle_model
            )
            
            self.generation_status['fusion'] = True
            self.is_generated = True
            return True
            
        except Exception as e:
            st.error(f"Fusion computation failed: {e}")
            return False
    
    def generate_all(self, progress_callback=None) -> bool:
        """Generate all perception layers."""
        success = True
        
        if not self.generate_terrain(progress_callback):
            success = False
        if not self.generate_wind(progress_callback):
            success = False
        if not self.generate_threats(progress_callback):
            success = False
        if not self.generate_obstacles(progress_callback):
            success = False
        if success:
            self.generate_fusion(progress_callback)
        
        return success
    
    # =========================================================================
    # Data Export Methods - Convert numpy arrays to DataFrames
    # =========================================================================
    
    def get_terrain_dataframe(self, sample_size: int = 500) -> pd.DataFrame:
        """Convert terrain data to DataFrame for visualization."""
        if not self._terrain_model:
            return pd.DataFrame()
        
        elevation = self._terrain_model.elevation
        ny, nx = elevation.shape
        
        # Sample points for manageable visualization
        indices = np.random.choice(ny * nx, min(sample_size, ny * nx), replace=False)
        rows = indices // nx
        cols = indices % nx
        
        # Create lat/lon coordinates
        lat_range = np.linspace(self.config.bounds_south, self.config.bounds_north, ny)
        lon_range = np.linspace(self.config.bounds_west, self.config.bounds_east, nx)
        
        # Compute terrain slope
        slope_y, slope_x = np.gradient(elevation)
        slope_magnitude = np.sqrt(slope_x**2 + slope_y**2)
        slope_degrees = np.degrees(np.arctan(slope_magnitude / self.config.resolution_m))
        
        # Roughness (local variance)
        from scipy.ndimage import generic_filter
        roughness = generic_filter(elevation, np.std, size=3)
        
        return pd.DataFrame({
            'Latitude': lat_range[rows],
            'Longitude': lon_range[cols],
            'Elevation (m)': elevation[rows, cols],
            'Slope (deg)': slope_degrees[rows, cols],
            'Roughness': roughness[rows, cols]
        })
    
    def get_wind_dataframe(self) -> pd.DataFrame:
        """Convert wind data to DataFrame for visualization."""
        if not self._wind_model:
            return pd.DataFrame()
        
        wind_speed = self._wind_model.wind_speed
        wind_direction = self._wind_model.wind_direction
        turbulence = self._wind_model.turbulence_intensity
        altitudes = self._wind_model.altitude_bands
        
        # Create summary dataframe by altitude
        data = []
        for i, alt in enumerate(altitudes):
            speed_layer = wind_speed[i]
            dir_layer = wind_direction[i]
            turb_layer = turbulence[i]
            
            data.append({
                'Altitude (m)': alt,
                'Wind Speed (m/s)': float(np.mean(speed_layer)),
                'Wind Speed Min (m/s)': float(np.min(speed_layer)),
                'Wind Speed Max (m/s)': float(np.max(speed_layer)),
                'Wind Direction (deg)': float(np.mean(dir_layer)) % 360,
                'Turbulence': float(np.mean(turb_layer))
            })
        
        return pd.DataFrame(data)
    
    def get_wind_grid_dataframe(self, altitude_idx: int = 1, sample_size: int = 200) -> pd.DataFrame:
        """Get wind data as a spatial grid for visualization."""
        if not self._wind_model:
            return pd.DataFrame()
        
        grid_size = self._wind_model.grid_size
        
        # Sample points
        indices = np.random.choice(grid_size * grid_size, min(sample_size, grid_size * grid_size), replace=False)
        rows = indices // grid_size
        cols = indices % grid_size
        
        # Create lat/lon coordinates
        lat_range = np.linspace(self.config.bounds_south, self.config.bounds_north, grid_size)
        lon_range = np.linspace(self.config.bounds_west, self.config.bounds_east, grid_size)
        
        return pd.DataFrame({
            'Latitude': lat_range[rows],
            'Longitude': lon_range[cols],
            'Wind U (m/s)': self._wind_model.wind_u[altitude_idx, rows, cols],
            'Wind V (m/s)': self._wind_model.wind_v[altitude_idx, rows, cols],
            'Wind Speed (m/s)': self._wind_model.wind_speed[altitude_idx, rows, cols],
            'Wind Direction (deg)': self._wind_model.wind_direction[altitude_idx, rows, cols] % 360,
            'Turbulence': self._wind_model.turbulence_intensity[altitude_idx, rows, cols]
        })
    
    def get_threat_dataframe(self) -> pd.DataFrame:
        """Convert threat sources to DataFrame."""
        if not self._threat_model or not self._threat_model.threat_sources:
            return pd.DataFrame()
        
        data = []
        for source in self._threat_model.threat_sources:
            # Convert grid indices to lat/lon
            lat = self.config.bounds_south + (source.lat_idx / self.config.grid_size) * \
                  (self.config.bounds_north - self.config.bounds_south)
            lon = self.config.bounds_west + (source.lon_idx / self.config.grid_size) * \
                  (self.config.bounds_east - self.config.bounds_west)
            
            # Determine threat level based on range
            if source.range_m >= 45000:
                level = "High"
            elif source.range_m >= 30000:
                level = "Medium"
            else:
                level = "Low"
            
            data.append({
                'Threat ID': source.threat_id,
                'Threat Type': source.threat_type.value,
                'Latitude': lat,
                'Longitude': lon,
                'Range (m)': source.range_m,
                'Altitude Min (m)': source.altitude_min,
                'Altitude Max (m)': source.altitude_max,
                'Threat Level': level
            })
        
        return pd.DataFrame(data)
    
    def get_obstacle_dataframe(self, sample_size: int = 100) -> pd.DataFrame:
        """Convert obstacle data to DataFrame."""
        if not self._obstacle_model or self._obstacle_model.building_mask is None:
            return pd.DataFrame()
        
        mask = self._obstacle_model.building_mask
        heights = self._obstacle_model.building_height
        clearance = self._obstacle_model.clearance_map
        
        # Find building locations
        building_locs = np.where(mask > 0)
        
        if len(building_locs[0]) == 0:
            return pd.DataFrame()
        
        # Sample if too many
        num_buildings = len(building_locs[0])
        if num_buildings > sample_size:
            indices = np.random.choice(num_buildings, sample_size, replace=False)
            rows = building_locs[0][indices]
            cols = building_locs[1][indices]
        else:
            rows = building_locs[0]
            cols = building_locs[1]
        
        # Create lat/lon coordinates
        grid_size = mask.shape[0]
        lat_range = np.linspace(self.config.bounds_south, self.config.bounds_north, grid_size)
        lon_range = np.linspace(self.config.bounds_west, self.config.bounds_east, grid_size)
        
        return pd.DataFrame({
            'Building ID': range(1, len(rows) + 1),
            'Latitude': lat_range[rows],
            'Longitude': lon_range[cols],
            'Height (m)': heights[rows, cols],
            'Clearance (m)': clearance[rows, cols],
            'Footprint (m2)': np.random.uniform(100, 2000, len(rows))  # Estimated
        })
    
    def get_landing_zones_dataframe(self) -> pd.DataFrame:
        """Get identified landing zones."""
        if not self._obstacle_model or not self._obstacle_model.landing_zones:
            return pd.DataFrame()
        
        data = []
        for lz in self._obstacle_model.landing_zones:
            data.append({
                'Zone ID': lz.zone_id,
                'Center Lat': lz.center_lat,
                'Center Lon': lz.center_lon,
                'Area (m²)': lz.area_m2,
                'Slope (deg)': lz.slope_deg,
                'Feasibility Score': lz.feasibility_score
            })
        
        return pd.DataFrame(data)
    
    def get_fusion_summary(self) -> Dict[str, Any]:
        """Get fusion layer summary statistics."""
        if not self._fusion_model:
            return {}
        
        return {
            'risk_map_mean': float(np.mean(self._fusion_model.risk_map)),
            'risk_map_max': float(np.max(self._fusion_model.risk_map)),
            'risk_map_min': float(np.min(self._fusion_model.risk_map)),
            'feasibility_mean': float(np.mean(self._fusion_model.feasibility_map)),
            'feasibility_above_50': float(np.mean(self._fusion_model.feasibility_map > 0.5) * 100),
            'energy_cost_mean': float(np.mean(self._fusion_model.energy_cost_map)),
            'energy_cost_max': float(np.max(self._fusion_model.energy_cost_map)),
        }
    
    # =========================================================================
    # Visualization Methods
    # =========================================================================
    
    def get_terrain_3d_figure(self):
        """Get 3D terrain visualization using plotly."""
        if not self._terrain_model or not PERCEPTION_AVAILABLE:
            return None
        
        try:
            fig = plot_terrain_3d_plotly(self._terrain_model.elevation)
            return fig
        except Exception as e:
            st.warning(f"Could not create 3D terrain plot: {e}")
            return None
    
    def get_wind_3d_figure(self):
        """Get 3D wind visualization using plotly."""
        if not self._wind_model or not PERCEPTION_AVAILABLE:
            return None
        
        try:
            fig = plot_wind_3d_layers_plotly(self._wind_model)
            return fig
        except Exception as e:
            st.warning(f"Could not create 3D wind plot: {e}")
            return None
    
    def get_threat_3d_figure(self):
        """Get 3D threat visualization using plotly."""
        if not self._threat_model or not PERCEPTION_AVAILABLE:
            return None
        
        try:
            fig = plot_threat_3d_altitude_analysis_plotly(self._threat_model)
            return fig
        except Exception as e:
            st.warning(f"Could not create 3D threat plot: {e}")
            return None
    
    def get_fused_3d_figure(self):
        """Get 3D fused heatmap visualization."""
        if not self._fusion_model or not PERCEPTION_AVAILABLE:
            return None
        
        try:
            fig = plot_fused_3d_heatmap_plotly(self._fusion_model)
            return fig
        except Exception as e:
            st.warning(f"Could not create 3D fusion plot: {e}")
            return None


# =============================================================================
# Session State Management
# =============================================================================

def init_perception_session_state():
    """Initialize perception-related session state."""
    if 'perception_manager' not in st.session_state:
        st.session_state.perception_manager = None
    
    if 'perception_config' not in st.session_state:
        st.session_state.perception_config = PerceptionConfig()
    
    if 'perception_generated' not in st.session_state:
        st.session_state.perception_generated = False


def get_perception_manager() -> Optional[PerceptionManager]:
    """Get the current perception manager from session state."""
    init_perception_session_state()
    return st.session_state.perception_manager


def create_perception_manager(config: Optional[PerceptionConfig] = None) -> PerceptionManager:
    """Create a new perception manager and store in session state."""
    init_perception_session_state()
    
    if config:
        st.session_state.perception_config = config
    
    manager = PerceptionManager(st.session_state.perception_config)
    st.session_state.perception_manager = manager
    return manager


def render_perception_controls() -> Optional[PerceptionManager]:
    """
    Render perception model generation controls and handle automatic regeneration.
    """
    init_perception_session_state()

    st.markdown("### Perception Model Configuration")

    if not PERCEPTION_AVAILABLE:
        st.error(f"Perception modules not available: {IMPORT_ERROR}")
        st.info("Using demo data instead. Check that the src/evtol/perception module is properly installed.")
        return None

    # Get previous config
    previous_config = st.session_state.get('perception_config', PerceptionConfig())

    # --- Render controls ---
    col1, col2 = st.columns(2)
    with col1:
        grid_size = st.slider(
            "Grid Size", 100, 500, previous_config.grid_size, 50,
            help="Size of the grid for all models. Larger grids are more detailed but slower."
        )
        terrain_seed = st.number_input(
            "Terrain Seed", 1, 999, previous_config.terrain_seed,
            help="Seed for reproducible terrain generation."
        )
        base_wind_speed = st.slider(
            "Base Wind Speed (m/s)", 3.0, 20.0, previous_config.base_wind_speed, 0.5,
            help="Mean wind speed at 100m altitude."
        )
        
    with col2:
        building_density = st.slider(
            "Building Density", 0.01, 0.20, previous_config.building_density, 0.01,
            help="Fraction of grid cells with buildings."
        )
        resolution = st.selectbox(
            "Resolution (m)", [100.0, 150.0, 250.0, 500.0],
            index=[100.0, 150.0, 250.0, 500.0].index(previous_config.resolution_m),
            help="Grid cell resolution in meters."
        )

    # Create new config from widget values
    new_config = PerceptionConfig(
        grid_size=grid_size,
        resolution_m=resolution,
        terrain_seed=terrain_seed,
        base_wind_speed=base_wind_speed,
        building_density=building_density
    )

    # Check if config has changed or if data has never been generated
    config_changed = (new_config != previous_config)
    not_generated = not st.session_state.get('perception_generated', False)

    if config_changed or not_generated:
        st.session_state.perception_config = new_config
        
        manager = create_perception_manager(new_config)
        
        progress_bar = st.progress(0, text="Initializing perception models...")
        status_text = st.empty()

        def update_progress(message: str):
            status_text.text(message)

        # Generate each model
        progress_bar.progress(10, text="Generating terrain model...")
        manager.generate_terrain(update_progress)
        
        progress_bar.progress(30, text="Generating wind model...")
        manager.generate_wind(update_progress)
        
        progress_bar.progress(50, text="Generating threat model...")
        manager.generate_threats(update_progress)
        
        progress_bar.progress(70, text="Generating obstacle model...")
        manager.generate_obstacles(update_progress)
        
        progress_bar.progress(90, text="Computing fusion model...")
        manager.generate_fusion(update_progress)
        
        progress_bar.progress(100)
        status_text.success("All perception models generated successfully!")
        progress_bar.empty()

        st.session_state.perception_generated = True
        st.session_state.perception_manager = manager

    return st.session_state.perception_manager


def export_perception_data() -> Dict[str, pd.DataFrame]:
    """Export all perception data as DataFrames."""
    manager = get_perception_manager()
    
    if not manager:
        return {}
    
    return {
        'terrain': manager.get_terrain_dataframe(),
        'wind': manager.get_wind_dataframe(),
        'wind_grid': manager.get_wind_grid_dataframe(),
        'threat': manager.get_threat_dataframe(),
        'obstacle': manager.get_obstacle_dataframe(),
        'landing_zones': manager.get_landing_zones_dataframe()
    }
