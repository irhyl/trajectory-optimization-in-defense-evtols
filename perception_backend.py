import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

class TerrainModel:
    def __init__(self, resolution):
        self.resolution = resolution
        # Generate grid
        x = np.linspace(0, 100, int(100/resolution))
        y = np.linspace(0, 100, int(100/resolution))
        self.X, self.Y = np.meshgrid(x, y)
        # Generate elevation data
        self.Z = 10 * np.sin(self.X/10) + 10 * np.cos(self.Y/10) + 5 * np.random.rand(*self.X.shape)

class WindModel:
    def __init__(self, intensity):
        self.intensity = intensity
        x = np.linspace(0, 100, 20)
        y = np.linspace(0, 100, 20)
        self.X, self.Y = np.meshgrid(x, y)
        self.U = intensity * np.cos(self.X/10)
        self.V = intensity * np.sin(self.Y/10)
        self.speeds = np.sqrt(self.U**2 + self.V**2)

class ThreatModel:
    def __init__(self, count, range_km):
        self.df = pd.DataFrame({
            'x': np.random.uniform(0, 100, count),
            'y': np.random.uniform(0, 100, count),
            'z': np.random.uniform(0, 50, count),
            'radius': np.random.uniform(5, 15, count),
            'type': np.random.choice(['Radar', 'SAM', 'Jammer'], count)
        })

class PerceptionManager:
    def __init__(self, terrain, wind, threat):
        self.terrain = terrain
        self.wind = wind
        self.threat = threat
        self.obstacles = pd.DataFrame({
            'Longitude': np.random.uniform(0, 100, 10),
            'Latitude': np.random.uniform(0, 100, 10),
            'Height (m)': np.random.uniform(10, 50, 10),
            'Clearance (m)': np.random.uniform(5, 20, 10)
        })
        self.landing_zones = pd.DataFrame({
            'Center Lon': np.random.uniform(0, 100, 5),
            'Center Lat': np.random.uniform(0, 100, 5),
            'Area (m²)': np.random.uniform(500, 2000, 5),
            'Feasibility Score': np.random.uniform(0.5, 1.0, 5)
        })

    def get_terrain_3d_figure(self):
        return go.Figure(data=[go.Surface(z=self.terrain.Z, x=self.terrain.X, y=self.terrain.Y)])

    def get_wind_3d_figure(self):
        fig = go.Figure()
        fig.add_trace(go.Heatmap(z=self.wind.speeds, x=self.wind.X[0,:], y=self.wind.Y[:,0], hoverinfo='z'))
        step = 2
        fig.add_trace(go.Scatter(
            x=self.wind.X[::step, ::step].flatten(),
            y=self.wind.Y[::step, ::step].flatten(),
            mode='markers',
            marker=dict(symbol='arrow', size=10, color='black', 
                        angle=np.degrees(np.arctan2(self.wind.V[::step, ::step], self.wind.U[::step, ::step])).flatten())
        ))
        return fig

    def get_threat_3d_figure(self):
        df = self.threat.df
        return go.Figure(data=[go.Scatter3d(
            x=df['x'], y=df['y'], z=df['z'], mode='markers',
            marker=dict(size=df['radius']*2, color=df['radius']),
            text=df['type']
        )])

    def get_fused_3d_figure(self):
        return go.Figure(data=[go.Surface(z=self.terrain.Z, x=self.terrain.X, y=self.terrain.Y, opacity=0.8)])

    def get_fusion_summary(self):
        return {
            'risk_map_mean': 0.45,
            'risk_map_max': 0.92,
            'feasibility_mean': 0.78,
            'energy_cost_mean': 12.5
        }

def render_perception_controls():
    c1, c2 = st.columns(2)
    with c1:
        res = st.slider("Terrain Resolution", 0.1, 5.0, 1.0)
        wind = st.slider("Wind Intensity", 0.0, 20.0, 5.0)
    with c2:
        threats = st.number_input("Threat Count", 1, 20, 5)
        rng = st.slider("Detection Range", 10, 100, 50)
    
    tm = TerrainModel(res)
    wm = WindModel(wind)
    thm = ThreatModel(threats, rng)
    manager = PerceptionManager(tm, wm, thm)
    st.session_state['perception_manager'] = manager
    return manager

def export_perception_data():
    if 'perception_manager' not in st.session_state:
        return {}
    mgr = st.session_state['perception_manager']
    
    return {
        'terrain': pd.DataFrame({'x': mgr.terrain.X.flatten(), 'y': mgr.terrain.Y.flatten(), 'z': mgr.terrain.Z.flatten()}).head(1000),
        'wind': pd.DataFrame({'x': mgr.wind.X.flatten(), 'y': mgr.wind.Y.flatten(), 'speed': mgr.wind.speeds.flatten()}).head(1000),
        'threat': mgr.threat.df, 'obstacle': mgr.obstacles, 'landing_zones': mgr.landing_zones
    }