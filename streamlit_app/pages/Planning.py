import streamlit as st
import sys
import os
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

# Ensure src is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

try:
    from planning_backend import render_planning_controls, get_planning_manager
    from perception_backend import render_perception_controls
except ImportError:
    # Fallback for local testing structure
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from planning_backend import render_planning_controls, get_planning_manager
    from perception_backend import render_perception_controls

# --- Page Configuration ---
st.set_page_config(
    page_title="Trajectory Planning - Defense eVTOLs",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Color Palettes ---
COLORS = {
    "trajectory": "Plasma",
    "waypoints": "#EF553B",
    "ground": "#F1E9D2",
    "text": "#333333"
}

# --- Header ---
st.title("Trajectory Planning")
st.markdown("""
<div style='font-family:Roboto,sans-serif;font-size:1.0rem;color:#555;'>
Configure mission parameters and generate optimal flight paths. 
This module integrates environmental data to ensure collision-free, energy-efficient trajectories.
</div>
<hr>
""", unsafe_allow_html=True)

# --- Backend Initialization ---
# Ensure perception manager exists for environment context
if 'perception_manager' not in st.session_state:
    # Initialize with defaults if not coming from Perception page
    from perception_backend import TerrainModel, WindModel, ThreatModel, PerceptionManager
    st.session_state['perception_manager'] = PerceptionManager(TerrainModel(1.0), WindModel(5.0), ThreatModel(5, 50))

perception_mgr = st.session_state['perception_manager']
planning_mgr = get_planning_manager()

# --- Controls ---
with st.expander("Mission Parameters & Constraints", expanded=True):
    params = render_planning_controls()

# --- Trajectory Generation ---
# Generate if recompute clicked or if no data exists yet
if params['recompute'] or 'trajectory_data' not in st.session_state:
    with st.spinner("Optimizing Trajectory..."):
        traj_df = planning_mgr.plan_trajectory(
            params['start'], 
            params['end'], 
            obstacles=perception_mgr.obstacles,
            steps=100
        )
        metrics = planning_mgr.calculate_metrics(traj_df)
        constraints_df = planning_mgr.check_constraints(traj_df, params['constraints'])
        
        st.session_state['trajectory_data'] = traj_df
        st.session_state['planning_metrics'] = metrics
        st.session_state['constraints_status'] = constraints_df

traj_df = st.session_state['trajectory_data']
metrics = st.session_state['planning_metrics']
constraints_df = st.session_state['constraints_status']

# --- Visualizations ---

# 1. 3D Trajectory Plot
st.markdown("<b>3D Mission Visualization</b>", unsafe_allow_html=True)

# Base figure from terrain (Context)
fig_3d = perception_mgr.get_terrain_3d_figure()

# Add Trajectory Trace
fig_3d.add_trace(go.Scatter3d(
    x=traj_df['x'], y=traj_df['y'], z=traj_df['z'],
    mode='lines',
    line=dict(color=traj_df['speed'], colorscale=COLORS['trajectory'], width=6),
    name='Planned Path',
    hoverinfo='text',
    text=[f"Time: {t:.1f}s<br>Alt: {z:.1f}m<br>Vel: {v:.1f}m/s" 
          for t, z, v in zip(traj_df['time'], traj_df['z'], traj_df['speed'])]
))

# Add Ground Shadow (Projection)
fig_3d.add_trace(go.Scatter3d(
    x=traj_df['x'], y=traj_df['y'], z=[0]*len(traj_df),
    mode='lines',
    line=dict(color='gray', width=3, dash='dot'),
    opacity=0.4,
    name='Ground Track',
    hoverinfo='none'
))

# Add Start/End Waypoints
fig_3d.add_trace(go.Scatter3d(
    x=[params['start'][0], params['end'][0]],
    y=[params['start'][1], params['end'][1]],
    z=[params['start'][2], params['end'][2]],
    mode='markers',
    marker=dict(size=8, color=COLORS['waypoints'], symbol='diamond'),
    name='Waypoints'
))

# Add Obstacles (Context)
obs_df = perception_mgr.obstacles
if not obs_df.empty:
    fig_3d.add_trace(go.Scatter3d(
        x=obs_df['Longitude'], y=obs_df['Latitude'], z=obs_df['Height (m)'],
        mode='markers',
        marker=dict(size=obs_df['Height (m)'], color='rgba(200, 100, 100, 0.5)', symbol='square'),
        name='Obstacles'
    ))

fig_3d.update_layout(
    template="plotly_white",
    margin=dict(l=10, r=10, t=10, b=10),
    font=dict(family="Roboto, sans-serif", size=12, color="#333"),
    scene=dict(
        xaxis_title="X (m)", yaxis_title="Y (m)", zaxis_title="Altitude (m)",
        aspectmode="data"
    ),
    legend=dict(yanchor="top", y=0.95, xanchor="left", x=0.05)
)
st.plotly_chart(fig_3d, use_container_width=True)

# 2. 2D Projections & Diagnostics
c1, c2 = st.columns([2, 1])

with c1:
    st.markdown("<b>Trajectory Projections</b>", unsafe_allow_html=True)
    tab_xy, tab_xz, tab_yz = st.tabs(["Top View (XY)", "Side View (XZ)", "Front View (YZ)"])
    
    with tab_xy:
        fig_xy = px.line(traj_df, x='x', y='y', title=None)
        fig_xy.update_layout(template="plotly_white", xaxis_title="X (m)", yaxis_title="Y (m)", margin=dict(t=20, b=20))
        fig_xy.update_traces(line_color="#636EFA", line_width=3)
        st.plotly_chart(fig_xy, use_container_width=True)
        
    with tab_xz:
        fig_xz = px.line(traj_df, x='x', y='z', title=None)
        fig_xz.update_layout(template="plotly_white", xaxis_title="X (m)", yaxis_title="Altitude (m)", margin=dict(t=20, b=20))
        fig_xz.update_traces(line_color="#636EFA", line_width=3)
        st.plotly_chart(fig_xz, use_container_width=True)

    with tab_yz:
        fig_yz = px.line(traj_df, x='y', y='z', title=None)
        fig_yz.update_layout(template="plotly_white", xaxis_title="Y (m)", yaxis_title="Altitude (m)", margin=dict(t=20, b=20))
        fig_yz.update_traces(line_color="#636EFA", line_width=3)
        st.plotly_chart(fig_yz, use_container_width=True)

with c2:
    st.markdown("<b>Mission Diagnostics</b>", unsafe_allow_html=True)
    
    # Cost Breakdown
    cost_data = pd.DataFrame({
        'Component': ['Energy', 'Time', 'Risk', 'Smoothness'],
        'Cost': [metrics['Energy Cost (kJ)'], metrics['Total Distance (m)']/10, metrics['Risk Score']*100, 15.5]
    })
    fig_cost = px.pie(cost_data, values='Cost', names='Component', hole=0.4, color_discrete_sequence=px.colors.qualitative.Pastel)
    fig_cost.update_layout(template="plotly_white", margin=dict(t=0, b=0, l=0, r=0), height=200, showlegend=True)
    st.plotly_chart(fig_cost, use_container_width=True)

    # Constraint Status
    st.markdown("<div style='font-size:0.9rem; margin-top:10px'><b>Constraint Validation</b></div>", unsafe_allow_html=True)
    for _, row in constraints_df.iterrows():
        color = "green" if row['Status'] == "Satisfied" else "red"
        st.markdown(f"""
        <div style='display:flex; justify-content:space-between; padding:5px; border-bottom:1px solid #eee;'>
            <span>{row['Constraint']}</span>
            <span style='color:{color}; font-weight:bold;'>{row['Status']}</span>
        </div>
        """, unsafe_allow_html=True)

# 3. Time Series Analysis
st.markdown("<b>Flight Telemetry</b>", unsafe_allow_html=True)
ts_vars = st.multiselect("Select Variables to Plot", 
                         ['speed', 'energy', 'vx', 'vy', 'vz', 'ax', 'ay', 'az'], 
                         default=['speed', 'energy'])

if ts_vars:
    fig_ts = px.line(traj_df, x='time', y=ts_vars, markers=False)
    fig_ts.update_layout(
        template="plotly_white", 
        xaxis_title="Time (s)", 
        yaxis_title="Value",
        legend_title="Variable",
        margin=dict(t=20, b=20),
        font=dict(family="Roboto, sans-serif", size=12, color="#333")
    )
    st.plotly_chart(fig_ts, use_container_width=True)

# --- Exports ---
st.markdown("---")
c_ex1, c_ex2 = st.columns([1, 5])
with c_ex1:
    st.download_button(
        "Download Trajectory CSV", 
        traj_df.to_csv(index=False).encode('utf-8'), 
        "trajectory_plan.csv",
        mime="text/csv",
        use_container_width=True
    )
with c_ex2:
    st.download_button(
        "Download Mission Report",
        "Mission Report Mock Data...",
        "mission_report.txt",
        use_container_width=False
    )