import streamlit as st
import sys
import os
import plotly.express as px

# Ensure src is in sys.path for backend imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

try:
    from perception_backend import render_perception_controls, export_perception_data
except ImportError:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from perception_backend import render_perception_controls, export_perception_data

# --- Page Configuration ---
st.set_page_config(
    page_title="Perception Models - Defense eVTOLs",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Color Palettes ---
COLOR_PALETTES = {
    "terrain": ["#F1E9D2", "#E2EDE3", "#C5D0C2"],
    "wind": ["#DDE7EF", "#C9D8E5", "#C8D4DD"],
    "threats": ["#E6CFD6", "#DDBEC7", "#E8C5C8"],
    "obstacles": ["#F0E6DB", "#EADFD5", "#E2CFC3", "#EFD8CC"],
    "landing_zones": ["#E7DDEF", "#DCD4E9", "#D7D3E9"],
    "fused": ["#D8D4DB", "#DCE5E1", "#DFE5EC"]
}

def create_colorscale(colors):
    """Converts a list of hex colors to a Plotly colorscale format."""
    return [[i / (len(colors) - 1), c] for i, c in enumerate(colors)]

def add_reset_view_button(fig, camera=None):
    """Adds a reset view button to a 3D Plotly figure."""
    if camera is None:
        camera = dict(eye=dict(x=1.5, y=1.5, z=1.1))
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                buttons=[
                    dict(
                        label="Reset View",
                        method="relayout",
                        args=[{"scene.camera": camera}]
                    )
                ],
                x=0.0, y=1.12, xanchor="left", yanchor="top"
            )
        ]
    )
    return fig


# --- Main Page ---
st.title("Perception Layer")
st.markdown("""
<div style='font-family:Roboto,sans-serif;font-size:1.1rem;color:#222;'>
<b>Perception Layer:</b> Real-time analysis of terrain, wind, threats, and obstacles using backend models. All data and visualizations are generated live from the backend.
</div>
<br>
<div style='font-family:Roboto,sans-serif;font-size:0.95rem;color:#444;'>
<b>Minimalist, professional, and interactive:</b> Visualizations use advanced Plotly features, overlays, and modern color palettes. Use toggles to show/hide overlays and explore the data interactively.
</div>
<hr>
""", unsafe_allow_html=True)

# --- Summary Cards ---
summary_placeholder = st.empty()

# --- Configuration and Backend ---
with st.expander("Model Configuration", expanded=True):
    manager = render_perception_controls()

if not manager:
    st.error("Perception backend not available. Check backend installation.")
    st.stop()

# Render Summary
summary = manager.get_fusion_summary()
with summary_placeholder.container():
    st.subheader("Perception Summary")
    cols = st.columns(4)
    cols[0].metric("Mean Risk", f"{summary.get('risk_map_mean', 0):.2f}")
    cols[1].metric("Max Risk", f"{summary.get('risk_map_max', 0):.2f}")
    cols[2].metric("Mean Feasibility", f"{summary.get('feasibility_mean', 0)*100:.1f}%")
    cols[3].metric("Mean Energy Cost", f"{summary.get('energy_cost_mean', 0):.2f}")

data = export_perception_data()

# --- Tabs ---
tabs = st.tabs([
    "Terrain", "Wind", "Threats", "Obstacles", "Landing Zones", "Fused Intelligence"
])

# --- Terrain Tab ---
with tabs[0]:
    st.markdown("<b>Terrain Model:</b> Elevation, slope, and roughness analysis.", unsafe_allow_html=True)
    st.markdown("""
    **Terrain Model Visualization**
    *Type:* 3D Surface Plot (Plotly)
    *What it shows:* The elevation of the terrain over the operational area. The surface is colored using a minimalist, pastel color palette for clarity and aesthetics. The axes represent latitude and longitude indices, and the height represents elevation in meters.
    *Purpose:* To give a clear, interactive view of the terrain, highlighting elevation changes, slopes, and roughness. Useful for understanding where high ground, valleys, or flat areas are located, which is critical for flight planning and landing zone selection.
    """)
    fig = manager.get_terrain_3d_figure()
    if fig:
        # Only apply global theme, not axis/colorbar/colorscale overrides
        fig.update_layout(
            template="simple_white",
            font=dict(family="Roboto, sans-serif", size=14, color="#222"),
            paper_bgcolor="#f8f9fa", plot_bgcolor="#f8f9fa",
            margin=dict(l=10, r=10, t=30, b=10),
            transition=dict(duration=500, easing="cubic-in-out")
        )
        fig = add_reset_view_button(fig)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True, "displaylogo": False})
    
    df = data.get('terrain')
    if df is not None and not df.empty:
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.download_button("Export Terrain CSV", df.to_csv(index=False).encode('utf-8'), "terrain.csv")

# --- Wind Tab ---
with tabs[1]:
    st.markdown("<b>Wind Model:</b> Multi-altitude wind speed, direction, and turbulence.", unsafe_allow_html=True)
    st.markdown("""
    **Wind Model Visualization**
    *Type:* 3D Layered Heatmap (Plotly)
    *What it shows:* Wind speed at multiple altitudes (typically 10m, 100m, 500m). Each layer is a 2D heatmap, colored with a soft, continuous scale, showing wind speed intensity across the grid. Interactive: you can hover to see wind speed at any point.
    *Purpose:* To visualize how wind conditions change with altitude and location. Helps in planning energy-efficient and safe flight paths, as wind can affect both trajectory and energy consumption.
    """)

    # --- Combined Terrain + Wind Vectors Visualization ---
    wind_model = getattr(manager, 'wind_model', None)
    if wind_model:
        altitudes = wind_model.altitude_bands
        altitude_idx = st.slider(
            "Select Altitude Band (m)",
            min_value=0,
            max_value=len(altitudes)-1,
            value=1,
            format="%d: %dm" % (1, int(altitudes[1])) if len(altitudes) > 1 else "%d: %dm",
            key="wind_altitude_slider"
        )
        fig = manager.get_terrain_wind_3d_figure(altitude_idx=altitude_idx)
        if fig:
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True, "displaylogo": False})

    df = data.get('wind')
    if df is not None and not df.empty:
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.download_button("Export Wind CSV", df.to_csv(index=False).encode('utf-8'), "wind.csv")

# --- Threats Tab ---
with tabs[2]:
    st.markdown("<b>Threat Model:</b> Radar and SAM locations, ranges, and threat levels.", unsafe_allow_html=True)
    st.markdown("""
    **Threat Model Visualization**
    *Type:* 3D Scatter/Line Plot (Plotly)
    *What it shows:* Threat level as a function of altitude at the center of the domain. The plot uses color and vertical position to indicate low, medium, and high threat zones. Horizontal colored bands (green/yellow/red) visually separate safe, caution, and danger altitudes.
    *Purpose:* To assess risk from threats (e.g., radar, SAMs) at different altitudes. Helps operators choose safe flight altitudes and routes.
    """)
    fig = manager.get_threat_3d_figure()
    if fig:
        fig.update_layout(
            template="simple_white",
            margin=dict(l=10, r=10, t=30, b=10),
            font=dict(family="Roboto, sans-serif", size=14, color="#222"),
            paper_bgcolor="#f8f9fa", plot_bgcolor="#f8f9fa",
            scene=dict(
                xaxis=dict(showgrid=True, gridcolor="#E5E5E5", zeroline=False, showticklabels=True, title="Longitude", backgroundcolor="#f8f9fa", tickfont=dict(size=12)),
                yaxis=dict(showgrid=True, gridcolor="#E5E5E5", zeroline=False, showticklabels=True, title="Latitude", backgroundcolor="#f8f9fa", tickfont=dict(size=12)),
                zaxis=dict(showgrid=True, gridcolor="#E5E5E5", zeroline=False, showticklabels=True, title="Altitude (m)", tickfont=dict(size=12)),
                aspectmode="data"
            ),
            transition=dict(duration=500, easing="cubic-in-out")
        )
        fig.update_traces(
            marker=dict(
                colorscale=create_colorscale(COLOR_PALETTES["threats"]),
                size=18,
                opacity=0.7,
                line=dict(width=2, color="#333"),
                colorbar=dict(title="Threat Level", thickness=14, len=0.5, tickfont=dict(size=12))
            ),
            hovertemplate="<b>Lon</b>: %{x}<br><b>Lat</b>: %{y}<br><b>Alt</b>: %{z:.2f} m"
        )
        fig = add_reset_view_button(fig)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True, "displaylogo": False})
    
    df = data.get('threat')
    if df is not None and not df.empty:
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.download_button("Export Threats CSV", df.to_csv(index=False).encode('utf-8'), "threats.csv")

# --- Obstacles Tab ---
with tabs[3]:
    st.markdown("<b>Obstacle Model:</b> Building locations, heights, and clearance analysis.", unsafe_allow_html=True)
    st.markdown("""
    **Obstacle Model Visualization**
    *Type:* 2D Scatter Plot (Plotly Express)
    *What it shows:* Locations of buildings/obstacles, with marker size representing building height and color representing clearance. Minimalist, clean design with soft color gradients.
    *Purpose:* To identify where obstacles are concentrated, their heights, and available clearance for safe navigation. Essential for collision avoidance and emergency landing planning.
    """)
    df = data.get('obstacle')
    if df is not None and not df.empty:
        fig = px.scatter_3d(
            df, x="Longitude", y="Latitude", z="Height (m)", size="Height (m)", color="Clearance (m)",
            color_continuous_scale=COLOR_PALETTES["obstacles"],
            template="simple_white",
            opacity=0.8,
            hover_data={"Longitude": True, "Latitude": True, "Height (m)": True, "Clearance (m)": True}
        )
        fig.update_layout(
            margin=dict(l=10, r=10, t=30, b=10),
            font=dict(family="Roboto, sans-serif", size=14, color="#222"),
            paper_bgcolor="#f8f9fa", plot_bgcolor="#f8f9fa",
            coloraxis_showscale=True,
            showlegend=False,
            scene=dict(
                xaxis=dict(showgrid=True, gridcolor="#E5E5E5", zeroline=False, showticklabels=True, title="Longitude", backgroundcolor="#f8f9fa", tickfont=dict(size=12)),
                yaxis=dict(showgrid=True, gridcolor="#E5E5E5", zeroline=False, showticklabels=True, title="Latitude", backgroundcolor="#f8f9fa", tickfont=dict(size=12)),
                zaxis=dict(showgrid=True, gridcolor="#E5E5E5", zeroline=False, showticklabels=True, title="Height (m)", tickfont=dict(size=12)),
                aspectmode="data"
            ),
            coloraxis_colorbar=dict(title="Clearance (m)", thickness=14, len=0.5, tickfont=dict(size=12)),
            transition=dict(duration=500, easing="cubic-in-out")
        )
        fig = add_reset_view_button(fig)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True, "displaylogo": False})
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.download_button("Export Obstacles CSV", df.to_csv(index=False).encode('utf-8'), "obstacles.csv")

# --- Landing Zones Tab ---
with tabs[4]:
    st.markdown("<b>Landing Zones:</b> Feasible landing areas identified by the perception models.", unsafe_allow_html=True)
    st.markdown("""
    **Landing Zones Visualization**
    *Type:* 2D Scatter Plot (Plotly Express)
    *What it shows:* Feasible landing zones, with marker size representing area and color representing slope. Hovering reveals zone ID, area, slope, and feasibility score.
    *Purpose:* To quickly identify and compare potential landing sites based on size, slope, and suitability. Supports mission planning and emergency procedures.
    """)
    df = data.get('landing_zones')
    if df is not None and not df.empty:
        fig = px.scatter(
            df, x="Center Lon", y="Center Lat", size="Area (m²)", color="Feasibility Score",
            color_continuous_scale=COLOR_PALETTES["landing_zones"],
            template="simple_white",
            opacity=0.85,
            hover_data={"Center Lon": True, "Center Lat": True, "Area (m²)": True, "Feasibility Score": True}
        )
        fig.update_layout(
            margin=dict(l=10, r=10, t=30, b=10),
            font=dict(family="Roboto, sans-serif", size=14, color="#222"),
            paper_bgcolor="#f8f9fa", plot_bgcolor="#f8f9fa",
            coloraxis_showscale=True,
            showlegend=False,
            coloraxis_colorbar=dict(title="Feasibility", thickness=14, len=0.5, tickfont=dict(size=12)),
            transition=dict(duration=500, easing="cubic-in-out")
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True, "displaylogo": False})
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.download_button("Export Landing Zones CSV", df.to_csv(index=False).encode('utf-8'), "landing_zones.csv")