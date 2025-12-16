import streamlit as st
import pandas as pd
import plotly.express as px
import sys
import pathlib

# Ensure the streamlit_app package root is importable when Streamlit runs pages
streamlit_app_root = pathlib.Path(__file__).resolve().parents[1]
if str(streamlit_app_root) not in sys.path:
    sys.path.insert(0, str(streamlit_app_root))

from planning_backend import render_planning_controls, get_planning_manager
import traceback

# --- Page Configuration ---
st.set_page_config(
    page_title="Planning Layer - Defense eVTOLs",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.title("Planning Layer")
st.markdown("""
<b>Planning Layer:</b> Trajectory optimization and mission planning using outputs from the Perception layer (terrain, wind, threats, obstacles, etc).<br>
All results are generated from backend models and CSV outputs.
<hr>
""", unsafe_allow_html=True)



# --- Model Configuration and Backend Controls ---
try:
    manager = render_planning_controls()
    if not manager:
        st.error("Planning backend not available. Check backend installation.")
        st.stop()
except Exception as e:
    st.error(f"Error initializing planning backend: {e}")
    st.exception(traceback.format_exc())
    st.stop()


# --- Summary Cards ---
st.subheader("Planning Summary")
cols = st.columns(4)
summary = getattr(manager, 'summary', {}) if manager else {}
cols[0].metric("Feasible Solutions", summary.get('feasible_solutions', '-'))
cols[1].metric("Pareto-optimal Solutions", summary.get('pareto_solutions', '-'))
cols[2].metric("Best Cost", summary.get('best_cost', '-'))
cols[3].metric("Best Risk", summary.get('best_risk', '-'))



# --- Get Planning Output DataFrames from Backend ---
try:
    moo_df = getattr(manager, 'moo_results', pd.DataFrame())
    pareto_df = getattr(manager, 'pareto_frontier', pd.DataFrame())
    vehicle_feasible_df = getattr(manager, 'vehicle_feasible_solutions', pd.DataFrame())
    vehicle_pareto_df = getattr(manager, 'vehicle_feasible_pareto_solutions', pd.DataFrame())
except Exception as e:
    st.error(f"Error retrieving planning results: {e}")
    st.exception(traceback.format_exc())
    moo_df = pareto_df = vehicle_feasible_df = vehicle_pareto_df = pd.DataFrame()

# --- Output Tabs ---
tabs = st.tabs([
    "MOO Results", "Pareto Frontier", "Vehicle Feasible Solutions", "Vehicle Feasible Pareto Solutions"
])

# --- MOO Results Tab ---
with tabs[0]:
    st.markdown("**Multi-Objective Optimization (MOO) Results**")
    if not moo_df.empty:
        st.dataframe(moo_df, use_container_width=True, hide_index=True)
        st.download_button("Export MOO Results CSV", moo_df.to_csv(index=False).encode('utf-8'), "moo_results_16_solutions.csv")
    else:
        st.warning("No MOO results available.")

# --- Pareto Frontier Tab ---
with tabs[1]:
    st.markdown("**Pareto Frontier Solutions**")
    if not pareto_df.empty:
        st.dataframe(pareto_df, use_container_width=True, hide_index=True)
        st.download_button("Export Pareto Frontier CSV", pareto_df.to_csv(index=False).encode('utf-8'), "pareto_frontier_solutions.csv")
        # Pareto front visualization (if columns exist)
        cols = pareto_df.columns
        if len(cols) >= 3:
            x, y, z = cols[:3]
            fig = px.scatter_3d(pareto_df, x=x, y=y, z=z, color=z, title="Pareto Frontier (First 3 Columns)")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No Pareto frontier solutions available.")

# --- Vehicle Feasible Solutions Tab ---
with tabs[2]:
    st.markdown("**Vehicle Feasible Solutions**")
    if not vehicle_feasible_df.empty:
        st.dataframe(vehicle_feasible_df, use_container_width=True, hide_index=True)
        st.download_button("Export Vehicle Feasible Solutions CSV", vehicle_feasible_df.to_csv(index=False).encode('utf-8'), "vehicle_feasible_solutions_moo.csv")
    else:
        st.warning("No vehicle feasible solutions available.")

# --- Vehicle Feasible Pareto Solutions Tab ---
with tabs[3]:
    st.markdown("**Vehicle Feasible Pareto Solutions**")
    if not vehicle_pareto_df.empty:
        st.dataframe(vehicle_pareto_df, use_container_width=True, hide_index=True)
        st.download_button("Export Vehicle Feasible Pareto Solutions CSV", vehicle_pareto_df.to_csv(index=False).encode('utf-8'), "vehicle_feasible_pareto_solutions.csv")
    else:
        st.warning("No vehicle feasible pareto solutions available.")
