import streamlit as st

st.set_page_config(
    page_title="Defense eVTOL Trajectory Optimization",
    layout="wide",
)

st.title("Defense eVTOL Trajectory Optimization")
st.markdown("""
Welcome to the Defense eVTOL Trajectory Optimization Suite.

### Modules
- **Perception**: Analyze terrain, wind, threats, and obstacles.
- **Planning**: Generate optimal trajectories based on perception data.

Select a module from the sidebar to begin.
""")
