import streamlit as st
import os

# Inject Optax-style CSS if available
css_path = os.path.join(os.path.dirname(__file__), '..', '.streamlit', 'optax_style.css')
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.sidebar.title("EVTOL Trajectory Optimization")

st.markdown("""
# Welcome
Use the sidebar to navigate to Home or Perception.
""")
