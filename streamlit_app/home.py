"""
Home Page - Chic Dashboard

Modern, minimalist introduction to the trajectory optimization system.
"""

import streamlit as st
from colors import COLOR_PALETTE, VISUALIZATION_COLORS

# Page configuration
st.set_page_config(
    page_title="Trajectory Optimization - Defense eVTOLs",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern glassmorphic CSS
st.markdown("""
<style>
    * {
        font-family: 'Segoe UI', 'Roboto', sans-serif;
    }
    
    .hero-container {
        background: linear-gradient(135deg, rgba(248, 248, 246, 0.8) 0%, rgba(223, 229, 236, 0.8) 100%);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(200, 211, 202, 0.3);
        border-radius: 20px;
        padding: 60px 40px;
        margin: 30px 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.08);
    }
    
    .hero-title {
        font-size: 3.5rem;
        font-weight: 700;
        color: #1a1a1a;
        margin: 0;
        letter-spacing: -1px;
    }
    
    .hero-subtitle {
        font-size: 1.3rem;
        color: #757575;
        font-weight: 300;
        margin-top: 12px;
    }
    
    .hero-description {
        font-size: 1.05rem;
        color: #5a5a5a;
        line-height: 1.7;
        margin-top: 20px;
        max-width: 800px;
    }
    
    .feature-card {
        background: rgba(248, 248, 246, 0.6);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(200, 211, 202, 0.2);
        border-radius: 16px;
        padding: 32px;
        margin: 16px 0;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.05);
        transition: all 0.3s ease;
    }
    
    .feature-card:hover {
        background: rgba(223, 229, 236, 0.6);
        border-color: rgba(201, 216, 229, 0.3);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.08);
        transform: translateY(-2px);
    }
    
    .feature-title {
        font-size: 1.3rem;
        font-weight: 600;
        color: #2d3748;
        margin: 0 0 12px 0;
    }
    
    .feature-description {
        color: #5a5a5a;
        font-size: 0.95rem;
        line-height: 1.6;
        margin: 0;
    }
    
    .stat-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 20px;
        margin: 40px 0;
    }
    
    .stat-card {
        background: rgba(248, 248, 246, 0.6);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(200, 211, 202, 0.2);
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.05);
    }
    
    .stat-number {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1a1a1a;
        margin: 0;
    }
    
    .stat-label {
        font-size: 0.9rem;
        color: #5a5a5a;
        margin-top: 8px;
    }
    
    .module-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 16px;
        margin: 30px 0;
    }
    
    .module-card {
        background: rgba(248, 248, 246, 0.6);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(200, 211, 202, 0.2);
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .module-card:hover {
        background: rgba(223, 229, 236, 0.6);
        border-color: rgba(201, 216, 229, 0.3);
        transform: translateY(-4px);
    }
    
    .module-name {
        font-size: 1.1rem;
        font-weight: 600;
        color: #2d3748;
        margin: 0;
    }
    
    .divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(200, 211, 202, 0.3), transparent);
        margin: 50px 0;
    }
    
    .footer {
        background: rgba(248, 248, 246, 0.6);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(200, 211, 202, 0.2);
        border-radius: 16px;
        padding: 40px;
        margin: 50px 0 0 0;
        text-align: center;
    }
    
    .footer-text {
        color: #5a5a5a;
        margin: 8px 0;
        font-size: 0.95rem;
    }
</style>
""", unsafe_allow_html=True)

# Hero Section
st.markdown("""
<div class="hero-container">
    <h1 class="hero-title">Trajectory Optimization</h1>
    <p class="hero-subtitle">Defense eVTOL Mission Planning & Analysis</p>
    <p class="hero-description">
        Professional system for generating optimal flight paths for defense eVTOLs using multi-objective optimization, 
        environmental perception, and real-time vehicle dynamics simulation.
    </p>
</div>
""", unsafe_allow_html=True)

# System Statistics
st.markdown("<h2 style='color: #2d3748;'>System Capabilities</h2>", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown("""
    <div class="stat-card">
        <p class="stat-number">4</p>
        <p class="stat-label">System Layers</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="stat-card">
        <p class="stat-number">4+</p>
        <p class="stat-label">Algorithms</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="stat-card">
        <p class="stat-number">20+</p>
        <p class="stat-label">Export Formats</p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="stat-card">
        <p class="stat-number">Real-time</p>
        <p class="stat-label">Simulation</p>
    </div>
    """, unsafe_allow_html=True)

# Architecture Overview
st.markdown("<h2 style='color: #2d3748;'>System Architecture</h2>", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="module-card">
        <p class="module-name">Perception</p>
        <p style="color: #757575; font-size: 0.85rem; margin: 8px 0;">
        Terrain, Wind, Threat, Obstacle Analysis
        </p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="module-card">
        <p class="module-name">Planning</p>
        <p style="color: #757575; font-size: 0.85rem; margin: 8px 0;">
        Trajectory Optimization & Routing
        </p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="module-card">
        <p class="module-name">Vehicle</p>
        <p style="color: #757575; font-size: 0.85rem; margin: 8px 0;">
        Dynamics & Battery Modeling
        </p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="module-card">
        <p class="module-name">Control</p>
        <p style="color: #757575; font-size: 0.85rem; margin: 8px 0;">
        Flight Control & Tracking
        </p>
    </div>
    """, unsafe_allow_html=True)

# Key Features
st.markdown("<h2 style='color: #2d3748;'>Key Features</h2>", unsafe_allow_html=True)

feature_cols = st.columns(3)

with feature_cols[0]:
    st.markdown("""
    <div class="feature-card">
        <p class="feature-title">Pipeline Integration</p>
        <p class="feature-description">
            Seamless workflow from mission planning through waypoint generation, perception analysis, trajectory optimization, and vehicle simulation.
        </p>
    </div>
    """, unsafe_allow_html=True)

with feature_cols[1]:
    st.markdown("""
    <div class="feature-card">
        <p class="feature-title">Multi-Objective Optimization</p>
        <p class="feature-description">
            Balance energy efficiency, safety constraints, mission time, and terrain avoidance using Pareto-optimal solutions.
        </p>
    </div>
    """, unsafe_allow_html=True)

with feature_cols[2]:
    st.markdown("""
    <div class="feature-card">
        <p class="feature-title">Advanced Visualization</p>
        <p class="feature-description">
            Interactive 3D plots, flight profiles, threat coverage analysis, and vehicle performance metrics.
        </p>
    </div>
    """, unsafe_allow_html=True)

feature_cols2 = st.columns(3)

with feature_cols2[0]:
    st.markdown("""
    <div class="feature-card">
        <p class="feature-title">Real-time Simulation</p>
        <p class="feature-description">
            Dynamic vehicle simulation with 6-DOF dynamics, battery state estimation, and motor performance modeling.
        </p>
    </div>
    """, unsafe_allow_html=True)

with feature_cols2[1]:
    st.markdown("""
    <div class="feature-card">
        <p class="feature-title">Control Analysis</p>
        <p class="feature-description">
            PID parameter optimization, trajectory tracking metrics, and flight stability analysis.
        </p>
    </div>
    """, unsafe_allow_html=True)

with feature_cols2[2]:
    st.markdown("""
    <div class="feature-card">
        <p class="feature-title">Flexible Exports</p>
        <p class="feature-description">
            Export mission waypoints, analysis results, and simulation data in multiple formats for external tools.
        </p>
    </div>
    """, unsafe_allow_html=True)

# Workflow Guide
st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
st.markdown("<h2 style='color: #2d3748;'>Getting Started</h2>", unsafe_allow_html=True)

st.markdown("""
<div class="feature-card">
    <p class="feature-title">Recommended Workflow</p>
    <p class="feature-description">
    <strong>Pipeline</strong> &rarr; Define mission parameters and generate waypoints<br>
    <strong>Perception</strong> &rarr; Analyze environmental factors<br>
    <strong>Planning</strong> &rarr; Optimize trajectories<br>
    <strong>Vehicle</strong> &rarr; Simulate vehicle dynamics<br>
    <strong>Control</strong> &rarr; Analyze flight control<br>
    <strong>Simulation</strong> &rarr; View complete mission analysis
    </p>
</div>
""", unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer">
    <p class="footer-text"><strong>Trajectory Optimization in Defense eVTOLs</strong></p>
    <p class="footer-text" style='font-size: 0.8rem;'>Professional Mission Planning & Analysis System | Build 1.0.0</p>
</div>
""", unsafe_allow_html=True)
