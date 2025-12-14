"""
Streamlit Styling and Theming Module

Provides centralized styling configuration for the entire application
with professional, minimalist design using Roboto font family.
"""

import streamlit as st
from pathlib import Path


def apply_global_styles():
    """Apply global CSS styling to the entire app."""
    css = """
    <style>
    /* Import Roboto Font */
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');
    
    /* Global Font Configuration */
    * {
        font-family: 'Roboto', sans-serif;
    }
    
    html, body, [class*="css"] {
        font-family: 'Roboto', sans-serif;
    }
    
    /* Main Container */
    [data-testid="stMainBlockContainer"] {
        background-color: #ffffff;
        padding: 2rem;
    }
    
    [data-testid="stAppViewContainer"] {
        background-color: #fafafa;
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Roboto', sans-serif;
        color: #1a1a1a;
        font-weight: 500;
        letter-spacing: -0.5px;
    }
    
    h1 {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    h2 {
        font-size: 1.75rem;
        font-weight: 600;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        border-bottom: 1px solid #e0e0e0;
        padding-bottom: 0.5rem;
    }
    
    h3 {
        font-size: 1.3rem;
        font-weight: 500;
    }
    
    /* Body Text */
    p, span, label, div {
        font-family: 'Roboto', sans-serif;
        color: #424242;
        line-height: 1.6;
    }
    
    /* Captions */
    [data-testid="stCaption"] {
        color: #757575;
        font-size: 0.875rem;
        font-weight: 300;
    }
    
    /* Buttons */
    button {
        font-family: 'Roboto', sans-serif;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        border-radius: 4px;
        border: none;
        padding: 0.75rem 1.5rem;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    button[kind="primary"] {
        background-color: #1565c0;
        color: white;
    }
    
    button[kind="primary"]:hover {
        background-color: #1e40af;
        box-shadow: 0 2px 8px rgba(21, 101, 192, 0.3);
    }
    
    button[kind="secondary"] {
        background-color: #f5f5f5;
        color: #1a1a1a;
        border: 1px solid #e0e0e0;
    }
    
    button[kind="secondary"]:hover {
        background-color: #eeeeee;
    }
    
    /* Input Fields */
    input, textarea, select {
        font-family: 'Roboto', sans-serif;
        border-radius: 4px;
        border: 1px solid #d0d0d0;
        padding: 0.75rem;
        font-size: 0.95rem;
    }
    
    input:focus, textarea:focus, select:focus {
        border-color: #1565c0;
        box-shadow: 0 0 0 3px rgba(21, 101, 192, 0.1);
        outline: none;
    }
    
    /* Tabs */
    [data-testid="stTabs"] [role="tablist"] {
        background-color: transparent;
        border-bottom: 2px solid #e0e0e0;
    }
    
    [data-testid="stTabs"] [role="tab"] {
        font-family: 'Roboto', sans-serif;
        font-weight: 500;
        color: #757575;
        border-radius: 0;
        border-bottom: 3px solid transparent;
        padding: 1rem 1.5rem;
    }
    
    [data-testid="stTabs"] [role="tab"][aria-selected="true"] {
        color: #1565c0;
        border-bottom-color: #1565c0;
    }
    
    /* Metric Cards */
    [data-testid="metric-container"] {
        background-color: #ffffff;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
        padding: 1.5rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    }
    
    [data-testid="metric-container"]:hover {
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        transition: box-shadow 0.3s ease;
    }
    
    /* Expanders */
    [data-testid="stExpander"] {
        border: 1px solid #e0e0e0;
        border-radius: 4px;
    }
    
    [data-testid="stExpander"] > summary {
        font-family: 'Roboto', sans-serif;
        font-weight: 500;
        color: #1a1a1a;
        padding: 1rem;
    }
    
    /* Dividers */
    hr {
        border: none;
        border-top: 1px solid #e0e0e0;
        margin: 2rem 0;
    }
    
    /* Code Blocks */
    pre, code {
        font-family: 'Roboto Mono', monospace;
        background-color: #f5f5f5;
        border-radius: 4px;
        padding: 0.5rem;
    }
    
    /* Alerts and Messages */
    [data-testid="stAlert"] {
        border-radius: 4px;
        font-family: 'Roboto', sans-serif;
    }
    
    .info-box {
        background-color: #e3f2fd;
        border-left: 4px solid #1565c0;
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
    }
    
    .success-box {
        background-color: #e8f5e9;
        border-left: 4px solid #2e7d32;
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
    }
    
    .warning-box {
        background-color: #fff3e0;
        border-left: 4px solid #f57c00;
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #fafafa;
        border-right: 1px solid #e0e0e0;
    }
    
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #1565c0;
    }
    
    /* DataFrames */
    [data-testid="stDataFrame"] {
        border-radius: 4px;
        border: 1px solid #e0e0e0;
    }
    
    /* Plots and Charts */
    [data-testid="plotly-graph"] {
        border-radius: 8px;
        border: 1px solid #e0e0e0;
        padding: 1rem;
        background-color: #ffffff;
    }
    
    /* Download Button */
    [data-testid="stDownloadButton"] button {
        background-color: #2e7d32;
        color: white;
    }
    
    [data-testid="stDownloadButton"] button:hover {
        background-color: #1b5e20;
    }
    
    /* Scrollbar Styling */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #bdbdbd;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #757575;
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def get_color_scheme():
    """Return the color scheme dictionary."""
    return {
        "primary": "#1565c0",
        "primary_light": "#42a5f5",
        "primary_dark": "#0d47a1",
        "secondary": "#2e7d32",
        "secondary_light": "#66bb6a",
        "secondary_dark": "#1b5e20",
        "neutral": "#757575",
        "neutral_light": "#bdbdbd",
        "neutral_dark": "#424242",
        "background": "#fafafa",
        "surface": "#ffffff",
        "border": "#e0e0e0",
        "error": "#c62828",
        "warning": "#f57c00",
        "success": "#2e7d32",
        "info": "#1565c0",
    }


def configure_page(page_title: str, layout: str = "wide"):
    """Configure Streamlit page settings."""
    st.set_page_config(
        page_title=page_title,
        layout=layout,
        initial_sidebar_state="expanded",
        menu_items={
            "About": "Trajectory Optimization in Defense eVTOLs - Professional System"
        }
    )
    apply_global_styles()


def display_header(title: str, description: str = None, subtitle: str = None):
    """Display a professional page header."""
    st.markdown(f"# {title}")
    
    if subtitle:
        st.markdown(f"### {subtitle}", unsafe_allow_html=True)
    
    if description:
        st.markdown(f"<p style='color: #757575; font-size: 1rem; margin-top: -10px;'>{description}</p>", 
                   unsafe_allow_html=True)
    
    st.divider()


def display_metric_row(metrics: dict):
    """Display metrics in a clean row layout."""
    cols = st.columns(len(metrics))
    
    for col, (label, value) in zip(cols, metrics.items()):
        with col:
            st.metric(label=label, value=value)


def display_info_box(content: str, box_type: str = "info"):
    """Display an information box."""
    style_map = {
        "info": "info-box",
        "success": "success-box",
        "warning": "warning-box",
    }
    style_class = style_map.get(box_type, "info-box")
    st.markdown(f"<div class='{style_class}'>{content}</div>", unsafe_allow_html=True)


def get_download_button_html(filename: str, file_content: str):
    """Generate download button HTML."""
    return f"""
    <div style='margin: 1rem 0;'>
        <button style='
            background-color: #2e7d32;
            color: white;
            padding: 0.75rem 1.5rem;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        '>Download {filename}</button>
    </div>
    """
