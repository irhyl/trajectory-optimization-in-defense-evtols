"""
Real-time Telemetry Streamer for Streamlit

Provides efficient real-time updates for SITL telemetry display.
Uses thread-safe data structures to avoid Streamlit session state issues.
"""

import streamlit as st
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Deque
from collections import deque
import time
import threading

from sitl_connector import SITLConnector, DemoSITLConnector, Telemetry, Waypoint, get_connector

# Thread-safe global storage for telemetry data (avoids session state in threads)
_telemetry_lock = threading.Lock()
_telemetry_buffer: deque = deque(maxlen=500)
_global_connector: Optional[SITLConnector] = None


def _buffer_telemetry(telem: Telemetry):
    """Thread-safe telemetry buffer callback."""
    with _telemetry_lock:
        _telemetry_buffer.append({
            'time': telem.timestamp,
            'lat': telem.latitude,
            'lon': telem.longitude,
            'alt': telem.altitude_rel,
            'groundspeed': telem.groundspeed,
            'vz': telem.vz,
            'roll': np.degrees(telem.roll),
            'pitch': np.degrees(telem.pitch),
            'yaw': np.degrees(telem.yaw),
            'battery': telem.battery_remaining,
            'voltage': telem.battery_voltage,
            'current': telem.battery_current,
            'armed': telem.armed,
            'mode': telem.mode
        })


def init_sitl_session_state():
    """Initialize SITL-related session state variables."""
    if 'sitl_connected' not in st.session_state:
        st.session_state.sitl_connected = False
    
    if 'mission_waypoints' not in st.session_state:
        st.session_state.mission_waypoints = []
    
    if 'sitl_mode' not in st.session_state:
        st.session_state.sitl_mode = 'demo'  # 'demo' or 'real'
    
    if 'connection_string' not in st.session_state:
        st.session_state.connection_string = 'tcp:127.0.0.1:5760'


def get_sitl_connector() -> Optional[SITLConnector]:
    """Get the global SITL connector."""
    global _global_connector
    return _global_connector


def connect_sitl(use_demo: bool = True, connection_string: str = 'tcp:127.0.0.1:5760') -> bool:
    """
    Connect to SITL simulator.
    
    Args:
        use_demo: If True, use demo mode (no real SITL required).
        connection_string: MAVLink connection string for real SITL.
    
    Returns:
        True if connection successful.
    """
    global _global_connector, _telemetry_buffer
    
    # Disconnect existing connection
    if _global_connector:
        _global_connector.disconnect()
        _global_connector = None
    
    # Clear telemetry buffer
    with _telemetry_lock:
        _telemetry_buffer.clear()
    
    # Create new connector
    connector = get_connector(use_demo=use_demo)
    
    if not use_demo:
        connector.config.connection_string = connection_string
    
    # Connect
    success = connector.connect()
    
    if success:
        _global_connector = connector
        st.session_state.sitl_connected = True
        st.session_state.sitl_mode = 'demo' if use_demo else 'real'
        
        # Add thread-safe telemetry callback
        connector.add_telemetry_callback(_buffer_telemetry)
    
    return success


def disconnect_sitl():
    """Disconnect from SITL simulator."""
    global _global_connector
    
    if _global_connector:
        _global_connector.disconnect()
        _global_connector = None
    
    with _telemetry_lock:
        _telemetry_buffer.clear()
    
    st.session_state.sitl_connected = False


def get_current_telemetry() -> Optional[Telemetry]:
    """Get current telemetry from connected SITL."""
    connector = get_sitl_connector()
    if connector and connector.is_connected:
        return connector.telemetry
    return None


def get_telemetry_dataframe(n_points: int = 100) -> pd.DataFrame:
    """
    Get recent telemetry as a DataFrame.
    
    Args:
        n_points: Number of recent points to include.
    
    Returns:
        DataFrame with telemetry data.
    """
    with _telemetry_lock:
        buffer = list(_telemetry_buffer)[-n_points:]
    
    if not buffer:
        return pd.DataFrame()
    
    df = pd.DataFrame(buffer)
    if 'time' in df.columns and len(df) > 0:
        # Convert to relative time
        df['time_rel'] = df['time'] - df['time'].iloc[0]
    return df


def create_waypoints_from_coords(coords: List[tuple], base_alt: float = 50.0) -> List[Waypoint]:
    """
    Create waypoint list from coordinate tuples.
    
    Args:
        coords: List of (lat, lon) or (lat, lon, alt) tuples.
        base_alt: Default altitude if not specified.
    
    Returns:
        List of Waypoint objects.
    """
    waypoints = []
    for i, coord in enumerate(coords):
        if len(coord) >= 3:
            lat, lon, alt = coord[0], coord[1], coord[2]
        else:
            lat, lon = coord[0], coord[1]
            alt = base_alt
        
        waypoints.append(Waypoint(
            seq=i,
            lat=lat,
            lon=lon,
            alt=alt
        ))
    
    return waypoints


def upload_mission_from_pipeline() -> bool:
    """
    Upload mission from Pipeline page data (if available).
    
    Returns:
        True if mission uploaded successfully.
    """
    global _global_connector
    
    if not _global_connector or not _global_connector.is_connected:
        return False
    
    # Check for pipeline waypoints in session state
    if 'pipeline_waypoints' in st.session_state:
        waypoints = st.session_state.pipeline_waypoints
    else:
        # Create demo mission
        waypoints = create_demo_mission()
    
    st.session_state.mission_waypoints = waypoints
    return _global_connector.upload_mission(waypoints)


def create_demo_mission() -> List[Waypoint]:
    """Create a demo mission for testing."""
    # Demo waypoints near China Lake / Edwards AFB area
    base_lat = 35.363261
    base_lon = -117.060185
    
    waypoints = [
        Waypoint(seq=0, lat=base_lat, lon=base_lon, alt=50),  # Home/Takeoff
        Waypoint(seq=1, lat=base_lat + 0.005, lon=base_lon + 0.005, alt=100),
        Waypoint(seq=2, lat=base_lat + 0.010, lon=base_lon, alt=150),
        Waypoint(seq=3, lat=base_lat + 0.005, lon=base_lon - 0.005, alt=100),
        Waypoint(seq=4, lat=base_lat, lon=base_lon, alt=50),  # Return
    ]
    
    return waypoints


# ==================== UI Components ====================

def render_connection_panel():
    """Render SITL connection control panel."""
    st.markdown("#### SITL Connection")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        mode = st.radio(
            "Connection Mode",
            ["Demo (No SITL Required)", "Real SITL (ArduPilot)"],
            key="sitl_mode_select",
            horizontal=True
        )
        use_demo = mode.startswith("Demo")
        
        if not use_demo:
            connection_str = st.text_input(
                "Connection String",
                value=st.session_state.connection_string,
                placeholder="tcp:127.0.0.1:5760",
                help="MAVLink connection string. Common options:\n"
                     "- tcp:127.0.0.1:5760 (SITL default)\n"
                     "- udp:127.0.0.1:14550\n"
                     "- /dev/ttyACM0 (serial)"
            )
            st.session_state.connection_string = connection_str
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        
        if st.session_state.sitl_connected:
            # Connected - show status and disconnect button
            connector = get_sitl_connector()
            if connector:
                telem = connector.telemetry
                st.success(f"Connected ({st.session_state.sitl_mode.upper()})")
                st.caption(f"Mode: {telem.mode}")
                st.caption(f"Armed: {'Yes' if telem.armed else 'No'}")
            
            if st.button("Disconnect", type="secondary", key="btn_disconnect"):
                disconnect_sitl()
                st.rerun()
        else:
            # Not connected - show connect button
            if st.button("Connect", type="primary", key="btn_connect"):
                conn_str = st.session_state.connection_string if not use_demo else ""
                with st.spinner("Connecting..."):
                    success = connect_sitl(use_demo=use_demo, connection_string=conn_str)
                if success:
                    st.success("Connected!")
                    st.rerun()
                else:
                    st.error("Connection failed. Check SITL is running.")


def render_vehicle_controls():
    """Render vehicle control buttons."""
    st.markdown("#### Vehicle Controls")
    
    connector = get_sitl_connector()
    if not connector or not connector.is_connected:
        st.warning("Connect to SITL first")
        return
    
    telem = connector.telemetry
    
    # Mode selection
    col1, col2 = st.columns(2)
    with col1:
        mode = st.selectbox(
            "Flight Mode",
            ["STABILIZE", "GUIDED", "AUTO", "LOITER", "RTL", "LAND"],
            index=["STABILIZE", "GUIDED", "AUTO", "LOITER", "RTL", "LAND"].index(telem.mode) 
                  if telem.mode in ["STABILIZE", "GUIDED", "AUTO", "LOITER", "RTL", "LAND"] else 0,
            key="mode_select"
        )
        if st.button("Set Mode", key="btn_set_mode"):
            if connector.set_mode(mode):
                st.success(f"Mode: {mode}")
            else:
                st.error("Mode change failed")
    
    with col2:
        takeoff_alt = st.number_input("Takeoff Altitude (m)", min_value=5, max_value=100, value=30, key="takeoff_alt")
    
    # Control buttons
    btn_col1, btn_col2, btn_col3, btn_col4 = st.columns(4)
    
    with btn_col1:
        if st.button("🔓 ARM", type="primary", key="btn_arm", disabled=telem.armed):
            if connector.set_mode("GUIDED"):
                if connector.arm():
                    st.success("Armed!")
                else:
                    st.error("Arm failed")
    
    with btn_col2:
        if st.button("🔒 DISARM", type="secondary", key="btn_disarm", disabled=not telem.armed):
            if connector.disarm(force=True):
                st.success("Disarmed")
    
    with btn_col3:
        if st.button("🚀 TAKEOFF", key="btn_takeoff", disabled=not telem.armed):
            if connector.takeoff(takeoff_alt):
                st.success(f"Taking off to {takeoff_alt}m")
    
    with btn_col4:
        if st.button("🛬 LAND", key="btn_land"):
            if connector.land():
                st.success("Landing...")


def render_telemetry_display():
    """Render live telemetry display."""
    st.markdown("#### Live Telemetry")
    
    connector = get_sitl_connector()
    if not connector or not connector.is_connected:
        st.info("Connect to SITL to see telemetry")
        return
    
    telem = connector.telemetry
    
    # Status metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        status_color = "🟢" if telem.is_connected else "🔴"
        st.metric("Status", f"{status_color} {telem.mode}")
    
    with col2:
        st.metric("Altitude", f"{telem.altitude_rel:.1f} m")
    
    with col3:
        st.metric("Ground Speed", f"{telem.groundspeed:.1f} m/s")
    
    with col4:
        st.metric("Battery", f"{telem.battery_remaining}%")
    
    with col5:
        armed_indicator = "🔴 ARMED" if telem.armed else "⚪ DISARMED"
        st.metric("Armed", armed_indicator)
    
    # Detailed telemetry
    with st.expander("Detailed Telemetry", expanded=False):
        detail_col1, detail_col2, detail_col3 = st.columns(3)
        
        with detail_col1:
            st.markdown("**Position**")
            st.text(f"Lat: {telem.latitude:.6f}°")
            st.text(f"Lon: {telem.longitude:.6f}°")
            st.text(f"Alt MSL: {telem.altitude_msl:.1f} m")
            st.text(f"Alt REL: {telem.altitude_rel:.1f} m")
        
        with detail_col2:
            st.markdown("**Velocity**")
            st.text(f"Vx: {telem.vx:.2f} m/s")
            st.text(f"Vy: {telem.vy:.2f} m/s")
            st.text(f"Vz: {telem.vz:.2f} m/s")
            st.text(f"GS: {telem.groundspeed:.1f} m/s")
        
        with detail_col3:
            st.markdown("**Attitude**")
            st.text(f"Roll: {np.degrees(telem.roll):.1f}°")
            st.text(f"Pitch: {np.degrees(telem.pitch):.1f}°")
            st.text(f"Yaw: {np.degrees(telem.yaw):.1f}°")
        
        gps_col1, gps_col2 = st.columns(2)
        with gps_col1:
            st.markdown("**GPS**")
            fix_types = {0: "No Fix", 1: "No Fix", 2: "2D", 3: "3D", 4: "DGPS", 5: "RTK Float", 6: "RTK Fixed"}
            st.text(f"Fix: {fix_types.get(telem.gps_fix, 'Unknown')}")
            st.text(f"Satellites: {telem.satellites}")
            st.text(f"HDOP: {telem.hdop:.2f}")
        
        with gps_col2:
            st.markdown("**Battery**")
            st.text(f"Voltage: {telem.battery_voltage:.2f} V")
            st.text(f"Current: {telem.battery_current:.1f} A")
            st.text(f"Remaining: {telem.battery_remaining}%")


def render_mission_panel():
    """Render mission upload and control panel."""
    st.markdown("#### Mission Control")
    
    connector = get_sitl_connector()
    if not connector or not connector.is_connected:
        st.warning("Connect to SITL first")
        return
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📤 Upload Demo Mission", key="btn_upload_mission"):
            waypoints = create_demo_mission()
            if connector.upload_mission(waypoints):
                st.session_state.mission_waypoints = waypoints
                st.success(f"Uploaded {len(waypoints)} waypoints")
            else:
                st.error("Upload failed")
    
    with col2:
        if st.button("▶️ Start Mission", key="btn_start_mission", disabled=len(st.session_state.mission_waypoints) == 0):
            if connector.start_mission():
                st.success("Mission started!")
            else:
                st.error("Failed to start mission")
    
    with col3:
        if st.button("🏠 Return to Launch", key="btn_rtl"):
            if connector.return_to_launch():
                st.success("RTL activated")
    
    # Display current waypoints
    if st.session_state.mission_waypoints:
        with st.expander(f"Mission Waypoints ({len(st.session_state.mission_waypoints)})", expanded=False):
            wp_data = []
            for wp in st.session_state.mission_waypoints:
                wp_data.append({
                    'Seq': wp.seq,
                    'Latitude': f"{wp.lat:.6f}",
                    'Longitude': f"{wp.lon:.6f}",
                    'Altitude (m)': wp.alt
                })
            st.dataframe(pd.DataFrame(wp_data), use_container_width=True)
