import numpy as np
import pandas as pd

class TrajectoryPlanner:
    def __init__(self, algorithm="Optimal Control"):
        self.algorithm = algorithm

    def plan_trajectory(self, start_pos, end_pos, obstacles=None, steps=100):
        """
        Generates a trajectory between start and end points.
        In a real system, this would run RRT*, A*, or NLP solvers.
        Here, we simulate a smooth trajectory with dynamic constraints.
        """
        t = np.linspace(0, 1, steps)
        
        # Cubic Bezier-like interpolation for smooth path
        # Adding some "noise" or deviation to simulate obstacle avoidance if obstacles exist
        deviation_x = 10 * np.sin(t * np.pi) if obstacles is not None else 0
        deviation_z = 5 * np.sin(t * np.pi) # Arc for altitude

        x = (1-t) * start_pos[0] + t * end_pos[0] + deviation_x * 0.2
        y = (1-t) * start_pos[1] + t * end_pos[1]
        z = (1-t) * start_pos[2] + t * end_pos[2] + deviation_z

        # Ensure z doesn't go below ground (simple check)
        z = np.maximum(z, 0)

        # Calculate derivatives (Velocity, Acceleration)
        dt = 1.0 # Normalized time step, scale for real physics
        vx = np.gradient(x, dt)
        vy = np.gradient(y, dt)
        vz = np.gradient(z, dt)
        
        ax = np.gradient(vx, dt)
        ay = np.gradient(vy, dt)
        az = np.gradient(vz, dt)

        speed = np.sqrt(vx**2 + vy**2 + vz**2)
        energy = 0.5 * speed**2 + 9.81 * z # Specific energy

        # Create DataFrame
        df = pd.DataFrame({
            'time': np.linspace(0, 120, steps), # Assume 2 minute flight
            'x': x, 'y': y, 'z': z,
            'vx': vx, 'vy': vy, 'vz': vz,
            'ax': ax, 'ay': ay, 'az': az,
            'speed': speed,
            'energy': energy
        })
        
        return df

    def calculate_metrics(self, trajectory_df):
        """Calculates cost and performance metrics."""
        total_dist = np.sum(np.sqrt(np.diff(trajectory_df['x'])**2 + 
                                    np.diff(trajectory_df['y'])**2 + 
                                    np.diff(trajectory_df['z'])**2))
        avg_speed = trajectory_df['speed'].mean()
        max_alt = trajectory_df['z'].max()
        total_energy = trajectory_df['energy'].sum() * 0.01 # Scaling factor
        
        return {
            "Total Distance (m)": round(total_dist, 2),
            "Avg Speed (m/s)": round(avg_speed, 2),
            "Max Altitude (m)": round(max_alt, 2),
            "Energy Cost (kJ)": round(total_energy, 2),
            "Risk Score": round(np.random.uniform(0.1, 0.4), 3) # Mock risk
        }

    def check_constraints(self, trajectory_df, constraints):
        """Checks trajectory against defined constraints."""
        violations = []
        
        # Max Altitude Check
        if trajectory_df['z'].max() > constraints.get('max_altitude', 100):
            violations.append({"Constraint": "Max Altitude", "Status": "Violated", "Value": f"{trajectory_df['z'].max():.1f} m"})
        else:
            violations.append({"Constraint": "Max Altitude", "Status": "Satisfied", "Value": "OK"})

        # Max Speed Check
        if trajectory_df['speed'].max() > constraints.get('max_speed', 30):
            violations.append({"Constraint": "Max Speed", "Status": "Violated", "Value": f"{trajectory_df['speed'].max():.1f} m/s"})
        else:
            violations.append({"Constraint": "Max Speed", "Status": "Satisfied", "Value": "OK"})

        # Obstacle Clearance (Mock)
        violations.append({"Constraint": "Obstacle Clearance", "Status": "Satisfied", "Value": "> 5m"})

        return pd.DataFrame(violations)

def render_planning_controls():
    """Renders the sidebar/expander controls for planning."""
    c1, c2, c3 = st.columns(3)
    
    with c1:
        algo = st.selectbox("Algorithm", ["Optimal Control (NLP)", "RRT*", "A* Search", "Hybrid A*"])
        start_x = st.number_input("Start X", 0, 100, 10)
        start_y = st.number_input("Start Y", 0, 100, 10)
        start_z = st.number_input("Start Z", 0, 50, 0)

    with c2:
        scenario = st.selectbox("Scenario", ["Urban Canyon", "Mountain Rescue", "Coastal Patrol"])
        end_x = st.number_input("End X", 0, 100, 90)
        end_y = st.number_input("End Y", 0, 100, 90)
        end_z = st.number_input("End Z", 0, 50, 0)

    with c3:
        max_alt = st.slider("Max Altitude Constraint (m)", 50, 200, 100)
        max_speed = st.slider("Max Speed Constraint (m/s)", 10, 50, 30)
        recompute = st.button("Recompute Trajectory", type="primary")

    return {
        "algorithm": algo,
        "start": (start_x, start_y, start_z),
        "end": (end_x, end_y, end_z),
        "constraints": {"max_altitude": max_alt, "max_speed": max_speed},
        "recompute": recompute
    }

def get_planning_manager():
    if 'planning_manager' not in st.session_state:
        st.session_state['planning_manager'] = TrajectoryPlanner()
    return st.session_state['planning_manager']