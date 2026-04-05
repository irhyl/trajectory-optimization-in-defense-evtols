"""
6-DoF Rigid Body Dynamics for Tiltrotor

This module implements the complete 6-DoF equations of motion for a rigid body
aircraft using quaternion attitude representation.

Equations of Motion
===================

1. Position Kinematics (NED frame):
   ṗ = R(q)ᵀ · V_B

   where R(q) is the DCM from NED to Body frame.

2. Translational Dynamics (Body frame):
   m(V̇_B + ω_B × V_B) = F_B

   Expanded:
   m·u̇ = Fx - m(q·w - r·v) + m·g·sin(θ)
   m·v̇ = Fy - m(r·u - p·w) - m·g·cos(θ)·sin(φ)
   m·ẇ = Fz - m(p·v - q·u) - m·g·cos(θ)·cos(φ)

3. Attitude Kinematics (Quaternion):
   q̇ = ½ · q ⊗ ω

   In matrix form:
   [q̇₀]   1   [0  -p  -q  -r] [q₀]
   [q̇₁] = - · [p   0   r  -q] [q₁]
   [q̇₂]   2   [q  -r   0   p] [q₂]
   [q̇₃]       [r   q  -p   0] [q₃]

4. Rotational Dynamics (Body frame):
   I·ω̇_B + ω_B × (I·ω_B) = M_B

   Expanded (for diagonal I):
   Ixx·ṗ = Mx + (Iyy - Izz)·q·r
   Iyy·q̇ = My + (Izz - Ixx)·r·p
   Izz·ṙ = Mz + (Ixx - Iyy)·p·q

Force Components
================

The total force in body frame is:
   F_B = F_gravity + F_propulsion + F_aerodynamic

- Gravity (transformed to body frame):
  F_g = R(q) · [0, 0, m·g]ᵀ

- Propulsion (from rotors):
  F_prop = Σᵢ T_i(Ω_i, θ_i, V) · n̂_i

- Aerodynamics (wing, fuselage):
  F_aero = ½ρV² · S · [CD, CY, CL]ᵀ (transformed to body)

Moment Components
=================

The total moment about CG is:
   M_B = M_propulsion + M_aerodynamic + M_gyroscopic

- Propulsion moments:
  M_prop = Σᵢ (r_i × F_i) + Q_i·ẑ_rotor

- Aerodynamic moments:
  M_aero = ½ρV² · S · c̄ · [Cl, Cm, Cn]ᵀ

- Gyroscopic (rotor angular momentum):
  M_gyro = Σᵢ ω_B × (I_rotor · Ω_i · ẑ_rotor)
"""

from __future__ import annotations
import numpy as np
from typing import Any
from dataclasses import dataclass, field
import logging

from .state import VehicleState, ControlInput, Quaternion
from ..config import TiltrotorConfig

logger = logging.getLogger(__name__)


# Physical constants
G = 9.80665  # Standard gravity [m/s²]


@dataclass
class ForcesMoments:
    """
    Container for forces and moments acting on the vehicle.

    All quantities in body frame unless otherwise noted.
    """
    # Forces [N] in body frame [Fx, Fy, Fz]
    forces: np.ndarray = field(default_factory=lambda: np.zeros(3))

    # Moments [N·m] in body frame [Mx, My, Mz]
    moments: np.ndarray = field(default_factory=lambda: np.zeros(3))

    # Component breakdown for analysis
    gravity_force: np.ndarray = field(default_factory=lambda: np.zeros(3))
    propulsion_force: np.ndarray = field(default_factory=lambda: np.zeros(3))
    propulsion_moment: np.ndarray = field(default_factory=lambda: np.zeros(3))
    aero_force: np.ndarray = field(default_factory=lambda: np.zeros(3))
    aero_moment: np.ndarray = field(default_factory=lambda: np.zeros(3))
    gyroscopic_moment: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def total(self) -> tuple[np.ndarray, np.ndarray]:
        """Return total forces and moments."""
        return self.forces, self.moments


class RigidBodyDynamics:
    """
    6-DoF rigid body dynamics with quaternion attitude.

    This class computes state derivatives given current state and control inputs.
    It integrates forces and moments from propulsion, aerodynamics, and gravity.

    Attributes:
        config: Vehicle configuration
        propulsion_model: Propulsion system model (rotors, motors)
        aero_model: Aerodynamics model (wing, fuselage)
    """

    def __init__(
        self,
        config: TiltrotorConfig,
        propulsion_model: Any | None = None,
        aero_model: Any | None = None,
    ):
        """
        Initialize rigid body dynamics.

        Args:
            config: Vehicle configuration
            propulsion_model: Propulsion model (optional, will use simplified if None)
            aero_model: Aerodynamics model (optional, will use simplified if None)
        """
        self.config = config
        self.propulsion = propulsion_model
        self.aero = aero_model

        # Cache mass properties
        self.mass = config.mass.total_mass
        self.inertia = config.mass.inertia_matrix
        self.inertia_inv = np.linalg.inv(self.inertia)
        self.cg = config.mass.cg_position

        logger.info(f"RigidBodyDynamics initialized: mass={self.mass:.1f}kg")

    def compute_derivatives(
        self,
        state: VehicleState,
        control: ControlInput,
        wind: np.ndarray = None,
    ) -> np.ndarray:
        """
        Compute state derivatives (ẋ = f(x, u)).

        This is the core dynamics function that computes:
        1. Position derivative from velocity kinematics
        2. Velocity derivative from translational dynamics
        3. Quaternion derivative from attitude kinematics
        4. Angular rate derivative from rotational dynamics
        5. Actuator dynamics (nacelle, rotor)
        6. Energy dynamics (battery)

        Args:
            state: Current vehicle state
            control: Control inputs
            wind: Wind velocity in NED frame [m/s] (optional)

        Returns:
            State derivative vector [23]
        """
        if wind is None:
            wind = np.zeros(3)

        # Get rotation matrix (NED → Body)
        R = state.attitude.to_dcm()

        # Wind in body frame
        wind_body = R @ wind

        # Compute aerodynamic velocity (body frame)
        # V_air = V_body - V_wind_body
        V_air = state.velocity - wind_body

        # === Compute Forces and Moments ===
        fm = self._compute_forces_moments(state, control, V_air, R)

        # === 1. Position Kinematics ===
        # ṗ = Rᵀ · V_B (velocity in NED frame)
        p_dot = R.T @ state.velocity

        # === 2. Translational Dynamics ===
        # m(V̇ + ω × V) = F
        # V̇ = F/m - ω × V
        omega = state.angular_velocity
        V_dot = fm.forces / self.mass - np.cross(omega, state.velocity)

        # === 3. Attitude Kinematics ===
        # q̇ = ½ · q ⊗ ω
        q = state.attitude
        q_dot = self._quaternion_derivative(q, omega)

        # === 4. Rotational Dynamics ===
        # I·ω̇ = M - ω × (I·ω)
        I_omega = self.inertia @ omega
        omega_dot = self.inertia_inv @ (fm.moments - np.cross(omega, I_omega))

        # === 5. Nacelle Dynamics ===
        # First-order with rate limiting
        nacelle_dot = self._nacelle_dynamics(state, control)
        nacelle_rate_dot = np.zeros(2)  # Second derivative (simplified)

        # === 6. Rotor Dynamics ===
        # First-order lag on rotor speed
        rotor_dot = self._rotor_dynamics(state, control)
        collective_dot = self._collective_dynamics(state, control)

        # === 7. Battery Dynamics ===
        soc_dot, temp_dot = self._battery_dynamics(state, control, fm)

        # === Assemble State Derivative ===
        x_dot = np.concatenate([
            p_dot,                    # 3: position
            V_dot,                    # 3: velocity
            q_dot,                    # 4: quaternion
            omega_dot,                # 3: angular velocity
            nacelle_dot,              # 2: nacelle angles
            nacelle_rate_dot,         # 2: nacelle rates
            rotor_dot,                # 2: rotor speeds
            collective_dot,           # 2: collective pitch
            [soc_dot],                # 1: battery SOC
            [temp_dot],               # 1: battery temperature
        ])

        return x_dot

    def _compute_forces_moments(
        self,
        state: VehicleState,
        control: ControlInput,
        V_air: np.ndarray,
        R: np.ndarray,
    ) -> ForcesMoments:
        """
        Compute all forces and moments in body frame.

        Args:
            state: Current state
            control: Control inputs
            V_air: Aerodynamic velocity (body frame)
            R: Rotation matrix (NED → Body)

        Returns:
            ForcesMoments container
        """
        fm = ForcesMoments()

        # === 1. Gravity ===
        # Transform gravity to body frame: F_g = R · [0, 0, m·g]
        gravity_ned = np.array([0.0, 0.0, self.mass * G])
        fm.gravity_force = R @ gravity_ned

        # === 2. Propulsion ===
        if self.propulsion is not None:
            prop_forces, prop_moments = self.propulsion.compute_forces_moments(
                state, control
            )
            fm.propulsion_force = prop_forces
            fm.propulsion_moment = prop_moments
        else:
            # Simplified propulsion model
            fm.propulsion_force, fm.propulsion_moment = self._simplified_propulsion(
                state, control
            )

        # === 3. Aerodynamics ===
        if self.aero is not None:
            aero_forces, aero_moments = self.aero.compute_forces_moments(
                state, control, V_air
            )
            fm.aero_force = aero_forces
            fm.aero_moment = aero_moments
        else:
            # Simplified aerodynamics
            fm.aero_force, fm.aero_moment = self._simplified_aero(state, V_air)

        # === 4. Gyroscopic Moments ===
        fm.gyroscopic_moment = self._compute_gyroscopic_moments(state)

        # === Total ===
        fm.forces = (fm.gravity_force + fm.propulsion_force + fm.aero_force)
        fm.moments = (fm.propulsion_moment + fm.aero_moment + fm.gyroscopic_moment)

        return fm

    def _simplified_propulsion(
        self,
        state: VehicleState,
        control: ControlInput,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Simplified propulsion model for testing.

        Models two tilting rotors with basic thrust/torque relations.
        """
        forces = np.zeros(3)
        moments = np.zeros(3)

        # Rotor positions in body frame
        r_left = self.config.propulsion.left_rotor.position
        r_right = self.config.propulsion.right_rotor.position

        for i, (rotor_cfg, pos) in enumerate([
            (self.config.propulsion.left_rotor, r_left),
            (self.config.propulsion.right_rotor, r_right),
        ]):
            # Get rotor state
            Omega = state.rotor_speeds[i]
            theta_col = state.collective_pitch[i]
            nacelle_angle = state.nacelle_angles[i]

            # Simplified thrust: T = k_T · Ω² · θ_col
            # For realistic eVTOL: ~100N per rad/s² at 10° collective
            k_T = 0.5  # Thrust coefficient
            T = k_T * Omega**2 * (theta_col + 0.1)  # Add bias for hover
            T = max(0, min(T, rotor_cfg.max_thrust if hasattr(rotor_cfg, 'max_thrust') else 5000))

            # Thrust direction in nacelle frame (along rotor axis)
            # In nacelle frame, thrust is along -z (up)
            T_nacelle = np.array([0, 0, -T])

            # Rotate by nacelle angle (rotation about y-axis)
            # nacelle_angle: 0° = horizontal (cruise), 90° = vertical (hover)
            c_nac = np.cos(nacelle_angle)
            s_nac = np.sin(nacelle_angle)
            R_nacelle = np.array([
                [c_nac, 0, s_nac],
                [0, 1, 0],
                [-s_nac, 0, c_nac]
            ])

            # Thrust in body frame
            T_body = R_nacelle @ T_nacelle

            # Torque reaction (simplified)
            # Q = k_Q · Ω² (opposes rotation direction)
            k_Q = 0.01  # Torque coefficient
            Q = k_Q * Omega**2

            # Direction depends on rotation direction
            rot_dir = rotor_cfg.rotation_direction
            Q_nacelle = np.array([0, 0, rot_dir * Q])
            Q_body = R_nacelle @ Q_nacelle

            # Add to totals
            forces += T_body

            # Moment from thrust about CG
            r = pos - self.cg
            moments += np.cross(r, T_body)

            # Add torque reaction
            moments += Q_body

        return forces, moments

    def _simplified_aero(
        self,
        state: VehicleState,
        V_air: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Simplified aerodynamic model.

        Includes:
        - Fuselage parasitic drag
        - Wing lift and drag (in cruise)
        - Basic stability derivatives
        """
        forces = np.zeros(3)
        moments = np.zeros(3)

        # Airspeed magnitude
        V = np.linalg.norm(V_air)
        if V < 1.0:
            return forces, moments

        # Dynamic pressure
        rho = self.config.aerodynamics.air_density_sl
        q_bar = 0.5 * rho * V**2

        # Angle of attack and sideslip
        u, v, w = V_air
        alpha = np.arctan2(w, u) if abs(u) > 0.1 else 0.0
        beta = np.arcsin(np.clip(v / V, -1.0, 1.0)) if V > 0.1 else 0.0

        # === Fuselage Drag ===
        S_fuse = self.config.aerodynamics.fuselage.frontal_area
        CD_fuse = self.config.aerodynamics.fuselage.cd_frontal
        D_fuse = q_bar * S_fuse * CD_fuse

        # Drag opposes velocity
        forces[0] -= D_fuse * u / V  # X component
        forces[1] -= D_fuse * v / V  # Y component
        forces[2] -= D_fuse * w / V  # Z component

        # === Wing Aerodynamics (significant in cruise) ===
        wing = self.config.aerodynamics.wing
        S_wing = wing.area

        # Only significant lift when nacelles are tilted forward
        nacelle_factor = np.cos(np.mean(state.nacelle_angles))  # 1 in cruise, 0 in hover

        if nacelle_factor > 0.3:  # Transition or cruise
            # Lift coefficient (linear + stall)
            CL0 = wing.cl0
            CL_alpha = wing.cl_alpha
            CL = CL0 + CL_alpha * alpha
            CL = np.clip(CL, -wing.cl_max, wing.cl_max)

            # Drag coefficient (parabolic polar)
            CD0 = wing.cd0
            k = wing.cd_k
            CD = CD0 + k * CL**2

            # Lift and drag in stability axes
            L = q_bar * S_wing * CL * nacelle_factor
            D = q_bar * S_wing * CD * nacelle_factor

            # Transform to body axes
            # Lift perpendicular to velocity, Drag parallel
            forces[0] -= D * np.cos(alpha) - L * np.sin(alpha)
            forces[2] -= D * np.sin(alpha) + L * np.cos(alpha)

            # Pitching moment
            CM0 = wing.cm0
            Cm = CM0 - 0.1 * CL  # Stability derivative
            M_pitch = q_bar * S_wing * wing.mean_chord * Cm * nacelle_factor
            moments[1] += M_pitch

        # === Sideslip Side-Force and Directional Stability ===
        # Contributions from fuselage and vertical tail.
        # CY_beta : side-force-due-to-sideslip  [1/rad]  (negative = restoring)
        # CN_beta : yaw-moment-due-to-sideslip  [1/rad]  (positive = stable)
        # Reference: Etkin & Reid (1996), Dynamics of Flight, §3.5.
        S_vtail = self.config.aerodynamics.fuselage.vertical_tail_area
        l_vtail = self.config.aerodynamics.fuselage.tail_arm
        CY_beta = -0.35   # Typical tiltrotor value [1/rad]
        CN_beta = 0.06    # Directional stability [1/rad]
        if nacelle_factor > 0.1 and V > 5.0:
            CY = CY_beta * beta
            CN = CN_beta * beta
            forces[1] += q_bar * S_vtail * CY                  # Side force
            moments[2] += q_bar * S_vtail * l_vtail * CN       # Yaw moment

        # === Stability Damping ===
        # Roll damping
        p, q, r = state.angular_velocity
        b = wing.span
        c = wing.mean_chord

        # Dimensional damping (simplified)
        Clp = -0.4  # Roll damping derivative
        Cmq = -10.0  # Pitch damping derivative
        Cnr = -0.1  # Yaw damping derivative

        if V > 5.0:
            moments[0] += q_bar * S_wing * b * Clp * (p * b / (2 * V))
            moments[1] += q_bar * S_wing * c * Cmq * (q * c / (2 * V))
            moments[2] += q_bar * S_wing * b * Cnr * (r * b / (2 * V))

        return forces, moments

    def _compute_gyroscopic_moments(self, state: VehicleState) -> np.ndarray:
        """
        Compute gyroscopic moments from spinning rotors.

        M_gyro = ω_body × (I_rotor · Ω_rotor · ẑ_rotor)

        This is significant in maneuvering flight and causes
        cross-coupling between pitch and roll.
        """
        moments = np.zeros(3)

        # Rotor polar moment of inertia (estimate)
        I_rotor = 2.0  # kg·m² (for each rotor)

        omega_body = state.angular_velocity

        for i in range(2):  # Left and right rotors
            Omega = state.rotor_speeds[i]
            nacelle_angle = state.nacelle_angles[i]
            rot_dir = 1.0 if i == 0 else -1.0  # Counter-rotating

            # Rotor angular momentum in nacelle frame
            H_nacelle = np.array([0, 0, I_rotor * Omega * rot_dir])

            # Rotate to body frame
            c_nac = np.cos(nacelle_angle)
            s_nac = np.sin(nacelle_angle)
            R_nacelle = np.array([
                [c_nac, 0, s_nac],
                [0, 1, 0],
                [-s_nac, 0, c_nac]
            ])
            H_body = R_nacelle @ H_nacelle

            # Gyroscopic moment
            moments += np.cross(omega_body, H_body)

        return moments

    def _quaternion_derivative(
        self,
        q: Quaternion,
        omega: np.ndarray,
    ) -> np.ndarray:
        """
        Compute quaternion derivative from angular velocity.

        q̇ = ½ · Ω(ω) · q

        where Ω(ω) is the skew-symmetric matrix form.
        """
        p, qb, r = omega  # Note: 'qb' to avoid confusion with quaternion

        # Quaternion multiplication matrix
        Omega = 0.5 * np.array([
            [0, -p, -qb, -r],
            [p,  0,  r, -qb],
            [qb, -r,  0,  p],
            [r,  qb, -p,  0]
        ])

        q_vec = q.array
        q_dot = Omega @ q_vec

        return q_dot

    def _nacelle_dynamics(
        self,
        state: VehicleState,
        control: ControlInput,
    ) -> np.ndarray:
        """
        Nacelle tilt dynamics (first-order with rate limiting).

        θ̇_nacelle = (θ_cmd - θ) / τ, rate-limited
        """
        tau = self.config.propulsion.left_nacelle.tilt_time_constant
        rate_limit = np.radians(self.config.propulsion.left_nacelle.tilt_rate_limit)

        # Desired rate
        theta_error = control.nacelle_angle_cmd - state.nacelle_angles
        theta_dot_desired = theta_error / tau

        # Apply rate limiting
        theta_dot = np.clip(theta_dot_desired, -rate_limit, rate_limit)

        return theta_dot

    def _rotor_dynamics(
        self,
        state: VehicleState,
        control: ControlInput,
    ) -> np.ndarray:
        """
        Rotor speed dynamics (first-order lag).

        Ω̇ = (Ω_cmd - Ω) / τ_motor
        """
        tau_motor = 0.5  # Motor time constant [s]

        Omega_error = control.rotor_speed_cmd - state.rotor_speeds
        Omega_dot = Omega_error / tau_motor

        return Omega_dot

    def _collective_dynamics(
        self,
        state: VehicleState,
        control: ControlInput,
    ) -> np.ndarray:
        """
        Collective pitch dynamics (fast actuator).
        """
        tau_swash = 0.1  # Swashplate time constant [s]

        # Add differential collective for yaw control
        cmd_left = control.collective_cmd[0] + control.differential_collective
        cmd_right = control.collective_cmd[1] - control.differential_collective

        theta_error = np.array([cmd_left, cmd_right]) - state.collective_pitch
        theta_dot = theta_error / tau_swash

        return theta_dot

    def _battery_dynamics(
        self,
        state: VehicleState,
        control: ControlInput,
        fm: ForcesMoments,
    ) -> tuple[float, float]:
        """
        Battery state dynamics.

        Returns:
            (soc_dot, temperature_dot)
        """
        # Estimate power consumption from propulsion forces
        # P ≈ T · v_induced + P_profile + P_avionics

        # Simplified power model
        P_hover = 80000  # W (80 kW hover power)
        P_cruise = 50000  # W (50 kW cruise power)

        # Blend based on nacelle angle
        nacelle_avg = np.mean(state.nacelle_angles)
        hover_factor = np.sin(nacelle_avg)  # 1 in hover, 0 in cruise

        P_prop = hover_factor * P_hover + (1 - hover_factor) * P_cruise
        P_prop *= control.throttle  # Scale by throttle

        # Add avionics power
        P_avionics = 2000  # W
        P_total = P_prop + P_avionics

        # SOC derivative
        # SOC_dot = -P / (V * Q)
        V_batt = state.battery_voltage
        Q_batt = self.config.battery.pack.capacity_kwh * 1000 * 3600  # Convert to Joules

        soc_dot = -P_total / Q_batt

        # Temperature dynamics (simplified)
        # Heat generation from internal resistance
        I = P_total / V_batt
        R_int = 0.1  # Ohms (pack level)
        P_heat = I**2 * R_int

        # Cooling
        T_ambient = 25.0  # °C
        h_cooling = 50.0  # W/K (heat transfer coefficient)

        # Thermal mass
        m_batt = self.config.battery.pack.mass
        cp_batt = 1000.0  # J/kg·K

        Q_dot = P_heat - h_cooling * (state.battery_temperature - T_ambient)
        temp_dot = Q_dot / (m_batt * cp_batt)

        return soc_dot, temp_dot

    def trim(
        self,
        airspeed: float,
        altitude: float,
        flight_path_angle: float = 0.0,
        nacelle_angle: float = None,
    ) -> tuple[VehicleState, ControlInput]:
        """
        Compute trim state and controls for steady flight.

        Args:
            airspeed: Target airspeed [m/s]
            altitude: Altitude [m]
            flight_path_angle: Climb/descent angle [rad]
            nacelle_angle: Fixed nacelle angle (optional, will optimize if None)

        Returns:
            (trim_state, trim_control)
        """
        # This would use numerical optimization to find trim
        # For now, return approximate values

        # Estimate nacelle angle from airspeed
        if nacelle_angle is None:
            if airspeed < 20:  # Hover
                nacelle_angle = np.pi / 2
            elif airspeed > 60:  # Full cruise
                nacelle_angle = 0.0
            else:  # Transition
                nacelle_angle = np.pi / 2 * (1 - (airspeed - 20) / 40)

        # Create trim state
        state = VehicleState(
            position=np.array([0, 0, -altitude]),
            velocity=np.array([airspeed * np.cos(flight_path_angle), 0,
                              airspeed * np.sin(flight_path_angle)]),
            attitude=Quaternion.from_euler(0, flight_path_angle, 0),
            nacelle_angles=np.array([nacelle_angle, nacelle_angle]),
            rotor_speeds=np.array([150.0, 150.0]),
        )

        # Estimate collective for weight support
        # T = mg → θ_col ≈ f(T, Ω, V)
        W = self.mass * G
        alpha = np.arctan2(state.velocity[2], max(abs(state.velocity[0]), 1e-6))
        T_per_rotor = W / 2 / max(np.cos(alpha), 0.01)

        # Simplified inverse thrust model
        Omega = 150.0  # rad/s
        k_T = 0.5
        theta_col = T_per_rotor / (k_T * Omega**2) - 0.1
        theta_col = np.clip(theta_col, 0.05, 0.3)

        state.collective_pitch = np.array([theta_col, theta_col])

        # Create trim control
        control = ControlInput(
            nacelle_angle_cmd=state.nacelle_angles.copy(),
            rotor_speed_cmd=state.rotor_speeds.copy(),
            collective_cmd=state.collective_pitch.copy(),
            throttle=0.7,
        )

        return state, control

    def linearize(
        self,
        state: VehicleState,
        control: ControlInput,
        epsilon: float = 1e-6,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Linearize dynamics about a trim point.

        Computes Jacobians A and B such that:
            δẋ ≈ A·δx + B·δu

        Args:
            state: Trim state
            control: Trim control
            epsilon: Finite difference step

        Returns:
            (A, B) - State and control Jacobians
        """
        x0 = state.to_array()
        u0 = control.to_array()

        n_states = len(x0)
        n_controls = len(u0)

        A = np.zeros((n_states, n_states))
        B = np.zeros((n_states, n_controls))

        # Nominal derivative
        f0 = self.compute_derivatives(state, control)

        # State Jacobian (A)
        for i in range(n_states):
            x_plus = x0.copy()
            x_plus[i] += epsilon
            state_plus = VehicleState.from_array(x_plus, state.time)
            f_plus = self.compute_derivatives(state_plus, control)
            A[:, i] = (f_plus - f0) / epsilon

        # Control Jacobian (B)
        for i in range(n_controls):
            u_plus = u0.copy()
            u_plus[i] += epsilon
            control_plus = ControlInput.from_array(u_plus)
            f_plus = self.compute_derivatives(state, control_plus)
            B[:, i] = (f_plus - f0) / epsilon

        return A, B
