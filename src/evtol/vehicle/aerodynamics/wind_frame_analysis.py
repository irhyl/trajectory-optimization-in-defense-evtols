"""
Wind Frame Aerodynamic Analysis and Multi-Rotor Interference

This module provides tools for:
1. Wind frame transformations (body ↔ wind frame)
2. Angle of attack and sideslip calculations
3. Multi-rotor interference wake effects
4. Aerodynamic force/moment transformations
5. Dynamic pressure and Reynolds number calculations

Theory
======

Wind Frame Kinematics:
    Body frame: [x_b (forward), y_b (right), z_b (down)]
    Wind frame: [x_w (freestream), y_w (sideslip), z_w (normal)]
    
    Transformations via rotation matrix with angle of attack (α) and
    sideslip (β):
        V_wind = R_w→b(α, β) · V_body
    
Angle of Attack: α = arctan(V_z / V_x)  [rad]
    - Positive: nose up relative to velocity
    - Range: [-π/2, π/2]

Sideslip Angle: β = arcsin(V_y / |V|)  [rad]
    - Positive: nose right relative to velocity
    - Range: [-π/2, π/2]

Multi-Rotor Interference (Quad-Rotor Configuration):

        Front-Left (FL)           Front-Right (FR)
                 ◆                         ◆
                 │                         │
        ─────────┼─────────────────────────┼─────────
                 │                         │
        ─────────┼─────────────────────────┼─────────
                 │                         │
                 ◆                         ◆
        Rear-Left (RL)            Rear-Right (RR)

Interference effects:
1. **Wake Deficit**: Downstream rotors operate in reduced inflow
   - Front rotors induce downwash at rear rotor disk
   - Estimated: Δv_induced ≈ -0.1 to -0.2 × v_center (10-20% power increase)

2. **Dynamic Coupling**: Unequal thrust changes induce transient moments
   - Power adjustment: P_i = P_0 · Π_j (w_ij)  where w_ij ∈ [0.9, 1.1]

3. **Aerodynamic Interference**: Fuselage/boom blocking reduces rotor inflow
   - Effect: ~5% power at hover, ~15% in forward flight

References
==========

[1] Stewart, D.F. (1981). "An Investigation of Main Rotor Interference
    Effects on a Tiltrotor Aircraft." NASA TM-81229.

[2] Stepniewski, W.Z., Keys, C.N. (1984). Rotorcraft Aerodynamics.
    Dover Publications. Chapters 9-10 (interference, multi-rotor).

[3] Stevens, B.L., Lewis, F.L., Johnson, E.N. (2015). Aircraft Control and
    Simulation, 3rd ed. Chapter 2 (coordinate transformations).

[4] Prouty, R.G. (2002). Helicopter Performance, Stability, and Control.
    Krieger Publishing. Chapter 5 (interference effects).
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class RotorPosition(Enum):
    """Rotor position identifiers for quad configuration."""
    FRONT_LEFT = 0
    FRONT_RIGHT = 1
    REAR_RIGHT = 2
    REAR_LEFT = 3


@dataclass
class WindFrameState:
    """Wind frame flight state."""
    # Velocity magnitudes
    V_total: float = 0.0       # Total velocity [m/s]
    V_x_wind: float = 0.0      # Freestream component [m/s]
    V_y_wind: float = 0.0      # Sideslip component [m/s]
    V_z_wind: float = 0.0      # Normal component [m/s]
    
    # Angles
    alpha: float = 0.0         # Angle of attack [rad]
    beta: float = 0.0          # Sideslip angle [rad]
    
    # Derived
    dynamic_pressure: float = 0.0  # q = 0.5·ρ·V² [Pa]
    mach: float = 0.0          # Mach number
    reynolds: float = 0.0      # Reynolds number
    

@dataclass
class InterferenceFactors:
    """Multi-rotor interference coefficients."""
    # Wake deficit matrix (4×4 for quad configuration)
    # w_ij = power reduction factor from rotor i onto rotor j
    # w_ii = 1.0 (self), w_ij < 1.0 (benefits downstream rotor)
    wake_matrix: np.ndarray = None
    
    # Fuselage interference factors
    fuselage_factor: float = 1.0   # Applied to all rotors equally
    
    # Overall combined factor
    combined: np.ndarray = None  # 4-element array of interference factors


class WindFrameTransform:
    """
    Bidirectional wind frame coordinate transformations.
    
    Provides:
    - Body ↔ Wind frame conversions
    - Angle of attack and sideslip computation
    - Rotation matrix generation
    - Euler angle relationships
    """
    
    @staticmethod
    def body_to_wind_frame(
        velocity_body: np.ndarray,
    ) -> tuple[float, float, float]:
        """
        Convert body-frame velocity to wind frame angles.
        
        Body frame: [u (forward), v (right), w (down)]
        Wind frame angles: (α, β, V_total)
        
        Args:
            velocity_body: Velocity [u, v, w] [m/s]
        
        Returns:
            (alpha, beta, V_total) where:
            - alpha: Angle of attack [rad]
            - beta: Sideslip angle [rad]
            - V_total: Total velocity magnitude [m/s]
        """
        u, v, w = velocity_body
        V_total = np.linalg.norm(velocity_body)
        
        if V_total < 0.01:
            return 0.0, 0.0, V_total
        
        # Angle of attack: α = arctan(w/u)
        alpha = np.arctan2(w, u)
        
        # Sideslip: β = arcsin(v/V)
        beta = np.arcsin(np.clip(v / V_total, -1.0, 1.0))
        
        return alpha, beta, V_total
    
    @staticmethod
    def wind_to_body_frame(
        alpha: float,
        beta: float,
        V_total: float = 1.0,
    ) -> np.ndarray:
        """
        Convert wind frame angles to body-frame velocity.
        
        Inverse of body_to_wind_frame.
        
        Args:
            alpha: Angle of attack [rad]
            beta: Sideslip angle [rad]
            V_total: Total velocity magnitude [m/s]
        
        Returns:
            velocity_body [u, v, w] [m/s]
        """
        # u = V·cos(α)·cos(β)
        # v = V·sin(β)
        # w = V·sin(α)·cos(β)
        
        cos_alpha = np.cos(alpha)
        sin_alpha = np.sin(alpha)
        cos_beta = np.cos(beta)
        sin_beta = np.sin(beta)
        
        u = V_total * cos_alpha * cos_beta
        v = V_total * sin_beta
        w = V_total * sin_alpha * cos_beta
        
        return np.array([u, v, w])
    
    @staticmethod
    def dcm_body_to_wind(alpha: float, beta: float) -> np.ndarray:
        """
        Direction cosine matrix from body to wind frame.
        
        Rotation sequence: Yaw (β) then Pitch (α)
        
        Args:
            alpha: Angle of attack [rad]
            beta: Sideslip angle [rad]
        
        Returns:
            3×3 DCM matrix: R_w→b
        """
        ca = np.cos(alpha)
        sa = np.sin(alpha)
        cb = np.cos(beta)
        sb = np.sin(beta)
        
        # Rotation matrix sequence:
        # 1. Sideslip rotation about z-axis
        # 2. Angle of attack rotation about y-axis
        
        R = np.array([
            [ca * cb,      -ca * sb,      -sa],
            [sb,           cb,            0.0],
            [sa * cb,      -sa * sb,      ca],
        ])
        
        return R
    
    @staticmethod
    def dcm_wind_to_body(alpha: float, beta: float) -> np.ndarray:
        """
        Direction cosine matrix from wind to body frame (inverse).
        
        Args:
            alpha: Angle of attack [rad]
            beta: Sideslip angle [rad]
        
        Returns:
            3×3 DCM matrix: R_b→w (inverse of dcm_body_to_wind)
        """
        R_w_to_b = WindFrameTransform.dcm_body_to_wind(alpha, beta)
        return R_w_to_b.T


class MultiRotorInterference:
    """
    Multi-rotor interference modeling for quad-rotor configurations.
    
    Accounts for:
    1. Wake deficit from leading rotors to trailing rotors
    2. Fuselage/boom blocking effects
    3. Dynamic coupling between rotors
    
    Based on momentum theory with empirical wake deficit factors.
    """
    
    def __init__(
        self,
        rotor_spacing: float = 2.0,  # Characteristic spacing [m]
        rotor_radius: float = 2.0,   # Rotor radius [m]
    ):
        """
        Initialize multi-rotor interference model.
        
        Args:
            rotor_spacing: Typical rotor-to-rotor distance [m]
            rotor_radius: Rotor disk radius [m]
        """
        self.rotor_spacing = rotor_spacing
        self.rotor_radius = rotor_radius
        
        # Quad rotor positions (normalized: [-1, 1] for front/back and left/right)
        self.rotor_positions_normalized = np.array([
            [-1, -1],  # Front-Left (FL)
            [+1, -1],  # Front-Right (FR)
            [+1, +1],  # Rear-Right (RR)
            [-1, +1],  # Rear-Left (RL)
        ])
        
        # Build base interference matrix (symmetric for hover)
        self._build_interference_matrix()
        
        logger.info(f"MultiRotorInterference initialized: spacing={rotor_spacing}m, "
                   f"radius={rotor_radius}m")
    
    def _build_interference_matrix(self):
        """
        Construct interference coefficient matrix.
        
        w_ij = power factor for rotor j influenced by rotor i
        - w_ii = 1.0 (self-interaction)
        - w_ij < 1.0 (wake benefits downstream rotor, reduces power needed)
        - w_ij > 1.0 (potential blockage increases power)
        """
        self.interference_matrix = np.zeros((4, 4))
        
        for i in range(4):
            for j in range(4):
                if i == j:
                    # Self: apply fuselage blockage factor
                    self.interference_matrix[i, j] = 1.0
                else:
                    # Compute distance between rotors
                    pos_i = self.rotor_positions_normalized[i]
                    pos_j = self.rotor_positions_normalized[j]
                    dist_normalized = np.linalg.norm(pos_i - pos_j)
                    
                    # Check if rotor j is downstream of rotor i
                    # Downstream: front rotors affect rear rotors primarily
                    is_downstream = (pos_j[0] > pos_i[0])  # j to the right (forward)
                    
                    if is_downstream:
                        # Wake benefit: reduce power needed at j
                        # Model: w_ij ≈ 0.85-0.95 (15-5% benefit)
                        wake_deficit = 0.05 * (2 - dist_normalized)
                        self.interference_matrix[i, j] = 1.0 - wake_deficit
                    else:
                        # Upstream or lateral: minimal effect or blockage
                        self.interference_matrix[i, j] = 1.0
    
    def compute_interference_factors(
        self,
        thrust_nominal: np.ndarray,
        flight_condition: str = "hover",
    ) -> InterferenceFactors:
        """
        Compute interference power factors for all rotors.
        
        Args:
            thrust_nominal: Nominal thrust per rotor (no interference) [N] (4-element)
            flight_condition: "hover", "forward", or "transition"
        
        Returns:
            InterferenceFactors with wake matrix and combined factors
        """
        # Apply fuselage blockage
        # In forward flight, fuselage blockage increases (~10-15%)
        if flight_condition == "forward":
            fuselage_factor = 1.10  # 10% power increase from fuselage blockage
        elif flight_condition == "transition":
            fuselage_factor = 1.05  # 5% intermediate
        else:  # hover
            fuselage_factor = 1.03  # 3% even in hover
        
        # Asymmetry based on thrust distribution
        # If thrusts are unequal, interference varies
        thrust_ratios = thrust_nominal / np.mean(thrust_nominal)
        
        # Apply thrust-dependent corrections
        # High-thrust rotor has increased interference (loads other rotors more)
        wake_matrix_effective = self.interference_matrix.copy()
        for i in range(4):
            for j in range(4):
                if i != j:
                    # Scale interference by thrust ratio
                    ratio_i = thrust_ratios[i]
                    # More thrust at i → more interference to j
                    scale_factor = 1.0 + 0.1 * (ratio_i - 1.0)
                    wake_matrix_effective[i, j] *= scale_factor
        
        # Compute combined factors (applied to power)
        combined_factors = np.ones(4)
        for j in range(4):
            # Sum of influences from all other rotors on rotor j
            interference_total = np.sum(wake_matrix_effective[:, j])
            combined_factors[j] = fuselage_factor * interference_total
        
        return InterferenceFactors(
            wake_matrix=wake_matrix_effective,
            fuselage_factor=fuselage_factor,
            combined=combined_factors,
        )


class AerodynamicsMetrics:
    """
    Compute aerodynamic metrics for flight analysis.
    """
    
    @staticmethod
    def compute_dynamic_pressure(
        velocity: float,
        rho: float = 1.225,
    ) -> float:
        """
        Compute dynamic pressure.
        
        q = 0.5 · ρ · V²  [Pa]
        
        Args:
            velocity: True airspeed [m/s]
            rho: Air density [kg/m³]
        
        Returns:
            Dynamic pressure [Pa]
        """
        return 0.5 * rho * velocity**2
    
    @staticmethod
    def compute_reynolds_number(
        velocity: float,
        characteristic_length: float = 0.15,
        dynamic_viscosity: float = 1.81e-5,
    ) -> float:
        """
        Compute Reynolds number.
        
        Re = ρ·V·L / μ = V·L / ν  [dimensionless]
        
        Args:
            velocity: True airspeed [m/s]
            characteristic_length: Chord or rotor radius [m]
            dynamic_viscosity: Dynamic viscosity [Pa·s]
        
        Returns:
            Reynolds number
        """
        nu = dynamic_viscosity / 1.225  # Kinematic viscosity [m²/s]
        return velocity * characteristic_length / max(nu, 1e-6)
    
    @staticmethod
    def compute_mach_number(
        velocity: float,
        altitude: float = 0.0,
    ) -> float:
        """
        Compute Mach number.
        
        M = V / a  where a = √(γ·R·T)
        
        Args:
            velocity: True airspeed [m/s]
            altitude: Altitude above sea level [m]
        
        Returns:
            Mach number
        """
        # Temperature lapse rate
        T = 15.0 - 0.0065 * altitude  # °C
        T_kelvin = T + 273.15
        
        # Speed of sound
        a = np.sqrt(1.4 * 287.05 * T_kelvin)
        
        return velocity / a
    
    @staticmethod
    def atmosphere_model(altitude: float) -> tuple[float, float, float]:
        """
        1976 Standard Atmosphere model.
        
        Args:
            altitude: Altitude above sea level [m]
        
        Returns:
            (rho, T, a) - density [kg/m³], temperature [°C], speed of sound [m/s]
        """
        if altitude < 11000:
            # Troposphere
            T = 15.0 - 0.0065 * altitude
            T_ratio = (T + 273.15) / 288.15
            rho = 1.225 * (T_ratio ** (-5.255))
        else:
            # Stratosphere (simplified)
            T = -56.5
            T_ratio = (T + 273.15) / 288.15
            rho = 1.225 * (T_ratio ** (-5.255))
        
        a = np.sqrt(1.4 * 287.05 * (T + 273.15))
        
        return rho, T, a
