"""
Fuselage Aerodynamics Model

This module models the aerodynamic forces on the fuselage and
nacelle fairings for a tiltrotor eVTOL.

Theory
======

The fuselage contributes primarily parasitic drag:
    D_f = q · Cd_f · S_ref

where Cd_f includes:
- Skin friction drag (turbulent boundary layer)
- Form drag (pressure drag from shape)
- Interference drag (wing-body junction)

For streamlined bodies:
    Cd_f ≈ Cf · (1 + 1.5/FR^1.5 + 7/FR³) · S_wet/S_ref

where:
- Cf: Skin friction coefficient
- FR: Fineness ratio (length/diameter)
- S_wet: Wetted area

Side Force:
    Y = q · Cy_β · β · S_ref

Moments:
- The fuselage typically destabilizes pitch (nose up with α)
- Provides weathercock stability in yaw

Nacelle Fairing:
    Additional drag from motor nacelles
    D_nacelle = q · Cd_nacelle · A_frontal
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
import logging

from ..config import FuselageConfig

logger = logging.getLogger(__name__)


@dataclass
class FuselageState:
    """Fuselage aerodynamic state."""
    # Coefficients
    CD: float = 0.0              # Drag coefficient
    CY: float = 0.0              # Side force coefficient
    Cm: float = 0.0              # Pitching moment coefficient
    Cn: float = 0.0              # Yawing moment coefficient

    # Forces (body frame)
    drag: float = 0.0            # N
    side_force: float = 0.0      # N

    # Moments (body frame)
    moment_pitch: float = 0.0    # N·m
    moment_yaw: float = 0.0      # N·m


class FuselageAerodynamics:
    """
    Fuselage and nacelle aerodynamic model.

    Models:
    - Fuselage parasitic drag
    - Nacelle drag (including tilting nacelles)
    - Fuselage side force in sideslip
    - Pitch destabilization from fuselage
    - Yaw weathercock from fuselage/nacelles

    Attributes:
        config: Fuselage configuration
        S_ref: Reference area [m²]
        l_ref: Reference length [m]
    """

    def __init__(
        self,
        config: FuselageConfig,
        S_ref: float = 1.0,
        l_ref: float = 1.0,
    ):
        """
        Initialize fuselage aerodynamics.

        Args:
            config: Fuselage configuration
            S_ref: Reference area for coefficients [m²]
            l_ref: Reference length for moments [m]
        """
        self.config = config
        self.S_ref = S_ref
        self.l_ref = l_ref

        # Fuselage geometry
        self.length = config.length
        self.width = config.width
        self.height = config.height

        # Fineness ratio
        d_eq = np.sqrt(self.width * self.height)  # Equivalent diameter
        self.FR = self.length / d_eq if d_eq > 0 else 5.0

        # Wetted area (approximate for streamlined fuselage)
        self.S_wet = np.pi * d_eq * self.length * 0.7  # 70% of cylinder

        # Frontal area
        self.A_frontal = np.pi * (d_eq / 2)**2 * 0.8

        # Drag coefficient
        self._compute_drag_coefficient()

        # State
        self.state = FuselageState()

        logger.info(f"FuselageAerodynamics initialized: L={self.length:.1f}m, "
                   f"FR={self.FR:.1f}, CD0={self.CD0:.4f}")

    def _compute_drag_coefficient(self):
        """Compute base drag coefficient."""
        # Skin friction coefficient (turbulent, Re ~ 10^7)
        Re = 1e7  # Approximate Reynolds number
        Cf = 0.074 / Re**0.2  # Prandtl-Schlichting

        # Form factor for streamlined body
        k = 1 + 1.5 / self.FR**1.5 + 7 / self.FR**3

        # Base drag coefficient
        self.CD0 = Cf * k * (self.S_wet / self.S_ref)

        # Add base drag if blunt rear
        self.CD0 += self.config.cd_frontal

        # Side force derivative (per radian of sideslip)
        self.CY_beta = -0.3  # Typical for fuselage

        # Moment derivatives
        self.Cm_alpha = 0.15   # Pitch up with alpha (destabilizing)
        self.Cn_beta = 0.05    # Weathercock stability

    def compute(
        self,
        V_inf: float,
        alpha: float,
        beta: float,
        rho: float = 1.225,
        nacelle_angle: float = 0.0,
    ) -> FuselageState:
        """
        Compute fuselage aerodynamic forces and moments.

        Args:
            V_inf: Freestream velocity [m/s]
            alpha: Angle of attack [rad]
            beta: Sideslip angle [rad]
            rho: Air density [kg/m³]
            nacelle_angle: Nacelle tilt angle [rad]

        Returns:
            FuselageState
        """
        state = FuselageState()

        if V_inf < 1.0:
            self.state = state
            return state

        q = 0.5 * rho * V_inf**2

        # Drag coefficient (increases with angle of attack)
        CD = self.CD0 * (1 + 2 * alpha**2 + beta**2)

        # Nacelle drag (increases when tilted due to frontal area)
        A_nacelle = 0.3  # m² (approximate nacelle frontal area)
        CD_nacelle = 0.1 * np.sin(nacelle_angle)**2 * (A_nacelle / self.S_ref)
        CD += CD_nacelle

        state.CD = CD
        state.drag = q * self.S_ref * CD

        # Side force
        state.CY = self.CY_beta * beta
        state.side_force = q * self.S_ref * state.CY

        # Pitching moment (destabilizing)
        state.Cm = self.Cm_alpha * alpha
        state.moment_pitch = q * self.S_ref * self.l_ref * state.Cm

        # Yawing moment (stabilizing weathercock)
        state.Cn = self.Cn_beta * beta
        state.moment_yaw = q * self.S_ref * self.l_ref * state.Cn

        self.state = state
        return state

    def get_forces_moments_body(
        self,
        V_body: np.ndarray,
        nacelle_angle: float = 0.0,
        rho: float = 1.225,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute fuselage forces and moments in body frame.

        Args:
            V_body: Velocity in body frame [u, v, w] [m/s]
            nacelle_angle: Nacelle tilt [rad]
            rho: Air density [kg/m³]

        Returns:
            (forces, moments) in body frame
        """
        u, v, w = V_body

        V_inf = np.sqrt(u**2 + v**2 + w**2)

        if V_inf < 1.0:
            return np.zeros(3), np.zeros(3)

        # Aerodynamic angles
        alpha = np.arctan2(w, u) if abs(u) > 0.1 else 0.0
        beta = np.arcsin(v / V_inf) if V_inf > 0.1 else 0.0

        state = self.compute(V_inf, alpha, beta, rho, nacelle_angle)

        # Forces in body frame
        # Drag opposes velocity
        D_body = state.drag * V_body / V_inf

        forces = -D_body  # Drag opposes motion
        forces[1] += state.side_force  # Side force

        moments = np.array([
            0.0,                  # No roll moment from fuselage
            state.moment_pitch,   # Pitch
            state.moment_yaw,     # Yaw
        ])

        return forces, moments


class NacelleInterference:
    """
    Models aerodynamic interference between nacelles and wing.

    Tiltrotor nacelles affect:
    - Local wing lift (upwash/downwash)
    - Induced drag
    - Nacelle-wing junction drag
    """

    def __init__(
        self,
        nacelle_span_position: float,  # y/b
        nacelle_diameter: float,
        wing_chord: float,
    ):
        """
        Initialize nacelle interference model.

        Args:
            nacelle_span_position: Spanwise position (y/b)
            nacelle_diameter: Nacelle diameter [m]
            wing_chord: Wing chord at nacelle [m]
        """
        self.y_b = nacelle_span_position
        self.d_nacelle = nacelle_diameter
        self.chord = wing_chord

        # Interference factors
        self.k_lift = 0.95   # Lift loss factor
        self.k_drag = 1.10   # Drag increase factor

    def get_interference_factors(
        self,
        nacelle_angle: float,
    ) -> tuple[float, float]:
        """
        Get lift and drag interference factors.

        Args:
            nacelle_angle: Nacelle tilt [rad]

        Returns:
            (lift_factor, drag_factor)
        """
        # Interference worst in transition
        transition_penalty = np.sin(2 * nacelle_angle)

        k_L = self.k_lift - 0.03 * transition_penalty
        k_D = self.k_drag + 0.05 * transition_penalty

        return k_L, k_D
