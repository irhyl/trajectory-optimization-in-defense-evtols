"""
Wing Aerodynamics Model

This module implements a finite wing aerodynamic model suitable
for tiltrotor eVTOL in transition and cruise flight.

Theory
======

Lift Coefficient:
    CL = CLα · (α - α₀) + CL_flap · δf

    where:
    - CLα: Lift curve slope (typically 0.8-0.9 of 2π for finite wing)
    - α₀: Zero-lift angle of attack
    - δf: Flap deflection

Finite Wing Correction (Lifting Line):
    CLα = CL_2D / (1 + CL_2D / (π·e·AR))

    where:
    - AR: Aspect ratio (b²/S)
    - e: Oswald efficiency factor

Drag Polar:
    CD = CD0 + CDi = CD0 + CL² / (π·e·AR)

Pitching Moment:
    Cm = Cm0 + Cmα·α + Cmδe·δe

Control Surfaces:
    - Flaps: High lift devices for transition
    - Ailerons: Roll control
    - Spoilers: (optional) Speed brakes

Ground Effect:
    Reduces induced drag when h/b < 1
    CDi_GE = CDi · φ(h/b)

Dynamic Stall:
    Onset when α > αstall and α̇ > 0
    Causes hysteresis in CL-α curve
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
import logging

from ..config import WingConfig

logger = logging.getLogger(__name__)


@dataclass
class WingState:
    """Wing aerodynamic state."""
    # Angle of attack
    alpha: float = 0.0           # rad
    alpha_eff: float = 0.0       # rad (effective, with flap effect)

    # Coefficients
    CL: float = 0.0              # Lift coefficient
    CD: float = 0.0              # Drag coefficient
    Cm: float = 0.0              # Pitching moment coefficient

    # Forces (body frame)
    lift: float = 0.0            # N
    drag: float = 0.0            # N
    side_force: float = 0.0      # N

    # Moments (body frame about CG)
    moment_pitch: float = 0.0    # N·m
    moment_roll: float = 0.0     # N·m
    moment_yaw: float = 0.0      # N·m

    # Derived
    L_D: float = 0.0             # Lift-to-drag ratio
    stalled: bool = False


class WingAerodynamics:
    """
    Finite wing aerodynamic model.

    Models a straight or moderately swept wing with:
    - Finite span effects (lifting line)
    - Control surfaces (flaps, ailerons)
    - Ground effect
    - Simple stall model

    The wing is assumed symmetric about the aircraft centerline.

    Attributes:
        config: Wing configuration
        S: Wing area [m²]
        b: Wing span [m]
        AR: Aspect ratio
        c_bar: Mean aerodynamic chord [m]
    """

    def __init__(self, config: WingConfig):
        """
        Initialize wing aerodynamics.

        Args:
            config: Wing configuration
        """
        self.config = config

        # Geometric parameters
        self.S = config.area
        self.b = config.span
        self.AR = self.b**2 / self.S
        self.c_bar = self.S / self.b  # MAC (simplified)

        # Aerodynamic centre position relative to CG in body frame.
        # WingConfig does not store a CG-relative position; the aerodynamic
        # centre of a wing is conventionally at 25% MAC forward of the wing
        # leading edge.  We place it at the airframe mid-station (x=0, y=0, z=0)
        # as a conservative default that callers can override by subclassing.
        self.r_ac = np.zeros(3)   # [x, y, z] body frame [m]

        # Aerodynamic derivatives
        self._compute_derivatives()

        # State
        self.state = WingState()

        logger.info(f"WingAerodynamics initialized: S={self.S:.1f}m², "
                   f"b={self.b:.1f}m, AR={self.AR:.1f}")

    def _compute_derivatives(self):
        """Compute aerodynamic derivatives from config."""
        cfg = self.config

        # 2D airfoil lift slope (typically 2π)
        CL_alpha_2D = 2 * np.pi

        # Finite wing correction (Prandtl lifting-line theory).
        # WingConfig does not store oswald_efficiency directly; we estimate it
        # from the Pamadi (2004) approximation: e ≈ 1.78(1 - 0.045·AR^0.68) - 0.64
        e = max(0.5, 1.78 * (1.0 - 0.045 * self.AR**0.68) - 0.64)
        self.CLa = CL_alpha_2D / (1 + CL_alpha_2D / (np.pi * e * self.AR))

        # Induced drag factor
        self.K = 1 / (np.pi * e * self.AR)

        # Zero-lift parameters.
        # WingConfig stores cl0 (zero-AoA CL) and cl_alpha; the zero-lift AoA
        # is  α₀ = -CL₀ / CLα  (by definition of the linear lift model).
        self.CL0 = cfg.cl0
        self.alpha_0 = -cfg.cl0 / max(cfg.cl_alpha, 1e-6)  # Zero-lift AoA [rad]

        # Drag polar — WingConfig uses cd0 (not cd_0)
        self.CD0 = cfg.cd0

        # Pitching moment — WingConfig uses cm0 (not cm_0)
        self.Cm0 = cfg.cm0
        self.Cma = -0.1  # Typically negative (stable)

        # Stall
        self.alpha_stall = np.radians(15)  # Stall angle
        self.CL_max = cfg.cl_max

        # Control derivatives
        self.CLdf = 0.8   # dCL/d(flap) [/rad]
        self.CLda = 0.3   # dCL/d(aileron) [/rad] (rolling moment)
        self.CDdf = 0.05  # Flap drag increment [/rad]

    def compute(
        self,
        V_inf: float,
        alpha: float,
        beta: float,
        p: float,
        q: float,
        r: float,
        delta_flap: float = 0.0,
        delta_aileron: float = 0.0,
        rho: float = 1.225,
        height_agl: float = None,
    ) -> WingState:
        """
        Compute wing aerodynamic forces and moments.

        Args:
            V_inf: Freestream velocity [m/s]
            alpha: Angle of attack [rad]
            beta: Sideslip angle [rad]
            p: Roll rate [rad/s]
            q: Pitch rate [rad/s]
            r: Yaw rate [rad/s]
            delta_flap: Flap deflection [rad] (positive down)
            delta_aileron: Aileron deflection [rad] (positive = right wing down)
            rho: Air density [kg/m³]
            height_agl: Height above ground for ground effect [m]

        Returns:
            WingState with forces and moments
        """
        state = WingState()
        state.alpha = alpha

        if V_inf < 1.0:
            self.state = state
            return state

        # Dynamic pressure
        q_bar = 0.5 * rho * V_inf**2

        # Effective angle of attack (flap increases effective camber)
        alpha_eff = alpha + 0.3 * delta_flap  # Flap effectiveness
        state.alpha_eff = alpha_eff

        # Lift coefficient
        CL = self.CLa * (alpha_eff - self.alpha_0) + self.CLdf * delta_flap

        # Rate effects (pitch damping)
        CL += (self.CLa * self.c_bar / (2 * V_inf)) * q

        # Stall model
        if abs(alpha) > self.alpha_stall:
            state.stalled = True
            # Post-stall: CL drops, CD increases
            if alpha > 0:
                CL = self.CL_max * np.cos(alpha - self.alpha_stall)
            else:
                CL = -self.CL_max * np.cos(alpha + self.alpha_stall)
        else:
            CL = np.clip(CL, -self.CL_max, self.CL_max)

        state.CL = CL

        # Drag coefficient (parabolic polar)
        CDi = self.K * CL**2  # Induced drag

        # Ground effect on induced drag
        if height_agl is not None and height_agl < self.b:
            ge_factor = self._ground_effect_factor(height_agl)
            CDi *= ge_factor

        CD0 = self.CD0 + self.CDdf * abs(delta_flap)
        state.CD = CD0 + CDi

        # Pitching moment
        state.Cm = self.Cm0 + self.Cma * alpha

        # Forces in wind frame
        L = q_bar * self.S * state.CL
        D = q_bar * self.S * state.CD

        # Transform to body frame
        ca, sa = np.cos(alpha), np.sin(alpha)
        Fx_wind = -D
        Fz_wind = -L

        state.drag = D
        state.lift = L

        # Transform wind-frame forces to body-frame (standard wind→body rotation)
        # x_body = cos(α)·Fx_wind + sin(α)·Fz_wind  (positive forward)
        # z_body = −sin(α)·Fx_wind + cos(α)·Fz_wind  (positive down in NED)
        Fx_body = ca * Fx_wind + sa * Fz_wind
        Fz_body = -sa * Fx_wind + ca * Fz_wind

        # Side force (from sideslip)
        CY = -0.3 * beta  # Side force derivative
        Fy = q_bar * self.S * CY
        state.side_force = Fy

        # Moments about CG
        # Pitching moment
        state.moment_pitch = q_bar * self.S * self.c_bar * state.Cm

        # Moment from lift at AC offset from CG
        x_ac = self.r_ac[0]  # x distance from CG to AC
        state.moment_pitch += -L * x_ac * np.cos(alpha)

        # Rolling moment (from aileron)
        Cl_roll = self.CLda * delta_aileron
        # Roll damping
        Cl_roll -= 0.4 * (p * self.b / (2 * V_inf))
        state.moment_roll = q_bar * self.S * self.b * Cl_roll

        # Yawing moment (from sideslip)
        Cn = 0.05 * beta  # Weathercock stability
        # Yaw damping
        Cn -= 0.1 * (r * self.b / (2 * V_inf))
        state.moment_yaw = q_bar * self.S * self.b * Cn

        # Store body-frame aerodynamic forces on state for use by callers
        # (e.g. rigid_body.py assembles total forces from these values)
        state.lift = L              # Lift magnitude [N]  (perpendicular to velocity)
        state.drag = D              # Drag magnitude [N]  (parallel to velocity)
        # NOTE: Fx_body / Fz_body already computed above; expose via state
        # so that the vehicle model does not need to re-do the rotation.
        # We re-use the side_force slot for Fy and add body-frame components.
        state.side_force = Fy

        # Overwrite force scalars with body-frame components so callers get
        # consistent sign conventions without extra rotation steps.
        # Positive x = forward, positive z = down (NED body convention).
        state.drag = -Fx_body   # Store as drag magnitude (positive = retarding)
        state.lift = -Fz_body   # Store as lift magnitude (positive = upward)

        # Lift-to-drag ratio
        if state.CD > 0:
            state.L_D = state.CL / state.CD

        self.state = state
        return state

    def _ground_effect_factor(self, height: float) -> float:
        """
        Compute ground effect factor for induced drag.

        Uses classical formula for rectangular wing.

        Args:
            height: Height above ground [m]

        Returns:
            Factor to multiply induced drag (< 1 in ground effect)
        """
        h_b = height / self.b

        if h_b >= 1:
            return 1.0

        # Empirical ground effect formula
        phi = (16 * h_b)**2 / (1 + (16 * h_b)**2)

        return phi

    def get_forces_moments_body(
        self,
        V_body: np.ndarray,
        omega_body: np.ndarray,
        delta_flap: float = 0.0,
        delta_aileron: float = 0.0,
        rho: float = 1.225,
        height_agl: float = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute wing forces and moments in body frame.

        Args:
            V_body: Velocity in body frame [u, v, w] [m/s]
            omega_body: Angular velocity [p, q, r] [rad/s]
            delta_flap: Flap deflection [rad]
            delta_aileron: Aileron deflection [rad]
            rho: Air density [kg/m³]
            height_agl: Height above ground [m]

        Returns:
            (forces, moments) in body frame
        """
        u, v, w = V_body
        p, q, r = omega_body

        # Airspeed
        V_inf = np.sqrt(u**2 + v**2 + w**2)

        if V_inf < 1.0:
            return np.zeros(3), np.zeros(3)

        # Aerodynamic angles
        alpha = np.arctan2(w, u) if abs(u) > 0.1 else 0.0
        beta = np.arcsin(v / V_inf) if V_inf > 0.1 else 0.0

        # Compute aerodynamics
        state = self.compute(
            V_inf, alpha, beta, p, q, r,
            delta_flap, delta_aileron, rho, height_agl
        )

        # Forces in body frame
        ca, sa = np.cos(alpha), np.sin(alpha)
        cb, sb = np.cos(beta), np.sin(beta)

        Fx = -state.drag * ca * cb - state.side_force * sb + state.lift * sa
        Fy = -state.drag * sb + state.side_force * cb
        Fz = -state.drag * sa * cb - state.lift * ca

        forces = np.array([Fx, Fy, Fz])
        moments = np.array([state.moment_roll, state.moment_pitch, state.moment_yaw])

        return forces, moments

    def get_trim_alpha(
        self,
        weight: float,
        V_inf: float,
        rho: float = 1.225,
    ) -> float:
        """
        Find angle of attack for level flight.

        Args:
            weight: Aircraft weight [N]
            V_inf: Airspeed [m/s]
            rho: Air density [kg/m³]

        Returns:
            Required angle of attack [rad]
        """
        q_bar = 0.5 * rho * V_inf**2
        CL_required = weight / (q_bar * self.S)

        # Invert CL equation
        alpha_required = CL_required / self.CLa + self.alpha_0

        return alpha_required
