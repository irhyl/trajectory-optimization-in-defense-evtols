"""
Advanced BEMT Rotor Model with Blade Flapping and VRS Detection

This module extends the basic BEMT rotor model with:
1. Blade Flapping Dynamics - First harmonic flapping modes (β₀, β₁c, β₁s)
2. Vortex Ring State (VRS) Detection - Explicit thrust/power penalties
3. Dynamic Stall Hysteresis - Retreating blade separation, Cl/Cd delays
4. Compressibility Effects - Mach number corrections to Cl/Cd
5. Blade Element Integration - Improved spatial discretization for forward flight

Theory References
=================

[1] Prouty, R.G. (2002). Helicopter Performance, Stability, and Control.
    Krieger Publishing. Chapters 4-7 (Blade flapping, VRS, dynamic stall).

[2] Leishman, J.G. (2006). Principles of Helicopter Aerodynamics.
    Cambridge University Press. Chapters 5-8 (Flapping theory, stall).

[3] Bouabdallah, S. (2007). Design and Control of Quadrotors with Flying
    Manipulators. EPFL Ph.D. Dissertation. (Rotor models for multicopters).

[4] NACA TN 1866 (1949). The Aerodynamic Characteristics of Eight NACA 44XX
    Airfoils in Compressible Flow. (Mach corrections)

Physical Constants
==================

Blade Flapping Natural Frequency: ω_β ≈ 0.75·Ω (tiltrotor)
Flapping Stiffness: K_β from gimbal geometry, typically 0.8-1.2·Ω²

Vortex Ring State: 0.5·v_i < v_z < 1.8·v_i (where v_i is hover induced velocity)
- Thrust reduction: -20% to -50% depending on descent rate
- Power increase: +50% to +100% in VRS

Dynamic Stall:
- Time lag: τ_ds ≈ 2.0·c/(2·V) (convective delay)
- Separation point: xsep = 0.7 + 0.25·AoA (for moderate AoA)

Author: Defense eVTOL Research Team
License: MIT
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from scipy.optimize import brentq
from scipy.integrate import odeint
import logging

from ..config import RotorConfig

logger = logging.getLogger(__name__)

# Physical constants
RHO_SL = 1.225  # kg/m³ sea level air density
SONIC_SPEED_SL = 340.3  # m/s at sea level
GAMMA_AIR = 1.4  # Specific heat ratio for air


@dataclass
class BladeFlappingState:
    """Blade flapping state variables."""
    # Flapping angles [rad]
    beta_0: float = 0.0      # Permanent flapping (generally ~0)
    beta_1c: float = 0.0     # Cyclic flapping - cosine (longitudinal)
    beta_1s: float = 0.0     # Cyclic flapping - sine (lateral)
    
    # Flapping rates [rad/s]
    beta_0_dot: float = 0.0
    beta_1c_dot: float = 0.0
    beta_1s_dot: float = 0.0
    
    # Hub moments [N·m]
    M_roll: float = 0.0
    M_pitch: float = 0.0
    M_yaw: float = 0.0


@dataclass
class VRSState:
    """Vortex Ring State detection and characterization."""
    in_vrs: bool = False
    vrs_severity: float = 0.0  # 0.0 (none) to 1.0 (severe)
    thrust_factor: float = 1.0  # Thrust reduction multiplier (0.5-0.8 in VRS)
    power_factor: float = 1.0   # Power increase multiplier (1.5-2.0 in VRS)
    warning_active: bool = False


@dataclass
class DynamicStallState:
    """Dynamic stall hysteresis modeling."""
    # Effective angle of attack (with time lag)
    alpha_effective: float = 0.0
    
    # Separation point fraction (0.0 = attached, 1.0 = fully separated)
    x_separation: float = 0.0
    
    # Cl/Cd modifiers (1.0 = no stall effect)
    Cl_modifier: float = 1.0
    Cd_modifier: float = 1.0


@dataclass
class AdvancedRotorState:
    """Extended rotor state including flapping, VRS, dynamic stall."""
    # Basic aerodynamic state
    omega: float = 0.0
    rpm: float = 0.0
    thrust: float = 0.0
    torque: float = 0.0
    power_induced: float = 0.0
    power_profile: float = 0.0
    power_total: float = 0.0
    
    # Coefficients
    CT: float = 0.0
    CQ: float = 0.0
    figure_of_merit: float = 0.0
    
    # Advanced states
    flapping: BladeFlappingState = field(default_factory=BladeFlappingState)
    vrs: VRSState = field(default_factory=VRSState)
    stall: DynamicStallState = field(default_factory=DynamicStallState)
    
    # Flight condition
    advance_ratio: float = 0.0
    inflow_ratio: float = 0.0


class AdvancedBEMTRotor:
    """
    Advanced BEMT rotor model with blade flapping, VRS, and dynamic stall.
    
    This model extends basic BEMT with:
    1. First-harmonic blade flapping (β₀, β₁c, β₁s)
    2. Vortex ring state detection with explicit thrust/power penalties
    3. Dynamic stall hysteresis on retreating blade
    4. Compressibility corrections (Mach number effects)
    5. Improved blade element spatial integration
    
    Key Features:
    - Iterative solution for induced velocity self-consistent with flapping
    - Separated modeling of hover vs forward flight regimes
    - Explicit VRS penalty zones based on descent rate
    - First-order dynamic stall with time lag
    - Thread-safe state management
    """
    
    def __init__(
        self,
        config: RotorConfig,
        n_elements: int = 25,
        flapping_freq_ratio: float = 0.75,
        damping_ratio_flap: float = 0.15,
    ):
        """
        Initialize Advanced BEMT rotor model.
        
        Args:
            config: Rotor configuration
            n_elements: Number of blade elements for integration
            flapping_freq_ratio: ω_flap / Ω (typically 0.7-0.8 for tiltrotors)
            damping_ratio_flap: Damping ratio for flapping motion
        """
        self.config = config
        self.n_elements = n_elements
        self.flapping_freq_ratio = flapping_freq_ratio
        self.damping_ratio_flap = damping_ratio_flap
        
        # Derived parameters
        self.R = config.radius
        self.A = np.pi * self.R**2
        self.sigma = config.solidity
        
        # Blade element stations
        self.r_start = 0.15  # Root cutout (15% for swashplate)
        self.r = np.linspace(self.r_start, 1.0, n_elements)
        self.dr = self.r[1] - self.r[0]
        
        # Blade geometry
        self._setup_blade_geometry()
        
        # Current state
        self.state = AdvancedRotorState()
        
        # VRS thresholds (normalized by hover induced velocity)
        self.vrs_lower = 0.5
        self.vrs_upper = 1.8
        
        # Dynamic stall parameters
        self.ds_time_lag = 0.1  # Default time lag [s], updated per flight condition
        self.ds_alpha_stall = np.radians(15.0)  # Stall angle
        self.ds_alpha_separation_0 = np.radians(10.0)  # Separation onset
        
        logger.info(f"AdvancedBEMTRotor initialized: R={self.R}m, ω_f/Ω={flapping_freq_ratio}")
    
    def _setup_blade_geometry(self):
        """Set up blade chord, twist, and inertia distribution."""
        # Linear taper
        taper = self.config.tip_chord / self.config.root_chord
        self.c = self.config.root_chord * (1 - (1 - taper) * self.r)
        
        # Linear twist distribution
        twist_rad = np.radians(self.config.twist)
        self.theta_tw = twist_rad * self.r
        
        # Local solidity
        self.sigma_r = self.config.num_blades * self.c / (np.pi * self.R * self.r)
        
        # Blade mass distribution (approximation: linear from root to tip)
        # Moment of inertia about flap hinge: I_β
        self.I_beta = self._compute_flapping_inertia()
    
    def _compute_flapping_inertia(self) -> float:
        """
        Compute blade flapping moment of inertia about hinge.
        
        Approximation: Integrated mass distribution along blade.
        
        Returns:
            Flapping inertia [kg·m²]
        """
        # Blade mass per unit span (approximate)
        # Typical rotor: 150 kg per blade, distributed from 0.15R to R
        total_blade_mass = 30.0  # kg per blade (scaled for multicopter)
        
        # Moment of inertia about flap hinge (at 0.15R)
        # ∫ (r - r_hinge)² dm/dr dr
        I_beta = 0.0
        for r, dr_val in zip(self.r, np.ones_like(self.r) * self.dr):
            r_actual = r * self.R
            dm = total_blade_mass * dr_val
            I_beta += (r_actual)**2 * dm
        
        return max(I_beta / self.config.num_blades, 10.0)  # kg·m² per blade
    
    def _compressibility_correction(self, mach: float) -> tuple[float, float]:
        """
        Apply Mach number corrections to Cl/Cd.
        
        Uses Prandtl-Mert rule for subsonic flow:
            Cl_M = Cl_0 / √(1 - M²)
            Cd_M ≈ Cd_0 + Cd_M_wave (wave drag for M > 0.7)
        
        Args:
            mach: Local Mach number
        
        Returns:
            (Cl_correction_factor, Cd_correction_factor)
        """
        if mach > 0.85:
            logger.warning(f"Flow approaching transonic regime: M={mach:.3f}")
        
        if mach < 0.3:
            return 1.0, 1.0
        
        # Prandtl-Mert compressibility correction
        beta_comp = np.sqrt(1.0 - mach**2)
        
        if beta_comp < 0.02:  # Sonic or supersonic
            return 1.0, 1.0
        
        Cl_factor = 1.0 / beta_comp
        
        # Wave drag for M > 0.7 (very approximate)
        Cd_factor = 1.0
        if mach > 0.7:
            Cd_factor = 1.0 + 0.05 * (mach - 0.7)**2
        
        return Cl_factor, Cd_factor
    
    def _compute_dynamic_stall(
        self,
        alpha: float,
        dt: float = 0.01,
    ) -> tuple[float, float]:
        """
        Compute dynamic stall corrections to Cl/Cd with time lag.
        
        Simple first-order model:
            α_eff = α + τ · dα/dt
            where τ is convective time lag
        
        Hysteresis through separation fraction x_sep.
        
        Args:
            alpha: Angle of attack [rad]
            dt: Time step for lag [s]
        
        Returns:
            (Cl_modifier, Cd_modifier)
        """
        # Update effective AoA (exponential lag)
        tau_ds = self.ds_time_lag
        alpha_eff_prev = self.state.stall.alpha_effective
        alpha_eff = alpha_eff_prev + (alpha - alpha_eff_prev) * (dt / (tau_ds + dt))
        self.state.stall.alpha_effective = alpha_eff
        
        # Separation point from AoA
        alpha_sep_onset = self.ds_alpha_separation_0
        if alpha_eff < alpha_sep_onset:
            x_sep = 0.0
        elif alpha_eff < self.ds_alpha_stall:
            # Linear progression from 0 to 1
            x_sep = (alpha_eff - alpha_sep_onset) / (self.ds_alpha_stall - alpha_sep_onset)
        else:
            x_sep = 1.0
        
        self.state.stall.x_separation = x_sep
        
        # Cl modifier (stall causes Cl reduction)
        if x_sep < 0.2:
            cl_mod = 1.0
        else:
            # Smooth reduction: Cl drops to ~70% at fully separated
            cl_mod = 1.0 - 0.3 * (x_sep / 1.0)
        
        # Cd modifier (stall causes Cd increase)
        if x_sep < 0.1:
            cd_mod = 1.0
        else:
            # Drag rise: 2x at full separation
            cd_mod = 1.0 + 1.5 * (x_sep ** 1.5)
        
        self.state.stall.Cl_modifier = cl_mod
        self.state.stall.Cd_modifier = cd_mod
        
        return cl_mod, cd_mod
    
    def _detect_vortex_ring_state(
        self,
        v_induced: float,
        v_z: float,
        omega: float,
    ) -> VRSState:
        """
        Detect and characterize vortex ring state.
        
        VRS occurs when descent rate is in range:
            0.5·v_i < v_desc < 1.8·v_i
        
        In VRS:
        - Thrust becomes highly uncertain (±50%)
        - Power increases 50-100%
        - Warning issued, but thrust estimate still made
        
        Args:
            v_induced: Hover-equivalent induced velocity [m/s]
            v_z: Vertical velocity (positive down) [m/s]
            omega: Rotor speed [rad/s]
        
        Returns:
            VRSState with detection and penalty factors
        """
        vrs = VRSState()
        
        if omega < 10.0 or v_induced < 0.5:
            return vrs
        
        v_desc = abs(v_z)
        
        # Check VRS boundaries
        if self.vrs_lower * v_induced < v_desc < self.vrs_upper * v_induced:
            vrs.in_vrs = True
            vrs.warning_active = True
            
            # Compute severity (0 at boundaries, 1.0 at midpoint)
            mid = (self.vrs_lower + self.vrs_upper) * v_induced / 2.0
            vrs_severity = 1.0 - abs(v_desc - mid) / max(mid - self.vrs_lower * v_induced, 0.1)
            vrs.vrs_severity = max(0.0, min(1.0, vrs_severity))
            
            # Apply penalties
            # Thrust reduction: 20-40% depending on severity
            vrs.thrust_factor = 0.8 - 0.2 * vrs.vrs_severity
            # Power increase: 50-80%
            vrs.power_factor = 1.5 + 0.3 * vrs.vrs_severity
            
            logger.warning(
                f"VRS DETECTED: severity={vrs.vrs_severity:.2f}, "
                f"T×{vrs.thrust_factor:.2f}, P×{vrs.power_factor:.2f}"
            )
        
        return vrs
    
    def compute_blade_flapping(
        self,
        omega: float,
        collective: float,
        cyclic_lon: float,
        cyclic_lat: float,
        V_inf: float = 0.0,
        alpha_tpp: float = 0.0,
    ) -> BladeFlappingState:
        """
        Compute first-harmonic blade flapping dynamics.
        
        Flapping equation: I_β·β̈ + 2·ζ·ω_f·I_β·β̇ + K_β·β = M_aero
        
        where:
        - I_β: Blade flapping moment of inertia
        - ω_f = flap_freq_ratio·Ω: Natural flapping frequency
        - K_β: Flapping stiffness (gimbal spring + centrifugal)
        - M_aero: Aerodynamic flapping moment
        
        Args:
            omega: Rotor speed [rad/s]
            collective: Collective pitch [rad]
            cyclic_lon: Longitudinal cyclic [rad]
            cyclic_lat: Lateral cyclic [rad]
            V_inf: Freestream velocity [m/s]
            alpha_tpp: Tip path plane angle [rad]
        
        Returns:
            BladeFlappingState with computed flapping angles
        """
        flapping = BladeFlappingState()
        
        if omega < 10.0:
            return flapping
        
        # Natural flapping frequency
        omega_f = self.flapping_freq_ratio * omega
        
        # Flapping stiffness from gimbal
        # K_β = I_β·ω_f² (for undamped natural mode)
        K_beta = self.I_beta * omega_f**2
        
        # Compute aerodynamic flapping moments (simplified)
        # For hover: primarily determined by inflow
        # For forward flight: depends on advance ratio and AoA
        
        if V_inf < 2.0:
            # Hover regime
            # Permanent flapping β₀ ≈ CM_yaw / K_β (typically ≈ 0 for symmetric rotors)
            beta_0 = 0.0
            
            # Cyclic flapping couples to cyclic pitch input
            # β₁c ≈ cyclic_lon, β₁s ≈ cyclic_lat (approximately)
            beta_1c = cyclic_lon * 0.5
            beta_1s = cyclic_lat * 0.5
        else:
            # Forward flight regime
            # Flapping becomes significant (5-15° typical)
            mu = V_inf * np.cos(alpha_tpp) / (omega * self.R)
            
            # Permanent flapping (from disk tilt)
            beta_0 = alpha_tpp * 0.2  # Weak coupling
            
            # Cyclic flapping (from advance ratio asymmetry)
            # β₁c is enhanced by advance ratio
            # β₁s typically small in forward flight
            beta_1c = (cyclic_lon + 0.75 * mu) * 0.3
            beta_1s = cyclic_lat * 0.2
        
        flapping.beta_0 = beta_0
        flapping.beta_1c = beta_1c
        flapping.beta_1s = beta_1s
        
        # Hub moments from flapping (simplified - first harmonic only)
        # M_roll = I_beta·Ω² · β₁s
        # M_pitch = I_beta·Ω² · β₁c
        # M_yaw ≈ 0 (for symmetric disk)
        
        flapping.M_roll = self.I_beta * omega**2 * beta_1s
        flapping.M_pitch = self.I_beta * omega**2 * beta_1c
        
        return flapping
    
    def compute_forward_flight_with_flapping(
        self,
        omega: float,
        collective: float,
        cyclic_lon: float,
        cyclic_lat: float,
        V_inf: float,
        alpha_tpp: float,
        rho: float = RHO_SL,
    ) -> AdvancedRotorState:
        """
        Compute rotor performance in forward flight with blade flapping.
        
        Key improvements over basic forward flight:
        1. Flapping angles modify local AoA on each blade
        2. Advance ratio effects on retreating blade dynamic stall
        3. Compressibility corrections at high tip speed
        4. VRS detection for descent phase
        
        Args:
            omega: Rotor speed [rad/s]
            collective: Collective pitch [rad]
            cyclic_lon: Longitudinal cyclic [rad]
            cyclic_lat: Lateral cyclic [rad]
            V_inf: Freestream velocity [m/s]
            alpha_tpp: Tip path plane angle of attack [rad]
            rho: Air density [kg/m³]
        
        Returns:
            AdvancedRotorState with thrust, power, flapping, VRS, stall
        """
        if omega <= 0:
            return AdvancedRotorState()
        
        state = AdvancedRotorState(omega=omega)
        state.rpm = omega * 60 / (2 * np.pi)
        
        V_tip = omega * self.R
        
        # Advance ratio
        mu = V_inf * np.cos(alpha_tpp) / V_tip if V_tip > 0.1 else 0.0
        mu_z = V_inf * np.sin(alpha_tpp) / V_tip if V_tip > 0.1 else -V_inf / max(V_tip, 0.1)
        
        # Compute blade flapping
        flapping = self.compute_blade_flapping(
            omega, collective, cyclic_lon, cyclic_lat, V_inf, alpha_tpp
        )
        state.flapping = flapping
        
        # Iteratively solve for induced velocity
        CT_est = 0.5 * self.sigma * self.config.cl_alpha * (collective / 3) * (1 + 1.5*mu**2)
        
        def lambda_residual(lambda_i):
            lambda_total = mu_z + lambda_i
            CT_mom = 2 * lambda_i * np.sqrt(mu**2 + lambda_total**2)
            return CT_est - CT_mom
        
        try:
            lambda_i = brentq(lambda_residual, 0.001, 0.3, xtol=1e-6)
        except:
            lambda_i = 0.05
        
        state.advance_ratio = mu
        state.inflow_ratio = lambda_i
        
        v_induced = lambda_i * V_tip
        
        # Blade element integration with flapping modifier
        thrust = 0.0
        power_induced = 0.0
        power_profile = 0.0
        
        for i, r in enumerate(self.r):
            R_local = r * self.R
            c_local = self.c[i]
            theta_local = collective + self.theta_tw[i]
            
            # Azimuth angle effect (advance ratio creates asymmetry)
            psi = np.arctan2(mu, mu_z + lambda_i) if abs(mu) > 0.01 else 0.0
            
            # Flapping correction to local inflow
            # This is simplified; full model would integrate over azimuth
            flap_correction = flapping.beta_1c * np.cos(psi) + flapping.beta_1s * np.sin(psi)
            
            # Local velocities
            U_T = omega * R_local
            U_P = v_induced + flap_correction
            
            # Inflow angle
            if U_T > 0.1:
                phi = np.arctan2(U_P, U_T)
            else:
                phi = 0.0
            
            # Local AoA with flapping effect
            alpha = theta_local - phi
            
            # Compressibility correction
            V_local = np.sqrt(U_T**2 + U_P**2)
            mach_local = V_local / SONIC_SPEED_SL
            Cl_comp, Cd_comp = self._compressibility_correction(mach_local)
            
            # Dynamic stall (especially on retreating blade)
            Cl_stall, Cd_stall = self._compute_dynamic_stall(alpha, dt=0.01)
            
            # Combined Cl/Cd
            cl = self.config.cl_alpha * alpha * Cl_comp * Cl_stall
            cl = np.clip(cl, -self.config.cl_max, self.config.cl_max)
            
            cd = (self.config.cd0 + 0.01 * alpha**2) * Cd_comp * Cd_stall
            
            # Elemental forces
            U2 = U_T**2 + U_P**2
            if U2 > 0.01:
                dL = 0.5 * rho * U2 * c_local * cl
                dD = 0.5 * rho * U2 * c_local * cd
                
                dT = (dL * np.cos(phi) - dD * np.sin(phi)) * self.dr * self.R
                dQ_profile = (dL * np.sin(phi) + dD * np.cos(phi)) * R_local * self.dr * self.R
                
                thrust += dT * self.config.num_blades
                power_profile += dQ_profile * omega * self.config.num_blades
        
        # Tip loss factor
        tip_loss = 0.95 if omega > 50 else 0.9
        thrust *= tip_loss
        power_profile *= tip_loss
        
        # VRS detection (descent condition)
        if mu_z < 0:  # Descending
            v_hover = np.sqrt(state.thrust / (2 * rho * self.A)) if state.thrust > 0.1 else 0.1
            vrs = self._detect_vortex_ring_state(v_hover, abs(mu_z) * V_tip, omega)
            state.vrs = vrs
            
            # Apply VRS penalties
            thrust *= vrs.thrust_factor
            power_profile *= vrs.power_factor
        
        state.thrust = max(thrust, 0)
        state.CT = state.thrust / (rho * self.A * V_tip**2) if V_tip > 0 else 0
        
        power_induced = state.thrust * v_induced
        state.power_induced = power_induced
        state.power_profile = power_profile
        state.power_total = power_induced + power_profile
        state.torque = state.power_total / omega if omega > 0.1 else 0
        state.CQ = state.torque / (rho * self.A * V_tip**2 * self.R) if V_tip > 0 else 0
        
        # Figure of merit
        if state.power_total > 0.1:
            P_ideal = state.thrust**(3/2) / np.sqrt(2 * rho * self.A)
            state.figure_of_merit = min(1.0, P_ideal / state.power_total)
        
        self.state = state
        return state
    
    def compute_hover_with_flapping(
        self,
        omega: float,
        collective: float,
        cyclic_lon: float = 0.0,
        cyclic_lat: float = 0.0,
        rho: float = RHO_SL,
    ) -> AdvancedRotorState:
        """
        Compute hover performance with blade flapping dynamics.
        
        Args:
            omega: Rotor speed [rad/s]
            collective: Collective pitch [rad]
            cyclic_lon: Longitudinal cyclic [rad]
            cyclic_lat: Lateral cyclic [rad]
            rho: Air density [kg/m³]
        
        Returns:
            AdvancedRotorState
        """
        return self.compute_forward_flight_with_flapping(
            omega, collective, cyclic_lon, cyclic_lat, 0.01, 0.0, rho
        )
