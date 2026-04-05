"""
Radar Cross-Section (RCS) Model

This module models the radar cross-section of a tiltrotor eVTOL
as a function of aspect angle, frequency, and rotor state.

Theory
======

RCS (σ) is the effective area that returns radar energy:
    σ = 4π × |S|² / |E_i|²

where S is the scattered field and E_i is the incident field.

For complex targets like aircraft, RCS depends on:
1. Aspect angle (azimuth φ, elevation θ)
2. Frequency (wavelength λ)
3. Polarization
4. Target motion (Doppler)

Components
==========

1. Body RCS (fuselage, wing, nacelles):
    σ_body(θ, φ) - Angular-dependent from geometry

2. Rotor RCS (blade flash):
    σ_rotor(t) = σ_blade × N_blades × |sin(Ω×t)|²
    Modulated at blade passage frequency

3. Engine inlet/exhaust (cavity returns):
    σ_cavity - High RCS from internal reflections

Reduction Techniques:
- Shaping (minimize flat surfaces)
- RAM (radar absorbing materials)
- Rotor blade design (swept tips, composite)

Frequencies of Interest:
- VHF (30-300 MHz): Long-range search
- UHF (300-1000 MHz): Acquisition
- L-band (1-2 GHz): Air surveillance
- S-band (2-4 GHz): Target tracking
- X-band (8-12 GHz): Fire control

Author: Defense eVTOL Research Team
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class RadarBand(Enum):
    """Radar frequency bands."""
    VHF = "VHF"      # 30-300 MHz
    UHF = "UHF"      # 300-1000 MHz
    L_BAND = "L"     # 1-2 GHz
    S_BAND = "S"     # 2-4 GHz
    C_BAND = "C"     # 4-8 GHz
    X_BAND = "X"     # 8-12 GHz
    KU_BAND = "Ku"   # 12-18 GHz
    KA_BAND = "Ka"   # 26-40 GHz


# Band center frequencies (Hz)
BAND_FREQUENCIES = {
    RadarBand.VHF: 100e6,
    RadarBand.UHF: 500e6,
    RadarBand.L_BAND: 1.5e9,
    RadarBand.S_BAND: 3.0e9,
    RadarBand.C_BAND: 6.0e9,
    RadarBand.X_BAND: 10.0e9,
    RadarBand.KU_BAND: 15.0e9,
    RadarBand.KA_BAND: 35.0e9,
}


@dataclass
class RCSState:
    """RCS model state."""
    # Total RCS by band (m²)
    rcs_total: dict[RadarBand, float] = field(default_factory=dict)

    # Component contributions (m²)
    rcs_body: float = 0.0
    rcs_wing: float = 0.0
    rcs_nacelle: float = 0.0
    rcs_rotor: float = 0.0
    rcs_cavity: float = 0.0

    # Rotor modulation
    rotor_phase: float = 0.0        # Current rotor phase (rad)
    blade_flash_active: bool = False

    # Aspect angles (deg)
    azimuth_deg: float = 0.0        # 0 = nose-on
    elevation_deg: float = 0.0       # 0 = level

    # dBsm values (for convenience)
    rcs_dbsm: dict[RadarBand, float] = field(default_factory=dict)


@dataclass
class RCSConfig:
    """RCS model configuration."""
    # Reference RCS values at X-band, broadside (m²)
    # These represent a stealthy design baseline
    rcs_body_reference: float = 0.1      # m² (small body)
    rcs_wing_reference: float = 0.5      # m² (thin wing)
    rcs_nacelle_reference: float = 0.2   # m² (streamlined)
    rcs_rotor_blade_reference: float = 0.01  # m² per blade
    rcs_cavity_reference: float = 0.3    # m² (inlet/exhaust)

    # Number of rotor blades per rotor
    num_blades: int = 4
    num_rotors: int = 2

    # RAM (Radar Absorbing Material) effectiveness
    ram_reduction_db: float = 10.0  # dB reduction from materials

    # Shaping effectiveness (angular variation)
    shaping_factor: float = 0.5  # 0=no shaping, 1=perfect shaping

    # Aspect angle sensitivity
    # RCS varies roughly as cos²(θ) for flat surfaces
    frontal_reduction_db: float = 15.0  # Nose-on reduction vs broadside


class RCSModel:
    """
    Radar Cross-Section model for tiltrotor eVTOL.

    This model computes RCS as a function of:
    - Aspect angle (azimuth and elevation)
    - Radar frequency band
    - Rotor state (RPM for blade flash)
    - Nacelle angle (changes exposed geometry)

    The model is suitable for:
    - Detection probability analysis
    - Trajectory optimization for minimum observability
    - Threat avoidance planning
    """

    def __init__(self, config: RCSConfig | None = None):
        """
        Initialize RCS model.

        Args:
            config: RCS configuration
        """
        self.config = config or RCSConfig()
        self.state = RCSState()

        # Pre-compute band-dependent scaling factors
        self._setup_frequency_scaling()

        # Initialize RCS tables
        self._setup_aspect_tables()

        logger.info("RCS model initialized")

    def _setup_frequency_scaling(self) -> None:
        """Set up frequency-dependent scaling."""
        # Rayleigh region (λ >> target size): σ ∝ 1/λ⁴
        # Resonance region (λ ~ target size): σ oscillates
        # Optical region (λ << target size): σ ~ constant

        # Reference wavelength (X-band, 3cm)
        lambda_ref = 0.03  # meters

        self._freq_scale = {}
        for band in RadarBand:
            freq = BAND_FREQUENCIES[band]
            lambda_band = 3e8 / freq  # c / f

            # Approximate scaling (simplified)
            if lambda_band > 0.3:  # VHF/UHF - Rayleigh-ish
                scale = (lambda_ref / lambda_band) ** 2
            elif lambda_band > 0.03:  # L/S/C - Resonance region
                scale = 1.0 + 0.5 * np.sin(2 * np.pi * lambda_band / 0.1)
            else:  # X/Ku/Ka - Optical region
                scale = (lambda_band / lambda_ref) ** 0.5

            self._freq_scale[band] = np.clip(scale, 0.1, 10.0)

    def _setup_aspect_tables(self) -> None:
        """Set up aspect angle RCS patterns."""
        # Azimuth angles (0 = nose, 90 = broadside, 180 = tail)
        self._azimuth_angles = np.linspace(0, 360, 361)

        # Elevation angles (-90 = below, 0 = level, 90 = above)
        self._elevation_angles = np.linspace(-90, 90, 181)

        # Body RCS pattern (egg-shaped)
        # Minimum at nose/tail, maximum at broadside
        az_rad = np.radians(self._azimuth_angles)
        self._body_pattern = (
            0.3 + 0.7 * np.abs(np.sin(az_rad))  # Nose=0.3, side=1.0
        )

        # Wing RCS pattern
        # Minimum edge-on, maximum flat-on (top/bottom look)
        self._wing_az_pattern = 1.0 - 0.8 * np.abs(np.cos(az_rad))  # Side=1, nose=0.2

    def compute_rcs(
        self,
        azimuth_deg: float,
        elevation_deg: float,
        rotor_rpm: float = 0.0,
        nacelle_angle_deg: float = 0.0,
        time: float = 0.0,
    ) -> RCSState:
        """
        Compute RCS for given aspect and operating state.

        Args:
            azimuth_deg: Azimuth angle (0 = nose-on, 90 = port side)
            elevation_deg: Elevation angle (0 = level, positive = above)
            rotor_rpm: Rotor RPM (for blade flash computation)
            nacelle_angle_deg: Nacelle tilt angle (90 = hover, 0 = cruise)
            time: Current time (for rotor phase)

        Returns:
            Updated RCS state
        """
        # Normalize angles
        azimuth_deg = azimuth_deg % 360
        elevation_deg = np.clip(elevation_deg, -90, 90)

        self.state.azimuth_deg = azimuth_deg
        self.state.elevation_deg = elevation_deg

        # Compute component RCS at X-band (reference)
        rcs_body = self._compute_body_rcs(azimuth_deg, elevation_deg)
        rcs_wing = self._compute_wing_rcs(azimuth_deg, elevation_deg)
        rcs_nacelle = self._compute_nacelle_rcs(azimuth_deg, elevation_deg, nacelle_angle_deg)
        rcs_rotor = self._compute_rotor_rcs(azimuth_deg, rotor_rpm, nacelle_angle_deg, time)
        rcs_cavity = self._compute_cavity_rcs(azimuth_deg, elevation_deg)

        # Apply RAM reduction
        ram_factor = 10 ** (-self.config.ram_reduction_db / 10)
        rcs_body *= ram_factor
        rcs_wing *= ram_factor
        rcs_nacelle *= ram_factor
        # Rotor blades and cavities harder to treat
        rcs_rotor *= ram_factor * 2
        rcs_cavity *= ram_factor * 3

        # Store component values
        self.state.rcs_body = rcs_body
        self.state.rcs_wing = rcs_wing
        self.state.rcs_nacelle = rcs_nacelle
        self.state.rcs_rotor = rcs_rotor
        self.state.rcs_cavity = rcs_cavity

        # Compute total for each band
        rcs_x_band = rcs_body + rcs_wing + rcs_nacelle + rcs_rotor + rcs_cavity

        for band in RadarBand:
            scale = self._freq_scale[band]
            rcs_total = rcs_x_band * scale
            self.state.rcs_total[band] = rcs_total

            # Convert to dBsm
            if rcs_total > 0:
                self.state.rcs_dbsm[band] = 10 * np.log10(rcs_total)
            else:
                self.state.rcs_dbsm[band] = -100  # Essentially zero

        return self.state

    def _compute_body_rcs(self, azimuth_deg: float, elevation_deg: float) -> float:
        """Compute fuselage body RCS."""
        az_rad = np.radians(azimuth_deg)
        el_rad = np.radians(elevation_deg)

        # Basic pattern (broadside maximum)
        horizontal_factor = 0.3 + 0.7 * np.abs(np.sin(az_rad))

        # Elevation effect (higher RCS from below due to belly)
        if elevation_deg < 0:
            vertical_factor = 1.0 + 0.3 * np.abs(np.sin(el_rad))  # Below
        else:
            vertical_factor = 0.8 + 0.2 * np.abs(np.sin(el_rad))  # Above

        # Shaping reduction at nose-on
        if abs(azimuth_deg) < 30 or abs(azimuth_deg - 180) < 30:
            shaping = 10 ** (-self.config.frontal_reduction_db / 10)
        else:
            shaping = 1.0

        return self.config.rcs_body_reference * horizontal_factor * vertical_factor * shaping

    def _compute_wing_rcs(self, azimuth_deg: float, elevation_deg: float) -> float:
        """Compute wing RCS."""
        az_rad = np.radians(azimuth_deg)
        el_rad = np.radians(elevation_deg)

        # Wing is thin - low RCS edge-on, high from above/below
        edge_on = np.abs(np.cos(az_rad))  # 1 at nose, 0 at side
        horizontal_factor = 1.0 - 0.9 * edge_on

        # From above/below, wing presents large flat surface
        vertical_factor = 1.0 + 2.0 * np.abs(np.sin(el_rad))

        return self.config.rcs_wing_reference * horizontal_factor * vertical_factor

    def _compute_nacelle_rcs(
        self,
        azimuth_deg: float,
        elevation_deg: float,
        nacelle_angle_deg: float,
    ) -> float:
        """Compute nacelle RCS (both nacelles)."""
        az_rad = np.radians(azimuth_deg)
        nac_rad = np.radians(nacelle_angle_deg)

        # Nacelles present different aspects depending on tilt
        # In hover (90°), nacelles point up - visible from above
        # In cruise (0°), nacelles point forward - visible from front

        # Effective aspect to nacelle
        # When looking from side, always see nacelle side
        side_factor = np.abs(np.sin(az_rad))

        # When looking from front/back, depends on nacelle angle
        front_factor = np.abs(np.cos(az_rad)) * np.cos(nac_rad)

        # When looking from above/below, depends on nacelle angle
        el_rad = np.radians(elevation_deg)
        vertical_factor = np.abs(np.sin(el_rad)) * np.sin(nac_rad)

        total_factor = side_factor + front_factor + vertical_factor

        # Two nacelles
        return 2 * self.config.rcs_nacelle_reference * np.clip(total_factor, 0.1, 3.0)

    def _compute_rotor_rcs(
        self,
        azimuth_deg: float,
        rotor_rpm: float,
        nacelle_angle_deg: float,
        time: float,
    ) -> float:
        """
        Compute rotor blade flash RCS.

        Rotor blades create periodic RCS spikes when a blade
        is perpendicular to the radar line of sight.
        """
        if rotor_rpm <= 0:
            return 0.0

        # Rotor phase
        omega = rotor_rpm * 2 * np.pi / 60  # rad/s
        phase = omega * time
        self.state.rotor_phase = phase % (2 * np.pi)

        # Blade flash occurs when blade is broadside to radar
        # This happens N_blades times per revolution
        blade_period = 2 * np.pi / self.config.num_blades
        blade_phase = (phase % blade_period) / blade_period * 2 * np.pi

        # Flash intensity (sinusoidal model)
        flash_intensity = np.abs(np.sin(blade_phase))

        # Blade visibility depends on aspect angle and nacelle position
        az_rad = np.radians(azimuth_deg)
        nac_rad = np.radians(nacelle_angle_deg)

        # Rotor disk visibility
        # In hover, disk is horizontal - visible from above
        # In cruise, disk is vertical - visible from front
        disk_visibility = (
            np.abs(np.sin(nac_rad)) * 0.5 +  # Horizontal disk contribution
            np.abs(np.cos(nac_rad)) * np.abs(np.cos(az_rad)) * 0.5  # Vertical disk
        )

        # Peak RCS per blade
        rcs_blade = self.config.rcs_rotor_blade_reference

        # Total (both rotors, all blades contributing at different phases)
        rcs_rotor = (
            self.config.num_rotors *
            self.config.num_blades *
            rcs_blade *
            flash_intensity *
            disk_visibility
        )

        self.state.blade_flash_active = flash_intensity > 0.7

        return rcs_rotor

    def _compute_cavity_rcs(self, azimuth_deg: float, elevation_deg: float) -> float:
        """Compute engine inlet/exhaust cavity RCS."""
        az_rad = np.radians(azimuth_deg)

        # Cavities visible primarily from front and rear
        front_visible = np.abs(np.cos(az_rad))

        # Front aspect (inlet) has higher RCS
        if abs(azimuth_deg) < 90 or abs(azimuth_deg) > 270:
            # Looking at front
            cavity_factor = front_visible * 1.5
        else:
            # Looking at rear (exhaust)
            cavity_factor = front_visible * 1.0

        return self.config.rcs_cavity_reference * cavity_factor

    def get_average_rcs(self, band: RadarBand = RadarBand.X_BAND) -> float:
        """Get average RCS across all aspects for given band."""
        # Sample multiple aspects
        total = 0.0
        count = 0

        for az in range(0, 360, 10):
            for el in range(-60, 61, 30):
                self.compute_rcs(az, el)
                total += self.state.rcs_total.get(band, 0.0)
                count += 1

        return total / count if count > 0 else 0.0

    def get_minimum_rcs_aspect(
        self,
        band: RadarBand = RadarBand.X_BAND,
        rotor_rpm: float = 0.0,
    ) -> tuple[float, float, float]:
        """
        Find aspect angle with minimum RCS.

        Useful for planning ingress routes.

        Returns:
            (azimuth_deg, elevation_deg, rcs_m2)
        """
        min_rcs = float('inf')
        min_az = 0
        min_el = 0

        for az in range(0, 360, 5):
            for el in range(-60, 61, 10):
                self.compute_rcs(az, el, rotor_rpm)
                rcs = self.state.rcs_total.get(band, float('inf'))
                if rcs < min_rcs:
                    min_rcs = rcs
                    min_az = az
                    min_el = el

        return float(min_az), float(min_el), min_rcs

    def get_state(self) -> RCSState:
        """Get current RCS state."""
        return self.state

    def to_dict(self) -> dict:
        """Convert state to dictionary."""
        return {
            'azimuth_deg': self.state.azimuth_deg,
            'elevation_deg': self.state.elevation_deg,
            'rcs_total_m2': {k.value: v for k, v in self.state.rcs_total.items()},
            'rcs_dbsm': {k.value: v for k, v in self.state.rcs_dbsm.items()},
            'rcs_body': self.state.rcs_body,
            'rcs_wing': self.state.rcs_wing,
            'rcs_rotor': self.state.rcs_rotor,
            'blade_flash_active': self.state.blade_flash_active,
        }
