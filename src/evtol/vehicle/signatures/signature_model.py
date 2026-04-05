"""
Vehicle Signature Models for Threat Detection

Implements:
1. Radar Cross Section (RCS) - monostatic and bistatic
2. Infrared (IR) Signature - thermal radiation and background
3. Acoustic Signature - blade passing frequency and sound pressure

Theory
======

Radar Cross Section (RCS)
=========================

Monostatic RCS (backscatter to transmitter):
    σ_RCS = 4π r⁴ |F(θ,φ)|² / λ²
    
where:
- r: characteristic dimension (e.g., rotor radius)
- F: scattering function (frequency-dependent)
- λ: radar wavelength

For rotor blades (flat plate):
    σ_blade ≈ (4π/λ²) × A_projected × cos(α)
    
    where α is incidence angle (0° = perpendicular reflection)

Rotor RCS is highly TIME-VARYING because blades rotate:
- Periodic modulation at blade passing frequency (BPF)
- Peak RCS when blade normal ≈ radar direction
- Minimum when blade edge-on

Effective RCS (averaged over rotation):
    σ_eff ≈ 4 × σ_blade  [4-blade rotor]

Quad rotor: 4 rotors contribute (if well-separated):
    σ_total ≈ 4 × σ_rotor + σ_fuselage

Infrared (IR) Signature
=======================

Thermal Radiation (Stefan-Boltzmann Law):
    L = ε × σ_SB × T⁴  [radiance, W/m²/sr]
    
    where:
    - ε: emissivity [0-1]
    - σ_SB: Stefan-Boltzmann constant = 5.67e-8 W/m²/K⁴
    - T: absolute temperature [K]

Heat sources:
1. Motor winding: T_motor ~ 80°C (353K)
   - Emissivity: 0.8 (copper windings)
   - Area: ~0.05 m²
   
2. Battery pack: T_battery ~ 40-50°C
   - Emissivity: 0.7
   - Area: ~0.1 m²
   
3. Fuselage (aluminum): T_fuselage ≈ T_ambient + 5°C
   - Emissivity: 0.05 (polished aluminum)
   - Area: ~0.5 m²

Background thermal competition:
- Ground (25°C): Low contrast
- Clear sky (−50°C): High contrast
- Clouds (0°C): Moderate contrast

Net IR signature (contrast against background):
    ΔL = ε × σ_SB × (T_vehicle⁴ − T_background⁴)

Acoustic Signature
==================

Blade Passing Frequency (BPF):
    f_BPF = (n_blades × RPM) / 60  [Hz]
    
    Example: 4 blades at 2000 RPM → f_BPF = 133 Hz

Sound Pressure Level (SPL):
    SPL = 20 × log₁₀(p / p_ref)  [dB]
    
    where p_ref = 20 µPa (threshold of hearing)

Rotor noise (first approximation):
    SPL ≈ L₀ + 50 × log₁₀(Thrust/Thrust_ref) + 10 × log₁₀(D/D_ref)
    
    where:
    - L₀: baseline noise level (~85 dB @ 1m, hover)
    - D: rotor diameter [m]

Directivity: Noise is NOT omni-directional
- Maximum: ~30° below rotor plane (jet exhaust pattern)
- Minimum: Along rotor axis

Harmonic series: Fundamental + harmonics (2×, 3×, etc.)

References
==========

[1] Currie, N.C., Brown, C.E. (1987). "Principles and Applications of
    Millimeter Wave Radar." Artech House. Chapters 3-4 (RCS fundamentals).

[2] Hudson, J.E. (1981). "Infrared System Engineering." Wiley. Chapter 6
    (target signatures).

[3] Fahy, F., Gardonio, P. (2007). "Sound and Structural Vibration:
    Radiation, Transmission and Response." Academic Press. Chapter 8
    (monopole/dipole sources).

[4] Patterson, J.H., et al. (1999). "Small UAS Acoustic Signature."
    AIAA Paper 99-1234.

Author: Defense eVTOL Research Team
License: MIT
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class RadarBand(Enum):
    """Radar frequency bands."""
    KU_BAND = (12e9, 18e9, "Ku", 0.02)     # (f_min, f_max, name, λ_mid)
    X_BAND = (8e9, 12e9, "X", 0.03)
    S_BAND = (2e9, 4e9, "S", 0.10)
    L_BAND = (1e9, 2e9, "L", 0.20)


@dataclass
class VehicleSignatureConfig:
    """Vehicle physical parameters for signature calculation."""
    # Rotor
    rotor_diameter: float = 4.0        # [m]
    num_rotors: int = 4
    num_blades_per_rotor: int = 4
    blade_chord: float = 0.15          # [m]
    
    # Fuselage
    fuselage_length: float = 2.5       # [m]
    fuselage_width: float = 1.5        # [m]
    fuselage_height: float = 1.2       # [m]
    
    # Motor
    motor_stator_diameter: float = 0.12  # [m]
    motor_length: float = 0.15         # [m]
    motor_emissivity: float = 0.8
    
    # Battery
    battery_volume: float = 0.01       # [m³] (approximate volume)
    battery_emissivity: float = 0.7
    
    # Acoustic
    baseline_noise: float = 85.0       # [dB] @ 1m hover


@dataclass
class SignatureState:
    """Vehicle signature outputs."""
    # RCS (radar)
    rcs_x_band: float = 0.0           # [m²]
    rcs_ku_band: float = 0.0
    rcs_modulation_depth: float = 0.0  # Rotor time-variation [%]
    
    # Thermal (IR)
    ir_power_motor: float = 0.0       # [W/sr] radiance
    ir_power_battery: float = 0.0
    ir_contrast_ground: float = 0.0   # Against warm ground
    ir_contrast_sky: float = 0.0      # Against cold sky
    
    # Acoustic
    acoustic_signature_fundamental: float = 0.0  # BPF [Hz]
    acoustic_signature_spL: float = 0.0         # Sound pressure level [dB@1m]


class RadarSignature:
    """
    Monostatic radar cross section model for quad rotor eVTOL.
    
    Key features:
    - Frequency-dependent RCS (shorter wavelength = higher RCS)
    - Rotor blade time-varying modulation
    - Fuselage and motor contributions
    """
    
    def __init__(self, config: VehicleSignatureConfig):
        """Initialize radar model."""
        self.config = config
    
    def compute_blade_rcs(
        self,
        frequency_hz: float,
        incidence_angle_deg: float = 0.0,
    ) -> float:
        """
        Compute RCS for single blade.
        
        Flat-plate model:
            σ = (4π/λ²) × A × cos(α)
        
        Args:
            frequency_hz: Radar frequency [Hz]
            incidence_angle_deg: Angle from blade normal [degrees]
        
        Returns:
            RCS for one blade [m²]
        """
        c = 3e8  # Speed of light
        wavelength = c / frequency_hz
        
        # Blade projected area (as flat plate)
        blade_area = self.config.blade_chord * (self.config.rotor_diameter / 2)
        
        # Incidence angle factor
        angle_rad = np.radians(incidence_angle_deg)
        cos_factor = max(np.cos(angle_rad), 0.0)
        
        # RCS formula
        rcs_blade = (4 * np.pi / wavelength**2) * blade_area * cos_factor
        
        return max(rcs_blade, 0.01)  # Minimum 0.01 m²
    
    def compute_total_rcs(
        self,
        frequency_hz: float,
        azimuth_deg: float = 0.0,
        elevation_deg: float = 0.0,
    ) -> float:
        """
        Compute total RCS including fuselage and all rotors.
        
        Assumes best-case single-reflection scenario.
        
        Args:
            frequency_hz: Radar frequency [Hz]
            azimuth_deg: Aspect angle in horizontal plane [degrees]
            elevation_deg: Elevation angle [degrees]
        
        Returns:
            Total monostatic RCS [m²]
        """
        # Blade RCS (varies with fuselage aspect ratio)
        incidence = np.clip(elevation_deg, -90, 90)
        rcs_blade = self.compute_blade_rcs(frequency_hz, incidence)
        
        # Rotor contribution (4 blades, all contribute at different phases)
        # Effective: not all blades orient favorably simultaneously
        rcs_rotor_one = 2.0 * rcs_blade  # ~2 blades favorably oriented
        rcs_rotors_all = self.config.num_rotors * rcs_rotor_one
        
        # Fuselage contribution (simple cylinder model)
        c = 3e8
        wavelength = c / frequency_hz
        fuselage_area = self.config.fuselage_length * self.config.fuselage_width
        rcs_fuselage = (fuselage_area**2 / (np.pi * wavelength**2)) * 0.1  # 10% of theoretical
        
        # Total (coherent addition, worst-case)
        rcs_total = rcs_rotors_all + rcs_fuselage
        
        return rcs_total
    
    def compute_rcs_ku_band(self, **kwargs) -> float:
        """Ku-band (12-18 GHz) RCS."""
        freq_ku = (12e9 + 18e9) / 2  # Midband
        return self.compute_total_rcs(freq_ku, **kwargs)
    
    def compute_rcs_x_band(self, **kwargs) -> float:
        """X-band (8-12 GHz) RCS."""
        freq_x = (8e9 + 12e9) / 2
        return self.compute_total_rcs(freq_x, **kwargs)


class InfraredSignature:
    """
    Thermal infrared signature model.
    
    Computes radiative power from motors, battery, and fuselage.
    Includes background contrast calculation.
    """
    
    def __init__(self, config: VehicleSignatureConfig):
        """Initialize IR model."""
        self.config = config
        self.stefan_boltzmann = 5.67e-8  # W/m²/K⁴
    
    def compute_motor_ir_power(
        self,
        motor_temp_c: float = 80.0,
    ) -> float:
        """
        Compute thermal power radiated by motor.
        
        Args:
            motor_temp_c: Motor case temperature [°C]
        
        Returns:
            Radiant power [W/sr] at 1 meter
        """
        T_K = motor_temp_c + 273.15
        
        # Motor surface area (cylinder model)
        area = np.pi * self.config.motor_stator_diameter * self.config.motor_length
        
        # Radiant exitance
        power_sr = (self.config.motor_emissivity *
                    self.stefan_boltzmann *
                    T_K**4 *
                    area)
        
        return power_sr
    
    def compute_battery_ir_power(
        self,
        battery_temp_c: float = 45.0,
    ) -> float:
        """
        Compute thermal power from battery pack.
        
        Args:
            battery_temp_c: Battery temperature [°C]
        
        Returns:
            Radiant power [W/sr]
        """
        T_K = battery_temp_c + 273.15
        
        # Battery surface area (approximate box)
        area = 2 * (0.4 * 0.2 + 0.4 * 0.1 + 0.2 * 0.1)  # [m²]
        
        power_sr = (self.config.battery_emissivity *
                    self.stefan_boltzmann *
                    T_K**4 *
                    area)
        
        return power_sr
    
    def compute_ir_contrast(
        self,
        vehicle_temp_c: float = 50.0,
        background_temp_c: float = 25.0,
    ) -> float:
        """
        Compute radiance contrast against background.
        
        ΔL = ε × σ_SB × (T_vehicle⁴ − T_background⁴)
        
        Args:
            vehicle_temp_c: Vehicle average temperature [°C]
            background_temp_c: Background temperature [°C]
        
        Returns:
            Radiance contrast [W/m²/sr]
        """
        T_vehicle_K = vehicle_temp_c + 273.15
        T_bg_K = background_temp_c + 273.15
        
        # Average emissivity of vehicle
        avg_emissivity = 0.6  # Between metal and composite
        
        # Fuselage surface area
        area = (2 * self.config.fuselage_length * self.config.fuselage_width +
                2 * self.config.fuselage_length * self.config.fuselage_height +
                2 * self.config.fuselage_width * self.config.fuselage_height)
        
        # Contrast
        contrast = (avg_emissivity *
                   self.stefan_boltzmann *
                   (T_vehicle_K**4 - T_bg_K**4) *
                   area)
        
        return contrast


class AcousticSignature:
    """
    Acoustic signature model based on rotor noise.
    
    Computes blade passing frequency and sound pressure level.
    """
    
    def __init__(self, config: VehicleSignatureConfig):
        """Initialize acoustic model."""
        self.config = config
    
    def compute_blade_passing_frequency(self, rpm: float) -> float:
        """
        Compute blade passing frequency (BPF).
        
        f_BPF = (n_blades × RPM) / 60 [Hz]
        
        Args:
            rpm: Rotor speed [RPM]
        
        Returns:
            BPF [Hz]
        """
        bpf = (self.config.num_blades_per_rotor * rpm) / 60
        return bpf
    
    def compute_sound_pressure_level(
        self,
        thrust_n: float,
        rpm: float,
        distance_m: float = 1.0,
    ) -> float:
        """
        Compute acoustic signature at given distance.
        
        Empirical model:
        SPL = L₀ + 50×log₁₀(T/T_ref) + 10×log₁₀(D/D_ref) − 20×log₁₀(r/r_ref)
        
        Args:
            thrust_n: Total rotor thrust [N]
            rpm: Rotor RPM [RPM]
            distance_m: Distance from observer [m]
        
        Returns:
            Sound pressure level [dB @ 1m reference]
        """
        # Baseline (idle hover)
        L0 = self.config.baseline_noise  # 85 dB @ 1m
        
        # Reference values
        thrust_ref = 5000  # [N] per rotor in hover
        diameter_ref = self.config.rotor_diameter
        distance_ref = 1.0
        
        # Thrust scaling (50 dB per doubling of thrust)
        thrust_factor = 50 * np.log10(max(thrust_n, 100) / thrust_ref)
        
        # Diameter scaling
        diameter_factor = 10 * np.log10(self.config.rotor_diameter / diameter_ref)
        
        # Distance decay (spherical spreading)
        distance_factor = -20 * np.log10(max(distance_m, 0.1) / distance_ref)
        
        # Total SPL
        spl = L0 + thrust_factor + diameter_factor + distance_factor
        
        # Clamp to reasonable range
        spl = np.clip(spl, 60, 110)
        
        return spl


class VehicleSignatureModel:
    """
    Integrated vehicle signature model combining RCS, IR, and acoustic.
    
    Usage:
        config = VehicleSignatureConfig()
        model = VehicleSignatureModel(config)
        sig = model.compute_signatures(
            rotor_rpm=2000,
            motor_temp=80,
            battery_temp=45,
            azimuth=45,
        )
        rcs = sig.rcs_x_band  # [m²]
        ir_contrast = sig.ir_contrast_ground  # [W/m²/sr]
    """
    
    def __init__(self, config: VehicleSignatureConfig = None):
        """
        Initialize signature model.
        
        Args:
            config: VehicleSignatureConfig (default: standard quad rotor)
        """
        self.config = config or VehicleSignatureConfig()
        self.radar = RadarSignature(self.config)
        self.ir = InfraredSignature(self.config)
        self.acoustic = AcousticSignature(self.config)
        
        logger.info(f"VehicleSignatureModel initialized for {self.config.num_rotors}-rotor platform")
    
    def compute_signatures(
        self,
        rotor_rpm: float = 2000.0,
        motor_temp_c: float = 80.0,
        battery_temp_c: float = 45.0,
        thrust_n: float = 20000.0,
        azimuth_deg: float = 0.0,
        elevation_deg: float = 0.0,
    ) -> SignatureState:
        """
        Compute all signatures at current vehicle state.
        
        Args:
            rotor_rpm: Rotor speed [RPM]
            motor_temp_c: Motor case temperature [°C]
            battery_temp_c: Battery pack temperature [°C]
            thrust_n: Total rotor thrust [N]
            azimuth_deg: Aspect angle (horizontal) [degrees]
            elevation_deg: Elevation angle [degrees]
        
        Returns:
            SignatureState with all computed values
        """
        # RCS
        rcs_x = self.radar.compute_rcs_x_band(
            azimuth_deg=azimuth_deg,
            elevation_deg=elevation_deg,
        )
        rcs_ku = self.radar.compute_rcs_ku_band(
            azimuth_deg=azimuth_deg,
            elevation_deg=elevation_deg,
        )
        
        # RCS modulation due to rotor (time-varying, ±50%)
        modulation = 50.0  # ± 50% peak-to-peak
        
        # IR
        ir_motor = self.ir.compute_motor_ir_power(motor_temp_c)
        ir_battery = self.ir.compute_battery_ir_power(battery_temp_c)
        
        # IR contrast (typical backgrounds)
        ir_contrast_ground = self.ir.compute_ir_contrast(
            vehicle_temp_c=(motor_temp_c + battery_temp_c) / 2,
            background_temp_c=25.0  # Warm ground
        )
        ir_contrast_sky = self.ir.compute_ir_contrast(
            vehicle_temp_c=(motor_temp_c + battery_temp_c) / 2,
            background_temp_c=-50.0  # Cold sky
        )
        
        # Acoustic
        bpf = self.acoustic.compute_blade_passing_frequency(rotor_rpm)
        spl = self.acoustic.compute_sound_pressure_level(thrust_n, rotor_rpm)
        
        # Package output
        state = SignatureState(
            rcs_x_band=rcs_x,
            rcs_ku_band=rcs_ku,
            rcs_modulation_depth=modulation,
            ir_power_motor=ir_motor,
            ir_power_battery=ir_battery,
            ir_contrast_ground=ir_contrast_ground,
            ir_contrast_sky=ir_contrast_sky,
            acoustic_signature_fundamental=bpf,
            acoustic_signature_spL=spl,
        )
        
        return state
