"""
Acoustic Signature Model

This module models the acoustic (noise) signature of the tiltrotor eVTOL
for detection vulnerability analysis and noise footprint planning.

Theory
======

Rotor Noise Sources:
1. Thickness Noise: Blade displacement of air (monopole)
2. Loading Noise: Fluctuating blade forces (dipole)
3. Blade-Vortex Interaction (BVI): Blade cutting tip vortices
4. Broadband Noise: Turbulence, trailing edge, blade-wake

For tiltrotors:
- Hover: High loading noise from concentrated thrust
- Transition: BVI noise increases during conversion
- Cruise: Lower noise, mainly propeller-mode

Acoustic Propagation:
    SPL(r) = SPL_ref - 20·log₁₀(r/r_ref) - α·r

where:
    α: Atmospheric absorption (dB/km)
    r: Distance from source

Electric Motor Noise:
    - Electromagnetic: Motor pole-passing frequency
    - Cooling fans: Broadband
    - Inverter PWM: High frequency

Author: Defense eVTOL Research Team
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class NoiseSource(Enum):
    """Acoustic noise source types."""
    THICKNESS = "thickness"
    LOADING = "loading"
    BVI = "bvi"
    BROADBAND = "broadband"
    MOTOR_EM = "motor_electromagnetic"
    INVERTER = "inverter"


@dataclass
class AcousticState:
    """Acoustic signature state."""
    # Overall SPL at reference distance (dB re 20 μPa at 1m)
    spl_total: float = 0.0

    # Component contributions (dB)
    spl_rotor: float = 0.0
    spl_motor: float = 0.0
    spl_airframe: float = 0.0

    # Detailed rotor noise breakdown
    spl_thickness: float = 0.0
    spl_loading: float = 0.0
    spl_bvi: float = 0.0
    spl_broadband: float = 0.0

    # A-weighted SPL (human perception)
    spl_a_weighted: float = 0.0

    # Peak frequencies (Hz)
    freq_fundamental: float = 0.0   # Blade passage frequency
    freq_peak: float = 0.0          # Loudest frequency

    # Detection metrics
    detection_range_km: float = 0.0       # Range for threshold detection
    ground_noise_footprint_m: float = 0.0  # Radius above threshold on ground

    # Directivity
    azimuth_peak_deg: float = 0.0
    elevation_peak_deg: float = 0.0


@dataclass
class AcousticConfig:
    """Acoustic model configuration."""
    # Rotor parameters (per rotor)
    num_rotors: int = 4             # Number of rotors
    num_blades: int = 3             # Blades per rotor
    rotor_radius_m: float = 1.5     # Rotor radius

    # Motor parameters
    motor_poles: int = 14           # Pole pairs

    # Reference conditions
    reference_distance_m: float = 30.0    # Reference for SPL measurement
    detection_threshold_db: float = 55.0   # SPL for detection

    # Atmospheric absorption (dB/km at standard conditions)
    absorption_125hz: float = 0.3
    absorption_500hz: float = 1.0
    absorption_2khz: float = 4.0
    absorption_8khz: float = 20.0

    # Baseline noise levels (empirical, from similar aircraft)
    base_spl_hover: float = 85.0     # dB at 30m
    base_spl_cruise: float = 75.0    # dB at 30m (propeller mode)

    # A-weighting correction for typical spectrum
    a_weight_correction: float = -5.0  # dB


class AcousticModel:
    """
    Acoustic signature model for tiltrotor eVTOL.

    This model computes sound pressure levels and noise footprint
    based on rotor operating conditions and flight state.

    Key features:
    - Multi-source noise model (rotor, motor, airframe)
    - Rotor noise breakdown (thickness, loading, BVI, broadband)
    - Atmospheric propagation with absorption
    - A-weighting for human perception
    - Ground noise footprint calculation

    For defense applications, acoustic signature affects:
    - Detection by ground personnel
    - Acoustic localization systems
    - Wildlife disturbance (conservation areas)
    """

    def __init__(self, config: AcousticConfig | None = None):
        """
        Initialize acoustic model.

        Args:
            config: Acoustic configuration
        """
        self.config = config or AcousticConfig()
        self.state = AcousticState()

        logger.info("Acoustic signature model initialized")

    def compute_signature(
        self,
        rotor_rpm: float,
        rotor_thrust: float,
        rotor_power: float,
        tip_mach: float,
        nacelle_angle_deg: float,
        airspeed_mps: float,
        altitude_m: float,
        descent_rate_mps: float = 0.0,
        temperature_c: float = 15.0,
        humidity_pct: float = 50.0,
    ) -> AcousticState:
        """
        Compute acoustic signature for given conditions.

        Args:
            rotor_rpm: Rotor angular speed
            rotor_thrust: Total thrust (N)
            rotor_power: Total shaft power (W)
            tip_mach: Blade tip Mach number
            nacelle_angle_deg: Nacelle tilt (90=hover, 0=cruise)
            airspeed_mps: Forward airspeed
            altitude_m: Altitude AGL
            descent_rate_mps: Descent rate (for BVI)
            temperature_c: Air temperature
            humidity_pct: Relative humidity

        Returns:
            Updated acoustic state
        """
        # Blade passage frequency
        bpf = self.config.num_blades * rotor_rpm / 60.0
        self.state.freq_fundamental = bpf

        # Compute rotor noise components
        self._compute_rotor_noise(
            rotor_rpm, rotor_thrust, rotor_power,
            tip_mach, nacelle_angle_deg, airspeed_mps,
            descent_rate_mps
        )

        # Compute motor/inverter noise
        self._compute_motor_noise(rotor_rpm, rotor_power)

        # Compute airframe noise
        self._compute_airframe_noise(airspeed_mps)

        # Combine sources (energy sum in dB)
        self.state.spl_total = self._sum_spl([
            self.state.spl_rotor,
            self.state.spl_motor,
            self.state.spl_airframe,
        ])

        # A-weighting
        self.state.spl_a_weighted = (
            self.state.spl_total + self.config.a_weight_correction
        )

        # Peak frequency (usually BPF or 2×BPF)
        self.state.freq_peak = 2 * bpf  # Typically 2nd harmonic

        # Compute detection range and footprint
        self._compute_propagation(altitude_m, temperature_c, humidity_pct)

        return self.state

    def _compute_rotor_noise(
        self,
        rpm: float,
        thrust: float,
        power: float,
        tip_mach: float,
        nacelle_angle_deg: float,
        airspeed_mps: float,
        descent_rate_mps: float,
    ) -> None:
        """Compute rotor noise components."""
        # Thrust per rotor
        thrust_per_rotor = thrust / self.config.num_rotors

        # Disk loading
        A_disk = np.pi * self.config.rotor_radius_m**2
        disk_loading = thrust_per_rotor / A_disk

        # Tip speed
        omega = rpm * 2 * np.pi / 60
        V_tip = omega * self.config.rotor_radius_m

        # === Thickness Noise ===
        # Monopole source, scales with M_tip^6
        spl_thickness_base = 60 + 60 * np.log10(tip_mach + 0.01)
        self.state.spl_thickness = max(0, spl_thickness_base)

        # === Loading Noise ===
        # Dipole source, scales with thrust and tip speed
        # SPL ∝ 10·log(T²·V_tip²)
        T_ref = 1000  # N reference thrust
        V_ref = 150   # m/s reference tip speed

        spl_loading = (
            self.config.base_spl_hover +
            10 * np.log10((thrust_per_rotor / T_ref)**2 + 0.01) +
            10 * np.log10((V_tip / V_ref)**2 + 0.01)
        )
        self.state.spl_loading = max(0, spl_loading)

        # === Blade-Vortex Interaction (BVI) ===
        # Critical during descent and transition
        # Descent angle affects wake impingement
        if airspeed_mps > 1:
            descent_angle_deg = np.degrees(np.arctan2(descent_rate_mps, airspeed_mps))
        else:
            descent_angle_deg = 0

        # BVI peaks at 5-15° descent angle
        bvi_factor = np.exp(-0.1 * (descent_angle_deg - 10)**2)

        # Also increased during transition (nacelle tilting)
        transition_factor = 1 + 0.5 * np.sin(np.radians(nacelle_angle_deg))

        spl_bvi_base = 75 + 10 * np.log10(disk_loading / 200 + 0.01)
        self.state.spl_bvi = spl_bvi_base * bvi_factor * transition_factor

        # === Broadband Noise ===
        # Turbulence ingestion, trailing edge, blade-wake
        # Scales with power (higher power = more turbulence)
        P_ref = 100000  # W reference
        spl_broadband = 65 + 5 * np.log10(power / P_ref + 0.01)
        self.state.spl_broadband = max(0, spl_broadband)

        # === Total Rotor Noise ===
        self.state.spl_rotor = self._sum_spl([
            self.state.spl_thickness,
            self.state.spl_loading,
            self.state.spl_bvi,
            self.state.spl_broadband,
        ])

    def _compute_motor_noise(self, rpm: float, power: float) -> None:
        """
        Compute motor and inverter noise.

        Electric motors produce less noise than internal combustion,
        but electromagnetic and cooling noise can be significant.
        """
        # Motor electromagnetic noise
        # Fundamental at pole passing frequency
        motor_rpm = rpm  # Direct drive assumed
        motor_em_freq = self.config.motor_poles * motor_rpm / 60  # Hz, EM excitation frequency

        # SPL scales with power
        P_ref = 100000  # W
        spl_motor_em = 55 + 5 * np.log10(power / P_ref + 0.01)

        # Inverter switching noise (typically >10 kHz, attenuated by A-weighting)
        spl_inverter = 50 + 3 * np.log10(power / P_ref + 0.01)

        # Combined motor system noise
        self.state.spl_motor = self._sum_spl([spl_motor_em, spl_inverter])

    def _compute_airframe_noise(self, airspeed_mps: float) -> None:
        """
        Compute airframe aerodynamic noise.

        Sources include landing gear, antennae, surface roughness.
        Negligible at low speed but grows with velocity.
        """
        V_ref = 100  # m/s

        if airspeed_mps > 10:
            # Sixth power law for aerodynamic noise
            self.state.spl_airframe = 50 + 60 * np.log10(airspeed_mps / V_ref)
        else:
            self.state.spl_airframe = 0

    def _compute_propagation(
        self,
        altitude_m: float,
        temperature_c: float,
        humidity_pct: float,
    ) -> None:
        """Compute propagation to ground and detection range."""
        # Atmospheric absorption at dominant frequency
        freq = self.state.freq_peak

        # Interpolate absorption coefficient
        if freq < 250:
            alpha = self.config.absorption_125hz
        elif freq < 1000:
            alpha = self.config.absorption_500hz
        elif freq < 4000:
            alpha = self.config.absorption_2khz
        else:
            alpha = self.config.absorption_8khz

        # Humidity correction (higher humidity = less absorption)
        humidity_factor = 1 - 0.01 * (humidity_pct - 50)
        alpha *= humidity_factor

        # === Detection Range ===
        # Range where SPL equals threshold
        # SPL(r) = SPL_ref - 20·log(r/r_ref) - α·r/1000
        spl_ref = self.state.spl_a_weighted
        r_ref = self.config.reference_distance_m
        threshold = self.config.detection_threshold_db

        # Solve iteratively
        for r in np.linspace(100, 10000, 100):
            spl_at_r = spl_ref - 20 * np.log10(r / r_ref) - alpha * r / 1000
            if spl_at_r <= threshold:
                self.state.detection_range_km = r / 1000
                break
        else:
            self.state.detection_range_km = 10.0  # Max range

        # === Ground Noise Footprint ===
        # Radius on ground above threshold at given altitude
        if altitude_m > 10:
            # Direct downward propagation
            spl_ground = spl_ref - 20 * np.log10(altitude_m / r_ref)
            spl_ground -= alpha * altitude_m / 1000

            if spl_ground > threshold:
                # Footprint radius where SPL drops to threshold
                excess_db = spl_ground - threshold
                self.state.ground_noise_footprint_m = (
                    altitude_m * 10**(excess_db / 40)  # Approximate geometry
                )
            else:
                self.state.ground_noise_footprint_m = 0
        else:
            self.state.ground_noise_footprint_m = self.state.detection_range_km * 1000

    def _sum_spl(self, levels: list[float]) -> float:
        """
        Sum sound pressure levels (energy addition).

        SPL_total = 10·log₁₀(Σ 10^(SPL_i/10))
        """
        # Filter out zeros and very low values
        valid_levels = [l for l in levels if l > 0]

        if not valid_levels:
            return 0.0

        # Energy sum
        energy_sum = sum(10**(l / 10) for l in valid_levels)
        return 10 * np.log10(energy_sum)

    def get_spl_at_distance(
        self,
        distance_m: float,
        frequency_hz: float | None = None,
    ) -> float:
        """
        Get SPL at given distance from aircraft.

        Args:
            distance_m: Distance from source
            frequency_hz: Frequency (for absorption), uses peak if None

        Returns:
            SPL in dB at given distance
        """
        if frequency_hz is None:
            frequency_hz = self.state.freq_peak

        # Get absorption
        if frequency_hz < 250:
            alpha = self.config.absorption_125hz
        elif frequency_hz < 1000:
            alpha = self.config.absorption_500hz
        elif frequency_hz < 4000:
            alpha = self.config.absorption_2khz
        else:
            alpha = self.config.absorption_8khz

        # Propagation
        r_ref = self.config.reference_distance_m
        spl = (
            self.state.spl_total -
            20 * np.log10(distance_m / r_ref) -
            alpha * distance_m / 1000
        )

        return max(0, spl)

    def get_directivity_pattern(
        self,
        num_points: int = 72,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Get approximate noise directivity pattern.

        Rotor noise is typically loudest in-plane and below.

        Returns:
            Tuple of (azimuth_angles, relative_spl) arrays
        """
        azimuths = np.linspace(0, 360, num_points)

        # Simplified directivity (rotor noise peaks near rotor plane)
        relative_spl = np.zeros(num_points)

        for i, az in enumerate(azimuths):
            # Thickness noise peaks in-plane
            thickness_dir = 1 - 0.3 * np.abs(np.sin(np.radians(az)))

            # Loading noise more uniform with slight increase below
            loading_dir = 1.0

            # Combined
            relative_spl[i] = (thickness_dir + loading_dir) / 2

        # Normalize to dB relative
        relative_spl = 10 * np.log10(relative_spl)
        relative_spl -= relative_spl.max()

        return azimuths, relative_spl

    def is_detectable_at_ground(self, altitude_m: float) -> bool:
        """Check if aircraft is acoustically detectable from ground."""
        spl_ground = self.get_spl_at_distance(altitude_m)
        return spl_ground > self.config.detection_threshold_db

    def get_state(self) -> AcousticState:
        """Get current acoustic state."""
        return self.state

    def to_dict(self) -> dict:
        """Convert state to dictionary."""
        return {
            'spl_total': self.state.spl_total,
            'spl_rotor': self.state.spl_rotor,
            'spl_motor': self.state.spl_motor,
            'spl_a_weighted': self.state.spl_a_weighted,
            'freq_fundamental': self.state.freq_fundamental,
            'detection_range_km': self.state.detection_range_km,
            'ground_footprint_m': self.state.ground_noise_footprint_m,
        }
