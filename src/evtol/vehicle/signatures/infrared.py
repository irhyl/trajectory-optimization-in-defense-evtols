"""
Infrared Signature Model

This module models the infrared (thermal) signature of the tiltrotor eVTOL
for detection vulnerability analysis and signature management.

Theory
======

IR signature arises from thermal emission following Stefan-Boltzmann law:
    P = ε·σ·A·T⁴

where:
    ε: Emissivity (0-1)
    σ: Stefan-Boltzmann constant (5.67×10⁻⁸ W/m²K⁴)
    A: Radiating area (m²)
    T: Temperature (K)

IR Bands (Atmospheric Windows):
    - SWIR (1-3 μm): Solar reflection, hot exhaust
    - MWIR (3-5 μm): Hot engine parts, exhaust plume
    - LWIR (8-14 μm): Warm aircraft skin, ground contrast

Sources for eVTOL:
==================

1. Motor/Inverter: 80-150°C operating temperature
   - Low compared to turbines
   - Radiates in LWIR

2. Battery Pack: 20-50°C (with cooling)
   - Low signature
   - Internal, partially shielded

3. Aerodynamic Heating: Minimal at eVTOL speeds (<250 kt)

4. Rotor Tip Heating: Compressive heating at high RPM
   - V_tip up to 200 m/s
   - Minimal IR contribution

5. Solar Reflection: Depends on surface finish
   - Can be significant in SWIR band

6. Exhaust: For electric, only cooling airflow
   - ~20-40°C above ambient
   - Very low compared to jet/turboprop

Author: Defense eVTOL Research Team
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


# Stefan-Boltzmann constant
STEFAN_BOLTZMANN = 5.67e-8  # W/m²K⁴


class IRBand(Enum):
    """Infrared spectral bands."""
    SWIR = "SWIR"    # 1-3 μm (short-wave)
    MWIR = "MWIR"    # 3-5 μm (mid-wave)
    LWIR = "LWIR"    # 8-14 μm (long-wave)


@dataclass
class IRState:
    """IR signature state."""
    # Radiant intensity by band (W/sr)
    intensity: dict[IRBand, float] = field(default_factory=dict)

    # Component temperatures (°C)
    temp_motor: float = 80.0
    temp_inverter: float = 60.0
    temp_battery: float = 35.0
    temp_skin: float = 25.0
    temp_coolant_exhaust: float = 30.0

    # Component contributions (W/sr)
    ir_motor: float = 0.0
    ir_inverter: float = 0.0
    ir_battery: float = 0.0
    ir_skin: float = 0.0
    ir_exhaust: float = 0.0

    # Contrast against background
    contrast_ratio: float = 1.0  # >1 = warmer than background

    # Solar reflection (for SWIR)
    solar_reflected: float = 0.0

    # Detection parameters
    detection_range_km: dict[IRBand, float] = field(default_factory=dict)


@dataclass
class IRConfig:
    """IR model configuration."""
    # Component emissivities
    emissivity_motor: float = 0.3       # Metallic (aluminum housing)
    emissivity_inverter: float = 0.4    # Mixed (heatsink)
    emissivity_battery: float = 0.9     # Dark enclosure
    emissivity_skin: float = 0.8        # Composite/painted
    emissivity_exhaust: float = 0.9     # Hot air

    # Visible areas (m²) - depends on aspect
    area_motor: float = 0.2             # Exposed motor area
    area_inverter: float = 0.1          # Usually internal
    area_battery: float = 0.0           # Internal, not visible
    area_skin: float = 10.0             # Total skin area
    area_exhaust: float = 0.1           # Cooling vent area

    # Solar reflection
    solar_reflectivity: float = 0.3     # Fraction reflected
    solar_irradiance: float = 1000.0    # W/m² (at surface)

    # Reference IR seeker parameters (for detection range estimate)
    seeker_sensitivity: float = 1e-12   # W at detector
    seeker_aperture: float = 0.1        # m diameter
    atmospheric_transmission: float = 0.7  # Per km at low altitude


class IRModel:
    """
    Infrared signature model for tiltrotor eVTOL.

    This model computes IR radiant intensity in different bands
    based on component temperatures and viewing geometry.

    Key features:
    - Multi-band (SWIR/MWIR/LWIR) signature
    - Temperature-dependent emission
    - Aspect-angle variation
    - Solar reflection component
    - Background contrast estimation

    For defense eVTOL, IR signature is low compared to turbine aircraft
    due to electric propulsion, but motors and cooling exhaust are still
    detectable by sensitive IR seekers.
    """

    def __init__(self, config: IRConfig | None = None):
        """
        Initialize IR model.

        Args:
            config: IR signature configuration
        """
        self.config = config or IRConfig()
        self.state = IRState()

        logger.info("IR signature model initialized")

    def compute_signature(
        self,
        temp_motor: float,
        temp_inverter: float,
        temp_battery: float,
        temp_coolant_exhaust: float,
        airspeed_mps: float = 0.0,
        altitude_m: float = 0.0,
        ambient_temp: float = 15.0,
        solar_angle_deg: float = 45.0,
        azimuth_deg: float = 0.0,
        elevation_deg: float = 0.0,
    ) -> IRState:
        """
        Compute IR signature for given conditions.

        Args:
            temp_motor: Motor temperature (°C)
            temp_inverter: Inverter temperature (°C)
            temp_battery: Battery temperature (°C)
            temp_coolant_exhaust: Cooling exhaust temperature (°C)
            airspeed_mps: Airspeed for skin heating (m/s)
            altitude_m: Altitude for atmospheric effects
            ambient_temp: Ambient temperature (°C)
            solar_angle_deg: Sun elevation angle
            azimuth_deg: Observer azimuth angle
            elevation_deg: Observer elevation angle

        Returns:
            Updated IR state
        """
        # Store temperatures
        self.state.temp_motor = temp_motor
        self.state.temp_inverter = temp_inverter
        self.state.temp_battery = temp_battery
        self.state.temp_coolant_exhaust = temp_coolant_exhaust

        # Skin temperature (aerodynamic heating minimal for eVTOL)
        # Recovery temperature: T_r = T_ambient × (1 + 0.2 × M²)
        mach = airspeed_mps / 340.0
        T_recovery = (ambient_temp + 273.15) * (1 + 0.2 * mach**2) - 273.15
        self.state.temp_skin = max(ambient_temp, T_recovery)

        # Compute visible areas based on aspect
        areas = self._compute_visible_areas(azimuth_deg, elevation_deg)

        # Compute thermal emission for each component
        self.state.ir_motor = self._compute_emission(
            temp_motor, self.config.emissivity_motor, areas['motor']
        )
        self.state.ir_inverter = self._compute_emission(
            temp_inverter, self.config.emissivity_inverter, areas['inverter']
        )
        self.state.ir_battery = self._compute_emission(
            temp_battery, self.config.emissivity_battery, areas['battery']
        )
        self.state.ir_skin = self._compute_emission(
            self.state.temp_skin, self.config.emissivity_skin, areas['skin']
        )
        self.state.ir_exhaust = self._compute_emission(
            temp_coolant_exhaust, self.config.emissivity_exhaust, areas['exhaust']
        )

        # Solar reflection (SWIR band primarily)
        if solar_angle_deg > 0:
            # Solar illumination
            sun_irradiance = self.config.solar_irradiance * np.sin(np.radians(solar_angle_deg))
            # Reflected toward observer (diffuse approximation)
            self.state.solar_reflected = (
                sun_irradiance *
                self.config.solar_reflectivity *
                areas['skin'] / np.pi  # Lambertian reflection
            )
        else:
            self.state.solar_reflected = 0.0

        # Compute band-specific intensities
        self._compute_band_intensities(ambient_temp)

        # Background contrast
        T_back_K = ambient_temp + 273.15
        T_skin_K = self.state.temp_skin + 273.15
        self.state.contrast_ratio = (T_skin_K / T_back_K) ** 4

        # Detection range estimates
        self._estimate_detection_ranges(altitude_m)

        return self.state

    def _compute_visible_areas(
        self,
        azimuth_deg: float,
        elevation_deg: float,
    ) -> dict[str, float]:
        """
        Compute visible projected areas for each component.

        Args:
            azimuth_deg: Observer azimuth
            elevation_deg: Observer elevation

        Returns:
            Dictionary of visible areas per component
        """
        az_rad = np.radians(azimuth_deg)
        el_rad = np.radians(elevation_deg)  # noqa: F841 (reserved for elevation-dependent visibility)

        # Motor visibility (at nacelles)
        # Motors visible from side and somewhat from front/back
        motor_factor = 0.5 + 0.5 * np.abs(np.sin(az_rad))

        # Inverter (usually inside fuselage)
        inverter_factor = 0.2  # Minimal external visibility

        # Battery (internal)
        battery_factor = 0.0  # Not visible

        # Skin (all-around visibility)
        skin_factor = 1.0

        # Exhaust (cooling vents - primarily at rear/sides)
        exhaust_factor = 0.3 + 0.5 * np.abs(np.cos(az_rad - np.pi))

        return {
            'motor': self.config.area_motor * motor_factor,
            'inverter': self.config.area_inverter * inverter_factor,
            'battery': self.config.area_battery * battery_factor,
            'skin': self.config.area_skin * skin_factor,
            'exhaust': self.config.area_exhaust * exhaust_factor,
        }

    def _compute_emission(
        self,
        temp_c: float,
        emissivity: float,
        area: float,
    ) -> float:
        """
        Compute thermal emission power.

        Args:
            temp_c: Temperature in Celsius
            emissivity: Surface emissivity
            area: Radiating area (m²)

        Returns:
            Radiant power (W)
        """
        T_kelvin = temp_c + 273.15

        # Stefan-Boltzmann law
        power = emissivity * STEFAN_BOLTZMANN * area * T_kelvin**4

        # Convert to intensity (W/sr) assuming Lambertian emission
        intensity = power / np.pi

        return intensity

    def _compute_band_intensities(self, ambient_temp: float) -> None:
        """Compute intensity in each IR band."""
        # Total thermal emission
        total_thermal = (
            self.state.ir_motor +
            self.state.ir_inverter +
            self.state.ir_battery +
            self.state.ir_skin +
            self.state.ir_exhaust
        )

        # Background thermal emission (for contrast)
        T_back = ambient_temp + 273.15
        background_emission = STEFAN_BOLTZMANN * T_back**4 * 10.0 / np.pi

        # Net emission above background
        net_thermal = max(0, total_thermal - background_emission * 0.8)

        # Band distribution (simplified based on Planck's law)
        # For motor at ~100°C (373K), peak emission ~8μm (LWIR)
        # For skin at ~25°C (298K), peak emission ~10μm (LWIR)

        self.state.intensity[IRBand.SWIR] = (
            self.state.solar_reflected +  # Dominant in SWIR
            0.01 * net_thermal  # Minimal thermal at these short wavelengths
        )

        self.state.intensity[IRBand.MWIR] = (
            0.15 * net_thermal  # Some thermal contribution
        )

        self.state.intensity[IRBand.LWIR] = (
            0.84 * net_thermal  # Dominant thermal emission band
        )

    def _estimate_detection_ranges(self, altitude_m: float) -> None:
        """
        Estimate detection range for IR seeker.

        Uses simplified range equation:
            Range = sqrt(I × τ × A_aperture / (4π × NEP))

        where:
            I: Source intensity (W/sr)
            τ: Atmospheric transmission
            A_aperture: Seeker aperture area
            NEP: Noise equivalent power
        """
        # Altitude effect on atmospheric transmission
        # Lower altitude = more water vapor = more attenuation
        if altitude_m < 1000:
            tau_factor = 0.8
        elif altitude_m < 3000:
            tau_factor = 0.9
        else:
            tau_factor = 1.0

        tau = self.config.atmospheric_transmission * tau_factor
        A_aperture = np.pi * (self.config.seeker_aperture / 2)**2
        sensitivity = self.config.seeker_sensitivity

        for band in IRBand:
            intensity = self.state.intensity.get(band, 0.0)

            if intensity > 0:
                # Range where received power equals sensitivity
                # P_received = I × τ^range × A / (4π × range²)
                # Simplified: range ≈ sqrt(I × τ × A / (4π × sensitivity))
                range_m = np.sqrt(intensity * tau * A_aperture / (4 * np.pi * sensitivity))
                self.state.detection_range_km[band] = range_m / 1000.0
            else:
                self.state.detection_range_km[band] = 0.0

    def get_total_intensity(self, band: IRBand = IRBand.MWIR) -> float:
        """Get total IR intensity in given band."""
        return self.state.intensity.get(band, 0.0)

    def get_contrast_temperature(self, band: IRBand = IRBand.LWIR) -> float:
        """
        Get apparent temperature difference from background.

        This is what IR seekers measure - the ΔT from background.
        """
        # Simplified: assume background at ambient
        # Real ΔT depends on specific geometry and background
        return self.state.temp_skin - 15.0  # Assume 15°C background

    def is_detectable(
        self,
        seeker_range_km: float,
        band: IRBand = IRBand.MWIR,
    ) -> bool:
        """Check if aircraft is detectable at given range."""
        detection_range = self.state.detection_range_km.get(band, 0.0)
        return seeker_range_km <= detection_range

    def get_state(self) -> IRState:
        """Get current IR state."""
        return self.state

    def to_dict(self) -> dict:
        """Convert state to dictionary."""
        return {
            'temp_motor': self.state.temp_motor,
            'temp_skin': self.state.temp_skin,
            'temp_exhaust': self.state.temp_coolant_exhaust,
            'intensity_swir': self.state.intensity.get(IRBand.SWIR, 0.0),
            'intensity_mwir': self.state.intensity.get(IRBand.MWIR, 0.0),
            'intensity_lwir': self.state.intensity.get(IRBand.LWIR, 0.0),
            'solar_reflected': self.state.solar_reflected,
            'contrast_ratio': self.state.contrast_ratio,
            'detection_range_mwir_km': self.state.detection_range_km.get(IRBand.MWIR, 0.0),
            'detection_range_lwir_km': self.state.detection_range_km.get(IRBand.LWIR, 0.0),
        }
