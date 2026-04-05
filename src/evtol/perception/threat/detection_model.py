"""
Detection Model for Defense eVTOL Trajectory Optimization.

This module implements physics-based radar detection modeling using
the radar range equation and probabilistic detection theory.

Mathematical Framework:

    1. Radar Range Equation (Two-way):
        P_r = (P_t × G² × λ² × σ) / ((4π)³ × R⁴)

    Where:
        P_r = Received power (W)
        P_t = Transmitted power (W)
        G   = Antenna gain
        λ   = Wavelength (m)
        σ   = Target RCS (m²)
        R   = Range (m)

    2. Signal-to-Noise Ratio:
        SNR = P_r / (k × T × B × F)

    Where:
        k = Boltzmann's constant
        T = System temperature (K)
        B = Bandwidth (Hz)
        F = Noise figure

    3. Detection Probability (Swerling Models):
        P_d = f(SNR, P_fa, fluctuation model)

    Swerling Cases:
        - Case 0: Non-fluctuating (Marcum Q-function)
        - Case I: Slow fluctuation, many scatterers
        - Case II: Fast fluctuation, many scatterers
        - Case III: Slow, one dominant + many small
        - Case IV: Fast, one dominant + many small

    4. Cumulative Detection (Track-While-Scan):
        P_d_cum = 1 - ∏(1 - P_d_i)

    5. Terrain Masking:
        Detection requires line-of-sight (LOS)
        P_d_terrain = P_d × LOS_factor

References:
    [1] Skolnik, M. "Radar Handbook" 3rd Ed.
    [2] Mahafza, B. "Radar Systems Analysis and Design"
    [3] Blake, L. "Radar Range-Performance Analysis"
    [4] Barton, D. "Modern Radar System Analysis"
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any
from enum import Enum

from .threat_types import (
    ThreatSystem,
    RadarSpecification,
    RCSProfile,
    EVTOL_RCS_PROFILE,
    AlertLevel,
)


# =============================================================================
# PHYSICAL CONSTANTS
# =============================================================================

# Speed of light (m/s)
C = 299792458.0

# Boltzmann constant (J/K)
K_BOLTZMANN = 1.380649e-23

# Reference temperature (K)
T_REF = 290.0

# Earth radius (m)
EARTH_RADIUS = 6371000.0


# =============================================================================
# SWERLING FLUCTUATION MODELS
# =============================================================================

class SwerlingCase(Enum):
    """
    Swerling target fluctuation models.

    Describes how target RCS fluctuates during observation.
    """
    CASE_0 = 0    # Non-fluctuating (steady target)
    CASE_I = 1    # Scan-to-scan fluctuation, Rayleigh
    CASE_II = 2   # Pulse-to-pulse fluctuation, Rayleigh
    CASE_III = 3  # Scan-to-scan, one dominant + many small
    CASE_IV = 4   # Pulse-to-pulse, one dominant + many small


# =============================================================================
# DETECTION MODEL CONFIGURATION
# =============================================================================

@dataclass
class DetectionModelConfig:
    """Configuration for detection model calculations."""

    # RCS profile
    rcs_profile: RCSProfile = field(default_factory=lambda: EVTOL_RCS_PROFILE)

    # Target fluctuation model
    swerling_case: SwerlingCase = SwerlingCase.CASE_I

    # Detection threshold
    probability_false_alarm: float = 1e-6
    probability_detection_required: float = 0.9

    # Integration
    pulses_integrated: int = 10
    coherent_integration: bool = True

    # Atmospheric effects
    include_atmospheric_loss: bool = True
    atmospheric_loss_db_per_km: float = 0.01  # Clear weather

    # Earth curvature
    include_earth_curvature: bool = True
    antenna_height_m: float = 5.0

    # Terrain masking
    include_terrain_masking: bool = True
    terrain_resolution_m: float = 30.0


# =============================================================================
# CORE DETECTION CALCULATIONS
# =============================================================================

class RadarDetectionModel:
    """
    Physics-based radar detection probability calculator.

    Implements full radar range equation with atmospheric effects,
    earth curvature, and statistical detection theory.
    """

    def __init__(self, config: DetectionModelConfig | None = None):
        """Initialize detection model with configuration."""
        self.config = config or DetectionModelConfig()

    def calculate_detection_probability(
        self,
        radar: RadarSpecification,
        target_range_m: float,
        target_altitude_m: float,
        target_rcs_sqm: float,
        aspect_azimuth_deg: float = 0.0,
        aspect_elevation_deg: float = 0.0,
        terrain_factor: float = 1.0,
    ) -> dict[str, float]:
        """
        Calculate detection probability for given geometry.

        Args:
            radar: Radar system specification
            target_range_m: Slant range to target (m)
            target_altitude_m: Target altitude MSL (m)
            target_rcs_sqm: Target RCS (m²)
            aspect_azimuth_deg: Target aspect angle in azimuth
            aspect_elevation_deg: Target aspect angle in elevation
            terrain_factor: 0-1 terrain masking factor

        Returns:
            Dictionary with detection metrics
        """
        # Get aspect-dependent RCS
        effective_rcs = self.config.rcs_profile.get_rcs(
            aspect_azimuth_deg,
            aspect_elevation_deg,
            radar.band,
        )

        # Calculate SNR
        snr_db = self._calculate_snr(
            radar,
            target_range_m,
            effective_rcs,
        )

        # Apply atmospheric loss
        if self.config.include_atmospheric_loss:
            range_km = target_range_m / 1000.0
            atm_loss_db = self.config.atmospheric_loss_db_per_km * range_km
            snr_db -= atm_loss_db

        # Check radar horizon
        horizon_range = self._radar_horizon_range(
            radar_height_m=self.config.antenna_height_m + radar.max_range_km_1sqm * 10,  # Elevated radar
            target_height_m=target_altitude_m,
        )

        horizon_factor = 1.0
        if self.config.include_earth_curvature:
            if target_range_m > horizon_range:
                horizon_factor = 0.0  # Beyond horizon
            elif target_range_m > horizon_range * 0.9:
                # Transitional region
                horizon_factor = (horizon_range - target_range_m) / (0.1 * horizon_range)

        # Apply integration gain
        integration_gain_db = self._integration_gain(
            self.config.pulses_integrated,
            self.config.coherent_integration,
        )
        snr_db += integration_gain_db

        # Calculate detection probability
        snr_linear = 10 ** (snr_db / 10)
        pd = self._swerling_pd(
            snr_linear,
            self.config.probability_false_alarm,
            self.config.swerling_case,
        )

        # Apply terrain and horizon factors
        pd_effective = pd * terrain_factor * horizon_factor

        # Calculate detection range (for reference)
        detection_range_km = self._detection_range(
            radar,
            target_rcs_sqm,
            self.config.probability_detection_required,
        )

        return {
            "probability_detection": pd_effective,
            "snr_db": snr_db,
            "effective_rcs_sqm": effective_rcs,
            "horizon_range_km": horizon_range / 1000.0,
            "detection_range_km": detection_range_km,
            "terrain_factor": terrain_factor,
            "horizon_factor": horizon_factor,
            "pd_raw": pd,
        }

    def _calculate_snr(
        self,
        radar: RadarSpecification,
        range_m: float,
        rcs_sqm: float,
    ) -> float:
        """
        Calculate signal-to-noise ratio using radar equation.

        SNR = (P_t × G² × λ² × σ) / ((4π)³ × R⁴ × k × T × B × F)
        """
        # Convert radar parameters
        pt_w = radar.peak_power_kw * 1000.0  # W
        g_linear = 10 ** (radar.antenna_gain_db / 10)
        wavelength = radar.wavelength_m
        nf_linear = 10 ** (radar.noise_figure_db / 10)

        # Estimate bandwidth from beamwidth (approximate)
        # B ≈ 1 / pulse_width, pulse_width ≈ 2/(beamwidth_rad × range × velocity)
        bandwidth = 1e6  # Assume 1 MHz bandwidth

        # Noise power
        noise_power = K_BOLTZMANN * T_REF * bandwidth * nf_linear

        # Received power (radar equation)
        numerator = pt_w * (g_linear ** 2) * (wavelength ** 2) * rcs_sqm
        denominator = ((4 * math.pi) ** 3) * (range_m ** 4)

        if denominator == 0:
            return 100.0  # Very high SNR at zero range

        pr = numerator / denominator

        # SNR in dB
        snr_linear = pr / noise_power
        snr_db = 10 * math.log10(max(snr_linear, 1e-20))

        return snr_db

    def _integration_gain(
        self,
        num_pulses: int,
        coherent: bool,
    ) -> float:
        """Calculate integration gain in dB."""
        if num_pulses <= 1:
            return 0.0

        if coherent:
            # Coherent integration: gain = N
            return 10 * math.log10(num_pulses)
        else:
            # Non-coherent integration: gain ≈ √N
            return 10 * math.log10(math.sqrt(num_pulses))

    def _radar_horizon_range(
        self,
        radar_height_m: float,
        target_height_m: float,
    ) -> float:
        """
        Calculate radar horizon range considering Earth curvature.

        R_horizon = √(2 × R_e × h_radar) + √(2 × R_e × h_target)

        Uses 4/3 Earth radius for refraction.
        """
        # Effective Earth radius (4/3 model for standard refraction)
        re_effective = EARTH_RADIUS * 4 / 3

        # Horizon distances
        d_radar = math.sqrt(2 * re_effective * max(radar_height_m, 0))
        d_target = math.sqrt(2 * re_effective * max(target_height_m, 0))

        return d_radar + d_target

    def _detection_range(
        self,
        radar: RadarSpecification,
        rcs_sqm: float,
        pd_required: float,
    ) -> float:
        """
        Calculate maximum detection range for given P_d.

        Uses radar-specific max_range_km_1sqm and scales by RCS.
        """
        # Scale reference range by RCS^0.25
        range_km = radar.max_range_km_1sqm * (rcs_sqm ** 0.25)

        # Adjust for P_d requirement (higher P_d = shorter range)
        if pd_required > 0.9:
            range_km *= 0.9
        elif pd_required < 0.5:
            range_km *= 1.2

        return range_km

    def _swerling_pd(
        self,
        snr: float,
        pfa: float,
        case: SwerlingCase,
    ) -> float:
        """
        Calculate detection probability for Swerling models.

        Simplified approximations for computational efficiency.
        """
        if snr <= 0:
            return 0.0

        # Required SNR for given Pfa (Albersheim's approximation)
        a = math.log(0.62 / pfa)
        snr_required = a + 0.12 * a * math.log(a)

        # Detection probability based on SNR margin
        snr_margin = snr / snr_required

        if case == SwerlingCase.CASE_0:
            # Non-fluctuating: sharp transition
            if snr_margin >= 1.0:
                pd = 1.0 - math.exp(-snr_margin + 1)
            else:
                pd = 0.5 * snr_margin ** 2

        elif case in [SwerlingCase.CASE_I, SwerlingCase.CASE_II]:
            # Rayleigh fluctuation: softer transition
            if snr_margin >= 1.0:
                pd = 1.0 - math.exp(-0.5 * snr_margin)
            else:
                pd = 0.3 * snr_margin

        elif case in [SwerlingCase.CASE_III, SwerlingCase.CASE_IV]:
            # Chi-square (4 DOF) fluctuation
            if snr_margin >= 1.0:
                pd = 1.0 - (1 + snr_margin) * math.exp(-snr_margin)
            else:
                pd = 0.4 * snr_margin ** 1.5

        else:
            pd = 0.5  # Default

        return min(max(pd, 0.0), 1.0)

    def calculate_cumulative_pd(
        self,
        pd_per_scan: list[float],
    ) -> float:
        """
        Calculate cumulative detection probability over multiple scans.

        P_d_cum = 1 - ∏(1 - P_d_i)
        """
        if not pd_per_scan:
            return 0.0

        prob_miss_all = 1.0
        for pd in pd_per_scan:
            prob_miss_all *= (1.0 - pd)

        return 1.0 - prob_miss_all

    def detection_timeline(
        self,
        radar: RadarSpecification,
        trajectory: list[tuple[float, float, float, float]],  # (lat, lon, alt, time)
        threat_position: tuple[float, float, float],  # (lat, lon, alt)
        scan_interval_s: float = 6.0,
    ) -> dict[str, Any]:
        """
        Calculate detection probability along a trajectory.

        Args:
            radar: Radar specification
            trajectory: List of (lat, lon, alt_m, time_s) points
            threat_position: Threat (lat, lon, alt_m)
            scan_interval_s: Radar scan interval

        Returns:
            Timeline of detection metrics
        """
        results = {
            "timeline": [],
            "first_detection_time": None,
            "max_pd": 0.0,
            "cumulative_pd": 0.0,
            "time_in_detection": 0.0,
        }

        pd_history = []
        last_scan_time = 0.0

        for lat, lon, alt, time in trajectory:
            # Calculate range
            range_m = self._haversine_m(
                lat, lon, threat_position[0], threat_position[1]
            )

            # Get RCS (simplified - assume nose-on)
            rcs = self.config.rcs_profile.get_rcs(0, 0, radar.band)

            # Calculate detection probability
            det_result = self.calculate_detection_probability(
                radar=radar,
                target_range_m=range_m,
                target_altitude_m=alt,
                target_rcs_sqm=rcs,
            )

            pd = det_result["probability_detection"]

            # Track scan times
            if time >= last_scan_time + scan_interval_s:
                pd_history.append(pd)
                last_scan_time = time

            # Update results
            results["timeline"].append({
                "time": time,
                "range_km": range_m / 1000.0,
                "pd": pd,
                "snr_db": det_result["snr_db"],
            })

            if pd > 0.5 and results["first_detection_time"] is None:
                results["first_detection_time"] = time

            results["max_pd"] = max(results["max_pd"], pd)

        # Calculate cumulative Pd
        results["cumulative_pd"] = self.calculate_cumulative_pd(pd_history)

        # Time in detection zone
        detection_count = sum(1 for t in results["timeline"] if t["pd"] > 0.5)
        if len(results["timeline"]) > 1:
            dt = results["timeline"][1]["time"] - results["timeline"][0]["time"]
            results["time_in_detection"] = detection_count * dt

        return results

    @staticmethod
    def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate great-circle distance in meters."""
        R = 6371000.0

        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)

        a = (math.sin(dlat / 2) ** 2 +
             math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        return R * c


# =============================================================================
# MULTI-THREAT DETECTION AGGREGATOR
# =============================================================================

class DetectionAggregator:
    """
    Aggregate detection probabilities from multiple threat systems.

    Models the IADS (Integrated Air Defense System) detection capability
    where multiple radars contribute to overall detection probability.
    """

    def __init__(self, config: DetectionModelConfig | None = None):
        """Initialize aggregator."""
        self.detection_model = RadarDetectionModel(config)
        self.config = config or DetectionModelConfig()

    def calculate_aggregate_pd(
        self,
        threats: list[ThreatSystem],
        target_lat: float,
        target_lon: float,
        target_alt_m: float,
        target_heading_deg: float = 0.0,
        terrain_factors: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """
        Calculate aggregate detection probability from all threats.

        Assumes independent detection attempts (conservative).

        Args:
            threats: List of threat systems
            target_lat: Target latitude
            target_lon: Target longitude
            target_alt_m: Target altitude MSL
            target_heading_deg: Target heading for aspect calculation
            terrain_factors: Optional per-threat terrain masking

        Returns:
            Aggregate detection assessment
        """
        terrain_factors = terrain_factors or {}

        individual_pds = []
        threat_details = []

        for threat in threats:
            if threat.radar is None:
                # Non-radar threat (visual detection)
                pd = self._visual_detection_pd(
                    threat, target_lat, target_lon, target_alt_m
                )
            else:
                # Radar detection
                range_m = self._haversine_m(
                    target_lat, target_lon,
                    threat.latitude, threat.longitude,
                )

                # Calculate aspect angle
                bearing = self._bearing_deg(
                    threat.latitude, threat.longitude,
                    target_lat, target_lon,
                )
                relative_bearing = (bearing - target_heading_deg + 180) % 360 - 180

                terrain_factor = terrain_factors.get(threat.threat_id, 1.0)

                result = self.detection_model.calculate_detection_probability(
                    radar=threat.radar,
                    target_range_m=range_m,
                    target_altitude_m=target_alt_m,
                    target_rcs_sqm=0.5,  # Nominal eVTOL RCS
                    aspect_azimuth_deg=relative_bearing,
                    terrain_factor=terrain_factor,
                )

                pd = result["probability_detection"]

            if pd > 0:
                individual_pds.append(pd)
                threat_details.append({
                    "threat_id": threat.threat_id,
                    "name": threat.name,
                    "category": threat.category.name,
                    "pd": pd,
                    "range_km": self._haversine_m(
                        target_lat, target_lon,
                        threat.latitude, threat.longitude,
                    ) / 1000.0,
                })

        # Aggregate (probability of detection by at least one)
        if individual_pds:
            prob_miss_all = 1.0
            for pd in individual_pds:
                prob_miss_all *= (1.0 - pd)
            aggregate_pd = 1.0 - prob_miss_all
        else:
            aggregate_pd = 0.0

        # Classify alert level
        alert_level = self._classify_alert_level(aggregate_pd)

        # Find dominant threat
        dominant = None
        if threat_details:
            dominant = max(threat_details, key=lambda x: x["pd"])

        return {
            "aggregate_pd": aggregate_pd,
            "num_threats_detecting": len(individual_pds),
            "alert_level": alert_level,
            "dominant_threat": dominant,
            "threat_details": threat_details,
        }

    def _visual_detection_pd(
        self,
        threat: ThreatSystem,
        target_lat: float,
        target_lon: float,
        target_alt_m: float,
    ) -> float:
        """Calculate visual detection probability."""
        range_m = self._haversine_m(
            target_lat, target_lon,
            threat.latitude, threat.longitude,
        )

        range_km = range_m / 1000.0

        # Visual detection range depends on conditions
        max_visual_range_km = 5.0  # Good visibility

        if range_km > max_visual_range_km:
            return 0.0

        # Altitude factor (higher = easier to see against sky)
        alt_factor = min(1.0, target_alt_m / 1000.0)

        # Range factor
        range_factor = 1.0 - (range_km / max_visual_range_km) ** 2

        return range_factor * alt_factor * 0.5  # Max 50% for visual

    def _classify_alert_level(self, pd: float) -> AlertLevel:
        """Classify alert level based on detection probability."""
        if pd < 0.1:
            return AlertLevel.NONE
        elif pd < 0.3:
            return AlertLevel.AWARENESS
        elif pd < 0.5:
            return AlertLevel.CAUTION
        elif pd < 0.7:
            return AlertLevel.WARNING
        elif pd < 0.9:
            return AlertLevel.CRITICAL
        else:
            return AlertLevel.LETHAL

    @staticmethod
    def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance in meters."""
        R = 6371000.0
        lat1_rad, lat2_rad = math.radians(lat1), math.radians(lat2)
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = (math.sin(dlat/2)**2 +
             math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2)
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

    @staticmethod
    def _bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate bearing from point 1 to point 2."""
        lat1_rad, lat2_rad = math.radians(lat1), math.radians(lat2)
        dlon = math.radians(lon2 - lon1)
        x = math.sin(dlon) * math.cos(lat2_rad)
        y = (math.cos(lat1_rad) * math.sin(lat2_rad) -
             math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(dlon))
        return (math.degrees(math.atan2(x, y)) + 360) % 360


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def calculate_detection_range(
    radar: RadarSpecification,
    rcs_sqm: float = 0.5,
    pd_required: float = 0.9,
) -> float:
    """
    Calculate maximum detection range for given RCS and P_d.

    Args:
        radar: Radar specification
        rcs_sqm: Target RCS in m²
        pd_required: Required detection probability

    Returns:
        Detection range in km
    """
    model = RadarDetectionModel()
    return model._detection_range(radar, rcs_sqm, pd_required)


def calculate_radar_horizon(
    radar_altitude_m: float,
    target_altitude_m: float,
) -> float:
    """
    Calculate radar horizon range.

    Args:
        radar_altitude_m: Radar altitude MSL
        target_altitude_m: Target altitude MSL

    Returns:
        Horizon range in km
    """
    model = RadarDetectionModel()
    range_m = model._radar_horizon_range(radar_altitude_m, target_altitude_m)
    return range_m / 1000.0


def create_detection_model(
    rcs_profile: RCSProfile | None = None,
    swerling_case: SwerlingCase = SwerlingCase.CASE_I,
    pfa: float = 1e-6,
) -> RadarDetectionModel:
    """
    Create configured detection model.

    Args:
        rcs_profile: Target RCS profile (uses eVTOL default if None)
        swerling_case: Target fluctuation model
        pfa: Probability of false alarm

    Returns:
        Configured RadarDetectionModel
    """
    config = DetectionModelConfig(
        rcs_profile=rcs_profile or EVTOL_RCS_PROFILE,
        swerling_case=swerling_case,
        probability_false_alarm=pfa,
    )
    return RadarDetectionModel(config)


def create_detection_aggregator() -> DetectionAggregator:
    """Create detection aggregator with default configuration."""
    return DetectionAggregator()
