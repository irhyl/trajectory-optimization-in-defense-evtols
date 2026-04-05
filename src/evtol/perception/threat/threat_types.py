"""
Threat Type Definitions for Defense eVTOL Trajectory Optimization.

This module provides comprehensive type definitions for modeling military
threat systems relevant to low-altitude eVTOL operations in contested
airspace. Based on open-source defense literature and academic research.

Threat Taxonomy:
    1. Surface-to-Air Missiles (SAM)
        - Strategic/Theater (S-300, S-400, HQ-9)
        - Medium Range (SA-6, SA-11, HQ-16)
        - Short Range (SA-15, SA-22, HQ-7)
        - MANPADS (SA-18, FN-6, QW-2)

    2. Anti-Aircraft Artillery (AAA)
        - Radar-guided (ZSU-23-4, Type 95)
        - Optically-guided (ZU-23-2)

    3. Radar Systems
        - Early Warning (P-18, JY-26)
        - Acquisition (ST-68, YLC-8)
        - Fire Control (SNR-125, HT-233)

    4. Electronic Warfare
        - Jammers (ground-based, airborne)
        - GPS denial zones

    5. Hostile Aircraft
        - Fighters (intercept capability)
        - Attack helicopters
        - Combat UAVs

Mathematical Framework:
    - Detection: P_d = f(RCS, range, altitude, aspect, terrain)
    - Engagement: P_e = f(weapon_envelope, reaction_time, tracking)
    - Kill: P_k = f(warhead, fuze, target_vulnerability, countermeasures)
    - Survival: P_s = 1 - P_d × P_e × P_k

References:
    [1] Kopp, C. "SAM System Lethality Analysis", Air Power Australia
    [2] Jane's Land-Based Air Defence
    [3] SIPRI Arms Transfer Database
    [4] Missile Defense Advocacy Alliance
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto


# ENUMERATIONS

class ThreatCategory(Enum):
    """Primary threat system categories."""

    SAM_STRATEGIC = auto()      # S-300, S-400, HQ-9 (long range)
    SAM_MEDIUM = auto()         # SA-6, SA-11, HQ-16 (medium range)
    SAM_SHORT = auto()          # SA-15, SA-22, HQ-7 (short range)
    SAM_MANPADS = auto()        # Man-portable (SA-18, FN-6)
    AAA_RADAR = auto()          # Radar-guided AAA
    AAA_OPTICAL = auto()        # Optically-guided AAA
    RADAR_EW = auto()           # Early warning radar
    RADAR_ACQ = auto()          # Acquisition radar
    RADAR_FC = auto()           # Fire control radar
    EW_JAMMER = auto()          # Electronic jammer
    EW_GPS_DENIAL = auto()      # GPS denial/spoofing
    AIRCRAFT_FIGHTER = auto()   # Fighter aircraft
    AIRCRAFT_HELI = auto()      # Attack helicopter
    AIRCRAFT_UAV = auto()       # Combat UAV
    GROUND_OBSERVER = auto()    # Visual observation post
    ACOUSTIC_SENSOR = auto()    # Acoustic detection


class ThreatMobility(Enum):
    """Mobility classification of threat systems."""

    FIXED = auto()              # Permanent installation
    SEMI_MOBILE = auto()        # Relocatable (hours)
    MOBILE = auto()             # Vehicle-mounted (minutes)
    MAN_PORTABLE = auto()       # Carried by personnel
    TOWED = auto()              # Towed by vehicle


class ThreatStatus(Enum):
    """Operational status of threat system."""

    ACTIVE = auto()             # Currently operational
    DORMANT = auto()            # Present but inactive
    SUSPECTED = auto()          # Intelligence-based, unconfirmed
    DESTROYED = auto()          # Neutralized
    UNKNOWN = auto()            # Status uncertain


class RadarBand(Enum):
    """Radar frequency bands (IEEE designation)."""

    HF = auto()       # 3-30 MHz (OTH radar)
    VHF = auto()      # 30-300 MHz (counter-stealth)
    UHF = auto()      # 300 MHz - 1 GHz
    L = auto()        # 1-2 GHz (long range search)
    S = auto()        # 2-4 GHz (medium range, weather)
    C = auto()        # 4-8 GHz (weather, tracking)
    X = auto()        # 8-12 GHz (fire control, tracking)
    KU = auto()       # 12-18 GHz (high resolution)
    K = auto()        # 18-27 GHz
    KA = auto()       # 27-40 GHz (very high resolution)


class GuidanceType(Enum):
    """Missile/weapon guidance methods."""

    COMMAND = auto()            # Command guidance (CLOS)
    SARH = auto()               # Semi-Active Radar Homing
    ARH = auto()                # Active Radar Homing
    IR = auto()                 # Infrared homing
    EO = auto()                 # Electro-optical
    LASER = auto()              # Laser-guided
    GPS_INS = auto()            # GPS/INS guided
    DUAL_MODE = auto()          # Multi-mode seeker


class WarheadType(Enum):
    """Warhead types for lethality calculation."""

    BLAST_FRAG = auto()         # Blast-fragmentation
    CONTINUOUS_ROD = auto()     # Continuous rod
    SHAPED_CHARGE = auto()      # Shaped charge
    HE = auto()                 # High explosive
    PROXIMITY_FRAG = auto()     # Proximity-fuzed fragmentation


class AlertLevel(Enum):
    """Threat alert levels for trajectory planning."""

    NONE = 0            # No threat
    AWARENESS = 1       # Long-range detection possible
    CAUTION = 2         # Within acquisition range
    WARNING = 3         # Tracking/engagement possible
    CRITICAL = 4        # Imminent engagement
    LETHAL = 5          # Inside lethal envelope


class CountermeasureType(Enum):
    """Available countermeasures."""

    CHAFF = auto()
    FLARE = auto()
    ECM_NOISE = auto()
    ECM_DECEPTIVE = auto()
    DIRCM = auto()              # Directed IR countermeasure
    MANEUVER = auto()
    TERRAIN_MASKING = auto()
    NOE_FLIGHT = auto()         # Nap-of-earth


class ThreatPriority(Enum):
    """Priority classification for threat systems affecting route planning."""

    CRITICAL = auto()   # Must be avoided at all costs (strategic SAM)
    HIGH = auto()       # High priority avoidance (medium SAM)
    MEDIUM = auto()     # Standard threat (short range SAM, AAA)
    LOW = auto()        # Low priority (MANPADS, observers)
    MINIMAL = auto()    # Minimal impact on routing


class RadarType(Enum):
    """Radar technology classification."""

    PHASED_ARRAY = auto()       # Active/Passive Electronically Scanned Array
    PULSE_DOPPLER = auto()      # Pulse-Doppler tracking
    FIRE_CONTROL = auto()       # Fire control radar
    VHF_ARRAY = auto()          # VHF band array (counter-stealth)
    PASSIVE = auto()            # Passive detection (IR, acoustic)
    CW = auto()                 # Continuous wave
    MECHANICALLY_SCANNED = auto()  # Mechanically scanned dish


class WeaponType(Enum):
    """Weapon system classification."""

    SURFACE_TO_AIR_MISSILE = auto()  # SAM systems
    AAA_GUN = auto()                 # Anti-aircraft artillery
    MANPADS = auto()                 # Man-portable air defense
    CIWS = auto()                    # Close-in weapon system
    DIRECTED_ENERGY = auto()         # Laser/HPM weapons


# =============================================================================
# RADAR CROSS SECTION (RCS)
# =============================================================================

@dataclass
class RCSProfile:
    """
    Radar Cross Section profile for the eVTOL platform.

    RCS varies with:
        - Aspect angle (nose-on, broadside, tail-on)
        - Frequency band
        - Polarization

    Values in square meters (m²).
    """

    # Nominal RCS by aspect (m²)
    nose_on: float = 0.1        # Front aspect
    broadside: float = 1.0      # Side aspect (maximum)
    tail_on: float = 0.3        # Rear aspect
    top: float = 2.0            # Top-down (relevant for look-down radars)
    bottom: float = 1.5         # Bottom-up

    # RCS variation by frequency band (multipliers)
    band_factors: dict[RadarBand, float] = field(default_factory=lambda: {
        RadarBand.VHF: 2.0,     # VHF sees larger RCS (Rayleigh region)
        RadarBand.UHF: 1.5,
        RadarBand.L: 1.2,
        RadarBand.S: 1.0,       # Reference
        RadarBand.C: 0.9,
        RadarBand.X: 0.8,
        RadarBand.KU: 0.7,
    })

    def get_rcs(
        self,
        azimuth_deg: float,
        elevation_deg: float,
        band: RadarBand = RadarBand.S,
    ) -> float:
        """
        Calculate RCS for given aspect angles and frequency.

        Args:
            azimuth_deg: Azimuth angle from nose (0=nose, 90=side, 180=tail)
            elevation_deg: Elevation angle (0=level, +90=above, -90=below)
            band: Radar frequency band

        Returns:
            RCS in m²
        """
        # Normalize azimuth to 0-180
        az = abs(azimuth_deg) % 360
        if az > 180:
            az = 360 - az

        # Interpolate between aspects
        if az < 30:
            # Nose region
            base_rcs = self.nose_on
        elif az < 60:
            # Nose-to-side transition
            t = (az - 30) / 30
            base_rcs = self.nose_on * (1 - t) + self.broadside * t
        elif az < 120:
            # Broadside region
            base_rcs = self.broadside
        elif az < 150:
            # Side-to-tail transition
            t = (az - 120) / 30
            base_rcs = self.broadside * (1 - t) + self.tail_on * t
        else:
            # Tail region
            base_rcs = self.tail_on

        # Elevation factor (top/bottom contribution)
        el = abs(elevation_deg)
        if el > 60:
            el_factor = 1.5  # Looking down/up sees larger RCS
        elif el > 30:
            el_factor = 1.0 + 0.5 * (el - 30) / 30
        else:
            el_factor = 1.0

        # Frequency band factor
        band_factor = self.band_factors.get(band, 1.0)

        return base_rcs * el_factor * band_factor


# eVTOL platform RCS (composite structure, moderate signature)
EVTOL_RCS_PROFILE = RCSProfile(
    nose_on=0.05,       # Small front aspect
    broadside=0.8,      # Side aspect (rotors visible)
    tail_on=0.2,
    top=1.5,            # Rotor disc visible from above
    bottom=1.0,
)


# =============================================================================
# RADAR SYSTEM SPECIFICATIONS
# =============================================================================

@dataclass
class RadarSpecification:
    """
    Radar system technical specifications.

    Based on radar range equation:
        R_max = [(P_t × G² × λ² × σ) / ((4π)³ × P_min)]^(1/4)

    Where:
        P_t = Transmitted power
        G = Antenna gain
        λ = Wavelength
        σ = Target RCS
        P_min = Minimum detectable signal
    """

    # Basic parameters
    name: str = "Generic Radar"
    band: RadarBand = RadarBand.S
    frequency_ghz: float = 3.0

    # Power and sensitivity
    peak_power_kw: float = 100.0
    average_power_kw: float = 5.0
    antenna_gain_db: float = 30.0
    noise_figure_db: float = 3.0
    min_detectable_signal_dbm: float = -110.0

    # Angular coverage
    azimuth_coverage_deg: float = 360.0
    azimuth_beamwidth_deg: float = 3.0
    elevation_min_deg: float = -3.0
    elevation_max_deg: float = 70.0
    elevation_beamwidth_deg: float = 10.0

    # Tracking capability
    max_tracks: int = 100
    track_update_rate_hz: float = 1.0

    # Performance factors
    integration_gain_db: float = 10.0     # Coherent integration
    clutter_rejection_db: float = 30.0    # MTI/Doppler processing

    # Derived maximum range for reference RCS (m²) at 1 m²
    max_range_km_1sqm: float = 100.0

    @property
    def wavelength_m(self) -> float:
        """Calculate wavelength from frequency."""
        c = 299792458.0  # Speed of light m/s
        return c / (self.frequency_ghz * 1e9)

    def detection_range_km(
        self,
        rcs_sqm: float,
        probability_detection: float = 0.9,
    ) -> float:
        """
        Calculate detection range using radar range equation.

        Detection range scales as RCS^(1/4):
            R = R_ref × (σ / σ_ref)^(1/4)

        Args:
            rcs_sqm: Target RCS in m²
            probability_detection: Required P_d (affects range)

        Returns:
            Detection range in km
        """
        # Reference: max_range_km_1sqm is for 1 m² RCS
        rcs_factor = (rcs_sqm / 1.0) ** 0.25

        # P_d factor (lower P_d allows longer range)
        # Swerling I/II: SNR increases ~3 dB per decade of P_d
        pd_factor = 1.0
        if probability_detection < 0.9:
            pd_factor = 1.1
        elif probability_detection > 0.95:
            pd_factor = 0.85

        return self.max_range_km_1sqm * rcs_factor * pd_factor


# =============================================================================
# WEAPON SYSTEM SPECIFICATIONS
# =============================================================================

@dataclass
class MissileSpecification:
    """Surface-to-air missile technical specifications."""

    name: str = "Generic SAM"

    # Range envelope
    max_range_km: float = 50.0
    min_range_km: float = 1.0
    max_altitude_m: float = 25000.0
    min_altitude_m: float = 30.0

    # Performance
    max_speed_mach: float = 3.0
    max_g_capability: float = 30.0
    flight_time_max_s: float = 60.0

    # Guidance
    guidance: GuidanceType = GuidanceType.SARH
    terminal_guidance: GuidanceType | None = None

    # Warhead
    warhead_type: WarheadType = WarheadType.BLAST_FRAG
    warhead_kg: float = 70.0
    lethal_radius_m: float = 20.0

    # Kill probability
    single_shot_pk: float = 0.7
    pk_vs_maneuvering: float = 0.5     # Against maneuvering target
    pk_with_ecm: float = 0.4            # Against ECM-equipped target

    # Reaction time
    reaction_time_s: float = 8.0        # Detection to launch
    reload_time_s: float = 30.0


@dataclass
class AAASpecification:
    """Anti-aircraft artillery specifications."""

    name: str = "Generic AAA"
    caliber_mm: float = 23.0

    # Range
    max_range_m: float = 2500.0
    effective_range_m: float = 1500.0
    max_altitude_m: float = 2000.0

    # Rate of fire
    rate_of_fire_rpm: float = 800.0
    burst_length_rounds: int = 50

    # Fire control
    radar_guided: bool = False
    optical_tracking: bool = True

    # Lethality
    pk_per_round: float = 0.001         # Per-round kill probability
    pk_per_burst: float = 0.15          # Per-burst kill probability


# =============================================================================
# THREAT SYSTEM DEFINITIONS
# =============================================================================

@dataclass
class ThreatEnvelope:
    """
    Three-dimensional threat engagement envelope.

    The envelope defines where a threat system can:
        1. Detect targets
        2. Track targets
        3. Engage targets (fire solution)
        4. Achieve lethal hits
    """

    # Horizontal range (km)
    detection_range_km: float = 100.0
    tracking_range_km: float = 80.0
    engagement_range_km: float = 50.0
    lethal_range_km: float = 40.0

    # Minimum range (dead zone)
    min_range_km: float = 0.5

    # Altitude limits (m)
    min_altitude_m: float = 30.0
    max_altitude_m: float = 25000.0

    # Azimuth coverage (degrees)
    azimuth_start_deg: float = 0.0      # Relative to system heading
    azimuth_end_deg: float = 360.0      # 360 = full coverage

    # Elevation limits (degrees)
    elevation_min_deg: float = -5.0
    elevation_max_deg: float = 85.0

    def contains_point(
        self,
        range_km: float,
        altitude_m: float,
        azimuth_deg: float,
        envelope_type: str = "engagement",
    ) -> bool:
        """
        Check if a point is within the specified envelope.

        Args:
            range_km: Horizontal range from threat
            altitude_m: Target altitude MSL
            azimuth_deg: Azimuth from threat (relative to heading)
            envelope_type: "detection", "tracking", "engagement", or "lethal"

        Returns:
            True if point is within envelope
        """
        # Select range based on envelope type
        max_range = {
            "detection": self.detection_range_km,
            "tracking": self.tracking_range_km,
            "engagement": self.engagement_range_km,
            "lethal": self.lethal_range_km,
        }.get(envelope_type, self.engagement_range_km)

        # Range check
        if range_km > max_range or range_km < self.min_range_km:
            return False

        # Altitude check
        if altitude_m < self.min_altitude_m or altitude_m > self.max_altitude_m:
            return False

        # Azimuth check
        az = azimuth_deg % 360
        if self.azimuth_end_deg >= self.azimuth_start_deg:
            # Normal case
            if az < self.azimuth_start_deg or az > self.azimuth_end_deg:
                return False
        else:
            # Wrapping case (e.g., 270-90 covers north)
            if az < self.azimuth_start_deg and az > self.azimuth_end_deg:
                return False

        return True


@dataclass
class ThreatSystem:
    """
    Complete threat system definition.

    This is the primary dataclass for representing any threat system
    in the operational environment.
    """

    # Identity
    threat_id: str = field(default_factory=lambda: f"THR-{uuid.uuid4().hex[:8].upper()}")
    name: str = "Unknown Threat"
    designator: str = ""                # NATO designator (e.g., SA-11)

    # Classification
    category: ThreatCategory = ThreatCategory.SAM_MEDIUM
    mobility: ThreatMobility = ThreatMobility.MOBILE
    status: ThreatStatus = ThreatStatus.ACTIVE

    # Position (WGS84)
    latitude: float = 0.0
    longitude: float = 0.0
    altitude_m: float = 0.0             # System altitude MSL
    heading_deg: float = 0.0            # System orientation (for sector coverage)

    # Envelope
    envelope: ThreatEnvelope = field(default_factory=ThreatEnvelope)

    # System specifications
    radar: RadarSpecification | None = None
    missile: MissileSpecification | None = None
    aaa: AAASpecification | None = None

    # Reaction characteristics
    reaction_time_s: float = 10.0       # Time from detection to engagement
    salvo_size: int = 2                 # Missiles per engagement
    reload_time_s: float = 30.0         # Time to reload
    simultaneous_engagements: int = 2   # Max concurrent engagements

    # Kill probability modifiers
    base_pk: float = 0.7
    pk_altitude_factor: float = 1.0     # Higher altitude = lower Pk for some systems
    pk_aspect_factor: float = 1.0       # Aspect angle effect
    pk_ecm_factor: float = 0.5          # Pk when ECM active

    # Confidence and intelligence
    confidence: float = 0.9             # 0-1, intelligence confidence
    last_confirmed: datetime | None = None
    source: str = "INTEL"               # Intelligence source

    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    notes: str = ""

    def detection_probability(
        self,
        range_km: float,
        altitude_m: float,
        rcs_sqm: float = 0.5,
        terrain_factor: float = 1.0,
    ) -> float:
        """
        Calculate probability of detection.

        P_d = f(SNR) where SNR depends on range, RCS, and radar parameters.
        Simplified model using detection range as reference.

        Args:
            range_km: Horizontal range to target
            altitude_m: Target altitude
            rcs_sqm: Target RCS
            terrain_factor: 0-1, terrain masking effect (0=masked, 1=clear)

        Returns:
            Detection probability 0-1
        """
        if self.radar is None:
            # Non-radar threat (visual, acoustic)
            if self.category in [ThreatCategory.GROUND_OBSERVER]:
                # Visual detection (range-limited)
                visual_range_km = 5.0
                if range_km > visual_range_km:
                    return 0.0
                return max(0, 1.0 - (range_km / visual_range_km) ** 2) * terrain_factor
            return 0.0

        # Get detection range for this RCS
        det_range_km = self.radar.detection_range_km(rcs_sqm)

        # Altitude effects (low altitude harder to detect)
        alt_factor = 1.0
        if altitude_m < 100:
            alt_factor = 0.5 + 0.5 * (altitude_m / 100)
        elif altitude_m < 500:
            alt_factor = 0.9

        # Calculate P_d based on range ratio
        if range_km > det_range_km:
            return 0.0

        # SNR-based model: P_d increases as range decreases
        range_ratio = range_km / det_range_km

        # Swerling I detection model approximation
        pd = 1.0 - math.exp(-3 * (1 - range_ratio ** 4))

        return min(1.0, max(0.0, pd * alt_factor * terrain_factor))

    def engagement_probability(
        self,
        range_km: float,
        altitude_m: float,
        target_speed_ms: float = 50.0,
        target_maneuvering: bool = False,
    ) -> float:
        """
        Calculate probability of successful engagement.

        Args:
            range_km: Horizontal range
            altitude_m: Target altitude
            target_speed_ms: Target speed in m/s
            target_maneuvering: Is target performing evasive maneuvers

        Returns:
            Engagement probability 0-1
        """
        # Check if within engagement envelope
        if not self.envelope.contains_point(range_km, altitude_m, 0, "engagement"):
            return 0.0

        # Base engagement probability
        pe = 0.9  # Assume good tracking

        # Range factor (outer edge of envelope is harder)
        range_ratio = range_km / self.envelope.engagement_range_km
        pe *= 1.0 - 0.3 * range_ratio

        # Speed factor (faster targets harder)
        if target_speed_ms > 100:
            pe *= 0.9
        if target_speed_ms > 200:
            pe *= 0.8

        # Maneuver factor
        if target_maneuvering:
            pe *= 0.7

        return max(0.0, min(1.0, pe))

    def kill_probability(
        self,
        range_km: float,
        altitude_m: float,
        target_maneuvering: bool = False,
        ecm_active: bool = False,
    ) -> float:
        """
        Calculate single-shot kill probability.

        Args:
            range_km: Horizontal range
            altitude_m: Target altitude
            target_maneuvering: Evasive maneuvers active
            ecm_active: Electronic countermeasures active

        Returns:
            Kill probability 0-1
        """
        pk = self.base_pk

        # Altitude factor (some systems less effective at low altitude)
        if altitude_m < 100:
            pk *= 0.7  # Low altitude clutter/masking
        elif altitude_m < 500:
            pk *= 0.9

        # Range factor
        if range_km > self.envelope.lethal_range_km * 0.8:
            pk *= 0.8  # Edge of envelope

        # Maneuver factor
        if target_maneuvering and self.missile:
            pk *= self.missile.pk_vs_maneuvering / self.missile.single_shot_pk

        # ECM factor
        if ecm_active:
            pk *= self.pk_ecm_factor

        return max(0.0, min(1.0, pk))

    def cumulative_pk(
        self,
        num_missiles: int,
        single_shot_pk: float,
    ) -> float:
        """
        Calculate cumulative Pk for multiple shots.

        P_k_cum = 1 - (1 - P_k)^n
        """
        return 1.0 - (1.0 - single_shot_pk) ** num_missiles


# =============================================================================
# THREAT SCENARIO CONFIGURATION
# =============================================================================

@dataclass
class ThreatScenarioConfig:
    """Configuration for threat scenario generation."""

    # Scenario identity
    scenario_id: str = field(default_factory=lambda: f"SCN-{uuid.uuid4().hex[:8].upper()}")
    name: str = "Unnamed Scenario"
    description: str = ""

    # Geographic bounds
    center_lat: float = 28.6
    center_lon: float = 77.2
    radius_km: float = 200.0

    # Threat density
    num_strategic_sam: int = 2
    num_medium_sam: int = 5
    num_short_sam: int = 10
    num_manpads: int = 20
    num_aaa: int = 15
    num_radars: int = 8
    num_observers: int = 30

    # Operational theater
    theater: str = "WESTERN"  # WESTERN, NORTHERN, EASTERN, etc.

    # Time validity
    valid_from: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    valid_until: datetime | None = None


@dataclass
class ThreatAssessment:
    """Assessment of threat exposure for a trajectory point."""

    position_lat: float
    position_lon: float
    position_alt_m: float

    # Cumulative probabilities
    probability_detection: float = 0.0
    probability_engagement: float = 0.0
    probability_kill: float = 0.0
    probability_survival: float = 1.0

    # Contributing threats
    threats_in_detection_range: int = 0
    threats_in_engagement_range: int = 0
    threats_in_lethal_range: int = 0

    # Dominant threat
    primary_threat_id: str | None = None
    primary_threat_distance_km: float = float('inf')

    # Alert level
    alert_level: AlertLevel = AlertLevel.NONE

    # Time in zone
    time_in_detection_s: float = 0.0
    time_in_engagement_s: float = 0.0
    time_in_lethal_s: float = 0.0


@dataclass
class ThreatFieldConfig:
    """Configuration for threat field generation."""

    # Grid parameters
    lat_min: float = 28.0
    lat_max: float = 29.0
    lon_min: float = 76.5
    lon_max: float = 77.5

    # Resolution
    lat_resolution: float = 0.01        # ~1.1 km
    lon_resolution: float = 0.01        # ~1.0 km at this latitude
    alt_levels_m: list[float] = field(default_factory=lambda: [
        50, 100, 200, 300, 500, 1000, 2000, 3000
    ])

    # Computation parameters
    rcs_sqm: float = 0.5                # Assumed platform RCS
    include_terrain_masking: bool = True

    # Output
    output_format: str = "numpy"        # numpy, geotiff, json


# =============================================================================
# PREDEFINED THREAT SYSTEM TEMPLATES
# =============================================================================

def create_sa6_gainful() -> ThreatSystem:
    """SA-6 Gainful (2K12 Kub) - Medium range SAM."""
    return ThreatSystem(
        name="SA-6 Gainful",
        designator="SA-6",
        category=ThreatCategory.SAM_MEDIUM,
        mobility=ThreatMobility.MOBILE,
        envelope=ThreatEnvelope(
            detection_range_km=75,
            tracking_range_km=60,
            engagement_range_km=24,
            lethal_range_km=20,
            min_range_km=4,
            min_altitude_m=50,
            max_altitude_m=14000,
        ),
        radar=RadarSpecification(
            name="1S91 Straight Flush",
            band=RadarBand.C,
            frequency_ghz=5.0,
            max_range_km_1sqm=75,
        ),
        missile=MissileSpecification(
            name="3M9",
            max_range_km=24,
            min_range_km=4,
            max_altitude_m=14000,
            min_altitude_m=50,
            max_speed_mach=2.8,
            guidance=GuidanceType.SARH,
            single_shot_pk=0.55,
        ),
        reaction_time_s=8,
        base_pk=0.55,
    )


def create_sa11_gadfly() -> ThreatSystem:
    """SA-11 Gadfly (9K37 Buk) - Medium range SAM."""
    return ThreatSystem(
        name="SA-11 Gadfly",
        designator="SA-11",
        category=ThreatCategory.SAM_MEDIUM,
        mobility=ThreatMobility.MOBILE,
        envelope=ThreatEnvelope(
            detection_range_km=100,
            tracking_range_km=85,
            engagement_range_km=35,
            lethal_range_km=30,
            min_range_km=3,
            min_altitude_m=25,
            max_altitude_m=22000,
        ),
        radar=RadarSpecification(
            name="9S35 Fire Dome",
            band=RadarBand.X,
            frequency_ghz=10.0,
            max_range_km_1sqm=100,
        ),
        missile=MissileSpecification(
            name="9M38",
            max_range_km=35,
            min_range_km=3,
            max_altitude_m=22000,
            min_altitude_m=25,
            max_speed_mach=3.0,
            guidance=GuidanceType.SARH,
            single_shot_pk=0.70,
        ),
        reaction_time_s=6,
        base_pk=0.70,
    )


def create_sa15_gauntlet() -> ThreatSystem:
    """SA-15 Gauntlet (9K330 Tor) - Short range SAM."""
    return ThreatSystem(
        name="SA-15 Gauntlet",
        designator="SA-15",
        category=ThreatCategory.SAM_SHORT,
        mobility=ThreatMobility.MOBILE,
        envelope=ThreatEnvelope(
            detection_range_km=25,
            tracking_range_km=20,
            engagement_range_km=12,
            lethal_range_km=10,
            min_range_km=1,
            min_altitude_m=10,
            max_altitude_m=6000,
        ),
        radar=RadarSpecification(
            name="Tor Fire Control",
            band=RadarBand.KU,
            frequency_ghz=15.0,
            max_range_km_1sqm=25,
        ),
        missile=MissileSpecification(
            name="9M330",
            max_range_km=12,
            min_range_km=1,
            max_altitude_m=6000,
            min_altitude_m=10,
            max_speed_mach=2.5,
            guidance=GuidanceType.COMMAND,
            single_shot_pk=0.75,
        ),
        reaction_time_s=5,
        base_pk=0.75,
    )


def create_sa18_igla() -> ThreatSystem:
    """SA-18 Grouse (9K38 Igla) - MANPADS."""
    return ThreatSystem(
        name="SA-18 Grouse",
        designator="SA-18",
        category=ThreatCategory.SAM_MANPADS,
        mobility=ThreatMobility.MAN_PORTABLE,
        envelope=ThreatEnvelope(
            detection_range_km=8,
            tracking_range_km=5,
            engagement_range_km=5.2,
            lethal_range_km=4.5,
            min_range_km=0.5,
            min_altitude_m=10,
            max_altitude_m=3500,
        ),
        radar=None,  # Passive IR
        missile=MissileSpecification(
            name="9M39",
            max_range_km=5.2,
            min_range_km=0.5,
            max_altitude_m=3500,
            min_altitude_m=10,
            max_speed_mach=1.9,
            guidance=GuidanceType.IR,
            single_shot_pk=0.45,
        ),
        reaction_time_s=3,
        base_pk=0.45,
    )


def create_hq9() -> ThreatSystem:
    """HQ-9 (Chinese) - Long range strategic SAM."""
    return ThreatSystem(
        name="HQ-9",
        designator="HQ-9",
        category=ThreatCategory.SAM_STRATEGIC,
        mobility=ThreatMobility.SEMI_MOBILE,
        envelope=ThreatEnvelope(
            detection_range_km=300,
            tracking_range_km=200,
            engagement_range_km=125,
            lethal_range_km=100,
            min_range_km=6,
            min_altitude_m=25,
            max_altitude_m=27000,
        ),
        radar=RadarSpecification(
            name="HT-233 PESA",
            band=RadarBand.C,
            frequency_ghz=5.5,
            max_range_km_1sqm=300,
        ),
        missile=MissileSpecification(
            name="HQ-9",
            max_range_km=125,
            min_range_km=6,
            max_altitude_m=27000,
            min_altitude_m=25,
            max_speed_mach=4.2,
            guidance=GuidanceType.ARH,
            single_shot_pk=0.80,
        ),
        reaction_time_s=10,
        base_pk=0.80,
    )


def create_hq16() -> ThreatSystem:
    """HQ-16 (Chinese) - Medium range SAM."""
    return ThreatSystem(
        name="HQ-16",
        designator="HQ-16",
        category=ThreatCategory.SAM_MEDIUM,
        mobility=ThreatMobility.MOBILE,
        envelope=ThreatEnvelope(
            detection_range_km=85,
            tracking_range_km=70,
            engagement_range_km=40,
            lethal_range_km=35,
            min_range_km=1.5,
            min_altitude_m=15,
            max_altitude_m=18000,
        ),
        radar=RadarSpecification(
            name="HQ-16 PESA",
            band=RadarBand.X,
            frequency_ghz=9.0,
            max_range_km_1sqm=85,
        ),
        missile=MissileSpecification(
            name="HQ-16",
            max_range_km=40,
            min_range_km=1.5,
            max_altitude_m=18000,
            min_altitude_m=15,
            max_speed_mach=3.5,
            guidance=GuidanceType.SARH,
            single_shot_pk=0.72,
        ),
        reaction_time_s=6,
        base_pk=0.72,
    )


def create_fn6() -> ThreatSystem:
    """FN-6 (Chinese) - MANPADS."""
    return ThreatSystem(
        name="FN-6",
        designator="FN-6",
        category=ThreatCategory.SAM_MANPADS,
        mobility=ThreatMobility.MAN_PORTABLE,
        envelope=ThreatEnvelope(
            detection_range_km=6,
            tracking_range_km=5,
            engagement_range_km=5.5,
            lethal_range_km=5.0,
            min_range_km=0.5,
            min_altitude_m=10,
            max_altitude_m=3800,
        ),
        radar=None,
        missile=MissileSpecification(
            name="FN-6",
            max_range_km=5.5,
            min_range_km=0.5,
            max_altitude_m=3800,
            min_altitude_m=10,
            max_speed_mach=2.0,
            guidance=GuidanceType.IR,
            single_shot_pk=0.50,
        ),
        reaction_time_s=3,
        base_pk=0.50,
    )


def create_zsu23_4() -> ThreatSystem:
    """ZSU-23-4 Shilka - Radar-guided AAA."""
    return ThreatSystem(
        name="ZSU-23-4 Shilka",
        designator="ZSU-23-4",
        category=ThreatCategory.AAA_RADAR,
        mobility=ThreatMobility.MOBILE,
        envelope=ThreatEnvelope(
            detection_range_km=20,
            tracking_range_km=8,
            engagement_range_km=2.5,
            lethal_range_km=2.0,
            min_range_km=0.0,
            min_altitude_m=0,
            max_altitude_m=1500,
        ),
        radar=RadarSpecification(
            name="RPK-2 Tobol",
            band=RadarBand.KU,
            frequency_ghz=14.0,
            max_range_km_1sqm=20,
        ),
        aaa=AAASpecification(
            name="AZP-23",
            caliber_mm=23,
            max_range_m=2500,
            effective_range_m=1500,
            max_altitude_m=1500,
            rate_of_fire_rpm=3400,  # 4 barrels × 850
            radar_guided=True,
            pk_per_burst=0.20,
        ),
        reaction_time_s=2,
        base_pk=0.20,
    )


# Template factory
THREAT_TEMPLATES: dict[str, callable] = {
    "SA-6": create_sa6_gainful,
    "SA-11": create_sa11_gadfly,
    "SA-15": create_sa15_gauntlet,
    "SA-18": create_sa18_igla,
    "HQ-9": create_hq9,
    "HQ-16": create_hq16,
    "FN-6": create_fn6,
    "ZSU-23-4": create_zsu23_4,
}


def create_threat_from_template(
    designator: str,
    latitude: float,
    longitude: float,
    heading_deg: float = 0.0,
    status: ThreatStatus = ThreatStatus.ACTIVE,
) -> ThreatSystem:
    """
    Create a positioned threat from a template.

    Args:
        designator: Template designator (e.g., "SA-11")
        latitude: Position latitude
        longitude: Position longitude
        heading_deg: System orientation
        status: Operational status

    Returns:
        Configured ThreatSystem
    """
    if designator not in THREAT_TEMPLATES:
        raise ValueError(f"Unknown threat template: {designator}")

    threat = THREAT_TEMPLATES[designator]()
    threat.latitude = latitude
    threat.longitude = longitude
    threat.heading_deg = heading_deg
    threat.status = status

    return threat
