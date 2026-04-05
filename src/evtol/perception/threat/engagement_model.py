"""
Engagement Model for Defense eVTOL Trajectory Optimization.

This module implements weapon engagement modeling including:
    - Engagement envelope geometry
    - Reaction time modeling
    - Kill probability (Pk) calculations
    - Cumulative kill probability for salvos
    - Survivability analysis

Mathematical Framework:

    1. Engagement Envelope:
        - Slant range limits: R_min < R < R_max
        - Altitude limits: h_min < h < h_max
        - Azimuth coverage: θ_start < θ < θ_end
        - Kinematic constraints: V_target, a_target

    2. Kill Probability:
        P_k = P_launch × P_intercept × P_fuze × P_warhead

        Where:
            P_launch = Probability of successful launch
            P_intercept = Probability missile reaches target
            P_fuze = Probability of fuze function
            P_warhead = Probability of lethal damage

    3. Single-Shot Kill Probability (SSKP):
        SSKP = P_k × (1 - P_CM)

        Where P_CM = Countermeasure effectiveness

    4. Cumulative Pk (Multiple Missiles):
        P_k_cum = 1 - ∏(1 - P_k_i)

    5. Engagement Timeline:
        t_total = t_detect + t_track + t_launch + t_flyout

    6. Terminal Guidance:
        Miss distance σ depends on:
            - Seeker accuracy
            - Target maneuver
            - ECM effectiveness

References:
    [1] Kopp, C. "SAM System Lethality"
    [2] Ball, R. "The Fundamentals of Aircraft Combat Survivability"
    [3] Washburn, A. "Search and Detection"
    [4] Nicholas, T. "Fundamentals of Missile Guidance"
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any
from enum import Enum, auto

from .threat_types import (
    ThreatSystem,
    AAASpecification,
    GuidanceType,
    CountermeasureType,
)


# =============================================================================
# ENGAGEMENT STATES
# =============================================================================

class EngagementPhase(Enum):
    """Phases of the engagement sequence."""

    NOT_IN_RANGE = auto()           # Target outside envelope
    DETECTION = auto()              # Target detected
    TRACKING = auto()               # Track established
    FIRE_SOLUTION = auto()          # Launch solution computed
    MISSILE_LAUNCH = auto()         # Missile in flight
    TERMINAL = auto()               # Terminal guidance
    INTERCEPT = auto()              # Intercept attempt
    MISS = auto()                   # Miss
    KILL = auto()                   # Target destroyed


class EngagementResult(Enum):
    """Possible engagement outcomes."""

    NO_ENGAGEMENT = auto()          # Target not engaged
    SUCCESSFUL_INTERCEPT = auto()   # Target destroyed
    MISS_MANEUVER = auto()          # Miss due to target maneuver
    MISS_ECM = auto()               # Miss due to countermeasures
    MISS_GUIDANCE = auto()          # Miss due to guidance error
    MISS_FUZE = auto()              # Miss due to fuze failure
    MISSILE_FAILURE = auto()        # Missile system failure


# =============================================================================
# ENGAGEMENT MODEL CONFIGURATION
# =============================================================================

@dataclass
class EngagementModelConfig:
    """Configuration for engagement calculations."""

    # System reliability
    missile_reliability: float = 0.95      # P(successful launch)
    guidance_reliability: float = 0.98     # P(guidance works)
    fuze_reliability: float = 0.97         # P(fuze functions)

    # Target characteristics
    target_vulnerability: float = 1.0      # 1.0 = nominal
    target_maneuver_g: float = 2.0         # Available g's for evasion
    target_max_speed_ms: float = 50.0      # Max target speed

    # Countermeasures
    chaff_effectiveness: float = 0.3       # P(chaff defeats engagement)
    flare_effectiveness: float = 0.4       # P(flare defeats IR seeker)
    ecm_effectiveness: float = 0.25        # P(ECM defeats engagement)
    maneuver_effectiveness: float = 0.2    # P(maneuver defeats engagement)

    # Engagement parameters
    max_simultaneous_engagements: int = 2
    salvo_doctrine: int = 2                # Missiles per target
    shoot_look_shoot: bool = True          # SLS doctrine

    # Environmental
    weather_factor: float = 1.0            # 1.0 = clear, <1 = degraded


# =============================================================================
# KILL PROBABILITY MODELS
# =============================================================================

@dataclass
class KillProbabilityFactors:
    """Breakdown of factors affecting kill probability."""

    base_pk: float = 0.7
    range_factor: float = 1.0
    altitude_factor: float = 1.0
    aspect_factor: float = 1.0
    maneuver_factor: float = 1.0
    ecm_factor: float = 1.0
    weather_factor: float = 1.0
    reliability_factor: float = 1.0

    @property
    def effective_pk(self) -> float:
        """Calculate effective Pk from all factors."""
        pk = self.base_pk
        pk *= self.range_factor
        pk *= self.altitude_factor
        pk *= self.aspect_factor
        pk *= self.maneuver_factor
        pk *= self.ecm_factor
        pk *= self.weather_factor
        pk *= self.reliability_factor
        return min(max(pk, 0.0), 0.99)  # Cap at 99%


class EngagementModel:
    """
    Complete engagement modeling for SAM and AAA systems.

    Models the full engagement sequence from detection through
    potential intercept, including countermeasure effects.
    """

    def __init__(self, config: EngagementModelConfig | None = None):
        """Initialize engagement model."""
        self.config = config or EngagementModelConfig()

    def calculate_engagement_probability(
        self,
        threat: ThreatSystem,
        target_lat: float,
        target_lon: float,
        target_alt_m: float,
        target_speed_ms: float = 50.0,
        target_heading_deg: float = 0.0,
        target_maneuvering: bool = False,
        ecm_active: bool = False,
        countermeasures: list[CountermeasureType] | None = None,
    ) -> dict[str, Any]:
        """
        Calculate probability of successful engagement.

        Args:
            threat: Threat system attempting engagement
            target_lat: Target latitude
            target_lon: Target longitude
            target_alt_m: Target altitude MSL
            target_speed_ms: Target speed in m/s
            target_heading_deg: Target heading
            target_maneuvering: Is target maneuvering
            ecm_active: Is ECM active
            countermeasures: Active countermeasures

        Returns:
            Engagement assessment dictionary
        """
        countermeasures = countermeasures or []

        # Calculate geometry
        range_km = self._calculate_range_km(
            target_lat, target_lon,
            threat.latitude, threat.longitude,
        )

        bearing = self._calculate_bearing(
            threat.latitude, threat.longitude,
            target_lat, target_lon,
        )
        relative_azimuth = (bearing - threat.heading_deg) % 360

        # Check if in engagement envelope
        in_envelope = threat.envelope.contains_point(
            range_km, target_alt_m, relative_azimuth, "engagement"
        )

        if not in_envelope:
            return {
                "can_engage": False,
                "phase": EngagementPhase.NOT_IN_RANGE,
                "probability_engagement": 0.0,
                "probability_kill": 0.0,
                "time_to_intercept": None,
                "factors": None,
            }

        # Calculate Pk factors
        factors = self._calculate_pk_factors(
            threat=threat,
            range_km=range_km,
            altitude_m=target_alt_m,
            relative_azimuth=relative_azimuth,
            target_speed_ms=target_speed_ms,
            target_maneuvering=target_maneuvering,
            ecm_active=ecm_active,
            countermeasures=countermeasures,
        )

        # Calculate time to intercept
        tti = self._calculate_time_to_intercept(
            threat, range_km, target_alt_m, target_speed_ms
        )

        # Calculate salvo Pk
        single_shot_pk = factors.effective_pk
        salvo_pk = self._calculate_salvo_pk(
            single_shot_pk, threat.salvo_size
        )

        return {
            "can_engage": True,
            "phase": EngagementPhase.FIRE_SOLUTION,
            "probability_engagement": 0.9,  # P(launch given in envelope)
            "single_shot_pk": single_shot_pk,
            "probability_kill": salvo_pk,
            "salvo_size": threat.salvo_size,
            "time_to_intercept": tti,
            "reaction_time": threat.reaction_time_s,
            "factors": {
                "base_pk": factors.base_pk,
                "range_factor": factors.range_factor,
                "altitude_factor": factors.altitude_factor,
                "aspect_factor": factors.aspect_factor,
                "maneuver_factor": factors.maneuver_factor,
                "ecm_factor": factors.ecm_factor,
                "weather_factor": factors.weather_factor,
                "reliability_factor": factors.reliability_factor,
            },
            "range_km": range_km,
            "altitude_m": target_alt_m,
        }

    def _calculate_pk_factors(
        self,
        threat: ThreatSystem,
        range_km: float,
        altitude_m: float,
        relative_azimuth: float,
        target_speed_ms: float,
        target_maneuvering: bool,
        ecm_active: bool,
        countermeasures: list[CountermeasureType],
    ) -> KillProbabilityFactors:
        """Calculate individual Pk factors."""

        factors = KillProbabilityFactors(base_pk=threat.base_pk)

        # Range factor
        # Pk degrades at edges of envelope
        lethal_range = threat.envelope.lethal_range_km
        min_range = threat.envelope.min_range_km

        if range_km < min_range * 1.5:
            # Too close - missile kinematics limited
            factors.range_factor = 0.7
        elif range_km > lethal_range:
            # Beyond lethal range but in engagement envelope
            factors.range_factor = 0.5
        elif range_km > lethal_range * 0.8:
            # Outer edge of lethal range
            factors.range_factor = 0.85
        else:
            # Optimal range
            factors.range_factor = 1.0

        # Altitude factor
        # Very low altitude: ground clutter, masking
        # Very high altitude: kinematic limitations
        min_alt = threat.envelope.min_altitude_m
        max_alt = threat.envelope.max_altitude_m

        if altitude_m < min_alt * 2:
            factors.altitude_factor = 0.6  # Near minimum altitude
        elif altitude_m < 500:
            factors.altitude_factor = 0.75  # Low altitude clutter
        elif altitude_m > max_alt * 0.8:
            factors.altitude_factor = 0.85  # High altitude kinematic
        else:
            factors.altitude_factor = 1.0

        # Aspect factor
        # Some guidance types are aspect-dependent
        if threat.missile:
            if threat.missile.guidance == GuidanceType.IR:
                # IR seekers prefer rear aspect (hot engines)
                if 90 < relative_azimuth < 270:
                    # Rear aspect
                    factors.aspect_factor = 1.0
                else:
                    # Front aspect (cooler)
                    factors.aspect_factor = 0.7
            elif threat.missile.guidance == GuidanceType.SARH:
                # SARH is aspect-independent
                factors.aspect_factor = 1.0
            else:
                factors.aspect_factor = 0.95

        # Maneuver factor
        if target_maneuvering:
            g_ratio = self.config.target_maneuver_g / 3.0  # Normalize to 3g
            factors.maneuver_factor = max(0.4, 1.0 - 0.3 * g_ratio)
        else:
            factors.maneuver_factor = 1.0

        # ECM factor
        if ecm_active:
            factors.ecm_factor = 1.0 - self.config.ecm_effectiveness
        else:
            factors.ecm_factor = 1.0

        # Countermeasure factor
        cm_survival = 1.0
        for cm in countermeasures:
            if cm == CountermeasureType.CHAFF:
                cm_survival *= (1.0 - self.config.chaff_effectiveness)
            elif cm == CountermeasureType.FLARE:
                if threat.missile and threat.missile.guidance == GuidanceType.IR:
                    cm_survival *= (1.0 - self.config.flare_effectiveness)
            elif cm == CountermeasureType.MANEUVER:
                cm_survival *= (1.0 - self.config.maneuver_effectiveness)

        factors.ecm_factor *= cm_survival

        # Weather factor
        factors.weather_factor = self.config.weather_factor

        # Reliability factor
        factors.reliability_factor = (
            self.config.missile_reliability *
            self.config.guidance_reliability *
            self.config.fuze_reliability
        )

        return factors

    def _calculate_time_to_intercept(
        self,
        threat: ThreatSystem,
        range_km: float,
        altitude_m: float,
        target_speed_ms: float,
    ) -> float:
        """
        Calculate time from launch to intercept.

        Includes:
            - Missile boost phase
            - Cruise/coast phase
            - Terminal phase
        """
        if threat.missile is None:
            # AAA - nearly instantaneous
            if threat.aaa:
                return range_km * 1000.0 / 900.0  # ~900 m/s muzzle velocity
            return 1.0

        # Missile speed (average across flight)
        # Assume Mach number converts to ~330 m/s per Mach
        avg_speed_ms = threat.missile.max_speed_mach * 330.0 * 0.7  # 70% of max

        # Slant range (include altitude)
        slant_range_m = math.sqrt((range_km * 1000) ** 2 + altitude_m ** 2)

        # Flight time
        flight_time = slant_range_m / avg_speed_ms

        # Add boost phase (typically 2-5 seconds)
        boost_time = 3.0

        # Total time to intercept
        return boost_time + flight_time

    def _calculate_salvo_pk(
        self,
        single_shot_pk: float,
        num_missiles: int,
    ) -> float:
        """
        Calculate cumulative Pk for missile salvo.

        P_k_cum = 1 - (1 - P_k)^n
        """
        return 1.0 - (1.0 - single_shot_pk) ** num_missiles

    def calculate_survivability(
        self,
        threats: list[ThreatSystem],
        trajectory: list[tuple[float, float, float, float]],  # (lat, lon, alt, time)
        target_speed_ms: float = 50.0,
        target_maneuvering: bool = False,
        ecm_active: bool = False,
    ) -> dict[str, Any]:
        """
        Calculate survivability along entire trajectory.

        Args:
            threats: All threat systems in environment
            trajectory: List of (lat, lon, alt_m, time_s) points
            target_speed_ms: Target speed
            target_maneuvering: Constant maneuvering
            ecm_active: ECM constantly active

        Returns:
            Survivability analysis
        """
        results = {
            "trajectory_points": len(trajectory),
            "survivability": 1.0,
            "cumulative_pk": 0.0,
            "engagements": [],
            "highest_threat": None,
            "time_in_lethal": 0.0,
        }

        cumulative_survival = 1.0

        for i, (lat, lon, alt, time) in enumerate(trajectory):
            point_pk = 0.0
            point_threats = []

            for threat in threats:
                eng_result = self.calculate_engagement_probability(
                    threat=threat,
                    target_lat=lat,
                    target_lon=lon,
                    target_alt_m=alt,
                    target_speed_ms=target_speed_ms,
                    target_maneuvering=target_maneuvering,
                    ecm_active=ecm_active,
                )

                if eng_result["can_engage"]:
                    pk = eng_result["probability_kill"]

                    # Time-based Pk (for continuous exposure)
                    # Assume threat can engage once per reaction_time
                    dt = 0.0
                    if i > 0:
                        dt = time - trajectory[i-1][3]

                    # Scale Pk by exposure time relative to reaction time
                    if dt > 0 and eng_result["reaction_time"] > 0:
                        engagement_probability = min(1.0, dt / eng_result["reaction_time"])
                        effective_pk = pk * engagement_probability
                    else:
                        effective_pk = pk

                    # Aggregate (independent threats)
                    point_pk = 1.0 - (1.0 - point_pk) * (1.0 - effective_pk)

                    point_threats.append({
                        "threat_id": threat.threat_id,
                        "name": threat.name,
                        "pk": effective_pk,
                        "range_km": eng_result["range_km"],
                    })

            # Update cumulative survivability
            cumulative_survival *= (1.0 - point_pk)

            if point_threats:
                results["engagements"].append({
                    "time": time,
                    "position": (lat, lon, alt),
                    "point_pk": point_pk,
                    "threats": point_threats,
                })

                # Track highest single-point threat
                max_threat = max(point_threats, key=lambda x: x["pk"])
                if (results["highest_threat"] is None or
                    max_threat["pk"] > results["highest_threat"]["pk"]):
                    results["highest_threat"] = max_threat

        results["survivability"] = cumulative_survival
        results["cumulative_pk"] = 1.0 - cumulative_survival

        # Calculate time in lethal zones
        lethal_points = sum(1 for e in results["engagements"] if e["point_pk"] > 0.1)
        if len(trajectory) > 1:
            dt = trajectory[1][3] - trajectory[0][3]
            results["time_in_lethal"] = lethal_points * dt

        return results

    def analyze_engagement_timeline(
        self,
        threat: ThreatSystem,
        target_trajectory: list[tuple[float, float, float, float]],
    ) -> dict[str, Any]:
        """
        Detailed timeline analysis for single threat engagement.

        Returns phase-by-phase breakdown of engagement sequence.
        """
        timeline = []

        detection_time = None
        tracking_time = None
        launch_time = None

        for lat, lon, alt, time in target_trajectory:
            range_km = self._calculate_range_km(
                lat, lon, threat.latitude, threat.longitude
            )

            # Check envelope progressively
            in_detection = threat.envelope.contains_point(range_km, alt, 0, "detection")
            in_tracking = threat.envelope.contains_point(range_km, alt, 0, "tracking")
            in_engagement = threat.envelope.contains_point(range_km, alt, 0, "engagement")
            in_lethal = threat.envelope.contains_point(range_km, alt, 0, "lethal")

            # Determine phase
            if in_lethal:
                phase = EngagementPhase.TERMINAL
            elif in_engagement:
                if launch_time and time >= launch_time:
                    phase = EngagementPhase.MISSILE_LAUNCH
                else:
                    phase = EngagementPhase.FIRE_SOLUTION
            elif in_tracking:
                phase = EngagementPhase.TRACKING
                if tracking_time is None:
                    tracking_time = time
            elif in_detection:
                phase = EngagementPhase.DETECTION
                if detection_time is None:
                    detection_time = time
            else:
                phase = EngagementPhase.NOT_IN_RANGE

            # Check for launch (after reaction time in tracking)
            if (phase == EngagementPhase.FIRE_SOLUTION and
                tracking_time and
                time >= tracking_time + threat.reaction_time_s and
                launch_time is None):
                launch_time = time

            timeline.append({
                "time": time,
                "range_km": range_km,
                "altitude_m": alt,
                "phase": phase.name,
                "in_detection": in_detection,
                "in_tracking": in_tracking,
                "in_engagement": in_engagement,
                "in_lethal": in_lethal,
            })

        return {
            "threat_id": threat.threat_id,
            "threat_name": threat.name,
            "timeline": timeline,
            "detection_time": detection_time,
            "tracking_time": tracking_time,
            "launch_time": launch_time,
            "reaction_time": threat.reaction_time_s,
        }

    @staticmethod
    def _calculate_range_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate great-circle distance in km."""
        R = 6371.0
        lat1_rad, lat2_rad = math.radians(lat1), math.radians(lat2)
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = (math.sin(dlat/2)**2 +
             math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2)
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

    @staticmethod
    def _calculate_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate bearing from point 1 to point 2."""
        lat1_rad, lat2_rad = math.radians(lat1), math.radians(lat2)
        dlon = math.radians(lon2 - lon1)
        x = math.sin(dlon) * math.cos(lat2_rad)
        y = (math.cos(lat1_rad) * math.sin(lat2_rad) -
             math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(dlon))
        return (math.degrees(math.atan2(x, y)) + 360) % 360


# =============================================================================
# AAA-SPECIFIC ENGAGEMENT MODEL
# =============================================================================

class AAAEngagementModel:
    """
    Specialized engagement model for anti-aircraft artillery.

    AAA has different characteristics:
        - Very short range
        - Very fast reaction time
        - Per-burst kill probability
        - Continuous engagement capability
    """

    def __init__(self, config: EngagementModelConfig | None = None):
        """Initialize AAA model."""
        self.config = config or EngagementModelConfig()

    def calculate_pk(
        self,
        aaa: AAASpecification,
        range_m: float,
        altitude_m: float,
        target_speed_ms: float = 50.0,
        crossing_angle_deg: float = 90.0,
    ) -> dict[str, Any]:
        """
        Calculate AAA kill probability.

        Args:
            aaa: AAA system specification
            range_m: Slant range to target (m)
            altitude_m: Target altitude
            target_speed_ms: Target speed
            crossing_angle_deg: Target crossing angle (90=beam, 0/180=head/tail)

        Returns:
            AAA engagement assessment
        """
        # Check range
        if range_m > aaa.max_range_m:
            return {
                "can_engage": False,
                "pk_per_burst": 0.0,
                "expected_bursts": 0,
            }

        # Range factor (effectiveness degrades with range)
        if range_m < aaa.effective_range_m:
            range_factor = 1.0
        else:
            range_factor = 1.0 - (range_m - aaa.effective_range_m) / (aaa.max_range_m - aaa.effective_range_m)

        # Crossing angle factor
        # Beam crossing is hardest to track
        crossing_factor = 1.0 - 0.3 * abs(math.sin(math.radians(crossing_angle_deg)))

        # Speed factor
        if target_speed_ms > 100:
            speed_factor = 0.7
        elif target_speed_ms > 50:
            speed_factor = 0.85
        else:
            speed_factor = 1.0

        # Altitude factor
        if altitude_m > aaa.max_altitude_m * 0.8:
            alt_factor = 0.7
        elif altitude_m < 200:
            alt_factor = 0.9  # Close to ground may have masking
        else:
            alt_factor = 1.0

        # Calculate effective Pk per burst
        pk_per_burst = (
            aaa.pk_per_burst *
            range_factor *
            crossing_factor *
            speed_factor *
            alt_factor
        )

        # Calculate expected bursts during exposure
        # Exposure time depends on speed and range
        exposure_time_s = (aaa.effective_range_m * 2) / target_speed_ms
        burst_duration_s = aaa.burst_length_rounds / (aaa.rate_of_fire_rpm / 60.0)
        expected_bursts = max(1, int(exposure_time_s / (burst_duration_s + 1.0)))

        # Cumulative Pk
        cumulative_pk = 1.0 - (1.0 - pk_per_burst) ** expected_bursts

        return {
            "can_engage": True,
            "pk_per_burst": pk_per_burst,
            "expected_bursts": expected_bursts,
            "cumulative_pk": cumulative_pk,
            "exposure_time_s": exposure_time_s,
            "factors": {
                "range": range_factor,
                "crossing": crossing_factor,
                "speed": speed_factor,
                "altitude": alt_factor,
            },
        }


# =============================================================================
# ENGAGEMENT ZONE MAPPING
# =============================================================================

class EngagementZoneMapper:
    """
    Generate engagement zone contours for visualization.

    Creates geographic representations of:
        - Detection zones
        - Tracking zones
        - Engagement envelopes
        - Lethal zones
    """

    def __init__(self, resolution_deg: float = 0.01):
        """Initialize mapper with angular resolution."""
        self.resolution = resolution_deg

    def generate_envelope_polygon(
        self,
        threat: ThreatSystem,
        envelope_type: str = "engagement",
        num_points: int = 72,
    ) -> list[tuple[float, float]]:
        """
        Generate polygon representing engagement envelope.

        Args:
            threat: Threat system
            envelope_type: "detection", "tracking", "engagement", "lethal"
            num_points: Number of polygon vertices

        Returns:
            List of (lat, lon) points forming polygon
        """
        # Get range for envelope type
        range_km = {
            "detection": threat.envelope.detection_range_km,
            "tracking": threat.envelope.tracking_range_km,
            "engagement": threat.envelope.engagement_range_km,
            "lethal": threat.envelope.lethal_range_km,
        }.get(envelope_type, threat.envelope.engagement_range_km)

        points = []

        # Generate points around threat
        for i in range(num_points):
            angle_deg = i * (360.0 / num_points)

            # Check if this angle is in coverage sector
            if not self._in_azimuth_coverage(
                angle_deg,
                threat.envelope.azimuth_start_deg,
                threat.envelope.azimuth_end_deg,
                threat.heading_deg,
            ):
                continue

            # Calculate point at this angle and range
            lat, lon = self._point_at_distance(
                threat.latitude,
                threat.longitude,
                range_km,
                angle_deg,
            )
            points.append((lat, lon))

        # Close polygon if we have points
        if points and len(points) > 2:
            points.append(points[0])

        return points

    def generate_sector_polygon(
        self,
        threat: ThreatSystem,
        envelope_type: str = "engagement",
    ) -> list[tuple[float, float]]:
        """
        Generate sector polygon for systems with limited azimuth coverage.

        Returns polygon with center at threat location.
        """
        range_km = {
            "detection": threat.envelope.detection_range_km,
            "tracking": threat.envelope.tracking_range_km,
            "engagement": threat.envelope.engagement_range_km,
            "lethal": threat.envelope.lethal_range_km,
        }.get(envelope_type, threat.envelope.engagement_range_km)

        points = [(threat.latitude, threat.longitude)]  # Center

        # Arc points
        start_az = (threat.heading_deg + threat.envelope.azimuth_start_deg) % 360
        end_az = (threat.heading_deg + threat.envelope.azimuth_end_deg) % 360

        # Handle wraparound
        if end_az < start_az:
            angles = list(range(int(start_az), 360, 5)) + list(range(0, int(end_az) + 1, 5))
        else:
            angles = list(range(int(start_az), int(end_az) + 1, 5))

        for angle in angles:
            lat, lon = self._point_at_distance(
                threat.latitude, threat.longitude, range_km, angle
            )
            points.append((lat, lon))

        # Close back to center
        points.append((threat.latitude, threat.longitude))

        return points

    def _in_azimuth_coverage(
        self,
        angle: float,
        start: float,
        end: float,
        heading: float,
    ) -> bool:
        """Check if angle is within azimuth coverage."""
        # Adjust for system heading
        rel_angle = (angle - heading) % 360

        if end >= start:
            return start <= rel_angle <= end
        else:
            return rel_angle >= start or rel_angle <= end

    @staticmethod
    def _point_at_distance(
        lat: float,
        lon: float,
        distance_km: float,
        bearing_deg: float,
    ) -> tuple[float, float]:
        """Calculate point at given distance and bearing."""
        R = 6371.0  # Earth radius km

        lat_rad = math.radians(lat)
        lon_rad = math.radians(lon)
        bearing_rad = math.radians(bearing_deg)

        d = distance_km / R

        new_lat = math.asin(
            math.sin(lat_rad) * math.cos(d) +
            math.cos(lat_rad) * math.sin(d) * math.cos(bearing_rad)
        )

        new_lon = lon_rad + math.atan2(
            math.sin(bearing_rad) * math.sin(d) * math.cos(lat_rad),
            math.cos(d) - math.sin(lat_rad) * math.sin(new_lat)
        )

        return (math.degrees(new_lat), math.degrees(new_lon))


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def calculate_kill_probability(
    threat: ThreatSystem,
    target_lat: float,
    target_lon: float,
    target_alt_m: float,
    maneuvering: bool = False,
    ecm: bool = False,
) -> float:
    """
    Quick Pk calculation for a single threat/target pair.

    Returns single-shot kill probability.
    """
    model = EngagementModel()
    result = model.calculate_engagement_probability(
        threat=threat,
        target_lat=target_lat,
        target_lon=target_lon,
        target_alt_m=target_alt_m,
        target_maneuvering=maneuvering,
        ecm_active=ecm,
    )
    return result.get("single_shot_pk", 0.0)


def calculate_salvo_pk(single_shot_pk: float, num_missiles: int) -> float:
    """Calculate cumulative Pk for missile salvo."""
    return 1.0 - (1.0 - single_shot_pk) ** num_missiles


def calculate_survivability(
    threats: list[ThreatSystem],
    trajectory: list[tuple[float, float, float, float]],
) -> float:
    """
    Calculate probability of surviving trajectory through threat environment.

    Returns survivability probability (0-1).
    """
    model = EngagementModel()
    result = model.calculate_survivability(threats, trajectory)
    return result["survivability"]


def create_engagement_model(
    reliability: float = 0.95,
    ecm_effectiveness: float = 0.25,
) -> EngagementModel:
    """Create configured engagement model."""
    config = EngagementModelConfig(
        missile_reliability=reliability,
        ecm_effectiveness=ecm_effectiveness,
    )
    return EngagementModel(config)


def create_zone_mapper() -> EngagementZoneMapper:
    """Create engagement zone mapper."""
    return EngagementZoneMapper()
