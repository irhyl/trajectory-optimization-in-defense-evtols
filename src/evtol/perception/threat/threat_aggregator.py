"""
Threat Aggregator for Defense eVTOL Trajectory Optimization.

This module implements multi-source threat aggregation, fusing
information from multiple threat models into a unified risk
assessment suitable for trajectory optimization.

Aggregation Framework:

    1. Multi-Threat Fusion:
        Combine detection/engagement probabilities from multiple
        threat systems using probabilistic OR:

        P(hit by any) = 1 - Π_i (1 - P_i)

    2. Time-Varying Threats:
        - Patrol aircraft schedules
        - Radar emission patterns
        - Alert state changes

    3. Uncertainty Propagation:
        - Threat position uncertainty
        - Parameter estimation errors
        - Intelligence confidence levels

    4. Risk Metrics:
        - Cumulative exposure
        - Maximum instantaneous risk
        - Probability of mission success
        - Expected casualties

Integration Points:
    - ThreatField: Spatial cost queries
    - TerrainMasking: LOS analysis
    - DetectionModel: Radar detection probability
    - EngagementModel: Weapon effectiveness

Output:
    - Aggregated risk maps
    - Time-varying threat profiles
    - Mission feasibility assessment
    - Optimal timing recommendations

References:
    [1] Kochenderfer, M. "Decision Making Under Uncertainty"
    [2] Marden, J. "Multi-Agent Systems"
    [3] Beard, R. "Small Unmanned Aircraft" Ch. 12

Author: Defense eVTOL Trajectory Optimization System
Version: 1.0.0
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from typing import Any
from datetime import datetime, timezone, timedelta
from enum import Enum, auto
import numpy as np

from .threat_types import (
    ThreatPriority,
    ThreatStatus,
)
from .threat_types import ThreatSystem
from .detection_model import RadarDetectionModel
from .engagement_model import EngagementModel
from .threat_field import ThreatField, ThreatFieldConfig, GridBounds


logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================

class AggregationMethod(Enum):
    """Method for combining multiple threat probabilities."""

    PROBABILISTIC_OR = auto()    # 1 - Π(1 - P_i), independent threats
    MAXIMUM = auto()             # max(P_i), conservative
    WEIGHTED_SUM = auto()        # Σ w_i × P_i, priority-weighted
    BAYESIAN = auto()            # Bayesian fusion with priors
    DEMPSTER_SHAFER = auto()     # Evidence-based fusion


class UncertaintyModel(Enum):
    """Model for handling threat uncertainty."""

    DETERMINISTIC = auto()       # No uncertainty considered
    GAUSSIAN = auto()            # Gaussian position/parameter uncertainty
    MONTE_CARLO = auto()         # Monte Carlo sampling
    WORST_CASE = auto()          # Conservative bounds


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class AggregatorConfig:
    """Configuration for threat aggregation."""

    # Aggregation method
    aggregation_method: AggregationMethod = AggregationMethod.PROBABILISTIC_OR

    # Uncertainty handling
    uncertainty_model: UncertaintyModel = UncertaintyModel.DETERMINISTIC
    position_uncertainty_m: float = 100.0      # GPS/intelligence accuracy
    parameter_uncertainty_pct: float = 10.0     # Parameter estimation error
    monte_carlo_samples: int = 100              # MC samples if used

    # Threat weighting by priority
    priority_weights: dict[ThreatPriority, float] = field(default_factory=lambda: {
        ThreatPriority.CRITICAL: 2.0,
        ThreatPriority.HIGH: 1.5,
        ThreatPriority.MEDIUM: 1.0,
        ThreatPriority.LOW: 0.5,
    })

    # Time-varying behavior
    enable_time_varying: bool = True
    time_step_s: float = 60.0                   # Update interval
    prediction_horizon_s: float = 3600.0        # 1 hour lookahead

    # Risk thresholds
    mission_abort_risk: float = 0.50            # Abort if risk exceeds
    acceptable_cumulative_risk: float = 0.10    # 10% max cumulative

    # Spatial resolution for aggregation
    aggregation_resolution_m: float = 500.0


@dataclass
class ThreatWeight:
    """Weighting for a single threat in aggregation."""

    threat_id: str
    base_weight: float = 1.0
    priority_multiplier: float = 1.0
    confidence_factor: float = 1.0      # 0-1 based on intel confidence
    recency_factor: float = 1.0         # Decay based on intel age

    @property
    def effective_weight(self) -> float:
        return (
            self.base_weight *
            self.priority_multiplier *
            self.confidence_factor *
            self.recency_factor
        )


# =============================================================================
# AGGREGATED RISK RESULT
# =============================================================================

@dataclass
class AggregatedRisk:
    """Result of threat aggregation at a point or along a path."""

    # Combined probabilities
    detection_probability: float
    engagement_probability: float
    kill_probability: float

    # Mission-level metrics
    survival_probability: float
    mission_success_probability: float

    # Contributing threats (sorted by contribution)
    threat_contributions: dict[str, float]
    dominant_threat: str | None

    # Risk classification
    risk_level: str                 # "LOW", "MEDIUM", "HIGH", "EXTREME"
    is_acceptable: bool
    abort_recommended: bool

    # Uncertainty bounds (if computed)
    confidence_interval: tuple[float, float] | None = None

    # Time-varying info
    timestamp: datetime | None = None
    valid_duration_s: float | None = None


@dataclass
class TimeVaryingRisk:
    """Risk profile over time for a position or path."""

    start_time: datetime
    end_time: datetime
    time_step_s: float

    # Risk at each time step
    timestamps: list[datetime]
    risk_values: list[float]

    # Statistics
    mean_risk: float
    max_risk: float
    min_risk: float

    # Optimal timing
    best_time: datetime | None
    best_risk: float | None

    # Windows of acceptability
    acceptable_windows: list[tuple[datetime, datetime]]


@dataclass
class MissionAssessment:
    """Complete mission risk assessment."""

    # Overall metrics
    mission_feasible: bool
    overall_risk: float
    survival_probability: float

    # Path-specific
    total_exposure: float
    max_instantaneous_risk: float
    critical_waypoints: list[int]

    # Threat summary
    primary_threats: list[str]
    threat_count: int

    # Recommendations
    route_quality: str              # "OPTIMAL", "ACCEPTABLE", "MARGINAL", "UNACCEPTABLE"
    recommendations: list[str]
    alternative_suggested: bool


# =============================================================================
# THREAT AGGREGATOR
# =============================================================================

class ThreatAggregator:
    """
    Multi-source threat aggregation engine.

    Combines detection/engagement/kill probabilities from multiple
    threat systems into unified risk metrics for trajectory planning.

    Example:
        >>> aggregator = ThreatAggregator(threats, config)
        >>>
        >>> # Point risk query
        >>> risk = aggregator.get_aggregated_risk(28.5, 77.0, 500)
        >>> print(f"Kill probability: {risk.kill_probability:.2%}")
        >>>
        >>> # Path assessment
        >>> assessment = aggregator.assess_mission(waypoints)
        >>> if not assessment.mission_feasible:
        >>>     print("Mission abort recommended")
        >>>
        >>> # Time-varying analysis
        >>> time_profile = aggregator.get_time_varying_risk(
        ...     lat=28.5, lon=77.0, alt=500,
        ...     start_time=datetime.now(timezone.utc),
        ...     duration_hours=6
        ... )
        >>> print(f"Best time: {time_profile.best_time}")

    Inputs:
        threats: List of ThreatSystem objects
        config: AggregatorConfig
        detection_model: Optional custom detection model
        engagement_model: Optional custom engagement model
        terrain_masking: Optional terrain masking model

    Outputs:
        - Aggregated risk at any position
        - Time-varying risk profiles
        - Mission feasibility assessments
        - Threat contribution breakdowns
    """

    def __init__(
        self,
        threats: list[ThreatSystem],
        config: AggregatorConfig | None = None,
        detection_model: RadarDetectionModel | None = None,
        engagement_model: EngagementModel | None = None,
        threat_field: ThreatField | None = None,
    ):
        self.threats = threats
        self.config = config or AggregatorConfig()

        # Models
        self.detection_model = detection_model or RadarDetectionModel()
        self.engagement_model = engagement_model or EngagementModel()
        self.threat_field = threat_field

        # Build threat index
        self._threat_index: dict[str, ThreatSystem] = {
            t.threat_id: t for t in threats
        }

        # Compute threat weights
        self._threat_weights = self._compute_threat_weights()

        # Cache for performance
        self._cache: dict[tuple, Any] = {}

        logger.info(f"ThreatAggregator initialized with {len(threats)} threats")

    def _compute_threat_weights(self) -> dict[str, ThreatWeight]:
        """Compute weights for each threat based on priority and confidence."""
        weights = {}

        for threat in self.threats:
            priority_mult = self.config.priority_weights.get(
                getattr(threat, 'priority', ThreatPriority.MEDIUM), 1.0
            )

            # Confidence from intelligence quality (simulated)
            confidence = getattr(threat, 'intelligence_confidence', 0.9)

            # Recency decay (simulated)
            recency = getattr(threat, 'intel_recency_factor', 1.0)

            weights[threat.threat_id] = ThreatWeight(
                threat_id=threat.threat_id,
                base_weight=1.0,
                priority_multiplier=priority_mult,
                confidence_factor=confidence,
                recency_factor=recency,
            )

        return weights

    # =========================================================================
    # CORE AGGREGATION
    # =========================================================================

    def get_aggregated_risk(
        self,
        latitude: float,
        longitude: float,
        altitude_m: float,
        timestamp: datetime | None = None,
        platform_rcs_dbsm: float = -10.0,
    ) -> AggregatedRisk:
        """
        Get aggregated threat risk at a position.

        Args:
            latitude: Position latitude
            longitude: Position longitude
            altitude_m: Altitude MSL in meters
            timestamp: Time for time-varying threats (None = now)
            platform_rcs_dbsm: Platform RCS in dBsm

        Returns:
            AggregatedRisk with combined probabilities
        """
        timestamp = timestamp or datetime.now(timezone.utc)

        # Collect individual threat contributions
        detection_probs = []
        engagement_probs = []
        kill_probs = []
        contributions = {}

        for threat in self.threats:
            if not self._is_threat_active(threat, timestamp):
                continue

            # Get individual probabilities
            p_det, p_eng, p_kill = self._evaluate_single_threat(
                threat, latitude, longitude, altitude_m, platform_rcs_dbsm
            )

            # Apply weight
            weight = self._threat_weights[threat.threat_id].effective_weight

            detection_probs.append((p_det, weight))
            engagement_probs.append((p_eng, weight))
            kill_probs.append((p_kill, weight))

            # Store contribution (weighted kill probability)
            contributions[threat.threat_id] = p_kill * weight

        # Aggregate probabilities
        p_detection = self._aggregate_probabilities(detection_probs)
        p_engagement = self._aggregate_probabilities(engagement_probs)
        p_kill = self._aggregate_probabilities(kill_probs)

        # Mission metrics
        p_survival = 1.0 - p_kill
        p_mission = p_survival  # Simplified; could include other factors

        # Find dominant threat
        dominant = max(contributions.items(), key=lambda x: x[1])[0] if contributions else None

        # Risk classification
        risk_level = self._classify_risk_level(p_kill)
        is_acceptable = p_kill <= self.config.acceptable_cumulative_risk
        abort_recommended = p_kill >= self.config.mission_abort_risk

        return AggregatedRisk(
            detection_probability=p_detection,
            engagement_probability=p_engagement,
            kill_probability=p_kill,
            survival_probability=p_survival,
            mission_success_probability=p_mission,
            threat_contributions=contributions,
            dominant_threat=dominant,
            risk_level=risk_level,
            is_acceptable=is_acceptable,
            abort_recommended=abort_recommended,
            timestamp=timestamp,
        )

    def _evaluate_single_threat(
        self,
        threat: ThreatSystem,
        latitude: float,
        longitude: float,
        altitude_m: float,
        platform_rcs_dbsm: float,
    ) -> tuple[float, float, float]:
        """Evaluate detection/engagement/kill for single threat."""

        # Compute range to threat
        range_m = self._compute_distance(
            latitude, longitude, 0,
            threat.latitude, threat.longitude, 0,
        )

        # Detection probability
        p_detection = 0.0
        if threat.radar and threat.radar.frequency_ghz > 0:
            try:
                rcs_sqm = 10 ** (platform_rcs_dbsm / 10.0)
                det_result = self.detection_model.calculate_detection_probability(
                    radar=threat.radar,
                    target_range_m=range_m,
                    target_altitude_m=altitude_m,
                    target_rcs_sqm=rcs_sqm,
                )
                p_detection = det_result.get('detection_probability', 0.0) if isinstance(det_result, dict) else 0.0
            except (ZeroDivisionError, ValueError):
                p_detection = 0.0

        # Engagement probability
        eng_result = self.engagement_model.calculate_engagement_probability(
            threat=threat,
            target_lat=latitude,
            target_lon=longitude,
            target_alt_m=altitude_m,
        )
        p_engagement = eng_result.get('probability_engagement', 0.0) if isinstance(eng_result, dict) else 0.0

        # Kill probability
        if isinstance(eng_result, dict) and 'probability_kill' in eng_result:
            p_kill = eng_result['probability_kill']
        else:
            # Combined probability
            p_kill = p_detection * p_engagement * self._get_sspk(threat)

        return p_detection, p_engagement, p_kill

    def _get_sspk(self, threat: ThreatSystem) -> float:
        """Get single-shot probability of kill."""
        if threat.missile:
            return getattr(threat.missile, 'single_shot_pk', threat.base_pk)
        return threat.base_pk

    def _is_threat_active(
        self,
        threat: ThreatSystem,
        timestamp: datetime
    ) -> bool:
        """Check if threat is active at given time."""
        if threat.status != ThreatStatus.ACTIVE:
            return False

        # Check time-varying schedules if enabled
        if self.config.enable_time_varying:
            # Could check patrol schedules, emission patterns, etc.
            pass

        return True

    def _aggregate_probabilities(
        self,
        probs_with_weights: list[tuple[float, float]]
    ) -> float:
        """
        Aggregate probabilities based on configured method.

        Args:
            probs_with_weights: List of (probability, weight) tuples

        Returns:
            Aggregated probability
        """
        if not probs_with_weights:
            return 0.0

        method = self.config.aggregation_method

        if method == AggregationMethod.PROBABILISTIC_OR:
            # P(at least one) = 1 - Π(1 - P_i)
            product = 1.0
            for p, _w in probs_with_weights:
                product *= (1.0 - p)
            return 1.0 - product

        elif method == AggregationMethod.MAXIMUM:
            return max(p for p, w in probs_with_weights)

        elif method == AggregationMethod.WEIGHTED_SUM:
            total_weight = sum(w for p, w in probs_with_weights)
            if total_weight == 0:
                return 0.0
            weighted_sum = sum(p * w for p, w in probs_with_weights)
            return min(1.0, weighted_sum / total_weight)

        elif method == AggregationMethod.BAYESIAN:
            # Simplified Bayesian update
            odds = 1.0
            for p, w in probs_with_weights:
                if 0 < p < 1:
                    odds *= (p / (1 - p)) ** w
            return odds / (1 + odds)

        elif method == AggregationMethod.DEMPSTER_SHAFER:
            # Simplified Dempster-Shafer
            belief = 0.0
            for p, _w in probs_with_weights:
                belief = belief + p * (1 - belief)
            return belief

        return 0.0

    def _classify_risk_level(self, p_kill: float) -> str:
        """Classify risk level from kill probability."""
        if p_kill >= 0.50:
            return "EXTREME"
        elif p_kill >= 0.20:
            return "HIGH"
        elif p_kill >= 0.05:
            return "MEDIUM"
        else:
            return "LOW"

    # =========================================================================
    # TIME-VARYING ANALYSIS
    # =========================================================================

    def get_time_varying_risk(
        self,
        latitude: float,
        longitude: float,
        altitude_m: float,
        start_time: datetime,
        duration_hours: float = 6.0,
        time_step_minutes: float = 5.0,
    ) -> TimeVaryingRisk:
        """
        Compute risk profile over time at a position.

        Args:
            latitude, longitude, altitude_m: Position
            start_time: Start of analysis window
            duration_hours: Duration to analyze
            time_step_minutes: Time resolution

        Returns:
            TimeVaryingRisk with temporal profile
        """
        end_time = start_time + timedelta(hours=duration_hours)
        step = timedelta(minutes=time_step_minutes)

        timestamps = []
        risk_values = []

        current = start_time
        while current <= end_time:
            timestamps.append(current)

            risk = self.get_aggregated_risk(
                latitude, longitude, altitude_m, current
            )
            risk_values.append(risk.kill_probability)

            current += step

        # Statistics
        mean_risk = sum(risk_values) / len(risk_values)
        max_risk = max(risk_values)
        min_risk = min(risk_values)

        # Find optimal time
        best_idx = risk_values.index(min_risk)
        best_time = timestamps[best_idx]

        # Find acceptable windows
        acceptable_windows = self._find_acceptable_windows(
            timestamps, risk_values
        )

        return TimeVaryingRisk(
            start_time=start_time,
            end_time=end_time,
            time_step_s=time_step_minutes * 60,
            timestamps=timestamps,
            risk_values=risk_values,
            mean_risk=mean_risk,
            max_risk=max_risk,
            min_risk=min_risk,
            best_time=best_time,
            best_risk=min_risk,
            acceptable_windows=acceptable_windows,
        )

    def _find_acceptable_windows(
        self,
        timestamps: list[datetime],
        risk_values: list[float],
    ) -> list[tuple[datetime, datetime]]:
        """Find time windows where risk is acceptable."""
        threshold = self.config.acceptable_cumulative_risk
        windows = []

        in_window = False
        window_start = None

        for t, r in zip(timestamps, risk_values):
            if r <= threshold:
                if not in_window:
                    window_start = t
                    in_window = True
            else:
                if in_window:
                    windows.append((window_start, t))
                    in_window = False

        # Close final window if still open
        if in_window and window_start:
            windows.append((window_start, timestamps[-1]))

        return windows

    # =========================================================================
    # MISSION ASSESSMENT
    # =========================================================================

    def assess_mission(
        self,
        waypoints: list[tuple[float, float, float]],
        speed_m_s: float = 50.0,
        start_time: datetime | None = None,
    ) -> MissionAssessment:
        """
        Comprehensive mission risk assessment.

        Args:
            waypoints: List of (lat, lon, alt) waypoints
            speed_m_s: Platform speed for timing calculations
            start_time: Mission start time

        Returns:
            MissionAssessment with feasibility and recommendations
        """
        start_time = start_time or datetime.now(timezone.utc)

        # Evaluate each waypoint
        waypoint_risks = []
        critical_waypoints = []
        threat_exposures: dict[str, float] = {}

        current_time = start_time

        for i, (lat, lon, alt) in enumerate(waypoints):
            risk = self.get_aggregated_risk(lat, lon, alt, current_time)
            waypoint_risks.append(risk.kill_probability)

            if risk.kill_probability >= self.config.mission_abort_risk:
                critical_waypoints.append(i)

            # Accumulate threat exposures
            for threat_id, contrib in risk.threat_contributions.items():
                threat_exposures[threat_id] = threat_exposures.get(threat_id, 0) + contrib

            # Update time based on segment travel
            if i < len(waypoints) - 1:
                next_lat, next_lon, next_alt = waypoints[i + 1]
                dist = self._compute_distance(
                    lat, lon, alt, next_lat, next_lon, next_alt
                )
                travel_time = dist / speed_m_s
                current_time += timedelta(seconds=travel_time)

        # Compute cumulative metrics
        max_risk = max(waypoint_risks)
        mean_risk = sum(waypoint_risks) / len(waypoint_risks)

        # Survival probability along path
        survival = 1.0
        for r in waypoint_risks:
            survival *= (1.0 - r)

        # Total exposure (integral approximation)
        total_exposure = sum(waypoint_risks)

        # Primary threats
        sorted_threats = sorted(
            threat_exposures.items(), key=lambda x: x[1], reverse=True
        )
        primary_threats = [t[0] for t in sorted_threats[:3]]

        # Feasibility determination
        mission_feasible = (
            max_risk < self.config.mission_abort_risk and
            survival > (1 - self.config.acceptable_cumulative_risk)
        )

        # Route quality assessment
        route_quality = self._assess_route_quality(
            max_risk, survival, len(critical_waypoints)
        )

        # Generate recommendations
        recommendations = self._generate_recommendations(
            waypoint_risks, critical_waypoints, primary_threats
        )

        return MissionAssessment(
            mission_feasible=mission_feasible,
            overall_risk=mean_risk,
            survival_probability=survival,
            total_exposure=total_exposure,
            max_instantaneous_risk=max_risk,
            critical_waypoints=critical_waypoints,
            primary_threats=primary_threats,
            threat_count=len(threat_exposures),
            route_quality=route_quality,
            recommendations=recommendations,
            alternative_suggested=not mission_feasible,
        )

    def _compute_distance(
        self,
        lat1: float, lon1: float, alt1: float,
        lat2: float, lon2: float, alt2: float,
    ) -> float:
        """Compute 3D distance between points."""
        # Haversine for horizontal
        R = 6371000
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlam = math.radians(lon2 - lon1)

        a = math.sin(dphi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

        horizontal = R * c
        vertical = abs(alt2 - alt1)

        return math.sqrt(horizontal**2 + vertical**2)

    def _assess_route_quality(
        self,
        max_risk: float,
        survival: float,
        n_critical: int,
    ) -> str:
        """Assess overall route quality."""
        if max_risk < 0.05 and survival > 0.95 and n_critical == 0:
            return "OPTIMAL"
        elif max_risk < 0.15 and survival > 0.85 and n_critical <= 1:
            return "ACCEPTABLE"
        elif max_risk < 0.30 and survival > 0.70:
            return "MARGINAL"
        else:
            return "UNACCEPTABLE"

    def _generate_recommendations(
        self,
        waypoint_risks: list[float],
        critical_waypoints: list[int],
        primary_threats: list[str],
    ) -> list[str]:
        """Generate tactical recommendations."""
        recommendations = []

        if critical_waypoints:
            recommendations.append(
                f"Consider rerouting around waypoints: {critical_waypoints}"
            )

        if primary_threats:
            recommendations.append(
                f"Primary threats to avoid: {', '.join(primary_threats[:3])}"
            )

        # Altitude recommendations
        high_risk_alts = [i for i, r in enumerate(waypoint_risks) if r > 0.1]
        if high_risk_alts:
            recommendations.append(
                "Consider altitude adjustments to exploit terrain masking"
            )

        # Timing recommendations
        if max(waypoint_risks) > 0.2:
            recommendations.append(
                "Consider timing mission to coincide with reduced threat activity"
            )

        if not recommendations:
            recommendations.append("Route appears optimized for threat avoidance")

        return recommendations

    # =========================================================================
    # UNCERTAINTY ANALYSIS
    # =========================================================================

    def get_risk_with_uncertainty(
        self,
        latitude: float,
        longitude: float,
        altitude_m: float,
        confidence_level: float = 0.95,
    ) -> tuple[AggregatedRisk, tuple[float, float]]:
        """
        Get risk with uncertainty bounds.

        Args:
            latitude, longitude, altitude_m: Position
            confidence_level: Confidence level for interval

        Returns:
            (nominal_risk, (lower_bound, upper_bound))
        """
        if self.config.uncertainty_model == UncertaintyModel.DETERMINISTIC:
            risk = self.get_aggregated_risk(latitude, longitude, altitude_m)
            return risk, (risk.kill_probability, risk.kill_probability)

        elif self.config.uncertainty_model == UncertaintyModel.MONTE_CARLO:
            return self._monte_carlo_uncertainty(
                latitude, longitude, altitude_m, confidence_level
            )

        elif self.config.uncertainty_model == UncertaintyModel.WORST_CASE:
            return self._worst_case_uncertainty(
                latitude, longitude, altitude_m
            )

        else:
            risk = self.get_aggregated_risk(latitude, longitude, altitude_m)
            return risk, (risk.kill_probability, risk.kill_probability)

    def _monte_carlo_uncertainty(
        self,
        latitude: float,
        longitude: float,
        altitude_m: float,
        confidence_level: float,
    ) -> tuple[AggregatedRisk, tuple[float, float]]:
        """Monte Carlo sampling for uncertainty."""
        n_samples = self.config.monte_carlo_samples
        samples = []

        for _ in range(n_samples):
            # Perturb position
            pos_std = self.config.position_uncertainty_m / 111000  # degrees
            lat_sample = latitude + np.random.normal(0, pos_std)
            lon_sample = longitude + np.random.normal(0, pos_std)
            alt_sample = altitude_m + np.random.normal(0, 50)  # 50m altitude uncertainty

            risk = self.get_aggregated_risk(lat_sample, lon_sample, alt_sample)
            samples.append(risk.kill_probability)

        # Compute confidence interval
        samples.sort()
        alpha = 1 - confidence_level
        lower_idx = int(alpha / 2 * n_samples)
        upper_idx = int((1 - alpha / 2) * n_samples)

        lower_bound = samples[lower_idx]
        upper_bound = samples[min(upper_idx, n_samples - 1)]

        # Nominal is median
        nominal = self.get_aggregated_risk(latitude, longitude, altitude_m)
        nominal.confidence_interval = (lower_bound, upper_bound)

        return nominal, (lower_bound, upper_bound)

    def _worst_case_uncertainty(
        self,
        latitude: float,
        longitude: float,
        altitude_m: float,
    ) -> tuple[AggregatedRisk, tuple[float, float]]:
        """Worst-case bounds considering uncertainty."""
        # Sample at uncertainty bounds
        delta = self.config.position_uncertainty_m / 111000

        risks = []
        for dlat in [-delta, 0, delta]:
            for dlon in [-delta, 0, delta]:
                for dalt in [-100, 0, 100]:
                    risk = self.get_aggregated_risk(
                        latitude + dlat,
                        longitude + dlon,
                        altitude_m + dalt,
                    )
                    risks.append(risk.kill_probability)

        lower_bound = min(risks)
        upper_bound = max(risks)

        nominal = self.get_aggregated_risk(latitude, longitude, altitude_m)
        nominal.confidence_interval = (lower_bound, upper_bound)

        return nominal, (lower_bound, upper_bound)

    # =========================================================================
    # THREAT FIELD INTEGRATION
    # =========================================================================

    def update_threat_field(
        self,
        bounds: GridBounds,
        resolution_m: float = 500.0,
    ):
        """
        Create/update integrated threat field.

        Args:
            bounds: Geographic bounds for field
            resolution_m: Grid resolution
        """
        config = ThreatFieldConfig(
            horizontal_resolution_m=resolution_m,
            detection_weight=0.2,
            engagement_weight=0.3,
            kill_weight=0.5,
        )

        self.threat_field = ThreatField(
            bounds=bounds,
            threats=self.threats,
            config=config,
            detection_model=self.detection_model,
            engagement_model=self.engagement_model,
        )

        self.threat_field.compute()
        logger.info("Threat field updated")

    def get_threat_field_cost(
        self,
        latitude: float,
        longitude: float,
        altitude_m: float,
    ) -> float:
        """Get cost from threat field if available."""
        if self.threat_field is None:
            raise RuntimeError("No threat field computed. Call update_threat_field() first.")

        return self.threat_field.get_cost(latitude, longitude, altitude_m)

    # =========================================================================
    # EXPORT
    # =========================================================================

    def get_threat_summary(self) -> dict[str, Any]:
        """Get summary of all threats."""
        by_category: dict[str, int] = {}
        by_priority: dict[str, int] = {}
        active_count = 0

        for threat in self.threats:
            cat = threat.category.name
            by_category[cat] = by_category.get(cat, 0) + 1

            pri = getattr(threat, 'priority', ThreatPriority.MEDIUM).name
            by_priority[pri] = by_priority.get(pri, 0) + 1

            if threat.status == ThreatStatus.ACTIVE:
                active_count += 1

        return {
            'total_threats': len(self.threats),
            'active_threats': active_count,
            'by_category': by_category,
            'by_priority': by_priority,
            'aggregation_method': self.config.aggregation_method.name,
        }

    def to_dict(self) -> dict[str, Any]:
        """Export aggregator configuration."""
        return {
            'config': {
                'aggregation_method': self.config.aggregation_method.name,
                'uncertainty_model': self.config.uncertainty_model.name,
                'mission_abort_risk': self.config.mission_abort_risk,
                'acceptable_cumulative_risk': self.config.acceptable_cumulative_risk,
            },
            'threats': self.get_threat_summary(),
            'weights': {
                tid: w.effective_weight
                for tid, w in self._threat_weights.items()
            },
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_aggregator(
    threats: list[ThreatSystem],
    method: str = 'probabilistic_or',
) -> ThreatAggregator:
    """
    Create threat aggregator with specified method.

    Args:
        threats: List of threat systems
        method: Aggregation method name

    Returns:
        Configured ThreatAggregator
    """
    method_map = {
        'probabilistic_or': AggregationMethod.PROBABILISTIC_OR,
        'maximum': AggregationMethod.MAXIMUM,
        'weighted_sum': AggregationMethod.WEIGHTED_SUM,
        'bayesian': AggregationMethod.BAYESIAN,
        'dempster_shafer': AggregationMethod.DEMPSTER_SHAFER,
    }

    config = AggregatorConfig(
        aggregation_method=method_map.get(method, AggregationMethod.PROBABILISTIC_OR)
    )

    return ThreatAggregator(threats, config)


def quick_risk_assessment(
    latitude: float,
    longitude: float,
    altitude_m: float,
    threats: list[ThreatSystem],
) -> dict[str, Any]:
    """
    Quick single-point risk assessment.

    Args:
        latitude, longitude, altitude_m: Position
        threats: List of threat systems

    Returns:
        Risk assessment dictionary
    """
    aggregator = ThreatAggregator(threats)
    risk = aggregator.get_aggregated_risk(latitude, longitude, altitude_m)

    return {
        'kill_probability': risk.kill_probability,
        'survival_probability': risk.survival_probability,
        'risk_level': risk.risk_level,
        'is_acceptable': risk.is_acceptable,
        'dominant_threat': risk.dominant_threat,
    }
