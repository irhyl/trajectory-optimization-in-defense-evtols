"""
Conflict Detection and Alert Generation for eVTOL Operations.

This module provides conflict detection between the ownship (eVTOL) and
tracked obstacles, generating alerts based on predicted trajectories
and separation minima violations.

Key Capabilities:
    - Trajectory prediction (linear, turn-rate based)
    - CPA/TTC computation for all tracked obstacles
    - Alert level classification (NONE → CRITICAL)
    - Conflict zone identification
    - Evasion vector suggestion

Alert Levels (per ICAO standards, adapted for eVTOL):
    - NONE: No conflict (> 2km or > 5 min)
    - ADVISORY: Awareness only (< 2km, < 5 min)
    - CAUTION: Monitor closely (< 500m, < 30s)
    - WARNING: Prepare maneuver (< 200m, < 15s)
    - CRITICAL: Immediate evasion (< 50m, < 5s)

References:
    - ICAO Doc 9863: Airborne Collision Avoidance System (ACAS)
    - RTCA DO-317B: Minimum Operational Performance Standards for ACAS
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass
from typing import Any
from datetime import datetime, timezone
import numpy as np
from numpy.typing import NDArray

from .obstacle_types import (
    AlertLevel,
    ConflictAlert,
    ConflictZone,
    SeparationMinima,
    DEFAULT_SEPARATION_MINIMA,
    TrackStatus,
    BoundingCylinder,
)
from .geometry import (
    ENUFrame,
    compute_cpa,
    compute_ttc_spheres,
    CPAResult,
    TTCResult,
)
from .tracker import TrackManager, ManagedTrack

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class ConflictDetectorConfig:
    """Configuration for conflict detection."""

    # Prediction horizons
    short_term_horizon_s: float = 30.0   # For immediate alerts
    medium_term_horizon_s: float = 120.0  # For planning
    long_term_horizon_s: float = 300.0    # For strategic awareness

    # Prediction time step
    prediction_dt_s: float = 1.0

    # Alert thresholds (distance in meters, time in seconds)
    critical_distance_m: float = 50.0
    critical_time_s: float = 5.0

    warning_distance_m: float = 200.0
    warning_time_s: float = 15.0

    caution_distance_m: float = 500.0
    caution_time_s: float = 30.0

    advisory_distance_m: float = 2000.0
    advisory_time_s: float = 300.0

    # Vertical separation thresholds
    vertical_separation_m: float = 100.0

    # Alert suppression
    min_alert_interval_s: float = 1.0    # Minimum time between alerts
    alert_hysteresis_m: float = 20.0      # Prevent alert flickering

    # Evasion suggestions
    suggest_evasion: bool = True
    min_evasion_clearance_m: float = 100.0


@dataclass
class OwnshipState:
    """Current state of the ownship (eVTOL)."""

    # Position (LLA)
    latitude: float
    longitude: float
    altitude_m: float

    # Velocity (ENU, m/s)
    velocity_east: float
    velocity_north: float
    velocity_up: float

    # Orientation
    heading_deg: float

    # Dimensions for collision detection
    length_m: float = 10.0
    width_m: float = 8.0
    height_m: float = 3.0

    # Safety buffer
    safety_buffer_m: float = 20.0

    @property
    def speed(self) -> float:
        """Horizontal speed in m/s."""
        return math.sqrt(self.velocity_east ** 2 + self.velocity_north ** 2)

    @property
    def effective_radius(self) -> float:
        """Effective bounding sphere radius including safety buffer."""
        max_dim = max(self.length_m, self.width_m, self.height_m)
        return max_dim / 2 + self.safety_buffer_m

    def to_enu(self, frame: ENUFrame) -> NDArray[np.float64]:
        """Convert position to ENU coordinates."""
        return frame.lla_to_enu(self.latitude, self.longitude, self.altitude_m)

    def get_velocity_enu(self) -> NDArray[np.float64]:
        """Get velocity as ENU vector."""
        return np.array([
            self.velocity_east,
            self.velocity_north,
            self.velocity_up
        ], dtype=np.float64)


# =============================================================================
# TRAJECTORY PREDICTOR
# =============================================================================

@dataclass
class PredictedState:
    """Predicted future state of an object."""

    time_offset_s: float  # Time from now
    position_enu: NDArray[np.float64]  # ENU position
    velocity_enu: NDArray[np.float64]  # ENU velocity
    uncertainty_m: float  # Position uncertainty (grows with time)


class TrajectoryPredictor:
    """
    Predicts future trajectories for ownship and obstacles.

    Supports multiple prediction models:
        - Linear (constant velocity)
        - Curved (constant turn rate)
        - Intent-based (if waypoints known)
    """

    def __init__(self, enu_frame: ENUFrame):
        self.enu_frame = enu_frame

    def predict_ownship(
        self,
        state: OwnshipState,
        horizon_s: float,
        dt_s: float = 1.0,
    ) -> list[PredictedState]:
        """
        Predict ownship trajectory.

        Args:
            state: Current ownship state
            horizon_s: Prediction horizon
            dt_s: Time step

        Returns:
            List of predicted states
        """
        predictions = []

        pos_enu = state.to_enu(self.enu_frame)
        vel_enu = state.get_velocity_enu()

        t = 0.0
        while t <= horizon_s:
            # Linear prediction
            pred_pos = pos_enu + vel_enu * t

            # Uncertainty grows with time
            uncertainty = 5.0 + 0.5 * t  # Base + growth rate

            predictions.append(PredictedState(
                time_offset_s=t,
                position_enu=pred_pos.copy(),
                velocity_enu=vel_enu.copy(),
                uncertainty_m=uncertainty,
            ))

            t += dt_s

        return predictions

    def predict_track(
        self,
        managed_track: ManagedTrack,
        horizon_s: float,
        dt_s: float = 1.0,
    ) -> list[PredictedState]:
        """
        Predict obstacle trajectory from Kalman filter state.

        Args:
            managed_track: Track with Kalman filter
            horizon_s: Prediction horizon
            dt_s: Time step

        Returns:
            List of predicted states
        """
        predictions = []

        kf = managed_track.filter
        pos_enu = kf.get_position()
        vel_enu = kf.get_velocity()

        # Get position uncertainty from covariance
        pos_cov = kf.P[:3, :3]
        base_uncertainty = float(np.sqrt(np.trace(pos_cov) / 3))

        t = 0.0
        while t <= horizon_s:
            # Linear prediction (CV model)
            pred_pos = pos_enu + vel_enu * t

            # Uncertainty grows based on velocity uncertainty
            vel_cov = kf.P[3:6, 3:6]
            vel_uncertainty = float(np.sqrt(np.trace(vel_cov) / 3))
            uncertainty = base_uncertainty + vel_uncertainty * t

            predictions.append(PredictedState(
                time_offset_s=t,
                position_enu=pred_pos.copy(),
                velocity_enu=vel_enu.copy(),
                uncertainty_m=uncertainty,
            ))

            t += dt_s

        return predictions


# =============================================================================
# CONFLICT ANALYSIS
# =============================================================================

@dataclass
class ConflictAnalysis:
    """Detailed analysis of a potential conflict."""

    track_id: str
    obstacle_id: str | None

    # CPA analysis
    cpa: CPAResult

    # TTC analysis
    ttc: TTCResult

    # Current state
    current_distance_m: float
    current_horizontal_m: float
    current_vertical_m: float

    # Alert determination
    alert_level: AlertLevel

    # Predicted conflict
    time_to_conflict_s: float
    distance_at_conflict_m: float

    # Separation violation
    horizontal_violation: bool
    vertical_violation: bool

    # Evasion suggestion
    suggested_evasion: NDArray[np.float64] | None = None


class ConflictDetector:
    """
    Detects and analyzes conflicts between ownship and obstacles.

    Workflow:
        1. Update ownship state
        2. Get tracked obstacles from TrackManager
        3. Predict trajectories
        4. Compute CPA/TTC for each pair
        5. Classify alert levels
        6. Generate ConflictAlert objects

    Example:
        >>> detector = ConflictDetector(tracker)
        >>>
        >>> # Each update cycle:
        >>> ownship = OwnshipState(lat=28.6, lon=77.2, alt=500, ...)
        >>> alerts = detector.detect_conflicts(ownship)
        >>>
        >>> for alert in alerts:
        ...     if alert.level >= AlertLevel.WARNING:
        ...         execute_evasion(alert)
    """

    def __init__(
        self,
        tracker: TrackManager,
        config: ConflictDetectorConfig | None = None,
        separation_minima: dict[AlertLevel, SeparationMinima] | None = None,
    ):
        self.tracker = tracker
        self.config = config or ConflictDetectorConfig()
        self.separation_minima = separation_minima or DEFAULT_SEPARATION_MINIMA

        self.enu_frame = tracker.enu_frame
        self.predictor = TrajectoryPredictor(self.enu_frame)

        # Alert state for hysteresis
        self._last_alerts: dict[str, ConflictAlert] = {}
        self._last_alert_time: dict[str, datetime] = {}

        logger.info("ConflictDetector initialized")

    def detect_conflicts(
        self,
        ownship: OwnshipState,
        include_static: bool = False,
    ) -> list[ConflictAlert]:
        """
        Detect conflicts with all tracked obstacles.

        Args:
            ownship: Current ownship state
            include_static: Include static obstacles (buildings, towers)

        Returns:
            List of ConflictAlert objects, sorted by severity
        """
        alerts = []

        # Get ownship trajectory prediction
        self.predictor.predict_ownship(
            ownship,
            self.config.medium_term_horizon_s,
            self.config.prediction_dt_s,
        )

        ownship_pos = ownship.to_enu(self.enu_frame)
        ownship_vel = ownship.get_velocity_enu()

        # Analyze each tracked obstacle
        for _track_id, managed_track in self.tracker.tracks.items():
            # Skip tentative tracks for alerts
            if managed_track.status == TrackStatus.TENTATIVE:
                continue

            analysis = self._analyze_conflict(
                ownship_pos, ownship_vel, ownship.effective_radius,
                managed_track,
            )

            if analysis.alert_level != AlertLevel.NONE:
                alert = self._create_alert(ownship, managed_track, analysis)
                alerts.append(alert)

        # Sort by severity (highest first)
        alerts.sort(key=lambda a: a.alert_level.value, reverse=True)

        # Update alert history
        for alert in alerts:
            self._last_alerts[alert.obstacle_id] = alert
            self._last_alert_time[alert.obstacle_id] = datetime.now(timezone.utc)

        return alerts

    def _analyze_conflict(
        self,
        ownship_pos: NDArray[np.float64],
        ownship_vel: NDArray[np.float64],
        ownship_radius: float,
        track: ManagedTrack,
    ) -> ConflictAnalysis:
        """Analyze potential conflict with a single track."""

        # Get track state
        track_pos = track.filter.get_position()
        track_vel = track.filter.get_velocity()

        # Estimate track radius from geometry (if available)
        track_radius = 50.0  # Default for aircraft

        # Current distances
        delta = track_pos - ownship_pos
        current_distance = float(np.linalg.norm(delta))
        current_horizontal = float(np.linalg.norm(delta[:2]))
        current_vertical = abs(delta[2])

        # Compute CPA
        cpa = compute_cpa(ownship_pos, ownship_vel, track_pos, track_vel)

        # Compute TTC with safety buffers
        ttc = compute_ttc_spheres(
            ownship_pos, ownship_vel, ownship_radius,
            track_pos, track_vel, track_radius,
            max_time=self.config.medium_term_horizon_s,
        )

        # Determine alert level
        alert_level = self._classify_alert(cpa, ttc, current_distance)

        # Check separation violations based on alert level
        # Use WARNING level separation as default for violation check
        warning_sep = self.separation_minima.get(
            AlertLevel.WARNING,
            SeparationMinima(horizontal_m=200.0, vertical_m=100.0, time_s=15.0)
        )
        horizontal_violation = cpa.horizontal_distance < warning_sep.horizontal_m
        vertical_violation = cpa.vertical_distance < warning_sep.vertical_m

        # Calculate evasion if needed
        suggested_evasion = None
        if self.config.suggest_evasion and alert_level >= AlertLevel.WARNING:
            suggested_evasion = self._suggest_evasion(
                ownship_pos, ownship_vel, track_pos, track_vel
            )

        return ConflictAnalysis(
            track_id=track.track_id,
            obstacle_id=track.obstacle_id,
            cpa=cpa,
            ttc=ttc,
            current_distance_m=current_distance,
            current_horizontal_m=current_horizontal,
            current_vertical_m=current_vertical,
            alert_level=alert_level,
            time_to_conflict_s=cpa.time_to_cpa,
            distance_at_conflict_m=cpa.distance_at_cpa,
            horizontal_violation=horizontal_violation,
            vertical_violation=vertical_violation,
            suggested_evasion=suggested_evasion,
        )

    def _classify_alert(
        self,
        cpa: CPAResult,
        ttc: TTCResult,
        current_distance: float,
    ) -> AlertLevel:
        """Classify alert level based on CPA and TTC."""

        # Not converging = lower priority
        if not cpa.is_converging:
            if current_distance < self.config.advisory_distance_m:
                return AlertLevel.ADVISORY
            return AlertLevel.NONE

        # CRITICAL: Imminent collision
        if (cpa.time_to_cpa < self.config.critical_time_s and
            cpa.distance_at_cpa < self.config.critical_distance_m):
            return AlertLevel.CRITICAL

        if ttc.will_collide and ttc.time_to_collision < self.config.critical_time_s:
            return AlertLevel.CRITICAL

        # WARNING: Prepare maneuver
        if (cpa.time_to_cpa < self.config.warning_time_s and
            cpa.distance_at_cpa < self.config.warning_distance_m):
            return AlertLevel.WARNING

        if ttc.will_collide and ttc.time_to_collision < self.config.warning_time_s:
            return AlertLevel.WARNING

        # CAUTION: Monitor closely
        if (cpa.time_to_cpa < self.config.caution_time_s and
            cpa.distance_at_cpa < self.config.caution_distance_m):
            return AlertLevel.CAUTION

        # ADVISORY: Awareness
        if (cpa.time_to_cpa < self.config.advisory_time_s and
            cpa.distance_at_cpa < self.config.advisory_distance_m):
            return AlertLevel.ADVISORY

        return AlertLevel.NONE

    def _suggest_evasion(
        self,
        own_pos: NDArray[np.float64],
        own_vel: NDArray[np.float64],
        track_pos: NDArray[np.float64],
        track_vel: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Suggest evasion direction vector.

        Strategy: Turn away from relative velocity vector,
        perpendicular to collision course.
        """
        # Relative position and velocity
        rel_pos = track_pos - own_pos
        rel_vel = track_vel - own_vel

        # Perpendicular direction in horizontal plane
        perp = np.array([-rel_pos[1], rel_pos[0], 0], dtype=np.float64)
        perp_norm = np.linalg.norm(perp)

        if perp_norm > 0.1:
            perp /= perp_norm
        else:
            # Fallback: climb
            perp = np.array([0, 0, 1], dtype=np.float64)

        # Scale to desired clearance
        evasion = perp * self.config.min_evasion_clearance_m

        return evasion

    def _create_alert(
        self,
        ownship: OwnshipState,
        track: ManagedTrack,
        analysis: ConflictAnalysis,
    ) -> ConflictAlert:
        """Create ConflictAlert from analysis."""

        # Get track position in LLA
        track_pos_enu = track.filter.get_position()
        track_lat, track_lon, track_alt = self.enu_frame.enu_to_lla(track_pos_enu)

        # Determine recommended action based on analysis
        recommended = ""
        if analysis.alert_level >= AlertLevel.WARNING:
            if analysis.suggested_evasion is not None:
                if analysis.suggested_evasion[2] > 0:
                    recommended = "climb"
                elif analysis.suggested_evasion[2] < 0:
                    recommended = "descend"
                elif analysis.suggested_evasion[0] > 0:
                    recommended = "turn_right"
                else:
                    recommended = "turn_left"

        return ConflictAlert(
            own_position=(ownship.latitude, ownship.longitude, ownship.altitude_m),
            obstacle_id=analysis.track_id,
            alert_level=analysis.alert_level,
            horizontal_distance_m=analysis.cpa.horizontal_distance,
            vertical_distance_m=analysis.cpa.vertical_distance,
            slant_distance_m=analysis.current_distance_m,
            time_to_cpa_s=analysis.cpa.time_to_cpa,
            distance_at_cpa_m=analysis.cpa.distance_at_cpa,
            time_to_collision_s=analysis.ttc.time_to_collision if analysis.ttc.will_collide else None,
            recommended_action=recommended,
        )

    def get_conflict_zones(
        self,
        ownship: OwnshipState,
        horizon_s: float = 60.0,
    ) -> list[ConflictZone]:
        """
        Identify zones where conflicts may occur.

        Returns spatial regions to avoid based on predicted trajectories.
        """
        zones = []

        ownship_pos = ownship.to_enu(self.enu_frame)
        ownship_vel = ownship.get_velocity_enu()

        for track_id, track in self.tracker.tracks.items():
            if track.status == TrackStatus.TENTATIVE:
                continue

            track_pos = track.filter.get_position()
            track_vel = track.filter.get_velocity()

            cpa = compute_cpa(ownship_pos, ownship_vel, track_pos, track_vel)

            if cpa.is_converging and cpa.time_to_cpa < horizon_s:
                # Create zone around CPA point
                cpa_enu = (cpa.position1_at_cpa + cpa.position2_at_cpa) / 2
                cpa_lat, cpa_lon, cpa_alt = self.enu_frame.enu_to_lla(cpa_enu)

                # Use CAUTION separation for zone sizing
                caution_sep = self.separation_minima.get(
                    AlertLevel.CAUTION,
                    SeparationMinima(horizontal_m=500.0, vertical_m=200.0, time_s=30.0)
                )

                # Determine alert level for zone
                zone_alert = self._classify_alert(cpa, TTCResult(
                    float('inf'), None, False, 0, 0, 0
                ), float(np.linalg.norm(track_pos - ownship_pos)))

                # Create bounding cylinder for zone geometry
                zone_geometry = BoundingCylinder(
                    center_lat=cpa_lat,
                    center_lon=cpa_lon,
                    base_alt_m=cpa_alt - caution_sep.vertical_m,
                    radius_m=caution_sep.horizontal_m * 2,
                    height_m=caution_sep.vertical_m * 4,
                )

                zone = ConflictZone(
                    zone_id=f"CZ-{track_id}",
                    obstacle_id=track_id,
                    geometry=zone_geometry,
                    alert_level=zone_alert,
                )
                zones.append(zone)

        return zones

    def get_all_alerts_summary(self) -> dict[str, Any]:
        """Get summary of current alert state."""
        alerts_by_level = {level.name: 0 for level in AlertLevel}

        for alert in self._last_alerts.values():
            alerts_by_level[alert.alert_level.name] += 1

        return {
            'total_alerts': len(self._last_alerts),
            'by_level': alerts_by_level,
            'highest_level': max(
                (a.alert_level for a in self._last_alerts.values()),
                default=AlertLevel.NONE
            ).name,
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_conflict_detector(
    tracker: TrackManager,
    critical_distance_m: float = 50.0,
    warning_distance_m: float = 200.0,
) -> ConflictDetector:
    """
    Create configured conflict detector.

    Args:
        tracker: TrackManager instance
        critical_distance_m: Distance for CRITICAL alerts
        warning_distance_m: Distance for WARNING alerts

    Returns:
        Configured ConflictDetector
    """
    config = ConflictDetectorConfig(
        critical_distance_m=critical_distance_m,
        warning_distance_m=warning_distance_m,
    )
    return ConflictDetector(tracker, config)


def check_immediate_conflicts(
    detector: ConflictDetector,
    ownship: OwnshipState,
) -> tuple[bool, ConflictAlert | None]:
    """
    Quick check for immediate conflicts.

    Args:
        detector: ConflictDetector instance
        ownship: Current ownship state

    Returns:
        (has_critical_alert, most_severe_alert)
    """
    alerts = detector.detect_conflicts(ownship)

    if not alerts:
        return False, None

    most_severe = alerts[0]  # Already sorted by severity
    has_critical = most_severe.alert_level == AlertLevel.CRITICAL

    return has_critical, most_severe
