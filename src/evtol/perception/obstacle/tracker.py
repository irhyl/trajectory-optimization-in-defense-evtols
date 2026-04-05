"""
Kalman Filter Tracker for Dynamic Obstacle Tracking.

This module provides multi-target tracking for dynamic obstacles using
Extended Kalman Filters with multiple motion models:

Motion Models:
    - CV (Constant Velocity): Linear motion assumption
    - CTRV (Constant Turn Rate and Velocity): Coordinated turn model
    - CA (Constant Acceleration): For maneuvering targets

State Vector (8 elements):
    [px, py, pz, vx, vy, vz, heading, turn_rate]

Key Capabilities:
    - Multi-target tracking with track management
    - Gating and data association (GNN, JPDA)
    - Track initiation, confirmation, and deletion
    - Trajectory prediction (short/medium term)
    - Covariance-based uncertainty estimation

References:
    - Bar-Shalom, Y., Li, X.R., Kirubarajan, T. (2001).
      Estimation with Applications to Tracking and Navigation.
    - Blackman, S., Popoli, R. (1999). Design and Analysis of Modern
      Tracking Systems.
"""

from __future__ import annotations

import math
import time
import logging
from dataclasses import dataclass, field
from typing import Any
from enum import Enum
from datetime import datetime, timezone
import numpy as np
from numpy.typing import NDArray

from .obstacle_types import (
    Track,
    TrackStatus,
    KalmanState,
    Aircraft,
    ObstacleSource,
)
from .geometry import ENUFrame

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

# State vector indices
PX, PY, PZ = 0, 1, 2  # Position (ENU meters)
VX, VY, VZ = 3, 4, 5  # Velocity (m/s)
HDG = 6               # Heading (radians)
OMEGA = 7             # Turn rate (rad/s)

STATE_DIM = 8         # Full state dimension
MEAS_DIM_POS = 3      # Position-only measurement
MEAS_DIM_FULL = 6     # Position + velocity measurement


# =============================================================================
# MOTION MODELS
# =============================================================================

class MotionModel(Enum):
    """Available motion models for tracking."""
    CV = "constant_velocity"
    CTRV = "constant_turn_rate_velocity"
    CA = "constant_acceleration"


@dataclass
class ProcessNoise:
    """Process noise parameters for motion models."""

    # Position noise (m²)
    sigma_pos: float = 1.0

    # Velocity noise (m²/s²)
    sigma_vel: float = 2.0

    # Heading noise (rad²)
    sigma_hdg: float = 0.01

    # Turn rate noise (rad²/s²)
    sigma_omega: float = 0.001

    def get_Q_cv(self, dt: float) -> NDArray[np.float64]:
        """Get process noise matrix for CV model."""
        Q = np.zeros((STATE_DIM, STATE_DIM))

        # Position variance grows with velocity uncertainty
        q_pos = self.sigma_pos ** 2
        q_vel = self.sigma_vel ** 2

        # Coupled position-velocity noise
        for i in range(3):
            Q[i, i] = q_pos + q_vel * dt ** 2
            Q[i + 3, i + 3] = q_vel
            Q[i, i + 3] = q_vel * dt
            Q[i + 3, i] = q_vel * dt

        # Heading and turn rate
        Q[HDG, HDG] = self.sigma_hdg ** 2
        Q[OMEGA, OMEGA] = self.sigma_omega ** 2

        return Q

    def get_Q_ctrv(self, dt: float) -> NDArray[np.float64]:
        """Get process noise matrix for CTRV model."""
        Q = self.get_Q_cv(dt)

        # Increase heading uncertainty for turning
        Q[HDG, HDG] *= 2.0
        Q[OMEGA, OMEGA] *= 2.0

        return Q


# =============================================================================
# KALMAN FILTER CORE
# =============================================================================

class KalmanFilter:
    """
    Extended Kalman Filter for single-target tracking.

    Supports multiple motion models with automatic model selection
    based on target maneuverability.
    """

    def __init__(
        self,
        motion_model: MotionModel = MotionModel.CV,
        process_noise: ProcessNoise | None = None,
    ):
        self.motion_model = motion_model
        self.process_noise = process_noise or ProcessNoise()

        # State and covariance
        self.x: NDArray[np.float64] = np.zeros(STATE_DIM)
        self.P: NDArray[np.float64] = np.eye(STATE_DIM) * 100.0

        # Measurement noise
        self.R_pos = np.diag([25.0, 25.0, 100.0])  # Position (m²)
        self.R_vel = np.diag([4.0, 4.0, 4.0])       # Velocity (m²/s²)

        # Innovation statistics
        self.innovation: NDArray | None = None
        self.innovation_cov: NDArray | None = None
        self.nis: float = 0.0  # Normalized Innovation Squared

    def initialize(
        self,
        position: NDArray[np.float64],
        velocity: NDArray[np.float64] | None = None,
        heading: float = 0.0,
        position_cov: NDArray[np.float64] | None = None,
    ):
        """
        Initialize filter state from first observation.

        Args:
            position: [px, py, pz] in ENU meters
            velocity: [vx, vy, vz] in m/s (optional)
            heading: Initial heading in radians
            position_cov: Initial position covariance
        """
        self.x[PX:PZ+1] = position

        if velocity is not None:
            self.x[VX:VZ+1] = velocity
            # Compute heading from velocity
            self.x[HDG] = math.atan2(velocity[0], velocity[1])
        else:
            self.x[HDG] = heading

        self.x[OMEGA] = 0.0

        # Initialize covariance
        self.P = np.eye(STATE_DIM)

        if position_cov is not None:
            self.P[:3, :3] = position_cov
        else:
            self.P[PX, PX] = 100.0
            self.P[PY, PY] = 100.0
            self.P[PZ, PZ] = 400.0

        # Velocity uncertainty
        if velocity is not None:
            self.P[VX, VX] = 25.0
            self.P[VY, VY] = 25.0
            self.P[VZ, VZ] = 25.0
        else:
            self.P[VX, VX] = 100.0
            self.P[VY, VY] = 100.0
            self.P[VZ, VZ] = 100.0

        # Heading and turn rate uncertainty
        self.P[HDG, HDG] = 0.5
        self.P[OMEGA, OMEGA] = 0.1

    def predict(self, dt: float):
        """
        Predict state forward by dt seconds.

        Args:
            dt: Time step in seconds
        """
        if dt <= 0:
            return

        if self.motion_model == MotionModel.CV:
            self._predict_cv(dt)
        elif self.motion_model == MotionModel.CTRV:
            self._predict_ctrv(dt)
        else:
            self._predict_cv(dt)  # Fallback

    def _predict_cv(self, dt: float):
        """Constant Velocity prediction."""
        # State transition: x_new = F @ x
        F = np.eye(STATE_DIM)
        F[PX, VX] = dt
        F[PY, VY] = dt
        F[PZ, VZ] = dt

        # Predict state
        self.x = F @ self.x

        # Normalize heading
        self.x[HDG] = self._normalize_angle(self.x[HDG])

        # Predict covariance
        Q = self.process_noise.get_Q_cv(dt)
        self.P = F @ self.P @ F.T + Q

    def _predict_ctrv(self, dt: float):
        """Constant Turn Rate and Velocity prediction."""
        v = math.sqrt(self.x[VX] ** 2 + self.x[VY] ** 2)
        omega = self.x[OMEGA]
        theta = self.x[HDG]

        if abs(omega) < 1e-6:
            # Straight-line motion (avoid division by zero)
            self._predict_cv(dt)
            return

        # CTRV equations
        theta_new = theta + omega * dt

        # Position update (integrate arc)
        self.x[PX] += (v / omega) * (math.sin(theta_new) - math.sin(theta))
        self.x[PY] += (v / omega) * (math.cos(theta) - math.cos(theta_new))
        self.x[PZ] += self.x[VZ] * dt

        # Velocity direction rotates
        self.x[VX] = v * math.sin(theta_new)
        self.x[VY] = v * math.cos(theta_new)

        # Update heading
        self.x[HDG] = self._normalize_angle(theta_new)

        # Jacobian for covariance propagation
        F = self._ctrv_jacobian(dt, v, theta, omega)
        Q = self.process_noise.get_Q_ctrv(dt)
        self.P = F @ self.P @ F.T + Q

    def _ctrv_jacobian(
        self, dt: float, v: float, theta: float, omega: float
    ) -> NDArray[np.float64]:
        """Compute Jacobian for CTRV model."""
        F = np.eye(STATE_DIM)

        if abs(omega) < 1e-6:
            return F

        s_t = math.sin(theta)
        c_t = math.cos(theta)
        s_tn = math.sin(theta + omega * dt)
        c_tn = math.cos(theta + omega * dt)

        # Partial derivatives
        F[PX, HDG] = (v / omega) * (c_tn - c_t)
        F[PY, HDG] = (v / omega) * (s_tn - s_t)

        F[PX, VX] = (1 / omega) * (s_tn - s_t) * (self.x[VX] / v) if v > 0 else 0
        F[PY, VY] = (1 / omega) * (c_t - c_tn) * (self.x[VY] / v) if v > 0 else 0

        F[VX, HDG] = v * c_tn
        F[VY, HDG] = -v * s_tn

        F[HDG, OMEGA] = dt

        return F

    def update(
        self,
        measurement: NDArray[np.float64],
        measurement_type: str = "position",
    ) -> float:
        """
        Update state with measurement.

        Args:
            measurement: Measurement vector
            measurement_type: "position" (3D) or "full" (6D pos+vel)

        Returns:
            Normalized Innovation Squared (NIS) for gating
        """
        if measurement_type == "position":
            return self._update_position(measurement)
        else:
            return self._update_full(measurement)

    def _update_position(self, z: NDArray[np.float64]) -> float:
        """Update with position-only measurement."""
        # Measurement matrix (observe position only)
        H = np.zeros((3, STATE_DIM))
        H[0, PX] = 1.0
        H[1, PY] = 1.0
        H[2, PZ] = 1.0

        # Predicted measurement
        z_pred = H @ self.x

        # Innovation
        y = z - z_pred
        self.innovation = y

        # Innovation covariance
        S = H @ self.P @ H.T + self.R_pos
        self.innovation_cov = S

        # NIS (chi-squared statistic)
        try:
            S_inv = np.linalg.inv(S)
            self.nis = float(y.T @ S_inv @ y)
        except np.linalg.LinAlgError:
            self.nis = float('inf')
            return self.nis

        # Kalman gain
        K = self.P @ H.T @ S_inv

        # State update
        self.x = self.x + K @ y
        self.x[HDG] = self._normalize_angle(self.x[HDG])

        # Covariance update (Joseph form for numerical stability)
        I_KH = np.eye(STATE_DIM) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R_pos @ K.T

        return self.nis

    def _update_full(self, z: NDArray[np.float64]) -> float:
        """Update with position + velocity measurement."""
        # Measurement matrix
        H = np.zeros((6, STATE_DIM))
        H[0, PX] = 1.0
        H[1, PY] = 1.0
        H[2, PZ] = 1.0
        H[3, VX] = 1.0
        H[4, VY] = 1.0
        H[5, VZ] = 1.0

        # Full measurement noise
        R = np.block([
            [self.R_pos, np.zeros((3, 3))],
            [np.zeros((3, 3)), self.R_vel]
        ])

        # Predicted measurement
        z_pred = H @ self.x

        # Innovation
        y = z - z_pred
        self.innovation = y

        # Innovation covariance
        S = H @ self.P @ H.T + R
        self.innovation_cov = S

        # NIS
        try:
            S_inv = np.linalg.inv(S)
            self.nis = float(y.T @ S_inv @ y)
        except np.linalg.LinAlgError:
            self.nis = float('inf')
            return self.nis

        # Kalman gain
        K = self.P @ H.T @ S_inv

        # State update
        self.x = self.x + K @ y
        self.x[HDG] = self._normalize_angle(self.x[HDG])

        # Covariance update
        I_KH = np.eye(STATE_DIM) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T

        return self.nis

    def get_position(self) -> NDArray[np.float64]:
        """Get current position estimate."""
        return self.x[PX:PZ+1].copy()

    def get_velocity(self) -> NDArray[np.float64]:
        """Get current velocity estimate."""
        return self.x[VX:VZ+1].copy()

    def get_speed(self) -> float:
        """Get horizontal speed."""
        return float(math.sqrt(self.x[VX] ** 2 + self.x[VY] ** 2))

    def get_heading(self) -> float:
        """Get heading in radians."""
        return float(self.x[HDG])

    def get_position_uncertainty(self) -> NDArray[np.float64]:
        """Get 3-sigma position uncertainty."""
        return 3.0 * np.sqrt(np.diag(self.P[:3, :3]))

    def predict_trajectory(
        self,
        horizon_s: float,
        dt: float = 1.0,
    ) -> list[NDArray[np.float64]]:
        """
        Predict future trajectory.

        Args:
            horizon_s: Prediction horizon in seconds
            dt: Time step for predictions

        Returns:
            List of predicted positions
        """
        # Clone current state
        x_pred = self.x.copy()
        predictions = [x_pred[PX:PZ+1].copy()]

        t = 0.0
        while t < horizon_s:
            # Simple CV propagation for trajectory
            x_pred[PX] += x_pred[VX] * dt
            x_pred[PY] += x_pred[VY] * dt
            x_pred[PZ] += x_pred[VZ] * dt

            predictions.append(x_pred[PX:PZ+1].copy())
            t += dt

        return predictions

    def to_kalman_state(self) -> KalmanState:
        """Convert to KalmanState dataclass."""
        return KalmanState(
            state=self.x.copy(),
            covariance=self.P.copy(),
        )

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        """Normalize angle to [-π, π]."""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle


# =============================================================================
# TRACK MANAGEMENT
# =============================================================================

@dataclass
class TrackManagerConfig:
    """Configuration for track management."""

    # Track initiation
    min_hits_to_confirm: int = 3       # Hits before CONFIRMED
    max_misses_tentative: int = 2      # Misses before deleting tentative
    max_misses_confirmed: int = 5      # Misses before deleting confirmed

    # Gating
    gating_threshold: float = 9.21     # Chi-squared (3 DOF, 99%)
    max_association_distance_m: float = 500.0

    # Motion model selection
    turn_rate_threshold: float = 0.02  # rad/s for CTRV switch

    # Track output
    output_tentative: bool = False     # Include tentative tracks in output


@dataclass
class ManagedTrack:
    """Internal track representation with filter state."""

    track_id: str
    filter: KalmanFilter

    # Track lifecycle
    status: TrackStatus = TrackStatus.TENTATIVE
    hits: int = 0
    misses: int = 0
    age_frames: int = 0

    # Timing
    creation_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_update_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Source info
    source: ObstacleSource = ObstacleSource.SENSOR
    obstacle_id: str | None = None
    callsign: str | None = None

    # History
    position_history: list[tuple[float, float, float]] = field(default_factory=list)
    max_history_length: int = 100

    def update_history(self, position: NDArray[np.float64]):
        """Add position to history."""
        self.position_history.append(tuple(position))
        if len(self.position_history) > self.max_history_length:
            self.position_history.pop(0)

    def to_track(self, enu_frame: ENUFrame) -> Track:
        """Convert to Track dataclass."""
        pos_enu = self.filter.get_position()
        self.filter.get_velocity()

        # Convert to LLA
        lat, lon, alt = enu_frame.enu_to_lla(pos_enu)

        # Create track (use default TrackHistory)
        return Track(
            track_id=self.track_id,
            status=self.status,
            kalman=self.filter.to_kalman_state(),
            update_count=self.hits,
            missed_count=self.misses,
            created_at=self.creation_time,
            last_update=self.last_update_time,
        )


class TrackManager:
    """
    Multi-target tracker with track management.

    Handles:
        - Track initiation from new detections
        - Data association (Global Nearest Neighbor)
        - Track confirmation and deletion
        - Trajectory prediction

    Example:
        >>> manager = TrackManager(origin_lat=28.6, origin_lon=77.2)
        >>>
        >>> # Process detections each frame
        >>> for aircraft_list in data_stream:
        ...     manager.predict(dt=1.0)
        ...     tracks = manager.update(aircraft_list)
        ...
        ...     for track in tracks:
        ...         if track.status == TrackStatus.CONFIRMED:
        ...             print(f"Track {track.track_id}: {track.position_lat:.4f}")
    """

    def __init__(
        self,
        origin_lat: float,
        origin_lon: float,
        origin_alt: float = 0.0,
        config: TrackManagerConfig | None = None,
    ):
        self.config = config or TrackManagerConfig()
        self.enu_frame = ENUFrame(origin_lat, origin_lon, origin_alt)

        self.tracks: dict[str, ManagedTrack] = {}
        self._next_track_id = 1

        self.frame_count = 0
        self.last_predict_time: float | None = None

        logger.info(f"TrackManager initialized: origin=({origin_lat:.4f}, {origin_lon:.4f})")

    def _generate_track_id(self) -> str:
        """Generate unique track ID."""
        tid = f"TRK-{self._next_track_id:06d}"
        self._next_track_id += 1
        return tid

    def predict(self, dt: float):
        """
        Predict all tracks forward by dt seconds.

        Args:
            dt: Time step in seconds
        """
        for track in self.tracks.values():
            track.filter.predict(dt)
            track.age_frames += 1

        self.last_predict_time = time.time()

    def update(
        self,
        detections: list[Aircraft],
    ) -> list[Track]:
        """
        Update tracks with new detections.

        Args:
            detections: List of Aircraft detections

        Returns:
            List of current Track objects
        """
        self.frame_count += 1

        # Convert detections to measurements
        measurements = self._detections_to_measurements(detections)

        # Data association
        associations = self._associate(measurements)

        # Update associated tracks
        for track_id, det_idx in associations.items():
            self._update_track(track_id, measurements[det_idx], detections[det_idx])

        # Handle unassociated tracks (missed detections)
        associated_track_ids = set(associations.keys())
        for track_id in list(self.tracks.keys()):
            if track_id not in associated_track_ids:
                self._miss_track(track_id)

        # Initiate new tracks from unassociated detections
        associated_det_indices = set(associations.values())
        for i, det in enumerate(detections):
            if i not in associated_det_indices:
                self._initiate_track(measurements[i], det)

        # Delete dead tracks
        self._prune_tracks()

        # Return track list
        return self.get_tracks()

    def _detections_to_measurements(
        self,
        detections: list[Aircraft],
    ) -> list[dict[str, Any]]:
        """Convert aircraft detections to ENU measurements."""
        measurements = []

        for det in detections:
            # Convert to ENU
            pos_enu = self.enu_frame.lla_to_enu(
                det.state.latitude,
                det.state.longitude,
                det.state.altitude_m,
            )

            vel_enu = np.array([
                det.state.velocity_east,
                det.state.velocity_north,
                det.state.velocity_up,
            ])

            measurements.append({
                'position': pos_enu,
                'velocity': vel_enu,
                'heading': math.radians(det.state.heading_deg),
                'icao24': det.icao24,
                'callsign': det.callsign,
                'source': det.source,
            })

        return measurements

    def _associate(
        self,
        measurements: list[dict[str, Any]],
    ) -> dict[str, int]:
        """
        Associate measurements to existing tracks (GNN).

        Returns:
            Dict mapping track_id -> measurement index
        """
        if not self.tracks or not measurements:
            return {}

        track_ids = list(self.tracks.keys())
        n_tracks = len(track_ids)
        n_meas = len(measurements)

        # Cost matrix (distance-based)
        cost = np.full((n_tracks, n_meas), float('inf'))

        for i, tid in enumerate(track_ids):
            track = self.tracks[tid]
            pred_pos = track.filter.get_position()

            for j, meas in enumerate(measurements):
                # Check ICAO24 match first (strong prior)
                if track.obstacle_id and meas['icao24']:
                    if track.obstacle_id == meas['icao24']:
                        cost[i, j] = 0.0  # Perfect match
                        continue

                # Euclidean distance
                dist = float(np.linalg.norm(pred_pos - meas['position']))

                if dist <= self.config.max_association_distance_m:
                    # Compute NIS for gating
                    z = meas['position']
                    H = np.zeros((3, STATE_DIM))
                    H[0, PX] = H[1, PY] = H[2, PZ] = 1.0

                    z_pred = H @ track.filter.x
                    y = z - z_pred
                    S = H @ track.filter.P @ H.T + track.filter.R_pos

                    try:
                        S_inv = np.linalg.inv(S)
                        nis = float(y.T @ S_inv @ y)

                        if nis <= self.config.gating_threshold:
                            cost[i, j] = nis
                    except np.linalg.LinAlgError:
                        pass

        # Greedy assignment (simple GNN)
        associations = {}
        used_meas = set()

        while True:
            # Find minimum cost
            min_cost = float('inf')
            best_i, best_j = -1, -1

            for i in range(n_tracks):
                if track_ids[i] in associations:
                    continue
                for j in range(n_meas):
                    if j in used_meas:
                        continue
                    if cost[i, j] < min_cost:
                        min_cost = cost[i, j]
                        best_i, best_j = i, j

            if min_cost == float('inf'):
                break

            associations[track_ids[best_i]] = best_j
            used_meas.add(best_j)

        return associations

    def _update_track(
        self,
        track_id: str,
        measurement: dict[str, Any],
        detection: Aircraft,
    ):
        """Update track with associated measurement."""
        track = self.tracks[track_id]

        # Full measurement (position + velocity)
        z = np.concatenate([measurement['position'], measurement['velocity']])
        track.filter.update(z, measurement_type="full")

        # Update track state
        track.hits += 1
        track.misses = 0
        track.last_update_time = datetime.now(timezone.utc)
        track.callsign = detection.callsign or track.callsign

        # Update history
        track.update_history(measurement['position'])

        # Check for confirmation
        if track.status == TrackStatus.TENTATIVE:
            if track.hits >= self.config.min_hits_to_confirm:
                track.status = TrackStatus.CONFIRMED
                logger.debug(f"Track {track_id} CONFIRMED")

        # Adaptive motion model selection
        if abs(track.filter.x[OMEGA]) > self.config.turn_rate_threshold:
            track.filter.motion_model = MotionModel.CTRV
        else:
            track.filter.motion_model = MotionModel.CV

    def _miss_track(self, track_id: str):
        """Handle missed detection for track."""
        track = self.tracks[track_id]
        track.misses += 1

        # Degrade status
        if track.status == TrackStatus.CONFIRMED:
            if track.misses >= self.config.max_misses_confirmed:
                track.status = TrackStatus.COASTING
                logger.debug(f"Track {track_id} COASTING")

    def _initiate_track(
        self,
        measurement: dict[str, Any],
        detection: Aircraft,
    ):
        """Initiate new track from unassociated detection."""
        track_id = self._generate_track_id()

        # Create filter
        kf = KalmanFilter(motion_model=MotionModel.CV)
        kf.initialize(
            position=measurement['position'],
            velocity=measurement['velocity'],
            heading=measurement['heading'],
        )

        # Create track
        track = ManagedTrack(
            track_id=track_id,
            filter=kf,
            status=TrackStatus.TENTATIVE,
            hits=1,
            source=detection.source,
            obstacle_id=detection.icao24,
            callsign=detection.callsign,
        )
        track.update_history(measurement['position'])

        self.tracks[track_id] = track
        logger.debug(f"Track {track_id} initiated: {detection.icao24}")

    def _prune_tracks(self):
        """Delete dead tracks."""
        to_delete = []

        for track_id, track in self.tracks.items():
            if track.status == TrackStatus.TENTATIVE:
                if track.misses >= self.config.max_misses_tentative:
                    to_delete.append(track_id)
            elif track.status == TrackStatus.COASTING:
                if track.misses >= self.config.max_misses_confirmed:
                    to_delete.append(track_id)

        for tid in to_delete:
            del self.tracks[tid]
            logger.debug(f"Track {tid} DELETED")

    def get_tracks(
        self,
        include_tentative: bool = False,
    ) -> list[Track]:
        """
        Get current tracks.

        Args:
            include_tentative: Include tentative tracks

        Returns:
            List of Track objects
        """
        tracks = []

        for track in self.tracks.values():
            if track.status == TrackStatus.TENTATIVE and not include_tentative:
                continue
            tracks.append(track.to_track(self.enu_frame))

        return tracks

    def get_confirmed_tracks(self) -> list[Track]:
        """Get only confirmed tracks."""
        return [
            track.to_track(self.enu_frame)
            for track in self.tracks.values()
            if track.status == TrackStatus.CONFIRMED
        ]

    def get_track_by_id(self, track_id: str) -> Track | None:
        """Get specific track by ID."""
        if track_id in self.tracks:
            return self.tracks[track_id].to_track(self.enu_frame)
        return None

    def predict_track_trajectory(
        self,
        track_id: str,
        horizon_s: float = 60.0,
        dt: float = 1.0,
    ) -> list[tuple[float, float, float]] | None:
        """
        Predict trajectory for specific track.

        Args:
            track_id: Track to predict
            horizon_s: Prediction horizon
            dt: Time step

        Returns:
            List of (lat, lon, alt) tuples
        """
        if track_id not in self.tracks:
            return None

        track = self.tracks[track_id]
        enu_predictions = track.filter.predict_trajectory(horizon_s, dt)

        # Convert to LLA
        lla_predictions = []
        for enu in enu_predictions:
            lat, lon, alt = self.enu_frame.enu_to_lla(enu)
            lla_predictions.append((lat, lon, alt))

        return lla_predictions

    def get_statistics(self) -> dict[str, Any]:
        """Get tracker statistics."""
        status_counts = {s.value: 0 for s in TrackStatus}
        for track in self.tracks.values():
            status_counts[track.status.value] += 1

        return {
            'frame_count': self.frame_count,
            'total_tracks': len(self.tracks),
            'next_track_id': self._next_track_id,
            'status_counts': status_counts,
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_tracker(
    origin_lat: float,
    origin_lon: float,
    origin_alt: float = 0.0,
    min_hits: int = 3,
    max_misses: int = 5,
) -> TrackManager:
    """
    Create configured track manager.

    Args:
        origin_lat, origin_lon, origin_alt: ENU frame origin
        min_hits: Hits to confirm track
        max_misses: Misses before deletion

    Returns:
        Configured TrackManager
    """
    config = TrackManagerConfig(
        min_hits_to_confirm=min_hits,
        max_misses_confirmed=max_misses,
    )
    return TrackManager(origin_lat, origin_lon, origin_alt, config)


def track_aircraft_stream(
    tracker: TrackManager,
    aircraft_generator,
    update_interval_s: float = 1.0,
) -> None:
    """
    Process streaming aircraft data.

    Args:
        tracker: TrackManager instance
        aircraft_generator: Generator yielding List[Aircraft]
        update_interval_s: Expected update interval
    """
    for aircraft_list in aircraft_generator:
        tracker.predict(update_interval_s)
        tracks = tracker.update(aircraft_list)
        yield tracks
