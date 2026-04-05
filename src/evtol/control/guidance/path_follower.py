"""
Path Follower - Guidance Layer.

Geometric path following algorithms for waypoint navigation.
Used when timing is not critical (unlike trajectory tracking).

Implements:
- Pure pursuit controller
- Cross-track error based guidance
- Line-of-sight guidance
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class Waypoint:
    """3D waypoint with optional parameters."""
    x: float           # North (m)
    y: float           # East (m)
    z: float           # Down (m, negative = altitude)

    # Optional parameters
    speed: float = None       # Desired speed at waypoint (m/s)
    heading: float = None     # Desired heading (rad)
    radius: float = 10.0      # Acceptance radius (m)
    flyover: bool = False     # True = fly over, False = fly by


@dataclass
class PathFollowerConfig:
    """Configuration for path follower."""
    lookahead_distance: float = 50.0    # Pure pursuit lookahead (m)
    min_lookahead: float = 20.0
    max_lookahead: float = 200.0
    lookahead_gain: float = 2.0         # Lookahead = gain * speed

    # Cross-track control
    cross_track_gain: float = 0.05      # rad per m error
    max_cross_track_heading: float = np.radians(45)  # Max correction

    # Path segment
    default_speed: float = 30.0         # m/s
    waypoint_tolerance: float = 15.0    # m


class PathFollower:
    """
    Geometric path follower for waypoint navigation.

    Input: List of waypoints
    Output: Heading and speed commands

    Features:
    - Pure pursuit for smooth path following
    - Automatic waypoint advancement
    - Cross-track error minimization
    """

    def __init__(self, config: PathFollowerConfig | None = None):
        self.config = config or PathFollowerConfig()

        self._waypoints: list[Waypoint] = []
        self._current_idx: int = 0
        self._completed: bool = False

    def load_waypoints(self, waypoints: list[Waypoint]) -> None:
        """Load waypoint list."""
        self._waypoints = waypoints
        self._current_idx = 0
        self._completed = False

    def load_from_array(
        self,
        positions: np.ndarray,
        speeds: np.ndarray | None = None,
        radii: np.ndarray | None = None,
    ) -> None:
        """
        Load waypoints from numpy arrays.

        Args:
            positions: (N, 3) array [x, y, z] or [x, y, alt]
            speeds: (N,) array of speeds
            radii: (N,) array of acceptance radii
        """
        waypoints = []
        for i in range(len(positions)):
            wp = Waypoint(
                x=positions[i, 0],
                y=positions[i, 1],
                z=-positions[i, 2] if positions[i, 2] > 0 else positions[i, 2],
            )
            if speeds is not None:
                wp.speed = speeds[i]
            if radii is not None:
                wp.radius = radii[i]
            waypoints.append(wp)

        self.load_waypoints(waypoints)

    def compute_guidance(
        self,
        x: float,
        y: float,
        z: float,
        heading: float,
        speed: float,
    ) -> tuple[float, float, float, float]:
        """
        Compute guidance commands using pure pursuit.

        Args:
            x, y, z: Current position (NED)
            heading: Current heading (rad)
            speed: Current speed (m/s)

        Returns:
            (heading_cmd, speed_cmd, altitude_cmd, cross_track_error)
        """
        if len(self._waypoints) == 0 or self._completed:
            return heading, 0.0, z, 0.0

        current_wp = self._waypoints[self._current_idx]

        # Check waypoint reached
        dist_to_wp = np.sqrt(
            (current_wp.x - x)**2 +
            (current_wp.y - y)**2
        )

        if dist_to_wp < current_wp.radius:
            self._advance_waypoint()
            if self._completed:
                return heading, 0.0, current_wp.z, 0.0
            current_wp = self._waypoints[self._current_idx]

        # Pure pursuit lookahead
        lookahead = np.clip(
            self.config.lookahead_gain * speed,
            self.config.min_lookahead,
            self.config.max_lookahead,
        )

        # Get lookahead point
        lookahead_pt, cross_track = self._get_lookahead_point(
            x, y, lookahead
        )

        # Heading to lookahead point
        dx = lookahead_pt[0] - x
        dy = lookahead_pt[1] - y
        heading_cmd = np.arctan2(dy, dx)

        # Speed command
        speed_cmd = current_wp.speed or self.config.default_speed

        # Altitude command (simple: use waypoint altitude)
        altitude_cmd = current_wp.z

        return heading_cmd, speed_cmd, altitude_cmd, cross_track

    def _get_lookahead_point(
        self,
        x: float,
        y: float,
        lookahead: float,
    ) -> tuple[np.ndarray, float]:
        """
        Get lookahead point on path.

        Returns point on path at lookahead distance and cross-track error.
        """
        if self._current_idx == 0:
            # First segment - use current position as start
            p0 = np.array([x, y])
        else:
            prev_wp = self._waypoints[self._current_idx - 1]
            p0 = np.array([prev_wp.x, prev_wp.y])

        current_wp = self._waypoints[self._current_idx]
        p1 = np.array([current_wp.x, current_wp.y])

        # Current position
        pos = np.array([x, y])

        # Project onto line segment
        v = p1 - p0
        segment_length = np.linalg.norm(v)

        if segment_length < 1e-6:
            return p1, np.linalg.norm(pos - p1)

        v_unit = v / segment_length

        # Vector from p0 to current position
        w = pos - p0

        # Projection
        proj = np.dot(w, v_unit)
        proj = np.clip(proj, 0, segment_length)

        # Closest point on segment
        closest = p0 + proj * v_unit

        # Cross-track error
        cross_track = np.linalg.norm(pos - closest)

        # Lookahead point
        lookahead_proj = proj + lookahead

        if lookahead_proj >= segment_length:
            # Extend to next segment if available
            if self._current_idx + 1 < len(self._waypoints):
                next_wp = self._waypoints[self._current_idx + 1]
                remaining = lookahead_proj - segment_length
                v_next = np.array([next_wp.x, next_wp.y]) - p1
                v_next_len = np.linalg.norm(v_next)
                if v_next_len > 1e-6:
                    lookahead_pt = p1 + (remaining / v_next_len) * v_next
                else:
                    lookahead_pt = p1
            else:
                lookahead_pt = p1
        else:
            lookahead_pt = p0 + lookahead_proj * v_unit

        return lookahead_pt, cross_track

    def _advance_waypoint(self) -> None:
        """Advance to next waypoint."""
        self._current_idx += 1
        if self._current_idx >= len(self._waypoints):
            self._current_idx = len(self._waypoints) - 1
            self._completed = True

    @property
    def is_completed(self) -> bool:
        """Check if path is complete."""
        return self._completed

    @property
    def current_waypoint_index(self) -> int:
        """Get current waypoint index."""
        return self._current_idx

    def get_distance_to_goal(self, x: float, y: float) -> float:
        """Get remaining distance to final waypoint."""
        if len(self._waypoints) == 0:
            return 0.0

        final = self._waypoints[-1]
        return np.sqrt((final.x - x)**2 + (final.y - y)**2)
