"""
Transition Mode Controller.

Controls vehicle during hover-to-cruise and cruise-to-hover transitions.
Blends between hover and cruise control strategies.
"""

from dataclasses import dataclass

import numpy as np

from ..outer_loop import (
    VelocityController,
    AltitudeController,
    HeadingController,
)
from ..inner_loop import AttitudeController, RateController
from ..controller_base import AttitudeCommand, MomentCommand


@dataclass
class TransitionModeConfig:
    """Transition mode configuration."""
    # Speed thresholds
    hover_speed: float = 10.0    # m/s - below this, hover-like control
    cruise_speed: float = 35.0   # m/s - above this, cruise-like control

    # Attitude limits (more relaxed than hover)
    max_roll: float = np.radians(35)
    max_pitch: float = np.radians(20)  # Limited pitch to maintain lift

    # Transition duration (for smooth blending)
    min_transition_time: float = 5.0  # seconds


@dataclass
class TransitionState:
    """Bundle of measured state passed into compute()."""
    airspeed: float
    vx: float
    vy: float
    vz: float
    alt: float
    roll: float
    pitch: float
    yaw: float
    p: float
    q: float
    r: float
    nacelle_angle: float  # rad


class TransitionMode:
    """
    Transition mode controller.

    Blends between hover and cruise control based on airspeed.

    Key challenges:
    - Wing starts providing lift as speed increases
    - Nacelles tilting changes thrust direction
    - Control authority shifts from attitude to surfaces
    """

    def __init__(self, config: TransitionModeConfig | None = None):
        self.config = config or TransitionModeConfig()

        self.velocity_ctrl = VelocityController()
        self.altitude_ctrl = AltitudeController()
        self.heading_ctrl = HeadingController()
        self.attitude_ctrl = AttitudeController()
        self.rate_ctrl = RateController()

        self._transition_progress = 0.0
        self._transition_time = 0.0

    def reset(self) -> None:
        """Reset all sub-controllers."""
        self.velocity_ctrl.reset()
        self.altitude_ctrl.reset()
        self.heading_ctrl.reset()
        self.attitude_ctrl.reset()
        self.rate_ctrl.reset()
        self._transition_progress = 0.0
        self._transition_time = 0.0

    def compute(
        self,
        speed_cmd: float,
        heading_cmd: float,
        alt_cmd: float,
        state: TransitionState,
        dt: float,
    ) -> tuple[float, MomentCommand, float]:
        """
        Compute transition mode control outputs.

        Args:
            speed_cmd:    Commanded airspeed [m/s]
            heading_cmd:  Commanded heading [rad]
            alt_cmd:      Commanded altitude [m, positive up]
            state:        Current vehicle state (TransitionState)
            dt:           Time step [s]

        Returns:
            (thrust_cmd, moment_cmd, pitch_trim)
        """
        self._update_transition(state.airspeed, dt)
        self.attitude_ctrl.set_mode("transition")

        # Altitude → Thrust (account for reduced vertical thrust during tilt)
        thrust_cmd = self.altitude_ctrl.compute(
            alt_cmd, state.alt, vz=state.vz, dt=dt
        )
        thrust_cmd = self.altitude_ctrl.compute_for_nacelle_angle(
            thrust_cmd, state.nacelle_angle
        )

        # Speed control via pitch (nose-down to accelerate)
        pitch_cmd = np.clip(
            -0.05 * (speed_cmd - state.airspeed),
            -self.config.max_pitch,
            self.config.max_pitch,
        )

        pitch_trim = self._compute_pitch_trim(state.nacelle_angle)
        pitch_cmd += pitch_trim

        # Heading: coordinated turn at speed, yaw-only at low speed
        if state.airspeed > 15:
            roll_cmd = self.heading_ctrl.compute_bank_for_turn(
                heading_cmd, state.yaw, state.airspeed, dt
            )
        else:
            roll_cmd = 0.0
        roll_cmd = np.clip(roll_cmd, -self.config.max_roll, self.config.max_roll)

        att_cmd = AttitudeCommand(
            roll_rad=roll_cmd, pitch_rad=pitch_cmd, yaw_rad=heading_cmd
        )
        p_cmd, q_cmd, r_cmd = self.attitude_ctrl.compute(
            att_cmd, state.roll, state.pitch, state.yaw, dt
        )
        moment_cmd = self.rate_ctrl.compute(
            p_cmd, q_cmd, r_cmd, state.p, state.q, state.r, dt
        )

        return thrust_cmd, moment_cmd, pitch_trim

    def _update_transition(self, airspeed: float, dt: float) -> None:
        """Update transition progress [0=hover, 1=cruise]."""
        self._transition_time += dt
        speed_range = self.config.cruise_speed - self.config.hover_speed
        if speed_range > 0:
            self._transition_progress = np.clip(
                (airspeed - self.config.hover_speed) / speed_range, 0.0, 1.0
            )

    def _compute_pitch_trim(self, nacelle_angle: float) -> float:
        """
        Compute pitch trim needed during nacelle tilt.

        As nacelles rotate from vertical (hover) to horizontal (cruise),
        the required pitch-up trim increases to maintain wing angle of attack.
        Returns trim angle in radians, clamped to [0, 8°].
        """
        nacelle_from_vertical = np.radians(90) - nacelle_angle
        return float(np.clip(0.05 * nacelle_from_vertical, 0.0, np.radians(8)))

    @property
    def transition_progress(self) -> float:
        """Transition progress: 0 = hover, 1 = cruise."""
        return self._transition_progress
