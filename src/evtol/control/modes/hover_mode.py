"""
Hover Mode Controller.

Controls vehicle in hover/VTOL mode with nacelles in vertical position.
Position and velocity control via body attitude.
"""

import numpy as np
from dataclasses import dataclass

from ..outer_loop import (
    PositionController,
    VelocityController,
    AltitudeController,
    HeadingController,
)
from ..inner_loop import AttitudeController, RateController
from ..controller_base import AttitudeCommand, MomentCommand


@dataclass
class HoverModeConfig:
    """Hover mode configuration."""
    # Position control gains (aggressive for hover)
    position_kp: float = 1.0

    # Velocity limits
    max_horizontal_speed: float = 15.0   # m/s
    max_vertical_speed: float = 8.0      # m/s

    # Attitude limits (more limited in hover)
    max_roll: float = np.radians(30)
    max_pitch: float = np.radians(30)
    max_yaw_rate: float = np.radians(45)  # rad/s


class HoverMode:
    """
    Hover mode controller.

    Full position control with altitude hold.
    Uses attitude to generate horizontal forces.

    Control chain:
    Position → Velocity → Attitude → Rate → Moments
    """

    def __init__(self, config: HoverModeConfig | None = None):
        self.config = config or HoverModeConfig()

        # Controllers
        self.position_ctrl = PositionController()
        self.velocity_ctrl = VelocityController()
        self.altitude_ctrl = AltitudeController()
        self.heading_ctrl = HeadingController()
        self.attitude_ctrl = AttitudeController()
        self.rate_ctrl = RateController()

        # Set hover gains on attitude controller
        self.attitude_ctrl.set_mode("hover")

    def reset(self) -> None:
        """Reset all controllers."""
        self.position_ctrl.reset()
        self.velocity_ctrl.reset()
        self.altitude_ctrl.reset()
        self.heading_ctrl.reset()
        self.attitude_ctrl.reset()
        self.rate_ctrl.reset()

    def compute(
        self,
        # Commands
        x_cmd: float,
        y_cmd: float,
        alt_cmd: float,
        heading_cmd: float,
        # Current state
        x: float,
        y: float,
        alt: float,
        Vx: float,
        Vy: float,
        Vz: float,
        roll: float,
        pitch: float,
        yaw: float,
        p: float,
        q: float,
        r: float,
        dt: float,
    ) -> tuple[float, MomentCommand]:
        """
        Compute hover mode control.

        Returns:
            (thrust_cmd, moment_cmd): Thrust (N) and moments (Nm)
        """
        # ===== Outer Loop =====

        # Position → Velocity
        from ..controller_base import PositionCommand
        pos_cmd = PositionCommand(x=x_cmd, y=y_cmd, z=-alt_cmd)
        Vx_cmd, Vy_cmd = self.position_ctrl.compute(pos_cmd, x, y, dt)

        # Limit velocity
        V_horiz = np.sqrt(Vx_cmd**2 + Vy_cmd**2)
        if V_horiz > self.config.max_horizontal_speed:
            scale = self.config.max_horizontal_speed / V_horiz
            Vx_cmd *= scale
            Vy_cmd *= scale

        # Velocity → Attitude
        pitch_cmd, roll_cmd = self.velocity_ctrl.compute(
            Vx_cmd, Vy_cmd, Vx, Vy, yaw, dt
        )

        # Heading → Yaw rate
        yaw_rate_cmd = self.heading_ctrl.compute_yaw_rate(heading_cmd, yaw, dt)

        # Altitude → Thrust
        thrust_cmd = self.altitude_ctrl.compute(alt_cmd, alt, vz=Vz, dt=dt)

        # Apply attitude limits
        roll_cmd = np.clip(roll_cmd, -self.config.max_roll, self.config.max_roll)
        pitch_cmd = np.clip(pitch_cmd, -self.config.max_pitch, self.config.max_pitch)

        # ===== Inner Loop =====

        # Attitude → Rate
        att_cmd = AttitudeCommand(roll_rad=roll_cmd, pitch_rad=pitch_cmd, yaw_rad=heading_cmd)
        p_cmd, q_cmd, r_cmd = self.attitude_ctrl.compute(att_cmd, roll, pitch, yaw, dt)

        # Override yaw rate with heading controller output
        r_cmd = yaw_rate_cmd

        # Rate → Moments
        moment_cmd = self.rate_ctrl.compute(p_cmd, q_cmd, r_cmd, p, q, r, dt)

        return thrust_cmd, moment_cmd

    def compute_station_keep(
        self,
        x: float,
        y: float,
        alt: float,
        Vx: float,
        Vy: float,
        Vz: float,
        roll: float,
        pitch: float,
        yaw: float,
        p: float,
        q: float,
        r: float,
        dt: float,
    ) -> tuple[float, MomentCommand]:
        """
        Station keeping (hold current position).

        Sets commands to current position.
        """
        return self.compute(
            x, y, alt, yaw,  # Hold current position and heading
            x, y, alt, Vx, Vy, Vz,
            roll, pitch, yaw, p, q, r,
            dt,
        )
