"""
Hover Mode Controller.

Controls vehicle in vertical flight with nacelles vertical (90 deg).
Rotors provide all lift and attitude control moments.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass

from ..outer_loop import VelocityController, AltitudeController, HeadingController
from ..inner_loop import AttitudeController, RateController
from ..controller_base import AttitudeCommand, MomentCommand


@dataclass
class HoverModeConfig:
    """Hover mode configuration."""
    max_horizontal_speed: float = 10.0          # m/s
    max_vertical_speed:   float = 5.0           # m/s
    max_roll:             float = np.radians(30) # rad
    max_pitch:            float = np.radians(30) # rad
    max_yaw_rate:         float = np.radians(45) # rad/s


class HoverMode:
    """
    Hover mode controller.

    Multicopter-style control:
      - Altitude held by collective thrust (outer loop: altitude -> climb rate -> thrust)
      - Horizontal position held by attitude tilt (outer loop: velocity -> roll/pitch cmd)
      - Yaw held by differential rotor torque (heading -> yaw rate -> moment_z)

    All four rotors provide lift; nacelles stay at 90 deg (vertical).
    """

    def __init__(self, config: HoverModeConfig | None = None):
        self.config = config or HoverModeConfig()

        self.velocity_ctrl = VelocityController()
        self.altitude_ctrl = AltitudeController()
        self.heading_ctrl  = HeadingController()
        self.attitude_ctrl = AttitudeController()
        self.rate_ctrl     = RateController()

        self.attitude_ctrl.set_mode("hover")

    def reset(self) -> None:
        """Reset all inner controllers."""
        self.velocity_ctrl.reset()
        self.altitude_ctrl.reset()
        self.heading_ctrl.reset()
        self.attitude_ctrl.reset()
        self.rate_ctrl.reset()

    def compute(
        self,
        # Commands
        Vx_cmd: float,
        Vy_cmd: float,
        alt_cmd: float,
        heading_cmd: float,
        # Current state
        Vx: float,
        Vy: float,
        Vz: float,
        alt: float,
        roll: float,
        pitch: float,
        yaw: float,
        p: float,
        q: float,
        r: float,
        dt: float,
    ) -> tuple[float, MomentCommand]:
        """
        Compute hover mode control outputs.

        Args:
            Vx_cmd, Vy_cmd: Commanded NED horizontal velocities (m/s)
            alt_cmd:        Commanded altitude (m)
            heading_cmd:    Commanded heading (rad)
            Vx, Vy, Vz:     Current NED velocities (m/s)
            alt:            Current altitude (m)
            roll,pitch,yaw: Current Euler angles (rad)
            p, q, r:        Current body angular rates (rad/s)
            dt:             Time step (s)

        Returns:
            (thrust_cmd_N, moment_cmd): Total thrust (N) and 3-axis moments (N*m)
        """
        # --- Velocity -> Attitude (outer loop) ---
        Vx_cmd = np.clip(Vx_cmd, -self.config.max_horizontal_speed, self.config.max_horizontal_speed)
        Vy_cmd = np.clip(Vy_cmd, -self.config.max_horizontal_speed, self.config.max_horizontal_speed)

        pitch_cmd, roll_cmd = self.velocity_ctrl.compute(Vx_cmd, Vy_cmd, Vx, Vy, yaw, dt)
        pitch_cmd = np.clip(pitch_cmd, -self.config.max_pitch, self.config.max_pitch)
        roll_cmd  = np.clip(roll_cmd,  -self.config.max_roll,  self.config.max_roll)

        # --- Altitude -> Thrust (outer loop) ---
        thrust_cmd = self.altitude_ctrl.compute(alt_cmd, alt, Vz, dt)

        # --- Heading -> Yaw moment (outer loop) ---
        yaw_rate_cmd = self.heading_ctrl.compute_yaw_rate(heading_cmd, yaw, dt)

        # --- Attitude -> Rate (inner loop) ---
        att_cmd = AttitudeCommand(roll_rad=roll_cmd, pitch_rad=pitch_cmd, yaw_rad=heading_cmd)
        p_cmd, q_cmd, r_cmd = self.attitude_ctrl.compute(att_cmd, roll, pitch, yaw, dt)

        # Override r_cmd with heading controller yaw rate
        r_cmd = yaw_rate_cmd

        # --- Rate -> Moments (innermost loop) ---
        moment_cmd = self.rate_ctrl.compute(p_cmd, q_cmd, r_cmd, p, q, r, dt)

        return thrust_cmd, moment_cmd
