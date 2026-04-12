"""
Transition Mode Controller.

Controls vehicle as it tilts nacelles from vertical (hover) to horizontal (cruise)
or vice versa.  Both rotors and wing provide lift during this phase.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass

from ..outer_loop import VelocityController, AltitudeController, HeadingController
from ..inner_loop import AttitudeController, RateController
from ..controller_base import AttitudeCommand, MomentCommand


@dataclass
class TransitionModeConfig:
    """Transition mode configuration."""
    # Speed corridor
    entry_speed_min:  float = 10.0           # m/s - minimum to enter
    exit_speed_fwd:   float = 25.0           # m/s - proceed to cruise
    exit_speed_back:  float = 8.0            # m/s - revert to hover

    # Attitude limits (tighter than cruise - blended aerodynamics)
    max_roll:         float = np.radians(30)
    max_pitch_up:     float = np.radians(20)
    max_pitch_down:   float = np.radians(15)

    # Altitude
    max_climb_rate:   float = 6.0            # m/s
    max_descent_rate: float = 4.0            # m/s


class TransitionMode:
    """
    Transition mode controller.

    Blends hover and cruise control strategies based on nacelle tilt progress.
    As nacelles tilt from 90 deg -> 0 deg:
      - Rotor contribution to lift decreases
      - Wing contribution to lift increases
      - Pitch authority shifts from multicopter to aircraft convention

    The blend factor alpha in [0,1] maps 0=pure hover to 1=pure cruise.
    """

    def __init__(self, config: TransitionModeConfig | None = None):
        self.config = config or TransitionModeConfig()

        # Shared sub-controllers (gains scheduled per mode)
        self.velocity_ctrl = VelocityController()
        self.altitude_ctrl = AltitudeController()
        self.heading_ctrl  = HeadingController()
        self.attitude_ctrl = AttitudeController()
        self.rate_ctrl     = RateController()

        self.attitude_ctrl.set_mode("transition")

    def reset(self) -> None:
        self.velocity_ctrl.reset()
        self.altitude_ctrl.reset()
        self.heading_ctrl.reset()
        self.attitude_ctrl.reset()
        self.rate_ctrl.reset()

    def compute(
        self,
        # Commands
        speed_cmd:   float,
        alt_cmd:     float,
        heading_cmd: float,
        # Current state
        airspeed:    float,
        Vx: float, Vy: float, Vz: float,
        alt:         float,
        roll: float, pitch: float, yaw: float,
        p: float,    q: float,    r: float,
        dt:          float,
        blend_alpha: float = 0.5,
    ) -> tuple[float, MomentCommand]:
        """
        Compute transition mode control.

        Args:
            speed_cmd:   Commanded airspeed (m/s)
            alt_cmd:     Commanded altitude (m)
            heading_cmd: Commanded heading (rad)
            airspeed:    Current airspeed (m/s)
            Vx,Vy,Vz:   Current NED velocities (m/s)
            alt:         Current altitude (m)
            roll,pitch,yaw: Current Euler angles (rad)
            p,q,r:       Current body angular rates (rad/s)
            dt:          Time step (s)
            blend_alpha: Nacelle tilt progress [0=hover, 1=cruise]

        Returns:
            (thrust_cmd_N, moment_cmd)
        """
        alpha = float(np.clip(blend_alpha, 0.0, 1.0))

        # --- Hover path: velocity -> attitude ---
        pitch_hover, roll_hover = self.velocity_ctrl.compute(
            speed_cmd, 0.0, Vx, Vy, yaw, dt
        )

        # --- Cruise path: airspeed -> pitch ---
        pitch_cruise = self.velocity_ctrl.compute_cruise(speed_cmd, airspeed, dt)
        roll_cruise  = self.heading_ctrl.compute_bank_for_turn(heading_cmd, yaw, airspeed, dt)

        # --- Blend attitude commands ---
        pitch_cmd = (1.0 - alpha) * pitch_hover + alpha * pitch_cruise
        roll_cmd  = (1.0 - alpha) * roll_hover  + alpha * roll_cruise

        pitch_cmd = np.clip(pitch_cmd, -self.config.max_pitch_down, self.config.max_pitch_up)
        roll_cmd  = np.clip(roll_cmd,  -self.config.max_roll,       self.config.max_roll)

        # --- Altitude -> Thrust (blended: rotor fraction reduces with alpha) ---
        thrust_cmd = self.altitude_ctrl.compute(alt_cmd, alt, Vz, dt)
        # At alpha=0 (hover): full rotor thrust. At alpha=1: wing carries ~70% of lift.
        rotor_fraction = 1.0 - 0.7 * alpha
        thrust_cmd *= rotor_fraction

        # --- Inner loop ---
        att_cmd    = AttitudeCommand(roll_rad=roll_cmd, pitch_rad=pitch_cmd, yaw_rad=heading_cmd)
        p_cmd, q_cmd, r_cmd = self.attitude_ctrl.compute(att_cmd, roll, pitch, yaw, dt)
        moment_cmd = self.rate_ctrl.compute(p_cmd, q_cmd, r_cmd, p, q, r, dt)

        return thrust_cmd, moment_cmd
