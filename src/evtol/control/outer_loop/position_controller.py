"""
Position Controller - Outer Loop.

Controls 2D horizontal position (x, y in NED frame).
Outputs velocity commands to the velocity controller.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass

from ..controller_base import PIDController, ControllerGains


@dataclass
class PositionGains:
    """Gains for horizontal position control."""
    north: ControllerGains = None
    east:  ControllerGains = None

    max_vel_cmd: float = 15.0   # m/s  maximum commanded velocity

    def __post_init__(self):
        gains = ControllerGains(
            Kp=0.8, Ki=0.01, Kd=0.05,
            output_min=-self.max_vel_cmd,
            output_max=self.max_vel_cmd,
            integrator_min=-2.0, integrator_max=2.0,
        )
        if self.north is None:
            self.north = gains
        if self.east is None:
            self.east = ControllerGains(
                Kp=0.8, Ki=0.01, Kd=0.05,
                output_min=-self.max_vel_cmd,
                output_max=self.max_vel_cmd,
                integrator_min=-2.0, integrator_max=2.0,
            )


class PositionController:
    """
    2D horizontal position controller.

    Input:  Position error [north_err, east_err] (m)
    Output: Velocity commands [Vx_cmd, Vy_cmd] (m/s) for VelocityController

    Uses P+I+D on position error in NED frame.
    Speed limited to max_vel_cmd for safety.
    """

    def __init__(self, gains: PositionGains | None = None):
        self.gains  = gains or PositionGains()
        self._n_ctrl = PIDController(self.gains.north, "pos_north")
        self._e_ctrl = PIDController(self.gains.east,  "pos_east")

    def reset(self) -> None:
        self._n_ctrl.reset()
        self._e_ctrl.reset()

    def compute(
        self,
        north_cmd: float,
        east_cmd:  float,
        north:     float,
        east:      float,
        dt:        float,
    ) -> tuple[float, float]:
        """
        Compute velocity commands from position error.

        Args:
            north_cmd, east_cmd: Commanded NED position (m)
            north, east:         Current NED position (m)
            dt:                  Time step (s)

        Returns:
            (Vx_cmd, Vy_cmd): Velocity commands in NED (m/s)
        """
        Vx_cmd = self._n_ctrl.compute(north_cmd, north, dt)
        Vy_cmd = self._e_ctrl.compute(east_cmd,  east,  dt)

        Vx_cmd = float(np.clip(Vx_cmd, -self.gains.max_vel_cmd, self.gains.max_vel_cmd))
        Vy_cmd = float(np.clip(Vy_cmd, -self.gains.max_vel_cmd, self.gains.max_vel_cmd))

        return Vx_cmd, Vy_cmd

    def compute_with_feedforward(
        self,
        north_cmd: float,
        east_cmd:  float,
        north:     float,
        east:      float,
        Vx_ff:     float,
        Vy_ff:     float,
        dt:        float,
    ) -> tuple[float, float]:
        """
        Position control with trajectory feedforward velocity.

        Args:
            Vx_ff, Vy_ff: Feedforward velocity from planned trajectory (m/s)

        Returns:
            (Vx_cmd, Vy_cmd): Total velocity commands (m/s)
        """
        Vx_fb, Vy_fb = self.compute(north_cmd, east_cmd, north, east, dt)
        Vx_cmd = float(np.clip(Vx_fb + Vx_ff, -self.gains.max_vel_cmd, self.gains.max_vel_cmd))
        Vy_cmd = float(np.clip(Vy_fb + Vy_ff, -self.gains.max_vel_cmd, self.gains.max_vel_cmd))
        return Vx_cmd, Vy_cmd
