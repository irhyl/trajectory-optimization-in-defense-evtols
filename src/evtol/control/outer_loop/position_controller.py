"""
Position Controller - Outer Loop.

Controls horizontal position (x, y) in NED frame.
Outputs velocity commands for velocity controller.

Used primarily in hover mode for station-keeping and
position tracking.
"""

import numpy as np
from dataclasses import dataclass

from ..controller_base import (
    PIDController,
    ControllerGains,
    PositionCommand,
)


@dataclass
class PositionGains:
    """Gains for position control."""
    north: ControllerGains = None  # X axis
    east: ControllerGains = None   # Y axis

    # Velocity limits (m/s)
    max_horizontal_velocity: float = 30.0

    def __post_init__(self):
        if self.north is None:
            self.north = ControllerGains(
                Kp=0.8,
                Ki=0.05,
                Kd=0.3,
                output_min=-30.0, output_max=30.0,  # m/s
                integrator_min=-5.0, integrator_max=5.0,
            )
        if self.east is None:
            self.east = ControllerGains(
                Kp=0.8,
                Ki=0.05,
                Kd=0.3,
                output_min=-30.0, output_max=30.0,
                integrator_min=-5.0, integrator_max=5.0,
            )


class PositionController:
    """
    Horizontal position controller.

    Input: Position commands (x, y) in NED frame
    Output: Velocity commands (Vx, Vy) for velocity controller

    Note: This controller is mainly active in hover mode.
    In cruise mode, we typically command velocity/heading directly.
    """

    def __init__(self, gains: PositionGains | None = None):
        self.gains = gains or PositionGains()

        self._north_ctrl = PIDController(self.gains.north, "pos_north")
        self._east_ctrl = PIDController(self.gains.east, "pos_east")

    def reset(self) -> None:
        """Reset controllers."""
        self._north_ctrl.reset()
        self._east_ctrl.reset()

    def compute(
        self,
        cmd: PositionCommand,
        x: float,
        y: float,
        dt: float,
    ) -> tuple[float, float]:
        """
        Compute velocity commands from position error.

        Args:
            cmd: Commanded position
            x: Current north position (m)
            y: Current east position (m)
            dt: Time step (s)

        Returns:
            (Vx_cmd, Vy_cmd): Velocity commands in NED frame (m/s)
        """
        # Position error
        Vx_cmd = self._north_ctrl.compute(cmd.x, x, dt)
        Vy_cmd = self._east_ctrl.compute(cmd.y, y, dt)

        # Limit total horizontal velocity
        V_horiz = np.sqrt(Vx_cmd**2 + Vy_cmd**2)
        if V_horiz > self.gains.max_horizontal_velocity:
            scale = self.gains.max_horizontal_velocity / V_horiz
            Vx_cmd *= scale
            Vy_cmd *= scale

        return Vx_cmd, Vy_cmd

    def compute_with_feedforward(
        self,
        cmd: PositionCommand,
        x: float,
        y: float,
        Vx_ff: float,
        Vy_ff: float,
        dt: float,
    ) -> tuple[float, float]:
        """
        Compute with feedforward velocity (for trajectory tracking).

        Args:
            cmd: Commanded position
            x, y: Current position
            Vx_ff, Vy_ff: Feedforward velocity from trajectory
            dt: Time step

        Returns:
            (Vx_cmd, Vy_cmd): Total velocity commands
        """
        Vx_fb, Vy_fb = self.compute(cmd, x, y, dt)

        # Add feedforward
        Vx_cmd = Vx_fb + Vx_ff
        Vy_cmd = Vy_fb + Vy_ff

        # Re-limit
        V_horiz = np.sqrt(Vx_cmd**2 + Vy_cmd**2)
        if V_horiz > self.gains.max_horizontal_velocity:
            scale = self.gains.max_horizontal_velocity / V_horiz
            Vx_cmd *= scale
            Vy_cmd *= scale

        return Vx_cmd, Vy_cmd
