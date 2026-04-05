"""
Attitude Controller - Inner Loop.

Controls roll, pitch, yaw angles to track commanded attitudes.
Outputs angular rate commands or moment commands.
"""

import numpy as np
from dataclasses import dataclass

from ..controller_base import (
    PIDController,
    ControllerGains,
    AttitudeCommand,
)


@dataclass
class AttitudeGains:
    """Gains for all three attitude axes."""
    roll: ControllerGains = None
    pitch: ControllerGains = None
    yaw: ControllerGains = None

    def __post_init__(self):
        if self.roll is None:
            self.roll = ControllerGains(
                Kp=8.0, Ki=0.5, Kd=2.0,
                output_min=-5.0, output_max=5.0,  # rad/s rate limit
                integrator_min=-1.0, integrator_max=1.0,
            )
        if self.pitch is None:
            self.pitch = ControllerGains(
                Kp=8.0, Ki=0.5, Kd=2.0,
                output_min=-5.0, output_max=5.0,
                integrator_min=-1.0, integrator_max=1.0,
            )
        if self.yaw is None:
            self.yaw = ControllerGains(
                Kp=4.0, Ki=0.2, Kd=1.0,
                output_min=-3.0, output_max=3.0,
                integrator_min=-0.5, integrator_max=0.5,
            )


class AttitudeController:
    """
    Attitude controller for roll, pitch, yaw.

    Input: Attitude commands (φ_cmd, θ_cmd, ψ_cmd)
    Output: Angular rate commands (p_cmd, q_cmd, r_cmd) for rate controller

    Features:
    - Gain scheduling based on flight mode
    - Euler angle wrapping for yaw
    - Attitude limits enforcement
    """

    # Attitude limits (rad)
    MAX_ROLL = np.radians(45)
    MAX_PITCH = np.radians(30)

    def __init__(self, gains: AttitudeGains | None = None):
        self.gains = gains or AttitudeGains()

        # Individual axis controllers
        self._roll_ctrl = PIDController(self.gains.roll, "roll")
        self._pitch_ctrl = PIDController(self.gains.pitch, "pitch")
        self._yaw_ctrl = PIDController(self.gains.yaw, "yaw")

        # Gain schedule storage
        self._gain_schedules: dict[str, AttitudeGains] = {
            'hover': AttitudeGains(
                roll=ControllerGains(Kp=8.0, Ki=0.5, Kd=2.0, output_min=-5.0, output_max=5.0),
                pitch=ControllerGains(Kp=8.0, Ki=0.5, Kd=2.0, output_min=-5.0, output_max=5.0),
                yaw=ControllerGains(Kp=4.0, Ki=0.2, Kd=1.0, output_min=-3.0, output_max=3.0),
            ),
            'transition': AttitudeGains(
                roll=ControllerGains(Kp=6.0, Ki=0.3, Kd=1.5, output_min=-4.0, output_max=4.0),
                pitch=ControllerGains(Kp=6.0, Ki=0.3, Kd=1.5, output_min=-4.0, output_max=4.0),
                yaw=ControllerGains(Kp=3.0, Ki=0.1, Kd=0.8, output_min=-2.0, output_max=2.0),
            ),
            'cruise': AttitudeGains(
                roll=ControllerGains(Kp=4.0, Ki=0.2, Kd=1.0, output_min=-3.0, output_max=3.0),
                pitch=ControllerGains(Kp=4.0, Ki=0.2, Kd=1.0, output_min=-3.0, output_max=3.0),
                yaw=ControllerGains(Kp=2.0, Ki=0.1, Kd=0.5, output_min=-1.5, output_max=1.5),
            ),
        }

    def reset(self) -> None:
        """Reset all controllers."""
        self._roll_ctrl.reset()
        self._pitch_ctrl.reset()
        self._yaw_ctrl.reset()

    def set_mode(self, mode: str) -> None:
        """Set gains based on flight mode."""
        if mode in self._gain_schedules:
            gains = self._gain_schedules[mode]
            self._roll_ctrl.set_gains(gains.roll)
            self._pitch_ctrl.set_gains(gains.pitch)
            self._yaw_ctrl.set_gains(gains.yaw)

    def compute(
        self,
        cmd: AttitudeCommand,
        roll: float,
        pitch: float,
        yaw: float,
        dt: float,
    ) -> tuple:
        """
        Compute angular rate commands from attitude error.

        Args:
            cmd: Commanded attitude
            roll: Current roll angle (rad)
            pitch: Current pitch angle (rad)
            yaw: Current yaw angle (rad)
            dt: Time step (s)

        Returns:
            (p_cmd, q_cmd, r_cmd): Angular rate commands (rad/s)
        """
        # Limit commanded attitude
        roll_cmd = np.clip(cmd.roll_rad, -self.MAX_ROLL, self.MAX_ROLL)
        pitch_cmd = np.clip(cmd.pitch_rad, -self.MAX_PITCH, self.MAX_PITCH)
        yaw_cmd = cmd.yaw_rad

        # Roll controller
        p_cmd = self._roll_ctrl.compute(roll_cmd, roll, dt)

        # Pitch controller
        q_cmd = self._pitch_ctrl.compute(pitch_cmd, pitch, dt)

        # Yaw controller with angle wrapping
        yaw_error = self._wrap_angle(yaw_cmd - yaw)
        r_cmd = self._yaw_ctrl.compute(yaw + yaw_error, yaw, dt)

        return p_cmd, q_cmd, r_cmd

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        """Wrap angle to [-π, π]."""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle

    def get_debug_info(self) -> dict:
        """Get debug information."""
        return {
            'roll': self._roll_ctrl.get_state(),
            'pitch': self._pitch_ctrl.get_state(),
            'yaw': self._yaw_ctrl.get_state(),
        }
