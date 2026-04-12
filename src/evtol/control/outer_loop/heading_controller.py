"""
Heading Controller - Outer Loop.

Controls yaw angle. Outputs yaw rate command (for hover) or bank angle (for cruise).
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass

from ..controller_base import PIDController, ControllerGains


@dataclass
class HeadingGains:
    """Gains for heading control."""
    heading: ControllerGains = None

    max_yaw_rate:   float = np.radians(45)   # rad/s
    max_bank_angle: float = np.radians(30)   # rad (for coordinated turns)
    g: float = 9.81

    def __post_init__(self):
        if self.heading is None:
            self.heading = ControllerGains(
                Kp=2.0, Ki=0.05, Kd=0.1,
                output_min=-self.max_yaw_rate,
                output_max=self.max_yaw_rate,
                integrator_min=-0.3, integrator_max=0.3,
            )


class HeadingController:
    """
    Heading controller.

    Hover/transition: yaw rate command via differential rotor torque.
    Cruise: coordinated turn - bank angle from desired heading error.
    """

    def __init__(self, gains: HeadingGains | None = None):
        self.gains = gains or HeadingGains()
        self._hdg_ctrl = PIDController(self.gains.heading, "heading")

    def reset(self) -> None:
        self._hdg_ctrl.reset()

    def compute_yaw_rate(
        self,
        heading_cmd: float,
        yaw:         float,
        dt:          float,
    ) -> float:
        """
        Compute yaw rate command from heading error.

        Args:
            heading_cmd: Commanded heading (rad)
            yaw:         Current yaw (rad)
            dt:          Time step (s)

        Returns:
            yaw_rate_cmd: Yaw rate command (rad/s)
        """
        # Wrap heading error to [-pi, pi]
        err = heading_cmd - yaw
        err = (err + np.pi) % (2 * np.pi) - np.pi
        yaw_rate_cmd = self._hdg_ctrl.compute(yaw + err, yaw, dt)
        return float(np.clip(yaw_rate_cmd, -self.gains.max_yaw_rate, self.gains.max_yaw_rate))

    def compute_bank_for_turn(
        self,
        heading_cmd: float,
        yaw:         float,
        airspeed:    float,
        dt:          float,
    ) -> float:
        """
        Compute bank angle for coordinated turn in cruise.

        For a level coordinated turn: tan(phi) = V * omega / g
        => phi = atan(V * omega / g)

        Args:
            heading_cmd: Commanded heading (rad)
            yaw:         Current yaw (rad)
            airspeed:    Current airspeed (m/s)
            dt:          Time step (s)

        Returns:
            roll_cmd: Bank angle command (rad)
        """
        yaw_rate_cmd = self.compute_yaw_rate(heading_cmd, yaw, dt)
        V = max(airspeed, 1.0)
        roll_cmd = np.arctan2(V * yaw_rate_cmd, self.gains.g)
        return float(np.clip(roll_cmd, -self.gains.max_bank_angle, self.gains.max_bank_angle))
