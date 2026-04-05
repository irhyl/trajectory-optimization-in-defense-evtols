"""
Heading Controller - Outer Loop.

Controls yaw/heading angle.
Outputs yaw command for attitude controller or bank angle for coordinated turns.
"""

import numpy as np
from dataclasses import dataclass

from ..controller_base import PIDController, ControllerGains


@dataclass
class HeadingGains:
    """Gains for heading control."""
    heading: ControllerGains = None
    turn_rate: ControllerGains = None

    # Limits
    max_turn_rate: float = np.radians(30)  # rad/s
    max_bank_for_turn: float = np.radians(35)  # rad

    def __post_init__(self):
        if self.heading is None:
            self.heading = ControllerGains(
                Kp=2.0,    # rad/s per rad error
                Ki=0.1,
                Kd=0.5,
                output_min=-np.radians(30), output_max=np.radians(30),
                integrator_min=-np.radians(5), integrator_max=np.radians(5),
            )
        if self.turn_rate is None:
            self.turn_rate = ControllerGains(
                Kp=0.5,    # Bank angle per rad/s error
                Ki=0.05,
                Kd=0.1,
                output_min=-np.radians(35), output_max=np.radians(35),
                integrator_min=-np.radians(5), integrator_max=np.radians(5),
            )


class HeadingController:
    """
    Heading/yaw controller.

    Two modes:
    1. Hover mode: Direct yaw rate command
    2. Cruise mode: Coordinated turn via bank angle

    For coordinated turns in cruise:
    - Bank angle creates centripetal acceleration
    - Turn rate = g * tan(bank) / V
    - Therefore: bank = atan(V * turn_rate / g)
    """

    GRAVITY = 9.81

    def __init__(self, gains: HeadingGains | None = None):
        self.gains = gains or HeadingGains()

        self._heading_ctrl = PIDController(self.gains.heading, "heading")
        self._turn_ctrl = PIDController(self.gains.turn_rate, "turn_rate")

    def reset(self) -> None:
        """Reset controllers."""
        self._heading_ctrl.reset()
        self._turn_ctrl.reset()

    def compute_yaw_rate(
        self,
        heading_cmd: float,
        heading: float,
        dt: float,
    ) -> float:
        """
        Compute yaw rate command (hover mode).

        Args:
            heading_cmd: Commanded heading (rad)
            heading: Current heading (rad)
            dt: Time step (s)

        Returns:
            yaw_rate_cmd: Yaw rate command (rad/s)
        """
        # Wrap heading error to [-π, π]
        heading_error = self._wrap_angle(heading_cmd - heading)

        # PID on wrapped error
        yaw_rate_cmd = self._heading_ctrl.compute(
            heading + heading_error,  # Target = current + wrapped error
            heading,
            dt,
        )

        return np.clip(yaw_rate_cmd, -self.gains.max_turn_rate, self.gains.max_turn_rate)

    def compute_bank_for_turn(
        self,
        heading_cmd: float,
        heading: float,
        airspeed: float,
        dt: float,
    ) -> float:
        """
        Compute bank angle for coordinated turn (cruise mode).

        Args:
            heading_cmd: Commanded heading (rad)
            heading: Current heading (rad)
            airspeed: Current airspeed (m/s)
            dt: Time step (s)

        Returns:
            bank_cmd: Bank angle command (rad)
        """
        # Get desired turn rate from heading error
        yaw_rate_cmd = self.compute_yaw_rate(heading_cmd, heading, dt)

        # Coordinated turn: bank = atan(V * r / g)
        if airspeed > 10:  # Only in forward flight
            bank_cmd = np.arctan(airspeed * yaw_rate_cmd / self.GRAVITY)
        else:
            bank_cmd = 0.0

        return np.clip(bank_cmd, -self.gains.max_bank_for_turn, self.gains.max_bank_for_turn)

    def compute_turn_rate_from_bank(
        self,
        bank: float,
        airspeed: float,
    ) -> float:
        """
        Calculate expected turn rate from current bank angle.

        Useful for turn rate feedback in cruise.
        """
        if airspeed > 10:
            return self.GRAVITY * np.tan(bank) / airspeed
        return 0.0

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        """Wrap angle to [-π, π]."""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle
