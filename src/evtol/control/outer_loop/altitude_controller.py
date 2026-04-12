"""
Altitude Controller - Outer Loop.

Cascaded altitude -> climb-rate -> thrust controller.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass

from ..controller_base import PIDController, ControllerGains


@dataclass
class AltitudeGains:
    """Gains for the cascaded altitude controller."""
    # Outer: altitude error -> climb rate command
    altitude: ControllerGains = None
    # Inner: climb rate error -> vertical acceleration command
    climb_rate: ControllerGains = None

    # Vehicle parameters (for feedforward)
    vehicle_weight_n: float = 2000.0    # N  (used as thrust feedforward)
    max_climb_rate:   float = 8.0       # m/s
    max_descent_rate: float = 10.0      # m/s
    max_thrust_n:     float = 4800.0    # N  (total available)

    def __post_init__(self):
        if self.altitude is None:
            self.altitude = ControllerGains(
                Kp=1.2, Ki=0.05, Kd=0.1,
                output_min=-self.max_descent_rate,
                output_max=self.max_climb_rate,
                integrator_min=-2.0, integrator_max=2.0,
            )
        if self.climb_rate is None:
            self.climb_rate = ControllerGains(
                Kp=300.0, Ki=20.0, Kd=30.0,
                output_min=0.0,
                output_max=self.max_thrust_n,
                integrator_min=-200.0, integrator_max=200.0,
            )


class AltitudeController:
    """
    Cascaded altitude controller.

    Outer loop: altitude error -> climb rate setpoint
    Inner loop: climb rate error -> thrust command (with weight feedforward)

    In hover and transition: thrust supports full vehicle weight.
    In cruise: wing provides lift, so thrust demand is reduced by caller.
    """

    def __init__(self, gains: AltitudeGains | None = None):
        self.gains = gains or AltitudeGains()
        self._alt_ctrl  = PIDController(self.gains.altitude,    "altitude")
        self._rate_ctrl = PIDController(self.gains.climb_rate,  "climb_rate")

    def reset(self) -> None:
        """Reset both loops."""
        self._alt_ctrl.reset()
        self._rate_ctrl.reset()

    def compute(
        self,
        alt_cmd: float,
        alt:     float,
        Vz:      float,
        dt:      float,
    ) -> float:
        """
        Compute thrust command from altitude error.

        Args:
            alt_cmd: Commanded altitude (m, positive up)
            alt:     Current altitude  (m, positive up)
            Vz:      Current vertical velocity (m/s, positive up)
            dt:      Time step (s)

        Returns:
            thrust_cmd: Total thrust command (N)
        """
        # Outer loop: altitude -> climb rate
        climb_rate_cmd = self._alt_ctrl.compute(alt_cmd, alt, dt)
        climb_rate_cmd = np.clip(
            climb_rate_cmd,
            -self.gains.max_descent_rate,
            self.gains.max_climb_rate,
        )

        # Inner loop: climb rate -> thrust (feedforward: vehicle weight)
        thrust_ff   = self.gains.vehicle_weight_n
        thrust_fb   = self._rate_ctrl.compute(climb_rate_cmd, Vz, dt)
        thrust_cmd  = thrust_ff + thrust_fb
        thrust_cmd  = np.clip(thrust_cmd, 0.0, self.gains.max_thrust_n)

        return thrust_cmd

    def compute_direct(
        self,
        climb_rate_cmd: float,
        vz:             float,
        dt:             float,
    ) -> float:
        """
        Compute thrust from a direct climb rate command (skips outer altitude loop).

        Used by cruise mode and trajectory tracking.
        """
        climb_rate_cmd = np.clip(
            climb_rate_cmd,
            -self.gains.max_descent_rate,
            self.gains.max_climb_rate,
        )
        thrust_ff  = self.gains.vehicle_weight_n
        thrust_fb  = self._rate_ctrl.compute(climb_rate_cmd, vz, dt)
        return float(np.clip(thrust_ff + thrust_fb, 0.0, self.gains.max_thrust_n))
