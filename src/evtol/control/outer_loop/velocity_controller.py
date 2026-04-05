"""
Velocity Controller - Outer Loop.

Controls horizontal velocity (Vx, Vy) in NED frame.
Outputs attitude commands (roll, pitch) to achieve desired velocity.

In hover mode: pitch forward to accelerate north, roll right to accelerate east
In cruise mode: attitude affects flight path angle and turn rate
"""

import numpy as np
from dataclasses import dataclass

from ..controller_base import (
    PIDController,
    ControllerGains,
)


@dataclass
class VelocityGains:
    """Gains for velocity control."""
    vx: ControllerGains = None  # North velocity
    vy: ControllerGains = None  # East velocity

    # Attitude limits for velocity control
    max_pitch_rad: float = np.radians(25)  # Max pitch for velocity control
    max_roll_rad: float = np.radians(35)   # Max bank angle

    def __post_init__(self):
        if self.vx is None:
            self.vx = ControllerGains(
                Kp=0.15,   # rad per m/s error
                Ki=0.02,
                Kd=0.05,
                output_min=-np.radians(25), output_max=np.radians(25),
                integrator_min=-np.radians(5), integrator_max=np.radians(5),
            )
        if self.vy is None:
            self.vy = ControllerGains(
                Kp=0.15,
                Ki=0.02,
                Kd=0.05,
                output_min=-np.radians(35), output_max=np.radians(35),
                integrator_min=-np.radians(5), integrator_max=np.radians(5),
            )


class VelocityController:
    """
    Horizontal velocity controller.

    Input: Velocity commands (Vx, Vy) in NED frame
    Output: Attitude commands (pitch, roll) for inner loop

    Physics:
    - In hover, tilting the vehicle creates horizontal acceleration
    - pitch_cmd ≈ atan(ax/g) for forward acceleration
    - roll_cmd ≈ atan(ay/g) for lateral acceleration
    """

    def __init__(self, gains: VelocityGains | None = None):
        self.gains = gains or VelocityGains()

        self._vx_ctrl = PIDController(self.gains.vx, "vel_north")
        self._vy_ctrl = PIDController(self.gains.vy, "vel_east")

    def reset(self) -> None:
        """Reset controllers."""
        self._vx_ctrl.reset()
        self._vy_ctrl.reset()

    def compute(
        self,
        Vx_cmd: float,
        Vy_cmd: float,
        Vx: float,
        Vy: float,
        yaw: float,
        dt: float,
    ) -> tuple[float, float]:
        """
        Compute attitude commands from velocity error.

        Args:
            Vx_cmd, Vy_cmd: Commanded velocities (m/s) in NED frame
            Vx, Vy: Current velocities in NED frame
            yaw: Current heading (rad) - needed for body frame conversion
            dt: Time step (s)

        Returns:
            (pitch_cmd, roll_cmd): Attitude commands (rad)
        """
        # In NED frame
        accel_north = self._vx_ctrl.compute(Vx_cmd, Vx, dt)  # Accel for north velocity
        accel_east = self._vy_ctrl.compute(Vy_cmd, Vy, dt)   # Accel for east velocity

        # FIXED: Proper NED-to-body frame transformation
        # For aggressive maneuvers (large yaw rates), simple 2D rotation breaks down.
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)

        # Rotate acceleration commands from NED to body frame
        accel_body_x = accel_north * cos_yaw + accel_east * sin_yaw
        accel_body_y = -accel_north * sin_yaw + accel_east * cos_yaw

        # Convert accelerations to required pitch/roll angles
        # Using small angle approximation: tan(θ) ≈ a/g
        g = 9.81
        pitch_cmd = -np.arctan2(accel_body_x, g)  # Negative: forward accel needs nose down
        roll_cmd = np.arctan2(accel_body_y, g)     # Positive: right accel needs right roll

        # Apply limits
        pitch_cmd = np.clip(pitch_cmd, -self.gains.max_pitch_rad, self.gains.max_pitch_rad)
        roll_cmd = np.clip(roll_cmd, -self.gains.max_roll_rad, self.gains.max_roll_rad)

        return pitch_cmd, roll_cmd

    def compute_cruise(
        self,
        V_cmd: float,
        V_current: float,
        dt: float,
    ) -> float:
        """
        Cruise mode: airspeed control via pitch.

        Args:
            V_cmd: Commanded airspeed (m/s)
            V_current: Current airspeed (m/s)
            dt: Time step

        Returns:
            pitch_cmd: Pitch angle command (rad)
        """
        # Use Vx controller for airspeed
        pitch_cmd = self._vx_ctrl.compute(V_cmd, V_current, dt)

        # In cruise, nose down = speed up
        # Invert sign: positive error (want faster) -> nose down (negative pitch)
        pitch_cmd = -pitch_cmd

        return np.clip(pitch_cmd, -self.gains.max_pitch_rad, self.gains.max_pitch_rad)
