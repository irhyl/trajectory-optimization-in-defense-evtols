"""
Cruise Mode Controller.

Controls vehicle in forward flight with nacelles in horizontal position.
Wing provides lift, propellers provide thrust.
"""

import numpy as np
from dataclasses import dataclass

from ..outer_loop import (
    VelocityController,
    AltitudeController,
    HeadingController,
)
from ..inner_loop import AttitudeController, RateController
from ..controller_base import AttitudeCommand, MomentCommand


@dataclass
class CruiseModeConfig:
    """Cruise mode configuration."""
    # Speed limits
    min_speed: float = 25.0     # m/s (stall margin)
    max_speed: float = 70.0     # m/s (structural/power limit)
    cruise_speed: float = 50.0  # m/s (optimal)

    # Attitude limits (aircraft-like)
    max_roll: float = np.radians(45)    # Steeper bank allowed
    max_pitch: float = np.radians(15)   # Limited by stall
    min_pitch: float = np.radians(-20)  # Dive limit

    # Turn parameters
    standard_rate_turn: float = np.radians(3)  # rad/s (3°/s)
    max_turn_rate: float = np.radians(15)      # rad/s

    # Altitude parameters
    max_climb_rate: float = 10.0    # m/s
    max_descent_rate: float = 15.0  # m/s


class CruiseMode:
    """
    Cruise mode controller.

    Aircraft-like control in forward flight:
    - Pitch controls airspeed (dive = faster, climb = slower)
    - Bank controls turn rate
    - Thrust controls altitude/climb rate
    """

    def __init__(self, config: CruiseModeConfig | None = None):
        self.config = config or CruiseModeConfig()

        # Controllers
        self.velocity_ctrl = VelocityController()
        self.altitude_ctrl = AltitudeController()
        self.heading_ctrl = HeadingController()
        self.attitude_ctrl = AttitudeController()
        self.rate_ctrl = RateController()

        # Set cruise gains
        self.attitude_ctrl.set_mode("cruise")

    def reset(self) -> None:
        """Reset controllers."""
        self.velocity_ctrl.reset()
        self.altitude_ctrl.reset()
        self.heading_ctrl.reset()
        self.attitude_ctrl.reset()
        self.rate_ctrl.reset()

    def compute(
        self,
        # Commands
        speed_cmd: float,
        heading_cmd: float,
        alt_cmd: float,
        # Current state
        airspeed: float,
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
        Compute cruise mode control.

        Returns:
            (thrust_cmd, moment_cmd): Thrust and moments
        """
        # ===== Speed Control via Pitch =====
        # In cruise, pitch attitude affects airspeed
        # (unlike hover where it affects horizontal velocity)

        speed_cmd = np.clip(speed_cmd, self.config.min_speed, self.config.max_speed)
        pitch_cmd = self.velocity_ctrl.compute_cruise(speed_cmd, airspeed, dt)
        pitch_cmd = np.clip(pitch_cmd, self.config.min_pitch, self.config.max_pitch)

        # ===== Altitude Control via Thrust =====
        # In cruise, throttle adjusts climb rate
        # Need less thrust than hover (wing provides lift)

        thrust_cmd = self.altitude_ctrl.compute(alt_cmd, alt, Vz, dt)

        # Scale thrust for cruise (wing lift reduces required thrust)
        # Simplified: assume wing provides ~80% of lift in cruise
        thrust_cmd *= 0.3  # Only need thrust for propulsion + some lift margin

        # ===== Heading Control via Bank =====
        # Coordinated turn: bank angle determines turn rate

        roll_cmd = self.heading_ctrl.compute_bank_for_turn(
            heading_cmd, yaw, airspeed, dt
        )
        roll_cmd = np.clip(roll_cmd, -self.config.max_roll, self.config.max_roll)

        # ===== Inner Loop =====

        att_cmd = AttitudeCommand(roll_rad=roll_cmd, pitch_rad=pitch_cmd, yaw_rad=heading_cmd)
        p_cmd, q_cmd, r_cmd = self.attitude_ctrl.compute(att_cmd, roll, pitch, yaw, dt)

        moment_cmd = self.rate_ctrl.compute(p_cmd, q_cmd, r_cmd, p, q, r, dt)

        return thrust_cmd, moment_cmd

    def compute_tracking(
        self,
        # Path following commands
        heading_cmd: float,
        alt_cmd: float,
        speed_cmd: float,
        climb_rate_cmd: float,
        # Current state
        airspeed: float,
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
        Cruise mode with explicit climb rate command (for trajectory tracking).
        """
        # Speed control
        speed_cmd = np.clip(speed_cmd, self.config.min_speed, self.config.max_speed)
        pitch_cmd = self.velocity_ctrl.compute_cruise(speed_cmd, airspeed, dt)

        # Modify pitch for climb rate
        climb_rate_cmd = np.clip(
            climb_rate_cmd,
            -self.config.max_descent_rate,
            self.config.max_climb_rate
        )

        # Pitch for climb: gamma = asin(Vz / V)
        if airspeed > 10:
            climb_pitch = np.arcsin(np.clip(climb_rate_cmd / airspeed, -0.5, 0.5))
            pitch_cmd += climb_pitch

        pitch_cmd = np.clip(pitch_cmd, self.config.min_pitch, self.config.max_pitch)

        # Altitude via thrust (with climb rate override)
        thrust_cmd = self.altitude_ctrl.compute_direct(
            climb_rate_cmd=climb_rate_cmd,
            vz=Vz,
            dt=dt,
        )
        thrust_cmd *= 0.3  # Cruise scaling

        # Heading via bank
        roll_cmd = self.heading_ctrl.compute_bank_for_turn(
            heading_cmd, yaw, airspeed, dt
        )
        roll_cmd = np.clip(roll_cmd, -self.config.max_roll, self.config.max_roll)

        # Inner loop
        att_cmd = AttitudeCommand(roll_rad=roll_cmd, pitch_rad=pitch_cmd, yaw_rad=heading_cmd)
        p_cmd, q_cmd, r_cmd = self.attitude_ctrl.compute(att_cmd, roll, pitch, yaw, dt)
        moment_cmd = self.rate_ctrl.compute(p_cmd, q_cmd, r_cmd, p, q, r, dt)

        return thrust_cmd, moment_cmd

    def check_stall_protection(self, airspeed: float, pitch: float) -> bool:
        """
        Check if stall protection should activate.

        Returns True if speed is too low and pitch is too high.
        """
        if airspeed < self.config.min_speed and pitch > np.radians(5):
            return True
        return False

    def get_optimal_speed(self) -> float:
        """Get optimal cruise speed."""
        return self.config.cruise_speed
