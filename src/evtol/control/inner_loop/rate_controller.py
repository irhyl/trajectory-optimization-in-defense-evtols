"""
Rate Controller - Inner Loop (innermost).

Controls angular rates (p, q, r) to track commanded rates.
Outputs moment commands for control allocation.
"""

from dataclasses import dataclass

from ..controller_base import (
    PIDController,
    ControllerGains,
    MomentCommand,
)


@dataclass
class RateGains:
    """Gains for angular rate control."""
    roll_rate: ControllerGains = None   # p
    pitch_rate: ControllerGains = None  # q
    yaw_rate: ControllerGains = None    # r

    def __post_init__(self):
        # Default gains for 2500 kg tiltrotor
        # Moments of inertia: Ixx=3000, Iyy=8000, Izz=10000 kg·m²
        if self.roll_rate is None:
            self.roll_rate = ControllerGains(
                Kp=3000.0,  # Nm per rad/s error
                Ki=300.0,
                Kd=100.0,
                output_min=-50000.0, output_max=50000.0,  # Max roll moment
                integrator_min=-5000.0, integrator_max=5000.0,
            )
        if self.pitch_rate is None:
            self.pitch_rate = ControllerGains(
                Kp=8000.0,  # Larger due to higher Iyy
                Ki=800.0,
                Kd=200.0,
                output_min=-80000.0, output_max=80000.0,
                integrator_min=-10000.0, integrator_max=10000.0,
            )
        if self.yaw_rate is None:
            self.yaw_rate = ControllerGains(
                Kp=10000.0,  # Largest due to Izz
                Ki=500.0,
                Kd=300.0,
                output_min=-60000.0, output_max=60000.0,
                integrator_min=-8000.0, integrator_max=8000.0,
            )


class RateController:
    """
    Angular rate controller.

    Input: Rate commands (p_cmd, q_cmd, r_cmd) from attitude controller
    Output: Moment commands (L, M, N) for control allocation

    This is the innermost control loop, running at highest frequency.
    """

    def __init__(self, gains: RateGains | None = None):
        self.gains = gains or RateGains()

        self._p_ctrl = PIDController(self.gains.roll_rate, "roll_rate")
        self._q_ctrl = PIDController(self.gains.pitch_rate, "pitch_rate")
        self._r_ctrl = PIDController(self.gains.yaw_rate, "yaw_rate")

    def reset(self) -> None:
        """Reset all controllers."""
        self._p_ctrl.reset()
        self._q_ctrl.reset()
        self._r_ctrl.reset()

    def compute(
        self,
        p_cmd: float,
        q_cmd: float,
        r_cmd: float,
        p: float,
        q: float,
        r: float,
        dt: float,
    ) -> MomentCommand:
        """
        Compute moment commands from rate error.

        Args:
            p_cmd, q_cmd, r_cmd: Commanded rates (rad/s)
            p, q, r: Current angular rates (rad/s)
            dt: Time step (s)

        Returns:
            MomentCommand with L, M, N moments
        """
        L = self._p_ctrl.compute(p_cmd, p, dt)
        M = self._q_ctrl.compute(q_cmd, q, dt)
        N = self._r_ctrl.compute(r_cmd, r, dt)

        return MomentCommand(L=L, M=M, N=N, T=0.0)

    def set_gains(self, gains: RateGains) -> None:
        """Update gains."""
        self.gains = gains
        self._p_ctrl.set_gains(gains.roll_rate)
        self._q_ctrl.set_gains(gains.pitch_rate)
        self._r_ctrl.set_gains(gains.yaw_rate)
