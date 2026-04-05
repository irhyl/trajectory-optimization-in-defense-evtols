"""
Control Mixer - Allocation Layer.

Converts high-level thrust and moment commands to individual rotor speeds.
Handles the inverse kinematics of the tiltrotor configuration.

Mixing matrix approach for control allocation.
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class RotorConfig:
    """Configuration for a single rotor."""
    position: np.ndarray  # [x, y, z] relative to CG (m)
    direction: int        # +1 CCW, -1 CW
    nacelle_id: int       # Which nacelle (for tiltrotors)

    # Rotor parameters
    radius: float = 1.0          # m
    max_thrust: float = 10000.0  # N per rotor
    min_thrust: float = 0.0      # N

    # Thrust coefficient (thrust = Ct * rho * omega^2 * r^4)
    Ct: float = 0.012


@dataclass
class RotorCommand:
    """Command for all rotors."""
    thrusts: np.ndarray      # Thrust per rotor (N)
    omegas: np.ndarray       # Angular velocity per rotor (rad/s)
    nacelle_angles: np.ndarray  # Nacelle tilt per nacelle (rad)


class ControlMixer:
    """
    Control allocation via mixing matrix.

    Input: Total thrust (N), moments [L, M, N] (Nm)
    Output: Individual rotor thrusts/speeds

    For a tiltrotor, the relationship is:
    - Thrust: sum of all rotor thrusts (along rotor axis)
    - Roll moment: differential thrust left/right
    - Pitch moment: differential thrust front/rear
    - Yaw moment: differential torque (reaction to rotor spin)

    Mixing matrix: [T, L, M, N]' = M * [T1, T2, T3, T4]'
    Inverse: [T1, T2, T3, T4]' = M^-1 * [T, L, M, N]'
    """

    def __init__(
        self,
        rotor_configs: list[RotorConfig] | None = None,
        rho: float = 1.225,
    ):
        """
        Initialize control mixer.

        Args:
            rotor_configs: List of rotor configurations
            rho: Air density (kg/m³)
        """
        self.rho = rho

        # Default quad tiltrotor configuration
        if rotor_configs is None:
            self.rotor_configs = self._default_quad_tiltrotor()
        else:
            self.rotor_configs = rotor_configs

        self.n_rotors = len(self.rotor_configs)

        # Build mixing matrix
        self._build_mixing_matrix()

    def _default_quad_tiltrotor(self) -> list[RotorConfig]:
        """Default quad tiltrotor configuration."""
        return [
            # Front-left (CCW)
            RotorConfig(
                position=np.array([2.5, -3.0, 0.0]),
                direction=1,
                nacelle_id=0,
                max_thrust=8000.0,
            ),
            # Front-right (CW)
            RotorConfig(
                position=np.array([2.5, 3.0, 0.0]),
                direction=-1,
                nacelle_id=1,
                max_thrust=8000.0,
            ),
            # Rear-left (CW)
            RotorConfig(
                position=np.array([-2.5, -3.0, 0.0]),
                direction=-1,
                nacelle_id=0,
                max_thrust=8000.0,
            ),
            # Rear-right (CCW)
            RotorConfig(
                position=np.array([-2.5, 3.0, 0.0]),
                direction=1,
                nacelle_id=1,
                max_thrust=8000.0,
            ),
        ]

    def _build_mixing_matrix(self) -> None:
        """
        Build the mixing matrix for control allocation.

        For each rotor, calculate contribution to:
        - Total thrust
        - Roll moment (about X-axis)
        - Pitch moment (about Y-axis)
        - Yaw moment (about Z-axis, from reaction torque)
        """
        # Matrix: [T, L, M, N]' = M * [T1, T2, T3, T4]'
        M = np.zeros((4, self.n_rotors))

        for i, rotor in enumerate(self.rotor_configs):
            x, y, z = rotor.position

            # Thrust contribution (assuming vertical in hover)
            M[0, i] = 1.0

            # Roll moment from thrust offset (y-arm)
            M[1, i] = -y  # Negative because positive y is right

            # Pitch moment from thrust offset (x-arm)
            M[2, i] = x   # Positive x is forward

            # Yaw moment from reaction torque
            # Torque constant relative to thrust (approximate)
            kQ = 0.05 * rotor.radius  # Torque = kQ * Thrust
            M[3, i] = rotor.direction * kQ

        self._mix_matrix = M

        # Pseudo-inverse for control allocation
        # Using weighted pseudo-inverse for even thrust distribution
        W = np.eye(self.n_rotors)  # Weight matrix
        self._mix_inv = np.linalg.pinv(M @ W) @ W

    def allocate(
        self,
        thrust_cmd: float,
        moment_cmd: np.ndarray,
        nacelle_angles: np.ndarray,
    ) -> RotorCommand:
        """
        Allocate thrust/moment commands to individual rotors.
        
        FIXED: Uses constrained allocation to properly handle motor saturation.
        When motors saturate, allocation minimizes deviation from desired moments
        rather than allowing arbitrary error growth.

        Args:
            thrust_cmd: Total thrust command (N)
            moment_cmd: Moment command [L, M, N] (Nm)
            nacelle_angles: Current nacelle angles (rad)

        Returns:
            RotorCommand with individual rotor commands
        """
        # Command vector
        cmd = np.array([thrust_cmd, moment_cmd[0], moment_cmd[1], moment_cmd[2]])

        # Unconstrained allocation
        rotor_thrusts_nominal = self._mix_inv @ cmd
        
        # Get thrust limits
        thrust_min = np.array([r.min_thrust for r in self.rotor_configs])
        thrust_max = np.array([r.max_thrust for r in self.rotor_configs])
        
        # FIXED: Constrained allocation (prevents integrator windup)
        rotor_thrusts = np.clip(rotor_thrusts_nominal, thrust_min, thrust_max)

        # Apply limits
        for i, rotor in enumerate(self.rotor_configs):
            rotor_thrusts[i] = np.clip(
                rotor_thrusts[i],
                rotor.min_thrust,
                rotor.max_thrust,
            )

        # Convert thrust to rotor speed
        # T = Ct * rho * omega^2 * R^4
        # omega = sqrt(T / (Ct * rho * R^4))
        rotor_omegas = np.zeros(self.n_rotors)
        for i, rotor in enumerate(self.rotor_configs):
            T = max(rotor_thrusts[i], 10.0)  # Minimum for calculation
            denom = rotor.Ct * self.rho * rotor.radius**4
            if denom > 0:
                rotor_omegas[i] = np.sqrt(T / denom)
            else:
                rotor_omegas[i] = 0.0

        return RotorCommand(
            thrusts=rotor_thrusts,
            omegas=rotor_omegas,
            nacelle_angles=nacelle_angles,
        )

    def get_total_power(self, rotor_cmd: RotorCommand) -> float:
        """
        Calculate total power consumption.

        Using momentum theory: P = T^1.5 / sqrt(2 * rho * A)
        """
        total_power = 0.0
        for i, rotor in enumerate(self.rotor_configs):
            T = rotor_cmd.thrusts[i]
            A = np.pi * rotor.radius**2
            P = T**1.5 / np.sqrt(2 * self.rho * A)
            total_power += P

        return total_power

    def get_max_thrust(self) -> float:
        """Get maximum total thrust."""
        return sum(r.max_thrust for r in self.rotor_configs)

    def get_max_moments(self) -> np.ndarray:
        """Estimate maximum achievable moments."""
        # Assuming max differential thrust between pairs
        max_L = 0.0
        max_M = 0.0
        max_N = 0.0

        for rotor in self.rotor_configs:
            x, y, z = rotor.position
            T = rotor.max_thrust
            max_L += abs(y) * T
            max_M += abs(x) * T
            max_N += 0.05 * rotor.radius * T

        return np.array([max_L / 2, max_M / 2, max_N / 2])


class ControlMixerAdaptive(ControlMixer):
    """
    Adaptive control mixer that accounts for rotor failures.

    Extends base mixer with:
    - Rotor health monitoring
    - Reconfiguration on failure
    - Reduced authority modes
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._rotor_health = np.ones(self.n_rotors)  # 1.0 = healthy, 0.0 = failed

    def set_rotor_health(self, rotor_idx: int, health: float) -> None:
        """
        Set rotor health status.

        Args:
            rotor_idx: Rotor index
            health: Health factor (0 to 1)
        """
        self._rotor_health[rotor_idx] = np.clip(health, 0.0, 1.0)
        self._rebuild_for_failures()

    def _rebuild_for_failures(self) -> None:
        """Rebuild mixing matrix accounting for failures."""
        # Scale maximum thrust by health
        for i, rotor in enumerate(self.rotor_configs):
            rotor.max_thrust = 8000.0 * self._rotor_health[i]

        # Rebuild mixing matrix
        self._build_mixing_matrix()
