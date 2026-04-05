"""
Nacelle Dynamics - Tiltrotor Nacelle Actuation System

This module models the nacelle tilting mechanism for tiltrotor eVTOL.
The nacelle transitions from helicopter mode (90°) to airplane mode (0°).

Physical System
===============

The nacelle is actuated by electric linear actuators that rotate
the entire rotor/motor assembly about a lateral axis.

Key Dynamics:
    J_n·θ̈ + c·θ̇ + k·θ = τ_actuator + τ_aero + τ_gyro

where:
    J_n = nacelle rotational inertia about tilt axis
    c = damping coefficient
    k = spring stiffness (for centering)
    τ_actuator = actuator torque
    τ_aero = aerodynamic hinge moment
    τ_gyro = gyroscopic moment from rotor

Rate Limiting:
    |θ̇| ≤ θ̇_max (typically 10°/s for safety)

Transition Phases
=================

1. Hover (θ = 90°): Rotor thrust vertical, full lift from rotors
2. Transition (0° < θ < 90°): Mixed rotor/wing lift
3. Cruise (θ = 0°): Rotor thrust horizontal, wing provides lift

The transition corridor is critical for tiltrotor design:
    - Wing stall must be avoided during deceleration
    - Vortex ring state must be avoided
    - Control authority transitions from rotor to aerodynamic
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
import logging

from ..config import NacelleConfig

logger = logging.getLogger(__name__)


@dataclass
class NacelleState:
    """Nacelle operating state."""
    # Position and rate
    angle: float = np.pi / 2       # rad (90° = hover)
    angle_rate: float = 0.0        # rad/s
    angle_accel: float = 0.0       # rad/s²

    # Command tracking
    angle_cmd: float = np.pi / 2   # rad
    tracking_error: float = 0.0    # rad

    # Actuator state
    actuator_torque: float = 0.0   # N·m
    actuator_power: float = 0.0    # W
    actuator_saturation: bool = False

    # External moments
    aero_moment: float = 0.0       # N·m
    gyro_moment: float = 0.0       # N·m

    # Mode
    mode: str = "hover"            # "hover", "transition", "cruise"


class NacelleDynamics:
    """
    Nacelle tilting dynamics for tiltrotor.

    Models the nacelle as a second-order system with rate limiting
    and position limits.

    The nacelle angle convention:
        - θ = 0°: Cruise (rotor axis horizontal, thrust forward)
        - θ = 90°: Hover (rotor axis vertical, thrust up)
        - θ = 95°: Maximum aft tilt (for rearward flight)

    Attributes:
        config: Nacelle configuration
        state: Current nacelle state
    """

    def __init__(self, config: NacelleConfig):
        """
        Initialize nacelle dynamics.

        Args:
            config: Nacelle configuration
        """
        self.config = config

        # Physical parameters derived from NacelleConfig.
        # NacelleConfig stores: tilt_range (tuple deg), tilt_rate_limit (deg/s),
        # tilt_time_constant (s), nacelle_mass (kg), pivot_position (ndarray).
        #
        # We estimate the second-order dynamics parameters from the nacelle mass
        # and the first-order time constant (τ = J / (c + k·τ)):
        #   J ≈ m · r²  with r ≈ 0.6 m (typical nacelle arm from pivot)
        _r_nacelle = 0.6   # m — effective radius from pivot to nacelle CoM
        self.J = config.nacelle_mass * _r_nacelle**2   # kg·m² (about tilt axis)

        # Critical damping from time constant: τ_1st = J / c  →  c = J / τ
        tau = config.tilt_time_constant  # s
        self.c = self.J / max(tau, 0.01)  # N·m·s/rad (damping)
        self.k = 0.0  # No physical centering spring on a tiltrotor actuator

        # Limits
        self.angle_min = np.radians(config.tilt_range[0])   # rad
        self.angle_max = np.radians(config.tilt_range[1])   # rad
        self.rate_limit = np.radians(config.tilt_rate_limit)  # rad/s
        # Max torque sized to drive nacelle at rate limit against full damping
        self.torque_max = self.c * self.rate_limit * 3.0    # N·m (3× safety margin)

        # Aerodynamic moment arm (nacelle depth from pivot)
        self._moment_arm = _r_nacelle * 0.5  # m

        # Controller gains (PD controller)
        self.Kp = 50.0   # Position gain
        self.Kd = 10.0   # Derivative gain

        # State
        self.state = NacelleState()

        logger.info(f"NacelleDynamics initialized: "
                   f"range=[{config.tilt_range[0]:.0f}°, {config.tilt_range[1]:.0f}°], "
                   f"rate_limit={config.tilt_rate_limit}°/s")

    def set_angle_command(self, angle_cmd: float):
        """
        Set nacelle angle command.

        Args:
            angle_cmd: Commanded nacelle angle [rad]
        """
        # Clamp to limits
        self.state.angle_cmd = np.clip(
            angle_cmd,
            self.angle_min,
            self.angle_max
        )

    def compute_derivatives(
        self,
        angle: float,
        angle_rate: float,
        angle_cmd: float,
        rotor_omega: float = 0.0,
        rotor_inertia: float = 0.0,
        V_inf: float = 0.0,
        alpha: float = 0.0,
    ) -> tuple[float, float]:
        """
        Compute nacelle angle derivatives.

        Args:
            angle: Current nacelle angle [rad]
            angle_rate: Current nacelle rate [rad/s]
            angle_cmd: Commanded angle [rad]
            rotor_omega: Rotor angular velocity [rad/s]
            rotor_inertia: Rotor polar moment of inertia [kg·m²]
            V_inf: Freestream velocity [m/s]
            alpha: Angle of attack [rad]

        Returns:
            (d_angle, d_angle_rate)
        """
        # Position error
        error = angle_cmd - angle

        # PD controller for actuator torque
        tau_cmd = self.Kp * error - self.Kd * angle_rate

        # Saturate actuator torque
        tau_actuator = np.clip(tau_cmd, -self.torque_max, self.torque_max)
        self.state.actuator_saturation = abs(tau_cmd) > self.torque_max

        # Gyroscopic moment from rotor
        # When nacelle tilts, rotor angular momentum precesses
        # τ_gyro = I_rotor × ω_rotor × θ̇_nacelle
        tau_gyro = rotor_inertia * rotor_omega * angle_rate
        self.state.gyro_moment = tau_gyro

        # Aerodynamic hinge moment (simplified)
        # Increases with airspeed and angle of attack
        q_dynamic = 0.5 * 1.225 * V_inf**2
        S_nacelle = 0.5  # m² (approximate nacelle area)
        C_m = 0.1 * np.sin(2 * alpha)  # Moment coefficient
        tau_aero = q_dynamic * S_nacelle * self._moment_arm * C_m
        self.state.aero_moment = tau_aero

        # Spring torque (centers to neutral)
        neutral_angle = np.radians(45)  # Transition position
        tau_spring = -self.k * (angle - neutral_angle)

        # Total external moment
        tau_total = tau_actuator + tau_spring - tau_gyro - tau_aero

        # Damping
        tau_damping = -self.c * angle_rate

        # Angular acceleration
        angle_accel = (tau_total + tau_damping) / self.J

        # Rate limiting (first-order rate limiter)
        # Limit the acceleration to enforce rate limit
        if angle_rate > self.rate_limit:
            angle_accel = min(angle_accel, 0)  # Can only slow down
        elif angle_rate < -self.rate_limit:
            angle_accel = max(angle_accel, 0)  # Can only slow down

        # Store state
        self.state.actuator_torque = tau_actuator
        self.state.angle_accel = angle_accel

        return angle_rate, angle_accel

    def step(
        self,
        angle_cmd: float,
        dt: float,
        rotor_omega: float = 0.0,
        rotor_inertia: float = 0.0,
        V_inf: float = 0.0,
        alpha: float = 0.0,
    ) -> NacelleState:
        """
        Integrate nacelle dynamics for one time step.

        Args:
            angle_cmd: Commanded angle [rad]
            dt: Time step [s]
            rotor_omega: Rotor angular velocity [rad/s]
            rotor_inertia: Rotor polar inertia [kg·m²]
            V_inf: Freestream velocity [m/s]
            alpha: Angle of attack [rad]

        Returns:
            Updated NacelleState
        """
        self.set_angle_command(angle_cmd)

        # Get derivatives
        d_angle, d_rate = self.compute_derivatives(
            self.state.angle,
            self.state.angle_rate,
            self.state.angle_cmd,
            rotor_omega,
            rotor_inertia,
            V_inf,
            alpha,
        )

        # Euler integration
        new_rate = self.state.angle_rate + d_rate * dt

        # Rate limiting
        new_rate = np.clip(new_rate, -self.rate_limit, self.rate_limit)

        new_angle = self.state.angle + new_rate * dt

        # Position limiting with zero rate at limits
        if new_angle >= self.angle_max:
            new_angle = self.angle_max
            new_rate = min(new_rate, 0)
        elif new_angle <= self.angle_min:
            new_angle = self.angle_min
            new_rate = max(new_rate, 0)

        # Update state
        self.state.angle = new_angle
        self.state.angle_rate = new_rate
        self.state.tracking_error = self.state.angle_cmd - new_angle

        # Power consumption
        self.state.actuator_power = abs(
            self.state.actuator_torque * self.state.angle_rate
        )

        # Mode determination
        angle_deg = np.degrees(new_angle)
        if angle_deg > 80:
            self.state.mode = "hover"
        elif angle_deg < 10:
            self.state.mode = "cruise"
        else:
            self.state.mode = "transition"

        return self.state

    def get_rotation_matrix(self) -> np.ndarray:
        """
        Get rotation matrix from nacelle frame to body frame.

        The nacelle rotates about the body Y-axis.

        Returns:
            3x3 rotation matrix R_nb
        """
        c = np.cos(self.state.angle)
        s = np.sin(self.state.angle)

        # Rotation about Y-axis
        R = np.array([
            [c, 0, s],
            [0, 1, 0],
            [-s, 0, c]
        ])

        return R

    def get_transition_schedule(
        self,
        V_current: float,
        V_target: float,
    ) -> float:
        """
        Get recommended nacelle angle for airspeed.

        This implements a conversion corridor that ensures:
        - Wing doesn't stall
        - Rotor doesn't enter vortex ring state

        Args:
            V_current: Current airspeed [m/s]
            V_target: Target airspeed [m/s] (determines direction)

        Returns:
            Recommended nacelle angle [rad]
        """
        # Transition corridor parameters
        V_max_hover = 5.0    # m/s (maximum for pure hover)
        V_cruise = 60.0      # m/s (full cruise speed)

        if V_current < V_max_hover:
            # Pure hover
            return np.radians(90)
        elif V_current > V_cruise * 0.9:
            # Full cruise
            return np.radians(0)
        else:
            # Linear interpolation in transition
            progress = (V_current - V_max_hover) / (V_cruise * 0.9 - V_max_hover)
            progress = np.clip(progress, 0, 1)
            return np.radians(90 * (1 - progress))


@dataclass
class TiltSchedule:
    """Nacelle tilt schedule for flight phase."""
    airspeed: float        # m/s
    nacelle_angle: float   # rad
    rotor_rpm: float       # RPM
    wing_flap: float       # rad


class TransitionController:
    """
    Controller for managing nacelle transitions.

    Coordinates:
    - Nacelle angle scheduling
    - Rotor RPM management
    - Wing flap settings
    - Envelope protection
    """

    def __init__(
        self,
        nacelle_left: NacelleDynamics,
        nacelle_right: NacelleDynamics,
    ):
        """
        Initialize transition controller.

        Args:
            nacelle_left: Left nacelle dynamics
            nacelle_right: Right nacelle dynamics
        """
        self.nacelle_left = nacelle_left
        self.nacelle_right = nacelle_right

        # Transition schedule (airspeed → nacelle angle)
        self.schedule = [
            TiltSchedule(0.0, np.radians(90), 500, np.radians(0)),
            TiltSchedule(10.0, np.radians(80), 480, np.radians(10)),
            TiltSchedule(20.0, np.radians(60), 450, np.radians(20)),
            TiltSchedule(30.0, np.radians(45), 420, np.radians(15)),
            TiltSchedule(40.0, np.radians(30), 400, np.radians(10)),
            TiltSchedule(50.0, np.radians(15), 380, np.radians(5)),
            TiltSchedule(60.0, np.radians(5), 360, np.radians(0)),
            TiltSchedule(70.0, np.radians(0), 350, np.radians(0)),
        ]

        # Differential tilt limits (for roll control)
        self.max_differential = np.radians(5)  # Maximum asymmetry

    def get_schedule_point(self, airspeed: float) -> TiltSchedule:
        """
        Interpolate schedule for given airspeed.

        Args:
            airspeed: Current airspeed [m/s]

        Returns:
            Interpolated TiltSchedule
        """
        # Find bracketing points
        for i in range(len(self.schedule) - 1):
            if self.schedule[i].airspeed <= airspeed <= self.schedule[i+1].airspeed:
                # Linear interpolation
                t = ((airspeed - self.schedule[i].airspeed) /
                     (self.schedule[i+1].airspeed - self.schedule[i].airspeed))

                return TiltSchedule(
                    airspeed=airspeed,
                    nacelle_angle=(1-t) * self.schedule[i].nacelle_angle +
                                  t * self.schedule[i+1].nacelle_angle,
                    rotor_rpm=(1-t) * self.schedule[i].rotor_rpm +
                              t * self.schedule[i+1].rotor_rpm,
                    wing_flap=(1-t) * self.schedule[i].wing_flap +
                              t * self.schedule[i+1].wing_flap,
                )

        # Below or above range
        if airspeed < self.schedule[0].airspeed:
            return self.schedule[0]
        else:
            return self.schedule[-1]

    def command_symmetric_tilt(
        self,
        angle_cmd: float,
        dt: float,
        rotor_omega: float = 0.0,
        rotor_inertia: float = 0.0,
        V_inf: float = 0.0,
    ) -> tuple[NacelleState, NacelleState]:
        """
        Command both nacelles to the same angle.

        Args:
            angle_cmd: Commanded angle [rad]
            dt: Time step [s]
            rotor_omega: Rotor speed [rad/s]
            rotor_inertia: Rotor inertia [kg·m²]
            V_inf: Airspeed [m/s]

        Returns:
            (left_state, right_state)
        """
        left = self.nacelle_left.step(
            angle_cmd, dt, rotor_omega, rotor_inertia, V_inf
        )
        right = self.nacelle_right.step(
            angle_cmd, dt, rotor_omega, rotor_inertia, V_inf
        )

        return left, right

    def command_differential_tilt(
        self,
        base_angle: float,
        differential: float,
        dt: float,
        rotor_omega: float = 0.0,
        rotor_inertia: float = 0.0,
        V_inf: float = 0.0,
    ) -> tuple[NacelleState, NacelleState]:
        """
        Command differential nacelle tilt for roll control.

        Args:
            base_angle: Base nacelle angle [rad]
            differential: Differential angle [rad] (left - right)
            dt: Time step [s]
            rotor_omega: Rotor speed [rad/s]
            rotor_inertia: Rotor inertia [kg·m²]
            V_inf: Airspeed [m/s]

        Returns:
            (left_state, right_state)
        """
        # Limit differential
        differential = np.clip(
            differential,
            -self.max_differential,
            self.max_differential
        )

        left_cmd = base_angle + differential / 2
        right_cmd = base_angle - differential / 2

        left = self.nacelle_left.step(
            left_cmd, dt, rotor_omega, rotor_inertia, V_inf
        )
        right = self.nacelle_right.step(
            right_cmd, dt, rotor_omega, rotor_inertia, V_inf
        )

        return left, right
