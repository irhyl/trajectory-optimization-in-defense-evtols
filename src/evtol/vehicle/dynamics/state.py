"""
Vehicle Dynamics State Representation

Defines the canonical state vector for 6-DoF rigid body dynamics integration.
All conventions follow aerospace NED (North-East-Down) body-frame standards.

Quaternion Convention
---------------------
Hamilton convention: q = [w, x, y, z], where w is the scalar part.
Identity (no rotation): q = [1, 0, 0, 0].

This is consistent with the quaternion derivative matrix used in rigid_body.py:
    q̇ = ½ · Ω(ω) · q

State Vector Layout (length 23)
---------------------------------
[0:3]   position         NED frame [m]
[3:6]   velocity         Body frame [u, v, w] [m/s]
[6:10]  attitude         Quaternion [w, x, y, z]
[10:13] angular_velocity Body frame [p, q, r] [rad/s]
[13:15] nacelle_angles   Left, right tilt [rad]  (0=cruise, π/2=hover)
[15:17] nacelle_rates    Left, right tilt rates [rad/s]
[17:19] rotor_speeds     Left, right [rad/s]
[19:21] collective_pitch Left, right [rad]
[21]    battery_soc      [0–1]
[22]    battery_temp     [°C]

References
----------
Diebel, J. (2006). Representing Attitude: Euler Angles, Unit Quaternions,
    and Rotation Vectors. Stanford University Technical Report.
Shuster, M.D. (1993). A Survey of Attitude Representations.
    J. Astronautical Sciences, 41(4), 439–517.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Quaternion helper class
# ---------------------------------------------------------------------------

class Quaternion:
    """
    Unit quaternion representing SO(3) attitude.

    Convention: q = [w, x, y, z]  (Hamilton, scalar-first).
    All operations maintain unit norm.

    Parameters
    ----------
    w, x, y, z : float
        Scalar and vector components.
    """

    __slots__ = ("_q",)

    def __init__(self, w: float = 1.0, x: float = 0.0,
                 y: float = 0.0, z: float = 0.0):
        q = np.array([w, x, y, z], dtype=np.float64)
        norm = np.linalg.norm(q)
        if norm < 1e-12:
            raise ValueError("Quaternion norm is zero; cannot normalise.")
        self._q = q / norm

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "Quaternion":
        """Construct from a 4-element array [w, x, y, z]."""
        arr = np.asarray(arr, dtype=np.float64).ravel()
        if arr.shape != (4,):
            raise ValueError(f"Expected shape (4,), got {arr.shape}")
        return cls(arr[0], arr[1], arr[2], arr[3])

    @classmethod
    def identity(cls) -> "Quaternion":
        """Return the identity quaternion (zero rotation)."""
        return cls(1.0, 0.0, 0.0, 0.0)

    @classmethod
    def from_euler(cls, roll: float, pitch: float, yaw: float) -> "Quaternion":
        """
        Construct from ZYX Euler angles (aerospace convention).

        Parameters
        ----------
        roll  : float  [rad] — rotation about x-axis
        pitch : float  [rad] — rotation about y-axis
        yaw   : float  [rad] — rotation about z-axis
        """
        cr, sr = np.cos(roll * 0.5), np.sin(roll * 0.5)
        cp, sp = np.cos(pitch * 0.5), np.sin(pitch * 0.5)
        cy, sy = np.cos(yaw * 0.5), np.sin(yaw * 0.5)
        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy
        return cls(w, x, y, z)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def array(self) -> np.ndarray:
        """Return quaternion as numpy array [w, x, y, z]."""
        return self._q.copy()

    @property
    def w(self) -> float:
        return float(self._q[0])

    @property
    def x(self) -> float:
        return float(self._q[1])

    @property
    def y(self) -> float:
        return float(self._q[2])

    @property
    def z(self) -> float:
        return float(self._q[3])

    # ------------------------------------------------------------------
    # SO(3) operations
    # ------------------------------------------------------------------

    def to_dcm(self) -> np.ndarray:
        """
        Convert to Direction Cosine Matrix (rotation from NED to body frame).

        R such that v_body = R @ v_ned.

        Derived from unit quaternion q = [w, x, y, z]:

            R = (w²-‖v‖²)I + 2vvᵀ + 2w[v]×

        where v = [x, y, z].
        """
        w, x, y, z = self._q
        x2, y2, z2 = x*x, y*y, z*z
        wx, wy, wz = w*x, w*y, w*z
        xy, xz, yz = x*y, x*z, y*z
        return np.array([
            [1.0 - 2*(y2 + z2),   2*(xy + wz),         2*(xz - wy)],
            [2*(xy - wz),          1.0 - 2*(x2 + z2),   2*(yz + wx)],
            [2*(xz + wy),          2*(yz - wx),          1.0 - 2*(x2 + y2)],
        ], dtype=np.float64)

    def to_euler(self) -> tuple[float, float, float]:
        """
        Convert to ZYX Euler angles (roll, pitch, yaw) [rad].

        Singularity at pitch = ±90° (gimbal lock); handled by
        clamping the arcsin argument.
        """
        w, x, y, z = self._q
        # roll
        sinr_cosp = 2.0 * (w*x + y*z)
        cosr_cosp = 1.0 - 2.0 * (x*x + y*y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        # pitch
        sinp = np.clip(2.0 * (w*y - z*x), -1.0, 1.0)
        pitch = np.arcsin(sinp)
        # yaw
        siny_cosp = 2.0 * (w*z + x*y)
        cosy_cosp = 1.0 - 2.0 * (y*y + z*z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        return roll, pitch, yaw

    def conjugate(self) -> "Quaternion":
        """Return the conjugate (inverse for unit quaternion)."""
        return Quaternion(self.w, -self.x, -self.y, -self.z)

    def __mul__(self, other: "Quaternion") -> "Quaternion":
        """Hamilton product q1 ⊗ q2."""
        w1, x1, y1, z1 = self._q
        w2, x2, y2, z2 = other._q
        return Quaternion(
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
        )

    def normalise(self) -> "Quaternion":
        """Return normalised copy (should already be unit, but guards drift)."""
        return Quaternion.from_array(self._q)

    def __repr__(self) -> str:
        return (f"Quaternion(w={self.w:.4f}, x={self.x:.4f}, "
                f"y={self.y:.4f}, z={self.z:.4f})")


# ---------------------------------------------------------------------------
# VehicleState
# ---------------------------------------------------------------------------

@dataclass
class VehicleState:
    """
    Complete 6-DoF vehicle state for dynamics integration.

    This is the low-level dynamics-layer state, distinct from the
    higher-level ``evtol.core.state.VehicleState`` which uses flat numpy
    arrays suitable for logging and inter-layer data transfer.

    Attributes
    ----------
    time             : float        Simulation time [s]
    position         : ndarray(3)   NED position [m]
    velocity         : ndarray(3)   Body-frame velocity [u, v, w] [m/s]
    attitude         : Quaternion   Attitude [w, x, y, z]
    angular_velocity : ndarray(3)   Body-frame angular rates [p, q, r] [rad/s]
    nacelle_angles   : ndarray(2)   [left, right] tilt angle [rad]
    nacelle_rates    : ndarray(2)   [left, right] tilt rate [rad/s]
    rotor_speeds     : ndarray(2)   [left, right] rotor angular speed [rad/s]
    collective_pitch : ndarray(2)   [left, right] blade collective pitch [rad]
    battery_soc      : float        State of charge [0–1]
    battery_voltage  : float        Terminal voltage [V]
    battery_temp     : float        Battery temperature [°C]
    """

    time: float = 0.0

    position: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64))
    velocity: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64))
    attitude: Quaternion = field(
        default_factory=Quaternion.identity)
    angular_velocity: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64))

    nacelle_angles: np.ndarray = field(
        default_factory=lambda: np.array([np.pi/2, np.pi/2], dtype=np.float64))
    nacelle_rates: np.ndarray = field(
        default_factory=lambda: np.zeros(2, dtype=np.float64))
    rotor_speeds: np.ndarray = field(
        default_factory=lambda: np.array([157.0, 157.0], dtype=np.float64))
    collective_pitch: np.ndarray = field(
        default_factory=lambda: np.array([0.175, 0.175], dtype=np.float64))  # ~10 deg

    battery_soc: float = 1.0
    battery_voltage: float = 400.0
    battery_temp: float = 25.0

    def __post_init__(self) -> None:
        self.position = np.asarray(self.position, dtype=np.float64)
        self.velocity = np.asarray(self.velocity, dtype=np.float64)
        self.angular_velocity = np.asarray(self.angular_velocity, dtype=np.float64)
        self.nacelle_angles = np.asarray(self.nacelle_angles, dtype=np.float64)
        self.nacelle_rates = np.asarray(self.nacelle_rates, dtype=np.float64)
        self.rotor_speeds = np.asarray(self.rotor_speeds, dtype=np.float64)
        self.collective_pitch = np.asarray(self.collective_pitch, dtype=np.float64)
        if not isinstance(self.attitude, Quaternion):
            self.attitude = Quaternion.from_array(self.attitude)

    # ------------------------------------------------------------------
    # Flat-array serialisation (for numerical integrators)
    # ------------------------------------------------------------------

    def to_array(self) -> np.ndarray:
        """
        Serialise state to a flat 23-element vector.

        Layout: [pos(3), vel(3), quat(4), omega(3),
                 nacelle_angle(2), nacelle_rate(2),
                 rotor_speed(2), collective(2), soc(1), batt_temp(1)]
        """
        return np.concatenate([
            self.position,
            self.velocity,
            self.attitude.array,
            self.angular_velocity,
            self.nacelle_angles,
            self.nacelle_rates,
            self.rotor_speeds,
            self.collective_pitch,
            [self.battery_soc],
            [self.battery_temp],
        ])

    @classmethod
    def from_array(cls, x: np.ndarray, time: float = 0.0) -> "VehicleState":
        """Deserialise from a flat 23-element vector."""
        if len(x) != 23:
            raise ValueError(f"State vector must have 23 elements, got {len(x)}")
        return cls(
            time=time,
            position=x[0:3].copy(),
            velocity=x[3:6].copy(),
            attitude=Quaternion.from_array(x[6:10]),
            angular_velocity=x[10:13].copy(),
            nacelle_angles=x[13:15].copy(),
            nacelle_rates=x[15:17].copy(),
            rotor_speeds=x[17:19].copy(),
            collective_pitch=x[19:21].copy(),
            battery_soc=float(x[21]),
            battery_temp=float(x[22]),
        )

    def copy(self) -> "VehicleState":
        """Deep copy."""
        return VehicleState(
            time=self.time,
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            attitude=Quaternion.from_array(self.attitude.array),
            angular_velocity=self.angular_velocity.copy(),
            nacelle_angles=self.nacelle_angles.copy(),
            nacelle_rates=self.nacelle_rates.copy(),
            rotor_speeds=self.rotor_speeds.copy(),
            collective_pitch=self.collective_pitch.copy(),
            battery_soc=self.battery_soc,
            battery_voltage=self.battery_voltage,
            battery_temp=self.battery_temp,
        )


# ---------------------------------------------------------------------------
# ControlInput
# ---------------------------------------------------------------------------

@dataclass
class ControlInput:
    """
    Control input vector for the tiltrotor vehicle.

    Attributes
    ----------
    rotor_speed_cmd      : ndarray(2)  Commanded rotor speeds [rad/s]
    collective_cmd       : ndarray(2)  Commanded collective pitch [rad]
    differential_collective : float   Differential collective for yaw [rad]
    nacelle_cmd          : ndarray(2)  Commanded nacelle angles [rad]
    throttle             : float       Throttle setting [0–1]
    aileron              : float       Aileron deflection [rad]
    elevator             : float       Elevator deflection [rad]
    rudder               : float       Rudder deflection [rad]
    """

    rotor_speed_cmd: np.ndarray = field(
        default_factory=lambda: np.array([157.0, 157.0], dtype=np.float64))
    collective_cmd: np.ndarray = field(
        default_factory=lambda: np.array([0.175, 0.175], dtype=np.float64))
    differential_collective: float = 0.0
    nacelle_cmd: np.ndarray = field(
        default_factory=lambda: np.array([np.pi/2, np.pi/2], dtype=np.float64))
    throttle: float = 0.5
    aileron: float = 0.0
    elevator: float = 0.0
    rudder: float = 0.0

    def __post_init__(self) -> None:
        self.rotor_speed_cmd = np.asarray(self.rotor_speed_cmd, dtype=np.float64)
        self.collective_cmd = np.asarray(self.collective_cmd, dtype=np.float64)
        self.nacelle_cmd = np.asarray(self.nacelle_cmd, dtype=np.float64)

    def copy(self) -> "ControlInput":
        return ControlInput(
            rotor_speed_cmd=self.rotor_speed_cmd.copy(),
            collective_cmd=self.collective_cmd.copy(),
            differential_collective=self.differential_collective,
            nacelle_cmd=self.nacelle_cmd.copy(),
            throttle=self.throttle,
            aileron=self.aileron,
            elevator=self.elevator,
            rudder=self.rudder,
        )


__all__ = ["Quaternion", "VehicleState", "ControlInput"]
