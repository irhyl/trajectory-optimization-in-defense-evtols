"""
Tiltrotor Vehicle Model - Standalone Implementation

This module provides a complete, self-contained tiltrotor vehicle model
that can be used without depending on other subpackages.

Features:
- Full 6-DoF dynamics with quaternion attitude
- Simplified rotor thrust model
- Basic battery discharge model
- Multi-domain signature estimates (RCS, IR, Acoustic)
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
import logging

# Import canonical state definitions — do NOT redefine these locally.
from ..core.state import VehicleState, FlightPhase

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Backward-compatibility shims so that any code that created a VehicleState
# via vehicle_model.VehicleState still works.  The canonical class lives in
# core.state and its fields differ slightly from the old local definition:
#   old: temp_battery / temp_motor
#   new: temperature_battery / temperature_motor
# We expose thin property aliases on the imported class so callers that use
# the old names do not break.  Python allows monkey-patching dataclasses.
# ---------------------------------------------------------------------------
if not hasattr(VehicleState, "temp_battery"):
    VehicleState.temp_battery = property(
        lambda self: self.temperature_battery,
        lambda self, v: setattr(self, "temperature_battery", v),
    )
if not hasattr(VehicleState, "temp_motor"):
    VehicleState.temp_motor = property(
        lambda self: self.temperature_motor,
        lambda self, v: setattr(self, "temperature_motor", v),
    )


@dataclass
class TiltrotorConfig:
    """Tiltrotor configuration."""
    # All fields with defaults must come after those without
    name: str = "Defense_eVTOL_V280"
    mass_kg: float = 2500.0
    Ixx: float = 3000.0
    Iyy: float = 8000.0
    Izz: float = 10000.0
    Ixz: float = 500.0
    wingspan_m: float = 12.0
    fuselage_length_m: float = 14.0
    num_rotors: int = 4
    rotor_radius_m: float = 1.5
    max_takeoff_weight_kg: float = 3000.0
    max_airspeed_mps: float = 150.0
    battery_capacity_wh: float = 200000.0  # 200 kWh
    battery_voltage_nom: float = 400.0
    dt_nominal: float = 0.01
    rotor_positions: list[tuple[float, float, float]] = field(default_factory=list)

    def __post_init__(self):
        if not self.rotor_positions:
            # Symmetric rotor positions for balanced moments
            # All rotors equidistant from CG
            self.rotor_positions = [
                (2.5, -3.0, 0.5),   # Front left (tilting)
                (2.5, 3.0, 0.5),    # Front right (tilting)
                (-2.5, -3.0, 0.5),  # Rear left (fixed, same distance)
                (-2.5, 3.0, 0.5),   # Rear right (fixed, same distance)
            ]


@dataclass
class ControlInputs:
    """Control input vector."""
    rotor_speed_cmd: np.ndarray = field(default_factory=lambda: np.ones(4) * 157.0)
    nacelle_angle_cmd: float = 90.0
    aileron: float = 0.0
    elevator: float = 0.0
    rudder: float = 0.0
    collective: float = 0.5

    # Autopilot commands (optional - if set, override direct rotor commands)
    altitude_cmd: float = None      # Commanded altitude (m, AGL positive up)
    airspeed_cmd: float = None      # Commanded airspeed (m/s)
    heading_cmd: float = None       # Commanded heading (deg)
    climb_rate_cmd: float = None    # Commanded climb rate (m/s)


@dataclass
class VehicleOutput:
    """Vehicle model outputs."""
    state: VehicleState = None
    phase: FlightPhase = FlightPhase.GROUND

    # Forces/moments
    lift_N: float = 0.0
    drag_N: float = 0.0
    total_thrust_N: float = 0.0
    aero_moment: np.ndarray = field(default_factory=lambda: np.zeros(3))

    # Propulsion
    rotor_thrusts: np.ndarray = field(default_factory=lambda: np.zeros(4))
    rotor_torques: np.ndarray = field(default_factory=lambda: np.zeros(4))
    rotor_powers: np.ndarray = field(default_factory=lambda: np.zeros(4))
    nacelle_angle: float = 90.0

    # Energy
    battery_soc: float = 1.0
    battery_voltage: float = 400.0
    total_power_w: float = 0.0
    range_remaining_km: float = 0.0
    endurance_remaining_min: float = 0.0

    # Thermal
    temp_battery: float = 25.0
    temp_motor: float = 50.0

    # Signatures
    rcs_dbsm: float = -10.0
    ir_intensity: float = 0.0
    acoustic_spl: float = 70.0

    warnings: list[str] = field(default_factory=list)


class TiltrotorVehicle:
    """
    Complete tiltrotor eVTOL vehicle model (standalone).

    This class provides physics-based simulation of a tiltrotor aircraft
    without external dependencies.

    Usage:
        vehicle = TiltrotorVehicle()
        vehicle.initialize(position=(0, 0, -100))

        controls = ControlInputs()
        output = vehicle.step(controls, dt=0.01)
    """

    def __init__(self, config: TiltrotorConfig | None = None):
        """Initialize vehicle model."""
        self.config = config or TiltrotorConfig()
        self.state = VehicleState()
        self.output = VehicleOutput()
        self.time = 0.0

        # FIXED: Single canonical configuration path (eliminate duck-typing)
        # Extract inertia tensor
        if hasattr(self.config, 'Ixx'):
            # Direct Ixx/Iyy/Izz fields (canonical format)
            self.inertia = np.diag([
                self.config.Ixx,
                self.config.Iyy,
                self.config.Izz
            ])
            if hasattr(self.config, 'Ixz'):
                self.inertia[0, 2] = self.inertia[2, 0] = self.config.Ixz
        elif hasattr(self.config, 'mass') and hasattr(self.config.mass, 'inertia_matrix'):
            self.inertia = self.config.mass.inertia_matrix
        else:
            raise ValueError(
                f"Invalid config: must have Ixx/Iyy/Izz fields or mass.inertia_matrix. "
                f"Config type: {type(self.config)}"
            )
        
        # Validate inertia matrix
        try:
            self.inertia_inv = np.linalg.inv(self.inertia)
        except np.linalg.LinAlgError:
            raise ValueError(f"Inertia matrix is singular: {self.inertia}")

        # Extract battery capacity
        if hasattr(self.config, 'battery_capacity_wh'):
            self.battery_energy_wh = self.config.battery_capacity_wh
        elif hasattr(self.config, 'battery') and hasattr(self.config.battery, 'capacity_wh'):
            self.battery_energy_wh = self.config.battery.capacity_wh
        else:
            logger.warning("Battery capacity not found in config, using default 200 kWh")
            self.battery_energy_wh = 200000.0
        
        # Extract vehicle mass
        if hasattr(self.config, 'mass_kg'):
            self.mass = self.config.mass_kg
        elif hasattr(self.config, 'mass') and hasattr(self.config.mass, 'total_mass'):
            self.mass = self.config.mass.total_mass
        else:
            logger.warning("Vehicle mass not found in config, using default 2500 kg")
            self.mass = 2500.0

        logger.info(f"TiltrotorVehicle '{self.config.name}' initialized (mass: {self.mass:.1f} kg)")

    def initialize(
        self,
        position: tuple[float, float, float] = (0.0, 0.0, 0.0),
        velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
        attitude_euler_deg: tuple[float, float, float] = (0.0, 0.0, 0.0),
        battery_soc: float = 1.0,
    ) -> None:
        """Initialize vehicle state."""
        self.state.position = np.array(position, dtype=np.float64)
        self.state.velocity = np.array(velocity, dtype=np.float64)

        roll, pitch, yaw = np.radians(attitude_euler_deg)
        self.state.attitude = self._euler_to_quaternion(roll, pitch, yaw)

        self.state.angular_rates = np.zeros(3, dtype=np.float64)
        self.state.battery_soc = battery_soc
        self.state.nacelle_angle = 90.0
        # Initialize rotor speeds at hover (~157 rad/s for 2500kg vehicle)
        self.state.rotor_speeds = np.ones(4, dtype=np.float64) * 157.0

        self.battery_energy_wh = battery_soc * self.config.battery_capacity_wh
        self.time = 0.0

    def step(
        self,
        controls: ControlInputs,
        dt: float | None = None,
        environment: dict | None = None,
    ) -> VehicleOutput:
        """
        Advance simulation by one timestep.

        Args:
            controls: Control inputs
            dt: Timestep (s)
            environment: Environmental conditions

        Returns:
            Vehicle outputs
        """
        dt = dt or self.config.dt_nominal
        env = environment or {}

        wind = np.array(env.get('wind_ned', [0, 0, 0]))
        density = env.get('density', 1.225)
        gravity = 9.81

        self.output.warnings = []

        # Current state for autopilot
        current_altitude = -self.state.position[2]  # NED to altitude
        current_climb_rate = -self.state.velocity[2]  # NED to climb rate
        current_airspeed = np.linalg.norm(self.state.velocity)  # Total airspeed

        # === AUTOPILOT: Convert altitude/speed commands to rotor speed ===
        omega_hover = 157.0  # rad/s for hover equilibrium
        rotor_speed_cmd = controls.rotor_speed_cmd.copy()

        # Airspeed limiter - reduce thrust if exceeding max airspeed
        max_airspeed = self.config.max_airspeed_mps  # 150 m/s
        if current_airspeed > max_airspeed:
            # Reduce rotor speed proportionally
            speed_factor = max_airspeed / current_airspeed
            rotor_speed_cmd = rotor_speed_cmd * speed_factor

        if controls.altitude_cmd is not None:
            # Altitude hold controller (proportional + derivative)
            alt_error = controls.altitude_cmd - current_altitude
            climb_rate_target = np.clip(alt_error * 0.5, -5.0, 5.0)  # 0.5 gain, max 5 m/s

            if controls.climb_rate_cmd is not None:
                climb_rate_target = controls.climb_rate_cmd

            climb_rate_error = climb_rate_target - current_climb_rate

            # Rotor speed adjustment accounting for nacelle angle
            # At hover (nacelle=90°), all thrust is vertical
            # At cruise (nacelle=10°), only sin(10°)=0.17 of thrust is vertical
            # Need to increase rotor speed to compensate for reduced vertical component
            nacelle_rad = np.radians(self.state.nacelle_angle)
            vertical_thrust_fraction = np.sin(nacelle_rad)
            vertical_thrust_fraction = max(vertical_thrust_fraction, 0.3)  # Limit to avoid division issues

            # Required omega for hover at current nacelle angle
            omega_required = omega_hover / np.sqrt(vertical_thrust_fraction)

            # Add climb rate adjustment
            delta_omega = climb_rate_error * 5.0  # Gain: 5 rad/s per m/s error
            rotor_speed_cmd = np.ones(4) * (omega_required + delta_omega)
            rotor_speed_cmd = np.clip(rotor_speed_cmd, 100, 300)

        # === 1. Nacelle Dynamics ===
        nacelle_rate = 15.0  # deg/s max
        nacelle_error = controls.nacelle_angle_cmd - self.state.nacelle_angle
        nacelle_cmd_rate = np.clip(nacelle_error / dt, -nacelle_rate, nacelle_rate)
        self.state.nacelle_angle += nacelle_cmd_rate * dt
        self.state.nacelle_angle = np.clip(self.state.nacelle_angle, 0, 95)

        # === 2. Rotor Dynamics ===
        tau_rotor = 0.2
        for i in range(4):
            omega_error = rotor_speed_cmd[i] - self.state.rotor_speeds[i]
            omega_dot = omega_error / tau_rotor
            self.state.rotor_speeds[i] += omega_dot * dt
            self.state.rotor_speeds[i] = np.clip(self.state.rotor_speeds[i], 0, 400)

        # === 3. Compute Forces ===
        R_nb = self._quaternion_to_rotation(self.state.attitude)
        V_air = self.state.velocity - wind
        airspeed = np.linalg.norm(V_air)

        # Rotor thrust - using momentum theory based approach
        # T = CT * rho * A * (Omega*R)^2  where CT ~ 0.012 at hover
        # For 2500 kg vehicle in hover, each rotor needs ~6130 N
        # With R=1.5m, A=7.07m^2, rho=1.225, solve for omega
        # omega_hover ~ 157 rad/s (about 1500 RPM)

        total_thrust_body = np.zeros(3, dtype=np.float64)
        total_moment = np.zeros(3, dtype=np.float64)
        rotor_powers = []

        rotor_disk_area = np.pi * self.config.rotor_radius_m**2  # m²
        hover_thrust_per_rotor = self.config.mass_kg * 9.81 / 4  # ~6130 N

        for i in range(4):
            # Tilt angle (front 2 tilt, rear 2 fixed)
            if i < 2:
                tilt_rad = np.radians(self.state.nacelle_angle)
            else:
                tilt_rad = np.radians(90)

            # Thrust direction in body frame
            thrust_dir = np.array([
                -np.cos(tilt_rad),
                0.0,
                -np.sin(tilt_rad),
            ], dtype=np.float64)

            # Thrust magnitude using realistic scaling
            # Normalized rotor speed: omega/omega_hover where omega_hover~157 rad/s
            omega = self.state.rotor_speeds[i]
            omega_hover = 157.0
            omega_ratio = omega / omega_hover

            # Thrust scales with omega^2
            thrust = hover_thrust_per_rotor * omega_ratio**2
            thrust = np.clip(thrust, 0, 15000)  # Limit max thrust

            thrust_vec = thrust_dir * thrust
            total_thrust_body += thrust_vec
            self.output.rotor_thrusts[i] = thrust

            # Power using momentum theory: P = T * sqrt(T / (2*rho*A))
            # For R=1.5m, A=7.07m², rho=1.225: sqrt(2*rho*A) = 4.16
            # P = T^1.5 / sqrt(2*rho*A) = k * T^1.5 where k = 1/sqrt(2*rho*A) ~ 0.24
            k_power = 1.0 / np.sqrt(2 * density * rotor_disk_area)  # ~0.24 for 1.5m rotor
            power = k_power * thrust**1.5
            torque = power / max(omega, 1.0)  # Q = P / omega

            self.output.rotor_torques[i] = torque
            self.output.rotor_powers[i] = power
            rotor_powers.append(power)

            # Moment from offset
            pos = np.array(self.config.rotor_positions[i], dtype=np.float64)
            total_moment += np.cross(pos, thrust_vec)

            # Reaction torque (counter-rotating pairs)
            yaw_sign = 1.0 if i % 2 == 0 else -1.0
            total_moment[2] += yaw_sign * torque * 0.1

        self.output.total_thrust_N = np.linalg.norm(total_thrust_body)
        self.output.nacelle_angle = self.state.nacelle_angle

        # Aerodynamic forces (simplified)
        S_wing = self.config.wingspan_m * 1.5  # ~18 m² wing area

        if airspeed > 5:
            # Lift from wing (acts in body Z direction, upward)
            # CL varies with angle of attack - simplified as function of nacelle angle
            # At hover (nacelle=90), no forward speed, no lift
            # At cruise (nacelle=0-30), wing generates lift
            wing_AoA = 5.0  # degrees
            CL = 0.5 * np.sin(2 * np.radians(wing_AoA))  # ~0.17
            lift = 0.5 * density * airspeed**2 * S_wing * CL

            # Drag (parasitic + induced)
            # Parasitic drag coefficient
            CD0 = 0.03  # Clean config
            # Add fuselage/nacelle drag
            CD_parasitic = CD0 + 0.02 * (1 - np.cos(np.radians(self.state.nacelle_angle)))
            # Induced drag
            AR = self.config.wingspan_m**2 / S_wing  # Aspect ratio
            e = 0.8  # Oswald efficiency
            CD_induced = CL**2 / (np.pi * AR * e)
            CD_total = CD_parasitic + CD_induced

            drag = 0.5 * density * airspeed**2 * S_wing * CD_total

            self.output.lift_N = lift
            self.output.drag_N = drag
        else:
            self.output.lift_N = 0
            self.output.drag_N = 0

        # Total force computation
        # Convert thrust from body to NED frame
        total_thrust_ned = R_nb @ total_thrust_body

        # Weight in NED (down is positive Z)
        weight_ned = np.array([0.0, 0.0, self.config.mass_kg * gravity], dtype=np.float64)

        # Aerodynamic forces in NED frame
        # Drag opposes velocity direction
        if airspeed > 5:
            velocity_dir = V_air / airspeed
            drag_ned = -self.output.drag_N * velocity_dir
            # Lift acts perpendicular to velocity, in vertical plane (simplified)
            lift_ned = np.array([0.0, 0.0, -self.output.lift_N], dtype=np.float64)
            aero_ned = drag_ned + lift_ned
        else:
            aero_ned = np.zeros(3, dtype=np.float64)

        # Total force in NED frame
        F_total_ned = total_thrust_ned + aero_ned + weight_ned

        # === 4. State Derivatives ===
        # Translational dynamics (in NED frame for simplicity)
        omega = self.state.angular_rates

        # Add velocity damping for numerical stability
        damping_vel = 0.02  # Small damping coefficient
        V_dot_ned = F_total_ned / self.config.mass_kg - damping_vel * self.state.velocity

        # Limit acceleration for stability
        V_dot_ned = np.clip(V_dot_ned, -50.0, 50.0)

        # Rotational dynamics with damping
        damping_omega = 0.5  # Angular rate damping
        omega_dot = self.inertia_inv @ (
            total_moment - np.cross(omega, self.inertia @ omega) - damping_omega * self.inertia @ omega
        )

        # Limit angular acceleration
        omega_dot = np.clip(omega_dot, -5.0, 5.0)

        # Quaternion kinematics
        q = self.state.attitude
        q_dot = 0.5 * self._quaternion_multiply(
            q, np.array([0, omega[0], omega[1], omega[2]])
        )

        # === 5. Integrate ===
        self.state.position += self.state.velocity * dt
        self.state.velocity += V_dot_ned * dt
        self.state.attitude += q_dot * dt
        self.state.attitude /= np.linalg.norm(self.state.attitude)
        self.state.angular_rates += omega_dot * dt

        # === 6. Energy Update ===
        total_power = sum(rotor_powers)
        self.output.total_power_w = total_power

        energy_used_wh = total_power * dt / 3600
        self.battery_energy_wh -= energy_used_wh
        self.battery_energy_wh = max(0, self.battery_energy_wh)

        self.state.battery_soc = self.battery_energy_wh / self.config.battery_capacity_wh
        self.output.battery_soc = self.state.battery_soc
        self.output.battery_voltage = self.config.battery_voltage_nom * (0.9 + 0.1 * self.state.battery_soc)

        if total_power > 0:
            time_remaining_s = self.battery_energy_wh * 3600 / total_power
            self.output.endurance_remaining_min = time_remaining_s / 60
            if airspeed > 5:
                self.output.range_remaining_km = (time_remaining_s / 3600) * airspeed * 3.6

        # === 7. Thermal Update ===
        motor_heat = total_power * 0.05
        self.state.temp_motor += motor_heat * 1e-6 * dt
        self.state.temp_motor = min(120, self.state.temp_motor)
        self.output.temp_motor = self.state.temp_motor
        self.output.temp_battery = self.state.temp_battery

        # === 8. Signatures ===
        # RCS (simplified)
        self.output.rcs_dbsm = -5 + 5 * np.sin(np.radians(self.state.nacelle_angle))

        # IR (simplified)
        self.output.ir_intensity = (self.state.temp_motor - 25) * 0.1

        # Acoustic (simplified)
        avg_rpm = np.mean(self.state.rotor_speeds) * 60 / (2 * np.pi)
        self.output.acoustic_spl = 60 + 20 * np.log10(avg_rpm / 100 + 0.1)

        # === 9. Flight Phase ===
        altitude = -self.state.position[2]
        self.output.phase = self._determine_phase(altitude, airspeed)

        # Update time
        self.time += dt
        self.output.state = self.state

        return self.output

    def _determine_phase(self, altitude: float, airspeed: float) -> FlightPhase:
        """Determine current flight phase."""
        nacelle = self.state.nacelle_angle
        vz = self.state.velocity[2]

        if altitude < 1:
            return FlightPhase.GROUND
        elif nacelle > 80:
            if vz > 0.5:
                return FlightPhase.DESCENT
            elif altitude < 5:
                return FlightPhase.GROUND   # landing roll-out
            else:
                return FlightPhase.HOVER
        elif nacelle < 20 and airspeed > 30:
            return FlightPhase.CRUISE
        elif airspeed > 30:
            return FlightPhase.TRANSITION_TO_CRUISE
        else:
            return FlightPhase.TRANSITION_TO_HOVER

    def _euler_to_quaternion(self, roll: float, pitch: float, yaw: float) -> np.ndarray:
        """Convert Euler angles to quaternion."""
        cr, cp, cy = np.cos([roll/2, pitch/2, yaw/2])
        sr, sp, sy = np.sin([roll/2, pitch/2, yaw/2])

        return np.array([
            cr*cp*cy + sr*sp*sy,
            sr*cp*cy - cr*sp*sy,
            cr*sp*cy + sr*cp*sy,
            cr*cp*sy - sr*sp*cy,
        ])

    def _quaternion_to_rotation(self, q: np.ndarray) -> np.ndarray:
        """Convert quaternion to rotation matrix."""
        w, x, y, z = q
        return np.array([
            [1-2*(y**2+z**2), 2*(x*y-w*z), 2*(x*z+w*y)],
            [2*(x*y+w*z), 1-2*(x**2+z**2), 2*(y*z-w*x)],
            [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x**2+y**2)],
        ])

    def _quaternion_multiply(self, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """Quaternion multiplication."""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
        ])

    def get_euler_attitude(self) -> tuple[float, float, float]:
        """Get attitude as Euler angles (deg)."""
        q = self.state.attitude

        sinr_cosp = 2 * (q[0]*q[1] + q[2]*q[3])
        cosr_cosp = 1 - 2 * (q[1]**2 + q[2]**2)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        sinp = 2 * (q[0]*q[2] - q[3]*q[1])
        pitch = np.arcsin(np.clip(sinp, -1, 1))

        siny_cosp = 2 * (q[0]*q[3] + q[1]*q[2])
        cosy_cosp = 1 - 2 * (q[2]**2 + q[3]**2)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return tuple(np.degrees([roll, pitch, yaw]))

    def get_state(self) -> VehicleState:
        """Get current vehicle state."""
        return self.state

    def to_dict(self) -> dict:
        """Convert vehicle state to dictionary."""
        euler = self.get_euler_attitude()
        return {
            'time': self.time,
            'phase': self.output.phase.value,
            'position_ned': self.state.position.tolist(),
            'velocity_ned': self.state.velocity.tolist(),
            'attitude_euler_deg': list(euler),
            'airspeed_mps': float(np.linalg.norm(self.state.velocity)),
            'altitude_m': float(-self.state.position[2]),
            'nacelle_angle_deg': float(self.state.nacelle_angle),
            'thrust_N': float(self.output.total_thrust_N),
            'power_w': float(self.output.total_power_w),
            'battery_soc': float(self.output.battery_soc),
            'rcs_dbsm': float(self.output.rcs_dbsm),
            'acoustic_spl': float(self.output.acoustic_spl),
        }
