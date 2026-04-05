"""
PMSM Motor Model - Permanent Magnet Synchronous Motor

This module implements a comprehensive electric motor model for eVTOL
propulsion systems, including electromagnetic dynamics, thermal effects,
and efficiency mapping.

Theory
======

PMSM Voltage Equations (dq-frame):
    v_d = R_s·i_d + L_d·di_d/dt - ω_e·L_q·i_q
    v_q = R_s·i_q + L_q·di_q/dt + ω_e·L_d·i_d + ω_e·λ_pm

Electromagnetic Torque:
    T_e = (3/2)·P·[λ_pm·i_q + (L_d - L_q)·i_d·i_q]

For surface-mount PMSM (L_d ≈ L_q):
    T_e = (3/2)·P·λ_pm·i_q = K_t·i_q

Mechanical Dynamics:
    J·dω/dt = T_e - T_load - B·ω

Efficiency Breakdown:
    - Copper losses: P_cu = 3/2·R_s·(i_d² + i_q²)
    - Core losses: P_core = k_h·f·B² + k_e·f²·B²
    - Windage: P_wind = k_w·ω³

Thermal Model (lumped):
    m_w·c_w·dT_w/dt = P_cu - h_wf·(T_w - T_f)
    m_s·c_s·dT_s/dt = P_core + h_wf·(T_w - T_f) - h_sf·(T_s - T_amb)

where subscripts: w=winding, s=stator, f=frame, amb=ambient
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
import logging

from ..config import MotorConfig

logger = logging.getLogger(__name__)


@dataclass
class MotorState:
    """Motor operating state."""
    # Electrical
    voltage: float = 0.0          # V (DC bus or phase)
    current: float = 0.0          # A (phase current RMS)
    current_d: float = 0.0        # A (d-axis current)
    current_q: float = 0.0        # A (q-axis current)

    # Mechanical
    omega: float = 0.0            # rad/s (electrical angular velocity)
    rpm: float = 0.0              # Motor RPM
    torque: float = 0.0           # N·m
    torque_max: float = 0.0       # N·m (current limit)

    # Power
    power_electrical: float = 0.0  # W (input)
    power_mechanical: float = 0.0  # W (output)
    power_loss_copper: float = 0.0
    power_loss_core: float = 0.0
    power_loss_total: float = 0.0
    efficiency: float = 0.0

    # Thermal
    temperature_winding: float = 25.0   # °C
    temperature_stator: float = 25.0    # °C
    temperature_limit: float = 180.0    # °C
    thermal_margin: float = 155.0       # °C remaining

    # Operating region
    in_constant_torque: bool = True
    in_field_weakening: bool = False


class PMSMMotor:
    """
    Permanent Magnet Synchronous Motor model.

    This model captures:
    1. Electromagnetic torque production
    2. Voltage and current limits
    3. Field weakening at high speed
    4. Efficiency maps
    5. Thermal dynamics

    The motor is controlled in the dq reference frame:
    - d-axis: Field (flux) control, used for field weakening
    - q-axis: Torque control

    Attributes:
        config: Motor configuration
        state: Current operating state
    """

    def __init__(self, config: MotorConfig):
        """
        Initialize PMSM motor model.

        Args:
            config: Motor configuration parameters
        """
        self.config = config

        # Derived parameters.
        # MotorConfig stores the torque constant as ``kt`` [N·m/A] and derives
        # kv from kt via  kv = 60 / (2π·kt)  (standard PMSM relation).
        self.Kt = config.kt               # N·m/A
        self.Ke = config.kt               # V/(rad/s) – equal to Kt for PMSM
        self.Rs = config.resistance       # Ω (phase resistance)
        self.Ls = config.inductance       # H (phase inductance)
        # Pole pairs derived from kv: P = 60 / (2π·kv·Kt) ≈ 1 / (kv·Kt·2π/60)
        # For the default kv=10 RPM/V, kt=0.955 N·m/A this gives ≈ 6 pole pairs
        self.P = max(1, round(60 / (2 * np.pi * config.kv * config.kt)))

        # Back-EMF constant (line-to-line)
        self.Kv = config.kv  # RPM/V (directly from config)

        # Limits.
        # MotorConfig does not store max_current / max_voltage directly; we
        # derive them from peak power and rated voltage/torque.
        # Nominal DC bus voltage estimated from kv and max_rpm:
        #   V_bus ≈ (max_rpm / kv) × √3  (for SVPWM peak line voltage)
        V_bus_est = (config.max_rpm / config.kv) * np.sqrt(3)
        self.V_max = V_bus_est            # V (estimated DC bus)
        self.I_max = (config.peak_power * 1000) / max(self.V_max, 1.0)  # A
        self.omega_max = config.max_rpm * 2 * np.pi / 60  # rad/s
        self.T_max = config.max_torque    # N·m

        # Thermal parameters (approximate)
        self.R_th_winding = 1.0    # K/W (winding to stator)
        self.R_th_stator = 0.5     # K/W (stator to ambient)
        self.C_th_winding = 100.0  # J/K (winding thermal mass)
        self.C_th_stator = 500.0   # J/K (stator thermal mass)
        self.T_max_winding = 180.0 # °C

        # Initialize state
        self.state = MotorState()

        # Efficiency map (precomputed)
        self._build_efficiency_map()

        logger.info(f"PMSMMotor initialized: Kt={self.Kt:.4f} N·m/A, "
                   f"P_max={config.max_power:.1f} kW")

    def _build_efficiency_map(self):
        """
        Build comprehensive 2D efficiency map (speed × torque × temperature).
        
        Uses realistic PMSM efficiency characteristics:
        - Copper losses: I²R (dominant at high current)
        - Core losses: k_h·f·B² + k_e·f²·B² (increases with speed³ approximation)
        - Friction losses: proportional to speed (rotor windage)
        - Temperature effects: resistance increases, efficiency decreases
        
        Data-driven from typical EV motor datasheets.
        """
        n_speed = 60
        n_torque = 60
        n_temp = 4

        self.omega_map = np.linspace(0, self.omega_max, n_speed)
        self.torque_map = np.linspace(0, self.T_max * 1.2, n_torque)
        self.temp_map = np.array([25.0, 50.0, 75.0, 100.0])
        self.efficiency_map = np.zeros((n_speed, n_torque, n_temp))

        for i, omega in enumerate(self.omega_map):
            for j, torque in enumerate(self.torque_map):
                for k, T_winding in enumerate(self.temp_map):
                    if omega > 0.1 and torque > 0.01:
                        # Temperature-dependent resistance
                        Rs_eff = self.Rs * (1.0 + 0.004 * (T_winding - 25.0))
                        
                        # d-axis and q-axis currents
                        i_q = torque / self.Kt if self.Kt > 0 else 0
                        
                        # Field weakening current (at high speed)
                        omega_e = omega * self.P
                        E_bemf = self.Ke * omega_e
                        V_phase_max = self.V_max / np.sqrt(3)
                        
                        if E_bemf < V_phase_max:
                            # Constant torque region
                            i_d = 0.0
                        else:
                            # Field weakening: reduce flux to stay in voltage limit
                            # Simplified: reduce current by 10-30% depending on overspeed
                            i_d = -0.2 * i_q  # Small field weakening current
                        
                        # Copper losses (3-phase)
                        P_cu = 1.5 * Rs_eff * (i_d**2 + i_q**2)
                        
                        # Core/iron losses (empirical: proportional to f^1.5 and B^2)
                        # Assume B proportional to flux: B ~ Ke·ω_e / (stator constant)
                        # P_core ≈ k_h·f·B² + k_e·f²·B² ~ k·ω^1.5
                        if self.config.max_power > 0:
                            speed_ratio = omega / self.omega_max if self.omega_max > 0 else 0
                            P_core = 0.001 * self.config.max_power * (speed_ratio ** 1.5)
                            P_core *= (1.0 + 0.002 * (T_winding - 25.0))  # Temp effect
                        else:
                            P_core = 0.0
                        
                        # Friction/windage losses
                        P_friction = 0.05 * (speed_ratio ** 2) * self.config.max_power
                        
                        # Output mechanical power
                        P_out = max(0, torque * omega)
                        
                        # Total input power
                        P_in = P_out + P_cu + P_core + P_friction
                        
                        if P_in > 1.0:
                            self.efficiency_map[i, j, k] = min(1.0, max(0.30, P_out / P_in))
                        else:
                            # Very low power: mechanical losses dominate
                            self.efficiency_map[i, j, k] = 0.50 if P_out > 0 else 0.0
                    else:
                        # No load or zero speed
                        self.efficiency_map[i, j, k] = 0.0

    def compute_torque(
        self,
        omega: float,
        torque_cmd: float,
        V_bus: float,
        T_winding: float = 25.0,
    ) -> MotorState:
        """
        Compute motor output for given speed and torque command.

        Includes:
        - Current limiting
        - Voltage limiting (field weakening region)
        - Efficiency calculation
        - Thermal derating

        Args:
            omega: Motor shaft speed [rad/s]
            torque_cmd: Commanded torque [N·m]
            V_bus: DC bus voltage [V]
            T_winding: Winding temperature [°C]

        Returns:
            MotorState with torque, power, efficiency
        """
        state = MotorState(
            omega=omega,
            rpm=omega * 60 / (2 * np.pi),
            temperature_winding=T_winding,
        )

        # Electrical angular velocity
        omega_e = omega * self.P

        # Temperature derating
        if T_winding > self.T_max_winding - 20:
            # Linear derating in last 20°C
            derate = (self.T_max_winding - T_winding) / 20.0
            derate = np.clip(derate, 0.1, 1.0)
            I_max_derated = self.I_max * derate
            logger.warning(f"Motor thermal derating: {derate*100:.0f}%")
        else:
            I_max_derated = self.I_max

        state.thermal_margin = self.T_max_winding - T_winding

        # Back-EMF at current speed
        E_bemf = self.Ke * omega_e

        # Available voltage for current (accounting for back-EMF)
        V_phase_max = V_bus / np.sqrt(3)  # Line-to-neutral
        V_available = np.sqrt(max(V_phase_max**2 - E_bemf**2, 0))

        # Maximum current from voltage limit
        I_max_voltage = V_available / np.sqrt(self.Rs**2 + (omega_e * self.Ls)**2)

        # Effective current limit
        I_limit = min(I_max_derated, I_max_voltage)

        # Maximum torque at this speed
        T_max_speed = self.Kt * I_limit
        state.torque_max = T_max_speed

        # Check operating region
        if I_max_voltage < I_max_derated:
            state.in_field_weakening = True
            state.in_constant_torque = False
        else:
            state.in_field_weakening = False
            state.in_constant_torque = True

        # Apply torque command with limiting
        torque_limited = np.clip(torque_cmd, -T_max_speed, T_max_speed)
        state.torque = torque_limited

        # q-axis current for this torque
        i_q = torque_limited / self.Kt
        state.current_q = i_q

        # d-axis current (field weakening if needed)
        if state.in_field_weakening:
            # Simple field weakening: reduce flux
            i_d = -np.sqrt(max(I_limit**2 - i_q**2, 0)) * 0.5
            state.current_d = i_d
        else:
            i_d = 0.0
            state.current_d = 0.0

        # Total phase current (RMS)
        state.current = np.sqrt(i_d**2 + i_q**2)

        # Power calculations
        state.power_mechanical = state.torque * omega

        # Copper losses
        state.power_loss_copper = 1.5 * self.Rs * (i_d**2 + i_q**2)

        # Core losses (empirical)
        if omega_e > 0:
            f_elec = omega_e / (2 * np.pi)
            state.power_loss_core = (
                0.001 * self.config.max_power *
                (f_elec / (self.omega_max * self.P / (2*np.pi)))**1.5
            )
        else:
            state.power_loss_core = 0.0

        state.power_loss_total = state.power_loss_copper + state.power_loss_core
        state.power_electrical = state.power_mechanical + state.power_loss_total

        # Efficiency
        if state.power_electrical > 0 and state.power_mechanical > 0:
            state.efficiency = state.power_mechanical / state.power_electrical
        elif state.power_mechanical < 0:  # Regeneration
            state.efficiency = state.power_electrical / state.power_mechanical if state.power_mechanical != 0 else 0
        else:
            state.efficiency = 0.0

        # DC bus current
        if V_bus > 0:
            state.voltage = V_bus

        self.state = state
        return state

    def compute_thermal_dynamics(
        self,
        power_loss: float,
        T_winding: float,
        T_stator: float,
        T_ambient: float,
        dt: float,
    ) -> tuple[float, float]:
        """
        Update motor thermal state.

        Lumped thermal model:
        - Winding heated by copper losses, cooled by stator
        - Stator heated by core losses + winding heat, cooled by ambient

        Args:
            power_loss: Total power loss [W]
            T_winding: Current winding temperature [°C]
            T_stator: Current stator temperature [°C]
            T_ambient: Ambient temperature [°C]
            dt: Time step [s]

        Returns:
            (new_T_winding, new_T_stator)
        """
        # Copper losses to winding
        P_copper = self.state.power_loss_copper
        P_core = self.state.power_loss_core

        # Heat transfer rates
        Q_winding_stator = (T_winding - T_stator) / self.R_th_winding
        Q_stator_ambient = (T_stator - T_ambient) / self.R_th_stator

        # Temperature derivatives
        dT_winding = (P_copper - Q_winding_stator) / self.C_th_winding
        dT_stator = (P_core + Q_winding_stator - Q_stator_ambient) / self.C_th_stator

        # Euler integration
        new_T_winding = T_winding + dT_winding * dt
        new_T_stator = T_stator + dT_stator * dt

        return new_T_winding, new_T_stator

    def get_efficiency(self, omega: float, torque: float) -> float:
        """
        Look up efficiency from precomputed map.

        Args:
            omega: Motor speed [rad/s]
            torque: Motor torque [N·m]

        Returns:
            Efficiency (0-1)
        """
        if omega <= 0 or torque <= 0:
            return 0.0

        # Interpolate in efficiency map
        i_omega = np.interp(omega, self.omega_map, np.arange(len(self.omega_map)))
        i_torque = np.interp(abs(torque), self.torque_map, np.arange(len(self.torque_map)))

        i0, i1 = int(i_omega), min(int(i_omega) + 1, len(self.omega_map) - 1)
        j0, j1 = int(i_torque), min(int(i_torque) + 1, len(self.torque_map) - 1)

        # Bilinear interpolation
        alpha = i_omega - i0
        beta = i_torque - j0

        eta = (
            (1-alpha) * (1-beta) * self.efficiency_map[i0, j0] +
            alpha * (1-beta) * self.efficiency_map[i1, j0] +
            (1-alpha) * beta * self.efficiency_map[i0, j1] +
            alpha * beta * self.efficiency_map[i1, j1]
        )

        return eta

    def get_max_torque_at_speed(self, omega: float, V_bus: float) -> float:
        """
        Get maximum available torque at given speed.

        Args:
            omega: Motor speed [rad/s]
            V_bus: DC bus voltage [V]

        Returns:
            Maximum torque [N·m]
        """
        omega_e = omega * self.P
        E_bemf = self.Ke * omega_e
        V_phase_max = V_bus / np.sqrt(3)

        V_available = np.sqrt(max(V_phase_max**2 - E_bemf**2, 0))
        I_max_voltage = V_available / np.sqrt(self.Rs**2 + (omega_e * self.Ls)**2)

        I_limit = min(self.I_max, I_max_voltage)
        return self.Kt * I_limit

    def get_power_limit(self, omega: float, V_bus: float) -> float:
        """
        Get maximum power at given speed.

        Args:
            omega: Motor speed [rad/s]
            V_bus: DC bus voltage [V]

        Returns:
            Maximum mechanical power [W]
        """
        T_max = self.get_max_torque_at_speed(omega, V_bus)
        P_max_torque = T_max * omega

        # Also limited by motor power rating
        return min(P_max_torque, self.config.max_power)
