"""
Thermal Management Model - Battery and Motor Cooling

This module handles thermal dynamics for the eVTOL powertrain,
including battery pack and motor thermal management.

Thermal Architecture
====================

For defense eVTOL, thermal management is critical for:
1. Battery performance (capacity/power derating at temperature extremes)
2. Motor performance (torque limits at high temperature)
3. Signature management (IR emissions)

Model Components:
    - Battery thermal mass and cooling
    - Motor thermal mass and cooling
    - Ambient conditions (altitude, speed effects)
    - Active cooling system (liquid cooling loop)

Heat Transfer Modes:
    Q_convection = h·A·(T_surface - T_ambient)
    Q_conduction = k·A·ΔT/Δx
    Q_radiation = ε·σ·A·(T⁴ - T_ambient⁴)

For liquid cooling:
    Q_coolant = ṁ·c_p·(T_out - T_in)
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class ThermalState:
    """Thermal system state."""
    # Component temperatures (°C)
    battery_temp: float = 25.0
    motor_temps: np.ndarray = field(default_factory=lambda: np.array([25.0, 25.0]))  # L/R motors
    inverter_temp: float = 25.0
    coolant_temp: float = 25.0

    # Heat flows (W)
    battery_heat: float = 0.0
    motor_heat: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0]))
    inverter_heat: float = 0.0
    cooling_power: float = 0.0  # Power removed by cooling system

    # Cooling system state
    coolant_flow_rate: float = 0.0   # kg/s
    cooling_pump_power: float = 0.0  # W

    # Ambient
    ambient_temp: float = 25.0
    altitude_m: float = 0.0
    airspeed_mps: float = 0.0

    # IR signature contribution
    ir_signature_factor: float = 1.0  # Relative to reference


@dataclass
class ThermalConfig:
    """Thermal system configuration."""
    # Battery thermal
    battery_mass_kg: float = 200.0
    battery_specific_heat: float = 1000.0  # J/kg·K
    battery_surface_area: float = 2.0      # m²
    battery_h_natural: float = 10.0        # W/m²·K natural convection

    # Motor thermal (per motor)
    motor_mass_kg: float = 15.0
    motor_specific_heat: float = 500.0     # J/kg·K (copper/steel)
    motor_surface_area: float = 0.3        # m²
    motor_max_temp: float = 150.0          # °C
    motor_h_natural: float = 25.0          # W/m²·K

    # Inverter thermal
    inverter_mass_kg: float = 5.0
    inverter_specific_heat: float = 400.0  # J/kg·K
    inverter_max_temp: float = 85.0        # °C

    # Liquid cooling system
    coolant_specific_heat: float = 3800.0  # J/kg·K (water-glycol)
    coolant_density: float = 1050.0        # kg/m³
    coolant_volume_liters: float = 10.0    # Liters
    max_flow_rate: float = 0.5             # kg/s
    radiator_effectiveness: float = 0.7
    radiator_area: float = 0.5             # m²
    pump_power_per_flow: float = 100.0     # W per kg/s

    # Forced convection (from airspeed)
    h_forced_per_mps: float = 5.0  # Additional W/m²·K per m/s

    # IR signature
    emissivity_battery: float = 0.9
    emissivity_motor: float = 0.3     # Metallic surfaces
    emissivity_exhaust: float = 0.8   # Cooling outlet


class ThermalModel:
    """
    Thermal management system model.

    This model tracks temperatures of major powertrain components
    and manages the liquid cooling system to maintain safe operating
    temperatures.

    Features:
    - Coupled thermal dynamics (battery, motors, inverters)
    - Active liquid cooling with pump control
    - Altitude and airspeed effects on cooling
    - IR signature estimation for stealth planning
    """

    def __init__(self, config: ThermalConfig | None = None, num_motors: int = 2):
        """
        Initialize thermal model.

        Args:
            config: Thermal configuration
            num_motors: Number of motors (default: 2 for tiltrotor)
        """
        self.config = config or ThermalConfig()
        self.num_motors = num_motors

        self.state = ThermalState(
            motor_temps=np.full(num_motors, 25.0),
            motor_heat=np.zeros(num_motors),
        )

        # Coolant thermal mass
        self.coolant_mass = (
            self.config.coolant_density *
            self.config.coolant_volume_liters / 1000
        )

        logger.info(f"Thermal model initialized: {num_motors} motors")

    def _get_convection_coefficient(self, surface_area: float, h_natural: float) -> float:
        """
        Get effective convection coefficient including forced convection.

        Args:
            surface_area: Surface area (m²)
            h_natural: Natural convection coefficient (W/m²·K)

        Returns:
            Effective h (W/m²·K)
        """
        # Forced convection from airspeed
        h_forced = self.config.h_forced_per_mps * self.state.airspeed_mps

        # Altitude effect (reduced air density reduces cooling)
        rho_ratio = np.exp(-self.state.altitude_m / 8500.0)

        # Combined (natural + forced, scaled by density)
        h_effective = (h_natural + h_forced) * np.sqrt(rho_ratio)

        return h_effective

    def _compute_radiator_cooling(self, coolant_temp: float) -> float:
        """
        Compute heat rejected by radiator.

        Args:
            coolant_temp: Coolant temperature (°C)

        Returns:
            Heat rejected (W)
        """
        # Airspeed-dependent cooling
        if self.state.airspeed_mps < 1.0:
            # No airflow - reduced effectiveness
            effectiveness = self.config.radiator_effectiveness * 0.2
        else:
            effectiveness = self.config.radiator_effectiveness

        # Temperature difference
        delta_T = coolant_temp - self.state.ambient_temp

        # Heat transfer (simplified effectiveness-NTU)
        h_rad = 50.0 + 10.0 * self.state.airspeed_mps  # W/m²·K

        Q_max = h_rad * self.config.radiator_area * delta_T
        Q_actual = effectiveness * Q_max

        return max(0, Q_actual)  # Only cooling, no heating

    def step(
        self,
        battery_heat: float,
        motor_heats: np.ndarray,
        inverter_heat: float,
        dt: float,
        ambient_temp: float = 25.0,
        altitude_m: float = 0.0,
        airspeed_mps: float = 0.0,
    ) -> ThermalState:
        """
        Advance thermal state by one time step.

        Args:
            battery_heat: Battery heat generation (W)
            motor_heats: Motor heat generation per motor (W)
            inverter_heat: Inverter heat generation (W)
            dt: Time step (s)
            ambient_temp: Ambient temperature (°C)
            altitude_m: Altitude (m)
            airspeed_mps: Airspeed (m/s)

        Returns:
            Updated thermal state
        """
        # Update ambient conditions
        self.state.ambient_temp = ambient_temp
        self.state.altitude_m = altitude_m
        self.state.airspeed_mps = airspeed_mps
        self.state.battery_heat = battery_heat
        self.state.motor_heat = motor_heats
        self.state.inverter_heat = inverter_heat

        # Determine cooling pump flow rate (simple proportional control)
        # Increase flow when components are hot
        max_component_temp = max(
            self.state.battery_temp,
            np.max(self.state.motor_temps),
            self.state.inverter_temp
        )

        # Target 40°C for components
        temp_error = max_component_temp - 40.0
        flow_fraction = np.clip(0.2 + 0.08 * temp_error, 0.1, 1.0)
        self.state.coolant_flow_rate = flow_fraction * self.config.max_flow_rate

        # Pump power
        self.state.cooling_pump_power = (
            self.state.coolant_flow_rate * self.config.pump_power_per_flow
        )

        # ===== Battery thermal dynamics =====
        h_batt = self._get_convection_coefficient(
            self.config.battery_surface_area,
            self.config.battery_h_natural
        )

        # Heat to coolant (liquid cooling)
        Q_batt_coolant = (
            self.state.coolant_flow_rate *
            self.config.coolant_specific_heat *
            max(0, self.state.battery_temp - self.state.coolant_temp) * 0.5
        )

        # Heat to ambient (convection)
        Q_batt_ambient = (
            h_batt * self.config.battery_surface_area *
            (self.state.battery_temp - ambient_temp)
        )

        # Net heat accumulation
        Q_batt_net = battery_heat - Q_batt_coolant - Q_batt_ambient
        dT_batt = Q_batt_net / (self.config.battery_mass_kg * self.config.battery_specific_heat)
        self.state.battery_temp += dT_batt * dt

        # ===== Motor thermal dynamics =====
        h_motor = self._get_convection_coefficient(
            self.config.motor_surface_area,
            self.config.motor_h_natural
        )

        for i in range(self.num_motors):
            # Heat to coolant
            Q_motor_coolant = (
                self.state.coolant_flow_rate / self.num_motors *
                self.config.coolant_specific_heat *
                max(0, self.state.motor_temps[i] - self.state.coolant_temp) * 0.3
            )

            # Heat to ambient
            Q_motor_ambient = (
                h_motor * self.config.motor_surface_area *
                (self.state.motor_temps[i] - ambient_temp)
            )

            Q_motor_net = motor_heats[i] - Q_motor_coolant - Q_motor_ambient
            dT_motor = Q_motor_net / (self.config.motor_mass_kg * self.config.motor_specific_heat)
            self.state.motor_temps[i] += dT_motor * dt

        # ===== Inverter thermal dynamics =====
        Q_inv_coolant = (
            self.state.coolant_flow_rate *
            self.config.coolant_specific_heat *
            max(0, self.state.inverter_temp - self.state.coolant_temp) * 0.4
        )

        Q_inv_net = inverter_heat - Q_inv_coolant
        dT_inv = Q_inv_net / (self.config.inverter_mass_kg * self.config.inverter_specific_heat)
        self.state.inverter_temp += dT_inv * dt

        # ===== Coolant dynamics =====
        # Heat absorbed from components
        Q_absorbed = Q_batt_coolant + np.sum([
            self.state.coolant_flow_rate / self.num_motors *
            self.config.coolant_specific_heat *
            max(0, self.state.motor_temps[i] - self.state.coolant_temp) * 0.3
            for i in range(self.num_motors)
        ]) + Q_inv_coolant

        # Heat rejected by radiator
        Q_radiator = self._compute_radiator_cooling(self.state.coolant_temp)
        self.state.cooling_power = Q_radiator

        # Net coolant temperature change
        Q_coolant_net = Q_absorbed - Q_radiator
        dT_coolant = Q_coolant_net / (self.coolant_mass * self.config.coolant_specific_heat)
        self.state.coolant_temp += dT_coolant * dt

        # ===== IR Signature =====
        self._update_ir_signature()

        return self.state

    def _update_ir_signature(self) -> None:
        """Update IR signature factor based on temperatures."""
        # Stefan-Boltzmann: P ∝ ε·σ·T⁴
        # Compare to reference at 25°C
        T_ref = 298.15  # 25°C in Kelvin

        # Weighted contribution from components
        # Motors are most visible (external)
        T_batt_K = self.state.battery_temp + 273.15
        T_motor_K = np.mean(self.state.motor_temps) + 273.15
        T_coolant_K = self.state.coolant_temp + 273.15  # Radiator exhaust

        ir_batt = self.config.emissivity_battery * (T_batt_K / T_ref)**4
        ir_motor = self.config.emissivity_motor * (T_motor_K / T_ref)**4
        ir_exhaust = self.config.emissivity_exhaust * (T_coolant_K / T_ref)**4

        # Weighted average (motors dominate external signature)
        self.state.ir_signature_factor = 0.2 * ir_batt + 0.5 * ir_motor + 0.3 * ir_exhaust

    def get_motor_torque_limit_factor(self) -> np.ndarray:
        """
        Get motor torque derating factor due to temperature.

        Returns:
            Torque limit factor per motor (0-1)
        """
        factors = np.ones(self.num_motors)

        for i, T in enumerate(self.state.motor_temps):
            if T > self.config.motor_max_temp:
                # Severe derating above max temp
                factors[i] = 0.0
            elif T > self.config.motor_max_temp - 20:
                # Linear derating from 20°C below max
                factors[i] = (self.config.motor_max_temp - T) / 20.0

        return factors

    def is_thermal_limited(self) -> tuple[bool, str]:
        """
        Check if any component is thermally limited.

        Returns:
            (is_limited, reason)
        """
        if np.any(self.state.motor_temps > self.config.motor_max_temp):
            idx = np.argmax(self.state.motor_temps)
            return True, f"Motor {idx} overheated ({self.state.motor_temps[idx]:.1f}°C)"

        if self.state.inverter_temp > self.config.inverter_max_temp:
            return True, f"Inverter overheated ({self.state.inverter_temp:.1f}°C)"

        if self.state.battery_temp > 55:
            return True, f"Battery overheated ({self.state.battery_temp:.1f}°C)"

        if self.state.battery_temp < -20:
            return True, f"Battery too cold ({self.state.battery_temp:.1f}°C)"

        return False, "Normal"

    def get_state(self) -> ThermalState:
        """Get current thermal state."""
        return self.state

    def reset(self, temperature: float = 25.0) -> None:
        """Reset all temperatures to given value."""
        self.state = ThermalState(
            battery_temp=temperature,
            motor_temps=np.full(self.num_motors, temperature),
            inverter_temp=temperature,
            coolant_temp=temperature,
            ambient_temp=temperature,
        )

    def to_dict(self) -> dict:
        """Convert state to dictionary."""
        return {
            'battery_temp': self.state.battery_temp,
            'motor_temps': self.state.motor_temps.tolist(),
            'inverter_temp': self.state.inverter_temp,
            'coolant_temp': self.state.coolant_temp,
            'cooling_power': self.state.cooling_power,
            'pump_power': self.state.cooling_pump_power,
            'ir_signature_factor': self.state.ir_signature_factor,
            'ambient_temp': self.state.ambient_temp,
        }
