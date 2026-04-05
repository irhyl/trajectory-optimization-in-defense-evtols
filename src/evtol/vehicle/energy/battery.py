"""
Electrochemical Battery Model - Equivalent Circuit with Thermal Coupling

This module implements a physics-based battery model suitable for
high-fidelity eVTOL simulation with defense mission profiles.

Model Architecture
==================

The battery is modeled using a 2-RC equivalent circuit:

    V_terminal = V_OC(SOC,T) - I·R0(T) - V_RC1 - V_RC2

Where:
    V_OC: Open-circuit voltage (function of SOC and temperature)
    R0: Ohmic resistance (temperature-dependent)
    V_RC1, V_RC2: Polarization voltages from RC pairs

The RC pairs capture:
    - RC1: Fast dynamics (charge transfer, ~1s time constant)
    - RC2: Slow dynamics (diffusion, ~100s time constant)

SOC Dynamics
============

    dSOC/dt = -η_c(I) · I / Q_nom

Where:
    η_c: Coulombic efficiency (function of current direction and magnitude)
    Q_nom: Nominal capacity (Ah)

Thermal Dynamics
================

    m·c_p·dT/dt = I²·R_total + Q_reversible - Q_cooling

Where:
    Q_reversible = I·T·∂V_OC/∂T (entropic heating/cooling)
    Q_cooling: Heat rejection to ambient

State of Health (SOH)
=====================

    dSOH/dt = -f(DOD, C_rate, T)

Capacity fade from:
    - Cycle aging (depth of discharge)
    - Calendar aging (storage temperature)
    - Rate-dependent degradation (high C-rates)

Chemistry: Li-ion NMC (high energy density for aviation)
Pack Configuration: 400V nominal, 100+ Ah capacity
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from scipy.interpolate import interp1d
import logging

logger = logging.getLogger(__name__)


@dataclass
class BatteryConfig:
    """Battery pack configuration."""
    # Chemistry
    chemistry: str = "NMC_622"  # LiNi0.6Mn0.2Co0.2O2

    # Pack topology
    cells_series: int = 96      # For ~400V nominal
    cells_parallel: int = 10    # For capacity

    # Cell parameters (NMC 21700 cell baseline)
    cell_capacity_ah: float = 5.0         # Ah per cell
    cell_voltage_nominal: float = 3.7     # V
    cell_voltage_max: float = 4.2         # V
    cell_voltage_min: float = 2.8         # V
    cell_mass_kg: float = 0.070           # kg per cell

    # Pack-level (computed from topology)
    @property
    def pack_voltage_nominal(self) -> float:
        return self.cells_series * self.cell_voltage_nominal

    @property
    def pack_voltage_max(self) -> float:
        return self.cells_series * self.cell_voltage_max

    @property
    def pack_voltage_min(self) -> float:
        return self.cells_series * self.cell_voltage_min

    @property
    def pack_capacity_ah(self) -> float:
        return self.cells_parallel * self.cell_capacity_ah

    @property
    def pack_capacity_wh(self) -> float:
        return self.pack_capacity_ah * self.pack_voltage_nominal

    @property
    def pack_mass_kg(self) -> float:
        cells = self.cells_series * self.cells_parallel
        cell_mass = cells * self.cell_mass_kg
        # Add 30% for pack structure, BMS, cooling
        return cell_mass * 1.3

    @property
    def specific_energy_wh_kg(self) -> float:
        return self.pack_capacity_wh / self.pack_mass_kg

    # Resistance parameters (per cell, at 25°C)
    R0_cell_ohm: float = 0.020           # Ohmic resistance
    R1_cell_ohm: float = 0.015           # RC1 resistance
    C1_cell_farad: float = 1000.0        # RC1 capacitance (~15s time constant)
    R2_cell_ohm: float = 0.010           # RC2 resistance
    C2_cell_farad: float = 10000.0       # RC2 capacitance (~100s time constant)

    # Pack resistance (cells in series add, parallel divide)
    @property
    def R0_pack_ohm(self) -> float:
        return self.R0_cell_ohm * self.cells_series / self.cells_parallel

    @property
    def R1_pack_ohm(self) -> float:
        return self.R1_cell_ohm * self.cells_series / self.cells_parallel

    @property
    def R2_pack_ohm(self) -> float:
        return self.R2_cell_ohm * self.cells_series / self.cells_parallel

    # Thermal parameters
    cell_specific_heat_j_kg_k: float = 1000.0  # J/kg·K
    thermal_resistance_k_w: float = 2.0        # K/W to ambient

    # Operating limits
    max_charge_c_rate: float = 2.0       # 2C max charge
    max_discharge_c_rate: float = 5.0    # 5C max discharge (high power bursts)
    continuous_c_rate: float = 2.0       # Continuous 2C

    min_soc: float = 0.10                # 10% minimum SOC
    max_soc: float = 0.95                # 95% maximum SOC

    min_temp_c: float = -20.0            # Minimum operating temp
    max_temp_c: float = 55.0             # Maximum operating temp
    optimal_temp_c: float = 25.0         # Optimal temperature

    # Degradation parameters
    cycle_degradation_per_fec: float = 0.0001  # SOH loss per full equivalent cycle
    calendar_degradation_per_day: float = 0.00001  # SOH loss per day at 25°C


@dataclass
class BatteryState:
    """Battery pack state."""
    # Charge state
    soc: float = 0.8                 # State of charge (0-1)
    soh: float = 1.0                 # State of health (0-1)

    # Voltages
    voltage_oc: float = 380.0        # Open circuit voltage (V)
    voltage_terminal: float = 380.0  # Terminal voltage (V)
    voltage_rc1: float = 0.0         # RC1 polarization voltage (V)
    voltage_rc2: float = 0.0         # RC2 polarization voltage (V)

    # Current and power
    current: float = 0.0             # Pack current (A, positive = discharge)
    power: float = 0.0               # Power (W, positive = discharge)
    c_rate: float = 0.0              # C-rate (1/h)

    # Thermal
    temperature: float = 25.0        # Average cell temperature (°C)
    heat_generation: float = 0.0     # Heat generation rate (W)

    # Energy tracking
    energy_discharged_wh: float = 0.0   # Total energy discharged (Wh)
    energy_charged_wh: float = 0.0      # Total energy charged (Wh)
    ah_throughput: float = 0.0          # Total Ah throughput (for aging)

    # Limits
    max_discharge_power: float = 200000.0  # Max discharge power (W)
    max_charge_power: float = 100000.0     # Max charge power (W)


class BatteryModel:
    """
    2-RC Equivalent Circuit Battery Model with Thermal Coupling.

    This model provides:
    - Accurate voltage prediction under dynamic loads
    - Temperature-dependent parameters
    - SOC estimation with coulomb counting
    - SOH tracking for mission planning
    - Power limits based on state

    Suitable for:
    - 6-DoF vehicle simulation
    - Energy-constrained trajectory optimization
    - Mission feasibility analysis

    Reference:
    - Plett, G. "Battery Management Systems, Volume I: Battery Modeling"
    """

    def __init__(self, config: BatteryConfig | None = None):
        """
        Initialize battery model.

        Args:
            config: Battery configuration (default: aviation-grade NMC pack)
        """
        self.config = config or BatteryConfig()
        self.state = BatteryState()

        # Initialize OCV-SOC lookup table
        self._setup_ocv_soc_table()

        # Initialize temperature-dependent resistance factors
        self._setup_temperature_factors()

        # Set initial state
        self._update_ocv()
        self.state.voltage_terminal = self.state.voltage_oc
        self._update_power_limits()

        logger.info(
            f"Battery model initialized: {self.config.pack_capacity_wh/1000:.1f} kWh, "
            f"{self.config.pack_voltage_nominal:.0f}V nominal, "
            f"{self.config.specific_energy_wh_kg:.0f} Wh/kg"
        )

    def _setup_ocv_soc_table(self) -> None:
        """Set up OCV-SOC relationship for NMC chemistry."""
        # Typical NMC OCV curve (per cell, 3.0-4.2V range)
        soc_points = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        ocv_cell = np.array([3.00, 3.40, 3.55, 3.62, 3.68, 3.73, 3.80, 3.88, 3.98, 4.10, 4.20])

        # Scale to pack voltage
        ocv_pack = ocv_cell * self.config.cells_series

        self._ocv_interp = interp1d(
            soc_points, ocv_pack,
            kind='cubic',
            bounds_error=False,
            fill_value=(ocv_pack[0], ocv_pack[-1])
        )

        # Store for derivative calculation (entropic coefficient)
        self._soc_points = soc_points
        self._ocv_pack = ocv_pack

    def _setup_temperature_factors(self) -> None:
        """Set up temperature-dependent resistance factors."""
        # Resistance increases at low temperatures, decreases at high
        # Arrhenius-type relationship
        self._temp_ref = 25.0  # Reference temperature (°C)
        self._activation_energy = 20000.0  # J/mol (typical for Li-ion)
        self._gas_constant = 8.314  # J/mol·K

    def _get_resistance_factor(self, temperature: float) -> float:
        """
        Get temperature-dependent resistance multiplier.

        Uses Arrhenius equation:
            R(T) = R_ref × exp(Ea/R × (1/T - 1/T_ref))

        Args:
            temperature: Cell temperature (°C)

        Returns:
            Resistance multiplier (1.0 at 25°C)
        """
        T = temperature + 273.15  # Convert to Kelvin
        T_ref = self._temp_ref + 273.15

        factor = np.exp(
            self._activation_energy / self._gas_constant *
            (1/T - 1/T_ref)
        )

        # Limit to reasonable range
        return np.clip(factor, 0.5, 5.0)

    def _update_ocv(self) -> None:
        """Update open-circuit voltage based on SOC."""
        self.state.voltage_oc = float(self._ocv_interp(self.state.soc))

    def _update_power_limits(self) -> None:
        """Update power limits based on current state."""
        # C-rate limits
        max_discharge_current = (
            self.config.max_discharge_c_rate * self.config.pack_capacity_ah
        )
        max_charge_current = (
            self.config.max_charge_c_rate * self.config.pack_capacity_ah
        )

        # Temperature derating
        T = self.state.temperature
        if T < 0:
            # Severely derate at low temps
            temp_factor = 0.3 + 0.7 * (T + 20) / 20
        elif T > 45:
            # Derate at high temps
            temp_factor = 1.0 - 0.5 * (T - 45) / 10
        else:
            temp_factor = 1.0
        temp_factor = np.clip(temp_factor, 0.1, 1.0)

        # SOC limits
        soc_discharge_factor = min(1.0, (self.state.soc - self.config.min_soc) / 0.1)
        soc_charge_factor = min(1.0, (self.config.max_soc - self.state.soc) / 0.1)

        # SOH derating
        soh_factor = self.state.soh

        # Compute limits
        max_discharge_current *= temp_factor * soc_discharge_factor * soh_factor
        max_charge_current *= temp_factor * soc_charge_factor * soh_factor

        # Convert to power using terminal voltage estimate
        self.state.max_discharge_power = max_discharge_current * self.state.voltage_terminal
        self.state.max_charge_power = max_charge_current * self.state.voltage_terminal

    def _compute_heat_generation(self, current: float) -> float:
        """
        Compute heat generation rate.

        Heat sources:
        1. Ohmic heating: I²·R_total
        2. Reversible heating: I·T·(∂V_OC/∂T)

        Args:
            current: Pack current (A)

        Returns:
            Heat generation rate (W)
        """
        # Resistance factor
        r_factor = self._get_resistance_factor(self.state.temperature)

        # Total resistance
        R_total = r_factor * (
            self.config.R0_pack_ohm +
            self.config.R1_pack_ohm +
            self.config.R2_pack_ohm
        )

        # Ohmic heating (always positive)
        Q_ohmic = current**2 * R_total

        # Reversible heating (entropic)
        # ∂V_OC/∂T ≈ -0.0003 to -0.0005 V/K for NMC
        dVdT = -0.0004 * self.config.cells_series  # Pack level
        T_kelvin = self.state.temperature + 273.15
        Q_reversible = current * T_kelvin * dVdT  # Negative during discharge

        return Q_ohmic + Q_reversible

    def step(self, power_demand: float, dt: float, ambient_temp: float = 25.0) -> BatteryState:
        """
        Advance battery state by one time step.

        Args:
            power_demand: Power demand (W, positive = discharge)
            dt: Time step (s)
            ambient_temp: Ambient temperature (°C)

        Returns:
            Updated battery state
        """
        # Estimate current from power demand
        # P = V·I, but V depends on I, so iterate
        current = self._solve_current(power_demand)

        # Resistance temperature factor
        r_factor = self._get_resistance_factor(self.state.temperature)

        # Update RC polarization voltages (first-order dynamics)
        R1 = self.config.R1_pack_ohm * r_factor
        C1 = self.config.C1_cell_farad / self.config.cells_series * self.config.cells_parallel
        tau1 = R1 * C1

        R2 = self.config.R2_pack_ohm * r_factor
        C2 = self.config.C2_cell_farad / self.config.cells_series * self.config.cells_parallel
        tau2 = R2 * C2

        # Exponential decay + current contribution
        self.state.voltage_rc1 = (
            self.state.voltage_rc1 * np.exp(-dt/tau1) +
            R1 * current * (1 - np.exp(-dt/tau1))
        )
        self.state.voltage_rc2 = (
            self.state.voltage_rc2 * np.exp(-dt/tau2) +
            R2 * current * (1 - np.exp(-dt/tau2))
        )

        # Update SOC (coulomb counting)
        coulombic_efficiency = 0.995 if current > 0 else 1.0  # Slightly less efficient discharge
        delta_ah = current * dt / 3600.0  # A·s to Ah
        delta_soc = delta_ah / (self.config.pack_capacity_ah * self.state.soh) * coulombic_efficiency

        self.state.soc = np.clip(
            self.state.soc - delta_soc,
            self.config.min_soc,
            self.config.max_soc
        )

        # Update OCV for new SOC
        self._update_ocv()

        # Compute terminal voltage
        R0 = self.config.R0_pack_ohm * r_factor
        self.state.voltage_terminal = (
            self.state.voltage_oc -
            current * R0 -
            self.state.voltage_rc1 -
            self.state.voltage_rc2
        )

        # Clamp terminal voltage
        self.state.voltage_terminal = np.clip(
            self.state.voltage_terminal,
            self.config.pack_voltage_min,
            self.config.pack_voltage_max
        )

        # Update thermal state
        self.state.heat_generation = self._compute_heat_generation(current)

        # Simple thermal model: T_dot = (Q_gen - Q_cooling) / (m·c_p)
        Q_cooling = (self.state.temperature - ambient_temp) / self.config.thermal_resistance_k_w
        dT_dt = (self.state.heat_generation - Q_cooling) / (
            self.config.pack_mass_kg * self.config.cell_specific_heat_j_kg_k
        )
        self.state.temperature += dT_dt * dt

        # Update degradation
        self._update_soh(current, dt)

        # Track energy
        power_actual = current * self.state.voltage_terminal
        if current > 0:
            self.state.energy_discharged_wh += power_actual * dt / 3600.0
        else:
            self.state.energy_charged_wh += abs(power_actual) * dt / 3600.0

        self.state.ah_throughput += abs(delta_ah)

        # Update current state
        self.state.current = current
        self.state.power = power_actual
        self.state.c_rate = abs(current) / self.config.pack_capacity_ah

        # Update power limits for next step
        self._update_power_limits()

        return self.state

    def _solve_current(self, power_demand: float) -> float:
        """
        Solve for current given power demand.

        Power equation: P = V_t · I = (V_OC - I·R_total) · I
        Quadratic: R·I² - V_OC·I + P = 0

        Args:
            power_demand: Requested power (W)

        Returns:
            Current (A)
        """
        r_factor = self._get_resistance_factor(self.state.temperature)
        R_total = r_factor * (
            self.config.R0_pack_ohm +
            self.config.R1_pack_ohm +
            self.config.R2_pack_ohm
        )

        V_oc = self.state.voltage_oc

        # Quadratic formula: I = (V_OC ± sqrt(V_OC² - 4·R·P)) / (2·R)
        discriminant = V_oc**2 - 4 * R_total * power_demand

        if discriminant < 0:
            # Power demand exceeds capability
            # Return maximum possible current
            if power_demand > 0:
                return V_oc / (2 * R_total)  # Max discharge current
            else:
                return -V_oc / (2 * R_total)  # Max charge current

        # Take the smaller root (less voltage drop)
        if power_demand >= 0:
            current = (V_oc - np.sqrt(discriminant)) / (2 * R_total)
        else:
            current = (V_oc + np.sqrt(discriminant)) / (2 * R_total)

        # Apply limits
        max_discharge = self.config.max_discharge_c_rate * self.config.pack_capacity_ah
        max_charge = self.config.max_charge_c_rate * self.config.pack_capacity_ah

        return np.clip(current, -max_charge, max_discharge)

    def _update_soh(self, current: float, dt: float) -> None:
        """
        Update state of health based on aging.

        Args:
            current: Pack current (A)
            dt: Time step (s)
        """
        # Cycle aging (based on Ah throughput)
        delta_ah = abs(current) * dt / 3600.0
        fec = delta_ah / (2 * self.config.pack_capacity_ah)  # Full equivalent cycles
        cycle_degradation = fec * self.config.cycle_degradation_per_fec

        # Temperature-dependent calendar aging
        T_factor = np.exp((self.state.temperature - 25) / 10)  # Arrhenius-type
        calendar_degradation = self.config.calendar_degradation_per_day * (dt / 86400) * T_factor

        # C-rate stress (high rates accelerate aging)
        c_rate = abs(current) / self.config.pack_capacity_ah
        rate_stress_factor = 1.0 + 0.5 * max(0, c_rate - 1.0)

        # Total degradation
        total_degradation = (cycle_degradation * rate_stress_factor + calendar_degradation)

        self.state.soh = max(0.0, self.state.soh - total_degradation)

    def get_remaining_energy_wh(self) -> float:
        """Get remaining usable energy in Wh."""
        usable_soc = self.state.soc - self.config.min_soc
        return usable_soc * self.config.pack_capacity_wh * self.state.soh

    def get_remaining_range_km(self, power_cruise_kw: float = 50.0) -> float:
        """
        Estimate remaining range at given cruise power.

        Args:
            power_cruise_kw: Cruise power consumption (kW)

        Returns:
            Estimated remaining range (km)
        """
        remaining_wh = self.get_remaining_energy_wh()
        endurance_hours = remaining_wh / (power_cruise_kw * 1000)

        # Assume 200 km/h cruise speed for tiltrotor
        cruise_speed_kmh = 200.0

        return endurance_hours * cruise_speed_kmh

    def get_endurance_seconds(self, power_w: float) -> float:
        """
        Get remaining endurance at given power level.

        Args:
            power_w: Power consumption (W)

        Returns:
            Remaining endurance (seconds)
        """
        remaining_wh = self.get_remaining_energy_wh()
        if power_w <= 0:
            return float('inf')
        return remaining_wh * 3600 / power_w

    def can_complete_mission(
        self,
        energy_required_wh: float,
        reserve_fraction: float = 0.2,
    ) -> tuple[bool, float]:
        """
        Check if battery can complete mission.

        Args:
            energy_required_wh: Energy required for mission (Wh)
            reserve_fraction: Reserve energy fraction

        Returns:
            (can_complete, margin) where margin is energy margin ratio
        """
        available = self.get_remaining_energy_wh()
        required_with_reserve = energy_required_wh * (1 + reserve_fraction)

        margin = available / required_with_reserve if required_with_reserve > 0 else float('inf')

        return margin >= 1.0, margin

    def reset(self, soc: float = 0.8, soh: float = 1.0, temperature: float = 25.0) -> None:
        """
        Reset battery to initial state.

        Args:
            soc: Initial state of charge
            soh: Initial state of health
            temperature: Initial temperature (°C)
        """
        self.state = BatteryState(
            soc=soc,
            soh=soh,
            temperature=temperature,
        )
        self._update_ocv()
        self.state.voltage_terminal = self.state.voltage_oc
        self._update_power_limits()

    def get_state(self) -> BatteryState:
        """Get current battery state."""
        return self.state

    def to_dict(self) -> dict:
        """Convert state to dictionary."""
        return {
            'soc': self.state.soc,
            'soh': self.state.soh,
            'voltage_oc': self.state.voltage_oc,
            'voltage_terminal': self.state.voltage_terminal,
            'current': self.state.current,
            'power': self.state.power,
            'c_rate': self.state.c_rate,
            'temperature': self.state.temperature,
            'heat_generation': self.state.heat_generation,
            'energy_discharged_wh': self.state.energy_discharged_wh,
            'remaining_energy_wh': self.get_remaining_energy_wh(),
            'max_discharge_power': self.state.max_discharge_power,
            'max_charge_power': self.state.max_charge_power,
        }
