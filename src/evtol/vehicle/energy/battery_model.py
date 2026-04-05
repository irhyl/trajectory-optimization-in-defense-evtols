"""
Battery and Energy Management System for eVTOL

This module implements realistic battery modeling with:
1. Voltage sag under load (voltage × current nonlinearity)
2. Thermal effects (temperature-dependent internal resistance)
3. Battery state of charge (SOC) tracking with cycle counting
4. Power budget and energy reserve monitoring
5. Thermal management integration

Theory
======

Battery Electrochemistry (Lithium-Ion):

Terminal Voltage Model:
    V_terminal = E₀(SOC) - I·R_internal(T, I) - ΔV_concentration

where:
- E₀(SOC): Open-circuit voltage (SOC-dependent)
- R_internal(T, I): Temperature and current-dependent internal resistance
- ΔV_concentration: Concentration polarization (rate-dependent voltage drop)

Typical LiPo Cell (3.7V nominal):
- Fully charged: 4.2V (100% SOC)
- Nominal: 3.7V (50% SOC)
- Depleted: 2.8V (0% SOC)

Internal Resistance Model:
    R(T, I) = R₀ + α·(T - T_ref)  [temperature dependent]
    
    Where R₀ also increases with aging and cycling.

Power Dissipation:
    P_loss = I²·R_internal  [heat generation in cell]
    
Thermal Model:
    C_thermal·dT/dt = P_loss - h·(T - T_ambient)
    
    Steady-state: ΔT = P_loss / h

State of Charge (SOC) Tracking:
    SOC(t) = SOC(0) - ∫ I(τ)/Q_capacity dτ
    
    Coulomb counting: Ampere-hour integration
    
Depth of Discharge (DoD):
    DoD = 100% - SOC  [percentage]
    
Cycle Life Model:
    N_cycles(DoD) ≈ A / (DoD^B)  where A, B from datasheet
    
    Example: OEM may spec 1000 cycles @ 100% DoD, → 5000 cycles @ 50% DoD

References
==========

[1] Chen, M., Rincon-Mora, G.A. (2006). "Accurate Electrical Battery Model
    Capable of Predicting Runtime and I-V Performance." IEEE Trans. Energy
    Conversion, vol. 21, no. 2.

[2] Plett, G.L. (2015). Battery Management Systems, Vol. 1: Battery Modeling.
    Artech House. Chapters 2-4 (equivalent circuit models).

[3] Vetter, J., et al. (2005). "Ageing mechanisms in lithium-ion batteries."
    Journal of Power Sources, vol. 147, pp. 269-281.

[4] NASA RP-1257 (1998). Design and Development of the Space Shuttle
    Auxiliary Power Unit. Section 4 (power system architecture).
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class BatteryConfig:
    """
    Battery pack configuration.
    
    Attributes:
        nominal_voltage: Nominal pack voltage [V]
        min_voltage: Cutoff voltage (minimum safe) [V]
        max_voltage: Maximum charge voltage [V]
        capacity_wh: Total energy capacity [Wh]
        capacity_ah: Total charge capacity [Ah]
        internal_resistance: Internal resistance @ 25°C [Ω]
        num_cells: Number of series cells
        num_parallel_modules: Number of parallel modules
        max_continuous_current: Max continuous discharge [A]
        max_peak_current: Max peak current (5-10s) [A]
        thermal_capacity: Thermal mass [J/°C]
        thermal_conductance: Heat dissipation [W/°C]
    """
    nominal_voltage: float = 48.0      # V (12S LiPo)
    min_voltage: float = 40.0          # V (cutoff)
    max_voltage: float = 50.4          # V (fully charged)
    capacity_wh: float = 2400.0        # Wh (energy)
    capacity_ah: float = 50.0          # Ah (charge)
    internal_resistance: float = 0.05  # Ω @ 25°C
    num_cells: int = 12                # Series
    num_parallel_modules: int = 2
    max_continuous_current: float = 200.0  # A
    max_peak_current: float = 300.0    # A
    thermal_capacity: float = 500.0    # J/°C
    thermal_conductance: float = 10.0  # W/°C


@dataclass
class BatteryState:
    """Battery pack operating state."""
    # Electrical state
    voltage: float = 48.0              # Terminal voltage [V]
    voltage_unloaded: float = 48.0     # Open-circuit voltage [V]
    current: float = 0.0               # Discharge current [A]
    power: float = 0.0                 # Power draw [W]
    
    # Energy state
    soc: float = 1.0                   # State of charge [0, 1]
    soh: float = 1.0                   # State of health [0, 1]
    energy_remaining: float = 2400.0   # [Wh]
    coulombs_remaining: float = 50.0   # [Ah]
    
    # Thermal state
    temperature: float = 25.0          # Pack temperature [°C]
    heat_dissipation: float = 0.0      # Heat generation [W]
    
    # Cycle tracking
    cycle_count: float = 0.0           # Equivalent full cycles
    depth_of_discharge: float = 0.0    # Current DoD [%]
    
    # Status
    is_charging: bool = False
    is_discharging: bool = True
    is_healthy: bool = True
    warning_flags: str = ""


class BatteryPack:
    """
    Lithium-ion battery pack model with voltage sag and thermal effects.
    
    Implements:
    - Nonlinear voltage vs SOC with load current
    - Temperature-dependent internal resistance
    - Thermal dynamics with heat dissipation
    - Energy accounting via Coulomb counting
    - Cycle life estimation
    
    Usage:
        battery = BatteryPack(config)
        state = battery.update(current_draw=100, dt=0.01)
        power_available = state.voltage * state.current
        energy_remaining_wh = state.energy_remaining
    """
    
    def __init__(self, config: BatteryConfig, ambient_temp: float = 25.0):
        """
        Initialize battery pack model.
        
        Args:
            config: Battery configuration
            ambient_temp: Ambient temperature [°C]
        """
        self.config = config
        self.T_ambient = ambient_temp
        
        # Initialize state at full charge
        self.state = BatteryState(
            soc=1.0,
            voltage=config.max_voltage,
            energy_remaining=config.capacity_wh,
            temperature=ambient_temp,
        )
        
        # OCV (Open Circuit Voltage) lookup table (SOC → V)
        self._build_ocv_table()
        
        logger.info(
            f"BatteryPack initialized: {config.capacity_wh}Wh, "
            f"{config.nominal_voltage}V nominal, "
            f"R_int={config.internal_resistance}Ω @ 25°C"
        )
    
    def _build_ocv_table(self):
        """
        Build open-circuit voltage vs SOC table.
        
        Typical LiPo curve:
        SOC    100%   90%    80%    70%    50%    30%    20%    10%    0%
        V/cell 4.20   4.15   4.08   4.02   3.80   3.65   3.60   3.50   2.80
        """
        self.soc_lookup = np.array([0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 1.0])
        
        # Voltage per cell (3.7V nominal for LiPo)
        # Scale by number of cells for pack voltage
        v_per_cell = np.array([2.80, 3.50, 3.60, 3.65, 3.80, 4.02, 4.08, 4.15, 4.20])
        self.ocv_lookup = v_per_cell * self.config.num_cells
        
        logger.debug(f"OCV lookup table: {len(self.soc_lookup)} points")
    
    def _get_ocv(self, soc: float) -> float:
        """
        Get open-circuit voltage for given SOC via interpolation.
        
        Args:
            soc: State of charge [0, 1]
        
        Returns:
            Open-circuit voltage [V]
        """
        soc = np.clip(soc, 0.0, 1.0)
        ocv = np.interp(soc, self.soc_lookup, self.ocv_lookup)
        return ocv
    
    def _get_internal_resistance(self, temperature: float, current: float) -> float:
        """
        Get temperature and current-dependent internal resistance.
        
        Model:
        - Base resistance at 25°C: R₀
        - Temperature effect: +0.5% per °C above 25°C
        - High current effect: increases slightly (electrochemical effects)
        
        Args:
            temperature: Cell temperature [°C]
            current: Discharge current [A]
        
        Returns:
            Internal resistance [Ω]
        """
        # Base temperature dependence
        dT = temperature - 25.0
        temp_factor = 1.0 + 0.005 * dT  # 0.5% per °C
        
        R_base = self.config.internal_resistance * temp_factor
        
        # Current-dependent resistance (high current effects)
        # Typically small for well-designed packs
        I_normalized = abs(current) / self.config.max_continuous_current
        current_factor = 1.0 + 0.1 * I_normalized**2  # Quadratic for high current
        
        R_total = R_base * current_factor
        
        return R_total
    
    def update(
        self,
        current_draw: float,
        dt: float = 0.01,
    ) -> BatteryState:
        """
        Update battery state for one time step.
        
        Implements:
        1. Voltage sag calculation (load-dependent)
        2. Thermal dynamics
        3. Energy accounting (Coulomb counting)
        4. SOH degradation
        
        Args:
            current_draw: Current being drawn [A] (positive = discharge)
            dt: Time step [s]
        
        Returns:
            Updated BatteryState
        """
        # Limit current to maximum
        current_limited = np.clip(current_draw, 0.0, self.config.max_peak_current)
        
        # Get open-circuit voltage at current SOC
        ocv = self._get_ocv(self.state.soc)
        
        # Get temperature-dependent internal resistance
        r_int = self._get_internal_resistance(self.state.temperature, current_limited)
        
        # Calculate terminal voltage (voltage sag)
        # V_terminal = OCV - I·R_internal
        voltage_drop = current_limited * r_int
        V_terminal = ocv - voltage_drop
        
        # Clamp to limits
        V_terminal = np.clip(
            V_terminal,
            self.config.min_voltage,
            self.config.max_voltage
        )
        
        # Power calculations
        P_out = V_terminal * current_limited  # Power delivered
        P_loss = current_limited**2 * r_int   # Heat in battery
        
        # Thermal dynamics: C·dT/dt = P_loss - h·(T - T_amb)
        tau_thermal = self.config.thermal_capacity / self.config.thermal_conductance
        T_ss = self.T_ambient + P_loss / self.config.thermal_conductance
        
        dT = (T_ss - self.state.temperature) * (dt / (tau_thermal + dt))
        temp_new = self.state.temperature + dT
        
        # Energy accounting (Coulomb counting)
        # dE = V·I·dt [Joules]
        # dSOC = -I·dt / Q_capacity
        energy_discharged = V_terminal * current_limited * dt / 3600  # Wh (convert from J)
        charge_discharged = current_limited * dt / 3600  # Ah
        
        soc_new = self.state.soc - (charge_discharged / self.config.capacity_ah)
        soc_new = np.clip(soc_new, 0.0, 1.0)
        
        energy_remaining = soc_new * self.config.capacity_wh
        coulombs_remaining = soc_new * self.config.capacity_ah
        
        # Depth of discharge (current)
        dod_current = (1.0 - soc_new) * 100
        
        # SOH degradation from cycling
        # Simplified: 0.1% degradation per full cycle equivalent
        # Accelerated if charged frequently at high DoD
        aging_rate = 0.001 * (1.0 + 2.0 * dod_current / 100)  # Per cycle
        charge_cycles_this_step = charge_discharged / self.config.capacity_ah
        soh_new = self.state.soh - (aging_rate * charge_cycles_this_step)
        soh_new = max(soh_new, 0.80)  # Minimum 80% SOH
        
        # Cycle count (fractional, per full cycle at current DoD)
        cycle_increment = charge_discharged / self.config.capacity_ah
        cycle_count_new = self.state.cycle_count + cycle_increment
        
        # Health warnings
        warning_flags = ""
        if V_terminal < self.config.nominal_voltage * 0.85:
            warning_flags += "LOW_VOLTAGE "
        if temp_new > 60.0:
            warning_flags += "HIGH_TEMP "
        if soc_new < 0.10:
            warning_flags += "LOW_SOC "
        if soh_new < 0.90:
            warning_flags += "DEGRADATION "
        
        # Update state
        self.state.current = current_limited
        self.state.power = P_out
        self.state.voltage = V_terminal
        self.state.voltage_unloaded = ocv
        
        self.state.soc = soc_new
        self.state.soh = soh_new
        self.state.energy_remaining = energy_remaining
        self.state.coulombs_remaining = coulombs_remaining
        
        self.state.temperature = temp_new
        self.state.heat_dissipation = P_loss
        
        self.state.cycle_count = cycle_count_new
        self.state.depth_of_discharge = dod_current
        
        self.state.is_discharging = current_limited > 1.0
        self.state.is_healthy = soc_new > 0.05 and soh_new > 0.80
        self.state.warning_flags = warning_flags
        
        # Check critical conditions
        if V_terminal < self.config.min_voltage or soc_new <= 0.0:
            logger.critical("Battery: Cutoff voltage reached or depleted")
            self.state.is_healthy = False
        
        return self.state
    
    def get_power_available(self) -> float:
        """
        Get instantaneous power available at current voltage.
        
        Returns:
            Power capacity [W] = V × I_max
        """
        return self.state.voltage * self.config.max_continuous_current
    
    def get_energy_remaining(self) -> float:
        """Get remaining energy [Wh]."""
        return self.state.energy_remaining
    
    def get_endurance_estimate(self, avg_power_draw: float) -> float:
        """
        Estimate flight endurance at given constant power draw.
        
        Args:
            avg_power_draw: Average power consumption [W]
        
        Returns:
            Estimated flight time [s]
        """
        if avg_power_draw < 100:  # Prevent division by zero
            return float('inf')
        
        # Simple estimate: E / P, but account for voltage sag
        # More accurate: integrate power over discharge curve
        endurance = self.state.energy_remaining * 3600 / avg_power_draw
        
        return endurance
    
    def estimate_cycle_life_remaining(self) -> float:
        """
        Estimate remaining cycle life based on current SOH.
        
        Datasheet example: 1000 cycles @ 100% DoD
        Assume roughly exponential relationship.
        
        Returns:
            Estimated remaining full cycles
        """
        if self.state.soh < 0.80:
            return 0.0  # Battery considered end-of-life
        
        # Simplified: assume 2000 total cycles available @ 80% SOH
        # Linear degradation assumption
        cycles_per_percent_soh = 20  # 2000 cycles / (100% - 80%)
        remaining_cycles = (self.state.soh - 0.80) * cycles_per_percent_soh
        
        return remaining_cycles
