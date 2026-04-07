"""
Battery Model - Electrochemical and Thermal Modeling

This module implements a comprehensive battery model that includes:
- State of charge (SOC) dynamics
- Temperature effects on performance
- Voltage modeling with internal resistance
- C-rate limitations and efficiency
- Thermal management
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass
from scipy.interpolate import interp1d

try:
    from ..utils.config import VehicleConfig
except ImportError:
    # Fallback for direct imports
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from utils.config import VehicleConfig


@dataclass
class BatteryState:
    """Battery state variables"""
    soc: float  # State of charge (0-1)
    temperature: float  # Temperature in Celsius
    voltage: float  # Terminal voltage in Volts
    current: float  # Current in Amperes
    power: float  # Power in Watts
    internal_resistance: float  # Internal resistance in Ohms


class BatteryModel:
    """
    Comprehensive battery model with electrochemical and thermal effects.
    
    This model includes:
    - SOC dynamics with coulombic efficiency
    - Temperature-dependent capacity and resistance
    - Voltage modeling with open-circuit voltage
    - C-rate limitations and power constraints
    - Thermal dynamics with cooling/heating
    """
    
    def __init__(self, config: VehicleConfig):
        """
        Initialize battery model with configuration.
        
        Args:
            config: Vehicle configuration containing battery parameters
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Battery parameters
        self.chemistry = config.battery.chemistry
        self.capacity_nominal = config.battery.capacity_nominal  # Ah
        self.voltage_nominal = config.battery.voltage_nominal  # V
        self.voltage_range = config.battery.voltage_range  # [V_min, V_max]
        
        # Thermal parameters
        self.mass = config.battery.thermal.mass  # kg
        self.specific_heat = config.battery.thermal.specific_heat  # J/kg·K
        self.thermal_conductivity = config.battery.thermal.thermal_conductivity  # W/m·K
        self.cooling_capacity = config.battery.thermal.cooling_capacity  # W
        
        # Limits
        self.max_discharge_rate = config.battery.limits.max_discharge_rate  # C
        self.max_charge_rate = config.battery.limits.max_charge_rate  # C
        self.min_soc = config.battery.limits.min_soc
        self.max_soc = config.battery.limits.max_soc
        self.min_temperature = config.battery.limits.min_temperature  # °C
        self.max_temperature = config.battery.limits.max_temperature  # °C
        
        # Current state
        self.state = BatteryState(
            soc=0.8,  # Default 80% SOC
            temperature=20.0,  # Default 20°C
            voltage=self.voltage_nominal,  # Will be updated by _calculate_ocv
            current=0.0,
            power=0.0,
            internal_resistance=0.1
        )
        
        # Load battery characteristics from dataset
        self._load_battery_characteristics()
        
        # Update initial voltage based on SOC
        self.state.voltage = self._calculate_ocv(self.state.soc)
        
        self.logger.info(f"Battery model initialized: {self.chemistry}, "
                        f"{self.capacity_nominal}Ah, {self.voltage_nominal}V")
    
    def _load_battery_characteristics(self) -> None:
        """Load battery characteristics from the dataset."""
        # This would load from the battery_specs.csv file
        # For now, we'll use synthetic data based on the chemistry
        
        if self.chemistry == "Li-ion_NMC":
            self._setup_li_ion_nmc_characteristics()
        elif self.chemistry == "Li-S":
            self._setup_li_s_characteristics()
        elif self.chemistry == "SolidState":
            self._setup_solid_state_characteristics()
        else:
            self.logger.warning(f"Unknown battery chemistry: {self.chemistry}")
            self._setup_li_ion_nmc_characteristics()  # Default
    
    def _setup_li_ion_nmc_characteristics(self) -> None:
        """Setup Li-ion NMC battery characteristics."""
        # Temperature-dependent capacity (from dataset)
        self.temp_capacity_data = {
            -20: 0.6,   # 60% capacity at -20°C
            0: 0.8,     # 80% capacity at 0°C
            20: 1.0,    # 100% capacity at 20°C
            40: 0.8     # 80% capacity at 40°C
        }
        
        # C-rate dependent efficiency
        self.c_rate_efficiency_data = {
            0.5: 0.96,  # 96% efficiency at 0.5C
            1.0: 0.95,  # 95% efficiency at 1C
            2.0: 0.93,  # 93% efficiency at 2C
            5.0: 0.87   # 87% efficiency at 5C
        }
        
        # Open circuit voltage vs SOC (scaled to pack voltage)
        # For 400V pack, assuming ~100 cells in series (400V/4V per cell)
        cells_in_series = int(self.voltage_nominal / 4.0)  # ~100 cells for 400V
        self.ocv_soc_data = np.array([
            [0.0, 3.0 * cells_in_series],   # 0% SOC, 300V
            [0.1, 3.2 * cells_in_series],   # 10% SOC, 320V
            [0.2, 3.4 * cells_in_series],   # 20% SOC, 340V
            [0.3, 3.5 * cells_in_series],   # 30% SOC, 350V
            [0.4, 3.6 * cells_in_series],   # 40% SOC, 360V
            [0.5, 3.7 * cells_in_series],   # 50% SOC, 370V
            [0.6, 3.8 * cells_in_series],   # 60% SOC, 380V
            [0.7, 3.9 * cells_in_series],   # 70% SOC, 390V
            [0.8, 4.0 * cells_in_series],   # 80% SOC, 400V
            [0.9, 4.1 * cells_in_series],   # 90% SOC, 410V
            [1.0, 4.2 * cells_in_series]    # 100% SOC, 420V
        ])
        
        # Internal resistance vs SOC and temperature (scaled for pack)
        # Resistance scales with number of cells in series
        cells_in_series = int(self.voltage_nominal / 4.0)
        self.resistance_data = {
            'soc': np.array([0.0, 0.2, 0.5, 0.8, 1.0]),
            'temp': np.array([-20, 0, 20, 40]),
            'resistance': np.array([
                np.array([0.2, 0.15, 0.1, 0.12, 0.2]) * cells_in_series,   # -20°C
                np.array([0.15, 0.1, 0.08, 0.1, 0.15]) * cells_in_series,  # 0°C
                np.array([0.1, 0.08, 0.05, 0.08, 0.1]) * cells_in_series,  # 20°C
                np.array([0.12, 0.1, 0.08, 0.1, 0.12]) * cells_in_series   # 40°C
            ])
        }
    
    def _setup_li_s_characteristics(self) -> None:
        """Setup Li-S battery characteristics."""
        # Li-S has higher energy density but different characteristics
        self.temp_capacity_data = {
            -20: 0.7,   # 70% capacity at -20°C
            0: 0.9,     # 90% capacity at 0°C
            20: 1.0,    # 100% capacity at 20°C
            40: 0.9     # 90% capacity at 40°C
        }
        
        self.c_rate_efficiency_data = {
            0.5: 0.96,  # 96% efficiency at 0.5C
            1.0: 0.95,  # 95% efficiency at 1C
            2.0: 0.93,  # 93% efficiency at 2C
            5.0: 0.87   # 87% efficiency at 5C
        }
        
        # Li-S has different voltage characteristics (scaled to pack voltage)
        cells_in_series = int(self.voltage_nominal / 3.0)  # ~133 cells for 400V
        self.ocv_soc_data = np.array([
            [0.0, 2.0 * cells_in_series],   # 0% SOC, 266V
            [0.1, 2.1 * cells_in_series],   # 10% SOC, 280V
            [0.2, 2.2 * cells_in_series],   # 20% SOC, 293V
            [0.3, 2.3 * cells_in_series],   # 30% SOC, 307V
            [0.4, 2.4 * cells_in_series],   # 40% SOC, 320V
            [0.5, 2.5 * cells_in_series],   # 50% SOC, 333V
            [0.6, 2.6 * cells_in_series],   # 60% SOC, 347V
            [0.7, 2.7 * cells_in_series],   # 70% SOC, 360V
            [0.8, 2.8 * cells_in_series],   # 80% SOC, 373V
            [0.9, 2.9 * cells_in_series],   # 90% SOC, 387V
            [1.0, 3.0 * cells_in_series]    # 100% SOC, 400V
        ])
    
    def _setup_solid_state_characteristics(self) -> None:
        """Setup solid state battery characteristics."""
        # Solid state batteries have different characteristics
        self.temp_capacity_data = {
            -20: 0.8,   # 80% capacity at -20°C
            0: 0.95,    # 95% capacity at 0°C
            20: 1.0,    # 100% capacity at 20°C
            40: 0.95    # 95% capacity at 40°C
        }
        
        self.c_rate_efficiency_data = {
            0.5: 0.98,  # 98% efficiency at 0.5C
            1.0: 0.97,  # 97% efficiency at 1C
            2.0: 0.95,  # 95% efficiency at 2C
            5.0: 0.90   # 90% efficiency at 5C
        }
        
        # Solid state voltage characteristics (scaled to pack voltage)
        cells_in_series = int(self.voltage_nominal / 3.5)  # ~114 cells for 400V
        self.ocv_soc_data = np.array([
            [0.0, 2.5 * cells_in_series],   # 0% SOC, 285V
            [0.1, 2.6 * cells_in_series],   # 10% SOC, 296V
            [0.2, 2.7 * cells_in_series],   # 20% SOC, 308V
            [0.3, 2.8 * cells_in_series],   # 30% SOC, 319V
            [0.4, 2.9 * cells_in_series],   # 40% SOC, 331V
            [0.5, 3.0 * cells_in_series],   # 50% SOC, 342V
            [0.6, 3.1 * cells_in_series],   # 60% SOC, 353V
            [0.7, 3.2 * cells_in_series],   # 70% SOC, 365V
            [0.8, 3.3 * cells_in_series],   # 80% SOC, 376V
            [0.9, 3.4 * cells_in_series],   # 90% SOC, 388V
            [1.0, 3.5 * cells_in_series]    # 100% SOC, 399V
        ])
    
    def set_initial_state(self, soc: float, temperature: float) -> None:
        """
        Set initial battery state.
        
        Args:
            soc: Initial state of charge (0-1)
            temperature: Initial temperature in Celsius
        """
        self.state.soc = np.clip(soc, self.min_soc, self.max_soc)
        self.state.temperature = np.clip(temperature, self.min_temperature, self.max_temperature)
        self.state.voltage = self._calculate_ocv(self.state.soc)
        self.state.current = 0.0
        self.state.power = 0.0
        self.state.internal_resistance = self._calculate_internal_resistance(
            self.state.soc, self.state.temperature
        )
        
        self.logger.info(f"Battery initial state: SOC={self.state.soc:.3f}, "
                        f"T={self.state.temperature:.1f}°C, V={self.state.voltage:.1f}V")
    
    def update_state(self, power_demand: float, dt: float) -> None:
        """
        Update battery state based on power demand.
        
        Args:
            power_demand: Power demand in Watts (positive for discharge)
            dt: Time step in seconds
        """
        # Calculate current based on power demand
        current = self._calculate_current_from_power(power_demand)
        
        # Apply current limits
        current = self._apply_current_limits(current)
        
        # Update SOC
        self._update_soc(current, dt)
        
        # Update temperature
        self._update_temperature(current, dt)
        
        # Update voltage and resistance
        self.state.voltage = self._calculate_terminal_voltage(current)
        self.state.internal_resistance = self._calculate_internal_resistance(
            self.state.soc, self.state.temperature
        )
        
        # Update power and current
        self.state.current = current
        self.state.power = self.state.voltage * current
        
        # Check for critical conditions
        self._check_critical_conditions()
    
    def _calculate_current_from_power(self, power_demand: float) -> float:
        """
        Calculate current from power demand using simplified method.
        
        Args:
            power_demand: Power demand in Watts
            
        Returns:
            Current in Amperes
        """
        if abs(power_demand) < 1e-6:  # Near zero power
            return 0.0
        
        # Use open circuit voltage for current calculation to avoid instability
        ocv = self._calculate_ocv(self.state.soc)
        if abs(ocv) < 1e-6:
            return 0.0
        
        # Simple current calculation
        current = power_demand / ocv
        
        # Apply reasonable limits to prevent instability
        max_current = self.capacity_nominal * self.max_discharge_rate
        current = np.clip(current, -max_current, max_current)
        
        return current
    
    def _apply_current_limits(self, current: float) -> float:
        """
        Apply current limits based on C-rate and temperature.
        
        Args:
            current: Desired current in Amperes
            
        Returns:
            Limited current in Amperes
        """
        # Get temperature-dependent capacity
        capacity_factor = self._get_temperature_capacity_factor(self.state.temperature)
        effective_capacity = self.capacity_nominal * capacity_factor
        
        # Calculate C-rate
        c_rate = abs(current) / effective_capacity
        
        # Apply C-rate limits
        if current > 0:  # Discharge
            max_c_rate = self.max_discharge_rate
        else:  # Charge
            max_c_rate = self.max_charge_rate
        
        if c_rate > max_c_rate:
            current = np.sign(current) * max_c_rate * effective_capacity
        
        return current
    
    def _update_soc(self, current: float, dt: float) -> None:
        """
        Update state of charge based on current.
        
        Args:
            current: Current in Amperes (positive for discharge)
            dt: Time step in seconds
        """
        # Get temperature-dependent capacity
        capacity_factor = self._get_temperature_capacity_factor(self.state.temperature)
        effective_capacity = self.capacity_nominal * capacity_factor
        
        # Calculate C-rate for efficiency lookup
        c_rate = abs(current) / effective_capacity
        
        # Get coulombic efficiency
        efficiency = self._get_coulombic_efficiency(c_rate)
        
        # Update SOC (negative current = charging, positive = discharging)
        soc_change = -current * dt / (effective_capacity * 3600)  # Convert Ah to As
        soc_change *= efficiency  # Apply coulombic efficiency
        
        self.state.soc += soc_change
        self.state.soc = np.clip(self.state.soc, self.min_soc, self.max_soc)
    
    def _update_temperature(self, current: float, dt: float) -> None:
        """
        Update battery temperature based on heat generation and cooling.
        
        Args:
            current: Current in Amperes
            dt: Time step in seconds
        """
        # Calculate heat generation (Joule heating + electrochemical heating)
        joule_heating = current**2 * self.state.internal_resistance
        electrochemical_heating = abs(current) * 0.1  # Simplified model
        heat_generation = joule_heating + electrochemical_heating
        
        # Calculate cooling (simplified thermal model)
        temp_diff = self.state.temperature - 20.0  # Ambient temperature
        cooling_rate = self.cooling_capacity * (temp_diff / 40.0)  # Proportional cooling
        
        # Update temperature
        net_heat = heat_generation - cooling_rate
        temp_change = net_heat * dt / (self.mass * self.specific_heat)
        
        self.state.temperature += temp_change
        self.state.temperature = np.clip(
            self.state.temperature, self.min_temperature, self.max_temperature
        )
    
    def _calculate_terminal_voltage(self, current: float) -> float:
        """
        Calculate terminal voltage based on current.
        
        Args:
            current: Current in Amperes
            
        Returns:
            Terminal voltage in Volts
        """
        # Open circuit voltage
        ocv = self._calculate_ocv(self.state.soc)
        
        # Simplified internal resistance (avoid complex interpolation)
        base_resistance = 0.1  # Base resistance in Ohms
        resistance = base_resistance * (self.voltage_nominal / 400.0)  # Scale with voltage
        
        # Voltage drop
        voltage_drop = current * resistance
        
        # Terminal voltage with reasonable bounds
        terminal_voltage = ocv - voltage_drop
        
        # Ensure voltage stays within reasonable bounds
        min_voltage = self.voltage_range[0] * 0.5  # Allow some margin
        max_voltage = self.voltage_range[1] * 1.5   # Allow some margin
        terminal_voltage = np.clip(terminal_voltage, min_voltage, max_voltage)
        
        return terminal_voltage
    
    def _calculate_ocv(self, soc: float) -> float:
        """
        Calculate open circuit voltage from SOC.
        
        Args:
            soc: State of charge (0-1)
            
        Returns:
            Open circuit voltage in Volts
        """
        # Interpolate from OCV-SOC curve
        soc_values = self.ocv_soc_data[:, 0]
        voltage_values = self.ocv_soc_data[:, 1]
        
        ocv = np.interp(soc, soc_values, voltage_values)
        return ocv
    
    def _calculate_internal_resistance(self, soc: float, temperature: float) -> float:
        """
        Calculate internal resistance from SOC and temperature.
        
        Args:
            soc: State of charge (0-1)
            temperature: Temperature in Celsius
            
        Returns:
            Internal resistance in Ohms
        """
        # Interpolate from resistance lookup table
        soc_values = self.resistance_data['soc']
        temp_values = self.resistance_data['temp']
        resistance_table = self.resistance_data['resistance']
        
        # Find temperature index
        temp_idx = np.searchsorted(temp_values, temperature)
        if temp_idx == 0:
            temp_idx = 1
        elif temp_idx >= len(temp_values):
            temp_idx = len(temp_values) - 1
        
        # Interpolate between temperature points
        if temp_idx < len(temp_values):
            temp_ratio = (temperature - temp_values[temp_idx-1]) / (temp_values[temp_idx] - temp_values[temp_idx-1])
            resistance_low = np.interp(soc, soc_values, resistance_table[temp_idx-1])
            resistance_high = np.interp(soc, soc_values, resistance_table[temp_idx])
            resistance = resistance_low + temp_ratio * (resistance_high - resistance_low)
        else:
            resistance = np.interp(soc, soc_values, resistance_table[-1])
        
        return resistance
    
    def _get_temperature_capacity_factor(self, temperature: float) -> float:
        """
        Get capacity factor based on temperature.
        
        Args:
            temperature: Temperature in Celsius
            
        Returns:
            Capacity factor (0-1)
        """
        temp_values = list(self.temp_capacity_data.keys())
        capacity_values = list(self.temp_capacity_data.values())
        
        capacity_factor = np.interp(temperature, temp_values, capacity_values)
        return capacity_factor
    
    def _get_coulombic_efficiency(self, c_rate: float) -> float:
        """
        Get coulombic efficiency based on C-rate.
        
        Args:
            c_rate: C-rate
            
        Returns:
            Coulombic efficiency (0-1)
        """
        c_rate_values = list(self.c_rate_efficiency_data.keys())
        efficiency_values = list(self.c_rate_efficiency_data.values())
        
        efficiency = np.interp(c_rate, c_rate_values, efficiency_values)
        return efficiency
    
    def _check_critical_conditions(self) -> None:
        """Check for critical battery conditions."""
        if self.state.soc < 0.05:  # 5% SOC
            self.logger.warning("Battery critically low - 5% SOC")
        
        if self.state.temperature > 55.0:  # 55°C
            self.logger.warning("Battery overheating - 55°C")
        
        if self.state.voltage < self.voltage_range[0]:
            self.logger.warning(f"Battery voltage below minimum: {self.state.voltage:.1f}V")
    
    def get_state_of_charge(self) -> float:
        """Get current state of charge."""
        return self.state.soc
    
    def get_temperature(self) -> float:
        """Get current temperature."""
        return self.state.temperature
    
    def get_voltage(self) -> float:
        """Get current terminal voltage."""
        return self.state.voltage
    
    def get_current(self) -> float:
        """Get current current."""
        return self.state.current
    
    def get_power(self) -> float:
        """Get current power."""
        return self.state.power
    
    def get_available_power(self) -> float:
        """Get maximum available power."""
        # Calculate maximum current at current SOC and temperature
        capacity_factor = self._get_temperature_capacity_factor(self.state.temperature)
        effective_capacity = self.capacity_nominal * capacity_factor
        max_current = self.max_discharge_rate * effective_capacity
        
        # Calculate maximum power
        max_voltage = self._calculate_terminal_voltage(-max_current)  # Negative for discharge
        max_power = max_voltage * max_current
        
        return max_power
    
    def get_remaining_energy(self) -> float:
        """Get remaining energy in Wh."""
        capacity_factor = self._get_temperature_capacity_factor(self.state.temperature)
        effective_capacity = self.capacity_nominal * capacity_factor
        remaining_capacity = effective_capacity * self.state.soc
        remaining_energy = remaining_capacity * self.state.voltage
        
        return remaining_energy
    
    def get_soc_derivative(self) -> float:
        """Get SOC derivative for integration."""
        if abs(self.state.current) < 1e-6:
            return 0.0
        
        capacity_factor = self._get_temperature_capacity_factor(self.state.temperature)
        effective_capacity = self.capacity_nominal * capacity_factor
        c_rate = abs(self.state.current) / effective_capacity
        efficiency = self._get_coulombic_efficiency(c_rate)
        
        soc_derivative = -self.state.current / (effective_capacity * 3600) * efficiency
        return soc_derivative
    
    def get_temperature_derivative(self) -> float:
        """Get temperature derivative for integration."""
        # Heat generation
        joule_heating = self.state.current**2 * self.state.internal_resistance
        electrochemical_heating = abs(self.state.current) * 0.1
        heat_generation = joule_heating + electrochemical_heating
        
        # Cooling
        temp_diff = self.state.temperature - 20.0
        cooling_rate = self.cooling_capacity * (temp_diff / 40.0)
        
        # Temperature derivative
        net_heat = heat_generation - cooling_rate
        temp_derivative = net_heat / (self.mass * self.specific_heat)
        
        return temp_derivative
