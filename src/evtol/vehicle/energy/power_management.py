"""
Power Management System for eVTOL

Coordinates:
1. Battery voltage and current limits
2. Motor controller efficiency
3. Thermal management across all subsystems
4. Power distribution and budgeting
5. Energy optimization strategies

Theory
======

Power Budget Hierarchy:

Battery (nominal) → DC Bus → Contractors → Motor Controllers

Power Flow:
    P_battery = V_terminal × I_discharge
    P_loss_dist = I_discrete × R_harness
    P_available_motor = P_battery - P_loss_dist
    P_motor_out = P_available_motor × η_controller × η_motor
    P_rotor_useful = P_motor_out - P_motor_loss

Energy Reserve Monitoring:

During high-intensity maneuvers (e.g., max climb), power demand exceeds
cruise levels. Battery voltage sags. System must:

1. Monitor voltage droop vs. target
2. Degrade low-priority systems if necessary
3. Maintain flight-critical margins

Performance Reserve:
    Power_margin = P_available - P_current_demand
    If margin < P_hover, helicopter cannot maintain hover.

Thermal Management:

Multiple heat sources:
- Battery Joule heating: I²·R_internal
- Motor copper losses: I²·R_phase
- Motor core losses: frequency-dependent
- Motor friction: windage
- Controller switching losses: f_sw × C_loss × V²

Total system temperature rise determines:
- Motor derating (reduced available torque)
- Battery voltage sag acceleration
- Component reliability (Arrhenius model)

References
==========

[1] Steinberg, M.L. (1992). "Optimal Power Distribution in Multi-Motor
    Fixed-Wing Aircraft." IEEE AES Systems Magazine.

[2] Chen, M., Rincon-Mora, G.A. (2006). Battery Management Systems.
    IEEE PELS Education Chapter, vol. 5.

[3] Sulligoi, G., et al. (2016). "Smart Power Management in Shipboard
    Microgrids." IEEE Trans. Power Systems, vol. 31, no. 2.

[4] NASA SP-8100 (1974). Space Vehicle Design Criteria: Power Distribution
    Architecture. KSC-14449.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class PowerMode(Enum):
    """System operating modes."""
    HOVER = "hover"
    CRUISE = "cruise"
    CLIMB = "climb"
    DESCENT = "descent"
    TRANSITION = "transition"
    LOW_POWER = "low_power"
    EMERGENCY = "emergency"


@dataclass
class PowerBusConfig:
    """
    Power bus and distribution configuration.
    
    Attributes:
        bus_voltage: Main DC bus voltage [V]
        num_controllers: Number of independent motor controllers
        harness_resistance: Main distribution harness [Ω]
        max_bus_current: Maximum bus current [A]
        controller_efficiency: Motor controller efficiency [0-1]
    """
    bus_voltage: float = 48.0
    num_controllers: int = 4
    harness_resistance: float = 0.02
    max_bus_current: float = 300.0
    controller_efficiency: float = 0.96


@dataclass
class MotorControllerState:
    """State of single motor controller."""
    motor_id: int = 0
    
    # Electrical
    input_voltage: float = 48.0
    input_current: float = 0.0
    output_voltage: float = 48.0  # After internal drops
    output_current: float = 0.0
    
    # Power
    power_in: float = 0.0       # [W]
    power_out: float = 0.0      # [W]
    power_loss: float = 0.0     # [W]
    
    # Thermal
    temperature: float = 25.0   # [°C]
    
    # Status
    is_enabled: bool = True
    throttle_cmd: float = 0.0   # [0-1]
    limiting_factor: str = ""


@dataclass
class SystemPowerState:
    """Overall system power state."""
    mode: PowerMode = PowerMode.HOVER
    
    # Battery-level
    battery_voltage: float = 48.0
    battery_current_total: float = 0.0
    battery_power_draw: float = 0.0
    battery_soc: float = 1.0
    battery_temp: float = 25.0
    
    # Bus-level
    bus_voltage: float = 48.0
    bus_current: float = 0.0
    bus_power: float = 0.0
    harness_loss: float = 0.0
    
    # Controllers
    controller_states: list = field(default_factory=list)
    total_rotor_power: float = 0.0
    
    # System status
    power_available_margin: float = 0.0
    thermal_margin: float = 0.0
    is_powered: bool = True
    warning_flags: str = ""


class PowerManagementSystem:
    """
    Coordinates battery, controllers, and motors.
    
    Manages:
    - Voltage distribution and droop compensation
    - Thermal feedback and derating
    - Power allocation across motors
    - Reserve monitoring
    
    Usage:
        pms = PowerManagementSystem(battery, config)
        pms.set_motor_commands([throttle_1, throttle_2, throttle_3, throttle_4])
        state = pms.update(battery_state, motor_states, dt=0.01)
        power_available = state.power_available_margin
    """
    
    def __init__(
        self,
        battery,
        bus_config: PowerBusConfig,
        motor_thermal_capacity: float = 200.0,  # [J/°C]
        controller_thermal_capacity: float = 50.0,  # [J/°C]
    ):
        """
        Initialize power management system.
        
        Args:
            battery: BatteryPack instance
            bus_config: Power bus configuration
            motor_thermal_capacity: Thermal mass of each motor [J/°C]
            controller_thermal_capacity: Thermal mass of each controller [J/°C]
        """
        self.battery = battery
        self.config = bus_config
        self.motor_c_thermal = motor_thermal_capacity
        self.controller_c_thermal = controller_thermal_capacity
        
        # Initialize controller states
        self.controller_states = [
            MotorControllerState(motor_id=i)
            for i in range(bus_config.num_controllers)
        ]
        
        self.state = SystemPowerState(
            controller_states=self.controller_states
        )
        
        # Power command targets [0-1]
        self.throttle_commands = np.zeros(bus_config.num_controllers)
        
        # Thermal state for motors and controllers
        self.motor_temperatures = np.ones(bus_config.num_controllers) * 25.0
        self.controller_temperatures = np.ones(bus_config.num_controllers) * 25.0
        
        # Power allocation history (for diagnostics)
        self.power_history = {
            'timestamp': [],
            'battery_power': [],
            'total_motor_power': [],
            'thermal_margin': [],
        }
        
        logger.info(
            f"PowerManagementSystem initialized: {bus_config.num_controllers} "
            f"controllers, {bus_config.bus_voltage}V bus, "
            f"{bus_config.controller_efficiency*100:.1f}% efficiency"
        )
    
    def set_motor_commands(self, throttle_cmds: np.ndarray | list):
        """
        Set throttle commands for all motors.
        
        Args:
            throttle_cmds: Motor throttle commands [0-1], length = num_controllers
        """
        if len(throttle_cmds) != self.config.num_controllers:
            raise ValueError(
                f"Expected {self.config.num_controllers} commands, "
                f"got {len(throttle_cmds)}"
            )
        
        self.throttle_commands = np.array(throttle_cmds, dtype=float)
        self.throttle_commands = np.clip(self.throttle_commands, 0.0, 1.0)
    
    def _estimate_motor_power(
        self,
        throttle: float,
        motor_rpm: float,
        motor_efficiency: float = 0.88,
    ) -> float:
        """
        Estimate power draw for single motor at given throttle.
        
        Simple model: P = T·ω where T comes from throttle command
        More accurate: use actual motor datasheet curve
        
        Args:
            throttle: Throttle command [0-1]
            motor_rpm: Current motor speed [RPM]
            motor_efficiency: Motor efficiency [0-1]
        
        Returns:
            Estimated power input [W]
        """
        # Nominal max power per motor
        P_max_motor = 12000  # [W] nominal for 48V PMSM
        
        # Power demand scales roughly with throttle^2 (similar to thrust)
        # At high throttle, efficiency drops due to saturation
        if throttle < 0.1:
            return 100.0  # Minimum standby power
        
        # Quadratic scaling for aerodynamic loading
        power_ideal = P_max_motor * (throttle**2.2)
        
        # Derate for efficiency
        power_input = power_ideal / motor_efficiency
        
        return power_input
    
    def update(
        self,
        battery_state,
        motor_speeds_rpm: np.ndarray,
        motor_efficiencies: np.ndarray,
        ambient_temp: float = 25.0,
        dt: float = 0.01,
    ) -> SystemPowerState:
        """
        Update power management state.
        
        Implements:
        1. Power allocation across motors
        2. Voltage droop and derating
        3. Thermal feedback
        4. Reserve monitoring
        
        Args:
            battery_state: Current battery state
            motor_speeds_rpm: Current motor speeds [RPM] (length = num_controllers)
            motor_efficiencies: Motor efficiency scalars [0-1] (length = num_controllers)
            ambient_temp: Ambient temperature [°C]
            dt: Time step [s]
        
        Returns:
            Updated SystemPowerState
        """
        # Battery parameters
        V_battery = battery_state.voltage
        
        # Voltage droop in harness
        # Allocate proportional to throttle (rough power distribution)
        P_estimate_total = sum(
            self._estimate_motor_power(self.throttle_commands[i], motor_speeds_rpm[i])
            for i in range(self.config.num_controllers)
        )
        
        # Current draw at bus (approximately)
        I_bus_estimate = max(P_estimate_total / V_battery, 0.1)
        I_bus_estimate = min(I_bus_estimate, self.config.max_bus_current)
        
        # Voltage droop in distribution harness
        V_droop = I_bus_estimate * self.config.harness_resistance
        V_bus = V_battery - V_droop
        
        # Power loss in harness
        P_harness_loss = I_bus_estimate**2 * self.config.harness_resistance
        
        # Update individual controllers
        total_power_out = 0.0
        motor_current_allocations = []
        
        for i in range(self.config.num_controllers):
            if not self.controller_states[i].is_enabled:
                continue
            
            throttle = self.throttle_commands[i]
            P_motor_demand = self._estimate_motor_power(
                throttle,
                motor_speeds_rpm[i],
                motor_efficiencies[i]
            )
            
            # Current for this motor (estimate)
            I_motor = P_motor_demand / V_bus if V_bus > 0 else 0.0
            I_motor = max(I_motor, 0.1)  # Minimum 0.1A
            
            # Power after controller losses
            P_controller_loss = P_motor_demand * (1.0 - self.config.controller_efficiency)
            P_motor_output = P_motor_demand - P_controller_loss
            
            # Thermal dynamics for controller
            tau_controller = self.controller_c_thermal / 5.0  # [s], assume 5W/°C dissipation
            T_controller_ss = ambient_temp + P_controller_loss / 5.0
            dT_controller = (T_controller_ss - self.controller_temperatures[i]) * (dt / max(tau_controller, 0.01))
            self.controller_temperatures[i] += dT_controller
            
            # Derating for controller temperature
            if self.controller_temperatures[i] > 85.0:
                derate = 1.0 - 0.05 * (self.controller_temperatures[i] - 85.0) / 15.0
                P_motor_output *= max(derate, 0.7)
                self.controller_states[i].limiting_factor = "CONTROLLER_TEMP"
            
            # Thermal dynamics for motor (simplified)
            tau_motor = self.motor_c_thermal / 10.0  # [W/°C dissipation]
            motor_loss = P_motor_output * (1.0 - motor_efficiencies[i])
            T_motor_ss = ambient_temp + motor_loss / 10.0
            dT_motor = (T_motor_ss - self.motor_temperatures[i]) * (dt / max(tau_motor, 0.01))
            self.motor_temperatures[i] += dT_motor
            
            # Derating for motor temperature
            if self.motor_temperatures[i] > 90.0:
                derate = 1.0 - 0.04 * (self.motor_temperatures[i] - 90.0) / 20.0
                P_motor_output *= max(derate, 0.75)
                self.controller_states[i].limiting_factor = "MOTOR_TEMP"
            
            # Check voltage/current limits
            if V_bus < self.battery.config.min_voltage * 1.05:
                self.controller_states[i].limiting_factor = "LOW_VOLTAGE"
            
            if I_motor > self.config.max_bus_current / self.config.num_controllers:
                I_motor = self.config.max_bus_current / self.config.num_controllers
                P_motor_output = I_motor * V_bus * self.config.controller_efficiency
                self.controller_states[i].limiting_factor = "CURRENT_LIMIT"
            
            # Update controller state
            self.controller_states[i].input_voltage = V_bus
            self.controller_states[i].input_current = I_motor
            self.controller_states[i].output_voltage = V_bus  # Approximate
            self.controller_states[i].output_current = P_motor_output / max(V_bus, 1.0)
            self.controller_states[i].power_in = P_motor_demand
            self.controller_states[i].power_out = P_motor_output
            self.controller_states[i].power_loss = P_motor_demand - P_motor_output
            self.controller_states[i].temperature = self.controller_temperatures[i]
            self.controller_states[i].throttle_cmd = throttle
            
            motor_current_allocations.append(I_motor)
            total_power_out += P_motor_output
        
        # Total current draw at battery
        I_battery_total = sum(motor_current_allocations) + 5.0  # +5A for avionics
        
        # Power margin (reserve for climbing)
        P_available_total = V_battery * self.config.max_bus_current
        power_margin = P_available_total - (I_battery_total * V_battery + P_harness_loss)
        
        # Thermal margin (min of all components - 100°C limit)
        all_temps = list(self.controller_temperatures) + list(self.motor_temperatures)
        max_temp = max(all_temps)
        thermal_margin = 100.0 - max_temp
        
        # Determine operating mode
        if V_battery < self.battery.config.min_voltage * 1.1:
            mode = PowerMode.EMERGENCY
            warning_flags = "CRITICAL_BATTERY "
        elif thermal_margin < 10.0:
            mode = PowerMode.DESCENT
            warning_flags = "THERMAL_LIMIT "
        elif battery_state.soc < 0.15:
            mode = PowerMode.DESCENT
            warning_flags = "LOW_SOC_MANAGER "
        else:
            mode = PowerMode.HOVER
            warning_flags = ""
        
        # Update system state
        self.state.mode = mode
        self.state.battery_voltage = V_battery
        self.state.battery_current_total = I_battery_total
        self.state.battery_power_draw = V_battery * I_battery_total
        self.state.battery_soc = battery_state.soc
        self.state.battery_temp = battery_state.temperature
        
        self.state.bus_voltage = V_bus
        self.state.bus_current = I_bus_estimate
        self.state.bus_power = V_bus * I_bus_estimate
        self.state.harness_loss = P_harness_loss
        
        self.state.total_rotor_power = total_power_out
        self.state.power_available_margin = power_margin
        self.state.thermal_margin = thermal_margin
        self.state.is_powered = V_bus > self.battery.config.min_voltage
        self.state.warning_flags = warning_flags.strip()
        
        # Log history
        self.power_history['battery_power'].append(self.state.battery_power_draw)
        self.power_history['total_motor_power'].append(total_power_out)
        self.power_history['thermal_margin'].append(thermal_margin)
        
        return self.state
    
    def get_power_available(self) -> float:
        """Get available power margin [W]."""
        return self.state.power_available_margin
    
    def get_thermal_margin(self) -> float:
        """Get thermal margin to 100°C limit [°C]."""
        return self.state.thermal_margin
    
    def get_reserve_endurance(self, hover_power: float = 5000.0) -> float:
        """
        Estimate how long system can maintain hover at current battery state.
        
        Args:
            hover_power: Power required to hover [W]
        
        Returns:
            Endurance [s]
        """
        if hover_power < 100:
            return float('inf')
        
        energy_remaining = self.battery.get_energy_remaining()
        endurance = energy_remaining * 3600 / hover_power
        
        return endurance
