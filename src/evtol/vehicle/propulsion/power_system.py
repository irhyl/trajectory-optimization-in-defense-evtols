"""
Power Distribution System - Electric Power Management

This module models the power distribution architecture for eVTOL,
including the DC bus, power electronics, and load management.

Architecture
============

Battery Pack → DC Bus → Motor Controllers → Motors
                ↓
           Auxiliary Loads (avionics, actuators, cooling)

Key Components:
1. DC Bus: High-voltage distribution (typically 400-800V)
2. Motor Controllers: Inverters (DC→3-phase AC)
3. Power Distribution Unit (PDU): Load switching and protection
4. DC-DC Converters: For auxiliary loads (12V, 28V)

Power Flow:
    P_bus = P_propulsion + P_actuators + P_avionics + P_cooling + P_losses

Bus Voltage Dynamics:
    C_bus · dV_bus/dt = I_battery - Σ I_loads

Protection:
    - Overcurrent protection
    - Undervoltage lockout
    - Thermal protection

Author: Defense eVTOL Research Team
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class LoadPriority(Enum):
    """Load shedding priority levels."""
    CRITICAL = 0     # Flight-critical (cannot shed)
    HIGH = 1         # Important but can reduce
    MEDIUM = 2       # Comfort/convenience
    LOW = 3          # Shed first


@dataclass
class PowerLoad:
    """Represents a power consumer on the bus."""
    name: str
    power_nominal: float      # W (rated power)
    power_current: float = 0  # W (current consumption)
    voltage_min: float = 350  # V (minimum operating voltage)
    voltage_max: float = 450  # V (maximum operating voltage)
    priority: LoadPriority = LoadPriority.MEDIUM
    enabled: bool = True
    efficiency: float = 0.95

    @property
    def current(self) -> float:
        """Current draw at nominal bus voltage."""
        return self.power_current / 400.0  # Approximate at 400V


@dataclass
class BusState:
    """DC bus operating state."""
    voltage: float = 400.0         # V
    current: float = 0.0           # A (total bus current)
    power_total: float = 0.0       # W
    power_available: float = 0.0   # W (from battery)

    # Load breakdown
    power_propulsion: float = 0.0  # W
    power_actuators: float = 0.0   # W
    power_avionics: float = 0.0    # W
    power_thermal: float = 0.0     # W
    power_auxiliary: float = 0.0   # W

    # Losses
    power_losses: float = 0.0      # W

    # Status
    load_fraction: float = 0.0     # 0-1
    undervoltage: bool = False
    overload: bool = False
    loads_shed: list[str] = field(default_factory=list)


class InverterModel:
    """
    Motor controller/inverter model.

    Converts DC bus power to 3-phase AC for motor.
    Includes efficiency modeling and thermal limits.
    """

    def __init__(
        self,
        power_rating: float,
        voltage_nominal: float = 400.0,
        switching_freq: float = 16000.0,
    ):
        """
        Initialize inverter model.

        Args:
            power_rating: Rated power [W]
            voltage_nominal: Nominal DC bus voltage [V]
            switching_freq: PWM switching frequency [Hz]
        """
        self.power_rating = power_rating
        self.V_nominal = voltage_nominal
        self.f_sw = switching_freq

        # Efficiency parameters
        self.eta_peak = 0.98          # Peak efficiency
        self.P_standby = 20.0         # Standby power [W]

        # Thermal
        self.T_junction = 25.0        # °C
        self.T_junction_max = 150.0   # °C
        self.R_th = 0.1               # K/W (junction to heatsink)

    def compute_output(
        self,
        V_bus: float,
        P_motor: float,
        motor_efficiency: float,
    ) -> tuple[float, float, float]:
        """
        Compute inverter power flow.

        Args:
            V_bus: DC bus voltage [V]
            P_motor: Motor mechanical power [W]
            motor_efficiency: Motor efficiency [0-1]

        Returns:
            (P_dc_input, P_loss, efficiency)
        """
        if P_motor <= 0:
            return self.P_standby, self.P_standby, 0.0

        # Motor electrical power
        P_motor_elec = P_motor / motor_efficiency

        # Inverter efficiency (decreases at low load)
        load_fraction = min(P_motor_elec / self.power_rating, 1.0)

        # Efficiency model (parabolic with peak at ~70% load)
        eta = self.eta_peak * (1 - 0.05 * (load_fraction - 0.7)**2)
        eta = max(eta, 0.85)  # Minimum efficiency

        # DC input power
        P_dc = P_motor_elec / eta

        # Losses
        P_loss = P_dc - P_motor_elec + self.P_standby

        # Update junction temperature (simplified)
        T_ambient = 40.0  # °C (assumed)
        self.T_junction = T_ambient + P_loss * self.R_th

        return P_dc, P_loss, eta


class PowerDistribution:
    """
    Power distribution unit model.

    Manages:
    1. DC bus voltage regulation
    2. Load switching and protection
    3. Power budgeting and load shedding
    4. Fault detection and isolation

    Attributes:
        voltage_nominal: Nominal bus voltage [V]
        voltage_limits: (min, max) operating voltage [V]
        power_capacity: Maximum continuous power [W]
    """

    def __init__(
        self,
        voltage_nominal: float = 400.0,
        voltage_min: float = 350.0,
        voltage_max: float = 450.0,
        power_capacity: float = 500000.0,  # 500 kW
        bus_capacitance: float = 0.01,     # F (bus capacitor)
    ):
        """
        Initialize power distribution system.

        Args:
            voltage_nominal: Nominal bus voltage [V]
            voltage_min: Minimum operating voltage [V]
            voltage_max: Maximum operating voltage [V]
            power_capacity: Maximum bus power [W]
            bus_capacitance: Bus capacitance [F]
        """
        self.V_nominal = voltage_nominal
        self.V_min = voltage_min
        self.V_max = voltage_max
        self.P_max = power_capacity
        self.C_bus = bus_capacitance

        # Loads
        self.loads: dict[str, PowerLoad] = {}
        self._setup_default_loads()

        # Inverters (one per motor)
        self.inverters: list[InverterModel] = []

        # State
        self.state = BusState(voltage=voltage_nominal)

        logger.info(f"PowerDistribution initialized: "
                   f"V_nom={voltage_nominal}V, P_max={power_capacity/1000:.0f}kW")

    def _setup_default_loads(self):
        """Set up default auxiliary loads."""
        self.add_load(PowerLoad(
            name="avionics",
            power_nominal=500,
            power_current=400,
            priority=LoadPriority.CRITICAL,
        ))

        self.add_load(PowerLoad(
            name="flight_computer",
            power_nominal=200,
            power_current=150,
            priority=LoadPriority.CRITICAL,
        ))

        self.add_load(PowerLoad(
            name="navigation",
            power_nominal=100,
            power_current=80,
            priority=LoadPriority.CRITICAL,
        ))

        self.add_load(PowerLoad(
            name="communication",
            power_nominal=150,
            power_current=100,
            priority=LoadPriority.HIGH,
        ))

        self.add_load(PowerLoad(
            name="actuators",
            power_nominal=2000,
            power_current=500,
            priority=LoadPriority.CRITICAL,
        ))

        self.add_load(PowerLoad(
            name="cooling",
            power_nominal=3000,
            power_current=1500,
            priority=LoadPriority.HIGH,
        ))

        self.add_load(PowerLoad(
            name="lighting",
            power_nominal=200,
            power_current=100,
            priority=LoadPriority.LOW,
        ))

        self.add_load(PowerLoad(
            name="payload",
            power_nominal=500,
            power_current=300,
            priority=LoadPriority.MEDIUM,
        ))

    def add_load(self, load: PowerLoad):
        """Add a power load."""
        self.loads[load.name] = load

    def add_inverter(self, power_rating: float):
        """Add a motor inverter."""
        self.inverters.append(InverterModel(power_rating, self.V_nominal))

    def compute_power_budget(
        self,
        power_propulsion: float,
        V_bus: float,
        motor_efficiencies: list[float],
    ) -> BusState:
        """
        Compute power distribution and bus state.

        Args:
            power_propulsion: Total propulsion power [W]
            V_bus: Current bus voltage [V]
            motor_efficiencies: Efficiency of each motor [0-1]

        Returns:
            Updated BusState
        """
        state = BusState(voltage=V_bus)

        # Check voltage limits
        if V_bus < self.V_min:
            state.undervoltage = True
            logger.warning(f"Undervoltage: V_bus={V_bus:.1f}V < {self.V_min}V")

        # Propulsion power through inverters
        if len(self.inverters) > 0 and len(motor_efficiencies) > 0:
            P_per_motor = power_propulsion / len(self.inverters)
            P_prop_total = 0

            for i, inv in enumerate(self.inverters):
                eta_motor = motor_efficiencies[i] if i < len(motor_efficiencies) else 0.95
                P_dc, P_loss, _ = inv.compute_output(V_bus, P_per_motor, eta_motor)
                P_prop_total += P_dc
                state.power_losses += P_loss

            state.power_propulsion = P_prop_total
        else:
            state.power_propulsion = power_propulsion / 0.95  # Assume 95% efficiency

        # Auxiliary loads
        for name, load in self.loads.items():
            if not load.enabled:
                continue

            # Check if load can operate at current voltage
            if V_bus < load.voltage_min:
                load.power_current = 0
                state.loads_shed.append(name)
                continue

            # Add to appropriate category
            if name in ["avionics", "flight_computer", "navigation", "communication"]:
                state.power_avionics += load.power_current
            elif name == "actuators":
                state.power_actuators += load.power_current
            elif name == "cooling":
                state.power_thermal += load.power_current
            else:
                state.power_auxiliary += load.power_current

            # Account for converter efficiency
            state.power_losses += load.power_current * (1 - load.efficiency)

        # Total power
        state.power_total = (
            state.power_propulsion +
            state.power_actuators +
            state.power_avionics +
            state.power_thermal +
            state.power_auxiliary +
            state.power_losses
        )

        # Bus current
        if V_bus > 0:
            state.current = state.power_total / V_bus

        # Load fraction
        state.load_fraction = state.power_total / self.P_max
        state.overload = state.load_fraction > 1.0

        if state.overload:
            logger.warning(f"Power overload: {state.power_total/1000:.1f}kW > "
                          f"{self.P_max/1000:.1f}kW")

        self.state = state
        return state

    def load_shedding(
        self,
        power_available: float,
        power_required: float,
    ) -> list[str]:
        """
        Perform load shedding to match available power.

        Sheds loads in order of priority (lowest first).

        Args:
            power_available: Available power from battery [W]
            power_required: Total power demand [W]

        Returns:
            List of shed load names
        """
        if power_required <= power_available:
            return []

        power_deficit = power_required - power_available
        shed_loads = []

        # Sort loads by priority (lowest priority first)
        sorted_loads = sorted(
            self.loads.items(),
            key=lambda x: x[1].priority.value,
            reverse=True
        )

        for name, load in sorted_loads:
            if load.priority == LoadPriority.CRITICAL:
                continue  # Never shed critical loads

            if power_deficit <= 0:
                break

            if load.enabled and load.power_current > 0:
                power_deficit -= load.power_current
                load.enabled = False
                shed_loads.append(name)
                logger.info(f"Load shed: {name} ({load.power_current:.0f}W)")

        if power_deficit > 0:
            logger.error(f"Cannot meet power demand even after shedding. "
                        f"Deficit: {power_deficit:.0f}W")

        return shed_loads

    def restore_loads(self, power_available: float):
        """
        Restore shed loads when power becomes available.

        Args:
            power_available: Available power [W]
        """
        # Sort by priority (highest first to restore)
        sorted_loads = sorted(
            self.loads.items(),
            key=lambda x: x[1].priority.value
        )

        power_remaining = power_available - self.state.power_total

        for name, load in sorted_loads:
            if not load.enabled and power_remaining > load.power_nominal:
                load.enabled = True
                power_remaining -= load.power_nominal
                logger.info(f"Load restored: {name}")

    def get_efficiency(self) -> float:
        """
        Get overall power distribution efficiency.

        Returns:
            Efficiency [0-1]
        """
        if self.state.power_total <= 0:
            return 1.0

        power_output = self.state.power_total - self.state.power_losses
        return power_output / self.state.power_total


@dataclass
class PowerSystemState:
    """Complete power system state."""
    # Bus
    bus_voltage: float = 400.0
    bus_current: float = 0.0

    # Power breakdown
    power_propulsion: float = 0.0
    power_auxiliary: float = 0.0
    power_total: float = 0.0

    # Battery interface
    battery_current: float = 0.0
    battery_power: float = 0.0

    # Efficiency
    efficiency_propulsion: float = 0.95
    efficiency_system: float = 0.90

    # Status
    healthy: bool = True
    faults: list[str] = field(default_factory=list)


class PowerSystem:
    """
    Integrated power system model.

    Combines:
    - Battery (external interface)
    - Power distribution
    - Motor controllers
    - Load management
    """

    def __init__(
        self,
        num_motors: int = 2,
        motor_power: float = 150000.0,  # W per motor
        bus_voltage: float = 400.0,
    ):
        """
        Initialize power system.

        Args:
            num_motors: Number of propulsion motors
            motor_power: Power rating per motor [W]
            bus_voltage: Nominal bus voltage [V]
        """
        # Power distribution
        self.pdu = PowerDistribution(
            voltage_nominal=bus_voltage,
            power_capacity=num_motors * motor_power * 1.2,
        )

        # Add inverters for each motor
        for _ in range(num_motors):
            self.pdu.add_inverter(motor_power)

        self.state = PowerSystemState(bus_voltage=bus_voltage)
        self.num_motors = num_motors

    def update(
        self,
        motor_powers: list[float],
        motor_efficiencies: list[float],
        battery_voltage: float,
        battery_power_available: float,
    ) -> PowerSystemState:
        """
        Update power system state.

        Args:
            motor_powers: Mechanical power per motor [W]
            motor_efficiencies: Efficiency per motor [0-1]
            battery_voltage: Battery terminal voltage [V]
            battery_power_available: Available battery power [W]

        Returns:
            Updated PowerSystemState
        """
        # Total propulsion power
        P_prop = sum(motor_powers)

        # Compute power budget
        bus_state = self.pdu.compute_power_budget(
            P_prop,
            battery_voltage,
            motor_efficiencies,
        )

        # Check if we need to shed loads
        if bus_state.power_total > battery_power_available:
            shed = self.pdu.load_shedding(
                battery_power_available,
                bus_state.power_total,
            )
            if shed:
                # Recompute with shed loads
                bus_state = self.pdu.compute_power_budget(
                    P_prop,
                    battery_voltage,
                    motor_efficiencies,
                )

        # Update state
        self.state.bus_voltage = battery_voltage
        self.state.bus_current = bus_state.current
        self.state.power_propulsion = bus_state.power_propulsion
        self.state.power_auxiliary = (
            bus_state.power_actuators +
            bus_state.power_avionics +
            bus_state.power_thermal +
            bus_state.power_auxiliary
        )
        self.state.power_total = bus_state.power_total
        self.state.battery_current = bus_state.current
        self.state.battery_power = bus_state.power_total

        # Efficiencies
        if P_prop > 0:
            self.state.efficiency_propulsion = P_prop / bus_state.power_propulsion
        self.state.efficiency_system = self.pdu.get_efficiency()

        # Health check
        self.state.healthy = not (bus_state.undervoltage or bus_state.overload)
        self.state.faults = []
        if bus_state.undervoltage:
            self.state.faults.append("undervoltage")
        if bus_state.overload:
            self.state.faults.append("overload")

        return self.state
