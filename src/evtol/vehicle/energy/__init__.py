"""
Energy Subpackage - Battery and Power Management

This subpackage contains energy system models for eVTOL:

Battery Model (battery_model.py):
- Lithium-ion equivalent circuit with voltage sag
- Temperature-dependent internal resistance
- SOC/SOH tracking and cycle counting
- Thermal dynamics with ambient coupling

Power Management (power_management.py):
- Multi-motor power distribution
- Voltage droop compensation
- Thermal derating (controller + motor)
- Reserve monitoring and power margins

Legacy modules (preserved for compatibility):
- battery.py: Previous battery model
- thermal.py: Previous thermal model
"""

# New comprehensive models
try:
    from .battery_model import BatteryPack, BatteryConfig, BatteryState
    from .power_management import (
        PowerManagementSystem,
        PowerBusConfig,
        PowerMode,
        MotorControllerState,
        SystemPowerState,
    )
except ImportError:
    pass

# Legacy models (backward compatibility)
try:
    from .battery import BatteryModel, BatteryState as BatteryStateLegacy
except ImportError:
    pass

try:
    from .thermal import ThermalModel, ThermalState
except ImportError:
    pass

__all__ = [
    # New models
    "BatteryPack",
    "BatteryConfig",
    "BatteryState",
    "PowerManagementSystem",
    "PowerBusConfig",
    "PowerMode",
    "MotorControllerState",
    "SystemPowerState",
]
