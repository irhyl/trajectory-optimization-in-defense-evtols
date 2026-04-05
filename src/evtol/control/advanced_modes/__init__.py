"""
Advanced Flight Modes Package - Phase 2.3

Exports:
- AdvancedFlightModeManager: Main mode selection and transition control
- EnergyOptimizer: Energy consumption analysis and mode recommendation
- ComprehensiveStallProtection: Stall detection and recovery
"""

from .advanced_modes import (
    AdvancedFlightModeManager,
    FlightModeState,
    SafetyLevel,
    AdvancedFlightModeState,
    BlendingCurve,
    MotorCommand,
    ThreatMap,
    VehicleState,
    MissionContext,
)

from .energy_optimizer import (
    EnergyOptimizer,
    VehicleEnergyModel,
    EnergyOptimizationResult,
)

from .stall_protection import (
    ComprehensiveStallProtection,
    StallDetector,
    StallProtectionController,
    StallRecoveryProfile,
    RecoveryPhase,
)

__all__ = [
    # Core manager
    'AdvancedFlightModeManager',
    
    # Types
    'FlightModeState',
    'SafetyLevel',
    'AdvancedFlightModeState',
    'BlendingCurve',
    'MotorCommand',
    'ThreatMap',
    'VehicleState',
    'MissionContext',
    
    # Energy optimization
    'EnergyOptimizer',
    'VehicleEnergyModel',
    'EnergyOptimizationResult',
    
    # Stall protection
    'ComprehensiveStallProtection',
    'StallDetector',
    'StallProtectionController',
    'StallRecoveryProfile',
    'RecoveryPhase',
]
