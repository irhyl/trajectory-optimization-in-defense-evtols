"""
Control Module - Hierarchical Flight Control System (Phase 2B)

PhD-grade cascaded control architecture with Lyapunov stability guarantees.

CONTROL HIERARCHY:
    Guidance Layer (0.1 Hz) → L-1 waypoint navigation, threat avoidance
         ↓
    Cascaded Loops (100 Hz inner, 10 Hz outer)
         ↓
    Outer Loop (Velocity) → Position/velocity error to attitude commands
    Inner Loop (Attitude) → Attitude error to control torques (quaternion-based)
         ↓
    Allocation Layer → Torques/forces to rotor thrusts + nacelle angles

LYAPUNOV STABILITY: Both inner and outer loops proven asymptotically stable
with exponential convergence.

Phase 2B Core:
- CascadedControlSystem: Main control law with Lyapunov functions
- GuidanceSystem: L-1 guidance + threat avoidance (0.1 Hz)
- TrajectoryGenerator: Smooth velocity profiling

Phase 1 (Legacy, backward compatible):
- PIDController, CascadedPID: Basic cascaded loops
- AttitudeController, RateController: Simplified versions
- Allocation: Thrust mixing
- Flight modes: Mode management

Main entry points:
1. New: CascadedControlSystem with GuidanceSystem (recommended)
2. Legacy: FlightController for backward compatibility
"""

# Phase 2B: New cascaded control + Lyapunov
try:
    from .cascaded_control import (
        CascadedControlSystem,
        InnerLoopGains,
        OuterLoopGains,
        AllocationGains,
        ControllerState,
        ControlMode,
    )
    from .guidance import (
        GuidanceSystem,
        TrajectoryGenerator,
        Waypoint,
        ThreatZone,
        NavigationMode,
        GuidanceState,
    )
except ImportError:
    pass

# Phase 1: Base classes (legacy)
try:
    from .controller_base import (
        ControllerBase,
        PIDController,
        CascadedPID,
        ControllerGains,
        AttitudeCommand,
        MomentCommand,
        VelocityCommand,
        PositionCommand,
    )
except ImportError:
    pass

# Inner loop controllers
from .inner_loop import AttitudeController, RateController

# Outer loop controllers
from .outer_loop import (
    PositionController,
    VelocityController,
    AltitudeController,
    HeadingController,
)

# Guidance layer
from .guidance import (
    TrajectoryTracker,
    PathFollower,
    MissionManager,
    MissionPhase,
)

# Allocation layer
from .allocation import (
    ControlMixer,
    RotorCommand,
    NacelleScheduler,
    NacelleConfig,
)

# Flight modes
from .modes import (
    FlightModeManager,
    FlightMode,
    HoverMode,
    TransitionMode,
    CruiseMode,
)

# Main flight controller
from .flight_controller import (
    FlightController,
    FlightControllerConfig,
    FlightControllerState,
)

# Phase 2B: Motor Control (new)
try:
    from .motor_controller import (
        MotorController,
        MotorControllerOrchestrator,
        MotorModel,
        MotorLimits,
        MotorCommand,
        MotorTelemetry,
        MotorStatus,
        MotorFailureMode,
        MotorControlMode,
        MotorHealthMonitor,
        MotorControllerState,
    )
    from .motor_interface import (
        PWMConverter,
        PWMSignal,
        MotorESCInterface,
        SimulatedESC,
        MotorCommandQueue,
        MotorControlInterface,
        ESCType,
    )
    from .sitl_simulator import (
        SITLSimulatorBridge,
        BasicVehicleSimulator,
        GazeboSITLBridge,
        VehicleState,
        SITLConfig,
        SimulatorMode,
        create_simulator_bridge,
    )
except ImportError:
    pass

__all__ = [
    # Base
    'ControllerBase',
    'PIDController',
    'CascadedPID',
    'ControllerGains',
    'AttitudeCommand',
    'MomentCommand',
    'VelocityCommand',
    'PositionCommand',

    # Inner loop
    'AttitudeController',
    'RateController',

    # Outer loop
    'PositionController',
    'VelocityController',
    'AltitudeController',
    'HeadingController',

    # Guidance
    'TrajectoryTracker',
    'PathFollower',
    'MissionManager',
    'MissionPhase',

    # Allocation
    'ControlMixer',
    'RotorCommand',
    'NacelleScheduler',
    'NacelleConfig',

    # Modes
    'FlightModeManager',
    'FlightMode',
    'HoverMode',
    'TransitionMode',
    'CruiseMode',

    # Main controller
    'FlightController',
    'FlightControllerConfig',
    'FlightControllerState',
    # Phase 2B: Motor Control
    'MotorController',
    'MotorControllerOrchestrator',
    'MotorModel',
    'MotorLimits',
    'MotorCommand',
    'MotorTelemetry',
    'MotorStatus',
    'MotorFailureMode',
    'MotorControlMode',
    'MotorHealthMonitor',
    'MotorControllerState',
    'PWMConverter',
    'PWMSignal',
    'MotorESCInterface',
    'SimulatedESC',
    'MotorCommandQueue',
    'MotorControlInterface',
    'ESCType',]
