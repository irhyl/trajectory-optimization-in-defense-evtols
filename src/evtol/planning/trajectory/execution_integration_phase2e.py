"""
Phase 2E Integration: Execution-to-Controller Bridge

Connects trajectory execution (Phase 2E) with flight control (Phase 2B):
- 50 Hz control loop orchestration
- State feedback from controller to executor
- Command dispatch to motor controllers
- Real-time monitoring and contingency escalation

**Architecture**

```
Phase 2C (Mission Plans)
         ↓
Phase 2D (Trajectory Execution)  
         ↓
Phase 2E (Advanced Control)
  - DifferentialFlatnessAnalyzer
  - OnlineReplanner
  - LearningAdapter
         ↓
Phase 2E Integration ← NEW (THIS MODULE)
  - ExecutionController (50 Hz orchestrator)
  - ControlLoopManager
  - StateSync
         ↓
Phase 2B (Flight Controller)
  - AttitudeController
  - RateController
  - MotorAllocation
         ↓
Vehicle Motors/Propulsion
```

**Control Loop (50 Hz = 20ms cycle)**

```
t=0ms:   Receive current state from Phase 2B
t=1ms:   ├─ Evaluate trajectory at time t
t=2ms:   ├─ Compute reference state (pos, vel, attitude)
t=3ms:   ├─ Synthesize motor thrusts (flatness control)
t=4ms:   ├─ Check contingencies (detect anomalies)
t=5ms:   ├─ Apply learning-based thresholds
t=6ms:   ├─ Monitor for replanning triggers
t=10ms:  └─ Dispatch commands to Phase 2B
t=11ms:  Receive execution state feedback
t=15ms:  Logging/telemetry (async)
t=20ms:  Loop complete, repeat (50 Hz)
```

**References**

[1] Beard & McLain (2012): "Small Unmanned Aircraft: Theory and Practice" Ch. 11-12
[2] Mellinger et al. (2010): "Minimum snap trajectory generation and control"
[3] Mahony et al. (2012): "Multirotor Aerial Vehicles: Modeling, Estimation, and Control"
"""

from __future__ import annotations
import logging
import numpy as np
import time
import threading
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Callable, Dict, Any, Tuple
from collections import deque

from ..core.state import State, Pose, Velocity
from .execution_engine import TrajectoryExecutionEngine
from .differential_flatness_analyzer import DifferentialFlatnessAnalyzer
from .online_replanning_interface import OnlineReplanner
from .learning_adaptation import ParameterLearner
from .contingency_management import ContingencyLevel

logger = logging.getLogger(__name__)


class LoopPhase(Enum):
    """Phase within 50 Hz control cycle."""
    STATE_READ = "state_read"              # Read state from controller
    TRAJECTORY_EVAL = "trajectory_eval"    # Evaluate trajectory
    CONTROL_SYNTHESIZE = "control_synthesize"  # Compute commands
    ANOMALY_CHECK = "anomaly_check"        # Detect anomalies
    LEARNING_UPDATE = "learning_update"    # Apply learned parameters
    REPLAN_CHECK = "replan_check"          # Check replanning triggers
    COMMAND_DISPATCH = "command_dispatch"  # Send to Phase 2B
    TELEMETRY = "telemetry"                # Logging (async)


@dataclass
class ControlCycleMetrics:
    """Performance metrics for single control cycle."""
    cycle_number: int
    timestamp: datetime
    
    # Timing breakdown (ms)
    state_read_ms: float = 0.0
    trajectory_eval_ms: float = 0.0
    control_synthesis_ms: float = 0.0
    anomaly_check_ms: float = 0.0
    learning_update_ms: float = 0.0
    replan_check_ms: float = 0.0
    command_dispatch_ms: float = 0.0
    total_cycle_ms: float = 0.0
    
    # State feedback
    position_error: float = 0.0             # m
    velocity_tracking_error: float = 0.0    # m/s
    attitude_error: float = 0.0             # deg
    
    # Control outputs
    total_thrust: float = 0.0               # N
    motor_saturation_ratio: float = 0.0     # [0, 1]
    
    # Contingencies
    active_anomalies: int = 0
    contingency_level: str = "NOMINAL"
    replan_requested: bool = False
    
    def cycle_overrun(self, budget_ms: float = 20.0) -> bool:
        """Check if cycle exceeded time budget."""
        return self.total_cycle_ms > budget_ms


@dataclass
class ControlCommand:
    """Command to Phase 2B flight controller."""
    motor_thrusts: np.ndarray              # [T0, T1, ...] (N)
    motor_indices: np.ndarray              # [0, 1, ...]
    timestamp: float                        # Command issue time (s)
    
    # Optional feedforward
    attitude_desired: Optional[np.ndarray] = None  # Desired rotation matrix
    angular_rate_desired: Optional[np.ndarray] = None  # Desired ω (body frame)
    acceleration_feedforward: Optional[np.ndarray] = None  # Feed-forward accel
    
    def to_pwm(self, thrust_to_pwm_func: Callable) -> np.ndarray:
        """Convert thrusts to PWM signals (1000-2000 µs typical)."""
        return thrust_to_pwm_func(self.motor_thrusts)


@dataclass
class ExecutionState:
    """Aggregate execution state for logging/monitoring."""
    cycle_number: int
    mission_time: float                     # s since mission start
    vehicle_state: State
    reference_state: State
    
    # Trajectory status
    trajectory_progress: float              # [0, 1] percent complete
    trajectory_feasible: bool
    
    # Control status
    control_command: ControlCommand
    control_mode: str                       # FEEDFORWARD_ONLY, FEEDBACK_LINEAR, LQR_OPTIMAL
    
    # Contingency status
    anomaly_detections: Dict[str, float]    # Anomaly type → severity
    contingency_level: str
    contingency_action: str                 # NONE, ADJUST_HEADING, etc.
    
    # Learning status
    learning_params_active: bool
    learning_confidence: float              # [0, 1]


class ControlLoopManager:
    """
    Manages 50 Hz control loop orchestration.
    
    Interfaces:
    - Input: Current state from Phase 2B (via callback)
    - Process: Phase 2E components (execution, control, learning)
    - Output: Motor commands to Phase 2B (via callback)
    """
    
    def __init__(
        self,
        execution_engine: TrajectoryExecutionEngine,
        flatness_analyzer: DifferentialFlatnessAnalyzer,
        replanner: OnlineReplanner,
        learning_adapter: ParameterLearner,
        control_rate_hz: int = 50,
    ):
        """
        Initialize control loop manager.
        
        Args:
            execution_engine: Phase 2D execution orchestrator
            flatness_analyzer: Phase 2E control synthesizer
            replanner: Phase 2E online replanning
            learning_adapter: Phase 2E learning sub-system
            control_rate_hz: Control loop frequency
        """
        self.execution = execution_engine
        self.flatness = flatness_analyzer
        self.replanner = replanner
        self.learning = learning_adapter
        
        self.control_rate = control_rate_hz
        self.cycle_time = 1.0 / control_rate_hz
        
        self.cycle_count = 0
        self.mission_start_time = None
        self.running = False
        
        # State feedback queue (for async updates)
        self.state_queue = deque(maxlen=2)
        self.command_callbacks = []
        self.state_callbacks = []
        
        # Metrics history
        self.metrics_history = deque(maxlen=1000)  # Last 20 seconds @ 50 Hz
        self.last_cycle_metrics: Optional[ControlCycleMetrics] = None
    
    def add_state_callback(self, callback: Callable[[State], None]):
        """Register callback for state updates from Phase 2B."""
        self.state_callbacks.append(callback)
    
    def add_command_callback(self, callback: Callable[[ControlCommand], None]):
        """Register callback for command dispatch to Phase 2B."""
        self.command_callbacks.append(callback)
    
    def on_state_received(self, state: State):
        """Called when Phase 2B publishes current state."""
        self.state_queue.append(state)
    
    def initialize_mission(
        self,
        mission_id: str,
        trajectory_waypoints: np.ndarray,
        smoothing_constraints: Optional[Any] = None,
    ):
        """
        Initialize mission for execution.
        
        Args:
            mission_id: Unique mission identifier
            trajectory_waypoints: [N, 3] waypoint array
            smoothing_constraints: Optional trajectory constraints
        """
        logger.info(f"Initializing mission {mission_id} with {len(trajectory_waypoints)} waypoints")
        
        # Load into execution engine
        self.execution.load_mission_trajectory(
            trajectory_waypoints,
            smoothing_constraints or {}
        )
        
        # Initialize replanner
        self.replanner.reset()
        
        # Reset learning state
        self.learning.recorder.start_mission(mission_id)
        
        self.mission_start_time = time.time()
        self.cycle_count = 0
        self.metrics_history.clear()
    
    def finalize_mission(self, success: bool, mission_metrics: Optional[Any] = None):
        """End mission and record learning data."""
        if mission_metrics and self.learning.recorder:
            self.learning.recorder.finalize_mission(mission_metrics)
        
        logger.info(f"Mission completed: success={success}, cycles={self.cycle_count}")
    
    def run_cycle(self) -> Tuple[ControlCommand, ControlCycleMetrics]:
        """
        Execute single 50 Hz control cycle.
        
        Returns:
            (ControlCommand to motor controller, cycle metrics)
        """
        cycle_start_time = time.time()
        metrics = ControlCycleMetrics(
            cycle_number=self.cycle_count,
            timestamp=datetime.now(),
        )
        
        try:
            # ===== PHASE 1: State Read (1 ms budget) =====
            phase_start = time.time()
            
            # Get latest state from queue
            if not self.state_queue:
                logger.warning("No state available, using cached")
                return None, metrics
            
            current_state = self.state_queue[-1]
            metrics.state_read_ms = (time.time() - phase_start) * 1000
            
            # ===== PHASE 2: Trajectory Evaluation (1 ms budget) =====
            phase_start = time.time()
            
            mission_time = self._get_mission_time()
            ref_state = self.execution.get_reference_at_time(mission_time)
            
            # Progress tracking
            trajectory_duration = self.execution.trajectory_duration if hasattr(self.execution, 'trajectory_duration') else 100.0
            progress = mission_time / trajectory_duration
            
            metrics.trajectory_eval_ms = (time.time() - phase_start) * 1000
            
            # ===== PHASE 3: Control Synthesis (3 ms budget) =====
            phase_start = time.time()
            
            motor_command = self.flatness.synthesize_control(
                trajectory=self.execution.current_trajectory,
                current_state=current_state,
                time=mission_time,
            )
            
            metrics.control_synthesis_ms = (time.time() - phase_start) * 1000
            metrics.total_thrust = np.sum(motor_command.thrusts)
            max_thrust = self.flatness.vehicle_config.propulsion.max_thrust_per_motor
            metrics.motor_saturation_ratio = np.max(motor_command.thrusts) / max_thrust
            
            # ===== PHASE 4: Anomaly/Contingency Check (2 ms budget) =====
            phase_start = time.time()
            
            contingency_event = self.execution.update_execution(
                actual_state=current_state,
                battery_energy=999.0,  # TODO: Get from state
                motor_temp=50.0,       # TODO: Get from state
                threat_level=0.0,      # TODO: Get from perception
            )
            
            if contingency_event != ContingencyLevel.NOMINAL:
                metrics.active_anomalies += 1
                metrics.contingency_level = contingency_event.name
                self.learning.recorder.record_anomaly(
                    anomaly_type="CONTINGENCY",
                    severity=0.7,
                    threshold_crossed=True,
                    response_action="UNKNOWN",
                    environmental_context={},
                )
            
            metrics.anomaly_check_ms = (time.time() - phase_start) * 1000
            
            # ===== PHASE 5: Learning Update (1 ms budget) =====
            phase_start = time.time()
            
            # Periodically apply learned parameters (every 100 cycles = 2s)
            if self.cycle_count % 100 == 0 and self.learning.learned_params:
                # Update detector weights (conservative blend)
                # TODO: Integrate with contingency manager
                pass
            
            metrics.learning_update_ms = (time.time() - phase_start) * 1000
            
            # ===== PHASE 6: Replanning Check (1 ms budget) =====
            phase_start = time.time()
            
            # Check replanning triggers
            should_replan = (
                metrics.contingency_level != "NOMINAL" or
                metrics.motor_saturation_ratio > 0.95 or
                progress > 0.9  # Near goal
            )
            
            if should_replan and not self.replanner.current_replan:
                metrics.replan_requested = True
                logger.info("Replanning triggered")
                # TODO: Launch async replan request
            
            metrics.replan_check_ms = (time.time() - phase_start) * 1000
            
            # ===== PHASE 7: Command Dispatch (1 ms budget) =====
            phase_start = time.time()
            
            control_cmd = ControlCommand(
                motor_thrusts=motor_command.thrusts,
                motor_indices=motor_command.motor_indices,
                timestamp=current_state.timestamp,
                attitude_desired=None,  # Optional feedforward
                acceleration_feedforward=None,
            )
            
            # Dispatch to Phase 2B callbacks
            for callback in self.command_callbacks:
                callback(control_cmd)
            
            metrics.command_dispatch_ms = (time.time() - phase_start) * 1000
            
            # ===== Compute errors =====
            pos_error = np.linalg.norm(current_state.pose.position - ref_state.pose.position)
            vel_error = np.linalg.norm(current_state.velocity.linear - ref_state.velocity.linear)
            
            metrics.position_error = pos_error
            metrics.velocity_tracking_error = vel_error
            metrics.total_cycle_ms = (time.time() - cycle_start_time) * 1000
            
            # Check cycle overrun
            if metrics.cycle_overrun():
                logger.warning(f"Control cycle overrun: {metrics.total_cycle_ms:.2f}ms > 20ms")
            
            # Store metrics
            self.metrics_history.append(metrics)
            self.last_cycle_metrics = metrics
            
            self.cycle_count += 1
            
            return control_cmd, metrics
            
        except Exception as e:
            logger.error(f"Control cycle exception: {e}", exc_info=True)
            return None, metrics
    
    def _get_mission_time(self) -> float:
        """Get elapsed time since mission start."""
        if not self.mission_start_time:
            return 0.0
        return time.time() - self.mission_start_time
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """Return control loop diagnostics."""
        if not self.last_cycle_metrics:
            return {}
        
        recent_cycles = list(self.metrics_history)[-100:]  # Last 100 cycles (2 sec)
        
        cycle_times = [m.total_cycle_ms for m in recent_cycles]
        
        return {
            "cycle_count": self.cycle_count,
            "current_rate_hz": 1000 / np.mean(cycle_times) if cycle_times else 0,
            "cycle_time_avg_ms": np.mean(cycle_times) if cycle_times else 0,
            "cycle_time_max_ms": np.max(cycle_times) if cycle_times else 0,
            "cycle_overruns": sum(1 for m in recent_cycles if m.cycle_overrun()),
            "position_error_avg_m": np.mean([m.position_error for m in recent_cycles]),
            "position_error_max_m": np.max([m.position_error for m in recent_cycles]),
            "motor_saturation_avg": np.mean([m.motor_saturation_ratio for m in recent_cycles]),
            "contingency_events": sum(1 for m in recent_cycles if m.active_anomalies > 0),
            "replans_requested": sum(1 for m in recent_cycles if m.replan_requested),
        }


class ExecutionController:
    """
    High-level mission execution controller.
    
    Wraps control loop manager with thread safety and mission lifecycle.
    """
    
    def __init__(self, control_loop: ControlLoopManager):
        self.loop = control_loop
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
    
    def start_mission(self, mission_id: str, waypoints: np.ndarray):
        """Start executing mission."""
        self.loop.initialize_mission(mission_id, waypoints)
        self.running = True
        
        logger.info(f"Starting mission {mission_id}")
    
    def stop_mission(self, success: bool = True):
        """Stop mission execution."""
        self.running = False
        self.loop.finalize_mission(success)
        
        logger.info(f"Stopped mission, success={success}")
    
    def execute_sync(self) -> Tuple[ControlCommand, ControlCycleMetrics]:
        """
        Execute single control cycle (blocking).
        
        Call this from 50 Hz external loop.
        """
        with self._lock:
            return self.loop.run_cycle()
    
    def execute_async(self, duration_s: float = 300.0):
        """
        Execute mission for fixed duration in background thread.
        
        Args:
            duration_s: Mission duration (s)
        """
        def mission_loop():
            start_time = time.time()
            
            while self.running and (time.time() - start_time) < duration_s:
                cmd, metrics = self.loop.run_cycle()
                
                # Sleep to maintain 50 Hz rate
                cycle_time = time.time() - (start_time + metrics.cycle_number * self.loop.cycle_time)
                sleep_time = self.loop.cycle_time - cycle_time
                if sleep_time > 0:
                    time.sleep(sleep_time)
            
            self.stop_mission(success=True)
        
        self.thread = threading.Thread(target=mission_loop, daemon=True)
        self.thread.start()
