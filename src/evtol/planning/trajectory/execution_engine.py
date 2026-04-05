"""
Phase 2D - Complete Trajectory Execution System

Integrates smooth trajectory generation, reference generation, and
contingency management for full mission execution pipeline.

Pipeline Flow
=============

1. Mission Planning (Phase 2C)
   ↓
   Optimal trajectory waypoints
   ↓
2. Trajectory Smoothing (Phase 2D-1)
   ↓
   Smooth, C² continuous curves
   ↓
3. Time Parametrization (Phase 2D-2)
   ↓
   Reference trajectory with time profile
   ↓
4. Reference Generation (Phase 2D-2)
   ↓
   Controller-rate reference signals
   ↓
5. Control Execution (Phase 2B)
   + Contingency Monitoring (Phase 2D-3)
   ↓
6. Anomaly Detection → Response (Phase 2D-3)
   ↓
   (Success) Mission Complete
   (Issue) → Replan via Phase 2C

Author: Defense eVTOL Research Team
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from enum import Enum, auto
from datetime import datetime, timedelta
import logging

from .smooth_trajectory_generator import (
    TrajectorySmootherEngine,
    SmoothingMethod,
    SmoothingConstraints,
    SmoothedTrajectory,
    TrajectoryReferenceGenerator,
)
from .contingency_management import (
    AnomalyDetector,
    ContingencyTrigger,
    ContingencyManager,
    ContingencyLevel,
    ResponseAction,
)
from ..core.trajectory import Trajectory
from ..core.state import State, Pose, Velocity

logger = logging.getLogger(__name__)


class ExecutionPhase(Enum):
    """Execution phase of mission."""
    IDLE = auto()
    SETUP = auto()
    EXECUTING = auto()
    CONTINGENCY = auto()
    COMPLETED = auto()
    ABORTED = auto()


@dataclass
class ExecutionMetrics:
    """Metrics for trajectory execution."""
    
    # Timing
    phase_start_time: datetime = field(default_factory=datetime.now)
    total_elapsed: float = 0.0  # s
    
    # Tracking
    mean_tracking_error: float = 0.0  # m
    max_tracking_error: float = 0.0  # m
    cumulative_error: float = 0.0  # m·s
    
    # Energy
    energy_consumed: float = 0.0  # Wh
    energy_predicted: float = 0.0  # Wh
    energy_error_percent: float = 0.0  # %
    
    # Performance
    control_cycles: int = 0
    anomalies_detected: int = 0
    contingencies_triggered: int = 0
    
    # Status
    phase: ExecutionPhase = ExecutionPhase.IDLE
    mission_progress: float = 0.0  # [0, 1]


class TrajectoryExecutionEngine:
    """
    Master execution engine for trajectory following.
    
    Orchestrates reference generation, control execution,
    and contingency management.
    """
    
    def __init__(
        self,
        smoothing_method: SmoothingMethod = SmoothingMethod.BSPLINE_QUINTIC,
        controller_update_rate: float = 50.0,  # Hz
    ):
        """
        Initialize execution engine.
        
        Args:
            smoothing_method: Trajectory smoothing method
            controller_update_rate: Controller update frequency [Hz]
        """
        self.smoothing_method = smoothing_method
        self.update_rate = controller_update_rate
        self.dt = 1.0 / controller_update_rate
        
        # Components (initialized on demand)
        self.smoother: TrajectorySmootherEngine | None = None
        self.ref_gen: TrajectoryReferenceGenerator | None = None
        self.contingency_mgr: ContingencyManager | None = None
        
        # Current trajectory
        self.current_trajectory: SmoothedTrajectory | None = None
        self.reference_trajs: list[State] = []
        
        # Execution state
        self.current_phase = ExecutionPhase.IDLE
        self.current_time = 0.0
        self.metrics = ExecutionMetrics()
        
        logger.info(f"TrajectoryExecutionEngine initialized: {smoothing_method.value} @ {controller_update_rate} Hz")
    
    def load_mission_trajectory(
        self,
        waypoints: np.ndarray,
        constraints: SmoothingConstraints | None = None,
    ) -> SmoothedTrajectory:
        """
        Load and prepare mission trajectory.
        
        Args:
            waypoints: Mission waypoints from Phase 2C
            constraints: Optional smoothing constraints
            
        Returns:
            Smoothed trajectory ready for execution
        """
        logger.info(f"Loading mission trajectory: {len(waypoints)} waypoints")
        
        # Initialize smoother if needed
        if self.smoother is None:
            self.smoother = TrajectorySmootherEngine(
                method=self.smoothing_method,
                constraints=constraints or SmoothingConstraints(),
            )
        
        # Smooth trajectory
        self.current_trajectory = self.smoother.smooth_waypoints(waypoints)
        
        # Generate references
        self.ref_gen = TrajectoryReferenceGenerator(
            self.current_trajectory,
            controller_update_rate=self.update_rate,
        )
        
        self.reference_trajs = self.ref_gen.get_all_references()[1]
        
        logger.info(f"Trajectory loaded: {self.current_trajectory.total_distance:.0f}m, "
                   f"{self.current_trajectory.total_time:.1f}s")
        
        return self.current_trajectory
    
    def initialize_execution(self) -> bool:
        """
        Initialize ready-to-execute state.
        
        Performs pre-flight checks and validation.
        
        Returns:
            True if ready to execute
        """
        if self.current_trajectory is None:
            logger.error("No trajectory loaded")
            return False
        
        # Initialize contingency manager
        def dummy_predictor(ref_state: State) -> State:
            return ref_state
        
        detector = AnomalyDetector(dummy_predictor)
        trigger = ContingencyTrigger()
        self.contingency_mgr = ContingencyManager(detector, trigger)
        
        # Reset metrics
        self.metrics = ExecutionMetrics(
            phase=ExecutionPhase.SETUP,
            phase_start_time=datetime.now(),
        )
        
        self.current_time = 0.0
        
        logger.info("Execution engine ready")
        return True
    
    def start_execution(self) -> bool:
        """
        Start trajectory execution.
        
        Returns:
            True if execution started successfully
        """
        if self.contingency_mgr is None:
            if not self.initialize_execution():
                return False
        
        self.current_phase = ExecutionPhase.EXECUTING
        self.metrics.phase = ExecutionPhase.EXECUTING
        
        logger.info("Trajectory execution started")
        return True
    
    def get_reference_at_time(self, t: float) -> State:
        """
        Get reference trajectory state at time t.
        
        Args:
            t: Time [s]
            
        Returns:
            Reference state for controller
        """
        if self.ref_gen is None:
            raise RuntimeError("Reference generator not initialized")
        
        # Clamp to trajectory duration
        t_clamped = np.clip(t, 0, self.current_trajectory.total_time)
        
        return self.ref_gen.get_reference_at_time(t_clamped)
    
    def update_execution(
        self,
        actual_state: State,
        battery_energy: float,
        motor_temp: float = 70.0,
        threat_level: float = 0.1,
    ) -> ContingencyLevel:
        """
        Update execution state (call once per control cycle).
        
        Monitors tracking, energy, and thermal performance.
        Triggers contingency responses if needed.
        
        Args:
            actual_state: Current vehicle state from sensors
            battery_energy: Battery energy remaining [Wh]
            motor_temp: Motor temperature [°C]
            threat_level: Current threat exposure [0, 1]
            
        Returns:
            Current contingency level
        """
        if self.current_phase != ExecutionPhase.EXECUTING:
            return ContingencyLevel.NOMINAL
        
        # Get reference for current time
        ref_state = self.get_reference_at_time(self.current_time)
        
        # Estimate predicted energy (simple model: linear interpolation)
        mission_progress = self.current_time / max(self.current_trajectory.total_time, 1.0)
        predicted_energy_consumed = mission_progress * self.current_trajectory.total_distance * 10  # 10 Wh/m
        
        # Update contingency manager
        event = self.contingency_mgr.update(
            actual_state=actual_state,
            reference_state=ref_state,
            battery_energy=battery_energy,
            predicted_energy=predicted_energy_consumed,
            motor_temp=motor_temp,
            threat_level=threat_level,
        )
        
        # Update metrics
        self._update_metrics(actual_state, ref_state, battery_energy)
        
        # Handle contingency response
        if event.level != ContingencyLevel.NOMINAL:
            self._handle_contingency(event)
        
        self.current_time += self.dt
        self.metrics.control_cycles += 1
        
        return event.level
    
    def _update_metrics(
        self,
        actual_state: State,
        reference_state: State,
        battery_energy: float,
    ):
        """Update execution metrics."""
        # Tracking error
        error = np.linalg.norm(
            actual_state.pose.position - reference_state.pose.position
        )
        
        self.metrics.mean_tracking_error = (
            (self.metrics.mean_tracking_error * (self.metrics.control_cycles - 1) + error) /
            self.metrics.control_cycles
        )
        self.metrics.max_tracking_error = max(self.metrics.max_tracking_error, error)
        self.metrics.cumulative_error += error * self.dt
        
        # Energy
        mission_progress = self.current_time / max(self.current_trajectory.total_time, 1.0)
        self.metrics.mission_progress = mission_progress
        
        # Elapsed time
        self.metrics.total_elapsed = self.current_time
    
    def _handle_contingency(self, event):
        """Handle contingency event."""
        logger.warning(f"Contingency triggered: {event.level.name} - {event.recommended_action.name}")
        
        self.metrics.contingencies_triggered += 1
        
        if event.recommended_action == ResponseAction.REDUCE_SPEED:
            logger.info("Reducing speed command issued")
        elif event.recommended_action == ResponseAction.CLIMB_ALTITUDE:
            logger.info("Climb altitude command issued")
        elif event.recommended_action == ResponseAction.RETURN_TO_BASE:
            self.current_phase = ExecutionPhase.CONTINGENCY
            self.metrics.phase = ExecutionPhase.CONTINGENCY
            logger.warning("RTB initiated")
        elif event.recommended_action == ResponseAction.REPLAN_MISSION:
            self.current_phase = ExecutionPhase.CONTINGENCY
            logger.warning("Replanning triggered")
    
    def complete_execution(self, success: bool = True) -> ExecutionMetrics:
        """
        Complete trajectory execution.
        
        Args:
            success: Whether mission completed successfully
            
        Returns:
            Execution metrics summary
        """
        self.current_phase = ExecutionPhase.COMPLETED if success else ExecutionPhase.ABORTED
        self.metrics.phase = self.current_phase
        
        logger.info(
            f"Execution complete: {self.current_phase.name}, "
            f"Progress: {self.metrics.mission_progress:.1%}, "
            f"Error: {self.metrics.mean_tracking_error:.1f}m"
        )
        
        return self.metrics
    
    def get_execution_status(self) -> dict:
        """Get current execution status."""
        return {
            'phase': self.current_phase.name,
            'time': self.current_time,
            'total_time': self.current_trajectory.total_time if self.current_trajectory else 0,
            'progress': self.metrics.mission_progress,
            'tracking_error': self.metrics.mean_tracking_error,
            'cycles': self.metrics.control_cycles,
            'contingencies': self.metrics.contingencies_triggered,
        }


class ExecutionValidator:
    """
    Validate trajectory before execution.
    
    Performs pre-flight checks to ensure safety and feasibility.
    """
    
    @staticmethod
    def validate_trajectory(
        trajectory: SmoothedTrajectory,
        constraints: SmoothingConstraints,
    ) -> tuple[bool, list[str]]:
        """
        Validate trajectory against constraints.
        
        Args:
            trajectory: Smoothed trajectory to validate
            constraints: Safety constraints
            
        Returns:
            (valid, messages)
        """
        messages = []
        valid = True
        
        # Check duration
        if trajectory.total_time <= 0:
            messages.append("Invalid trajectory duration")
            valid = False
        
        # Check distance
        if trajectory.total_distance <= 0:
            messages.append("Invalid trajectory distance")
            valid = False
        
        # Check speed profile
        max_speed_profile = np.max(trajectory.speed_profile)
        if max_speed_profile > constraints.max_speed:
            messages.append(f"Speed profile exceeds limit: {max_speed_profile:.1f} m/s")
            valid = False
        
        return valid, messages
