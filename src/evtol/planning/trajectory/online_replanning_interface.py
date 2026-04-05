"""
Online Replanning Interface

Mid-flight trajectory replanning when contingencies occur or plan becomes suboptimal:
- Trigger detection: threshold crossings, threat changes, environmental shifts
- Replanning request: hand-off current state → Phase 2C mission planner
- Trajectory hand-off: smooth transition between old and new plans
- Feasibility checking: ensure new plan can be executed immediately

**Architecture**

```
Phase 2E Execution                Phase 2C Mission Planning
┌─────────────────────────┐     ┌──────────────────────────┐
│ ExecutionEngine         │     │ MissionPlanner           │
│  - execute_trajectory() │◄───►│  - optimize_trajectory() │
│  - detected anomaly→    │     │  - handles replanning    │
│    trigger replan       │     │                          │
└─────────────────────────┘     └──────────────────────────┘
         ▲
         │ ReplanRequest
         │
    OnlineReplanner
    ┌────────────  ┐
    │ - trigger    │
    │ - feasibility|
    │ - hand-off   │
    └───────────── ┘
```

**State Capture for Replanning**

When anomaly occurs (e.g., energy drop, threat):
1. Capture current state: position, velocity, remaining energy, time
2. Update goal constraints: adjusted energy budget, threat avoidance
3. Request new plan from Phase 2C with same timeline/objectives

**Trajectory Hand-off**

Smooth switch from old trajectory to new:
1. Plan switch time t_switch (typically 2-3 seconds for smoothing)
2. Blend old trajectory (t ≤ t_switch) with new
3. Ensure continuity: position, velocity, acceleration match
4. Execute blended trajectory during replanning compute

**References**

[1] Kar et al. (2017): "Real-time motion planning with temporal logic constraints"
    IEEE TASE, https://doi.org/10.1109/TASE.2017.7973385

[2] Wzorek & Doherty (2012): "Receding horizon task execution and replanning"
    IROS, https://doi.org/10.1109/IROS.2012.6385820

[3] Goerzen et al. (2010): "A survey of motion planning algorithms"
    Journal of Intelligent & Robotic Systems
"""

from __future__ import annotations
import logging
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from typing import Optional, List, Tuple, Dict, Any, Callable  # noqa: F401
import threading
import time

from ..core.state import State, Pose, Velocity

logger = logging.getLogger(__name__)


class ReplanTrigger(Enum):
    """Why replanning was triggered."""
    ENERGY_CRITICAL = "energy_critical"           # Battery dropping
    THREAT_AVOIDANCE = "threat_avoidance"         # New threat detected
    ENVIRONMENT_CHANGE = "environment_change"     # Wind, weather shift
    TRACKING_DEGRADED = "tracking_degraded"       # Persistent error
    TRAJECTORY_INFEASIBLE = "trajectory_infeasible"  # Cannot execute plan
    USER_REQUEST = "user_request"                 # Manual trigger
    CONTINGENCY_RESPONSE = "contingency_response" # From Phase 2D


@dataclass
class ReplanRequest:
    """Request for mid-flight trajectory replanning."""
    trigger: ReplanTrigger
    current_state: State
    time_remaining: float                  # Seconds until original goal
    energy_remaining: float                # Wh
    
    # Constraints for replanning
    threat_update: Optional[np.ndarray] = None  # Updated threat field
    wind_update: Optional[np.ndarray] = None    # Updated wind field
    goal_update: Optional[np.ndarray] = None    # Updated goal position
    
    # Urgency
    priority: int = 5                      # [1-10] 1=low, 10=immediate
    allow_reduced_coverage: bool = False   # Can skip waypoints?
    max_delay: float = 5.0                 # Max planning time (s)
    
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ReplanResult:
    """Result of replanning (success/failure)."""
    success: bool
    new_trajectory: Optional[Any] = None   # SmoothedTrajectory object
    new_waypoints: Optional[np.ndarray] = None
    planning_time: float = 0.0             # Time spent planning (s)
    quality_metric: float = 0.0            # [0, 1] plan quality vs original
    
    # Hand-off parameters
    switch_time: float = 0.0               # When to switch (s)
    blend_window: float = 2.0              # Blending duration (s)
    
    error_message: str = ""


@dataclass
class BlendedTrajectory:
    """Smooth blend between old and new trajectories."""
    old_trajectory: Any                    # Previous trajectory
    new_trajectory: Any                    # New trajectory
    switch_time: float                     # When transition occurs (absolute time)
    blend_duration: float                  # Duration of blending [1-3s typical]
    
    def evaluate_at_time(self, t: float) -> np.ndarray:
        """Get position from blended trajectory."""
        if t < self.switch_time:
            # Pre-switch: use old trajectory
            return self.old_trajectory.evaluate_at_time(t)
        elif t > self.switch_time + self.blend_duration:
            # Post-blend: use new trajectory
            phase = (t - self.switch_time - self.blend_duration)
            return self.new_trajectory.evaluate_at_time(phase)
        else:
            # Blend region
            blend_param = (t - self.switch_time) / self.blend_duration
            
            # Old trajectory at blend end
            old_pos = self.old_trajectory.evaluate_at_time(self.switch_time + self.blend_duration)
            
            # New trajectory at blend start (remapped time)
            new_pos = self.new_trajectory.evaluate_at_time(0.0 + blend_param * self.blend_duration)
            
            # Smooth blend (cubic Hermite)
            s = blend_param
            h00 = 2*s**3 - 3*s**2 + 1
            h10 = s**3 - 2*s**2 + s
            h01 = -2*s**3 + 3*s**2
            h11 = s**3 - s**2
            
            # Velocity at switch point (continuity)
            old_vel = self.old_trajectory.evaluate_at_time(self.switch_time + self.blend_duration - 0.01)
            old_vel = (old_pos - old_vel) / 0.01
            
            new_vel_initial = 0.01  # Small initial velocity from new plan
            
            blended_pos = (
                h00 * old_pos + h10 * old_vel +
                h01 * new_pos + h11 * new_vel_initial
            )
            
            return blended_pos


class FeasibilityChecker:
    """Verify new trajectory can be executed with current resources."""
    
    @staticmethod
    def check_energy_feasibility(
        waypoints: np.ndarray,
        energy_remaining: float,
        vehicle_mass: float,
        cruise_speed: float = 15.0,
        reserve_margin: float = 0.1,
    ) -> Tuple[bool, float]:
        """
        Check if trajectory can be completed with remaining energy.
        
        Args:
            waypoints: Path waypoints [m]
            energy_remaining: Battery energy [Wh]
            vehicle_mass: Vehicle mass [kg]
            cruise_speed: Typical cruise speed [m/s]
            reserve_margin: Margin fraction (0.1 = 10% reserve)
        
        Returns:
            (feasible, useable_energy)
        """
        # Estimate energy consumption
        total_distance = np.sum(np.linalg.norm(np.diff(waypoints, axis=0), axis=1))
        
        # Rough model: E = m·g·h + drag (simplified)
        height_change = np.abs(np.diff(waypoints[:, 2])).sum()
        potential_energy = vehicle_mass * 9.81 * height_change / 1000 / 3.6  # Convert to Wh
        
        # Kinetic energy from speed changes
        speeds = np.linalg.norm(np.diff(waypoints, axis=0), axis=1) / 10.0  # Rough estimate
        kinetic_energy = np.sum(vehicle_mass * speeds**2 / 2) / 1000 / 3.6
        
        # Total estimated consumption (with margin for losses)
        estimated_consumption = (potential_energy + kinetic_energy) * 1.3
        
        useable_energy = energy_remaining * (1.0 - reserve_margin)
        feasible = estimated_consumption <= useable_energy
        
        logger.info(
            f"Energy feasibility: required={estimated_consumption:.1f}Wh, "
            f"available={useable_energy:.1f}Wh, feasible={feasible}"
        )
        
        return feasible, useable_energy
    
    @staticmethod
    def check_time_feasibility(
        waypoints: np.ndarray,
        time_remaining: float,
        cruise_speed: float = 15.0,
        min_speed: float = 5.0,
        max_speed: float = 25.0,
    ) -> Tuple[bool, float]:
        """
        Check if trajectory can be completed in remaining time.
        
        Args:
            waypoints: Path waypoints [m]
            time_remaining: Time available [s]
            cruise_speed: Nominal speed [m/s]
            min_speed: Minimum speed [m/s]
            max_speed: Maximum speed [m/s]
        
        Returns:
            (feasible, time_required)
        """
        # Distance along path
        distances = np.linalg.norm(np.diff(waypoints, axis=0), axis=1)
        total_distance = np.sum(distances)
        
        # Estimate time at different speeds
        time_at_cruise = total_distance / cruise_speed
        time_at_max = total_distance / max_speed
        
        # Use max speed if time constrained
        required_speed = total_distance / time_remaining if time_remaining > 0 else cruise_speed
        feasible = max_speed >= required_speed >= min_speed
        
        logger.info(
            f"Time feasibility: required_time={time_at_cruise:.1f}s, "
            f"available={time_remaining:.1f}s, feasible={feasible}"
        )
        
        return feasible, time_at_cruise


class OnlineReplanner:
    """
    Orchestrates mid-flight replanning with Phase 2C mission planner.
    
    **Workflow**:
    1. Detect trigger (anomaly / environment change)
    2. Capture current state
    3. Request replanning from Phase 2C
    4. While waiting: execute blended trajectory
    5. On completion: switch to new plan
    """
    
    def __init__(
        self,
        mission_planner_callback: Optional[Callable] = None,
        max_replans_per_mission: int = 3,
        replan_timeout: float = 10.0,
    ):
        """
        Initialize online replanner.
        
        Args:
            mission_planner_callback: Function to call for replanning
            max_replans_per_mission: Safety limit on replans
            replan_timeout: Max time to wait for plan (s)
        """
        self.mission_planner = mission_planner_callback
        self.max_replans = max_replans_per_mission
        self.replan_timeout = replan_timeout
        
        self.feasibility_checker = FeasibilityChecker()
        self.replan_count = 0
        self.last_replan_time = None
        
        self.current_replan: Optional[ReplanResult] = None
        self._replan_lock = threading.Lock()
    
    def request_replan(
        self,
        trigger: ReplanTrigger,
        current_state: State,
        time_remaining: float,
        energy_remaining: float,
        threat_field: Optional[Any] = None,
        wind_field: Optional[Any] = None,
        goal_position: Optional[np.ndarray] = None,
        priority: int = 5,
    ) -> Optional[ReplanRequest]:
        """
        Generate replanning request from current execution state.
        
        Args:
            trigger: Why replanning triggered
            current_state: Current vehicle state
            time_remaining: Seconds until goal
            energy_remaining: Wh
            threat_field: Updated threat model
            wind_field: Updated wind model
            goal_position: Updated goal (if moving target)
            priority: Urgency [1-10]
        
        Returns:
            ReplanRequest if feasible, None if max replans exceeded
        """
        # Safety check: limit replans
        if self.replan_count >= self.max_replans:
            logger.warning(f"Max replans ({self.max_replans}) exceeded, denying replan")
            return None
        
        # Rate limiting: don't replan too frequently
        if self.last_replan_time:
            time_since = time.time() - self.last_replan_time
            if time_since < 10.0:  # Minimum 10s between replans
                logger.warning(f"Replan rate limit: {time_since:.1f}s since last replan")
                return None
        
        # Build request
        request = ReplanRequest(
            trigger=trigger,
            current_state=current_state,
            time_remaining=time_remaining,
            energy_remaining=energy_remaining,
            threat_update=threat_field if threat_field else None,
            wind_update=wind_field if wind_field else None,
            goal_update=goal_position,
            priority=priority,
            max_delay=self.replan_timeout,
        )
        
        logger.info(
            f"Replan request generated: trigger={trigger.value}, "
            f"energy={energy_remaining:.0f}Wh, time={time_remaining:.0f}s"
        )
        
        return request
    
    def execute_replan_async(
        self,
        request: ReplanRequest,
    ) -> threading.Thread:
        """
        Launch replanning in background thread.
        
        Calls mission_planner callback with request, expects ReplanResult.
        
        Args:
            request: Replanning request
        
        Returns:
            Thread object (check thread.is_alive() to monitor progress)
        """
        def replan_worker():
            if not self.mission_planner:
                logger.error("No mission planner callback provided")
                return
            
            try:
                start_time = time.time()
                
                # Call planner (assumed to return ReplanResult or trajectory)
                result = self.mission_planner(request)
                
                planning_time = time.time() - start_time
                
                # Wrap result if needed
                if not isinstance(result, ReplanResult):
                    result = ReplanResult(
                        success=result is not None,
                        new_trajectory=result,
                        planning_time=planning_time,
                    )
                else:
                    result.planning_time = planning_time
                
                # Validate feasibility
                if result.success and result.new_waypoints is not None:
                    energy_feasible, _ = self.feasibility_checker.check_energy_feasibility(
                        result.new_waypoints,
                        request.energy_remaining,
                        vehicle_mass=100.0,  # TODO: Get from config
                    )
                    
                    time_feasible, _ = self.feasibility_checker.check_time_feasibility(
                        result.new_waypoints,
                        request.time_remaining,
                    )
                    
                    if not (energy_feasible and time_feasible):
                        logger.warning("New plan fails feasibility checks")
                        result.success = False
                
                with self._replan_lock:
                    self.current_replan = result
                    self.replan_count += 1
                    self.last_replan_time = time.time()
                
                logger.info(
                    f"Replan completed: success={result.success}, "
                    f"time={result.planning_time:.3f}s, quality={result.quality_metric:.2f}"
                )
                
            except Exception as e:
                logger.error(f"Replan failed with exception: {e}")
                with self._replan_lock:
                    self.current_replan = ReplanResult(
                        success=False,
                        error_message=str(e),
                    )
        
        thread = threading.Thread(target=replan_worker, daemon=True)
        thread.start()
        return thread
    
    def get_replan_status(self) -> Optional[ReplanResult]:
        """Check if replanning has completed."""
        with self._replan_lock:
            return self.current_replan
    
    def apply_new_trajectory(
        self,
        old_trajectory: Any,
        old_time: float,
        replan_result: ReplanResult,
    ) -> Optional[BlendedTrajectory]:
        """
        Apply new trajectory with smooth hand-off.
        
        Args:
            old_trajectory: Currently executing trajectory
            old_time: Current time in old trajectory
            replan_result: Result from replanning
        
        Returns:
            BlendedTrajectory for smooth transition, or None if invalid
        """
        if not replan_result.success or replan_result.new_trajectory is None:
            logger.error("Cannot apply invalid replan result")
            return None
        
        # Compute switch time
        switch_time = old_time + replan_result.switch_time
        blend_duration = replan_result.blend_window
        
        blended = BlendedTrajectory(
            old_trajectory=old_trajectory,
            new_trajectory=replan_result.new_trajectory,
            switch_time=switch_time,
            blend_duration=blend_duration,
        )
        
        logger.info(
            f"Applied new trajectory: switch_time={switch_time:.1f}s, "
            f"blend_duration={blend_duration:.1f}s"
        )
        
        return blended
    
    def reset(self):
        """Reset replan counter for new mission."""
        self.replan_count = 0
        self.last_replan_time = None
        self.current_replan = None
