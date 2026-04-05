"""
Contingency Planning

This module handles contingency planning and management for
mission execution under failures and unexpected situations.

Contingency Types:
-----------------
1. Emergency landing (engine failure, low battery)
2. Threat avoidance (new threat detected)
3. Weather diversion (adverse conditions)
4. Mission abort (unable to complete objectives)
5. Communication loss (lost link procedures)

Each contingency has:
- Trigger conditions
- Response actions
- Alternative destinations
- Priority level

Author: Defense eVTOL Research Team
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Any
from collections.abc import Callable
from enum import Enum, auto
import logging

from ..core.trajectory import Trajectory
from .planner import MissionPlan, MissionLeg, MissionObjective, MissionPhase

logger = logging.getLogger(__name__)


class ContingencyType(Enum):
    """Types of contingency situations."""
    EMERGENCY_LANDING = auto()
    LOW_BATTERY = auto()
    ENGINE_FAILURE = auto()
    THREAT_DETECTED = auto()
    THREAT_ENGAGED = auto()
    WEATHER_DIVERSION = auto()
    OBSTACLE_DETECTED = auto()
    MISSION_ABORT = auto()
    COMMUNICATION_LOSS = auto()
    GPS_DEGRADED = auto()
    SENSOR_FAILURE = auto()
    AIRSPACE_VIOLATION = auto()


class ContingencyPriority(Enum):
    """Contingency priority levels."""
    CRITICAL = 1  # Immediate action required
    HIGH = 2      # Action required soon
    MEDIUM = 3    # Action required when convenient
    LOW = 4       # Informational


@dataclass
class ContingencyTrigger:
    """
    Condition that triggers a contingency.

    Attributes:
        type: Type of contingency
        condition: Callable that evaluates trigger condition
        threshold: Trigger threshold value
        description: Human-readable description
    """
    type: ContingencyType
    condition: Callable[[dict[str, Any]], bool]
    threshold: float = 0.0
    description: str = ""

    def evaluate(self, state: dict[str, Any]) -> bool:
        """
        Evaluate if trigger condition is met.

        Args:
            state: Current system state dictionary

        Returns:
            True if trigger is activated
        """
        try:
            return self.condition(state)
        except Exception as e:
            logger.warning(f"Error evaluating trigger: {e}")
            return False


@dataclass
class AlternateLandingSite:
    """
    Alternate landing site for emergencies.

    Attributes:
        name: Site identifier
        position: Site position [lat, lon, alt]
        type: Site type (base, forward, emergency)
        runway_heading: Runway heading if applicable
        facilities: Available facilities
        priority: Site priority (lower = better)
    """
    name: str
    position: np.ndarray
    type: str = "emergency"
    runway_heading: float | None = None
    facilities: list[str] = field(default_factory=list)
    priority: int = 1

    def __post_init__(self):
        self.position = np.asarray(self.position)

    def distance_from(self, position: np.ndarray) -> float:
        """Compute distance to site."""
        dlat = (self.position[0] - position[0]) * 110540
        dlon = (self.position[1] - position[1]) * 111320 * np.cos(np.radians(position[0]))
        return np.sqrt(dlat**2 + dlon**2)


@dataclass
class ContingencyPlan:
    """
    Plan for responding to a contingency.

    Attributes:
        type: Contingency type this plan addresses
        priority: Response priority
        actions: List of actions to take
        alternate_destination: Alternate destination if applicable
        trajectory: Pre-planned contingency trajectory
        valid_region: Region where plan is valid [min_lat, max_lat, min_lon, max_lon]
    """
    type: ContingencyType
    priority: ContingencyPriority = ContingencyPriority.MEDIUM
    actions: list[str] = field(default_factory=list)
    alternate_destination: AlternateLandingSite | None = None
    trajectory: Trajectory | None = None
    valid_region: tuple[float, float, float, float] | None = None

    def is_applicable(self, position: np.ndarray) -> bool:
        """Check if plan is applicable at position."""
        if self.valid_region is None:
            return True

        min_lat, max_lat, min_lon, max_lon = self.valid_region
        return (min_lat <= position[0] <= max_lat and
                min_lon <= position[1] <= max_lon)


class ContingencyManager:
    """
    Manages contingency planning and response.

    Responsibilities:
    1. Monitor system state for trigger conditions
    2. Select appropriate contingency plan
    3. Execute contingency response
    4. Update mission plan if needed

    Attributes:
        triggers: Registered contingency triggers
        plans: Available contingency plans
        landing_sites: Alternate landing sites
        active_contingency: Currently active contingency (if any)
    """

    def __init__(self):
        self.triggers: list[ContingencyTrigger] = []
        self.plans: dict[ContingencyType, list[ContingencyPlan]] = {}
        self.landing_sites: list[AlternateLandingSite] = []
        self.active_contingency: ContingencyType | None = None
        self._trigger_history: list[tuple[float, ContingencyType]] = []

    def register_trigger(self, trigger: ContingencyTrigger) -> None:
        """Register a contingency trigger."""
        self.triggers.append(trigger)

    def register_plan(self, plan: ContingencyPlan) -> None:
        """Register a contingency plan."""
        if plan.type not in self.plans:
            self.plans[plan.type] = []
        self.plans[plan.type].append(plan)

    def add_landing_site(self, site: AlternateLandingSite) -> None:
        """Add an alternate landing site."""
        self.landing_sites.append(site)

    def monitor(self, state: dict[str, Any]) -> ContingencyType | None:
        """
        Monitor state and check for triggered contingencies.

        Args:
            state: Current system state

        Returns:
            Triggered contingency type or None
        """
        for trigger in self.triggers:
            if trigger.evaluate(state):
                logger.warning(f"Contingency triggered: {trigger.type.name}")
                self._trigger_history.append((state.get('time', 0), trigger.type))
                return trigger.type

        return None

    def get_plan(
        self,
        contingency_type: ContingencyType,
        position: np.ndarray,
    ) -> ContingencyPlan | None:
        """
        Get appropriate contingency plan.

        Args:
            contingency_type: Type of contingency
            position: Current position

        Returns:
            Best applicable plan or None
        """
        if contingency_type not in self.plans:
            return None

        applicable_plans = [
            plan for plan in self.plans[contingency_type]
            if plan.is_applicable(position)
        ]

        if not applicable_plans:
            return None

        # Sort by priority
        applicable_plans.sort(key=lambda p: p.priority.value)

        return applicable_plans[0]

    def get_nearest_landing_site(
        self,
        position: np.ndarray,
        min_facilities: list[str] | None = None,
    ) -> AlternateLandingSite | None:
        """
        Find nearest alternate landing site.

        Args:
            position: Current position
            min_facilities: Required facilities

        Returns:
            Nearest suitable landing site
        """
        candidates = self.landing_sites.copy()

        # Filter by facilities
        if min_facilities:
            candidates = [
                site for site in candidates
                if all(f in site.facilities for f in min_facilities)
            ]

        if not candidates:
            return None

        # Sort by distance
        candidates.sort(key=lambda s: s.distance_from(position))

        return candidates[0]

    def plan_emergency_route(
        self,
        current_position: np.ndarray,
        target_site: AlternateLandingSite,
        planner: Any,  # PathPlanner
    ) -> Trajectory | None:
        """
        Plan emergency route to landing site.

        Args:
            current_position: Current aircraft position
            target_site: Target landing site
            planner: Path planner to use

        Returns:
            Emergency trajectory or None
        """
        try:
            result = planner.plan(current_position, target_site.position)
            if result.success:
                return result.trajectory
        except Exception as e:
            logger.error(f"Failed to plan emergency route: {e}")

        return None

    def execute_contingency(
        self,
        contingency_type: ContingencyType,
        current_state: dict[str, Any],
        mission: MissionPlan,
    ) -> tuple[bool, MissionPlan | None]:
        """
        Execute contingency response.

        Args:
            contingency_type: Type of contingency
            current_state: Current system state
            mission: Current mission plan

        Returns:
            (success, updated_mission)
        """
        position = current_state.get('position', np.zeros(3))

        # Get plan
        plan = self.get_plan(contingency_type, position)
        if plan is None:
            logger.error(f"No contingency plan for {contingency_type.name}")
            return False, None

        self.active_contingency = contingency_type

        # Execute based on type
        if contingency_type in [ContingencyType.EMERGENCY_LANDING,
                                ContingencyType.ENGINE_FAILURE,
                                ContingencyType.LOW_BATTERY]:
            return self._execute_emergency_landing(plan, current_state, mission)

        elif contingency_type in [ContingencyType.THREAT_DETECTED,
                                  ContingencyType.THREAT_ENGAGED]:
            return self._execute_threat_avoidance(plan, current_state, mission)

        elif contingency_type == ContingencyType.MISSION_ABORT:
            return self._execute_mission_abort(plan, current_state, mission)

        elif contingency_type == ContingencyType.WEATHER_DIVERSION:
            return self._execute_weather_diversion(plan, current_state, mission)

        else:
            logger.warning(f"No specific handler for {contingency_type.name}")
            return False, None

    def _execute_emergency_landing(
        self,
        plan: ContingencyPlan,
        state: dict[str, Any],
        mission: MissionPlan,
    ) -> tuple[bool, MissionPlan | None]:
        """Execute emergency landing procedure."""
        position = state.get('position', np.zeros(3))

        # Find nearest landing site
        site = self.get_nearest_landing_site(position)
        if site is None and plan.alternate_destination:
            site = plan.alternate_destination

        if site is None:
            logger.error("No landing site available for emergency")
            return False, None

        logger.info(f"Emergency landing at {site.name}")

        # Create emergency mission
        emergency_mission = MissionPlan(
            name=f"Emergency_{mission.name}",
            status="emergency",
        )

        current_obj = MissionObjective(
            name="Current",
            position=position,
        )

        landing_obj = MissionObjective(
            name=site.name,
            position=site.position,
            type="emergency_landing",
        )

        leg = MissionLeg(
            id=0,
            start=current_obj,
            end=landing_obj,
            phase=MissionPhase.DESCENT,
        )
        leg.estimate_time()

        emergency_mission.add_leg(leg)

        # Use pre-planned trajectory if available
        if plan.trajectory:
            leg.trajectory = plan.trajectory

        return True, emergency_mission

    def _execute_threat_avoidance(
        self,
        plan: ContingencyPlan,
        state: dict[str, Any],
        mission: MissionPlan,
    ) -> tuple[bool, MissionPlan | None]:
        """Execute threat avoidance maneuver."""
        state.get('position', np.zeros(3))
        state.get('threat_position')

        # Execute evasion actions
        for action in plan.actions:
            logger.info(f"Executing: {action}")

        # If pre-planned evasion trajectory exists
        if plan.trajectory:
            # Modify current leg with evasion trajectory
            current_leg_idx = state.get('current_leg', 0)
            if current_leg_idx < len(mission.legs):
                mission.legs[current_leg_idx].trajectory = plan.trajectory

        return True, mission

    def _execute_mission_abort(
        self,
        plan: ContingencyPlan,
        state: dict[str, Any],
        mission: MissionPlan,
    ) -> tuple[bool, MissionPlan | None]:
        """Execute mission abort procedure."""
        position = state.get('position', np.zeros(3))

        logger.warning("Aborting mission - returning to base")

        # Find home base (first objective usually)
        home = mission.objectives[0] if mission.objectives else None

        if home is None:
            # Use nearest landing site
            site = self.get_nearest_landing_site(position)
            if site:
                home = MissionObjective(
                    name=site.name,
                    position=site.position,
                )

        if home is None:
            return False, None

        # Create abort mission
        abort_mission = MissionPlan(
            name=f"Abort_{mission.name}",
            status="abort",
        )

        current_obj = MissionObjective(
            name="Current",
            position=position,
        )

        leg = MissionLeg(
            id=0,
            start=current_obj,
            end=home,
            phase=MissionPhase.EGRESS,
        )
        leg.estimate_time()

        abort_mission.add_leg(leg)

        return True, abort_mission

    def _execute_weather_diversion(
        self,
        plan: ContingencyPlan,
        state: dict[str, Any],
        mission: MissionPlan,
    ) -> tuple[bool, MissionPlan | None]:
        """Execute weather diversion."""
        position = state.get('position', np.zeros(3))

        # Find diversion site
        if plan.alternate_destination:
            diversion_site = plan.alternate_destination
        else:
            diversion_site = self.get_nearest_landing_site(
                position,
                min_facilities=['hangar'],  # Need shelter
            )

        if diversion_site is None:
            return False, None

        logger.info(f"Weather diversion to {diversion_site.name}")

        # Create diversion mission
        diversion_mission = MissionPlan(
            name=f"Diversion_{mission.name}",
            status="diversion",
        )

        current_obj = MissionObjective(
            name="Current",
            position=position,
        )

        diversion_obj = MissionObjective(
            name=diversion_site.name,
            position=diversion_site.position,
            type="diversion",
        )

        leg = MissionLeg(
            id=0,
            start=current_obj,
            end=diversion_obj,
            phase=MissionPhase.CRUISE,
        )
        leg.estimate_time()

        diversion_mission.add_leg(leg)

        return True, diversion_mission

    def clear_contingency(self) -> None:
        """Clear active contingency."""
        self.active_contingency = None


def create_standard_triggers() -> list[ContingencyTrigger]:
    """
    Create standard contingency triggers.

    Returns:
        List of common triggers
    """
    triggers = []

    # Low battery
    triggers.append(ContingencyTrigger(
        type=ContingencyType.LOW_BATTERY,
        condition=lambda s: s.get('battery_level', 1.0) < 0.2,
        threshold=0.2,
        description="Battery level below 20%",
    ))

    # Critical battery
    triggers.append(ContingencyTrigger(
        type=ContingencyType.EMERGENCY_LANDING,
        condition=lambda s: s.get('battery_level', 1.0) < 0.1,
        threshold=0.1,
        description="Battery level critical (<10%)",
    ))

    # Engine failure
    triggers.append(ContingencyTrigger(
        type=ContingencyType.ENGINE_FAILURE,
        condition=lambda s: any(
            not healthy for healthy in s.get('motor_health', [True, True, True, True])
        ),
        description="Motor failure detected",
    ))

    # Threat proximity
    triggers.append(ContingencyTrigger(
        type=ContingencyType.THREAT_DETECTED,
        condition=lambda s: s.get('threat_distance', float('inf')) < 5000,
        threshold=5000,
        description="Threat within 5km",
    ))

    # Communication loss
    triggers.append(ContingencyTrigger(
        type=ContingencyType.COMMUNICATION_LOSS,
        condition=lambda s: s.get('time_since_last_contact', 0) > 60,
        threshold=60,
        description="No communication for 60 seconds",
    ))

    return triggers
