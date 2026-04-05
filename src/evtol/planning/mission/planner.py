"""
Multi-Leg Mission Planning

This module implements mission-level planning for defense eVTOL operations.

Mission Structure:
-----------------
Mission
├── Leg 1: Base → Waypoint A (reconnaissance)
├── Leg 2: Waypoint A → Target (strike)
├── Leg 3: Target → Waypoint B (egress)
└── Leg 4: Waypoint B → Base (return)

Each leg has:
- Start/end waypoints
- Mission phase (takeoff, cruise, loiter, strike, landing)
- Constraints (timing, altitude, threat exposure)
- Trajectory (optimized path)

Author: Defense eVTOL Research Team
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from enum import Enum, auto
import logging

from ..core.trajectory import Trajectory
from ..core.constraints import ConstraintSet
from ..algorithms.base import PathPlanner, PlanningConfig

logger = logging.getLogger(__name__)


class MissionPhase(Enum):
    """Mission phase types."""
    PREFLIGHT = auto()
    TAKEOFF = auto()
    CLIMB = auto()
    CRUISE = auto()
    DESCENT = auto()
    LOITER = auto()
    RECONNAISSANCE = auto()
    STRIKE = auto()
    ESCORT = auto()
    INGRESS = auto()
    EGRESS = auto()
    APPROACH = auto()
    LANDING = auto()
    POSTFLIGHT = auto()


@dataclass
class MissionObjective:
    """
    Mission objective definition.

    Attributes:
        name: Objective identifier
        position: Target position [lat, lon, alt]
        priority: Objective priority (1 = highest)
        type: Type of objective (waypoint, target, etc.)
        time_window: Optional time window [earliest, latest]
        duration: Required time at objective (for loiter)
        completed: Whether objective has been completed
    """
    name: str
    position: np.ndarray
    priority: int = 1
    type: str = "waypoint"
    time_window: tuple[float, float] | None = None
    duration: float = 0.0
    completed: bool = False

    def __post_init__(self):
        self.position = np.asarray(self.position)

    def is_time_valid(self, time: float) -> bool:
        """Check if time is within valid window."""
        if self.time_window is None:
            return True
        return self.time_window[0] <= time <= self.time_window[1]


@dataclass
class MissionLeg:
    """
    Single leg of a mission.

    A leg connects two waypoints with a specific trajectory.

    Attributes:
        id: Leg identifier
        start: Start objective
        end: End objective
        phase: Mission phase for this leg
        trajectory: Optimized trajectory (filled during planning)
        constraints: Leg-specific constraints
        estimated_time: Estimated duration [s]
        estimated_energy: Estimated energy consumption [fraction]
        status: Leg status
    """
    id: int
    start: MissionObjective
    end: MissionObjective
    phase: MissionPhase = MissionPhase.CRUISE
    trajectory: Trajectory | None = None
    constraints: ConstraintSet | None = None
    estimated_time: float = 0.0
    estimated_energy: float = 0.0
    status: str = "planned"

    @property
    def distance(self) -> float:
        """Straight-line distance between endpoints."""
        # Approximate distance in meters
        dlat = (self.end.position[0] - self.start.position[0]) * 110540
        dlon = (self.end.position[1] - self.start.position[1]) * 111320 * np.cos(
            np.radians(self.start.position[0])
        )
        dalt = self.end.position[2] - self.start.position[2]
        return np.sqrt(dlat**2 + dlon**2 + dalt**2)

    def estimate_time(self, avg_speed: float = 40.0) -> float:
        """Estimate leg duration based on average speed."""
        self.estimated_time = self.distance / avg_speed
        return self.estimated_time


@dataclass
class MissionPlan:
    """
    Complete mission plan.

    Contains all legs, objectives, and overall mission parameters.

    Attributes:
        name: Mission name
        legs: List of mission legs
        objectives: All mission objectives
        start_time: Mission start time
        energy_budget: Maximum energy consumption (fraction)
        constraints: Mission-level constraints
    """
    name: str
    legs: list[MissionLeg] = field(default_factory=list)
    objectives: list[MissionObjective] = field(default_factory=list)
    start_time: float = 0.0
    energy_budget: float = 0.8  # 80% of battery
    constraints: ConstraintSet | None = None
    status: str = "planning"

    def __len__(self) -> int:
        return len(self.legs)

    @property
    def total_distance(self) -> float:
        """Total straight-line distance."""
        return sum(leg.distance for leg in self.legs)

    @property
    def total_time(self) -> float:
        """Total estimated mission time."""
        return sum(leg.estimated_time for leg in self.legs)

    @property
    def total_energy(self) -> float:
        """Total estimated energy consumption."""
        return sum(leg.estimated_energy for leg in self.legs)

    def add_leg(self, leg: MissionLeg) -> None:
        """Add a leg to the mission."""
        self.legs.append(leg)

    def add_objective(self, objective: MissionObjective) -> None:
        """Add an objective."""
        self.objectives.append(objective)

    def get_trajectory(self) -> Trajectory | None:
        """Get concatenated trajectory for entire mission."""
        if not self.legs or not all(leg.trajectory for leg in self.legs):
            return None

        all_segments = []
        all_states = []

        for leg in self.legs:
            if leg.trajectory:
                all_segments.extend(leg.trajectory.segments)
                if not all_states:
                    all_states.extend(leg.trajectory.states)
                else:
                    # Skip first state to avoid duplicate
                    all_states.extend(leg.trajectory.states[1:])

        if not all_segments:
            return None

        return Trajectory(segments=all_segments, states=all_states)

    def get_waypoints(self) -> np.ndarray:
        """Get all waypoint positions."""
        waypoints = [self.legs[0].start.position]
        for leg in self.legs:
            waypoints.append(leg.end.position)
        return np.array(waypoints)

    def validate(self) -> tuple[bool, list[str]]:
        """
        Validate mission plan.

        Returns:
            (is_valid, list of issues)
        """
        issues = []

        # Check leg connectivity
        for i in range(len(self.legs) - 1):
            if not np.allclose(self.legs[i].end.position, self.legs[i + 1].start.position):
                issues.append(f"Legs {i} and {i+1} are not connected")

        # Check energy budget
        if self.total_energy > self.energy_budget:
            issues.append(f"Energy consumption {self.total_energy:.2f} exceeds budget {self.energy_budget:.2f}")

        # Check time windows
        current_time = self.start_time
        for leg in self.legs:
            current_time += leg.estimated_time
            if not leg.end.is_time_valid(current_time):
                issues.append(f"Arrival at {leg.end.name} at t={current_time:.0f} outside time window")

        return len(issues) == 0, issues

    def summary(self) -> str:
        """Generate mission summary."""
        lines = [
            f"Mission: {self.name}",
            "=" * 50,
            f"Legs: {len(self.legs)}",
            f"Objectives: {len(self.objectives)}",
            f"Total distance: {self.total_distance/1000:.1f} km",
            f"Estimated time: {self.total_time/60:.1f} min",
            f"Estimated energy: {self.total_energy*100:.1f}%",
            "",
            "Legs:",
        ]

        for leg in self.legs:
            lines.append(
                f"  {leg.id}: {leg.start.name} → {leg.end.name} "
                f"({leg.phase.name}, {leg.distance/1000:.1f}km, {leg.estimated_time/60:.1f}min)"
            )

        return "\n".join(lines)


class MissionPlanner:
    """
    Mission-level trajectory planner.

    Orchestrates planning for multi-leg missions:
    1. Generates mission structure from objectives
    2. Plans each leg using appropriate planner
    3. Optimizes overall mission (timing, resource allocation)
    4. Validates feasibility

    Attributes:
        leg_planner: Planner for individual legs
        planning_config: Planning configuration
    """

    def __init__(
        self,
        leg_planner: PathPlanner,
        planning_config: PlanningConfig | None = None,
    ):
        """
        Initialize mission planner.

        Args:
            leg_planner: Path planner for individual legs
            planning_config: Planning configuration
        """
        self.leg_planner = leg_planner
        self.config = planning_config or PlanningConfig()

    def plan_mission(
        self,
        objectives: list[MissionObjective],
        start_position: np.ndarray,
        return_to_start: bool = True,
        optimize_order: bool = False,
    ) -> MissionPlan:
        """
        Plan a complete mission.

        Args:
            objectives: Mission objectives in order
            start_position: Starting position
            return_to_start: Whether to return to start
            optimize_order: Whether to optimize objective ordering

        Returns:
            Complete MissionPlan
        """
        # Create start objective
        start = MissionObjective(
            name="Start",
            position=start_position,
            type="base",
        )

        # Optionally optimize order (TSP-like)
        if optimize_order:
            objectives = self._optimize_order(objectives, start_position)

        # Create mission
        mission = MissionPlan(
            name=f"Mission_{len(objectives)}_objectives",
            objectives=[start] + objectives,
        )

        # Generate legs
        current = start
        for i, objective in enumerate(objectives):
            phase = self._determine_phase(i, len(objectives), objective)

            leg = MissionLeg(
                id=i,
                start=current,
                end=objective,
                phase=phase,
            )
            leg.estimate_time()

            mission.add_leg(leg)
            current = objective

        # Return leg if requested
        if return_to_start:
            return_leg = MissionLeg(
                id=len(objectives),
                start=current,
                end=start,
                phase=MissionPhase.EGRESS,
            )
            return_leg.estimate_time()
            mission.add_leg(return_leg)

        # Plan each leg
        for leg in mission.legs:
            self._plan_leg(leg)

        # Estimate energy
        self._estimate_energy(mission)

        return mission

    def _plan_leg(self, leg: MissionLeg) -> None:
        """Plan a single mission leg."""
        from ..core.state import State, Pose, Velocity
        try:
            start_state = State(
                pose=Pose(position=np.asarray(leg.start.position, dtype=float)),
                velocity=Velocity(),
            )
            goal_state = State(
                pose=Pose(position=np.asarray(leg.end.position, dtype=float)),
                velocity=Velocity(),
            )
            result = self.leg_planner.plan(start_state, goal_state)

            if result.success:
                leg.trajectory = result.trajectory
                leg.estimated_time = result.trajectory.duration
                leg.status = "planned"
            else:
                leg.status = "failed"
                logger.warning(f"Failed to plan leg {leg.id}: planning_time={result.planning_time:.2f}s")

        except Exception as e:
            leg.status = "error"
            logger.error(f"Error planning leg {leg.id}: {e}")

    def _determine_phase(
        self,
        leg_index: int,
        total_legs: int,
        objective: MissionObjective,
    ) -> MissionPhase:
        """Determine mission phase for a leg based on context."""
        if leg_index == 0:
            return MissionPhase.INGRESS
        elif leg_index == total_legs - 1:
            return MissionPhase.EGRESS
        elif objective.type == "target":
            return MissionPhase.STRIKE
        elif objective.type == "recon":
            return MissionPhase.RECONNAISSANCE
        elif objective.type == "loiter":
            return MissionPhase.LOITER
        else:
            return MissionPhase.CRUISE

    def _optimize_order(
        self,
        objectives: list[MissionObjective],
        start: np.ndarray,
    ) -> list[MissionObjective]:
        """
        Optimize objective ordering (simple nearest-neighbor TSP).

        For more sophisticated ordering, use proper TSP solvers.
        """
        if len(objectives) <= 2:
            return objectives

        remaining = list(range(len(objectives)))
        ordered = []
        current_pos = start

        while remaining:
            # Find nearest unvisited objective
            min_dist = np.inf
            nearest_idx = remaining[0]

            for idx in remaining:
                obj = objectives[idx]
                dist = np.linalg.norm(obj.position - current_pos)

                # Weight by priority
                weighted_dist = dist / obj.priority

                if weighted_dist < min_dist:
                    min_dist = weighted_dist
                    nearest_idx = idx

            ordered.append(objectives[nearest_idx])
            current_pos = objectives[nearest_idx].position
            remaining.remove(nearest_idx)

        return ordered

    def _estimate_energy(self, mission: MissionPlan) -> None:
        """Estimate energy consumption for mission."""
        # Simple model: energy proportional to distance and altitude changes
        base_consumption = 0.0001  # Per meter
        climb_factor = 0.0005  # Per meter of climb

        for leg in mission.legs:
            distance_energy = leg.distance * base_consumption

            alt_change = leg.end.position[2] - leg.start.position[2]
            if alt_change > 0:
                climb_energy = alt_change * climb_factor
            else:
                climb_energy = 0.0

            # Phase-specific multipliers
            phase_factor = 1.0
            if leg.phase in [MissionPhase.TAKEOFF, MissionPhase.CLIMB]:
                phase_factor = 1.5
            elif leg.phase == MissionPhase.LOITER:
                phase_factor = 0.8
            elif leg.phase == MissionPhase.STRIKE:
                phase_factor = 1.2

            leg.estimated_energy = (distance_energy + climb_energy) * phase_factor

    def replan_from(
        self,
        mission: MissionPlan,
        current_position: np.ndarray,
        current_leg_index: int,
    ) -> MissionPlan:
        """
        Replan mission from current position.

        Used for dynamic replanning during mission execution.

        Args:
            mission: Original mission plan
            current_position: Current aircraft position
            current_leg_index: Index of current leg

        Returns:
            Updated MissionPlan
        """
        # Create new mission starting from current position
        remaining_objectives = [
            leg.end for leg in mission.legs[current_leg_index:]
        ]

        return self.plan_mission(
            objectives=remaining_objectives,
            start_position=current_position,
            return_to_start=False,  # Already included if needed
        )


def create_reconnaissance_mission(
    base_position: np.ndarray,
    targets: list[np.ndarray],
    loiter_time: float = 60.0,
) -> MissionPlan:
    """
    Create a reconnaissance mission.

    Args:
        base_position: Base/start position
        targets: Target positions to observe
        loiter_time: Time to spend at each target

    Returns:
        MissionPlan for reconnaissance
    """
    objectives = []
    for i, target in enumerate(targets):
        obj = MissionObjective(
            name=f"Target_{i+1}",
            position=np.asarray(target),
            priority=1,
            type="recon",
            duration=loiter_time,
        )
        objectives.append(obj)

    # Create simple planner (will be replaced with proper planner)
    from ..algorithms.graph import AStarPlanner
    planner = AStarPlanner()

    mission_planner = MissionPlanner(planner)

    return mission_planner.plan_mission(
        objectives=objectives,
        start_position=base_position,
        return_to_start=True,
    )


def create_strike_mission(
    base_position: np.ndarray,
    ingress_waypoints: list[np.ndarray],
    target: np.ndarray,
    egress_waypoints: list[np.ndarray],
) -> MissionPlan:
    """
    Create a strike mission with specific ingress/egress routes.

    Args:
        base_position: Base position
        ingress_waypoints: Ingress route waypoints
        target: Target position
        egress_waypoints: Egress route waypoints

    Returns:
        MissionPlan for strike mission
    """
    objectives = []

    # Ingress waypoints
    for i, wp in enumerate(ingress_waypoints):
        objectives.append(MissionObjective(
            name=f"Ingress_{i+1}",
            position=np.asarray(wp),
            type="waypoint",
        ))

    # Target
    objectives.append(MissionObjective(
        name="Target",
        position=np.asarray(target),
        priority=1,
        type="target",
    ))

    # Egress waypoints
    for i, wp in enumerate(egress_waypoints):
        objectives.append(MissionObjective(
            name=f"Egress_{i+1}",
            position=np.asarray(wp),
            type="waypoint",
        ))

    from ..algorithms.graph import AStarPlanner
    planner = AStarPlanner()

    mission_planner = MissionPlanner(planner)

    return mission_planner.plan_mission(
        objectives=objectives,
        start_position=base_position,
        return_to_start=True,
        optimize_order=False,  # Preserve specified order
    )
