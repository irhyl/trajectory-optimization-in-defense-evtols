"""
Autonomy Orchestrator - Planning Layer.

Coordinates between the perception layer (traversability cost field),
planning algorithms (RRT* + NSGA-III), and control layer (trajectory tracking).

This is the single integration point that ties the four-layer architecture:
  Perception  ->  Orchestrator  ->  Planning  ->  Control

Usage::

    orchestrator = AutonomyOrchestrator(config)
    orchestrator.set_perception_data(cost_field, threat_map)
    plan = orchestrator.plan_mission(start, goal, mission_config)
    orchestrator.execute(plan)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from .rrt_star import RRTStar
from .nsga3_optimizer import NSGA3Optimizer
from .mission import MissionPlanner

logger = logging.getLogger(__name__)


@dataclass
class OrchestratorConfig:
    """Configuration for the autonomy orchestrator."""
    # Planning
    max_planning_time_s:   float = 30.0    # wall-clock limit for path search
    replanning_interval_s: float = 5.0     # how often to check for replan trigger
    safety_altitude_m:     float = 50.0    # minimum cruise altitude
    # NSGA-III
    n_objectives:          int   = 3        # energy, time, threat
    population_size:       int   = 100
    n_generations:         int   = 200
    # Outputs
    output_dir:            str   = "outputs/planning_dataset"


@dataclass
class OrchestratorState:
    """Runtime state of the orchestrator."""
    is_planning:        bool  = False
    is_executing:       bool  = False
    current_mission_id: int   = 0
    replan_count:       int   = 0
    last_plan_time_s:   float = 0.0


class AutonomyOrchestrator:
    """
    Integration hub for the four-layer defense eVTOL autonomy stack.

    Responsibilities:
    1. Accept perception outputs (cost field, threats, obstacles)
    2. Invoke RRT* for feasible path search
    3. Post-process with NSGA-III multi-objective optimisation
    4. Deliver trajectory to control layer for tracking
    5. Monitor execution and trigger replanning when needed
    """

    def __init__(self, config: OrchestratorConfig | None = None):
        self.config  = config or OrchestratorConfig()
        self.state   = OrchestratorState()

        # Sub-systems (lazy-initialised)
        self._rrt:     Optional[RRTStar]       = None
        self._nsga:    Optional[NSGA3Optimizer] = None
        self._mission: Optional[MissionPlanner] = None

        # Perception data (updated externally)
        self._cost_field:  Optional[np.ndarray] = None
        self._threat_map:  Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Perception interface
    # ------------------------------------------------------------------

    def set_perception_data(
        self,
        cost_field: np.ndarray,
        threat_map: Optional[np.ndarray] = None,
    ) -> None:
        """
        Update the traversability cost field from the perception layer.

        Args:
            cost_field: 3-D array (x, y, z) of traversability costs [0, 1].
            threat_map: Optional 3-D threat probability field [0, 1].
        """
        self._cost_field = cost_field
        self._threat_map = threat_map
        logger.debug("Orchestrator: perception data updated. "
                     "cost_field shape=%s", cost_field.shape)

    # ------------------------------------------------------------------
    # Planning interface
    # ------------------------------------------------------------------

    def plan_mission(
        self,
        start:          np.ndarray,
        goal:           np.ndarray,
        mission_config: Optional[dict] = None,
    ) -> Optional[dict]:
        """
        Plan a full mission trajectory from start to goal.

        Sequence:
            1. Run RRT* to find a collision-free path
            2. Optimise with NSGA-III over energy / time / threat objectives
            3. Return the best trajectory for the control layer

        Args:
            start: Start position [x, y, z] in NED metres.
            goal:  Goal position  [x, y, z] in NED metres.
            mission_config: Optional overrides for mission parameters.

        Returns:
            dict with keys 'waypoints', 'velocity_profile', 'pareto_front'
            or None if planning fails.
        """
        self.state.is_planning = True
        cfg = mission_config or {}

        try:
            # Step 1: RRT* path search
            if self._rrt is None:
                self._rrt = RRTStar()
            logger.info("Orchestrator: running RRT* from %s to %s", start, goal)
            path = self._rrt.plan(start, goal, cost_field=self._cost_field)
            if path is None:
                logger.warning("Orchestrator: RRT* found no path")
                return None

            # Step 2: NSGA-III multi-objective optimisation
            if self._nsga is None:
                self._nsga = NSGA3Optimizer(
                    n_objectives=self.config.n_objectives,
                    population_size=self.config.population_size,
                    n_generations=self.config.n_generations,
                )

            pareto_front = self._nsga.optimize(
                paths=[path],
                threat_map=self._threat_map,
                energy_weight=cfg.get("energy_weight", 1.0),
            )

            # Step 3: Select best trajectory (minimum weighted sum)
            best = pareto_front[0] if pareto_front else {"waypoints": path}

            self.state.current_mission_id += 1
            self.state.last_plan_time_s   = 0.0  # reset timer
            logger.info("Orchestrator: plan complete — %d waypoints", len(best.get("waypoints", [])))
            return best

        except Exception as exc:
            logger.error("Orchestrator: planning failed — %s", exc)
            return None
        finally:
            self.state.is_planning = False

    # ------------------------------------------------------------------
    # Replanning interface
    # ------------------------------------------------------------------

    def check_replan_trigger(
        self,
        current_pos: np.ndarray,
        goal:        np.ndarray,
        threat_delta: float = 0.0,
    ) -> bool:
        """
        Decide whether replanning is needed.

        Args:
            current_pos:  Current vehicle position [x, y, z].
            goal:         Mission goal [x, y, z].
            threat_delta: Change in threat level since last plan (0-1).

        Returns:
            True if replanning should be triggered.
        """
        # Replan if threat has increased significantly
        if threat_delta > 0.3:
            logger.info("Orchestrator: replan triggered by threat_delta=%.2f", threat_delta)
            self.state.replan_count += 1
            return True
        return False

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_status(self) -> dict:
        """Return orchestrator status summary."""
        return {
            "is_planning":        self.state.is_planning,
            "is_executing":       self.state.is_executing,
            "current_mission_id": self.state.current_mission_id,
            "replan_count":       self.state.replan_count,
            "perception_ready":   self._cost_field is not None,
        }
