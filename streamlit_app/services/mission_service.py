"""Mission planning service for Streamlit front-end."""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure the core src package is importable when Streamlit runs from streamlit_app
ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from evtol.planning.base import Waypoint  # type: ignore  # pylint: disable=import-error
from evtol.planning.config import PlanningConfig  # type: ignore  # pylint: disable=import-error
from evtol.planning.mission.planning import MissionPlanner  # type: ignore  # pylint: disable=import-error


class MissionService:
    """Wrapper around MissionPlanner with JSON-backed history."""

    def __init__(self, config_path: Optional[str | Path] = None, history_limit: int = 25) -> None:
        self.config = PlanningConfig(config_path)
        self.planner = MissionPlanner(self.config)
        self.history_limit = history_limit
        self.history_path = ROOT_DIR / "outputs" / "mission-history.json"
        self.history_path.parent.mkdir(parents=True, exist_ok=True)

    def load_history(self) -> List[Dict[str, Any]]:
        """Load mission history from disk."""
        if not self.history_path.exists():
            return []
        try:
            with self.history_path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        except json.JSONDecodeError:
            return []

    def plan_and_store(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Plan a mission using backend and persist it."""
        origin_wp = self._dict_to_waypoint(payload["origin"])
        destination_wps = [self._dict_to_waypoint(dest) for dest in payload["destinations"]]
        if not destination_wps:
            raise ValueError("At least one destination waypoint is required")

        launch_iso = payload["launch_iso"]
        mission_id = payload.get("mission_id") or self._generate_mission_id(payload.get("mission_name", "mission"), payload.get("callsign"))
        plan = self.planner.plan_mission(origin_wp, destination_wps, launch_iso, constraints=payload.get("constraints"))

        entry: Dict[str, Any] = {
            "mission_id": mission_id,
            "submitted_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "mission": {
                "name": payload.get("mission_name"),
                "callsign": payload.get("callsign"),
                "mission_type": payload.get("mission_type"),
                "objective": payload.get("objective"),
                "launch_iso": launch_iso,
            },
            "origin": payload["origin"],
            "destinations": payload["destinations"],
            "constraints": payload.get("constraints"),
            "vehicles": payload.get("vehicles"),
            "environment": payload.get("environment"),
            "plan": plan,
        }

        history = self.load_history()
        history.insert(0, entry)
        trimmed = history[: self.history_limit]
        with self.history_path.open("w", encoding="utf-8") as handle:
            json.dump(trimmed, handle, indent=2)

        return entry

    def get_mission(self, mission_id: str) -> Optional[Dict[str, Any]]:
        """Fetch a single mission from history."""
        for record in self.load_history():
            if record.get("mission_id") == mission_id:
                return record
        return None

    @staticmethod
    def _dict_to_waypoint(payload: Dict[str, Any]) -> Waypoint:
        return Waypoint(
            float(payload["lat"]),
            float(payload["lon"]),
            float(payload.get("alt_m", payload.get("alt", 100.0))),
        )

    @staticmethod
    def _generate_mission_id(name: str, callsign: Optional[str]) -> str:
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        slug_source = callsign or name or "mission"
        slug = re.sub(r"[^a-z0-9]+", "-", slug_source.lower()).strip("-") or "mission"
        return f"{slug}-{timestamp}"


__all__ = ["MissionService"]
