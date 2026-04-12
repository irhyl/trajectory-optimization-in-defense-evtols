"""
Flight Mode Manager - State machine for hover / transition / cruise modes.

Manages mode transitions with hysteresis, blending, and safety guards.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional


class FlightMode(Enum):
    """Primary flight modes for the tiltrotor eVTOL."""
    HOVER      = auto()   # Vertical flight, rotors lift  (nacelles 90 deg)
    TRANSITION = auto()   # Mixed lift - rotors + wing     (nacelles 0-90 deg)
    CRUISE     = auto()   # Wing-borne forward flight       (nacelles 0 deg)
    EMERGENCY  = auto()   # Failsafe descent


@dataclass
class ModeTransitionConfig:
    """Thresholds and timing for automatic mode transitions."""
    hover_to_transition_min:  float = 10.0   # m/s - enter TRANSITION above this
    transition_to_cruise_min: float = 25.0   # m/s - enter CRUISE above this
    cruise_to_transition_max: float = 20.0   # m/s - leave CRUISE below this
    transition_to_hover_max:  float = 8.0    # m/s - return to HOVER below this
    hysteresis_time:          float = 2.0    # s  - hold desired before committing
    min_mode_dwell_time:      float = 3.0    # s  - minimum time in each mode
    min_cruise_altitude:      float = 30.0   # m  - altitude guard for cruise entry


@dataclass
class FlightModeState:
    """Runtime state of the mode manager."""
    mode:                FlightMode          = FlightMode.HOVER
    prev_mode:           FlightMode          = FlightMode.HOVER
    mode_elapsed_s:      float               = 0.0
    transition_progress: float               = 0.0
    n_transitions:       int                 = 0
    hysteresis_timer_s:  float               = 0.0
    pending_mode:        Optional[FlightMode] = None


class FlightModeManager:
    """
    Flight mode state machine for the tiltrotor eVTOL.

    Determines the current flight mode (HOVER / TRANSITION / CRUISE / EMERGENCY)
    based on airspeed, altitude, and mission context.  Enforces:
      - Minimum dwell time (no chattering)
      - Hysteresis band on airspeed transitions
      - Altitude guard for cruise entry
      - Emergency override path
    """

    def __init__(self, config: Optional[ModeTransitionConfig] = None):
        self.config = config or ModeTransitionConfig()
        self.state  = FlightModeState()
        self._history: list[tuple[float, FlightMode]] = []

    def reset(self) -> None:
        """Reset to initial HOVER state."""
        self.state    = FlightModeState()
        self._history = []

    def update(
        self,
        t:         float,
        airspeed:  float,
        altitude:  float,
        dt:        float,
        emergency: bool = False,
    ) -> FlightMode:
        """
        Update mode state machine and return current mode.

        Args:
            t:         Current time (s)
            airspeed:  Vehicle airspeed (m/s)
            altitude:  Vehicle altitude AGL (m)
            dt:        Time step (s)
            emergency: Force emergency mode if True

        Returns:
            Current FlightMode
        """
        if emergency:
            self._set_mode(FlightMode.EMERGENCY, t)
            return self.state.mode

        self.state.mode_elapsed_s += dt
        desired = self._desired_mode(airspeed, altitude)

        if desired != self.state.mode:
            if desired == self.state.pending_mode:
                self.state.hysteresis_timer_s += dt
                if (self.state.hysteresis_timer_s >= self.config.hysteresis_time and
                        self.state.mode_elapsed_s  >= self.config.min_mode_dwell_time):
                    self._set_mode(desired, t)
            else:
                self.state.pending_mode       = desired
                self.state.hysteresis_timer_s = 0.0
        else:
            self.state.pending_mode       = None
            self.state.hysteresis_timer_s = 0.0

        self._update_transition_progress(airspeed)

        if not self._history or self._history[-1][1] != self.state.mode:
            self._history.append((t, self.state.mode))

        return self.state.mode

    def force_mode(self, mode: FlightMode, t: float = 0.0) -> None:
        """Force immediate mode change (bypasses hysteresis)."""
        self._set_mode(mode, t)

    @property
    def current_mode(self) -> FlightMode:
        return self.state.mode

    @property
    def transition_progress(self) -> float:
        return self.state.transition_progress

    @property
    def mode_history(self) -> list[tuple[float, FlightMode]]:
        return list(self._history)

    @property
    def n_transitions(self) -> int:
        return self.state.n_transitions

    def _desired_mode(self, airspeed: float, altitude: float) -> FlightMode:
        cfg = self.config
        cur = self.state.mode
        if cur == FlightMode.HOVER:
            if airspeed >= cfg.hover_to_transition_min:
                return FlightMode.TRANSITION
        elif cur == FlightMode.TRANSITION:
            if airspeed < cfg.transition_to_hover_max:
                return FlightMode.HOVER
            if airspeed >= cfg.transition_to_cruise_min and altitude >= cfg.min_cruise_altitude:
                return FlightMode.CRUISE
        elif cur == FlightMode.CRUISE:
            if airspeed < cfg.cruise_to_transition_max:
                return FlightMode.TRANSITION
        return cur

    def _set_mode(self, mode: FlightMode, t: float) -> None:
        if mode != self.state.mode:
            self.state.prev_mode      = self.state.mode
            self.state.mode           = mode
            self.state.mode_elapsed_s = 0.0
            self.state.n_transitions += 1
            self.state.pending_mode   = None

    def _update_transition_progress(self, airspeed: float) -> None:
        cfg = self.config
        if self.state.mode == FlightMode.HOVER:
            self.state.transition_progress = 0.0
        elif self.state.mode == FlightMode.CRUISE:
            self.state.transition_progress = 1.0
        elif self.state.mode == FlightMode.TRANSITION:
            span = cfg.transition_to_cruise_min - cfg.hover_to_transition_min
            frac = (airspeed - cfg.hover_to_transition_min) / span if span > 0 else 0.0
            self.state.transition_progress = float(np.clip(frac, 0.0, 1.0))
        else:
            self.state.transition_progress = 0.0

    def get_debug_info(self) -> dict:
        return {
            'mode':                self.state.mode.name,
            'prev_mode':           self.state.prev_mode.name,
            'mode_elapsed_s':      self.state.mode_elapsed_s,
            'transition_progress': self.state.transition_progress,
            'n_transitions':       self.state.n_transitions,
            'hysteresis_timer_s':  self.state.hysteresis_timer_s,
            'pending_mode':        self.state.pending_mode.name if self.state.pending_mode else None,
        }
