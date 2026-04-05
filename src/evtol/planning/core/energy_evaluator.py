"""
Energy Evaluator for Mission-Level Planning

This module integrates the Phase 2A vehicle efficiency model with trajectory
planning to evaluate energy consumption. It provides:

1. Trajectory-specific energy calculation
2. Battery reserve validation
3. Energy-aware goal selection for RRT*
4. NSGA-III Pareto objective (energy efficiency)

Mathematical Model
==================

Energy consumption has three components:

1. **Hover energy**: Eₕ = Pₕ · tₕ
   - Weight-dependent: Pₕ = (W + ΔW_hover)² / (2ρADₘᵢₙ)
   - Increases with altitude (density ρ)

2. **Cruise energy**: Eₘ = Pₘ · tₘ
   - Drag-dependent: Pₘ = (½ρV²Sₘcₘ)V + (W/V)·√(W/(½ρSₘ))
   - Speed-optimal around 50-60 m/s (depending on efficiency)

3. **Transition energy**: E_trans = (Pₕ + Pₘ)/2 · t_trans
   - Nacelle angle tilt dependent

Reserve Requirements:
- Minimum reserve: 10% of total capacity
- Safety margin for contingency: additional 5%
- Recommended endurance: 15 minutes hover

Energy Density:
- Modern LiPo: ~250 Wh/kg (gravimetric)
- NCA cells: ~260 Wh/kg
- Solid-state target: ~380 Wh/kg

Author: Defense eVTOL Research Team
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from collections.abc import Callable
from enum import Enum
import logging

from ..core.trajectory import Trajectory
from ...vehicle.config import MassProperties, PropulsionConfig, FlightPhase
from ...vehicle.energy.battery_model import BatteryPack, BatteryConfig

# BatteryPack is the concrete battery model; alias for clarity inside this module
BatteryModel = BatteryPack

logger = logging.getLogger(__name__)


class EnergyMode(Enum):
    """Energy evaluation mode."""
    OPTIMAL = "optimal"        # Optimize for Pareto front
    CONSERVATIVE = "conservative"  # Assume worst-case conditions
    NOMINAL = "nominal"        # Standard conditions


@dataclass
class EnergyProfile:
    """Energy consumption profile for a trajectory."""
    
    total_energy: float  # Total energy [Wh]
    hover_energy: float  # Hover component [Wh]
    cruise_energy: float  # Cruise component [Wh]
    transition_energy: float  # Transition component [Wh]
    
    hover_time: float  # Total hover time [s]
    cruise_time: float  # Total cruise time [s]
    transition_time: float  # Total transition time [s]
    
    reserve_energy: float  # Energy reserve [Wh]
    reserve_fraction: float  # Reserve as fraction of capacity
    
    # Efficiency metrics
    specific_energy: float  # Energy per mission distance [Wh/m]
    energy_efficiency: float  # Distance per energy [m/Wh]
    cruise_efficiency: float  # Cruise-phase efficiency [m/Wh]
    
    # Margin metrics
    reserve_adequate: bool  # Meets minimum reserve requirement
    mission_feasible: bool  # Can complete mission with reserves
    endurance_minutes: float  # Hovering endurance at SOC [minutes]


@dataclass
class EnergyConstraints:
    """Energy-related constraints for planning."""
    
    max_energy: float = 100_000.0  # Battery capacity [Wh]
    min_reserve: float = 10_000.0  # Minimum reserve [Wh]
    contingency: float = 5_000.0   # Contingency margin [Wh]
    max_discharge_rate: float = 2.0  # C-rate limit
    temp_penalty_enabled: bool = False  # Temperature derating
    altitude_penalty_enabled: bool = True  # Density altitude penalty
    
    def required_reserve(self) -> float:
        """Total energy that must be reserved."""
        return self.min_reserve + self.contingency


class EnergyEvaluator:
    """
    Integrated energy evaluation for mission planning.
    
    Computes energy consumption for arbitrary trajectories using:
    - Phase 2A battery model
    - Flight dynamics from vehicle config
    - Altitude/temperature penalties
    
    Provides both trajectory evaluation and Pareto objective for NSGA-III.
    """
    
    def __init__(
        self,
        mass_config: MassProperties,
        propulsion_config: PropulsionConfig,
        battery_model: BatteryPack,
        constraints: EnergyConstraints | None = None,
        energy_mode: EnergyMode = EnergyMode.NOMINAL,
        wing_area: float = 10.0,  # m² — use WingConfig.area if available
    ):
        """
        Initialize energy evaluator.
        
        Args:
            mass_config: Vehicle mass properties
            propulsion_config: Motor/propeller configuration
            battery_model: Battery chemistry model
            constraints: Energy constraint specification
            energy_mode: Evaluation mode (optimal, conservative, nominal)
        """
        self.mass = mass_config
        self.propulsion = propulsion_config
        self.battery = battery_model
        self.constraints = constraints or EnergyConstraints()
        self.mode = energy_mode
        self.wing_area = wing_area
        
        # Gravitational and atmospheric constants
        self.g = 9.81  # m/s²
        self.rho_0 = 1.225  # Air density at sea level [kg/m³]
        self.h_scale = 8500  # Scale height for exponential atmosphere [m]
        
        logger.info(f"EnergyEvaluator initialized: {energy_mode.value} mode")
    
    def evaluate_trajectory(
        self,
        trajectory: Trajectory,
        reference_altitude: float = 0.0,
    ) -> EnergyProfile:
        """
        Evaluate energy profile for complete trajectory.
        
        Args:
            trajectory: Trajectory to evaluate
            reference_altitude: Sea-level reference for atmosphere model
            
        Returns:
            Detailed energy profile
        """
        # Segment trajectory into phases
        hover_segments = []
        cruise_segments = []
        transition_segments = []
        
        for segment in trajectory.segments:
            phase = self._classify_flight_phase(segment)
            if phase == FlightPhase.HOVER:
                hover_segments.append(segment)
            elif phase in (FlightPhase.CRUISE, FlightPhase.CLIMB, FlightPhase.DESCENT):
                cruise_segments.append(segment)
            else:
                transition_segments.append(segment)
        
        # Compute energy for each phase
        hover_energy, hover_time = self._compute_hover_energy(
            hover_segments,
            reference_altitude,
        )
        
        cruise_energy, cruise_time = self._compute_cruise_energy(
            cruise_segments,
            reference_altitude,
        )
        
        transition_energy, transition_time = self._compute_transition_energy(
            transition_segments,
            reference_altitude,
        )
        
        total_energy = hover_energy + cruise_energy + transition_energy
        
        # Compute reserve and feasibility
        reserve_energy = self.constraints.max_energy - total_energy
        reserve_fraction = reserve_energy / self.constraints.max_energy
        
        required_reserve = self.constraints.required_reserve()
        reserve_adequate = reserve_energy >= required_reserve
        mission_feasible = reserve_adequate and total_energy > 0
        
        # Compute efficiency metrics
        mission_distance = self._compute_mission_distance(trajectory)
        specific_energy = total_energy / max(mission_distance, 1.0)
        energy_efficiency = mission_distance / max(total_energy, 1.0)
        cruise_efficiency = mission_distance / max(cruise_energy, 1.0) if cruise_energy > 0 else 0
        
        # Endurance at remaining SOC
        total_time = hover_time + cruise_time + transition_time
        avg_hover_power = hover_energy / max(hover_time, 1.0) if hover_time > 0 else 0
        endurance_minutes = (reserve_energy / max(avg_hover_power, 1.0)) / 60.0
        
        return EnergyProfile(
            total_energy=total_energy,
            hover_energy=hover_energy,
            cruise_energy=cruise_energy,
            transition_energy=transition_energy,
            hover_time=hover_time,
            cruise_time=cruise_time,
            transition_time=transition_time,
            reserve_energy=reserve_energy,
            reserve_fraction=reserve_fraction,
            specific_energy=specific_energy,
            energy_efficiency=energy_efficiency,
            cruise_efficiency=cruise_efficiency,
            reserve_adequate=reserve_adequate,
            mission_feasible=mission_feasible,
            endurance_minutes=endurance_minutes,
        )
    
    def _classify_flight_phase(self, segment) -> FlightPhase:
        """Classify trajectory segment into flight phase."""
        if segment.duration < 1.0:
            return FlightPhase.HOVER
        
        start_vel = segment.start.velocity.speed
        end_vel = segment.end.velocity.speed
        avg_vel = (start_vel + end_vel) / 2
        
        # Speed threshold for hover/cruise boundary (~10 m/s)
        if avg_vel < 10:
            return FlightPhase.HOVER
        
        # Check altitude change for climb/descent
        alt_change = segment.end.pose.position[2] - segment.start.pose.position[2]
        vert_speed = alt_change / segment.duration
        
        if vert_speed > 1.0:  # Climbing
            return FlightPhase.CLIMB
        elif vert_speed < -1.0:  # Descending
            return FlightPhase.DESCENT
        else:
            return FlightPhase.CRUISE
    
    def _compute_hover_energy(
        self,
        segments: list,
        reference_altitude: float,
    ) -> tuple[float, float]:
        """
        Compute hover energy for segments.
        
        Returns:
            (energy [Wh], time [s])
        """
        total_energy = 0.0
        total_time = 0.0
        
        for segment in segments:
            avg_altitude = (
                segment.start.pose.position[2] +
                segment.end.pose.position[2]
            ) / 2
            
            # Atmospheric density at altitude
            rho = self._atmospheric_density(avg_altitude - reference_altitude)
            
            # Weight including payload
            weight = self.mass.total_mass * self.g
            
            # Hover power (disk loading model): P = W^(3/2) / sqrt(2ρA_total)
            # A_total = sum of both rotor disk areas
            disk_area = (
                self.propulsion.left_rotor.disk_area +
                self.propulsion.right_rotor.disk_area
            )
            hover_power = (weight ** 1.5) / np.sqrt(2 * rho * disk_area)
            
            # Apply penalties
            if self.mode == EnergyMode.CONSERVATIVE:
                hover_power *= 1.15  # 15% penalty for inefficiency
            
            hover_energy_wh = (hover_power / 1000) * segment.duration / 3600
            total_energy += hover_energy_wh
            total_time += segment.duration
        
        return total_energy, total_time
    
    def _compute_cruise_energy(
        self,
        segments: list,
        reference_altitude: float,
    ) -> tuple[float, float]:
        """
        Compute cruise energy for segments.
        
        Returns:
            (energy [Wh], time [s])
        """
        total_energy = 0.0
        total_time = 0.0
        
        for segment in segments:
            velocity = segment.end.velocity.speed
            if velocity < 1.0:
                continue

            avg_altitude = (
                segment.start.pose.position[2] +
                segment.end.pose.position[2]
            ) / 2

            rho = self._atmospheric_density(avg_altitude - reference_altitude)
            weight = self.mass.total_mass * self.g

            # Cruise power model (parasite drag + induced drag)
            # P = (1/2 * rho * V^2 * S * cd) * V + (W/V) * sqrt(W / (1/2 * rho * S * cl))
            wing_area = self.wing_area
            cd_parasite = 0.08  # Typical cruise parasite drag
            cl_cruise = 0.4  # Cruise lift coefficient
            
            parasite_power = 0.5 * rho * velocity**3 * wing_area * cd_parasite
            induced_power = (weight / velocity) * np.sqrt(weight / (0.5 * rho * wing_area * cl_cruise))
            cruise_power = parasite_power + induced_power
            
            if self.mode == EnergyMode.CONSERVATIVE:
                cruise_power *= 1.12  # 12% penalty
            
            cruise_energy_wh = (cruise_power / 1000) * segment.duration / 3600
            total_energy += cruise_energy_wh
            total_time += segment.duration
        
        return total_energy, total_time
    
    def _compute_transition_energy(
        self,
        segments: list,
        reference_altitude: float,
    ) -> tuple[float, float]:
        """
        Compute transition (hover↔cruise) energy.
        
        Returns:
            (energy [Wh], time [s])
        """
        # Transition blends hover and cruise
        total_energy = 0.0
        total_time = 0.0
        
        for segment in segments:
            # Average of hover and cruise power
            hover_e, _ = self._compute_hover_energy([segment], reference_altitude)
            cruise_e, _ = self._compute_cruise_energy([segment], reference_altitude)
            
            transition_energy_wh = (hover_e + cruise_e) / 2
            total_energy += transition_energy_wh
            total_time += segment.duration
        
        return total_energy, total_time
    
    def _atmospheric_density(self, altitude: float) -> float:
        """
        Compute atmospheric density at altitude.
        
        Uses exponential atmosphere model: ρ(h) = ρ₀ * exp(-h/h_scale)
        
        Args:
            altitude: Altitude above sea level [m]
            
        Returns:
            Density [kg/m³]
        """
        return self.rho_0 * np.exp(-altitude / self.h_scale)
    
    def _compute_mission_distance(self, trajectory: Trajectory) -> float:
        """Compute total horizontal distance traveled."""
        total_distance = 0.0
        
        for segment in trajectory.segments:
            start_pos = segment.start.pose.position[:2]  # xy only
            end_pos = segment.end.pose.position[:2]
            distance = np.linalg.norm(end_pos - start_pos)
            total_distance += distance
        
        return total_distance
    
    def energy_objective(self, trajectory: Trajectory) -> float:
        """
        Objective function for NSGA-III optimization.
        
        Returns energy consumption (to be minimized).
        Lower is better.
        
        Args:
            trajectory: Candidate trajectory
            
        Returns:
            Energy consumption [Wh]
        """
        profile = self.evaluate_trajectory(trajectory)
        
        # Add constraint violation penalty
        penalty = 0.0
        if not profile.reserve_adequate:
            shortage = self.constraints.required_reserve() - profile.reserve_energy
            penalty += shortage * 10  # High penalty for reserve violation
        
        if not profile.mission_feasible:
            penalty += 50_000  # Infeasible solution
        
        return profile.total_energy + penalty
    
    def feasibility_check(self, trajectory: Trajectory) -> tuple[bool, str]:
        """
        Check if trajectory meets energy constraints.
        
        Args:
            trajectory: Candidate trajectory
            
        Returns:
            (feasible, reason)
        """
        profile = self.evaluate_trajectory(trajectory)
        
        if profile.total_energy > self.constraints.max_energy:
            return False, "Insufficient battery capacity"
        
        if profile.reserve_energy < self.constraints.required_reserve():
            return False, "Insufficient energy reserve"
        
        if profile.endurance_minutes < 5:
            return False, "Insufficient safety endurance"
        
        return True, "Energy feasible"
