"""
Energy Optimizer - Phase 2.3

Predicts energy consumption for each flight mode and recommends
the most efficient mode selection for mission completion.

Models:
- Hover power consumption (rotor-dependent)
- Cruise power consumption (speed-dependent)
- Transition costs (blending overhead)
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class VehicleEnergyModel:
    """Physics-based energy consumption model for eVTOL."""
    # Vehicle parameters
    mass_kg: float = 2500.0
    hover_thrust_N: float = 24525.0  # = m*g
    max_cruise_speed_ms: float = 30.0
    design_cruise_speed_ms: float = 20.0
    
    # Rotor parameters
    rotor_power_factor: float = 1.2    # Factor for induced + profile power
    rotor_radius_m: float = 2.5
    
    # Aerodynamic parameters
    wing_area_m2: float = 50.0
    drag_coefficient: float = 0.05
    air_density_kgm3: float = 1.225
    
    # Motor efficiency
    rotor_motor_efficiency: float = 0.85
    forward_motor_efficiency: float = 0.90
    
    # Transmission
    drivetrain_efficiency: float = 0.95
    
    def hover_power_kw(self, altitude_m: float = 0.0, wind_speed_ms: float = 0.0) -> float:
        """
        Calculate hover power consumption.
        
        Uses simplified momentum theory:
        P_hover = T * v_i
        where v_i = induced velocity
        
        Args:
            altitude_m: Altitude (affects air density)
            wind_speed_ms: Wind speed (increases power needed)
            
        Returns:
            Power consumption in kW
        """
        # Air density correction for altitude
        rho = self.air_density_kgm3 * np.exp(-altitude_m / 8500.0)
        
        # Induced velocity for hover
        disk_area = np.pi * self.rotor_radius_m**2
        v_induced = np.sqrt(self.hover_thrust_N / (2 * rho * disk_area))
        
        # Induced power
        p_induced = self.hover_thrust_N * v_induced
        
        # Wind effect (headwind increases power, tailwind decreases)
        wind_factor = 1.0 + 0.1 * max(0, wind_speed_ms)
        
        # Total power with efficiency
        p_total_w = p_induced * self.rotor_power_factor * wind_factor / self.rotor_motor_efficiency
        
        return p_total_w / 1000.0  # Convert to kW
    
    def cruise_power_kw(self, speed_ms: float, altitude_m: float = 0.0, wind_speed_ms: float = 0.0) -> float:
        """
        Calculate cruise power consumption.
        
        Combines:
        1. Rotor power (residual hover support)
        2. Forward motor power (propulsion against drag)
        
        Args:
            speed_ms: Cruise speed (m/s)
            altitude_m: Altitude
            wind_speed_ms: Wind speed
            
        Returns:
            Power consumption in kW
        """
        # Airspeed relative to wind
        v_air = speed_ms + wind_speed_ms  # Simplified headwind model
        v_air = max(v_air, speed_ms * 0.5)  # Min 50% speed
        
        # Drag force
        drag_force_n = 0.5 * self.air_density_kgm3 * self.drag_coefficient * self.wing_area_m2 * v_air**2
        
        # Forward propulsion power
        p_forward_w = drag_force_n * v_air / (self.forward_motor_efficiency * self.drivetrain_efficiency)
        
        # Residual rotor power (5-10% of hover at cruise)
        residual_rotor_power_kw = self.hover_power_kw(altitude_m) * 0.07
        
        p_total_w = p_forward_w + residual_rotor_power_kw * 1000.0
        
        return p_total_w / 1000.0  # Convert to kW
    
    def transition_power_kw(self, progress: float, altitude_m: float = 0.0) -> float:
        """
        Estimate power during transition (blend of hover and cruise).
        
        Args:
            progress: Transition progress [0,1] (0=hover, 1=cruise)
            altitude_m: Altitude
            
        Returns:
            Average power consumption in kW
        """
        p_hover = self.hover_power_kw(altitude_m)
        p_cruise = self.cruise_power_kw(self.design_cruise_speed_ms * 0.5, altitude_m)  # Half speed in transition
        
        # Blend (transition power is higher than simple average)
        blending_factor = 1.1  # 10% overhead during blending
        return (p_hover * (1 - progress) + p_cruise * progress) * blending_factor
    
    def energy_to_fly_distance_kwh(self, distance_m: float, speed_ms: float, altitude_m: float = 0.0) -> float:
        """
        Calculate energy required to fly a distance.
        
        Args:
            distance_m: Distance to fly (m)
            speed_ms: Average speed (m/s)
            altitude_m: Altitude (m)
            
        Returns:
            Energy required (kWh)
        """
        if speed_ms < 0.5:
            return float('inf')  # Can't hover indefinitely
        
        time_hours = distance_m / (speed_ms * 3600.0)
        power_kw = self.cruise_power_kw(speed_ms, altitude_m)
        
        return power_kw * time_hours
    
    def max_hover_time_s(self, battery_capacity_kwh: float, altitude_m: float = 0.0, soc: float = 1.0) -> float:
        """
        Calculate maximum hover duration.
        
        Args:
            battery_capacity_kwh: Battery capacity (kWh)
            altitude_m: Altitude (m)
            soc: State of charge [0,1]
            
        Returns:
            Maximum hover time (seconds)
        """
        available_energy = battery_capacity_kwh * soc
        hover_power = self.hover_power_kw(altitude_m)
        
        if hover_power < 0.1:
            return float('inf')
        
        return (available_energy / hover_power) * 3600.0  # Convert to seconds
    
    def max_cruise_range_m(self, battery_capacity_kwh: float, cruise_speed_ms: float, altitude_m: float = 0.0, soc: float = 1.0) -> float:
        """
        Calculate maximum cruise range.
        
        Args:
            battery_capacity_kwh: Battery capacity (kWh)
            cruise_speed_ms: Cruise speed (m/s)
            altitude_m: Altitude (m)
            soc: State of charge [0,1]
            
        Returns:
            Maximum range (meters)
        """
        available_energy = battery_capacity_kwh * soc
        cruise_power = self.cruise_power_kw(cruise_speed_ms, altitude_m)
        
        if cruise_power < 0.1:
            return float('inf')
        
        time_hours = available_energy / cruise_power
        return cruise_speed_ms * time_hours * 3600.0  # time_hours to seconds


@dataclass
class EnergyOptimizationResult:
    """Result of energy optimization analysis."""
    recommended_mode: str  # 'HOVER', 'TRANSITION', 'CRUISE'
    mode_confidence: float  # [0,1]
    energy_score: float  # [0,1] efficiency score
    
    hover_available_time_s: float
    cruise_available_range_m: float
    transition_energy_cost_kwh: float
    
    reasoning: str


class EnergyOptimizer:
    """
    Recommends optimal flight mode based on energy analysis.
    
    Considers:
    - Remaining battery capacity
    - Distance to destination
    - Time available
    - Power consumption of each mode
    """
    
    def __init__(self, vehicle_model: Optional[VehicleEnergyModel] = None):
        self.vehicle = vehicle_model or VehicleEnergyModel()
        self.battery_capacity_kwh = 100.0  # Default: 100 kWh battery
    
    def optimize_mode(
        self,
        distance_remaining_m: float,
        time_remaining_s: float,
        battery_soc: float,
        current_altitude_m: float = 0.0,
        wind_speed_ms: float = 0.0,
        threat_level: float = 0.0,
    ) -> EnergyOptimizationResult:
        """
        Optimize mode selection based on energy and mission constraints.
        
        Returns:
            EnergyOptimizationResult with recommendation
        """
        
        # Available energy
        available_kwh = battery_soc * self.battery_capacity_kwh
        
        # Calculate metrics for each mode
        
        # 1. HOVER analysis
        hover_time = self.vehicle.max_hover_time_s(
            self.battery_capacity_kwh, current_altitude_m, battery_soc
        )
        hover_available_time_s = hover_time
        
        # 2. CRUISE analysis
        cruise_range = self.vehicle.max_cruise_range_m(
            self.battery_capacity_kwh,
            self.vehicle.design_cruise_speed_ms,
            current_altitude_m,
            battery_soc
        )
        cruise_available_range_m = cruise_range
        
        # 3. TRANSITION analysis (temporary, not sustained)
        transition_energy = self.vehicle.transition_power_kw(0.5, current_altitude_m) * (5.0 / 3600.0)  # 5s transition
        
        # Decision logic
        
        # Option A: Can we reach destination in cruise?
        if distance_remaining_m < cruise_range * 0.9:  # 90% safety margin
            # Calculate energy needed
            cruise_power = self.vehicle.cruise_power_kw(
                self.vehicle.design_cruise_speed_ms, current_altitude_m, wind_speed_ms
            )
            time_needed_hours = distance_remaining_m / (self.vehicle.design_cruise_speed_ms * 3600.0)
            energy_needed_kwh = cruise_power * time_needed_hours
            
            if energy_needed_kwh < available_kwh * 0.8:  # 80% battery reserve
                return EnergyOptimizationResult(
                    recommended_mode='CRUISE',
                    mode_confidence=0.9,
                    energy_score=0.85,
                    hover_available_time_s=hover_available_time_s,
                    cruise_available_range_m=cruise_available_range_m,
                    transition_energy_cost_kwh=transition_energy,
                    reasoning=f"Sufficient energy to reach destination via cruise. Range: {cruise_available_range_m/1000:.1f}km"
                )
        
        # Option B: Transition better than hover?
        hover_power = self.vehicle.hover_power_kw(current_altitude_m)
        cruise_power = self.vehicle.cruise_power_kw(
            self.vehicle.design_cruise_speed_ms, current_altitude_m
        )
        
        if cruise_power < hover_power * 0.75:  # Cruise is ≥25% more efficient
            if threat_level < 0.5:  # No high threat
                return EnergyOptimizationResult(
                    recommended_mode='TRANSITION',
                    mode_confidence=0.75,
                    energy_score=0.7,
                    hover_available_time_s=hover_available_time_s,
                    cruise_available_range_m=cruise_available_range_m,
                    transition_energy_cost_kwh=transition_energy,
                    reasoning=f"Cruise is significantly more efficient. Switch to TRANSITION for gradual acceleration."
                )
        
        # Option C: Default to hover for safety/control
        return EnergyOptimizationResult(
            recommended_mode='HOVER',
            mode_confidence=0.6,
            energy_score=0.3,
            hover_available_time_s=hover_available_time_s,
            cruise_available_range_m=cruise_available_range_m,
            transition_energy_cost_kwh=transition_energy,
            reasoning=f"Conservative choice for safety. Hover time available: {hover_available_time_s/60:.1f}min"
        )


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    'EnergyOptimizer',
    'VehicleEnergyModel',
    'EnergyOptimizationResult',
]
