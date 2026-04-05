"""
Vehicle Configuration - Comprehensive Parameter Definitions

This module defines all configuration dataclasses for the tiltrotor vehicle model.
Parameters are based on realistic defense eVTOL specifications similar to:
- Bell V-280 Valor (scaled down for electric)
- AgustaWestland AW609
- Joby S4 (for electric propulsion reference)

All units are SI unless otherwise noted.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Any
from enum import Enum
import json
from pathlib import Path


# Import canonical FlightPhase — do NOT redefine locally.
# The canonical definition lives in core.state and has the full set of phases.
from ..core.state import FlightPhase  # noqa: F401  (re-exported for compatibility)


class NacelleMode(Enum):
    """Nacelle configuration mode."""
    VERTICAL = "vertical"      # 90° - Full hover
    TRANSITION = "transition"  # 0° < angle < 90°
    HORIZONTAL = "horizontal"  # 0° - Full cruise


@dataclass
class MassProperties:
    """
    Vehicle mass properties.

    Reference: Based on scaled V-280 Valor for electric propulsion.

    Attributes:
        empty_mass: Operational empty mass [kg]
        battery_mass: Battery pack mass [kg]
        payload_mass: Mission payload [kg]
        fuel_mass: Reserve fuel for hybrid (0 for pure electric) [kg]
        cg_position: Center of gravity in body frame [m]
        inertia_tensor: Moments and products of inertia [kg·m²]
    """
    empty_mass: float = 800.0          # kg - airframe, motors, avionics
    battery_mass: float = 400.0        # kg - high energy density pack
    payload_mass: float = 150.0        # kg - sensors, weapons, cargo
    fuel_mass: float = 0.0             # kg - pure electric

    # CG position in body frame [x, y, z] (forward, right, down)
    cg_position: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))

    # Inertia tensor [Ixx, Iyy, Izz, Ixy, Ixz, Iyz] in kg·m²
    # Typical tiltrotor: long fuselage → high Iyy, wide rotors → high Ixx
    inertia_tensor: np.ndarray = field(default_factory=lambda: np.array([
        800.0,    # Ixx - roll
        2500.0,   # Iyy - pitch
        2800.0,   # Izz - yaw
        0.0,      # Ixy
        50.0,     # Ixz
        0.0       # Iyz
    ]))

    @property
    def total_mass(self) -> float:
        """Total vehicle mass [kg]."""
        return self.empty_mass + self.battery_mass + self.payload_mass + self.fuel_mass

    @property
    def inertia_matrix(self) -> np.ndarray:
        """Full 3x3 inertia tensor."""
        Ixx, Iyy, Izz, Ixy, Ixz, Iyz = self.inertia_tensor
        return np.array([
            [Ixx, -Ixy, -Ixz],
            [-Ixy, Iyy, -Iyz],
            [-Ixz, -Iyz, Izz]
        ])


@dataclass
class RotorConfig:
    """
    Individual rotor configuration.

    Based on scaled tiltrotor proprotor design.

    Attributes:
        radius: Rotor radius [m]
        num_blades: Number of blades
        chord: Average blade chord [m]
        twist: Linear blade twist [deg]
        solidity: Rotor solidity σ = Nb·c/(πR)
        tip_mach_limit: Maximum tip Mach number
        collective_range: [min, max] collective pitch [deg]
        rpm_range: [min, max] rotor RPM
    """
    radius: float = 2.0                    # m
    num_blades: int = 4
    chord: float = 0.15                    # m (average)
    root_chord: float = 0.20               # m
    tip_chord: float = 0.10                # m (tapered)
    twist: float = -12.0                   # deg (root to tip)

    # Airfoil characteristics
    cl_alpha: float = 5.73                 # 1/rad (thin airfoil theory)
    cl_max: float = 1.4                    # Maximum lift coefficient
    cd0: float = 0.01                      # Zero-lift drag

    # Operating limits
    tip_mach_limit: float = 0.85           # Compressibility limit
    collective_range: tuple[float, float] = (-5.0, 25.0)  # deg
    rpm_range: tuple[float, float] = (800.0, 3000.0)      # RPM

    # Rotor location in body frame [x, y, z]
    position: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))

    # Direction of rotation (+1 = CCW from above, -1 = CW)
    rotation_direction: int = 1

    @property
    def solidity(self) -> float:
        """Rotor solidity σ = Nb·c/(πR)."""
        return self.num_blades * self.chord / (np.pi * self.radius)

    @property
    def disk_area(self) -> float:
        """Rotor disk area [m²]."""
        return np.pi * self.radius ** 2


@dataclass
class NacelleConfig:
    """
    Nacelle/pylon configuration for tiltrotor.

    Attributes:
        tilt_range: [min, max] nacelle tilt angle [deg]
                    0° = horizontal (cruise), 90° = vertical (hover)
        tilt_rate_limit: Maximum tilt rate [deg/s]
        tilt_time_constant: First-order dynamics time constant [s]
    """
    tilt_range: tuple[float, float] = (0.0, 95.0)  # deg, slight over-tilt for decel
    tilt_rate_limit: float = 10.0                   # deg/s
    tilt_time_constant: float = 0.3                 # s

    # Nacelle mass (affects CG shift during tilt)
    nacelle_mass: float = 50.0     # kg each

    # Position of nacelle pivot in body frame
    pivot_position: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))


@dataclass
class MotorConfig:
    """
    PMSM (Permanent Magnet Synchronous Motor) configuration.

    Based on high-power aerospace electric motors.

    Attributes:
        max_power: Maximum continuous power [kW]
        peak_power: Peak power (30s rating) [kW]
        max_torque: Maximum torque [N·m]
        max_rpm: Maximum motor RPM
        efficiency_peak: Peak efficiency
        kv: Motor velocity constant [RPM/V]
        kt: Torque constant [N·m/A]
        resistance: Phase resistance [Ω]
        inductance: Phase inductance [H]
        thermal_mass: Thermal mass [J/K]
        max_temperature: Maximum winding temperature [°C]
    """
    max_power: float = 150.0       # kW continuous
    peak_power: float = 200.0      # kW peak (30s)
    max_torque: float = 500.0      # N·m
    max_rpm: float = 4000.0        # RPM

    efficiency_peak: float = 0.95
    efficiency_map: np.ndarray | None = None  # 2D lookup [torque, rpm] → η

    # Electromagnetic parameters
    kv: float = 10.0               # RPM/V
    kt: float = 0.955              # N·m/A (kt = 60/(2π·kv))
    resistance: float = 0.02       # Ω per phase
    inductance: float = 0.0005     # H per phase

    # Thermal parameters
    thermal_mass: float = 5000.0   # J/K
    thermal_resistance: float = 0.1  # K/W to ambient
    max_temperature: float = 150.0   # °C


@dataclass
class PropulsionConfig:
    """
    Complete propulsion system configuration.

    For tiltrotor: 2 main proprotors + optional pusher.
    """
    # Left rotor (body-left wingtip)
    left_rotor: RotorConfig = field(default_factory=lambda: RotorConfig(
        position=np.array([0.5, -3.5, -0.5]),  # Forward, left, up
        rotation_direction=1  # CCW from above
    ))

    # Right rotor (body-right wingtip)
    right_rotor: RotorConfig = field(default_factory=lambda: RotorConfig(
        position=np.array([0.5, 3.5, -0.5]),   # Forward, right, up
        rotation_direction=-1  # CW from above (counter-rotating)
    ))

    # Left nacelle
    left_nacelle: NacelleConfig = field(default_factory=lambda: NacelleConfig(
        pivot_position=np.array([0.0, -3.5, 0.0])
    ))

    # Right nacelle
    right_nacelle: NacelleConfig = field(default_factory=lambda: NacelleConfig(
        pivot_position=np.array([0.0, 3.5, 0.0])
    ))

    # Motor configuration (same for both)
    motor: MotorConfig = field(default_factory=MotorConfig)

    # Gear ratio (motor RPM / rotor RPM)
    gear_ratio: float = 2.0
    gear_efficiency: float = 0.98


@dataclass
class WingConfig:
    """
    Wing aerodynamic configuration.

    Tiltrotor wing: high aspect ratio, moderate sweep.
    """
    span: float = 10.0              # m (tip to tip)
    chord_root: float = 1.5         # m
    chord_tip: float = 0.8          # m
    sweep_le: float = 5.0           # deg (leading edge sweep)
    dihedral: float = 2.0           # deg
    twist: float = -3.0             # deg (washout)
    incidence: float = 2.0          # deg (wing setting angle)

    # Airfoil data
    airfoil: str = "NACA 23015"
    cl_alpha: float = 5.5           # 1/rad
    cl_max: float = 1.5
    cl0: float = 0.3                # Zero-AoA lift coefficient
    cd0: float = 0.008              # Zero-lift drag
    cd_k: float = 0.04              # Induced drag factor k (CD = CD0 + k·CL²)
    cm0: float = -0.05              # Zero-lift pitching moment

    # Control surfaces
    flap_span_fraction: float = 0.6   # Fraction of span with flaps
    flap_chord_fraction: float = 0.25  # Fraction of chord
    flap_max_deflection: float = 40.0  # deg

    aileron_span_fraction: float = 0.3
    aileron_chord_fraction: float = 0.2
    aileron_max_deflection: float = 25.0  # deg

    @property
    def area(self) -> float:
        """Wing planform area [m²]."""
        return 0.5 * self.span * (self.chord_root + self.chord_tip)

    @property
    def aspect_ratio(self) -> float:
        """Wing aspect ratio."""
        return self.span ** 2 / self.area

    @property
    def mean_chord(self) -> float:
        """Mean aerodynamic chord [m]."""
        taper = self.chord_tip / self.chord_root
        return (2/3) * self.chord_root * (1 + taper + taper**2) / (1 + taper)


@dataclass
class FuselageConfig:
    """Fuselage aerodynamic configuration."""
    length: float = 8.0             # m
    width: float = 1.8              # m
    height: float = 2.0             # m

    # Aerodynamic reference areas
    frontal_area: float = 2.5       # m² (for drag)
    wetted_area: float = 40.0       # m²

    # Drag coefficients
    cd_frontal: float = 0.3         # Based on frontal area

    # Tail configuration
    horizontal_tail_area: float = 3.0   # m²
    vertical_tail_area: float = 2.5     # m²
    tail_arm: float = 4.0               # m (distance from CG to tail AC)


@dataclass
class ControlSurfaceConfig:
    """Control surface configuration."""
    # Elevator
    elevator_area: float = 2.0      # m²
    elevator_arm: float = 4.0       # m
    elevator_max_deflection: float = 25.0  # deg
    elevator_effectiveness: float = 0.4    # d(CL)/d(δe)

    # Rudder
    rudder_area: float = 1.5        # m²
    rudder_arm: float = 4.0         # m
    rudder_max_deflection: float = 30.0    # deg
    rudder_effectiveness: float = 0.3      # d(CY)/d(δr)

    # Rate limits
    surface_rate_limit: float = 60.0  # deg/s


@dataclass
class AerodynamicsConfig:
    """Complete aerodynamics configuration."""
    wing: WingConfig = field(default_factory=WingConfig)
    fuselage: FuselageConfig = field(default_factory=FuselageConfig)
    control_surfaces: ControlSurfaceConfig = field(default_factory=ControlSurfaceConfig)

    # Atmosphere
    air_density_sl: float = 1.225   # kg/m³ at sea level
    speed_of_sound_sl: float = 340.0  # m/s at sea level

    # Ground effect parameters
    ground_effect_height: float = 10.0  # m (significant below this)


@dataclass
class BatteryCellConfig:
    """Individual battery cell configuration."""
    chemistry: str = "NMC811"       # LiNiMnCoO2 high energy
    capacity_ah: float = 5.0        # Ah per cell
    voltage_nominal: float = 3.7    # V
    voltage_max: float = 4.2        # V
    voltage_min: float = 3.0        # V

    # Internal resistance model (SOC and temperature dependent)
    resistance_base: float = 0.02   # Ω at 25°C, 50% SOC

    # Thermal parameters
    mass: float = 0.070             # kg per cell
    specific_heat: float = 1000.0   # J/kg·K

    # Discharge characteristics
    max_c_rate_continuous: float = 3.0   # 3C continuous
    max_c_rate_peak: float = 8.0         # 8C for 10s


@dataclass
class BatteryPackConfig:
    """Battery pack configuration."""
    cell: BatteryCellConfig = field(default_factory=BatteryCellConfig)

    # Pack topology (series x parallel)
    cells_series: int = 108          # ~400V nominal (108 × 3.7V)
    cells_parallel: int = 20         # 100Ah capacity (20 × 5Ah)

    # Pack-level parameters
    pack_overhead_mass: float = 50.0  # kg (BMS, housing, cooling)
    cooling_power: float = 2000.0     # W maximum cooling capacity

    @property
    def total_cells(self) -> int:
        return self.cells_series * self.cells_parallel

    @property
    def capacity_kwh(self) -> float:
        """Total pack capacity [kWh]."""
        return (self.cell.capacity_ah * self.cells_parallel *
                self.cell.voltage_nominal * self.cells_series) / 1000

    @property
    def voltage_nominal(self) -> float:
        """Nominal pack voltage [V]."""
        return self.cell.voltage_nominal * self.cells_series

    @property
    def mass(self) -> float:
        """Total battery mass [kg]."""
        return self.total_cells * self.cell.mass + self.pack_overhead_mass


@dataclass
class BatteryConfig:
    """Complete battery configuration."""
    pack: BatteryPackConfig = field(default_factory=BatteryPackConfig)

    # Operating limits
    soc_min: float = 0.10           # 10% minimum (reserve)
    soc_max: float = 0.95           # 95% maximum (longevity)
    temperature_min: float = 0.0     # °C
    temperature_max: float = 45.0    # °C

    # Thermal management
    heating_power: float = 1000.0    # W (for cold start)
    cooling_power: float = 3000.0    # W

    # Reserve requirements (defense mission)
    reserve_fraction: float = 0.20   # 20% reserve for RTB


@dataclass
class RCSConfig:
    """Radar Cross Section configuration."""
    # Base RCS values at different aspects [m²]
    # Aspect angles: 0°=nose, 90°=broadside, 180°=tail
    base_rcs_nose: float = 0.5       # m² (reduced by design)
    base_rcs_broadside: float = 2.0  # m² (wing/nacelle contribution)
    base_rcs_tail: float = 0.8       # m²
    base_rcs_top: float = 3.0        # m² (rotor disk visible)
    base_rcs_bottom: float = 2.5     # m²

    # Frequency bands of interest
    frequency_bands: dict[str, tuple[float, float]] = field(default_factory=lambda: {
        "L": (1e9, 2e9),      # L-band (1-2 GHz)
        "S": (2e9, 4e9),      # S-band (2-4 GHz)
        "C": (4e9, 8e9),      # C-band (4-8 GHz)
        "X": (8e9, 12e9),     # X-band (8-12 GHz)
        "Ku": (12e9, 18e9),   # Ku-band (12-18 GHz)
    })

    # Rotor modulation (blade flash) - adds to RCS at rotor harmonics
    blade_flash_amplitude: float = 0.3  # m² peak


@dataclass
class IRConfig:
    """Infrared signature configuration."""
    # Emissivity values
    fuselage_emissivity: float = 0.85
    engine_emissivity: float = 0.90
    rotor_emissivity: float = 0.80

    # Temperature sources
    motor_operating_temp: float = 80.0    # °C
    battery_operating_temp: float = 35.0  # °C
    exhaust_temp: float = 0.0             # °C (pure electric - none!)

    # Suppression features
    ir_suppression_factor: float = 0.7    # 30% reduction from design


@dataclass
class AcousticConfig:
    """Acoustic signature configuration."""
    # Reference noise levels at 100m distance
    rotor_noise_hover: float = 75.0    # dBA
    rotor_noise_cruise: float = 70.0   # dBA
    motor_noise: float = 55.0          # dBA

    # Blade passage frequency
    bpf_multiplier: float = 1.0        # Adjusts rotor harmonic content

    # Directivity (nose-on typically quieter)
    directivity_index: float = 3.0     # dB variation with aspect


@dataclass
class SignatureConfig:
    """Complete signature configuration."""
    rcs: RCSConfig = field(default_factory=RCSConfig)
    ir: IRConfig = field(default_factory=IRConfig)
    acoustic: AcousticConfig = field(default_factory=AcousticConfig)


@dataclass
class FlightEnvelopeConfig:
    """Flight envelope limits."""
    # Speed limits
    v_ne: float = 150.0       # m/s - never exceed
    v_max_cruise: float = 120.0  # m/s - max cruise
    v_min_cruise: float = 40.0   # m/s - min cruise (stall margin)

    # Altitude limits
    altitude_max: float = 6000.0   # m
    altitude_ceiling: float = 4500.0  # m (service ceiling)

    # Rate limits
    climb_rate_max: float = 15.0    # m/s
    descent_rate_max: float = 20.0  # m/s

    # Load factor limits
    n_max_positive: float = 3.5     # g
    n_max_negative: float = -1.5    # g

    # Bank angle limit
    bank_angle_max: float = 60.0    # deg

    # Transition corridor
    transition_speed_min: float = 30.0   # m/s
    transition_speed_max: float = 80.0   # m/s


@dataclass
class TiltrotorConfig:
    """
    Complete tiltrotor vehicle configuration.

    This is the top-level configuration class that aggregates all
    subsystem configurations.
    """
    name: str = "Defense eVTOL Tiltrotor"
    designation: str = "TR-150"

    mass: MassProperties = field(default_factory=MassProperties)
    propulsion: PropulsionConfig = field(default_factory=PropulsionConfig)
    aerodynamics: AerodynamicsConfig = field(default_factory=AerodynamicsConfig)
    battery: BatteryConfig = field(default_factory=BatteryConfig)
    signatures: SignatureConfig = field(default_factory=SignatureConfig)
    envelope: FlightEnvelopeConfig = field(default_factory=FlightEnvelopeConfig)

    # Simulation parameters
    integration_dt: float = 0.01     # s (100 Hz simulation)
    control_dt: float = 0.02         # s (50 Hz control)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        # Custom serialization for numpy arrays
        def serialize(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif hasattr(obj, '__dataclass_fields__'):
                return {k: serialize(getattr(obj, k))
                       for k in obj.__dataclass_fields__}
            elif isinstance(obj, dict):
                return {k: serialize(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [serialize(v) for v in obj]
            elif isinstance(obj, Enum):
                return obj.value
            else:
                return obj
        return serialize(self)

    def save(self, path: Path) -> None:
        """Save configuration to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "TiltrotorConfig":
        """
        Load configuration from a JSON file previously written by ``save()``.

        The loader performs a recursive merge: values present in the file
        override the dataclass defaults, while any keys absent from the file
        fall back to defaults.  Numpy arrays stored as lists are restored.
        """
        with open(path) as f:
            data = json.load(f)

        def _apply(obj: object, d: dict) -> None:
            """Recursively overwrite dataclass fields from dict ``d``."""
            if not isinstance(d, dict):
                return
            for key, value in d.items():
                if not hasattr(obj, key):
                    continue
                current = getattr(obj, key)
                if isinstance(current, np.ndarray):
                    setattr(obj, key, np.array(value, dtype=float))
                elif hasattr(current, "__dataclass_fields__"):
                    _apply(current, value)
                elif isinstance(current, Enum):
                    # Re-instantiate enum by value
                    setattr(obj, key, type(current)(value))
                else:
                    setattr(obj, key, value)

        cfg = cls()
        _apply(cfg, data)
        return cfg


# Alias for convenience
VehicleConfig = TiltrotorConfig
