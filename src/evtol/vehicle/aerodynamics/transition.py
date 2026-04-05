"""
Transition Aerodynamics - Tiltrotor Conversion Corridor Model

This module handles the complex aerodynamics during tiltrotor conversion
from helicopter mode (nacelle 90°) to airplane mode (nacelle 0°).

Theory
======

During transition, lift comes from both rotors and wing:
    L_total = L_rotor × sin(θ_nacelle) + L_wing × q̄

The conversion corridor defines safe operating envelope:
    - Minimum airspeed to maintain wing lift
    - Maximum airspeed to avoid rotor compressibility
    - Nacelle rate limits to avoid excessive attitude changes

Blending Functions
==================

As nacelle tilts forward, rotor wake interaction with wing changes:
    - Download factor: Wing in rotor wake (hover) → reduced wing lift
    - Upload factor: Rotor in wing upwash (cruise) → increased thrust

    K_download(θ_nac) = 1 - 0.05 × (θ_nac / 90°)²
    K_upload(θ_nac) = 1 + 0.02 × (1 - θ_nac / 90°)²

Conversion Corridor (V-280 Valor typical):
    - Hover: θ_nac = 90°, V = 0-30 kts
    - Conversion: θ_nac = 90° → 0°, V = 80-160 kts
    - Cruise: θ_nac = 0°, V > 150 kts
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class TransitionState:
    """State during tiltrotor transition."""
    # Nacelle angle (90° = hover, 0° = cruise)
    nacelle_angle_deg: float = 90.0
    nacelle_rate_dps: float = 0.0  # deg/s

    # Lift contributions
    lift_rotor: float = 0.0    # N from rotors
    lift_wing: float = 0.0     # N from wing
    lift_total: float = 0.0    # Total N

    # Drag contributions
    drag_rotor: float = 0.0    # N
    drag_fuselage: float = 0.0 # N
    drag_wing: float = 0.0     # N (induced + profile)
    drag_total: float = 0.0    # Total N

    # Interaction factors
    download_factor: float = 1.0  # Wing download from rotor
    upload_factor: float = 1.0    # Rotor upload from wing

    # Corridor status
    within_corridor: bool = True
    corridor_margin: float = 1.0  # Distance to corridor boundary (1 = centered)


@dataclass
class ConversionCorridorConfig:
    """Configuration for conversion corridor limits."""
    # Nacelle limits
    nacelle_min_deg: float = 0.0     # Full airplane mode
    nacelle_max_deg: float = 95.0    # Slightly past vertical for hover
    nacelle_rate_max_dps: float = 8.0  # Max tilt rate

    # Speed boundaries by nacelle angle (deg: (V_min_kts, V_max_kts))
    corridor_boundaries: dict[float, tuple[float, float]] = field(default_factory=lambda: {
        90.0: (0.0, 60.0),      # Hover mode
        75.0: (40.0, 100.0),    # Early conversion
        60.0: (60.0, 140.0),    # Mid conversion
        45.0: (80.0, 180.0),    # Late conversion
        30.0: (100.0, 220.0),   # Near airplane
        15.0: (130.0, 280.0),   # Almost airplane
        0.0: (150.0, 350.0),    # Full airplane
    })

    # Altitude corrections (higher altitude = higher speeds)
    altitude_correction_factor: float = 0.02  # Per 1000m

    # Safety margins
    margin_min_speed_kts: float = 10.0   # Buffer below minimum
    margin_max_speed_kts: float = 20.0   # Buffer below maximum


class TransitionAerodynamics:
    """
    Tiltrotor transition aerodynamics model.

    This model blends rotor and wing aerodynamics during conversion,
    accounting for mutual interference effects and enforcing the
    conversion corridor constraints.

    Reference: Bell V-280 Valor flight test data (declassified portions)
    """

    def __init__(
        self,
        corridor_config: ConversionCorridorConfig | None = None,
    ):
        """
        Initialize transition model.

        Args:
            corridor_config: Conversion corridor configuration
        """
        self.corridor = corridor_config or ConversionCorridorConfig()
        self.state = TransitionState()

        # Pre-compute corridor interpolation
        self._setup_corridor_interpolation()

        logger.info("Transition aerodynamics model initialized")

    def _setup_corridor_interpolation(self) -> None:
        """Set up interpolation for corridor boundaries."""
        angles = sorted(self.corridor.corridor_boundaries.keys(), reverse=True)
        self._corridor_angles = np.array(angles)

        v_mins = []
        v_maxs = []
        for angle in angles:
            v_min, v_max = self.corridor.corridor_boundaries[angle]
            v_mins.append(v_min)
            v_maxs.append(v_max)

        self._v_min_kts = np.array(v_mins)
        self._v_max_kts = np.array(v_maxs)

    def get_corridor_limits(
        self,
        nacelle_angle_deg: float,
        altitude_m: float = 0.0,
    ) -> tuple[float, float]:
        """
        Get min/max airspeed for current nacelle angle.

        Args:
            nacelle_angle_deg: Current nacelle angle (0-90°)
            altitude_m: Current altitude for density correction

        Returns:
            (V_min_kts, V_max_kts) airspeed limits
        """
        # Interpolate corridor boundaries
        nacelle_angle_deg = np.clip(nacelle_angle_deg, 0.0, 90.0)

        v_min = np.interp(nacelle_angle_deg, self._corridor_angles[::-1],
                         self._v_min_kts[::-1])
        v_max = np.interp(nacelle_angle_deg, self._corridor_angles[::-1],
                         self._v_max_kts[::-1])

        # Altitude correction (higher altitude requires higher TAS)
        alt_factor = 1.0 + self.corridor.altitude_correction_factor * (altitude_m / 1000.0)
        v_min *= alt_factor
        v_max *= alt_factor

        return float(v_min), float(v_max)

    def check_corridor(
        self,
        nacelle_angle_deg: float,
        airspeed_kts: float,
        altitude_m: float = 0.0,
    ) -> tuple[bool, float, str]:
        """
        Check if current state is within conversion corridor.

        Args:
            nacelle_angle_deg: Current nacelle angle
            airspeed_kts: Current airspeed
            altitude_m: Current altitude

        Returns:
            (within_corridor, margin, message)
        """
        v_min, v_max = self.get_corridor_limits(nacelle_angle_deg, altitude_m)

        # Add safety margins
        v_min_safe = v_min + self.corridor.margin_min_speed_kts
        v_max_safe = v_max - self.corridor.margin_max_speed_kts

        if airspeed_kts < v_min:
            margin = (airspeed_kts - v_min) / self.corridor.margin_min_speed_kts
            return False, margin, f"Below minimum speed ({v_min:.0f} kts)"

        if airspeed_kts > v_max:
            margin = (v_max - airspeed_kts) / self.corridor.margin_max_speed_kts
            return False, margin, f"Above maximum speed ({v_max:.0f} kts)"

        # Compute margin (distance to nearest boundary, normalized)
        margin_low = (airspeed_kts - v_min) / (v_min_safe - v_min) if airspeed_kts < v_min_safe else 1.0
        margin_high = (v_max - airspeed_kts) / (v_max - v_max_safe) if airspeed_kts > v_max_safe else 1.0
        margin = min(margin_low, margin_high)

        return True, margin, "Within corridor"

    def compute_download_factor(
        self,
        nacelle_angle_deg: float,
        rotor_thrust: float,
        wing_area: float,
        airspeed: float,
        rho: float = 1.225,
    ) -> float:
        """
        Compute wing download factor due to rotor wake.

        In hover, the rotor wake impinges on the wing, reducing effective lift.
        As nacelle tilts forward and airspeed increases, this effect diminishes.

        Args:
            nacelle_angle_deg: Nacelle angle (90 = hover)
            rotor_thrust: Total rotor thrust (N)
            wing_area: Wing area (m²)
            airspeed: Airspeed (m/s)
            rho: Air density (kg/m³)

        Returns:
            Download factor (1.0 = no download, <1 = reduced lift)
        """
        # Nacelle angle effect (download strongest in hover)
        theta = np.radians(nacelle_angle_deg)
        nac_factor = np.sin(theta) ** 2  # 1.0 at 90°, 0 at 0°

        # Dynamic pressure ratio (download decreases with speed)
        if airspeed > 1.0:
            # Disk loading
            disk_area = 2.0 * (np.pi * 2.0**2)  # Approximate two rotors
            disk_loading = rotor_thrust / disk_area
            q_dyn = 0.5 * rho * airspeed**2  # noqa: F841 (dynamic pressure, for reference)

            # Download velocity ratio
            v_download = np.sqrt(disk_loading / (2 * rho))
            speed_factor = 1.0 / (1.0 + (airspeed / v_download)**2)
        else:
            speed_factor = 1.0

        # Maximum download is about 5-10% of rotor thrust
        download_fraction = 0.08 * nac_factor * speed_factor

        # Return as multiplicative factor on wing lift
        return 1.0 - download_fraction

    def compute_upload_factor(
        self,
        nacelle_angle_deg: float,
        wing_circulation: float,
        rotor_radius: float,
    ) -> float:
        """
        Compute rotor upload factor due to wing upwash.

        In airplane mode, the wing creates upwash that increases
        effective angle of attack at the rotor disk.

        Args:
            nacelle_angle_deg: Nacelle angle (0 = airplane)
            wing_circulation: Wing bound circulation (m²/s)
            rotor_radius: Rotor radius (m)

        Returns:
            Upload factor (1.0 = no upload, >1 = increased thrust)
        """
        # Upload effect increases as nacelle tilts forward
        theta = np.radians(nacelle_angle_deg)
        nac_factor = np.cos(theta) ** 2  # 0 at 90°, 1 at 0°

        # Upwash-induced angle change (simplified)
        # In reality, this depends on rotor position relative to wing
        delta_alpha = 0.02 * nac_factor * np.sign(wing_circulation)

        # Thrust increase ~ 2π × Δα for linear regime
        upload_factor = 1.0 + 2 * np.pi * abs(delta_alpha) * nac_factor

        return min(upload_factor, 1.05)  # Cap at 5% increase

    def blend_forces(
        self,
        nacelle_angle_deg: float,
        rotor_thrust: float,
        rotor_drag: float,
        wing_lift: float,
        wing_drag: float,
        fuselage_drag: float,
        download_factor: float = 1.0,
        upload_factor: float = 1.0,
    ) -> tuple[float, float, float]:
        """
        Blend rotor and wing forces during transition.

        Args:
            nacelle_angle_deg: Current nacelle angle
            rotor_thrust: Rotor thrust magnitude (N)
            rotor_drag: Rotor H-force/drag (N)
            wing_lift: Wing lift (N)
            wing_drag: Wing drag (N)
            fuselage_drag: Fuselage drag (N)
            download_factor: Wing download factor
            upload_factor: Rotor upload factor

        Returns:
            (total_lift, total_drag, pitching_moment_contribution)
        """
        theta = np.radians(nacelle_angle_deg)

        # Rotor thrust components
        # Thrust vector tilts with nacelle
        rotor_lift = rotor_thrust * np.sin(theta) * upload_factor
        rotor_forward = rotor_thrust * np.cos(theta)  # Forward thrust component

        # Wing lift with download correction
        effective_wing_lift = wing_lift * download_factor

        # Total lift
        total_lift = rotor_lift + effective_wing_lift

        # Total drag
        # In hover, rotor doesn't create traditional drag
        # In cruise, H-force is the rotor drag
        total_drag = wing_drag + fuselage_drag + rotor_drag

        # Net forward force (thrust - drag)
        # This is what accelerates/decelerates the aircraft
        net_force = rotor_forward - total_drag  # noqa: F841

        # Pitching moment contribution (simplified)
        # Depends on rotor position relative to CG
        # Positive = nose up
        rotor_arm = 0.5  # Approximate moment arm (m)
        wing_arm = -0.3  # Wing AC aft of CG
        pitch_moment = (rotor_forward * rotor_arm * np.sin(theta) +
                       effective_wing_lift * wing_arm)

        # Update state
        self.state.lift_rotor = rotor_lift
        self.state.lift_wing = effective_wing_lift
        self.state.lift_total = total_lift
        self.state.drag_rotor = rotor_drag
        self.state.drag_wing = wing_drag
        self.state.drag_fuselage = fuselage_drag
        self.state.drag_total = total_drag
        self.state.download_factor = download_factor
        self.state.upload_factor = upload_factor

        return total_lift, total_drag, pitch_moment

    def compute_transition_state(
        self,
        nacelle_angle_deg: float,
        nacelle_rate_dps: float,
        airspeed_mps: float,
        altitude_m: float,
        rotor_thrust: float,
        rotor_torque: float,
        wing_area: float = 15.0,
        wing_CL: float = 0.5,
        rho: float = 1.225,
    ) -> TransitionState:
        """
        Compute complete transition aerodynamic state.

        Args:
            nacelle_angle_deg: Nacelle tilt angle
            nacelle_rate_dps: Nacelle tilt rate
            airspeed_mps: Airspeed in m/s
            altitude_m: Altitude in meters
            rotor_thrust: Total rotor thrust
            rotor_torque: Total rotor torque
            wing_area: Wing reference area (m²)
            wing_CL: Wing lift coefficient
            rho: Air density

        Returns:
            Complete TransitionState
        """
        # Convert airspeed to knots for corridor check
        airspeed_kts = airspeed_mps * 1.94384

        # Check corridor
        within_corridor, margin, _ = self.check_corridor(
            nacelle_angle_deg, airspeed_kts, altitude_m
        )

        # Compute wing lift
        q = 0.5 * rho * airspeed_mps**2
        wing_lift = q * wing_area * wing_CL if airspeed_mps > 5.0 else 0.0

        # Compute download factor
        download = self.compute_download_factor(
            nacelle_angle_deg, rotor_thrust, wing_area, airspeed_mps, rho
        )

        # Compute upload factor (simplified - assume circulation ~ CL)
        upload = self.compute_upload_factor(
            nacelle_angle_deg, wing_CL * wing_area * airspeed_mps, 2.0
        )

        # Estimate drags (simplified)
        CD_wing = 0.02 + 0.05 * wing_CL**2  # Profile + induced
        wing_drag = q * wing_area * CD_wing if airspeed_mps > 5.0 else 0.0

        CD_fuse = 0.025
        fuselage_area = 3.0  # m² frontal
        fuselage_drag = q * fuselage_area * CD_fuse if airspeed_mps > 5.0 else 0.0

        # Rotor H-force (drag)
        # In hover this is near zero, in forward flight it's significant
        theta = np.radians(nacelle_angle_deg)
        rotor_drag = 0.03 * rotor_thrust * np.cos(theta) * (airspeed_mps / 50.0)

        # Blend forces
        self.blend_forces(
            nacelle_angle_deg, rotor_thrust, rotor_drag,
            wing_lift, wing_drag, fuselage_drag,
            download, upload
        )

        # Update state
        self.state.nacelle_angle_deg = nacelle_angle_deg
        self.state.nacelle_rate_dps = nacelle_rate_dps
        self.state.within_corridor = within_corridor
        self.state.corridor_margin = margin

        return self.state

    def get_recommended_nacelle_angle(
        self,
        airspeed_kts: float,
        altitude_m: float = 0.0,
    ) -> float:
        """
        Get recommended nacelle angle for given airspeed.

        Returns the nacelle angle that places the aircraft in the
        center of the conversion corridor for the given speed.

        Args:
            airspeed_kts: Target airspeed
            altitude_m: Current altitude

        Returns:
            Recommended nacelle angle (degrees)
        """
        # Find nacelle angle where airspeed is centered in corridor
        best_angle = 90.0
        best_margin = -1.0

        for angle in np.linspace(0, 90, 91):
            v_min, v_max = self.get_corridor_limits(angle, altitude_m)
            if v_min <= airspeed_kts <= v_max:
                # Compute margin (how centered in corridor)
                center = (v_min + v_max) / 2
                margin = 1.0 - abs(airspeed_kts - center) / ((v_max - v_min) / 2)
                if margin > best_margin:
                    best_margin = margin
                    best_angle = angle

        return best_angle

    def get_nacelle_rate_command(
        self,
        current_angle_deg: float,
        target_angle_deg: float,
        airspeed_kts: float,
        altitude_m: float = 0.0,
    ) -> float:
        """
        Get safe nacelle rate command.

        Ensures nacelle tilts at a rate that keeps aircraft within
        the conversion corridor.

        Args:
            current_angle_deg: Current nacelle angle
            target_angle_deg: Desired nacelle angle
            airspeed_kts: Current airspeed
            altitude_m: Current altitude

        Returns:
            Safe nacelle rate (deg/s, positive = toward hover)
        """
        # Basic rate toward target
        error = target_angle_deg - current_angle_deg

        # Limit rate
        max_rate = self.corridor.nacelle_rate_max_dps

        # Check if target would put us outside corridor
        within, margin, _ = self.check_corridor(target_angle_deg, airspeed_kts, altitude_m)

        if not within:
            # Reduce rate proportionally to margin
            max_rate *= max(0.0, margin + 1.0) / 2.0

        # Apply rate limit
        rate = np.clip(error, -max_rate, max_rate)

        return rate

    def get_state(self) -> TransitionState:
        """Get current transition state."""
        return self.state

    def to_dict(self) -> dict:
        """Convert state to dictionary."""
        return {
            'nacelle_angle_deg': self.state.nacelle_angle_deg,
            'nacelle_rate_dps': self.state.nacelle_rate_dps,
            'lift_rotor': self.state.lift_rotor,
            'lift_wing': self.state.lift_wing,
            'lift_total': self.state.lift_total,
            'drag_total': self.state.drag_total,
            'download_factor': self.state.download_factor,
            'upload_factor': self.state.upload_factor,
            'within_corridor': self.state.within_corridor,
            'corridor_margin': self.state.corridor_margin,
        }
