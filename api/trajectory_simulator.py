#!/usr/bin/env python3
"""
FRC 2026 Projectile Trajectory Simulator
=========================================
A comprehensive physics simulator for calculating optimal shooting trajectories
with realistic effects including air resistance and Magnus effect (backspin).

Author: FRC Trajectory Tools
License: MIT
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, List, Optional, Callable
from enum import Enum
import warnings


class GamePiece(Enum):
    """Common FRC game piece types with their physical properties."""
    # 2026 game piece
    FUEL = "fuel"
    # Previous years for reference
    NOTE_2024 = "note_2024"  # 2024 Crescendo
    CARGO_2022 = "cargo_2022"  # 2022 Rapid React
    POWER_CELL_2020 = "power_cell_2020"  # 2020 Infinite Recharge
    CUSTOM = "custom"


@dataclass
class GamePieceProperties:
    """Physical properties of a game piece."""
    name: str
    mass: float  # kg
    radius: float  # meters
    drag_coefficient: float  # dimensionless
    lift_coefficient: float  # for Magnus effect
    moment_of_inertia: float  # kg·m² (for spin dynamics)

    @classmethod
    def from_game_piece(cls, piece: GamePiece) -> 'GamePieceProperties':
        """Get properties for standard FRC game pieces."""
        properties = {
            GamePiece.FUEL: cls(
                name="Coral (2026)",
                mass=0.227,  # ~0.5 lb
                radius=0.15,  # 15 cm radius
                drag_coefficient=0.47,  # sphere-like
                lift_coefficient=0.25,
                moment_of_inertia=0.00097  # solid sphere approximation
            ),
            GamePiece.NOTE_2024: cls(
                name="Note (2024)",
                mass=0.235,  # 235g
                radius=0.178,  # 14 inch diameter / 2
                drag_coefficient=0.5,  # ring shape
                lift_coefficient=0.2,
                moment_of_inertia=0.0037
            ),
            GamePiece.CARGO_2022: cls(
                name="Cargo (2022)",
                mass=0.27,  # 270g
                radius=0.12,  # 9.5 inch diameter / 2
                drag_coefficient=0.47,
                lift_coefficient=0.25,
                moment_of_inertia=0.0019
            ),
            GamePiece.POWER_CELL_2020: cls(
                name="Power Cell (2020)",
                mass=0.141,  # 5oz
                radius=0.0889,  # 7 inch diameter / 2
                drag_coefficient=0.47,
                lift_coefficient=0.25,
                moment_of_inertia=0.00056
            ),
        }
        return properties.get(piece, properties[GamePiece.FUEL])


@dataclass
class EnvironmentConditions:
    """Environmental conditions affecting trajectory."""
    air_density: float = 1.225  # kg/m³ at sea level, 15°C
    gravity: float = 9.81  # m/s²
    temperature_celsius: float = 20.0
    altitude_meters: float = 0.0

    def __post_init__(self):
        """Adjust air density based on altitude and temperature."""
        # Barometric formula approximation
        scale_height = 8500  # meters
        temp_kelvin = self.temperature_celsius + 273.15
        standard_temp = 288.15  # K (15°C)

        # Adjust for altitude
        pressure_ratio = np.exp(-self.altitude_meters / scale_height)
        # Adjust for temperature
        temp_ratio = standard_temp / temp_kelvin

        self.air_density = 1.225 * pressure_ratio * temp_ratio


@dataclass
class LaunchParameters:
    """Parameters for the projectile launch."""
    position: Tuple[float, float]  # (x, y) in meters
    velocity: float  # m/s
    angle: float  # degrees from horizontal
    spin_rate: float = 0.0  # rad/s (positive = backspin)

    @property
    def velocity_vector(self) -> Tuple[float, float]:
        """Get initial velocity as (vx, vy) vector."""
        angle_rad = np.radians(self.angle)
        return (
            self.velocity * np.cos(angle_rad),
            self.velocity * np.sin(angle_rad)
        )


@dataclass
class Target:
    """Target definition for the shooter."""
    name: str
    position: Tuple[float, float]  # (x, y) center in meters
    entry_radius: float  # meters (opening size)
    funnel_radius: float  # meters (outer funnel size)
    height_at_funnel: float  # height at funnel edge

    def get_entry_bounds(self) -> Tuple[float, float]:
        """Get the left and right x-coordinates of the entry opening."""
        return (
            self.position[0] - self.entry_radius,
            self.position[0] + self.entry_radius
        )

    def is_valid_entry(self, x: float, y: float, tolerance: float = 0.05) -> bool:
        """Check if a point is within valid entry zone."""
        entry_left, entry_right = self.get_entry_bounds()
        # Check if x is within entry bounds and y is close to entry height
        if entry_left <= x <= entry_right:
            return abs(y - self.position[1]) <= tolerance
        return False


@dataclass
class TrajectoryPoint:
    """A single point along the trajectory."""
    time: float
    x: float
    y: float
    vx: float
    vy: float
    spin: float  # current spin rate

    @property
    def speed(self) -> float:
        return np.sqrt(self.vx ** 2 + self.vy ** 2)

    @property
    def position(self) -> Tuple[float, float]:
        return (self.x, self.y)

    @property
    def velocity(self) -> Tuple[float, float]:
        return (self.vx, self.vy)


@dataclass
class TrajectoryResult:
    """Complete result of a trajectory simulation."""
    points: List[TrajectoryPoint]
    launch_params: LaunchParameters
    target: Optional[Target]
    hit_target: bool = False
    impact_point: Optional[Tuple[float, float]] = None
    flight_time: float = 0.0
    max_height: float = 0.0
    range_distance: float = 0.0
    entry_velocity: Optional[float] = None
    entry_angle: Optional[float] = None

    def get_arrays(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get trajectory as numpy arrays (t, x, y)."""
        t = np.array([p.time for p in self.points])
        x = np.array([p.x for p in self.points])
        y = np.array([p.y for p in self.points])
        return t, x, y


class PhysicsEngine:
    """
    Comprehensive physics engine for projectile simulation.

    Includes:
    - Gravitational acceleration
    - Quadratic air drag
    - Magnus effect (from spin)
    - Spin decay due to air resistance
    """

    def __init__(
            self,
            game_piece: GamePieceProperties,
            environment: EnvironmentConditions
    ):
        self.piece = game_piece
        self.env = environment

        # Precompute constants
        self.cross_section = np.pi * game_piece.radius ** 2
        self.drag_factor = 0.5 * environment.air_density * self.cross_section * game_piece.drag_coefficient
        self.magnus_factor = 0.5 * environment.air_density * self.cross_section * game_piece.lift_coefficient

    def compute_drag_force(self, vx: float, vy: float) -> Tuple[float, float]:
        """
        Compute air drag force using quadratic drag model.
        F_drag = -0.5 * ρ * A * Cd * v² * v_hat
        """
        speed = np.sqrt(vx ** 2 + vy ** 2)
        if speed < 1e-10:
            return (0.0, 0.0)

        drag_magnitude = self.drag_factor * speed ** 2

        # Force opposes velocity
        fx = -drag_magnitude * (vx / speed)
        fy = -drag_magnitude * (vy / speed)

        return (fx, fy)

    def compute_magnus_force(self, vx: float, vy: float, spin: float) -> Tuple[float, float]:
        """
        Compute Magnus force due to spin.

        For backspin (positive spin), the Magnus force lifts the ball.
        F_magnus = 0.5 * ρ * A * Cl * v² * (ω × v_hat)

        In 2D with spin around z-axis:
        - Positive spin (backspin) creates upward lift when moving forward
        """
        speed = np.sqrt(vx ** 2 + vy ** 2)
        if speed < 1e-10 or abs(spin) < 1e-10:
            return (0.0, 0.0)

        # Magnus coefficient depends on spin parameter S = ω*r/v
        spin_param = abs(spin) * self.piece.radius / speed

        # Empirical Magnus coefficient (Kutta-Joukowski approximation)
        # Cl typically varies with spin parameter
        effective_cl = self.piece.lift_coefficient * min(spin_param, 0.5) * 2

        magnus_magnitude = self.magnus_factor * speed ** 2 * effective_cl

        # Direction: perpendicular to velocity, determined by spin direction
        # For positive spin (backspin), force is perpendicular and "up" relative to velocity
        # Cross product of spin axis (0,0,ω) with velocity (vx,vy,0) gives (-ω*vy, ω*vx, 0)
        sign = np.sign(spin)
        fx = -sign * magnus_magnitude * (vy / speed)
        fy = sign * magnus_magnitude * (vx / speed)

        return (fx, fy)

    def compute_spin_decay(self, spin: float, vx: float, vy: float, dt: float) -> float:
        """
        Compute spin decay due to air resistance on rotating body.
        Uses exponential decay model with velocity-dependent time constant.
        """
        speed = np.sqrt(vx ** 2 + vy ** 2)

        # Spin decay time constant (empirical, ~2-5 seconds typically)
        # Faster movement = more air interaction = faster decay
        tau = 3.0 / (1.0 + 0.1 * speed)

        return spin * np.exp(-dt / tau)

    def compute_acceleration(
            self,
            x: float, y: float,
            vx: float, vy: float,
            spin: float
    ) -> Tuple[float, float]:
        """Compute total acceleration on the projectile."""
        # Gravity (always down)
        ax = 0.0
        ay = -self.env.gravity

        # Air drag
        fx_drag, fy_drag = self.compute_drag_force(vx, vy)
        ax += fx_drag / self.piece.mass
        ay += fy_drag / self.piece.mass

        # Magnus effect
        fx_magnus, fy_magnus = self.compute_magnus_force(vx, vy, spin)
        ax += fx_magnus / self.piece.mass
        ay += fy_magnus / self.piece.mass

        return (ax, ay)


class TrajectorySimulator:
    """
    Main trajectory simulator with various solving methods.
    """

    def __init__(
            self,
            physics: PhysicsEngine,
            dt: float = 0.001,  # Time step in seconds
            max_time: float = 5.0  # Maximum simulation time
    ):
        self.physics = physics
        self.dt = dt
        self.max_time = max_time

    def simulate(
            self,
            launch: LaunchParameters,
            target: Optional[Target] = None,
            method: str = "rk4"
    ) -> TrajectoryResult:
        """
        Simulate the trajectory from launch to ground impact or target.

        Args:
            launch: Launch parameters (position, velocity, angle, spin)
            target: Optional target to check for hits
            method: Integration method ("euler", "rk4", "adaptive")

        Returns:
            TrajectoryResult with full trajectory data
        """
        # Initial conditions
        x, y = launch.position
        vx, vy = launch.velocity_vector
        spin = launch.spin_rate
        t = 0.0

        points = [TrajectoryPoint(t, x, y, vx, vy, spin)]
        max_height = y

        # Integration loop
        while t < self.max_time and y >= 0:
            if method == "euler":
                x, y, vx, vy, spin = self._euler_step(x, y, vx, vy, spin)
            elif method == "rk4":
                x, y, vx, vy, spin = self._rk4_step(x, y, vx, vy, spin)
            else:
                x, y, vx, vy, spin = self._rk4_step(x, y, vx, vy, spin)

            t += self.dt
            points.append(TrajectoryPoint(t, x, y, vx, vy, spin))
            max_height = max(max_height, y)

            # Check if we've passed the target
            if target and y <= target.position[1] and len(points) > 2:
                prev = points[-2]
                if prev.y > target.position[1]:
                    # Interpolate to find exact crossing point
                    alpha = (target.position[1] - prev.y) / (y - prev.y)
                    cross_x = prev.x + alpha * (x - prev.x)
                    cross_vx = prev.vx + alpha * (vx - prev.vx)
                    cross_vy = prev.vy + alpha * (vy - prev.vy)

                    entry_left, entry_right = target.get_entry_bounds()
                    if entry_left <= cross_x <= entry_right:
                        # Hit the target!
                        result = TrajectoryResult(
                            points=points,
                            launch_params=launch,
                            target=target,
                            hit_target=True,
                            impact_point=(cross_x, target.position[1]),
                            flight_time=t,
                            max_height=max_height,
                            range_distance=cross_x - launch.position[0],
                            entry_velocity=np.sqrt(cross_vx ** 2 + cross_vy ** 2),
                            entry_angle=np.degrees(np.arctan2(cross_vy, cross_vx))
                        )
                        return result

        # Didn't hit target or simulation ended
        result = TrajectoryResult(
            points=points,
            launch_params=launch,
            target=target,
            hit_target=False,
            impact_point=(x, max(0, y)),
            flight_time=t,
            max_height=max_height,
            range_distance=x - launch.position[0]
        )
        return result

    def _euler_step(
            self,
            x: float, y: float,
            vx: float, vy: float,
            spin: float
    ) -> Tuple[float, float, float, float, float]:
        """Simple Euler integration step."""
        ax, ay = self.physics.compute_acceleration(x, y, vx, vy, spin)

        x_new = x + vx * self.dt
        y_new = y + vy * self.dt
        vx_new = vx + ax * self.dt
        vy_new = vy + ay * self.dt
        spin_new = self.physics.compute_spin_decay(spin, vx, vy, self.dt)

        return x_new, y_new, vx_new, vy_new, spin_new

    def _rk4_step(
            self,
            x: float, y: float,
            vx: float, vy: float,
            spin: float
    ) -> Tuple[float, float, float, float, float]:
        """Fourth-order Runge-Kutta integration step."""
        dt = self.dt

        # k1
        ax1, ay1 = self.physics.compute_acceleration(x, y, vx, vy, spin)

        # k2
        x2 = x + 0.5 * dt * vx
        y2 = y + 0.5 * dt * vy
        vx2 = vx + 0.5 * dt * ax1
        vy2 = vy + 0.5 * dt * ay1
        ax2, ay2 = self.physics.compute_acceleration(x2, y2, vx2, vy2, spin)

        # k3
        x3 = x + 0.5 * dt * vx2
        y3 = y + 0.5 * dt * vy2
        vx3 = vx + 0.5 * dt * ax2
        vy3 = vy + 0.5 * dt * ay2
        ax3, ay3 = self.physics.compute_acceleration(x3, y3, vx3, vy3, spin)

        # k4
        x4 = x + dt * vx3
        y4 = y + dt * vy3
        vx4 = vx + dt * ax3
        vy4 = vy + dt * ay3
        ax4, ay4 = self.physics.compute_acceleration(x4, y4, vx4, vy4, spin)

        # Combine
        x_new = x + (dt / 6.0) * (vx + 2 * vx2 + 2 * vx3 + vx4)
        y_new = y + (dt / 6.0) * (vy + 2 * vy2 + 2 * vy3 + vy4)
        vx_new = vx + (dt / 6.0) * (ax1 + 2 * ax2 + 2 * ax3 + ax4)
        vy_new = vy + (dt / 6.0) * (ay1 + 2 * ay2 + 2 * ay3 + ay4)
        spin_new = self.physics.compute_spin_decay(spin, vx, vy, dt)

        return x_new, y_new, vx_new, vy_new, spin_new


@dataclass
class ShooterConfig:
    """Configuration for a flywheel shooter to estimate backspin."""
    flywheel_diameter: float  # inches
    flywheel_rpm: float  # RPM
    ball_diameter: float  # inches
    hood_type: str = "static"  # "static", "adjustable", "dual_wheel"
    compression: float = 0.5  # inches of compression
    hood_material: str = "foam"  # "foam", "polycarbonate", "rubber"
    num_flywheels: int = 1  # 1 for single, 2 for dual

    def estimate_backspin_rpm(self) -> float:
        """
        Estimate the backspin RPM imparted to the ball.

        For a hooded single flywheel shooter:
        - The flywheel accelerates one side of the ball
        - The static hood provides friction on the other side
        - This creates differential velocity = backspin

        Key factors:
        - Higher compression = more spin transfer
        - Foam hood = more grip = more spin
        - Larger flywheel relative to ball = more spin

        Returns:
            Estimated backspin in RPM
        """
        # Flywheel surface speed (inches/sec)
        flywheel_radius = self.flywheel_diameter / 2
        flywheel_surface_speed = self.flywheel_rpm * 2 * np.pi * flywheel_radius / 60

        ball_radius = self.ball_diameter / 2

        # Spin transfer efficiency depends on several factors
        # Base efficiency for a static hooded shooter
        if self.hood_type == "static":
            base_efficiency = 0.35
        elif self.hood_type == "adjustable":
            base_efficiency = 0.30
        elif self.hood_type == "dual_wheel":
            # Dual opposing wheels can create more spin
            base_efficiency = 0.50
        else:
            base_efficiency = 0.30

        # Compression factor: more compression = more spin transfer
        # Normalized to typical 0.5" compression
        compression_factor = min(1.5, 0.7 + 0.6 * (self.compression / 0.5))

        # Material factor
        material_factors = {
            "foam": 1.2,  # High grip
            "rubber": 1.1,
            "polycarbonate": 0.8,  # Lower grip, more slip
            "delrin": 0.7,
        }
        material_factor = material_factors.get(self.hood_material, 1.0)

        # Combined efficiency
        total_efficiency = base_efficiency * compression_factor * material_factor
        total_efficiency = min(0.7, total_efficiency)  # Cap at realistic max

        # Calculate backspin
        # ω_ball = (v_flywheel - v_hood) / r_ball * efficiency
        # For static hood, v_hood = 0
        # But ball center also moves, so effective differential is reduced

        # Simplified model: backspin ≈ flywheel_rpm * (r_flywheel/r_ball) * efficiency
        backspin_rpm = self.flywheel_rpm * (flywheel_radius / ball_radius) * total_efficiency

        return backspin_rpm

    def estimate_exit_velocity(self) -> float:
        """
        Estimate ball exit velocity in m/s.

        Exit velocity is typically 50-70% of flywheel surface speed
        due to slip and energy transfer inefficiency.
        """
        flywheel_radius_m = (self.flywheel_diameter / 2) * 0.0254  # to meters
        flywheel_surface_speed = self.flywheel_rpm * 2 * np.pi * flywheel_radius_m / 60

        # Exit velocity efficiency (typically 0.5-0.7)
        if self.num_flywheels == 2:
            efficiency = 0.65  # Dual flywheels are more efficient
        else:
            efficiency = 0.55  # Single flywheel with hood

        # Compression helps energy transfer
        compression_boost = min(1.1, 1.0 + 0.1 * (self.compression / 0.5))

        return flywheel_surface_speed * efficiency * compression_boost

    @classmethod
    def from_254_2017(cls) -> 'ShooterConfig':
        """
        Create config based on Team 254's 2017 Steamworks shooter.

        From technical binder:
        - Four 4" Fairlane wheels
        - 3500 RPM wheel speed
        - Fixed hood at 14°
        - 0.5" foam compression
        - Twin backspin flywheels (dual horizontal)
        """
        return cls(
            flywheel_diameter=4.0,
            flywheel_rpm=3500,
            ball_diameter=5.0,  # Fuel ball was ~5" diameter
            hood_type="static",
            compression=0.5,
            hood_material="foam",
            num_flywheels=2  # Twin flywheels
        )

    @classmethod
    def create_single_flywheel(
            cls,
            flywheel_diameter: float,
            flywheel_rpm: float,
            ball_diameter: float,
            compression: float = 0.5
    ) -> 'ShooterConfig':
        """Create a typical single flywheel hooded shooter config."""
        return cls(
            flywheel_diameter=flywheel_diameter,
            flywheel_rpm=flywheel_rpm,
            ball_diameter=ball_diameter,
            hood_type="static",
            compression=compression,
            hood_material="foam",
            num_flywheels=1
        )


class TrajectoryOptimizer:
    """
    Optimizer to find optimal launch parameters to hit a target.
    """

    def __init__(self, simulator: TrajectorySimulator):
        self.simulator = simulator

    def find_optimal_angle(
            self,
            launch_x: float,
            launch_y: float,
            velocity: float,
            spin: float,
            target: Target,
            prefer_high: bool = True,
            tolerance: float = 0.01,
            max_iterations: int = 100
    ) -> Optional[float]:
        """
        Find the optimal launch angle to hit the target.

        Uses binary search / Newton's method to find the angle.

        Args:
            launch_x, launch_y: Launch position
            velocity: Launch velocity magnitude
            spin: Spin rate
            target: Target to hit
            prefer_high: If True, prefer high arc trajectory
            tolerance: Acceptable error in target hit (meters)
            max_iterations: Maximum optimization iterations

        Returns:
            Optimal angle in degrees, or None if no solution found
        """
        # First, do a coarse sweep to find approximate solution
        angles = np.linspace(5, 85, 81)  # 1 degree increments
        best_angle = None
        best_error = float('inf')

        for angle in angles:
            launch = LaunchParameters(
                position=(launch_x, launch_y),
                velocity=velocity,
                angle=angle,
                spin_rate=spin
            )
            result = self.simulator.simulate(launch, target)

            if result.impact_point:
                # Error is distance from target center at target height
                error_x = abs(result.impact_point[0] - target.position[0])

                # Check if within entry bounds
                entry_left, entry_right = target.get_entry_bounds()
                if entry_left <= result.impact_point[0] <= entry_right:
                    if error_x < best_error:
                        best_error = error_x
                        best_angle = angle

        if best_angle is None:
            return None

        # Refine with binary search
        angle_low = max(5, best_angle - 5)
        angle_high = min(85, best_angle + 5)

        for _ in range(max_iterations):
            if angle_high - angle_low < 0.01:
                break

            angle_mid = (angle_low + angle_high) / 2

            launch = LaunchParameters(
                position=(launch_x, launch_y),
                velocity=velocity,
                angle=angle_mid,
                spin_rate=spin
            )
            result = self.simulator.simulate(launch, target)

            if result.impact_point:
                if result.impact_point[0] < target.position[0]:
                    angle_low = angle_mid
                else:
                    angle_high = angle_mid

        return (angle_low + angle_high) / 2

    def find_minimum_velocity(
            self,
            launch_x: float,
            launch_y: float,
            spin: float,
            target: Target,
            velocity_range: Tuple[float, float] = (5.0, 30.0),
            tolerance: float = 0.1
    ) -> Optional[Tuple[float, float]]:
        """
        Find the minimum velocity and corresponding angle to hit the target.

        Returns:
            Tuple of (velocity, angle) or None if no solution
        """
        v_low, v_high = velocity_range

        while v_high - v_low > tolerance:
            v_mid = (v_low + v_high) / 2

            angle = self.find_optimal_angle(
                launch_x, launch_y, v_mid, spin, target
            )

            if angle is not None:
                v_high = v_mid
            else:
                v_low = v_mid

        # Get the final angle for the found velocity
        final_velocity = v_high
        final_angle = self.find_optimal_angle(
            launch_x, launch_y, final_velocity, spin, target
        )

        if final_angle is not None:
            return (final_velocity, final_angle)
        return None

    def find_optimal_velocity(
            self,
            launch_x: float,
            launch_y: float,
            angle: float,
            spin: float,
            target: Target,
            velocity_range: Tuple[float, float] = (5.0, 30.0),
            tolerance: float = 0.05
    ) -> Optional[float]:
        """
        Find the optimal velocity to hit the target at a given angle.

        Uses binary search to find the velocity that lands the projectile
        closest to the target center.

        Args:
            launch_x, launch_y: Launch position
            angle: Fixed launch angle in degrees
            spin: Spin rate in rad/s
            target: Target to hit
            velocity_range: (min, max) velocity to search
            tolerance: Acceptable error in velocity (m/s)

        Returns:
            Optimal velocity in m/s, or None if no solution found
        """
        v_low, v_high = velocity_range
        best_velocity = None
        best_error = float('inf')

        # First, coarse sweep to find approximate range
        velocities = np.linspace(v_low, v_high, 50)

        for v in velocities:
            launch = LaunchParameters(
                position=(launch_x, launch_y),
                velocity=v,
                angle=angle,
                spin_rate=spin
            )
            result = self.simulator.simulate(launch, target)

            if result.impact_point:
                # Calculate error from target
                dx = result.impact_point[0] - target.position[0]
                dy = result.impact_point[1] - target.position[1]
                error = np.sqrt(dx ** 2 + dy ** 2)

                # Check if within entry bounds
                entry_left, entry_right = target.get_entry_bounds()
                if entry_left <= result.impact_point[0] <= entry_right:
                    if result.hit_target and error < best_error:
                        best_error = error
                        best_velocity = v

        if best_velocity is None:
            return None

        # Refine with binary search
        v_low = max(velocity_range[0], best_velocity - 2)
        v_high = min(velocity_range[1], best_velocity + 2)

        for _ in range(50):  # Max iterations
            if v_high - v_low < tolerance:
                break

            v_mid = (v_low + v_high) / 2

            launch = LaunchParameters(
                position=(launch_x, launch_y),
                velocity=v_mid,
                angle=angle,
                spin_rate=spin
            )
            result = self.simulator.simulate(launch, target)

            if result.impact_point:
                if result.impact_point[0] < target.position[0]:
                    # Undershooting - need more velocity
                    v_low = v_mid
                else:
                    # Overshooting - need less velocity
                    v_high = v_mid
            else:
                # No valid trajectory, try higher velocity
                v_low = v_mid

        return (v_low + v_high) / 2

    def find_optimal_parameters(
            self,
            launch_x: float,
            launch_y: float,
            spin: float,
            target: Target,
            velocity_range: Tuple[float, float] = (5.0, 25.0),
            angle_range: Tuple[float, float] = (20.0, 75.0),
            optimize_for: str = "accuracy"
    ) -> Optional[Tuple[float, float, TrajectoryResult]]:
        """
        Find optimal velocity AND angle combination to hit target.

        Args:
            launch_x, launch_y: Launch position
            spin: Spin rate in rad/s
            target: Target to hit
            velocity_range: (min, max) velocity to search
            angle_range: (min, max) angle to search
            optimize_for: "accuracy", "min_velocity", "flattest", or "entry_angle"

        Returns:
            Tuple of (velocity, angle, result) or None
        """
        best_params = None
        best_score = float('inf') if optimize_for != "flattest" else float('-inf')
        best_result = None

        # Grid search
        velocities = np.linspace(velocity_range[0], velocity_range[1], 30)
        angles = np.linspace(angle_range[0], angle_range[1], 60)

        for v in velocities:
            for a in angles:
                launch = LaunchParameters(
                    position=(launch_x, launch_y),
                    velocity=v,
                    angle=a,
                    spin_rate=spin
                )
                result = self.simulator.simulate(launch, target)

                if result.hit_target:
                    # Calculate score based on optimization goal
                    if optimize_for == "accuracy":
                        # Minimize distance from center
                        score = abs(result.impact_point[0] - target.position[0])
                    elif optimize_for == "min_velocity":
                        # Minimize velocity (with accuracy constraint)
                        score = v
                    elif optimize_for == "flattest":
                        # Maximize entry angle (closer to horizontal = flatter)
                        # Entry angle is negative when coming down
                        score = result.entry_angle if result.entry_angle else -90
                    elif optimize_for == "entry_angle":
                        # Optimize for steeper entry (better for funnels)
                        score = abs(result.entry_angle) if result.entry_angle else 0
                    else:
                        score = v  # Default to min velocity

                    is_better = (score < best_score) if optimize_for != "flattest" else (score > best_score)

                    if is_better:
                        best_score = score
                        best_params = (v, a)
                        best_result = result

        if best_params:
            return (best_params[0], best_params[1], best_result)
        return None

    def compute_error_envelope(
            self,
            launch: LaunchParameters,
            target: Target,
            velocity_error: float = 0.5,  # m/s
            angle_error: float = 1.0,  # degrees
            spin_error: float = 5.0,  # rad/s
            n_samples: int = 50
    ) -> List[TrajectoryResult]:
        """
        Compute trajectories for error bounds to visualize shot consistency.

        Returns list of trajectory results for various error combinations.
        """
        results = []

        # Generate error combinations
        v_errors = np.linspace(-velocity_error, velocity_error, 5)
        a_errors = np.linspace(-angle_error, angle_error, 5)

        for dv in v_errors:
            for da in a_errors:
                error_launch = LaunchParameters(
                    position=launch.position,
                    velocity=launch.velocity + dv,
                    angle=launch.angle + da,
                    spin_rate=launch.spin_rate
                )
                result = self.simulator.simulate(error_launch, target)
                results.append(result)

        return results


# Utility functions for common calculations

def compute_no_drag_angle(
        launch_x: float, launch_y: float,
        target_x: float, target_y: float,
        velocity: float,
        gravity: float = 9.81,
        prefer_high: bool = True
) -> Optional[float]:
    """
    Compute launch angle analytically (no air resistance).

    This is the classic projectile motion formula:
    θ = arctan((v² ± √(v⁴ - g(gx² + 2yv²))) / (gx))

    Returns angle in degrees.
    """
    dx = target_x - launch_x
    dy = target_y - launch_y

    v2 = velocity ** 2
    v4 = velocity ** 4
    g = gravity

    discriminant = v4 - g * (g * dx ** 2 + 2 * dy * v2)

    if discriminant < 0:
        return None  # Target unreachable

    sqrt_disc = np.sqrt(discriminant)

    if prefer_high:
        angle_rad = np.arctan2(v2 + sqrt_disc, g * dx)
    else:
        angle_rad = np.arctan2(v2 - sqrt_disc, g * dx)

    return np.degrees(angle_rad)


def meters_to_inches(m: float) -> float:
    """Convert meters to inches."""
    return m * 39.3701


def inches_to_meters(inches: float) -> float:
    """Convert inches to meters."""
    return inches / 39.3701


def fps_to_mps(fps: float) -> float:
    """Convert feet per second to meters per second."""
    return fps * 0.3048


def mps_to_fps(mps: float) -> float:
    """Convert meters per second to feet per second."""
    return mps / 0.3048


def rpm_to_rads(rpm: float) -> float:
    """Convert RPM to radians per second."""
    return rpm * 2 * np.pi / 60


def rads_to_rpm(rads: float) -> float:
    """Convert radians per second to RPM."""
    return rads * 60 / (2 * np.pi)


if __name__ == "__main__":
    # Quick demo
    print("FRC Trajectory Simulator - Core Module")
    print("=" * 60)

    # =========================================
    # BACKSPIN ESTIMATION EXAMPLE
    # =========================================
    print("\n" + "=" * 60)
    print("BACKSPIN ESTIMATION FOR HOODED FLYWHEEL SHOOTERS")
    print("=" * 60)

    # Example 1: Team 254's 2017 Steamworks shooter
    print("\n--- Team 254 (2017) Steamworks Shooter ---")
    shooter_254 = ShooterConfig.from_254_2017()
    backspin_254 = shooter_254.estimate_backspin_rpm()
    exit_vel_254 = shooter_254.estimate_exit_velocity()
    print(f"  Flywheel: {shooter_254.flywheel_diameter}\" @ {shooter_254.flywheel_rpm} RPM")
    print(f"  Ball diameter: {shooter_254.ball_diameter}\"")
    print(f"  Hood: {shooter_254.hood_type} with {shooter_254.compression}\" {shooter_254.hood_material}")
    print(f"  Estimated backspin: {backspin_254:.0f} RPM")
    print(f"  Estimated exit velocity: {exit_vel_254:.1f} m/s ({mps_to_fps(exit_vel_254):.1f} ft/s)")

    # Example 2: Generic single flywheel shooter for 2026
    print("\n--- Generic Single Flywheel (2026 estimate) ---")
    # Assuming Coral is ~8" diameter
    shooter_2026 = ShooterConfig.create_single_flywheel(
        flywheel_diameter=4.0,
        flywheel_rpm=4000,
        ball_diameter=8.0,  # Estimated Coral size
        compression=0.75
    )
    backspin_2026 = shooter_2026.estimate_backspin_rpm()
    exit_vel_2026 = shooter_2026.estimate_exit_velocity()
    print(f"  Flywheel: {shooter_2026.flywheel_diameter}\" @ {shooter_2026.flywheel_rpm} RPM")
    print(f"  Ball diameter: {shooter_2026.ball_diameter}\"")
    print(f"  Compression: {shooter_2026.compression}\"")
    print(f"  Estimated backspin: {backspin_2026:.0f} RPM")
    print(f"  Estimated exit velocity: {exit_vel_2026:.1f} m/s ({mps_to_fps(exit_vel_2026):.1f} ft/s)")

    # Example 3: How flywheel RPM affects backspin
    print("\n--- Backspin vs Flywheel RPM (4\" wheel, 5\" ball) ---")
    for rpm in [2000, 3000, 4000, 5000]:
        shooter = ShooterConfig.create_single_flywheel(4.0, rpm, 5.0, 0.5)
        print(f"  {rpm} RPM flywheel → {shooter.estimate_backspin_rpm():.0f} RPM backspin")

    # =========================================
    # TRAJECTORY SIMULATION
    # =========================================
    print("\n" + "=" * 60)
    print("TRAJECTORY SIMULATION")
    print("=" * 60)

    # Create game piece and environment
    piece = GamePieceProperties.from_game_piece(GamePiece.FUEL)
    env = EnvironmentConditions(temperature_celsius=22, altitude_meters=100)

    print(f"\nGame Piece: {piece.name}")
    print(f"  Mass: {piece.mass:.3f} kg ({piece.mass * 1000:.0f}g)")
    print(f"  Radius: {meters_to_inches(piece.radius):.2f} in")
    print(f"  Drag Coefficient: {piece.drag_coefficient}")
    print(f"\nEnvironment:")
    print(f"  Air Density: {env.air_density:.4f} kg/m³")
    print(f"  Gravity: {env.gravity:.2f} m/s²")

    # Create physics engine and simulator
    physics = PhysicsEngine(piece, env)
    simulator = TrajectorySimulator(physics, dt=0.001)
    optimizer = TrajectoryOptimizer(simulator)

    # Define target
    target = Target(
        name="Hub (2026 REBUILT)",
        position=(0, 1.828),  # 72 inches height
        entry_radius=0.302,  # 11.9 inch radius (inner hole)
        funnel_radius=0.529,  # 20.85 inch radius (outer funnel)
        height_at_funnel=1.828  # Simplified for top-down entry
    )

    # Test launch with estimated backspin
    estimated_spin = rpm_to_rads(backspin_2026)
    launch = LaunchParameters(
        position=(-3.0, 0.5),  # 3m back, 0.5m height
        velocity=12.0,  # m/s
        angle=50.0,
        spin_rate=estimated_spin
    )

    print(f"\nLaunch Parameters:")
    print(f"  Position: ({launch.position[0]:.2f}, {launch.position[1]:.2f}) m")
    print(f"  Velocity: {launch.velocity:.1f} m/s ({mps_to_fps(launch.velocity):.1f} ft/s)")
    print(f"  Angle: {launch.angle:.1f}°")
    print(f"  Backspin: {rads_to_rpm(launch.spin_rate):.0f} RPM")

    # Simulate
    result = simulator.simulate(launch, target)

    print(f"\nSimulation Result:")
    print(f"  Hit Target: {result.hit_target}")
    print(f"  Flight Time: {result.flight_time:.3f} s")
    print(f"  Max Height: {result.max_height:.3f} m")
    print(f"  Range: {result.range_distance:.3f} m")
    if result.impact_point:
        print(f"  Impact Point: ({result.impact_point[0]:.3f}, {result.impact_point[1]:.3f}) m")

    # =========================================
    # OPTIMIZATION
    # =========================================
    print("\n" + "=" * 60)
    print("TRAJECTORY OPTIMIZATION")
    print("=" * 60)

    # Find optimal angle for given velocity
    print(f"\n--- Finding optimal angle for v={launch.velocity} m/s ---")
    optimal_angle = optimizer.find_optimal_angle(
        launch.position[0], launch.position[1],
        launch.velocity, launch.spin_rate, target
    )
    if optimal_angle:
        print(f"  Optimal angle: {optimal_angle:.1f}°")

    # Find optimal velocity for given angle
    print(f"\n--- Finding optimal velocity for angle={launch.angle}° ---")
    optimal_velocity = optimizer.find_optimal_velocity(
        launch.position[0], launch.position[1],
        launch.angle, launch.spin_rate, target
    )
    if optimal_velocity:
        print(f"  Optimal velocity: {optimal_velocity:.2f} m/s ({mps_to_fps(optimal_velocity):.1f} ft/s)")

    # Find minimum velocity to hit target
    print(f"\n--- Finding minimum velocity to hit target ---")
    min_result = optimizer.find_minimum_velocity(
        launch.position[0], launch.position[1],
        launch.spin_rate, target
    )
    if min_result:
        print(f"  Minimum velocity: {min_result[0]:.2f} m/s at {min_result[1]:.1f}°")

    # Find globally optimal parameters
    print(f"\n--- Finding optimal velocity AND angle ---")
    optimal = optimizer.find_optimal_parameters(
        launch.position[0], launch.position[1],
        launch.spin_rate, target,
        optimize_for="accuracy"
    )
    if optimal:
        v, a, res = optimal
        print(f"  Best accuracy: v={v:.2f} m/s, angle={a:.1f}°")
        print(f"    Entry velocity: {res.entry_velocity:.1f} m/s")
        print(f"    Entry angle: {res.entry_angle:.1f}°")

    # Compare with no-drag trajectory
    ideal_angle = compute_no_drag_angle(
        launch.position[0], launch.position[1],
        target.position[0], target.position[1],
        launch.velocity
    )
    if ideal_angle and optimal_angle:
        print(f"\n--- Physics Comparison ---")
        print(f"  Ideal angle (no drag): {ideal_angle:.1f}°")
        print(f"  Real angle (with drag/spin): {optimal_angle:.1f}°")
        print(f"  Difference: {optimal_angle - ideal_angle:+.1f}°")

    print("\n" + "=" * 60)
    print("✓ All tests completed!")
    print("=" * 60)
