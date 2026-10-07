import math
import unittest

from api.physics3d import FlightParameters, integrate_trajectory, launch_state
from api.trajectory_simulator import (
    EnvironmentConditions,
    GamePieceProperties,
    LaunchParameters,
    PhysicsEngine,
    Target,
    TrajectorySimulator,
)


class LegacyAdapterTests(unittest.TestCase):
    def setUp(self):
        self.piece = GamePieceProperties(
            name="test",
            mass=0.215,
            radius=0.075,
            drag_coefficient=0.0,
            lift_coefficient=0.0,
            moment_of_inertia=0.000484,
        )
        self.env = EnvironmentConditions()
        self.physics = PhysicsEngine(self.piece, self.env)
        self.simulator = TrajectorySimulator(self.physics, dt=0.002, max_time=3.0)
        self.launch = LaunchParameters((0.0, 1.0), 10.0, 30.0, 0.0)
        self.flight_params = FlightParameters(
            mass=self.piece.mass,
            radius=self.piece.radius,
            drag_coefficient=0.0,
            lift_coefficient=0.0,
            air_density=self.env.air_density,
            gravity=self.env.gravity,
        )

    def test_legacy_rk4_matches_3d_xz_projection(self):
        legacy = self.simulator.simulate(self.launch, method="rk4")
        vx, vz = self.launch.velocity_vector
        core = integrate_trajectory(
            launch_state((0, 0, 1), (vx, 0, vz), (0, 0, 0)),
            self.flight_params,
            method="rk4",
            dt=self.simulator.dt,
            max_time=self.simulator.max_time,
        )
        self.assertAlmostEqual(legacy.points[-1].x, core[-1].state[0], places=8)
        self.assertAlmostEqual(legacy.points[-1].y, core[-1].state[2], places=8)

    def test_legacy_adaptive_alias_uses_rk45(self):
        adaptive = self.simulator.simulate(self.launch, method="adaptive")
        rk45 = self.simulator.simulate(self.launch, method="rk45")
        self.assertAlmostEqual(adaptive.flight_time, rk45.flight_time, places=10)
        self.assertAlmostEqual(adaptive.points[-1].x, rk45.points[-1].x, places=8)

    def test_legacy_euler_is_rejected(self):
        with self.assertRaises(ValueError):
            self.simulator.simulate(self.launch, method="euler")

    def test_positive_legacy_backspin_produces_upward_magnus_effect(self):
        self.piece.lift_coefficient = 0.25
        spun = self.simulator.simulate(LaunchParameters((0, 1), 10, 20, 200))
        unspun = self.simulator.simulate(LaunchParameters((0, 1), 10, 20, 0))
        self.assertGreater(spun.max_height, unspun.max_height)

    def test_environment_gravity_is_forwarded(self):
        self.physics.env.gravity = 3.0
        result = self.simulator.simulate(LaunchParameters((0, 1), 2, 0, 0))
        self.assertGreater(result.flight_time, math.sqrt(2 / 9.81))

    def _target_crossings(self):
        disc = 5.0**2 - 4 * 4.905 * 0.5
        t_up = (5.0 - math.sqrt(disc)) / 9.81
        t_down = (5.0 + math.sqrt(disc)) / 9.81
        return 2.0 * t_up, 2.0 * t_down

    def _crossing_launch(self):
        return LaunchParameters(
            (0, 0.5), math.hypot(2, 5), math.degrees(math.atan2(5, 2)), 0
        )

    def test_target_uses_descending_crossing_not_ascending_crossing(self):
        x_up, _ = self._target_crossings()
        target = Target("test", (x_up, 1.0), 0.05, 0.05, 1.0)
        result = self.simulator.simulate(self._crossing_launch(), target)
        self.assertFalse(result.hit_target)

    def test_target_hit_reduces_to_horizontal_clearance_at_y_zero(self):
        _, x_down = self._target_crossings()
        target = Target("test", (x_down, 1.0), 0.05, 0.05, 1.0)
        result = self.simulator.simulate(self._crossing_launch(), target)
        self.assertTrue(result.hit_target)


if __name__ == "__main__":
    unittest.main()
