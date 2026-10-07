import math
import unittest

from api.trajectory_simulator import (
    EnvironmentConditions,
    GamePiece,
    GamePieceProperties,
    PhysicsEngine,
)


class PhysicsModelTests(unittest.TestCase):
    def setUp(self):
        self.piece = GamePieceProperties.from_game_piece(GamePiece.FUEL)
        self.env = EnvironmentConditions(temperature_celsius=20.0, altitude_meters=0.0)
        self.physics = PhysicsEngine(self.piece, self.env)

    def test_fuel_uses_nominal_mass_not_maximum_spec_mass(self):
        # 2026 FUEL is specified at about 0.203-0.227 kg.
        # Without measured ball-by-ball data, use the midpoint as the nominal simulation mass.
        self.assertAlmostEqual(self.piece.mass, 0.215, places=3)

    def test_drag_force_matches_quadratic_drag_equation(self):
        vx, vy = 10.0, 0.0
        fx, fy = self.physics.compute_drag_force(vx, vy)

        area = math.pi * self.piece.radius ** 2
        expected = 0.5 * self.env.air_density * area * self.piece.drag_coefficient * vx ** 2
        self.assertAlmostEqual(fx, -expected, places=10)
        self.assertAlmostEqual(fy, 0.0, places=10)

    def test_magnus_force_uses_lift_coefficient_once(self):
        vx, vy = 10.0, 0.0
        spin_parameter = 0.25
        spin = spin_parameter * vx / self.piece.radius

        fx, fy = self.physics.compute_magnus_force(vx, vy, spin)

        area = math.pi * self.piece.radius ** 2
        effective_cl = self.piece.lift_coefficient * min(spin_parameter, 0.5) * 2.0
        expected_lift = 0.5 * self.env.air_density * area * effective_cl * vx ** 2

        self.assertAlmostEqual(fx, 0.0, places=10)
        self.assertAlmostEqual(fy, expected_lift, places=10)

    def test_uncalibrated_fuel_spin_is_constant_by_default(self):
        initial_spin = 200.0
        remaining_spin = self.physics.compute_spin_decay(initial_spin, 12.0, 0.0, 1.0)
        self.assertAlmostEqual(remaining_spin, initial_spin, places=12)

    def test_spin_decay_can_be_enabled_with_a_calibrated_time_constant(self):
        self.assertTrue(
            hasattr(self.piece, "spin_decay_time_constant"),
            "GamePieceProperties should expose an optional calibrated spin-decay time constant",
        )

        self.piece.spin_decay_time_constant = 2.0
        remaining_spin = self.physics.compute_spin_decay(200.0, 12.0, 0.0, 1.0)
        self.assertAlmostEqual(remaining_spin, 200.0 * math.exp(-0.5), places=12)

    def test_positive_backspin_produces_upward_lift_for_forward_motion(self):
        _, fy = self.physics.compute_magnus_force(10.0, 0.0, 100.0)
        self.assertGreater(fy, 0.0)

    def test_zero_spin_has_zero_magnus_force(self):
        self.assertEqual(self.physics.compute_magnus_force(10.0, 2.0, 0.0), (0.0, 0.0))

    def test_default_environment_is_about_20c_sea_level_density(self):
        self.assertAlmostEqual(self.env.air_density, 1.204, places=3)


if __name__ == "__main__":
    unittest.main()
