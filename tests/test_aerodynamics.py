import unittest

from api.aerodynamics import (
    DEFAULT_DYNAMIC_VISCOSITY,
    evaluate_drag_model,
    evaluate_lift_model,
    normalize_drag_model,
    normalize_lift_model,
    reynolds_number,
    spin_parameter,
)


class AerodynamicsTests(unittest.TestCase):
    def test_reynolds_number_uses_rho_v_d_over_mu(self):
        actual = reynolds_number(
            air_density=1.204,
            speed=12.0,
            diameter=0.15,
            dynamic_viscosity=DEFAULT_DYNAMIC_VISCOSITY,
        )
        self.assertAlmostEqual(actual, 1.204 * 12.0 * 0.15 / 1.81e-5, places=9)

    def test_spin_parameter_is_finite_and_zero_without_airflow(self):
        self.assertEqual(spin_parameter(radius=0.075, perpendicular_spin=200, speed=0), 0)
        self.assertEqual(spin_parameter(radius=0.075, perpendicular_spin=200, speed=1e-13), 0)
        self.assertAlmostEqual(
            spin_parameter(radius=0.075, perpendicular_spin=80, speed=12),
            0.5,
            places=12,
        )

    def test_constant_and_1d_drag_models_evaluate_and_clamp(self):
        constant = normalize_drag_model({"kind": "constant", "coefficient": 0.47}, 0.1)
        self.assertEqual(
            evaluate_drag_model(constant, 120000),
            {"coefficient": 0.47, "clamped": False},
        )
        model = normalize_drag_model(
            {"kind": "table1d", "reynolds": [100000, 200000], "coefficients": [0.5, 0.3]},
            0.47,
        )
        self.assertEqual(evaluate_drag_model(model, 150000), {"coefficient": 0.4, "clamped": False})
        self.assertEqual(evaluate_drag_model(model, 50000), {"coefficient": 0.5, "clamped": True})

    def test_legacy_1d_and_2d_lift_models_interpolate(self):
        legacy = normalize_lift_model({"kind": "legacy-spin-cap", "maxCoefficient": 0.25}, 0.1)
        self.assertEqual(
            evaluate_lift_model(legacy, 120000, 0.25),
            {"coefficient": 0.125, "clamped": False},
        )
        self.assertEqual(
            evaluate_lift_model(legacy, 120000, 0.75),
            {"coefficient": 0.25, "clamped": False},
        )
        one_d = normalize_lift_model(
            {"kind": "table1d", "spinParameters": [0, 1], "coefficients": [0, 0.4]},
            0.25,
        )
        self.assertEqual(
            evaluate_lift_model(one_d, 120000, 0.5),
            {"coefficient": 0.2, "clamped": False},
        )
        self.assertEqual(
            evaluate_lift_model(one_d, 120000, 2),
            {"coefficient": 0.4, "clamped": True},
        )
        two_d = normalize_lift_model({
            "kind": "table2d",
            "reynolds": [100000, 200000],
            "spinParameters": [0, 1],
            "coefficients": [[0, 0.2], [0.2, 0.6]],
        }, 0.25)
        self.assertEqual(
            evaluate_lift_model(two_d, 150000, 0.5),
            {"coefficient": 0.25, "clamped": False},
        )

    def test_invalid_tables_are_rejected(self):
        with self.assertRaises(ValueError):
            normalize_drag_model({"kind": "table1d", "reynolds": [100000], "coefficients": [0.4]}, 0.47)
        with self.assertRaises(ValueError):
            normalize_drag_model({"kind": "table1d", "reynolds": [200000, 100000], "coefficients": [0.3, 0.4]}, 0.47)
        with self.assertRaises(ValueError):
            normalize_drag_model({"kind": "table1d", "reynolds": [100000, 100000], "coefficients": [0.3, 0.4]}, 0.47)
        with self.assertRaises(ValueError):
            normalize_lift_model({
                "kind": "table2d",
                "reynolds": [100000, 200000],
                "spinParameters": [0, 1],
                "coefficients": [[0, 0.1]],
            }, 0.25)


if __name__ == "__main__":
    unittest.main()
