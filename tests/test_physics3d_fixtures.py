import unittest

from scripts.generate_physics3d_fixtures import CASE_NAMES, build_fixture


class Physics3DFixtureTests(unittest.TestCase):
    def test_case_names_are_stable(self):
        self.assertEqual(
            CASE_NAMES,
            [
                "vacuum",
                "drag_only",
                "backspin",
                "sidespin",
                "matching_wind",
                "robot_velocity",
                "spin_decay",
                "rk45",
                "calibrated_drag_wind_robot",
                "calibrated_lift_wind_robot",
            ],
        )

    def test_fixture_schema_and_case_order(self):
        fixture = build_fixture()
        self.assertEqual(fixture["schema"], "physics3d-golden-v1")
        self.assertEqual([case["name"] for case in fixture["cases"]], CASE_NAMES)
        drag_case = next(case for case in fixture["cases"] if case["name"] == "calibrated_drag_wind_robot")
        lift_case = next(case for case in fixture["cases"] if case["name"] == "calibrated_lift_wind_robot")
        self.assertEqual(drag_case["params"]["dragModel"]["kind"], "table1d")
        self.assertEqual(lift_case["params"]["liftModel"]["kind"], "table2d")
        self.assertNotEqual(drag_case["params"]["wind"], [0.0, 0.0, 0.0])
        self.assertNotEqual(lift_case["params"]["wind"], [0.0, 0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
