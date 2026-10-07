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
            ],
        )

    def test_fixture_schema_and_case_order(self):
        fixture = build_fixture()
        self.assertEqual(fixture["schema"], "physics3d-golden-v1")
        self.assertEqual([case["name"] for case in fixture["cases"]], CASE_NAMES)


if __name__ == "__main__":
    unittest.main()
