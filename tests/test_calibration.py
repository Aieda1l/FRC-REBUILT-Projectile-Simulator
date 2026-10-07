import unittest

from api.calibration import parse_calibration_profile


PROFILE = {
    "schema": "frc-projectile-calibration-v1",
    "name": "Test FUEL profile",
    "gamePiece": {"diameter": 0.15, "massReference": 0.215},
    "environment": {"dynamicViscosity": 1.81e-5},
    "dragModel": {
        "kind": "table1d",
        "reynolds": [50000, 200000],
        "coefficients": [0.5, 0.3],
    },
    "liftModel": {
        "kind": "table1d",
        "spinParameters": [0, 1],
        "coefficients": [0, 0.3],
    },
    "spinDecayTimeConstant": None,
    "domain": {"reynolds": [50000, 200000], "spinParameter": [0, 1]},
    "validation": {"rms3d": 0.04},
}


class CalibrationProfileTests(unittest.TestCase):
    def test_profile_v1_parses_and_normalizes_models(self):
        profile = parse_calibration_profile(PROFILE)
        self.assertEqual(profile["schema"], "frc-projectile-calibration-v1")
        self.assertEqual(profile["dragModel"]["kind"], "table1d")
        self.assertEqual(profile["liftModel"]["kind"], "table1d")

    def test_unknown_schema_is_rejected(self):
        with self.assertRaises(ValueError):
            parse_calibration_profile({**PROFILE, "schema": "future-v2"})

    def test_reversed_domain_is_rejected(self):
        with self.assertRaises(ValueError):
            parse_calibration_profile({
                **PROFILE,
                "domain": {
                    **PROFILE["domain"],
                    "reynolds": [200000, 50000],
                },
            })


if __name__ == "__main__":
    unittest.main()
