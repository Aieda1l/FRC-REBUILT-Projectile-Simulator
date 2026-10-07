import math
import unittest

from api.uncertainty import create_seeded_rng, evaluate_shot_uncertainty, sample_distribution, sample_shot_params

TOP_Z = 72.0 * 0.0254

def clean_vacuum_params():
    launch_x, launch_y, x_cross, vx = -3.0, 0.5, -0.30, 2.0
    t = (x_cross - launch_x) / vx
    vz = (TOP_Z - launch_y + 0.5 * 9.81 * t * t) / t
    return {
        "launchX": launch_x, "launchY": launch_y, "velocity": math.hypot(vx, vz),
        "angleDeg": math.degrees(math.atan2(vz, vx)), "spinRPM": 0.0,
        "mass": 0.215, "radius": 0.075, "dragCoeff": 0.47, "liftCoeff": 0.25,
        "airDensity": 1.204, "gravity": 9.81, "enableDrag": False, "enableMagnus": False,
        "targetX": 0.0, "targetLateralY": 0.0, "robotVelocity": [0.0, 0.0, 0.0], "wind": [0.0, 0.0, 0.0],
    }

class UncertaintyTests(unittest.TestCase):
    def test_seeded_rng_is_stable_across_languages(self):
        rng = create_seeded_rng(1)
        expected = [0.23645552527159452, 0.3692706737201661, 0.5042420323006809, 0.7048832636792213]
        for value in expected:
            self.assertAlmostEqual(rng(), value, places=15)

    def test_distribution_sampling_and_validation(self):
        self.assertEqual(sample_distribution({"kind": "fixed", "value": 2}, lambda: 0.5), 2)
        self.assertEqual(sample_distribution({"kind": "uniform", "min": 2, "max": 6}, lambda: 0.25), 3)
        self.assertEqual(sample_distribution({"kind": "normal", "mean": 4, "sigma": 0}, lambda: 0.5), 4)
        with self.assertRaises(ValueError):
            sample_distribution({"kind": "normal", "mean": 0, "sigma": -1}, lambda: 0.5)
        with self.assertRaises(ValueError):
            sample_distribution({"kind": "uniform", "min": 5, "max": 2}, lambda: 0.5)

    def test_sample_shot_params_uses_additive_and_multiplier_semantics(self):
        base = clean_vacuum_params()
        sampled = sample_shot_params(base, {
            "velocity": {"kind": "fixed", "value": 1}, "angleDeg": {"kind": "fixed", "value": -2},
            "spinRPM": {"kind": "fixed", "value": 50}, "mass": {"kind": "fixed", "value": 0.01},
            "dragMultiplier": {"kind": "fixed", "value": 1.1}, "liftMultiplier": {"kind": "fixed", "value": 0.8},
            "robotVelocity": [{"kind": "fixed", "value": 0.2}, {"kind": "fixed", "value": -0.1}, {"kind": "fixed", "value": 0}],
            "wind": [{"kind": "fixed", "value": 0.5}, {"kind": "fixed", "value": 0}, {"kind": "fixed", "value": 0}],
        }, create_seeded_rng(7))
        self.assertEqual(sampled["velocity"], base["velocity"] + 1)
        self.assertEqual(sampled["angleDeg"], base["angleDeg"] - 2)
        self.assertEqual(sampled["spinRPM"], 50)
        self.assertAlmostEqual(sampled["mass"], 0.225)
        self.assertAlmostEqual(sampled["dragCoeff"], 0.47 * 1.1)
        self.assertAlmostEqual(sampled["liftCoeff"], 0.25 * 0.8)
        self.assertEqual(sampled["robotVelocity"], [0.2, -0.1, 0.0])
        self.assertEqual(sampled["wind"], [0.5, 0.0, 0.0])

    def test_fixed_zero_uncertainty_is_a_clean_entry(self):
        result = evaluate_shot_uncertainty(
            clean_vacuum_params(),
            {"velocity": {"kind": "fixed", "value": 0}, "angleDeg": {"kind": "fixed", "value": 0}, "spinRPM": {"kind": "fixed", "value": 0}},
            sample_count=8, seed=2026, dt=0.002,
        )
        self.assertEqual(result["counts"]["clean-entry"], 8)
        self.assertEqual(result["probabilities"]["clean-entry"], 1)
        self.assertAlmostEqual(sum(result["probabilities"].values()), 1.0, places=12)
        self.assertTrue(math.isfinite(result["clearance"]["p10"]))
        self.assertTrue(math.isfinite(result["entryVelocity"]["median"]))
        self.assertTrue(math.isfinite(result["entryAngle"]["median"]))

    def test_invalid_sample_counts_fail_fast(self):
        with self.assertRaises(ValueError):
            evaluate_shot_uncertainty(clean_vacuum_params(), {}, sample_count=0)
        with self.assertRaises(ValueError):
            evaluate_shot_uncertainty(clean_vacuum_params(), {}, sample_count=10001)

if __name__ == "__main__":
    unittest.main()
