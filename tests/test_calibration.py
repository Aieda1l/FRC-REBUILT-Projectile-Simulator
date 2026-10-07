import math
import unittest

import calibration.fitting as fitting

from calibration.fitting import (
    fit_drag_model,
    fit_lift_model,
    fit_spin_decay,
    parse_calibration_profile,
    split_shots,
    validate_profile,
    _flight_parameters_for_shot,
)
from api.physics3d import FlightParameters, integrate_trajectory, launch_state


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

    def test_profile_v1_accepts_2d_drag_and_signed_lift_tables(self):
        profile = parse_calibration_profile({
            **PROFILE,
            "dragModel": {
                "kind": "table2d",
                "reynolds": [50000, 200000],
                "spinParameters": [0, 1],
                "coefficients": [[0.5, 0.45], [0.35, 0.3]],
            },
            "liftModel": {
                "kind": "table1d",
                "spinParameters": [0, 1],
                "coefficients": [-0.1, 0.3],
            },
        })
        self.assertEqual(profile["dragModel"]["kind"], "table2d")
        self.assertEqual(profile["liftModel"]["coefficients"], [-0.1, 0.3])

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


def synthetic_shot(shot_id, speed, spin, params, duration=0.3):
    initial = launch_state((0, 0, 1), (speed, 0, 4), (0, -spin, 0))
    samples = integrate_trajectory(
        initial,
        params,
        dt=0.002,
        max_time=duration,
        terminal_height=None,
    )
    observations = []
    for time in (0.1, 0.2, 0.3):
        index = round(time / 0.002)
        observations.append({
            "time": time,
            "position": samples[index].state[:3].tolist(),
        })
    return {
        "id": shot_id,
        "position": [0, 0, 1],
        "muzzleVelocity": [speed, 0, 4],
        "spin": [0, -spin, 0],
        "robotVelocity": [0, 0, 0],
        "wind": [0, 0, 0],
        "observations": observations,
    }


def make_profile(drag_model, lift_model):
    return {
        "schema": "frc-projectile-calibration-v1",
        "name": "synthetic",
        "gamePiece": {"diameter": 0.15, "massReference": 0.215},
        "environment": {"dynamicViscosity": 1.81e-5},
        "dragModel": drag_model,
        "liftModel": lift_model,
        "spinDecayTimeConstant": None,
        "domain": {"reynolds": [0, 500000], "spinParameter": [0, 2]},
        "validation": {},
    }


class CalibrationFittingTests(unittest.TestCase):
    def setUp(self):
        self.base = {
            "mass": 0.215,
            "radius": 0.075,
            "air_density": 1.204,
            "gravity": 9.81,
            "dynamic_viscosity": 1.81e-5,
        }


    def test_flight_parameter_reconstruction_preserves_buoyancy_opt_out(self):
        base = FlightParameters(enable_buoyancy=False)
        shot = {
            "position": [0, 0, 1],
            "muzzleVelocity": [10, 0, 4],
            "spin": [0, 0, 0],
        }
        rebuilt = _flight_parameters_for_shot(
            shot,
            base,
            drag_model=base.drag_model,
            lift_model=base.lift_model,
        )
        self.assertFalse(rebuilt.enable_buoyancy)

    def test_split_is_deterministic_for_seed(self):
        shots = [{"id": f"s{index}"} for index in range(12)]
        a_train, a_validation = split_shots(shots, 0.25, 2026)
        b_train, b_validation = split_shots(shots, 0.25, 2026)
        self.assertEqual([shot["id"] for shot in a_train], [shot["id"] for shot in b_train])
        self.assertEqual([shot["id"] for shot in a_validation], [shot["id"] for shot in b_validation])
        self.assertFalse(
            set(shot["id"] for shot in a_train)
            & set(shot["id"] for shot in a_validation)
        )


    def test_spin_partition_uses_dimensionless_spin_parameter(self):
        low_s_high_raw_spin = {
            "id": "low-s",
            "position": [0, 0, 1],
            "muzzleVelocity": [100, 0, 0],
            "spin": [0, -20, 0],
        }
        high_s_low_raw_spin = {
            "id": "high-s",
            "position": [0, 0, 1],
            "muzzleVelocity": [1, 0, 0],
            "spin": [0, -10, 0],
        }
        drag, spinning = fitting.partition_shots_by_spin_parameter(
            [low_s_high_raw_spin, high_s_low_raw_spin],
            self.base,
            0.05,
        )
        self.assertEqual([shot["id"] for shot in drag], ["low-s"])
        self.assertEqual([shot["id"] for shot in spinning], ["high-s"])

    def test_spin_partition_includes_threshold_boundary_and_rejects_invalid_threshold(self):
        boundary = {
            "id": "boundary",
            "position": [0, 0, 1],
            "muzzleVelocity": [10, 0, 0],
            "spin": [0, -10, 0],
        }
        drag, spinning = fitting.partition_shots_by_spin_parameter(
            [boundary],
            self.base,
            0.075,
        )
        self.assertEqual([shot["id"] for shot in drag], ["boundary"])
        self.assertEqual(spinning, [])
        for invalid in (-0.01, float("inf"), float("nan")):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    fitting.partition_shots_by_spin_parameter([boundary], self.base, invalid)

    def test_constant_drag_fit_recovers_synthetic_coefficient(self):
        truth = FlightParameters(
            **self.base,
            drag_coefficient=0.36,
            lift_coefficient=0,
            enable_magnus=False,
        )
        shots = [
            synthetic_shot("d8", 8, 0, truth),
            synthetic_shot("d12", 12, 0, truth),
            synthetic_shot("d16", 16, 0, truth),
        ]
        fitted = fit_drag_model(shots, self.base, model_kind="constant")
        self.assertEqual(fitted["kind"], "constant")
        self.assertAlmostEqual(fitted["coefficient"], 0.36, delta=0.02)

    def test_fitted_lift_reduces_held_out_position_error(self):
        drag_model = {"kind": "constant", "coefficient": 0.36}
        truth = FlightParameters(
            **self.base,
            drag_model=drag_model,
            lift_model={
                "kind": "table1d",
                "spinParameters": [0, 1],
                "coefficients": [0, 0.28],
            },
        )
        train = [
            synthetic_shot("l1", 10, 50, truth),
            synthetic_shot("l2", 11, 80, truth),
            synthetic_shot("l3", 12, 110, truth),
            synthetic_shot("l4", 13, 140, truth),
        ]
        held_out = [synthetic_shot("held", 11.5, 95, truth)]
        fitted_lift = fit_lift_model(train, self.base, drag_model, model_kind="table1d")
        fitted_metrics = validate_profile(
            held_out,
            make_profile(drag_model, fitted_lift),
            self.base,
        )
        zero_metrics = validate_profile(
            held_out,
            make_profile(
                drag_model,
                {"kind": "table1d", "spinParameters": [0, 1], "coefficients": [0, 0]},
            ),
            self.base,
        )
        self.assertLess(fitted_metrics["rms3d"], zero_metrics["rms3d"])



    def test_lift_fit_can_recover_negative_coefficients(self):
        drag_model = {"kind": "constant", "coefficient": 0.36}
        truth = FlightParameters(
            **self.base,
            drag_model=drag_model,
            lift_model={
                "kind": "table1d",
                "spinParameters": [0, 1],
                "coefficients": [-0.2, -0.2],
            },
        )
        train = [
            synthetic_shot("n1", 10, 50, truth),
            synthetic_shot("n2", 11, 80, truth),
            synthetic_shot("n3", 12, 110, truth),
            synthetic_shot("n4", 13, 140, truth),
        ]
        held_out = [synthetic_shot("negative-held", 11.5, 95, truth)]
        fitted = fit_lift_model(train, self.base, drag_model, model_kind="table1d")
        self.assertLess(min(fitted["coefficients"]), 0)
        fitted_metrics = validate_profile(
            held_out,
            make_profile(drag_model, fitted),
            self.base,
        )
        zero_metrics = validate_profile(
            held_out,
            make_profile(
                drag_model,
                {"kind": "table1d", "spinParameters": [0, 1], "coefficients": [0, 0]},
            ),
            self.base,
        )
        self.assertLess(fitted_metrics["rms3d"], zero_metrics["rms3d"])

    def test_validation_reports_entry_angle_and_clean_entry_confusion(self):
        def hub_shot(shot_id, x_cross, observed_result):
            launch_x, launch_z, vx = -3.0, 0.5, 2.0
            top_z = 72.0 * 0.0254
            crossing_time = (x_cross - launch_x) / vx
            vz = (top_z - launch_z + 0.5 * 9.81 * crossing_time ** 2) / crossing_time
            truth = FlightParameters(
                **{**self.base, "air_density": 0.0},
                drag_coefficient=0.0,
                lift_coefficient=0.0,
            )
            initial = launch_state(
                (launch_x, 0.0, launch_z),
                (vx, 0.0, vz),
                (0.0, 0.0, 0.0),
            )
            samples = integrate_trajectory(
                initial,
                truth,
                dt=0.002,
                max_time=1.8,
                terminal_height=None,
            )
            shot = {
                "id": shot_id,
                "position": [launch_x, 0.0, launch_z],
                "muzzleVelocity": [vx, 0.0, vz],
                "spin": [0.0, 0.0, 0.0],
                "robotVelocity": [0.0, 0.0, 0.0],
                "wind": [0.0, 0.0, 0.0],
                "airDensity": 0.0,
                "targetX": 0.0,
                "targetLateralY": 0.0,
                "observedHubResult": observed_result,
                "observations": [{
                    "time": 1.8,
                    "position": samples[-1].state[:3].tolist(),
                }],
            }
            if observed_result == "clean-entry":
                vertical_at_top = vz - 9.81 * crossing_time
                shot["observedEntryAngle"] = math.degrees(math.atan2(vertical_at_top, vx))
            return shot

        profile = make_profile(
            {"kind": "constant", "coefficient": 0.0},
            {"kind": "table1d", "spinParameters": [0, 1], "coefficients": [0, 0]},
        )
        metrics = validate_profile(
            [
                hub_shot("clean", -0.30, "clean-entry"),
                hub_shot("miss", 1.20, "miss"),
            ],
            profile,
            self.base,
        )
        self.assertAlmostEqual(metrics["entryAngleRms"], 0.0, delta=0.05)
        self.assertEqual(metrics["cleanEntryConfusion"], {
            "truePositive": 1,
            "trueNegative": 1,
            "falsePositive": 0,
            "falseNegative": 0,
        })

    def test_spin_decay_is_not_inferred_without_spin_measurements(self):
        truth = FlightParameters(**self.base)
        shots = [synthetic_shot("s1", 10, 100, truth)]
        self.assertIsNone(fit_spin_decay(shots, self.base))

    def test_table_drag_fit_rejects_insufficient_independent_data(self):
        truth = FlightParameters(**self.base, drag_coefficient=0.4, enable_magnus=False)
        shots = [
            synthetic_shot("d1", 10, 0, truth),
            synthetic_shot("d2", 11, 0, truth),
        ]
        with self.assertRaises(ValueError):
            fit_drag_model(shots, self.base, model_kind="table1d")


if __name__ == "__main__":
    unittest.main()
