import asyncio
import unittest

from fastapi import HTTPException
from pydantic import ValidationError

from api.main import Sim3DRequest, SimRequest, simulate, simulate3d


class ApiTests(unittest.TestCase):
    def test_legacy_simulate_response_shape_is_unchanged(self):
        response = asyncio.run(simulate(SimRequest(
            velocity=10, angle=45, spin_rate=0, launch_x=0, launch_y=1
        )))
        self.assertEqual(set(response), {"success", "points", "hit"})
        self.assertEqual(set(response["points"][0]), {"x", "y"})

    def test_simulate3d_defaults_to_rk45_and_returns_full_vectors(self):
        response = asyncio.run(simulate3d(Sim3DRequest(
            position=(0, 0, 1),
            muzzle_velocity=(2, 3, 4),
            spin=(0, 0, 0),
            enable_drag=False,
            enable_magnus=False,
            max_time=0.1,
        )))
        self.assertEqual(response["method"], "rk45")
        self.assertEqual(response["samples"][0]["position"], [0.0, 0.0, 1.0])
        self.assertEqual(response["samples"][0]["velocity"], [2.0, 3.0, 4.0])
        self.assertEqual(len(response["samples"][0]["spin"]), 3)

    def test_simulate3d_adds_robot_velocity_and_matching_wind_removes_drag(self):
        response = asyncio.run(simulate3d(Sim3DRequest(
            position=(0, 0, 1),
            muzzle_velocity=(10, 0, 0),
            spin=(0, 0, 0),
            robot_velocity=(1, 2, 0),
            wind=(11, 2, 0),
            gravity=0,
            max_time=0.1,
        )))
        self.assertEqual(response["samples"][0]["velocity"], [11.0, 2.0, 0.0])
        self.assertAlmostEqual(response["samples"][-1]["velocity"][0], 11.0, places=8)
        self.assertAlmostEqual(response["samples"][-1]["velocity"][1], 2.0, places=8)

    def test_simulate3d_physical_overrides_affect_result(self):
        response = asyncio.run(simulate3d(Sim3DRequest(
            position=(0, 0, 1),
            muzzle_velocity=(0, 0, 0),
            spin=(0, 0, 0),
            gravity=3.0,
            enable_drag=False,
            enable_magnus=False,
            max_time=0.1,
        )))
        self.assertAlmostEqual(response["samples"][-1]["velocity"][2], -0.3, places=6)

    def test_simulate3d_converts_integration_error_to_http_400(self):
        request = Sim3DRequest(
            position=(0, 0, 2),
            muzzle_velocity=(40, 15, 25),
            spin=(0, -800, 300),
            wind=(3, -2, 0),
            method="rk45",
            dt=0.2,
            min_step=0.2,
            max_step=0.2,
            max_time=0.2,
            rtol=1e-16,
            atol=1e-16,
        )
        with self.assertRaises(HTTPException) as ctx:
            asyncio.run(simulate3d(request))
        self.assertEqual(ctx.exception.status_code, 400)


    def test_simulate3d_rejects_inverted_step_bounds_at_model_validation(self):
        with self.assertRaises(ValidationError):
            Sim3DRequest(
                position=(0, 0, 1),
                muzzle_velocity=(1, 0, 0),
                spin=(0, 0, 0),
                min_step=0.1,
                max_step=0.01,
            )

    def test_simulate3d_rejects_nonfinite_vector_values(self):
        with self.assertRaises(ValidationError):
            Sim3DRequest(
                position=(0, 0, float("nan")),
                muzzle_velocity=(1, 0, 0),
                spin=(0, 0, 0),
            )

    def test_simulate3d_rejects_wrong_vector_length(self):
        with self.assertRaises(ValidationError):
            Sim3DRequest(
                position=(0, 0),
                muzzle_velocity=(1, 0, 0),
                spin=(0, 0, 0),
            )


    def test_simulate3d_accepts_calibrated_aerodynamic_models(self):
        response = asyncio.run(simulate3d(Sim3DRequest(
            position=(0, 0, 1),
            muzzle_velocity=(10, 0, 0),
            spin=(0, 0, 0),
            gravity=0,
            drag_coefficient=0,
            drag_model={"kind": "constant", "coefficient": 0.5},
            enable_magnus=False,
            max_time=0.1,
        )))
        self.assertLess(response["samples"][-1]["velocity"][0], 10.0)

    def test_simulate3d_rejects_malformed_aerodynamic_model_as_http_400(self):
        request = Sim3DRequest(
            position=(0, 0, 1),
            muzzle_velocity=(10, 0, 0),
            spin=(0, 0, 0),
            drag_model={"kind": "table1d", "reynolds": [100000], "coefficients": [0.5]},
        )
        with self.assertRaises(HTTPException) as ctx:
            asyncio.run(simulate3d(request))
        self.assertEqual(ctx.exception.status_code, 400)


if __name__ == "__main__":
    unittest.main()
