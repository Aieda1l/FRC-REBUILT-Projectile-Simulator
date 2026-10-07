"""3-D flight invariants. These fail before the Milestone 2 engine exists."""
import math
import unittest

import numpy as np

from api.physics3d import (
    FlightParameters,
    derivatives,
    integrate_trajectory,
    launch_state,
    rk4_step,
)


class Physics3DTests(unittest.TestCase):
    def setUp(self):
        self.vacuum = FlightParameters(enable_drag=False, enable_magnus=False)

    def test_launch_adds_robot_field_velocity(self):
        state = launch_state(
            position=(0, 0, 1),
            muzzle_velocity=(10, 0, 5),
            spin=(0, -100, 0),
            robot_velocity=(1, 2, 0),
        )
        np.testing.assert_allclose(state[3:6], [11, 2, 5])
        np.testing.assert_allclose(state[6:], [0, -100, 0])

    def test_gravity_only_rk4_matches_analytic_solution(self):
        initial = launch_state((0, 0, 2), (10, 3, 5), (0, 0, 0))
        samples = integrate_trajectory(initial, self.vacuum, method="rk4", dt=0.01, max_time=0.4)
        last = samples[-1]
        self.assertAlmostEqual(last.time, 0.4, places=10)
        np.testing.assert_allclose(
            last.state[:3],
            [4, 1.2, 2 + 5 * 0.4 - 0.5 * 9.81 * 0.4 ** 2],
            atol=1e-9
        )

    def test_rk45_matches_rk4_with_adaptive_steps(self):
        initial = launch_state((0, 0, 2), (12, 3, 8), (0, -100, 0))
        base = FlightParameters()
        reference = integrate_trajectory(initial, base, method="rk4", dt=0.001, max_time=0.5)
        adaptive = integrate_trajectory(
            initial, base, method="rk45", dt=0.2, max_step=0.2,
            max_time=0.5, rtol=1e-7, atol=1e-9
        )
        np.testing.assert_allclose(adaptive[-1].state, reference[-1].state, atol=2e-4)

    def test_magnus_backspin_lifts_and_sidespin_deflects(self):
        params = FlightParameters(enable_drag=False)
        backspin = launch_state((0, 0, 2), (10, 0, 0), (0, -100, 0))
        sidespin = launch_state((0, 0, 2), (10, 0, 0), (0, 0, 100))
        self.assertGreater(derivatives(backspin, params)[5], -params.gravity)
        self.assertGreater(derivatives(sidespin, params)[4], 0)

    def test_only_spin_perpendicular_to_airflow_produces_lift(self):
        params = FlightParameters(enable_drag=False)
        axis_aligned = launch_state((0, 0, 2), (10, 0, 0), (100, 0, 0))
        self.assertAlmostEqual(derivatives(axis_aligned, params)[5], -params.gravity)

    def test_drag_uses_air_relative_velocity(self):
        initial = launch_state((0, 0, 2), (10, 0, 0), (0, 0, 0))
        still_air = derivatives(initial, FlightParameters(enable_magnus=False))
        matching_wind = derivatives(
            initial,
            FlightParameters(enable_magnus=False, wind=(10, 0, 0)),
        )
        self.assertLess(still_air[3], 0)
        self.assertAlmostEqual(matching_wind[3], 0)

    def test_calibrated_spin_decay_is_part_of_ode(self):
        params = FlightParameters(
            enable_drag=False, enable_magnus=False, spin_decay_time_constant=2
        )
        initial = launch_state((0, 0, 2), (0, 0, 0), (0, -200, 0))
        state = rk4_step(initial, params, 0.5)
        self.assertAlmostEqual(state[7], -200 * math.exp(-0.25), delta=0.001)

    def test_ground_crossing_is_interpolated_and_terminated(self):
        initial = launch_state((0, 0, 1), (2, 0, 0), (0, 0, 0))
        samples = integrate_trajectory(initial, self.vacuum, dt=0.06, max_time=3)
        expected_time = math.sqrt(2 / self.vacuum.gravity)
        self.assertAlmostEqual(samples[-1].state[2], 0.0, places=9)
        self.assertAlmostEqual(samples[-1].time, expected_time, delta=0.003)
        self.assertAlmostEqual(samples[-1].state[0], 2 * expected_time, delta=0.006)

    def test_rejects_nonpositive_time_step(self):
        initial = launch_state((0, 0, 1), (1, 0, 0), (0, 0, 0))
        with self.assertRaises(ValueError):
            integrate_trajectory(initial, self.vacuum, dt=0)

    def test_rejects_bad_method(self):
        initial = launch_state((0, 0, 1), (1, 0, 0), (0, 0, 0))
        with self.assertRaises(ValueError):
            integrate_trajectory(initial, self.vacuum, method="euler")


if __name__ == "__main__":
    unittest.main()
