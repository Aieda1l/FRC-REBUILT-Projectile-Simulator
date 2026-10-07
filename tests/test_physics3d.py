"""3-D flight invariants. These fail before the Milestone 2 engine exists."""
import math
import unittest

import numpy as np

from api.physics3d import (
    FlightParameters,
    IntegrationError,
    aerodynamic_diagnostics,
    derivatives,
    integrate_trajectory,
    launch_state,
    rk4_step,
)


class Physics3DTests(unittest.TestCase):
    def setUp(self):
        self.vacuum = FlightParameters(enable_drag=False, enable_magnus=False, enable_buoyancy=False)


    def test_launch_state_has_canonical_order_and_float64(self):
        state = launch_state((1, 2, 3), (4, 5, 6), (7, 8, 9), (10, 11, 12))
        np.testing.assert_allclose(state, [1, 2, 3, 14, 16, 18, 7, 8, 9])
        self.assertEqual(state.dtype, np.float64)

    def test_rejects_wrong_vector_length_and_nonfinite_values(self):
        with self.assertRaises(ValueError):
            launch_state((0, 0), (1, 0, 0), (0, 0, 0))
        with self.assertRaises(ValueError):
            launch_state((0, 0, float("nan")), (1, 0, 0), (0, 0, 0))

    def test_rejects_invalid_physical_parameters(self):
        with self.assertRaises(ValueError):
            FlightParameters(mass=0)
        with self.assertRaises(ValueError):
            FlightParameters(spin_decay_time_constant=0)

    def test_zero_relative_airflow_has_no_aerodynamic_acceleration(self):
        state = launch_state((0, 0, 1), (10, 0, 0), (0, -100, 0))
        dv = derivatives(state, FlightParameters(wind=(10, 0, 0)))
        np.testing.assert_allclose(dv[3:6], [0, 0, -9.81])

    def test_magnus_acceleration_is_perpendicular_to_airflow(self):
        params = FlightParameters(enable_drag=False, enable_buoyancy=False, wind=(1.0, -2.0, 0.5))
        state = launch_state((0, 0, 2), (11, 3, 4), (20, -100, 30))
        dv = derivatives(state, params)
        u = state[3:6] - np.asarray(params.wind)
        magnus_accel = dv[3:6] - np.array([0.0, 0.0, -params.gravity])
        self.assertAlmostEqual(float(np.dot(magnus_accel, u)), 0.0, places=12)

    def test_launch_adds_robot_field_velocity(self):
        state = launch_state(
            position=(0, 0, 1),
            muzzle_velocity=(10, 0, 5),
            spin=(0, -100, 0),
            robot_velocity=(1, 2, 0),
        )
        np.testing.assert_allclose(state[3:6], [11, 2, 5])
        np.testing.assert_allclose(state[6:], [0, -100, 0])



    def test_buoyancy_reduces_effective_downward_acceleration(self):
        params = FlightParameters(enable_drag=False, enable_magnus=False)
        state = launch_state((0, 0, 2), (0, 0, 0), (0, 0, 0))
        volume = (4.0 / 3.0) * math.pi * params.radius ** 3
        expected = (
            -params.gravity
            + params.air_density * volume * params.gravity / params.mass
        )
        self.assertAlmostEqual(derivatives(state, params)[5], expected, places=12)
        self.assertEqual(
            derivatives(
                state,
                FlightParameters(
                    enable_drag=False,
                    enable_magnus=False,
                    enable_buoyancy=False,
                ),
            )[5],
            -params.gravity,
        )
        self.assertEqual(
            derivatives(
                state,
                FlightParameters(
                    enable_drag=False,
                    enable_magnus=False,
                    air_density=0,
                ),
            )[5],
            -params.gravity,
        )
        self.assertEqual(
            derivatives(
                state,
                FlightParameters(
                    enable_drag=False,
                    enable_magnus=False,
                    gravity=0,
                ),
            )[5],
            0,
        )

    def test_signed_lift_reverses_magnus_acceleration(self):
        state = launch_state((0, 0, 2), (10, 0, 0), (0, -100, 0))
        positive_params = FlightParameters(
            enable_drag=False,
            enable_buoyancy=False,
            lift_model={
                "kind": "table1d",
                "spinParameters": [0, 1],
                "coefficients": [0.2, 0.2],
            },
        )
        negative_params = FlightParameters(
            enable_drag=False,
            enable_buoyancy=False,
            lift_model={
                "kind": "table1d",
                "spinParameters": [0, 1],
                "coefficients": [-0.2, -0.2],
            },
        )
        positive = derivatives(state, positive_params)[5] + positive_params.gravity
        negative = derivatives(state, negative_params)[5] + negative_params.gravity
        self.assertGreater(positive, 0)
        self.assertLess(negative, 0)
        self.assertAlmostEqual(positive, -negative, places=12)

    def test_2d_drag_uses_current_spin_parameter(self):
        drag_model = {
            "kind": "table2d",
            "reynolds": [50000, 200000],
            "spinParameters": [0, 1],
            "coefficients": [[0.2, 0.8], [0.2, 0.8]],
        }
        params = FlightParameters(
            enable_magnus=False,
            enable_buoyancy=False,
            drag_model=drag_model,
        )
        unspun = derivatives(
            launch_state((0, 0, 2), (10, 0, 0), (0, 0, 0)),
            params,
        )[3]
        spun = derivatives(
            launch_state((0, 0, 2), (10, 0, 0), (0, -100, 0)),
            params,
        )[3]
        self.assertLess(spun, unspun)

    def test_zero_relative_airflow_has_no_drag_or_magnus_with_2d_drag(self):
        params = FlightParameters(
            wind=(10, 0, 0),
            enable_buoyancy=False,
            drag_model={
                "kind": "table2d",
                "reynolds": [50000, 200000],
                "spinParameters": [0, 1],
                "coefficients": [[0.2, 0.8], [0.2, 0.8]],
            },
        )
        state = launch_state((0, 0, 2), (10, 0, 0), (0, -100, 0))
        np.testing.assert_allclose(derivatives(state, params)[3:6], [0, 0, -9.81])
        self.assertTrue(aerodynamic_diagnostics(state, params)["dragClamped"])

    def test_rk4_fourth_order_convergence(self):
        initial = launch_state((0, 0, 2), (12, 2, 8), (0, -120, 30))
        params = FlightParameters(wind=(1, -0.5, 0))
        reference = integrate_trajectory(initial, params, dt=0.0005, max_time=0.4, terminal_height=None)[-1].state
        coarse = integrate_trajectory(initial, params, dt=0.04, max_time=0.4, terminal_height=None)[-1].state
        medium = integrate_trajectory(initial, params, dt=0.02, max_time=0.4, terminal_height=None)[-1].state
        fine = integrate_trajectory(initial, params, dt=0.01, max_time=0.4, terminal_height=None)[-1].state
        e1 = np.linalg.norm(coarse - reference)
        e2 = np.linalg.norm(medium - reference)
        e3 = np.linalg.norm(fine - reference)
        self.assertGreater(e1 / e2, 10.0)
        self.assertGreater(e2 / e3, 10.0)

    def test_max_time_is_exact_when_no_terminal_height(self):
        samples = integrate_trajectory(
            launch_state((0, 0, 2), (3, 0, 4), (0, 0, 0)),
            self.vacuum, dt=0.03, max_time=0.2, terminal_height=None,
        )
        self.assertAlmostEqual(samples[-1].time, 0.2, places=12)

    def test_crossing_direction_ignores_ascending_pass(self):
        initial = launch_state((0, 0, 0.5), (2, 0, 5), (0, 0, 0))
        descending = integrate_trajectory(
            initial, self.vacuum, dt=0.02, max_time=2,
            terminal_height=1.0, terminal_direction=-1,
        )
        self.assertLess(descending[-1].state[5], 0)
        self.assertAlmostEqual(descending[-1].state[2], 1.0, places=10)

    def test_rejects_bad_terminal_direction(self):
        initial = launch_state((0, 0, 1), (1, 0, 0), (0, 0, 0))
        with self.assertRaises(ValueError):
            integrate_trajectory(initial, self.vacuum, terminal_direction=2)

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


    def test_rk45_reduces_step_for_tighter_tolerance(self):
        initial = launch_state((0, 0, 2), (12, 3, 8), (0, -100, 20))
        params = FlightParameters(wind=(1, -0.5, 0))
        loose = integrate_trajectory(initial, params, method="rk45", dt=0.2, max_step=0.2, max_time=0.5, terminal_height=None, rtol=1e-3, atol=1e-6)
        tight = integrate_trajectory(initial, params, method="rk45", dt=0.2, max_step=0.2, max_time=0.5, terminal_height=None, rtol=1e-8, atol=1e-10)
        self.assertGreater(len(tight), len(loose))

    def test_rk45_hits_max_time_exactly(self):
        initial = launch_state((0, 0, 2), (3, 0, 4), (0, 0, 0))
        samples = integrate_trajectory(initial, self.vacuum, method="rk45", dt=0.03, max_time=0.2, terminal_height=None)
        self.assertAlmostEqual(samples[-1].time, 0.2, places=12)

    def test_rk45_raises_when_tolerance_cannot_be_met_at_min_step(self):
        initial = launch_state((0, 0, 2), (40, 15, 25), (0, -800, 300))
        params = FlightParameters(wind=(3, -2, 0))
        with self.assertRaises(IntegrationError):
            integrate_trajectory(initial, params, method="rk45", dt=0.2, min_step=0.2, max_step=0.2, max_time=0.2, rtol=1e-16, atol=1e-16, terminal_height=None)

    def test_rk45_rejects_invalid_tolerances_and_step_bounds(self):
        initial = launch_state((0, 0, 1), (1, 0, 1), (0, 0, 0))
        with self.assertRaises(ValueError):
            integrate_trajectory(initial, self.vacuum, method="rk45", rtol=0)
        with self.assertRaises(ValueError):
            integrate_trajectory(initial, self.vacuum, method="rk45", min_step=0.1, max_step=0.01)

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
        params = FlightParameters(enable_drag=False, enable_buoyancy=False)
        backspin = launch_state((0, 0, 2), (10, 0, 0), (0, -100, 0))
        sidespin = launch_state((0, 0, 2), (10, 0, 0), (0, 0, 100))
        self.assertGreater(derivatives(backspin, params)[5], -params.gravity)
        self.assertGreater(derivatives(sidespin, params)[4], 0)

    def test_only_spin_perpendicular_to_airflow_produces_lift(self):
        params = FlightParameters(enable_drag=False, enable_buoyancy=False)
        axis_aligned = launch_state((0, 0, 2), (10, 0, 0), (100, 0, 0))
        self.assertAlmostEqual(derivatives(axis_aligned, params)[5], -params.gravity)

    def test_drag_uses_air_relative_velocity(self):
        initial = launch_state((0, 0, 2), (10, 0, 0), (0, 0, 0))
        still_air = derivatives(initial, FlightParameters(enable_magnus=False, enable_buoyancy=False))
        matching_wind = derivatives(
            initial,
            FlightParameters(enable_magnus=False, enable_buoyancy=False, wind=(10, 0, 0)),
        )
        self.assertLess(still_air[3], 0)
        self.assertAlmostEqual(matching_wind[3], 0)

    def test_calibrated_spin_decay_is_part_of_ode(self):
        params = FlightParameters(
            enable_drag=False, enable_magnus=False, spin_decay_time_constant=2
        )
        initial = launch_state((0, 0, 2), (0, 0, 0), (0, -200, 0))
        state = rk4_step(initial, params, 0.25)
        state = rk4_step(state, params, 0.25)
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


    def test_advanced_models_override_scalar_fallbacks(self):
        state = launch_state((0, 0, 1), (10, 0, 0), (0, -100, 0))
        diagnostics = aerodynamic_diagnostics(
            state,
            FlightParameters(
                drag_coefficient=0.99,
                lift_coefficient=0.99,
                drag_model={"kind": "constant", "coefficient": 0.12},
                lift_model={
                    "kind": "table2d",
                    "reynolds": [50000, 200000],
                    "spinParameters": [0, 1],
                    "coefficients": [[0, 0.1], [0, 0.3]],
                },
            ),
        )
        self.assertEqual(diagnostics["dragCoefficient"], 0.12)
        self.assertGreater(diagnostics["liftCoefficient"], 0)
        self.assertLess(diagnostics["liftCoefficient"], 0.99)
        self.assertFalse(diagnostics["dragClamped"])
        self.assertFalse(diagnostics["liftClamped"])

    def test_aerodynamic_diagnostics_are_finite_at_zero_relative_airflow(self):
        state = launch_state((0, 0, 1), (10, 0, 0), (0, -100, 0))
        diagnostics = aerodynamic_diagnostics(
            state,
            FlightParameters(wind=(10, 0, 0)),
        )
        self.assertEqual(diagnostics, {
            "reynolds": 0.0,
            "spinParameter": 0.0,
            "dragCoefficient": 0.47,
            "liftCoefficient": 0.0,
            "dragClamped": False,
            "liftClamped": False,
        })


if __name__ == "__main__":
    unittest.main()
