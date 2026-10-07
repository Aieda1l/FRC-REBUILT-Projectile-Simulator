#!/usr/bin/env python3
"""Generate deterministic Python-reference fixtures for JS physics parity."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api.physics3d import (
    FlightParameters,
    derivatives,
    integrate_trajectory,
    launch_state,
    rk4_step,
)

CASE_NAMES = [
    "vacuum",
    "drag_only",
    "backspin",
    "sidespin",
    "matching_wind",
    "robot_velocity",
    "spin_decay",
    "rk45",
]


def _list(values):
    return [float(value) for value in values]


def build_fixture():
    cases = []

    vacuum_initial = launch_state((0, 0, 2), (10, 3, 5), (0, 0, 0))
    vacuum_params = FlightParameters(enable_drag=False, enable_magnus=False)
    vacuum_samples = integrate_trajectory(
        vacuum_initial,
        vacuum_params,
        method="rk4",
        dt=0.05,
        max_time=0.2,
        terminal_height=None,
    )
    cases.append({
        "name": "vacuum",
        "operation": "trajectory",
        "initialState": _list(vacuum_initial),
        "params": {"enableDrag": False, "enableMagnus": False},
        "options": {"method": "rk4", "dt": 0.05, "maxTime": 0.2, "terminalHeight": None},
        "expectedSamples": [
            {"time": float(sample.time), "state": _list(sample.state)}
            for sample in vacuum_samples
        ],
    })

    state = launch_state((0, 0, 2), (12, 2, 5), (0, 0, 0))
    cases.append({
        "name": "drag_only",
        "operation": "derivatives",
        "state": _list(state),
        "params": {"enableMagnus": False},
        "expected": _list(derivatives(state, FlightParameters(enable_magnus=False))),
    })

    state = launch_state((0, 0, 2), (10, 0, 0), (0, -100, 0))
    cases.append({
        "name": "backspin",
        "operation": "derivatives",
        "state": _list(state),
        "params": {"enableDrag": False},
        "expected": _list(derivatives(state, FlightParameters(enable_drag=False))),
    })

    state = launch_state((0, 0, 2), (10, 0, 0), (0, 0, 100))
    cases.append({
        "name": "sidespin",
        "operation": "derivatives",
        "state": _list(state),
        "params": {"enableDrag": False},
        "expected": _list(derivatives(state, FlightParameters(enable_drag=False))),
    })

    state = launch_state((0, 0, 2), (10, 0, 0), (0, 0, 0))
    cases.append({
        "name": "matching_wind",
        "operation": "derivatives",
        "state": _list(state),
        "params": {"wind": [10.0, 0.0, 0.0]},
        "expected": _list(derivatives(state, FlightParameters(wind=(10, 0, 0)))),
    })

    launched = launch_state((1, 2, 3), (10, 0, 5), (0, -100, 20), (1, 2, 0))
    cases.append({
        "name": "robot_velocity",
        "operation": "launch",
        "position": [1.0, 2.0, 3.0],
        "muzzleVelocity": [10.0, 0.0, 5.0],
        "spin": [0.0, -100.0, 20.0],
        "robotVelocity": [1.0, 2.0, 0.0],
        "expected": _list(launched),
    })

    state = launch_state((0, 0, 2), (8, 1, 4), (0, -200, 0))
    spin_params = FlightParameters(
        enable_drag=False,
        enable_magnus=False,
        spin_decay_time_constant=2.0,
    )
    cases.append({
        "name": "spin_decay",
        "operation": "rk4Step",
        "state": _list(state),
        "params": {
            "enableDrag": False,
            "enableMagnus": False,
            "spinDecayTimeConstant": 2.0,
        },
        "dt": 0.1,
        "expected": _list(rk4_step(state, spin_params, 0.1)),
    })

    initial = launch_state((0, 0, 2), (12, 3, 8), (0, -100, 20))
    rk45_params = FlightParameters(wind=(1, -0.5, 0))
    rk45_samples = integrate_trajectory(
        initial,
        rk45_params,
        method="rk45",
        dt=0.2,
        max_step=0.2,
        max_time=0.5,
        terminal_height=None,
        rtol=1e-7,
        atol=1e-9,
    )
    cases.append({
        "name": "rk45",
        "operation": "trajectoryFinal",
        "initialState": _list(initial),
        "params": {"wind": [1.0, -0.5, 0.0]},
        "options": {
            "method": "rk45",
            "dt": 0.2,
            "maxStep": 0.2,
            "maxTime": 0.5,
            "terminalHeight": None,
            "rtol": 1e-7,
            "atol": 1e-9,
        },
        "expectedFinal": _list(rk45_samples[-1].state),
    })

    assert [case["name"] for case in cases] == CASE_NAMES
    return {"schema": "physics3d-golden-v1", "cases": cases}


def main():
    output = Path("tests/fixtures/physics3d_golden.json")
    output.write_text(json.dumps(build_fixture(), indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
