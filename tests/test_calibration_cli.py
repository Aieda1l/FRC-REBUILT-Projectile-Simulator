import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from api.physics3d import FlightParameters, integrate_trajectory, launch_state


ROOT = Path(__file__).resolve().parents[1]


def recorded_shot(shot_id, speed, spin):
    params = FlightParameters(
        drag_coefficient=0.36,
        lift_coefficient=0.0,
        enable_magnus=False,
    )
    initial = launch_state((0, 0, 1), (speed, 0, 4), (0, -spin, 0))
    samples = integrate_trajectory(
        initial,
        params,
        dt=0.002,
        max_time=0.2,
        terminal_height=None,
    )
    return {
        "id": shot_id,
        "position": [0, 0, 1],
        "muzzleVelocity": [speed, 0, 4],
        "spin": [0, -spin, 0],
        "observations": [
            {"time": 0.1, "position": samples[50].state[:3].tolist()},
            {"time": 0.2, "position": samples[100].state[:3].tolist()},
        ],
    }


def run_cli(shots, *extra):
    tempdir = tempfile.TemporaryDirectory()
    base = Path(tempdir.name)
    input_path = base / "shots.json"
    output_path = base / "profile.json"
    input_path.write_text(json.dumps({"shots": shots}))
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/calibrate_fuel.py",
            str(input_path),
            "--output",
            str(output_path),
            *extra,
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    profile = json.loads(output_path.read_text()) if output_path.exists() else None
    return tempdir, completed, profile


class CalibrationCliTests(unittest.TestCase):
    def test_cli_reports_low_spin_selection_and_uses_zero_lift_fallback(self):
        tempdir, completed, profile = run_cli(
            [
                recorded_shot("drag", 10, 0),
                recorded_shot("spin", 10, 20),
            ],
            "--validation-fraction",
            "0",
        )
        self.addCleanup(tempdir.cleanup)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn(
            "Calibration selection: drag=1 spinning=1 max_drag_spin_parameter=0.05",
            completed.stdout,
        )
        self.assertIn("Lift fitting skipped", completed.stdout)
        self.assertEqual(
            profile["liftModel"],
            {"kind": "legacy-spin-cap", "maxCoefficient": 0.0, "saturationSpin": 0.5},
        )

    def test_cli_fails_when_training_subset_has_no_low_spin_shots(self):
        tempdir, completed, profile = run_cli(
            [
                recorded_shot("high", 10, 20),
                recorded_shot("low-held-out", 10, 0),
            ],
            "--validation-fraction",
            "0.5",
            "--seed",
            "2026",
        )
        self.addCleanup(tempdir.cleanup)
        self.assertNotEqual(completed.returncode, 0)
        self.assertIsNone(profile)
        combined = completed.stdout + completed.stderr
        self.assertIn("0.05", combined)
        self.assertIn("low-spin", combined)
        self.assertIn("--drag-max-spin-parameter", combined)

    def test_cli_rejects_negative_drag_spin_threshold(self):
        tempdir, completed, _ = run_cli(
            [recorded_shot("drag", 10, 0)],
            "--validation-fraction",
            "0",
            "--drag-max-spin-parameter",
            "-0.1",
        )
        self.addCleanup(tempdir.cleanup)
        self.assertNotEqual(completed.returncode, 0)


if __name__ == "__main__":
    unittest.main()
