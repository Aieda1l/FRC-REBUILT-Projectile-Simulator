#!/usr/bin/env python3
"""Fit a versioned FUEL calibration profile from recorded JSON or planar CSV shots."""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path

from calibration.fitting import (
    dataset_domain,
    fit_drag_model,
    fit_lift_model,
    fit_spin_decay,
    partition_shots_by_spin_parameter,
    split_shots,
    validate_profile,
)


def load_dataset(path: Path):
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text())
        if isinstance(data, list):
            return data, {}
        return data["shots"], data.get("baseParams", {})

    grouped = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            shot_id = row["shot_id"]
            shot = grouped.setdefault(shot_id, {
                "id": shot_id,
                "position": [float(row["launch_x"]), 0.0, float(row["launch_z"])],
                "muzzleVelocity": [float(row["vx"]), 0.0, float(row["vz"])],
                "spin": [0.0, float(row.get("spin_y", 0.0)), 0.0],
                "observations": [],
            })
            shot["observations"].append({
                "time": float(row["time"]),
                "position": [float(row["x"]), 0.0, float(row["z"])],
            })
    return list(grouped.values()), {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--drag-model", choices=("constant", "table1d"), default="constant")
    parser.add_argument("--lift-model", choices=("table1d", "table2d"), default="table1d")
    parser.add_argument("--drag-max-spin-parameter", type=float, default=0.05)
    args = parser.parse_args()
    if (
        not math.isfinite(args.drag_max_spin_parameter)
        or args.drag_max_spin_parameter < 0
    ):
        parser.error("--drag-max-spin-parameter must be finite and non-negative")

    shots, base = load_dataset(args.input)
    base = {
        "mass": 0.215,
        "radius": 0.075,
        "air_density": 1.204,
        "gravity": 9.81,
        "dynamic_viscosity": 1.81e-5,
        **base,
    }
    train, validation = split_shots(shots, args.validation_fraction, args.seed)
    drag_shots, spinning = partition_shots_by_spin_parameter(
        train,
        base,
        args.drag_max_spin_parameter,
    )
    print(
        "Calibration selection: "
        f"drag={len(drag_shots)} "
        f"spinning={len(spinning)} "
        f"max_drag_spin_parameter={args.drag_max_spin_parameter:g}"
    )
    if not drag_shots:
        parser.error(
            "no low-spin training shots at or below "
            f"--drag-max-spin-parameter {args.drag_max_spin_parameter:g}; "
            "collect low-spin data or explicitly raise --drag-max-spin-parameter"
        )

    drag = fit_drag_model(drag_shots, base, args.drag_model)
    if len(spinning) >= 2:
        lift = fit_lift_model(spinning, base, drag, args.lift_model)
    else:
        print("Lift fitting skipped: fewer than two spinning training shots.")
        lift = {"kind": "legacy-spin-cap", "maxCoefficient": 0.0, "saturationSpin": 0.5}
    decay = fit_spin_decay(train, base)
    domain = dataset_domain(train, base)
    profile = {
        "schema": "frc-projectile-calibration-v1",
        "name": args.input.stem,
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "gamePiece": {
            "diameter": 2 * float(base["radius"]),
            "massReference": float(base["mass"]),
        },
        "environment": {"dynamicViscosity": float(base["dynamic_viscosity"])},
        "dragModel": drag,
        "liftModel": lift,
        "spinDecayTimeConstant": decay,
        "domain": domain,
        "validation": {},
    }
    if validation:
        profile["validation"] = validate_profile(validation, profile, base)
    args.output.write_text(json.dumps(profile, indent=2) + "\n")


if __name__ == "__main__":
    main()
