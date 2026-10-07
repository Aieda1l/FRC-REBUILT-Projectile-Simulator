"""Calibration profile schema and fitting utilities."""

from __future__ import annotations

import copy
import math
import random
from typing import Any, Dict, Iterable, Sequence

import numpy as np
from scipy.optimize import least_squares

from api.aerodynamics import (
    evaluate_drag_model,
    evaluate_lift_model,
    normalize_drag_model,
    normalize_lift_model,
    reynolds_number,
    spin_parameter,
)
from api.physics3d import (
    FlightParameters,
    aerodynamic_diagnostics,
    integrate_trajectory,
    launch_state,
)
from api.uncertainty import CLASSIFICATIONS, classify_hub_samples

CALIBRATION_SCHEMA = "frc-projectile-calibration-v1"


def _finite(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _positive(value: Any, name: str) -> float:
    number = _finite(value, name)
    if number <= 0:
        raise ValueError(f"{name} must be positive")
    return number


def _range2(value: Any, name: str) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} must contain [min, max]")
    low = _finite(value[0], f"{name}[0]")
    high = _finite(value[1], f"{name}[1]")
    if low < 0 or high < low:
        raise ValueError(f"{name} must be non-negative and ordered")
    return [low, high]


def parse_calibration_profile(value: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("calibration profile must be an object")
    if value.get("schema") != CALIBRATION_SCHEMA:
        raise ValueError(f"unsupported calibration schema: {value.get('schema')}")
    name = value.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("calibration profile name must be non-empty")
    if "dragModel" not in value or "liftModel" not in value:
        raise ValueError("calibration profile requires dragModel and liftModel")

    game_piece_source = value.get("gamePiece") or {}
    environment_source = value.get("environment") or {}
    decay = value.get("spinDecayTimeConstant")
    if decay is not None:
        decay = _positive(decay, "spinDecayTimeConstant")

    return {
        "schema": CALIBRATION_SCHEMA,
        "name": name.strip(),
        "createdAt": value.get("createdAt"),
        "gamePiece": {
            **copy.deepcopy(game_piece_source),
            "diameter": _positive(game_piece_source.get("diameter"), "gamePiece.diameter"),
            "massReference": _positive(
                game_piece_source.get("massReference"), "gamePiece.massReference"
            ),
        },
        "environment": {
            **copy.deepcopy(environment_source),
            "dynamicViscosity": _positive(
                environment_source.get("dynamicViscosity", 1.81e-5),
                "environment.dynamicViscosity",
            ),
        },
        "dragModel": normalize_drag_model(value["dragModel"], 0.0),
        "liftModel": normalize_lift_model(value["liftModel"], 0.0),
        "spinDecayTimeConstant": decay,
        "domain": {
            "reynolds": _range2((value.get("domain") or {}).get("reynolds"), "domain.reynolds"),
            "spinParameter": _range2(
                (value.get("domain") or {}).get("spinParameter"),
                "domain.spinParameter",
            ),
        },
        "validation": copy.deepcopy(value.get("validation") or {}),
    }


def split_shots(shots: Sequence[Dict[str, Any]], validation_fraction: float, seed: int):
    if not 0 <= validation_fraction < 1:
        raise ValueError("validation_fraction must be in [0, 1)")
    items = list(shots)
    if not items:
        return [], []
    indices = list(range(len(items)))
    random.Random(int(seed)).shuffle(indices)
    validation_count = int(round(len(items) * validation_fraction))
    if validation_fraction > 0 and len(items) > 1:
        validation_count = max(1, validation_count)
    validation_count = min(validation_count, max(0, len(items) - 1))
    validation_indices = set(indices[:validation_count])
    train = [shot for index, shot in enumerate(items) if index not in validation_indices]
    validation = [shot for index, shot in enumerate(items) if index in validation_indices]
    return train, validation


def _base_kwargs(base_params: Any) -> Dict[str, Any]:
    if isinstance(base_params, FlightParameters):
        return {
            "mass": base_params.mass,
            "radius": base_params.radius,
            "drag_coefficient": base_params.drag_coefficient,
            "lift_coefficient": base_params.lift_coefficient,
            "air_density": base_params.air_density,
            "dynamic_viscosity": base_params.dynamic_viscosity,
            "gravity": base_params.gravity,
            "wind": base_params.wind,
            "enable_drag": base_params.enable_drag,
            "enable_magnus": base_params.enable_magnus,
            "spin_decay_time_constant": base_params.spin_decay_time_constant,
        }
    if not isinstance(base_params, dict):
        raise ValueError("base_params must be a FlightParameters or dict")
    return dict(base_params)


def _shot_initial_conditions(shot: Dict[str, Any], base_params: Any):
    base = _base_kwargs(base_params)
    position = shot.get("position")
    muzzle = shot.get("muzzleVelocity")
    spin = shot.get("spin")
    if position is None or muzzle is None or spin is None:
        raise ValueError("shot requires position, muzzleVelocity, and spin")
    robot = shot.get("robotVelocity", (0.0, 0.0, 0.0))
    wind = shot.get("wind", base.get("wind", (0.0, 0.0, 0.0)))
    radius = float(shot.get("radius", base.get("radius", 0.075)))
    air_density = float(shot.get("airDensity", base.get("air_density", 1.204)))
    viscosity = float(
        shot.get("dynamicViscosity", base.get("dynamic_viscosity", 1.81e-5))
    )
    velocity = np.asarray(muzzle, dtype=float) + np.asarray(robot, dtype=float)
    relative = velocity - np.asarray(wind, dtype=float)
    speed = float(np.linalg.norm(relative))
    omega = np.asarray(spin, dtype=float)
    if speed <= 1e-12:
        perpendicular_spin = 0.0
    else:
        u_hat = relative / speed
        perpendicular_spin = float(
            np.linalg.norm(omega - float(np.dot(omega, u_hat)) * u_hat)
        )
    reynolds = reynolds_number(
        air_density=air_density,
        speed=speed,
        diameter=2.0 * radius,
        dynamic_viscosity=viscosity,
    )
    spin_value = spin_parameter(
        radius=radius,
        perpendicular_spin=perpendicular_spin,
        speed=speed,
    )
    return reynolds, spin_value


def dataset_domain(shots: Sequence[Dict[str, Any]], base_params: Any) -> Dict[str, list[float]]:
    if not shots:
        return {"reynolds": [0.0, 0.0], "spinParameter": [0.0, 0.0]}
    values = [_shot_initial_conditions(shot, base_params) for shot in shots]
    reynolds_values = [value[0] for value in values]
    spin_values = [value[1] for value in values]
    return {
        "reynolds": [min(reynolds_values), max(reynolds_values)],
        "spinParameter": [min(spin_values), max(spin_values)],
    }


def _flight_parameters_for_shot(
    shot: Dict[str, Any],
    base_params: Any,
    *,
    drag_model: Dict[str, Any],
    lift_model: Dict[str, Any],
    spin_decay_time_constant: float | None = None,
    enable_magnus: bool | None = None,
) -> FlightParameters:
    kwargs = _base_kwargs(base_params)
    kwargs.pop("drag_model", None)
    kwargs.pop("lift_model", None)
    if "mass" in shot:
        kwargs["mass"] = float(shot["mass"])
    if "radius" in shot:
        kwargs["radius"] = float(shot["radius"])
    elif "diameter" in shot:
        kwargs["radius"] = float(shot["diameter"]) / 2.0
    if "airDensity" in shot:
        kwargs["air_density"] = float(shot["airDensity"])
    if "dynamicViscosity" in shot:
        kwargs["dynamic_viscosity"] = float(shot["dynamicViscosity"])
    kwargs["wind"] = tuple(shot.get("wind", kwargs.get("wind", (0.0, 0.0, 0.0))))
    kwargs["drag_model"] = drag_model
    kwargs["lift_model"] = lift_model
    kwargs["spin_decay_time_constant"] = spin_decay_time_constant
    if enable_magnus is not None:
        kwargs["enable_magnus"] = enable_magnus
    return FlightParameters(**kwargs)


def _simulate_observations(
    shot: Dict[str, Any],
    base_params: Any,
    *,
    drag_model: Dict[str, Any],
    lift_model: Dict[str, Any],
    spin_decay_time_constant: float | None = None,
    enable_magnus: bool | None = None,
):
    observations = shot.get("observations") or []
    if not observations:
        raise ValueError(f"shot {shot.get('id', '<unknown>')} has no observations")
    max_time = max(float(observation["time"]) for observation in observations)
    params = _flight_parameters_for_shot(
        shot,
        base_params,
        drag_model=drag_model,
        lift_model=lift_model,
        spin_decay_time_constant=spin_decay_time_constant,
        enable_magnus=enable_magnus,
    )
    initial = launch_state(
        shot["position"],
        shot["muzzleVelocity"],
        shot["spin"],
        shot.get("robotVelocity", (0.0, 0.0, 0.0)),
    )
    samples = integrate_trajectory(
        initial,
        params,
        dt=min(0.002, max(0.0005, max_time / 100.0 if max_time else 0.002)),
        max_time=max_time,
        terminal_height=None,
    )
    return params, samples


def _interpolated_state(samples, time: float) -> np.ndarray:
    time = float(time)
    if time <= samples[0].time:
        return samples[0].state
    if time >= samples[-1].time:
        return samples[-1].state
    for left, right in zip(samples, samples[1:]):
        if left.time <= time <= right.time:
            span = right.time - left.time
            if span <= 1e-15:
                return left.state
            alpha = (time - left.time) / span
            return left.state + alpha * (right.state - left.state)
    return samples[-1].state


def _trajectory_residuals(
    shots: Sequence[Dict[str, Any]],
    base_params: Any,
    drag_model: Dict[str, Any],
    lift_model: Dict[str, Any],
    *,
    enable_magnus: bool | None = None,
) -> np.ndarray:
    residuals: list[float] = []
    for shot in shots:
        _, samples = _simulate_observations(
            shot,
            base_params,
            drag_model=drag_model,
            lift_model=lift_model,
            enable_magnus=enable_magnus,
        )
        for observation in shot["observations"]:
            predicted = _interpolated_state(samples, observation["time"])[:3]
            observed = np.asarray(observation["position"], dtype=float)
            if observed.shape != (3,) or not np.all(np.isfinite(observed)):
                raise ValueError("observation position must contain three finite values")
            weight = float(observation.get("weight", 1.0))
            residuals.extend(((predicted - observed) * weight).tolist())
    return np.asarray(residuals, dtype=float)


def fit_drag_model(
    shots: Sequence[Dict[str, Any]],
    base_params: Any,
    model_kind: str = "constant",
) -> Dict[str, Any]:
    shots = list(shots)
    if not shots:
        raise ValueError("at least one shot is required for drag fitting")
    zero_lift = {"kind": "legacy-spin-cap", "maxCoefficient": 0.0, "saturationSpin": 0.5}

    if model_kind == "constant":
        def residual(x):
            return _trajectory_residuals(
                shots,
                base_params,
                {"kind": "constant", "coefficient": float(x[0])},
                zero_lift,
                enable_magnus=False,
            )

        result = least_squares(residual, x0=[0.47], bounds=([0.0], [3.0]))
        return {"kind": "constant", "coefficient": float(result.x[0])}

    if model_kind == "table1d":
        reynolds_values = sorted({
            round(_shot_initial_conditions(shot, base_params)[0], 9)
            for shot in shots
        })
        if len(reynolds_values) < 3:
            raise ValueError("table1d drag fitting requires at least three distinct Reynolds conditions")
        axes = [
            float(reynolds_values[0]),
            float(reynolds_values[len(reynolds_values) // 2]),
            float(reynolds_values[-1]),
        ]

        def residual(x):
            model = {
                "kind": "table1d",
                "reynolds": axes,
                "coefficients": [float(value) for value in x],
            }
            base_residual = _trajectory_residuals(
                shots, base_params, model, zero_lift, enable_magnus=False
            )
            smooth = 0.05 * (x[2:] - 2 * x[1:-1] + x[:-2])
            return np.concatenate((base_residual, smooth))

        result = least_squares(
            residual,
            x0=np.full(3, 0.47),
            bounds=(np.zeros(3), np.full(3, 3.0)),
        )
        return normalize_drag_model({
            "kind": "table1d",
            "reynolds": axes,
            "coefficients": result.x.tolist(),
        }, 0.47)

    raise ValueError("model_kind must be 'constant' or 'table1d'")


def fit_lift_model(
    shots: Sequence[Dict[str, Any]],
    base_params: Any,
    drag_model: Dict[str, Any],
    model_kind: str = "table1d",
) -> Dict[str, Any]:
    shots = list(shots)
    if len(shots) < 2:
        raise ValueError("at least two spinning shots are required for lift fitting")
    conditions = [_shot_initial_conditions(shot, base_params) for shot in shots]
    spin_values = sorted({round(value[1], 9) for value in conditions})
    if len(spin_values) < 2:
        raise ValueError("lift fitting requires at least two distinct spin parameters")

    if model_kind == "table1d":
        axes = (
            [float(spin_values[0]), float(spin_values[-1])]
            if len(spin_values) == 2
            else [
                float(spin_values[0]),
                float(spin_values[len(spin_values) // 2]),
                float(spin_values[-1]),
            ]
        )
        count = len(axes)

        def residual(x):
            model = {
                "kind": "table1d",
                "spinParameters": axes,
                "coefficients": [float(value) for value in x],
            }
            base_residual = _trajectory_residuals(shots, base_params, drag_model, model)
            if count < 3:
                return base_residual
            smooth = 0.05 * (x[2:] - 2 * x[1:-1] + x[:-2])
            return np.concatenate((base_residual, smooth))

        result = least_squares(
            residual,
            x0=np.full(count, 0.15),
            bounds=(np.zeros(count), np.full(count, 3.0)),
        )
        return normalize_lift_model({
            "kind": "table1d",
            "spinParameters": axes,
            "coefficients": result.x.tolist(),
        }, 0.25)

    if model_kind == "table2d":
        reynolds_values = sorted({round(value[0], 9) for value in conditions})
        if len(shots) < 4 or len(reynolds_values) < 2:
            raise ValueError(
                "table2d lift fitting requires at least four shots spanning two Reynolds conditions"
            )
        r_axis = [float(reynolds_values[0]), float(reynolds_values[-1])]
        s_axis = [float(spin_values[0]), float(spin_values[-1])]

        def residual(x):
            model = {
                "kind": "table2d",
                "reynolds": r_axis,
                "spinParameters": s_axis,
                "coefficients": [
                    [float(x[0]), float(x[1])],
                    [float(x[2]), float(x[3])],
                ],
            }
            return _trajectory_residuals(shots, base_params, drag_model, model)

        result = least_squares(
            residual,
            x0=np.full(4, 0.15),
            bounds=(np.zeros(4), np.full(4, 3.0)),
        )
        return normalize_lift_model({
            "kind": "table2d",
            "reynolds": r_axis,
            "spinParameters": s_axis,
            "coefficients": [
                [float(result.x[0]), float(result.x[1])],
                [float(result.x[2]), float(result.x[3])],
            ],
        }, 0.25)

    raise ValueError("model_kind must be 'table1d' or 'table2d'")


def fit_spin_decay(
    shots: Sequence[Dict[str, Any]],
    base_params: Any,
) -> float | None:
    pairs: list[tuple[float, float, float]] = []
    for shot in shots:
        initial = float(np.linalg.norm(np.asarray(shot.get("spin", (0, 0, 0)), dtype=float)))
        if initial <= 0:
            continue
        for observation in shot.get("observations") or []:
            if "spin" not in observation:
                continue
            measured = float(np.linalg.norm(np.asarray(observation["spin"], dtype=float)))
            time = float(observation["time"])
            if time > 0 and measured > 0:
                pairs.append((time, initial, measured))
    if len(pairs) < 2:
        return None

    def residual(x):
        tau = float(x[0])
        return np.asarray([
            initial * math.exp(-time / tau) - measured
            for time, initial, measured in pairs
        ])

    result = least_squares(residual, x0=[2.0], bounds=([0.01], [100.0]))
    return float(result.x[0])


def validate_profile(
    shots: Sequence[Dict[str, Any]],
    profile: Dict[str, Any],
    base_params: Any,
) -> Dict[str, Any]:
    normalized = parse_calibration_profile(profile)
    errors: list[np.ndarray] = []
    vertical_errors: list[float] = []
    downrange_errors: list[float] = []
    hub_plane_errors: list[float] = []
    entry_angle_errors: list[float] = []
    clean_entry_confusion = {
        "truePositive": 0,
        "trueNegative": 0,
        "falsePositive": 0,
        "falseNegative": 0,
    }
    labeled_hub_results = 0
    reynolds_values: list[float] = []
    spin_values: list[float] = []
    clamped = 0
    diagnostic_count = 0

    for shot in shots:
        params, samples = _simulate_observations(
            shot,
            {
                **_base_kwargs(base_params),
                "dynamic_viscosity": normalized["environment"]["dynamicViscosity"],
            },
            drag_model=normalized["dragModel"],
            lift_model=normalized["liftModel"],
            spin_decay_time_constant=normalized["spinDecayTimeConstant"],
        )
        for observation in shot["observations"]:
            state = _interpolated_state(samples, observation["time"])
            predicted = state[:3]
            observed = np.asarray(observation["position"], dtype=float)
            delta = predicted - observed
            errors.append(delta)
            downrange_errors.append(float(delta[0]))
            vertical_errors.append(float(delta[2]))
            diagnostics = aerodynamic_diagnostics(state, params)
            reynolds_values.append(float(diagnostics["reynolds"]))
            spin_values.append(float(diagnostics["spinParameter"]))
            diagnostic_count += 1
            outside = (
                diagnostics["reynolds"] < normalized["domain"]["reynolds"][0]
                or diagnostics["reynolds"] > normalized["domain"]["reynolds"][1]
                or diagnostics["spinParameter"] < normalized["domain"]["spinParameter"][0]
                or diagnostics["spinParameter"] > normalized["domain"]["spinParameter"][1]
                or diagnostics["dragClamped"]
                or diagnostics["liftClamped"]
            )
            if outside:
                clamped += 1
            if observation.get("hubPlane"):
                hub_plane_errors.append(float(np.linalg.norm(delta)))

        observed_hub_result = shot.get("observedHubResult")
        observed_entry_angle = shot.get("observedEntryAngle")
        if observed_hub_result is not None or observed_entry_angle is not None:
            interaction = classify_hub_samples(
                samples,
                float(shot.get("targetX", 0.0)),
                float(shot.get("targetLateralY", 0.0)),
                params.radius,
            )
            if observed_hub_result is not None:
                if observed_hub_result not in CLASSIFICATIONS:
                    raise ValueError(
                        "observedHubResult must be clean-entry, rim-collision, "
                        "funnel-collision, or miss"
                    )
                labeled_hub_results += 1
                predicted_clean = interaction["classification"] == "clean-entry"
                observed_clean = observed_hub_result == "clean-entry"
                if predicted_clean and observed_clean:
                    clean_entry_confusion["truePositive"] += 1
                elif predicted_clean and not observed_clean:
                    clean_entry_confusion["falsePositive"] += 1
                elif not predicted_clean and observed_clean:
                    clean_entry_confusion["falseNegative"] += 1
                else:
                    clean_entry_confusion["trueNegative"] += 1

            if observed_entry_angle is not None:
                top_crossing = interaction.get("topCrossing")
                if top_crossing is not None:
                    state = top_crossing["state"]
                    predicted_angle = math.degrees(
                        math.atan2(
                            float(state[5]),
                            math.hypot(float(state[3]), float(state[4])),
                        )
                    )
                    measured_angle = _finite(observed_entry_angle, "observedEntryAngle")
                    entry_angle_errors.append(predicted_angle - measured_angle)

    if not errors:
        raise ValueError("validation requires at least one observation")
    matrix = np.asarray(errors)
    squared_norms = np.sum(matrix ** 2, axis=1)
    metrics = {
        "rms3d": float(np.sqrt(np.mean(squared_norms))),
        "rmsVertical": float(np.sqrt(np.mean(np.square(vertical_errors)))),
        "rmsDownrange": float(np.sqrt(np.mean(np.square(downrange_errors)))),
        "hubPlaneRms": (
            float(np.sqrt(np.mean(np.square(hub_plane_errors))))
            if hub_plane_errors else None
        ),
        "entryAngleRms": (
            float(np.sqrt(np.mean(np.square(entry_angle_errors))))
            if entry_angle_errors else None
        ),
        "cleanEntryConfusion": (
            clean_entry_confusion if labeled_hub_results else None
        ),
        "reynoldsRange": [min(reynolds_values), max(reynolds_values)],
        "spinParameterRange": [min(spin_values), max(spin_values)],
        "clampedFraction": clamped / diagnostic_count if diagnostic_count else 0.0,
    }
    return metrics
