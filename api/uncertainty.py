"""Seeded uncertainty sampling and conservative Monte Carlo shot evaluation."""

from __future__ import annotations

import copy
import math
from typing import Any, Callable, Dict, Sequence

import numpy as np

from .physics3d import FlightParameters, integrate_trajectory, launch_state

CLASSIFICATIONS = ("clean-entry", "rim-collision", "funnel-collision", "miss")
UINT32 = 2 ** 32
EPS = 1e-12
INCH_TO_METER = 0.0254
TOP_ACROSS_FLATS = 41.727 * INCH_TO_METER
TOP_Z = 72.0 * INCH_TO_METER
BOTTOM_SIDE = 18.92 * INCH_TO_METER
PANEL_HEIGHT = 17.90 * INCH_TO_METER
SQRT3 = math.sqrt(3.0)


def _finite(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def create_seeded_rng(seed: int = 0) -> Callable[[], float]:
    state = int(seed) & 0xFFFFFFFF

    def rng() -> float:
        nonlocal state
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        return state / UINT32

    return rng


def _unit_sample(rng: Callable[[], float]) -> float:
    value = _finite(rng(), "rng result")
    if value < 0 or value >= 1:
        raise ValueError("rng result must be in [0, 1)")
    return value


def sample_distribution(definition: Dict[str, Any], rng: Callable[[], float]) -> float:
    if not isinstance(definition, dict):
        raise ValueError("distribution must be an object")
    kind = definition.get("kind")
    if kind == "fixed":
        return _finite(definition.get("value"), "value")
    if kind == "uniform":
        low = _finite(definition.get("min"), "min")
        high = _finite(definition.get("max"), "max")
        if high < low:
            raise ValueError("max must be greater than or equal to min")
        return low + (high - low) * _unit_sample(rng)
    if kind == "normal":
        mean = _finite(definition.get("mean", 0), "mean")
        sigma = _finite(definition.get("sigma"), "sigma")
        if sigma < 0:
            raise ValueError("sigma must be non-negative")
        low = -math.inf if definition.get("min") is None else _finite(definition["min"], "min")
        high = math.inf if definition.get("max") is None else _finite(definition["max"], "max")
        if high < low:
            raise ValueError("max must be greater than or equal to min")
        if sigma == 0:
            return max(low, min(high, mean))
        candidate = mean
        for _ in range(64):
            u1 = max(_unit_sample(rng), np.finfo(float).eps)
            u2 = _unit_sample(rng)
            z = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
            candidate = mean + sigma * z
            if low <= candidate <= high:
                return candidate
        return max(low, min(high, candidate))
    raise ValueError(f"unknown distribution kind: {kind}")


def _scale_drag_model(model: Dict[str, Any] | None, factor: float):
    if model is None:
        return None
    out = copy.deepcopy(model)
    if out["kind"] == "constant":
        out["coefficient"] *= factor
    elif out["kind"] == "table1d":
        out["coefficients"] = [value * factor for value in out["coefficients"]]
    elif out["kind"] == "table2d":
        out["coefficients"] = [
            [value * factor for value in row] for row in out["coefficients"]
        ]
    else:
        raise ValueError(f"cannot scale drag model kind: {out.get('kind')}")
    return out


def _scale_lift_model(model: Dict[str, Any] | None, factor: float):
    if model is None:
        return None
    out = copy.deepcopy(model)
    if out["kind"] == "legacy-spin-cap":
        out["maxCoefficient"] *= factor
    elif out["kind"] == "table1d":
        out["coefficients"] = [value * factor for value in out["coefficients"]]
    elif out["kind"] == "table2d":
        out["coefficients"] = [
            [value * factor for value in row] for row in out["coefficients"]
        ]
    else:
        raise ValueError(f"cannot scale lift model kind: {out.get('kind')}")
    return out


def _sample_optional(definition, rng, name):
    if definition is None:
        return None
    try:
        return sample_distribution(definition, rng)
    except ValueError as exc:
        raise ValueError(f"{name}: {exc}") from exc


def _perturbed_vector(base, definitions, rng, name):
    values = list(base or (0.0, 0.0, 0.0))
    if len(values) != 3:
        raise ValueError(f"{name} must contain exactly three values")
    if definitions is None:
        return [_finite(value, f"{name}[{index}]") for index, value in enumerate(values)]
    if not isinstance(definitions, (list, tuple)) or len(definitions) != 3:
        raise ValueError(f"{name} uncertainty must contain exactly three distributions")
    return [
        _finite(values[index], f"{name}[{index}]")
        + _sample_optional(definitions[index], rng, f"{name}[{index}]")
        for index in range(3)
    ]


def sample_shot_params(base_params, uncertainty=None, rng=None):
    if not isinstance(base_params, dict):
        raise ValueError("base_params must be an object")
    if uncertainty is None:
        uncertainty = {}
    if not isinstance(uncertainty, dict):
        raise ValueError("uncertainty must be an object")
    rng = rng or create_seeded_rng(0)
    out = copy.deepcopy(base_params)
    for key in ("velocity", "angleDeg", "spinRPM", "mass"):
        delta = _sample_optional(uncertainty.get(key), rng, key)
        if delta is not None:
            out[key] = _finite(out.get(key), key) + delta
    if out["velocity"] <= 0:
        raise ValueError("sampled velocity must be positive")
    if out["mass"] <= 0:
        raise ValueError("sampled mass must be positive")

    drag_factor = _sample_optional(uncertainty.get("dragMultiplier"), rng, "dragMultiplier")
    if drag_factor is not None:
        if drag_factor < 0:
            raise ValueError("dragMultiplier must be non-negative")
        if out.get("dragModel") is not None:
            out["dragModel"] = _scale_drag_model(out["dragModel"], drag_factor)
        if out.get("dragCoeff") is not None:
            out["dragCoeff"] *= drag_factor

    lift_factor = _sample_optional(uncertainty.get("liftMultiplier"), rng, "liftMultiplier")
    if lift_factor is not None:
        if lift_factor < 0:
            raise ValueError("liftMultiplier must be non-negative")
        if out.get("liftModel") is not None:
            out["liftModel"] = _scale_lift_model(out["liftModel"], lift_factor)
        if out.get("liftCoeff") is not None:
            out["liftCoeff"] *= lift_factor

    out["robotVelocity"] = _perturbed_vector(
        out.get("robotVelocity", [0, 0, 0]), uncertainty.get("robotVelocity"), rng, "robotVelocity"
    )
    out["wind"] = _perturbed_vector(
        out.get("wind", [0, 0, 0]), uncertainty.get("wind"), rng, "wind"
    )
    return out


def _geometry(center_x, center_y):
    top_apothem = TOP_ACROSS_FLATS / 2
    bottom_apothem = SQRT3 * BOTTOM_SIDE / 2
    bottom_z = TOP_Z - PANEL_HEIGHT
    slope = (top_apothem - bottom_apothem) / (TOP_Z - bottom_z)
    normals = [(math.cos(i * math.pi / 3), math.sin(i * math.pi / 3)) for i in range(6)]

    def vertices(apothem, z):
        radius = 2 * apothem / SQRT3
        return [
            (
                center_x + radius * math.cos(math.pi / 6 + i * math.pi / 3),
                center_y + radius * math.sin(math.pi / 6 + i * math.pi / 3),
                z,
            )
            for i in range(6)
        ]

    return {
        "center_x": center_x, "center_y": center_y, "top_z": TOP_Z, "bottom_z": bottom_z,
        "top_apothem": top_apothem, "bottom_apothem": bottom_apothem, "slope": slope,
        "normals": normals, "top_vertices": vertices(top_apothem, TOP_Z),
    }


def _interpolate_sample(a, b, alpha):
    alpha = max(0.0, min(1.0, alpha))
    return {
        "time": a.time + alpha * (b.time - a.time),
        "state": a.state + alpha * (b.state - a.state),
    }


def _descending_crossing(samples, height, after_time=-math.inf):
    for a, b in zip(samples, samples[1:]):
        if b.time < after_time - EPS:
            continue
        za, zb = float(a.state[2] - height), float(b.state[2] - height)
        if za > EPS and zb <= EPS and b.state[5] < 0:
            crossing = _interpolate_sample(a, b, za / (za - zb))
            crossing["state"] = crossing["state"].copy()
            crossing["state"][2] = height
            if crossing["time"] + EPS >= after_time:
                return crossing
    return None


def _raw_hex_clearance(x, y, apothem, geometry):
    local_x, local_y = x - geometry["center_x"], y - geometry["center_y"]
    return min(apothem - (nx * local_x + ny * local_y) for nx, ny in geometry["normals"])


def _point_segment_distance(px, py, ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    denom = dx * dx + dy * dy
    if denom <= EPS:
        return math.hypot(px - ax, py - ay)
    projection = ((px - ax) * dx + (py - ay) * dy) / denom
    t = max(0.0, min(1.0, projection))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def _min_boundary_distance(x, y, vertices):
    return min(
        _point_segment_distance(
            x, y, vertices[i][0], vertices[i][1],
            vertices[(i + 1) % len(vertices)][0], vertices[(i + 1) % len(vertices)][1],
        )
        for i in range(len(vertices))
    )


def _panel_clearance(state, normal, geometry, ball_radius):
    x, y, z = state[:3]
    nx, ny = normal
    apothem = geometry["bottom_apothem"] + geometry["slope"] * (z - geometry["bottom_z"])
    numerator = apothem - (nx * (x - geometry["center_x"]) + ny * (y - geometry["center_y"]))
    return numerator / math.sqrt(1 + geometry["slope"] ** 2) - ball_radius


def classify_hub_samples(samples, center_x, center_y, ball_radius):
    geometry = _geometry(center_x, center_y)
    top = _descending_crossing(samples, geometry["top_z"])
    if top is None:
        return {"classification": "miss", "clearanceMargin": -math.inf, "topCrossing": None}
    x, y = top["state"][0], top["state"][1]
    top_clearance = _raw_hex_clearance(x, y, geometry["top_apothem"], geometry) - ball_radius
    if top_clearance < -EPS:
        distance = _min_boundary_distance(x, y, geometry["top_vertices"])
        if distance <= ball_radius + EPS:
            return {"classification": "rim-collision", "clearanceMargin": top_clearance, "topCrossing": top}
        return {"classification": "miss", "clearanceMargin": top_clearance, "topCrossing": top}

    bottom = _descending_crossing(samples, geometry["bottom_z"], top["time"] + EPS)
    if bottom is None:
        return {"classification": "miss", "clearanceMargin": top_clearance, "topCrossing": top}

    path = [top]
    path.extend(
        {"time": sample.time, "state": sample.state}
        for sample in samples
        if top["time"] + EPS < sample.time < bottom["time"] - EPS
    )
    path.append(bottom)
    minimum = top_clearance
    for a, b in zip(path, path[1:]):
        for normal in geometry["normals"]:
            c0 = _panel_clearance(a["state"], normal, geometry, ball_radius)
            c1 = _panel_clearance(b["state"], normal, geometry, ball_radius)
            minimum = min(minimum, c0, c1)
            if c0 < -EPS or (c0 >= -EPS and c1 < -EPS):
                return {"classification": "funnel-collision", "clearanceMargin": min(minimum, -EPS), "topCrossing": top}

    bottom_clearance = _raw_hex_clearance(
        bottom["state"][0], bottom["state"][1], geometry["bottom_apothem"], geometry
    ) - ball_radius
    minimum = min(minimum, bottom_clearance)
    if bottom_clearance < -EPS:
        return {"classification": "funnel-collision", "clearanceMargin": bottom_clearance, "topCrossing": top}
    return {"classification": "clean-entry", "clearanceMargin": minimum, "topCrossing": top}


def _simulate_shot(params, dt):
    angle = math.radians(_finite(params["angleDeg"], "angleDeg"))
    velocity = _finite(params["velocity"], "velocity")
    spin = _finite(params["spinRPM"], "spinRPM") * 2 * math.pi / 60
    initial = launch_state(
        (params["launchX"], 0.0, params["launchY"]),
        (velocity * math.cos(angle), 0.0, velocity * math.sin(angle)),
        (0.0, -spin, 0.0),
        params.get("robotVelocity", (0.0, 0.0, 0.0)),
    )
    flight = FlightParameters(
        mass=params["mass"], radius=params["radius"],
        drag_coefficient=params.get("dragCoeff", 0.47),
        lift_coefficient=params.get("liftCoeff", 0.25),
        air_density=params.get("airDensity", 1.204),
        dynamic_viscosity=params.get("dynamicViscosity", 1.81e-5),
        drag_model=params.get("dragModel"), lift_model=params.get("liftModel"),
        gravity=params.get("gravity", 9.81), wind=tuple(params.get("wind", (0.0, 0.0, 0.0))),
        enable_drag=params.get("enableDrag", True), enable_magnus=params.get("enableMagnus", True),
        enable_buoyancy=params.get("enableBuoyancy", True),
        spin_decay_time_constant=params.get("spinDecayTimeConstant"),
    )
    samples = integrate_trajectory(initial, flight, method="rk4", dt=dt, max_time=5.0)
    interaction = classify_hub_samples(
        samples, params.get("targetX", 0.0), params.get("targetLateralY", 0.0), params["radius"]
    )
    entry_velocity = entry_angle = None
    top = interaction["topCrossing"]
    if interaction["classification"] == "clean-entry" and top is not None:
        state = top["state"]
        entry_velocity = float(np.linalg.norm(state[3:6]))
        entry_angle = math.degrees(math.atan2(state[5], math.hypot(state[3], state[4])))
    return interaction, entry_velocity, entry_angle


def _percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return ordered[low]
    t = position - low
    return ordered[low] + t * (ordered[high] - ordered[low])


def _summary(values):
    finite_values = [float(value) for value in values if value is not None and math.isfinite(value)]
    return {
        "p10": _percentile(finite_values, 0.10),
        "median": _percentile(finite_values, 0.50),
        "p90": _percentile(finite_values, 0.90),
    }


def evaluate_shot_uncertainty(
    base_params,
    uncertainty=None,
    *,
    sample_count=256,
    seed=2026,
    dt=0.002,
):
    if not isinstance(sample_count, int) or isinstance(sample_count, bool) or not 1 <= sample_count <= 10000:
        raise ValueError("sample_count must be an integer from 1 to 10000")
    dt = _finite(dt, "dt")
    if dt <= 0:
        raise ValueError("dt must be positive")
    uncertainty = uncertainty or {}
    rng = create_seeded_rng(seed)
    counts = {key: 0 for key in CLASSIFICATIONS}
    clearances, entry_velocities, entry_angles = [], [], []
    for _ in range(sample_count):
        sampled = sample_shot_params(base_params, uncertainty, rng)
        interaction, entry_velocity, entry_angle = _simulate_shot(sampled, dt)
        counts[interaction["classification"]] += 1
        if math.isfinite(interaction["clearanceMargin"]):
            clearances.append(interaction["clearanceMargin"])
        if entry_velocity is not None:
            entry_velocities.append(entry_velocity)
        if entry_angle is not None:
            entry_angles.append(entry_angle)
    return {
        "sampleCount": sample_count,
        "seed": int(seed) & 0xFFFFFFFF,
        "counts": counts,
        "probabilities": {key: counts[key] / sample_count for key in CLASSIFICATIONS},
        "clearance": _summary(clearances),
        "entryVelocity": _summary(entry_velocities),
        "entryAngle": _summary(entry_angles),
    }
