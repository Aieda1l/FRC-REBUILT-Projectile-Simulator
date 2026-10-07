"""Pure aerodynamic coefficient models shared by calibration and flight dynamics."""

from __future__ import annotations

import math
from typing import Any, Dict, Sequence

DEFAULT_DYNAMIC_VISCOSITY = 1.81e-5
_EPS = 1e-12


def _finite(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _non_negative(value: float, name: str) -> float:
    value = _finite(value, name)
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _positive(value: float, name: str) -> float:
    value = _finite(value, name)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _axis(values: Sequence[float], name: str) -> list[float]:
    if not isinstance(values, (list, tuple)) or len(values) < 2:
        raise ValueError(f"{name} must contain at least two values")
    out = [_non_negative(value, f"{name}[{index}]") for index, value in enumerate(values)]
    if any(out[index] <= out[index - 1] for index in range(1, len(out))):
        raise ValueError(f"{name} must be strictly increasing")
    return out


def _coefficients(values: Sequence[float], expected_length: int, name: str = "coefficients") -> list[float]:
    if not isinstance(values, (list, tuple)) or len(values) != expected_length:
        raise ValueError(f"{name} must contain exactly {expected_length} values")
    return [_non_negative(value, f"{name}[{index}]") for index, value in enumerate(values)]


def reynolds_number(*, air_density: float, speed: float, diameter: float,
                    dynamic_viscosity: float = DEFAULT_DYNAMIC_VISCOSITY) -> float:
    return (
        _non_negative(air_density, "air_density")
        * _non_negative(speed, "speed")
        * _positive(diameter, "diameter")
        / _positive(dynamic_viscosity, "dynamic_viscosity")
    )


def spin_parameter(*, radius: float, perpendicular_spin: float, speed: float) -> float:
    radius = _positive(radius, "radius")
    omega = abs(_finite(perpendicular_spin, "perpendicular_spin"))
    speed = _non_negative(speed, "speed")
    if speed <= _EPS:
        return 0.0
    return radius * omega / speed


def normalize_drag_model(model: Dict[str, Any] | None, fallback_coefficient: float) -> Dict[str, Any]:
    fallback = _non_negative(fallback_coefficient, "fallback_coefficient")
    source = model or {"kind": "constant", "coefficient": fallback}
    kind = source.get("kind")
    if kind == "constant":
        return {"kind": kind, "coefficient": _non_negative(source.get("coefficient"), "coefficient")}
    if kind == "table1d":
        reynolds = _axis(source.get("reynolds"), "reynolds")
        return {
            "kind": kind,
            "reynolds": reynolds,
            "coefficients": _coefficients(source.get("coefficients"), len(reynolds)),
        }
    raise ValueError(f"unknown drag model kind: {kind}")


def normalize_lift_model(model: Dict[str, Any] | None, fallback_coefficient: float) -> Dict[str, Any]:
    fallback = _non_negative(fallback_coefficient, "fallback_coefficient")
    source = model or {
        "kind": "legacy-spin-cap",
        "maxCoefficient": fallback,
        "saturationSpin": 0.5,
    }
    kind = source.get("kind")
    if kind == "legacy-spin-cap":
        return {
            "kind": kind,
            "maxCoefficient": _non_negative(source.get("maxCoefficient"), "maxCoefficient"),
            "saturationSpin": _positive(source.get("saturationSpin", 0.5), "saturationSpin"),
        }
    if kind == "table1d":
        spin_parameters = _axis(source.get("spinParameters"), "spinParameters")
        return {
            "kind": kind,
            "spinParameters": spin_parameters,
            "coefficients": _coefficients(source.get("coefficients"), len(spin_parameters)),
        }
    if kind == "table2d":
        reynolds = _axis(source.get("reynolds"), "reynolds")
        spin_parameters = _axis(source.get("spinParameters"), "spinParameters")
        matrix = source.get("coefficients")
        if not isinstance(matrix, (list, tuple)) or len(matrix) != len(reynolds):
            raise ValueError(f"coefficients must contain exactly {len(reynolds)} rows")
        return {
            "kind": kind,
            "reynolds": reynolds,
            "spinParameters": spin_parameters,
            "coefficients": [
                _coefficients(row, len(spin_parameters), f"coefficients[{index}]")
                for index, row in enumerate(matrix)
            ],
        }
    raise ValueError(f"unknown lift model kind: {kind}")


def _bracket(values: Sequence[float], query: float) -> tuple[int, int, float, bool]:
    query = _non_negative(query, "query")
    if query < values[0]:
        return 0, 0, 0.0, True
    last = len(values) - 1
    if query > values[last]:
        return last, last, 0.0, True
    if query == values[0]:
        return 0, 0, 0.0, False
    if query == values[last]:
        return last, last, 0.0, False
    for index in range(last):
        if values[index] <= query <= values[index + 1]:
            fraction = (query - values[index]) / (values[index + 1] - values[index])
            return index, index + 1, fraction, False
    raise ValueError("query could not be bracketed")


def _linear(values: Sequence[float], outputs: Sequence[float], query: float) -> tuple[float, bool]:
    low, high, fraction, clamped = _bracket(values, query)
    if low == high:
        return outputs[low], clamped
    return outputs[low] + fraction * (outputs[high] - outputs[low]), clamped


def evaluate_drag_model(model: Dict[str, Any], reynolds: float) -> Dict[str, Any]:
    if model["kind"] == "constant":
        return {"coefficient": model["coefficient"], "clamped": False}
    value, clamped = _linear(model["reynolds"], model["coefficients"], reynolds)
    return {"coefficient": value, "clamped": clamped}


def evaluate_lift_model(model: Dict[str, Any], reynolds: float, spin_parameter_value: float) -> Dict[str, Any]:
    spin_parameter_value = _non_negative(spin_parameter_value, "spin_parameter")
    if model["kind"] == "legacy-spin-cap":
        return {
            "coefficient": model["maxCoefficient"]
            * min(spin_parameter_value / model["saturationSpin"], 1.0),
            "clamped": False,
        }
    if model["kind"] == "table1d":
        value, clamped = _linear(
            model["spinParameters"], model["coefficients"], spin_parameter_value
        )
        return {"coefficient": value, "clamped": clamped}

    r_low, r_high, r_fraction, r_clamped = _bracket(model["reynolds"], reynolds)
    s_low, s_high, s_fraction, s_clamped = _bracket(
        model["spinParameters"], spin_parameter_value
    )

    def row_value(row: int) -> float:
        if s_low == s_high:
            return model["coefficients"][row][s_low]
        a = model["coefficients"][row][s_low]
        b = model["coefficients"][row][s_high]
        return a + s_fraction * (b - a)

    low_value = row_value(r_low)
    value = (
        low_value
        if r_low == r_high
        else low_value + r_fraction * (row_value(r_high) - low_value)
    )
    return {"coefficient": value, "clamped": r_clamped or s_clamped}
