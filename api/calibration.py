"""Calibration profile schema and fitting utilities."""

from __future__ import annotations

import copy
import math
from typing import Any, Dict

from .aerodynamics import normalize_drag_model, normalize_lift_model

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
