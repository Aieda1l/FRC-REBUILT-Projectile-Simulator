"""Canonical 3-D projectile dynamics for the FRC trajectory simulator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .aerodynamics import (
    DEFAULT_DYNAMIC_VISCOSITY,
    evaluate_drag_model,
    evaluate_lift_model,
    normalize_drag_model,
    normalize_lift_model,
    reynolds_number,
    spin_parameter as dimensionless_spin_parameter,
)

_EPS = 1e-12


@dataclass(frozen=True)
class FlightSample:
    time: float
    state: np.ndarray


@dataclass
class FlightParameters:
    mass: float = 0.215
    radius: float = 0.075
    drag_coefficient: float = 0.47
    lift_coefficient: float = 0.25
    air_density: float = 1.204
    dynamic_viscosity: float = DEFAULT_DYNAMIC_VISCOSITY
    drag_model: Optional[Dict[str, Any]] = None
    lift_model: Optional[Dict[str, Any]] = None
    gravity: float = 9.81
    wind: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    enable_drag: bool = True
    enable_magnus: bool = True
    spin_decay_time_constant: Optional[float] = None

    def __post_init__(self) -> None:
        if not np.isfinite(self.mass) or self.mass <= 0:
            raise ValueError("mass must be finite and positive")
        if not np.isfinite(self.radius) or self.radius <= 0:
            raise ValueError("radius must be finite and positive")
        for name in ("drag_coefficient", "lift_coefficient", "air_density", "gravity"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not np.isfinite(self.dynamic_viscosity) or self.dynamic_viscosity <= 0:
            raise ValueError("dynamic_viscosity must be finite and positive")
        self.drag_model = normalize_drag_model(self.drag_model, self.drag_coefficient)
        self.lift_model = normalize_lift_model(self.lift_model, self.lift_coefficient)
        self.wind = tuple(_vector3(self.wind, "wind"))
        if self.spin_decay_time_constant is not None:
            if (not np.isfinite(self.spin_decay_time_constant)
                    or self.spin_decay_time_constant <= 0):
                raise ValueError("spin_decay_time_constant must be finite and positive")


class IntegrationError(RuntimeError):
    """Raised when an adaptive integration request cannot be satisfied."""


def _vector3(values: Sequence[float], name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.shape != (3,) or not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain exactly three finite values")
    return arr


def _state9(state: Sequence[float]) -> np.ndarray:
    arr = np.asarray(state, dtype=np.float64)
    if arr.shape != (9,) or not np.all(np.isfinite(arr)):
        raise ValueError("state must contain exactly nine finite values")
    return arr


def launch_state(
    position: Sequence[float],
    muzzle_velocity: Sequence[float],
    spin: Sequence[float],
    robot_velocity: Sequence[float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    position_v = _vector3(position, "position")
    muzzle_v = _vector3(muzzle_velocity, "muzzle_velocity")
    spin_v = _vector3(spin, "spin")
    robot_v = _vector3(robot_velocity, "robot_velocity")
    return np.concatenate((position_v, muzzle_v + robot_v, spin_v)).astype(np.float64)


def _aerodynamic_state(state: np.ndarray, params: FlightParameters) -> dict:
    velocity = state[3:6]
    omega = state[6:9]
    relative_velocity = velocity - np.asarray(params.wind, dtype=np.float64)
    speed = float(np.linalg.norm(relative_velocity))
    if speed <= _EPS:
        drag = evaluate_drag_model(params.drag_model, 0.0)
        lift = evaluate_lift_model(params.lift_model, 0.0, 0.0)
        return {
            "speed": speed,
            "u_hat": np.zeros(3, dtype=np.float64),
            "omega_perp": np.zeros(3, dtype=np.float64),
            "omega_perp_mag": 0.0,
            "diagnostics": {
                "reynolds": 0.0,
                "spinParameter": 0.0,
                "dragCoefficient": drag["coefficient"],
                "liftCoefficient": lift["coefficient"],
                "dragClamped": drag["clamped"],
                "liftClamped": lift["clamped"],
            },
        }

    u_hat = relative_velocity / speed
    omega_perp = omega - float(np.dot(omega, u_hat)) * u_hat
    omega_perp_mag = float(np.linalg.norm(omega_perp))
    reynolds = reynolds_number(
        air_density=params.air_density,
        speed=speed,
        diameter=2.0 * params.radius,
        dynamic_viscosity=params.dynamic_viscosity,
    )
    spin_value = dimensionless_spin_parameter(
        radius=params.radius,
        perpendicular_spin=omega_perp_mag,
        speed=speed,
    )
    drag = evaluate_drag_model(params.drag_model, reynolds)
    lift = evaluate_lift_model(params.lift_model, reynolds, spin_value)
    return {
        "speed": speed,
        "u_hat": u_hat,
        "omega_perp": omega_perp,
        "omega_perp_mag": omega_perp_mag,
        "diagnostics": {
            "reynolds": reynolds,
            "spinParameter": spin_value,
            "dragCoefficient": drag["coefficient"],
            "liftCoefficient": lift["coefficient"],
            "dragClamped": drag["clamped"],
            "liftClamped": lift["clamped"],
        },
    }


def aerodynamic_diagnostics(state: np.ndarray, params: FlightParameters) -> dict:
    return _aerodynamic_state(_state9(state), params)["diagnostics"]


def derivatives(state: np.ndarray, params: FlightParameters) -> np.ndarray:
    y = _state9(state)
    velocity = y[3:6]
    omega = y[6:9]
    acceleration = np.array([0.0, 0.0, -params.gravity], dtype=np.float64)
    aero = _aerodynamic_state(y, params)

    if aero["speed"] > _EPS:
        area = np.pi * params.radius ** 2
        dynamic_area = 0.5 * params.air_density * area * aero["speed"] ** 2
        diagnostics = aero["diagnostics"]

        if params.enable_drag and diagnostics["dragCoefficient"] > 0:
            acceleration += (
                -dynamic_area * diagnostics["dragCoefficient"] * aero["u_hat"] / params.mass
            )

        if (
            params.enable_magnus
            and diagnostics["liftCoefficient"] > 0
            and aero["omega_perp_mag"] > _EPS
        ):
            lift_direction = np.cross(aero["omega_perp"], aero["u_hat"])
            lift_norm = float(np.linalg.norm(lift_direction))
            if lift_norm > _EPS:
                lift_direction /= lift_norm
                acceleration += (
                    dynamic_area
                    * diagnostics["liftCoefficient"]
                    * lift_direction
                    / params.mass
                )

    if params.spin_decay_time_constant is None:
        spin_derivative = np.zeros(3, dtype=np.float64)
    else:
        spin_derivative = -omega / params.spin_decay_time_constant

    return np.concatenate((velocity, acceleration, spin_derivative))


def rk4_step(state: np.ndarray, params: FlightParameters, dt: float) -> np.ndarray:
    y = _state9(state)
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and positive")
    k1 = derivatives(y, params)
    k2 = derivatives(y + 0.5 * dt * k1, params)
    k3 = derivatives(y + 0.5 * dt * k2, params)
    k4 = derivatives(y + dt * k3, params)
    return y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _crosses_height(
    a: np.ndarray,
    b: np.ndarray,
    height: float,
    direction: int,
) -> bool:
    za = float(a[2] - height)
    zb = float(b[2] - height)
    if direction == -1:
        return za > 0.0 and zb <= 0.0
    if direction == 1:
        return za < 0.0 and zb >= 0.0
    return (za > 0.0 and zb <= 0.0) or (za < 0.0 and zb >= 0.0)


def _interpolate_crossing(
    t0: float,
    y0: np.ndarray,
    t1: float,
    y1: np.ndarray,
    height: float,
) -> FlightSample:
    dz = float(y1[2] - y0[2])
    if abs(dz) <= _EPS:
        alpha = 0.0
    else:
        alpha = float((height - y0[2]) / dz)
    alpha = min(1.0, max(0.0, alpha))
    state = y0 + alpha * (y1 - y0)
    state = state.copy()
    state[2] = height
    return FlightSample(t0 + alpha * (t1 - t0), state)


def _rk45_step(
    state: np.ndarray,
    params: FlightParameters,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    y = _state9(state)
    k1 = derivatives(y, params)
    k2 = derivatives(y + dt * (1/5) * k1, params)
    k3 = derivatives(y + dt * ((3/40)*k1 + (9/40)*k2), params)
    k4 = derivatives(y + dt * ((44/45)*k1 + (-56/15)*k2 + (32/9)*k3), params)
    k5 = derivatives(y + dt * ((19372/6561)*k1 + (-25360/2187)*k2 + (64448/6561)*k3 + (-212/729)*k4), params)
    k6 = derivatives(y + dt * ((9017/3168)*k1 + (-355/33)*k2 + (46732/5247)*k3 + (49/176)*k4 + (-5103/18656)*k5), params)
    y5 = y + dt * ((35/384)*k1 + (500/1113)*k3 + (125/192)*k4 + (-2187/6784)*k5 + (11/84)*k6)
    k7 = derivatives(y5, params)
    y4 = y + dt * (
        (5179/57600)*k1
        + (7571/16695)*k3
        + (393/640)*k4
        + (-92097/339200)*k5
        + (187/2100)*k6
        + (1/40)*k7
    )
    return y5, y5 - y4


def integrate_trajectory(
    initial_state: np.ndarray,
    params: FlightParameters,
    method: str = "rk4",
    dt: float = 0.001,
    max_time: float = 5.0,
    *,
    rtol: float = 1e-6,
    atol: float = 1e-9,
    min_step: float = 1e-5,
    max_step: float = 0.05,
    terminal_height: Optional[float] = 0.0,
    terminal_direction: int = -1,
) -> List[FlightSample]:
    y = _state9(initial_state).copy()
    if method not in ("rk4", "rk45"):
        raise ValueError("method must be 'rk4' or 'rk45'")
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and positive")
    if not np.isfinite(max_time) or max_time < 0:
        raise ValueError("max_time must be finite and non-negative")
    if terminal_direction not in (-1, 0, 1):
        raise ValueError("terminal_direction must be -1, 0, or 1")
    if terminal_height is not None and not np.isfinite(terminal_height):
        raise ValueError("terminal_height must be finite or None")
    if not np.isfinite(rtol) or rtol <= 0 or not np.isfinite(atol) or atol <= 0:
        raise ValueError("rtol and atol must be finite and positive")
    if (not np.isfinite(min_step) or min_step <= 0
            or not np.isfinite(max_step) or max_step <= 0
            or min_step > max_step):
        raise ValueError("step bounds must be positive with min_step <= max_step")

    t = 0.0
    samples = [FlightSample(t, y.copy())]
    if method == "rk4":
        while t < max_time - 1e-15:
            step = min(dt, max_time - t)
            y_next = rk4_step(y, params, step)
            t_next = t + step
            if terminal_height is not None and _crosses_height(
                y, y_next, terminal_height, terminal_direction
            ):
                samples.append(_interpolate_crossing(t, y, t_next, y_next, terminal_height))
                break
            samples.append(FlightSample(t_next, y_next.copy()))
            y = y_next
            t = t_next
        return samples

    step = min(max(dt, min_step), max_step)
    while t < max_time - 1e-15:
        remaining = max_time - t
        trial_step = min(step, remaining)
        y_next, error = _rk45_step(y, params, trial_step)
        scale = atol + rtol * np.maximum(np.abs(y), np.abs(y_next))
        error_norm = float(np.sqrt(np.mean((error / scale) ** 2)))

        if error_norm <= 1.0:
            t_next = t + trial_step
            if terminal_height is not None and _crosses_height(
                y, y_next, terminal_height, terminal_direction
            ):
                samples.append(_interpolate_crossing(t, y, t_next, y_next, terminal_height))
                break
            samples.append(FlightSample(t_next, y_next.copy()))
            y = y_next
            t = t_next

        factor = 5.0 if error_norm == 0.0 else 0.9 * error_norm ** (-0.2)
        factor = min(5.0, max(0.2, factor))
        proposed = trial_step * factor

        if error_norm > 1.0 and trial_step <= min_step * (1.0 + 1e-12):
            raise IntegrationError("RK45 tolerance cannot be met at min_step")
        step = min(max_step, max(min_step, proposed))

    return samples
