from typing import Any, Dict, Literal, Optional, Tuple

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .physics3d import (
    FlightParameters,
    IntegrationError,
    integrate_trajectory,
    launch_state,
)
from .trajectory_simulator import (
    EnvironmentConditions,
    GamePiece,
    GamePieceProperties,
    LaunchParameters,
    PhysicsEngine,
    TrajectorySimulator,
)

app = FastAPI()

Vector3 = Tuple[float, float, float]


class SimRequest(BaseModel):
    velocity: float
    angle: float
    spin_rate: float
    launch_x: float
    launch_y: float


class Sim3DRequest(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)

    position: Vector3
    muzzle_velocity: Vector3
    spin: Vector3
    robot_velocity: Vector3 = (0.0, 0.0, 0.0)
    wind: Vector3 = (0.0, 0.0, 0.0)
    method: Literal["rk4", "rk45"] = "rk45"

    dt: float = Field(default=0.02, gt=0)
    max_time: float = Field(default=5.0, ge=0)
    rtol: float = Field(default=1e-6, gt=0)
    atol: float = Field(default=1e-9, gt=0)
    min_step: float = Field(default=1e-5, gt=0)
    max_step: float = Field(default=0.05, gt=0)

    mass: float = Field(default=0.215, gt=0)
    radius: float = Field(default=0.075, gt=0)
    drag_coefficient: float = Field(default=0.47, ge=0)
    lift_coefficient: float = Field(default=0.25, ge=0)
    air_density: float = Field(default=1.204, ge=0)
    dynamic_viscosity: float = Field(default=1.81e-5, gt=0)
    drag_model: Optional[Dict[str, Any]] = None
    lift_model: Optional[Dict[str, Any]] = None
    gravity: float = Field(default=9.81, ge=0)
    spin_decay_time_constant: Optional[float] = Field(default=None, gt=0)
    enable_drag: bool = True
    enable_magnus: bool = True

    @model_validator(mode="after")
    def validate_step_bounds(self):
        if self.min_step > self.max_step:
            raise ValueError("min_step must be less than or equal to max_step")
        return self


@app.post("/api/simulate")
async def simulate(data: SimRequest):
    piece = GamePieceProperties.from_game_piece(GamePiece.FUEL)
    env = EnvironmentConditions()
    physics = PhysicsEngine(piece, env)
    sim = TrajectorySimulator(physics)

    launch = LaunchParameters(
        position=(data.launch_x, data.launch_y),
        velocity=data.velocity,
        angle=data.angle,
        spin_rate=data.spin_rate,
    )
    result = sim.simulate(launch)

    return {
        "success": True,
        "points": [{"x": p.x, "y": p.y} for p in result.points],
        "hit": result.hit_target,
    }


@app.post("/api/simulate3d")
async def simulate3d(data: Sim3DRequest):
    try:
        params = FlightParameters(
            mass=data.mass,
            radius=data.radius,
            drag_coefficient=data.drag_coefficient,
            lift_coefficient=data.lift_coefficient,
            air_density=data.air_density,
            dynamic_viscosity=data.dynamic_viscosity,
            drag_model=data.drag_model,
            lift_model=data.lift_model,
            gravity=data.gravity,
            wind=data.wind,
            enable_drag=data.enable_drag,
            enable_magnus=data.enable_magnus,
            spin_decay_time_constant=data.spin_decay_time_constant,
        )
        initial = launch_state(
            data.position,
            data.muzzle_velocity,
            data.spin,
            data.robot_velocity,
        )
        samples = integrate_trajectory(
            initial,
            params,
            method=data.method,
            dt=data.dt,
            max_time=data.max_time,
            rtol=data.rtol,
            atol=data.atol,
            min_step=data.min_step,
            max_step=data.max_step,
        )
    except (IntegrationError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    serialized = []
    for sample in samples:
        state = sample.state
        serialized.append({
            "time": float(sample.time),
            "position": [float(value) for value in state[0:3]],
            "velocity": [float(value) for value in state[3:6]],
            "spin": [float(value) for value in state[6:9]],
            "speed": float(np.linalg.norm(state[3:6])),
        })

    final = samples[-1]
    horizontal_delta = final.state[0:2] - initial[0:2]
    return {
        "success": True,
        "method": data.method,
        "samples": serialized,
        "flight_time": float(final.time),
        "range": float(np.linalg.norm(horizontal_delta)),
        "max_height": float(max(sample.state[2] for sample in samples)),
    }
