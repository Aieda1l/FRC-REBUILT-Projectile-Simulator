from fastapi import FastAPI
from pydantic import BaseModel
from .trajectory_simulator import PhysicsEngine, GamePieceProperties, EnvironmentConditions, LaunchParameters, \
    TrajectorySimulator, GamePiece

app = FastAPI()


class SimRequest(BaseModel):
    velocity: float
    angle: float
    spin_rate: float
    launch_x: float
    launch_y: float


@app.post("/api/simulate")
async def simulate(data: SimRequest):
    # Reuse your existing classes
    piece = GamePieceProperties.from_game_piece(GamePiece.CORAL)
    env = EnvironmentConditions()
    physics = PhysicsEngine(piece, env)
    sim = TrajectorySimulator(physics)

    launch = LaunchParameters(
        position=(data.launch_x, data.launch_y),
        velocity=data.velocity,
        angle=data.angle,
        spin_rate=data.spin_rate
    )

    result = sim.simulate(launch)

    # Return what the frontend needs
    return {
        "success": True,
        "points": [{"x": p.x, "y": p.y} for p in result.points],
        "hit": result.hit_target
    }