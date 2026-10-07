# FRC 2026 Projectile Trajectory Simulator

A web-based physics simulator for FIRST Robotics Competition (FRC) teams to calculate and optimize shooting
trajectories. It models gravity, quadratic air drag, backspin/Magnus lift, and interactive error analysis while keeping
uncalibrated FUEL-specific aerodynamic assumptions explicit.

![React](https://img.shields.io/badge/React-18.0+-61DAFB.svg?logo=react&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.68+-009688.svg?logo=fastapi&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.9+-3776AB.svg?logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## Features

### Physics Modeling

- **Gravitational acceleration**: Standard configurable 9.81 m/s²; altitude and temperature adjust air density, not gravity.
- **Quadratic air drag**: Standard $\frac{1}{2}\rho A C_d v^2$ force law.
- **Magnus effect**: Arbitrary 3-D spin vectors using the dimensionless spin parameter $S=\omega_\perp r/v$.
- **Spin decay**: Disabled by default for FUEL until a measured decay time constant is available; an optional calibrated time constant is integrated as part of the ODE.
- **Environment**: Air-density correction for temperature/altitude in the legacy Python adapter plus field-axis wind vectors in the canonical 3-D core.
- **Numerical solvers**: True whole-state RK4 and adaptive Dormand-Prince RK45. The browser uses fixed-step RK4 by default; the new 3-D API defaults to RK45.

### 3-D Engine and Coordinates

The canonical engine uses a right-handed field coordinate system: **x** is forward/downrange, **y** is left lateral, and **z** is up. State order is x, y, z, vx, vy, vz, omega_x, omega_y, omega_z in SI units.

Air drag and Magnus lift use air-relative velocity v - wind. Only the component of spin perpendicular to that airflow contributes to Magnus lift. Launch state can also include robot field velocity, so a moving robot's velocity is added to the shooter-relative exit velocity before integration.

The React simulator remains visually 2-D in this milestone: its existing downrange/height controls map to the x/z plane, and positive backspin maps to the negative-y spin axis. The browser keeps RK4 for fast local optimization sweeps while POST /api/simulate3d defaults to adaptive RK45.

### Interactive Web GUI

- **Real-time Visualization**: Instant SVG trajectory plotting using React.
- **Backspin Estimator**: Calculate approximate RPM based on flywheel specs (diameter, gearing, compression).
- **Optimization Tools**:
    - Auto-calculate optimal launch angle.
    - Find minimum required velocity.
    - "Best Fit" mode to optimize both velocity and angle simultaneously.
- **Error Envelope**: Visualize how small inconsistencies in shooter speed or angle affect accuracy.

## Project Structure

This project is structured as a modern full-stack application designed for Vercel deployment:

```text
frc-simulator/
├── api/                   # Python Backend (FastAPI) & Physics Engine
│   ├── main.py            # API entry point
│   └── trajectory_simulator.py  # Core physics logic
├── src/                   # Frontend (React + Vite)
│   ├── components/
│   │   └── TrajectorySimulator.jsx # Main UI & Client-side simulation
│   ├── App.jsx
│   └── index.css          # Tailwind styling
├── public/                # Static assets
└── vercel.json            # Deployment configuration
```

## Installation & Local Development

### Prerequisites

- Node.js (v18+)
- Python (v3.9+)

### 1. Setup

Clone the repository and install dependencies:

```bash
git clone https://github.com/your-username/frc-trajectory-simulator.git
cd frc-trajectory-simulator

# Install Frontend Dependencies
npm install

# Install Backend Dependencies (Optional for local API testing)
pip install -r requirements.txt
```

### 2. Run Locally

To run the frontend (which handles the simulation visualization):

```bash
npm run dev
```

Open `http://localhost:5173` in your browser.

> **Note:** The current React component performs physics calculations client-side for zero-latency feedback. The Python
> API is set up to allow for advanced server-side calculations or data logging in the future.

## Deployment

This project is optimized for **Vercel**.

1. Install the Vercel CLI:
   ```bash
   npm install -g vercel
   ```
2. Deploy:
   ```bash
   vercel
   ```
3. Use default settings:
    - Build Command: `npm run build`
    - Output Directory: `dist`

Alternatively, push to GitHub and connect your repository to Vercel. The included `vercel.json` will automatically
handle the Python/React hybrid build.

## Physics Model Details

### Air Drag

The simulator uses the standard quadratic drag model:
$$F_{drag} = -\frac{1}{2} \rho A C_d v^2 \hat{v}$$

### Magnus Effect

The Magnus force from backspin creates lift perpendicular to the velocity:
$F_{magnus} = \frac{1}{2} \rho A C_l v^2 \hat{m}$

The current 2-D baseline uses $S=|\omega|r/v$ and ramps $C_l$ linearly to the configured cap by $S=0.5$. The configured FUEL lift coefficient is applied exactly once.

### Calibration Status

The 2026 FUEL defaults are intentionally conservative rather than presented as measured constants:

- Nominal mass is **0.215 kg**, the midpoint of the official ~0.203-0.227 kg range.
- $C_d=0.47$ is an **uncalibrated sphere-like baseline**; real foam-ball drag can vary with Reynolds number, wear, and surface condition.
- $C_l=0.25$ is an **uncalibrated lift cap** for the simple spin-parameter model.
- FUEL spin decay is **off by default** because no FUEL-specific spin-down data is available.
- HUB hit detection uses an interpolated descending crossing and approximate ball-center clearance; full 3-D hex-edge/rim/funnel contact remains a future milestone.
- Flywheel-to-exit-speed/backspin calculations remain rough launcher estimates and should be replaced by measured exit conditions when possible.

Reference background: [NASA sphere drag](https://www1.grc.nasa.gov/beginners-guide-to-aeronautics/drag-of-a-sphere/) and [FIRST 2026 season materials](https://www.firstinspires.org/resources/library/frc/season-materials).

## Python API Usage

While the web interface is the primary tool, you can still use the physics engine programmatically for data analysis or
scriptable optimizations.

```python
from api.trajectory_simulator import (
    PhysicsEngine, TrajectorySimulator,
    GamePieceProperties, GamePiece, EnvironmentConditions,
    LaunchParameters, Target
)

# Setup
piece = GamePieceProperties.from_game_piece(GamePiece.FUEL)
env = EnvironmentConditions()
physics = PhysicsEngine(piece, env)
sim = TrajectorySimulator(physics)

# Run Simulation
launch = LaunchParameters(
    position=(-3.0, 0.5),
    velocity=12.0,
    angle=45.0,
    spin_rate=209  # rad/s (~2000 RPM)
)
result = sim.simulate(launch)

print(f"Range: {result.range_distance:.2f}m")
print(f"Hit Target: {result.hit_target}")
```

## Tips for FRC Teams

1. **Flywheel Tuning**: Use the "Backspin Calculator" toggle. Enter your wheel diameter and compression to see estimated
   backspin.
2. **Error Envelopes**: Don't just find the perfect angle. Turn on the "Error Envelope" to see if a +/- 2° variance
   causes a miss. A robust shot is better than a perfect theoretical shot.
3. **Ideal vs. Real**: Toggle "Show Ideal" to see how much gravity-only physics differs from the drag+lift model. This
   helps explain why standard kinematic equations fail for light game pieces like the 2024 Note or 2026 Fuel.

## License

MIT License - Free to use for all FRC teams.

---
*Good luck at competition!* 🤖